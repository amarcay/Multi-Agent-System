# MCP
from langchain_mcp_adapters.client import MultiServerMCPClient
# Langgraph
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.chat_history import InMemoryChatMessageHistory

from datetime import datetime


class WebAgent:
    """Agent de recherche web utilisant MCP et un LLM cloud."""

    def __init__(
        self,
        llm,
        mcp_url: str = "http://localhost:8003/mcp",
        transport: str = "streamable_http"
    ):
        """
        Initialise le WebAgent.

        Args:
            llm: Le modèle de langage à utiliser (ex: ChatGoogleGenerativeAI)
            mcp_url: URL du serveur MCP pour la recherche web
            transport: Type de transport MCP
        """
        self.llm = llm
        self.mcp_url = mcp_url
        self.transport = transport
        self.client = None
        self.agent = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialise la connexion MCP et crée l'agent.
        Doit être appelé avant d'utiliser l'agent.
        """
        if self._initialized:
            return

        print("\t--- Connexion au serveur MCP Web... ---")

        self.client = MultiServerMCPClient({
            "WEB-Server": {
                "url": self.mcp_url,
                "transport": self.transport,
            }
        })

        # Chargement des outils
        web_tools = await self.client.get_tools(server_name="WEB-Server")
        assert web_tools[0].name == "web_search", "L'outil web_search n'est pas disponible"

        print("\t--- Connexion MCP OK ---")

        # Récupération de la date/heure actuelles
        jours_semaine = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
        aujourdhui = datetime.today()
        today_date = f"{jours_semaine[aujourdhui.weekday()]} {aujourdhui.day} {aujourdhui.year}"
        hour = datetime.now().strftime('%H:%M')

        # Création de l'agent
        self.agent = create_agent(
            self.llm,
            tools=web_tools,
            debug=False,
            system_prompt=f"""
            Tu es un assistant qui doit absolument utiliser l'outil web_search pour répondre aux questions.
            Ne réponds jamais sans utiliser cet outil.
            Tu ne dois jamais t'excuser.
            Aujourd'hui, nous sommes le {today_date}, il est {hour}.
            Si tu reçois des données anonymisées (sous la forme "*******"), tu dois rester le plus général possible dans ta réponse,
            ne pas entrer dans les détails et adapter ta requête.
            """
        )

        self._initialized = True
        print("\t--- WebAgent initialisé ---")

    async def search(self, query: str, tags: list[str] | None = None) -> str:
        """
        Effectue une recherche web avec la requête donnée.

        Args:
            query: La requête de recherche
            tags: Tags optionnels pour le tracing (ex: ["final_answer"])

        Returns:
            Le contenu de la réponse de l'agent
        """
        if not self._initialized:
            await self.initialize()

        sub_state = {"messages": [HumanMessage(content=query)]}

        if tags:
            result = await self.agent.with_config(tags=tags).ainvoke(sub_state)
        else:
            result = await self.agent.ainvoke(sub_state)

        return result['messages'][-1].content

    async def invoke(
        self,
        state: dict,
        config: RunnableConfig,
        chat_history: InMemoryChatMessageHistory,
        llm_for_query_extraction = None
    ) -> dict:
        """
        Méthode complète pour utiliser l'agent comme nœud dans un graphe LangGraph.

        Args:
            state: L'état du graphe contenant les messages
            config: Configuration du runtime (conservé pour compatibilité LangGraph)
            chat_history: Historique de la conversation
            llm_for_query_extraction: LLM optionnel pour extraire la requête de recherche

        Returns:
            Dictionnaire avec la clé "messages" contenant la réponse
        """
        del config  # Conservé pour compatibilité interface LangGraph
        if not self._initialized:
            await self.initialize()

        # Récupération de la date pour le prompt
        jours_semaine = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
        aujourdhui = datetime.today()
        today_date = f"{jours_semaine[aujourdhui.weekday()]} {aujourdhui.day} {aujourdhui.year}"

        # Utiliser web_sub_query si disponible (après anonymisation potentielle)
        query_to_search = state.get("web_sub_query")

        if not query_to_search and llm_for_query_extraction:
            user_query_hist = state["messages"][-3:]
            user_query_system = (
                f"""Tu es un agent assistant spécialisé dans la recherche web. Aujourd'hui, nous sommes le {today_date}.
                Ta tâche est d'analyser les messages reçus (y compris le contexte et la question de l'utilisateur) afin de déterminer la recherche web la plus pertinente à effectuer pour répondre précisément à la demande.
                Pour chaque question ou besoin d'information détecté, identifie les mots-clés et sujets importants, puis formule une seule requête de recherche web parfaitement adaptée.
                Si plusieurs axes de recherche sont possibles, choisis uniquement le plus pertinent et ne propose qu'une seule requête.
                Sois synthétique, professionnel et va droit au but dans la formulation de la requête de recherche.

                Exemple :
                - input : "Je cherche des informations récentes sur la réglementation européenne concernant l'intelligence artificielle."
                - output : "dernières actualités réglementation européenne intelligence artificielle"

                Les messages :
                """
            )
            from langchain_core.messages import SystemMessage
            query_list = [SystemMessage(content=user_query_system)] + list(user_query_hist)
            result_query = await llm_for_query_extraction.ainvoke(query_list)
            query_to_search = result_query.content
        elif not query_to_search:
            # Fallback: utiliser le dernier message utilisateur
            last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
            query_to_search = last_human.content if last_human else ""

        print(f"Agent_WEB_Cloud recherche : '{query_to_search}'")

        # Effectuer la recherche
        web_response = await self.search(query_to_search, tags=["final_answer"])

        final_message = AIMessage(content=web_response, name="Agent_WEB_Cloud")

        # Ajouter à l'historique
        original_user_message = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        if original_user_message:
            chat_history.add_messages([original_user_message, final_message])

        print(f"✅ Agent WEB Cloud a terminé : {web_response[:100]}...")
        return {"messages": [final_message]}

    @property
    def is_initialized(self) -> bool:
        """Retourne True si l'agent est initialisé."""
        return self._initialized
