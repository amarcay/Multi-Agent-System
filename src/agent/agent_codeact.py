"""
Agent CodeAct utilisant langgraph-codeact avec exécution dans un sandbox Docker.
Capable d'exécuter du code Python pour des calculs, traitement de données, etc.
"""

from langgraph_codeact import create_codeact
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.chat_history import InMemoryChatMessageHistory
from src.sandbox.docker_executor import DockerCodeExecutor

from datetime import datetime
from typing import Any


class CodeActAgent:
    """Agent CodeAct avec sandbox Docker pour l'exécution de code."""

    def __init__(self, llm, sandbox_executor: DockerCodeExecutor | None = None):
        self.llm = llm
        self._executor = sandbox_executor or DockerCodeExecutor()
        self._agent = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialise l'agent CodeAct avec les outils et le sandbox."""
        if self._initialized:
            return

        print("\t--- Initialisation de l'Agent CodeAct... ---")

        tools = [self._search_web, self._read_file, self._write_file]

        eval_fn = self._create_eval_fn()

        code_act = create_codeact(self.llm, tools, eval_fn)
        self._agent = code_act.compile()

        self._initialized = True
        print("\t--- Agent CodeAct initialisé ---")

    def _create_eval_fn(self):
        """Crée la fonction eval pour langgraph-codeact."""
        executor = self._executor

        def eval_fn(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            """Exécute du code Python dans le sandbox Docker."""
            return executor.execute(code, _locals)

        return eval_fn

    @staticmethod
    def _search_web(query: str) -> str:
        """Effectue une recherche web pour trouver des informations actuelles.

        Args:
            query: La requête de recherche en langage naturel

        Returns:
            Les résultats de la recherche web
        """
        return f"[Recherche web non disponible dans le sandbox. Utilisez la requête directement: {query}]"

    @staticmethod
    def _read_file(path: str) -> str:
        """Lit le contenu d'un fichier dans le sandbox.

        Args:
            path: Chemin du fichier à lire dans le sandbox

        Returns:
            Le contenu du fichier
        """
        return f"[Lecture fichier: {path} - disponible uniquement dans le conteneur Docker]"

    @staticmethod
    def _write_file(path: str, content: str) -> str:
        """Écrit du contenu dans un fichier dans le sandbox.

        Args:
            path: Chemin du fichier à écrire dans le sandbox
            content: Contenu à écrire

        Returns:
            Confirmation de l'écriture
        """
        return f"[Écriture fichier: {path} - disponible uniquement dans le conteneur Docker]"

    async def invoke(
        self,
        state: dict,
        config: RunnableConfig,
        chat_history: InMemoryChatMessageHistory,
    ) -> dict:
        """Invoque l'agent CodeAct comme nœud dans un graphe LangGraph.

        Args:
            state: L'état du graphe contenant les messages
            config: Configuration du runtime
            chat_history: Historique de la conversation

        Returns:
            Dictionnaire avec la clé "messages" contenant la réponse
        """
        if not self._initialized:
            await self.initialize()

        # Récupérer le dernier message utilisateur
        user_query = state["messages"][-1]

        # Construire le contexte
        jours_semaine = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
        aujourdhui = datetime.today()
        today_date = f"{jours_semaine[aujourdhui.weekday()]} {aujourdhui.day} {aujourdhui.year}"
        hour = datetime.now().strftime('%H:%M')

        system_prompt = (
            f"Tu es un assistant capable d'exécuter du code Python pour résoudre des problèmes. "
            f"Aujourd'hui, nous sommes le {today_date}, il est {hour}. "
            f"Tu peux écrire et exécuter du code Python pour effectuer des calculs, "
            f"du traitement de données, des analyses, etc. "
            f"Utilise le code Python quand c'est pertinent pour répondre précisément. "
            f"Tu ne dois jamais t'excuser."
        )

        # Historique limité
        history_messages = list(chat_history.messages)[-15:]

        messages_to_send = [
            SystemMessage(content=system_prompt),
            *history_messages,
            user_query,
        ]

        try:
            result = await self._agent.ainvoke(
                {"messages": messages_to_send},
            )
            response_content = result["messages"][-1].content
        except Exception as e:
            print(f"Erreur Agent CodeAct: {e}")
            response_content = f"Désolé, une erreur est survenue lors de l'exécution du code : {e}"

        ai_message = AIMessage(content=response_content, name="Agent_CodeAct_Cloud")
        chat_history.add_messages([user_query, ai_message])

        print(f"Agent CodeAct Cloud : {response_content[:200]}...")
        return {"messages": [ai_message]}

    @property
    def is_initialized(self) -> bool:
        """Retourne True si l'agent est initialisé."""
        return self._initialized

    def cleanup(self) -> None:
        """Nettoyage des ressources."""
        if self._executor:
            self._executor.cleanup()
