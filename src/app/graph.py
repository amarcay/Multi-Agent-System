# --- Imports ---
import warnings
import json
import re

# Langgraph
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import Literal, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

# Variables d'environnement
from dotenv import load_dotenv
load_dotenv()
import os

# PII Presidio
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer
from presidio_anonymizer.entities import OperatorConfig
from src.presidio.recognizer import *
from src.agent.agent_web import WebAgent
from src.agent.agent_codeact import CodeActAgent
from src.memory.memory_manager import MemoryManager
from src.memory.memory_extractor import extract_facts, generate_summary

nlp_config = {"nlp_engine_name": "spacy", "models": [{"lang_code": "fr", "model_name": "fr_core_news_lg"}]}
provider = NlpEngineProvider(nlp_configuration=nlp_config)
nlp_engine_fr = provider.create_engine()

registry = RecognizerRegistry(supported_languages=["fr"])
registry.load_predefined_recognizers(languages=["fr"])
registry.add_recognizer(FrNirRecognizer())
registry.add_recognizer(FrPhoneRecognizer())
registry.add_recognizer(FrZIPcodeRecognizer())
registry.add_recognizer(FrAdresseRecognizer())
registry.add_recognizer(FrIbanRecognizer())

unnecessary = ['DateRecognizer','MedicalLicenseRecognizer']
[registry.remove_recognizer(rec) for rec in unnecessary]

context_enhancer = LemmaContextAwareEnhancer(context_prefix_count=5, 
                                             context_suffix_count=2,
                                             context_similarity_factor=0.45,
                                             min_score_with_context_similarity=0.4)

unnecessary_entity_types = ['ORGANIZATION', 'URL', 'CRYPTO']
for entity_type in unnecessary_entity_types:
    recognizers_to_remove = [
        rec for rec in registry.recognizers 
        if entity_type in rec.supported_entities
    ]
    for rec in recognizers_to_remove:
        registry.remove_recognizer(rec.name)

analyzer = AnalyzerEngine(
    registry=registry,
    nlp_engine=nlp_engine_fr,
    supported_languages=["fr"],
    context_aware_enhancer=context_enhancer
)        

def filtrer_FP(results, texte):
    faux_positifs = []
    for res in results:
        if res.entity_type == "LOCATION":
            extrait = texte[res.start:res.end].strip()
            if re.match(r"^FR\d{2}", extrait):
                faux_positifs.append(res)
            elif extrait.upper() in ["IBAN", "FR", "IBAN FR76"]:
                faux_positifs.append(res)
    return [r for r in results if r not in faux_positifs]

anonymizer = AnonymizerEngine()

# --- Init des modèles ---

# Ollama
from langchain_ollama import ChatOllama
# Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Agents "locaux"
llm_supervisor_local = ChatOpenAI(model="gpt-4.1") #ChatOllama(model="supervisor")
llm_simple_local = ChatOllama(model="llama3.2:3b")

# Agents "cloud"
llm_simple_cloud = ChatGoogleGenerativeAI(model="gemini-2.5-flash",convert_system_message_to_human=True)
llm_web_cloud = ChatOpenAI(model="gpt-4.1")
llm_codeact_cloud = ChatGoogleGenerativeAI(model="gemini-2.5-flash", convert_system_message_to_human=True)
llm_memory_extractor = ChatOllama(model="llama3.2:3b")


# --- Initialisation ---

from datetime import datetime
jours_semaine = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
aujourdhui = datetime.today()
today_date = f"{jours_semaine[aujourdhui.weekday()]} {aujourdhui.day} {aujourdhui.year}"
hour = datetime.now().strftime('%H:%M')

warnings.filterwarnings('ignore')

chats_by_session_id = {}

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    """Récupère ou crée un historique de chat pour une session donnée."""
    if session_id not in chats_by_session_id:
        chats_by_session_id[session_id] = InMemoryChatMessageHistory()
    return chats_by_session_id[session_id]

def check_config(config: RunnableConfig):
    """Vérifie que la configuration contient bien le thread_id."""
    if "configurable" not in config or "thread_id" not in config["configurable"]:
        raise ValueError(
            "Assurez-vous que la config contient : {'configurable': {'thread_id': 'une_valeur'}}"
        )
        
def has_document_context(messages: list) -> bool:
    """Détecte si les messages contiennent un contexte documentaire OpenWebUI."""
    for msg in messages:
        if isinstance(msg, SystemMessage):
            content = msg.content
            if '<source' in content and ('id=' in content or '<context' in content):
                return True
    return False

# --- Définition des états et des routes ---
class Router(TypedDict):
    """Définit la prochaine étape choisie par le superviseur."""
    next: Literal["Agent_WEB_Cloud", "Agent_RAG_Document", "Agent_Simple_Cloud", "Agent_Simple_Local", "Agent_CodeAct_Cloud", "FINISH"]

class Webquery(TypedDict):
    """Représente une requête web avec le champ 'query' de type str."""
    query: str

class State(TypedDict):
    """Représente l'état du graphe, incluant les nouvelles étapes."""
    input: str
    messages: Annotated[list, add_messages]
    next: str
    web_sub_query: str | None
    is_confidential: bool | None
    anonymized_text: str | None
    human_approved: bool | None
    retrieved_memories: list[str] | None

# --- Création du Graphe et des Agents ---

async def make_graph(checkpointer: AsyncSqliteSaver | None = None, memory_manager: MemoryManager | None = None):
    print("--- Création du graph... ---\n")
    print("\t--- Création des Agents... ---\n")

    # Création du WebAgent
    web_agent = WebAgent(llm=llm_web_cloud)
    await web_agent.initialize()

    # Création du CodeActAgent
    codeact_agent = CodeActAgent(llm=llm_codeact_cloud)
    try:
        await codeact_agent.initialize()
    except Exception as e:
        print(f"⚠️ CodeActAgent non disponible (Docker requis): {e}")
        codeact_agent = None

    print("\t\t--- Création des Nœuds... ---\n")

    # --- Définition des Nœuds du Graphe ---

    async def confidentiality_check_node(state: State) -> dict:
        """
        Vérifie la confidentialité de la requête avec Presidio. S'il y a des PII, 
        il met à jour l'état, ce qui déclenchera l'anonymisation automatique.
        """
        print("--- Executing Confidentiality Check Node (Presidio) ---")

        text_to_analyze = state.get("web_sub_query")

        if not text_to_analyze:
            last_message = state["messages"][-1]
            if isinstance(last_message, HumanMessage):
                if isinstance(last_message.content, list):
                    text_part = next((p for p in last_message.content if p.get("type") == "text"), None)
                    if text_part:
                        text_to_analyze = text_part["text"]
                else:
                    text_to_analyze = last_message.content

        if not text_to_analyze:
            return {"is_confidential": False, "web_sub_query": None, "anonymized_text": None}

        try:
            analyzer_results = analyzer.analyze(
                text=text_to_analyze, 
                language="fr", 
                score_threshold=0.7
            )
        
            analyzer_results = filtrer_FP(analyzer_results, text_to_analyze)

            update_dict = {
                "web_sub_query": text_to_analyze,
            }

            if not analyzer_results:
                print("--- Aucune PII détectée ---")
                update_dict["is_confidential"] = False
                update_dict["anonymized_text"] = text_to_analyze
            else:
                print(f"--- PII détectées : {[r.entity_type for r in analyzer_results]} ---")
            
                anonymization_operators = {
                    "DEFAULT": OperatorConfig(
                        "mask",
                        {
                            "type": "fixed",
                            "masking_char": "*",
                            "chars_to_mask": 50,
                            "from_end": False
                        }
                    )
                }
            
                anonymized_result = anonymizer.anonymize(
                    text=text_to_analyze, 
                    analyzer_results=analyzer_results,
                    operators=anonymization_operators
                )
            
                print(f"Texte anonymisé : {anonymized_result.text}")
                update_dict["is_confidential"] = True
                update_dict["anonymized_text"] = anonymized_result.text

            return update_dict

        except Exception as e:
            print(f"--- Exception inattendue lors de l'analyse Presidio : {e} ---")
            return {
                "is_confidential": False, 
                "web_sub_query": text_to_analyze, 
                "anonymized_text": text_to_analyze
            }


    async def anonymizer_node(state: State) -> dict:
        """
        Anonymise la requête en utilisant le résultat pré-calculé par confidentiality_check_node.
        """
        print("--- Executing Anonymizer Node (Presidio) ---")

        if not state.get("human_approved"):
            print("--- Pas d'approbation humaine, anonymisation ignorée. ---")
            return {}

        anonymized_text = state.get("anonymized_text")
        if not anonymized_text:
            return {}

        if state.get("web_sub_query"):
            print(f"Anonymizing web_sub_query...")
            return {"web_sub_query": anonymized_text}
        else:
            print(f"Anonymizing main message history...")

            updated_messages = list(state["messages"])

            if not updated_messages or not isinstance(updated_messages[-1], HumanMessage):
                return {}

            message_to_replace = updated_messages[-1]
            is_multimodal = isinstance(message_to_replace.content, list)

            if is_multimodal:
                new_content_parts = [part for part in message_to_replace.content if part.get("type") != "text"]
                new_content_parts.insert(0, {"type": "text", "text": anonymized_text})
                anonymized_content = new_content_parts
            else:
                anonymized_content = anonymized_text

            anonymized_message = HumanMessage(content=anonymized_content, id=message_to_replace.id)
            updated_messages[-1] = anonymized_message

            return {"messages": updated_messages}

    async def human_approval_node(state: State) -> dict:
        """
        Pause le graphe et attend une entrée humaine.
        L'API interceptera cette interruption et enverra les données au client.
        """
        print("--- Interruption pour approbation humaine ---")

        decision = interrupt(
            {
                "question": "J'ai détecté des informations qui semblent personnelles. "
                           "Souhaitez-vous que je les anonymise avant de continuer sur le web ?",
                "options": ["oui", "non"]
            }
        )

        print(f"--- Décision Humaine Reçue via resume : {decision} ---")

        if str(decision).lower().strip() in ["yes", "oui"]:
            print("--- Approbation reçue. Continuation avec l'anonymisation. ---")
            return {"human_approved": True}
        else:
            print("--- Refus reçu. Annulation de l'anonymisation. ---")
            return {"human_approved": False}

    members = ["Agent_Simple_Local", "Agent_WEB_Cloud", "Agent_Simple_Cloud", "Agent_RAG_Document", "Agent_CodeAct_Cloud"]
    
    def supervisor_node(state: State, config: RunnableConfig):
        """Choisit le prochain agent à exécuter en fonction de la demande utilisateur."""
        check_config(config)
        session_id = config["configurable"]["session_id"]
        chat_history = get_chat_history(session_id)
        incoming_messages = state.get("messages", [])
        
        if has_document_context(incoming_messages):
            print("CONTEXTE DOCUMENTAIRE DÉTECTÉ → Routage automatique vers Agent RAG Document")
            return {"next": "Agent_RAG_Document"}
            
        system_prompt = f"""
Aujourd'hui, nous sommes le {today_date}, il est {hour}.
Vous êtes un superviseur chargé de router les requêtes vers l'agent le plus approprié parmi : {members}.

## Compétences de chaque agent :
- **Agent_RAG_Document** : analyse de documents fournis en contexte (PDFs, sources, fichiers)
- **Agent_WEB_Cloud** : recherche web, données actuelles (météo, actualités, lieux, commerces, prix, événements, informations à jour)
- **Agent_Simple_Local** : réponses instantanées pour tâches simples ne nécessitant PAS d'informations externes (reformulations, explications de concepts connus, conversations générales)
- **Agent_Simple_Cloud** : analyses complexes, raisonnement avancé, synthèses élaborées (sans données sensibles)
- **Agent_CodeAct_Cloud** : exécution de code Python, calculs mathématiques, traitement de données, analyses numériques, résolution de problèmes nécessitant du code

## Règles de routage (dans l'ordre de priorité) :

1. Document fourni en contexte → **Agent_RAG_Document**
2. Requête contenant des PII ou données sensibles → **Agent_Simple_Local**
3. Nécessite calculs, code Python, traitement de données, analyses numériques → **Agent_CodeAct_Cloud**
4. Requête nécessitant des informations du monde réel actuelles (lieux, commerces, restaurants, horaires, météo, actualités, prix, événements, personnes, entreprises) → **Agent_WEB_Cloud**
5. Tâche complexe nécessitant un raisonnement approfondi → **Agent_Simple_Cloud**
6. Tâche simple basée uniquement sur des connaissances générales → **Agent_Simple_Local**

## Exemples :

Utilisateur : "Quel temps fait-il à Paris aujourd'hui ?"
{{"next": "Agent_WEB_Cloud"}}

Utilisateur : "Trouve-moi un bon restaurant italien à Rennes"
{{"next": "Agent_WEB_Cloud"}}

Utilisateur : "Quels sont les horaires d'ouverture de la piscine municipale ?"
{{"next": "Agent_WEB_Cloud"}}

Utilisateur : "Résume ce texte : 'Le développement durable...'"
{{"next": "Agent_Simple_Local"}}

Utilisateur : "Explique-moi la théorie de la relativité en termes simples."
{{"next": "Agent_Simple_Local"}}

Utilisateur : "Analyse les implications philosophiques de l'IA sur le marché du travail"
{{"next": "Agent_Simple_Cloud"}}

Utilisateur : "Calcule la somme des 100 premiers nombres premiers"
{{"next": "Agent_CodeAct_Cloud"}}

Utilisateur : "Écris un script Python qui trie une liste de nombres"
{{"next": "Agent_CodeAct_Cloud"}}

Utilisateur : "Combien font 15 + 27 ?"
{{"next": "Agent_CodeAct_Cloud"}}

Utilisateur : "Bonjour, comment vas-tu ?"
{{"next": "Agent_Simple_Local"}}

## Question clé à se poser :
"Cette requête nécessite-t-elle des informations qui peuvent changer ou qui dépendent du monde réel actuel ?"
- Si OUI → **Agent_WEB_Cloud**
- Si NON → évaluer la complexité pour choisir entre Local et Cloud

Répondez **uniquement** avec un objet JSON valide :
{{"next": "nom_de_l_agent"}}
"""

        messages = [SystemMessage(content=system_prompt), state["messages"][-1]]
    
        response = llm_supervisor_local.with_structured_output(Router, method="json_mode").invoke(messages)
        print(f"Superviseur a choisi : {response['next']}")
        return {"next": response['next']}

    async def agent_rag_document_node(state: State, config: RunnableConfig) -> dict:
        """Agent spécialisé pour analyser les documents fournis via OpenWebUI."""
        print("--- Executing Agent RAG Document ---")
        check_config(config)
        session_id = config["configurable"]["session_id"]
        chat_history = get_chat_history(session_id)
        
        incoming_messages = state.get("messages", [])
        
        system_messages = [msg for msg in incoming_messages if isinstance(msg, SystemMessage)]
        user_query = next((msg for msg in reversed(incoming_messages) if isinstance(msg, HumanMessage)), None)
        
        if not user_query:
            return {"messages": [AIMessage(content="Aucune question détectée.", name="Agent_RAG_Document")]}
        
        rag_system_prompt = (
            f"""Tu es un assistant expert en analyse de documents. Aujourd'hui, nous sommes le {today_date}.
            IMPORTANT :
            - Un contexte documentaire t'a été fourni dans les messages système précédents.
            - Analyse ce contexte avec précision pour répondre à la question de l'utilisateur.
            - Cite les sources en utilisant le format [id] quand c'est approprié.
            - Si l'information demandée n'est pas dans le contexte, indique-le clairement.
            - Sois précis et factuel dans tes réponses.
            Ne t'excuse jamais et réponds de manière directe et professionnelle.
            """
        )
        
        messages_to_send = [
            SystemMessage(content=rag_system_prompt),
            *system_messages,
            *list(chat_history.messages)[-10:],  # Limité à 10 derniers messages pour éviter saturation
            user_query
        ]
        
        print(f"📄 Agent RAG Document traite {len(system_messages)} message(s) système avec contexte")
        print(f"📋 Question : {user_query.content[:100]}...")
        
        response = await llm_simple_local.with_config(tags=["final_answer"]).ainvoke(messages_to_send)
        
        ai_message = AIMessage(content=response.content, name="Agent_RAG_Document")
        chat_history.add_messages([user_query, ai_message])
        
        print(f"Agent RAG Document : {response.content[:200]}...")
        return {"messages": [ai_message]}

    async def agent_web_cloud_node(state: State, config: RunnableConfig) -> dict:
        """Exécute l'agent WEB Cloud avec support de la requête anonymisée."""
        print("--- Executing Agent WEB Cloud ---")
        check_config(config)
        session_id = config["configurable"]["session_id"]
        chat_history = get_chat_history(session_id)

        return await web_agent.invoke(
            state=state,
            config=config,
            chat_history=chat_history,
            llm_for_query_extraction=llm_supervisor_local
        )

    async def agent_simple_local_node(state: State, config: RunnableConfig) -> dict:
        """Exécute une conversation simple sur un modèle local."""
        print("--- Executing Agent Simple Local ---")
        check_config(config)
        session_id = config["configurable"]["session_id"]
        chat_history = get_chat_history(session_id)
        user_query = state["messages"][-1]
        memory_context = _build_memory_context(state)

        system_prompt = (
            f"""Tu es un assistant conversationnel. Aujourd'hui, nous sommes le {today_date}, il est {hour}.
            En te basant sur l'historique complet de la conversation ci-dessous,
            réponds à la dernière question de l'utilisateur.
            Utilise les informations des messages précédents si c'est pertinent.
            Tu ne dois jamais t'excuser.{memory_context}
            """
        )

        messages_with_history = [SystemMessage(content=system_prompt)] + list(chat_history.messages)[-15:] + [user_query]
        response = await llm_simple_local.with_config(tags=["final_answer"]).ainvoke(messages_with_history)
        ai_message = AIMessage(content=response.content, name="Agent_Simple_Local")
        chat_history.add_messages([user_query, ai_message])

        print(f"Agent Simple Local : {response.content[:200]}...")
        return {"messages": [ai_message]}
    
    async def agent_simple_cloud_node(state: State, config: RunnableConfig) -> dict:
        """Exécute une conversation simple sur un modèle cloud."""
        print("--- Executing Agent Simple Cloud ---")
        check_config(config)
        session_id = config["configurable"]["session_id"]
        chat_history = get_chat_history(session_id)
        user_query = state["messages"][-1]
        memory_context = _build_memory_context(state)

        system_prompt = (
            f"""Tu es un assistant conversationnel. Aujourd'hui, nous sommes le {today_date}, il est {hour}.
            En te basant sur l'historique complet de la conversation ci-dessous,
            réponds à la dernière question de l'utilisateur.
            Utilise les informations des messages précédents si c'est pertinent.
            Tu ne dois jamais t'excuser.{memory_context}
            """
        )

        messages_with_history = [SystemMessage(content=system_prompt)] + list(chat_history.messages)[-15:] + [user_query]

        # IMPORTANT: Ajouter le tag final_answer
        response = await llm_simple_cloud.with_config(tags=["final_answer"]).ainvoke(messages_with_history)
        ai_message = AIMessage(content=response.content, name="Agent_Simple_Cloud")
        chat_history.add_messages([user_query, ai_message])

        print(f"Agent Simple Cloud : {response.content[:200]}...")
        return {"messages": [ai_message]}
    
    # --- Nœuds Mémoire ---

    async def memory_retrieval_node(state: State, config: RunnableConfig) -> dict:
        """Recherche les mémoires pertinentes pour la requête utilisateur."""
        if not memory_manager:
            return {"retrieved_memories": []}

        print("--- Executing Memory Retrieval Node ---")
        check_config(config)
        session_id = config["configurable"]["session_id"]

        last_message = state["messages"][-1]
        query = last_message.content if isinstance(last_message.content, str) else str(last_message.content)

        try:
            # Recherche de faits pertinents (session + global)
            session_memories = await memory_manager.retrieve_memories(session_id, query, top_k=3)
            global_memories = await memory_manager.retrieve_all_memories(query, top_k=2)

            # Dédupliquer
            all_memories = list(dict.fromkeys(session_memories + global_memories))

            # Récupérer les résumés récents
            summaries = await memory_manager.get_recent_summaries(session_id, limit=1)
            if summaries:
                all_memories.append(f"[Résumé précédent] {summaries[0]}")

            print(f"--- {len(all_memories)} mémoire(s) récupérée(s) ---")
            return {"retrieved_memories": all_memories}
        except Exception as e:
            print(f"--- Erreur lors de la récupération des mémoires : {e} ---")
            return {"retrieved_memories": []}

    async def memory_extraction_node(state: State, config: RunnableConfig) -> dict:
        """Extrait les faits et génère un résumé après la réponse de l'agent."""
        if not memory_manager:
            return {}

        print("--- Executing Memory Extraction Node ---")
        check_config(config)
        session_id = config["configurable"]["session_id"]

        # Prendre les derniers messages de la conversation
        recent_messages = state["messages"][-4:]

        try:
            # Extraire les faits en parallèle avec le résumé
            facts = await extract_facts(recent_messages, llm_memory_extractor)
            if facts:
                await memory_manager.add_facts(session_id, facts)
                print(f"--- {len(facts)} fait(s) extrait(s) et sauvegardé(s) ---")

            summary = await generate_summary(recent_messages, llm_memory_extractor)
            if summary:
                await memory_manager.add_summary(session_id, summary)
                print(f"--- Résumé sauvegardé ---")
        except Exception as e:
            print(f"--- Erreur lors de l'extraction mémoire : {e} ---")

        return {}

    def _build_memory_context(state: State) -> str:
        """Construit le contexte mémoire à injecter dans les prompts agents."""
        memories = state.get("retrieved_memories")
        if not memories:
            return ""

        memory_lines = "\n".join(f"- {m}" for m in memories)
        return f"\n\n## Informations mémorisées des conversations précédentes :\n{memory_lines}\n"

    # --- Nœud CodeAct ---

    async def agent_codeact_cloud_node(state: State, config: RunnableConfig) -> dict:
        """Exécute l'agent CodeAct pour les tâches nécessitant du code Python."""
        print("--- Executing Agent CodeAct Cloud ---")
        check_config(config)
        session_id = config["configurable"]["session_id"]
        chat_history = get_chat_history(session_id)

        if codeact_agent is None:
            ai_message = AIMessage(
                content="L'agent CodeAct n'est pas disponible. Docker est requis pour exécuter du code. "
                        "Veuillez installer Docker et redémarrer le serveur.",
                name="Agent_CodeAct_Cloud"
            )
            chat_history.add_messages([state["messages"][-1], ai_message])
            return {"messages": [ai_message]}

        return await codeact_agent.invoke(
            state=state,
            config=config,
            chat_history=chat_history,
        )

    print("\t\t--- Création des Nœuds OK ---\n")
    print("\t--- Création des Agents OK ---\n")

    # --- Assemblage du Graphe ---
    builder = StateGraph(State)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("memory_retrieval", memory_retrieval_node)
    builder.add_node("Agent_Simple_Local", agent_simple_local_node)
    builder.add_node("Agent_RAG_Document", agent_rag_document_node)
    builder.add_node("Agent_Simple_Cloud", agent_simple_cloud_node)
    builder.add_node("Agent_WEB_Cloud", agent_web_cloud_node)
    builder.add_node("Agent_CodeAct_Cloud", agent_codeact_cloud_node)
    builder.add_node("Agent_Confidentiel", confidentiality_check_node)
    builder.add_node("anonymizer", anonymizer_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("memory_extraction", memory_extraction_node)

    builder.set_entry_point("supervisor")

    # Supervisor → memory_retrieval
    builder.add_edge("supervisor", "memory_retrieval")

    def should_check_confidentiality(state: State):
        """Vérifie si on doit passer par le check de confidentialité."""
        destination = state.get("next")
        if destination in ["Agent_WEB_Cloud", "Agent_Simple_Cloud", "Agent_CodeAct_Cloud"]:
            return "Agent_Confidentiel"
        else:
            return destination

    builder.add_conditional_edges(
        "memory_retrieval",
        should_check_confidentiality,
        {
            "Agent_Confidentiel": "Agent_Confidentiel",
            "Agent_Simple_Local": "Agent_Simple_Local",
            "Agent_RAG_Document": "Agent_RAG_Document",
        }
    )
      
    def after_confidentiality_check(state: State):
        """Décide s'il faut demander l'approbation humaine ou continuer directement."""
        if state.get("is_confidential"):
            print("--- PII détectées. Routage vers l'approbation humaine. ---")
            return "human_approval"
        else:
            print("--- Aucune PII détectée. Continuation directe. ---")
            return state.get("next")
            
    builder.add_conditional_edges(
        "Agent_Confidentiel",
        after_confidentiality_check,
        {
            "human_approval": "human_approval",
            "Agent_WEB_Cloud": "Agent_WEB_Cloud",
            "Agent_Simple_Cloud": "Agent_Simple_Cloud",
            "Agent_CodeAct_Cloud": "Agent_CodeAct_Cloud",
        }
    )

    def after_human_approval(state: State):
        """Décide si on anonymise ou si on continue sans anonymisation."""
        destination = state.get("next")
        if state.get("human_approved"):
            print(f"--- Utilisateur a approuvé l'anonymisation → anonymizer ---")
            return "anonymizer"
        else:
            print(f"--- Utilisateur a refusé l'anonymisation → {destination} ---")
            return destination

    builder.add_conditional_edges(
        "human_approval",
        after_human_approval,
        {
            "anonymizer": "anonymizer",
            "Agent_WEB_Cloud": "Agent_WEB_Cloud",
            "Agent_Simple_Cloud": "Agent_Simple_Cloud",
            "Agent_CodeAct_Cloud": "Agent_CodeAct_Cloud",
        }
    )

    def route_after_anonymization(state: State):
        """Une fois l'anonymisation faite, on continue vers la destination initiale."""
        return state.get("next")

    builder.add_conditional_edges(
        "anonymizer",
        route_after_anonymization,
        {
            "Agent_WEB_Cloud": "Agent_WEB_Cloud",
            "Agent_Simple_Cloud": "Agent_Simple_Cloud",
            "Agent_CodeAct_Cloud": "Agent_CodeAct_Cloud",
        }
    )

    # Agents → memory_extraction → END
    builder.add_edge("Agent_RAG_Document", "memory_extraction")
    builder.add_edge("Agent_Simple_Local", "memory_extraction")
    builder.add_edge("Agent_Simple_Cloud", "memory_extraction")
    builder.add_edge("Agent_WEB_Cloud", "memory_extraction")
    builder.add_edge("Agent_CodeAct_Cloud", "memory_extraction")
    builder.add_edge("memory_extraction", END)

    graph = builder.compile(checkpointer=checkpointer).with_config(run_name="MAS-ASI-V1-HITL-Production")
    print("--- Création du Graphe OK ---\n")
    return graph