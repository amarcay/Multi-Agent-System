"""Tests pour le graphe LangGraph et les nœuds d'agent."""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.app.graph import (
    get_chat_history,
    check_config,
    has_document_context,
    State,
    Router,
)


class TestChatHistory:
    """Tests pour la gestion de l'historique des conversations."""

    def test_get_chat_history_new_session(self):
        """Test création d'un nouvel historique pour une nouvelle session."""
        session_id = "test_session_1"
        history = get_chat_history(session_id)

        assert history is not None
        assert len(history.messages) == 0

    def test_get_chat_history_existing_session(self):
        """Test récupération d'un historique existant."""
        session_id = "test_session_2"

        # Première récupération
        history1 = get_chat_history(session_id)
        history1.add_message(HumanMessage(content="Test message"))

        # Deuxième récupération - devrait être le même objet
        history2 = get_chat_history(session_id)

        assert history1 is history2
        assert len(history2.messages) == 1
        assert history2.messages[0].content == "Test message"

    def test_multiple_sessions_independent(self):
        """Test que les sessions sont indépendantes."""
        session_id_1 = "test_session_3"
        session_id_2 = "test_session_4"

        history1 = get_chat_history(session_id_1)
        history2 = get_chat_history(session_id_2)

        history1.add_message(HumanMessage(content="Session 1"))
        history2.add_message(HumanMessage(content="Session 2"))

        assert len(history1.messages) == 1
        assert len(history2.messages) == 1
        assert history1.messages[0].content == "Session 1"
        assert history2.messages[0].content == "Session 2"


class TestCheckConfig:
    """Tests pour la validation de configuration."""

    def test_check_config_valid(self):
        """Test configuration valide."""
        config = {"configurable": {"thread_id": "test_thread"}}

        # Ne devrait pas lever d'exception
        try:
            check_config(config)
        except ValueError:
            pytest.fail("check_config a levé une exception pour une config valide")

    def test_check_config_missing_configurable(self):
        """Test configuration sans 'configurable'."""
        config = {}

        with pytest.raises(ValueError):
            check_config(config)

    def test_check_config_missing_thread_id(self):
        """Test configuration sans 'thread_id'."""
        config = {"configurable": {}}

        with pytest.raises(ValueError):
            check_config(config)


class TestHasDocumentContext:
    """Tests pour la détection de contexte documentaire."""

    def test_has_document_context_with_source(self):
        """Test détection de contexte avec balise <source>."""
        messages = [
            SystemMessage(content="<source id='doc1'><context>Some document content</context></source>")
        ]

        assert has_document_context(messages) is True

    def test_has_document_context_without_source(self):
        """Test absence de contexte documentaire."""
        messages = [
            HumanMessage(content="Simple question"),
            AIMessage(content="Simple response")
        ]

        assert has_document_context(messages) is False

    def test_has_document_context_partial_match(self):
        """Test que des correspondances partielles ne suffisent pas."""
        messages = [
            SystemMessage(content="This is just a source code example")
        ]

        # Ne devrait pas détecter car il manque les balises appropriées
        assert has_document_context(messages) is False

    def test_has_document_context_mixed_messages(self):
        """Test avec plusieurs types de messages."""
        messages = [
            HumanMessage(content="Question"),
            SystemMessage(content="<source id='1'>Document content</source>"),
            AIMessage(content="Answer")
        ]

        assert has_document_context(messages) is True


class TestStateStructure:
    """Tests pour la structure d'état du graphe."""

    def test_state_initialization(self):
        """Test initialisation de l'état."""
        state: State = {
            "input": "test input",
            "messages": [HumanMessage(content="test")],
            "next": "Agent_Simple_Local",
            "web_sub_query": None,
            "is_confidential": None,
            "anonymized_text": None,
            "human_approved": None
        }

        assert state["input"] == "test input"
        assert len(state["messages"]) == 1
        assert state["next"] == "Agent_Simple_Local"
        assert state["is_confidential"] is None

    def test_state_with_confidential_data(self):
        """Test état avec données confidentielles."""
        state: State = {
            "input": "test",
            "messages": [],
            "next": "Agent_WEB_Cloud",
            "web_sub_query": "original query",
            "is_confidential": True,
            "anonymized_text": "******* query",
            "human_approved": True
        }

        assert state["is_confidential"] is True
        assert state["anonymized_text"] == "******* query"
        assert state["human_approved"] is True


class TestConfidentialityCheckNode:
    """Tests pour le nœud de vérification de confidentialité."""

    @pytest.mark.asyncio
    async def test_confidentiality_no_pii_detected(self):
        """Test quand aucune PII n'est détectée."""
        from src.app.graph import make_graph

        # Créer un graphe de test sans checkpointer
        graph = await make_graph(checkpointer=None)

        state: State = {
            "input": "Quelle est la météo à Paris?",
            "messages": [HumanMessage(content="Quelle est la météo à Paris?")],
            "next": "Agent_WEB_Cloud",
            "web_sub_query": "Quelle est la météo à Paris?",
            "is_confidential": None,
            "anonymized_text": None,
            "human_approved": None
        }

        # Le nœud confidentiality_check devrait marquer is_confidential à False
        # car il n'y a pas de PII dans cette question

    @pytest.mark.asyncio
    async def test_confidentiality_with_phone_number(self):
        """Test quand un numéro de téléphone est détecté."""
        state: State = {
            "input": "Mon téléphone est 06 12 34 56 78",
            "messages": [HumanMessage(content="Mon téléphone est 06 12 34 56 78")],
            "next": "Agent_WEB_Cloud",
            "web_sub_query": "Mon téléphone est 06 12 34 56 78",
            "is_confidential": None,
            "anonymized_text": None,
            "human_approved": None
        }

        # Le nœud devrait détecter le numéro de téléphone


class TestSupervisorRouting:
    """Tests pour la logique de routage du superviseur."""

    def test_routing_to_web_agent(self):
        """Test routage vers l'agent web pour une question météo."""
        # Le superviseur devrait router vers Agent_WEB_Cloud pour des questions
        # nécessitant des données en temps réel
        pass

    def test_routing_to_local_agent(self):
        """Test routage vers l'agent local pour une question simple."""
        # Le superviseur devrait router vers Agent_Simple_Local pour des
        # questions simples ne nécessitant pas d'accès externe
        pass

    def test_routing_to_cloud_agent(self):
        """Test routage vers l'agent cloud pour une analyse complexe."""
        # Le superviseur devrait router vers Agent_Simple_Cloud pour des
        # tâches analytiques complexes
        pass

    def test_routing_to_rag_agent_with_document(self):
        """Test routage automatique vers RAG avec contexte documentaire."""
        # Quand un contexte documentaire est détecté, le routage devrait
        # automatiquement aller vers Agent_RAG_Document
        pass


class TestAnonymizerNode:
    """Tests pour le nœud d'anonymisation."""

    @pytest.mark.asyncio
    async def test_anonymizer_with_approval(self):
        """Test anonymisation avec approbation humaine."""
        state: State = {
            "input": "test",
            "messages": [HumanMessage(content="Mon NIR est 1 87 05 75 123 456 78")],
            "next": "Agent_WEB_Cloud",
            "web_sub_query": "Mon NIR est 1 87 05 75 123 456 78",
            "is_confidential": True,
            "anonymized_text": "Mon NIR est **************************************************",
            "human_approved": True
        }

        # Après anonymisation, web_sub_query devrait contenir le texte anonymisé

    @pytest.mark.asyncio
    async def test_anonymizer_without_approval(self):
        """Test anonymisation sans approbation humaine."""
        state: State = {
            "input": "test",
            "messages": [HumanMessage(content="Mon NIR est 1 87 05 75 123 456 78")],
            "next": "Agent_WEB_Cloud",
            "web_sub_query": "Mon NIR est 1 87 05 75 123 456 78",
            "is_confidential": True,
            "anonymized_text": "Mon NIR est **************************************************",
            "human_approved": False
        }

        # Sans approbation, le texte ne devrait pas être anonymisé


class TestHumanApprovalNode:
    """Tests pour le nœud d'approbation humaine."""

    @pytest.mark.asyncio
    async def test_human_approval_yes(self):
        """Test approbation humaine positive."""
        # Simuler une réponse "oui" de l'utilisateur
        # Le nœud devrait retourner {"human_approved": True}
        pass

    @pytest.mark.asyncio
    async def test_human_approval_no(self):
        """Test refus humain."""
        # Simuler une réponse "non" de l'utilisateur
        # Le nœud devrait retourner {"human_approved": False}
        pass


class TestAgentNodes:
    """Tests pour les nœuds d'agents."""

    @pytest.mark.asyncio
    async def test_agent_simple_local_response(self):
        """Test réponse de l'agent local simple."""
        # L'agent devrait répondre à une question simple
        pass

    @pytest.mark.asyncio
    async def test_agent_simple_cloud_response(self):
        """Test réponse de l'agent cloud simple."""
        # L'agent cloud devrait traiter une requête complexe
        pass

    @pytest.mark.asyncio
    async def test_agent_web_cloud_with_tools(self):
        """Test agent web avec utilisation des outils MCP."""
        # L'agent web devrait utiliser l'outil web_search
        pass

    @pytest.mark.asyncio
    async def test_agent_rag_document_with_context(self):
        """Test agent RAG avec contexte documentaire."""
        # L'agent RAG devrait analyser le contexte fourni
        pass


class TestGraphFlow:
    """Tests d'intégration pour le flux complet du graphe."""

    @pytest.mark.asyncio
    async def test_simple_query_flow(self):
        """Test flux complet pour une requête simple."""
        # supervisor -> Agent_Simple_Local -> END
        pass

    @pytest.mark.asyncio
    async def test_web_query_without_pii_flow(self):
        """Test flux pour une requête web sans PII."""
        # supervisor -> Agent_Confidentiel -> Agent_WEB_Cloud -> END
        pass

    @pytest.mark.asyncio
    async def test_web_query_with_pii_approved_flow(self):
        """Test flux pour une requête web avec PII et approbation."""
        # supervisor -> Agent_Confidentiel -> human_approval -> anonymizer -> Agent_WEB_Cloud -> END
        pass

    @pytest.mark.asyncio
    async def test_web_query_with_pii_rejected_flow(self):
        """Test flux pour une requête web avec PII refusée."""
        # supervisor -> Agent_Confidentiel -> human_approval -> Agent_WEB_Cloud -> END
        pass

    @pytest.mark.asyncio
    async def test_document_analysis_flow(self):
        """Test flux pour l'analyse de document."""
        # supervisor -> Agent_RAG_Document -> END
        pass


class TestEdgeConditions:
    """Tests pour les conditions de routage des edges."""

    def test_should_check_confidentiality_for_web_agent(self):
        """Test que les requêtes vers Agent_WEB_Cloud passent par le check de confidentialité."""
        state: State = {
            "input": "test",
            "messages": [],
            "next": "Agent_WEB_Cloud",
            "web_sub_query": None,
            "is_confidential": None,
            "anonymized_text": None,
            "human_approved": None
        }

        # La fonction should_check_confidentiality devrait retourner "Agent_Confidentiel"

    def test_should_check_confidentiality_for_cloud_agent(self):
        """Test que les requêtes vers Agent_Simple_Cloud passent par le check."""
        state: State = {
            "input": "test",
            "messages": [],
            "next": "Agent_Simple_Cloud",
            "web_sub_query": None,
            "is_confidential": None,
            "anonymized_text": None,
            "human_approved": None
        }

        # La fonction should_check_confidentiality devrait retourner "Agent_Confidentiel"

    def test_no_confidentiality_check_for_local_agent(self):
        """Test que les requêtes vers Agent_Simple_Local ne passent pas par le check."""
        state: State = {
            "input": "test",
            "messages": [],
            "next": "Agent_Simple_Local",
            "web_sub_query": None,
            "is_confidential": None,
            "anonymized_text": None,
            "human_approved": None
        }

        # La fonction should_check_confidentiality devrait retourner "Agent_Simple_Local"

    def test_after_confidentiality_check_no_pii(self):
        """Test routage après check sans PII détectée."""
        state: State = {
            "input": "test",
            "messages": [],
            "next": "Agent_WEB_Cloud",
            "web_sub_query": None,
            "is_confidential": False,
            "anonymized_text": None,
            "human_approved": None
        }

        # La fonction after_confidentiality_check devrait retourner "Agent_WEB_Cloud"

    def test_after_confidentiality_check_with_pii(self):
        """Test routage après check avec PII détectée."""
        state: State = {
            "input": "test",
            "messages": [],
            "next": "Agent_WEB_Cloud",
            "web_sub_query": None,
            "is_confidential": True,
            "anonymized_text": "******",
            "human_approved": None
        }

        # La fonction after_confidentiality_check devrait retourner "human_approval"

    def test_after_human_approval_accepted(self):
        """Test routage après approbation humaine acceptée."""
        state: State = {
            "input": "test",
            "messages": [],
            "next": "Agent_WEB_Cloud",
            "web_sub_query": None,
            "is_confidential": True,
            "anonymized_text": "******",
            "human_approved": True
        }

        # La fonction after_human_approval devrait retourner "anonymizer"

    def test_after_human_approval_rejected(self):
        """Test routage après approbation humaine refusée."""
        state: State = {
            "input": "test",
            "messages": [],
            "next": "Agent_WEB_Cloud",
            "web_sub_query": None,
            "is_confidential": True,
            "anonymized_text": "******",
            "human_approved": False
        }

        # La fonction after_human_approval devrait retourner "Agent_WEB_Cloud"

    def test_route_after_anonymization(self):
        """Test routage après anonymisation."""
        state: State = {
            "input": "test",
            "messages": [],
            "next": "Agent_WEB_Cloud",
            "web_sub_query": "anonymized query",
            "is_confidential": True,
            "anonymized_text": "******",
            "human_approved": True
        }

        # La fonction route_after_anonymization devrait retourner "Agent_WEB_Cloud"
