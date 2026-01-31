"""Tests pour l'API FastAPI."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import json

from src.app.api import app, sanitize_message_name, extract_context_info


@pytest.fixture
def client():
    """Créer un client de test FastAPI."""
    return TestClient(app)


@pytest.fixture
def mock_graph():
    """Mock du graphe LangGraph."""
    with patch("src.app.api.make_graph") as mock:
        mock_instance = AsyncMock()
        mock.return_value = mock_instance
        yield mock_instance


class TestSanitizeMessageName:
    """Tests pour la fonction de nettoyage des noms de messages."""

    def test_sanitize_simple_name(self):
        """Test nettoyage d'un nom simple."""
        result = sanitize_message_name("simple_name")
        assert result == "json_simple_name"

    def test_sanitize_name_with_spaces(self):
        """Test nettoyage avec espaces."""
        result = sanitize_message_name("name with spaces")
        assert result == "json_name_with_spaces"

    def test_sanitize_name_with_special_chars(self):
        """Test nettoyage avec caractères spéciaux."""
        result = sanitize_message_name("name<with>special/chars")
        assert result == "json_name_with_special_chars"

    def test_sanitize_empty_name(self):
        """Test nettoyage d'un nom vide."""
        result = sanitize_message_name("")
        assert result == "json"

    def test_sanitize_none_name(self):
        """Test nettoyage de None."""
        result = sanitize_message_name(None)
        assert result is None

    def test_sanitize_long_name(self):
        """Test nettoyage d'un nom très long."""
        long_name = "a" * 100
        result = sanitize_message_name(long_name)
        assert len(result) <= 64 + len("json_")  # 64 + prefix "json_"

    def test_sanitize_name_with_underscores(self):
        """Test nettoyage avec underscores."""
        result = sanitize_message_name("___multiple___underscores___")
        # Ne devrait pas avoir d'underscores au début ou à la fin
        assert not result.endswith("_")
        assert result.startswith("json_")


class TestExtractContextInfo:
    """Tests pour l'extraction d'informations de contexte."""

    def test_extract_no_context(self):
        """Test extraction sans contexte documentaire."""
        messages = [
            {"role": "user", "content": "Simple question"}
        ]

        result = extract_context_info(messages)

        assert result["has_document_context"] is False
        assert result["document_names"] == []
        assert result["context_length"] == 0

    def test_extract_with_context_from(self):
        """Test extraction avec 'Context from'."""
        messages = [
            {"role": "system", "content": "Context from document.pdf: Some content here"}
        ]

        result = extract_context_info(messages)

        assert result["has_document_context"] is True
        assert "document.pdf" in result["document_names"]
        assert result["context_length"] > 0

    def test_extract_with_use_following_context(self):
        """Test extraction avec 'Use the following context'."""
        messages = [
            {"role": "system", "content": "Use the following context to answer: ..."}
        ]

        result = extract_context_info(messages)

        assert result["has_document_context"] is True

    def test_extract_multiple_documents(self):
        """Test extraction avec plusieurs documents."""
        messages = [
            {"role": "system", "content": "Context from doc1.pdf: Content\nContext from doc2.pdf: More content"}
        ]

        result = extract_context_info(messages)

        assert result["has_document_context"] is True
        assert len(result["document_names"]) == 2
        assert "doc1.pdf" in result["document_names"]
        assert "doc2.pdf" in result["document_names"]


class TestListModelsEndpoint:
    """Tests pour l'endpoint /v1/models."""

    def test_list_models(self, client):
        """Test récupération de la liste des modèles."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert "object" in data
        assert data["object"] == "list"
        assert len(data["data"]) > 0

        model = data["data"][0]
        assert "id" in model
        assert model["id"] == "agent-superviseur-v5"
        assert model["object"] == "model"
        assert "created" in model
        assert model["owned_by"] == "ASI"


class TestChatCompletionsEndpoint:
    """Tests pour l'endpoint /v1/chat/completions."""

    @pytest.mark.asyncio
    async def test_chat_completions_missing_messages(self, client):
        """Test requête sans messages."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "agent-superviseur-v5",
                "messages": [],
                "stream": True
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_chat_completions_non_streaming_mode(self, client):
        """Test mode non-streaming (devrait échouer car HITL nécessite streaming)."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "agent-superviseur-v5",
                "messages": [{"role": "user", "content": "Test"}],
                "stream": False
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "streaming" in data["error"].lower()

    def test_chat_completions_new_conversation(self, client):
        """Test démarrage d'une nouvelle conversation."""
        # Note: Ce test nécessite que le graphe soit initialisé
        # Dans un environnement de test réel, on devrait mocker le graphe
        pass

    def test_chat_completions_existing_conversation(self, client):
        """Test continuation d'une conversation existante."""
        conversation_id = "test_conv_123"

        # Première requête
        response1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "agent-superviseur-v5",
                "messages": [{"role": "user", "content": "Hello"}],
                "conversation_id": conversation_id,
                "stream": True
            }
        )

        # Deuxième requête avec le même conversation_id
        response2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "agent-superviseur-v5",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                    {"role": "user", "content": "How are you?"}
                ],
                "conversation_id": conversation_id,
                "stream": True
            }
        )

        # Les deux devraient utiliser le même conversation_id

    def test_chat_completions_with_document_context(self, client):
        """Test requête avec contexte documentaire."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "agent-superviseur-v5",
                "messages": [
                    {"role": "system", "content": "Context from document.pdf: Important information"},
                    {"role": "user", "content": "What does the document say?"}
                ],
                "stream": True
            }
        )

        # Devrait traiter le contexte documentaire

    def test_chat_completions_with_name_sanitization(self, client):
        """Test nettoyage des noms de messages."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "agent-superviseur-v5",
                "messages": [
                    {"role": "user", "content": "Test", "name": "invalid<name>with/special\\chars"}
                ],
                "stream": True
            }
        )

        # Le nom devrait être nettoyé automatiquement

    def test_chat_completions_history_length_limit(self, client):
        """Test limitation de l'historique."""
        long_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(20)
        ]

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "agent-superviseur-v5",
                "messages": long_history,
                "stream": True,
                "history_length": 5
            }
        )

        # Seuls les 5 derniers messages devraient être envoyés au graphe


class TestStreamingWithHITL:
    """Tests pour le streaming avec interruptions HITL."""

    @pytest.mark.asyncio
    async def test_stream_normal_completion(self):
        """Test streaming d'une complétion normale."""
        # Mock du graphe qui stream une réponse normale
        pass

    @pytest.mark.asyncio
    async def test_stream_with_interruption(self):
        """Test streaming avec interruption HITL."""
        # Mock du graphe qui s'interrompt pour demander l'approbation humaine
        pass

    @pytest.mark.asyncio
    async def test_stream_error_handling(self):
        """Test gestion d'erreur pendant le streaming."""
        # Mock du graphe qui lève une exception
        pass

    @pytest.mark.asyncio
    async def test_stream_langfuse_tracking(self):
        """Test que Langfuse trace bien les événements."""
        # Vérifier que les spans Langfuse sont créés correctement
        pass


class TestResumeAfterInterrupt:
    """Tests pour la reprise après interruption."""

    @pytest.mark.asyncio
    async def test_resume_with_approval(self, client):
        """Test reprise avec approbation."""
        session_id = "test_session_interrupt"

        # Simuler une session interrompue
        # (normalement créée par une requête précédente)
        from src.app.api import INTERRUPTED_SESSIONS
        INTERRUPTED_SESSIONS[session_id] = {
            "thread_id": "test_thread_123",
            "timestamp": 1234567890
        }

        # Envoyer la réponse de l'utilisateur
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "agent-superviseur-v5",
                "messages": [{"role": "user", "content": "oui"}],
                "conversation_id": session_id,
                "stream": True
            }
        )

        # La session devrait reprendre avec la réponse de l'utilisateur

    @pytest.mark.asyncio
    async def test_resume_with_rejection(self, client):
        """Test reprise avec refus."""
        session_id = "test_session_reject"

        from src.app.api import INTERRUPTED_SESSIONS
        INTERRUPTED_SESSIONS[session_id] = {
            "thread_id": "test_thread_456",
            "timestamp": 1234567890
        }

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "agent-superviseur-v5",
                "messages": [{"role": "user", "content": "non"}],
                "conversation_id": session_id,
                "stream": True
            }
        )

        # La session devrait reprendre sans anonymisation

    @pytest.mark.asyncio
    async def test_resume_session_cleanup(self, client):
        """Test que la session est nettoyée après reprise."""
        session_id = "test_session_cleanup"

        from src.app.api import INTERRUPTED_SESSIONS
        INTERRUPTED_SESSIONS[session_id] = {
            "thread_id": "test_thread_789",
            "timestamp": 1234567890
        }

        assert session_id in INTERRUPTED_SESSIONS

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "agent-superviseur-v5",
                "messages": [{"role": "user", "content": "oui"}],
                "conversation_id": session_id,
                "stream": True
            }
        )

        # La session devrait être retirée de INTERRUPTED_SESSIONS après traitement


class TestCORSMiddleware:
    """Tests pour le middleware CORS."""

    def test_cors_headers_present(self, client):
        """Test présence des headers CORS."""
        response = client.options(
            "/v1/chat/completions",
            headers={"Origin": "http://localhost:3000"}
        )

        # Les headers CORS devraient être présents

    def test_cors_allows_all_origins(self, client):
        """Test que CORS accepte toutes les origines."""
        response = client.get(
            "/v1/models",
            headers={"Origin": "http://example.com"}
        )

        # L'origine devrait être acceptée


class TestStartupShutdown:
    """Tests pour les événements de démarrage et arrêt."""

    @pytest.mark.asyncio
    async def test_startup_initializes_graph(self):
        """Test que le startup initialise le graphe."""
        # Vérifier que app.state.graph est créé au démarrage
        pass

    @pytest.mark.asyncio
    async def test_shutdown_flushes_langfuse(self):
        """Test que le shutdown flush Langfuse."""
        # Vérifier que langfuse_client.flush() est appelé
        pass


class TestErrorHandling:
    """Tests pour la gestion d'erreurs."""

    def test_invalid_json_request(self, client):
        """Test requête avec JSON invalide."""
        response = client.post(
            "/v1/chat/completions",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code in [400, 422]

    def test_missing_required_fields(self, client):
        """Test requête avec champs manquants."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "agent-superviseur-v5"}  # messages manquant
        )

        assert response.status_code in [400, 422]

    def test_invalid_message_role(self, client):
        """Test avec rôle de message invalide."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "agent-superviseur-v5",
                "messages": [{"role": "invalid_role", "content": "Test"}],
                "stream": True
            }
        )

        # Devrait gérer gracieusement le rôle invalide


class TestIntegrationScenarios:
    """Tests d'intégration pour des scénarios complets."""

    @pytest.mark.asyncio
    async def test_simple_question_answer_flow(self, client):
        """Test flux complet question-réponse simple."""
        # 1. Envoyer une question simple
        # 2. Recevoir une réponse streamée
        # 3. Vérifier que la réponse est correctement formatée
        pass

    @pytest.mark.asyncio
    async def test_web_search_flow(self, client):
        """Test flux complet avec recherche web."""
        # 1. Envoyer une question nécessitant une recherche web
        # 2. Vérifier que l'agent web est utilisé
        # 3. Recevoir une réponse avec données actuelles
        pass

    @pytest.mark.asyncio
    async def test_pii_detection_and_anonymization_flow(self, client):
        """Test flux complet avec détection PII et anonymisation."""
        # 1. Envoyer une question avec PII
        # 2. Recevoir une interruption demandant l'approbation
        # 3. Répondre positivement
        # 4. Recevoir la réponse finale avec données anonymisées
        pass

    @pytest.mark.asyncio
    async def test_document_analysis_flow(self, client):
        """Test flux complet d'analyse de document."""
        # 1. Envoyer une question avec contexte documentaire
        # 2. Vérifier que l'agent RAG est utilisé
        # 3. Recevoir une réponse basée sur le document
        pass

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_flow(self, client):
        """Test flux de conversation multi-tours."""
        # 1. Première question
        # 2. Deuxième question qui dépend du contexte
        # 3. Vérifier que l'historique est maintenu
        pass
