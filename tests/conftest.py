"""Configuration pytest et fixtures partagées pour tous les tests."""
import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Generator

# Configuration pytest
def pytest_configure(config):
    """Configuration globale de pytest."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as an async test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Créer une boucle d'événements pour les tests async."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_chat_history():
    """Reset l'historique des chats entre chaque test."""
    from src.app.graph import chats_by_session_id
    chats_by_session_id.clear()
    yield
    chats_by_session_id.clear()


@pytest.fixture(autouse=True)
def reset_interrupted_sessions():
    """Reset les sessions interrompues entre chaque test."""
    try:
        from src.app.api import INTERRUPTED_SESSIONS
        INTERRUPTED_SESSIONS.clear()
        yield
        INTERRUPTED_SESSIONS.clear()
    except ImportError:
        # Si l'import échoue (dans certains tests), ignorer
        yield


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock des variables d'environnement pour les tests."""
    env_vars = {
        "MISTRAL_API_KEY": "test_mistral_key",
        "GOOGLE_API_KEY": "test_google_key",
        "SERPER_API_KEY": "test_serper_key",
        "LANGSMITH_API_KEY": "test_langsmith_key",
        "LANGFUSE_SECRET_KEY": "test_langfuse_secret",
        "LANGFUSE_PUBLIC_KEY": "test_langfuse_public",
        "LANGFUSE_HOST": "http://localhost:3001",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def sample_messages():
    """Fixture avec des messages d'exemple pour les tests."""
    return [
        {"role": "user", "content": "Bonjour!"},
        {"role": "assistant", "content": "Bonjour! Comment puis-je vous aider?"},
        {"role": "user", "content": "Quelle est la météo à Paris?"},
    ]


@pytest.fixture
def sample_pii_messages():
    """Fixture avec des messages contenant des PII."""
    return [
        {
            "role": "user",
            "content": "Mon numéro de téléphone est 06 12 34 56 78 et mon email est test@example.com"
        }
    ]


@pytest.fixture
def sample_document_context():
    """Fixture avec un contexte documentaire."""
    return {
        "role": "system",
        "content": """<source id='doc1'>
        <context>
        Ceci est un document de test contenant des informations importantes.
        Le projet MAS est un système multi-agent qui utilise LangGraph.
        </context>
        </source>"""
    }


@pytest.fixture
def mock_ollama_response():
    """Mock d'une réponse Ollama."""
    mock_response = Mock()
    mock_response.content = "Ceci est une réponse générée par Ollama."
    return mock_response


@pytest.fixture
def mock_google_response():
    """Mock d'une réponse Google Gemini."""
    mock_response = Mock()
    mock_response.content = "Ceci est une réponse générée par Google Gemini."
    return mock_response


@pytest.fixture
def mock_mcp_tools():
    """Mock des outils MCP."""
    mock_tool = Mock()
    mock_tool.name = "web_search"
    mock_tool.description = "Search the web for information"
    return [mock_tool]


@pytest.fixture
def mock_langfuse_client():
    """Mock du client Langfuse."""
    with pytest.MonkeyPatch.context() as m:
        mock_client = MagicMock()
        mock_client.start_as_current_span = MagicMock()
        m.setattr("src.app.api.langfuse_client", mock_client)
        yield mock_client


@pytest.fixture
def test_session_config():
    """Configuration de session pour les tests."""
    return {
        "configurable": {
            "session_id": "test_session_123",
            "thread_id": "test_thread_456"
        }
    }


@pytest.fixture
async def mock_graph():
    """Mock du graphe LangGraph pour les tests."""
    mock = AsyncMock()
    mock.ainvoke = AsyncMock(return_value={"messages": []})
    mock.astream = AsyncMock()
    mock.astream_events = AsyncMock()
    mock.aget_state = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_analyzer_results_no_pii():
    """Mock de résultats d'analyse Presidio sans PII."""
    return []


@pytest.fixture
def mock_analyzer_results_with_phone():
    """Mock de résultats d'analyse Presidio avec numéro de téléphone."""
    from presidio_analyzer import RecognizerResult

    return [
        RecognizerResult(
            entity_type="PHONE_NUMBER",
            start=27,
            end=41,
            score=0.95
        )
    ]


@pytest.fixture
def mock_analyzer_results_with_nir():
    """Mock de résultats d'analyse Presidio avec NIR."""
    from presidio_analyzer import RecognizerResult

    return [
        RecognizerResult(
            entity_type="FR_SSN",
            start=12,
            end=33,
            score=0.98
        )
    ]


@pytest.fixture
def sample_chat_completion_request():
    """Fixture avec une requête de chat completion valide."""
    return {
        "model": "agent-superviseur-v5",
        "messages": [
            {"role": "user", "content": "Quelle est la météo à Paris?"}
        ],
        "stream": True,
        "temperature": 0.7,
        "history_length": 5
    }


@pytest.fixture
def sample_chat_completion_request_with_pii():
    """Fixture avec une requête contenant des PII."""
    return {
        "model": "agent-superviseur-v5",
        "messages": [
            {"role": "user", "content": "Mon téléphone est 06 12 34 56 78, peux-tu vérifier la météo?"}
        ],
        "stream": True
    }


@pytest.fixture
def sample_streaming_chunks():
    """Fixture avec des chunks de streaming simulés."""
    from langchain_core.messages import AIMessageChunk

    return [
        AIMessageChunk(content="Voici "),
        AIMessageChunk(content="la "),
        AIMessageChunk(content="réponse "),
        AIMessageChunk(content="complète."),
    ]


# Helpers pour les assertions
class AssertionHelpers:
    """Helpers pour faciliter les assertions dans les tests."""

    @staticmethod
    def assert_valid_pii_detection(results, expected_types):
        """Vérifie que les types de PII attendus sont détectés."""
        detected_types = {r.entity_type for r in results}
        for expected_type in expected_types:
            assert expected_type in detected_types, f"Type PII '{expected_type}' non détecté"

    @staticmethod
    def assert_message_format(message, expected_role, expected_content_partial=None):
        """Vérifie le format d'un message."""
        assert "role" in message
        assert message["role"] == expected_role
        assert "content" in message

        if expected_content_partial:
            assert expected_content_partial in message["content"]

    @staticmethod
    def assert_streaming_response_format(chunk):
        """Vérifie le format d'un chunk de streaming."""
        data = chunk
        if isinstance(chunk, str):
            import json
            data = json.loads(chunk.replace("data: ", ""))

        assert "id" in data
        assert "object" in data
        assert data["object"] == "chat.completion.chunk"
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "delta" in data["choices"][0]


@pytest.fixture
def assertion_helpers():
    """Fournit les helpers d'assertion."""
    return AssertionHelpers


# Markers personnalisés
def pytest_collection_modifyitems(config, items):
    """Modifie la collection de tests pour ajouter des markers automatiques."""
    for item in items:
        # Ajouter le marker 'asyncio' aux tests async
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)

        # Ajouter le marker 'slow' aux tests qui utilisent le vrai modèle NLP
        if "analyzer_engine" in item.fixturenames:
            item.add_marker(pytest.mark.slow)

        # Ajouter le marker 'integration' aux tests de scénarios complets
        if "integration" in item.name.lower() or "flow" in item.name.lower():
            item.add_marker(pytest.mark.integration)


# Configuration pour capturer les logs
@pytest.fixture(autouse=True)
def configure_logging(caplog):
    """Configure les logs pour les tests."""
    import logging
    caplog.set_level(logging.INFO)
    return caplog


# Fixture pour nettoyer les ressources async
@pytest.fixture
async def cleanup_tasks():
    """Nettoie les tâches async après chaque test."""
    yield

    # Annuler toutes les tâches en cours
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()

    # Attendre que toutes les tâches soient terminées
    await asyncio.gather(*tasks, return_exceptions=True)


# Mock des dépendances externes
@pytest.fixture
def mock_external_apis(monkeypatch):
    """Mock toutes les APIs externes."""
    # Mock Ollama
    mock_ollama = AsyncMock()
    mock_ollama.ainvoke.return_value = Mock(content="Réponse Ollama mockée")

    # Mock Google
    mock_google = AsyncMock()
    mock_google.ainvoke.return_value = Mock(content="Réponse Google mockée")

    # Mock MCP
    mock_mcp = AsyncMock()
    mock_mcp.get_tools.return_value = [Mock(name="web_search")]

    return {
        "ollama": mock_ollama,
        "google": mock_google,
        "mcp": mock_mcp
    }


# Configuration pour les tests parallèles
def pytest_addoption(parser):
    """Ajoute des options de ligne de commande pour pytest."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Exécuter les tests lents"
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Exécuter les tests d'intégration"
    )


def pytest_runtest_setup(item):
    """Configuration avant l'exécution de chaque test."""
    # Skip les tests lents si --run-slow n'est pas spécifié
    if "slow" in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("Need --run-slow option to run")

    # Skip les tests d'intégration si --run-integration n'est pas spécifié
    if "integration" in item.keywords and not item.config.getoption("--run-integration"):
        pytest.skip("Need --run-integration option to run")
