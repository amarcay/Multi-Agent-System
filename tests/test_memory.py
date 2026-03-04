"""Tests pour le système de mémoire persistante."""
import pytest
import asyncio
import tempfile
import shutil
import os
from unittest.mock import AsyncMock, Mock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.memory.memory_manager import MemoryManager
from src.memory.memory_extractor import extract_facts, generate_summary, _format_messages


# --- Fixtures ---

@pytest.fixture
def temp_chroma_dir():
    """Crée un répertoire temporaire pour ChromaDB."""
    tmpdir = tempfile.mkdtemp(prefix="test_chroma_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def memory_manager(temp_chroma_dir):
    """Crée un MemoryManager avec un répertoire temporaire."""
    mm = MemoryManager(persist_dir=temp_chroma_dir)
    yield mm
    mm.cleanup()


@pytest.fixture
def mock_llm():
    """Mock d'un LLM pour les tests d'extraction."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def sample_conversation():
    """Messages d'exemple pour les tests."""
    return [
        HumanMessage(content="Bonjour, je m'appelle Jean et j'habite à Paris."),
        AIMessage(content="Bonjour Jean ! Comment puis-je vous aider aujourd'hui ?"),
        HumanMessage(content="Je travaille comme développeur Python."),
        AIMessage(content="C'est un métier passionnant ! Avez-vous des questions ?"),
    ]


# --- Tests MemoryManager ---

class TestMemoryManager:
    """Tests pour la classe MemoryManager."""

    @pytest.mark.asyncio
    async def test_init(self, memory_manager, temp_chroma_dir):
        """Test l'initialisation du MemoryManager."""
        assert memory_manager._persist_dir == temp_chroma_dir
        assert memory_manager._facts_collection is not None
        assert memory_manager._summaries_collection is not None

    @pytest.mark.asyncio
    async def test_add_fact(self, memory_manager):
        """Test l'ajout d'un fait."""
        await memory_manager.add_fact("session_1", "L'utilisateur s'appelle Jean")
        results = await memory_manager.retrieve_memories("session_1", "comment s'appelle l'utilisateur")
        assert len(results) >= 1
        assert "Jean" in results[0]

    @pytest.mark.asyncio
    async def test_add_facts_batch(self, memory_manager):
        """Test l'ajout de plusieurs faits."""
        facts = [
            "L'utilisateur habite à Paris",
            "L'utilisateur est développeur Python",
            "L'utilisateur aime le café",
        ]
        await memory_manager.add_facts("session_1", facts)
        results = await memory_manager.retrieve_memories("session_1", "où habite l'utilisateur", top_k=3)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_add_facts_empty(self, memory_manager):
        """Test l'ajout d'une liste vide de faits."""
        await memory_manager.add_facts("session_1", [])
        results = await memory_manager.retrieve_memories("session_1", "test")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_add_summary(self, memory_manager):
        """Test l'ajout d'un résumé."""
        await memory_manager.add_summary("session_1", "Discussion sur les habitudes alimentaires de Jean.")
        summaries = await memory_manager.get_recent_summaries("session_1")
        assert len(summaries) == 1
        assert "Jean" in summaries[0]

    @pytest.mark.asyncio
    async def test_retrieve_memories_empty(self, memory_manager):
        """Test la recherche sans données."""
        results = await memory_manager.retrieve_memories("session_1", "test")
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_memories_by_session(self, memory_manager):
        """Test que la recherche est filtrée par session."""
        await memory_manager.add_fact("session_1", "Jean habite à Paris")
        await memory_manager.add_fact("session_2", "Marie habite à Lyon")

        results_1 = await memory_manager.retrieve_memories("session_1", "où habite")
        results_2 = await memory_manager.retrieve_memories("session_2", "où habite")

        assert any("Paris" in r for r in results_1)
        assert any("Lyon" in r for r in results_2)

    @pytest.mark.asyncio
    async def test_retrieve_all_memories(self, memory_manager):
        """Test la recherche cross-session."""
        await memory_manager.add_fact("session_1", "Jean habite à Paris")
        await memory_manager.add_fact("session_2", "Marie habite à Lyon")

        results = await memory_manager.retrieve_all_memories("où habite", top_k=5)
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_get_recent_summaries_empty(self, memory_manager):
        """Test les résumés vides."""
        summaries = await memory_manager.get_recent_summaries("session_1")
        assert summaries == []

    @pytest.mark.asyncio
    async def test_get_recent_summaries_limit(self, memory_manager):
        """Test la limite des résumés."""
        for i in range(5):
            await memory_manager.add_summary("session_1", f"Résumé numéro {i}")

        summaries = await memory_manager.get_recent_summaries("session_1", limit=2)
        assert len(summaries) == 2

    @pytest.mark.asyncio
    async def test_semantic_search_relevance(self, memory_manager):
        """Test que la recherche sémantique retourne des résultats pertinents."""
        await memory_manager.add_fact("session_1", "L'utilisateur préfère le thé vert")
        await memory_manager.add_fact("session_1", "L'utilisateur travaille en informatique")
        await memory_manager.add_fact("session_1", "L'utilisateur a un chat nommé Félix")

        results = await memory_manager.retrieve_memories(
            "session_1", "quelle boisson aime l'utilisateur", top_k=1
        )
        assert len(results) == 1
        assert "thé" in results[0].lower()

    @pytest.mark.asyncio
    async def test_persistence(self, temp_chroma_dir):
        """Test que les données survivent à la recréation du manager."""
        mm1 = MemoryManager(persist_dir=temp_chroma_dir)
        await mm1.add_fact("session_1", "Fait persistant de test")
        mm1.cleanup()

        mm2 = MemoryManager(persist_dir=temp_chroma_dir)
        results = await mm2.retrieve_memories("session_1", "fait persistant")
        mm2.cleanup()

        assert len(results) >= 1
        assert "persistant" in results[0].lower()

    @pytest.mark.asyncio
    async def test_add_fact_with_metadata(self, memory_manager):
        """Test l'ajout d'un fait avec métadonnées."""
        await memory_manager.add_fact(
            "session_1",
            "L'utilisateur parle français",
            metadata={"source": "conversation", "confidence": "high"},
        )
        results = await memory_manager.retrieve_memories("session_1", "langue")
        assert len(results) >= 1


# --- Tests MemoryExtractor ---

class TestMemoryExtractor:
    """Tests pour les fonctions d'extraction de mémoire."""

    def test_format_messages(self, sample_conversation):
        """Test le formatage des messages."""
        result = _format_messages(sample_conversation)
        assert "Utilisateur:" in result
        assert "Assistant:" in result
        assert "Jean" in result
        assert "Paris" in result

    def test_format_messages_ignores_system(self):
        """Test que les messages système sont ignorés."""
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="Hello"),
        ]
        result = _format_messages(messages)
        assert "System prompt" not in result
        assert "Hello" in result

    def test_format_messages_empty(self):
        """Test avec une liste vide."""
        assert _format_messages([]) == ""

    @pytest.mark.asyncio
    async def test_extract_facts_success(self, sample_conversation, mock_llm):
        """Test l'extraction de faits réussie."""
        mock_llm.ainvoke.return_value = Mock(
            content='{"facts": ["Jean habite à Paris", "Jean est développeur Python"]}'
        )
        facts = await extract_facts(sample_conversation, mock_llm)
        assert len(facts) == 2
        assert "Jean habite à Paris" in facts

    @pytest.mark.asyncio
    async def test_extract_facts_json_in_markdown(self, sample_conversation, mock_llm):
        """Test l'extraction avec JSON dans un bloc markdown."""
        mock_llm.ainvoke.return_value = Mock(
            content='```json\n{"facts": ["Fait 1"]}\n```'
        )
        facts = await extract_facts(sample_conversation, mock_llm)
        assert len(facts) == 1

    @pytest.mark.asyncio
    async def test_extract_facts_empty(self, mock_llm):
        """Test avec des messages vides."""
        facts = await extract_facts([], mock_llm)
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_facts_no_relevant(self, sample_conversation, mock_llm):
        """Test quand aucun fait n'est trouvé."""
        mock_llm.ainvoke.return_value = Mock(content='{"facts": []}')
        facts = await extract_facts(sample_conversation, mock_llm)
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_facts_error(self, sample_conversation, mock_llm):
        """Test la gestion d'erreur."""
        mock_llm.ainvoke.side_effect = Exception("LLM error")
        facts = await extract_facts(sample_conversation, mock_llm)
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_facts_invalid_json(self, sample_conversation, mock_llm):
        """Test avec JSON invalide."""
        mock_llm.ainvoke.return_value = Mock(content="pas du json du tout")
        facts = await extract_facts(sample_conversation, mock_llm)
        assert facts == []

    @pytest.mark.asyncio
    async def test_generate_summary_success(self, sample_conversation, mock_llm):
        """Test la génération de résumé réussie."""
        mock_llm.ainvoke.return_value = Mock(
            content="Jean, développeur Python de Paris, s'est présenté."
        )
        summary = await generate_summary(sample_conversation, mock_llm)
        assert "Jean" in summary

    @pytest.mark.asyncio
    async def test_generate_summary_empty(self, mock_llm):
        """Test avec des messages vides."""
        summary = await generate_summary([], mock_llm)
        assert summary == ""

    @pytest.mark.asyncio
    async def test_generate_summary_error(self, sample_conversation, mock_llm):
        """Test la gestion d'erreur."""
        mock_llm.ainvoke.side_effect = Exception("LLM error")
        summary = await generate_summary(sample_conversation, mock_llm)
        assert summary == ""

    @pytest.mark.asyncio
    async def test_generate_summary_only_system_messages(self, mock_llm):
        """Test avec uniquement des messages système."""
        messages = [SystemMessage(content="System only")]
        summary = await generate_summary(messages, mock_llm)
        assert summary == ""
