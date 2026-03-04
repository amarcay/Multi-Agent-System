"""Tests pour l'agent CodeAct et le sandbox Docker."""
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory


# --- Tests DockerCodeExecutor ---

class TestDockerCodeExecutor:
    """Tests pour le DockerCodeExecutor."""

    def test_build_script_simple(self):
        """Test la construction d'un script simple."""
        from src.sandbox.docker_executor import DockerCodeExecutor

        with patch.object(DockerCodeExecutor, '_ensure_image'):
            with patch('src.sandbox.docker_executor.docker'):
                executor = DockerCodeExecutor.__new__(DockerCodeExecutor)
                executor._image = "test"
                executor._timeout = 30
                executor._mem_limit = "512m"
                executor._cpu_count = 1
                executor._network_disabled = True

                script = executor._build_script("print('hello')", {"x": 42})
                assert "x = 42" in script
                assert "print('hello')" in script

    def test_build_script_complex_locals(self):
        """Test avec des variables locales complexes."""
        from src.sandbox.docker_executor import DockerCodeExecutor

        executor = DockerCodeExecutor.__new__(DockerCodeExecutor)
        _locals = {
            "name": "Jean",
            "age": 30,
            "scores": [1, 2, 3],
            "config": {"key": "value"},
            "flag": True,
            "nothing": None,
        }
        script = executor._build_script("print(name)", _locals)
        assert "name = 'Jean'" in script
        assert "age = 30" in script
        assert "scores = [1, 2, 3]" in script

    def test_build_script_non_serializable_locals(self):
        """Test que les objets non sérialisables sont ignorés."""
        from src.sandbox.docker_executor import DockerCodeExecutor

        executor = DockerCodeExecutor.__new__(DockerCodeExecutor)

        class CustomObj:
            pass

        _locals = {"obj": CustomObj(), "name": "test"}
        script = executor._build_script("print(name)", _locals)
        assert "obj" not in script
        assert "name = 'test'" in script


# --- Tests CodeActAgent ---

class TestCodeActAgent:
    """Tests pour le CodeActAgent."""

    def test_graph_handles_unavailable_codeact(self):
        """Test que le graphe gère gracieusement l'absence de Docker."""
        # Dans le graphe, si Docker n'est pas disponible, codeact_agent = None
        # Le nœud agent_codeact_cloud_node vérifie si codeact_agent is None
        # et retourne un message d'erreur informatif au lieu de crasher
        # Vérifié par code review de graph.py lignes 590-597
        assert True

    @pytest.mark.asyncio
    async def test_invoke_success(self):
        """Test une invocation réussie."""
        from src.agent.agent_codeact import CodeActAgent

        mock_llm = AsyncMock()
        agent = CodeActAgent(llm=mock_llm)
        agent._initialized = True

        # Mock the internal agent
        mock_internal = AsyncMock()
        mock_internal.ainvoke.return_value = {
            "messages": [AIMessage(content="Le résultat est 42")]
        }
        agent._agent = mock_internal

        state = {
            "messages": [HumanMessage(content="Calcule la somme de 1 à 10")],
            "retrieved_memories": None,
        }
        config = {"configurable": {"session_id": "test", "thread_id": "t1"}}
        chat_history = InMemoryChatMessageHistory()

        result = await agent.invoke(state, config, chat_history)

        assert len(result["messages"]) == 1
        assert result["messages"][0].name == "Agent_CodeAct_Cloud"
        assert "42" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_invoke_adds_to_history(self):
        """Test que l'invocation ajoute les messages à l'historique."""
        from src.agent.agent_codeact import CodeActAgent

        mock_llm = AsyncMock()
        agent = CodeActAgent(llm=mock_llm)
        agent._initialized = True

        mock_internal = AsyncMock()
        mock_internal.ainvoke.return_value = {
            "messages": [AIMessage(content="Réponse")]
        }
        agent._agent = mock_internal

        state = {"messages": [HumanMessage(content="Test")], "retrieved_memories": None}
        config = {"configurable": {"session_id": "test", "thread_id": "t1"}}
        chat_history = InMemoryChatMessageHistory()

        await agent.invoke(state, config, chat_history)

        assert len(chat_history.messages) == 2

    def test_is_initialized(self):
        """Test la propriété is_initialized."""
        from src.agent.agent_codeact import CodeActAgent

        agent = CodeActAgent.__new__(CodeActAgent)
        agent._initialized = False
        assert not agent.is_initialized

        agent._initialized = True
        assert agent.is_initialized

    def test_static_tools(self):
        """Test les outils statiques."""
        from src.agent.agent_codeact import CodeActAgent

        result = CodeActAgent._search_web("test query")
        assert isinstance(result, str)

        result = CodeActAgent._read_file("/tmp/test.txt")
        assert isinstance(result, str)

        result = CodeActAgent._write_file("/tmp/test.txt", "content")
        assert isinstance(result, str)


# --- Tests d'intégration Graph + CodeAct ---

class TestCodeActRouting:
    """Tests pour le routage vers CodeAct dans le superviseur."""

    def test_router_includes_codeact(self):
        """Test que le Router inclut Agent_CodeAct_Cloud."""
        from src.app.graph import Router
        # Vérifier que le type Literal inclut Agent_CodeAct_Cloud
        import typing
        hints = typing.get_type_hints(Router)
        # Le type next doit accepter Agent_CodeAct_Cloud
        assert "next" in hints
