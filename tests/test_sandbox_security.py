"""Tests de sécurité pour le sandbox Docker."""
import pytest
from unittest.mock import patch, MagicMock, Mock


class TestDockerExecutorSecurity:
    """Tests de sécurité pour le DockerCodeExecutor."""

    def test_network_disabled_by_default(self):
        """Test que le réseau est désactivé par défaut."""
        from src.sandbox.docker_executor import DockerCodeExecutor

        with patch('src.sandbox.docker_executor.docker') as mock_docker:
            mock_docker.from_env.return_value = MagicMock()
            mock_docker.from_env.return_value.images.get.return_value = True

            executor = DockerCodeExecutor()
            assert executor._network_disabled is True

    def test_memory_limit(self):
        """Test que la limite mémoire est configurée."""
        from src.sandbox.docker_executor import DockerCodeExecutor

        with patch('src.sandbox.docker_executor.docker') as mock_docker:
            mock_docker.from_env.return_value = MagicMock()
            mock_docker.from_env.return_value.images.get.return_value = True

            executor = DockerCodeExecutor(mem_limit="256m")
            assert executor._mem_limit == "256m"

    def test_cpu_limit(self):
        """Test que la limite CPU est configurée."""
        from src.sandbox.docker_executor import DockerCodeExecutor

        with patch('src.sandbox.docker_executor.docker') as mock_docker:
            mock_docker.from_env.return_value = MagicMock()
            mock_docker.from_env.return_value.images.get.return_value = True

            executor = DockerCodeExecutor(cpu_count=1)
            assert executor._cpu_count == 1

    def test_timeout_configured(self):
        """Test que le timeout est configuré."""
        from src.sandbox.docker_executor import DockerCodeExecutor

        with patch('src.sandbox.docker_executor.docker') as mock_docker:
            mock_docker.from_env.return_value = MagicMock()
            mock_docker.from_env.return_value.images.get.return_value = True

            executor = DockerCodeExecutor(timeout=15)
            assert executor._timeout == 15

    def test_execute_runs_as_sandbox_user(self):
        """Test que le code s'exécute avec l'utilisateur sandbox."""
        from src.sandbox.docker_executor import DockerCodeExecutor

        with patch('src.sandbox.docker_executor.docker') as mock_docker:
            mock_client = MagicMock()
            mock_docker.from_env.return_value = mock_client
            mock_client.images.get.return_value = True

            mock_container = MagicMock()
            mock_container.wait.return_value = {"StatusCode": 0}
            mock_container.logs.return_value = b"output"
            mock_client.containers.run.return_value = mock_container

            executor = DockerCodeExecutor()
            executor.execute("print('hello')", {})

            call_kwargs = mock_client.containers.run.call_args
            assert call_kwargs.kwargs.get("user") == "sandbox" or \
                   (call_kwargs[1] if len(call_kwargs) > 1 else {}).get("user") == "sandbox"

    def test_execute_with_read_only_filesystem(self):
        """Test que le filesystem est en lecture seule."""
        from src.sandbox.docker_executor import DockerCodeExecutor

        with patch('src.sandbox.docker_executor.docker') as mock_docker:
            mock_client = MagicMock()
            mock_docker.from_env.return_value = mock_client
            mock_client.images.get.return_value = True

            mock_container = MagicMock()
            mock_container.wait.return_value = {"StatusCode": 0}
            mock_container.logs.return_value = b"output"
            mock_client.containers.run.return_value = mock_container

            executor = DockerCodeExecutor()
            executor.execute("print('hello')", {})

            call_kwargs = mock_client.containers.run.call_args
            assert call_kwargs.kwargs.get("read_only") is True or \
                   (call_kwargs[1] if len(call_kwargs) > 1 else {}).get("read_only") is True

    def test_container_removed_after_execution(self):
        """Test que le conteneur est supprimé après exécution."""
        from src.sandbox.docker_executor import DockerCodeExecutor

        with patch('src.sandbox.docker_executor.docker') as mock_docker:
            mock_client = MagicMock()
            mock_docker.from_env.return_value = mock_client
            mock_client.images.get.return_value = True

            mock_container = MagicMock()
            mock_container.wait.return_value = {"StatusCode": 0}
            mock_container.logs.return_value = b"output"
            mock_client.containers.run.return_value = mock_container

            executor = DockerCodeExecutor()
            executor.execute("print('hello')", {})

            mock_container.remove.assert_called_once_with(force=True)

    def test_execute_error_returns_error_message(self):
        """Test que les erreurs d'exécution retournent un message d'erreur."""
        from src.sandbox.docker_executor import DockerCodeExecutor

        with patch('src.sandbox.docker_executor.docker') as mock_docker:
            mock_client = MagicMock()
            mock_docker.from_env.return_value = mock_client
            mock_client.images.get.return_value = True

            mock_container = MagicMock()
            mock_container.wait.return_value = {"StatusCode": 1}
            mock_container.logs.side_effect = [
                b"",  # stdout
                b"NameError: name 'x' is not defined",  # stderr
            ]
            mock_client.containers.run.return_value = mock_container

            executor = DockerCodeExecutor()
            output, _ = executor.execute("print(x)", {})

            assert "Erreur" in output or "error" in output.lower()

    def test_build_script_no_code_injection(self):
        """Test que les variables locales ne permettent pas l'injection de code."""
        from src.sandbox.docker_executor import DockerCodeExecutor

        executor = DockerCodeExecutor.__new__(DockerCodeExecutor)
        _locals = {
            "malicious": "'; import os; os.system('rm -rf /')",
        }
        script = executor._build_script("print('safe')", _locals)
        # repr() should properly escape the string
        assert "import os" not in script.split("malicious = ")[0]

    def test_cleanup(self):
        """Test le nettoyage du client Docker."""
        from src.sandbox.docker_executor import DockerCodeExecutor

        with patch('src.sandbox.docker_executor.docker') as mock_docker:
            mock_client = MagicMock()
            mock_docker.from_env.return_value = mock_client
            mock_client.images.get.return_value = True

            executor = DockerCodeExecutor()
            executor.cleanup()
            mock_client.close.assert_called_once()
