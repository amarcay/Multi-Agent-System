"""
Exécuteur de code Python dans un conteneur Docker isolé.
Fournit un sandbox sécurisé pour l'agent CodeAct.
"""

import docker
import tempfile
import os
from typing import Any


class DockerCodeExecutor:
    """Exécute du code Python dans un conteneur Docker isolé."""

    def __init__(
        self,
        image: str = "mas-sandbox",
        timeout: int = 30,
        mem_limit: str = "512m",
        cpu_count: int = 1,
        network_disabled: bool = True,
    ):
        self._image = image
        self._timeout = timeout
        self._mem_limit = mem_limit
        self._cpu_count = cpu_count
        self._network_disabled = network_disabled
        self._client = docker.from_env()
        self._ensure_image()

    def _ensure_image(self) -> None:
        """Vérifie que l'image Docker existe, sinon la construit."""
        try:
            self._client.images.get(self._image)
        except docker.errors.ImageNotFound:
            dockerfile_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "Dockerfile.sandbox",
            )
            if os.path.exists(dockerfile_path):
                print(f"Construction de l'image Docker '{self._image}'...")
                self._client.images.build(
                    path=os.path.dirname(dockerfile_path),
                    dockerfile="Dockerfile.sandbox",
                    tag=self._image,
                )
                print(f"Image '{self._image}' construite avec succès.")
            else:
                raise FileNotFoundError(
                    f"Dockerfile.sandbox introuvable à {dockerfile_path}. "
                    f"Construisez l'image manuellement : docker build -f Dockerfile.sandbox -t {self._image} ."
                )

    def execute(self, code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Exécute du code Python dans un conteneur Docker isolé.

        Cette signature est conforme à la fonction `eval` de langgraph-codeact.

        Args:
            code: Code Python à exécuter
            _locals: Variables locales (non utilisées dans Docker, conservées pour compatibilité)

        Returns:
            Tuple (output_string, variables_dict)
        """
        # Créer un fichier temporaire avec le code
        script_content = self._build_script(code, _locals)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="codeact_"
        ) as f:
            f.write(script_content)
            script_path = f.name

        try:
            container = self._client.containers.run(
                self._image,
                command=["python", "/tmp/script.py"],
                volumes={script_path: {"bind": "/tmp/script.py", "mode": "ro"}},
                mem_limit=self._mem_limit,
                cpu_count=self._cpu_count,
                network_disabled=self._network_disabled,
                read_only=True,
                tmpfs={"/tmp/work": "size=50m"},
                user="sandbox",
                detach=True,
                stderr=True,
                stdout=True,
            )

            try:
                result = container.wait(timeout=self._timeout)
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            except Exception as e:
                container.kill()
                return f"Erreur: Timeout ({self._timeout}s) dépassé ou erreur d'exécution: {e}", _locals
            finally:
                container.remove(force=True)

            exit_code = result.get("StatusCode", -1)
            if exit_code != 0:
                output = f"Erreur (code {exit_code}):\n{stderr}" if stderr else f"Erreur (code {exit_code})"
            else:
                output = stdout.strip()
                if stderr:
                    output += f"\n[stderr]: {stderr.strip()}"

            return output, _locals

        except docker.errors.ContainerError as e:
            return f"Erreur conteneur: {e}", _locals
        except docker.errors.ImageNotFound:
            return f"Image Docker '{self._image}' introuvable. Construisez-la avec: docker build -f Dockerfile.sandbox -t {self._image} .", _locals
        except docker.errors.APIError as e:
            return f"Erreur Docker API: {e}", _locals
        finally:
            os.unlink(script_path)

    def _build_script(self, code: str, _locals: dict[str, Any]) -> str:
        """Construit le script Python à exécuter dans le conteneur."""
        # Injecter les variables locales sérialisables
        preamble_lines = []
        for key, value in _locals.items():
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                preamble_lines.append(f"{key} = {repr(value)}")

        preamble = "\n".join(preamble_lines)
        return f"""{preamble}

{code}
"""

    def cleanup(self) -> None:
        """Nettoyage du client Docker."""
        self._client.close()
