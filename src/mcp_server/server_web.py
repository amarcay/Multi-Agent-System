import contextlib
import logging
import os
from collections.abc import AsyncIterator

import anyio
import click
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

import mcp.types as types

# --- Configuration initiale ---
load_dotenv()
logger = logging.getLogger(__name__)

# Récupération de la clé API pour la recherche web
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
if not SERPER_API_KEY:
    raise ValueError("La variable d'environnement SERPER_API_KEY n'est pas définie !")

# Initialisation du client de recherche. On le fait une seule fois ici.
search_wrapper = GoogleSerperAPIWrapper()


@click.command()
@click.option("--port", default=8003, help="Port HTTP")
@click.option("--log-level", default="INFO", help="Niveau de log")
def main(port: int, log_level: str):
    """Fonction principale pour lancer le serveur."""

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    app = Server("WEB-Server")

    @app.call_tool()
    async def call_search_tool(name: str, arguments: dict) -> list[types.TextContent]:
        """Exécute l'outil de recherche web lorsqu'il est appelé."""
        if name != "web_search":
            # Sécurité au cas où d'autres outils seraient appelés par erreur
            return [types.TextContent(type="text", text=f"Erreur : Outil inconnu '{name}'.")]

        query = arguments.get("query")
        if not query:
            return [types.TextContent(type="text", text="Erreur : Le paramètre 'query' est manquant.")]
        
        logger.info(f"Tool '{name}' called with query: '{query}'")

        # search_wrapper.run() est une fonction synchrone (bloquante).
        # Dans un environnement async, il est préférable de l'exécuter dans un thread séparé
        # pour ne pas bloquer la boucle d'événements principale.
        try:
            result = await anyio.to_thread.run_sync(search_wrapper.run, query)
            return [types.TextContent(type="text", text=result)]
        except Exception as e:
            error_message = f"Une erreur est survenue lors de la recherche web : {e}"
            logger.error(error_message)
            return [types.TextContent(type="text", text=error_message)]


    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        """Définit et liste les outils disponibles sur ce serveur."""
        return [
            types.Tool(
                name="web_search",
                description="Réaliser une recherche WEB pour trouver des informations ou répondre à une question.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "La recherche web à exécuter."
                        }
                    },
                    "required": ["query"],
                },
            )
        ]

    # --- Configuration du serveur Starlette/Uvicorn (identique au code de l'image) ---
    session_manager = StreamableHTTPSessionManager(
        app=app,
        event_store=None,
        json_response=False,
        stateless=True,
    )

    async def handle_streamable_http(scope: Scope, receive: Receive, send: Send):
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            logger.info(f"Serveur démarré sur http://0.0.0.0:{port}")
            try:
                yield
            finally:
                logger.info("Arrêt du serveur...")

    starlette_app = Starlette(
        debug=True,
        routes=[Mount("/mcp", app=handle_streamable_http)],
        lifespan=lifespan,
    )

    import uvicorn
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
