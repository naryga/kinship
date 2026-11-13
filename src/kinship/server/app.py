"""The Kinship MCP server as an ASGI app.

- `POST /mcp`: MCP over Streamable HTTP (stateless, JSON responses). Requires a
  bearer token from `KINSHIP_TOKENS`.
- `GET /healthz`: unauthenticated liveness check that also pings the database.

Stateless mode means every HTTP request is self-contained: there is no
server-side MCP session to pin to a machine or to lose on restart, which suits
a small hosted app with read-only tools.
"""

from __future__ import annotations

import logging

import anyio.to_thread
from mcp.server.mcpserver import MCPServer
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from kinship import __version__
from kinship.config import Settings, get_settings
from kinship.rag.embedder import Embedder, EmbedderConfigError, get_embedder
from kinship.repo import SessionFactory, database_is_up, default_session_factory
from kinship.server import prompts, resources, tools
from kinship.server.auth import BearerAuthMiddleware
from kinship.server.context import OwnerContextMiddleware

MCP_PATH = "/mcp"
HEALTH_PATH = "/healthz"

INSTRUCTIONS = (
    "Kinship serves the user's own LinkedIn network: their connections, companies, "
    "messages, positions and recommendations. All tools are read-only."
)


logger = logging.getLogger(__name__)

FROM_SETTINGS = "from-settings"
"""`create_app(embedder=FROM_SETTINGS)`: build the query embedder from configuration."""


def configured_embedder(settings: Settings) -> Embedder | None:
    """The query embedder that `settings` describe, or None if they describe none.

    Without `OPENAI_API_KEY` (and without `KINSHIP_EMBEDDER=fake`) the server
    still runs: `semantic_search` then answers from full-text search alone.
    The fake embedder is used only when it is configured by name.
    """
    try:
        return get_embedder(settings)
    except EmbedderConfigError as error:
        logger.warning("No query embedder; semantic_search is full-text only. %s", error)
        return None


def build_mcp(session_factory: SessionFactory, embedder: Embedder | None = None) -> MCPServer:
    """The MCP server with every tool, resource and prompt module registered.

    `embedder` embeds search queries; tools get it from `current_embedder()`.
    """
    mcp = MCPServer(
        "kinship",
        instructions=INSTRUCTIONS,
        version=__version__,
        middleware=[OwnerContextMiddleware(session_factory, embedder)],
    )
    tools.register_all(mcp)
    resources.register_all(mcp)
    prompts.register_all(mcp)

    @mcp.custom_route(HEALTH_PATH, methods=["GET"], include_in_schema=False)
    async def healthz(request: Request) -> Response:
        if await anyio.to_thread.run_sync(database_is_up, session_factory):
            return JSONResponse({"status": "ok"})
        return JSONResponse({"status": "unavailable", "database": "down"}, status_code=503)

    return mcp


def create_app(
    settings: Settings | None = None,
    session_factory: SessionFactory | None = None,
    *,
    host: str = "127.0.0.1",
    embedder: Embedder | str | None = FROM_SETTINGS,
) -> Starlette:
    """Build the ASGI app.

    `session_factory` defaults to the configured `DATABASE_URL`; tests pass one
    bound to their own engine. `host` is the bind address: for a loopback host
    the SDK also enables DNS-rebinding protection (Host/Origin checks).

    `embedder` embeds search queries. By default it is built once, here, from
    `settings` (see `configured_embedder`); pass an `Embedder` to inject one, or
    None for a server without vector search.
    """
    settings = settings or get_settings()
    session_factory = session_factory or default_session_factory()
    if isinstance(embedder, str):
        if embedder != FROM_SETTINGS:
            raise ValueError("embedder must be an Embedder, None or FROM_SETTINGS")
        embedder = configured_embedder(settings)

    mcp = build_mcp(session_factory, embedder)
    app = mcp.streamable_http_app(
        streamable_http_path=MCP_PATH,
        json_response=True,
        stateless_http=True,
        host=host,
    )
    app.add_middleware(BearerAuthMiddleware, tokens=settings.token_owner_ids)
    return app
