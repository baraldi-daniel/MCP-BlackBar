
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastmcp import FastMCP

from server.tools import register_tools
from server.utils import header_store

logger = logging.getLogger(__name__)


class MCPPathAndHeaderMiddleware:
    """
    Corrige /mcp -> /mcp/ sem HTTP 307.

    Preserva os headers da requisição para
    autenticação delegada do usuário.

    Implementação ASGI compatível com streaming.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict,
        receive: Any,
        send: Any,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope.get("path") == "/mcp":
            scope = dict(scope)
            scope["path"] = "/mcp/"
            scope["raw_path"] = b"/mcp/"

        headers = {
            key.decode("latin-1").lower():
            value.decode("latin-1")
            for key, value in scope.get(
                "headers", []
            )
        }

        context_token = header_store.set(headers)

        try:
            await self.app(scope, receive, send)
        finally:
            header_store.reset(context_token)


mcp_server = FastMCP("BlackBar MCP")

register_tools(mcp_server)

mcp_app = mcp_server.http_app(
    path="/",
    transport="streamable-http",
)

app = FastAPI(
    title="BlackBar MCP",
    lifespan=mcp_app.lifespan,
)

app.add_middleware(
    MCPPathAndHeaderMiddleware
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "BlackBar MCP",
    }


app.mount("/mcp", mcp_app)

logger.info(
    "BlackBar MCP registered at /mcp (no redirect)"
)
