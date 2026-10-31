
from __future__ import annotations

import os
from contextvars import ContextVar

from databricks.sdk import WorkspaceClient


header_store: ContextVar[
    dict[str, str] | None
] = ContextVar(
    "blackbar_request_headers",
    default=None,
)


def get_workspace_client() -> WorkspaceClient:
    """
    Autenticação da aplicação Databricks
    usando a identidade do próprio App.
    """
    return WorkspaceClient()


def get_user_authenticated_workspace_client(
) -> WorkspaceClient:
    """
    Autenticação delegada do usuário.

    No Databricks Apps exige o token encaminhado
    pelo proxy. Nunca substitui silenciosamente
    o usuário pelo service principal.
    """

    if not os.getenv("DATABRICKS_APP_NAME"):
        return WorkspaceClient()

    headers = header_store.get() or {}

    token = headers.get(
        "x-forwarded-access-token", ""
    ).strip()

    if not token:
        raise PermissionError(
            "Missing x-forwarded-access-token "
            "for end-user authentication"
        )

    host = os.getenv(
        "DATABRICKS_HOST", ""
    ).strip()

    if not host:
        host = get_workspace_client().config.host

    if not host:
        raise RuntimeError(
            "Databricks workspace host "
            "is not configured"
        )

    return WorkspaceClient(
        host=host,
        token=token,
    )
