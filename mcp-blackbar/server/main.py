
from __future__ import annotations

import argparse
import logging
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BlackBar MCP"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(
            os.getenv(
                "DATABRICKS_APP_PORT",
                "8000",
            )
        ),
    )

    args = parser.parse_args()

    log_level = os.getenv(
        "BLACKBAR_LOG_LEVEL",
        "INFO",
    ).lower()

    logging.basicConfig(
        level=log_level.upper()
    )

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=args.port,
        log_level=log_level,
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
