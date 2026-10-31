from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import (
    StatementState,
)


@dataclass
class ParsedElement:
    element_id: str
    text: str
    page: int
    bbox: tuple[float, float, float, float]
    element_type: str = "text"


@dataclass
class ParsedDocument:
    elements: list[ParsedElement] = field(
        default_factory=list
    )


def _quote_sql(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _extract_bbox(item: dict) -> tuple:
    bbox = item.get("bbox")

    if isinstance(bbox, dict):
        if all(
            key in bbox
            for key in ("x0", "y0", "x1", "y1")
        ):
            return tuple(
                float(bbox[key])
                for key in ("x0", "y0", "x1", "y1")
            )

        if all(
            key in bbox
            for key in ("x", "y", "width", "height")
        ):
            x = float(bbox["x"])
            y = float(bbox["y"])
            w = float(bbox["width"])
            h = float(bbox["height"])

            return (x, y, x + w, y + h)

    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return tuple(float(value) for value in bbox)

    raise ValueError("Missing or invalid element bbox")


def _extract_elements(value: Any) -> list[dict]:
    if isinstance(value, str):
        value = json.loads(value)

    if isinstance(value, dict):
        if isinstance(value.get("elements"), list):
            return value["elements"]

        if isinstance(value.get("pages"), list):
            result = []

            for page_index, page in enumerate(
                value["pages"],
                start=1,
            ):
                for item in page.get("elements", []):
                    result.append({
                        **item,
                        "page": item.get(
                            "page",
                            page_index,
                        ),
                    })

            return result

        if isinstance(value.get("document"), dict):
            return _extract_elements(value["document"])

    raise ValueError(
        "Unsupported ai_parse_document response structure"
    )


def _normalize(value: Any) -> ParsedDocument:
    raw_elements = _extract_elements(value)
    elements = []

    for index, item in enumerate(raw_elements):
        if not isinstance(item, dict):
            continue

        text = item.get(
            "text",
            item.get("content", ""),
        )

        if not isinstance(text, str) or not text.strip():
            continue

        bbox = _extract_bbox(item)

        elements.append(
            ParsedElement(
                element_id=str(
                    item.get("id", index)
                ),
                text=text,
                page=int(
                    item.get(
                        "page",
                        item.get("page_number", 1),
                    )
                ),
                bbox=bbox,
                element_type=str(
                    item.get("type", "text")
                ),
            )
        )

    return ParsedDocument(elements=elements)


def parse_document(
    volume_path: str,
    client: WorkspaceClient | None = None,
) -> ParsedDocument:
    from server.storage import validate_volume_path

    validate_volume_path(volume_path)

    if client is None:
        client = WorkspaceClient()

    warehouse_id = os.environ[
        "BLACKBAR_WAREHOUSE_ID"
    ]

    version = os.getenv(
        "BLACKBAR_PARSER_VERSION",
        "2.0",
    )

    timeout = int(
        os.getenv("BLACKBAR_SQL_TIMEOUT", "240")
    )

    if version != "2.0":
        raise ValueError(
            "This implementation requires parser version 2.0"
        )

    # Binary file is loaded from the authorized UC Volume.
    # The parser returns a structured JSON document.
    sql = f"""
    SELECT
      to_json(
        ai_parse_document(
          content,
          map('version', '2.0')
        )
      ) AS parsed
    FROM read_files(
      {_quote_sql(volume_path)},
      format => 'binaryFile'
    )
    """

    execution = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
        wait_timeout="0s",
    )

    statement_id = execution.statement_id

    if not statement_id:
        raise RuntimeError("SQL statement ID missing")

    deadline = time.monotonic() + timeout

    while True:
        status = execution.status
        state = status.state if status else None

        if state == StatementState.SUCCEEDED:
            break

        if state in (
            StatementState.FAILED,
            StatementState.CANCELED,
            StatementState.CLOSED,
        ):
            raise RuntimeError(
                f"SQL parsing failed: {state}"
            )

        if time.monotonic() >= deadline:
            client.statement_execution.cancel_execution(
                statement_id
            )
            raise TimeoutError(
                "ai_parse_document SQL timeout"
            )

        time.sleep(2)

        execution = (
            client.statement_execution.get_statement(
                statement_id
            )
        )

    if execution.result is None:
        execution = (
            client.statement_execution.get_statement(
                statement_id
            )
        )

    result = execution.result

    if result is None or not result.data_array:
        raise RuntimeError(
            "ai_parse_document returned no data"
        )

    if len(result.data_array) != 1:
        raise RuntimeError(
            "Expected exactly one parsed document"
        )

    row = result.data_array[0]

    if not row or not row[0]:
        raise RuntimeError("Empty parser result")

    return _normalize(row[0])