
from __future__ import annotations

from typing import Any

from databricks.sdk import WorkspaceClient

from server.model import select_elements


DEFAULT_INSTRUCTION = (
    "Mask personal and sensitive information, including "
    "person names, addresses, phone numbers, email addresses, "
    "identification documents, financial identifiers, "
    "medical identifiers and other private information."
)


def normalize_elements(
    elements: list[Any],
) -> list[dict]:
    normalized = []
    seen = set()

    for index, element in enumerate(elements):
        if isinstance(element, dict):
            element_id = element.get(
                "element_id",
                element.get("id", index),
            )
            text = element.get(
                "text",
                element.get("content", ""),
            )
        else:
            element_id = getattr(
                element,
                "element_id",
                index,
            )
            text = getattr(
                element,
                "text",
                "",
            )

        if not isinstance(text, str):
            continue

        if not text.strip():
            continue

        element_id = str(element_id)

        if element_id in seen:
            raise ValueError(
                f"Duplicate element ID: {element_id}"
            )

        seen.add(element_id)

        normalized.append({
            "element_id": element_id,
            "text": text,
        })

    return normalized


def detect_sensitive(
    elements: list[Any],
    instruction: str = DEFAULT_INSTRUCTION,
    endpoint: str = "system.ai.claude-opus-4-8",
    client: WorkspaceClient | None = None,
) -> list[dict]:
    """
    Select sensitive content using the existing Claude client.

    Returns:
        [
            {
                "element_id": "...",
                "original": "...",
                "category": "..."
            }
        ]
    """

    normalized = normalize_elements(elements)

    if not normalized:
        return []

    if not instruction.strip():
        raise ValueError("Instruction cannot be empty")

    selections = select_elements(
        elements=normalized,
        instruction=instruction,
        endpoint=endpoint,
        client=client,
    )

    by_id = {
        element["element_id"]: element["text"]
        for element in normalized
    }

    validated = []
    seen = set()

    for selection in selections:
        element_id = str(
            selection.get("element_id", "")
        )
        original = selection.get("original")

        if element_id not in by_id:
            raise ValueError(
                "Unknown sensitive element ID"
            )

        if (
            not isinstance(original, str)
            or not original
            or original not in by_id[element_id]
        ):
            raise ValueError(
                "Sensitive text could not be verified"
            )

        key = (element_id, original)

        if key in seen:
            continue

        seen.add(key)

        validated.append({
            "element_id": element_id,
            "original": original,
            "category": str(
                selection.get("category", "other")
            ),
        })

    return validated


def sensitive_element_ids(
    selections: list[dict],
) -> set[str]:
    return {
        str(selection["element_id"])
        for selection in selections
    }
