from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RedactionRegion:
    page: int
    bbox: tuple[float, float, float, float]
    element_id: str


@dataclass
class LocalizationResult:
    replacements: list[RedactionRegion] = field(
        default_factory=list
    )
    unresolved: list[dict] = field(
        default_factory=list
    )


def _valid_bbox(element: Any):
    values = tuple(
        float(value)
        for value in element.bbox
    )

    if len(values) != 4:
        raise ValueError("Bounding box must contain 4 values")

    x0, y0, x1, y1 = values

    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid bounding box")

    return values


def localize_replacements(
    elements: list,
    selections: list[dict],
) -> LocalizationResult:
    by_id = {
        str(element.element_id): element
        for element in elements
    }

    result = LocalizationResult()
    seen = set()

    for selection in selections:
        element_id = str(
            selection.get("element_id", "")
        )

        original = selection.get("original")
        element = by_id.get(element_id)

        if element is None:
            result.unresolved.append({
                "element_id": element_id,
                "reason": "element_not_found",
            })
            continue

        if (
            not isinstance(original, str)
            or not original
            or original not in element.text
        ):
            result.unresolved.append({
                "element_id": element_id,
                "reason": "substring_not_found",
            })
            continue

        try:
            bbox = _valid_bbox(element)
            page = int(element.page)

            if page < 1:
                raise ValueError("Invalid page")

        except (ValueError, TypeError):
            result.unresolved.append({
                "element_id": element_id,
                "reason": "invalid_geometry",
            })
            continue

        if element_id in seen:
            continue

        seen.add(element_id)

        result.replacements.append(
            RedactionRegion(
                page=page,
                bbox=bbox,
                element_id=element_id,
            )
        )

    return result