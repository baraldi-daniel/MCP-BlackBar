from __future__ import annotations

from PIL import Image, ImageDraw


def redact_regions(
    image: Image.Image,
    boxes: list[tuple],
    padding: int = 3,
) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)

    width, height = result.size

    for bbox in boxes:
        if len(bbox) != 4:
            raise ValueError("Invalid bounding box")

        x0, y0, x1, y1 = (
            float(value)
            for value in bbox
        )

        if x1 <= x0 or y1 <= y0:
            raise ValueError("Invalid bounding box")

        left = max(0, int(x0) - padding)
        top = max(0, int(y0) - padding)
        right = min(width, int(x1) + padding)
        bottom = min(height, int(y1) + padding)

        draw.rectangle(
            (left, top, right, bottom),
            fill=(0, 0, 0),
        )

    return result