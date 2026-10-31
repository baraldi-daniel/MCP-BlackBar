from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageOps

from server.conversion import save_image
from server.faces import mask_faces
from server.redact import redact_regions


SUPPORTED_IMAGES = {
    ".png", ".jpg", ".jpeg",
    ".webp", ".bmp", ".tif", ".tiff",
}


def _pixel_box(
    bbox,
    width: int,
    height: int,
    units: str,
):
    x0, y0, x1, y1 = (
        float(value)
        for value in bbox
    )

    if units == "normalized":
        x0 *= width
        x1 *= width
        y0 *= height
        y1 *= height

    elif units != "pixel":
        raise ValueError(
            f"Unsupported bbox units: {units}"
        )

    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid bounding box")

    return (x0, y0, x1, y1)


def replace_image(
    input_path: str,
    output_path: str,
    localization,
    mask_people: bool = True,
) -> dict:
    if Path(input_path).suffix.lower() not in SUPPORTED_IMAGES:
        raise ValueError("Unsupported image format")

    with Image.open(input_path) as source:
        image = ImageOps.exif_transpose(
            source
        ).convert("RGB")

    width, height = image.size

    units = os.getenv(
        "BLACKBAR_BBOX_UNITS",
        "pixel",
    )

    boxes = []

    for region in localization.replacements:
        if region.page != 1:
            raise ValueError(
                "Image redaction only supports page 1"
            )

        boxes.append(
            _pixel_box(
                region.bbox,
                width,
                height,
                units,
            )
        )

    result = redact_regions(
        image,
        boxes,
    )

    face_boxes = []

    if mask_people:
        result, face_boxes, diagnostics = mask_faces(
            result,
            return_diagnostics=True,
        )

        if not diagnostics["yunet_available"]:
            raise RuntimeError(
                "YuNet unavailable"
            )

    if not boxes and not face_boxes:
        return {
            "status": "review_required",
            "output_ready": False,
            "reason": "no_redaction_regions",
        }

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_image(
        result,
        output_path,
    )

    return {
        "status": "completed",
        "output_ready": True,
        "media_type": "image",
        "text_regions": len(boxes),
        "faces_detected": len(face_boxes),
        "faces_masked_verified": mask_people,
        "manual_review_required": True,
    }