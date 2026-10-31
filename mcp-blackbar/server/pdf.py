
from __future__ import annotations

import os
from pathlib import Path

import fitz
from PIL import Image

from server.faces import mask_faces
from server.localize import localize_replacements
from server.model import select_elements
from server.parser import parse_document
from server.redact import redact_regions


def process_pdf(
    workspace,
    volume_path: str,
    source: str,
    output: str,
    instruction: str,
    endpoint: str,
    mask_people: bool = True,
) -> dict:
    """
    Permanently redact a PDF.

    OCR: Databricks ai_parse_document v2.
    Classification: Claude.
    Rendering: PyMuPDF.
    Faces: YuNet + Haar.

    Output is an image-only PDF.
    """

    maximum_pages = int(
        os.getenv(
            "BLACKBAR_MAX_PAGES",
            "50",
        )
    )

    dpi = int(
        os.getenv(
            "BLACKBAR_RENDER_DPI",
            "150",
        )
    )

    units = os.getenv(
        "BLACKBAR_BBOX_UNITS",
        "pixel",
    )

    if dpi <= 0:
        raise ValueError("Invalid render DPI")

    if units not in ("pixel", "normalized"):
        raise ValueError("Unsupported bbox units")

    parsed = parse_document(
        volume_path=volume_path,
        client=workspace,
    )

    elements = [
        {
            "element_id": str(element.element_id),
            "text": element.text,
            "page": element.page,
            "element_type": element.element_type,
        }
        for element in parsed.elements
    ]

    selections = select_elements(
        elements=elements,
        instruction=instruction,
        endpoint=endpoint,
        client=workspace,
    )

    localization = localize_replacements(
        parsed.elements,
        selections,
    )

    if localization.unresolved:
        return {
            "status": "review_required",
            "output_ready": False,
            "reason": "unresolved_pdf_locations",
            "unresolved_count": len(
                localization.unresolved
            ),
        }

    source_document = fitz.open(source)
    output_document = fitz.open()

    total_regions = 0
    total_faces = 0

    try:
        if source_document.page_count == 0:
            raise ValueError("Empty PDF")

        if source_document.page_count > maximum_pages:
            raise ValueError(
                f"PDF exceeds {maximum_pages} pages"
            )

        for page_number in range(
            source_document.page_count
        ):
            source_page = source_document[
                page_number
            ]

            pixmap = source_page.get_pixmap(
                matrix=fitz.Matrix(
                    dpi / 72,
                    dpi / 72,
                ),
                alpha=False,
            )

            image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )

            page_regions = [
                region
                for region in localization.replacements
                if region.page == page_number + 1
            ]

            boxes = []

            for region in page_regions:
                x0, y0, x1, y1 = region.bbox

                if units == "normalized":
                    x0 *= image.width
                    x1 *= image.width
                    y0 *= image.height
                    y1 *= image.height

                elif units == "pixel":
                    # Coordinates are expected to be in
                    # the same pixel space as the rendered
                    # page. Do not silently rescale.
                    pass

                if (
                    x0 < 0
                    or y0 < 0
                    or x1 > image.width
                    or y1 > image.height
                    or x1 <= x0
                    or y1 <= y0
                ):
                    raise ValueError(
                        "PDF bbox outside rendered page"
                    )

                boxes.append(
                    (x0, y0, x1, y1)
                )

            image = redact_regions(
                image,
                boxes,
                padding=3,
            )

            total_regions += len(boxes)

            if mask_people:
                image, faces, diagnostics = mask_faces(
                    image,
                    return_diagnostics=True,
                )

                if not diagnostics.get(
                    "yunet_available",
                    False,
                ):
                    raise RuntimeError(
                        "YuNet unavailable"
                    )

                total_faces += len(faces)

            # Encode the fully rendered and masked page.
            # Original PDF text and vector content are
            # not copied into the destination PDF.
            import io

            buffer = io.BytesIO()
            image.save(
                buffer,
                format="PNG",
            )

            new_page = output_document.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height,
            )

            new_page.insert_image(
                new_page.rect,
                stream=buffer.getvalue(),
            )

        if total_regions == 0 and total_faces == 0:
            return {
                "status": "review_required",
                "output_ready": False,
                "reason": "no_pdf_redaction_regions",
            }

        Path(output).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_document.save(
            output,
            garbage=4,
            deflate=True,
        )

    finally:
        output_document.close()
        source_document.close()

    return {
        "status": "completed",
        "output_ready": True,
        "media_type": "document",
        "pages_processed": page_number + 1,
        "text_regions": total_regions,
        "faces_detected": total_faces,
        "manual_review_required": True,
    }
