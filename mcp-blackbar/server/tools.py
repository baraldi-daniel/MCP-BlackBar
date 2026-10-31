
"""
BlackBar MCP tools.

Supported media:
- Images
- Audio
- Video
- PDF documents

Processing:
- Databricks ai_parse_document v2
- Claude via Databricks AI Gateway
- Whisper transcription
- YuNet + Haar face detection
- FFmpeg conversion

Storage:
- Unity Catalog Volumes
- Original files are never overwritten

No dependency on server.validation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

from databricks.sdk import WorkspaceClient

from server.storage import (
    validate_volume_path,
    destination,
    download,
    upload,
)

from server.conversion import (
    IMAGES,
    AUDIOS,
    VIDEOS,
    DOCUMENTS,
    kind,
    validate_conversion,
)

from server.parser import parse_document
from server.model import select_elements
from server.localize import localize_replacements
from server.media import replace_image
from server.audio import process_audio
from server.video import process_video
from server.pdf import process_pdf


logger = logging.getLogger("blackbar")

DEFAULT_MODEL = "system.ai.claude-opus-4-8"

SUPPORTED = IMAGES | AUDIOS | VIDEOS | DOCUMENTS


class ProcessingStageError(RuntimeError):
    """Wrap an exception raised during a processing stage."""

    def __init__(
        self,
        stage: str,
        cause: Exception,
    ):
        self.stage = stage
        self.original_error_type = type(cause).__name__
        super().__init__(str(cause))


def _run_stage(
    stage: str,
    operation: Callable[[], Any],
) -> Any:
    logger.info("Stage started: %s", stage)

    try:
        result = operation()

        logger.info(
            "Stage completed: %s",
            stage,
        )

        return result

    except Exception as exc:
        logger.exception(
            "Stage failed: %s",
            stage,
        )

        raise ProcessingStageError(
            stage,
            exc,
        ) from exc


def _public_report(
    result: dict,
    source_path: str,
    output_path: str | None = None,
) -> dict:
    """
    Return safe structured metadata.

    Sensitive text, transcripts and local paths
    are intentionally excluded.
    """

    if not isinstance(result, dict):
        raise TypeError("Invalid processing result")

    ready = bool(
        result.get("output_ready", False)
    )

    response = {
        "status": result.get(
            "status",
            "completed" if ready else "review_required",
        ),
        "output_ready": ready,
        "source_path": source_path,
        "output_path": output_path if ready else None,
        "media_type": result.get("media_type"),
        "manual_review_required": result.get(
            "manual_review_required",
            True,
        ),
    }

    allowed_fields = (
        "reason",
        "redacted_words",
        "redacted_intervals",
        "text_regions",
        "faces_detected",
        "frames_processed",
        "pages_processed",
        "audio_processed",
        "output_bytes",
        "unresolved_count",
    )

    for field in allowed_fields:
        if field in result:
            response[field] = result[field]

    return response


def _review_required(
    reason: str,
    media_type: str,
    **details,
) -> dict:
    return {
        "status": "review_required",
        "output_ready": False,
        "reason": reason,
        "media_type": media_type,
        "manual_review_required": True,
        **details,
    }


def _mask_image(
    workspace: WorkspaceClient,
    volume_path: str,
    local_source: str,
    local_output: str,
    instruction: str,
    endpoint: str,
    mask_people: bool,
) -> dict:
    """
    Image redaction:
      SQL OCR -> Claude -> localization
      -> black bars -> face masking.
    """

    parsed = _run_stage(
        "ai_parse_document",
        lambda: parse_document(
            volume_path=volume_path,
            client=workspace,
        ),
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

    selections = _run_stage(
        "claude_selection",
        lambda: select_elements(
            elements=elements,
            instruction=instruction,
            endpoint=endpoint,
            client=workspace,
        ),
    )

    localization = _run_stage(
        "localization",
        lambda: localize_replacements(
            parsed.elements,
            selections,
        ),
    )

    if localization.unresolved:
        return _review_required(
            "unresolved_text_locations",
            "image",
            unresolved_count=len(
                localization.unresolved
            ),
        )

    result = _run_stage(
        "image_redaction",
        lambda: replace_image(
            input_path=local_source,
            output_path=local_output,
            localization=localization,
            mask_people=mask_people,
        ),
    )

    if not isinstance(result, dict):
        raise RuntimeError(
            "Image processor returned invalid result"
        )

    if (
        result.get("output_ready")
        and mask_people
        and not result.get(
            "faces_masked_verified",
            False,
        )
    ):
        return _review_required(
            "face_masking_not_verified",
            "image",
        )

    return result


def _process(
    volume_path: str,
    instruction: str,
    mode: str = "redact",
    output_directory: str = "",
    output_filename: str = "",
    mask_people: bool = True,
) -> dict:
    """
    Process media from Unity Catalog Volumes.

    The original source is never overwritten.

    A successful technical processing result may be
    uploaded with manual_review_required=True.
    That flag does not mean anonymity is guaranteed.
    """

    source = validate_volume_path(
        volume_path
    )

    if mode == "mask":
        mode = "redact"

    if mode not in ("redact", "replace"):
        raise ValueError(
            "mode must be redact, mask or replace"
        )

    if mode == "replace":
        raise ValueError(
            "Replacement mode is not implemented "
            "by these processors; use redact or mask"
        )

    if (
        not isinstance(instruction, str)
        or not instruction.strip()
        or len(instruction) > 4000
    ):
        raise ValueError(
            "Instruction must contain 1-4000 characters"
        )

    media_type = kind(source)

    if media_type not in (
        "image",
        "audio",
        "video",
        "document",
    ):
        raise ValueError(
            f"Unsupported media type: {media_type}"
        )

    source_parent = str(
        Path(source).parent
    )

    if not output_directory:
        output_directory = source_parent

    if not output_filename:
        original = Path(source)

        output_filename = (
            f"{original.stem}_masked"
            f"{original.suffix}"
        )

    target = destination(
        source,
        output_directory,
        output_filename,
    )

    validate_conversion(
        source,
        target,
    )

    endpoint = os.getenv(
        "BLACKBAR_MODEL_ENDPOINT",
        DEFAULT_MODEL,
    ).strip()

    if not endpoint:
        raise ValueError(
            "BLACKBAR_MODEL_ENDPOINT is not configured"
        )

    workspace = WorkspaceClient()

    with tempfile.TemporaryDirectory(
        prefix="blackbar-"
    ) as directory:

        source_extension = (
            Path(source).suffix.lower()
        )

        target_extension = (
            Path(target).suffix.lower()
        )

        local_source = str(
            Path(directory)
            / f"source{source_extension}"
        )

        local_output = str(
            Path(directory)
            / f"output{target_extension}"
        )

        _run_stage(
            "volume_download",
            lambda: download(
                workspace,
                source,
                local_source,
            ),
        )

        if not Path(local_source).is_file():
            raise RuntimeError(
                "Source download failed"
            )

        if media_type == "image":

            result = _mask_image(
                workspace=workspace,
                volume_path=source,
                local_source=local_source,
                local_output=local_output,
                instruction=instruction,
                endpoint=endpoint,
                mask_people=mask_people,
            )

        elif media_type == "audio":

            result = _run_stage(
                "audio_processing",
                lambda: process_audio(
                    workspace,
                    local_source,
                    local_output,
                    instruction,
                    endpoint,
                ),
            )

        elif media_type == "video":

            result = _run_stage(
                "video_processing",
                lambda: process_video(
                    workspace,
                    local_source,
                    local_output,
                    instruction,
                    endpoint,
                ),
            )

        elif media_type == "document":

            result = _run_stage(
                "pdf_processing",
                lambda: process_pdf(
                    workspace=workspace,
                    volume_path=source,
                    source=local_source,
                    output=local_output,
                    instruction=instruction,
                    endpoint=endpoint,
                    mask_people=mask_people,
                ),
            )

        else:
            raise ValueError(
                f"Unsupported media type: {media_type}"
            )

        if not isinstance(result, dict):
            raise RuntimeError(
                "Processor returned invalid result"
            )

        # Never publish failed or incomplete redactions.
        if (
            result.get("status") != "completed"
            or not result.get(
                "output_ready",
                False,
            )
        ):
            return _public_report(
                result,
                source,
            )

        output_file = Path(local_output)

        if not output_file.is_file():
            raise RuntimeError(
                "Processor did not create output file"
            )

        if output_file.stat().st_size <= 0:
            raise RuntimeError(
                "Output file is empty"
            )

        output_bytes = (
            output_file.stat().st_size
        )

        # Manual review is advisory here.
        # It is not used to silently discard a
        # successfully processed output.
        #
        # The output must still pass every validation
        # performed inside its media processor.

        _run_stage(
            "volume_upload",
            lambda: upload(
                workspace,
                target,
                local_output,
            ),
        )

        completed = {
            **result,
            "status": "completed",
            "output_ready": True,
            "output_bytes": output_bytes,
            "manual_review_required": result.get(
                "manual_review_required",
                True,
            ),
        }

        return _public_report(
            completed,
            source,
            target,
        )


def _error_response(
    exc: Exception,
) -> dict:
    """
    Return structured errors without exposing
    credentials or sensitive media content.
    """

    logger.error(
        "BlackBar request failed: %s",
        type(exc).__name__,
    )

    response = {
        "status": "error",
        "output_ready": False,
        "output_path": None,
        "error_type": type(exc).__name__,
        "message": "Media processing failed",
    }

    if isinstance(
        exc,
        ProcessingStageError,
    ):
        response["stage"] = exc.stage
        response["error_type"] = (
            exc.original_error_type
        )

    return response


def register_tools(mcp):
    """Register all MCP tools."""

    @mcp.tool()
    def health() -> dict:
        """
        Check BlackBar MCP configuration.
        """

        return {
            "status": "ok",
            "service": "BlackBar MCP",
            "version": "0.8.0",
            "parser": "ai_parse_document_v2",
            "model_endpoint": os.getenv(
                "BLACKBAR_MODEL_ENDPOINT",
                DEFAULT_MODEL,
            ),
            "warehouse_configured": bool(
                os.getenv(
                    "BLACKBAR_WAREHOUSE_ID"
                )
            ),
            "face_detection": "yunet_haar",
            "audio_transcription": "faster_whisper",
        }

    @mcp.tool()
    def list_supported_formats() -> dict:
        """
        List supported input and output formats.
        """

        return {
            "images": sorted(IMAGES),
            "audio": sorted(AUDIOS),
            "video": sorted(VIDEOS),
            "documents": sorted(DOCUMENTS),
            "conversion_scope": "same_media_type",
        }

    @mcp.tool()
    async def mask_document(
        volume_path: str,
        instruction: str,
        mode: str = "redact",
        output_directory: str = "",
        output_filename: str = "",
        mask_people: bool = True,
    ) -> dict:
        """
        Permanently mask sensitive information
        in an image or PDF from a UC Volume.
        """

        try:
            media_type = kind(
                volume_path
            )

            if media_type not in (
                "image",
                "document",
            ):
                raise ValueError(
                    "mask_document accepts images and PDFs"
                )

            return await asyncio.to_thread(
                _process,
                volume_path,
                instruction,
                mode,
                output_directory,
                output_filename,
                mask_people,
            )

        except Exception as exc:
            return _error_response(exc)

    @mcp.tool()
    async def mask_media(
        volume_path: str,
        instruction: str,
        mode: str = "redact",
        output_directory: str = "",
        output_filename: str = "",
        mask_people: bool = True,
    ) -> dict:
        """
        Redact sensitive content in an image,
        audio file, video or PDF stored in
        Unity Catalog Volumes.

        Supports changing output formats within
        the same media type.

        Example:
        source:
          /Volumes/catalog/schema/volume/audio.m4a

        output_directory:
          /Volumes/catalog/schema/volume

        output_filename:
          audioalterado.mp3
        """

        try:
            return await asyncio.to_thread(
                _process,
                volume_path,
                instruction,
                mode,
                output_directory,
                output_filename,
                mask_people,
            )

        except Exception as exc:
            return _error_response(exc)
