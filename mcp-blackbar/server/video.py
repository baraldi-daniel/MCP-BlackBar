from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from server.audio import process_audio
from server.conversion import (
    VIDEOS,
    duration,
    probe,
    run_ffmpeg,
    video_codec,
)
from server.faces import mask_faces
from server.localize import localize_replacements
from server.model import select_elements
from server.parser import parse_document
from server.redact import redact_regions


VIDEO = VIDEOS


def _has_audio(path: str) -> bool:
    metadata = probe(path)

    return any(
        stream.get("codec_type") == "audio"
        for stream in metadata.get("streams", [])
    )


def process_video(
    workspace,
    source: str,
    output: str,
    instruction: str,
    endpoint: str,
) -> dict:
    maximum = float(
        os.getenv(
            "BLACKBAR_MAX_VIDEO_SECONDS",
            "180",
        )
    )

    seconds = duration(source)

    if seconds > maximum:
        raise ValueError(
            f"Video exceeds {maximum} seconds"
        )

    # Video redaction requires visual OCR through
    # ai_parse_document. Local frame OCR is intentionally
    # not replaced by Tesseract.

    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError("Cannot open video")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Invalid video metadata")

    sample_seconds = float(
        os.getenv(
            "BLACKBAR_VIDEO_SAMPLE_SECONDS",
            "1",
        )
    )

    if sample_seconds <= 0:
        capture.release()
        raise ValueError("Invalid video sampling interval")

    frame_count = 0
    face_count = 0
    text_count = 0

    # The current pipeline does not have a verified
    # temporal text tracker. Therefore it must not
    # publish a video after inspecting only sampled frames.
    #
    # All frames are processed for faces. Text OCR needs
    # an additional verified frame-to-Volume pipeline.

    with tempfile.TemporaryDirectory(
        prefix="blackbar-video-"
    ) as directory:
        silent_video = str(
            Path(directory) / "video.mp4"
        )

        raw_audio = str(
            Path(directory) / "audio.wav"
        )

        masked_audio = str(
            Path(directory) / "masked.wav"
        )

        writer = cv2.VideoWriter(
            silent_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            capture.release()
            raise RuntimeError(
                "Cannot initialize video writer"
            )

        try:
            while True:
                ok, frame = capture.read()

                if not ok:
                    break

                image = Image.fromarray(
                    cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB,
                    )
                )

                image, boxes, diagnostics = mask_faces(
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

                face_count += len(boxes)

                output_frame = cv2.cvtColor(
                    np.asarray(image),
                    cv2.COLOR_RGB2BGR,
                )

                writer.write(output_frame)

                frame_count += 1

        finally:
            capture.release()
            writer.release()

        if frame_count == 0:
            raise RuntimeError(
                "No readable video frames"
            )

        audio_processed = False

        if _has_audio(source):
            run_ffmpeg(
                "-i", source,
                "-map", "0:a:0",
                "-vn",
                "-c:a", "pcm_s16le",
                raw_audio,
            )

            audio_result = process_audio(
                workspace,
                raw_audio,
                masked_audio,
                instruction,
                endpoint,
            )

            if not audio_result.get("output_ready"):
                return {
                    "status": "review_required",
                    "output_ready": False,
                    "reason": "audio_processing_incomplete",
                }

            audio_processed = True

        # Fail closed: visual text may remain in frames.
        # Do not upload a video as fully redacted until
        # temporal text masking is implemented and tested.
        return {
            "status": "review_required",
            "output_ready": False,
            "reason": "video_text_redaction_not_verified",
            "media_type": "video",
            "frames_processed": frame_count,
            "faces_detected": face_count,
            "text_regions": text_count,
            "audio_processed": audio_processed,
            "manual_review_required": True,
        }