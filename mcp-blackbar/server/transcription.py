
from __future__ import annotations

import math
import os
from pathlib import Path

from faster_whisper import WhisperModel

from server.conversion import duration


_MODEL = None


def get_model() -> WhisperModel:
    global _MODEL

    if _MODEL is None:
        _MODEL = WhisperModel(
            os.getenv(
                "BLACKBAR_WHISPER_MODEL",
                "small",
            ),
            device=os.getenv(
                "BLACKBAR_WHISPER_DEVICE",
                "cpu",
            ),
            compute_type=os.getenv(
                "BLACKBAR_WHISPER_COMPUTE",
                "int8",
            ),
        )

    return _MODEL


def transcribe_audio(path: str) -> dict:
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(path)

    seconds = duration(path)

    maximum = float(
        os.getenv(
            "BLACKBAR_MAX_AUDIO_SECONDS",
            "1800",
        )
    )

    if seconds > maximum:
        raise ValueError(
            f"Audio duration exceeds {maximum} seconds"
        )

    segments, info = get_model().transcribe(
        path,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        language=os.getenv(
            "BLACKBAR_AUDIO_LANGUAGE"
        ) or None,
    )

    words = []
    segment_data = []

    for segment in segments:
        start = float(segment.start)
        end = float(segment.end)

        segment_data.append({
            "start": start,
            "end": end,
            "text": segment.text.strip(),
        })

        for word in segment.words or []:
            if word.start is None or word.end is None:
                continue

            word_start = float(word.start)
            word_end = float(word.end)

            if not (
                math.isfinite(word_start)
                and math.isfinite(word_end)
                and word_end > word_start
            ):
                continue

            words.append({
                "element_id": str(len(words)),
                "text": word.word,
                "start": word_start,
                "end": word_end,
            })

    return {
        "language": info.language,
        "duration": seconds,
        "text": " ".join(
            segment["text"]
            for segment in segment_data
        ),
        "segments": segment_data,
        "words": words,
    }
