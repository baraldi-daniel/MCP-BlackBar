
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageOps


IMAGES = {
    ".png", ".jpg", ".jpeg",
    ".webp", ".bmp", ".tif", ".tiff",
}

AUDIOS = {
    ".mp3", ".m4a", ".wav", ".aac",
    ".flac", ".ogg", ".oga", ".opus",
    ".wma", ".aif", ".aiff", ".ac3",
    ".caf", ".mp2", ".amr", ".mka",
}

VIDEOS = {
    ".mp4", ".mov", ".mkv",
    ".webm", ".avi",
}

DOCUMENTS = {".pdf"}

OUTPUT_AUDIO = AUDIOS - {".amr", ".mka"}


def kind(path: str) -> str:
    extension = Path(path).suffix.lower()

    if extension in IMAGES:
        return "image"
    if extension in AUDIOS:
        return "audio"
    if extension in VIDEOS:
        return "video"
    if extension in DOCUMENTS:
        return "document"

    raise ValueError(
        f"Unsupported extension: {extension}"
    )


def validate_conversion(source: str, output: str) -> str:
    media_type = kind(source)
    extension = Path(output).suffix.lower()

    allowed = {
        "image": IMAGES,
        "audio": OUTPUT_AUDIO,
        "video": VIDEOS,
        "document": DOCUMENTS,
    }[media_type]

    if extension not in allowed:
        raise ValueError(
            f"Unsupported conversion: {media_type} -> {extension}"
        )

    if source == output:
        raise ValueError("Cannot overwrite source")

    return media_type


def run_ffmpeg(*arguments: str) -> None:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        *arguments,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )

    if result.returncode:
        raise RuntimeError(
            f"FFmpeg failed: {result.stderr[-2000:]}"
        )


def probe(path: str) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_streams",
            "-show_format",
            "-of", "json",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )

    return json.loads(result.stdout)


def duration(path: str) -> float:
    value = float(
        probe(path)["format"]["duration"]
    )

    if not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid media duration")

    return value


def audio_codec(extension: str) -> list[str]:
    codecs = {
        ".mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
        ".m4a": ["-c:a", "aac", "-b:a", "192k"],
        ".aac": ["-c:a", "aac", "-b:a", "192k"],
        ".wav": ["-c:a", "pcm_s16le"],
        ".flac": ["-c:a", "flac"],
        ".ogg": ["-c:a", "libvorbis", "-q:a", "5"],
        ".oga": ["-c:a", "libvorbis", "-q:a", "5"],
        ".opus": ["-c:a", "libopus", "-b:a", "128k"],
        ".wma": ["-c:a", "wmav2", "-b:a", "192k"],
        ".aif": ["-c:a", "pcm_s16be"],
        ".aiff": ["-c:a", "pcm_s16be"],
        ".ac3": ["-c:a", "ac3", "-b:a", "192k"],
        ".caf": ["-c:a", "pcm_s16le"],
        ".mp2": ["-c:a", "mp2", "-b:a", "192k"],
    }

    if extension not in codecs:
        raise ValueError(
            f"Unsupported audio output: {extension}"
        )

    return codecs[extension]


def video_codec(extension: str) -> list[str]:
    if extension == ".webm":
        return [
            "-c:v", "libvpx-vp9",
            "-crf", "32",
            "-b:v", "0",
            "-c:a", "libopus",
        ]

    if extension == ".avi":
        return [
            "-c:v", "mpeg4",
            "-q:v", "3",
            "-c:a", "libmp3lame",
            "-b:a", "192k",
        ]

    return [
        "-c:v", "libx264",
        "-crf", "20",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
    ]


def save_image(image: Image.Image, output: str) -> None:
    extension = Path(output).suffix.lower()

    if extension not in IMAGES:
        raise ValueError("Unsupported image output")

    Path(output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = ImageOps.exif_transpose(image)

    if extension in (".jpg", ".jpeg", ".bmp"):
        if image.mode in ("RGBA", "LA"):
            rgba = image.convert("RGBA")
            background = Image.new(
                "RGB", rgba.size, "white"
            )
            background.paste(
                rgba,
                mask=rgba.getchannel("A"),
            )
            image = background
        else:
            image = image.convert("RGB")

    formats = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".tif": "TIFF",
        ".tiff": "TIFF",
    }

    image.save(
        output,
        format=formats.get(
            extension,
            extension[1:].upper(),
        ),
    )


def convert_redacted(source: str, output: str) -> None:
    media_type = validate_conversion(source, output)

    Path(output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if media_type == "image":
        with Image.open(source) as image:
            save_image(image, output)
        return

    if media_type == "audio":
        run_ffmpeg(
            "-i", source,
            "-map", "0:a:0",
            "-vn",
            *audio_codec(Path(output).suffix.lower()),
            output,
        )
        return

    if media_type == "video":
        run_ffmpeg(
            "-i", source,
            "-map", "0:v:0",
            "-map", "0:a?",
            *video_codec(Path(output).suffix.lower()),
            output,
        )
        return

    raise ValueError(
        "PDF must be processed through server.pdf"
    )
