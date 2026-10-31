from __future__ import annotations

from pathlib import Path

from databricks.sdk import WorkspaceClient

from server.conversion import validate_conversion


def _parts(path: str) -> list[str]:
    if not isinstance(path, str):
        raise ValueError("Path must be a string")

    if "\x00" in path or "\\" in path:
        raise ValueError("Invalid path")

    parts = path.split("/")

    if len(parts) < 5 or parts[0] != "":
        raise ValueError("Invalid Volume path")

    if parts[1] != "Volumes":
        raise ValueError("Path must start with /Volumes/")

    if any(part in ("", ".", "..") for part in parts[2:]):
        raise ValueError("Invalid path component")

    return parts


def validate_volume_path(path: str) -> str:
    parts = _parts(path)

    if len(parts) < 6:
        raise ValueError("Volume file path required")

    if not Path(path).suffix:
        raise ValueError("File extension required")

    return path


def validate_volume_directory(path: str) -> str:
    path = path.rstrip("/")
    parts = _parts(path)

    if len(parts) < 5:
        raise ValueError("Invalid Volume directory")

    return path


def destination(
    source: str,
    output_directory: str,
    output_filename: str,
    extension: str | None = None,
) -> str:
    validate_volume_path(source)

    if not output_directory:
        output_directory = source.rsplit("/", 1)[0]

    validate_volume_directory(output_directory)

    if not output_filename:
        output_filename = (
            Path(source).stem
            + "_masked"
            + Path(source).suffix
        )

    if (
        "/" in output_filename
        or "\\" in output_filename
        or output_filename in (".", "..")
    ):
        raise ValueError("Invalid output filename")

    result = (
        output_directory.rstrip("/")
        + "/"
        + output_filename
    )

    validate_volume_path(result)
    validate_conversion(source, result)

    if result == source:
        raise ValueError("Cannot overwrite source")

    return result


def download(
    workspace: WorkspaceClient,
    remote_path: str,
    local_path: str,
) -> str:
    validate_volume_path(remote_path)

    Path(local_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    workspace.files.download_to(
        remote_path,
        local_path,
        overwrite=True,
    )

    target = Path(local_path)

    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("Download failed")

    return local_path


def upload(
    workspace: WorkspaceClient,
    remote_path: str,
    local_path: str,
) -> str:
    validate_volume_path(remote_path)

    source = Path(local_path)

    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError("Output file missing")

    workspace.files.create_directory(
        remote_path.rsplit("/", 1)[0]
    )

    workspace.files.upload_from(
        remote_path,
        local_path,
        overwrite=False,
    )

    return remote_path