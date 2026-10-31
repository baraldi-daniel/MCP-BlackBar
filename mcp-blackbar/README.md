
# BlackBar MCP

AI-assisted sensitive information redaction for
images, audio, video and PDF files on Databricks.

## Overview

BlackBar MCP is a Model Context Protocol (MCP)
server that processes files stored in Unity Catalog
Volumes.

Users can request masking through an AI agent
connected to the Databricks-hosted MCP application.

The original file is preserved. Successful operations
produce a separate output file.

## Technology Stack

| Component | Technology |
|---|---|
| Hosting | Databricks Apps |
| MCP | FastMCP 2.x |
| API runtime | FastAPI / Uvicorn |
| Storage | Unity Catalog Volumes |
| Language model | Claude via Databricks AI Gateway |
| OCR | ai_parse_document v2 |
| Audio transcription | faster-whisper |
| Audio conversion | FFmpeg / imageio-ffmpeg |
| Image processing | Pillow / OpenCV |
| PDF processing | PyMuPDF |

## Project Structure

```text
blackbar-mcp/
├── app.yaml
├── manifest.yaml
├── pyproject.toml
├── CLAUDE.md
├── README.md
├── LICENCE
├── NOTICE
└── server/
    ├── __init__.py
    ├── app.py
    ├── main.py
    ├── utils.py
    ├── tools.py
    ├── storage.py
    ├── conversion.py
    ├── parser.py
    ├── model.py
    ├── localize.py
    ├── media.py
    ├── faces.py
    ├── audio.py
    ├── transcription.py
    ├── video.py
    ├── pdf.py
    └── sensitive.py
```

## MCP Tools

### health

Returns the server's configuration status.

### list_supported_formats

Lists the media formats recognized by the current
deployment.

### mask_document

Requests sensitive-content redaction for supported
images and PDF documents.

### mask_media

Requests sensitive-content redaction for supported
images, audio, video and PDF documents.

The available capabilities depend on the installed
processing modules.

## Example: Automatic Audio Redaction

Input:

/Volumes/unimed_fortaleza/unimed_ia/docs/Rua Conselheiro Moreira de Barros 2.m4a

Output:

/Volumes/unimed_fortaleza/unimed_ia/docs/audioalterado.mp3

Example tool arguments:

```json
{
  "volume_path": "/Volumes/unimed_fortaleza/unimed_ia/docs/Rua Conselheiro Moreira de Barros 2.m4a",
  "instruction": "Detect and silence sensitive personal information while preserving other spoken content.",
  "mode": "mask",
  "output_directory": "/Volumes/unimed_fortaleza/unimed_ia/docs",
  "output_filename": "audioalterado.mp3"
}
```

The tool should transcribe the recording, identify
sensitive words, align timestamps, silence selected
intervals and create the requested output.

No output should be considered complete until the
tool confirms successful processing and upload.

## Configuration

Databricks Apps uses app.yaml to define the startup
command and application environment variables.

Startup:

```bash
uv sync
exec uv run --no-sync blackbar-mcp
```

Key settings:

| Variable | Purpose |
|---|---|
| BLACKBAR_MODEL_ENDPOINT | Claude model endpoint |
| BLACKBAR_WAREHOUSE_ID | SQL Warehouse for OCR |
| BLACKBAR_PARSER_VERSION | Document parser version |
| BLACKBAR_WHISPER_MODEL | Whisper model size |
| BLACKBAR_MAX_AUDIO_SECONDS | Audio duration limit |
| BLACKBAR_MAX_VIDEO_SECONDS | Video duration limit |
| BLACKBAR_LOG_LEVEL | Application logging |

The configured Claude endpoint is:

system.ai.claude-opus-4-8

Avoid hardcoding secrets in app.yaml.

## Deployment

1. Open the project folder in Databricks Workspace.
2. Confirm app.yaml and pyproject.toml are present.
3. Verify that the server/ Python modules exist.
4. Open Compute > Apps.
5. Select the target application.
6. Deploy the updated workspace folder.
7. Open the App logs.
8. Confirm that the application starts successfully.

Expected startup messages include:

```text
StreamableHTTP session manager started
Application startup complete
```

## MCP Connection

The MCP endpoint is:

```text
https://<your-app-url>/mcp
```

The app must accept requests to /mcp without
an unexpected redirect.

The caller requires valid Databricks authentication
and the relevant application permissions.

Test the health tool before processing media.

## Troubleshooting

### MCP initialize response is empty

Verify:
- Application startup logs
- Correct deployed source folder
- /mcp route configuration
- Middleware and ASGI lifespan
- Authentication and permissions

### HTTP 307 at POST /mcp

Inspect the mount path and trailing-slash routing.
The client must reach a valid MCP endpoint without
depending on redirects.

### HTTP 502 Bad Gateway

Inspect application logs for startup exceptions,
including missing Python modules or failed imports.

### FileNotFoundError during audio_processing

Possible causes:
- FFmpeg executable unavailable
- Whisper model/cache files unavailable
- Missing local temporary file
- Another missing runtime resource

The processing stage is important: an error in
audio_processing does not prove that the UC
Volume source file is missing.

### review_required

Processing was not verified as sufficiently
complete for a successful output.

Do not represent it as completed masking.

## Security and Privacy

Files may contain personal and sensitive data.

- Protect access to Unity Catalog Volumes.
- Use appropriate data-processing permissions.
- Avoid logging raw transcripts or PII.
- Keep original files intact.
- Validate output paths before writing.
- Do not overwrite existing files.
- Review generated outputs before external sharing.

Whisper, OCR, face detection and language models
can miss sensitive information.

BlackBar MCP does not guarantee complete or
regulatory-compliant anonymization.

## Limitations

Capabilities depend on the deployed modules,
available model permissions and runtime resources.

Automatic detection may have false negatives
or incorrect timestamp alignment.

Video sampling cannot by itself guarantee detection
of every sensitive frame.

Large files may exceed configured processing limits.

Successful conversion alone is not proof that
sensitive information has been fully removed.

## Licensing

See LICENCE for the project's license.

Third-party packages remain governed by their
own licenses. See NOTICE.

## Development Guidance

See CLAUDE.md for architecture rules, security
requirements and implementation constraints.
