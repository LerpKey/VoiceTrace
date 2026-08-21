# VoiceTrace

[English](README.md) · [Chinese](README.zh-CN.md)

A local-first voice transcription and recording review workspace.

VoiceTrace turns long recordings into text that can be replayed, searched, and checked. It is designed for meetings, interviews, classes, and personal recordings. Audio and transcripts stay on the local machine by default. Cloud services are optional, and the workspace uploads only speech intervals selected by local voice activity detection by default.

## Features

- Resumable long-recording pipeline with silence splitting, voice activity detection (VAD), overlap deduplication, and Markdown rendering;
- Cloud ASR by default, with CPU VAD and CPU ERes2NetV2 speaker association;
- Optional local Qwen3-ASR support, without making local ASR hardware a default requirement;
- Speaker identification, display-name management, and cross-segment speaker memory;
- Timeline playback, transcript reading, topic browsing, sentence favorites, and Markdown export;
- Local FastAPI service, browser review UI, CLI entry point, and Windows launcher;
- API keys read from environment variables and never written into source code or runtime artifacts.

## Current status

Version `0.1.0` is still under development and currently targets personal use on a Windows machine. The project does not yet provide multi-user accounts, remote audio storage, or a deployment configuration suitable for direct public exposure.

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), and Node.js 22.13 or newer.

```powershell
uv sync
cd audio-review-ui
npm ci
cd ..
uv run voice-trace --help
```

Start the recording review workspace:

```powershell
start-voicetrace-workspace.bat
```

Or start it manually:

```powershell
uv run voice-trace audio-review --data-dir data
```

The default path uses CPU FunASR models for VAD and speaker feature extraction, so CUDA is not required. To enable local Qwen3-ASR, install the optional local dependencies:

```powershell
uv sync --extra local
```

See the [audio review user guide](docs/reference/audio-review-user-guide.md) for model preparation, API key configuration, and review workflows.

## Operating modes

- Cloud enhancement: set `DASHSCOPE_API_KEY`. The workspace filters speech intervals with local CPU VAD, uploads only those intervals, and associates speakers with CPU ERes2NetV2.
- Local ASR: disabled by default. Install the `local` extra, prepare the Qwen3-ASR model, and pass `--local` explicitly. This path requires CUDA.
- Text review and summaries: set `DEEPSEEK_API_KEY` when needed. The original transcript is never overwritten by the summary flow.

Models are stored in `data/models/audio` by default. Use `VOICETRACE_AUDIO_MODEL_DIR` or `--model-dir` to select another directory.

## Project structure

```text
research_kb/                 Python runtime, API, and CLI entry points
audio-review-ui/             Browser review UI
docs/                        User guides and model notes
tests/                       Unit and governance tests
data/models/audio/           Local model directory placeholder; weights are not committed
```

## Development and checks

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
cd audio-review-ui
npm test
npm run lint
```

Keep development benchmarks, models, caches, recordings, and transcripts local. Do not commit `.env` files, API keys, private keys, recordings, model weights, or personal transcripts.

## Documentation

- [Audio review user guide](docs/reference/audio-review-user-guide.md) · [Chinese](docs/reference/audio-review-user-guide.zh-CN.md)
- [Model packaging](docs/model-packaging.md) · [Chinese](docs/model-packaging.zh-CN.md)

## License

This repository does not currently include a license file. Until the maintainers choose a license and add `LICENSE`, the project should not be treated as granting the public permission to use, modify, or distribute it.
