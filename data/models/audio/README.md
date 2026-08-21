# Local Audio Models

[English](README.md) · [Chinese](README.zh-CN.md)

This is the default directory for local models. Model files are not committed to Git. Use `VOICETRACE_AUDIO_MODEL_DIR` or `--model-dir` to select another directory.

```text
audio/
├─ Qwen3-ASR-1.7B/   # optional local ASR, about 4.38 GiB
├─ fsmn-vad/         # low-cost speech interval detection, about 4 MB
└─ ERes2NetV2/       # cross-segment speaker features, about 71 MB
```

Cloud full-transcription mode can run without these models. Speech-only cloud mode requires `fsmn-vad` and `ERes2NetV2`; local ASR additionally requires `Qwen3-ASR-1.7B`.

The runtime downloads missing models on first use, or reuses models already present in the selected directory.
