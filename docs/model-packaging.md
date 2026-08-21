# Local Audio Model Packaging

[English](model-packaging.md) · [Chinese](model-packaging.zh-CN.md)

This is the default directory for local audio models. Model files are not committed to Git. Use `VOICETRACE_AUDIO_MODEL_DIR` or `--model-dir` to select another directory.

## Product default path

The normal product entry point exposes a low-cost cloud enhancement option and enables it by default. The pipeline uses CPU VAD to select speech intervals, CPU ERes2NetV2 to associate speakers across chunks, and sends only those speech intervals to cloud ASR. Full cloud upload and local Qwen3-ASR are not default workspace paths.

## Model layers

| Model | Default distribution | Purpose |
|---|---|---|
| `fsmn-vad` | Preinstall with a production package | Identify actual speech intervals and reduce cloud upload and cost |
| `ERes2NetV2` | Preinstall with a production package | Associate the same speaker across audio chunks on CPU |
| `Qwen3-ASR-1.7B` | Do not preinstall | Optional local ASR when explicitly enabled |

The source repository does not include model weights. A production distribution may ship the first two small models as a separate model resource package and let the installer copy them into the user's selected model root. That is different from committing weights to Git.

## Directory convention

```text
<selected model root>/
├─ fsmn-vad/
├─ ERes2NetV2/
└─ Qwen3-ASR-1.7B/    # optional
```

During development, set `VOICETRACE_AUDIO_MODEL_DIR` or pass `--model-dir`. The CLI also defaults to the low-cost cloud path. `--no-cloud --local --no-allow-cloud-upload --full-cloud` explicitly selects the local-only debugging path; full cloud upload should be used only when it has been deliberately authorized with `--full-cloud`.

When required small models are missing, the runtime downloads them on first use. A production installer should prefer bundled resources to avoid a long first launch. Future model updates can add a version manifest, hash verification, and atomic replacement without changing the current directory convention.
