"""FFmpeg-backed, non-destructive audio preprocessing."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from research_kb.audio.core import sha256_file
from research_kb.audio.domain import AudioChunk, RecordingSource

_SILENCE_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([0-9.]+)")


class AudioPreprocessError(RuntimeError):
    """Raised when a source cannot be decoded or enhanced safely."""


def ffmpeg_executable() -> str:
    """Resolve the project-managed FFmpeg binary."""
    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]
    except ImportError as error:
        raise AudioPreprocessError(
            "imageio-ffmpeg is required; run uv sync before transcription"
        ) from error
    return str(imageio_ffmpeg.get_ffmpeg_exe())


def _run_ffmpeg(arguments: list[str], *, timeout_seconds: float | None = None) -> str:
    command = [ffmpeg_executable(), "-hide_banner", "-nostdin", *arguments]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-12:])
        raise AudioPreprocessError(f"FFmpeg failed ({result.returncode}): {tail}")
    return result.stderr


def detect_silence_midpoints(
    source: RecordingSource,
    *,
    noise_db: int = -45,
    minimum_seconds: float = 1.0,
) -> tuple[int, ...]:
    """Return midpoints of stable silent spans without writing output."""
    stderr = _run_ffmpeg(
        [
            "-i",
            source.path,
            "-af",
            f"silencedetect=noise={noise_db}dB:d={minimum_seconds}",
            "-f",
            "null",
            "-",
        ]
    )
    starts = [float(match.group(1)) for match in _SILENCE_START.finditer(stderr)]
    ends = [float(match.group(1)) for match in _SILENCE_END.finditer(stderr)]
    midpoints: list[int] = []
    for start, end in zip(starts, ends, strict=False):
        if end > start:
            midpoints.append(round((start + end) * 500))
    return tuple(midpoints)


def _filter_chain(*, enhanced: bool) -> str:
    filters = [
        "pan=mono|c0=0.5*c0+0.5*c1",
        "highpass=f=70",
        "lowpass=f=7600",
    ]
    if enhanced:
        filters.extend(
            [
                "afftdn=nr=8:nf=-40:tn=1",
                "loudnorm=I=-20:LRA=11:TP=-2",
            ]
        )
    else:
        filters.append("dynaudnorm=f=500:g=7:p=0.9:m=10")
    return ",".join(filters)


def encode_chunk(
    source: RecordingSource,
    chunk: AudioChunk,
    output_path: Path,
    *,
    enhanced: bool,
) -> str:
    """Create one mono 16 kHz lossless derivative and return its SHA-256."""
    target = output_path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    duration_seconds = chunk.duration_ms / 1000
    _run_ffmpeg(
        [
            "-y",
            "-ss",
            f"{chunk.start_ms / 1000:.3f}",
            "-i",
            source.path,
            "-t",
            f"{duration_seconds:.3f}",
            "-vn",
            "-af",
            _filter_chain(enhanced=enhanced),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "flac",
            str(target),
        ]
    )
    return sha256_file(target)


def extract_clip(
    source_path: Path,
    output_path: Path,
    *,
    start_ms: int,
    end_ms: int,
    wav: bool = True,
) -> None:
    """Extract a timestamped clip from an already normalized derivative."""
    if end_ms <= start_ms:
        raise ValueError("clip end must be after start")
    target = output_path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.casefold() == ".mp3":
        codec = ["-c:a", "libmp3lame", "-b:a", "48k"]
    else:
        codec = ["-c:a", "pcm_s16le"] if wav else ["-c:a", "flac"]
    _run_ffmpeg(
        [
            "-y",
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-i",
            str(source_path),
            "-t",
            f"{(end_ms - start_ms) / 1000:.3f}",
            "-ar",
            "16000",
            "-ac",
            "1",
            *codec,
            str(target),
        ]
    )
