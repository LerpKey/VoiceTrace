"""Structured facts for reproducible audio transcription."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DecisionKind = Literal[
    "cloud_agreement",
    "cloud_primary",
    "local_primary",
    "third_pass_majority",
    "unclear",
    "non_speech",
    "text_context",
]


class RecordingSource(BaseModel):
    """Immutable identity and timeline for one source recording."""

    model_config = ConfigDict(frozen=True)

    recording_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    duration_ms: int = Field(gt=0)
    recorded_at: datetime
    codec: str = "aac"
    sample_rate_hz: int = Field(default=48_000, gt=0)
    channels: int = Field(default=2, gt=0)

    @property
    def source_path(self) -> Path:
        """Return the source path without mutating it."""
        return Path(self.path)


class AudioChunk(BaseModel):
    """One complete-coverage cloud transcription chunk."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    recording_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    overlap_before_ms: int = Field(default=0, ge=0)
    enhanced_path: str | None = None
    light_path: str | None = None
    enhanced_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    light_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_bounds(self) -> AudioChunk:
        """Reject empty or inverted chunk ranges."""
        if self.end_ms <= self.start_ms:
            raise ValueError("audio chunk end must be after start")
        if self.overlap_before_ms > self.end_ms - self.start_ms:
            raise ValueError("chunk overlap cannot exceed chunk duration")
        return self

    @property
    def duration_ms(self) -> int:
        """Return the encoded chunk duration."""
        return self.end_ms - self.start_ms


class ProviderCandidate(BaseModel):
    """One provider's evidence for a time range."""

    model_config = ConfigDict(frozen=True)

    provider: Literal["cloud", "local", "third_pass"]
    model: str = Field(min_length=1)
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    speaker_id: str | None = None
    task_id: str | None = None


class TranscriptSegment(BaseModel):
    """One adjudicated utterance with complete provider provenance."""

    model_config = ConfigDict(frozen=True)

    segment_id: str = Field(min_length=1)
    recording_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    speaker: str = Field(min_length=1)
    text: str
    decision: DecisionKind
    confidence: float = Field(ge=0, le=1)
    flags: tuple[str, ...] = ()
    candidates: tuple[ProviderCandidate, ...] = ()

    @model_validator(mode="after")
    def validate_bounds(self) -> TranscriptSegment:
        """Reject inverted transcript ranges."""
        if self.end_ms <= self.start_ms:
            raise ValueError("transcript segment end must be after start")
        return self


class SpeakerProfile(BaseModel):
    """Anonymous global speaker identity used across recordings."""

    model_config = ConfigDict(frozen=True)

    speaker: str = Field(min_length=1)
    local_speaker_keys: tuple[str, ...]
    recording_ids: tuple[str, ...]
    sample_count: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)


class TranscriptDocument(BaseModel):
    """Authoritative structured transcript from which Markdown is rendered."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    title: str = "Three-day office recording transcript"
    generated_at: datetime
    sources: tuple[RecordingSource, ...]
    speakers: tuple[SpeakerProfile, ...]
    segments: tuple[TranscriptSegment, ...]

    @model_validator(mode="after")
    def validate_timeline(self) -> TranscriptDocument:
        """Ensure every segment is attached to a source and stays in bounds."""
        durations = {source.recording_id: source.duration_ms for source in self.sources}
        previous: tuple[int, int] | None = None
        order = {source.recording_id: index for index, source in enumerate(self.sources)}
        for segment in self.segments:
            if segment.recording_id not in durations:
                raise ValueError("transcript segment references an unknown recording")
            if segment.end_ms > durations[segment.recording_id]:
                raise ValueError("transcript segment exceeds source duration")
            current = (order[segment.recording_id], segment.start_ms)
            if previous is not None and current < previous:
                raise ValueError("transcript segments must be chronologically ordered")
            previous = current
        return self


class ProcessingManifest(BaseModel):
    """Resumable state and cost audit without credentials or signed URLs."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["planned", "running", "completed", "failed"]
    created_at: datetime
    updated_at: datetime
    source_hashes: dict[str, str]
    chunks: tuple[AudioChunk, ...]
    local_model: str
    cloud_model: str | None = None
    cloud_fallback_model: str | None = None
    cloud_task_ids: dict[str, str] = Field(default_factory=dict)
    cloud_billed_seconds: int = Field(default=0, ge=0)
    estimated_cost_cny: float = Field(default=0, ge=0)
    cost_cap_cny: float = Field(gt=0)
    text_review_model: str | None = None
    text_review_requests: int = Field(default=0, ge=0)
    text_review_input_tokens: int = Field(default=0, ge=0)
    text_review_output_tokens: int = Field(default=0, ge=0)
    text_review_cost_cny: float = Field(default=0, ge=0)
    text_review_cost_cap_cny: float = Field(default=0, ge=0)
    completed_steps: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
