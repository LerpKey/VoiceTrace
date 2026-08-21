"""Resumable orchestration for dual-track long-recording transcription."""


from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from research_kb.audio.core import (
    deduplicate_overlap,
    has_sensitive_disagreement,
    inspect_recording,
    plan_chunks,
    render_markdown,
    sha256_file,
    text_agreement,
    union_intervals,
)
from research_kb.audio.domain import (
    AudioChunk,
    ProcessingManifest,
    ProviderCandidate,
    RecordingSource,
    SpeakerProfile,
    TranscriptDocument,
    TranscriptSegment,
)
from research_kb.audio.preprocess import (
    detect_silence_midpoints,
    encode_chunk,
    extract_clip,
)
from research_kb.audio.pricing import PRICE_PER_SECOND_CNY
from research_kb.audio.providers import (
    CloudResult,
    DashScopeFileTranscriber,
    DashScopeFlashTranscriber,
    LocalQwenTranscriber,
    ProviderSentence,
    TranscriptionProviderError,
    write_provider_payload,
)
from research_kb.audio.speaker import (
    LocalSpeakerSamples,
    SpeakerEmbedder,
    anonymous_speaker_label,
    resolve_local_segments,
    resolve_speakers,
)
from research_kb.audio.text_review import (
    DeepSeekTranscriptReviewer,
    TranscriptTextAnalysis,
    TranscriptTextReviewError,
    estimate_deepseek_cost_cny,
    render_topic_markdown,
)
from research_kb.storage.json import write_json_model
from research_kb.storage.text import write_text_atomic

LOCAL_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
SPEAKER_MODEL_ID = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
PRIMARY_CLOUD_MODEL = "qwen3-asr-flash-filetrans"
FALLBACK_CLOUD_MODEL = "fun-asr"
SHORT_CLOUD_MODEL = "qwen3-asr-flash"
THIRD_PASS_MODEL = "fun-asr-flash-2026-06-15"
LOCAL_SPEAKER_CLUSTERING_VERSION = "vad-event-complete-v2-diagnostics"


class RecordingTranscriptionError(RuntimeError):
    """Raised when the pipeline cannot safely produce a transcript."""


@dataclass(frozen=True)
class TranscriptionOptions:
    """Explicit authorization and output choices for one transcription run."""

    source_directory: Path
    output_path: Path
    model_directory: Path | None = None
    # Cloud ASR is the default quality path. Local Qwen3-ASR remains opt-in.
    run_local: bool = False
    run_cloud: bool = True
    allow_cloud_upload: bool = False
    resume: bool = True
    cost_cap_cny: float = 30.0
    chunk_minutes: int = 90
    overlap_seconds: int = 12
    run_third_pass: bool = True
    run_deepseek_text: bool = False
    deepseek_cost_cap_cny: float = 0.6
    expected_meetings: int | None = None
    speech_only_cloud: bool = False
    speech_region_gap_seconds: int = 30


def _ensure_model(model_id: str, directory: Path) -> Path:
    """Download a pinned local copy through ModelScope, then HF mirror fallback."""
    target = directory.expanduser().resolve()
    marker = target / ".download_complete"
    recognizable_model = target.is_dir() and (
        any(target.glob("*.safetensors")) or any(target.glob("*.bin")) or any(target.glob("*.pt"))
    )
    if marker.is_file() or recognizable_model:
        return target
    target.mkdir(parents=True, exist_ok=True)
    try:
        import modelscope  # type: ignore[import-untyped]

        modelscope.snapshot_download(model_id, local_dir=str(target))
        marker.touch()
        return target
    except Exception as modelscope_error:  # noqa: BLE001 - third-party download errors vary
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            from huggingface_hub import snapshot_download as huggingface_download

            huggingface_download(repo_id=model_id, local_dir=target)
            marker.touch()
            return target
        except Exception as huggingface_error:  # noqa: BLE001 - third-party errors vary
            raise RecordingTranscriptionError(
                f"could not download local model {model_id}"
            ) from ExceptionGroup("model download failures", [modelscope_error, huggingface_error])


def _chunk_path(root: Path, chunk: AudioChunk, *, enhanced: bool) -> Path:
    suffix = "enhanced" if enhanced else "light"
    return root / "chunks" / chunk.recording_id / f"{chunk.chunk_id}.{suffix}.flac"


def _load_cloud_cache(path: Path, *, expected_input_sha256: str) -> CloudResult | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("input_sha256") != expected_input_sha256:
        return None
    return CloudResult(
        model=str(payload["model"]),
        task_id=str(payload["task_id"]),
        billed_seconds=int(payload["billed_seconds"]),
        sentences=tuple(
            ProviderSentence(
                start_ms=int(sentence["start_ms"]),
                end_ms=int(sentence["end_ms"]),
                text=str(sentence["text"]),
                speaker_id=(
                    str(sentence["speaker_id"]) if sentence.get("speaker_id") is not None else None
                ),
                task_id=str(payload["task_id"]),
            )
            for sentence in payload["sentences"]
        ),
        sanitized_payload=cast(dict[str, Any], payload.get("provider_payload", {})),
    )


def _cloud_cache_payload(result: CloudResult, *, input_sha256: str) -> dict[str, Any]:
    return {
        "input_sha256": input_sha256,
        "model": result.model,
        "task_id": result.task_id,
        "billed_seconds": result.billed_seconds,
        "sentences": [
            {
                "start_ms": sentence.start_ms,
                "end_ms": sentence.end_ms,
                "text": sentence.text,
                "speaker_id": sentence.speaker_id,
            }
            for sentence in result.sentences
        ],
        "provider_payload": result.sanitized_payload,
    }


def _split_long_interval(
    start: int, end: int, *, maximum_ms: int = 45_000
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        next_end = min(cursor + maximum_ms, end)
        ranges.append((cursor, next_end))
        cursor = next_end
    return ranges


def _subtract_intervals(
    interval: tuple[int, int], occupied: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Return parts of an interval not already covered by provider evidence."""
    pieces = [interval]
    for occupied_start, occupied_end in sorted(occupied):
        next_pieces: list[tuple[int, int]] = []
        for start, end in pieces:
            if occupied_end <= start or occupied_start >= end:
                next_pieces.append((start, end))
                continue
            if start < occupied_start:
                next_pieces.append((start, occupied_start))
            if occupied_end < end:
                next_pieces.append((occupied_end, end))
        pieces = next_pieces
    return [(start, end) for start, end in pieces if end - start >= 500]


def _evidence_int(item: Mapping[str, object], key: str) -> int:
    value = item[key]
    if not isinstance(value, (str, int, float)):
        raise RecordingTranscriptionError(f"evidence field {key} is not numeric")
    return int(value)


def _unique_billed_seconds(results: Mapping[str, CloudResult]) -> int:
    """Count each provider request once even when a compatibility alias exists."""
    seen: set[str] = set()
    total = 0
    for key, result in results.items():
        request_key = result.task_id or f"anonymous:{key}"
        if request_key in seen:
            continue
        seen.add(request_key)
        total += result.billed_seconds
    return total


def _speaker_for_range(
    speaker_mapping: Mapping[str, str], recording_id: str, start_ms: int, end_ms: int
) -> str | None:
    """Resolve a sentence to its VAD event by midpoint, then overlap."""
    midpoint = (start_ms + end_ms) // 2
    matches: list[tuple[int, int, str]] = []
    for key, label in speaker_mapping.items():
        parts = key.split("|")
        if len(parts) != 3 or parts[0] != recording_id:
            continue
        try:
            event_start, event_end = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        overlap = min(end_ms, event_end) - max(start_ms, event_start)
        if event_start <= midpoint < event_end:
            return label
        if overlap > 0:
            matches.append((overlap, -event_start, label))
    best = max(matches, default=None)
    return best[2] if best is not None else None


class RecordingTranscriber:
    """Coordinate preprocessing, two ASR passes, speaker memory, and rendering."""

    def __init__(
        self,
        options: TranscriptionOptions,
        *,
        api_key: str | None = None,
        deepseek_api_key: str | None = None,
        deepseek_base_url: str = "https://api.deepseek.com",
        deepseek_model: str = "deepseek-v4-flash",
        deepseek_timeout_seconds: float = 60,
    ) -> None:
        self.options = options
        self.source_input = options.source_directory.expanduser().resolve()
        self.source_directory = (
            self.source_input.parent if self.source_input.is_file() else self.source_input
        )
        self.output_path = options.output_path.expanduser().resolve()
        self.derivatives_directory = (
            self.output_path.parent / "transcription"
            if self.source_input.is_file()
            else self.source_directory / "transcription"
        )
        self.work_directory = (
            self.output_path.parent / "transcription-local-only"
            if options.run_local and not options.run_cloud
            else self.derivatives_directory
        )
        self.models_directory = (
            options.model_directory.expanduser().resolve()
            if options.model_directory is not None
            else (
                Path.cwd().resolve() / "data" / "models" / "audio"
                if self.source_input.is_file()
                else self.derivatives_directory.parent / "models" / "audio"
            )
        )
        self.manifest_path = self.work_directory / "processing_manifest.json"
        self.transcript_path = self.work_directory / "transcript.json"
        self.text_analysis_path = self.work_directory / "text_analysis.json"
        self.topic_index_path = self.output_path.with_name("topic-index.md")
        self.api_key = api_key
        self.deepseek_api_key = deepseek_api_key
        self.deepseek_base_url = deepseek_base_url
        self.deepseek_model = deepseek_model
        self.deepseek_timeout_seconds = deepseek_timeout_seconds
        self._text_analysis: TranscriptTextAnalysis | None = None
        self._text_review_warning: str | None = None
        if options.run_cloud and not options.allow_cloud_upload:
            raise RecordingTranscriptionError("cloud mode requires --allow-cloud-upload")
        if options.run_cloud and not api_key:
            raise RecordingTranscriptionError("DASHSCOPE_API_KEY is required for cloud mode")
        if not options.run_cloud and not options.run_local:
            raise RecordingTranscriptionError("at least one transcription mode is required")
        if options.run_deepseek_text and not deepseek_api_key:
            raise RecordingTranscriptionError(
                "DEEPSEEK_API_KEY is required for DeepSeek text review"
            )
        if options.cost_cap_cny <= 0:
            raise RecordingTranscriptionError("cost cap must be positive")
        if options.deepseek_cost_cap_cny <= 0:
            raise RecordingTranscriptionError("DeepSeek cost cap must be positive")
        if options.speech_only_cloud and not options.run_cloud:
            raise RecordingTranscriptionError(
                "speech-only cloud mode requires cloud ASR"
            )
        if options.speech_region_gap_seconds < 0:
            raise RecordingTranscriptionError("speech region gap must not be negative")

    def _sources(self) -> tuple[RecordingSource, ...]:
        supported = {".m4a", ".mp3", ".wav", ".flac"}
        if self.source_input.is_file():
            files = [self.source_input] if self.source_input.suffix.casefold() in supported else []
        else:
            files = sorted(
                path
                for path in self.source_directory.iterdir()
                if path.is_file() and path.suffix.casefold() in supported
            )
        if not files:
            raise RecordingTranscriptionError("input contains no supported audio recordings")
        return tuple(inspect_recording(path, index=index) for index, path in enumerate(files, 1))

    def _validate_resume_sources(self, sources: tuple[RecordingSource, ...]) -> None:
        if not (self.options.resume and self.manifest_path.is_file()):
            return
        previous = ProcessingManifest.model_validate_json(self.manifest_path.read_bytes())
        current = {source.filename: source.sha256 for source in sources}
        if previous.source_hashes != current:
            raise RecordingTranscriptionError(
                "source recordings changed since the resumable manifest was created"
            )

    def _prepare_chunks(self, sources: tuple[RecordingSource, ...]) -> tuple[AudioChunk, ...]:
        prepared: list[AudioChunk] = []
        expected_chunks: dict[str, AudioChunk] = {}
        if self.options.resume and self.manifest_path.is_file():
            previous = ProcessingManifest.model_validate_json(self.manifest_path.read_bytes())
            expected_chunks = {chunk.chunk_id: chunk for chunk in previous.chunks}
        for source in sources:
            silence_cache = (
                self.derivatives_directory / "analysis" / f"{source.recording_id}.silence.json"
            )
            if self.options.resume and silence_cache.is_file():
                cached = json.loads(silence_cache.read_text(encoding="utf-8"))
                silence_points = tuple(int(value) for value in cached.get("midpoints_ms", ()))
            else:
                silence_points = detect_silence_midpoints(source)
                write_provider_payload(silence_cache, {"midpoints_ms": list(silence_points)})
            chunks = plan_chunks(
                source,
                target_ms=self.options.chunk_minutes * 60_000,
                overlap_ms=self.options.overlap_seconds * 1000,
                preferred_boundaries_ms=silence_points,
            )
            for chunk in chunks:
                enhanced_path = _chunk_path(self.derivatives_directory, chunk, enhanced=True)
                light_path = _chunk_path(self.derivatives_directory, chunk, enhanced=False)
                expected = expected_chunks.get(chunk.chunk_id)
                reusable_light = (
                    self.options.resume and light_path.is_file() and light_path.stat().st_size > 0
                )
                if not reusable_light:
                    light_hash = encode_chunk(source, chunk, light_path, enhanced=False)
                else:
                    light_hash = sha256_file(light_path)
                    if expected and light_hash != expected.light_sha256:
                        light_hash = encode_chunk(source, chunk, light_path, enhanced=False)
                if self.options.speech_only_cloud:
                    # Sparse recordings only need the light full-timeline copy for
                    # VAD/local ASR. Enhanced audio is produced later for the much
                    # smaller speech regions, avoiding a second 17-hour derivative.
                    enhanced_path = light_path
                    enhanced_hash = light_hash
                else:
                    reusable_enhanced = (
                        self.options.resume
                        and enhanced_path.is_file()
                        and enhanced_path.stat().st_size > 0
                    )
                    if not reusable_enhanced:
                        enhanced_hash = encode_chunk(source, chunk, enhanced_path, enhanced=True)
                    else:
                        enhanced_hash = sha256_file(enhanced_path)
                        if expected and enhanced_hash != expected.enhanced_sha256:
                            enhanced_hash = encode_chunk(
                                source, chunk, enhanced_path, enhanced=True
                            )
                prepared.append(
                    chunk.model_copy(
                        update={
                            "enhanced_path": str(enhanced_path),
                            "light_path": str(light_path),
                            "enhanced_sha256": enhanced_hash,
                            "light_sha256": light_hash,
                        }
                    )
                )
        return tuple(prepared)

    def _speech_cloud_chunks(
        self,
        sources: tuple[RecordingSource, ...],
        chunks: tuple[AudioChunk, ...],
    ) -> tuple[AudioChunk, ...]:
        """Encode only VAD-backed contiguous regions while retaining source offsets."""
        if not self.options.speech_only_cloud:
            return chunks
        vad_intervals = self._vad_intervals(chunks)
        regions: list[AudioChunk] = []
        source_by_id = {source.recording_id: source for source in sources}
        for source in sources:
            padded = [
                (max(0, start - 1_200), min(source.duration_ms, end + 1_200))
                for start, end in vad_intervals.get(source.recording_id, [])
            ]
            merged = union_intervals(
                padded,
                merge_gap_ms=self.options.speech_region_gap_seconds * 1_000,
            )
            split_regions = [
                part
                for start, end in merged
                for part in _split_long_interval(start, end, maximum_ms=90 * 60_000)
            ]
            for index, (start, end) in enumerate(split_regions, 1):
                region = AudioChunk(
                    chunk_id=f"{source.recording_id}_speech_{index:04d}",
                    recording_id=source.recording_id,
                    start_ms=start,
                    end_ms=end,
                )
                path = (
                    self.derivatives_directory
                    / "speech-regions"
                    / source.recording_id
                    / f"{region.chunk_id}.enhanced.flac"
                )
                if self.options.resume and path.is_file() and path.stat().st_size > 0:
                    digest = sha256_file(path)
                else:
                    digest = encode_chunk(
                        source_by_id[source.recording_id],
                        region,
                        path,
                        enhanced=True,
                    )
                regions.append(
                    region.model_copy(
                        update={
                            "enhanced_path": str(path),
                            "light_path": str(path),
                            "enhanced_sha256": digest,
                            "light_sha256": digest,
                        }
                    )
                )
        write_provider_payload(
            self.work_directory / "providers" / "cloud" / "speech-regions.json",
            {
                "vad_model": VAD_MODEL_ID,
                "merge_gap_seconds": self.options.speech_region_gap_seconds,
                "regions": [region.model_dump(mode="json") for region in regions],
            },
        )
        return tuple(regions)

    def _write_manifest(
        self,
        *,
        sources: tuple[RecordingSource, ...],
        chunks: tuple[AudioChunk, ...],
        status: str,
        cloud_model: str | None,
        cloud_results: dict[str, CloudResult],
        completed_steps: tuple[str, ...],
        errors: tuple[str, ...] = (),
    ) -> None:
        now = datetime.now(UTC)
        created = now
        if self.manifest_path.is_file():
            try:
                created = ProcessingManifest.model_validate_json(
                    self.manifest_path.read_bytes()
                ).created_at
            except Exception:  # noqa: BLE001 - corrupt legacy manifests fall back safely
                created = now
        billed = _unique_billed_seconds(cloud_results)
        text_usage = self._text_analysis.usage if self._text_analysis else ()
        text_requests = len(text_usage)
        text_input_tokens = sum(
            item.prompt_cache_hit_tokens + item.prompt_cache_miss_tokens for item in text_usage
        )
        text_output_tokens = sum(item.output_tokens for item in text_usage)
        text_cost = self._text_analysis.estimated_cost_cny if self._text_analysis else 0.0
        if self.options.run_deepseek_text:
            cache_directory = self.work_directory / "providers" / "deepseek"
            cached_payloads = []
            for cache_path in cache_directory.glob("*.json"):
                try:
                    payload = json.loads(cache_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if payload.get("model") == self.deepseek_model:
                    cached_payloads.append(payload)
            if cached_payloads:
                hit = sum(int(item.get("prompt_cache_hit_tokens", 0)) for item in cached_payloads)
                miss = sum(int(item.get("prompt_cache_miss_tokens", 0)) for item in cached_payloads)
                output = sum(int(item.get("output_tokens", 0)) for item in cached_payloads)
                text_requests = len(cached_payloads)
                text_input_tokens = hit + miss
                text_output_tokens = output
                text_cost = estimate_deepseek_cost_cny(hit=hit, miss=miss, output=output)
        manifest = ProcessingManifest(
            status=cast(Any, status),
            created_at=created,
            updated_at=now,
            source_hashes={source.filename: source.sha256 for source in sources},
            chunks=chunks,
            local_model=LOCAL_MODEL_ID,
            cloud_model=cloud_model,
            cloud_fallback_model=FALLBACK_CLOUD_MODEL,
            cloud_task_ids={key: result.task_id for key, result in cloud_results.items()},
            cloud_billed_seconds=billed,
            estimated_cost_cny=round(billed * PRICE_PER_SECOND_CNY + text_cost, 4),
            cost_cap_cny=self.options.cost_cap_cny,
            text_review_model=(self._text_analysis.model if self._text_analysis else None),
            text_review_requests=text_requests,
            text_review_input_tokens=text_input_tokens,
            text_review_output_tokens=text_output_tokens,
            text_review_cost_cny=round(text_cost, 6),
            text_review_cost_cap_cny=(
                self.options.deepseek_cost_cap_cny if self.options.run_deepseek_text else 0.0
            ),
            completed_steps=completed_steps,
            errors=((self._text_review_warning,) if self._text_review_warning else ()) + errors,
        )
        write_json_model(manifest, self.manifest_path)

    def _cloud_pass(
        self, chunks: tuple[AudioChunk, ...]
    ) -> tuple[str | None, dict[str, CloudResult], list[dict[str, object]]]:
        if not self.options.run_cloud:
            return None, {}, []
        if not chunks:
            return PRIMARY_CLOUD_MODEL, {}, []
        assert self.api_key is not None
        provider = DashScopeFileTranscriber(api_key=self.api_key)
        flash_provider = DashScopeFlashTranscriber(api_key=self.api_key)
        pilot_cache = self.work_directory / "providers" / "cloud" / "pilot.json"
        historical_pilot = None
        if self.options.resume and pilot_cache.is_file():
            historical_pilot = _load_cloud_cache(
                pilot_cache,
                expected_input_sha256=str(
                    json.loads(pilot_cache.read_text(encoding="utf-8")).get("input_sha256", "")
                ),
            )
        existing: dict[str, CloudResult] = {}
        for chunk in chunks:
            if self.options.resume:
                cached = _load_cloud_cache(
                    self.work_directory / "providers" / "cloud" / f"{chunk.chunk_id}.json",
                    expected_input_sha256=cast(str, chunk.enhanced_sha256),
                )
                if cached is not None:
                    existing[chunk.chunk_id] = cached
        historical_seconds = historical_pilot.billed_seconds if historical_pilot else 0
        conservative_total_seconds = historical_seconds + sum(
            cached.billed_seconds
            if (cached := existing.get(chunk.chunk_id)) is not None
            else (chunk.duration_ms + 999) // 1000
            for chunk in chunks
        )
        if conservative_total_seconds * PRICE_PER_SECOND_CNY > self.options.cost_cap_cny:
            raise RecordingTranscriptionError(
                "all planned cloud speech regions would exceed the cost cap"
            )
        model = historical_pilot.model if historical_pilot else PRIMARY_CLOUD_MODEL
        results: dict[str, CloudResult] = {}
        if historical_pilot is not None:
            # Retain the historical request in accounting, never submit it again.
            results["pilot"] = historical_pilot
        raw_sentences: list[dict[str, object]] = []
        for index, chunk in enumerate(chunks):
            cache_path = self.work_directory / "providers" / "cloud" / f"{chunk.chunk_id}.json"
            input_hash = cast(str, chunk.enhanced_sha256)
            result = existing.get(chunk.chunk_id)
            if result is None:
                projected = _unique_billed_seconds(results) + (chunk.duration_ms + 999) // 1000
                if projected * PRICE_PER_SECOND_CNY > self.options.cost_cap_cny:
                    raise RecordingTranscriptionError("cloud cost cap would be exceeded")
                chunk_path = Path(cast(str, chunk.enhanced_path))
                if model == SHORT_CLOUD_MODEL:
                    result = self._short_cloud_chunk(chunk, flash_provider)
                else:
                    try:
                        result = provider.transcribe(
                            chunk_path,
                            model=model,
                            diarization=not model.startswith("qwen3-asr"),
                            context_text="Office recording in the source language. It may contain phone calls, commute noise, and a few English terms. Transcribe faithfully sentence by sentence; do not translate.",
                        )
                    except TranscriptionProviderError as primary_error:
                        if index != 0 or historical_pilot is not None:
                            raise
                        model = FALLBACK_CLOUD_MODEL
                        try:
                            result = provider.transcribe(chunk_path, model=model, diarization=True)
                        except TranscriptionProviderError as fallback_error:
                            model = SHORT_CLOUD_MODEL
                            try:
                                result = self._short_cloud_chunk(chunk, flash_provider)
                            except TranscriptionProviderError as short_error:
                                raise RecordingTranscriptionError(
                                    "all cloud ASR models failed their first formal chunk"
                                ) from ExceptionGroup(
                                    "cloud model selection failures",
                                    [primary_error, fallback_error, short_error],
                                )
                write_provider_payload(
                    cache_path,
                    _cloud_cache_payload(result, input_sha256=input_hash),
                )
            results[chunk.chunk_id] = result
            if index == 0 and historical_pilot is None:
                # Compatibility alias to the same formal result, not a request.
                results["pilot"] = result
            for provider_sentence in result.sentences:
                raw_sentences.append(
                    {
                        "recording_id": chunk.recording_id,
                        "chunk_id": chunk.chunk_id,
                        "start_ms": chunk.start_ms + provider_sentence.start_ms,
                        "end_ms": chunk.start_ms + provider_sentence.end_ms,
                        "text": provider_sentence.text,
                        "speaker_id": provider_sentence.speaker_id,
                        "task_id": provider_sentence.task_id,
                        "model": result.model,
                    }
                )
        deduped: list[dict[str, object]] = []
        by_recording: dict[str, list[dict[str, object]]] = defaultdict(list)
        for evidence_item in raw_sentences:
            by_recording[str(evidence_item["recording_id"])].append(evidence_item)
        for recording_id in sorted(by_recording):
            deduped.extend(deduplicate_overlap(by_recording[recording_id]))
        return model, results, deduped

    def _short_cloud_chunk(
        self, chunk: AudioChunk, provider: DashScopeFlashTranscriber
    ) -> CloudResult:
        """Last-resort <=5-minute cloud transcription when file models are unavailable."""
        sentences: list[ProviderSentence] = []
        request_ids: list[str] = []
        payloads: list[dict[str, Any]] = []
        for start in range(0, chunk.duration_ms, 5 * 60_000):
            end = min(start + 5 * 60_000, chunk.duration_ms)
            clip = (
                self.work_directory
                / "providers"
                / "cloud-short-clips"
                / f"{chunk.chunk_id}_{start:010d}.flac"
            )
            if not (self.options.resume and clip.is_file()):
                extract_clip(
                    Path(cast(str, chunk.enhanced_path)),
                    clip,
                    start_ms=start,
                    end_ms=end,
                    wav=False,
                )
            result = provider.transcribe(
                clip,
                model=SHORT_CLOUD_MODEL,
                context_text="Office recording in the source language. Transcribe faithfully, keep spoken language and numbers, and do not translate.",
            )
            request_ids.append(result.request_id)
            payloads.append(result.sanitized_payload)
            if result.text:
                sentences.append(
                    ProviderSentence(
                        start_ms=start,
                        end_ms=end,
                        text=result.text,
                        task_id=result.request_id,
                    )
                )
        return CloudResult(
            model=SHORT_CLOUD_MODEL,
            task_id=",".join(request_ids),
            billed_seconds=(chunk.duration_ms + 999) // 1000,
            sentences=tuple(sentences),
            sanitized_payload={"results": payloads},
        )

    def _vad_intervals(self, chunks: tuple[AudioChunk, ...]) -> dict[str, list[tuple[int, int]]]:
        cache_path = self.work_directory / "providers" / "local" / "vad-intervals.json"
        input_hashes = {chunk.chunk_id: chunk.light_sha256 for chunk in chunks}
        if self.options.resume and cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("model") == VAD_MODEL_ID and cached.get("input_hashes") == input_hashes:
                return {
                    str(recording_id): [(int(interval[0]), int(interval[1])) for interval in values]
                    for recording_id, values in cached.get("intervals", {}).items()
                }
        vad_directory = _ensure_model(VAD_MODEL_ID, self.models_directory / "fsmn-vad")
        try:
            from funasr import AutoModel  # type: ignore[import-untyped]
        except ImportError as error:
            raise RecordingTranscriptionError("funasr is required for conservative VAD") from error
        model = AutoModel(model=str(vad_directory), device="cpu", disable_update=True)
        intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for chunk in chunks:
            result = model.generate(input=cast(str, chunk.light_path))
            values = result[0].get("value", []) if result else []
            effective_start = chunk.start_ms + chunk.overlap_before_ms
            for value in values:
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    continue
                start = max(effective_start, chunk.start_ms + int(value[0]) - 800)
                end = min(chunk.end_ms, chunk.start_ms + int(value[1]) + 800)
                if end - start >= 500:
                    intervals[chunk.recording_id].append((start, end))
        del model
        gc.collect()
        merged = {key: union_intervals(value) for key, value in intervals.items()}
        write_provider_payload(
            cache_path,
            {
                "model": VAD_MODEL_ID,
                "input_hashes": input_hashes,
                "intervals": {
                    key: [list(value) for value in values] for key, values in merged.items()
                },
            },
        )
        return merged

    def _local_pass(
        self,
        sources: tuple[RecordingSource, ...],
        chunks: tuple[AudioChunk, ...],
        cloud_sentences: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if not self.options.run_local:
            return []
        cache_path = self.work_directory / "providers" / "local" / "qwen3-asr.json"
        if self.options.resume and cache_path.is_file():
            return cast(
                list[dict[str, object]],
                json.loads(cache_path.read_text(encoding="utf-8"))["sentences"],
            )
        model_directory = _ensure_model(LOCAL_MODEL_ID, self.models_directory / "Qwen3-ASR-1.7B")
        vad_intervals = self._vad_intervals(chunks)
        chunks_by_recording: dict[str, list[AudioChunk]] = defaultdict(list)
        for chunk in chunks:
            chunks_by_recording[chunk.recording_id].append(chunk)
        requested: list[tuple[str, int, int, str | None]] = []
        cloud_ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for cloud_sentence in cloud_sentences:
            recording_id = str(cloud_sentence["recording_id"])
            start = _evidence_int(cloud_sentence, "start_ms")
            end = _evidence_int(cloud_sentence, "end_ms")
            cloud_ref = (
                f"{cloud_sentence['chunk_id']}:"
                f"{cloud_sentence['start_ms']}:{cloud_sentence['end_ms']}"
            )
            requested.append((recording_id, start, end, cloud_ref))
            cloud_ranges[recording_id].append((start, end))
        # VAD can add speech missed by cloud ASR, but it never deletes or coalesces
        # the exact cloud sentence ranges used for model-to-model comparison.
        for source in sources:
            occupied = union_intervals(cloud_ranges[source.recording_id], merge_gap_ms=0)
            for interval in vad_intervals.get(source.recording_id, []):
                for uncovered in _subtract_intervals(interval, occupied):
                    maximum_ms = 45_000 if cloud_sentences else 15_000
                    for start, end in _split_long_interval(*uncovered, maximum_ms=maximum_ms):
                        requested.append((source.recording_id, start, end, None))
        requested.sort(key=lambda item: (item[0], item[1], item[2]))
        clips: list[tuple[str, Path, int, int, str | None]] = []
        for source in sources:
            for recording_id, start, end, requested_ref in requested:
                if recording_id != source.recording_id:
                    continue
                owner = next(
                    (
                        chunk
                        for chunk in chunks_by_recording[recording_id]
                        if chunk.start_ms <= start and chunk.end_ms >= end
                    ),
                    None,
                )
                if owner is None:
                    continue
                kind = "cloud" if requested_ref else "vad"
                clip_path = (
                    self.work_directory
                    / "local-clips"
                    / recording_id
                    / f"{start:010d}_{end:010d}.{kind}.wav"
                )
                if not (self.options.resume and clip_path.is_file()):
                    extract_clip(
                        Path(cast(str, owner.light_path)),
                        clip_path,
                        start_ms=start - owner.start_ms,
                        end_ms=end - owner.start_ms,
                    )
                clips.append((recording_id, clip_path, start, end, requested_ref))
        provider = LocalQwenTranscriber(model_directory=model_directory)
        provider_sentences = provider.transcribe(
            [(path, start, end) for _, path, start, end, _ in clips]
        )
        provider.unload()
        local: list[dict[str, object]] = []
        for (recording_id, _, _, _, recognized_ref), recognized in zip(
            clips, provider_sentences, strict=True
        ):
            if not recognized.text and recognized_ref is None:
                continue
            local.append(
                {
                    "recording_id": recording_id,
                    "start_ms": recognized.start_ms,
                    "end_ms": recognized.end_ms,
                    "text": recognized.text,
                    "model": LOCAL_MODEL_ID,
                    "cloud_ref": recognized_ref,
                }
            )
        write_provider_payload(cache_path, {"model": LOCAL_MODEL_ID, "sentences": local})
        return local

    def _local_only_speaker_mapping(
        self,
        sources: tuple[RecordingSource, ...],
        local_sentences: list[dict[str, object]],
        chunks: tuple[AudioChunk, ...] = (),
    ) -> tuple[dict[str, str], tuple[SpeakerProfile, ...]]:
        if chunks and not local_sentences and not self.options.speech_only_cloud:
            return {}, ()
        source_order = {source.recording_id: index for index, source in enumerate(sources)}
        event_clips: list[tuple[str, str, int, int, tuple[Path, ...]]] = []
        if chunks:
            intervals = self._vad_intervals(chunks)
            for source in sources:
                for start_ms, end_ms in intervals.get(source.recording_id, []):
                    if end_ms - start_ms < 3_000:
                        continue
                    windows: list[tuple[int, int]] = []
                    if end_ms - start_ms <= 6_000:
                        windows.append((start_ms, end_ms))
                    else:
                        cursor = start_ms
                        while cursor + 6_000 <= end_ms:
                            windows.append((cursor, cursor + 6_000))
                            cursor += 3_000
                        if windows and windows[-1][1] < end_ms:
                            windows.append((end_ms - 6_000, end_ms))
                    if len(windows) > 5:
                        picks = np.linspace(0, len(windows) - 1, 5, dtype=int).tolist()
                        windows = [windows[index] for index in picks]
                    event_key = f"{source.recording_id}|{start_ms}|{end_ms}"
                    clip_paths: list[Path] = []
                    for _index, (window_start, window_end) in enumerate(windows):
                        clip = (
                            self.work_directory
                            / "speaker-events"
                            / source.recording_id
                            / (
                                f"{start_ms:010d}_{end_ms:010d}_"
                                f"{window_start:010d}_{window_end:010d}.wav"
                            )
                        )
                        if not (self.options.resume and clip.is_file()):
                            extract_clip(
                                source.source_path,
                                clip,
                                start_ms=window_start,
                                end_ms=window_end,
                            )
                        clip_paths.append(clip)
                    event_clips.append(
                        (
                            event_key,
                            source.recording_id,
                            start_ms,
                            end_ms,
                            tuple(clip_paths),
                        )
                    )
        else:
            # Compatibility for callers that only have sentence-level local
            # evidence; the full pipeline always supplies VAD chunks.
            for sentence in local_sentences:
                recording_id = str(sentence["recording_id"])
                start_ms = _evidence_int(sentence, "start_ms")
                end_ms = _evidence_int(sentence, "end_ms")
                if not end_ms - start_ms >= 3_000:
                    continue
                kind = "cloud" if sentence.get("cloud_ref") is not None else "vad"
                clip = (
                    self.work_directory
                    / "local-clips"
                    / recording_id
                    / f"{start_ms:010d}_{end_ms:010d}.{kind}.wav"
                )
                if clip.is_file():
                    event_clips.append(
                        (
                            f"{recording_id}|{start_ms}|{end_ms}",
                            recording_id,
                            start_ms,
                            end_ms,
                            (clip,),
                        )
                    )
        signature_payload = {
            "version": LOCAL_SPEAKER_CLUSTERING_VERSION,
            "sources": [(source.recording_id, source.sha256) for source in sources],
            "events": [
                (key, recording_id, start_ms, end_ms, [str(path) for path in clips])
                for key, recording_id, start_ms, end_ms, clips in event_clips
            ],
            "window_ms": 6_000,
            "step_ms": 3_000,
            "max_windows": 5,
        }
        input_signature = hashlib.sha256(
            json.dumps(signature_payload, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        cache_path = self.work_directory / "providers" / "local" / "speakers.json"
        if self.options.resume and cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("input_signature") == input_signature
                and cached.get("clustering_version") == LOCAL_SPEAKER_CLUSTERING_VERSION
            ):
                return (
                    {str(key): str(value) for key, value in cached.get("mapping", {}).items()},
                    tuple(
                        SpeakerProfile.model_validate(profile)
                        for profile in cached.get("profiles", ())
                    ),
                )
        speaker_directory = _ensure_model(SPEAKER_MODEL_ID, self.models_directory / "ERes2NetV2")
        samples: list[LocalSpeakerSamples] = []
        for key, recording_id, start_ms, _end_ms, clips in event_clips:
            samples.append(
                LocalSpeakerSamples(
                    key=key,
                    recording_id=recording_id,
                    first_order=(source_order[recording_id], start_ms),
                    clips=clips,
                )
            )
        if not samples:
            return {}, ()
        candidate_diagnostics: list[dict[str, object]] = []
        mapping, profiles = resolve_local_segments(
            tuple(samples),
            embedder=SpeakerEmbedder(model_directory=speaker_directory),
            embedding_cache_path=(
                self.work_directory / "providers" / "local" / "speaker-embeddings.json"
            ),
            cache_metadata={
                "model": SPEAKER_MODEL_ID,
                "source_hashes": {source.recording_id: source.sha256 for source in sources},
                "window_ms": 6_000,
                "step_ms": 3_000,
                "max_windows": 5,
            },
            diagnostics=candidate_diagnostics,
        )
        write_provider_payload(
            cache_path,
            {
                "model": SPEAKER_MODEL_ID,
                "clustering_version": LOCAL_SPEAKER_CLUSTERING_VERSION,
                "input_signature": input_signature,
                "mapping": mapping,
                "profiles": [profile.model_dump(mode="json") for profile in profiles],
                "candidates": candidate_diagnostics,
            },
        )
        return mapping, profiles

    def _speaker_mapping(
        self,
        sources: tuple[RecordingSource, ...],
        chunks: tuple[AudioChunk, ...],
        cloud_sentences: list[dict[str, object]],
    ) -> tuple[dict[str, str], tuple[SpeakerProfile, ...]]:
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        source_order = {source.recording_id: index for index, source in enumerate(sources)}
        for sentence in cloud_sentences:
            speaker_id = sentence.get("speaker_id")
            if speaker_id is None:
                continue
            key = f"{sentence['recording_id']}|{sentence['chunk_id']}|{speaker_id}"
            grouped[key].append(sentence)
        input_signature = hashlib.sha256(
            json.dumps(
                [
                    {
                        "recording_id": item["recording_id"],
                        "chunk_id": item["chunk_id"],
                        "start_ms": item["start_ms"],
                        "end_ms": item["end_ms"],
                        "speaker_id": item.get("speaker_id"),
                        "text": item["text"],
                    }
                    for item in cloud_sentences
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cache_path = self.work_directory / "providers" / "local" / "cloud-speakers.json"
        if self.options.resume and cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("model") == SPEAKER_MODEL_ID
                and cached.get("input_signature") == input_signature
            ):
                return (
                    {str(key): str(value) for key, value in cached.get("mapping", {}).items()},
                    tuple(
                        SpeakerProfile.model_validate(profile)
                        for profile in cached.get("profiles", [])
                    ),
                )
        speaker_directory = _ensure_model(SPEAKER_MODEL_ID, self.models_directory / "ERes2NetV2")
        sample_sets: list[LocalSpeakerSamples] = []
        for key, values in grouped.items():
            eligible = sorted(
                (
                    value
                    for value in values
                    if 3_000
                    <= _evidence_int(value, "end_ms") - _evidence_int(value, "start_ms")
                    <= 15_000
                ),
                key=lambda value: _evidence_int(value, "end_ms") - _evidence_int(value, "start_ms"),
                reverse=True,
            )[:3]
            if not eligible:
                continue
            clips: list[Path] = []
            for index, sentence in enumerate(eligible, 1):
                chunk = chunks_by_id[str(sentence["chunk_id"])]
                clip = (
                    self.work_directory / "speaker-clips" / f"{key.replace('|', '_')}_{index}.wav"
                )
                if not (self.options.resume and clip.is_file()):
                    extract_clip(
                        Path(cast(str, chunk.light_path)),
                        clip,
                        start_ms=max(0, _evidence_int(sentence, "start_ms") - chunk.start_ms),
                        end_ms=min(
                            chunk.duration_ms,
                            _evidence_int(sentence, "end_ms") - chunk.start_ms,
                        ),
                    )
                clips.append(clip)
            recording_id = str(eligible[0]["recording_id"])
            sample_sets.append(
                LocalSpeakerSamples(
                    key=key,
                    recording_id=recording_id,
                    first_order=(
                        source_order[recording_id],
                        _evidence_int(eligible[0], "start_ms"),
                    ),
                    clips=tuple(clips),
                )
            )
        if not sample_sets:
            mapping: dict[str, str] = {}
            profiles: list[SpeakerProfile] = []
        else:
            resolved_mapping, resolved_profiles = resolve_speakers(
                tuple(sample_sets),
                embedder=SpeakerEmbedder(model_directory=speaker_directory),
            )
            mapping = dict(resolved_mapping)
            profiles = list(resolved_profiles)
        unresolved = sorted(
            (key for key in grouped if key not in mapping),
            key=lambda key: (
                source_order[str(grouped[key][0]["recording_id"])],
                min(_evidence_int(item, "start_ms") for item in grouped[key]),
            ),
        )
        for key in unresolved:
            values = grouped[key]
            label = anonymous_speaker_label(len(profiles))
            recording_id = str(values[0]["recording_id"])
            mapping[key] = label
            profiles.append(
                SpeakerProfile(
                    speaker=label,
                    local_speaker_keys=(key,),
                    recording_ids=(recording_id,),
                    sample_count=0,
                    confidence=0.25,
                )
            )
        result_profiles = tuple(profiles)
        write_provider_payload(
            cache_path,
            {
                "model": SPEAKER_MODEL_ID,
                "input_signature": input_signature,
                "mapping": mapping,
                "profiles": [profile.model_dump(mode="json") for profile in result_profiles],
            },
        )
        return mapping, result_profiles

    def _third_pass(
        self,
        sources: tuple[RecordingSource, ...],
        chunks: tuple[AudioChunk, ...],
        cloud_sentences: list[dict[str, object]],
        local_sentences: list[dict[str, object]],
        cloud_results: dict[str, CloudResult],
    ) -> dict[str, dict[str, object]]:
        """Recheck disagreements with bounded concurrency and a 10% duration cap."""
        if not self.options.run_cloud or not self.options.run_third_pass:
            return {}
        assert self.api_key is not None
        api_key = self.api_key
        local_by_ref = {
            str(item["cloud_ref"]): item
            for item in local_sentences
            if item.get("cloud_ref") is not None
        }
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        allowance_ms = sum(source.duration_ms for source in sources) // 10
        reviewed_ms = 0
        base_billed_seconds = _unique_billed_seconds(cloud_results)
        selected: list[tuple[dict[str, object], str, str, int]] = []
        for cloud in cloud_sentences:
            cloud_ref = f"{cloud['chunk_id']}:{cloud['start_ms']}:{cloud['end_ms']}"
            local = local_by_ref.get(cloud_ref)
            if local is None:
                continue
            cloud_text = str(cloud["text"])
            local_text = str(local["text"])
            if text_agreement(cloud_text, local_text) >= 0.95 and not has_sensitive_disagreement(
                cloud_text, local_text
            ):
                continue
            duration_ms = _evidence_int(cloud, "end_ms") - _evidence_int(cloud, "start_ms")
            if duration_ms <= 0 or duration_ms > 300_000:
                continue
            if reviewed_ms + duration_ms > allowance_ms:
                break
            projected_seconds = base_billed_seconds + (reviewed_ms + duration_ms + 999) // 1000
            if projected_seconds * PRICE_PER_SECOND_CNY > self.options.cost_cap_cny:
                break
            selected.append((cloud, cloud_ref, local_text, duration_ms))
            reviewed_ms += duration_ms

        def recognize(
            item: tuple[dict[str, object], str, str, int],
        ) -> tuple[dict[str, object], str, int, dict[str, Any]]:
            cloud, cloud_ref, local_text, duration_ms = item
            cloud_text = str(cloud["text"])
            safe_ref = cloud_ref.replace(":", "_")
            cache_path = self.work_directory / "providers" / "third-pass" / f"{safe_ref}.json"
            if self.options.resume and cache_path.is_file():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                payload = cast(dict[str, Any], cached)
            else:
                chunk = chunk_by_id[str(cloud["chunk_id"])]
                clip_path = self.work_directory / "third-pass-clips" / f"{safe_ref}.mp3"
                if not (
                    self.options.resume and clip_path.is_file() and clip_path.stat().st_size > 0
                ):
                    extract_clip(
                        Path(cast(str, chunk.light_path)),
                        clip_path,
                        start_ms=_evidence_int(cloud, "start_ms") - chunk.start_ms,
                        end_ms=_evidence_int(cloud, "end_ms") - chunk.start_ms,
                        wav=False,
                    )
                last_error: TranscriptionProviderError | None = None
                for attempt in range(3):
                    try:
                        result = DashScopeFlashTranscriber(api_key=api_key).transcribe(
                            clip_path,
                            model=THIRD_PASS_MODEL,
                            context_text=(
                                f"Review the office recording in its source language. Candidate one: {cloud_text}; candidate two: {local_text}. Do not translate."
                            ),
                        )
                        break
                    except TranscriptionProviderError as error:
                        last_error = error
                        if attempt == 2:
                            raise
                        time.sleep(2**attempt)
                else:  # pragma: no cover - the retry loop either succeeds or raises
                    assert last_error is not None
                    raise last_error
                payload = {
                    "model": result.model,
                    "request_id": result.request_id,
                    "text": result.text,
                    "provider_payload": result.sanitized_payload,
                }
                write_provider_payload(cache_path, payload)
            return cloud, cloud_ref, duration_ms, payload

        evidence: dict[str, dict[str, object]] = {}
        with ThreadPoolExecutor(max_workers=6, thread_name_prefix="third-pass") as executor:
            recognized = executor.map(recognize, selected)
            for cloud, cloud_ref, duration_ms, payload in recognized:
                evidence[cloud_ref] = {
                    "model": str(payload["model"]),
                    "request_id": str(payload.get("request_id", "")),
                    "text": str(payload.get("text", "")),
                    "recording_id": str(cloud["recording_id"]),
                    "start_ms": _evidence_int(cloud, "start_ms"),
                    "end_ms": _evidence_int(cloud, "end_ms"),
                }
                cloud_results[f"third:{cloud_ref}"] = CloudResult(
                    model=str(payload["model"]),
                    task_id=str(payload.get("request_id", "")),
                    billed_seconds=(duration_ms + 999) // 1000,
                    sentences=(),
                    sanitized_payload=cast(dict[str, Any], payload.get("provider_payload", {})),
                )
        return evidence

    def _adjudicate(
        self,
        sources: tuple[RecordingSource, ...],
        cloud_sentences: list[dict[str, object]],
        local_sentences: list[dict[str, object]],
        third_pass: dict[str, dict[str, object]],
        speaker_mapping: dict[str, str],
    ) -> tuple[TranscriptSegment, ...]:
        local_by_recording: dict[str, list[dict[str, object]]] = defaultdict(list)
        local_by_cloud_ref: dict[str, dict[str, object]] = {}
        for sentence in local_sentences:
            local_by_recording[str(sentence["recording_id"])].append(sentence)
            cloud_ref = sentence.get("cloud_ref")
            if cloud_ref is not None:
                local_by_cloud_ref[str(cloud_ref)] = sentence
        provider_group_votes: dict[str, Counter[str]] = defaultdict(Counter)
        for cloud in cloud_sentences:
            if cloud.get("speaker_id") is None:
                continue
            group_key = f"{cloud['recording_id']}|{cloud['chunk_id']}|{cloud['speaker_id']}"
            cloud_ref = f"{cloud['chunk_id']}:{cloud['start_ms']}:{cloud['end_ms']}"
            local = local_by_cloud_ref.get(cloud_ref)
            if local is None:
                continue
            label = _speaker_for_range(
                speaker_mapping,
                str(local["recording_id"]),
                _evidence_int(local, "start_ms"),
                _evidence_int(local, "end_ms"),
            )
            if label is not None:
                provider_group_votes[group_key][label] += max(
                    1, _evidence_int(local, "end_ms") - _evidence_int(local, "start_ms")
                )
        provider_group_mapping: dict[str, str] = {}
        for group_key, votes in provider_group_votes.items():
            ranked = votes.most_common(2)
            if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                provider_group_mapping[group_key] = ranked[0][0]
        source_order = {source.recording_id: index for index, source in enumerate(sources)}
        segments: list[TranscriptSegment] = []
        for index, cloud in enumerate(
            sorted(
                cloud_sentences,
                key=lambda value: (
                    source_order[str(value["recording_id"])],
                    _evidence_int(value, "start_ms"),
                ),
            ),
            1,
        ):
            cloud_ref = f"{cloud['chunk_id']}:{cloud['start_ms']}:{cloud['end_ms']}"
            exact_item = local_by_cloud_ref.get(cloud_ref)
            exact = [exact_item] if exact_item is not None else []
            overlaps = exact or [
                local
                for local in local_by_recording[str(cloud["recording_id"])]
                if _evidence_int(local, "start_ms") < _evidence_int(cloud, "end_ms")
                and _evidence_int(local, "end_ms") > _evidence_int(cloud, "start_ms")
            ]
            local_text = "".join(str(item["text"]) for item in overlaps)
            cloud_text = str(cloud["text"])
            third = third_pass.get(cloud_ref)
            if exact and not local_text and third is not None and not str(third.get("text", "")):
                # Independent local and third-pass models both found no words;
                # suppress a likely single-provider hallucination.
                continue
            emitted_text = cloud_text
            agreement = text_agreement(cloud_text, local_text) if local_text else 0.0
            sensitive = has_sensitive_disagreement(cloud_text, local_text) if local_text else False
            if agreement >= 0.95 and not sensitive:
                decision = "cloud_agreement"
                confidence = 0.95
                flags: tuple[str, ...] = ()
            elif local_text:
                third_text = str(third.get("text", "")) if third else ""
                agrees_cloud = bool(third_text) and text_agreement(third_text, cloud_text) >= 0.95
                agrees_local = bool(third_text) and text_agreement(third_text, local_text) >= 0.95
                if agrees_cloud and not has_sensitive_disagreement(third_text, cloud_text):
                    decision = "third_pass_majority"
                    confidence = 0.9
                    flags = ("models_disagree", "third_pass_cloud_majority")
                elif agrees_local and not has_sensitive_disagreement(third_text, local_text):
                    decision = "third_pass_majority"
                    confidence = 0.9
                    emitted_text = local_text
                    flags = ("models_disagree", "third_pass_local_majority")
                else:
                    decision = "unclear"
                    confidence = max(0.3, min(0.49, agreement))
                    emitted_text = f"[Uncertain: {cloud_text}]" if cloud_text else "[Unclear]"
                    flags = ("models_disagree", "no_majority") + (
                        ("sensitive_difference",) if sensitive else ()
                    )
            else:
                decision = "cloud_primary"
                confidence = 0.6
                flags = ("local_missing",)
            if cloud.get("speaker_id") is not None:
                speaker_key = (
                    f"{cloud['recording_id']}|{cloud['chunk_id']}|{cloud.get('speaker_id')}"
                )
                speaker = speaker_mapping.get(
                    speaker_key, provider_group_mapping.get(speaker_key, "")
                )
            else:
                speaker = ""
                if self.options.speech_only_cloud:
                    speaker = _speaker_for_range(
                        speaker_mapping,
                        str(cloud["recording_id"]),
                        _evidence_int(cloud, "start_ms"),
                        _evidence_int(cloud, "end_ms"),
                    ) or ""
            mapped_overlaps = [
                item
                for item in overlaps
                if _speaker_for_range(
                    speaker_mapping,
                    str(item["recording_id"]),
                    _evidence_int(item, "start_ms"),
                    _evidence_int(item, "end_ms"),
                )
            ]
            if not speaker and mapped_overlaps:
                strongest = max(
                    mapped_overlaps,
                    key=lambda item: (
                        min(_evidence_int(item, "end_ms"), _evidence_int(cloud, "end_ms"))
                        - max(_evidence_int(item, "start_ms"), _evidence_int(cloud, "start_ms"))
                    ),
                )
                speaker_key = (
                    f"{strongest['recording_id']}|{strongest['start_ms']}|{strongest['end_ms']}"
                )
                speaker = (
                    _speaker_for_range(
                        speaker_mapping,
                        str(strongest["recording_id"]),
                        _evidence_int(strongest, "start_ms"),
                        _evidence_int(strongest, "end_ms"),
                    )
                    or ""
                )
            speaker = speaker or "Speaker (unconfirmed)"
            candidates = [
                ProviderCandidate(
                    provider="cloud",
                    model=str(cloud["model"]),
                    text=cloud_text,
                    start_ms=_evidence_int(cloud, "start_ms"),
                    end_ms=_evidence_int(cloud, "end_ms"),
                    speaker_id=(
                        str(cloud["speaker_id"]) if cloud.get("speaker_id") is not None else None
                    ),
                    task_id=(str(cloud["task_id"]) if cloud.get("task_id") else None),
                )
            ]
            if local_text:
                candidates.append(
                    ProviderCandidate(
                        provider="local",
                        model=LOCAL_MODEL_ID,
                        text=local_text,
                        start_ms=min(_evidence_int(item, "start_ms") for item in overlaps),
                        end_ms=max(_evidence_int(item, "end_ms") for item in overlaps),
                    )
                )
            if third and str(third.get("text", "")):
                candidates.append(
                    ProviderCandidate(
                        provider="third_pass",
                        model=str(third["model"]),
                        text=str(third["text"]),
                        start_ms=_evidence_int(cloud, "start_ms"),
                        end_ms=_evidence_int(cloud, "end_ms"),
                        task_id=str(third.get("request_id", "")) or None,
                    )
                )
            segments.append(
                TranscriptSegment(
                    segment_id=f"segment_{index:07d}",
                    recording_id=str(cloud["recording_id"]),
                    start_ms=_evidence_int(cloud, "start_ms"),
                    end_ms=_evidence_int(cloud, "end_ms"),
                    speaker=speaker,
                    text=emitted_text,
                    decision=cast(Any, decision),
                    confidence=confidence,
                    flags=flags,
                    candidates=tuple(candidates),
                )
            )
        for local in local_sentences:
            if local.get("cloud_ref") is not None:
                continue
            local_text = str(local.get("text", "")).strip()
            if not local_text:
                continue
            start_ms = _evidence_int(local, "start_ms")
            end_ms = _evidence_int(local, "end_ms")
            recording_id = str(local["recording_id"])
            resolved_speaker = _speaker_for_range(speaker_mapping, recording_id, start_ms, end_ms)
            speaker = resolved_speaker or "Speaker (unconfirmed)"
            speaker_flags = () if speaker != "Speaker (unconfirmed)" else ("speaker_unconfirmed",)
            segments.append(
                TranscriptSegment(
                    segment_id="local_supplement",
                    recording_id=str(local["recording_id"]),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speaker=speaker,
                    text=f"[Uncertain: {local_text}]",
                    decision="local_primary",
                    confidence=0.4,
                    flags=("local_only", "vad_supplement", *speaker_flags),
                    candidates=(
                        ProviderCandidate(
                            provider="local",
                            model=str(local.get("model", LOCAL_MODEL_ID)),
                            text=local_text,
                            start_ms=start_ms,
                            end_ms=end_ms,
                        ),
                    ),
                )
            )
        ordered = sorted(
            segments,
            key=lambda segment: (
                source_order[segment.recording_id],
                segment.start_ms,
                segment.end_ms,
            ),
        )
        return tuple(
            segment.model_copy(update={"segment_id": f"segment_{index:07d}"})
            for index, segment in enumerate(ordered, 1)
        )

    def _adjudicate_local_only(
        self,
        sources: tuple[RecordingSource, ...],
        local_sentences: list[dict[str, object]],
        speaker_mapping: dict[str, str],
    ) -> tuple[TranscriptSegment, ...]:
        """Create a local-only transcript without pretending single-model consensus."""
        source_order = {source.recording_id: index for index, source in enumerate(sources)}
        segments: list[TranscriptSegment] = []
        for local in sorted(
            local_sentences,
            key=lambda value: (
                source_order[str(value["recording_id"])],
                _evidence_int(value, "start_ms"),
                _evidence_int(value, "end_ms"),
            ),
        ):
            text = str(local.get("text", "")).strip()
            if not text:
                continue
            recording_id = str(local["recording_id"])
            start_ms = _evidence_int(local, "start_ms")
            end_ms = _evidence_int(local, "end_ms")
            speaker = _speaker_for_range(speaker_mapping, recording_id, start_ms, end_ms)
            speaker = speaker or "Speaker (unconfirmed)"
            flags = ("local_only",) + (
                ("speaker_unconfirmed",) if speaker == "Speaker (unconfirmed)" else ()
            )
            segments.append(
                TranscriptSegment(
                    segment_id="local_only",
                    recording_id=recording_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speaker=speaker,
                    text=text,
                    decision="local_primary",
                    confidence=0.7,
                    flags=flags,
                    candidates=(
                        ProviderCandidate(
                            provider="local",
                            model=str(local.get("model", LOCAL_MODEL_ID)),
                            text=text,
                            start_ms=start_ms,
                            end_ms=end_ms,
                        ),
                    ),
                )
            )
        return tuple(
            segment.model_copy(update={"segment_id": f"segment_{index:07d}"})
            for index, segment in enumerate(segments, 1)
        )

    def _review_text(
        self,
        document: TranscriptDocument,
        cloud_results: Mapping[str, CloudResult],
    ) -> TranscriptDocument:
        """Run the bounded context-only review and persist its independent topic layer."""
        if not self.options.run_deepseek_text:
            return document
        assert self.deepseek_api_key is not None
        asr_cost = _unique_billed_seconds(cloud_results) * PRICE_PER_SECOND_CNY
        remaining = self.options.cost_cap_cny - asr_cost
        review_cap = min(self.options.deepseek_cost_cap_cny, remaining)
        if review_cap <= 0:
            self._text_review_warning = "Text review fallback: total cost budget is insufficient"
            return document
        reviewer = DeepSeekTranscriptReviewer(
            api_key=self.deepseek_api_key,
            base_url=self.deepseek_base_url,
            model=self.deepseek_model,
            timeout_seconds=self.deepseek_timeout_seconds,
            cost_cap_cny=review_cap,
            cache_directory=self.work_directory / "providers" / "deepseek",
            expected_meeting_count=self.options.expected_meetings,
            allow_consolidation_fallback=True,
            allow_window_cost_fallback=True,
        )
        try:
            reviewed, analysis = reviewer.review(document)
        except TranscriptTextReviewError as error:
            self._text_review_warning = f"Text review fallback: {error}"
            return document
        if asr_cost + analysis.estimated_cost_cny > self.options.cost_cap_cny:
            self._text_review_warning = "Text review fallback: combined ASR and text review cost exceeds the cap"
            return document
        self._text_review_warning = (
            f"Text review complete: {analysis.window_fallback_count} windows kept the original recognition because of the cost cap"
            if analysis.window_fallback_count
            else None
        )
        self._text_analysis = analysis
        write_json_model(analysis, self.text_analysis_path)
        write_text_atomic(render_topic_markdown(reviewed, analysis), self.topic_index_path)
        return reviewed

    def run(self) -> TranscriptDocument:
        """Execute or resume the complete transcription workflow."""
        self.work_directory.mkdir(parents=True, exist_ok=True)
        sources = self._sources()
        self._validate_resume_sources(sources)
        chunks = self._prepare_chunks(sources)
        cloud_chunks = self._speech_cloud_chunks(sources, chunks)
        cloud_model = None
        cloud_results: dict[str, CloudResult] = {}
        completed_steps_list = ["preprocess"]
        self._write_manifest(
            sources=sources,
            chunks=chunks,
            status="running",
            cloud_model=cloud_model,
            cloud_results=cloud_results,
            completed_steps=tuple(completed_steps_list),
        )
        try:
            cloud_model, cloud_results, cloud_sentences = self._cloud_pass(cloud_chunks)
            if self.options.run_cloud:
                completed_steps_list.append("cloud")
            self._write_manifest(
                sources=sources,
                chunks=chunks,
                status="running",
                cloud_model=cloud_model,
                cloud_results=cloud_results,
                completed_steps=tuple(completed_steps_list),
            )
            local_sentences = self._local_pass(sources, chunks, cloud_sentences)
            if self.options.run_local:
                completed_steps_list.append("local")
            self._write_manifest(
                sources=sources,
                chunks=chunks,
                status="running",
                cloud_model=cloud_model,
                cloud_results=cloud_results,
                completed_steps=tuple(completed_steps_list),
            )
            third_pass = self._third_pass(
                sources, cloud_chunks, cloud_sentences, local_sentences, cloud_results
            )
            if self.options.run_cloud and cloud_sentences:
                if self.options.speech_only_cloud:
                    # Provider IDs restart for every sparse upload, so treating
                    # them as identities would create one apparent person per
                    # region. Cluster the continuous local clips instead.
                    speaker_mapping, profiles = self._local_only_speaker_mapping(
                        sources, local_sentences, chunks
                    )
                elif any(sentence.get("speaker_id") is not None for sentence in cloud_sentences):
                    speaker_mapping, profiles = self._speaker_mapping(
                        sources, cloud_chunks, cloud_sentences
                    )
                else:
                    speaker_mapping, profiles = self._local_only_speaker_mapping(
                        sources, local_sentences, chunks
                    )
                segments = self._adjudicate(
                    sources,
                    cloud_sentences,
                    local_sentences,
                    third_pass,
                    speaker_mapping,
                )
            else:
                speaker_mapping, profiles = self._local_only_speaker_mapping(
                    sources, local_sentences, chunks
                )
                segments = self._adjudicate_local_only(sources, local_sentences, speaker_mapping)
            completed_steps_list.append("speakers")
            self._write_manifest(
                sources=sources,
                chunks=chunks,
                status="running",
                cloud_model=cloud_model,
                cloud_results=cloud_results,
                completed_steps=tuple(completed_steps_list),
            )
            base_title = (
                Path(sources[0].filename).stem + " recording transcript"
                if len(sources) == 1
                else "Multi-day office recording transcript"
            )
            document = TranscriptDocument(
                title=(base_title + " (local only)" if not self.options.run_cloud else base_title),
                generated_at=datetime.now(UTC),
                sources=sources,
                speakers=profiles,
                segments=segments,
            )
            document = self._review_text(document, cloud_results)
            if self.options.run_deepseek_text and self._text_review_warning is None:
                completed_steps_list.append("deepseek_text")
                self._write_manifest(
                    sources=sources,
                    chunks=chunks,
                    status="running",
                    cloud_model=cloud_model,
                    cloud_results=cloud_results,
                    completed_steps=tuple(completed_steps_list),
                )
            write_json_model(document, self.transcript_path)
            write_text_atomic(
                render_markdown(
                    document,
                    structured_path=(
                        "transcription-local-only/transcript.json"
                        if not self.options.run_cloud
                        else "transcription/transcript.json"
                    ),
                ),
                self.output_path,
            )
            completed_steps_list.append("render")
            completed_steps = tuple(completed_steps_list)
            self._write_manifest(
                sources=sources,
                chunks=chunks,
                status="completed",
                cloud_model=cloud_model,
                cloud_results=cloud_results,
                completed_steps=completed_steps,
            )
            return document
        except BaseException as error:
            self._write_manifest(
                sources=sources,
                chunks=chunks,
                status="failed",
                cloud_model=cloud_model,
                cloud_results=cloud_results,
                completed_steps=tuple(completed_steps_list),
                errors=(str(error),),
            )
            raise
