"""Deterministic tests for long-recording transcription invariants."""


from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from research_kb.audio.core import (
    deduplicate_overlap,
    format_elapsed,
    has_sensitive_disagreement,
    inspect_recording,
    plan_chunks,
    read_mp4_timeline,
    render_markdown,
    sanitize_external_payload,
    text_agreement,
    union_intervals,
)
from research_kb.audio.domain import (
    RecordingSource,
    TranscriptDocument,
    TranscriptSegment,
)
from research_kb.audio.speaker import (
    LocalSpeakerSamples,
    anonymous_speaker_label,
    resolve_local_segments,
    resolve_speakers,
)


def _atom(kind: bytes, payload: bytes) -> bytes:
    size = len(payload) + 8
    return size.to_bytes(4, "big") + kind + payload


def _source(*, duration_ms: int = 10_000, recorded_at: datetime | None = None) -> RecordingSource:
    return RecordingSource(
        recording_id="recording_01",
        path="recording.m4a",
        filename="recording.m4a",
        sha256="a" * 64,
        size_bytes=100,
        duration_ms=duration_ms,
        recorded_at=recorded_at or datetime(2026, 8, 3, 8, 33, tzinfo=timezone(timedelta(hours=8))),
    )


def test_reads_real_mvhd_atom_and_ignores_fake_media_bytes(tmp_path: Path) -> None:
    created = int(
        (
            datetime(2026, 8, 3, 0, 33, 58, tzinfo=UTC) - datetime(1904, 1, 1, tzinfo=UTC)
        ).total_seconds()
    )
    mvhd = (
        b"\0\0\0\0"
        + created.to_bytes(4, "big")
        + created.to_bytes(4, "big")
        + (1_000).to_bytes(4, "big")
        + (31_908_160).to_bytes(4, "big")
        + bytes(80)
    )
    fake = b"noise-mvhd" + bytes(50)
    path = tmp_path / "sample.m4a"
    path.write_bytes(
        _atom(b"ftyp", b"M4A ") + _atom(b"mdat", fake) + _atom(b"moov", _atom(b"mvhd", mvhd))
    )

    recorded_at, duration_ms = read_mp4_timeline(path)

    assert recorded_at.isoformat() == "2026-08-03T08:33:58+08:00"
    assert duration_ms == 31_908_160


def test_inspects_non_mp4_audio_and_reads_filename_clock(tmp_path: Path) -> None:
    import soundfile

    path = tmp_path / "录音机-20260805-1320.wav"
    soundfile.write(path, np.zeros(16_000, dtype=np.float32), 16_000)

    source = inspect_recording(path, index=1)

    assert source.duration_ms == 1_000
    assert source.sample_rate_hz == 16_000
    assert source.channels == 1
    assert source.recorded_at.isoformat() == "2026-08-05T13:20:00+08:00"


def test_chunk_plan_covers_timeline_and_prefers_nearby_silence() -> None:
    source = _source(duration_ms=11_000)
    chunks = plan_chunks(
        source,
        target_ms=5_000,
        overlap_ms=1_000,
        preferred_boundaries_ms=(4_800, 9_100),
        boundary_window_ms=300,
    )

    assert [(item.start_ms, item.end_ms) for item in chunks] == [
        (0, 4_800),
        (3_800, 9_100),
        (8_100, 11_000),
    ]
    assert chunks[1].overlap_before_ms == 1_000
    assert all(left.end_ms - right.start_ms == 1_000 for left, right in pairwise(chunks))


def test_overlap_dedup_alignment_and_numeric_conflict() -> None:
    items: list[dict[str, object]] = [
        {"start_ms": 1_000, "end_ms": 2_000, "text": "预算是 120 元。"},
        {"start_ms": 1_500, "end_ms": 2_100, "text": "预算是120元"},
        {"start_ms": 3_000, "end_ms": 4_000, "text": "预算是 210 元。"},
    ]

    assert len(deduplicate_overlap(items)) == 2
    assert text_agreement("你好，世界！", "你好 世界") == 1
    assert has_sensitive_disagreement("预算 120 元", "预算 210 元")
    assert not has_sensitive_disagreement("预算 120 元", "共 120 元")

    noisy_overlap: list[dict[str, object]] = [
        {
            "chunk_id": "chunk_1",
            "start_ms": 5_387_470,
            "end_ms": 5_390_230,
            "text": "你是你你这个东西是空白的。",
        },
        {
            "chunk_id": "chunk_2",
            "start_ms": 5_388_560,
            "end_ms": 5_390_080,
            "text": "你你这个东西是空白的。",
        },
    ]
    deduplicated = deduplicate_overlap(noisy_overlap)
    assert len(deduplicated) == 1
    assert deduplicated[0]["text"] == "你是你你这个东西是空白的。"


def test_vad_ranges_merge_only_with_configured_gap() -> None:
    assert union_intervals([(100, 300), (250, 500), (900, 1_100)], merge_gap_ms=399) == [
        (100, 500),
        (900, 1_100),
    ]
    assert union_intervals([(100, 300), (900, 1_100)], merge_gap_ms=600) == [(100, 1_100)]


def test_markdown_uses_absolute_clock_and_compresses_long_silence() -> None:
    source = _source(duration_ms=30 * 60_000)
    document = TranscriptDocument(
        generated_at=datetime.now(UTC),
        sources=(source,),
        speakers=(),
        segments=(
            TranscriptSegment(
                segment_id="segment_1",
                recording_id=source.recording_id,
                start_ms=12 * 60_000,
                end_ms=12 * 60_000 + 5_000,
                speaker="说话人 A",
                text="嗯，开始吧。",
                decision="cloud_agreement",
                confidence=0.95,
            ),
        ),
    )

    markdown = render_markdown(document)

    assert "[08:33:00–08:45:00] [无可辨识语音]" in markdown
    assert "[08:45:00–08:45:05] 说话人 A：嗯，开始吧。" in markdown
    assert "[08:45:05–09:03:00] [无可辨识语音]" in markdown
    assert format_elapsed(source.duration_ms) == "00:30:00"


def test_signed_urls_and_credentials_are_never_persisted() -> None:
    payload = {
        "file_url": "oss://private/file.flac",
        "authorization": "Bearer secret",
        "nested": {
            "transcription_url": "https://example/result?Signature=secret",
            "message": "https://example/x?Expires=1&Signature=secret",
        },
    }

    cleaned = sanitize_external_payload(payload)

    assert cleaned == {"nested": {"message": "https://example/x?[redacted]&[redacted]"}}
    assert "secret" not in str(cleaned)


class _FakeEmbedder:
    def __init__(self, vectors: dict[str, np.ndarray[Any, np.dtype[np.float32]]]) -> None:
        self.vectors = vectors

    def embed(self, clips: tuple[Path, ...]) -> np.ndarray[Any, np.dtype[np.float32]]:
        return self.vectors[clips[0].stem]


def test_speaker_clustering_is_stable_and_requires_multiple_cross_day_samples() -> None:
    assert anonymous_speaker_label(26) == "说话人 AA"
    samples = (
        LocalSpeakerSamples("day1|0", "day1", (0, 10), (Path("alice.wav"), Path("a2.wav"))),
        LocalSpeakerSamples("day2|0", "day2", (1, 10), (Path("alice2.wav"), Path("a3.wav"))),
        LocalSpeakerSamples("visitor|0", "day2", (1, 20), (Path("visitor.wav"),)),
    )
    embedder = _FakeEmbedder(
        {
            "alice": np.array([1.0, 0.0], dtype=np.float32),
            "alice2": np.array([0.99, 0.01], dtype=np.float32),
            "visitor": np.array([1.0, 0.0], dtype=np.float32),
        }
    )

    mapping, profiles = resolve_speakers(samples, embedder=embedder)  # type: ignore[arg-type]

    assert mapping["day1|0"] == mapping["day2|0"] == "说话人 A"
    assert mapping["visitor|0"] == "说话人 B"
    assert profiles[0].recording_ids == ("day1", "day2")
    assert profiles[0].sample_count == 4


def test_local_event_clustering_does_not_chain_merge_adjacent_voices(tmp_path: Path) -> None:
    """Complete linkage keeps a transitive similarity chain split."""
    samples = tuple(
        LocalSpeakerSamples(
            key=name,
            recording_id="recording",
            first_order=(0, index),
            clips=(tmp_path / f"{name}.wav",),
        )
        for index, name in enumerate(("a1", "a2", "b1"))
    )

    class Embedder:
        def embed(self, clips: tuple[Path, ...]) -> np.ndarray[Any, np.dtype[np.float32]]:
            vectors = {
                "a1": np.array([1.0, 0.0], dtype=np.float32),
                "a2": np.array([0.94, 0.34], dtype=np.float32),
                "b1": np.array([0.64, 0.77], dtype=np.float32),
            }
            return vectors[clips[0].stem]

    mapping, profiles = resolve_local_segments(samples, embedder=Embedder())  # type: ignore[arg-type]

    assert mapping["a1"] == mapping["a2"]
    assert "b1" not in mapping
    assert profiles[0].confidence >= 0.8


def test_local_event_diagnostics_keep_external_similarity_and_margin(
    tmp_path: Path,
) -> None:
    names = ("a1", "a2", "b1", "b2")
    samples = tuple(
        LocalSpeakerSamples(
            key=f"recording|{index * 10_000}|{index * 10_000 + 6_000}",
            recording_id="recording",
            first_order=(0, index),
            clips=(tmp_path / f"{name}.wav",),
        )
        for index, name in enumerate(names)
    )
    vectors = {
        "a1": np.array([1.0, 0.0], dtype=np.float32),
        "a2": np.array([0.99, 0.1], dtype=np.float32),
        "b1": np.array([0.0, 1.0], dtype=np.float32),
        "b2": np.array([0.1, 0.99], dtype=np.float32),
    }

    class Embedder:
        def embed(self, clips: tuple[Path, ...]) -> np.ndarray[Any, np.dtype[np.float32]]:
            return vectors[clips[0].stem]

    diagnostics: list[dict[str, object]] = []
    mapping, _ = resolve_local_segments(
        samples,
        embedder=Embedder(),
        diagnostics=diagnostics,
    )  # type: ignore[arg-type]

    accepted = [item for item in diagnostics if item["accepted"]]
    assert len(accepted) == 2
    assert mapping[samples[0].key] == mapping[samples[1].key]
    assert mapping[samples[2].key] == mapping[samples[3].key]
    for item in accepted:
        assert item["event_ranges"]
        assert item["window_count"] == 2
        assert isinstance(item["nearest_external_similarity"], float)
        assert isinstance(item["similarity_margin"], float)
        assert item["similarity_margin"] >= 0.12


def test_local_event_diagnostics_identify_single_and_mixed_events(tmp_path: Path) -> None:
    samples = (
        LocalSpeakerSamples(
            key="recording|0|6000",
            recording_id="recording",
            first_order=(0, 0),
            clips=(tmp_path / "single.wav",),
        ),
        LocalSpeakerSamples(
            key="recording|10000|22000",
            recording_id="recording",
            first_order=(0, 1),
            clips=(tmp_path / "mixed_a.wav", tmp_path / "mixed_b.wav"),
        ),
    )

    class Embedder:
        def embed(self, clips: tuple[Path, ...]) -> np.ndarray[Any, np.dtype[np.float32]]:
            return (
                np.array([1.0, 0.0], dtype=np.float32)
                if clips[0].stem in {"single", "mixed_a"}
                else np.array([0.0, 1.0], dtype=np.float32)
            )

    diagnostics: list[dict[str, object]] = []
    mapping, profiles = resolve_local_segments(
        samples,
        embedder=Embedder(),
        diagnostics=diagnostics,
    )  # type: ignore[arg-type]

    assert mapping == {}
    assert profiles == ()
    reasons = {str(item["reason"]) for item in diagnostics}
    assert "single_event_insufficient_evidence" in reasons
    assert "mixed_event_window_similarity_below_0.70" in reasons


def test_local_event_diagnostics_reject_low_intra_cluster_similarity(tmp_path: Path) -> None:
    samples = tuple(
        LocalSpeakerSamples(
            key=f"recording|{index * 10_000}|{index * 10_000 + 6_000}",
            recording_id="recording",
            first_order=(0, index),
            clips=(tmp_path / f"event{index}.wav",),
        )
        for index in range(2)
    )
    vectors = (
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.7, np.sqrt(0.51)], dtype=np.float32),
    )

    class Embedder:
        def embed(self, clips: tuple[Path, ...]) -> np.ndarray[Any, np.dtype[np.float32]]:
            return vectors[int(clips[0].stem.removeprefix("event"))]

    diagnostics: list[dict[str, object]] = []
    mapping, profiles = resolve_local_segments(
        samples,
        embedder=Embedder(),
        within_recording_distance=0.40,
        diagnostics=diagnostics,
    )  # type: ignore[arg-type]

    assert mapping == {}
    assert profiles == ()
    assert len(diagnostics) == 1
    candidate = diagnostics[0]
    assert candidate["accepted"] is False
    assert candidate["reason"] == "intra_cluster_similarity_below_0.80"
    assert candidate["intra_cluster_min_similarity"] == pytest.approx(0.7, abs=1e-6)
    assert candidate["event_ranges"]


def test_local_event_diagnostics_reject_insufficient_similarity_margin(tmp_path: Path) -> None:
    sine = np.sqrt(0.19)
    a1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    a2 = np.array([0.9, sine, 0.0, 0.0], dtype=np.float32)
    b1 = np.array(
        [0.85, (0.75 - 0.765) / sine, np.sqrt(1 - 0.85**2 - ((0.75 - 0.765) / sine) ** 2), 0.0],
        dtype=np.float32,
    )
    b2 = np.array(
        [0.75, (0.85 - 0.675) / sine, b1[2], 0.0],
        dtype=np.float32,
    )
    vectors = {
        name: vector
        for name, vector in zip(("a1", "a2", "b1", "b2"), (a1, a2, b1, b2), strict=True)
    }
    samples = tuple(
        LocalSpeakerSamples(
            key=f"recording|{index * 10_000}|{index * 10_000 + 6_000}",
            recording_id="recording",
            first_order=(0, index),
            clips=(tmp_path / f"{name}.wav",),
        )
        for index, name in enumerate(vectors)
    )

    class Embedder:
        def embed(self, clips: tuple[Path, ...]) -> np.ndarray[Any, np.dtype[np.float32]]:
            return vectors[clips[0].stem]

    diagnostics: list[dict[str, object]] = []
    mapping, profiles = resolve_local_segments(
        samples,
        embedder=Embedder(),
        diagnostics=diagnostics,
    )  # type: ignore[arg-type]

    assert mapping == {}
    assert profiles == ()
    assert len(diagnostics) == 2
    for candidate in diagnostics:
        assert candidate["accepted"] is False
        assert candidate["reason"] == "similarity_margin_below_0.12"
        assert candidate["intra_cluster_min_similarity"] == pytest.approx(0.9, abs=1e-6)
        assert candidate["nearest_external_similarity"] == pytest.approx(0.85, abs=1e-6)
        assert candidate["similarity_margin"] == pytest.approx(0.05, abs=1e-6)


def test_invalid_chunk_and_out_of_bounds_transcript_are_rejected() -> None:
    source = _source()
    with pytest.raises(ValueError, match="chunk target"):
        plan_chunks(source, target_ms=1_000, overlap_ms=1_000)
    with pytest.raises(ValueError, match="exceeds source duration"):
        TranscriptDocument(
            generated_at=datetime.now(UTC),
            sources=(source,),
            speakers=(),
            segments=(
                TranscriptSegment(
                    segment_id="bad",
                    recording_id=source.recording_id,
                    start_ms=9_000,
                    end_ms=11_000,
                    speaker="说话人 ?",
                    text="[听不清]",
                    decision="unclear",
                    confidence=0.1,
                ),
            ),
        )

