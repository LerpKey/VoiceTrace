"""Pure, testable audio-transcription planning and rendering logic."""


from __future__ import annotations

import hashlib
import mmap
import re
import string
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

from research_kb.audio.domain import (
    AudioChunk,
    RecordingSource,
    TranscriptDocument,
    TranscriptSegment,
)

_MP4_EPOCH = datetime(1904, 1, 1, tzinfo=UTC)
_CHINA_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")
_COMPARISON_IGNORED = str.maketrans(
    "", "", string.whitespace + string.punctuation + "，。！？；：、（）【】《》“”‘’…—"
)
_SIGNED_QUERY_KEYS = re.compile(
    r"(?i)(?:Expires|OSSAccessKeyId|Signature|x-oss-signature)=[^&\s\"']+"
)


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mp4_timeline(path: Path) -> tuple[datetime, int]:
    """Read creation time and duration from an MP4/M4A mvhd atom."""
    with (
        path.open("rb") as source,
        mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data,
    ):
        position = _find_mvhd_payload(data)
        if position is None or position + 32 > len(data):
            raise ValueError(f"missing mvhd atom: {path}")
        version = data[position]
        if version == 0:
            created_seconds = int.from_bytes(data[position + 4 : position + 8], "big")
            timescale = int.from_bytes(data[position + 12 : position + 16], "big")
            duration_units = int.from_bytes(data[position + 16 : position + 20], "big")
        elif version == 1:
            created_seconds = int.from_bytes(data[position + 4 : position + 12], "big")
            timescale = int.from_bytes(data[position + 20 : position + 24], "big")
            duration_units = int.from_bytes(data[position + 24 : position + 32], "big")
        else:
            raise ValueError(f"unsupported mvhd version {version}: {path}")
    if timescale <= 0 or duration_units <= 0:
        raise ValueError(f"invalid mvhd timing values: {path}")
    recorded_at = (_MP4_EPOCH + timedelta(seconds=created_seconds)).astimezone(_CHINA_TIMEZONE)
    duration_ms = round(duration_units * 1000 / timescale)
    return recorded_at, duration_ms


def _atom_ranges(data: mmap.mmap, start: int, end: int) -> Iterable[tuple[bytes, int, int]]:
    """Yield validated ISO BMFF atoms without matching bytes inside media data."""
    cursor = start
    while cursor + 8 <= end:
        size = int.from_bytes(data[cursor : cursor + 4], "big")
        kind = bytes(data[cursor + 4 : cursor + 8])
        header = 8
        if size == 1:
            if cursor + 16 > end:
                return
            size = int.from_bytes(data[cursor + 8 : cursor + 16], "big")
            header = 16
        elif size == 0:
            size = end - cursor
        atom_end = cursor + size
        if size < header or atom_end > end:
            return
        yield kind, cursor + header, atom_end
        cursor = atom_end


def _find_mvhd_payload(data: mmap.mmap) -> int | None:
    for kind, payload, atom_end in _atom_ranges(data, 0, len(data)):
        if kind != b"moov":
            continue
        for child_kind, child_payload, _ in _atom_ranges(data, payload, atom_end):
            if child_kind == b"mvhd":
                return child_payload
    return None


def inspect_recording(path: Path, *, index: int) -> RecordingSource:
    """Build an immutable source record from a supported audio file."""
    resolved = path.expanduser().resolve()
    if resolved.suffix.casefold() in {".m4a", ".mp4"}:
        recorded_at, duration_ms = read_mp4_timeline(resolved)
        codec = "aac"
        sample_rate_hz = 48_000
        channels = 2
    else:
        try:
            import soundfile  # type: ignore[import-untyped]

            info = soundfile.info(str(resolved))
        except Exception as error:
            raise ValueError(f"could not inspect audio stream: {resolved}") from error
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"audio stream has no duration: {resolved}")
        duration_ms = round(info.frames * 1000 / info.samplerate)
        sample_rate_hz = int(info.samplerate)
        channels = int(info.channels)
        codec = str(info.subtype or info.format or resolved.suffix.lstrip(".")).casefold()
        recorded_at = _recorded_at_from_filename(resolved)
    return RecordingSource(
        recording_id=f"recording_{index:02d}",
        path=str(resolved),
        filename=resolved.name,
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
        duration_ms=duration_ms,
        recorded_at=recorded_at,
        codec=codec,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
    )


def _recorded_at_from_filename(path: Path) -> datetime:
    """Prefer an embedded YYYYMMDD-HHMM[SS] filename timestamp for non-MP4 audio."""
    match = re.search(r"(?<!\d)(20\d{6})[-_ ]?(\d{4}(?:\d{2})?)(?!\d)", path.stem)
    if match:
        compact = match.group(1) + match.group(2)
        pattern = "%Y%m%d%H%M%S" if len(compact) == 14 else "%Y%m%d%H%M"
        return datetime.strptime(compact, pattern).replace(tzinfo=_CHINA_TIMEZONE)
    return datetime.fromtimestamp(path.stat().st_mtime, tz=_CHINA_TIMEZONE)


def plan_chunks(
    source: RecordingSource,
    *,
    target_ms: int = 90 * 60 * 1000,
    overlap_ms: int = 12_000,
    preferred_boundaries_ms: Sequence[int] = (),
    boundary_window_ms: int = 30_000,
) -> tuple[AudioChunk, ...]:
    """Create complete coverage chunks, preferring nearby silence boundaries."""
    if target_ms <= overlap_ms or overlap_ms < 0:
        raise ValueError("chunk target must exceed non-negative overlap")
    preferred = sorted(set(preferred_boundaries_ms))
    chunks: list[AudioChunk] = []
    start = 0
    index = 1
    while start < source.duration_ms:
        ideal_end = min(start + target_ms, source.duration_ms)
        end = ideal_end
        if ideal_end < source.duration_ms:
            candidates = [
                value
                for value in preferred
                if abs(value - ideal_end) <= boundary_window_ms and value > start + overlap_ms
            ]
            if candidates:
                end = min(candidates, key=lambda value: abs(value - ideal_end))
        overlap_before = 0 if not chunks else overlap_ms
        chunks.append(
            AudioChunk(
                chunk_id=f"{source.recording_id}_chunk_{index:03d}",
                recording_id=source.recording_id,
                start_ms=start,
                end_ms=end,
                overlap_before_ms=overlap_before,
            )
        )
        if end >= source.duration_ms:
            break
        start = end - overlap_ms
        index += 1
    return tuple(chunks)


def normalize_for_comparison(text: str) -> str:
    """Normalize only for comparison; emitted transcript text remains untouched."""
    return text.casefold().translate(_COMPARISON_IGNORED)


def text_agreement(left: str, right: str) -> float:
    """Return a deterministic character similarity score."""
    normalized_left = normalize_for_comparison(left)
    normalized_right = normalize_for_comparison(right)
    if not normalized_left and not normalized_right:
        return 1.0
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def has_sensitive_disagreement(left: str, right: str) -> bool:
    """Treat numeric differences as requiring independent confirmation."""
    return re.findall(r"\d+(?:\.\d+)?", left) != re.findall(r"\d+(?:\.\d+)?", right)


def _object_int(value: object) -> int:
    if not isinstance(value, (str, int, float)):
        raise TypeError("timeline value must be numeric")
    return int(value)


def deduplicate_overlap(items: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Remove repeated cloud sentences emitted in adjacent chunk overlaps."""
    kept: list[dict[str, object]] = []
    for item in sorted(
        items,
        key=lambda value: (
            _object_int(value["start_ms"]),
            _object_int(value["end_ms"]),
        ),
    ):
        duplicate = False
        for kept_index in range(len(kept) - 1, max(-1, len(kept) - 9), -1):
            previous = kept[kept_index]
            previous_chunk = previous.get("chunk_id")
            current_chunk = item.get("chunk_id")
            if (
                previous_chunk is not None
                and current_chunk is not None
                and previous_chunk == current_chunk
            ):
                continue
            delta = abs(_object_int(previous["start_ms"]) - _object_int(item["start_ms"]))
            overlap = min(_object_int(previous["end_ms"]), _object_int(item["end_ms"])) - max(
                _object_int(previous["start_ms"]), _object_int(item["start_ms"])
            )
            shorter_duration = min(
                _object_int(previous["end_ms"]) - _object_int(previous["start_ms"]),
                _object_int(item["end_ms"]) - _object_int(item["start_ms"]),
            )
            temporal_match = delta <= 2_500 or (
                shorter_duration > 0 and overlap / shorter_duration >= 0.5
            )
            agreement = text_agreement(str(previous["text"]), str(item["text"]))
            sensitive_conflict = has_sensitive_disagreement(
                str(previous["text"]), str(item["text"])
            )
            if temporal_match and agreement >= 0.72 and not sensitive_conflict:
                if len(normalize_for_comparison(str(item["text"]))) > len(
                    normalize_for_comparison(str(previous["text"]))
                ):
                    kept[kept_index] = item
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return sorted(
        kept,
        key=lambda value: (
            _object_int(value["start_ms"]),
            _object_int(value["end_ms"]),
        ),
    )


def sanitize_external_payload(value: object) -> object:
    """Remove signed URLs and secret-like query material before persistence."""
    if isinstance(value, dict):
        cleaned: dict[str, object] = {}
        for key, child in value.items():
            if key.lower() in {"file_url", "transcription_url", "api_key", "authorization"}:
                continue
            cleaned[key] = sanitize_external_payload(child)
        return cleaned
    if isinstance(value, list):
        return [sanitize_external_payload(child) for child in value]
    if isinstance(value, str):
        return _SIGNED_QUERY_KEYS.sub("[redacted]", value)
    return value


def format_elapsed(milliseconds: int) -> str:
    """Format an elapsed source timestamp as HH:MM:SS."""
    total_seconds = max(0, milliseconds // 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render_markdown(
    document: TranscriptDocument,
    *,
    long_gap_ms: int = 10 * 60 * 1000,
    structured_path: str = "transcription/transcript.json",
) -> str:
    """Render a faithful human transcript from the structured authority."""
    lines = [f"# {document.title}", "", "## 说明", ""]
    lines.extend(
        [
            "- 本文按录音时间顺序逐句转写，保留口头语、重复和不完整句。",
            "- 说话人标签基于匿名声纹聚类；无法可靠判断的内容使用明确标记。",
            f"- 结构化事实、候选文本和模型溯源保存在 `{structured_path}`。",
            "",
        ]
    )
    segments_by_recording: dict[str, list[TranscriptSegment]] = {
        source.recording_id: [] for source in document.sources
    }
    for segment in document.segments:
        segments_by_recording[segment.recording_id].append(segment)
    for source in document.sources:
        local_segments = segments_by_recording[source.recording_id]
        end_at = source.recorded_at + timedelta(milliseconds=source.duration_ms)
        lines.extend(
            [
                f"## {source.recorded_at:%Y-%m-%d}",
                "",
                f"录音：`{source.filename}`  ",
                f"时间：{source.recorded_at:%H:%M:%S}–{end_at:%H:%M:%S}  ",
                f"时长：{format_elapsed(source.duration_ms)}",
                "",
            ]
        )
        cursor = 0
        for segment in local_segments:
            if segment.start_ms - cursor >= long_gap_ms:
                start_clock = source.recorded_at + timedelta(milliseconds=cursor)
                end_clock = source.recorded_at + timedelta(milliseconds=segment.start_ms)
                lines.append(f"[{start_clock:%H:%M:%S}–{end_clock:%H:%M:%S}] [无可辨识语音]")
                lines.append("")
            start_clock = source.recorded_at + timedelta(milliseconds=segment.start_ms)
            end_clock = source.recorded_at + timedelta(milliseconds=segment.end_ms)
            lines.append(
                f"[{start_clock:%H:%M:%S}–{end_clock:%H:%M:%S}] {segment.speaker}：{segment.text}"
            )
            lines.append("")
            cursor = max(cursor, segment.end_ms)
        if source.duration_ms - cursor >= long_gap_ms:
            start_clock = source.recorded_at + timedelta(milliseconds=cursor)
            lines.append(f"[{start_clock:%H:%M:%S}–{end_at:%H:%M:%S}] [无可辨识语音]")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def union_intervals(
    intervals: Iterable[tuple[int, int]], *, merge_gap_ms: int = 500
) -> list[tuple[int, int]]:
    """Merge overlapping VAD/provider ranges."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + merge_gap_ms:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
