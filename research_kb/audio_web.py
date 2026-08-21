"""Local web API for listening to long recordings with transcripts."""

# ruff: noqa: RUF001 - Chinese interface text intentionally uses Chinese punctuation.

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, cast
from urllib.parse import quote
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, field_validator

from research_kb.audio.core import inspect_recording, render_markdown
from research_kb.audio.domain import TranscriptDocument
from research_kb.audio.pricing import PRICE_PER_SECOND_CNY
from research_kb.audio.text_review import (
    DeepSeekTranscriptReviewer,
    TranscriptTextReviewError,
    estimate_deepseek_cost_cny,
    render_topic_markdown,
)
from research_kb.config import AppSettings
from research_kb.storage.json import write_json_model
from research_kb.storage.text import write_text_atomic

SUPPORTED_AUDIO = {".m4a", ".mp3", ".wav", ".flac"}
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
WINDOW_MS = 30 * 60_000
WEAK_TOPIC_GAP_MS = 2 * 60_000
WEAK_TOPIC_MAX_SPAN_MS = 20 * 60_000
ASR_PRICE_PER_SECOND_CNY = PRICE_PER_SECOND_CNY
MAX_JOB_ATTEMPTS = 3
MAX_SUMMARY_ATTEMPTS = 3
SUMMARY_RESULT_FILENAME = "summary_result.json"
SUMMARY_MARKDOWN_FILENAME = "总结_话题索引.md"
JOB_STEP_LABELS = {
    "preprocess": "预处理与降噪",
    "cloud": "云端语音识别",
    "local": "本地语音识别",
    "speakers": "整理说话人",
    "deepseek_text": "DeepSeek 文本整理",
    "render": "生成转写结果",
}
_JOB_STATE_LOCK = threading.RLock()
_TRANSCRIPTION_RUN_LOCK = threading.Lock()
_JOB_PROCESSES: dict[str, subprocess.Popen[str]] = {}
_JOB_THREADS: dict[str, threading.Thread] = {}
_SUMMARY_THREADS: dict[str, threading.Thread] = {}


@dataclass(frozen=True)
class RecordingEntry:
    """One source recording and its optional structured artifacts."""

    id: str
    title: str
    audio_path: Path
    duration_ms: int
    recorded_at: str
    sha256: str
    status: str
    transcript_path: Path | None = None
    source_recording_id: str | None = None
    analysis_path: Path | None = None
    summary_path: Path | None = None
    manifest_path: Path | None = None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _entry_id(sha256: str) -> str:
    return f"rec-{sha256[:16]}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _elapsed_clock(milliseconds: int) -> str:
    """Format an elapsed recording position for exported Markdown."""
    total_seconds = max(0, milliseconds) // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _markdown_text(value: object, fallback: str = "") -> str:
    """Keep exported headings and metadata on one safe, readable Markdown line."""
    cleaned = " ".join(str(value or fallback).replace("\r", "").split())
    return cleaned.replace("#", "\\#")


def _download_stem(value: object, fallback: str = "recording") -> str:
    cleaned = " ".join(str(value or fallback).replace("\r", "").split())
    return "".join("_" if character in '<>:"/\\|?*' else character for character in cleaned)


def _markdown_quote(value: object) -> list[str]:
    lines = str(value or "").replace("\r", "").splitlines() or [""]
    return [f"> {line}" if line else ">" for line in lines]


def _markdown_download(content: str, filename: str) -> Response:
    encoded_filename = quote(filename)
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"},
    )


def _render_transcript_export(catalog: RecordingCatalog, entry: RecordingEntry) -> str:
    segments = sorted(
        catalog.segments(entry),
        key=lambda item: (int(item.get("start_ms", 0)), int(item.get("end_ms", 0))),
    )
    lines = [
        f"# {_markdown_text(catalog.effective_title(entry), '未命名录音')}｜完整对话",
        "",
        f"> 录音日期：{_markdown_text(entry.recorded_at, '未记录')}",
        f"> 总时长：{_elapsed_clock(entry.duration_ms)}",
        f"> 语音条数：{len(segments)}",
        "",
        "## 对话正文",
        "",
    ]
    for segment in segments:
        start_ms = int(segment.get("start_ms", 0))
        end_ms = int(segment.get("end_ms", start_ms))
        speaker = _markdown_text(segment.get("speaker"), "说话人（未确认）")
        lines.extend(
            [
                f"### {_elapsed_clock(start_ms)}–{_elapsed_clock(end_ms)} · {speaker}",
                "",
                *_markdown_quote(segment.get("text", "")),
                "",
            ]
        )
    if not segments:
        lines.extend(["> 这份录音没有可导出的语音内容。", ""])
    lines.extend(
        [
            "---",
            "",
            "导出说明：以上内容按录音内时间顺序排列，保留完整转写；时间格式为 `时:分:秒`。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_summary_export(
    catalog: RecordingCatalog, entry: RecordingEntry, topics: list[dict[str, Any]]
) -> str:
    state = catalog.summary_state(entry)
    prompt = str(state.get("prompt", "")).strip()
    lines = [
        f"# {_markdown_text(catalog.effective_title(entry), '未命名录音')}｜总结",
        "",
        f"> 录音日期：{_markdown_text(entry.recorded_at, '未记录')}",
        f"> 总结主题：{len(topics)} 个",
        "",
        "## 本次整理要求",
        "",
    ]
    lines.extend(_markdown_quote(prompt or "按通用规则整理实质话题。"))
    lines.extend(["", "## 总结内容", ""])
    if not topics:
        lines.extend(["> 没有找到符合本次整理要求的内容。", ""])
    for index, topic in enumerate(topics, 1):
        keywords = [
            _markdown_text(word) for word in topic.get("keywords", []) if _markdown_text(word)
        ]
        lines.extend(
            [
                f"### {index:02d}. {_markdown_text(topic.get('title'), '未命名主题')}",
                "",
                (
                    f"**时间**：`{_elapsed_clock(int(topic.get('start_ms', 0)))}–"
                    f"{_elapsed_clock(int(topic.get('end_ms', 0)))}`"
                ),
                f"**关键词**：{'、'.join(keywords) if keywords else '无'}",
                "",
                str(topic.get("summary", "")).strip() or "暂无摘要。",
                "",
            ]
        )
    lines.extend(
        [
            "---",
            "",
            "导出说明：本文件只包含按本次提示词筛选后的总结内容；未纳入总结的语音仍保留在完整对话导出和左侧转写中。",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_prompts_payload(path: Path) -> dict[str, Any]:
    """Read the reusable prompt library, seeding the two useful starting templates."""
    try:
        payload = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        payload = {}
    prompts = payload.get("prompts")
    if not isinstance(prompts, list):
        prompts = []
    valid: list[dict[str, str]] = []
    for item in prompts:
        if not isinstance(item, dict):
            continue
        try:
            prompt = SummaryPromptPayload.model_validate(
                {"name": item.get("name", ""), "prompt": item.get("prompt", "")}
            )
        except ValueError:
            continue
        prompt_id = str(item.get("id", "")).strip()
        if not prompt_id:
            prompt_id = f"prompt-{uuid4().hex}"
        valid.append(
            {
                "id": prompt_id,
                "name": prompt.name,
                "prompt": prompt.prompt,
                "created_at": str(item.get("created_at", "")),
                "updated_at": str(item.get("updated_at", "")),
            }
        )
    if not valid:
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        valid = [{**item, "created_at": now, "updated_at": now} for item in DEFAULT_SUMMARY_PROMPTS]
    return {"schema_version": "1.0", "prompts": valid}


def _save_summary_prompts(path: Path, prompts: list[dict[str, Any]]) -> None:
    _write_json(path, {"schema_version": "1.0", "prompts": prompts})


class RecordingStartTimeUpdate(BaseModel):
    """A timezone-aware manual start-time calibration."""

    recorded_at: datetime

    @field_validator("recorded_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("recorded_at must include a timezone")
        return value


class SpeakerOverrideUpdate(BaseModel):
    """One recording-wide speaker label replacement."""

    source_speaker: str
    display_name: str

    @field_validator("source_speaker", "display_name")
    @classmethod
    def validate_label(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("speaker labels must not be empty")
        if len(cleaned) > 80:
            raise ValueError("speaker labels must not exceed 80 characters")
        return cleaned


class RecordingTitleUpdate(BaseModel):
    """A local display name for one recording."""

    title: str

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("recording title must not be empty")
        if len(cleaned) > 160:
            raise ValueError("recording title must not exceed 160 characters")
        return cleaned


class FavoriteNoteUpdate(BaseModel):
    """A durable note attached to one favorite sentence."""

    note: str

    @field_validator("note")
    @classmethod
    def validate_note(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) > 2_000:
            raise ValueError("favorite note must not exceed 2000 characters")
        return cleaned


class JobContinueRequest(BaseModel):
    """An explicit recovery strategy selected by the local user."""

    strategy: str | None = None


class SummaryRequest(BaseModel):
    """User-authored instructions for one on-demand DeepSeek summary."""

    prompt: str = ""

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) > 4_000:
            raise ValueError("summary prompt must not exceed 4000 characters")
        return cleaned


class SummaryPromptPayload(BaseModel):
    """A reusable user-authored summary instruction."""

    name: str
    prompt: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("summary prompt name must not be empty")
        if len(cleaned) > 80:
            raise ValueError("summary prompt name must not exceed 80 characters")
        return cleaned

    @field_validator("prompt")
    @classmethod
    def validate_prompt_body(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) > 4_000:
            raise ValueError("summary prompt must not exceed 4000 characters")
        return cleaned


DEFAULT_SUMMARY_PROMPTS = (
    {
        "id": "prompt-complete-meeting",
        "name": "完整会议",
        "prompt": (
            "请按录音时间顺序完整梳理会议内容，不遗漏议题、决定、分工、数字和待办；"
            "保留不确定信息，不要擅自补全。"
        ),
    },
    {
        "id": "prompt-finance-live",
        "name": "财经直播",
        "prompt": (
            "只整理主播的财经相关内容：市场观点、个股/行业、宏观数据、风险提示和操作逻辑。"
            "删除感谢礼物、寒暄、唱歌、闲聊和其他非财经内容，不要把这些内容列入总结。"
        ),
    },
)


def _topic_strength(topic: dict[str, Any], segment_count: int, evidence_count: int) -> str:
    title = str(topic.get("title", "")).strip()
    if title.startswith(("疑似", "[疑似")):
        return "weak"
    if evidence_count < 2 or evidence_count / max(1, segment_count) < 0.2:
        return "weak"
    return "strong"


def _weak_topic_groups(segments: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for segment in segments:
        start = int(segment.get("start_ms", 0))
        previous_end = int(current[-1].get("end_ms", start)) if current else start
        group_start = int(current[0].get("start_ms", start)) if current else start
        if current and (
            start - previous_end > WEAK_TOPIC_GAP_MS
            or start - group_start >= WEAK_TOPIC_MAX_SPAN_MS
        ):
            groups.append(current)
            current = []
        current.append(segment)
    if current:
        groups.append(current)
    return groups


def _weak_topic_payload(entry: RecordingEntry, segments: list[dict[str, Any]]) -> dict[str, Any]:
    first = segments[0]
    last = segments[-1]
    return {
        "id": f"weak-{entry.id}-{first.get('segment_id')}",
        "title": "零散对话与过渡内容",
        "summary": "这段语音没有形成集中的明确话题。内容已按原始时间顺序完整保留。",
        "keywords": [],
        "evidence_segment_ids": [],
        "start_ms": int(first.get("start_ms", 0)),
        "end_ms": int(last.get("end_ms", first.get("end_ms", 0))),
        "segment_count": len(segments),
        "strength": "weak",
    }


def _analysis_topic_payload(
    topic: dict[str, Any], segments: list[dict[str, Any]]
) -> dict[str, Any]:
    first = segments[0]
    last = segments[-1]
    segment_ids = {str(segment.get("segment_id")) for segment in segments}
    evidence = [
        str(segment_id)
        for segment_id in topic.get("evidence_segment_ids", [])
        if str(segment_id) in segment_ids
    ]
    return {
        "id": topic.get("topic_id") or f"topic-{first.get('segment_id')}",
        "title": topic.get("title") or "未命名话题",
        "summary": topic.get("summary", ""),
        "keywords": topic.get("keywords", []),
        "evidence_segment_ids": evidence,
        "start_ms": int(first.get("start_ms", 0)),
        "end_ms": int(last.get("end_ms", first.get("end_ms", 0))),
        "segment_count": len(segments),
        "strength": _topic_strength(topic, len(segments), len(evidence)),
    }


class RecordingCatalog:
    """Discover immutable transcript artifacts and expose windowed reading data."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.resolve()
        self.review_root = self.data_root / "audio-review"
        self.upload_root = self.review_root / "uploads"
        self.time_overrides_path = self.review_root / "recording-time-overrides.json"
        self.catalog_preferences_path = self.review_root / "recording-catalog.json"
        self.speaker_overrides_path = self.review_root / "speaker-overrides.json"
        self.favorites_path = self.review_root / "favorites.json"
        self.summary_prompts_path = self.review_root / "summary-prompts.json"
        self.activity_log_path = self.review_root / "activity-log.jsonl"
        self._entries: dict[str, RecordingEntry] = {}
        self._documents: dict[Path, dict[str, Any]] = {}
        self._time_overrides: dict[str, str] = {}
        self._title_overrides: dict[str, str] = {}
        self._hidden_recordings: set[str] = set()
        self._speaker_overrides: dict[str, dict[str, str]] = {}
        self._favorites: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self.refresh()

    def _load_time_overrides(self) -> dict[str, str]:
        try:
            payload = _read_json(self.time_overrides_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        recordings = payload.get("recordings", {})
        if not isinstance(recordings, dict):
            return {}
        return {
            str(recording_id): value
            for recording_id, value in recordings.items()
            if isinstance(value, str)
        }

    def _load_catalog_preferences(self) -> tuple[dict[str, str], set[str]]:
        try:
            payload = _read_json(self.catalog_preferences_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return {}, set()
        titles = payload.get("titles", {})
        hidden = payload.get("hidden", [])
        valid_titles = (
            {
                str(recording_id): title
                for recording_id, title in titles.items()
                if isinstance(title, str)
            }
            if isinstance(titles, dict)
            else {}
        )
        valid_hidden = (
            {str(recording_id) for recording_id in hidden if isinstance(recording_id, str)}
            if isinstance(hidden, list)
            else set()
        )
        return valid_titles, valid_hidden

    def _save_catalog_preferences(self) -> None:
        _write_json(
            self.catalog_preferences_path,
            {
                "schema_version": "1.0",
                "titles": self._title_overrides,
                "hidden": sorted(self._hidden_recordings),
            },
        )

    def _load_speaker_overrides(self) -> dict[str, dict[str, str]]:
        try:
            payload = _read_json(self.speaker_overrides_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        recordings = payload.get("recordings", {})
        if not isinstance(recordings, dict):
            return {}
        loaded: dict[str, dict[str, str]] = {}
        for recording_id, mappings in recordings.items():
            if not isinstance(mappings, dict):
                continue
            valid = {
                str(source): str(display)
                for source, display in mappings.items()
                if isinstance(source, str) and isinstance(display, str)
            }
            if valid:
                loaded[str(recording_id)] = valid
        return loaded

    def _load_favorites(self) -> dict[str, dict[str, dict[str, Any]]]:
        try:
            payload = _read_json(self.favorites_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        recordings = payload.get("recordings", {})
        if not isinstance(recordings, dict):
            return {}
        loaded: dict[str, dict[str, dict[str, Any]]] = {}
        for recording_id, favorites in recordings.items():
            if not isinstance(favorites, dict):
                continue
            valid = {
                str(segment_id): favorite
                for segment_id, favorite in favorites.items()
                if isinstance(segment_id, str) and isinstance(favorite, dict)
            }
            if valid:
                loaded[str(recording_id)] = valid
        return loaded

    def _load_upload_titles(self) -> dict[str, str]:
        """Recover original client filenames for normalized upload artifacts."""
        candidates: dict[str, tuple[float, str]] = {}
        if not self.upload_root.is_dir():
            return {}
        for job_path in self.upload_root.glob("job-*/job.json"):
            try:
                job = _read_json(job_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            sha256 = str(job.get("sha256", "")).strip().casefold()
            filename = Path(str(job.get("filename", "")).strip()).name
            if len(sha256) != 64 or not filename:
                continue
            created_at = float(job.get("created_at", 0) or 0)
            previous = candidates.get(sha256)
            if previous is None or created_at >= previous[0]:
                candidates[sha256] = (created_at, filename)
        return {sha256: value[1] for sha256, value in candidates.items()}

    def refresh(self) -> None:
        """Refresh the catalog while preferring completed, non-local-only transcripts."""
        discovered: dict[str, RecordingEntry] = {}
        documents: dict[Path, dict[str, Any]] = {}
        upload_titles = self._load_upload_titles()
        transcript_paths = sorted(
            self.data_root.rglob("transcript.json"),
            key=lambda path: ("local-only" in str(path).casefold(), len(path.parts)),
        )
        for transcript_path in transcript_paths:
            try:
                original_document = _read_json(transcript_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            reading_path = transcript_path.with_name("reading_view.json")
            if reading_path.is_file():
                try:
                    document = _read_json(reading_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    document = original_document
            else:
                document = original_document
            documents[transcript_path.resolve()] = document
            analysis_path = transcript_path.with_name("text_analysis.json")
            summary_path = transcript_path.with_name(SUMMARY_RESULT_FILENAME)
            manifest_path = transcript_path.with_name("processing_manifest.json")
            manifest_status = "completed"
            if manifest_path.is_file():
                try:
                    manifest_status = str(_read_json(manifest_path).get("status", "completed"))
                except (OSError, ValueError, json.JSONDecodeError):
                    manifest_status = "completed"
            if manifest_status == "failed" and analysis_path.is_file():
                manifest_status = "available_with_warning"
            for source in document.get("sources", []):
                if not isinstance(source, dict):
                    continue
                sha = str(source.get("sha256", ""))
                source_id = str(source.get("recording_id", ""))
                audio_path = Path(str(source.get("path", ""))).expanduser()
                if len(sha) != 64 or not source_id or not audio_path.is_file():
                    continue
                entry = RecordingEntry(
                    id=_entry_id(sha),
                    title=upload_titles.get(sha.casefold())
                    or str(source.get("filename") or audio_path.name),
                    audio_path=audio_path.resolve(),
                    duration_ms=int(source.get("duration_ms", 0)),
                    recorded_at=str(source.get("recorded_at", "")),
                    sha256=sha,
                    status=manifest_status,
                    transcript_path=transcript_path.resolve(),
                    source_recording_id=source_id,
                    analysis_path=analysis_path.resolve() if analysis_path.is_file() else None,
                    summary_path=summary_path.resolve() if summary_path.is_file() else None,
                    manifest_path=manifest_path.resolve() if manifest_path.is_file() else None,
                )
                discovered.setdefault(entry.id, entry)

        for manifest_path in self.data_root.rglob("processing_manifest.json"):
            if manifest_path.with_name("transcript.json").is_file():
                continue
            try:
                manifest = _read_json(manifest_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            source_dir = manifest_path.parent.parent
            hashes = manifest.get("source_hashes", {})
            if not isinstance(hashes, dict):
                hashes = {}
            for index, audio_path in enumerate(
                sorted(
                    path
                    for path in source_dir.iterdir()
                    if path.suffix.casefold() in SUPPORTED_AUDIO
                ),
                1,
            ):
                sha = str(hashes.get(audio_path.name, hashes.get(f"recording_{index:02d}", "")))
                try:
                    source = inspect_recording(audio_path, index=index)
                except (OSError, ValueError):
                    continue
                if len(sha) != 64:
                    sha = source.sha256
                entry = RecordingEntry(
                    id=_entry_id(sha),
                    title=audio_path.name,
                    audio_path=audio_path.resolve(),
                    duration_ms=source.duration_ms,
                    recorded_at=source.recorded_at.isoformat(),
                    sha256=sha,
                    status=str(manifest.get("status", "running")),
                    manifest_path=manifest_path.resolve(),
                )
                discovered.setdefault(entry.id, entry)
        with self._lock:
            self._entries = discovered
            self._documents = documents
            self._time_overrides = self._load_time_overrides()
            self._title_overrides, self._hidden_recordings = self._load_catalog_preferences()
            self._speaker_overrides = self._load_speaker_overrides()
            self._favorites = self._load_favorites()

    def entries(
        self, *, include_hidden: bool = False, query: str | None = None
    ) -> list[RecordingEntry]:
        with self._lock:
            normalized_query = (query or "").strip().casefold()
            entries = [
                entry
                for entry in self._entries.values()
                if (include_hidden or entry.id not in self._hidden_recordings)
                and (
                    not normalized_query
                    or normalized_query in self.effective_title(entry).casefold()
                    or normalized_query in entry.title.casefold()
                )
            ]
            return sorted(
                entries,
                key=lambda item: (self.effective_recorded_at(item), self.effective_title(item)),
            )

    def get(self, recording_id: str) -> RecordingEntry:
        with self._lock:
            entry = self._entries.get(recording_id)
        if entry is None:
            raise KeyError(recording_id)
        return entry

    def effective_recorded_at(self, entry: RecordingEntry) -> str:
        with self._lock:
            return self._time_overrides.get(entry.id, entry.recorded_at)

    def effective_title(self, entry: RecordingEntry) -> str:
        with self._lock:
            return self._title_overrides.get(entry.id, entry.title)

    def is_hidden(self, entry: RecordingEntry) -> bool:
        with self._lock:
            return entry.id in self._hidden_recordings

    def set_title(self, recording_id: str, title: str | None) -> RecordingEntry:
        with self._lock:
            entry = self._entries.get(recording_id)
            if entry is None:
                raise KeyError(recording_id)
            previous = self._title_overrides.get(recording_id, entry.title)
            normalized = title if title != entry.title else None
            if normalized is None:
                changed = self._title_overrides.pop(recording_id, None) is not None
            else:
                changed = self._title_overrides.get(recording_id) != normalized
                self._title_overrides[recording_id] = normalized
            self._save_catalog_preferences()
            if changed:
                self._append_activity(
                    entry,
                    "recording_title_reset" if normalized is None else "recording_title_updated",
                    {
                        "previous_title": previous,
                        "title": normalized or entry.title,
                    },
                )
            return entry

    def set_hidden(self, recording_id: str, hidden: bool) -> RecordingEntry:
        with self._lock:
            entry = self._entries.get(recording_id)
            if entry is None:
                raise KeyError(recording_id)
            changed = (recording_id in self._hidden_recordings) != hidden
            if hidden:
                self._hidden_recordings.add(recording_id)
            else:
                self._hidden_recordings.discard(recording_id)
            self._save_catalog_preferences()
            if changed:
                self._append_activity(
                    entry,
                    "recording_hidden" if hidden else "recording_restored",
                    {"title": self.effective_title(entry)},
                )
            return entry

    def set_recorded_at(self, recording_id: str, recorded_at: str | None) -> RecordingEntry:
        with self._lock:
            entry = self._entries.get(recording_id)
            if entry is None:
                raise KeyError(recording_id)
            if recorded_at is None:
                self._time_overrides.pop(recording_id, None)
            else:
                self._time_overrides[recording_id] = recorded_at
            _write_json(
                self.time_overrides_path,
                {"schema_version": "1.0", "recordings": self._time_overrides},
            )
            return entry

    def document(self, entry: RecordingEntry) -> dict[str, Any]:
        if entry.transcript_path is None:
            return {}
        with self._lock:
            return self._documents.get(entry.transcript_path, {})

    def source_segments(self, entry: RecordingEntry) -> list[dict[str, Any]]:
        source_id = entry.source_recording_id
        return [
            segment
            for segment in self.document(entry).get("segments", [])
            if isinstance(segment, dict) and segment.get("recording_id") == source_id
        ]

    def segments(self, entry: RecordingEntry) -> list[dict[str, Any]]:
        with self._lock:
            overrides = dict(self._speaker_overrides.get(entry.id, {}))
        segments: list[dict[str, Any]] = []
        for source in self.source_segments(entry):
            segment = dict(source)
            original_speaker = str(
                source.get("speaker") or "\u8bf4\u8bdd\u4eba\uff08\u672a\u786e\u8ba4\uff09"
            )
            segment["original_speaker"] = original_speaker
            segment["speaker"] = overrides.get(original_speaker, original_speaker)
            segments.append(segment)
        return segments

    def speakers(self, entry: RecordingEntry) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for segment in self.source_segments(entry):
            speaker = str(
                segment.get("speaker") or "\u8bf4\u8bdd\u4eba\uff08\u672a\u786e\u8ba4\uff09"
            )
            counts[speaker] = counts.get(speaker, 0) + 1
        with self._lock:
            overrides = dict(self._speaker_overrides.get(entry.id, {}))
        return [
            {
                "source_speaker": speaker,
                "display_name": overrides.get(speaker, speaker),
                "segment_count": count,
                "is_overridden": speaker in overrides,
            }
            for speaker, count in counts.items()
        ]

    def set_speaker_override(
        self, entry: RecordingEntry, source_speaker: str, display_name: str | None
    ) -> None:
        speaker_counts = {
            speaker["source_speaker"]: int(speaker["segment_count"])
            for speaker in self.speakers(entry)
        }
        if source_speaker not in speaker_counts:
            raise ValueError("speaker not found")
        with self._lock:
            mappings = self._speaker_overrides.setdefault(entry.id, {})
            previous = mappings.get(source_speaker, source_speaker)
            normalized = display_name if display_name != source_speaker else None
            if normalized is None:
                changed = mappings.pop(source_speaker, None) is not None
            else:
                changed = mappings.get(source_speaker) != normalized
                mappings[source_speaker] = normalized
            if not mappings:
                self._speaker_overrides.pop(entry.id, None)
            _write_json(
                self.speaker_overrides_path,
                {"schema_version": "1.0", "recordings": self._speaker_overrides},
            )
            if changed:
                self._append_activity(
                    entry,
                    (
                        "speaker_override_removed"
                        if normalized is None
                        else "speaker_override_updated"
                    ),
                    {
                        "source_speaker": source_speaker,
                        "previous_display_name": previous,
                        "display_name": normalized or source_speaker,
                        "affected_segment_count": speaker_counts[source_speaker],
                    },
                )

    def favorites(self, entry: RecordingEntry) -> list[dict[str, Any]]:
        segments = {str(segment.get("segment_id")): segment for segment in self.segments(entry)}
        with self._lock:
            stored = dict(self._favorites.get(entry.id, {}))
        hydrated: list[dict[str, Any]] = []
        for segment_id, favorite in stored.items():
            segment = segments.get(segment_id)
            if segment is None:
                hydrated.append(dict(favorite))
                continue
            hydrated.append(
                {
                    **favorite,
                    "segment_id": segment_id,
                    "start_ms": int(segment.get("start_ms", 0)),
                    "end_ms": int(segment.get("end_ms", 0)),
                    "speaker": str(segment.get("speaker", "")),
                    "text": str(segment.get("text", "")),
                }
            )
        for favorite in hydrated:
            favorite.setdefault("note", "")
            favorite.setdefault("note_updated_at", None)
        return sorted(hydrated, key=lambda item: int(item.get("start_ms", 0)))

    def all_favorites(self) -> list[dict[str, Any]]:
        """Return favorites from every recording in chronological order."""
        aggregated: list[dict[str, Any]] = []
        for entry in self.entries(include_hidden=True):
            for favorite in self.favorites(entry):
                aggregated.append(
                    {
                        "recording_id": entry.id,
                        "recording_title": self.effective_title(entry),
                        "recorded_at": self.effective_recorded_at(entry),
                        **favorite,
                    }
                )
        return aggregated

    def set_favorite(self, entry: RecordingEntry, segment_id: str, favorite: bool) -> None:
        segments = {str(segment.get("segment_id")): segment for segment in self.segments(entry)}
        segment = segments.get(segment_id)
        if segment is None:
            raise ValueError("segment not found")
        with self._lock:
            recording_favorites = self._favorites.setdefault(entry.id, {})
            if favorite and segment_id not in recording_favorites:
                created_at = datetime.now(UTC).isoformat()
                recording_favorites[segment_id] = {
                    "segment_id": segment_id,
                    "created_at": created_at,
                    "note": "",
                    "note_updated_at": None,
                    "start_ms": int(segment.get("start_ms", 0)),
                    "end_ms": int(segment.get("end_ms", 0)),
                    "speaker": str(segment.get("speaker", "")),
                    "text": str(segment.get("text", "")),
                }
                self._append_activity(
                    entry,
                    "favorite_added",
                    {
                        "segment_id": segment_id,
                        "start_ms": int(segment.get("start_ms", 0)),
                        "end_ms": int(segment.get("end_ms", 0)),
                        "speaker": str(segment.get("speaker", "")),
                        "text": str(segment.get("text", "")),
                    },
                )
            elif not favorite and segment_id in recording_favorites:
                removed = recording_favorites.pop(segment_id)
                self._append_activity(entry, "favorite_removed", dict(removed))
            else:
                return
            if not recording_favorites:
                self._favorites.pop(entry.id, None)
            _write_json(
                self.favorites_path,
                {"schema_version": "1.0", "recordings": self._favorites},
            )

    def set_favorite_note(
        self, entry: RecordingEntry, segment_id: str, note: str
    ) -> dict[str, Any]:
        """Persist a note for one existing favorite without changing the transcript."""
        with self._lock:
            recording_favorites = self._favorites.get(entry.id, {})
            favorite = recording_favorites.get(segment_id)
            if favorite is None:
                raise ValueError("favorite not found")
            previous = str(favorite.get("note", ""))
            if previous == note:
                return next(
                    item for item in self.favorites(entry) if item["segment_id"] == segment_id
                )
            updated_at = datetime.now(UTC).isoformat()
            favorite["note"] = note
            favorite["note_updated_at"] = updated_at
            _write_json(
                self.favorites_path,
                {"schema_version": "1.0", "recordings": self._favorites},
            )
            self._append_activity(
                entry,
                "favorite_note_updated" if note else "favorite_note_cleared",
                {
                    "segment_id": segment_id,
                    "previous_note": previous,
                    "note": note,
                },
            )
        return next(item for item in self.favorites(entry) if item["segment_id"] == segment_id)

    def _append_activity(self, entry: RecordingEntry, action: str, details: dict[str, Any]) -> None:
        event = {
            "event_id": f"evt-{uuid4().hex}",
            "created_at": datetime.now(UTC).isoformat(),
            "recording_id": entry.id,
            "recording_title": self.effective_title(entry),
            "action": action,
            "details": details,
        }
        self.activity_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.activity_log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def activity(self, entry: RecordingEntry, limit: int) -> list[dict[str, Any]]:
        try:
            lines = self.activity_log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        events: list[dict[str, Any]] = []
        for line in reversed(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("recording_id") != entry.id:
                continue
            events.append(event)
            if len(events) >= limit:
                break
        return events

    def topics(self, entry: RecordingEntry) -> list[dict[str, Any]]:
        analysis_path = entry.summary_path or entry.analysis_path
        if analysis_path is None:
            return []
        segments = sorted(
            self.segments(entry),
            key=lambda item: (int(item.get("start_ms", 0)), int(item.get("end_ms", 0))),
        )
        if not segments:
            return []
        try:
            payload = _read_json(analysis_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return []
        analysis = payload.get("analysis", payload)
        if not isinstance(analysis, dict):
            return []
        segment_indexes = {
            str(segment.get("segment_id")): index for index, segment in enumerate(segments)
        }
        ranges: list[tuple[int, int, dict[str, Any]]] = []
        for topic in analysis.get("topics", []):
            if not isinstance(topic, dict):
                continue
            first = segment_indexes.get(str(topic.get("start_segment_id")))
            last = segment_indexes.get(str(topic.get("end_segment_id")))
            if first is None or last is None:
                continue
            if last < first:
                first, last = last, first
            ranges.append((first, last, topic))
        ranges.sort(key=lambda item: (item[0], item[1]))

        topics: list[dict[str, Any]] = []
        for first, last, topic in ranges:
            if last < 0:
                continue
            assigned = segments[first : last + 1]
            if assigned:
                topics.append(_analysis_topic_payload(topic, assigned))
        return topics

    def density(self, entry: RecordingEntry, bins: int = 240) -> list[float]:
        if entry.duration_ms <= 0:
            return []
        width = entry.duration_ms / bins
        audible = [0.0] * bins
        for segment in self.segments(entry):
            start = max(0, int(segment.get("start_ms", 0)))
            end = min(entry.duration_ms, int(segment.get("end_ms", start)))
            first = min(bins - 1, int(start / width))
            last = min(bins - 1, int(max(start, end - 1) / width))
            for index in range(first, last + 1):
                left = index * width
                right = left + width
                audible[index] += max(0.0, min(end, right) - max(start, left))
        return [round(min(1.0, value / width), 3) for value in audible]

    def summary_state(self, entry: RecordingEntry) -> dict[str, Any]:
        """Return the durable state of the user-triggered summary operation."""
        if entry.transcript_path is None:
            return {"status": "unavailable"}
        state_path = entry.transcript_path.with_name("summary_request.json")
        result_path = entry.transcript_path.with_name(SUMMARY_RESULT_FILENAME)
        result_prompt = ""
        if result_path.is_file():
            try:
                result_prompt = str(_read_json(result_path).get("prompt", ""))
            except (OSError, ValueError, json.JSONDecodeError):
                result_prompt = ""
        try:
            state = _read_json(state_path)
        except (OSError, ValueError, json.JSONDecodeError):
            state = {"status": "idle"}
        status = str(state.get("status", "idle"))
        default_progress = 100 if status == "completed" else 0
        return {
            "status": status,
            "prompt": str(state.get("prompt", "")) or result_prompt,
            "source": "summary"
            if result_path.is_file()
            else ("text_review" if entry.analysis_path is not None else "none"),
            "error": state.get("error"),
            "progress_percent": max(
                0, min(100, int(state.get("progress_percent", default_progress) or 0))
            ),
            "stage": str(state.get("stage", "")),
            "attempt_count": int(state.get("attempt_count", 0) or 0),
            "max_attempts": int(
                state.get("max_attempts", MAX_SUMMARY_ATTEMPTS) or MAX_SUMMARY_ATTEMPTS
            ),
            "retry_count": int(state.get("retry_count", 0) or 0),
            "can_retry": bool(state.get("can_retry", status == "failed")),
            "cost_cny": state.get("cost_cny"),
            "topic_count": int(state.get("topic_count", 0) or 0),
            "requested_at": state.get("requested_at"),
            "finished_at": state.get("finished_at"),
        }

    def reading_blocks(
        self, entry: RecordingEntry, start_ms: int, end_ms: int
    ) -> list[dict[str, Any]]:
        segments = [
            item
            for item in self.segments(entry)
            if int(item.get("end_ms", 0)) > start_ms and int(item.get("start_ms", 0)) < end_ms
        ]
        topic_starts = {int(topic["start_ms"]) for topic in self.topics(entry)}
        blocks: list[dict[str, Any]] = []
        current: list[dict[str, Any]] = []
        char_count = 0
        for segment in segments:
            segment_start = int(segment.get("start_ms", 0))
            text = str(segment.get("text", "")).strip()
            previous_end = int(current[-1].get("end_ms", 0)) if current else segment_start
            block_start = (
                int(current[0].get("start_ms", segment_start)) if current else segment_start
            )
            split = bool(
                current
                and (
                    segment_start - previous_end > 12_000
                    or segment_start - block_start >= 90_000
                    or char_count + len(text) > 320
                    or segment_start in topic_starts
                )
            )
            if split:
                blocks.append(_block_payload(current))
                current = []
                char_count = 0
            current.append(segment)
            char_count += len(text)
        if current:
            blocks.append(_block_payload(current))
        return blocks


def _block_payload(segments: list[dict[str, Any]]) -> dict[str, Any]:
    sentences = []
    for segment in segments:
        candidates = [
            candidate for candidate in segment.get("candidates", []) if isinstance(candidate, dict)
        ]
        sentences.append(
            {
                "id": segment.get("segment_id"),
                "start_ms": segment.get("start_ms"),
                "end_ms": segment.get("end_ms"),
                "speaker": segment.get("speaker"),
                "original_speaker": segment.get("original_speaker", segment.get("speaker")),
                "text": segment.get("text"),
                "confidence": segment.get("confidence"),
                "flags": segment.get("flags", []),
                "candidates": candidates,
            }
        )
    return {
        "id": f"block-{sentences[0]['id']}",
        "start_ms": sentences[0]["start_ms"],
        "end_ms": sentences[-1]["end_ms"],
        "sentences": sentences,
    }


def _entry_payload(
    catalog: RecordingCatalog, entry: RecordingEntry, *, detail: bool
) -> dict[str, Any]:
    segments = catalog.segments(entry)
    payload: dict[str, Any] = {
        "id": entry.id,
        "title": catalog.effective_title(entry),
        "original_title": entry.title,
        "title_overridden": catalog.effective_title(entry) != entry.title,
        "hidden": catalog.is_hidden(entry),
        "duration_ms": entry.duration_ms,
        "recorded_at": catalog.effective_recorded_at(entry),
        "original_recorded_at": entry.recorded_at,
        "start_time_overridden": catalog.effective_recorded_at(entry) != entry.recorded_at,
        "status": entry.status,
        "sha256": entry.sha256,
        "segment_count": len(segments),
        "favorite_count": len(catalog.favorites(entry)),
        "has_transcript": entry.transcript_path is not None,
        "has_topics": bool(catalog.topics(entry)),
    }
    if detail:
        topics = catalog.topics(entry)
        payload.update(
            density=catalog.density(entry),
            topics=topics,
            topic_segment_count=sum(int(topic.get("segment_count", 0)) for topic in topics),
            summary=catalog.summary_state(entry),
        )
        if entry.manifest_path is not None:
            try:
                manifest = _read_json(entry.manifest_path)
            except (OSError, ValueError, json.JSONDecodeError):
                manifest = {}
            payload["processing"] = {
                "status": manifest.get("status", entry.status),
                "completed_steps": manifest.get("completed_steps", []),
                "estimated_cost_cny": manifest.get("estimated_cost_cny", 0),
                "cost_cap_cny": manifest.get("cost_cap_cny"),
                "errors": manifest.get("errors", []),
            }
    return payload


def _parse_range(header: str | None, size: int) -> tuple[int, int, bool]:
    if not header:
        return 0, size - 1, False
    if not header.startswith("bytes=") or "," in header:
        raise ValueError("unsupported range")
    start_text, end_text = header[6:].split("-", 1)
    if not start_text:
        length = int(end_text)
        if length <= 0:
            raise ValueError("invalid suffix")
        return max(0, size - length), size - 1, True
    start = int(start_text)
    end = min(size - 1, int(end_text) if end_text else size - 1)
    if start < 0 or start >= size or end < start:
        raise ValueError("range outside file")
    return start, end, True


def _file_chunks(path: Path, start: int, end: int) -> Iterator[bytes]:
    remaining = end - start + 1
    with path.open("rb") as stream:
        stream.seek(start)
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _write_job(job_path: Path, payload: dict[str, Any]) -> None:
    _write_json(job_path, payload)


def _job_steps(job: dict[str, Any]) -> tuple[str, ...]:
    steps = ["preprocess"]
    if bool(job.get("run_cloud_enabled", job.get("allow_cloud_upload"))):
        steps.append("cloud")
    steps.extend(("local", "speakers"))
    if bool(job.get("text_review_enabled", job.get("allow_cloud_upload"))):
        steps.append("deepseek_text")
    steps.append("render")
    return tuple(steps)


def _cached_text_usage(job_path: Path) -> tuple[int, int, int]:
    hit = miss = output = 0
    cache_paths = job_path.parent.glob("transcription/**/providers/deepseek/*.json")
    for cache_path in cache_paths:
        try:
            payload = _read_json(cache_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        hit += int(payload.get("prompt_cache_hit_tokens", 0) or 0)
        miss += int(payload.get("prompt_cache_miss_tokens", 0) or 0)
        output += int(payload.get("output_tokens", 0) or 0)
    return hit, miss, output


def _cloud_cost_override_decision(
    job_path: Path,
    job: dict[str, Any],
    message: str,
    estimated_cost_cny: float,
) -> dict[str, Any] | None:
    """Offer a bounded cloud continuation using already-derived speech regions."""
    if (
        str(job.get("status", "queued")) not in {"failed", "cancelled"}
        or job.get("decision_cancelled_at")
        or "all planned cloud speech regions would exceed the cost cap" not in message.casefold()
        or not bool(job.get("allow_cloud_upload"))
    ):
        return None
    region_paths = list(
        job_path.parent.glob("transcription/**/providers/cloud/speech-regions.json")
    )
    if not region_paths:
        return None
    try:
        payload = _read_json(max(region_paths, key=lambda path: path.stat().st_mtime_ns))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    regions = payload.get("regions")
    if not isinstance(regions, list) or not regions:
        return None
    planned_seconds = 60
    for region in regions:
        if not isinstance(region, dict):
            return None
        try:
            duration_ms = int(region["end_ms"]) - int(region["start_ms"])
        except (KeyError, TypeError, ValueError):
            return None
        if duration_ms <= 0:
            return None
        planned_seconds += (duration_ms + 999) // 1_000
    asr_cost_cap = planned_seconds * ASR_PRICE_PER_SECOND_CNY
    text_cost_cap = (
        float(job.get("text_review_cost_cap_cny", 0.3) or 0.3)
        if bool(job.get("text_review_enabled", True))
        else 0.0
    )
    required_cap = math.ceil((asr_cost_cap + text_cost_cap) * 100) / 100
    current_cap = float(job.get("cost_cap_cny", 3) or 3)
    if required_cap <= current_cap:
        return None
    return {
        "strategy": "continue_cloud_with_higher_cap",
        "title": "提高上限并继续云端识别",
        "description": "复用已完成的预处理、VAD 和语音分块, 继续使用云端 ASR, 不切换为仅本地。",
        "impact": (
            f"按官网原价估算, 云端 ASR 最多约 ¥{asr_cost_cap:.4f}, "
            f"文本整理最多 ¥{text_cost_cap:.2f}; 新的任务硬上限为 ¥{required_cap:.2f}。"
            "实际账单可能因免费额度或优惠更低。"
        ),
        "continue_label": f"提高至 ¥{required_cap:.2f} 并继续云端",
        "can_continue": True,
        "additional_external_cost_cny": round(max(0.0, required_cap - estimated_cost_cny), 4),
        "cost_cap_cny": required_cap,
        "planned_cloud_seconds": planned_seconds,
    }


def _full_restart_decision(
    job_path: Path,
    job: dict[str, Any],
    estimated_cost_cny: float,
) -> dict[str, Any] | None:
    """Offer an explicit reset when a failed checkpoint or derived cache is suspect."""
    if (
        str(job.get("status", "queued")) not in {"failed", "cancelled"}
        or job.get("decision_cancelled_at")
        or not any(path.is_file() for path in job_path.parent.glob("recording.*"))
    ):
        return None
    run_cloud = bool(job.get("run_cloud_enabled", job.get("allow_cloud_upload")))
    remaining_budget = round(
        max(0.0, float(job.get("cost_cap_cny", 3) or 3) - estimated_cost_cny),
        4,
    )
    external_impact = (
        f"可能新增外部费用最多 ¥{remaining_budget:.4f}"
        if run_cloud
        else "仅运行本地模型, 不产生外部费用"
    )
    return {
        "strategy": "restart_from_scratch",
        "title": "清理断点并从头开始",
        "description": (
            "删除本任务已生成的转写、模型缓存和处理清单, 保留上传的原始录音, "
            "重新执行文件检查和全部处理步骤。"
        ),
        "impact": (
            f"不会删除原始录音; {external_impact}。"
            "如果同一原录音本身损坏, 从头开始仍会报告相同错误。"
        ),
        "continue_label": "清理缓存并从头开始",
        "can_continue": True,
        "additional_external_cost_cny": remaining_budget if run_cloud else 0.0,
    }


def _recovery_decision(
    job_path: Path,
    job: dict[str, Any],
    completed_steps: list[str],
    message: str,
    estimated_cost_cny: float,
    text_review_cost_cny: float,
) -> dict[str, Any] | None:
    status = str(job.get("status", "queued"))
    if status not in {"failed", "cancelled", "completed"} or job.get("decision_cancelled_at"):
        return None
    source_exists = any(path.is_file() for path in job_path.parent.glob("recording.*"))
    if not source_exists:
        return {
            "strategy": "cannot_continue",
            "title": "源录音缺失, 不能安全继续",
            "description": "任务目录中找不到原始录音。请保留记录并重新添加原文件。",
            "impact": "不会运行模型, 也不会产生新费用。",
            "continue_label": "无法继续",
            "can_continue": False,
            "additional_external_cost_cny": 0.0,
        }

    normalized = message.casefold()
    remaining_budget = round(
        max(0.0, float(job.get("cost_cap_cny", 3) or 3) - estimated_cost_cny),
        4,
    )

    def decision(
        strategy: str,
        title: str,
        description: str,
        impact: str,
        *,
        additional_cost: float,
    ) -> dict[str, Any]:
        return {
            "strategy": strategy,
            "title": title,
            "description": description,
            "impact": impact,
            "continue_label": "按建议继续",
            "can_continue": True,
            "additional_external_cost_cny": round(additional_cost, 4),
        }

    unrecoverable_markers = (
        "unsupported audio",
        "no supported audio",
        "audio file exceeds",
        "invalid audio",
        "corrupt audio",
        "missing mvhd atom",
        "invalid mvhd timing values",
        "unsupported mvhd version",
        "could not inspect audio stream",
        "source audio is missing",
        "no such file",
    )
    if any(marker in normalized for marker in unrecoverable_markers):
        return {
            "strategy": "cannot_continue",
            "title": "录音文件无法继续处理",
            "description": "文件缺失、损坏或格式不受支持, 需要重新添加可读取的原录音。",
            "impact": "不会运行模型, 也不会产生新费用。",
            "continue_label": "无法继续",
            "can_continue": False,
            "additional_external_cost_cny": 0.0,
        }

    deepseek_markers = (
        "deepseek",
        "text review",
        "window json",
        "topic stubs",
        "consolidation",
    )
    if any(marker in normalized for marker in deepseek_markers):
        if status == "completed":
            text_remaining = max(
                0.0,
                float(job.get("text_review_cost_cap_cny", 0.3) or 0.3) - text_review_cost_cny,
            )
            repair = decision(
                "retry_text_review_only",
                "仅修复文本整理",
                "复用已完成的 ASR、时间戳、说话人和 DeepSeek 窗口缓存, 只重新校验并完成话题整理。",
                "不会重新上传音频或产生 ASR 费用; DeepSeek 最多新增 "
                f"¥{min(remaining_budget, text_remaining):.4f}。失败时仍保留当前转写。",
                additional_cost=min(remaining_budget, text_remaining),
            )
            repair["continue_label"] = "仅修复文本整理"
            return repair
        return decision(
            "finalize_without_text_review",
            "跳过文本整理, 直接完成转写",
            "保留已完成的 ASR、时间戳和说话人结果, 关闭 DeepSeek 后从断点生成最终文档。",
            "不再调用 DeepSeek; 预计新增外部费用为 ¥0。原始识别文本仍会完整保留。",
            additional_cost=0.0,
        )

    if status == "completed":
        if "文本整理已完成" in message and "费用上限" in message and remaining_budget > 0:
            extend = decision(
                "extend_text_review_budget",
                "是否完成剩余文本窗口",
                "当前整理结果已经可用; 如继续, 只处理因文本子预算停止的窗口并复用全部缓存。",
                "不会重新上传音频或产生 ASR 费用; DeepSeek 新增费用最多 "
                f"¥{remaining_budget:.4f}, 总费用仍不超过 ¥3。",
                additional_cost=remaining_budget,
            )
            extend["continue_label"] = "追加预算完成剩余窗口"
            extend["text_review_cost_cap_cny"] = round(text_review_cost_cny + remaining_budget, 6)
            return extend
        return None

    if "cost cap" in normalized:
        if "cloud" in completed_steps:
            return decision(
                "finalize_without_text_review",
                "复用已完成云端识别并直接收尾",
                "云端识别已经完成; 关闭额外文本整理, 只执行剩余本地步骤和文档生成。",
                "不会新增云端或 DeepSeek 调用; 预计新增外部费用为 ¥0。",
                additional_cost=0.0,
            )
        return decision(
            "continue_local_only",
            "切换为仅本地模式继续",
            "费用上限不足以安全完成云端识别; 保留已有缓存, 改用本地识别完成剩余内容。",
            "不再上传或调用云端, 预计新增外部费用为 ¥0; 准确度可能低于云端增强。",
            additional_cost=0.0,
        )

    cloud_configuration_markers = (
        "dashscope_api_key",
        "requires --allow-cloud-upload",
        "model not found",
        "model does not exist",
        "unauthorized",
        "forbidden",
        "invalid parameter",
        "bad request",
    )
    if any(marker in normalized for marker in cloud_configuration_markers):
        return decision(
            "continue_local_only",
            "切换为仅本地模式继续",
            "当前云端凭据或上传授权不可用; 不再调用云端, 使用本地缓存和本地模型完成。",
            "预计新增外部费用为 ¥0; 准确度可能低于云端增强。",
            additional_cost=0.0,
        )

    if "speech-only cloud mode requires" in normalized:
        if bool(job.get("allow_cloud_upload")):
            return decision(
                "resume_fixed_pipeline",
                "使用已修正的本地 VAD + 云端识别继续",
                "旧任务的模式组合错误; 新版会同时启用本地切分和云端识别并复用已有缓存。",
                f"只处理未完成区间, 仍受 ¥3 总上限约束; 剩余外部预算最多 ¥{remaining_budget:.4f}。",
                additional_cost=remaining_budget,
            )
        return decision(
            "continue_local_only",
            "按仅本地模式继续",
            "任务未授权云端上传; 移除错误的云端语音区间选项, 使用本地模型从断点继续。",
            "不会上传录音, 预计新增外部费用为 ¥0。",
            additional_cost=0.0,
        )

    local_resource_markers = (
        "cuda out of memory",
        "cuda is not available",
        "torch.cuda",
        "qwen-asr is required",
        "ffmpeg executable",
        "disk full",
        "no space",
        "permission denied",
    )
    if any(marker in normalized for marker in local_resource_markers):
        return {
            "strategy": "cannot_continue",
            "title": "本机资源不足, 暂不能安全继续",
            "description": "请先释放显存或磁盘空间, 再重新添加同一文件; 已有缓存会按哈希复用。",
            "impact": "当前不会运行模型, 也不会产生新费用。",
            "continue_label": "暂无法继续",
            "can_continue": False,
            "additional_external_cost_cny": 0.0,
        }

    run_cloud = bool(job.get("run_cloud_enabled", job.get("allow_cloud_upload")))
    return decision(
        "resume_checkpoint",
        "从最后成功断点继续",
        "自动重试已经停止; 再次运行只处理尚未成功的区间, 成功缓存不会重做。",
        (
            f"仍受 ¥3 总上限约束; 可能使用的剩余外部预算最多 ¥{remaining_budget:.4f}。"
            if run_cloud
            else "仅运行本地模型, 预计新增外部费用为 ¥0。"
        ),
        additional_cost=remaining_budget if run_cloud else 0.0,
    )


def _public_job(job_path: Path) -> dict[str, Any]:
    job = _read_json(job_path)
    manifest: dict[str, Any] = {}
    manifest_path: Path | None = None
    manifest_paths = list(job_path.parent.glob("transcription/**/processing_manifest.json"))
    if manifest_paths:
        manifest_path = max(manifest_paths, key=lambda path: path.stat().st_mtime_ns)
        try:
            manifest = _read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            manifest = {}

    completed_steps = [
        str(step)
        for step in manifest.get("completed_steps", job.get("completed_steps", []))
        if str(step) in JOB_STEP_LABELS
    ]
    status = str(job.get("status", "queued"))
    cancel_requested = bool(job.get("cancel_requested"))
    expected_steps = _job_steps(job)
    operation = str(job.get("operation") or "")
    manifest_messages = [str(item) for item in manifest.get("errors", []) if str(item).strip()]
    latest_manifest_message = manifest_messages[-1] if manifest_messages else ""
    text_review_requested = bool(job.get("text_review_enabled")) or any(
        (
            "deepseek_text" in completed_steps,
            bool(manifest.get("text_review_model")),
            bool(manifest.get("text_review_input_tokens")),
            bool(manifest.get("text_review_output_tokens")),
        )
    )
    analysis_exists = bool(
        manifest_path is not None and manifest_path.with_name("text_analysis.json").is_file()
    )
    text_review_status = "not_requested"
    if status in {"queued", "running"} and (
        operation == "text_review_repair" or text_review_requested
    ):
        text_review_status = "running"
    elif text_review_requested and status == "failed":
        text_review_status = "failed"
    elif text_review_requested and status == "completed":
        if "个窗口因费用上限保留原始识别" in latest_manifest_message:
            text_review_status = "partial"
        elif latest_manifest_message or not analysis_exists:
            text_review_status = "fallback"
        else:
            text_review_status = "completed"

    if text_review_status in {"fallback", "partial"}:
        completed_steps = [step for step in completed_steps if step != "deepseek_text"]
    core_transcript_ready = status == "completed" and "render" in completed_steps
    if operation == "text_review_repair" and status == "queued":
        progress = 90
        stage = "等待修复文本整理"
    elif operation == "text_review_repair" and status == "running":
        progress = 95
        stage = "仅修复文本整理"
    elif cancel_requested and status in {"queued", "running"}:
        progress = min(95, 5 + round(90 * len(completed_steps) / max(1, len(expected_steps))))
        stage = "正在取消"
    elif status == "cancelled":
        progress = min(95, 5 + round(90 * len(completed_steps) / max(1, len(expected_steps))))
        stage = "已取消"
    elif status == "completed":
        progress = 100
        if text_review_status == "fallback":
            stage = "转写可用 · 文本整理待处理"
        elif text_review_status == "partial":
            stage = "转写可用 · 文本整理部分完成"
        else:
            stage = "转写完成"
    elif status == "queued":
        progress = 0
        stage = "等待断点恢复" if int(job.get("recovery_count", 0) or 0) else "等待开始"
    else:
        progress = min(95, 5 + round(90 * len(completed_steps) / max(1, len(expected_steps))))
        stage = "处理失败" if status == "failed" else "准备与预处理"
        if status != "failed":
            next_step = next((step for step in expected_steps if step not in completed_steps), None)
            if next_step is not None:
                stage = JOB_STEP_LABELS[next_step]
        if status == "running" and bool(job.get("retrying")):
            attempt = int(job.get("attempt_count", 1) or 1)
            maximum = int(job.get("max_attempts", MAX_JOB_ATTEMPTS) or MAX_JOB_ATTEMPTS)
            stage = f"自动断点重试 {attempt}/{maximum}"

    cached_hit, cached_miss, cached_output = _cached_text_usage(job_path)
    manifest_input = int(manifest.get("text_review_input_tokens", 0) or 0)
    manifest_output = int(manifest.get("text_review_output_tokens", 0) or 0)
    cached_input = cached_hit + cached_miss
    text_input = max(manifest_input, cached_input)
    text_output = max(manifest_output, cached_output)
    text_cost = float(manifest.get("text_review_cost_cny", 0) or 0)
    if cached_input or cached_output:
        text_cost = max(
            text_cost,
            estimate_deepseek_cost_cny(
                hit=cached_hit,
                miss=cached_miss,
                output=cached_output,
            ),
        )
    current_billed_seconds = int(manifest.get("cloud_billed_seconds", 0) or 0)
    current_asr_cost = current_billed_seconds * ASR_PRICE_PER_SECOND_CNY
    baseline_billed_seconds = int(job.get("restart_baseline_cloud_billed_seconds", 0) or 0)
    billed_seconds = baseline_billed_seconds + current_billed_seconds
    asr_cost = float(job.get("restart_baseline_asr_cost_cny", 0) or 0) + current_asr_cost
    current_total_cost = max(
        float(manifest.get("estimated_cost_cny", 0) or 0),
        current_asr_cost + text_cost,
    )
    baseline_cost = float(job.get("restart_baseline_cost_cny", 0) or 0)
    baseline_text_cost = float(job.get("restart_baseline_text_review_cost_cny", 0) or 0)
    text_cost += baseline_text_cost
    total_cost = max(baseline_cost + current_total_cost, asr_cost + text_cost)
    public_error = job.get("error")
    warning = None
    if manifest_messages:
        if status == "failed":
            public_error = manifest_messages[-1]
        elif status == "completed":
            warning = manifest_messages[-1]
    recovery_decision = _recovery_decision(
        job_path,
        job,
        completed_steps,
        str(public_error or warning or job.get("last_error") or ""),
        total_cost,
        text_cost,
    )
    recovery_options = [recovery_decision] if recovery_decision is not None else []
    cloud_override = _cloud_cost_override_decision(
        job_path,
        job,
        str(public_error or warning or job.get("last_error") or ""),
        total_cost,
    )
    if cloud_override is not None and all(
        option.get("strategy") != cloud_override["strategy"] for option in recovery_options
    ):
        recovery_options.append(cloud_override)
    full_restart = _full_restart_decision(job_path, job, total_cost)
    if (
        full_restart is not None
        and (recovery_decision is None or bool(recovery_decision.get("can_continue")))
        and all(option.get("strategy") != full_restart["strategy"] for option in recovery_options)
    ):
        recovery_options.append(full_restart)

    return {
        **job,
        "status": status,
        "error": public_error,
        "warning": warning,
        "core_transcript_ready": core_transcript_ready,
        "text_review_status": text_review_status,
        "recovery_decision": recovery_decision,
        "recovery_options": recovery_options,
        "stage": stage,
        "progress_percent": progress,
        "completed_steps": completed_steps,
        "cloud_billed_seconds": billed_seconds,
        "asr_cost_cny": round(asr_cost, 6),
        "text_review_input_tokens": text_input,
        "text_review_output_tokens": text_output,
        "text_review_total_tokens": text_input + text_output,
        "text_review_cost_cny": round(text_cost, 6),
        "text_review_cost_cap_cny": float(
            job.get(
                "text_review_cost_cap_cny",
                manifest.get("text_review_cost_cap_cny", 0),
            )
            or 0
        ),
        "estimated_cost_cny": round(total_cost, 6),
        "cost_cap_cny": float(job.get("cost_cap_cny", manifest.get("cost_cap_cny", 3)) or 3),
    }


def _job_failure_message(job_path: Path, stdout: str, stderr: str) -> str:
    manifest_paths = list(job_path.parent.glob("transcription/**/processing_manifest.json"))
    if manifest_paths:
        try:
            manifest = _read_json(max(manifest_paths, key=lambda path: path.stat().st_mtime_ns))
            messages = [
                str(item).strip() for item in manifest.get("errors", []) if str(item).strip()
            ]
            if messages:
                return messages[-1][-2000:]
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return (stderr or stdout or "transcription process failed").strip()[-2000:]


def _is_retryable_failure(message: str) -> bool:
    normalized = message.casefold()
    non_retryable_markers = (
        "api_key is required",
        "requires --allow-cloud-upload",
        "speech-only cloud mode requires",
        "unsupported audio",
        "no supported audio",
        "at least one transcription mode",
        "cost cap",
        "audio file exceeds",
        "deepseek",
        "window json",
        "topic stubs",
        "consolidation",
        "text review",
        "model not found",
        "model does not exist",
        "unauthorized",
        "forbidden",
        "invalid parameter",
        "bad request",
        "cuda out of memory",
        "cuda is not available",
        "disk full",
        "no space",
    )
    return not any(marker in normalized for marker in non_retryable_markers)


def _transcription_command(
    source_path: Path,
    *,
    run_cloud: bool,
    run_text_review: bool,
    cost_cap_cny: float,
    text_review_cost_cap_cny: float,
) -> tuple[list[str], Path]:
    output = source_path.parent / "transcription" / "转文字文档.md"
    command = [
        sys.executable,
        "-m",
        "research_kb",
        "transcribe-recordings",
        str(source_path),
        "--output",
        str(output),
        "--resume",
        "--no-third-pass",
        "--deepseek-cost-cap-cny",
        str(text_review_cost_cap_cny),
        "--cost-cap-cny",
        str(cost_cap_cny),
        "--json",
    ]
    command.extend(
        ["--no-local", "--cloud", "--allow-cloud-upload", "--speech-only-cloud"]
        if run_cloud
        else ["--local", "--no-cloud"]
    )
    command.append("--deepseek-text" if run_text_review else "--no-deepseek-text")
    return command, output


def _run_transcription_job(job_path: Path, source_path: Path, allow_cloud: bool) -> None:
    # One heavyweight pipeline at a time prevents concurrent uploads from exhausting
    # GPU memory or multiplying pressure on the same cloud API.
    with _TRANSCRIPTION_RUN_LOCK:
        while True:
            with _JOB_STATE_LOCK:
                job = _read_json(job_path)
                if bool(job.get("cancel_requested")):
                    job.update(
                        status="cancelled",
                        finished_at=time.time(),
                        error=None,
                        retrying=False,
                    )
                    _write_job(job_path, job)
                    return
                maximum = int(job.get("max_attempts", MAX_JOB_ATTEMPTS) or MAX_JOB_ATTEMPTS)
                attempt = int(job.get("attempt_count", 0) or 0) + 1
                job.update(
                    status="running",
                    started_at=job.get("started_at") or time.time(),
                    attempt_count=attempt,
                    max_attempts=maximum,
                    retrying=attempt > 1,
                    error=None,
                )
                _write_job(job_path, job)
            run_cloud = bool(job.get("run_cloud_enabled", allow_cloud))
            run_text_review = bool(job.get("text_review_enabled", False)) and run_cloud
            command, output = _transcription_command(
                source_path,
                run_cloud=run_cloud,
                run_text_review=run_text_review,
                cost_cap_cny=float(job.get("restart_cost_cap_cny", job.get("cost_cap_cny", 3))),
                text_review_cost_cap_cny=float(
                    job.get(
                        "restart_text_review_cost_cap_cny",
                        job.get("text_review_cost_cap_cny", 0.3),
                    )
                ),
            )
            job_id = str(job.get("id") or job_path.parent.name)
            process: subprocess.Popen[str] | None = None
            stdout = ""
            stderr = ""
            returncode = -1
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                with _JOB_STATE_LOCK:
                    _JOB_PROCESSES[job_id] = process
                    latest = _read_json(job_path)
                    if bool(latest.get("cancel_requested")) and process.poll() is None:
                        process.terminate()
                stdout, stderr = process.communicate()
                returncode = int(process.returncode or 0)
            except OSError as error:
                stderr = str(error)
            finally:
                with _JOB_STATE_LOCK:
                    _JOB_PROCESSES.pop(job_id, None)

            with _JOB_STATE_LOCK:
                job = _read_json(job_path)
                if bool(job.get("cancel_requested")):
                    job.update(
                        status="cancelled",
                        finished_at=time.time(),
                        error=None,
                        retrying=False,
                    )
                    _write_job(job_path, job)
                    return
                if returncode == 0:
                    job.update(
                        status="completed",
                        finished_at=time.time(),
                        output=output.as_posix() if output.is_file() else None,
                        error=None,
                        retrying=False,
                    )
                    _write_job(job_path, job)
                    return

                message = _job_failure_message(job_path, stdout, stderr)
                if attempt < maximum and _is_retryable_failure(message):
                    job.update(
                        status="running",
                        retry_count=attempt,
                        retrying=True,
                        last_error=message,
                        error=None,
                    )
                    _write_job(job_path, job)
                    continue
                job.update(
                    status="failed",
                    finished_at=time.time(),
                    output=output.as_posix() if output.is_file() else None,
                    error=message,
                    last_error=message,
                    retrying=False,
                )
                _write_job(job_path, job)
                return


def _text_review_artifacts(job_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    manifest_paths = list(job_path.parent.glob("transcription/**/processing_manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError("completed transcription manifest is missing")
    manifest_path = max(manifest_paths, key=lambda path: path.stat().st_mtime_ns)
    work_directory = manifest_path.parent
    transcript_path = work_directory / "transcript.json"
    if not transcript_path.is_file():
        raise FileNotFoundError("completed transcript.json is missing")
    output_path = job_path.parent / "transcription" / "转文字文档.md"
    topic_path = output_path.with_name(f"{output_path.stem}_话题索引.md")
    cache_directory = work_directory / "providers" / "deepseek"
    return manifest_path, transcript_path, output_path, topic_path, cache_directory


def _run_text_review_repair(job_path: Path) -> None:
    """Repair only the derived text layer; never invoke or upload to ASR providers."""
    with _TRANSCRIPTION_RUN_LOCK:
        with _JOB_STATE_LOCK:
            job = _read_json(job_path)
            job.update(
                status="running",
                operation="text_review_repair",
                started_at=time.time(),
                error=None,
                repair_error=None,
            )
            _write_job(job_path, job)
        try:
            manifest_path, transcript_path, output_path, topic_path, cache_directory = (
                _text_review_artifacts(job_path)
            )
            manifest = _read_json(manifest_path)
            settings = AppSettings()
            if settings.deepseek_api_key is None:
                raise TranscriptTextReviewError(
                    "DEEPSEEK_API_KEY is required for text-review repair"
                )
            document = TranscriptDocument.model_validate_json(
                transcript_path.read_text(encoding="utf-8")
            )
            reviewer = DeepSeekTranscriptReviewer(
                api_key=settings.deepseek_api_key.get_secret_value(),
                model=str(job.get("text_review_model") or "deepseek-v4-flash"),
                cost_cap_cny=float(job.get("text_review_cost_cap_cny", 0.3) or 0.3),
                cache_directory=cache_directory,
                allow_consolidation_fallback=True,
                allow_window_cost_fallback=True,
            )
            reviewed, analysis = reviewer.review(document)
            write_json_model(analysis, transcript_path.with_name("text_analysis.json"))
            write_text_atomic(render_topic_markdown(reviewed, analysis), topic_path)
            write_json_model(reviewed, transcript_path)
            write_text_atomic(
                render_markdown(reviewed, structured_path="transcription/transcript.json"),
                output_path,
            )
            input_tokens = sum(
                item.prompt_cache_hit_tokens + item.prompt_cache_miss_tokens
                for item in analysis.usage
            )
            output_tokens = sum(item.output_tokens for item in analysis.usage)
            errors = [
                str(item)
                for item in manifest.get("errors", [])
                if not str(item).startswith("文本整理已回退")
            ]
            if analysis.window_fallback_count:
                errors.append(
                    f"文本整理已完成: {analysis.window_fallback_count} 个窗口因费用上限保留原始识别"
                )
            completed_steps = list(
                dict.fromkeys([*manifest.get("completed_steps", []), "deepseek_text"])
            )
            asr_cost = int(manifest.get("cloud_billed_seconds", 0) or 0) * ASR_PRICE_PER_SECOND_CNY
            manifest.update(
                completed_steps=completed_steps,
                errors=errors,
                text_review_model=analysis.model,
                text_review_requests=len(analysis.usage),
                text_review_input_tokens=input_tokens,
                text_review_output_tokens=output_tokens,
                text_review_cost_cny=analysis.estimated_cost_cny,
                estimated_cost_cny=round(asr_cost + analysis.estimated_cost_cny, 6),
                updated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            )
            _write_json(manifest_path, manifest)
        except Exception as error:
            message = (
                str(error)
                if isinstance(error, TranscriptTextReviewError)
                else f"text-review repair failed ({type(error).__name__})"
            )
            try:
                manifest_path, *_rest = _text_review_artifacts(job_path)
                manifest = _read_json(manifest_path)
                errors = [
                    str(item)
                    for item in manifest.get("errors", [])
                    if not str(item).startswith("文本整理已回退")
                ]
                manifest["errors"] = [*errors, f"文本整理已回退: {message}"]
                _write_json(manifest_path, manifest)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            with _JOB_STATE_LOCK:
                job = _read_json(job_path)
                job.update(
                    status="completed",
                    finished_at=time.time(),
                    operation=None,
                    repair_error=message,
                    error=None,
                )
                _write_job(job_path, job)
            return
        with _JOB_STATE_LOCK:
            job = _read_json(job_path)
            job.update(
                status="completed",
                finished_at=time.time(),
                operation=None,
                repair_error=None,
                error=None,
                text_review_repaired_at=time.time(),
            )
            for key in ("decision_cancelled_at", "last_error"):
                job.pop(key, None)
            _write_job(job_path, job)


def _start_text_review_repair_thread(job_path: Path) -> bool:
    job_id = job_path.parent.name
    with _JOB_STATE_LOCK:
        existing = _JOB_THREADS.get(job_id)
        if existing is not None and existing.is_alive():
            return False

        def run() -> None:
            try:
                _run_text_review_repair(job_path)
            finally:
                with _JOB_STATE_LOCK:
                    _JOB_THREADS.pop(job_id, None)

        thread = threading.Thread(target=run, name=f"text-review-{job_id}", daemon=True)
        _JOB_THREADS[job_id] = thread
    thread.start()
    return True


def _run_recording_summary(
    catalog: RecordingCatalog, entry: RecordingEntry, prompt: str, state_path: Path
) -> None:
    """Generate only the derived summary layer for an existing transcript."""
    with _TRANSCRIPTION_RUN_LOCK:

        def update_state(**changes: Any) -> None:
            with _JOB_STATE_LOCK:
                state = _read_json(state_path)
                state.update(changes)
                _write_json(state_path, state)

        def report_progress(progress_percent: int, stage: str) -> None:
            update_state(
                status="running",
                progress_percent=max(0, min(99, progress_percent)),
                stage=stage,
            )

        with _JOB_STATE_LOCK:
            state = _read_json(state_path)
            state.update(
                status="running",
                started_at=time.time(),
                progress_percent=5,
                stage="正在读取完整转写",
                error=None,
            )
            _write_json(state_path, state)
        try:
            if entry.transcript_path is None:
                raise TranscriptTextReviewError("transcript is not ready")
            settings = AppSettings()
            if settings.deepseek_api_key is None:
                raise TranscriptTextReviewError("DEEPSEEK_API_KEY is required for summary")
            document = TranscriptDocument.model_validate_json(
                entry.transcript_path.read_text(encoding="utf-8")
            )
            reviewer = DeepSeekTranscriptReviewer(
                api_key=settings.deepseek_api_key.get_secret_value(),
                base_url=settings.deepseek_base_url,
                model=settings.deepseek_model,
                timeout_seconds=settings.deepseek_timeout_seconds,
                cost_cap_cny=0.3,
                cache_directory=entry.transcript_path.parent / "providers" / "deepseek",
                allow_consolidation_fallback=True,
                allow_window_cost_fallback=True,
                user_prompt=prompt or None,
                progress_callback=report_progress,
            )
            report_progress(12, "正在准备文本分析")
            reviewed, analysis = reviewer.review(document)
            report_progress(92, "正在保存总结结果")
            result_path = entry.transcript_path.with_name(SUMMARY_RESULT_FILENAME)
            _write_json(
                result_path,
                {
                    "schema_version": "1.0",
                    "prompt": prompt,
                    "model": analysis.model,
                    "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "analysis": analysis.model_dump(mode="json"),
                },
            )
            topic_path = entry.transcript_path.with_name(SUMMARY_MARKDOWN_FILENAME)
            write_text_atomic(render_topic_markdown(reviewed, analysis), topic_path)
            # Refresh the catalog before publishing ``completed``. The UI stops polling as
            # soon as it sees that state, so publishing in the opposite order can expose a
            # completed request with the previous (empty) topic index.
            catalog.refresh()
            update_state(
                status="completed",
                finished_at=time.time(),
                progress_percent=100,
                stage="总结完成",
                model=analysis.model,
                cost_cny=analysis.estimated_cost_cny,
                topic_count=len(analysis.topics),
                prompt=prompt,
                error=None,
                can_retry=True,
            )
        except Exception as error:
            message = (
                str(error)
                if isinstance(error, TranscriptTextReviewError)
                else (f"summary failed ({type(error).__name__})")
            )
            with _JOB_STATE_LOCK:
                failed_attempts = int(_read_json(state_path).get("attempt_count", 0) or 0)
            update_state(
                status="failed",
                finished_at=time.time(),
                stage="总结失败，可重试",
                error=message,
                can_retry=failed_attempts < MAX_SUMMARY_ATTEMPTS,
            )


def _start_recording_summary_thread(
    catalog: RecordingCatalog, entry: RecordingEntry, prompt: str, state_path: Path
) -> bool:
    with _JOB_STATE_LOCK:
        existing = _SUMMARY_THREADS.get(entry.id)
        if existing is not None and existing.is_alive():
            return False

        def run() -> None:
            try:
                _run_recording_summary(catalog, entry, prompt, state_path)
            finally:
                with _JOB_STATE_LOCK:
                    _SUMMARY_THREADS.pop(entry.id, None)

        thread = threading.Thread(target=run, name=f"summary-{entry.id}", daemon=True)
        _SUMMARY_THREADS[entry.id] = thread
        thread.start()
        return True


def _start_transcription_thread(job_path: Path, source_path: Path, allow_cloud: bool) -> bool:
    job_id = job_path.parent.name
    with _JOB_STATE_LOCK:
        existing = _JOB_THREADS.get(job_id)
        if existing is not None and existing.is_alive():
            return False

        def run() -> None:
            try:
                _run_transcription_job(job_path, source_path, allow_cloud)
            finally:
                with _JOB_STATE_LOCK:
                    _JOB_THREADS.pop(job_id, None)

        thread = threading.Thread(target=run, name=f"transcribe-{job_id}", daemon=True)
        _JOB_THREADS[job_id] = thread
        thread.start()
        return True


def _recover_incomplete_jobs(upload_root: Path) -> None:
    for job_path in upload_root.glob("job-*/job.json"):
        try:
            with _JOB_STATE_LOCK:
                job = _read_json(job_path)
                previous_status = str(job.get("status", "queued"))
                if previous_status not in {"queued", "running"}:
                    continue
                source_path = next(
                    (path for path in job_path.parent.glob("recording.*") if path.is_file()),
                    None,
                )
                if source_path is None:
                    continue
                if bool(job.get("cancel_requested")):
                    job.update(status="cancelled", finished_at=time.time(), error=None)
                    _write_job(job_path, job)
                    continue
                job.update(
                    status="queued",
                    attempt_count=max(
                        0,
                        int(job.get("attempt_count", 0) or 0)
                        - (1 if previous_status == "running" else 0),
                    ),
                    recovery_count=int(job.get("recovery_count", 0) or 0) + 1,
                    recovered_at=time.time(),
                    retrying=False,
                )
                _write_job(job_path, job)
            if job.get("operation") == "text_review_repair":
                _start_text_review_repair_thread(job_path)
            else:
                _start_transcription_thread(
                    job_path,
                    source_path,
                    bool(job.get("allow_cloud_upload")),
                )
        except (OSError, ValueError, json.JSONDecodeError):
            continue


def _recover_incomplete_summaries(catalog: RecordingCatalog) -> None:
    """Turn orphaned in-process summaries into an explicit retryable failure."""
    for entry in catalog.entries(include_hidden=True):
        if entry.transcript_path is None:
            continue
        state_path = entry.transcript_path.with_name("summary_request.json")
        try:
            with _JOB_STATE_LOCK:
                state = _read_json(state_path)
                if str(state.get("status")) not in {"queued", "running"}:
                    continue
                state.update(
                    status="failed",
                    finished_at=time.time(),
                    stage="工作台重启，任务已中断",
                    error="工作台重启，中断了本次总结；可以点击重试。",
                    can_retry=True,
                )
                _write_json(state_path, state)
        except (OSError, ValueError, json.JSONDecodeError):
            continue


def _job_path(upload_root: Path, job_id: str) -> Path:
    if not job_id.startswith("job-") or any(part in job_id for part in ("/", "\\", "..")):
        raise HTTPException(404, "job not found")
    path = upload_root / job_id / "job.json"
    if not path.is_file():
        raise HTTPException(404, "job not found")
    return path


def _matching_upload_job(
    upload_root: Path,
    *,
    sha256: str,
    allow_cloud_upload: bool,
    exclude_id: str,
) -> tuple[Path, Path] | None:
    candidates: list[tuple[tuple[int, int, float], Path, Path]] = []
    for job_path in upload_root.glob("job-*/job.json"):
        if job_path.parent.name == exclude_id:
            continue
        try:
            job = _read_json(job_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        # “移除记录”只隐藏任务卡。隐藏过的失败任务不能拦截同一文件的新上传,
        # 否则用户无法通过重新添加原录音获得全新的处理任务。
        if job.get("dismissed_at"):
            continue
        if job.get("sha256") != sha256 or bool(job.get("allow_cloud_upload")) != allow_cloud_upload:
            continue
        source_path = next(
            (path for path in job_path.parent.glob("recording.*") if path.is_file()),
            None,
        )
        if source_path is None:
            continue
        status = str(job.get("status", "queued"))
        status_priority = (
            4 if status == "completed" else 3 if status in {"queued", "running"} else 2
        )
        manifest_paths = list(job_path.parent.glob("transcription/**/processing_manifest.json"))
        completed_count = 0
        if manifest_paths:
            try:
                manifest = _read_json(max(manifest_paths, key=lambda path: path.stat().st_mtime_ns))
                completed_count = len(manifest.get("completed_steps", []))
            except (OSError, ValueError, json.JSONDecodeError):
                completed_count = 0
        candidates.append(
            (
                (
                    status_priority,
                    completed_count,
                    float(job.get("retry_requested_at", job.get("created_at", 0)) or 0),
                ),
                job_path,
                source_path,
            )
        )
    if not candidates:
        return None
    _score, job_path, source_path = max(candidates, key=lambda item: item[0])
    return job_path, source_path


def create_app(data_root: Path | None = None) -> FastAPI:
    """Create the loopback-only application."""
    root = (data_root or Path("data")).resolve()
    catalog = RecordingCatalog(root)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        _recover_incomplete_jobs(catalog.upload_root)
        _recover_incomplete_summaries(catalog)
        yield

    app = FastAPI(
        title="语迹 VoiceTrace",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "Range"],
        expose_headers=["Accept-Ranges", "Content-Range", "Content-Length"],
    )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "recordings": len(catalog.entries())}

    @app.get("/api/summary-prompts")
    def summary_prompts() -> list[dict[str, Any]]:
        with _JOB_STATE_LOCK:
            payload = _summary_prompts_payload(catalog.summary_prompts_path)
            if not catalog.summary_prompts_path.is_file():
                _save_summary_prompts(catalog.summary_prompts_path, payload["prompts"])
            return cast(list[dict[str, Any]], payload["prompts"])

    @app.post("/api/summary-prompts", status_code=201)
    def create_summary_prompt(request: SummaryPromptPayload) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        item = {
            "id": f"prompt-{uuid4().hex}",
            "name": request.name,
            "prompt": request.prompt,
            "created_at": now,
            "updated_at": now,
        }
        with _JOB_STATE_LOCK:
            payload = _summary_prompts_payload(catalog.summary_prompts_path)
            payload["prompts"].append(item)
            _save_summary_prompts(catalog.summary_prompts_path, payload["prompts"])
        return item

    @app.put("/api/summary-prompts/{prompt_id}")
    def update_summary_prompt(prompt_id: str, request: SummaryPromptPayload) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        with _JOB_STATE_LOCK:
            payload = _summary_prompts_payload(catalog.summary_prompts_path)
            prompts = payload["prompts"]
            item = next((item for item in prompts if item.get("id") == prompt_id), None)
            if item is None:
                raise HTTPException(404, "summary prompt not found")
            item.update(name=request.name, prompt=request.prompt, updated_at=now)
            _save_summary_prompts(catalog.summary_prompts_path, prompts)
            return cast(dict[str, Any], item)

    @app.get("/api/recordings")
    def recordings(q: str | None = None, include_hidden: bool = False) -> list[dict[str, Any]]:
        catalog.refresh()
        return [
            _entry_payload(catalog, entry, detail=False)
            for entry in catalog.entries(include_hidden=include_hidden, query=q)
        ]

    @app.get("/api/favorites")
    def all_favorites() -> list[dict[str, Any]]:
        catalog.refresh()
        return catalog.all_favorites()

    @app.get("/api/recordings/{recording_id}")
    def recording(recording_id: str) -> dict[str, Any]:
        catalog.refresh()
        try:
            entry = catalog.get(recording_id)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        return _entry_payload(catalog, entry, detail=True)

    @app.get("/api/recordings/{recording_id}/export/summary.md", response_model=None)
    def export_summary(recording_id: str) -> Response:
        catalog.refresh()
        try:
            entry = catalog.get(recording_id)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        if entry.transcript_path is None:
            raise HTTPException(409, "transcript is not ready")
        topics = catalog.topics(entry)
        state = catalog.summary_state(entry)
        if not topics and state.get("status") != "completed":
            raise HTTPException(409, "summary is not ready")
        content = _render_summary_export(catalog, entry, topics)
        return _markdown_download(
            content,
            f"{_download_stem(catalog.effective_title(entry))}-总结.md",
        )

    @app.get("/api/recordings/{recording_id}/export/transcript.md", response_model=None)
    def export_transcript(recording_id: str) -> Response:
        catalog.refresh()
        try:
            entry = catalog.get(recording_id)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        if entry.transcript_path is None:
            raise HTTPException(409, "transcript is not ready")
        content = _render_transcript_export(catalog, entry)
        return _markdown_download(
            content,
            f"{_download_stem(catalog.effective_title(entry))}-完整对话.md",
        )

    @app.post("/api/recordings/{recording_id}/summary", status_code=202)
    def summarize_recording(recording_id: str, request: SummaryRequest) -> dict[str, Any]:
        catalog.refresh()
        try:
            entry = catalog.get(recording_id)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        if entry.transcript_path is None:
            raise HTTPException(409, "transcript is not ready")
        state_path = entry.transcript_path.with_name("summary_request.json")
        with _JOB_STATE_LOCK:
            try:
                previous = _read_json(state_path)
            except (OSError, ValueError, json.JSONDecodeError):
                previous = {}
            if str(previous.get("status")) in {"queued", "running"}:
                raise HTTPException(409, "summary is already running")
            previous_attempts = int(previous.get("attempt_count", 0) or 0)
            if (
                str(previous.get("status")) == "failed"
                and previous_attempts >= MAX_SUMMARY_ATTEMPTS
            ):
                raise HTTPException(409, "summary retry limit reached")
            attempt_count = previous_attempts + 1 if str(previous.get("status")) == "failed" else 1
            state = {
                "status": "queued",
                "prompt": request.prompt,
                "requested_at": time.time(),
                "error": None,
                "progress_percent": 0,
                "stage": "等待开始",
                "attempt_count": attempt_count,
                "max_attempts": MAX_SUMMARY_ATTEMPTS,
                "retry_count": max(0, attempt_count - 1),
                "can_retry": False,
            }
            _write_json(state_path, state)
        if not _start_recording_summary_thread(catalog, entry, request.prompt, state_path):
            with _JOB_STATE_LOCK:
                state.update(
                    status="failed",
                    finished_at=time.time(),
                    stage="无法启动总结，可重试",
                    error="总结任务未能启动，请稍后重试。",
                    can_retry=True,
                )
                _write_json(state_path, state)
            raise HTTPException(409, "summary is already running")
        return {"recording_id": recording_id, **state}

    @app.put("/api/recordings/{recording_id}/title")
    def set_recording_title(recording_id: str, update: RecordingTitleUpdate) -> dict[str, Any]:
        try:
            entry = catalog.set_title(recording_id, update.title)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        return _entry_payload(catalog, entry, detail=True)

    @app.delete("/api/recordings/{recording_id}/title")
    def reset_recording_title(recording_id: str) -> dict[str, Any]:
        try:
            entry = catalog.set_title(recording_id, None)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        return _entry_payload(catalog, entry, detail=True)

    @app.delete("/api/recordings/{recording_id}")
    def hide_recording(recording_id: str) -> dict[str, Any]:
        try:
            entry = catalog.set_hidden(recording_id, True)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        return _entry_payload(catalog, entry, detail=True)

    @app.post("/api/recordings/{recording_id}/restore")
    def restore_recording(recording_id: str) -> dict[str, Any]:
        try:
            entry = catalog.set_hidden(recording_id, False)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        return _entry_payload(catalog, entry, detail=True)

    @app.put("/api/recordings/{recording_id}/start-time")
    def set_recording_start_time(
        recording_id: str, update: RecordingStartTimeUpdate
    ) -> dict[str, Any]:
        try:
            entry = catalog.set_recorded_at(
                recording_id, update.recorded_at.isoformat(timespec="seconds")
            )
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        return _entry_payload(catalog, entry, detail=True)

    @app.delete("/api/recordings/{recording_id}/start-time")
    def reset_recording_start_time(recording_id: str) -> dict[str, Any]:
        try:
            entry = catalog.set_recorded_at(recording_id, None)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        return _entry_payload(catalog, entry, detail=True)

    @app.get("/api/recordings/{recording_id}/speakers")
    def speakers(recording_id: str) -> list[dict[str, Any]]:
        try:
            entry = catalog.get(recording_id)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        return catalog.speakers(entry)

    @app.put("/api/recordings/{recording_id}/speaker-overrides")
    def set_speaker_override(
        recording_id: str, update: SpeakerOverrideUpdate
    ) -> list[dict[str, Any]]:
        try:
            entry = catalog.get(recording_id)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        try:
            catalog.set_speaker_override(entry, update.source_speaker, update.display_name)
        except ValueError as error:
            raise HTTPException(404, "speaker not found") from error
        return catalog.speakers(entry)

    @app.delete("/api/recordings/{recording_id}/speaker-overrides")
    def reset_speaker_override(recording_id: str, source_speaker: str) -> list[dict[str, Any]]:
        try:
            entry = catalog.get(recording_id)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        try:
            catalog.set_speaker_override(entry, source_speaker, None)
        except ValueError as error:
            raise HTTPException(404, "speaker not found") from error
        return catalog.speakers(entry)

    @app.get("/api/recordings/{recording_id}/favorites")
    def favorites(recording_id: str) -> list[dict[str, Any]]:
        try:
            entry = catalog.get(recording_id)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        return catalog.favorites(entry)

    @app.put("/api/recordings/{recording_id}/favorites/{segment_id}")
    def add_favorite(recording_id: str, segment_id: str) -> list[dict[str, Any]]:
        try:
            entry = catalog.get(recording_id)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        try:
            catalog.set_favorite(entry, segment_id, True)
        except ValueError as error:
            raise HTTPException(404, "segment not found") from error
        return catalog.favorites(entry)

    @app.delete("/api/recordings/{recording_id}/favorites/{segment_id}")
    def remove_favorite(recording_id: str, segment_id: str) -> list[dict[str, Any]]:
        try:
            entry = catalog.get(recording_id)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        try:
            catalog.set_favorite(entry, segment_id, False)
        except ValueError as error:
            raise HTTPException(404, "segment not found") from error
        return catalog.favorites(entry)

    @app.put("/api/recordings/{recording_id}/favorites/{segment_id}/note")
    def update_favorite_note(
        recording_id: str, segment_id: str, update: FavoriteNoteUpdate
    ) -> dict[str, Any]:
        try:
            entry = catalog.get(recording_id)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        try:
            return catalog.set_favorite_note(entry, segment_id, update.note)
        except ValueError as error:
            raise HTTPException(404, "favorite not found") from error

    @app.get("/api/recordings/{recording_id}/activity")
    def activity(recording_id: str, limit: int = 100) -> list[dict[str, Any]]:
        try:
            entry = catalog.get(recording_id)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        if limit < 1 or limit > 500:
            raise HTTPException(400, "limit must be between 1 and 500")
        return catalog.activity(entry, limit)

    @app.get("/api/recordings/{recording_id}/blocks")
    def blocks(recording_id: str, start_ms: int = 0, end_ms: int | None = None) -> dict[str, Any]:
        try:
            entry = catalog.get(recording_id)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        if start_ms < 0:
            raise HTTPException(400, "start_ms must be non-negative")
        bounded_end = min(entry.duration_ms, end_ms or start_ms + WINDOW_MS)
        if bounded_end <= start_ms:
            raise HTTPException(400, "end_ms must be after start_ms")
        return {
            "start_ms": start_ms,
            "end_ms": bounded_end,
            "blocks": catalog.reading_blocks(entry, start_ms, bounded_end),
        }

    @app.get("/api/recordings/{recording_id}/audio", response_model=None)
    def audio(
        recording_id: str, request: Request
    ) -> StreamingResponse | FileResponse | JSONResponse:
        try:
            entry = catalog.get(recording_id)
        except KeyError as error:
            raise HTTPException(404, "recording not found") from error
        size = entry.audio_path.stat().st_size
        try:
            start, end, partial = _parse_range(request.headers.get("range"), size)
        except (ValueError, TypeError):
            return JSONResponse(
                {"detail": "requested range is not satisfiable"},
                status_code=416,
                headers={"Content-Range": f"bytes */{size}"},
            )
        media_type = mimetypes.guess_type(entry.audio_path.name)[0] or "application/octet-stream"
        if not partial:
            return FileResponse(
                entry.audio_path, media_type=media_type, headers={"Accept-Ranges": "bytes"}
            )
        length = end - start + 1
        return StreamingResponse(
            _file_chunks(entry.audio_path, start, end),
            status_code=206,
            media_type=media_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Content-Length": str(length),
            },
        )

    @app.post("/api/uploads", status_code=202)
    async def upload(
        background_tasks: BackgroundTasks,
        file: Annotated[UploadFile, File()],
        # Default to the product's low-cost path: VAD filters silence/noise
        # before the selected speech intervals are sent to cloud ASR.
        allow_cloud_upload: Annotated[bool, Form()] = True,
    ) -> dict[str, Any]:
        suffix = Path(file.filename or "").suffix.casefold()
        if suffix not in SUPPORTED_AUDIO:
            raise HTTPException(415, "unsupported audio format")
        upload_id = f"job-{int(time.time())}-{os.urandom(4).hex()}"
        target_dir = catalog.upload_root / upload_id
        target_dir.mkdir(parents=True, exist_ok=False)
        temporary = target_dir / "upload.part"
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "audio file exceeds 2 GB")
                    digest.update(chunk)
                    stream.write(chunk)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        source_path = target_dir / f"recording{suffix}"
        temporary.replace(source_path)
        sha256 = digest.hexdigest()
        with _JOB_STATE_LOCK:
            matched = _matching_upload_job(
                catalog.upload_root,
                sha256=sha256,
                allow_cloud_upload=allow_cloud_upload,
                exclude_id=upload_id,
            )
            if matched is not None:
                matched_job_path, matched_source_path = matched
                matched_job = _read_json(matched_job_path)
                matched_status = str(matched_job.get("status", "queued"))
                shutil.rmtree(target_dir)
                if matched_status in {"failed", "cancelled"}:
                    for key in (
                        "cancel_requested",
                        "cancel_requested_at",
                        "dismissed_at",
                        "error",
                        "finished_at",
                        "output",
                        "started_at",
                        "attempt_count",
                        "retry_count",
                        "retrying",
                        "last_error",
                        "decision_cancelled_at",
                        "decision_strategy",
                        "decision_requested_at",
                    ):
                        matched_job.pop(key, None)
                    matched_job.update(
                        filename=file.filename,
                        size_bytes=size,
                        status="queued",
                        max_attempts=MAX_JOB_ATTEMPTS,
                        run_cloud_enabled=allow_cloud_upload,
                        text_review_enabled=allow_cloud_upload,
                        retry_requested_at=time.time(),
                    )
                    _write_job(matched_job_path, matched_job)
                    background_tasks.add_task(
                        _run_transcription_job,
                        matched_job_path,
                        matched_source_path,
                        allow_cloud_upload,
                    )
                return _public_job(matched_job_path)
        job_path = target_dir / "job.json"
        job = {
            "id": upload_id,
            "filename": file.filename,
            "sha256": sha256,
            "size_bytes": size,
            "status": "queued",
            "max_attempts": MAX_JOB_ATTEMPTS,
            "allow_cloud_upload": allow_cloud_upload,
            "run_cloud_enabled": allow_cloud_upload,
            # Cloud authorization also opts into the one-time initial text review.
            # Later user-triggered summaries are stored separately and never touch
            # this pipeline flag or its manifest fields.
            "text_review_enabled": allow_cloud_upload,
            "cost_cap_cny": 3,
            "text_review_cost_cap_cny": 0.3 if allow_cloud_upload else 0,
            "created_at": time.time(),
        }
        _write_job(job_path, job)
        background_tasks.add_task(_run_transcription_job, job_path, source_path, allow_cloud_upload)
        return _public_job(job_path)

    @app.get("/api/jobs")
    def jobs() -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for job_path in catalog.upload_root.glob("job-*/job.json"):
            try:
                record = _public_job(job_path)
                if not record.get("dismissed_at"):
                    records.append(record)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        records.sort(
            key=lambda item: float(item.get("retry_requested_at", item.get("created_at", 0)) or 0),
            reverse=True,
        )
        return records[:20]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        return _public_job(_job_path(catalog.upload_root, job_id))

    @app.post("/api/jobs/{job_id}/continue", status_code=202)
    def continue_job(job_id: str, request: JobContinueRequest | None = None) -> dict[str, Any]:
        job_path = _job_path(catalog.upload_root, job_id)
        public = _public_job(job_path)
        decision = public.get("recovery_decision")
        requested_strategy = request.strategy if request is not None else None
        if requested_strategy:
            options = public.get("recovery_options", [])
            decision = next(
                (
                    option
                    for option in options
                    if isinstance(option, dict)
                    and str(option.get("strategy")) == requested_strategy
                ),
                None,
            )
        if not isinstance(decision, dict) or not bool(decision.get("can_continue")):
            raise HTTPException(409, "this job cannot be continued safely")
        strategy = str(decision["strategy"])
        text_review_strategies = {"retry_text_review_only", "extend_text_review_budget"}
        source_path: Path | None = None
        if strategy not in text_review_strategies:
            source_path = next(
                (path for path in job_path.parent.glob("recording.*") if path.is_file()),
                None,
            )
            if source_path is None:
                raise HTTPException(409, "source audio is missing")
        with _JOB_STATE_LOCK:
            stored = _read_json(job_path)
            allowed_statuses = (
                {"completed"} if strategy in text_review_strategies else {"failed", "cancelled"}
            )
            if str(stored.get("status", "queued")) not in allowed_statuses:
                raise HTTPException(409, "this recovery strategy is not valid for the job status")
            if strategy == "finalize_without_text_review":
                stored["run_cloud_enabled"] = bool(
                    stored.get("run_cloud_enabled", stored.get("allow_cloud_upload"))
                )
                stored["text_review_enabled"] = False
            elif strategy == "continue_local_only":
                stored["run_cloud_enabled"] = False
                stored["text_review_enabled"] = False
            elif strategy == "continue_cloud_with_higher_cap":
                stored["run_cloud_enabled"] = True
                stored["text_review_enabled"] = True
                stored["cost_cap_cny"] = float(decision["cost_cap_cny"])
            elif strategy == "resume_fixed_pipeline":
                stored["run_cloud_enabled"] = bool(stored.get("allow_cloud_upload"))
                stored["text_review_enabled"] = bool(stored.get("allow_cloud_upload"))
            elif strategy in text_review_strategies:
                stored["operation"] = "text_review_repair"
                stored["text_review_enabled"] = True
                if strategy == "extend_text_review_budget":
                    stored["text_review_cost_cap_cny"] = float(decision["text_review_cost_cap_cny"])
            elif strategy == "restart_from_scratch":
                transcription_root = job_path.parent / "transcription"
                if transcription_root.is_symlink():
                    raise HTTPException(409, "transcription cache path is not a directory")
                stored["restart_baseline_cost_cny"] = float(
                    public.get("estimated_cost_cny", 0) or 0
                )
                stored["restart_baseline_asr_cost_cny"] = float(public.get("asr_cost_cny", 0) or 0)
                stored["restart_baseline_text_review_cost_cny"] = float(
                    public.get("text_review_cost_cny", 0) or 0
                )
                stored["restart_baseline_cloud_billed_seconds"] = int(
                    public.get("cloud_billed_seconds", 0) or 0
                )
                stored["restart_cost_cap_cny"] = max(
                    0.0,
                    float(stored.get("cost_cap_cny", 3) or 3)
                    - float(public.get("estimated_cost_cny", 0) or 0),
                )
                stored["restart_text_review_cost_cap_cny"] = float(
                    stored.get("text_review_cost_cap_cny", 0.3) or 0.3
                )
                if transcription_root.is_dir():
                    shutil.rmtree(transcription_root)
                stored["run_cloud_enabled"] = bool(stored.get("allow_cloud_upload"))
                stored["text_review_enabled"] = bool(stored.get("allow_cloud_upload"))
            stored.update(
                status="queued",
                attempt_count=0,
                retry_count=0,
                retrying=False,
                max_attempts=MAX_JOB_ATTEMPTS,
                error=None,
                decision_strategy=strategy,
                decision_requested_at=time.time(),
                retry_requested_at=time.time(),
            )
            for key in (
                "cancel_requested",
                "cancel_requested_at",
                "decision_cancelled_at",
                "dismissed_at",
                "finished_at",
                "output",
                "last_error",
            ):
                stored.pop(key, None)
            _write_job(job_path, stored)
        if strategy in text_review_strategies:
            _start_text_review_repair_thread(job_path)
        else:
            assert source_path is not None
            _start_transcription_thread(
                job_path,
                source_path,
                bool(stored.get("allow_cloud_upload")),
            )
        return _public_job(job_path)

    @app.post("/api/jobs/{job_id}/decision/cancel")
    def cancel_failed_job(job_id: str) -> dict[str, Any]:
        job_path = _job_path(catalog.upload_root, job_id)
        with _JOB_STATE_LOCK:
            stored = _read_json(job_path)
            status = str(stored.get("status", "queued"))
            if status not in {"failed", "cancelled", "completed"}:
                raise HTTPException(409, "only terminal jobs can dismiss a recovery decision")
            stored.update(decision_cancelled_at=time.time(), retrying=False)
            if status in {"failed", "cancelled"}:
                stored.update(
                    status="cancelled",
                    finished_at=stored.get("finished_at") or time.time(),
                )
            _write_job(job_path, stored)
        return _public_job(job_path)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        job_path = _job_path(catalog.upload_root, job_id)
        with _JOB_STATE_LOCK:
            job = _read_json(job_path)
            status = str(job.get("status", "queued"))
            if status not in {"queued", "running"}:
                raise HTTPException(409, "only queued or running jobs can be cancelled")
            job.update(cancel_requested=True, cancel_requested_at=time.time())
            process = _JOB_PROCESSES.get(job_id)
            if status == "queued" and process is None:
                job.update(status="cancelled", finished_at=time.time(), error=None)
            _write_job(job_path, job)
            if process is not None and process.poll() is None:
                process.terminate()
        return _public_job(job_path)

    @app.delete("/api/jobs/{job_id}")
    def dismiss_job(job_id: str) -> dict[str, Any]:
        job_path = _job_path(catalog.upload_root, job_id)
        with _JOB_STATE_LOCK:
            job = _read_json(job_path)
            if str(job.get("status", "queued")) in {"queued", "running"}:
                raise HTTPException(409, "cancel an active job before removing its record")
            job["dismissed_at"] = time.time()
            _write_job(job_path, job)
        return _public_job(job_path)

    return app


def serve_audio_review(*, data_root: Path, frontend_directory: Path, open_browser: bool) -> None:
    """Start the frontend when available, then serve the local API."""
    frontend: subprocess.Popen[str] | None = None
    if frontend_directory.is_dir() and (frontend_directory / "node_modules").is_dir():
        npm_command = _npm_command()
        try:
            if npm_command is None:
                raise OSError("npm runtime was not found")
            env = _frontend_environment(npm_command)
            frontend = subprocess.Popen(
                [*npm_command, "run", "dev"], cwd=frontend_directory, env=env, text=True
            )
        except OSError:
            frontend = None
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open("http://localhost:3000")).start()
    try:
        import uvicorn

        uvicorn.run(create_app(data_root), host="127.0.0.1", port=8765, log_level="info")
    finally:
        if frontend is not None:
            frontend.terminate()


def _npm_command() -> list[str] | None:
    """Locate npm without requiring a system-wide Node installation."""
    configured = os.environ.get("NPM_EXECUTABLE")
    if configured:
        return [configured]
    system_npm = shutil.which("npm")
    if system_npm:
        return [system_npm]
    home = Path.home()
    node_candidates = sorted(
        (home / ".cache" / "codex-runtimes").glob("*/dependencies/node/bin/node.exe")
    )
    npm_candidates = sorted(
        (home / ".workbuddy" / "binaries" / "node" / "versions").glob(
            "*/node_modules/npm/bin/npm-cli.js"
        )
    )
    if node_candidates and npm_candidates:
        return [str(node_candidates[-1]), str(npm_candidates[-1])]
    return None


def _frontend_environment(npm_command: list[str]) -> dict[str, str]:
    """Expose a bundled Node executable to scripts launched by npm."""
    environment = {**os.environ, "WRANGLER_LOG_PATH": ".wrangler/wrangler.log"}
    executable = Path(npm_command[0])
    if executable.name.casefold() in {"node", "node.exe"}:
        node_directory = str(executable.parent)
        current_path = environment.get("PATH", "")
        path_entries = current_path.split(os.pathsep) if current_path else []
        if node_directory.casefold() not in {entry.casefold() for entry in path_entries}:
            environment["PATH"] = os.pathsep.join((node_directory, *path_entries))
    return environment
