"""Tests for the local long-recording playback API."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from research_kb import audio_web
from research_kb.audio.text_review import TranscriptTextAnalysis, TranscriptTopic
from research_kb.audio_web import RecordingCatalog, _parse_range, create_app


def _write_catalog(tmp_path: Path) -> tuple[Path, bytes]:
    data = tmp_path / "data"
    transcription = data / "sample" / "transcription"
    transcription.mkdir(parents=True)
    audio = data / "sample" / "recording.mp3"
    payload = b"0123456789abcdefghijklmnopqrstuvwxyz"
    audio.write_bytes(payload)
    sha = "a" * 64
    document = {
        "schema_version": "1.0",
        "title": "sample",
        "generated_at": datetime.now(UTC).isoformat(),
        "sources": [
            {
                "recording_id": "recording_01",
                "path": str(audio),
                "filename": "recording.mp3",
                "sha256": sha,
                "size_bytes": len(payload),
                "duration_ms": 200_000,
                "recorded_at": "2026-08-05T13:20:00+08:00",
            }
        ],
        "speakers": [],
        "segments": [
            {
                "segment_id": "segment_1",
                "recording_id": "recording_01",
                "start_ms": 1_000,
                "end_ms": 5_000,
                "speaker": "Speaker A",
                "text": "First sentence.",
                "confidence": 0.8,
                "flags": [],
                "candidates": [{"provider": "local", "model": "test", "text": "First sentence"}],
            },
            {
                "segment_id": "segment_2",
                "recording_id": "recording_01",
                "start_ms": 6_000,
                "end_ms": 9_000,
                "speaker": "Speaker B",
                "text": "Second sentence.",
                "confidence": 0.7,
                "flags": ["unclear"],
                "candidates": [],
            },
            {
                "segment_id": "segment_3",
                "recording_id": "recording_01",
                "start_ms": 25_000,
                "end_ms": 31_000,
                "speaker": "Speaker B",
                "text": "A new paragraph after a long silence.",
                "confidence": 0.6,
                "flags": [],
                "candidates": [],
            },
        ],
    }
    (transcription / "transcript.json").write_text(
        json.dumps(document, ensure_ascii=False), encoding="utf-8"
    )
    analysis = {
        "topics": [
            {
                "topic_id": "topic_1",
                "title": "Opening",
                "summary": "Test topic",
                "keywords": ["test"],
                "start_segment_id": "segment_1",
                "end_segment_id": "segment_2",
                "evidence_segment_ids": ["segment_1", "segment_2"],
            }
        ]
    }
    (transcription / "text_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False), encoding="utf-8"
    )
    return data, payload


def test_catalog_groups_readable_blocks_and_maps_topics(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    catalog = RecordingCatalog(data)
    entry = catalog.entries()[0]

    assert entry.status == "completed"
    assert len(catalog.segments(entry)) == 3
    topics = catalog.topics(entry)
    assert topics[0]["start_ms"] == 1_000
    assert topics[0]["strength"] == "strong"
    assert topics[0]["segment_count"] == 2
    assert len(topics) == 1
    assert sum(topic["segment_count"] for topic in topics) == 2
    blocks = catalog.reading_blocks(entry, 0, 60_000)
    assert [len(block["sentences"]) for block in blocks] == [2, 1]
    assert blocks[0]["sentences"][0]["candidates"][0]["provider"] == "local"
    density = catalog.density(entry, bins=4)
    assert len(density) == 4
    assert density[0] > 0


def test_markdown_exports_include_summary_and_complete_dialogue(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    client = TestClient(create_app(data))
    recording_id = client.get("/api/recordings").json()[0]["id"]

    summary = client.get(f"/api/recordings/{recording_id}/export/summary.md")
    transcript = client.get(f"/api/recordings/{recording_id}/export/transcript.md")

    assert summary.status_code == 200
    assert summary.headers["content-type"].startswith("text/markdown")
    assert "# recording.mp3 | Summary" in summary.text
    assert "### 01. Opening" in summary.text
    assert "Test topic" in summary.text
    assert "attachment;" in summary.headers["content-disposition"]
    assert transcript.status_code == 200
    assert "# recording.mp3 | Full transcript" in transcript.text
    assert "### 00:00:01–00:00:05 · Speaker A" in transcript.text
    assert "First sentence." in transcript.text
    assert "### 00:00:25–00:00:31 · Speaker B" in transcript.text
    catalog = RecordingCatalog(data)
    empty_summary = audio_web._render_summary_export(catalog, catalog.entries()[0], [])
    assert "No content matched this prompt" in empty_summary


def test_catalog_recovers_original_upload_filename_from_job_metadata(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-original-name"
    audio_web._write_job(
        job_dir / "job.json",
        {
            "id": "job-original-name",
            "filename": "recorder-20260812-0846.m4a",
            "sha256": "a" * 64,
            "created_at": 123.0,
            "status": "completed",
        },
    )

    listing = TestClient(create_app(data)).get("/api/recordings").json()

    assert listing[0]["title"] == "recorder-20260812-0846.m4a"
    assert listing[0]["original_title"] == "recorder-20260812-0846.m4a"
    assert listing[0]["title_overridden"] is False


def test_api_lists_windows_and_streams_byte_ranges(tmp_path: Path) -> None:
    data, payload = _write_catalog(tmp_path)
    client = TestClient(create_app(data))

    health = client.get("/api/health")
    assert health.json() == {"status": "ok", "recordings": 1}
    listing = client.get("/api/recordings").json()
    recording_id = listing[0]["id"]
    assert listing[0]["has_topics"] is True
    detail = client.get(f"/api/recordings/{recording_id}").json()
    assert len(detail["density"]) == 240
    assert detail["topics"][0]["title"] == "Opening"
    assert detail["topic_segment_count"] == 2
    assert detail["segment_count"] == 3
    window = client.get(
        f"/api/recordings/{recording_id}/blocks", params={"start_ms": 0, "end_ms": 40_000}
    ).json()
    assert len(window["blocks"]) == 2

    partial = client.get(f"/api/recordings/{recording_id}/audio", headers={"Range": "bytes=5-11"})
    assert partial.status_code == 206
    assert partial.content == payload[5:12]
    assert partial.headers["content-range"] == f"bytes 5-11/{len(payload)}"
    suffix = client.get(f"/api/recordings/{recording_id}/audio", headers={"Range": "bytes=-4"})
    assert suffix.content == payload[-4:]
    whole = client.get(f"/api/recordings/{recording_id}/audio")
    assert whole.content == payload
    invalid = client.get(f"/api/recordings/{recording_id}/audio", headers={"Range": "bytes=99-100"})
    assert invalid.status_code == 416
    assert client.get("/api/recordings/missing").status_code == 404
    assert client.get("/api/recordings/missing/blocks").status_code == 404


def test_catalog_starts_without_topics_until_user_requests_summary(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    (data / "sample" / "transcription" / "text_analysis.json").unlink()
    catalog = RecordingCatalog(data)
    entry = catalog.entries()[0]

    topics = catalog.topics(entry)
    assert topics == []


def test_summary_endpoint_runs_on_demand_and_persists_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, _ = _write_catalog(tmp_path)
    transcription = data / "sample" / "transcription"
    (transcription / "text_analysis.json").unlink()
    captured: dict[str, object] = {}
    audio_path = data / "sample" / "recording.mp3"
    document = audio_web.TranscriptDocument.model_validate(
        {
            "generated_at": datetime.now(UTC),
            "sources": [
                {
                    "recording_id": "recording_01",
                    "path": str(audio_path),
                    "filename": "recording.mp3",
                    "sha256": "a" * 64,
                    "size_bytes": audio_path.stat().st_size,
                    "duration_ms": 200_000,
                    "recorded_at": "2026-08-05T13:20:00+08:00",
                }
            ],
            "speakers": [],
            "segments": [
                {
                    "segment_id": "segment_1",
                    "recording_id": "recording_01",
                    "start_ms": 1_000,
                    "end_ms": 5_000,
                    "speaker": "Speaker A",
                    "text": "Test transcript",
                    "decision": "local_primary",
                    "confidence": 0.8,
                    "candidates": [
                        {
                            "provider": "local",
                            "model": "test",
                            "text": "Test transcript",
                            "start_ms": 1_000,
                            "end_ms": 5_000,
                        }
                    ],
                }
            ],
        }
    )
    audio_web.write_json_model(document, transcription / "transcript.json")
    analysis = TranscriptTextAnalysis(
        generated_at=datetime.now(UTC),
        model="deepseek-test",
        source_transcript_sha256="a" * 64,
        decisions=(),
        topics=(
            TranscriptTopic(
                topic_id="topic_0001",
                title="Finance content",
                start_segment_id="segment_1",
                end_segment_id="segment_1",
                summary="Keep finance views.",
                keywords=("finance",),
                evidence_segment_ids=("segment_1",),
            ),
        ),
        meetings=(),
        usage=(),
        estimated_cost_cny=0.01,
        cost_cap_cny=0.3,
        user_prompt="Keep only finance content",
    )

    class FakeReviewer:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def review(self, received: object) -> tuple[object, TranscriptTextAnalysis]:
            assert received == document
            return document, analysis

    secret = SimpleNamespace(get_secret_value=lambda: "temporary")
    monkeypatch.setattr(
        audio_web,
        "AppSettings",
        lambda: SimpleNamespace(
            deepseek_api_key=secret,
            deepseek_base_url="https://example.test",
            deepseek_model="deepseek-test",
            deepseek_timeout_seconds=10,
        ),
    )
    monkeypatch.setattr(audio_web, "DeepSeekTranscriptReviewer", FakeReviewer)
    client = TestClient(create_app(data))
    recording_id = client.get("/api/recordings").json()[0]["id"]
    analysis_path = transcription / "text_analysis.json"
    original_analysis_exists = analysis_path.exists()
    manifest_path = transcription / "processing_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "completed_steps": ["preprocess", "local", "speakers", "render"],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    original_manifest = manifest_path.read_bytes()
    original_transcript = (transcription / "transcript.json").read_bytes()

    response = client.post(
        f"/api/recordings/{recording_id}/summary",
        json={"prompt": "Keep only finance content"},
    )

    assert response.status_code == 202
    deadline = time.time() + 3
    detail: dict[str, object] = {}
    while time.time() < deadline:
        detail = client.get(f"/api/recordings/{recording_id}").json()
        if detail.get("summary", {}).get("status") == "completed":  # type: ignore[union-attr]
            break
        time.sleep(0.01)
    assert detail["summary"]["status"] == "completed"  # type: ignore[index]
    assert detail["topics"][0]["title"] == "Finance content"  # type: ignore[index]
    assert captured["user_prompt"] == "Keep only finance content"
    saved = json.loads((transcription / "summary_result.json").read_text(encoding="utf-8"))
    assert saved["prompt"] == "Keep only finance content"
    assert saved["analysis"]["user_prompt"] == "Keep only finance content"
    assert (transcription / "topic-index.md").is_file()
    assert analysis_path.exists() is original_analysis_exists
    assert (transcription / "processing_manifest.json").read_bytes() == original_manifest
    assert (transcription / "transcript.json").read_bytes() == original_transcript


def test_summary_prompt_library_supports_create_update_and_reload(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    client = TestClient(create_app(data))

    initial = client.get("/api/summary-prompts")
    assert initial.status_code == 200
    assert {item["name"] for item in initial.json()} >= {"Complete meeting", "Finance livestream"}

    created = client.post(
        "/api/summary-prompts",
        json={"name": "Project retrospective", "prompt": "Keep questions, decisions, and owners."},
    )
    assert created.status_code == 201
    prompt = created.json()
    updated = client.put(
        f"/api/summary-prompts/{prompt['id']}",
        json={"name": "Project retrospective", "prompt": "Keep questions, decisions, owners, and deadlines."},
    )
    assert updated.status_code == 200
    assert updated.json()["prompt"].endswith("deadlines.")

    reloaded = TestClient(create_app(data)).get("/api/summary-prompts")
    assert any(item["prompt"].endswith("deadlines.") for item in reloaded.json())
    assert (
        client.put(
            "/api/summary-prompts/prompt-does-not-exist",
            json={"name": "Missing", "prompt": "Invalid"},
        ).status_code
        == 404
    )


def test_api_persists_and_resets_recording_start_time(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    client = TestClient(create_app(data))
    original_document = (data / "sample" / "transcription" / "transcript.json").read_bytes()
    recording_id = client.get("/api/recordings").json()[0]["id"]

    updated = client.put(
        f"/api/recordings/{recording_id}/start-time",
        json={"recorded_at": "2026-08-05T08:20:00+08:00"},
    )
    assert updated.status_code == 200
    assert updated.json()["recorded_at"] == "2026-08-05T08:20:00+08:00"
    assert updated.json()["original_recorded_at"] == "2026-08-05T13:20:00+08:00"
    assert updated.json()["start_time_overridden"] is True
    assert (data / "audio-review" / "recording-time-overrides.json").is_file()
    assert (data / "sample" / "transcription" / "transcript.json").read_bytes() == original_document

    restarted = TestClient(create_app(data))
    assert restarted.get("/api/recordings").json()[0]["recorded_at"] == (
        "2026-08-05T08:20:00+08:00"
    )
    reset = restarted.delete(f"/api/recordings/{recording_id}/start-time")
    assert reset.status_code == 200
    assert reset.json()["recorded_at"] == "2026-08-05T13:20:00+08:00"
    assert reset.json()["start_time_overridden"] is False

    assert (
        client.put(
            f"/api/recordings/{recording_id}/start-time",
            json={"recorded_at": "2026-08-05T08:20:00"},
        ).status_code
        == 422
    )
    assert (
        client.put(
            "/api/recordings/missing/start-time",
            json={"recorded_at": "2026-08-05T08:20:00+08:00"},
        ).status_code
        == 404
    )


def test_api_applies_recording_wide_speaker_overrides_without_changing_source(
    tmp_path: Path,
) -> None:
    data, _ = _write_catalog(tmp_path)
    transcript_path = data / "sample" / "transcription" / "transcript.json"
    original_document = transcript_path.read_bytes()
    client = TestClient(create_app(data))
    recording_id = client.get("/api/recordings").json()[0]["id"]

    speakers = client.get(f"/api/recordings/{recording_id}/speakers")
    assert speakers.status_code == 200
    assert speakers.json() == [
        {
            "source_speaker": "Speaker A",
            "display_name": "Speaker A",
            "segment_count": 1,
            "is_overridden": False,
        },
        {
            "source_speaker": "Speaker B",
            "display_name": "Speaker B",
            "segment_count": 2,
            "is_overridden": False,
        },
    ]

    updated = client.put(
        f"/api/recordings/{recording_id}/speaker-overrides",
        json={"source_speaker": "Speaker B", "display_name": "Project owner"},
    )
    assert updated.status_code == 200
    assert updated.json()[1]["display_name"] == "Project owner"
    window = client.get(
        f"/api/recordings/{recording_id}/blocks",
        params={"start_ms": 0, "end_ms": 40_000},
    ).json()
    sentences = [sentence for block in window["blocks"] for sentence in block["sentences"]]
    assert [sentence["speaker"] for sentence in sentences] == [
        "Speaker A",
        "Project owner",
        "Project owner",
    ]
    assert sentences[1]["original_speaker"] == "Speaker B"
    assert transcript_path.read_bytes() == original_document

    restarted = TestClient(create_app(data))
    persisted = restarted.get(f"/api/recordings/{recording_id}/speakers").json()
    assert persisted[1]["display_name"] == "Project owner"
    reset = restarted.request(
        "DELETE",
        f"/api/recordings/{recording_id}/speaker-overrides",
        params={"source_speaker": "Speaker B"},
    )
    assert reset.status_code == 200
    assert reset.json()[1]["display_name"] == "Speaker B"
    assert (
        restarted.put(
            f"/api/recordings/{recording_id}/speaker-overrides",
            json={"source_speaker": "Missing", "display_name": "Someone"},
        ).status_code
        == 404
    )


def test_api_persists_favorites_and_keeps_append_only_activity_log(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    client = TestClient(create_app(data))
    recording_id = client.get("/api/recordings").json()[0]["id"]

    created = client.put(f"/api/recordings/{recording_id}/favorites/segment_2")
    assert created.status_code == 200
    assert created.json()[0]["segment_id"] == "segment_2"
    assert created.json()[0]["text"] == "Second sentence."
    assert client.put(f"/api/recordings/{recording_id}/favorites/segment_2").status_code == 200

    restarted = TestClient(create_app(data))
    favorites = restarted.get(f"/api/recordings/{recording_id}/favorites").json()
    assert [favorite["segment_id"] for favorite in favorites] == ["segment_2"]
    removed = restarted.delete(f"/api/recordings/{recording_id}/favorites/segment_2")
    assert removed.status_code == 200
    assert removed.json() == []

    activity = restarted.get(f"/api/recordings/{recording_id}/activity").json()
    assert [event["action"] for event in activity] == ["favorite_removed", "favorite_added"]
    assert len({event["event_id"] for event in activity}) == 2
    log_path = data / "audio-review" / "activity-log.jsonl"
    assert log_path.is_file()
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 2
    assert (
        restarted.put(f"/api/recordings/{recording_id}/favorites/missing-segment").status_code
        == 404
    )


def test_api_aggregates_favorites_and_persists_editable_notes(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    transcript_path = data / "sample" / "transcription" / "transcript.json"
    original_document = transcript_path.read_bytes()
    client = TestClient(create_app(data))
    recording = client.get("/api/recordings").json()[0]
    recording_id = recording["id"]
    assert client.put(f"/api/recordings/{recording_id}/favorites/segment_2").status_code == 200

    aggregated = client.get("/api/favorites")
    assert aggregated.status_code == 200
    assert aggregated.json() == [
        {
            "recording_id": recording_id,
            "recording_title": recording["title"],
            "recorded_at": recording["recorded_at"],
            "segment_id": "segment_2",
            "start_ms": 6_000,
            "end_ms": 9_000,
            "speaker": "Speaker B",
            "text": "Second sentence.",
            "created_at": aggregated.json()[0]["created_at"],
            "note": "",
            "note_updated_at": None,
        }
    ]

    updated = client.put(
        f"/api/recordings/{recording_id}/favorites/segment_2/note",
        json={"note": "Follow up on this issue and add the missing context."},
    )
    assert updated.status_code == 200
    assert updated.json()["note"] == "Follow up on this issue and add the missing context."
    assert updated.json()["note_updated_at"]
    assert transcript_path.read_bytes() == original_document

    restarted = TestClient(create_app(data))
    persisted = restarted.get("/api/favorites").json()[0]
    assert persisted["note"] == "Follow up on this issue and add the missing context."
    assert persisted["recording_title"] == "recording.mp3"

    cleared = restarted.put(
        f"/api/recordings/{recording_id}/favorites/segment_2/note",
        json={"note": ""},
    )
    assert cleared.status_code == 200
    assert cleared.json()["note"] == ""
    actions = [
        event["action"]
        for event in restarted.get(f"/api/recordings/{recording_id}/activity").json()
    ]
    assert actions[:3] == ["favorite_note_cleared", "favorite_note_updated", "favorite_added"]
    assert (
        restarted.put(
            f"/api/recordings/{recording_id}/favorites/missing-segment/note",
            json={"note": "Missing"},
        ).status_code
        == 404
    )
    assert (
        restarted.put(
            f"/api/recordings/{recording_id}/favorites/segment_2/note",
            json={"note": "x" * 2_001},
        ).status_code
        == 422
    )


def test_api_renames_searches_hides_and_restores_recordings_non_destructively(
    tmp_path: Path,
) -> None:
    data, _ = _write_catalog(tmp_path)
    transcript_path = data / "sample" / "transcription" / "transcript.json"
    original_document = transcript_path.read_bytes()
    client = TestClient(create_app(data))
    recording_id = client.get("/api/recordings").json()[0]["id"]

    renamed = client.put(
        f"/api/recordings/{recording_id}/title",
        json={"title": "August 5 meeting recording"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "August 5 meeting recording"
    assert renamed.json()["original_title"] == "recording.mp3"
    assert renamed.json()["title_overridden"] is True
    assert transcript_path.read_bytes() == original_document
    assert client.get("/api/recordings", params={"q": "meeting"}).json()[0]["id"] == recording_id
    assert client.get("/api/recordings", params={"q": "missing"}).json() == []

    removed = client.delete(f"/api/recordings/{recording_id}")
    assert removed.status_code == 200
    assert removed.json()["hidden"] is True
    assert client.get("/api/recordings").json() == []
    hidden = client.get("/api/recordings", params={"include_hidden": "true"}).json()
    assert hidden[0]["title"] == "August 5 meeting recording"
    assert hidden[0]["hidden"] is True
    assert transcript_path.read_bytes() == original_document

    restarted = TestClient(create_app(data))
    assert restarted.get("/api/recordings").json() == []
    restored = restarted.post(f"/api/recordings/{recording_id}/restore")
    assert restored.status_code == 200
    assert restored.json()["hidden"] is False
    reset = restarted.delete(f"/api/recordings/{recording_id}/title")
    assert reset.status_code == 200
    assert reset.json()["title"] == "recording.mp3"
    assert reset.json()["title_overridden"] is False
    assert transcript_path.read_bytes() == original_document

    actions = [
        event["action"]
        for event in restarted.get(f"/api/recordings/{recording_id}/activity").json()
    ]
    assert actions[:4] == [
        "recording_title_reset",
        "recording_restored",
        "recording_hidden",
        "recording_title_updated",
    ]
    assert (
        restarted.put(f"/api/recordings/{recording_id}/title", json={"title": "  "}).status_code
        == 422
    )
    assert (
        restarted.put("/api/recordings/missing/title", json={"title": "missing"}).status_code == 404
    )


def test_window_validation_and_upload_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data, _ = _write_catalog(tmp_path)
    monkeypatch.setattr(audio_web, "_run_transcription_job", lambda *_args: None)
    client = TestClient(create_app(data))
    recording_id = client.get("/api/recordings").json()[0]["id"]
    assert (
        client.get(f"/api/recordings/{recording_id}/blocks", params={"start_ms": -1}).status_code
        == 400
    )
    assert (
        client.get(
            f"/api/recordings/{recording_id}/blocks",
            params={"start_ms": 10_000, "end_ms": 5_000},
        ).status_code
        == 400
    )
    assert client.post("/api/uploads", files={"file": ("bad.txt", b"bad")}).status_code == 415
    uploaded = client.post(
        "/api/uploads",
        files={"file": ("new.m4a", b"audio")},
        data={"allow_cloud_upload": "false"},
    )
    assert uploaded.status_code == 202
    job = uploaded.json()
    assert job["status"] == "queued"
    assert job["progress_percent"] == 0
    assert job["stage"] == "Waiting to start"
    assert client.get(f"/api/jobs/{job['id']}").json()["sha256"]
    assert client.get("/api/jobs").json()[0]["id"] == job["id"]
    assert client.get("/api/jobs/../bad").status_code == 404

    cloud_upload = client.post(
        "/api/uploads",
        files={"file": ("cloud.m4a", b"cloud audio")},
        data={"allow_cloud_upload": "true"},
    )
    assert cloud_upload.status_code == 202
    assert cloud_upload.json()["text_review_enabled"] is True

    duplicate = client.post(
        "/api/uploads",
        files={"file": ("same-again.m4a", b"audio")},
        data={"allow_cloud_upload": "false"},
    )
    assert duplicate.status_code == 202
    assert duplicate.json()["id"] == job["id"]
    assert len(list((data / "audio-review" / "uploads").glob("job-*"))) == 2


def test_dismissed_failed_upload_starts_a_fresh_job_by_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, _ = _write_catalog(tmp_path)
    upload_root = data / "audio-review" / "uploads"
    existing_dir = upload_root / "job-existing"
    existing_dir.mkdir(parents=True)
    (existing_dir / "recording.wav").write_bytes(b"same audio")
    audio_web._write_job(
        existing_dir / "job.json",
        {
            "id": "job-existing",
            "filename": "old.wav",
            "sha256": hashlib.sha256(b"same audio").hexdigest(),
            "status": "failed",
            "allow_cloud_upload": False,
            "error": "old error",
            "dismissed_at": 1,
            "created_at": 1,
        },
    )
    monkeypatch.setattr(audio_web, "_run_transcription_job", lambda *_args: None)

    response = TestClient(create_app(data)).post(
        "/api/uploads",
        files={"file": ("new.wav", b"same audio")},
        data={"allow_cloud_upload": "false"},
    )

    assert response.status_code == 202
    assert response.json()["id"] != "job-existing"
    assert response.json()["status"] == "queued"
    stored = audio_web._read_json(existing_dir / "job.json")
    assert stored["filename"] == "old.wav"
    assert stored["status"] == "failed"
    assert stored["dismissed_at"] == 1
    assert len(list(upload_root.glob("job-*"))) == 2


def test_hash_resume_prefers_the_job_with_more_completed_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, _ = _write_catalog(tmp_path)
    upload_root = data / "audio-review" / "uploads"
    digest = hashlib.sha256(b"same audio").hexdigest()
    for job_id, status, steps, created_at in (
        ("job-rich", "failed", ["preprocess", "cloud", "local", "speakers"], 1),
        ("job-new", "cancelled", ["preprocess", "cloud"], 2),
    ):
        job_dir = upload_root / job_id
        work_dir = job_dir / "transcription" / "transcription"
        work_dir.mkdir(parents=True)
        (job_dir / "recording.wav").write_bytes(b"same audio")
        audio_web._write_job(
            job_dir / "job.json",
            {
                "id": job_id,
                "filename": f"{job_id}.wav",
                "sha256": digest,
                "status": status,
                "allow_cloud_upload": True,
                "created_at": created_at,
            },
        )
        (work_dir / "processing_manifest.json").write_text(
            json.dumps(
                {"status": status if status == "failed" else "running", "completed_steps": steps}
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(audio_web, "_run_transcription_job", lambda *_args: None)

    response = TestClient(create_app(data)).post(
        "/api/uploads",
        files={"file": ("retry.wav", b"same audio")},
        data={"allow_cloud_upload": "true"},
    )

    assert response.status_code == 202
    assert response.json()["id"] == "job-rich"
    assert response.json()["status"] == "queued"


def test_failed_job_can_be_dismissed_without_deleting_its_files(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-failed"
    job_dir.mkdir(parents=True)
    job_path = job_dir / "job.json"
    source_path = job_dir / "recording.wav"
    source_path.write_bytes(b"original audio")
    audio_web._write_job(
        job_path,
        {
            "id": "job-failed",
            "filename": "failed.wav",
            "status": "failed",
            "error": "incompatible options",
            "created_at": 1,
        },
    )
    client = TestClient(create_app(data))

    removed = client.delete("/api/jobs/job-failed")

    assert removed.status_code == 200
    assert removed.json()["dismissed_at"] > 0
    assert client.get("/api/jobs").json() == []
    assert client.get("/api/jobs/job-failed").json()["status"] == "failed"
    assert job_path.is_file()
    assert source_path.read_bytes() == b"original audio"


def test_failed_job_can_clear_checkpoint_and_restart_from_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-restart"
    transcription_root = job_dir / "transcription"
    work_dir = transcription_root / "transcription"
    work_dir.mkdir(parents=True)
    (work_dir / "processing_manifest.json").write_text(
        json.dumps({"completed_steps": ["preprocess"]}), encoding="utf-8"
    )
    source = job_dir / "recording.m4a"
    source.write_bytes(b"original audio")
    job_path = job_dir / "job.json"
    audio_web._write_job(
        job_path,
        {
            "id": "job-restart",
            "filename": "meeting.m4a",
            "status": "failed",
            "allow_cloud_upload": False,
            "error": "transcription failed after partial checkpoint",
            "created_at": 1,
        },
    )
    started: list[tuple[Path, Path, bool]] = []
    monkeypatch.setattr(
        audio_web,
        "_start_transcription_thread",
        lambda job, audio, cloud: started.append((job, audio, cloud)) or True,
    )

    client = TestClient(create_app(data))
    public = client.get("/api/jobs/job-restart").json()
    assert any(
        option["strategy"] == "restart_from_scratch" for option in public["recovery_options"]
    )

    response = client.post(
        "/api/jobs/job-restart/continue",
        json={"strategy": "restart_from_scratch"},
    )

    assert response.status_code == 202
    assert not transcription_root.exists()
    assert source.read_bytes() == b"original audio"
    stored = audio_web._read_json(job_path)
    assert stored["status"] == "queued"
    assert stored["run_cloud_enabled"] is False
    assert stored["text_review_enabled"] is False
    assert stored["decision_strategy"] == "restart_from_scratch"
    assert started == [(job_path, source, False)]


def test_invalid_mp4_container_does_not_offer_a_repeatable_resume(
    tmp_path: Path,
) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-invalid-m4a"
    job_dir.mkdir(parents=True)
    (job_dir / "recording.m4a").write_bytes(b"not an m4a container")
    audio_web._write_job(
        job_dir / "job.json",
        {
            "id": "job-invalid-m4a",
            "filename": "recording.m4a",
            "status": "failed",
            "error": "missing mvhd atom: recording.m4a",
        },
    )

    job = TestClient(create_app(data)).get("/api/jobs/job-invalid-m4a").json()

    assert job["recovery_decision"]["strategy"] == "cannot_continue"
    assert job["recovery_decision"]["can_continue"] is False
    assert job["recovery_options"] == [job["recovery_decision"]]


def test_running_job_can_request_cancellation(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-running"
    job_dir.mkdir(parents=True)
    job_path = job_dir / "job.json"
    audio_web._write_job(
        job_path,
        {
            "id": "job-running",
            "filename": "running.wav",
            "status": "running",
            "created_at": 1,
        },
    )

    class RunningProcess:
        terminated = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

    process = RunningProcess()
    audio_web._JOB_PROCESSES["job-running"] = process  # type: ignore[assignment]
    try:
        response = TestClient(create_app(data)).post("/api/jobs/job-running/cancel")
    finally:
        audio_web._JOB_PROCESSES.pop("job-running", None)

    assert response.status_code == 200
    assert response.json()["cancel_requested"] is True
    assert response.json()["stage"] == "Cancelling"
    assert process.terminated is True
    stored = audio_web._read_json(job_path)
    assert stored["cancel_requested_at"] > 0


def test_queued_job_is_cancelled_without_waiting_for_running_job(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-queued"
    job_dir.mkdir(parents=True)
    audio_web._write_job(
        job_dir / "job.json",
        {"id": "job-queued", "filename": "queued.wav", "status": "queued"},
    )

    response = TestClient(create_app(data)).post("/api/jobs/job-queued/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.json()["stage"] == "Cancelled"


def test_failed_deepseek_job_offers_zero_cost_finalize_decision(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-deepseek"
    work_dir = job_dir / "transcription" / "transcription"
    work_dir.mkdir(parents=True)
    (job_dir / "recording.wav").write_bytes(b"audio")
    audio_web._write_job(
        job_dir / "job.json",
        {
            "id": "job-deepseek",
            "filename": "meeting.wav",
            "status": "failed",
            "allow_cloud_upload": True,
            "text_review_enabled": True,
            "error": "DeepSeek returned invalid window JSON",
        },
    )
    audio_web._write_job(
        work_dir / "processing_manifest.json",
        {
            "completed_steps": ["preprocess", "cloud", "local", "speakers"],
            "cloud_billed_seconds": 100,
        },
    )

    job = TestClient(create_app(data)).get("/api/jobs/job-deepseek").json()

    decision = job["recovery_decision"]
    assert decision["strategy"] == "finalize_without_text_review"
    assert decision["can_continue"] is True
    assert decision["additional_external_cost_cny"] == 0
    assert "DeepSeek" in decision["impact"]


def test_completed_deepseek_fallback_offers_text_only_repair(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-completed-review"
    work_dir = job_dir / "transcription" / "transcription"
    work_dir.mkdir(parents=True)
    (job_dir / "recording.wav").write_bytes(b"audio")
    audio_web._write_job(
        job_dir / "job.json",
        {
            "id": "job-completed-review",
            "filename": "meeting.wav",
            "status": "completed",
            "allow_cloud_upload": True,
            "text_review_enabled": True,
            "cost_cap_cny": 3,
            "text_review_cost_cap_cny": 0.3,
        },
    )
    audio_web._write_job(
        work_dir / "processing_manifest.json",
        {
            "completed_steps": [
                "preprocess",
                "cloud",
                "local",
                "speakers",
                "deepseek_text",
                "render",
            ],
            "cloud_billed_seconds": 100,
            "text_review_cost_cny": 0.08,
            "errors": ["Text review fallback: DeepSeek returned invalid window JSON"],
        },
    )

    job = TestClient(create_app(data)).get("/api/jobs/job-completed-review").json()

    decision = job["recovery_decision"]
    assert decision["strategy"] == "retry_text_review_only"
    assert decision["continue_label"] == "Repair text review"
    assert decision["additional_external_cost_cny"] == pytest.approx(0.22)
    assert "Audio will not be uploaded again and no ASR cost will be added" in decision["impact"]
    assert job["core_transcript_ready"] is True
    assert job["text_review_status"] == "fallback"
    assert job["stage"] == "Transcript ready · text review pending"
    assert "deepseek_text" not in job["completed_steps"]


def test_completed_partial_text_review_offers_bounded_text_only_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-partial-review"
    work_dir = job_dir / "transcription" / "transcription"
    work_dir.mkdir(parents=True)
    (job_dir / "recording.wav").write_bytes(b"audio")
    audio_web._write_job(
        job_dir / "job.json",
        {
            "id": "job-partial-review",
            "filename": "meeting.wav",
            "status": "completed",
            "allow_cloud_upload": True,
        },
    )
    audio_web._write_job(
        work_dir / "processing_manifest.json",
        {
            "completed_steps": [
                "preprocess",
                "cloud",
                "local",
                "speakers",
                "deepseek_text",
                "render",
            ],
            "errors": ["Text review complete: 2 windows kept the original recognition because of the cost cap"],
        },
    )

    job = TestClient(create_app(data)).get("/api/jobs/job-partial-review").json()

    assert job["warning"].startswith("Text review complete")
    decision = job["recovery_decision"]
    assert decision["strategy"] == "extend_text_review_budget"
    assert decision["continue_label"] == "Add budget and finish remaining windows"
    assert decision["additional_external_cost_cny"] <= 3
    assert "Audio will not be uploaded again and no ASR cost will be added" in decision["impact"]
    assert job["core_transcript_ready"] is True
    assert job["text_review_status"] == "partial"
    assert job["stage"] == "Transcript ready · text review partially complete"
    assert "deepseek_text" not in job["completed_steps"]
    repair_started: list[Path] = []
    transcription_started: list[object] = []
    monkeypatch.setattr(
        audio_web,
        "_start_text_review_repair_thread",
        lambda path: repair_started.append(path) or True,
    )
    monkeypatch.setattr(
        audio_web,
        "_start_transcription_thread",
        lambda *args: transcription_started.append(args) or True,
    )

    continued = TestClient(create_app(data)).post("/api/jobs/job-partial-review/continue")

    assert continued.status_code == 202
    stored = audio_web._read_json(job_dir / "job.json")
    assert stored["text_review_cost_cap_cny"] <= 3
    assert repair_started == [job_dir / "job.json"]
    assert transcription_started == []


def test_completed_fallback_continue_starts_only_text_review_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-completed-review"
    work_dir = job_dir / "transcription" / "transcription"
    work_dir.mkdir(parents=True)
    (job_dir / "recording.wav").write_bytes(b"audio")
    job_path = job_dir / "job.json"
    audio_web._write_job(
        job_path,
        {
            "id": "job-completed-review",
            "filename": "meeting.wav",
            "status": "completed",
            "allow_cloud_upload": True,
            "text_review_enabled": True,
        },
    )
    audio_web._write_job(
        work_dir / "processing_manifest.json",
        {
            "completed_steps": ["preprocess", "cloud", "local", "speakers", "render"],
            "errors": ["Text review fallback: DeepSeek returned invalid window JSON"],
        },
    )
    repair_started: list[Path] = []
    transcription_started: list[object] = []
    monkeypatch.setattr(
        audio_web,
        "_start_text_review_repair_thread",
        lambda path: repair_started.append(path) or True,
    )
    monkeypatch.setattr(
        audio_web,
        "_start_transcription_thread",
        lambda *args: transcription_started.append(args) or True,
    )

    response = TestClient(create_app(data)).post("/api/jobs/job-completed-review/continue")

    assert response.status_code == 202
    assert response.json()["stage"] == "Waiting for text review repair"
    stored = audio_web._read_json(job_path)
    assert stored["operation"] == "text_review_repair"
    assert repair_started == [job_path]
    assert transcription_started == []


def test_text_review_repair_writes_only_derived_text_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-repair-worker"
    work_dir = job_dir / "transcription" / "transcription"
    work_dir.mkdir(parents=True)
    transcript_path = work_dir / "transcript.json"
    document = audio_web.TranscriptDocument.model_validate(
        {
            "generated_at": datetime.now(UTC),
            "sources": [
                {
                    "recording_id": "recording_01",
                    "path": str(job_dir / "recording.wav"),
                    "filename": "recording.wav",
                    "sha256": "a" * 64,
                    "size_bytes": 5,
                    "duration_ms": 10_000,
                    "recorded_at": datetime.now(UTC),
                }
            ],
            "speakers": [],
            "segments": [
                {
                    "segment_id": "segment_0000001",
                    "recording_id": "recording_01",
                    "start_ms": 0,
                    "end_ms": 1_000,
                    "speaker": "Speaker A",
                    "text": "Test transcript",
                    "decision": "local_primary",
                    "confidence": 0.9,
                }
            ],
        }
    )
    audio_web.write_json_model(document, transcript_path)
    manifest_path = work_dir / "processing_manifest.json"
    audio_web._write_json(
        manifest_path,
        {
            "completed_steps": ["preprocess", "cloud", "local", "speakers", "render"],
            "cloud_billed_seconds": 100,
            "errors": ["Text review fallback: invalid window JSON"],
        },
    )
    job_path = job_dir / "job.json"
    audio_web._write_job(
        job_path,
        {
            "id": "job-repair-worker",
            "filename": "meeting.wav",
            "status": "queued",
            "text_review_cost_cap_cny": 0.3,
        },
    )
    analysis = TranscriptTextAnalysis(
        generated_at=datetime.now(UTC),
        model="deepseek-test",
        source_transcript_sha256="a" * 64,
        decisions=(),
        topics=(),
        meetings=(),
        usage=(),
        estimated_cost_cny=0.01,
        cost_cap_cny=0.3,
    )
    secret = SimpleNamespace(get_secret_value=lambda: "temporary")
    monkeypatch.setattr(audio_web, "AppSettings", lambda: SimpleNamespace(deepseek_api_key=secret))
    monkeypatch.setattr(
        audio_web,
        "DeepSeekTranscriptReviewer",
        lambda **_kwargs: SimpleNamespace(review=lambda _document: (document, analysis)),
    )

    audio_web._run_text_review_repair(job_path)

    stored = audio_web._read_json(job_path)
    manifest = audio_web._read_json(manifest_path)
    assert stored["status"] == "completed"
    assert stored["repair_error"] is None
    assert manifest["cloud_billed_seconds"] == 100
    assert manifest["errors"] == []
    assert manifest["text_review_model"] == "deepseek-test"
    assert (work_dir / "text_analysis.json").is_file()
    assert (job_dir / "transcription" / "transcript.md").is_file()
    assert (job_dir / "transcription" / "topic-index.md").is_file()


def test_text_review_repair_failure_keeps_completed_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, _ = _write_catalog(tmp_path)
    source_transcript = data / "sample" / "transcription" / "transcript.json"
    job_dir = data / "audio-review" / "uploads" / "job-repair-no-key"
    work_dir = job_dir / "transcription" / "transcription"
    work_dir.mkdir(parents=True)
    transcript_path = work_dir / "transcript.json"
    original = source_transcript.read_text(encoding="utf-8")
    transcript_path.write_text(original, encoding="utf-8")
    manifest_path = work_dir / "processing_manifest.json"
    audio_web._write_json(manifest_path, {"completed_steps": ["render"], "errors": []})
    job_path = job_dir / "job.json"
    audio_web._write_job(
        job_path,
        {"id": "job-repair-no-key", "filename": "meeting.wav", "status": "queued"},
    )
    monkeypatch.setattr(audio_web, "AppSettings", lambda: SimpleNamespace(deepseek_api_key=None))

    audio_web._run_text_review_repair(job_path)

    stored = audio_web._read_json(job_path)
    manifest = audio_web._read_json(manifest_path)
    assert stored["status"] == "completed"
    assert "DEEPSEEK_API_KEY" in stored["repair_error"]
    assert "DEEPSEEK_API_KEY" in manifest["errors"][0]
    assert transcript_path.read_text(encoding="utf-8") == original


def test_continue_endpoint_applies_diagnosed_strategy_and_resets_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-deepseek"
    job_dir.mkdir(parents=True)
    source = job_dir / "recording.wav"
    source.write_bytes(b"audio")
    job_path = job_dir / "job.json"
    audio_web._write_job(
        job_path,
        {
            "id": "job-deepseek",
            "filename": "meeting.wav",
            "status": "failed",
            "allow_cloud_upload": True,
            "text_review_enabled": True,
            "error": "DeepSeek consolidation failed",
            "attempt_count": 3,
            "retry_count": 2,
        },
    )
    started: list[tuple[Path, Path, bool]] = []
    monkeypatch.setattr(
        audio_web,
        "_start_transcription_thread",
        lambda job, audio, cloud: started.append((job, audio, cloud)) or True,
    )

    response = TestClient(create_app(data)).post("/api/jobs/job-deepseek/continue")

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    stored = audio_web._read_json(job_path)
    assert stored["text_review_enabled"] is False
    assert stored["run_cloud_enabled"] is True
    assert stored["attempt_count"] == 0
    assert stored["retry_count"] == 0
    assert stored["decision_strategy"] == "finalize_without_text_review"
    assert started == [(job_path, source, True)]


def test_cloud_cost_cap_decision_continues_locally_without_new_external_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-cap"
    job_dir.mkdir(parents=True)
    (job_dir / "recording.wav").write_bytes(b"audio")
    audio_web._write_job(
        job_dir / "job.json",
        {
            "id": "job-cap",
            "filename": "long.wav",
            "status": "failed",
            "allow_cloud_upload": True,
            "error": "all planned cloud speech regions would exceed the cost cap",
        },
    )
    audio_web._write_json(
        job_dir / "transcription" / "transcription" / "providers" / "cloud" / "speech-regions.json",
        {
            "regions": [
                {"start_ms": 0, "end_ms": 15_000_000},
            ]
        },
    )
    monkeypatch.setattr(audio_web, "_start_transcription_thread", lambda *_args: True)
    client = TestClient(create_app(data))

    job = client.get("/api/jobs/job-cap").json()
    diagnosed = job["recovery_decision"]
    response = client.post("/api/jobs/job-cap/continue")

    assert diagnosed["strategy"] == "continue_local_only"
    assert diagnosed["additional_external_cost_cny"] == 0
    assert [option["strategy"] for option in job["recovery_options"]] == [
        "continue_local_only",
        "continue_cloud_with_higher_cap",
        "restart_from_scratch",
    ]
    assert job["recovery_options"][1]["cost_cap_cny"] == 3.62
    assert response.status_code == 202
    stored = audio_web._read_json(job_dir / "job.json")
    assert stored["run_cloud_enabled"] is False
    assert stored["text_review_enabled"] is False


def test_cloud_cost_cap_decision_can_raise_bound_and_resume_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-cloud-cap"
    job_dir.mkdir(parents=True)
    source = job_dir / "recording.wav"
    source.write_bytes(b"audio")
    job_path = job_dir / "job.json"
    audio_web._write_job(
        job_path,
        {
            "id": "job-cloud-cap",
            "filename": "long.wav",
            "status": "failed",
            "allow_cloud_upload": True,
            "run_cloud_enabled": True,
            "text_review_enabled": True,
            "cost_cap_cny": 3,
            "text_review_cost_cap_cny": 0.3,
            "error": "all planned cloud speech regions would exceed the cost cap",
        },
    )
    audio_web._write_json(
        job_dir / "transcription" / "transcription" / "providers" / "cloud" / "speech-regions.json",
        {"regions": [{"start_ms": 0, "end_ms": 15_000_000}]},
    )
    started: list[tuple[Path, Path, bool]] = []
    monkeypatch.setattr(
        audio_web,
        "_start_transcription_thread",
        lambda job, audio, cloud: started.append((job, audio, cloud)) or True,
    )
    client = TestClient(create_app(data))

    response = client.post(
        "/api/jobs/job-cloud-cap/continue",
        json={"strategy": "continue_cloud_with_higher_cap"},
    )

    assert response.status_code == 202
    stored = audio_web._read_json(job_path)
    assert stored["run_cloud_enabled"] is True
    assert stored["text_review_enabled"] is True
    assert stored["cost_cap_cny"] == 3.62
    assert stored["decision_strategy"] == "continue_cloud_with_higher_cap"
    assert started == [(job_path, source, True)]


def test_unavailable_cloud_model_recommends_local_fallback(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-cloud-model"
    job_dir.mkdir(parents=True)
    (job_dir / "recording.wav").write_bytes(b"audio")
    audio_web._write_job(
        job_dir / "job.json",
        {
            "id": "job-cloud-model",
            "filename": "meeting.wav",
            "status": "failed",
            "allow_cloud_upload": True,
            "error": "cloud model not found",
        },
    )

    decision = (
        TestClient(create_app(data)).get("/api/jobs/job-cloud-model").json()["recovery_decision"]
    )

    assert decision["strategy"] == "continue_local_only"
    assert decision["additional_external_cost_cny"] == 0


def test_unrecoverable_missing_audio_only_allows_cancellation(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-missing"
    job_dir.mkdir(parents=True)
    audio_web._write_job(
        job_dir / "job.json",
        {
            "id": "job-missing",
            "filename": "missing.wav",
            "status": "failed",
            "error": "source audio is missing",
        },
    )
    client = TestClient(create_app(data))

    job = client.get("/api/jobs/job-missing").json()
    continued = client.post("/api/jobs/job-missing/continue")
    cancelled = client.post("/api/jobs/job-missing/decision/cancel")

    assert job["recovery_decision"]["can_continue"] is False
    assert continued.status_code == 409
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["recovery_decision"] is None


def test_active_job_must_be_cancelled_before_dismissal(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-active"
    job_dir.mkdir(parents=True)
    audio_web._write_job(
        job_dir / "job.json",
        {"id": "job-active", "filename": "active.wav", "status": "queued"},
    )

    response = TestClient(create_app(data)).delete("/api/jobs/job-active")

    assert response.status_code == 409


def test_job_progress_merges_manifest_tokens_and_costs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    job_dir = data / "audio-review" / "uploads" / "job-2"
    transcription = job_dir / "transcription" / "transcription"
    transcription.mkdir(parents=True)
    audio_web._write_job(
        job_dir / "job.json",
        {
            "id": "job-2",
            "filename": "meeting.m4a",
            "status": "running",
            "allow_cloud_upload": True,
            "cost_cap_cny": 3,
            "created_at": 2,
        },
    )
    (transcription / "processing_manifest.json").write_text(
        json.dumps(
            {
                "status": "running",
                "completed_steps": ["preprocess", "cloud", "local"],
                "cloud_billed_seconds": 600,
                "estimated_cost_cny": 0.16,
                "cost_cap_cny": 3,
                "text_review_input_tokens": 12_000,
                "text_review_output_tokens": 1_200,
                "text_review_cost_cny": 0.028,
                "text_review_cost_cap_cny": 0.3,
            }
        ),
        encoding="utf-8",
    )
    older = data / "audio-review" / "uploads" / "job-1"
    older.mkdir()
    audio_web._write_job(
        older / "job.json",
        {"id": "job-1", "filename": "old.wav", "status": "completed", "created_at": 1},
    )

    jobs = TestClient(create_app(data)).get("/api/jobs").json()

    assert [job["id"] for job in jobs] == ["job-2", "job-1"]
    current = jobs[0]
    assert current["stage"] == "Organizing speakers"
    assert current["progress_percent"] == 50
    assert current["completed_steps"] == ["preprocess", "cloud", "local"]
    assert current["cloud_billed_seconds"] == 600
    assert current["text_review_input_tokens"] == 12_000
    assert current["text_review_output_tokens"] == 1_200
    assert current["text_review_total_tokens"] == 13_200
    assert current["text_review_cost_cny"] == 0.028
    assert current["estimated_cost_cny"] == 0.16


def test_failed_job_prefers_manifest_error_over_http_log_tail(tmp_path: Path) -> None:
    data = tmp_path / "data"
    job_dir = data / "audio-review" / "uploads" / "job-error"
    transcription = job_dir / "transcription"
    transcription.mkdir(parents=True)
    audio_web._write_job(
        job_dir / "job.json",
        {
            "id": "job-error",
            "filename": "meeting.m4a",
            "status": "failed",
            "error": 'HTTP Request: POST /chat/completions "200 OK"',
        },
    )
    (transcription / "processing_manifest.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "errors": ["DeepSeek consolidation coverage was incomplete"],
            }
        ),
        encoding="utf-8",
    )

    job = TestClient(create_app(data)).get("/api/jobs/job-error").json()

    assert job["error"] == "DeepSeek consolidation coverage was incomplete"


def test_job_progress_reads_live_token_cache_before_manifest_finishes(tmp_path: Path) -> None:
    job_dir = tmp_path / "job-live"
    cache = job_dir / "transcription" / "providers" / "deepseek"
    cache.mkdir(parents=True)
    job_path = job_dir / "job.json"
    audio_web._write_job(
        job_path,
        {
            "id": "job-live",
            "filename": "live.m4a",
            "status": "running",
            "allow_cloud_upload": True,
            "text_review_enabled": True,
            "cost_cap_cny": 3,
        },
    )
    (cache / "window.json").write_text(
        json.dumps(
            {
                "prompt_cache_hit_tokens": 1_000,
                "prompt_cache_miss_tokens": 2_000,
                "output_tokens": 500,
            }
        ),
        encoding="utf-8",
    )

    job = audio_web._public_job(job_path)

    assert job["stage"] == "Preprocessing and denoising"
    assert job["progress_percent"] == 5
    assert job["text_review_input_tokens"] == 3_000
    assert job["text_review_output_tokens"] == 500
    assert job["text_review_total_tokens"] == 3_500
    assert job["text_review_cost_cny"] > 0


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, (0, 9, False)),
        ("bytes=2-", (2, 9, True)),
        ("bytes=-3", (7, 9, True)),
    ],
)
def test_parse_range(header: str | None, expected: tuple[int, int, bool]) -> None:
    assert _parse_range(header, 10) == expected


@pytest.mark.parametrize("header", ["items=1-2", "bytes=1-2,4-5", "bytes=-0", "bytes=20-"])
def test_parse_range_rejects_invalid_requests(header: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        _parse_range(header, 10)


def test_finds_bundled_npm_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPM_EXECUTABLE", "npm-test")
    assert audio_web._npm_command() == ["npm-test"]


def test_frontend_environment_exposes_bundled_node_to_npm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", r"C:\Windows\System32")
    node = Path(r"C:\runtime\node\bin\node.exe")

    environment = audio_web._frontend_environment([str(node), r"C:\runtime\npm-cli.js"])

    assert environment["PATH"].split(os.pathsep)[0] == str(node.parent)
    assert environment["PATH"].endswith(r"C:\Windows\System32")


def test_serve_audio_review_passes_bundled_node_path_to_frontend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    frontend = tmp_path / "frontend"
    (frontend / "node_modules").mkdir(parents=True)
    node = Path(r"C:\runtime\node\bin\node.exe")
    captured: dict[str, object] = {}

    class FrontendProcess:
        def terminate(self) -> None:
            captured["terminated"] = True

    def start_frontend(command: list[str], **kwargs: object) -> FrontendProcess:
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return FrontendProcess()

    monkeypatch.setenv("PATH", r"C:\Windows\System32")
    monkeypatch.setattr(audio_web, "_npm_command", lambda: [str(node), "npm-cli.js"])
    monkeypatch.setattr(audio_web.subprocess, "Popen", start_frontend)
    monkeypatch.setattr(uvicorn, "run", lambda *_args, **_kwargs: None)

    audio_web.serve_audio_review(
        data_root=tmp_path / "data",
        frontend_directory=frontend,
        open_browser=False,
    )

    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert str(environment["PATH"]).split(os.pathsep)[0] == str(node.parent)
    assert captured["command"] == [str(node), "npm-cli.js", "run", "dev"]
    assert captured["terminated"] is True


def test_catalog_includes_resumable_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    source_dir = data / "pending" / "input"
    transcription = source_dir / "transcription"
    transcription.mkdir(parents=True)
    audio = source_dir / "pending.m4a"
    audio.write_bytes(b"pending")
    manifest = {
        "status": "running",
        "source_hashes": {"recording_01": "b" * 64},
        "completed_steps": ["preprocess"],
        "estimated_cost_cny": 0.2,
        "cost_cap_cny": 3,
        "errors": [],
    }
    (transcription / "processing_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        audio_web,
        "inspect_recording",
        lambda *_args, **_kwargs: SimpleNamespace(
            sha256="b" * 64,
            duration_ms=12_000,
            recorded_at=datetime(2026, 8, 7, tzinfo=UTC),
        ),
    )

    catalog = RecordingCatalog(data)
    entry = catalog.entries()[0]
    assert entry.status == "running"
    assert catalog.document(entry) == {}
    assert catalog.topics(entry) == []
    client = TestClient(create_app(data))
    detail = client.get(f"/api/recordings/{entry.id}").json()
    assert detail["processing"]["completed_steps"] == ["preprocess"]
    assert detail["has_transcript"] is False


def test_background_job_records_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "recording.m4a"
    source.write_bytes(b"audio")
    job_path = tmp_path / "job.json"
    audio_web._write_job(job_path, {"status": "queued"})
    captured: list[list[str]] = []

    class FakeProcess:
        def __init__(self, command: list[str], returncode: int, stdout: str, stderr: str) -> None:
            captured.append(command)
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

        def communicate(self) -> tuple[str, str]:
            return self.stdout, self.stderr

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 1

    monkeypatch.setattr(
        audio_web.subprocess,
        "Popen",
        lambda command, **_kwargs: FakeProcess(command, 0, "ok", ""),
    )
    audio_web._run_transcription_job(job_path, source, allow_cloud=True)
    completed = audio_web._read_json(job_path)
    assert completed["status"] == "completed"
    assert "--allow-cloud-upload" in captured[0]
    assert "--speech-only-cloud" in captured[0]
    assert "--no-deepseek-text" in captured[0]

    audio_web._write_job(job_path, {"status": "queued"})

    monkeypatch.setattr(
        audio_web.subprocess,
        "Popen",
        lambda command, **_kwargs: FakeProcess(command, 2, "", "failed"),
    )
    audio_web._run_transcription_job(job_path, source, allow_cloud=False)
    failed = audio_web._read_json(job_path)
    assert failed["status"] == "failed"
    assert failed["error"] == "failed"
    assert failed["attempt_count"] == 3
    assert failed["retry_count"] == 2
    assert "--no-cloud" in captured[-1]
    assert "--speech-only-cloud" not in captured[-1]
    assert "--no-deepseek-text" in captured[-1]


def test_background_job_retries_from_checkpoint_then_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "recording.wav"
    source.write_bytes(b"audio")
    job_path = tmp_path / "job.json"
    audio_web._write_job(job_path, {"id": "job-retry", "status": "queued"})
    attempts = 0

    class RetryProcess:
        def __init__(self) -> None:
            nonlocal attempts
            attempts += 1
            self.returncode = 2 if attempts == 1 else 0

        def communicate(self) -> tuple[str, str]:
            return ("ok", "temporary network error")

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 1

    monkeypatch.setattr(audio_web.subprocess, "Popen", lambda *_args, **_kwargs: RetryProcess())

    audio_web._run_transcription_job(job_path, source, allow_cloud=False)

    completed = audio_web._read_json(job_path)
    assert completed["status"] == "completed"
    assert completed["attempt_count"] == 2
    assert completed["retry_count"] == 1
    assert attempts == 2


def test_diagnosed_resume_modes_build_the_promised_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []

    class SuccessProcess:
        returncode = 0

        def __init__(self, command: list[str]) -> None:
            captured.append(command)

        def communicate(self) -> tuple[str, str]:
            return "ok", ""

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            return None

    monkeypatch.setattr(
        audio_web.subprocess,
        "Popen",
        lambda command, **_kwargs: SuccessProcess(command),
    )
    source = tmp_path / "recording.wav"
    source.write_bytes(b"audio")
    job_path = tmp_path / "job.json"
    audio_web._write_job(
        job_path,
        {
            "status": "queued",
            "allow_cloud_upload": True,
            "run_cloud_enabled": True,
            "text_review_enabled": False,
        },
    )
    audio_web._run_transcription_job(job_path, source, allow_cloud=True)
    assert "--cloud" in captured[-1]
    assert "--no-deepseek-text" in captured[-1]

    audio_web._write_job(
        job_path,
        {
            "status": "queued",
            "allow_cloud_upload": True,
            "run_cloud_enabled": False,
            "text_review_enabled": False,
        },
    )
    audio_web._run_transcription_job(job_path, source, allow_cloud=True)
    assert "--no-cloud" in captured[-1]
    assert "--allow-cloud-upload" not in captured[-1]
    assert "--no-deepseek-text" in captured[-1]


def test_configuration_failure_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "recording.wav"
    source.write_bytes(b"audio")
    job_path = tmp_path / "job.json"
    audio_web._write_job(job_path, {"id": "job-config", "status": "queued"})
    attempts = 0

    class ConfigurationFailure:
        returncode = 2

        def __init__(self) -> None:
            nonlocal attempts
            attempts += 1

        def communicate(self) -> tuple[str, str]:
            return "", "DASHSCOPE_API_KEY is required for cloud mode"

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            return None

    monkeypatch.setattr(
        audio_web.subprocess,
        "Popen",
        lambda *_args, **_kwargs: ConfigurationFailure(),
    )

    audio_web._run_transcription_job(job_path, source, allow_cloud=True)

    failed = audio_web._read_json(job_path)
    assert failed["status"] == "failed"
    assert failed["attempt_count"] == 1
    assert attempts == 1


def test_failure_message_prefers_manifest_and_classifies_retryability(tmp_path: Path) -> None:
    job_path = tmp_path / "job.json"
    audio_web._write_job(job_path, {"status": "running"})
    manifest_path = tmp_path / "transcription" / "processing_manifest.json"
    manifest_path.parent.mkdir()
    audio_web._write_job(
        manifest_path,
        {"errors": ["", "DASHSCOPE_API_KEY is required for cloud mode"]},
    )

    message = audio_web._job_failure_message(job_path, "stdout tail", "HTTP 200")

    assert message == "DASHSCOPE_API_KEY is required for cloud mode"
    assert audio_web._is_retryable_failure(message) is False
    assert audio_web._is_retryable_failure("cloud model not found") is False
    assert audio_web._is_retryable_failure("DeepSeek invalid response") is False
    assert audio_web._is_retryable_failure("temporary network timeout") is True


def test_public_job_exposes_retry_and_restart_recovery_stage(tmp_path: Path) -> None:
    job_path = tmp_path / "job.json"
    audio_web._write_job(
        job_path,
        {
            "status": "running",
            "retrying": True,
            "attempt_count": 2,
            "max_attempts": 3,
        },
    )
    assert audio_web._public_job(job_path)["stage"] == "Automatic checkpoint retry 2/3"

    audio_web._write_job(job_path, {"status": "queued", "recovery_count": 1})
    assert audio_web._public_job(job_path)["stage"] == "Waiting to resume from checkpoint"


def test_heavy_transcription_jobs_are_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active = 0
    maximum_active = 0
    counter_lock = threading.Lock()

    class SlowProcess:
        returncode = 0

        def __init__(self) -> None:
            nonlocal active, maximum_active
            with counter_lock:
                active += 1
                maximum_active = max(maximum_active, active)

        def communicate(self) -> tuple[str, str]:
            nonlocal active
            time.sleep(0.03)
            with counter_lock:
                active -= 1
            return "ok", ""

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            return None

    monkeypatch.setattr(audio_web.subprocess, "Popen", lambda *_args, **_kwargs: SlowProcess())
    threads = []
    for index in range(2):
        directory = tmp_path / f"job-{index}"
        directory.mkdir()
        source = directory / "recording.wav"
        source.write_bytes(b"audio")
        job_path = directory / "job.json"
        audio_web._write_job(job_path, {"id": f"job-{index}", "status": "queued"})
        thread = threading.Thread(
            target=audio_web._run_transcription_job,
            args=(job_path, source, False),
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert maximum_active == 1


def test_startup_recovers_interrupted_job_from_its_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, _ = _write_catalog(tmp_path)
    job_dir = data / "audio-review" / "uploads" / "job-orphan"
    job_dir.mkdir(parents=True)
    source = job_dir / "recording.wav"
    source.write_bytes(b"audio")
    job_path = job_dir / "job.json"
    audio_web._write_job(
        job_path,
        {
            "id": "job-orphan",
            "status": "running",
            "allow_cloud_upload": False,
            "attempt_count": 1,
        },
    )
    started: list[tuple[Path, Path, bool]] = []
    monkeypatch.setattr(
        audio_web,
        "_start_transcription_thread",
        lambda job, audio, cloud: started.append((job, audio, cloud)) or True,
    )

    with TestClient(create_app(data)) as client:
        assert client.get("/api/health").status_code == 200

    recovered = audio_web._read_json(job_path)
    assert recovered["status"] == "queued"
    assert recovered["attempt_count"] == 0
    assert recovered["recovery_count"] == 1
    assert started == [(job_path, source, False)]


def test_background_job_finishes_as_cancelled_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "recording.wav"
    source.write_bytes(b"audio")
    job_path = tmp_path / "job.json"
    audio_web._write_job(job_path, {"id": "job-cancel", "status": "queued"})

    class CancelledProcess:
        returncode = 1

        def communicate(self) -> tuple[str, str]:
            job = audio_web._read_json(job_path)
            job["cancel_requested"] = True
            audio_web._write_job(job_path, job)
            return "", "terminated"

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            return None

    monkeypatch.setattr(audio_web.subprocess, "Popen", lambda *_args, **_kwargs: CancelledProcess())

    audio_web._run_transcription_job(job_path, source, allow_cloud=False)

    cancelled = audio_web._read_json(job_path)
    assert cancelled["status"] == "cancelled"
    assert cancelled["error"] is None


def test_json_and_job_validation(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root is not an object"):
        audio_web._read_json(invalid)
    data, _ = _write_catalog(tmp_path)
    client = TestClient(create_app(data))
    assert client.get("/api/jobs/job-missing").status_code == 404


def test_catalog_prefers_non_destructive_reading_view(tmp_path: Path) -> None:
    data, _ = _write_catalog(tmp_path)
    transcription = data / "sample" / "transcription"
    original = json.loads((transcription / "transcript.json").read_text(encoding="utf-8"))
    reading = json.loads(json.dumps(original))
    reading["segments"][0]["text"] = "Edited display text."
    (transcription / "reading_view.json").write_text(
        json.dumps(reading, ensure_ascii=False), encoding="utf-8"
    )

    catalog = RecordingCatalog(data)
    assert catalog.segments(catalog.entries()[0])[0]["text"] == "Edited display text."
    assert (
        json.loads((transcription / "transcript.json").read_text(encoding="utf-8"))["segments"][0][
            "text"
        ]
        == "First sentence."
    )
