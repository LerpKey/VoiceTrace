"""Mocked runtime tests for the resumable audio providers and orchestrator."""


from __future__ import annotations

import json
import sys
import types
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import requests
from typer.testing import CliRunner

from research_kb.audio import pipeline, preprocess, providers
from research_kb.audio.domain import (
    AudioChunk,
    RecordingSource,
    SpeakerProfile,
    TranscriptDocument,
)
from research_kb.audio.providers import CloudResult, ProviderSentence
from research_kb.audio.speaker import (
    LocalSpeakerSamples,
    SpeakerEmbedder,
    resolve_local_segments,
)
from research_kb.cli import app


def _source(tmp_path: Path, *, duration_ms: int = 120_000) -> RecordingSource:
    path = tmp_path / "source.m4a"
    path.write_bytes(b"source")
    return RecordingSource(
        recording_id="recording_01",
        path=str(path),
        filename=path.name,
        sha256="a" * 64,
        size_bytes=6,
        duration_ms=duration_ms,
        recorded_at=datetime(2026, 8, 3, 8, 33, tzinfo=UTC),
    )


def _wav(path: Path, *, seconds: int = 1) -> Path:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(bytes(16_000 * 2 * seconds))
    return path


class _Response:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> object:
        return self.payload


class _Session:
    def __init__(self, *, gets: list[_Response], posts: list[_Response]) -> None:
        self.gets = gets
        self.posts = posts
        self.get_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.post_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def get(self, *args: object, **kwargs: object) -> _Response:
        self.get_calls.append((args, kwargs))
        return self.gets.pop(0)

    def post(self, *args: object, **kwargs: object) -> _Response:
        self.post_calls.append((args, kwargs))
        return self.posts.pop(0)


def test_preprocess_parses_silence_and_builds_both_filter_chains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    chunk = AudioChunk(
        chunk_id="chunk_1", recording_id=source.recording_id, start_ms=1_000, end_ms=3_000
    )
    calls: list[list[str]] = []

    def fake_run(arguments: list[str], *, timeout_seconds: float | None = None) -> str:
        del timeout_seconds
        calls.append(arguments)
        return "silence_start: 1.0\nsilence_end: 3.0\n"

    monkeypatch.setattr(preprocess, "_run_ffmpeg", fake_run)
    monkeypatch.setattr(preprocess, "sha256_file", lambda _path: "b" * 64)

    assert preprocess.detect_silence_midpoints(source) == (2_000,)
    assert preprocess.encode_chunk(source, chunk, tmp_path / "enhanced.flac", enhanced=True) == (
        "b" * 64
    )
    preprocess.encode_chunk(source, chunk, tmp_path / "light.flac", enhanced=False)
    preprocess.extract_clip(
        tmp_path / "light.flac", tmp_path / "clip.wav", start_ms=100, end_ms=500
    )
    preprocess.extract_clip(
        tmp_path / "light.flac",
        tmp_path / "review.mp3",
        start_ms=100,
        end_ms=500,
        wav=False,
    )

    rendered = " ".join(" ".join(call) for call in calls)
    assert "afftdn=nr=8" in rendered
    assert "loudnorm=I=-20" in rendered
    assert "dynaudnorm" in rendered
    assert "libmp3lame" in rendered
    with pytest.raises(ValueError, match="after start"):
        preprocess.extract_clip(tmp_path / "x", tmp_path / "y", start_ms=1, end_ms=1)


def test_ffmpeg_resolution_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.SimpleNamespace(
        get_ffmpeg_exe=lambda: "ffmpeg-test", get_ffmpeg_version=lambda: "test"
    )
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", module)
    monkeypatch.setattr(
        preprocess.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=0, stderr="ok"),
    )
    assert preprocess.ffmpeg_executable() == "ffmpeg-test"
    assert preprocess._run_ffmpeg(["-version"]) == "ok"
    monkeypatch.setattr(
        preprocess.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=1, stderr="bad"),
    )
    with pytest.raises(preprocess.AudioPreprocessError, match="FFmpeg failed"):
        preprocess._run_ffmpeg(["-bad"])


def test_file_provider_upload_poll_parse_and_sanitize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _wav(tmp_path / "sample.wav")
    policy = {
        "data": {
            "max_file_size_mb": 10,
            "upload_dir": "tmp/id",
            "oss_access_key_id": "access",
            "policy": "policy",
            "signature": "signature",
            "x_oss_object_acl": "private",
            "x_oss_forbid_overwrite": "true",
            "upload_host": "https://upload.example",
        }
    }
    task = {
        "output": {
            "task_status": "SUCCEEDED",
            "results": [
                {
                    "subtask_status": "SUCCEEDED",
                    "transcription_url": "https://result.example/file?Signature=secret",
                }
            ],
        },
        "usage": {"duration": 0},
    }
    result_payload = {
        "transcripts": [
            {
                "sentences": [
                    {
                        "begin_time": 10,
                        "end_time": 900,
                        "text": " 你好 ",
                        "speaker_id": 2,
                    }
                ]
            }
        ],
        "file_url": "oss://secret",
    }
    session = _Session(
        gets=[_Response(policy), _Response(task), _Response(result_payload)],
        posts=[_Response({}), _Response({"output": {"task_id": "task-1"}})],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)
    client = providers.DashScopeFileTranscriber(api_key="key", poll_seconds=0)

    result = client.transcribe(audio, model="fun-asr", diarization=True)

    assert result.task_id == "task-1"
    assert result.billed_seconds == 1
    assert result.sentences[0].text == "你好"
    assert result.sentences[0].speaker_id == "2"
    assert "file_url" not in result.sanitized_payload


def test_qwen3_file_provider_uses_current_request_and_result_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _wav(tmp_path / "sample.wav")
    session = _Session(
        gets=[
            _Response(
                {
                    "output": {
                        "task_status": "SUCCEEDED",
                        "result": {"transcription_url": "https://result.example/current"},
                    },
                    "usage": {"seconds": 1},
                }
            ),
            _Response(
                {
                    "transcripts": [
                        {"sentences": [{"begin_time": 100, "end_time": 100, "text": "嗯"}]}
                    ]
                }
            ),
        ],
        posts=[_Response({"output": {"task_id": "task-qwen3"}})],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)
    client = providers.DashScopeFileTranscriber(api_key="key", poll_seconds=0)
    monkeypatch.setattr(client, "upload_temporary", lambda _path, **_kwargs: "oss://private/audio")

    result = client.transcribe(
        audio,
        model="qwen3-asr-flash-filetrans",
        diarization=True,
        context_text="不应提交给当前接口",
    )

    submitted = session.post_calls[0][1]["json"]
    assert isinstance(submitted, dict)
    assert submitted["input"] == {"file_url": "oss://private/audio"}
    assert submitted["parameters"] == {
        "channel_id": [0],
        "enable_itn": False,
        "enable_words": True,
        "language": "zh",
    }
    assert result.sentences[0].end_ms == 101
    assert result.sentences[0].speaker_id is None


def test_file_provider_treats_no_words_task_as_empty_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _wav(tmp_path / "silent.wav")
    session = _Session(
        gets=[
            _Response(
                {
                    "output": {
                        "task_status": "FAILED",
                        "code": "ASR_RESPONSE_HAVE_NO_WORDS",
                        "message": "ASR_RESPONSE_HAVE_NO_WORDS",
                        "results": [{"file_url": "https://secret.example/?Signature=x"}],
                    }
                }
            )
        ],
        posts=[_Response({"output": {"task_id": "task-empty"}})],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)
    client = providers.DashScopeFileTranscriber(api_key="key", poll_seconds=0)
    monkeypatch.setattr(client, "upload_temporary", lambda _path, **_kwargs: "oss://audio")

    result = client.transcribe(audio, model="fun-asr", diarization=True)

    assert result.sentences == ()
    assert result.billed_seconds == 1
    assert "file_url" not in json.dumps(result.sanitized_payload)


def test_flash_provider_success_and_no_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _wav(tmp_path / "clip.wav")
    success = _Session(
        gets=[],
        posts=[
            _Response(
                {
                    "request_id": "r1",
                    "output": {"output": {"sentence": {"text": "复核文本"}}},
                }
            )
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: success)
    client = providers.DashScopeFlashTranscriber(api_key="key")
    assert client.transcribe(audio, model=pipeline.THIRD_PASS_MODEL).text == "复核文本"

    no_words = _Session(
        gets=[],
        posts=[
            _Response(
                {"code": "CLIENT_ERROR", "message": "ASR_RESPONSE_HAVE_NO_WORDS"},
                status_code=400,
            )
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: no_words)
    assert (
        providers.DashScopeFlashTranscriber(api_key="key")
        .transcribe(audio, model=pipeline.THIRD_PASS_MODEL)
        .text
        == ""
    )


def test_qwen3_flash_provider_uses_audio_content_and_asr_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _wav(tmp_path / "clip.wav")
    session = _Session(
        gets=[],
        posts=[
            _Response(
                {
                    "request_id": "r-qwen3",
                    "output": {"choices": [{"message": {"content": "当前接口文本"}}]},
                }
            )
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    result = providers.DashScopeFlashTranscriber(api_key="key").transcribe(
        audio, model="qwen3-asr-flash", context_text="不会作为文本消息发送"
    )

    submitted = session.post_calls[0][1]["json"]
    assert isinstance(submitted, dict)
    messages = submitted["input"]["messages"]
    assert len(messages) == 1
    assert set(messages[0]["content"][0]) == {"audio"}
    assert submitted["parameters"] == {"asr_options": {"language": "zh", "enable_itn": False}}
    assert result.text == "当前接口文本"


def test_local_provider_batches_and_unloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeModel:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> FakeModel:
            return cls()

        def transcribe(self, **_kwargs: object) -> list[types.SimpleNamespace]:
            return [types.SimpleNamespace(text=" 本地文本 ")]

    monkeypatch.setitem(sys.modules, "qwen_asr", types.SimpleNamespace(Qwen3ASRModel=FakeModel))
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    client = providers.LocalQwenTranscriber(model_directory=tmp_path, batch_size=1)
    result = client.transcribe([(tmp_path / "clip.wav", 10, 20)])
    assert result[0].text == "本地文本"
    client.unload()


def _options(tmp_path: Path, *, cloud: bool = True) -> pipeline.TranscriptionOptions:
    return pipeline.TranscriptionOptions(
        source_directory=tmp_path,
        output_path=tmp_path / "transcript.md",
        run_local=True,
        run_cloud=cloud,
        allow_cloud_upload=cloud,
        cost_cap_cny=30,
        chunk_minutes=1,
        overlap_seconds=1,
    )


def test_local_only_uses_separate_state_and_shared_derivatives(tmp_path: Path) -> None:
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path, cloud=False))
    assert transcriber.work_directory == tmp_path / "transcription-local-only"
    assert transcriber.derivatives_directory == tmp_path / "transcription"
    assert transcriber.manifest_path.parent == transcriber.work_directory


def test_single_file_input_and_deepseek_configuration_validation(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "rec-20260805-1320.wav")
    output = tmp_path / "results" / "transcript.md"
    file_options = pipeline.TranscriptionOptions(
        source_directory=audio,
        output_path=output,
        run_local=True,
        run_cloud=False,
    )
    transcriber = pipeline.RecordingTranscriber(file_options)

    assert transcriber.source_directory == audio.parent
    assert transcriber.derivatives_directory == output.parent / "transcription"
    assert transcriber.work_directory == output.parent / "transcription-local-only"
    assert transcriber.models_directory == Path.cwd().resolve() / "data" / "models" / "audio"
    sources = transcriber._sources()
    assert len(sources) == 1
    assert sources[0].filename == audio.name
    assert sources[0].duration_ms == 1_000

    deepseek_options = pipeline.TranscriptionOptions(
        source_directory=audio,
        output_path=output,
        run_local=True,
        run_cloud=False,
        run_deepseek_text=True,
    )
    with pytest.raises(pipeline.RecordingTranscriptionError, match="DEEPSEEK_API_KEY"):
        pipeline.RecordingTranscriber(deepseek_options)

    invalid_cap = pipeline.TranscriptionOptions(
        source_directory=audio,
        output_path=output,
        run_local=True,
        run_cloud=False,
        deepseek_cost_cap_cny=0,
    )
    with pytest.raises(pipeline.RecordingTranscriptionError, match="DeepSeek cost cap"):
        pipeline.RecordingTranscriber(invalid_cap)


def test_local_only_adjudication_keeps_plain_text_and_local_provenance(tmp_path: Path) -> None:
    source = _source(tmp_path, duration_ms=20_000)
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path, cloud=False))
    local = [
        {
            "recording_id": source.recording_id,
            "start_ms": 1_000,
            "end_ms": 6_000,
            "text": "本地原话",
            "model": pipeline.LOCAL_MODEL_ID,
        }
    ]
    key = f"{source.recording_id}|1000|6000"
    segments = transcriber._adjudicate_local_only((source,), local, {key: "说话人 A"})
    assert segments[0].text == "本地原话"
    assert segments[0].speaker == "说话人 A"
    assert segments[0].flags == ("local_only",)
    assert {candidate.provider for candidate in segments[0].candidates} == {"local"}


def test_sparse_provider_speaker_group_inherits_local_voice_cluster(tmp_path: Path) -> None:
    source = _source(tmp_path, duration_ms=20_000)
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path), api_key="key")
    cloud = [
        {
            "recording_id": source.recording_id,
            "chunk_id": "speech_1",
            "start_ms": 1_000,
            "end_ms": 2_000,
            "text": "短句",
            "speaker_id": "0",
            "task_id": "task",
            "model": "fun-asr",
        },
        {
            "recording_id": source.recording_id,
            "chunk_id": "speech_1",
            "start_ms": 3_000,
            "end_ms": 8_000,
            "text": "足够长的句子",
            "speaker_id": "0",
            "task_id": "task",
            "model": "fun-asr",
        },
    ]
    local = [
        {
            "recording_id": source.recording_id,
            "start_ms": 1_000,
            "end_ms": 2_000,
            "text": "短句",
            "cloud_ref": "speech_1:1000:2000",
        },
        {
            "recording_id": source.recording_id,
            "start_ms": 3_000,
            "end_ms": 8_000,
            "text": "足够长的句子",
            "cloud_ref": "speech_1:3000:8000",
        },
    ]
    mapping = {f"{source.recording_id}|3000|8000": "说话人 A"}

    segments = transcriber._adjudicate((source,), cloud, local, {}, mapping)

    assert [segment.speaker for segment in segments] == ["说话人 A", "说话人 A"]


def test_local_segment_clustering_requires_cross_day_evidence(tmp_path: Path) -> None:
    vectors = {
        "a1": np.array([1.0, 0.0], dtype=np.float32),
        "a2": np.array([0.99, 0.01], dtype=np.float32),
        "a3": np.array([1.0, 0.0], dtype=np.float32),
        "a4": np.array([0.99, -0.01], dtype=np.float32),
        "b1": np.array([0.0, 1.0], dtype=np.float32),
        "b2": np.array([0.0, 1.0], dtype=np.float32),
    }

    class FakeEmbedder:
        def embed(self, clips: tuple[Path, ...]) -> np.ndarray:
            return vectors[clips[0].stem]

    samples = tuple(
        LocalSpeakerSamples(
            key=name,
            recording_id=recording,
            first_order=(day, index),
            clips=(tmp_path / f"{name}.wav",),
        )
        for index, (name, recording, day) in enumerate(
            (
                ("a1", "day1", 0),
                ("a2", "day1", 0),
                ("b1", "day1", 0),
                ("a3", "day2", 1),
                ("a4", "day2", 1),
                ("b2", "day2", 1),
            )
        )
    )
    embedding_cache = tmp_path / "embeddings.json"
    mapping, profiles = resolve_local_segments(  # type: ignore[arg-type]
        samples,
        embedder=FakeEmbedder(),
        embedding_cache_path=embedding_cache,
        minimum_identity_samples=2,
    )
    assert mapping["a1"] == mapping["a3"]
    assert "b1" not in mapping
    assert "b2" not in mapping
    assert any(len(profile.recording_ids) == 2 for profile in profiles)
    assert embedding_cache.is_file()

    class CacheOnlyEmbedder:
        def embed(self, _clips: tuple[Path, ...]) -> np.ndarray:
            raise AssertionError("cached embeddings should be reused")

    cached_mapping, _ = resolve_local_segments(  # type: ignore[arg-type]
        samples,
        embedder=CacheOnlyEmbedder(),
        embedding_cache_path=embedding_cache,
        minimum_identity_samples=2,
    )
    assert cached_mapping == mapping


def test_local_only_speaker_mapping_writes_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, duration_ms=20_000)
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path, cloud=False))
    clip = (
        transcriber.work_directory
        / "local-clips"
        / source.recording_id
        / "0000001000_0000006000.vad.wav"
    )
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"clip")
    sentences = [
        {
            "recording_id": source.recording_id,
            "start_ms": 1_000,
            "end_ms": 6_000,
            "text": "本地文本",
        }
    ]
    profile = SpeakerProfile(
        speaker="说话人 A",
        local_speaker_keys=(f"{source.recording_id}|1000|6000",),
        recording_ids=(source.recording_id,),
        sample_count=5,
        confidence=0.75,
    )
    monkeypatch.setattr(pipeline, "_ensure_model", lambda _model, directory: directory)

    def fake_resolve(
        samples: tuple[LocalSpeakerSamples, ...], **kwargs: object
    ) -> tuple[dict[str, str], tuple[SpeakerProfile, ...]]:
        assert kwargs["embedding_cache_path"] == (
            transcriber.work_directory / "providers" / "local" / "speaker-embeddings.json"
        )
        return {samples[0].key: "说话人 A"}, (profile,)

    monkeypatch.setattr(pipeline, "resolve_local_segments", fake_resolve)
    mapping, profiles = transcriber._local_only_speaker_mapping((source,), sentences)
    assert mapping == {f"{source.recording_id}|1000|6000": "说话人 A"}
    assert profiles == (profile,)

    monkeypatch.setattr(
        pipeline,
        "resolve_local_segments",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cache missed")),
    )
    assert transcriber._local_only_speaker_mapping((source,), sentences) == (
        mapping,
        profiles,
    )


def test_local_speaker_edges_keep_single_identity_and_reject_short_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class SingleEmbedder:
        def embed(self, _clips: tuple[Path, ...]) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    sample = LocalSpeakerSamples(
        key="single",
        recording_id="day1",
        first_order=(0, 0),
        clips=(tmp_path / "single.wav",),
    )
    mapping, profiles = resolve_local_segments(  # type: ignore[arg-type]
        (sample,), embedder=SingleEmbedder(), minimum_identity_samples=1
    )
    assert mapping == {"single": "说话人 A"}
    assert profiles[0].sample_count == 1

    source = _source(tmp_path, duration_ms=20_000)
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path, cloud=False))
    monkeypatch.setattr(pipeline, "_ensure_model", lambda _model, directory: directory)
    assert transcriber._local_only_speaker_mapping(
        (source,),
        [
            {
                "recording_id": source.recording_id,
                "start_ms": 1_000,
                "end_ms": 2_000,
                "text": "太短",
            }
        ],
    ) == ({}, ())


def test_local_speaker_mapping_uses_vad_event_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, duration_ms=30_000)
    chunk = AudioChunk(
        chunk_id="chunk",
        recording_id=source.recording_id,
        start_ms=0,
        end_ms=30_000,
        light_path=source.path,
        enhanced_path=source.path,
        light_sha256="a" * 64,
        enhanced_sha256="a" * 64,
    )
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path, cloud=False))
    monkeypatch.setattr(
        transcriber,
        "_vad_intervals",
        lambda _chunks: {source.recording_id: [(1_000, 20_000)]},
    )
    monkeypatch.setattr(pipeline, "_ensure_model", lambda _model, directory: directory)
    monkeypatch.setattr(
        pipeline,
        "extract_clip",
        lambda _source, target, **_kwargs: (
            target.parent.mkdir(parents=True, exist_ok=True) or target.write_bytes(b"clip")
        ),
    )
    captured: list[LocalSpeakerSamples] = []

    def fake_resolve(
        samples: tuple[LocalSpeakerSamples, ...], **_kwargs: object
    ) -> tuple[dict[str, str], tuple[SpeakerProfile, ...]]:
        captured.extend(samples)
        return {}, ()

    monkeypatch.setattr(pipeline, "resolve_local_segments", fake_resolve)
    mapping, profiles = transcriber._local_only_speaker_mapping(
        (source,),
        [{"recording_id": source.recording_id, "start_ms": 1_000, "end_ms": 20_000}],
        (chunk,),
    )

    assert mapping == {}
    assert profiles == ()
    assert len(captured) == 1
    assert captured[0].key == f"{source.recording_id}|1000|20000"
    assert len(captured[0].clips) == 5
    assert "0000001000_0000007000" in captured[0].clips[0].name


def test_pipeline_prepare_chunks_and_resume_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path), api_key="key")
    monkeypatch.setattr(pipeline, "detect_silence_midpoints", lambda _source: (59_000,))

    def fake_encode(
        _source: RecordingSource, _chunk: AudioChunk, path: Path, *, enhanced: bool
    ) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("enhanced" if enhanced else "light", encoding="utf-8")
        return ("b" if enhanced else "c") * 64

    monkeypatch.setattr(pipeline, "encode_chunk", fake_encode)
    chunks = transcriber._prepare_chunks((source,))
    assert len(chunks) == 3
    assert chunks[0].end_ms == 59_000
    assert chunks[0].enhanced_sha256 == "b" * 64


def test_sparse_cloud_regions_merge_vad_and_keep_absolute_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    options = _options(tmp_path)
    options = pipeline.TranscriptionOptions(
        **{**options.__dict__, "speech_only_cloud": True, "speech_region_gap_seconds": 30}
    )
    transcriber = pipeline.RecordingTranscriber(options, api_key="key")
    light = tmp_path / "timeline.light.flac"
    light.write_bytes(b"light")
    chunk = AudioChunk(
        chunk_id="recording_01_0001",
        recording_id=source.recording_id,
        start_ms=0,
        end_ms=source.duration_ms,
        light_path=str(light),
        light_sha256="a" * 64,
    )
    monkeypatch.setattr(
        transcriber,
        "_vad_intervals",
        lambda _chunks: {
            source.recording_id: [(10_000, 20_000), (40_000, 50_000), (100_000, 110_000)]
        },
    )

    def fake_encode(
        _source: RecordingSource, _chunk: AudioChunk, path: Path, *, enhanced: bool
    ) -> str:
        assert enhanced
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"speech")
        return "d" * 64

    monkeypatch.setattr(pipeline, "encode_chunk", fake_encode)

    regions = transcriber._speech_cloud_chunks((source,), (chunk,))

    assert [(region.start_ms, region.end_ms) for region in regions] == [
        (8_800, 51_200),
        (98_800, 111_200),
    ]
    assert regions[0].light_path == regions[0].enhanced_path
    assert regions[0].light_sha256 == "d" * 64
    payload = json.loads(
        (transcriber.work_directory / "providers" / "cloud" / "speech-regions.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["merge_gap_seconds"] == 30


def test_cloud_pass_falls_back_and_deduplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    enhanced = tmp_path / "enhanced.flac"
    enhanced.write_bytes(b"audio")
    chunk = AudioChunk(
        chunk_id="recording_01_chunk_001",
        recording_id="recording_01",
        start_ms=0,
        end_ms=60_000,
        enhanced_path=str(enhanced),
        light_path=str(enhanced),
        enhanced_sha256="b" * 64,
        light_sha256="b" * 64,
    )

    class FakeFile:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def transcribe(self, _path: Path, *, model: str, **_kwargs: object) -> CloudResult:
            if model == pipeline.PRIMARY_CLOUD_MODEL:
                raise providers.TranscriptionProviderError("missing")
            return CloudResult(
                model=model,
                task_id="task",
                billed_seconds=60,
                sentences=(ProviderSentence(1_000, 2_000, "云端", "0", "task"),),
                sanitized_payload={},
            )

    monkeypatch.setattr(pipeline, "DashScopeFileTranscriber", FakeFile)
    monkeypatch.setattr(
        pipeline,
        "extract_clip",
        lambda _src, dst, **_kwargs: (
            dst.parent.mkdir(parents=True, exist_ok=True) or dst.write_bytes(b"pilot")
        ),
    )
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path), api_key="key")
    model, results, sentences = transcriber._cloud_pass((chunk,))
    assert model == pipeline.FALLBACK_CLOUD_MODEL
    assert set(results) == {"pilot", chunk.chunk_id}
    assert sentences[0]["start_ms"] == 1_000


def test_local_speaker_third_pass_and_adjudication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, duration_ms=60_000)
    light = _wav(tmp_path / "light.wav", seconds=1)
    chunk = AudioChunk(
        chunk_id="recording_01_chunk_001",
        recording_id=source.recording_id,
        start_ms=0,
        end_ms=60_000,
        enhanced_path=str(light),
        light_path=str(light),
        enhanced_sha256="b" * 64,
        light_sha256="c" * 64,
    )
    cloud = [
        {
            "recording_id": source.recording_id,
            "chunk_id": chunk.chunk_id,
            "start_ms": 1_000,
            "end_ms": 5_000,
            "text": "预算一百二十元",
            "speaker_id": "0",
            "task_id": "task",
            "model": "fun-asr",
        }
    ]
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path), api_key="key")
    monkeypatch.setattr(transcriber, "_vad_intervals", lambda _chunks: {source.recording_id: []})
    monkeypatch.setattr(pipeline, "_ensure_model", lambda _model, directory: directory)
    monkeypatch.setattr(
        pipeline,
        "extract_clip",
        lambda _src, dst, **_kwargs: dst.parent.mkdir(parents=True, exist_ok=True) or _wav(dst),
    )

    class FakeLocal:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def transcribe(self, clips: list[tuple[Path, int, int]]) -> tuple[ProviderSentence, ...]:
            return tuple(ProviderSentence(start, end, "预算一百三十元") for _, start, end in clips)

        def unload(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "LocalQwenTranscriber", FakeLocal)
    local = transcriber._local_pass((source,), (chunk,), cloud)
    assert local[0]["cloud_ref"] is not None

    class FakeFlash:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def transcribe(self, *_args: object, **_kwargs: object) -> providers.FlashResult:
            return providers.FlashResult(
                model=pipeline.THIRD_PASS_MODEL,
                request_id="third",
                text="预算一百三十元",
                sanitized_payload={},
            )

    monkeypatch.setattr(pipeline, "DashScopeFlashTranscriber", FakeFlash)
    cloud_results: dict[str, CloudResult] = {}
    third = transcriber._third_pass((source,), (chunk,), cloud, local, cloud_results)
    segments = transcriber._adjudicate(
        (source,), cloud, local, third, {f"{source.recording_id}|{chunk.chunk_id}|0": "说话人 A"}
    )
    assert segments[0].text == "预算一百三十元"
    assert segments[0].decision == "third_pass_majority"
    assert segments[0].speaker == "说话人 A"


def test_run_writes_structured_and_markdown_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, duration_ms=10_000)
    chunk = AudioChunk(
        chunk_id="chunk", recording_id=source.recording_id, start_ms=0, end_ms=10_000
    )
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path), api_key="key")
    monkeypatch.setattr(transcriber, "_sources", lambda: (source,))
    monkeypatch.setattr(transcriber, "_prepare_chunks", lambda _sources: (chunk,))
    monkeypatch.setattr(transcriber, "_cloud_pass", lambda _chunks: ("fun-asr", {}, []))
    monkeypatch.setattr(transcriber, "_local_pass", lambda *_args: [])
    monkeypatch.setattr(transcriber, "_third_pass", lambda *_args: {})
    monkeypatch.setattr(transcriber, "_adjudicate", lambda *_args: ())
    monkeypatch.setattr(transcriber, "_write_manifest", lambda **_kwargs: None)

    document = transcriber.run()

    assert document.sources == (source,)
    assert transcriber.transcript_path.is_file()
    assert transcriber.output_path.is_file()
    assert (
        json.loads(transcriber.transcript_path.read_text(encoding="utf-8"))["schema_version"]
        == "1.0"
    )


def test_model_download_resume_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = tmp_path / "ready"
    ready.mkdir()
    (ready / "model.pt").write_bytes(b"model")
    assert pipeline._ensure_model("model", ready) == ready

    downloaded = tmp_path / "downloaded"

    def model_scope(_model: str, *, local_dir: str) -> None:
        Path(local_dir, "weights.bin").write_bytes(b"weights")

    monkeypatch.setitem(
        sys.modules, "modelscope", types.SimpleNamespace(snapshot_download=model_scope)
    )
    assert pipeline._ensure_model("model", downloaded) == downloaded
    assert (downloaded / ".download_complete").is_file()

    fallback = tmp_path / "fallback"

    def fail_scope(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("scope failed")

    def hub_download(*, repo_id: str, local_dir: Path) -> None:
        assert repo_id == "model"
        Path(local_dir, "weights.bin").write_bytes(b"weights")

    monkeypatch.setitem(
        sys.modules, "modelscope", types.SimpleNamespace(snapshot_download=fail_scope)
    )
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=hub_download)
    )
    assert pipeline._ensure_model("model", fallback) == fallback


def test_transcriber_validation_sources_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(pipeline.RecordingTranscriptionError, match="allow-cloud-upload"):
        pipeline.RecordingTranscriber(
            pipeline.TranscriptionOptions(tmp_path, tmp_path / "out.md"), api_key="key"
        )
    with pytest.raises(pipeline.RecordingTranscriptionError, match="API_KEY"):
        pipeline.RecordingTranscriber(
            pipeline.TranscriptionOptions(tmp_path, tmp_path / "out.md", allow_cloud_upload=True)
        )
    with pytest.raises(pipeline.RecordingTranscriptionError, match="at least one"):
        pipeline.RecordingTranscriber(
            pipeline.TranscriptionOptions(
                tmp_path, tmp_path / "out.md", run_local=False, run_cloud=False
            )
        )
    with pytest.raises(pipeline.RecordingTranscriptionError, match="requires cloud ASR"):
        pipeline.RecordingTranscriber(
            pipeline.TranscriptionOptions(
                tmp_path,
                tmp_path / "out.md",
                run_cloud=False,
                speech_only_cloud=True,
                run_local=True,
            )
        )
    with pytest.raises(pipeline.RecordingTranscriptionError, match="must not be negative"):
        pipeline.RecordingTranscriber(
            pipeline.TranscriptionOptions(
                tmp_path,
                tmp_path / "out.md",
                allow_cloud_upload=True,
                speech_region_gap_seconds=-1,
            ),
            api_key="key",
        )
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path), api_key="key")
    with pytest.raises(pipeline.RecordingTranscriptionError, match="no supported audio"):
        transcriber._sources()

    source = _source(tmp_path)
    monkeypatch.setattr(pipeline, "inspect_recording", lambda _path, *, index: source)
    assert transcriber._sources() == (source,)
    chunk = AudioChunk(chunk_id="chunk", recording_id=source.recording_id, start_ms=0, end_ms=1_000)
    transcriber._write_manifest(
        sources=(source,),
        chunks=(chunk,),
        status="running",
        cloud_model="fun-asr",
        cloud_results={},
        completed_steps=("preprocess",),
    )
    assert json.loads(transcriber.manifest_path.read_text(encoding="utf-8"))["status"] == (
        "running"
    )
    transcriber._validate_resume_sources((source,))
    changed = source.model_copy(update={"sha256": "d" * 64})
    with pytest.raises(pipeline.RecordingTranscriptionError, match="changed"):
        transcriber._validate_resume_sources((changed,))


def test_manifest_accumulates_deepseek_cost_across_cached_retries(tmp_path: Path) -> None:
    source = _source(tmp_path)
    options = pipeline.TranscriptionOptions(
        source_directory=tmp_path,
        output_path=tmp_path / "out.md",
        run_local=True,
        run_cloud=False,
        run_deepseek_text=True,
    )
    transcriber = pipeline.RecordingTranscriber(options, deepseek_api_key="key")
    cache_directory = transcriber.work_directory / "providers" / "deepseek"
    cache_directory.mkdir(parents=True)
    for index in range(2):
        (cache_directory / f"window-{index}.json").write_text(
            json.dumps(
                {
                    "model": transcriber.deepseek_model,
                    "prompt_cache_hit_tokens": 10,
                    "prompt_cache_miss_tokens": 20,
                    "output_tokens": 30,
                }
            ),
            encoding="utf-8",
        )
    chunk = AudioChunk(chunk_id="chunk", recording_id=source.recording_id, start_ms=0, end_ms=1_000)

    transcriber._write_manifest(
        sources=(source,),
        chunks=(chunk,),
        status="running",
        cloud_model=None,
        cloud_results={},
        completed_steps=("preprocess",),
    )

    manifest = json.loads(transcriber.manifest_path.read_text(encoding="utf-8"))
    assert manifest["text_review_requests"] == 2
    assert manifest["text_review_input_tokens"] == 60
    assert manifest["text_review_output_tokens"] == 60
    assert manifest["text_review_cost_cny"] > 0


def test_text_review_failure_falls_back_to_original_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    options = pipeline.TranscriptionOptions(
        source_directory=tmp_path,
        output_path=tmp_path / "out.md",
        run_local=True,
        run_cloud=False,
        run_deepseek_text=True,
    )
    transcriber = pipeline.RecordingTranscriber(options, deepseek_api_key="key")
    document = TranscriptDocument(
        generated_at=datetime.now(UTC), sources=(source,), speakers=(), segments=()
    )

    class BrokenReviewer:
        def review(self, _document: TranscriptDocument) -> None:
            raise pipeline.TranscriptTextReviewError("invalid topic coverage")

    monkeypatch.setattr(
        pipeline,
        "DeepSeekTranscriptReviewer",
        lambda **_kwargs: BrokenReviewer(),
    )

    reviewed = transcriber._review_text(document, {})

    assert reviewed == document
    assert transcriber._text_review_warning == "文本整理已回退：invalid topic coverage"


def test_short_cloud_chunk_and_vad_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    chunk = AudioChunk(
        chunk_id="chunk",
        recording_id="recording_01",
        start_ms=0,
        end_ms=360_000,
        light_path=str(audio),
        enhanced_path=str(audio),
        light_sha256="a" * 64,
        enhanced_sha256="b" * 64,
    )
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path), api_key="key")
    monkeypatch.setattr(
        pipeline,
        "extract_clip",
        lambda _src, dst, **_kwargs: (
            dst.parent.mkdir(parents=True, exist_ok=True) or dst.write_bytes(b"clip")
        ),
    )

    class Flash:
        def transcribe(self, *_args: object, **_kwargs: object) -> providers.FlashResult:
            return providers.FlashResult("short", "request", "文本", {})

    result = transcriber._short_cloud_chunk(chunk, Flash())  # type: ignore[arg-type]
    assert result.billed_seconds == 360
    assert len(result.sentences) == 2

    class VadModel:
        def generate(self, **_kwargs: object) -> list[dict[str, object]]:
            return [{"value": [[100, 1_000], [1_200, 2_000], ["bad"]]}]

    monkeypatch.setitem(
        sys.modules, "funasr", types.SimpleNamespace(AutoModel=lambda **_kwargs: VadModel())
    )
    monkeypatch.setattr(pipeline, "_ensure_model", lambda _model, directory: directory)
    intervals = transcriber._vad_intervals((chunk,))
    assert intervals["recording_01"] == [(0, 2_800)]


def test_speaker_mapping_extracts_samples_and_embedder_handles_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, duration_ms=20_000)
    audio = _wav(tmp_path / "audio.wav")
    chunk = AudioChunk(
        chunk_id="chunk",
        recording_id=source.recording_id,
        start_ms=0,
        end_ms=20_000,
        light_path=str(audio),
        enhanced_path=str(audio),
        light_sha256="a" * 64,
        enhanced_sha256="b" * 64,
    )
    cloud = [
        {
            "recording_id": source.recording_id,
            "chunk_id": chunk.chunk_id,
            "start_ms": 1_000,
            "end_ms": 6_000,
            "text": "你好",
            "speaker_id": "0",
            "task_id": "task",
            "model": "fun-asr",
        },
        {
            "recording_id": source.recording_id,
            "chunk_id": chunk.chunk_id,
            "start_ms": 7_000,
            "end_ms": 12_000,
            "text": "继续",
            "speaker_id": "0",
            "task_id": "task",
            "model": "fun-asr",
        },
    ]
    monkeypatch.setattr(pipeline, "_ensure_model", lambda _model, directory: directory)
    monkeypatch.setattr(
        pipeline,
        "extract_clip",
        lambda _src, dst, **_kwargs: dst.parent.mkdir(parents=True, exist_ok=True) or _wav(dst),
    )
    profile = SpeakerProfile(
        speaker="说话人 A",
        local_speaker_keys=("key",),
        recording_ids=(source.recording_id,),
        sample_count=2,
        confidence=0.8,
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_speakers",
        lambda samples, *, embedder: ({samples[0].key: "说话人 A"}, (profile,)),
    )
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path), api_key="key")
    mapping, profiles = transcriber._speaker_mapping((source,), (chunk,), cloud)
    assert next(iter(mapping.values())) == "说话人 A"
    assert profiles == (profile,)

    monkeypatch.setattr(
        pipeline,
        "_ensure_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cache missed")),
    )
    assert transcriber._speaker_mapping((source,), (chunk,), cloud) == (mapping, profiles)

    class FakeSpeakerModel:
        def generate(self, **_kwargs: object) -> list[dict[str, object]]:
            import torch

            return [{"spk_embedding": torch.tensor([[3.0, 4.0]])}]

    speaker_kwargs: dict[str, object] = {}

    def fake_auto_model(**kwargs: object) -> FakeSpeakerModel:
        speaker_kwargs.update(kwargs)
        return FakeSpeakerModel()

    monkeypatch.setitem(
        sys.modules,
        "funasr",
        types.SimpleNamespace(AutoModel=fake_auto_model),
    )
    vector = SpeakerEmbedder(model_directory=tmp_path).embed((audio,))
    assert vector.tolist() == pytest.approx([0.6, 0.8])
    assert speaker_kwargs["device"] == "cpu"


def test_adjudication_unclear_missing_and_hallucination_suppression(tmp_path: Path) -> None:
    source = _source(tmp_path, duration_ms=20_000)
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path), api_key="key")
    base = {
        "recording_id": source.recording_id,
        "chunk_id": "chunk",
        "start_ms": 1_000,
        "end_ms": 2_000,
        "text": "云端文本",
        "speaker_id": None,
        "task_id": "task",
        "model": "fun-asr",
    }
    ref = "chunk:1000:2000"
    local = [
        {
            "recording_id": source.recording_id,
            "start_ms": 1_000,
            "end_ms": 2_000,
            "text": "本地完全不同",
            "model": pipeline.LOCAL_MODEL_ID,
            "cloud_ref": ref,
        }
    ]
    unclear = transcriber._adjudicate((source,), [base], local, {}, {})
    assert unclear[0].decision == "unclear"
    assert unclear[0].text.startswith("[疑似：")
    missing = transcriber._adjudicate((source,), [base], [], {}, {})
    assert missing[0].decision == "cloud_primary"
    empty_local = [{**local[0], "text": ""}]
    assert transcriber._adjudicate((source,), [base], empty_local, {ref: {"text": ""}}, {}) == ()


def test_adjudication_includes_uncertain_vad_supplement(tmp_path: Path) -> None:
    source = _source(tmp_path, duration_ms=20_000)
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path), api_key="key")
    local = [
        {
            "recording_id": source.recording_id,
            "start_ms": 3_000,
            "end_ms": 5_000,
            "text": "本地补漏文本",
            "model": pipeline.LOCAL_MODEL_ID,
            "cloud_ref": None,
        }
    ]
    segments = transcriber._adjudicate((source,), [], local, {}, {})
    assert len(segments) == 1
    assert segments[0].speaker == "说话人（未确认）"
    assert segments[0].text == "[疑似：本地补漏文本]"
    assert segments[0].decision == "local_primary"
    assert "vad_supplement" in segments[0].flags


def test_speaker_mapping_assigns_letter_to_short_unresolved_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, duration_ms=20_000)
    audio = _wav(tmp_path / "light.wav")
    chunk = AudioChunk(
        chunk_id="recording_01_chunk_001",
        recording_id=source.recording_id,
        start_ms=0,
        end_ms=20_000,
        light_path=str(audio),
        enhanced_path=str(audio),
        light_sha256="a" * 64,
        enhanced_sha256="b" * 64,
    )
    cloud = [
        {
            "recording_id": source.recording_id,
            "chunk_id": chunk.chunk_id,
            "start_ms": 1_000,
            "end_ms": 2_000,
            "text": "短句",
            "speaker_id": "7",
            "task_id": "task",
            "model": "fun-asr",
        }
    ]
    monkeypatch.setattr(pipeline, "_ensure_model", lambda _model, directory: directory)
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path), api_key="key")
    mapping, profiles = transcriber._speaker_mapping((source,), (chunk,), cloud)
    key = f"{source.recording_id}|{chunk.chunk_id}|7"
    assert mapping[key] == "说话人 A"
    assert profiles[0].confidence == 0.25
    assert profiles[0].sample_count == 0


def test_cli_transcribe_command_uses_runtime_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, duration_ms=1_000)

    class FakeTranscriber:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def run(self) -> TranscriptDocument:
            return TranscriptDocument(
                generated_at=datetime.now(UTC), sources=(source,), speakers=(), segments=()
            )

    monkeypatch.setattr(pipeline, "RecordingTranscriber", FakeTranscriber)
    result = CliRunner().invoke(
        app,
        [
            "transcribe-recordings",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.md"),
            "--no-cloud",
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["sources"] == 1


def test_provider_validation_and_error_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="API key"):
        providers.DashScopeFileTranscriber(api_key="")
    with pytest.raises(ValueError, match="API key"):
        providers.DashScopeFlashTranscriber(api_key="")
    with pytest.raises(providers.TranscriptionProviderError, match="non-object"):
        providers._json_object(_Response([]))  # type: ignore[arg-type]

    class InvalidJson(_Response):
        def json(self) -> object:
            raise requests.JSONDecodeError("bad", "x", 0)

    with pytest.raises(providers.TranscriptionProviderError, match="non-JSON"):
        providers._json_object(InvalidJson({}))  # type: ignore[arg-type]

    audio = _wav(tmp_path / "audio.wav")
    failure = _Session(gets=[], posts=[_Response({"code": "BAD", "message": "no"}, 400)])
    monkeypatch.setattr(requests, "Session", lambda: failure)
    with pytest.raises(providers.TranscriptionProviderError, match="short ASR failed"):
        providers.DashScopeFlashTranscriber(api_key="key").transcribe(
            audio, model=pipeline.THIRD_PASS_MODEL, context_text="上下文"
        )

    too_large = tmp_path / "large.flac"
    too_large.write_bytes(bytes(7_500_001))
    monkeypatch.setattr(requests, "Session", lambda: _Session(gets=[], posts=[]))
    with pytest.raises(providers.TranscriptionProviderError, match="10 MB"):
        providers.DashScopeFlashTranscriber(api_key="key").transcribe(
            too_large, model=pipeline.THIRD_PASS_MODEL
        )


def test_upload_policy_rejections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = _wav(tmp_path / "audio.wav")
    monkeypatch.setattr(
        requests,
        "Session",
        lambda: _Session(gets=[_Response({"code": "DENIED"}, 403)], posts=[]),
    )
    with pytest.raises(providers.TranscriptionProviderError, match="policy failed"):
        providers.DashScopeFileTranscriber(api_key="key").upload_temporary(audio, model="fun-asr")

    monkeypatch.setattr(
        requests,
        "Session",
        lambda: _Session(gets=[_Response({"data": {"max_file_size_mb": 0.000001}})], posts=[]),
    )
    with pytest.raises(providers.TranscriptionProviderError, match="upload limit"):
        providers.DashScopeFileTranscriber(api_key="key").upload_temporary(audio, model="fun-asr")


def test_flash_provider_reads_choices_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _wav(tmp_path / "audio.wav")
    session = _Session(
        gets=[],
        posts=[
            _Response(
                {
                    "request_id": "request",
                    "output": {"choices": [{"message": {"content": "choice transcript"}}]},
                }
            )
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)
    result = providers.DashScopeFlashTranscriber(api_key="key").transcribe(
        audio, model=pipeline.THIRD_PASS_MODEL
    )
    assert result.text == "choice transcript"


def test_file_provider_reports_failed_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = _wav(tmp_path / "audio.wav")
    session = _Session(
        gets=[_Response({"output": {"task_status": "FAILED"}})],
        posts=[_Response({"output": {"task_id": "task"}})],
    )
    provider = providers.DashScopeFileTranscriber(api_key="key")
    monkeypatch.setattr(provider, "upload_temporary", lambda *_args, **_kwargs: "oss://audio")
    provider._session = session
    with pytest.raises(providers.TranscriptionProviderError, match="ended as FAILED"):
        provider.transcribe(audio, model="fun-asr", diarization=True)


def test_file_provider_rejects_invalid_submission_and_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _wav(tmp_path / "audio.wav")
    provider = providers.DashScopeFileTranscriber(api_key="key")
    monkeypatch.setattr(provider, "upload_temporary", lambda *_args, **_kwargs: "oss://audio")
    provider._session = _Session(gets=[], posts=[_Response({"code": "BAD"}, 400)])
    with pytest.raises(providers.TranscriptionProviderError, match="submission failed"):
        provider.transcribe(
            audio,
            model=pipeline.PRIMARY_CLOUD_MODEL,
            diarization=True,
            context_text="context",
        )

    provider._session = _Session(
        gets=[_Response({"code": "BAD"}, 500)],
        posts=[_Response({"output": {"task_id": "task"}})],
    )
    with pytest.raises(providers.TranscriptionProviderError, match="task query failed"):
        provider.transcribe(audio, model="fun-asr", diarization=True)


def test_run_persists_failure_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source(tmp_path)
    chunk = AudioChunk(
        chunk_id="chunk",
        recording_id=source.recording_id,
        start_ms=0,
        end_ms=source.duration_ms,
    )
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path), api_key="key")
    monkeypatch.setattr(transcriber, "_sources", lambda: (source,))
    monkeypatch.setattr(transcriber, "_validate_resume_sources", lambda _sources: None)
    monkeypatch.setattr(transcriber, "_prepare_chunks", lambda _sources: (chunk,))

    def fail_cloud(_chunks: object) -> object:
        raise pipeline.RecordingTranscriptionError("cloud failed")

    manifests: list[dict[str, object]] = []
    monkeypatch.setattr(transcriber, "_cloud_pass", fail_cloud)
    monkeypatch.setattr(
        transcriber,
        "_write_manifest",
        lambda **kwargs: manifests.append(kwargs),
    )
    with pytest.raises(pipeline.RecordingTranscriptionError, match="cloud failed"):
        transcriber.run()
    assert manifests[-1]["status"] == "failed"
    assert manifests[-1]["errors"] == ("cloud failed",)


def test_cli_transcribe_command_reports_pipeline_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailedTranscriber:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise pipeline.RecordingTranscriptionError("pipeline failed")

    monkeypatch.setattr(pipeline, "RecordingTranscriber", FailedTranscriber)
    result = CliRunner().invoke(
        app,
        [
            "transcribe-recordings",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.md"),
            "--no-cloud",
            "--json",
        ],
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"] == "pipeline failed"


def test_file_provider_validates_task_result_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _wav(tmp_path / "audio.wav")

    def assert_error(*, gets: list[_Response], posts: list[_Response], match: str) -> None:
        provider = providers.DashScopeFileTranscriber(api_key="key")
        monkeypatch.setattr(provider, "upload_temporary", lambda *_args, **_kwargs: "oss://audio")
        provider._session = _Session(gets=gets, posts=posts)
        with pytest.raises(providers.TranscriptionProviderError, match=match):
            provider.transcribe(audio, model="fun-asr", diarization=True)

    assert_error(gets=[], posts=[_Response({"output": {}})], match="omitted task_id")
    succeeded = {"output": {"task_status": "SUCCEEDED", "results": []}}
    assert_error(
        gets=[_Response(succeeded)],
        posts=[_Response({"output": {"task_id": "task"}})],
        match="no result object",
    )
    failed_subtask = {
        "output": {
            "task_status": "SUCCEEDED",
            "results": [{"subtask_status": "FAILED", "code": "BAD", "message": "bad"}],
        }
    }
    assert_error(
        gets=[_Response(failed_subtask)],
        posts=[_Response({"output": {"task_id": "task"}})],
        match="subtask failed",
    )
    invalid_url = {
        "output": {
            "task_status": "SUCCEEDED",
            "results": [{"subtask_status": "SUCCEEDED", "transcription_url": "http://x"}],
        }
    }
    assert_error(
        gets=[_Response(invalid_url)],
        posts=[_Response({"output": {"task_id": "task"}})],
        match="valid HTTPS",
    )
    download_error = {
        "output": {
            "task_status": "SUCCEEDED",
            "results": [
                {
                    "subtask_status": "SUCCEEDED",
                    "transcription_url": "https://example.test/result",
                }
            ],
        }
    }
    assert_error(
        gets=[_Response(download_error), _Response({}, 500)],
        posts=[_Response({"output": {"task_id": "task"}})],
        match="result download failed",
    )


def test_cloud_pass_uses_short_model_after_both_file_models_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    chunk = AudioChunk(
        chunk_id="chunk",
        recording_id="recording_01",
        start_ms=0,
        end_ms=60_000,
        light_path=str(audio),
        enhanced_path=str(audio),
        light_sha256="a" * 64,
        enhanced_sha256="b" * 64,
    )

    class FailedFile:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def transcribe(self, *_args: object, **_kwargs: object) -> CloudResult:
            raise providers.TranscriptionProviderError("unavailable")

    class WorkingFlash:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def transcribe(self, *_args: object, **_kwargs: object) -> providers.FlashResult:
            return providers.FlashResult(pipeline.SHORT_CLOUD_MODEL, "request", "短模型", {})

    monkeypatch.setattr(pipeline, "DashScopeFileTranscriber", FailedFile)
    monkeypatch.setattr(pipeline, "DashScopeFlashTranscriber", WorkingFlash)
    monkeypatch.setattr(
        pipeline,
        "extract_clip",
        lambda _src, dst, **_kwargs: (
            dst.parent.mkdir(parents=True, exist_ok=True) or dst.write_bytes(b"clip")
        ),
    )
    transcriber = pipeline.RecordingTranscriber(_options(tmp_path), api_key="key")
    model, results, sentences = transcriber._cloud_pass((chunk,))
    assert model == pipeline.SHORT_CLOUD_MODEL
    assert results["pilot"].model == pipeline.SHORT_CLOUD_MODEL
    assert sentences[0]["text"] == "短模型"


def test_explicit_audio_model_directory_overrides_default(tmp_path: Path) -> None:
    model_directory = tmp_path / "models"
    options = pipeline.TranscriptionOptions(
        source_directory=tmp_path,
        output_path=tmp_path / "out.md",
        model_directory=model_directory,
        run_local=True,
        run_cloud=False,
    )
    transcriber = pipeline.RecordingTranscriber(options, api_key=None)
    assert transcriber.models_directory == model_directory.resolve()
