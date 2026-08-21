"""Tests for constrained transcript review and topic capture."""

# ruff: noqa: RUF001 - Chinese uncertainty punctuation is intentional test data.

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import research_kb.audio.text_review as text_review
from research_kb.audio.domain import (
    ProviderCandidate,
    RecordingSource,
    TranscriptDocument,
    TranscriptSegment,
)
from research_kb.audio.text_review import (
    DeepSeekTranscriptReviewer,
    ReviewResponse,
    TranscriptTextReviewError,
    render_topic_markdown,
)


def _document() -> TranscriptDocument:
    source = RecordingSource(
        recording_id="recording_01",
        path="meeting.mp3",
        filename="meeting.mp3",
        sha256="a" * 64,
        size_bytes=10,
        duration_ms=20_000,
        recorded_at=datetime(2026, 8, 5, 13, 20, tzinfo=UTC),
        codec="mp3",
        sample_rate_hz=44_100,
        channels=2,
    )
    segments = (
        TranscriptSegment(
            segment_id="segment_0000001",
            recording_id=source.recording_id,
            start_ms=1_000,
            end_ms=5_000,
            speaker="说话人 A",
            text="[疑似：指数空间不够]",
            decision="unclear",
            confidence=0.4,
            flags=("models_disagree", "no_majority"),
            candidates=(
                ProviderCandidate(
                    provider="cloud",
                    model="cloud",
                    text="指数空间不够",
                    start_ms=1_000,
                    end_ms=5_000,
                ),
                ProviderCandidate(
                    provider="local",
                    model="local",
                    text="存储空间不够",
                    start_ms=1_000,
                    end_ms=5_000,
                ),
            ),
        ),
        TranscriptSegment(
            segment_id="segment_0000002",
            recording_id=source.recording_id,
            start_ms=6_000,
            end_ms=9_000,
            speaker="说话人 B",
            text="预算是120元",
            decision="unclear",
            confidence=0.4,
            flags=("models_disagree", "no_majority", "sensitive_difference"),
            candidates=(
                ProviderCandidate(
                    provider="cloud",
                    model="cloud",
                    text="预算是120元",
                    start_ms=6_000,
                    end_ms=9_000,
                ),
                ProviderCandidate(
                    provider="local",
                    model="local",
                    text="预算是130元",
                    start_ms=6_000,
                    end_ms=9_000,
                ),
            ),
        ),
        TranscriptSegment(
            segment_id="segment_0000003",
            recording_id=source.recording_id,
            start_ms=12_000,
            end_ms=18_000,
            speaker="说话人 C",
            text="第二场开始讨论交付",
            decision="cloud_primary",
            confidence=0.6,
            candidates=(
                ProviderCandidate(
                    provider="cloud",
                    model="cloud",
                    text="第二场开始讨论交付",
                    start_ms=12_000,
                    end_ms=18_000,
                ),
            ),
        ),
    )
    return TranscriptDocument(
        title="连续会议",
        generated_at=datetime.now(UTC),
        sources=(source,),
        speakers=(),
        segments=segments,
    )


def test_deepseek_review_selects_only_existing_text_and_splits_meetings(
    tmp_path: Path,
) -> None:
    responses = [
        {
            "decisions": [
                {
                    "segment_id": "segment_0000001",
                    "choice": "local",
                    "confidence": 0.91,
                    "reason_code": "context_storage",
                },
                {
                    "segment_id": "segment_0000002",
                    "choice": "unclear",
                    "confidence": 0.96,
                    "reason_code": "numeric_conflict",
                },
            ],
            "topics": [
                {
                    "title": "存储容量与预算",
                    "start_segment_id": "segment_0000001",
                    "end_segment_id": "segment_0000002",
                    "summary": "讨论存储容量和预算。",
                    "keywords": ["存储", "预算"],
                    "evidence_segment_ids": ["segment_0000001", "segment_0000002"],
                },
                {
                    "title": "交付安排",
                    "start_segment_id": "segment_0000003",
                    "end_segment_id": "segment_0000003",
                    "summary": "第二场讨论交付。",
                    "keywords": ["交付"],
                    "evidence_segment_ids": ["segment_0000003"],
                },
            ],
        },
        {
            "topic_groups": [
                {
                    "source_topic_ids": ["topic_stub_0001"],
                    "title": "存储容量与预算",
                    "summary": "讨论存储容量和预算。",
                    "keywords": ["存储", "预算"],
                },
                {
                    "source_topic_ids": ["topic_stub_0002"],
                    "title": "交付安排",
                    "summary": "讨论交付安排。",
                    "keywords": ["交付"],
                },
            ],
            "meetings": [
                {
                    "source_topic_ids": ["topic_stub_0001"],
                    "title": "第一场会议",
                    "summary": "存储与预算。",
                },
                {
                    "source_topic_ids": ["topic_stub_0002"],
                    "title": "第二场会议",
                    "summary": "交付安排。",
                },
            ],
        },
    ]

    def requester(_request: dict[str, object]) -> ReviewResponse:
        payload = responses.pop(0)
        return ReviewResponse(
            json.dumps(payload, ensure_ascii=False),
            prompt_cache_miss_tokens=1_000,
            output_tokens=200,
        )

    reviewed, analysis = DeepSeekTranscriptReviewer(
        api_key="temporary",
        requester=requester,
        batch_characters=20_000,
        cache_directory=tmp_path / "cache",
        expected_meeting_count=2,
    ).review(_document())

    assert reviewed.segments[0].text == "存储空间不够"
    assert reviewed.segments[0].decision == "text_context"
    assert "no_majority" not in reviewed.segments[0].flags
    assert {candidate.text for candidate in reviewed.segments[0].candidates} == {
        "指数空间不够",
        "存储空间不够",
    }
    assert reviewed.segments[1].text == "预算是120元"
    assert len(analysis.meetings) == 2
    assert analysis.expected_meeting_count == 2
    assert analysis.estimated_cost_cny > 0
    markdown = render_topic_markdown(reviewed, analysis)
    assert "第一场会议" in markdown
    assert "segment_0000001" in markdown

    cached_reviewed, cached_analysis = DeepSeekTranscriptReviewer(
        api_key="temporary",
        requester=lambda _request: (_ for _ in ()).throw(AssertionError("cache missed")),
        batch_characters=20_000,
        cache_directory=tmp_path / "cache",
        expected_meeting_count=2,
    ).review(_document())
    assert cached_reviewed.segments[0].text == reviewed.segments[0].text
    assert cached_analysis.estimated_cost_cny == analysis.estimated_cost_cny


def test_deepseek_accepts_descriptive_reason_code_longer_than_forty_characters() -> None:
    responses = [
        json.dumps(
            {
                "decisions": [
                    {
                        "segment_id": "segment_0000001",
                        "choice": "local",
                        "confidence": 0.91,
                        "reason_code": "proper_name_and_brand_resolved_by_context",
                    }
                ],
                "topics": [],
            }
        )
    ]
    reviewed, analysis = DeepSeekTranscriptReviewer(
        api_key="temporary",
        requester=lambda _request: responses.pop(0),
        batch_characters=20_000,
    ).review(_document())

    assert reviewed.segments[0].text == "存储空间不够"
    assert analysis.decisions[0].reason_code == "proper_name_and_brand_resolved_by_context"


def test_deepseek_review_rejects_missing_candidate_and_invalid_topic_cover() -> None:
    missing_candidate = json.dumps(
        {
            "decisions": [
                {
                    "segment_id": "segment_0000003",
                    "choice": "local",
                    "confidence": 0.9,
                    "reason_code": "bad",
                }
            ],
            "topics": [],
        }
    )
    reviewer = DeepSeekTranscriptReviewer(
        api_key="temporary", requester=lambda _request: missing_candidate
    )
    with pytest.raises(TranscriptTextReviewError, match="missing ASR candidate"):
        reviewer.review(_document())


def test_deepseek_review_enforces_pre_request_cost_cap() -> None:
    called = False

    def requester(_request: dict[str, object]) -> str:
        nonlocal called
        called = True
        return "{}"

    reviewer = DeepSeekTranscriptReviewer(
        api_key="temporary",
        requester=requester,
        cost_cap_cny=0.000001,
        batch_characters=20_000,
    )
    with pytest.raises(TranscriptTextReviewError, match="cost cap"):
        reviewer.review(_document())
    assert called is False


def test_deepseek_reuses_paid_cache_even_when_a_new_request_would_exceed_cap(
    tmp_path: Path,
) -> None:
    response = json.dumps({"decisions": [], "topics": []})
    initial = DeepSeekTranscriptReviewer(
        api_key="key",
        requester=lambda _request: ReviewResponse(
            response,
            prompt_cache_miss_tokens=500,
            output_tokens=100,
        ),
        batch_characters=20_000,
        cache_directory=tmp_path,
    )
    initial.review(_document())

    cached = DeepSeekTranscriptReviewer(
        api_key="key",
        requester=lambda _request: (_ for _ in ()).throw(AssertionError("cache missed")),
        cost_cap_cny=0.000001,
        batch_characters=20_000,
        cache_directory=tmp_path,
    )
    _reviewed, analysis = cached.review(_document())

    assert len(analysis.usage) == 1


def test_deepseek_keeps_completed_windows_when_text_cost_cap_is_reached() -> None:
    response = json.dumps({"decisions": [], "topics": []})
    calls = 0

    def requester(_request: dict[str, object]) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return response
        raise TranscriptTextReviewError("DeepSeek text-review cost cap would be exceeded")

    reviewer = DeepSeekTranscriptReviewer(
        api_key="key",
        requester=requester,
        batch_characters=2_000,
        allow_window_cost_fallback=True,
    )
    document = _document().model_copy(update={"segments": _document().segments * 30})
    _reviewed, analysis = reviewer.review(document)

    assert calls == 2
    assert analysis.window_fallback_count > 0


def test_deepseek_real_request_adapter_captures_usage_and_disables_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    prompt_cache_hit_tokens=30,
                    prompt_cache_miss_tokens=70,
                    completion_tokens=20,
                ),
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(text_review, "OpenAI", lambda **_kwargs: fake_client)
    reviewer = DeepSeekTranscriptReviewer(api_key="temporary")

    response = reviewer._request({"messages": [], "max_tokens": 123})

    assert response.content == '{"ok":true}'
    assert response.prompt_cache_hit_tokens == 30
    assert response.prompt_cache_miss_tokens == 70
    assert response.output_tokens == 20
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert captured["temperature"] == 0


def test_deepseek_real_request_rejects_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyCompletions:
        def create(self, **_kwargs: object) -> object:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=" "))],
                usage=SimpleNamespace(),
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=EmptyCompletions()))
    monkeypatch.setattr(text_review, "OpenAI", lambda **_kwargs: fake_client)

    with pytest.raises(TranscriptTextReviewError, match="empty response"):
        DeepSeekTranscriptReviewer(api_key="temporary")._request({"messages": [], "max_tokens": 10})


def test_deepseek_empty_transcript_needs_no_request() -> None:
    empty = _document().model_copy(update={"segments": ()})
    reviewer = DeepSeekTranscriptReviewer(
        api_key="temporary",
        requester=lambda _request: (_ for _ in ()).throw(AssertionError("unexpected request")),
    )

    reviewed, analysis = reviewer.review(empty)

    assert reviewed.segments == ()
    assert analysis.decisions == ()
    assert analysis.topics == ()
    assert analysis.meetings == ()
    assert analysis.usage == ()


def test_user_prompt_guides_relevance_and_is_recorded() -> None:
    requests: list[dict[str, object]] = []

    def request(payload: dict[str, object]) -> str:
        requests.append(payload)
        return json.dumps({"decisions": [], "topics": []}, ensure_ascii=False)

    prompt = "只保留财经内容，删除感谢礼物、唱歌和闲聊。"
    _, analysis = DeepSeekTranscriptReviewer(
        api_key="temporary",
        user_prompt=prompt,
        requester=request,
    ).review(_document())

    assert analysis.user_prompt == prompt
    assert len(requests) == 1
    system_prompt = str(requests[0]["messages"][0]["content"])  # type: ignore[index]
    assert prompt in system_prompt
    assert "Omit portions explicitly marked irrelevant" in system_prompt


def test_deepseek_unconstrained_review_uses_model_meeting_groups() -> None:
    responses = iter(
        [
            {
                "decisions": [],
                "topics": [
                    {
                        "title": "容量",
                        "start_segment_id": "segment_0000001",
                        "end_segment_id": "segment_0000002",
                        "summary": "讨论容量。",
                        "evidence_segment_ids": ["segment_0000001"],
                    },
                    {
                        "title": "交付",
                        "start_segment_id": "segment_0000003",
                        "end_segment_id": "segment_0000003",
                        "summary": "讨论交付。",
                        "evidence_segment_ids": ["segment_0000003"],
                    },
                ],
            },
            {
                "topic_groups": [
                    {"source_topic_ids": ["topic_stub_0001"], "title": "容量"},
                    {"source_topic_ids": ["topic_stub_0002"], "title": "交付"},
                ],
                "meetings": [
                    {
                        "source_topic_ids": ["topic_stub_0001"],
                        "title": "容量会议",
                    },
                    {
                        "source_topic_ids": ["topic_stub_0002"],
                        "title": "交付会议",
                    },
                ],
            },
        ]
    )
    reviewer = DeepSeekTranscriptReviewer(
        api_key="temporary",
        batch_characters=20_000,
        requester=lambda _request: json.dumps(next(responses), ensure_ascii=False),
    )

    _, analysis = reviewer.review(_document())

    assert [meeting.title for meeting in analysis.meetings] == ["容量会议", "交付会议"]
    assert analysis.meetings[0].summary == "容量"
    assert analysis.meetings[1].topic_ids == ("topic_0002",)


def test_deepseek_transport_error_is_redacted() -> None:
    reviewer = DeepSeekTranscriptReviewer(
        api_key="temporary-secret",
        requester=lambda _request: (_ for _ in ()).throw(RuntimeError("temporary-secret")),
    )

    with pytest.raises(TranscriptTextReviewError, match="RuntimeError") as caught:
        reviewer.review(_document())
    assert "temporary-secret" not in str(caught.value)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"api_key": ""}, "API key"),
        ({"api_key": "key", "cost_cap_cny": 0}, "cost cap"),
        ({"api_key": "key", "batch_characters": 1_999}, "batch size"),
        ({"api_key": "key", "expected_meeting_count": 0}, "meeting count"),
    ],
)
def test_deepseek_reviewer_validates_configuration(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DeepSeekTranscriptReviewer(**kwargs)  # type: ignore[arg-type]


def test_deepseek_preserves_explicit_review_errors() -> None:
    reviewer = DeepSeekTranscriptReviewer(
        api_key="key",
        requester=lambda _request: (_ for _ in ()).throw(
            TranscriptTextReviewError("bounded failure")
        ),
    )
    with pytest.raises(TranscriptTextReviewError, match="bounded failure"):
        reviewer.review(_document())


@pytest.mark.parametrize(
    ("response", "match"),
    [
        ("{", "invalid window JSON"),
        (
            json.dumps(
                {
                    "decisions": [
                        {
                            "segment_id": "missing",
                            "choice": "cloud",
                            "confidence": 0.8,
                            "reason_code": "context",
                        }
                    ],
                    "topics": [],
                }
            ),
            "invalid decision segment",
        ),
        (
            json.dumps(
                {
                    "decisions": [],
                    "topics": [
                        {
                            "title": "outside",
                            "start_segment_id": "missing",
                            "end_segment_id": "segment_0000001",
                            "summary": "outside",
                        }
                    ],
                }
            ),
            "exceeded its review window",
        ),
        (
            json.dumps(
                {
                    "decisions": [],
                    "topics": [
                        {
                            "title": "inverted",
                            "start_segment_id": "segment_0000002",
                            "end_segment_id": "segment_0000001",
                            "summary": "inverted",
                        }
                    ],
                }
            ),
            "inverted segment range",
        ),
        (
            json.dumps(
                {
                    "decisions": [],
                    "topics": [
                        {
                            "title": "bad evidence",
                            "start_segment_id": "segment_0000001",
                            "end_segment_id": "segment_0000001",
                            "summary": "bad evidence",
                            "evidence_segment_ids": ["missing"],
                        }
                    ],
                }
            ),
            "cited an unknown segment",
        ),
    ],
)
def test_deepseek_rejects_untraceable_window_output(response: str, match: str) -> None:
    reviewer = DeepSeekTranscriptReviewer(
        api_key="key", requester=lambda _request: response, batch_characters=20_000
    )
    with pytest.raises(TranscriptTextReviewError, match=match):
        reviewer.review(_document())


def test_deepseek_rejects_invalid_consolidation_and_meeting_count() -> None:
    window = json.dumps(
        {
            "decisions": [],
            "topics": [
                {
                    "title": "topic",
                    "start_segment_id": "segment_0000001",
                    "end_segment_id": "segment_0000003",
                    "summary": "summary",
                }
            ],
        }
    )
    invalid_responses = iter([window, "{}"])
    invalid = DeepSeekTranscriptReviewer(
        api_key="key",
        requester=lambda _request: next(invalid_responses),
        batch_characters=20_000,
    )
    with pytest.raises(TranscriptTextReviewError, match="invalid consolidation JSON"):
        invalid.review(_document())

    one_meeting = json.dumps(
        {
            "topic_groups": [{"source_topic_ids": ["topic_stub_0001"], "title": "topic"}],
            "meetings": [{"source_topic_ids": ["topic_stub_0001"], "title": "meeting"}],
        }
    )
    wrong_count_responses = iter([window, one_meeting])
    wrong_count = DeepSeekTranscriptReviewer(
        api_key="key",
        requester=lambda _request: next(wrong_count_responses),
        batch_characters=20_000,
        expected_meeting_count=2,
    )
    with pytest.raises(TranscriptTextReviewError, match="wrong number of meetings"):
        wrong_count.review(_document())


def test_deepseek_can_fallback_locally_when_consolidation_json_is_truncated() -> None:
    window = json.dumps(
        {
            "decisions": [
                {
                    "segment_id": "segment_0000001",
                    "choice": "local",
                    "confidence": 0.8,
                    "reason_code": "context",
                }
            ],
            "topics": [
                {
                    "title": "存储空间",
                    "start_segment_id": "segment_0000001",
                    "end_segment_id": "segment_0000003",
                    "summary": "讨论存储与交付。",
                }
            ],
        }
    )
    responses = iter([window, '{"topic_groups": ['])
    reviewer = DeepSeekTranscriptReviewer(
        api_key="key",
        requester=lambda _request: next(responses),
        batch_characters=20_000,
        allow_consolidation_fallback=True,
    )

    reviewed, analysis = reviewer.review(_document())

    assert reviewed.segments[0].text == "存储空间不够"
    assert analysis.topics[0].title == "存储空间"
    assert analysis.meetings[0].title == "meeting 录音"


def test_deepseek_can_fallback_locally_when_consolidation_request_exceeds_cap() -> None:
    window = json.dumps(
        {
            "decisions": [],
            "topics": [
                {
                    "title": "存储空间",
                    "start_segment_id": "segment_0000001",
                    "end_segment_id": "segment_0000003",
                    "summary": "讨论存储与交付。",
                }
            ],
        }
    )
    calls = 0

    def requester(_request: dict[str, object]) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return window
        raise TranscriptTextReviewError("DeepSeek text-review cost cap would be exceeded")

    _reviewed, analysis = DeepSeekTranscriptReviewer(
        api_key="key",
        requester=requester,
        batch_characters=20_000,
        allow_consolidation_fallback=True,
    ).review(_document())

    assert calls == 2
    assert analysis.topics[0].title == "存储空间"
    assert analysis.meetings[0].title == "meeting 录音"


def test_deepseek_can_fallback_when_consolidation_coverage_is_incomplete() -> None:
    window = json.dumps(
        {
            "decisions": [],
            "topics": [
                {
                    "title": "存储空间",
                    "start_segment_id": "segment_0000001",
                    "end_segment_id": "segment_0000003",
                    "summary": "讨论存储与交付。",
                }
            ],
        }
    )
    incomplete = json.dumps(
        {
            "topic_groups": [],
            "meetings": [],
        }
    )
    responses = iter([window, incomplete])
    reviewer = DeepSeekTranscriptReviewer(
        api_key="key",
        requester=lambda _request: next(responses),
        batch_characters=20_000,
        allow_consolidation_fallback=True,
    )

    _reviewed, analysis = reviewer.review(_document())

    assert analysis.topics[0].title == "存储空间"
    assert analysis.meetings[0].title == "meeting 录音"


def test_deepseek_rejects_reordered_topic_cover() -> None:
    with pytest.raises(TranscriptTextReviewError, match="exactly once and in order"):
        DeepSeekTranscriptReviewer._validate_ordered_cover(
            [("topic_stub_0002",), ("topic_stub_0001",)],
            ("topic_stub_0001", "topic_stub_0002"),
        )


def test_review_helpers_cover_empty_and_non_applicable_choices() -> None:
    document = _document()
    empty_document = document.model_copy(update={"segments": ()})
    assert text_review._meetings_from_largest_gaps(empty_document, (), count=1) == ()
    with pytest.raises(TranscriptTextReviewError, match="exceeds audible segments"):
        text_review._meetings_from_largest_gaps(document, (), count=4)

    analysis = text_review.TranscriptTextAnalysis(
        generated_at=datetime.now(UTC),
        model="deepseek-test",
        source_transcript_sha256="b" * 64,
        decisions=(
            text_review.TextReviewDecision(
                segment_id="segment_0000003",
                choice="equivalent",
                confidence=0.9,
                reason_code="already_plain",
            ),
        ),
        topics=(),
        meetings=(),
        usage=(),
        estimated_cost_cny=0,
        cost_cap_cny=1,
    )
    reviewed = text_review.apply_text_review(document, analysis)
    assert reviewed == document

    equivalent_analysis = analysis.model_copy(
        update={
            "decisions": (
                text_review.TextReviewDecision(
                    segment_id="segment_0000001",
                    choice="equivalent",
                    confidence=0.9,
                    reason_code="equivalent_candidates",
                ),
            )
        }
    )
    equivalent_reviewed = text_review.apply_text_review(document, equivalent_analysis)
    assert equivalent_reviewed.segments[0].text == document.segments[0].candidates[0].text

    empty_candidate = document.segments[2].candidates[0].model_copy(update={"text": ""})
    unavailable = document.model_copy(
        update={
            "segments": (
                *document.segments[:2],
                document.segments[2].model_copy(update={"candidates": (empty_candidate,)}),
            )
        }
    )
    cloud_choice = analysis.model_copy(
        update={
            "decisions": (
                text_review.TextReviewDecision(
                    segment_id="segment_0000003",
                    choice="cloud",
                    confidence=0.9,
                    reason_code="unavailable",
                ),
            )
        }
    )
    assert text_review.apply_text_review(unavailable, cloud_choice) == unavailable
    assert text_review._candidate_text(document.segments[2], "local") is None

    markdown = render_topic_markdown(document, analysis)
    assert "DeepSeek" in markdown


def test_deepseek_batches_on_recording_boundary() -> None:
    document = _document()
    second_recording = document.segments[2].model_copy(update={"recording_id": "recording_02"})
    split_document = document.model_copy(
        update={"segments": (*document.segments[:2], second_recording)}
    )
    reviewer = DeepSeekTranscriptReviewer(api_key="key", batch_characters=20_000)
    assert [len(batch) for batch in reviewer._batches(split_document)] == [2, 1]

