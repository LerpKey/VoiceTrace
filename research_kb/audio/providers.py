"""Cloud and local providers for long-recording transcription."""

from __future__ import annotations

import base64
import json
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

import requests

from research_kb.audio.core import sanitize_external_payload


class TranscriptionProviderError(RuntimeError):
    """Raised when an ASR provider cannot return trustworthy evidence."""


@dataclass(frozen=True)
class ProviderSentence:
    """Provider-neutral timestamped sentence."""

    start_ms: int
    end_ms: int
    text: str
    speaker_id: str | None = None
    task_id: str | None = None


@dataclass(frozen=True)
class CloudResult:
    """Sanitized result and billable duration from one cloud task."""

    model: str
    task_id: str
    billed_seconds: int
    sentences: tuple[ProviderSentence, ...]
    sanitized_payload: dict[str, Any]


@dataclass(frozen=True)
class FlashResult:
    """One synchronous short-audio recognition used for fallback/adjudication."""

    model: str
    request_id: str
    text: str
    sanitized_payload: dict[str, Any]


def _json_object(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except requests.JSONDecodeError as error:
        raise TranscriptionProviderError(
            f"provider returned non-JSON HTTP {response.status_code}"
        ) from error
    if not isinstance(payload, dict):
        raise TranscriptionProviderError("provider returned a non-object JSON payload")
    return cast(dict[str, Any], payload)


class DashScopeFileTranscriber:
    """REST client for temporary upload plus asynchronous file transcription."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com",
        timeout_seconds: float = 120,
        poll_seconds: float = 5,
        max_wait_seconds: float = 10_800,
    ) -> None:
        if not api_key:
            raise ValueError("DashScope API key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._poll_seconds = poll_seconds
        self._max_wait_seconds = max_wait_seconds
        self._session = requests.Session()

    @property
    def _authorization_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def upload_temporary(self, path: Path, *, model: str) -> str:
        """Upload one derivative using short-lived OSS credentials."""
        source = path.expanduser().resolve()
        policy_response = self._session.get(
            f"{self._base_url}/api/v1/uploads",
            params={"action": "getPolicy", "model": model},
            headers={**self._authorization_headers, "Content-Type": "application/json"},
            timeout=self._timeout_seconds,
        )
        policy_payload = _json_object(policy_response)
        if policy_response.status_code != 200:
            raise TranscriptionProviderError(
                "temporary upload policy failed: "
                f"{policy_payload.get('code', policy_response.status_code)}"
            )
        data = policy_payload.get("data")
        if not isinstance(data, dict):
            raise TranscriptionProviderError("temporary upload policy omitted data")
        maximum_mb = float(data.get("max_file_size_mb", 0))
        if maximum_mb and source.stat().st_size > maximum_mb * 1024 * 1024:
            raise TranscriptionProviderError(
                f"temporary upload limit is {maximum_mb:g} MB for {model}"
            )
        key = f"{data['upload_dir']}/{source.name}"
        form = {
            "OSSAccessKeyId": str(data["oss_access_key_id"]),
            "policy": str(data["policy"]),
            "Signature": str(data["signature"]),
            "key": key,
            "x-oss-object-acl": str(data["x_oss_object_acl"]),
            "x-oss-forbid-overwrite": str(data["x_oss_forbid_overwrite"]),
            "success_action_status": "200",
        }
        with source.open("rb") as audio:
            upload_response = self._session.post(
                str(data["upload_host"]),
                data=form,
                files={"file": (source.name, audio, "audio/flac")},
                timeout=max(self._timeout_seconds, 600),
            )
        if upload_response.status_code != 200:
            raise TranscriptionProviderError(
                f"temporary audio upload failed with HTTP {upload_response.status_code}"
            )
        return f"oss://{key}"

    def transcribe(
        self,
        path: Path,
        *,
        model: str,
        diarization: bool,
        language_hints: tuple[str, ...] = ("zh", "en"),
        context_text: str | None = None,
    ) -> CloudResult:
        """Upload, submit, poll, download, sanitize, and parse one chunk."""
        oss_url = self.upload_temporary(path, model=model)
        qwen3_filetrans = model.startswith("qwen3-asr")
        if qwen3_filetrans:
            # Qwen3-ASR uses the current singular file_url contract and does not
            # support diarization. Keep the request minimal so legacy Fun-ASR
            # parameters cannot make a valid task fail after submission.
            parameters: dict[str, Any] = {
                "channel_id": [0],
                "enable_itn": False,
                "enable_words": True,
                "language": language_hints[0] if language_hints else "zh",
            }
            input_payload: dict[str, Any] = {"file_url": oss_url}
        else:
            parameters = {
                "channel_id": [0],
                "diarization_enabled": diarization,
                "language_hints": list(
                    language_hints if model.startswith("qwen-audio") else ("zh",)
                ),
                "special_word_filter": {
                    "filter_with_signed": {"word_list": []},
                    "filter_with_empty": {"word_list": []},
                    "system_reserved_filter": False,
                },
            }
            input_payload = {"file_urls": [oss_url]}
            if context_text and model.startswith("qwen-audio"):
                input_payload["context"] = [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": context_text[:400]}],
                    }
                ]
        submit_response = self._session.post(
            f"{self._base_url}/api/v1/services/audio/asr/transcription",
            headers={
                **self._authorization_headers,
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
                "X-DashScope-OssResourceResolve": "enable",
            },
            json={"model": model, "input": input_payload, "parameters": parameters},
            timeout=self._timeout_seconds,
        )
        submit_payload = _json_object(submit_response)
        output = submit_payload.get("output")
        if submit_response.status_code != 200 or not isinstance(output, dict):
            code = submit_payload.get("code") or submit_response.status_code
            message = submit_payload.get("message") or "submission failed"
            raise TranscriptionProviderError(f"cloud submission failed [{code}]: {message}")
        task_id = str(output.get("task_id", ""))
        if not task_id:
            raise TranscriptionProviderError("cloud submission omitted task_id")
        deadline = time.monotonic() + self._max_wait_seconds
        query_payload: dict[str, Any]
        while True:
            query_response = self._session.get(
                f"{self._base_url}/api/v1/tasks/{task_id}",
                headers=self._authorization_headers,
                timeout=self._timeout_seconds,
            )
            query_payload = _json_object(query_response)
            query_output = query_payload.get("output")
            if query_response.status_code != 200 or not isinstance(query_output, dict):
                raise TranscriptionProviderError(
                    "cloud task query failed: "
                    f"{query_payload.get('code', query_response.status_code)}"
                )
            status = str(query_output.get("task_status", ""))
            if status in {"SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"}:
                break
            if time.monotonic() >= deadline:
                raise TranscriptionProviderError(f"cloud task timed out: {task_id}")
            time.sleep(self._poll_seconds)
        if status != "SUCCEEDED":
            code = str(query_output.get("code", ""))
            message = str(query_output.get("message", ""))
            if code == "ASR_RESPONSE_HAVE_NO_WORDS" or message == "ASR_RESPONSE_HAVE_NO_WORDS":
                try:
                    import soundfile  # type: ignore[import-untyped]

                    audio_info = soundfile.info(str(path))
                    billed_seconds = (
                        audio_info.frames + audio_info.samplerate - 1
                    ) // audio_info.samplerate
                except Exception:  # noqa: BLE001 - soundfile backends raise varied errors
                    billed_seconds = 0
                sanitized = sanitize_external_payload(query_payload)
                return CloudResult(
                    model=model,
                    task_id=task_id,
                    billed_seconds=billed_seconds,
                    sentences=(),
                    sanitized_payload=(
                        cast(dict[str, Any], sanitized) if isinstance(sanitized, dict) else {}
                    ),
                )
            detail = f" [{code}]: {message}" if code or message else ""
            raise TranscriptionProviderError(f"cloud task ended as {status}{detail}: {task_id}")
        if qwen3_filetrans:
            raw_result = query_output.get("result")
            if not isinstance(raw_result, dict):
                raise TranscriptionProviderError("cloud task returned no result object")
            result_info = cast(dict[str, Any], raw_result)
        else:
            results = query_output.get("results")
            if not isinstance(results, list) or not results or not isinstance(results[0], dict):
                raise TranscriptionProviderError("cloud task returned no result object")
            result_info = cast(dict[str, Any], results[0])
            if result_info.get("subtask_status") != "SUCCEEDED":
                raise TranscriptionProviderError(
                    "cloud subtask failed "
                    f"[{result_info.get('code')}]: {result_info.get('message')}"
                )
        result_url = result_info.get("transcription_url")
        if not isinstance(result_url, str) or urlsplit(result_url).scheme != "https":
            raise TranscriptionProviderError("cloud task omitted a valid HTTPS result URL")
        result_response = self._session.get(result_url, timeout=self._timeout_seconds)
        result_payload = _json_object(result_response)
        if result_response.status_code != 200:
            raise TranscriptionProviderError(
                f"cloud result download failed with HTTP {result_response.status_code}"
            )
        sentences: list[ProviderSentence] = []
        transcripts = result_payload.get("transcripts")
        if isinstance(transcripts, list):
            for transcript in transcripts:
                if not isinstance(transcript, dict):
                    continue
                raw_sentences = transcript.get("sentences")
                if not isinstance(raw_sentences, list):
                    continue
                for sentence in raw_sentences:
                    if not isinstance(sentence, dict):
                        continue
                    text = str(sentence.get("text", "")).strip()
                    begin = int(sentence.get("begin_time", 0))
                    end = int(sentence.get("end_time", 0))
                    if text and end >= begin:
                        end = max(begin + 1, end)
                        raw_speaker = sentence.get("speaker_id")
                        sentences.append(
                            ProviderSentence(
                                start_ms=begin,
                                end_ms=end,
                                text=text,
                                speaker_id=(str(raw_speaker) if raw_speaker is not None else None),
                                task_id=task_id,
                            )
                        )
        usage = query_payload.get("usage")
        raw_provider_seconds = (
            usage.get("duration", usage.get("seconds", 0)) if isinstance(usage, dict) else 0
        )
        provider_seconds = int(raw_provider_seconds or 0)
        try:
            import soundfile

            audio_info = soundfile.info(str(path))
            input_seconds = (audio_info.frames + audio_info.samplerate - 1) // audio_info.samplerate
        except Exception:  # noqa: BLE001 - soundfile backends raise varied errors
            input_seconds = provider_seconds
        # The API may report detected-speech duration rather than full uploaded
        # duration. Cost control always uses the more conservative value.
        billed_seconds = max(provider_seconds, input_seconds)
        sanitized = sanitize_external_payload(result_payload)
        if not isinstance(sanitized, dict):
            raise TranscriptionProviderError("sanitized result payload is not an object")
        return CloudResult(
            model=model,
            task_id=task_id,
            billed_seconds=billed_seconds,
            sentences=tuple(sentences),
            sanitized_payload=cast(dict[str, Any], sanitized),
        )


class DashScopeFlashTranscriber:
    """Synchronous <=5 minute ASR client using an inline, private Data URI."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com",
        timeout_seconds: float = 600,
    ) -> None:
        if not api_key:
            raise ValueError("DashScope API key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._session = requests.Session()

    def transcribe(
        self,
        path: Path,
        *,
        model: str,
        context_text: str | None = None,
    ) -> FlashResult:
        """Return text from Fun-ASR-Flash or a compatible short-audio model."""
        source = path.expanduser().resolve()
        raw = source.read_bytes()
        if len(raw) > 7_500_000:
            raise TranscriptionProviderError(
                "short-audio input exceeds the safe 10 MB base64 request limit"
            )
        mime = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
        }.get(source.suffix.casefold(), "audio/flac")
        data_uri = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        qwen3_asr = model.startswith("qwen3-asr")
        messages: list[dict[str, Any]] = []
        if context_text and not qwen3_asr:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": context_text[:400]}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    [{"audio": data_uri}]
                    if qwen3_asr
                    else [{"type": "input_audio", "input_audio": {"data": data_uri}}]
                ),
            }
        )
        parameters: dict[str, Any]
        if qwen3_asr:
            parameters = {
                "asr_options": {
                    "language": "zh",
                    "enable_itn": False,
                }
            }
        else:
            parameters = {"format": source.suffix.lstrip("."), "sample_rate": "16000"}
        response = self._session.post(
            f"{self._base_url}/api/v1/services/aigc/multimodal-generation/generation",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "X-DashScope-SSE": "disable",
            },
            json={
                "model": model,
                "input": {"messages": messages},
                "parameters": parameters,
            },
            timeout=self._timeout_seconds,
        )
        payload = _json_object(response)
        if response.status_code != 200:
            code = payload.get("code") or response.status_code
            message = payload.get("message") or "recognition failed"
            if code == "CLIENT_ERROR" and message == "ASR_RESPONSE_HAVE_NO_WORDS":
                sanitized = sanitize_external_payload(payload)
                return FlashResult(
                    model=model,
                    request_id=str(payload.get("request_id", "")),
                    text="",
                    sanitized_payload=(
                        cast(dict[str, Any], sanitized) if isinstance(sanitized, dict) else {}
                    ),
                )
            raise TranscriptionProviderError(f"short ASR failed [{code}]: {message}")
        output = payload.get("output")
        text = ""
        if isinstance(output, dict):
            text = str(output.get("text", "")).strip()
            nested = output.get("output")
            if not text and isinstance(nested, dict):
                sentence = nested.get("sentence")
                if isinstance(sentence, dict):
                    text = str(sentence.get("text", "")).strip()
            if not text:
                choices = output.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    message = choices[0].get("message")
                    if isinstance(message, dict):
                        content = message.get("content", "")
                        text = str(content).strip()
        sanitized = sanitize_external_payload(payload)
        if not isinstance(sanitized, dict):
            raise TranscriptionProviderError("sanitized short result is not an object")
        return FlashResult(
            model=model,
            request_id=str(payload.get("request_id", "")),
            text=text,
            sanitized_payload=cast(dict[str, Any], sanitized),
        )


class LocalQwenTranscriber:
    """GPU-local Qwen3-ASR provider, loaded lazily."""

    def __init__(self, *, model_directory: Path, batch_size: int = 4) -> None:
        self.model_directory = model_directory.expanduser().resolve()
        self.batch_size = batch_size
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        # qwen-asr imports its optional Japanese forced aligner at package import
        # time. DyNet cannot open nagisa's model through a non-ASCII Windows
        # prefix, while plain ASR does not call that aligner. Avoid initializing
        # the unrelated component; forced alignment remains deliberately disabled.
        if not sys.prefix.isascii() and "nagisa" not in sys.modules:
            stub = types.ModuleType("nagisa")

            def unavailable_tagger(_text: str) -> Any:
                raise RuntimeError("Japanese forced alignment is disabled")

            stub.__dict__["tagging"] = unavailable_tagger
            sys.modules["nagisa"] = stub
        try:
            import torch
            from qwen_asr import Qwen3ASRModel  # type: ignore[import-untyped]
        except ImportError as error:
            raise TranscriptionProviderError(
                "qwen-asr and CUDA PyTorch are required for local transcription"
            ) from error
        if not torch.cuda.is_available():
            raise TranscriptionProviderError("CUDA is not available to local Qwen3-ASR")
        self._model = Qwen3ASRModel.from_pretrained(
            str(self.model_directory),
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
            max_inference_batch_size=self.batch_size,
            max_new_tokens=2048,
        )
        return self._model

    def transcribe(self, clips: list[tuple[Path, int, int]]) -> tuple[ProviderSentence, ...]:
        """Transcribe local clips while retaining their source offsets."""
        if not clips:
            return ()
        model = self._load()
        sentences: list[ProviderSentence] = []
        for offset in range(0, len(clips), self.batch_size):
            batch = clips[offset : offset + self.batch_size]
            try:
                results = model.transcribe(
                    audio=[str(path) for path, _, _ in batch],
                    language=[None] * len(batch),
                )
            except Exception as error:
                raise TranscriptionProviderError("local Qwen3-ASR inference failed") from error
            for result, (_, start_ms, end_ms) in zip(results, batch, strict=True):
                text = str(result.text).strip()
                sentences.append(ProviderSentence(start_ms=start_ms, end_ms=end_ms, text=text))
        return tuple(sentences)

    def unload(self) -> None:
        """Release model references before loading speaker models."""
        self._model = None
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_provider_payload(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist sanitized provider JSON."""
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    temporary.replace(target)
