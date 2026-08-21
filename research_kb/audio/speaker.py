"""Anonymous cross-recording speaker-memory implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from research_kb.audio.domain import SpeakerProfile


class SpeakerResolutionError(RuntimeError):
    """Raised when the local speaker model cannot provide embeddings."""


@dataclass(frozen=True)
class LocalSpeakerSamples:
    """Representative clips for one provider-local speaker ID."""

    key: str
    recording_id: str
    first_order: tuple[int, int]
    clips: tuple[Path, ...]


@dataclass
class _Cluster:
    label: str
    members: list[LocalSpeakerSamples]
    embeddings: list[np.ndarray[Any, np.dtype[np.float32]]]

    @property
    def centroid(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        centroid = np.mean(np.stack(self.embeddings), axis=0)
        norm = float(np.linalg.norm(centroid))
        return cast(
            np.ndarray[Any, np.dtype[np.float32]],
            centroid if norm == 0 else centroid / norm,
        )


def _similarity(
    left: np.ndarray[Any, np.dtype[np.float32]],
    right: np.ndarray[Any, np.dtype[np.float32]],
) -> float:
    return float(np.dot(left, right))


def _minimum_similarity(embeddings: list[np.ndarray[Any, np.dtype[np.float32]]]) -> float:
    if len(embeddings) < 2:
        return 1.0
    return min(
        _similarity(left, right)
        for index, left in enumerate(embeddings)
        for right in embeddings[index + 1 :]
    )


def _event_range(item: LocalSpeakerSamples) -> dict[str, object]:
    parts = item.key.split("|")
    try:
        start_ms, end_ms = int(parts[-2]), int(parts[-1])
    except (IndexError, ValueError):
        start_ms, end_ms = item.first_order[1], item.first_order[1]
    return {
        "recording_id": item.recording_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "window_count": len(item.clips),
    }


def anonymous_speaker_label(index: int) -> str:
    """Return a stable spreadsheet-style anonymous speaker label."""
    letters = ""
    value = index
    while True:
        value, remainder = divmod(value, 26)
        letters = chr(ord("A") + remainder) + letters
        if value == 0:
            break
        value -= 1
    return f"说话人 {letters}"


class SpeakerEmbedder:
    """ERes2NetV2 embeddings loaded only after Qwen ASR is released."""

    def __init__(self, *, model_directory: Path) -> None:
        self.model_directory = model_directory.expanduser().resolve()
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from funasr import AutoModel  # type: ignore[import-untyped]
        except ImportError as error:
            raise SpeakerResolutionError("funasr is required for speaker embeddings") from error
        try:
            self._model = AutoModel(
                model="ERes2NetV2",
                model_path=str(self.model_directory),
                device="cuda:0",
                disable_update=True,
            )
        except Exception as error:
            raise SpeakerResolutionError("ERes2NetV2 speaker model failed to load") from error
        return self._model

    def embed(self, clips: tuple[Path, ...]) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Return a robust normalized centroid for multiple clean excerpts."""
        model = self._load()
        embeddings: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        for clip in clips:
            try:
                result = model.generate(input=str(clip))
                raw = result[0]["spk_embedding"]
                if hasattr(raw, "detach"):
                    raw = raw.detach().cpu().numpy()
                vector = np.asarray(raw, dtype=np.float32).reshape(-1)
            except Exception as error:
                raise SpeakerResolutionError(f"speaker embedding failed: {clip.name}") from error
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                embeddings.append(vector / norm)
        if not embeddings:
            raise SpeakerResolutionError("speaker model returned no usable embeddings")
        centroid = np.median(np.stack(embeddings), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(centroid))
        return cast(
            np.ndarray[Any, np.dtype[np.float32]],
            centroid if norm == 0 else centroid / norm,
        )


def resolve_speakers(
    samples: tuple[LocalSpeakerSamples, ...],
    *,
    embedder: SpeakerEmbedder,
    threshold: float = 0.65,
) -> tuple[dict[str, str], tuple[SpeakerProfile, ...]]:
    """Greedily cluster acoustic identities, preferring false splits to false merges."""
    clusters: list[_Cluster] = []
    mapping: dict[str, str] = {}
    for item in sorted(samples, key=lambda sample: sample.first_order):
        embedding = embedder.embed(item.clips)
        best: _Cluster | None = None
        best_similarity = -1.0
        for cluster in clusters:
            similarity = float(np.dot(embedding, cluster.centroid))
            if similarity > best_similarity:
                best_similarity = similarity
                best = cluster
        # Cross-day merges require at least two representative clips. This prevents
        # a single distant/noisy utterance from collapsing two people into one identity.
        enough_evidence = len(item.clips) >= 2
        if best is None or best_similarity < threshold or not enough_evidence:
            best = _Cluster(
                label=anonymous_speaker_label(len(clusters)),
                members=[],
                embeddings=[],
            )
            clusters.append(best)
        best.members.append(item)
        best.embeddings.append(embedding)
        mapping[item.key] = best.label
    profiles = tuple(
        SpeakerProfile(
            speaker=cluster.label,
            local_speaker_keys=tuple(member.key for member in cluster.members),
            recording_ids=tuple(sorted({member.recording_id for member in cluster.members})),
            sample_count=sum(len(member.clips) for member in cluster.members),
            confidence=(
                0.95 if len({member.recording_id for member in cluster.members}) > 1 else 0.8
            ),
        )
        for cluster in clusters
    )
    return mapping, profiles


def resolve_local_segments(
    samples: tuple[LocalSpeakerSamples, ...],
    *,
    embedder: SpeakerEmbedder,
    embedding_cache_path: Path | None = None,
    within_recording_distance: float = 0.20,
    cross_recording_threshold: float = 0.84,
    minimum_identity_samples: int = 2,
    cache_metadata: dict[str, object] | None = None,
    diagnostics: list[dict[str, object]] | None = None,
) -> tuple[dict[str, str], tuple[SpeakerProfile, ...]]:
    """Cluster locally segmented speech without using provider speaker IDs.

    Each item is one independent VAD event. Its clips are windows from that
    event, rather than ASR sentence fragments. Complete linkage prevents a
    chain of merely similar events from collapsing two people.

    ``diagnostics`` is optional so existing callers can keep the historical
    two-value return while the pipeline persists the detailed audit evidence.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering  # type: ignore[import-untyped]
    except ImportError as error:
        raise SpeakerResolutionError("scikit-learn is required for local diarization") from error
    ordered = sorted(samples, key=lambda sample: sample.first_order)
    cached_embeddings: dict[str, list[float]] = {}
    if embedding_cache_path is not None and embedding_cache_path.is_file():
        payload = json.loads(embedding_cache_path.read_text(encoding="utf-8"))
        if cache_metadata is None or payload.get("metadata") == cache_metadata:
            cached_embeddings = {
                str(key): [float(value) for value in values]
                for key, values in payload.get("embeddings", {}).items()
            }
    embeddings: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
    window_embeddings: dict[str, list[np.ndarray[Any, np.dtype[np.float32]]]] = {}
    window_cache_keys: dict[str, list[str]] = {}
    for item in ordered:
        vectors: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        for index, clip in enumerate(item.clips):
            cache_key = f"{item.key}|{clip.stem or index}"
            cached = cached_embeddings.get(cache_key) or (
                cached_embeddings.get(item.key) if len(item.clips) == 1 else None
            )
            vector = (
                np.asarray(cached, dtype=np.float32)
                if cached is not None
                else embedder.embed((clip,))
            )
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vectors.append(vector / norm)
        if not vectors:
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "event_ranges": [_event_range(item)],
                        "window_count": len(item.clips),
                        "intra_cluster_min_similarity": None,
                        "nearest_external_similarity": None,
                        "similarity_margin": None,
                        "accepted": False,
                        "reason": "no_usable_window_embedding",
                    }
                )
            continue
        window_embeddings[item.key] = vectors
        window_cache_keys[item.key] = [
            f"{item.key}|{clip.stem or index}" for index, clip in enumerate(item.clips)
        ]
        if len(vectors) > 1 and _minimum_similarity(vectors) < 0.70:
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "event_ranges": [_event_range(item)],
                        "window_count": len(vectors),
                        "intra_cluster_min_similarity": round(_minimum_similarity(vectors), 6),
                        "nearest_external_similarity": None,
                        "similarity_margin": None,
                        "accepted": False,
                        "reason": "mixed_event_window_similarity_below_0.70",
                    }
                )
            continue
        center = np.median(np.stack(vectors), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(center))
        embeddings[item.key] = center if norm == 0 else center / norm
    if embedding_cache_path is not None:
        embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = embedding_cache_path.with_suffix(embedding_cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "format": "window-v1",
                    "metadata": cache_metadata or {},
                    "embeddings": {
                        cache_key: value.tolist()
                        for key, values in window_embeddings.items()
                        for cache_key, value in zip(window_cache_keys[key], values, strict=True)
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        temporary.replace(embedding_cache_path)
    valid_items = [item for item in ordered if item.key in embeddings]
    day_clusters: list[_Cluster] = []
    for recording_id in dict.fromkeys(item.recording_id for item in ordered):
        recording_samples = [item for item in valid_items if item.recording_id == recording_id]
        if not recording_samples:
            continue
        matrix = np.stack([embeddings[item.key] for item in recording_samples])
        if len(recording_samples) == 1:
            labels = np.zeros(1, dtype=np.int64)
        else:
            labels = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="complete",
                distance_threshold=within_recording_distance,
            ).fit_predict(matrix)
        provisional = [
            _Cluster(
                label="",
                members=[
                    item
                    for item, item_label in zip(recording_samples, labels, strict=True)
                    if int(item_label) == int(label)
                ],
                embeddings=[
                    embeddings[item.key]
                    for item, item_label in zip(recording_samples, labels, strict=True)
                    if int(item_label) == int(label)
                ],
            )
            for label in sorted(set(int(value) for value in labels))
        ]
        # Do not attach singleton clusters by a second, looser rule: they are
        # useful diagnostics but are not identity evidence.
        day_clusters.extend(provisional)

    global_clusters: list[_Cluster] = []
    for day_cluster in sorted(
        day_clusters, key=lambda cluster: min(member.first_order for member in cluster.members)
    ):
        embedding = day_cluster.centroid
        candidates = [
            cluster
            for cluster in global_clusters
            if day_cluster.members[0].recording_id
            not in {member.recording_id for member in cluster.members}
        ]
        global_best = max(
            candidates,
            key=lambda cluster: float(np.dot(embedding, cluster.centroid)),
            default=None,
        )
        similarity = (
            _similarity(embedding, global_best.centroid) if global_best is not None else -1.0
        )
        cross_pairs = (
            [
                _similarity(embedding_item, existing_item)
                for embedding_item in day_cluster.embeddings
                for existing_item in global_best.embeddings
            ]
            if global_best is not None
            else []
        )
        enough_evidence = (
            len(day_cluster.members) >= 2
            and global_best is not None
            and len(global_best.members) >= 2
            and similarity >= cross_recording_threshold
            and min(cross_pairs, default=-1.0) >= 0.80
        )
        target = global_best
        if not enough_evidence:
            target = _Cluster(
                label=anonymous_speaker_label(len(global_clusters)),
                members=[],
                embeddings=[],
            )
            global_clusters.append(target)
        assert target is not None
        target.members.extend(day_cluster.members)
        target.embeddings.extend(day_cluster.embeddings)

    retained_clusters = []
    for cluster in global_clusters:
        internal = _minimum_similarity(cluster.embeddings)
        external = max(
            (
                _similarity(embedding, other_embedding)
                for other in global_clusters
                if other is not cluster and len(other.members) >= minimum_identity_samples
                for embedding in cluster.embeddings
                for other_embedding in other.embeddings
            ),
            default=None,
        )
        margin = internal - external if external is not None else None
        accepted = (
            len(cluster.members) >= minimum_identity_samples
            and internal >= 0.80
            and (margin is None or margin >= 0.12)
        )
        if len(cluster.members) < minimum_identity_samples:
            reason = "single_event_insufficient_evidence"
        elif internal < 0.80:
            reason = "intra_cluster_similarity_below_0.80"
        elif margin is not None and margin < 0.12:
            reason = "similarity_margin_below_0.12"
        else:
            reason = "accepted"
        if diagnostics is not None:
            diagnostics.append(
                {
                    "event_ranges": [_event_range(member) for member in cluster.members],
                    "window_count": sum(len(member.clips) for member in cluster.members),
                    "intra_cluster_min_similarity": round(internal, 6),
                    "nearest_external_similarity": (
                        round(external, 6) if external is not None else None
                    ),
                    "similarity_margin": round(margin, 6) if margin is not None else None,
                    "accepted": accepted,
                    "reason": reason,
                }
            )
        if accepted:
            retained_clusters.append(cluster)
    for index, cluster in enumerate(retained_clusters):
        cluster.label = anonymous_speaker_label(index)
    mapping = {
        member.key: cluster.label for cluster in retained_clusters for member in cluster.members
    }
    profiles = tuple(
        SpeakerProfile(
            speaker=cluster.label,
            local_speaker_keys=tuple(member.key for member in cluster.members),
            recording_ids=tuple(sorted({member.recording_id for member in cluster.members})),
            sample_count=sum(len(member.clips) for member in cluster.members),
            confidence=_minimum_similarity(cluster.embeddings),
        )
        for cluster in retained_clusters
    )
    return mapping, profiles
