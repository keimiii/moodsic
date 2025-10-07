"""Cluster-aware song lookup consistent with MATCH design docs.

This module mirrors the behaviour described in ``docs/Stage IV — Inference``:

* Inputs are valence/arousal in the reference space ``[-1, 1]``.
* We reuse the trained ``StandardScaler`` + ``GaussianMixture`` artefacts to
  gate candidate songs (top-1 cluster, widen to top-2 when the leading
  posterior drops below the configured threshold).
* Songs are ranked by Euclidean distance in reference space and returned with
  their full metadata for downstream consumption (e.g., notebooks or APIs).

The goal is to provide a lightweight helper that can be reused outside the
original notebooks when only valence/arousal coordinates are available.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np


@dataclass
class SongCandidate:
    """Song metadata paired with its distance to the target VA."""

    data: Dict[str, Any]
    distance: float


@dataclass
class ClusterMatchResult:
    """Full result bundle from :meth:`ClusteredSongMatcher.recommend`."""

    target_valence: float
    target_arousal: float
    gated_clusters: List[int]
    cluster_posteriors: Dict[int, float]
    best_song: SongCandidate
    top_songs: List[SongCandidate]
    total_candidates: int


class ClusteredSongMatcher:
    """Reusable matcher that honours the retrieval policies in the docs."""

    def __init__(
        self,
        songs: Sequence[Dict[str, Any]],
        scaler,
        gmm,
        *,
        valence_field: str = "valence_ref",
        arousal_field: str = "arousal_ref",
        cluster_field: str = "cluster",
        top2_threshold: float = 0.55,
    ) -> None:
        if not songs:
            raise ValueError("songs must contain at least one entry")

        self.valence_field = valence_field
        self.arousal_field = arousal_field
        self.cluster_field = cluster_field
        self.top2_threshold = float(top2_threshold)

        self.songs: List[Dict[str, Any]] = []
        for song in songs:
            normalised = dict(song)
            normalised[self.valence_field] = _to_float(
                song.get(self.valence_field), f"{self.valence_field}"
            )
            normalised[self.arousal_field] = _to_float(
                song.get(self.arousal_field), f"{self.arousal_field}"
            )
            cluster_raw = song.get(self.cluster_field)
            if cluster_raw in (None, ""):
                raise ValueError(f"Song missing cluster id in '{self.cluster_field}': {song}")
            try:
                normalised[self.cluster_field] = int(cluster_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid cluster id '{cluster_raw}' in '{self.cluster_field}'"
                ) from exc
            self.songs.append(normalised)

        self.scaler = scaler
        self.gmm = gmm

        clusters = {song[self.cluster_field] for song in self.songs}
        if not clusters:
            raise ValueError("No clusters detected in provided songs")

        self._centroids: Dict[int, Tuple[float, float]] = {}
        for cluster_id in clusters:
            members = [s for s in self.songs if s[self.cluster_field] == cluster_id]
            if not members:
                continue
            avg_v = sum(m[self.valence_field] for m in members) / len(members)
            avg_a = sum(m[self.arousal_field] for m in members) / len(members)
            self._centroids[cluster_id] = (avg_v, avg_a)

    @classmethod
    def from_artifacts(
        cls,
        songs_csv: Path | str,
        artifacts_dir: Path | str,
        *,
        scaler_name: str = "scaler.pkl",
        gmm_name: str = "gmm.pkl",
        valence_field: str = "valence_ref",
        arousal_field: str = "arousal_ref",
        cluster_field: str = "cluster",
        top2_threshold: float = 0.55,
    ) -> "ClusteredSongMatcher":
        """Instantiate from DEAM artefacts on disk."""

        songs = _load_csv_as_dicts(songs_csv)
        artifacts_path = Path(artifacts_dir)
        scaler = joblib.load(artifacts_path / scaler_name)
        gmm = joblib.load(artifacts_path / gmm_name)
        return cls(
            songs,
            scaler,
            gmm,
            valence_field=valence_field,
            arousal_field=arousal_field,
            cluster_field=cluster_field,
            top2_threshold=top2_threshold,
        )

    def recommend(
        self,
        valence: float,
        arousal: float,
        *,
        top_k: Optional[int] = 5,
    ) -> ClusterMatchResult:
        """Return the closest cluster and songs for ``(valence, arousal)``."""

        v = _to_float(valence, "valence")
        a = _to_float(arousal, "arousal")

        gated_clusters, cluster_posteriors = self._gate_clusters(v, a)
        candidates = [
            song
            for song in self.songs
            if song[self.cluster_field] in gated_clusters
        ]

        if not candidates:
            raise RuntimeError("No songs available after cluster gating")

        ranked = self._rank_by_distance(candidates, v, a)
        limit = len(ranked) if not top_k or top_k <= 0 else min(top_k, len(ranked))
        top_songs = ranked[:limit]

        return ClusterMatchResult(
            target_valence=v,
            target_arousal=a,
            gated_clusters=gated_clusters,
            cluster_posteriors=cluster_posteriors,
            best_song=top_songs[0],
            top_songs=top_songs,
            total_candidates=len(candidates),
        )

    # ------------------------------------------------------------------
    def _gate_clusters(
        self,
        valence: float,
        arousal: float,
    ) -> Tuple[List[int], Dict[int, float]]:
        features = np.array([[valence, arousal]], dtype=np.float64)
        transformed = self.scaler.transform(features)
        posteriors = self.gmm.predict_proba(transformed)[0]
        order = np.argsort(posteriors)[::-1]

        gated = [int(order[0])]
        if posteriors[order[0]] < self.top2_threshold and order.size > 1:
            gated.append(int(order[1]))

        posterior_map = {int(idx): float(posteriors[idx]) for idx in order}
        return gated, posterior_map

    def _rank_by_distance(
        self,
        songs: Iterable[Dict[str, Any]],
        valence: float,
        arousal: float,
    ) -> List[SongCandidate]:
        ranked: List[SongCandidate] = []
        for song in songs:
            v = song[self.valence_field]
            a = song[self.arousal_field]
            dist = math.hypot(valence - v, arousal - a)
            ranked.append(SongCandidate(data=dict(song), distance=dist))
        ranked.sort(key=lambda candidate: candidate.distance)
        return ranked

    @property
    def centroids(self) -> Dict[int, Tuple[float, float]]:
        return dict(self._centroids)


def _load_csv_as_dicts(csv_path: Path | str) -> List[Dict[str, Any]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader, start=1):
            if not row:
                continue
            rows.append(dict(row))

    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows


def _to_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Value for {label!r} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Value for {label!r} must be finite, got {value!r}")
    return number


__all__ = [
    "ClusteredSongMatcher",
    "ClusterMatchResult",
    "SongCandidate",
]
