"""Song-level matcher with GMM station gating, dwell time, and recent memory."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Iterable, Optional, Sequence

import joblib
import numpy as np
import pandas as pd


@dataclass
class MatchResult:
    """Container describing the currently selected song."""

    song: Optional[pd.Series]
    switch: bool
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "song": None if self.song is None else self.song.to_dict(),
            "switch": self.switch,
            "timestamp": self.timestamp,
        }


class SongMatcher:
    """Retrieve songs using GMM gating, dwell enforcement, and repeat avoidance."""

    def __init__(
        self,
        songs_df: pd.DataFrame,
        scaler,
        gmm,
        *,
        min_dwell_time: float = 25.0,
        recent_k: int = 5,
        top2_threshold: float = 0.55,
        valence_col: str = "valence_ref",
        arousal_col: str = "arousal_ref",
        cluster_col: Optional[str] = "cluster",
        clock: Callable[[], float] | None = None,
    ) -> None:
        if songs_df.empty:
            raise ValueError("songs_df must contain at least one entry")

        self.songs = songs_df.reset_index(drop=True)
        self.scaler = scaler
        self.gmm = gmm
        self.min_dwell = float(min_dwell_time)
        self.recent_maxlen = int(recent_k)
        self.top2_threshold = float(top2_threshold)
        self.valence_col = valence_col
        self.arousal_col = arousal_col
        self.cluster_col = cluster_col
        self.clock = clock or time.time

        if "song_id" not in self.songs.columns:
            raise ValueError("songs_df must include a 'song_id' column")

        self._recent: Optional[Deque[str | int]]
        self._recent = deque(maxlen=self.recent_maxlen) if self.recent_maxlen > 0 else None
        self._current: Optional[pd.Series] = None
        self._current_start: Optional[float] = None

        missing = [c for c in (valence_col, arousal_col) if c not in self.songs.columns]
        if missing:
            raise ValueError(f"songs_df missing required columns: {missing}")

        if cluster_col and cluster_col not in self.songs.columns:
            raise ValueError(
                f"songs_df missing cluster column '{cluster_col}' required for gating"
            )

    # ------------------------------------------------------------------
    @classmethod
    def from_artifacts(
        cls,
        songs_df: pd.DataFrame,
        artifacts_dir: Path | str,
        *,
        scaler_name: str = "scaler.pkl",
        gmm_name: str = "gmm.pkl",
        **kwargs,
    ) -> "SongMatcher":
        artifacts_path = Path(artifacts_dir)
        scaler = joblib.load(artifacts_path / scaler_name)
        gmm = joblib.load(artifacts_path / gmm_name)
        return cls(songs_df, scaler, gmm, **kwargs)

    # ------------------------------------------------------------------
    def _gate_clusters(self, v_ref: float, a_ref: float) -> Sequence[int]:
        features = np.array([[v_ref, a_ref]], dtype=np.float64)
        X = self.scaler.transform(features)
        posteriors = self.gmm.predict_proba(X)[0]
        order = np.argsort(posteriors)[::-1]
        top1 = order[0]
        if posteriors[top1] < self.top2_threshold and posteriors.size > 1:
            return order[:2]
        return order[:1]

    def _eligible_candidates(self, clusters: Iterable[int]) -> pd.DataFrame:
        if not self.cluster_col:
            return self.songs
        clusters = tuple(clusters)
        if not clusters:
            return self.songs
        subset = self.songs[self.songs[self.cluster_col].isin(clusters)]
        return subset if not subset.empty else self.songs

    def _compute_distances(self, df: pd.DataFrame, v_ref: float, a_ref: float) -> np.ndarray:
        xy = df[[self.valence_col, self.arousal_col]].to_numpy(dtype=np.float64)
        targets = np.array([v_ref, a_ref], dtype=np.float64)
        return np.linalg.norm(xy - targets, axis=1)

    def _can_switch(self, now: float) -> bool:
        if self._current is None or self._current_start is None:
            return True
        return (now - self._current_start) >= self.min_dwell

    # ------------------------------------------------------------------
    def recommend(
        self,
        v_ref: float,
        a_ref: float,
        *,
        now: Optional[float] = None,
        top_k: int = 20,
    ) -> MatchResult:
        now = now if now is not None else self.clock()

        if not self._can_switch(now):
            return MatchResult(song=self._current, switch=False, timestamp=now)  # type: ignore[arg-type]

        clusters = self._gate_clusters(v_ref, a_ref)
        candidates = self._eligible_candidates(clusters)
        distances = self._compute_distances(candidates, v_ref, a_ref)

        # Narrow to top_k by distance
        if top_k is not None and top_k > 0 and candidates.shape[0] > top_k:
            idx = np.argpartition(distances, top_k - 1)[:top_k]
            subset = candidates.iloc[idx]
            d_subset = distances[idx]
        else:
            subset = candidates
            d_subset = distances

        selection = self._select_with_recent(subset, d_subset)
        switched = self._current is None or selection.get("song_id") != getattr(self._current, "song_id", None)
        if switched:
            self._current = selection
            self._current_start = now
            song_id = selection.get("song_id")
            if song_id is not None and self._recent is not None:
                self._recent.append(song_id)
        return MatchResult(song=self._current, switch=switched, timestamp=now)  # type: ignore[arg-type]

    def _select_with_recent(self, df: pd.DataFrame, distances: np.ndarray) -> pd.Series:
        if self._recent is not None and "song_id" in df.columns:
            mask = ~df["song_id"].isin(self._recent)
            if mask.any():
                df = df[mask]
                distances = distances[mask.to_numpy()]
        if df.empty:
            raise RuntimeError("No candidates after applying recent-song filter")
        best_idx = int(np.argmin(distances))
        return df.iloc[best_idx]

    # ------------------------------------------------------------------
    @property
    def current_song(self) -> Optional[pd.Series]:
        return self._current

    @property
    def recent_history(self) -> Sequence[str | int]:
        if self._recent is None:
            return tuple()
        return tuple(self._recent)

    def reset(self) -> None:
        self._current = None
        self._current_start = None
        if self._recent is not None:
            self._recent.clear()


__all__ = ["SongMatcher", "MatchResult"]
