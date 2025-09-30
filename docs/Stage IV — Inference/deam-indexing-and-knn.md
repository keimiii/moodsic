# DEAM Song-Level Matching (POC)

Default for the academic POC: song-level retrieval using DEAM static
annotations `[1, 9]` with a simple linear-scan k-NN. We pre-scale those
annotations into the shared reference space (`valence_ref`, `arousal_ref`) so
the runtime can operate without additional conversions. Station gating is
implemented via a Gaussian Mixture Model (GMM) over song-level V/A.

- [✅] Keep a dataframe of songs with static `[valence, arousal]` (stored as `valence_ref`, `arousal_ref`) and metadata
- [✅] Linear-scan k-NN at query time; enforce dwell-time and recent-song memory
- [✅] GMM “station” gating from the DEAM clustering notebook
- [ ] Optional: validate shortlist quality on sample queries

Code: `utils/song_matcher.py`

Tests: `tests/test_song_matcher.py`

Extracted from [project_overview.md](file:///Users/desmondchoy/Projects/emo-rec/docs/project_overview.md).

## Reference implementation (GMM gate + linear-scan k-NN)

```python
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class MatchResult:
    song: Optional[pd.Series]
    switch: bool
    timestamp: float

class SongMatcher:
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
        self.top2_threshold = float(top2_threshold)
        self.valence_col = valence_col
        self.arousal_col = arousal_col
        self.cluster_col = cluster_col
        self.clock = clock or time.time

        self._recent: Deque[str | int] | None = deque(maxlen=int(recent_k)) if recent_k > 0 else None
        self._current: Optional[pd.Series] = None
        self._current_start: Optional[float] = None

    def _gate_clusters(self, v_ref: float, a_ref: float) -> Sequence[int]:
        features = np.array([[v_ref, a_ref]], dtype=np.float64)
        probs = self.gmm.predict_proba(self.scaler.transform(features))[0]
        order = np.argsort(probs)[::-1]
        top1 = order[0]
        if probs[top1] < self.top2_threshold and probs.size > 1:
            return order[:2]
        return order[:1]

    def recommend(
        self,
        v_ref: float,
        a_ref: float,
        *,
        now: Optional[float] = None,
        top_k: int = 20,
    ) -> MatchResult:
        timestamp = now if now is not None else self.clock()
        if self._current is not None and self._current_start is not None:
            if (timestamp - self._current_start) < self.min_dwell:
                return MatchResult(song=self._current, switch=False, timestamp=timestamp)

        clusters = self._gate_clusters(v_ref, a_ref)
        candidates = self.songs if not self.cluster_col else self.songs[self.songs[self.cluster_col].isin(clusters)]
        distances = np.linalg.norm(
            candidates[[self.valence_col, self.arousal_col]].to_numpy(dtype=np.float64)
            - np.array([v_ref, a_ref], dtype=np.float64),
            axis=1,
        )

        if top_k and candidates.shape[0] > top_k:
            keep = np.argpartition(distances, top_k - 1)[:top_k]
            candidates = candidates.iloc[keep]
            distances = distances[keep]

        if self._recent is not None and "song_id" in candidates.columns:
            mask = ~candidates["song_id"].isin(self._recent)
            if mask.any():
                candidates = candidates[mask]
                distances = distances[mask.to_numpy()]

        selection = candidates.iloc[int(np.argmin(distances))]
        switched = self._current is None or selection.get("song_id") != getattr(self._current, "song_id", None)
        if switched:
            self._current = selection
            self._current_start = timestamp
            if self._recent is not None:
                self._recent.append(selection.get("song_id"))
        return MatchResult(song=self._current, switch=switched, timestamp=timestamp)
```

## Notes

- Dynamic per-frame annotations exist in `[-10, 10]`, but the POC uses static
  song-level `[1, 9]` and linear-scan distances
- Use `utils.emotion_scale_aligner.EmotionScaleAligner` to convert DEAM static
  values into reference space when curating `songs_df`.
