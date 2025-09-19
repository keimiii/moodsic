# Retrieval: Dwell and Variety

- [✅] Gate by GMM station (top-1; widen to top-2 if unsure) — `utils/song_matcher.py:97-105`, `tests/test_song_matcher.py:34-64`
- [✅] Enforce minimum dwell time (20–30s; example default 25s) — `utils/song_matcher.py:121-161`, `tests/test_song_matcher.py:67-84`
- [✅] Maintain recent-song memory to avoid repetition — `utils/song_matcher.py:163-172`, `tests/test_song_matcher.py:86-102`
- [ ] Log switching events for offline analysis

Extracted from [project_overview.md](file:///Users/desmondchoy/Projects/emo-rec/docs/project_overview.md).

## Policy

- Compute stabilized V/A in reference space; scale to DEAM static `[1, 9]` for distance.
- GMM station gate using `predict_proba`; if top posterior < 0.55, widen to top-2 clusters.
- Within selected cluster set, linear-scan k-NN over songs by Euclidean distance.
- Choose the nearest candidate not in recent memory; allow repeat if all blocked.
- Only switch when minimum dwell time has elapsed.

## Reference implementation

```python
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
    ) -> None:
        self.songs = songs_df.reset_index(drop=True)
        self.scaler = scaler
        self.gmm = gmm
        self.min_dwell = float(min_dwell_time)
        self._recent = deque(maxlen=int(recent_k))
        self._current = None
        self._current_start = None
        self.top2_threshold = float(top2_threshold)

    def _gate_clusters(self, v_ref: float, a_ref: float) -> Sequence[int]:
        X = self.scaler.transform([[v_ref, a_ref]])
        probs = self.gmm.predict_proba(X)[0]
        order = np.argsort(probs)[::-1]
        top1 = order[0]
        if probs[top1] < self.top2_threshold and probs.size > 1:
            return order[:2]
        return order[:1]

    def recommend(self, v_ref: float, a_ref: float, *, now: Optional[float] = None) -> MatchResult:
        timestamp = now if now is not None else time.time()
        if self._current is not None and self._current_start is not None:
            if (timestamp - self._current_start) < self.min_dwell:
                return MatchResult(song=self._current, switch=False, timestamp=timestamp)

        clusters = self._gate_clusters(v_ref, a_ref)
        candidates = self.songs[self.songs["cluster"].isin(clusters)]
        distances = np.linalg.norm(
            candidates[["valence_ref", "arousal_ref"]].to_numpy(dtype=float)
            - np.array([v_ref, a_ref], dtype=float),
            axis=1,
        )

        mask = ~candidates["song_id"].isin(self._recent)
        if mask.any():
            candidates = candidates[mask]
            distances = distances[mask.to_numpy()]

        selection = candidates.iloc[int(np.argmin(distances))]
        switched = self._current is None or selection["song_id"] != self._current.get("song_id")
        if switched:
            self._current = selection
            self._current_start = timestamp
            self._recent.append(selection["song_id"])
        return MatchResult(song=self._current, switch=switched, timestamp=timestamp)
```
