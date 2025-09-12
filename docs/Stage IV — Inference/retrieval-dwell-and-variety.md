# Retrieval: Dwell and Variety

- [ ] Gate by GMM station (top-1; widen to top-2 if unsure)
- [ ] Enforce minimum dwell time (20–30s; example default 25s)
- [ ] Maintain recent-song memory to avoid repetition
- [ ] Log switching events for offline analysis

Extracted from [project_overview.md](file:///Users/desmondchoy/Projects/emo-rec/docs/project_overview.md).

## Policy

- Compute stabilized V/A in reference space; scale to DEAM static `[1, 9]` for distance.
- GMM station gate using `predict_proba`; if top posterior < 0.55, widen to top-2 clusters.
- Within selected cluster set, linear-scan k-NN over songs by Euclidean distance.
- Choose the nearest candidate not in recent memory; allow repeat if all blocked.
- Only switch when minimum dwell time has elapsed.

## Reference implementations

```python
class SongMatcher:
    def __init__(self, songs_df: pd.DataFrame, gmm, scaler, min_dwell_time: float = 25.0, recent_k: int = 5):
        self.songs = songs_df.reset_index(drop=True)  # columns: song_id, valence, arousal, gmm_cluster (optional)
        self.gmm = gmm
        self.scaler = scaler
        self.min_dwell = min_dwell_time
        self.recent_songs = deque(maxlen=recent_k)
        self.current = None
        self.current_start = None

    def _gate_clusters(self, v_ref: float, a_ref: float, top2_threshold: float = 0.55):
        X = self.scaler.transform([[v_ref, a_ref]])
        post = self.gmm.predict_proba(X)[0]
        order = np.argsort(post)[::-1]
        top1, top2 = order[0], order[1]
        if post[top1] < top2_threshold:
            return {top1, top2}
        return {top1}
```

```python
class SongLevelMusicMatcher:
    def __init__(self, deam_songs_df, gmm, scaler):
        self.songs = deam_songs_df
        self.gmm = gmm
        self.scaler = scaler
        self.recent_songs = deque(maxlen=10)
        self.current_song = None
        self.song_start_time = None
        self.min_dwell_time = 20  # seconds
```
