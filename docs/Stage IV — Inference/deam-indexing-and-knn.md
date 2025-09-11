# DEAM Song-Level Matching (POC)

Default for the academic POC: song-level retrieval using DEAM static
annotations `[1, 9]` with a simple linear-scan k-NN. KD-Tree and dynamic
segmentation remain optional for larger catalogs.

- [✅] Keep a dataframe of songs with static `[valence, arousal]` and metadata
- [✅] Linear-scan k-NN at query time; enforce dwell-time and recent-song memory
- [✅] Optional GMM “station” gating from the DEAM clustering notebook
- [ ] Optional: validate shortlist quality on sample queries

Extracted from [project_overview.md](file:///Users/desmondchoy/Projects/emo-rec/docs/project_overview.md).

## Reference implementation (linear-scan k-NN)

```python
import numpy as np
import pandas as pd
from utils.emotion_scale_aligner import EmotionScaleAligner

class SongMatcher:
    def __init__(self, songs_df: pd.DataFrame, min_dwell_time: float = 25.0, recent_k: int = 5):
        self.songs = songs_df.reset_index(drop=True)  # columns: song_id, valence, arousal, [cluster]
        self.min_dwell = min_dwell_time
        self.recent_songs = []
        self.current = None
        self.current_start = None
        self.aligner = EmotionScaleAligner()

    def _can_switch(self, now: float) -> bool:
        return self.current is None or self.current_start is None or (now - self.current_start) >= self.min_dwell

    def recommend(self, v_ref: float, a_ref: float, now: float, k: int = 20, cluster_id: int | None = None):
        if not self._can_switch(now):
            return self.current
        vq, aq = self.aligner.reference_to_deam_static(v_ref, a_ref)
        cand = self.songs
        if cluster_id is not None and 'cluster' in cand.columns:
            cand = cand[cand['cluster'] == cluster_id]
        xy = cand[['valence', 'arousal']].to_numpy()
        d = np.linalg.norm(xy - np.array([vq, aq]), axis=1)
        top_idx = np.argpartition(d, min(k, len(d)-1))[:k]
        top = cand.iloc[top_idx]
        # Prefer songs not in recent list
        top = top[~top['song_id'].isin(self.recent_songs)] or top
        best = top.iloc[np.argmin(d[top_idx])]
        if self.current is None or best['song_id'] != self.current['song_id']:
            self.current = best
            self.current_start = now
            self.recent_songs.append(best['song_id'])
            self.recent_songs = self.recent_songs[-5:]
        return self.current
```

## Optional: Dynamic Segmentation + KD-Tree

- Segment songs into 10s windows (50% overlap) using dynamic annotations `[-10, 10]`.
- Build a KD-Tree over segment means for fast k-NN when scaling beyond song-level.
- Maintain segment metadata: `song_id`, `start_time`, `end_time`, `valence`, `arousal`.
