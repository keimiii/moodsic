# DEAM Song-Level Matching (POC)

Default for the academic POC: song-level retrieval using DEAM static
annotations `[1, 9]` with a simple linear-scan k-NN. Station gating is
implemented via a Gaussian Mixture Model (GMM) over song-level V/A.

- [✅] Keep a dataframe of songs with static `[valence, arousal]` and metadata
- [✅] Linear-scan k-NN at query time; enforce dwell-time and recent-song memory
- [✅] GMM “station” gating from the DEAM clustering notebook
- [ ] Optional: validate shortlist quality on sample queries

Extracted from [project_overview.md](file:///Users/desmondchoy/Projects/emo-rec/docs/project_overview.md).

## Reference implementation (GMM gate + linear-scan k-NN)

```python
import numpy as np
import pandas as pd
from utils.emotion_scale_aligner import EmotionScaleAligner

class SongMatcher:
    def __init__(self, songs_df: pd.DataFrame, gmm, scaler, min_dwell_time: float = 25.0, recent_k: int = 5):
        self.songs = songs_df.reset_index(drop=True)  # columns: song_id, valence, arousal, [gmm_cluster]
        self.gmm = gmm
        self.scaler = scaler
        self.min_dwell = min_dwell_time
        self.recent_songs = []
        self.current = None
        self.current_start = None
        self.aligner = EmotionScaleAligner()

    def _can_switch(self, now: float) -> bool:
        return self.current is None or self.current_start is None or (now - self.current_start) >= self.min_dwell

    def recommend(self, v_ref: float, a_ref: float, now: float, k: int = 20, top2_threshold: float = 0.55):
        if not self._can_switch(now):
            return self.current
        # Gate clusters using GMM in reference space
        post = self.gmm.predict_proba(self.scaler.transform([[v_ref, a_ref]]))[0]
        order = np.argsort(post)[::-1]
        cluster_set = {order[0]} if post[order[0]] >= top2_threshold else {order[0], order[1]}
        cand = self.songs
        if 'gmm_cluster' in cand.columns:
            cand = cand[cand['gmm_cluster'].isin(cluster_set)]
        # Compute distances in DEAM static [1, 9]
        vq, aq = self.aligner.reference_to_deam_static(v_ref, a_ref)
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

## Notes

- Dynamic per-frame annotations exist in `[-10, 10]`, but the POC uses static
  song-level `[1, 9]` and linear-scan distances
