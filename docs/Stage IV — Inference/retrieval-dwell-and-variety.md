# Retrieval: Dwell and Variety

- [ ] Query k=20 nearest segments per stabilized frame
- [ ] Enforce minimum dwell time (20–30s; example default 25s)
- [ ] Maintain recent-song memory to avoid repetition
- [ ] Log switching events for offline analysis

Extracted from [project_overview.md](file:///Users/desmondchoy/Projects/emo-rec/docs/project_overview.md).

## Policy

- k-NN over KD-Tree using stabilized V-A as query (after FE→DEAM scaling).
- Choose the nearest candidate not in recent memory; allow repeat if all blocked.
- Only switch when minimum dwell time has elapsed.

## Reference implementations

```python
class SegmentMatcher:
    def __init__(self, segments_df: pd.DataFrame, min_dwell_time: float = 25.0, recent_k: int = 5):
        self.segments = segments_df.reset_index(drop=True)
        self.kd_tree = KDTree(self.segments[["valence", "arousal"]].values)
        self.min_dwell = min_dwell_time
        self.recent_songs = deque(maxlen=recent_k)
        self.current = None
        self.current_start = None
```

```python
class SegmentLevelMusicMatcher:
    def __init__(self, deam_processor):
        self.segments = deam_processor.segments_metadata
        self.kd_tree = deam_processor.kd_tree
        self.recent_songs = deque(maxlen=10)
        self.current_segment = None
        self.segment_start_time = None
        self.min_dwell_time = 20  # seconds
```
