# DEAM Indexing and kNN

- [ ] Segment DEAM into 10s windows with 50% overlap (2 Hz annotations)
- [ ] Build KD-Tree over segment means in V-A space
- [ ] Persist `segments_metadata` and KD-Tree for reuse
- [ ] Validate index quality on sample queries

Extracted from [project_overview.md](file:///Users/desmondchoy/Projects/emo-rec/docs/project_overview.md).

## Segmentation

- Window size: 10 seconds
- Overlap: 50% (step = window × 0.5)
- Labels: mean valence and arousal per window (DEAM dynamic annotations in [-10, 10])

## Indexing

- Build KD-Tree on `[[valence, arousal], ...]` for fast k-NN.
- Store alongside a dataframe with: `song_id`, `start_time`, `end_time`, `valence`, `arousal`.

## Reference implementation

```python
class DEAMSegmentProcessor:
    def __init__(self, deam_path, window_size=10, overlap=0.5):
        self.deam_path = deam_path
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = 2  # Hz

    def process_dataset(self):
        annotations = pd.read_csv(f"{self.deam_path}/annotations_dynamic.csv")
        segments = []
        for song_id, song_data in annotations.groupby('song_id'):
            segments.extend(self._extract_segments(song_id, song_data))
        segments_df = pd.DataFrame(segments)
        self.kd_tree = KDTree(segments_df[['valence', 'arousal']].values)
        self.segments_metadata = segments_df
        return segments_df
```
