# Scaling and Matching (POC default)

- [✅] Implemented unified EmotionScaleAligner for all scale conversions
- [✅] Song-level matching over DEAM static `[1, 9]` (no segmentation)
- [ ] Verify mapping with plots

## Unified Scale Alignment

All scale conversions are now handled by the EmotionScaleAligner class:

```python
from utils.emotion_scale_aligner import EmotionScaleAligner

# Initialize aligner
aligner = EmotionScaleAligner()

# FindingEmo → DEAM static conversion
v_deam, a_deam = aligner.findingemo_to_deam_static(v_fe, a_fe)
```

- Note: For this academic POC we use DEAM static annotations `[1, 9]` (dynamic `[-10, 10]` also available).
- FE ranges: Valence `[-3, 3]`, Arousal `[0, 6]` → DEAM static ranges `[1, 9]`.
- All conversions maintain consistency and handle edge cases automatically.

## DEAM Segmentation (optional/future)
- Window size: `10s`
- Overlap: `50%`
- Sampling rate used: `2 Hz`
- Optional indexing: KD-Tree over segment means.
- Persist segment metadata: `song_id`, `start_time`, `end_time`, `valence`, `arousal`.

## Cluster-Gated Song Selection (from notebook)
- Train a `StandardScaler` and `GaussianMixture(K≈5, covariance_type='diag')` on
  DEAM song-level valence–arousal mapped to reference `[-1, 1]`.
- At runtime, transform stabilized `(v_ref, a_ref)` with the scaler and use
  `predict_proba` to obtain soft cluster posteriors.
- Choose top-1 cluster (or mix top-2 by posterior) and select within-cluster
  songs via simple distance to the query point (naive k-NN), with dwell-time
  and recent-song memory to avoid thrash.
