# Scaling and Matching (POC default)

- [ ] Full EmotionScaleAligner adoption across scale conversions (aligner exists, but training/eval code paths still use inline math)
- [✅] Song-level matching operates in reference space `[-1, 1]` using the DEAM cluster catalog (no segmentation yet)
- [ ] Verify mapping with plots

## Unified Scale Alignment

`EmotionScaleAligner` centralizes the conversion formulas, but several callers still duplicate the math. Migrating them remains on the backlog.

```python
from utils.emotion_scale_aligner import EmotionScaleAligner

# Initialize aligner
aligner = EmotionScaleAligner()

# FindingEmo → DEAM static conversion
v_deam, a_deam = aligner.findingemo_to_deam_static(v_fe, a_fe)
```

- Note: For this academic POC we use DEAM static annotations `[1, 9]` (dynamic `[-10, 10]` also available).
- FE ranges: Valence `[-3, 3]`, Arousal `[0, 6]` → DEAM static ranges `[1, 9]`.
- The aligner maintains consistency and handles edge cases automatically.
- Legacy normalization paths still apply manual conversions (e.g., `src/data/datasets.py`, `src/evaluation/evaluator.py`, `src/utils/metrics.py`); they should be moved over to the aligner.

## GMM Station Gating (Song-Level)
- Train a `StandardScaler` and `GaussianMixture(K≈5, covariance_type='diag')` on
  DEAM song-level valence–arousal in reference space `[-1, 1]` (as exported to `data/DEAM/deam_gmm_clusters.csv`).
- At runtime, transform stabilized `(v_ref, a_ref)` with the scaler and use `predict_proba` for soft posteriors.
- If the top posterior is < 0.55, widen the gate to include the top-2 clusters.
- Within the selected cluster set, rank songs by Euclidean distance between the
  stabilized V/A and each song’s song-level V/A; select top-1 (or top-N).

## Cluster-Gated Song Selection (from notebook)
- Train a `StandardScaler` and `GaussianMixture(K≈5, covariance_type='diag')` on
  DEAM song-level valence–arousal mapped to reference `[-1, 1]`.
- At runtime, transform stabilized `(v_ref, a_ref)` with the scaler and use `predict_proba` to obtain soft cluster posteriors.
- Choose top-1 cluster (or mix top-2 by posterior) and select within-cluster songs via simple distance to the query point (naive k-NN), with dwell-time and recent-song memory to avoid thrash (see `utils/song_matcher.py`).
