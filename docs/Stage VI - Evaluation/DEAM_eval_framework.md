# DEAM Quadrant Evaluation Framework

## Checklist
- [ ] Export the five-component DEAM GMM (means, covariances, priors, quadrant labels) into a versioned bundle under `results/clustering/` with provenance metadata.
- [ ] Implement `utils/deam_clusters.py` (or equivalent) to load the bundle and expose `predict_proba` plus quadrant lookup helpers for inference and evaluation.
- [ ] Ship `scripts/evaluation/evaluate_deam_quadrants.py` to score pipeline Parquet exports and emit `deam_quadrant_accuracy_<timestamp>` artifacts.
- [ ] Document each evaluator run (timestamp, input Parquet, tolerance settings) alongside VEATIC results in `docs/Stage VI - Evaluation/` to keep reporting consistent.

## Why This Exists

The DEAM evaluation framework extends the end-to-end pipeline by validating that
video-derived emotions align with the Gaussian mixture clusters that back our
song recommendations. The VEATIC evaluation confirms the regression quality of
valence/arousal scores; DEAM pushes that further by checking whether those
scores land in the correct emotional quadrants used for playlist selection.
Without a reproducible implementation, we risk diverging cluster definitions,
silent regressions in song matching accuracy, and inconsistent reporting across
runs. This guide documents the two missing pieces needed to operationalize the
framework described in `docs/project_overview.md:785` and referenced by the
clustering notebook `notebooks/Inference/e2e_video_to_music_clusters.ipynb`.

For broader evaluation context, see `docs/Stage VI - Evaluation/VEATIC_eval_framework.md`.

## Objectives

- Provide a versioned GMM parameter bundle (means, covariances, priors, and
  quadrant labels) that downstream code can load deterministically.
- Implement a lightweight evaluator that turns pipeline predictions into
  quadrant-level accuracy metrics (`1` for matches, `0` for mismatches) and
  aggregates the results for reporting under `results/evaluation/`.

## Component A — Persisted GMM Parameter Bundle

### Intuition

The music matcher relies on a five-component Gaussian mixture model trained on
DEAM embeddings to represent emotion “stations”. Each component is mapped to one
of the four valence–arousal quadrants. Freewheeling re-training (or notebook-only
clusters) makes evaluation irreproducible; persisting the parameters decouples
inference from exploratory work.

### Requirements

- **Artifacts:** Serialize the fitted GMM parameters (component means,
  covariances, mixture weights) alongside metadata that records the quadrant for
  each component and the reference space limits (expected `[-1, 1]`). A compact
  `JSON` + `npz` pair works: `clusters_meta.json` with quadrant labels and
  scaling info, `clusters_params.npz` with arrays.
- **Storage location:** Create `results/clustering/deam_gmm_<timestamp>/` (or a
  similar versioned directory) containing the artifacts plus a `README.md`
  summarizing training data, preprocessing, and the mapping logic. Keep the
  latest symlink (e.g., `results/clustering/deam_gmm_latest`) for ease of
  consumption.
- **Loading helper:** Add a small utility (e.g.,
  `utils/deam_clusters.py`) that loads the artifacts and exposes an object with
  `predict_proba(valence, arousal)` and `quadrant_for_component(idx)` methods.
  This avoids duplicating loading logic across the pipeline and evaluator.
- **Version tracking:** Record the git commit and notebook path that produced
  the bundle in the metadata so future updates can diff behavior before
  adoption.

### Implementation Tips

- Re-run `e2e_video_to_music_clusters.ipynb` with the final pre-processing
  pipeline and export the parameters directly in the notebook to avoid drift.
- Enforce deterministic random seeds in the notebook when refitting; capture
  the seed in the metadata file.
- Validate that each component’s centroid signs match its quadrant label before
  freezing the bundle.

## Component B — Quadrant Accuracy Evaluator

### Intuition

We evaluate the retrieval layer by asking: *Did the video’s stabilized emotion
land in the same quadrant as the cluster we matched it to?* If not, the song
recommendation may be semantically off even when raw valence/arousal scores look
reasonable. The evaluator formalizes this check and produces roll-ups that can
be trended across experiments.

### Workflow

1. **Inputs:**
   - Pipeline Parquet export (e.g.,
     `results/inference/pipeline_results_<timestamp>.parquet`) containing the
     stabilized valence/arousal predictions per video.
   - Persisted GMM bundle from Component A.
   - Optional CSV of manual overrides or filtering rules (e.g., videos to
     exclude).
2. **Quadrant labeling:** Derive the video’s quadrant using the sign of valence
   and arousal after applying any tolerance band (e.g., treat values within
   ±0.05 of zero as neutral and fall back to highest absolute axis). Persist the
   tolerance in the evaluator’s config so comparisons stay consistent.
3. **Cluster lookup:** Use the loading helper to obtain the highest-probability
   cluster (or follow the production selection logic if different). Fetch the
   cluster’s quadrant label from metadata.
4. **Scoring:** Emit `1` if the quadrants match, else `0`. Capture the raw
   predictions, cluster id, and quadrant labels to support debugging.
5. **Aggregation:** Compute overall accuracy and optionally per-quadrant
   precision/recall to surface imbalances. Write a CSV to
   `results/evaluation/deam_quadrant_accuracy_<timestamp>.csv` plus a JSON of run
   parameters mirroring the VEATIC documentation style.
6. **Reporting:** Update `docs/Stage VI - Evaluation/` summaries with the new
   accuracy metric alongside VEATIC MAE results.

### Implementation Notes

- Keep the evaluator as a small script (e.g.,
  `scripts/evaluation/evaluate_deam_quadrants.py`) with a function that can be
  imported into tests or notebooks. Accept CLI arguments for the inference
  Parquet path, cluster bundle directory, tolerance threshold, and output stem.
- Reuse the existing `EmotionScaleAligner` utilities to ensure quadrant checks
  occur in the reference space.
- Provide unit tests that feed a tiny synthetic Parquet file and a mocked GMM
  bundle to verify scoring and aggregation logic.
- When production matching uses additional heuristics (e.g., station gating),
  mirror those steps before scoring so evaluation reflects user-visible
  behavior.

## Related Documentation

- `docs/Stage VI - Evaluation/VEATIC_eval_framework.md` — Overall evaluation
  philosophy and workflow, including VEATIC metrics.
- `docs/project_overview.md` (Section 7.4) — High-level description of the DEAM
  evaluation framework and its place in the architecture.
- `notebooks/Inference/e2e_video_to_music_clusters.ipynb` — Source of the GMM
  training logic and quadrant mapping experiments.
- `scripts/evaluation/aggregate_veatic_metrics.py` — Reference implementation of
  a timestamped evaluation pipeline (structure and output conventions).

## Open Questions / Next Steps

- Decide whether quadrants should treat near-axis predictions as “neutral” and
  allow a dedicated fifth quadrant label. If yes, update both the cluster
  metadata and evaluator to stay consistent.
- Clarify how station gating (top-1 vs. top-2 expansion) impacts accuracy
  computation—should the evaluator simulate gating or score only the final
  cluster? Make this choice explicit in the README.
- Determine retention policy for historical bundles: keep all versions, or
  archive after major refits to prevent clutter?

With these components in place, the DEAM evaluation becomes reproducible and
auditable, closing the loop between emotion prediction quality and music
matching accuracy. <!-- Over-engineering: Captures both implementation steps and rationale without automating beyond current scope. -->
