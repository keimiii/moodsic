# DEAM Quadrant Evaluation Framework

## Status — Under Consideration

We are re-evaluating the quadrant-matching plan before investing further implementation effort because:
- The proposed metric compares the fused valence/arousal pair to the quadrant baked into the same GMM bundle, so it largely becomes a wiring sanity check rather than an independent validation of personalization quality.
- Unlike the VEATIC framework (`docs/Stage VI - Evaluation/VEATIC_eval_framework.md`), this approach does not benchmark against a held-out, human-labeled dataset; it therefore cannot demonstrate generalization in the way our professors expect for rigorous evaluation.
- It offers limited insight into whether the recommended songs truly reflect the user’s affect—posterior gaps or centroid distances stay unreported, and no external evidence corroborates the match.
- The script would still be useful as a smoke test for bundle drift or scale mismatches, but that benefit may not justify the build cost for a POC unless paired with richer storytelling.

For progress tracking we are keeping the original workflow below, but it should be treated as a reference draft until we decide whether to proceed.

## Alternative Direction — Explainable Stations (XAI Focus)

The explainability-first path repurposes the same DEAM assets to produce transparent, human-friendly station narratives—an angle that scores well in an academic POC because it demonstrates user-centred AI and interpretability.

### What We Can Deliver
- **Station profile cards:** Aggregate each cluster’s centroid, valence/arousal ranges, dominant genres, and top-confidence exemplar tracks. Persist these summaries alongside the GMM bundle so the app can display “You landed in *Sunlit Rush*: upbeat rock & chiptune (avg VA ≈ +0.40/+0.36).”
- **Why-this-song blurbs:** When recommending a track, surface the user’s fused VA, the station persona, and a few supporting metadata tags (tempo adjectives, instrumentation, last.fm labels) drawn from `data/DEAM/metadata_2014.csv` and related files.
- **Confidence storytelling:** Expose posterior margins and the runner-up station with its persona, making the recommendation rationale auditable (“Brooding Focus trailed by 9 %, so we stayed with Sunlit Rush”).
- **Professor-ready documentation:** Publish a short appendix summarizing each station’s narrative, exemplar songs, and metadata provenance. This doubles as interpretability evidence during the demo or report defence.

### Implementation Hooks
- Extend `scripts/clustering/export_deam_gmm.py` (or equivalent) to compute per-cluster genre counts, tag frequencies, and exemplar lists when producing the bundle. Store them in `clusters_meta.json` so downstream code and docs remain in sync.
- Build a lightweight helper (e.g., `utils/deam_station_explainers.py`) that formats the persona strings and exposes a `why_this_song` function for the UI or logs.
- Update evaluation docs to show how these explanations tie back to personalization goals, referencing the same fused VA outputs used in VEATIC.

### Why Professors Will Appreciate It
- Aligns with XAI and responsible AI criteria common in Master’s programmes: every recommendation comes with a reproducible rationale grounded in dataset metadata.
- Demonstrates thoughtful user experience design by translating latent clusters into accessible language, rather than stopping at numerical metrics.
- Provides tangible artefacts (profile cards, documentation, demo messaging) that can be showcased during assessment, signalling polish beyond baseline engineering.

## Legacy Quadrant Accuracy Draft (Reference)

The original plan remains below for archival purposes while we evaluate the explainability-first alternative.

## Checklist
- [x] Export the five-component DEAM GMM (means, covariances, priors, quadrant labels) into a versioned bundle under `results/clustering/` with provenance metadata.
- [x] Implement `scripts/clustering/deam_clusters.py` to annotate inference Parquet exports with DEAM posteriors, components, and quadrants.
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
  similar versioned directory) and drop the artifacts there alongside the
  serialized metadata. Keep an optional pointer (for example,
  `results/clustering/deam_gmm_latest`) only if it helps wiring.
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
3. **Cluster lookup:** Run `scripts/clustering/deam_clusters.py` (or import
   `annotate_parquet_with_clusters`) to append the top component, full posterior
   vector, and quadrant label to the Parquet. Downstream steps can then operate
   directly on the enriched columns.
4. **Scoring:** Emit `1` if the quadrants match, else `0`. Capture the raw
   predictions, cluster id, and quadrant labels to support debugging.
5. **Aggregation:** Compute overall accuracy and optionally per-quadrant
   precision/recall using the appended posterior columns (e.g., reuse the
   0.55 top-two-gating threshold when interpreting scores). Write a CSV to
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
- `docs/Stage IV — Inference/retrieval-dwell-and-variety.md` — Runtime gating,
  dwell, and de-duplication policy the evaluator mirrors.

## Open Questions / Next Steps

- Lock in four quadrants only; near-axis predictions should fall back to the
  standard sign-based quadrant so we avoid introducing a neutral label.
- Mirror the production station gating when scoring accuracy: run the same
  top-1 / top-2 gating logic, identify the cluster tied to the recommended
  song, and log the gating parameters in the run metadata.
- Drop historical GMM bundles once a newer export is adopted; no archival
  backlog needed for this POC.

With these components in place, the DEAM evaluation becomes reproducible and
auditable, closing the loop between emotion prediction quality and music
matching accuracy. <!-- Over-engineering: Captures both implementation steps and rationale without automating beyond current scope. -->
