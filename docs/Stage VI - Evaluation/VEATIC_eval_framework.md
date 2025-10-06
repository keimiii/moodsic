# VEATIC Evaluation Framework

## Checklist
- [x] Ground-truth VEATIC label CSVs available under `data/VEATIC/rating_averaged/` (loaded on the fly; no regeneration step needed).
- [x] Inference exports generated for stabilized and unstabilized modes with matching video sets.
- [x] Aggregation script implemented: `scripts/evaluation/aggregate_veatic_metrics.py` (produces per-video and aggregate CSVs + run params JSON).
- [x] Reporting templates established under `docs/Stage VI - Evaluation/` (Round I summary) and CSVs under `results/evaluation/`.

## Purpose
This document defines how we evaluate the PERCEIVE → STABILIZE stack on VEATIC videos.
It captures the why and how of the process so future contributors can reproduce results
without digging through notebooks or past threads.

Our primary goals are:
- quantify pathway quality (scene vs. face vs. fusion) using aggregated mean absolute error (MAE)
- measure the benefit of stabilization (EMA + gating) by comparing stabilized vs. unstabilized runs
- provide diagnostics (coverage, uncertainty, jitter) that explain why performance shifts

## Data & Artifacts
Two supported input formats (current default first):
1. **Consolidated Parquet summary (preferred)** – Single artifact at `results/inference/pipeline_results_<timestamp>.parquet`
   written by the pipeline export. Contains one row per video per mode with JSON payloads and metadata.
   Use `scripts/evaluation/aggregate_veatic_metrics.py` to produce:
   - `results/evaluation/veatic_per_video_<timestamp>.csv`
   - `results/evaluation/veatic_aggregate_<timestamp>.csv`
   - `results/evaluation/veatic_run_params_<timestamp>.json`
2. **Legacy JSON directories (still supported)** – Paired JSON files living under
   `results/inference/stabilized/` and `results/inference/unstabilized/` from the notebook export; contents mirror
   the fields consumed by the aggregator (means, variances, coverage). When available, prefer the Parquet path above.

Ground-truth labels live in `data/VEATIC/rating_averaged/{video_id}_{metric}.csv` and provide the clip-level
reference means we evaluate against. Maintain identical video sets across stabilized/unstabilized modes so comparisons
are paired.

## Evaluation Questions
We answer two headline questions:
1. **Pathway quality:** Does fusion achieve lower MAE than the individual scene or face pathways across VEATIC?
2. **Stabilization gain:** Does stabilization (EMA + gating) improve or hurt MAE relative to raw outputs?

Supporting diagnostics:
- How often does each pathway contribute predictions (`coverage`), and do coverage gaps explain MAE swings?
- When fusion underperforms, is it due to overweighting a noisy pathway or low face coverage?
- Do variances shrink under stabilization (indicating smoother response) without excessive lag?

## Metrics & Definitions
- **Per-video MAE:** `MAE_pathway(metric) = mean(|mean_{pathway,video} - mean_{label,video}|)` for
  each metric (valence, arousal). Use the ground-truth means computed from label CSVs.
- **Aggregated MAE:** Average the per-video MAEs across the dataset (optionally report both mean and
  median to expose skew).
- **Coverage:** The fraction of sampled timestamps where a pathway produced a prediction. Report mean and
  standard deviation across videos; low coverage can invalidate comparisons.
- **Variance diagnostics:** Average per-sample variance from the JSON (`var_*`). Helpful for tracking
  stabilization impact.
- **Stabilization deltas:** `ΔMAE = MAE_stabilized - MAE_unstabilized` per pathway,
  plus aggregate statistics (mean/median Δ, 95% CI via bootstrap).
- **Significance test:** Use paired bootstrap resampling or a paired t-test to quantify whether MAE
  differences are statistically meaningful.

> MAE remains the primary metric because it is scale-robust, directly interpretable, and consistent with the
> project proposal and overview docs. Secondary metrics (MSE, Spearman / Pearson correlations) can be added
> once the pipeline is reliable, but MAE answers our immediate questions with the least noise.

## Workflow
1. **Activate env.** `source .venv/bin/activate.fish`
2. **Choose run artifact.** Point to a single Parquet summary, e.g. `results/inference/pipeline_results_20251006_144126.parquet`.
3. **Aggregate.** Run:
   `python scripts/evaluation/aggregate_veatic_metrics.py results/inference/pipeline_results_<timestamp>.parquet`
   This writes per-video and aggregate CSVs plus a JSON of run parameters to `results/evaluation/`.
4. **Analyze.** Use the aggregate CSV to compare pathways and stabilization deltas (mean, median, std, 95% CI).
5. **Diagnostics (optional).** Plot MAE histograms/KDEs, flag fusion underperformers, and correlate coverage vs. MAE.
6. **Report.** Summarize results in a short Markdown note in `docs/Stage VI - Evaluation/` and reference the CSVs.

Legacy path (if Parquet unavailable): follow the same steps but inventory IDs from `results/inference/stabilized/` and
`unstabilized/`, then aggregate via a notebook or a helper script that mirrors the Parquet aggregator’s logic.

## Implementation Notes
- **Activation:** Always `source .venv/bin/activate.fish` before running Python so dependencies resolve.
- **Libraries:** Standard library + `pathlib`, `json`, `csv`, and `numpy`/`pandas` are sufficient. Follow project
  import style (stdlib first, then third-party).
- **Null handling:** Face pathway arrays may contain `null` entries when no face was detected. Filter them before
  computing statistics if you need per-sample analysis. The provided `coverage` field already reflects usable samples.
- **Performance:** With 124 videos and ~60 samples each, pure Python loops are fine. If you scale to per-frame
  diagnostics later, vectorize with `numpy` or `pandas`.
- **Reproducibility:** Persist intermediate results (per-video MAE tables) to `results/evaluation/` with timestamps
  to avoid mixing runs. Include the inference export timestamp or git commit hash in filenames.
- **Safety checks:**
  - Warn when coverage < 0.5 for any pathway; fusion MAE may be skewed.
  - Ensure each stabilized video references the same source path as the unstabilized counterpart.
  - Detect mismatched label lengths (should equal video frame count reported in JSON).

## Interpretation Guidance
- **Fusion underperforms scene & face:** likely indicates noisy face predictions or incorrect weighting. Inspect
  per-sample valence/arousal and variance traces to see which pathway dominated. Consider adjusting variance floors
  or the `scene_weight`/`face_weight` fallback.
- **Stabilization increases MAE:** may happen when the EMA lags sharp emotional shifts. Check the `uncertainty_threshold`
  and history window; a smaller `stabilizer_window` or lower `alpha` may help. Also verify the gating isn’t freezing
  outputs due to persistently high variance.
- **Face coverage is low:** the fusion result will lean on scene predictions. Evaluate MediaPipe detection quality or
  adjust cropping parameters if coverage stays low on certain video types (e.g., wide shots).

## Extensions & Future Work
- **Frame-aligned evaluation:** Once we trust the pipeline, compute MAE over matched timestamps instead of clip means.
  This will capture whether stabilization smooths too aggressively around rapid transitions.
- **Correlation metrics:** Add Spearman’s ρ and Pearson r, mirroring the project proposal targets, to measure how
  well temporal trends align even when means differ.
- **Uncertainty-aware scoring:** Weight per-video MAE by inverse variance to account for prediction confidence.
- **Visualization dashboard:** Automate plots (MAE bars, coverage heatmaps) and embed them in Streamlit or a
  notebook for faster iterative analysis.

By following this plan we deliver a reproducible, statistically grounded view of pathway performance and stabilization
impact on VEATIC, aligning future work with the project’s documented success criteria.

## Reports
- Round I: `docs/Stage VI - Evaluation/eval_res_round_i.md`
- Latest run artifacts (for reference):
  - `results/evaluation/veatic_run_params_20251006_144126.json`
  - `results/evaluation/veatic_aggregate_20251006_144126.csv`
  - `results/evaluation/veatic_per_video_20251006_144126.csv`
