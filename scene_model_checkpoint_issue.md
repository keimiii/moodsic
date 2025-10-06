# Scene Model Checkpoint Regression – Investigation Summary

## Problem Statement
The VEATIC inference pipeline no longer reproduces the documented Round I metrics. Re-running `scripts/evaluation/run_inference_pipeline.py` with the published weights yields substantially different scene and fusion errors between runs, signaling that the scene pathway is no longer loading its trained head correctly.

## Detection
The divergence appeared when comparing new Parquet exports (e.g. `pipeline_results_20251005_210203.parquet`, `pipeline_results_20251005_210828.parquet`, `pipeline_results_20251005_213534.parquet`, `pipeline_results_20251005_213923.parquet`) against one another and against `veatic_aggregate_round_i.csv`. Per-video scene MAE flipped sign and magnitude, and successive runs with identical CLI flags produced inconsistent aggregates.

## Suspected Root Causes
- The tightened `_maybe_load_weights` routine rejects the legacy two-output head checkpoint (`clip_vit-b32_improved_head.pth`), leaving the scene adapter uninitialized.
- The fastai learner pickle (`clip_vit-b32_model_improved_learner.pkl`) still encodes the legacy architecture and depends on training-time modules (`pathlib._local`, `clip`, fastai transforms) that are unavailable during inference, so automatic fallback also fails.
- Without loaded weights, the scene stream outputs effectively random predictions, which in turn destabilize fusion metrics.

## Implemented Changes
1. Hardened `SceneCLIPAdapter._maybe_load_weights` to fail loudly when checkpoints are missing or incompatible.
2. Updated `run_inference_pipeline.py` to offer an `--n` limit for quicker experiments and to register fastai globals for torch deserialization.
3. Redirected the loader to consume the learner pickle exclusively, removing the silent `.pth` fallback.

## Outcomes
- The loader now surfaces errors instead of silently continuing, but successive pipeline runs still differ because no compatible weights are actually loaded.
- Scene mean valence/arousal remains unstable (differences up to ~0.29 / ~0.23 across runs), proving that the regression head remains randomly initialized.
- Fusion and face medians drift as well, since fusion inherits noisy scene predictions.

## Next Steps
- Regenerate a checkpoint whose state dict matches the current split-head architecture (two single-output regressors). Replace the files under `scene/checkpoints/` with this export and rerun inference to confirm stability.
- Alternatively, port the training code back into the inference environment so the existing learner pickle can be loaded (restoring missing modules like `pathlib._local`, `clip`, and fastai transforms), then serialize the updated heads.
- After new weights are in place, re-run the pipeline (with and without `--n`) to verify deterministic outputs and reconcile `veatic_aggregate_round_i.csv`.
