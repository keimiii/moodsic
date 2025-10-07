# Scene Model Checkpoint Regression – Resolution Summary

## Incident Recap
Round I VEATIC inference runs started producing divergent scene MAE after the scene adapter stopped loading its trained regression head. Successive Parquet exports (e.g. `pipeline_results_20251005_210203.parquet` through `_213923.parquet`) disagreed with one another and with `veatic_aggregate_round_i.csv`, confirming the checkpoint regression.

## Remediation Actions
- Retrained and re-exported the CLIP ViT-B32 scene head with the corrected two-output architecture, yielding `clip_vit-b32_improved_fixed.{pth,pkl}` (`32f2ea00`, `076328e7`).
- Tightened `SceneCLIPAdapter` weight handling to call `get_image_features`, require explicit loads when `weights_path` is set, and downgrade aux-head failures to warnings (`4457535d`).
- Simplified the evaluation pipeline to consume the Parquet artifact directly and removed the silent `.pth` fallback, ensuring only compatible exports are ever loaded (`4457535d`).

## Verification
- `scripts/evaluation/run_inference_pipeline.py` with the new checkpoint now completes without loader warnings and produces deterministic Parquet exports (latest: `results/inference/pipeline_results_20251006_144126.parquet`).
- Aggregation of that run (`results/evaluation/veatic_aggregate_20251006_144126.csv`) reports stable scene metrics (valence MAE 0.2383, arousal MAE 0.2033) that align with Round I expectations and no longer drift between executions.
- Fusion metrics realigned once the scene stream stabilized, so downstream evaluation artefacts remain consistent with the reference documentation.

## Follow-ups
- Keep `clip_vit-b32_improved_fixed.{pth,pkl}` as the default scene adapter export and archive the legacy `clip_vit-b32_improved_head.pth` to avoid accidental reuse.
- Add a lightweight regression test to guarantee `SceneCLIPAdapter._maybe_load_weights` fails fast if a future export regresses the schema.
