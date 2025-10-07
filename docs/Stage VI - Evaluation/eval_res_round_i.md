# VEATIC Evaluation – Round I

## Parameters Used (batch_video_fusion_export.ipynb)
- `VIDEO_DIR`: `data/VEATIC/videos`
- `OUTPUT_ROOT`: `results/inference`
- `VIDEO_EXTENSIONS`: `.mp4`, `.mov`, `.mkv`, `.avi`
- `SCENE_TTA`: 3
- `FACE_TTA`: 3
- `TARGET_SAMPLE_FPS`: 1.0 Hz
- `MAX_FRAMES`: `None`
- `USE_VARIANCE_WEIGHTING`: `True`
- `SCENE_WEIGHT`: 0.6
- `FACE_WEIGHT`: 0.4
- `FACE_SCORE_THRESHOLD`: `None`
- `FACE_MAX_SIGMA`: `None`
- `BRIGHTNESS_THRESHOLD`: `None`
- Stabilizer (enabled for stabilized exports):
  - `STABILIZER_MODE`: `both`
  - `UNCERTAINTY_THRESHOLD`: 0.4
  - `STABILIZER_WINDOW`: 60 frames
  - `STABILIZER_ALPHA`: 0.7
- Batch export invoked twice: once with `enable_stabilizer=False`, once with `enable_stabilizer=True`.

### Model Architectures Loaded
- **Scene pathway** — `SceneCLIPAdapter`: frozen CLIP ViT-B/32 vision encoder paired with dropout-enabled MLP heads for valence/arousal regression (MC-dropout TTA = 3 passes during export).
- **Face pathway** — `EmoNetAdapter`: stacked hourglass EmoNet backbone (FAN-style conv blocks) producing expression + VA logits, run with 3× EmoNet TTA and face alignment via `EmoNetSingleFaceProcessor`.

## Aggregated Results (aggregate_veatic_metrics.py)
Raw data sources backing the summaries:
- `results/evaluation/veatic_run_params_20251006_144126.json`
- `results/evaluation/veatic_aggregate_20251006_144126.csv`
- `results/evaluation/veatic_per_video_20251006_144126.csv`

### Stabilized MAE Summary (dataset-level)
| Pathway | Valence Mean MAE | Valence 95% CI | Valence Median MAE | Arousal Mean MAE | Arousal 95% CI | Arousal Median MAE | Coverage Mean | Coverage Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Face | 0.208 | 0.185–0.234 | 0.181 | 0.168 | 0.148–0.189 | 0.152 | 0.861 | 0.171 |
| Scene | 0.238 | 0.210–0.268 | 0.221 | 0.203 | 0.178–0.230 | 0.162 | – | – |
| Fusion | 0.193 | 0.169–0.217 | 0.182 | 0.161 | 0.140–0.184 | 0.121 | – | – |

### Unstabilized MAE Summary (dataset-level)
| Pathway | Valence Mean MAE | Valence 95% CI | Valence Median MAE | Arousal Mean MAE | Arousal 95% CI | Arousal Median MAE | Coverage Mean | Coverage Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Face | 0.209 | 0.184–0.237 | 0.181 | 0.168 | 0.147–0.189 | 0.151 | 0.861 | 0.171 |
| Scene | 0.239 | 0.210–0.268 | 0.217 | 0.203 | 0.176–0.232 | 0.166 | – | – |
| Fusion | 0.193 | 0.171–0.217 | 0.180 | 0.161 | 0.138–0.182 | 0.124 | – | – |

The aggregate CSV (`results/evaluation/veatic_aggregate_20251006_144126.csv`) retains pathway-level MAE standard deviations, stabilization deltas (mean/median plus 95% CIs), and variance estimates. The paired per-video export (`results/evaluation/veatic_per_video_20251006_144126.csv`) adds clip-level deltas, coverage flags, and booleans for when face or scene outperform fusion.

## Analysis
- **Fusion leads both axes**: With `SCENE_WEIGHT=0.6` / `FACE_WEIGHT=0.4`, fusion attains the lowest dataset mean MAE on valence (0.193) and arousal (0.161), beating face-only (0.208/0.168) and scene-only (0.238/0.203).
- **Stabilization impact ~zero**: Mean ΔMAE hovers in ±0.0001–0.0007 across pathways/metrics (e.g., fusion valence −0.00056; fusion arousal +0.00028). Medians shift only slightly (e.g., fusion arousal median improves from 0.124 to 0.121).
- **Pathway ordering**: Fusion < Face < Scene for both valence and arousal means; face coverage remains high and unchanged (≈0.861 ± 0.171).
- **CIs are tight**: 95% CIs for fusion are narrow (valence 0.169–0.217; arousal 0.140–0.184), suggesting stable performance under current settings.

## Potential Next Steps
- Keep 0.6/0.4 blend as default given fusion’s lead on both axes; explore light tuning (e.g., 0.55/0.45) to check robustness.
- Add per-clip logged weights in `scripts/evaluation/run_inference_pipeline.py` to analyze when scene dominates and its effect on arousal.
- Consider pathway-specific stabilization (disable for arousal) since deltas are negligible and median improvements are modest.
- Extend aggregation with coverage-stratified slices to confirm fusion’s gains persist in low-coverage segments.
