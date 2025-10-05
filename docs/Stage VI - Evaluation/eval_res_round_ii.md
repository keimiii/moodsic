# VEATIC Evaluation – Round II

## Parameters Used (batch_video_fusion_export.ipynb)
- `VIDEO_DIR`: `data/VEATIC/videos`
- `OUTPUT_ROOT`: `results/inference`
- `VIDEO_EXTENSIONS`: `.mp4`, `.mov`, `.mkv`, `.avi`
- `SCENE_TTA`: 3
- `FACE_TTA`: 3
- `TARGET_SAMPLE_FPS`: 1.0 Hz
- `MAX_FRAMES`: `None`
- `USE_VARIANCE_WEIGHTING`: `True`
- `SCENE_WEIGHT`: 0.5
- `FACE_WEIGHT`: 0.5
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
- `results/evaluation/veatic_run_params_20251005_161034.json`
- `results/evaluation/veatic_aggregate_20251005_161034.csv`
- `results/evaluation/veatic_per_video_20251005_161034.csv`

### Stabilized MAE Summary (dataset-level)
| Pathway | Valence Mean MAE | Valence 95% CI | Valence Median MAE | Arousal Mean MAE | Arousal 95% CI | Arousal Median MAE | Coverage Mean | Coverage Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Face | 0.209 | 0.185–0.234 | 0.180 | 0.168 | 0.148–0.189 | 0.151 | 0.861 | 0.171 |
| Scene | 0.244 | 0.216–0.273 | 0.216 | 0.433 | 0.398–0.468 | 0.444 | – | – |
| Fusion | 0.212 | 0.186–0.238 | 0.184 | 0.240 | 0.211–0.269 | 0.241 | – | – |
| Ground Truth (clip means)* | -0.127 | – | -0.185 | 0.141 | – | 0.115 | – | – |

### Unstabilized MAE Summary (dataset-level)
| Pathway | Valence Mean MAE | Valence 95% CI | Valence Median MAE | Arousal Mean MAE | Arousal 95% CI | Arousal Median MAE | Coverage Mean | Coverage Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Face | 0.208 | 0.183–0.237 | 0.182 | 0.168 | 0.147–0.189 | 0.152 | 0.861 | 0.171 |
| Scene | 0.240 | 0.213–0.269 | 0.224 | 0.432 | 0.396–0.468 | 0.425 | – | – |
| Fusion | 0.211 | 0.187–0.237 | 0.187 | 0.236 | 0.207–0.264 | 0.229 | – | – |
| Ground Truth (clip means)* | -0.127 | – | -0.185 | 0.141 | – | 0.115 | – | – |

\*Ground-truth row captures the dataset-level mean and median of VEATIC clip labels (not an error metric).

The aggregate CSV (`results/evaluation/veatic_aggregate_20251005_161034.csv`) retains pathway-level MAE standard deviations, stabilization deltas (mean/median plus 95% CIs), and variance estimates. The paired per-video export (`results/evaluation/veatic_per_video_20251005_161034.csv`) adds clip-level deltas, coverage flags, and booleans for when face or scene outperform fusion.

## Analysis
- **Weight parity trade-offs**: Equalizing scene/face priors to 0.5 narrows the stabilized valence gap to 0.003 MAE (fusion 0.212 vs. face 0.209) but expands the arousal deficit to 0.072 MAE (0.240 vs. 0.168). Scene-only arousal balloons to 0.433, so the fusion stack is still absorbing scene noise rather than amplifying face wins.
- **Stabilizer drift**: Mean ΔMAE remains near-zero, yet the direction skews negative—fusion arousal worsens by 0.0039 on average with 148/248 clips drifting upward (`results/evaluation/veatic_aggregate_20251005_161034.csv`, `results/evaluation/veatic_per_video_20251005_161034.csv`). Valence deltas are tiny (0.0007) but consistently positive, signaling that the smoother is adding latency more often than it reduces spikes.
- **Pathway ordering**: Face retains the lead across every slice (0.168/0.168 arousal MAE stabilized/unstabilized), fusion sits in the middle, and scene trails badly. Fusion still beats scene on 75/124 valence clips and 118/124 arousal clips with stabilization, underscoring that the blend stays preferable to scene-only inference despite the larger scene weight.
- **Coverage sensitivity**: Fusion’s valence minus face gap correlates negatively with face coverage (ρ≈-0.33) and positively with scene MAE (ρ≈0.52), while arousal shows an even stronger link (ρ≈-0.36 / 0.73). With median face coverage at 0.86, the equal-weight blend is disproportionately penalized whenever scene variance spikes, exactly the segments the old 0.6/0.4 prior partially muted.
- **Low-coverage pocket**: The same six clips with face coverage <0.5 (IDs 18, 24, 26, 4, 5, 9) remain the main fusion rescue cases—fusion wins valence on four and arousal on two despite weak face support, confirming that fallback logic is useful but rarely invoked.
- **Delta balance**: Per-clip ΔMAE histograms stay symmetric (scene/arousal splits 124/124 between improvements and regressions), yet fusion’s arousal tail shows more large positives than negatives, hinting that stabilization should become opt-in for arousal-heavy analyses until weighting adapts.

## Potential Next Steps
- Reintroduce face-weight favoritism (e.g., 0.6/0.4 or adaptive caps) to restore arousal performance while monitoring valence drift in `aggregate_veatic_metrics.py` outputs.
- Add per-clip weight logging during `scripts/run_inference_pipeline.py` runs to quantify how often scene logits dominate when variance is high.
- Gate stabilization by pathway or signal type—disable for arousal exports until ΔMAE confidence intervals shrink below ±0.001.
- Revisit scene preprocessing (exposure normalization, CLIP prompt tuning) to contain the 0.43 arousal MAE before blending.
- Extend aggregation notebooks with coverage-stratified summaries so low-coverage segments don’t obscure improvements from weighting tweaks.
<!-- Over-engineering check: Document mirrors Round I structure with focused stats; further automation could pipe tables directly from notebooks if iteration speed becomes a bottleneck. -->
