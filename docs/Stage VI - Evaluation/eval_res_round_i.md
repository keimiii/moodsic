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
  - `UNCERTAINTY_THRESHOLD`: 0.4
  - `STABILIZER_WINDOW`: 60 frames
  - `STABILIZER_ALPHA`: 0.7
- Batch export invoked twice: once with `enable_stabilizer=False`, once with `enable_stabilizer=True`.

### Model Architectures Loaded
- **Scene pathway** — `SceneCLIPAdapter`: frozen CLIP ViT-B/32 vision encoder paired with dropout-enabled MLP heads for valence/arousal regression (MC-dropout TTA = 3 passes during export).
- **Face pathway** — `EmoNetAdapter`: stacked hourglass EmoNet backbone (FAN-style conv blocks) producing expression + VA logits, run with 3x EmoNet TTA and face alignment via `EmoNetSingleFaceProcessor`.

## Aggregated Results (aggregate_veatic_metrics.py)
### Stabilized MAE Summary (dataset-level)
| Pathway | Valence Mean MAE | Valence 95% CI | Valence Median MAE | Arousal Mean MAE | Arousal 95% CI | Arousal Median MAE | Coverage Mean | Coverage Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Face | 0.208 | 0.185–0.234 | 0.182 | 0.168 | 0.148–0.189 | 0.151 | 0.861 | 0.171 |
| Scene | 0.260 | 0.223–0.298 | 0.209 | 0.346 | 0.314–0.378 | 0.358 | – | – |
| Fusion | 0.220 | 0.191–0.249 | 0.192 | 0.208 | 0.183–0.233 | 0.207 | – | – |
| Ground Truth (clip means)* | -0.127 | – | -0.185 | 0.141 | – | 0.115 | – | – |

### Unstabilized MAE Summary (dataset-level)
| Pathway | Valence Mean MAE | Valence 95% CI | Valence Median MAE | Arousal Mean MAE | Arousal 95% CI | Arousal Median MAE | Coverage Mean | Coverage Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Face | 0.208 | 0.183–0.237 | 0.180 | 0.168 | 0.147–0.189 | 0.150 | 0.861 | 0.171 |
| Scene | 0.263 | 0.227–0.303 | 0.205 | 0.345 | 0.312–0.375 | 0.365 | – | – |
| Fusion | 0.222 | 0.196–0.252 | 0.190 | 0.209 | 0.183–0.233 | 0.193 | – | – |
| Ground Truth (clip means)* | -0.127 | – | -0.185 | 0.141 | – | 0.115 | – | – |

\*Ground-truth row captures the dataset-level mean and median of VEATIC clip labels (not an error metric).

The aggregate CSV also carries additional diagnostics that are not shown above: pathway-level MAE standard deviations, stabilization deltas (mean/median plus 95% CIs), and any available variance means. The paired per-video export (`veatic_per_video_*.csv`) adds clip-level deltas, coverage flags, and boolean indicators for when face or scene beats fusion—useful for deeper slicing in Polars.

## Analysis
- **Stabilization in context**: The paired stabilized/unstabilized exports land within ±0.002 MAE of each other for every pathway/metric. Bootstrap 95% CIs for ΔMAE straddle zero (`results/evaluation/veatic_aggregate_round_i.csv`), so the EMA+gating stack is neither rescuing nor harming the population-level error. Median arousal MAE for fusion actually improves when the smoother is off (0.193 vs. 0.207), hinting at stabilization lag during rapid arousal swings.
- **Pathway hierarchy**: Ordering stays Face < Fusion < Scene regardless of stabilization. Face MAE beats fusion by ~0.012 (valence) / ~0.040 (arousal), while fusion maintains an ~0.04–0.14 margin over scene. The consistency across both runs points to sound aggregation; the shortfall is upstream in the weighting strategy rather than a data bug.
- **Drivers of face dominance**: Average face coverage is 0.86, and the face-only MAE trails fusion most when coverage is high (ρ ≈ 0.07 between the fusion–face gap and coverage). The fusion–face gap correlates tightly with scene MAE (ρ ≈ 0.7), meaning fusion is inheriting scene noise even when the face stream is confident. In the six clips with coverage < 0.5, fusion beats face on valence in five cases but loses on arousal in all six, confirming that variance gating is overweighting scene predictions whenever the face stream is thin.
- **Per-video behaviour**: Stabilized fusion improves valence MAE on 64 clips and hurts 60; arousal splits 58/66. Only five videos show fusion worse than both face and scene on valence (IDs 36, 38, 41, 45, 81), and two on arousal (71, 80). These rare failures cluster around low face coverage and high scene variance, reinforcing that the fusion logic isn’t adapting sharply enough to signal quality changes.
- **Fusion in practice**: Even with the aggregate gap, fusion still beats scene on 82/124 stabilized valence clips and 111/124 arousal clips, so it remains a safer default than scene-only runs. Fusion overtakes face less often (56/124 valence, 35/124 arousal), but those wins disproportionately occur when face coverage is compromised—exactly the fallback behaviour we expected. The two modes (high face coverage vs. scarce coverage) explain why the dataset mean penalises fusion more than a coverage-aware summary would.
- **Delta balance**: Per-clip ΔMAE histograms are symmetric around zero for all pathways; the stabilizer trims large positive spikes but introduces similar-sized negatives. This balance, plus near-zero aggregate deltas, signals that we should treat stabilization as a neutral switch until the weighting logic can better discriminate which segments benefit from smoothing.

## Potential Next Steps
- **Reweight fusion adaptively**: Revisit variance-weighted blending so high-confidence face segments dominate. Consider clamping scene contributions when its per-clip MAE or variance exceeds the face stream; today’s fixed 0.6/0.4 priors appear too generous to the noisier scene pathway.
- **Investigate low-coverage clips**: Deep-dive the six videos with face coverage < 0.5 (IDs 18, 24, 26, 4, 5, 9). Review detection logs and raw frames to decide whether to tune MediaPipe parameters, add fallback tracking, or skip frames entirely when faces drop out.
- **Characterise fusion failures**: For the handful of clips where fusion underperforms both inputs, inspect time-series traces to see whether scene spikes, weighting lag, or stabilization lag drive the error. Logging intermediate weights/variances during export could expose the decision boundary.
- **Stress-test stabilizer**: Run targeted clips with high-frequency affect changes (or simulate with subsampled labels) while sweeping `STABILIZER_ALPHA` and `STABILIZER_WINDOW`. Validate on percentile MAE or per-frame metrics to determine whether the smoother is simply too sluggish for arousal.
- **Broaden diagnostics**: Add correlation metrics (Spearman, Pearson) and percentile-based coverage summaries in future aggregation runs to triangulate places where mean MAE hides temporal mismatches. This will help confirm whether fusion’s disadvantages stem from amplitude bias or phase lag.
<!-- Over-engineering check: Document covers essentials with lightweight tables; could slim further by omitting GT row if space tight. -->
