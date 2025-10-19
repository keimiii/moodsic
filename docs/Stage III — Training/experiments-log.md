# Experiments Log

- [✅] Define run template and ablation checklist
- [✅] Populate runs with metrics and checkpoints
- [✅] Summarize best-performing models

## Phase 0 - Scene Model Runs (Cross-ref: `docs/Stage III — Training/scene_model_ablation.md`)

| Notebook | Backbone & Head | Test MAE (Valence / Arousal / Avg) | Notes |
| --- | --- | --- | --- |
| `dinov3_baseline.ipynb` | Frozen DINOv3 ViT-B/16 -> Linear(768->2) tanh | 1.1527 / 1.3455 / 1.2491 | Original baseline; ~70/15/15 split after filtering 695/142/174 missing frames. |
| `dinov3_mlp.ipynb` | Frozen DINOv3 ViT-B/16 -> LN + GELU MLP -> tanh | 1.1323 / 1.3653 / 1.2488 | Added 512-d hidden layer; no material gain vs baseline. |
| `efficientnet_baseline.ipynb` | Frozen EfficientNet-B0 -> Linear(1280->2) tanh | 1.5471 / 1.4360 / 1.4915 | Underfits; warns on PIL truncation during dataload. |
| `resnet_baseline.ipynb` | Frozen ResNet-50 -> Linear(2048->2) tanh | 1.4178 / 1.3671 / 1.3925 | Marginally better than EfficientNet; still >1.39 MAE. |
| `resnet_101.ipynb` | Frozen ResNet-101 -> Linear(2048->2) tanh | 1.4284 / 1.3401 / 1.3842 | Slightly improved arousal MAE; valence stagnates. |
| `CLIP_ViT-B32.ipynb` | Frozen CLIP ViT-B/32 -> Linear(512->2) tanh | 1.1274 / 1.4613 / 1.2942 | First CLIP baseline; improves ~0.10 avg MAE over ResNet-50. |
| `CLIP_ViT-B32_improved.ipynb` | Frozen CLIP ViT-B/32 -> LN + GELU -> Dropout(0.15) -> 128-d head + aux Emo8 | 1.0684 / 1.2792 / 1.1738 | Auxiliary Emo8 branch + deeper head nets modest gains. |
| `scene_model_training.ipynb` | Frozen CLIP ViT-B/32 -> EmotionHead (BN + Dropout 0.3) | 0.5475 / 0.7970 / 0.6722 | Stratified loader (`split_indices.json`); largest jump in MAE (-0.50 avg) with stronger head + regularization. |

### Key Observations

- CLIP backbones consistently beat DINOv3 and CNN baselines by >=0.2 MAE_avg thanks to better global semantics (`scene_model_ablation.md` rows 8-9, 14-16).
- The EmotionHead configuration in `scene_model_training.ipynb` cuts average MAE in half relative to the best previous CLIP head, while also using stratified splits (train/val/test=13149/1632/1651, skipping 703/100/81 missing files).
- Extra hidden capacity without auxiliary supervision (e.g., DINOv3 MLP) plateaus, indicating bottleneck in backbone feature quality rather than head depth.
- ResNet/ EfficientNet runs suffer from higher valence error despite similar arousal curves, suggesting their frozen features miss valence cues even with comparable training stability.

### CLIP vs DINO Findings

- CLIP's text-aligned pretraining keeps frozen features sensitive to affective cues, so fine-tuning on the 13k-image FindingEmo split reaches the lowest test MAE_avg in the study (`docs/Stage III — Training/scene_model_ablation.md:8`, `docs/Stage III — Training/scene_model_ablation.md:22`).
- The EmotionHead stack (BatchNorm, ReLU, Dropout 0.3) provides enough capacity and regularisation to drive CLIP's MAE_avg down to 0.6722, a 0.5766 drop versus the strongest DINO variant at 1.2488 (`docs/Stage III — Training/scene_model_ablation.md:9`, `docs/Stage III — Training/scene_model_ablation.md:22`).
- Stratified split caching reduces class imbalance and missing-frame churn (703/100/81 skips) so the CLIP EmotionHead stays data-efficient where DINO runs degrade on the same split (`docs/Stage III — Training/scene_model_ablation.md:7`).
- DINO's regression-only heads underfit valence cues, with Spearman scores topping out near 0.35/0.15, whereas CLIP's improved head reaches 0.6643/0.3247 before shifting focus to MAE with the EmotionHead (`docs/Stage III — Training/scene_model_ablation.md:23`, `docs/Stage III — Training/scene_model_ablation.md:25`).

## Run Entry Template

```text
Run ID: PHASE0-YYYYMMDD-HHMM
Commit: <git-sha>
Data: FindingEmo split vX (seed S)
Model: SceneCLIPAdapter (clip-vit-base-patch32, dropout=0.3 heads)
Train: epochs=10(frozen heads)+5(partial unfreeze), bs=..., lr=...
Loss: MSE; Metrics: MAE_v=..., MAE_a=..., MAE_avg=...; Spearman ρ (optional)
Notes: early_stopping=patience5; lr_find.valley=...
Checkpoint: scene/checkpoints/clip_vit-b32_improved_fixed.pkl
```

```text
Run ID: PHASE1-EMONET-YYYYMMDD-HHMM
Commit: <git-sha>
Setup: models/emonet/ present; checkpoints in models/emonet/pretrained/
Calibration: models/emonet/calibration.json (a_v=..., b_v=..., a_a=..., b_a=...)
TTA: 5
Metrics: MAE_v=..., MAE_a=..., MAE_avg=...; Spearman ρ (optional) (on FE validation)
Notes: clamped to FE ranges
```

```text
Run ID: PHASE2-FUSION-YYYYMMDD-HHMM
Validation: MSE loss; MAE per dim; Spearman ρ (optional)
Fusion: inverse-variance weighting baseline (no fixed-weight search)
Divergence: scene-face distance=...
Stability: jitter_reduction=...%, gating_freq=...
```

## Ablations (from architecture)

- Scene-only vs Face-enhanced (MAE reduction, divergence reduction, multi-person scenes).
- EMA vs EMA+Uncertainty Gating (jitter reduction, gating false positives).
- Station gating ablations (top-1 vs top-2 threshold 0.55; impact on alignment, variety, user preference).

## Best Models Summary

- Phase 0 best: `scene_model_training.ipynb` (CLIP EmotionHead) - MAE_v=0.5475, MAE_a=0.7970, MAE_avg=0.6722.
- Fusion baseline: inverse-variance weighting - validation loss=...
- EmoNet calibration best: (scale_v=..., scale_a=..., shift_v=..., shift_a=...) - val MAE: v=..., a=...
- Calibration ablation: p-value=..., effect_size=..., significant=True/False
