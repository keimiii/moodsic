# Experiments Log

- [✅] Define run template and ablation checklist
- [ ] Populate runs with metrics and checkpoints
- [ ] Summarize best-performing models

## Run Entry Template

```text
Run ID: PHASE0-YYYYMMDD-HHMM
Commit: <git-sha>
Data: FindingEmo split vX (seed S)
Model: SceneEmotionRegressor (dropout=0.3)
Train: epochs=10(frozen)+5(unfrozen), bs=..., lr=...
Loss/Metric: combined (CCC+MSE); CCC_v=..., CCC_a=...; MAE_v=..., MAE_a=...
Notes: early_stopping=patience5; lr_find.valley=...
Checkpoint: checkpoints/scene_model.pth
```

```text
Run ID: PHASE1-EMONET-YYYYMMDD-HHMM
Commit: <git-sha>
Setup: models/emonet/ present; checkpoints in models/emonet/pretrained/
Calibration: models/emonet/calibration.json (a_v=..., b_v=..., a_a=..., b_a=...)
TTA: 5
Metrics: CCC_v=..., CCC_a=...; MAE_v=..., MAE_a=... (on FE validation)
Notes: clamped to FE ranges
```

```text
Run ID: PHASE2-FUSION-YYYYMMDD-HHMM
Validation: combined loss; CCC/MAE per dim
Fusion: inverse-variance weighting baseline (no fixed-weight search)
Divergence: scene-face distance=...
Stability: jitter_reduction=...%, gating_freq=...
```

## Ablations (from architecture)

- Scene-only vs Face-enhanced (CCC improvement, divergence reduction, multi-person scenes).
- EMA vs EMA+Uncertainty Gating (jitter reduction, gating false positives).
- Station gating ablations (top-1 vs top-2 threshold 0.55; impact on alignment, variety, user preference).

## Best Models Summary

- Phase 0 best: <run-id> — CCC_v=..., CCC_a=...
- Fusion baseline: inverse-variance weighting — validation loss=...
- EmoNet calibration best: (scale_v=..., scale_a=..., shift_v=..., shift_a=...) — val CCC: v=..., a=...
- Calibration ablation: p-value=..., effect_size=..., significant=True/False
