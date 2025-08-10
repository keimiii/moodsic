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
Run ID: PHASE1-YYYYMMDD-HHMM
Commit: <git-sha>
Data: Face crops vX (seed S)
Model: FaceEmotionRegressor (dropout=0.3)
Train: epochs=8, bs=..., lr=...
Metrics: CCC_v=..., CCC_a=...; MAE_v=..., MAE_a=...
Checkpoint: checkpoints/face_model.pth
```

```text
Run ID: PHASE2-SEARCH-YYYYMMDD-HHMM
Validation: combined loss; CCC/MAE per dim
Search Space: scene_w ∈ {0.3,0.4,0.5,0.6,0.7}
Best Weights: (scene_w=?, face_w=?) with score=?
Divergence: scene-face distance=...
Stability: jitter_reduction=...%, gating_freq=...
```

## Ablations (from architecture)

- Scene-only vs Face-enhanced (CCC improvement, divergence reduction, multi-person scenes).
- EMA vs EMA+Uncertainty Gating (jitter reduction, gating false positives).
- Segment-level vs Whole-video retrieval (alignment accuracy, variety, user preference).

## Best Models Summary

- Phase 0 best: <run-id> — CCC_v=..., CCC_a=...
- Phase 1 best: <run-id> — CCC_v=..., CCC_a=...
- Fusion best: weights=(..., ...) — validation loss=...
