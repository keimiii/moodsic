# Hyperparameters

- [✅] Seed defaults for MVP
- [ ] Sweep α (EMA), τ (gating), fusion weights
- [ ] Record LR finder outputs and chosen LR per phase
- [ ] Confirm dropout rates and MC sampling count

## Core Defaults (from architecture)

- Scene model dropout: `0.3`
- Face model dropout: `0.3`
- MC Dropout samples: `5` (training/eval); demo uses `10` for robustness
- EMA alpha: `0.7`
- Uncertainty threshold τ: `0.4` (demo shows `0.5`)
- Fusion fixed weights (if not variance-weighted): `scene=0.6`, `face=0.4`
- DEAM segment window: `10s`, overlap: `0.5`, sampling: `2 Hz`
- Minimum dwell time: `20–25s`

## Learning Rate & Scheduling

- LR selection via LR finder (`learn.lr_find().valley`).
- Schedules:
  - Phase 0: `fit_one_cycle(10, lr_max=lr)` with frozen backbone; then unfreeze and `fit_one_cycle(5, lr_max=slice(lr/100, lr/10))`.
  - Phase 1: `fit_one_cycle(8, lr_max=lr)`.
- Early stopping: `patience=5`.

## Sweep Ranges

- Fusion weight search: `scene_w ∈ {0.3, 0.4, 0.5, 0.6, 0.7}`, `face_w = 1 - scene_w`.
- EMA alpha: `0.5–0.8`.
- Uncertainty τ: `0.3–0.7`.
- MC samples: `3–10` (compute-bound).

## Batch/Compute Notes

- Use face crops (224×224) and resized frames (224×224) as per preprocessing.
- Record effective batch size, gradient accumulation if used, and GPU memory footprint per phase.
