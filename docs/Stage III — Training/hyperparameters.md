# Hyperparameters

- [✅] Seed defaults for MVP
- [ ] Sweep α (EMA) and τ (gating)
- [ ] Record LR finder outputs and chosen LR per phase
- [ ] Confirm dropout rates and MC sampling count

## Core Defaults (from architecture)

- Scene model dropout: `0.3`
- Scene MC Dropout samples: `5` (training/eval); demo uses `10` for robustness
- EmoNet TTA samples: `5` (PERCEIVE)
- EMA alpha: `0.7`
- Uncertainty threshold τ: `0.4` (demo shows `0.5`)
- Fusion fixed weights (if not variance-weighted): `scene=0.6`, `face=0.4`
- Retrieval (POC): Song-level DEAM static `[1, 9]`; simple k-NN (linear scan)
- Minimum dwell time: `20–25s`; recent-song memory enabled

## Learning Rate & Scheduling

- LR selection via LR finder (`learn.lr_find().valley`).
- Schedules:
  - Phase 0: `fit_one_cycle(10, lr_max=lr)` with frozen backbone; then unfreeze and `fit_one_cycle(5, lr_max=slice(lr/100, lr/10))`.
  - Phase 1: (N/A for face; EmoNet is fixed, no training.)
- Early stopping: `patience=5` (scene model).

## Sweep Ranges

- EMA alpha: `0.5–0.8`.
- Uncertainty τ: `0.3–0.7`.
- MC samples: `3–10` (compute-bound).

## Batch/Compute Notes

- Use face crops (224×224) and resized frames (224×224) as per preprocessing.
- Record effective batch size, gradient accumulation if used, and GPU memory footprint per phase.
