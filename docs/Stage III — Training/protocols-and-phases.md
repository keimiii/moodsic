# Training Protocols and Phases

- [✅] Define Phase 0/1/2 training flow (freeze → unfreeze; face; fusion)
- [✅] Implement dataloaders for scene dataset (fastai DataBlock in notebooks; face path uses EmoNet — no face dataloaders)
  - Implemented in scene training notebooks with fastai DataBlock/DataLoaders
    (see notebooks/scene/* "DataBlock & DataLoaders" sections). Splits present:
    `data/train.csv`, `data/valid.csv`, `data/test.csv`.
- [✅] Run LR finder for Phase 0 and apply one-cycle schedule
  - Used in notebooks (`learn.lr_find().valley` + `fit_one_cycle`).
  - TODO: Record selected LRs per run in `docs/Stage III — Training/experiments-log.md` and `docs/Stage III — Training/hyperparameters.md`.
- [ ] Validate fusion baseline on held-out set
  - Baseline fusion implemented and tested; run dataset-level validation and log.
  - Helpful: `scripts/fusion_threshold_tuning.py` for stability gating sweeps.

References (key artifacts in repo):
- Scene inference adapter with MC Dropout heads: `models/scene/clip_vit_scene_adapter.py`
- Scene training notebooks + heads saved: `notebooks/scene/*.ipynb`, checkpoints in `scene/checkpoints/*_head.pth`
- Face expert (EmoNet) adapter: `models/face/emonet_adapter.py`
- Cross-domain calibration layer + trainer + evaluator: `models/calibration/{cross_domain.py,trainer.py,evaluation.py}`
- Calibration results/checkpoint: `models/emonet/evaluation/results/*`
- Fusion (variance-weighted) + stabilizer (EMA + uncertainty gating): `models/fusion.py` (with tests in `tests/test_fusion.py`)

## Phase 0 — Scene Baseline

- Backbone: CLIP/ViT features with regression heads and dropout.
- Freeze backbone initially; then unfreeze for fine-tune.
- Early stopping used.
- LR policy: LR finder → one-cycle.

Status: Completed (trained via notebooks; heads saved under `scene/checkpoints/`). Inference adapter implemented in `models/scene/clip_vit_scene_adapter.py`.

```python
class PhaseTrainer:
    def train_phase_0(self):
        scene_dls = self._prepare_scene_dataloaders()
        scene_model = SceneEmotionRegressor()
        learn = Learner(
            scene_dls,
            scene_model,
            loss_func=F.mse_loss,
            metrics=[mae],
            cbs=[EarlyStoppingCallback(patience=5)]
        )
        lr = learn.lr_find().valley
        learn.fit_one_cycle(10, lr_max=lr)
        scene_model.backbone.requires_grad_(True)
        learn.fit_one_cycle(5, lr_max=slice(lr/100, lr/10))
        return learn
```

## Phase 1 — EmoNet Integration + Domain Calibration

- Use EmoNet as a fixed face expert via the adapter (alignment, normalization, calibration).
- **NEW**: Train CrossDomainCalibration layer to correct face→scene domain shift.
- Validate calibration parameters on a small FE validation split; clamp outputs to FE ranges.
- Provide TTA-based uncertainty (e.g., tta=5) for fusion and gating.

Status: Completed
- Face expert adapter: `models/face/emonet_adapter.py` (TTA, optional eye alignment; optional calibration application; clamps to [-1, 1]).
- Calibration implemented: `models/calibration/{cross_domain.py,trainer.py,evaluation.py}`.
- Trained checkpoint + evaluation metrics present: `models/emonet/evaluation/results/` (e.g., `calibration_emonet2findingemo.pt`, `test_metrics*.json`).

```python
# Train domain calibration (optional)
from models.calibration import CrossDomainCalibration, CalibrationTrainer

calibration = CrossDomainCalibration(l2_reg=1e-4)
trainer = CalibrationTrainer(calibration)

# Use subset of FindingEmo with EmoNet predictions + ground truth
trainer.fit(emonet_predictions, findingemo_labels, val_split=0.2)

# Run ablation study to validate effectiveness
evaluator = CalibrationEvaluator()
results = evaluator.ablation_study(emonet_predictions, findingemo_labels, n_runs=5)

# Only keep calibration if statistically significant improvement
if results['mae_avg']['significant']:
    print("✓ Calibration improves performance (lower MAE)")
else:
    print("✗ No significant improvement - use without calibration")
```

## Phase 2 — Fusion Integration

- Combine scene and face via variance-weighted averaging.

```python
# No fixed-weight search; use inverse-variance fusion as the baseline.
# Implemented in models/fusion.py with tests in tests/test_fusion.py
# Example usage (inference-time):
from models.fusion import SceneFaceFusion

fusion = SceneFaceFusion(scene_predictor=scene_adapter, face_expert=face_adapter,
                         face_processor=face_detector,
                         use_variance_weighting=True)
result = fusion.perceive_and_fuse(frame_bgr)
v, a = result.fused.valence, result.fused.arousal
```

## Stabilization During Training/Eval

- EMA smoothing with uncertainty gating to assess stability and responsiveness.

Implemented via optional stabilizer inside `SceneFaceFusion`.

```python
from models.fusion import SceneFaceFusion

fusion = SceneFaceFusion(scene_predictor=scene_adapter, face_expert=face_adapter,
                         face_processor=face_detector,
                         enable_stabilizer=True,
                         stabilizer_alpha=0.7,
                         uncertainty_threshold=0.4)
res = fusion.perceive_and_fuse(frame_bgr)
# res.stability_variance / res.stability_jitter available when enabled
```

## Next Actions (To‑Do)

- Validate fusion baseline on a held-out set and log results in
  `docs/Stage III — Training/experiments-log.md` (use `data/test.csv` and scene/face adapters).
- Record Phase 0 LR finder selections in `hyperparameters.md` and `experiments-log.md`.
- Optional: extract the Phase 0 training flow from notebooks into a script/CLI for reproducibility.
