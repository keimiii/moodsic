# Training Protocols and Phases

- [✅] Define Phase 0/1/2 training flow (freeze → unfreeze; face; fusion)
- [ ] Implement dataloaders for scene dataset (face path uses EmoNet; no face dataloaders needed)
- [ ] Run LR finder per phase and record selected LRs
- [ ] Validate fusion baseline on held-out set

## Phase 0 — Scene Baseline

- Backbone: CLIP/ViT features with regression heads and dropout.
- Freeze backbone initially; then unfreeze for fine-tune.
- Early stopping used.
- LR policy: LR finder → one-cycle.

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
```

## Stabilization During Training/Eval

- EMA smoothing with uncertainty gating to assess stability and responsiveness.

```python
stabilizer = AdaptiveStabilizer(alpha=0.7, uncertainty_threshold=0.4)
```
