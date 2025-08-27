# Training Protocols and Phases

- [✅] Define Phase 0/1/2 training flow (freeze → unfreeze; face; fusion)
- [ ] Implement dataloaders for scene dataset (face path uses EmoNet; no face dataloaders needed)
- [ ] Run LR finder per phase and record selected LRs
- [ ] Validate fusion on held-out set and store final weights

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
            loss_func=self._combined_loss,
            metrics=[self._ccc_metric, mae],
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
if results['ccc_avg']['significant']:
    print("✓ Calibration improves performance")
else:
    print("✗ No significant improvement - use without calibration")
```

## Phase 2 — Fusion Optimization

- Combine scene and face via variance-weighted averaging; optionally tune fixed weights on validation set.

```python
def optimize_fusion_weights(self, scene_model, face_expert, val_data):
    best_weights, best_score = None, float('inf')
    for scene_w in np.arange(0.3, 0.8, 0.1):
        face_w = 1 - scene_w
        fusion = SceneFaceFusion(scene_model, face_expert, SingleFaceProcessor())
        fusion.scene_weight, fusion.face_weight = scene_w, face_w
        val_loss = self._evaluate_fusion(fusion, val_data)
        if val_loss < best_score:
            best_score, best_weights = val_loss, (scene_w, face_w)
    return best_weights
```

## Stabilization During Training/Eval

- EMA smoothing with uncertainty gating to assess stability and responsiveness.

```python
stabilizer = AdaptiveStabilizer(alpha=0.7, uncertainty_threshold=0.4)
```
