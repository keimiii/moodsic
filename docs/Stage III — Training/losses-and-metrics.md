# Losses and Metrics

- [✅] Define combined loss (CCC + MSE)
- [✅] Primary metrics (CCC, MAE)
- [ ] Track scene–face divergence during validation
- [ ] Log stability metrics and targets

## Combined Loss

- Objective mixes agreement-focused CCC with MSE.

```python
def _combined_loss(self, pred, target):
    # 70% CCC + 30% MSE
    ccc_loss = 2 - self._ccc(pred[:, 0], target[:, 0]) - self._ccc(pred[:, 1], target[:, 1])
    mse_loss = F.mse_loss(pred, target)
    return 0.7 * ccc_loss + 0.3 * mse_loss
```

### CCC Formula

```python
def _ccc(self, pred, true):
    pred_mean = pred.mean(); true_mean = true.mean()
    covariance = ((pred - pred_mean) * (true - true_mean)).mean()
    pred_var = pred.var(); true_var = true.var()
    return (2 * covariance) / (pred_var + true_var + (pred_mean - true_mean)**2 + 1e-8)
```

## Training/Eval Metrics

- CCC (primary) and MAE reported during training for both valence and arousal.
- Scene–face divergence: mean Euclidean distance between scene and face predictions where both available.

## Stability Metrics (for pipeline evaluation)

- Jitter Reduction Rate: target 40–60% reduction after stabilization.
- Gating Activation Frequency: proportion of frames held by uncertainty gating.
- Response Time to Emotional Shifts: delay from change onset to stabilized output.

```python
stability = stabilizer.get_stability_metrics()  # variance and jitter over history
```

## Calibration Validation Metrics

- **Ablation Study**: Statistical comparison (paired t-tests) of with/without calibration performance
- **Parameter Monitoring**: Track learned scale/shift parameters; remove layer if near identity (1,1,0,0)
- **Generalization Test**: Validate on unseen datasets (EMOTIC, ArtPhoto) to prevent overfitting
- **Bias Visualization**: Bland-Altman plots showing systematic error reduction

```python
# Calibration validation workflow
evaluator = CalibrationEvaluator()
results = evaluator.ablation_study(predictions, labels, n_runs=5)

for metric in ['ccc_v', 'ccc_a', 'mae_v', 'mae_a']:
    if results[metric]['significant']:
        print(f"✓ {metric}: +{results[metric]['improvement']:.3f} (p={results[metric]['p_value']:.3f})")
```

## Retrieval-Stage Metrics (POC)

- Song-level emotional distance between query and retrieved songs.
- Switching frequency and dwell time distribution to ensure minimum dwell constraints are met.
- Optional (future): segment-level analyses if dynamic annotations/segmentation are enabled.
