# Losses and Metrics

- [✅] Loss: MSE (reference space [-1, 1])
- [✅] Primary metrics: V/A MAE
- [✅] Track scene–face divergence during validation
  - VEATIC exports (`results/inference/pipeline_results_20251006_144126.parquet`) preserve per-frame scene/face series, and the aggregator in `scripts/evaluation/aggregate_veatic_metrics.py:285` surfaces clip-level means; the paired run reports ≈0.45 mean Euclidean gap between the streams pre-stabilization.
- [✅] Log stability metrics and targets
  - The same aggregation pass writes per-video `var_*` columns and dataset summaries (`results/evaluation/veatic_aggregate_20251006_144126.csv:1`), giving fused variance means of 8.16e-05 (stabilized) vs. 8.18e-05 (raw) for sanity checks against the 40–60% jitter-reduction target.

## Training Loss

- Use Mean Squared Error (MSE) for regression in reference space [-1, 1].
- Prefer smooth L2 during training; MAE is used for reporting.

```python
def _loss(self, pred, target):
    return F.mse_loss(pred, target)
```

## Why Not CCC?

- CCC relies on Pearson correlation and can become undefined when either
  predictions or targets have near-zero variance; its denominator approaches
  zero and torchmetrics returns NaN with a warning. To avoid instability and
  improve interpretability, CCC has been removed from loss/metrics.

## Training/Eval Metrics

- MAE (primary) for both valence and arousal; report per-dimension and average.
- Optional diagnostics: Spearman’s ρ, Pearson r.
- Scene–face divergence: mean Euclidean distance between scene and face predictions where both available (audited via the VEATIC Parquet run noted above).

## Stability Metrics (for pipeline evaluation)

- Jitter Reduction Rate: target 40–60% reduction after stabilization.
- Gating Activation Frequency: proportion of frames held by uncertainty gating.
- Response Time to Emotional Shifts: delay from change onset to stabilized output.

```python
stability = stabilizer.get_stability_metrics()  # variance and jitter over history
```

## Calibration Validation Metrics

- Ablation Study: Paired tests comparing with/without calibration; lower MAE wins.
- Parameter Monitoring: Track learned scale/shift parameters; remove layer if near identity (1,1,0,0).
- Generalization Test: Validate on unseen datasets (EMOTIC, ArtPhoto) to prevent overfitting.
- Bias Visualization: Bland-Altman plots showing systematic error reduction.

```python
# Calibration validation workflow
evaluator = CalibrationEvaluator()
results = evaluator.ablation_study(predictions, labels, n_runs=5)

for metric in ['mae_v', 'mae_a']:
    if results[metric]['significant']:
        print(f"✓ {metric}: {results[metric]['improvement']:.3f} (p={results[metric]['p_value']:.3f})")
```

## Retrieval-Stage Metrics (POC)

- Song-level emotional distance between query and retrieved songs.
- Switching frequency and dwell time distribution to ensure minimum dwell constraints are met.
