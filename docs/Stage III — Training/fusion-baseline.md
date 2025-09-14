# Fusion Baseline

- [✅] Implement variance-weighted fusion as the baseline

## Method

- Fuse scene and face predictions using inverse-variance weights when MC Dropout variances are available; otherwise fall back to fixed weights.

```python
class SceneFaceFusion:
    def _variance_weighted_fusion(self, pred1, var1, pred2, var2):
        weight1 = 1 / (var1 + 1e-6); weight2 = 1 / (var2 + 1e-6)
        total = weight1 + weight2
        w1, w2 = weight1 / total, weight2 / total
        fused_pred = w1 * pred1 + w2 * pred2
        fused_var = 1 / total
        return fused_pred, fused_var
```

## Validation Protocol

- Use held-out validation split from FindingEmo.
- Evaluate variance-weighted fusion with MSE loss and report per-dimension MAE (primary). Optionally report Spearman’s ρ.
- Track scene–face divergence to detect context overfitting.
- Assess stability metrics after EMA+gating to ensure the baseline maintains responsiveness.
