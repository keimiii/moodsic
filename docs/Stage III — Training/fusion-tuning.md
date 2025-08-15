# Fusion Tuning

- [✅] Implement variance-weighted fusion
- [ ] Run fixed-weight grid search on validation set
- [ ] Compare variance-weighted vs fixed weights
- [ ] Document best weights and stability trade-offs

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

## Fixed-Weight Search (Phase 2)

```python
def optimize_fusion_weights(self, scene_model, face_expert, val_data):
    best_w, best_score = None, float('inf')
    for scene_w in np.arange(0.3, 0.8, 0.1):
        face_w = 1 - scene_w
        fusion = SceneFaceFusion(scene_model, face_expert, SingleFaceProcessor())
        fusion.scene_weight, fusion.face_weight = scene_w, face_w
        val_loss = self._evaluate_fusion(fusion, val_data)
        if val_loss < best_score:
            best_score, best_w = val_loss, (scene_w, face_w)
    return best_w
```

## Validation Protocol

- Use held-out validation split from FindingEmo.
- Evaluate with the combined loss (CCC+MSE) and report per-dimension CCC and MAE.
- Track scene–face divergence to detect context overfitting.
- Assess stability metrics after EMA+gating to ensure tuning does not harm responsiveness.
