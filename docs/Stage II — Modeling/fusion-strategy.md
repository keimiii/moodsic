# Fusion Strategy

- [ ] Implement variance-weighted fusion (inverse variance weighting)
- [ ] Provide fallback simple weighted average (defaults: 0.6 scene / 0.4 face)
- [ ] Handle no-face-detected path gracefully (scene-only)
- [ ] Optional grid search to optimize fusion weights on validation

## Approach
Combine scene and face predictions via variance-weighted averaging when uncertainty estimates are available. This trusts more confident predictions.

- Primary: inverse-variance weighting
- Fallback: fixed linear combination (`scene_weight = 0.6`, `face_weight = 0.4`)
- If no face is detected, use scene prediction as-is

## Reference Implementation

```python
import torch
import cv2

class SceneFaceFusion:
    def __init__(self, scene_model, face_model, face_processor):
        self.scene_model = scene_model
        self.face_model = face_model
        self.face_processor = face_processor
        self.scene_weight = 0.6
        self.face_weight = 0.4

    def predict(self, frame, use_variance_weighting=True, n_mc_samples=5):
        scene_tensor = self._preprocess_frame(frame)
        scene_mean, scene_var = self.scene_model(scene_tensor, n_samples=n_mc_samples)

        face_crop = self.face_processor.extract_primary_face(frame)
        if face_crop is not None:
            face_tensor = self._preprocess_frame(face_crop)
            face_mean, face_var = self.face_model(face_tensor, n_samples=n_mc_samples)
            if use_variance_weighting:
                final_pred, final_var = self._variance_weighted_fusion(scene_mean, scene_var, face_mean, face_var)
            else:
                final_pred = self.scene_weight * scene_mean + self.face_weight * face_mean
                final_var = self.scene_weight * scene_var + self.face_weight * face_var
        else:
            final_pred, final_var = scene_mean, scene_var

        v = final_pred[0].item()
        a = final_pred[1].item()
        var = (final_var[0].item(), final_var[1].item())
        return v, a, var

    def _variance_weighted_fusion(self, pred1, var1, pred2, var2):
        w1 = 1 / (var1 + 1e-6)
        w2 = 1 / (var2 + 1e-6)
        tot = w1 + w2
        w1, w2 = w1 / tot, w2 / tot
        fused_pred = w1 * pred1 + w2 * pred2
        fused_var = 1 / tot
        return fused_pred, fused_var

    def _preprocess_frame(self, frame):
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        import torch
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t - mean) / std
        return t.unsqueeze(0)
```

## Notes
- Divergence monitoring (scene vs. face) can be computed downstream for analysis
- Weight optimization can be performed via grid search over validation data
