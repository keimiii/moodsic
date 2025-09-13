# Fusion Strategy

- [✅] Implement variance-weighted fusion (inverse variance weighting)
- [✅] Provide fallback simple weighted average (defaults: 0.6 scene / 0.4 face)
- [✅] Handle no-face-detected path gracefully (scene-only)

## Approach
Combine scene and face predictions via variance-weighted averaging when uncertainty estimates are available. This trusts more confident predictions.

- Primary: inverse-variance weighting
- Fallback: fixed linear combination (`scene_weight = 0.6`, `face_weight = 0.4`)
- If no face is detected, use scene prediction as-is
- Fused variance is defined only under inverse-variance weighting; in
  fixed-weight fallback we do not propagate variance (treat as unknown/None).
- Default sampling: `scene_mc_samples = 5`, `face_tta = 5`.

Guardrails (optional): The fusion module supports optional gating of the face
path by detection score, per-dimension sigma, and frame brightness. See
`uncertainty-and-gating.md` for recommended thresholds and rationale.

## Reference Implementation

```python
import torch
import cv2

class SceneFaceFusion:
    def __init__(self, scene_model, face_expert, face_processor):
        self.scene_model = scene_model
        self.face_expert = face_expert  # EmoNet adapter
        self.face_processor = face_processor
        self.scene_weight = 0.6
        self.face_weight = 0.4

    def predict(self, frame, use_variance_weighting=True, n_mc_samples=5):
        scene_tensor = self._preprocess_frame(frame)
        scene_mean, scene_var = self.scene_model(scene_tensor, n_samples=n_mc_samples)

        face_crop = self.face_processor.extract_primary_face(frame)
        if face_crop is not None:
            # Adapter handles alignment, normalization, calibration; returns mean/var via TTA
            v_face, a_face, (v_var_face, a_var_face) = self.face_expert.predict(face_crop, tta=n_mc_samples)
            face_mean = torch.stack([torch.tensor(v_face), torch.tensor(a_face)])
            face_var = torch.stack([torch.tensor(v_var_face), torch.tensor(a_var_face)])
            if use_variance_weighting:
                final_pred, final_var = self._variance_weighted_fusion(scene_mean, scene_var, face_mean, face_var)
            else:
                final_pred = self.scene_weight * scene_mean + self.face_weight * face_mean
                # In fixed-weight fallback we leave variance unset (None)
                final_var = None
        else:
            final_pred, final_var = scene_mean, scene_var

        v = final_pred[0].item()
        a = final_pred[1].item()
        if final_var is None:
            var = (None, None)
        else:
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

Note: In the codebase, the public API is `perceive_and_fuse(frame_bgr) ->
FusionResult`, which exposes per-path predictions and the fused result along
with optional stabilizer metrics. The fusion math and fallbacks here match the
implementation.
