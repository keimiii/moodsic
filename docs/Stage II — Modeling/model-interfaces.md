# Model Interfaces

- [ ] Define input/output shapes and types for each component
- [ ] Specify device/batching behavior
- [ ] Handle no-face and edge cases explicitly
- [ ] Checkpoint/version naming conventions

## SceneEmotionRegressor
- Input: `pixel_values` tensor from CLIP processor, shape `[B, 3, H, W]` (CLIP-normalized)
- Output scale: reference space `[-1, 1]` for both valence and arousal
- Output (deterministic): tuple `(valence, arousal)` each of shape `[B]`
- Output (MC): tuple `(mean, var)` each of shape `[2, B]` where index `0` is valence, `1` is arousal

```python
v, a = scene_model(pixel_values)                    # shapes: [B], [B]
mean, var = scene_model(pixel_values, n_samples=5)  # shapes: [2, B], [2, B]
```

## Face Expert: EmoNet Adapter
- Input: primary face crop as BGR/RGB `np.ndarray` of shape `[H, W, 3]` (adapter handles alignment + resize + normalization)
- Output scale: reference space `[-1, 1]` for both valence and arousal
- Output: `(valence: float, arousal: float, variance: (float, float))`
- Uncertainty: via TTA (e.g., `tta=5`) rather than MC Dropout

```python
v, a, (v_var, a_var) = face_expert.predict(face_bgr, tta=5)
```

## SingleFaceProcessor
- Input: BGR image `np.ndarray` of shape `[H, W, 3]`
- Output: Cropped BGR face `np.ndarray` of shape `[224, 224, 3]` or `None` if no face

```python
face = face_processor.extract_primary_face(bgr_frame)  # None if not found
```

## SceneFaceFusion
- Input: raw BGR frame `np.ndarray`
- Output: `FusionResult` with fields:
  - `scene: Optional[EmotionPrediction]`
  - `face: Optional[EmotionPrediction]`
  - `fused: EmotionPrediction` (fields: `valence`, `arousal`, `var_valence`, `var_arousal`)
  - optional `face_bbox`, `face_score`, and stability metrics when enabled
- Behavior:
  - If face not detected → falls back to scene prediction (and variance)
  - If `use_variance_weighting` → inverse-variance fusion; else fixed weights (defaults 0.6/0.4).
    In fixed-weight fallback, fused variance may be unset (None).

```python
res = fusion.perceive_and_fuse(frame_bgr)
v, a = res.fused.valence, res.fused.arousal
v_var, a_var = res.fused.var_valence, res.fused.var_arousal  # may be None in fallback
```

## Stabilizer (AdaptiveStabilizer)
- Input per frame: `valence: float`, `arousal: float`, `variance: Optional[Tuple[float, float]]`
- Output: stabilized `(valence: float, arousal: float)`
- State: maintains EMA and last stable outputs

```python
sv, sa = stabilizer.update(v, a, variance=(v_var, a_var))
metrics = stabilizer.get_stability_metrics()
```

## EmotionPipeline (Unified Interface)
- Input: Raw model outputs (EmoNet valence/arousal)
- Output: Target-scale predictions with optional domain calibration
- Pipeline: `raw → scale_alignment → domain_calibration → final`

```python
from utils.emotion_pipeline import EmotionPipeline
pipeline = EmotionPipeline(calibration_layer=calibration, enable_calibration=True)

# Complete processing
v_final, a_final = pipeline.emonet_to_findingemo(v_emonet, a_emonet)
```

## Notes
- Internal contract: All model adapters (scene and face) and fusion operate in the reference space `[-1, 1]`.
- Scale alignment (e.g., FE↔reference, DEAM static `[1, 9]`↔reference) is handled at boundaries (training/eval, retrieval), not inside model interfaces.
- Domain calibration (see `cross-domain-calibration.md`) optionally corrects systematic biases between emotion contexts
- Device/batching: models accept batched tensors; fusion path uses batch size `1` at runtime
