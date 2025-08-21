# Model Interfaces

- [ ] Define input/output shapes and types for each component
- [ ] Specify device/batching behavior
- [ ] Handle no-face and edge cases explicitly
- [ ] Checkpoint/version naming conventions

## SceneEmotionRegressor
- Input: `pixel_values` tensor from CLIP processor, shape `[B, 3, H, W]` (CLIP-normalized)
- Output (deterministic): tuple `(valence, arousal)` each of shape `[B]`
- Output (MC): tuple `(mean, var)` each of shape `[2, B]` where index `0` is valence, `1` is arousal

```python
v, a = scene_model(pixel_values)                    # shapes: [B], [B]
mean, var = scene_model(pixel_values, n_samples=5)  # shapes: [2, B], [2, B]
```

## Face Expert: EmoNet Adapter
- Input: primary face crop as BGR/RGB `np.ndarray` of shape `[H, W, 3]` (adapter handles alignment + resize + normalization)
- Output: `(valence_fe: float, arousal_fe: float, variance: (float, float))` in FindingEmo ranges
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
- Output: `(valence: float, arousal: float, variance: (float, float))`
- Behavior:
  - If face not detected → returns scene prediction and variance
  - If `use_variance_weighting` → inverse-variance fusion; else fixed weights (defaults 0.6/0.4)

```python
v, a, (v_var, a_var) = fusion.predict(frame, use_variance_weighting=True, n_mc_samples=5)
```

## Stabilizer (AdaptiveStabilizer)
- Input per frame: `valence: float`, `arousal: float`, `variance: Optional[Tuple[float, float]]`
- Output: stabilized `(valence: float, arousal: float)`
- State: maintains EMA and last stable outputs

```python
sv, sa = stabilizer.update(v, a, variance=(v_var, a_var))
metrics = stabilizer.get_stability_metrics()
```

## Notes
- Scale alignment (FE→DEAM static [1, 9] for this POC) is handled in the matching stage, not in model interfaces
- Device/batching: models accept batched tensors; fusion path uses batch size `1` at runtime
