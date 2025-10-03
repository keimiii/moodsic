# Scene Model

- [x] Choose backbone (CLIP ViT) and freeze strategy
- [x] Implement regression heads with dropout for valence and arousal
- [x] Enable MC Dropout style sampling for uncertainty
- [x] Define CLIP preprocessing and batching

## Summary
The production scene path is implemented as `SceneCLIPAdapter` in
`models/scene/clip_vit_scene_adapter.py`. It wraps a frozen CLIP ViT backbone,
adds lightweight regression heads, and exposes a `predict` API that returns
valence, arousal, and per-dimension variance. Training notebooks export the
dropout heads, and inference loads them automatically from
`scene/checkpoints/clip_vit-b32_model_improved_learner.pkl` (or a caller supplied
checkpoint).

## Backbone and Preprocessing
- Backbone: `openai/clip-vit-base-patch32` vision encoder from Hugging Face
  transformers (configurable via `model_name`).
- Parameters remain frozen during inference; fine tuning is handled in
  notebooks before weights are saved.
- Preprocessing uses `CLIPImageProcessor.from_pretrained(model_name)` to
  produce `pixel_values` tensors from BGR numpy frames (the adapter handles the
  BGR to RGB conversion internally).

## Adapter API
- Input: BGR `np.ndarray` of shape `[H, W, 3]`.
- Output: `(valence: float, arousal: float, (var_valence: float, var_arousal: float))`
  in reference space `[-1, 1]`.
- Sampling: `tta` argument controls the number of stochastic passes (default 5).
- Invalid input (missing frame or wrong shape) returns neutral zeros and zero
  variance.

```python
from models.scene import SceneCLIPAdapter

scene_adapter = SceneCLIPAdapter(tta=5, auto_load_best=True)
v, a, (v_var, a_var) = scene_adapter.predict(frame_bgr)
```

## Training Notes
- Training scripts live in `notebooks/scene/` and export learned heads to
  `scene/checkpoints/*`.
- `SceneCLIPAdapter` auto loads
  `scene/checkpoints/clip_vit-b32_model_improved_learner.pkl` unless a custom
  `weights_path` is provided.
- MC Dropout is achieved by leaving dropout layers in train mode during sampling
  while keeping linear and layer norm modules in eval mode.
- Batch inference is supported via CLIP processors, but runtime defaults to
  batch size 1 (per frame processing).

## Integration Checklist
- [x] Adapter wired into `SceneFaceFusion` via `scene_predictor` argument.
- [x] Tests in `tests/test_perceive_e2e_flow.py` import `SceneCLIPAdapter` and
  exercise the predict path (skipped when torch or transformers are absent).
- [x] Runtime driver (`utils/runtime_driver.py`) instantiates the adapter when
  scene perception is enabled.
