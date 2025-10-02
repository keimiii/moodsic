# Model Interfaces

- [x] Define input/output shapes and types for each component (see sections below for the implemented adapters and processors).
- [x] Specify device/batching behavior (documented per interface; all runtime paths default to batch size `1`).
- [x] Handle no-face and edge cases explicitly (EmoNet adapter and face processor degrade gracefully to neutral predictions/`None`).
- [x] Checkpoint/version naming conventions (scene adapter loads `scene/checkpoints/clip_vit-b32_model_learner.pkl`; EmoNet adapter expects `models/emonet/pretrained/emonet_*.pth`).

## Scene Adapter: SceneCLIPAdapter
- Location: `models/scene/clip_vit_scene_adapter.py`
- Constructor highlights:
  - `model_name="openai/clip-vit-base-patch32"`
  - `tta` controls the default number of MC samples; `auto_load_best=True` loads `scene/checkpoints/clip_vit-b32_model_learner.pkl`
- `device="auto"` prefers CUDA -> MPS -> CPU
- Input: raw frame `np.ndarray` in BGR order, shape `[H, W, 3]`
- Output scale: reference space `[-1, 1]` for valence and arousal
- Output: `(valence: float, arousal: float, (var_valence: float, var_arousal: float))`
- Behavior:
  - Invalid inputs (non-BGR arrays/`None`) return neutral zeros with zero variance
  - Runs CLIP preprocessing internally, then MC Dropout over regression heads to estimate variance

```python
from models.scene import SceneCLIPAdapter

scene_model = SceneCLIPAdapter(tta=5, auto_load_best=True)
v, a, (v_var, a_var) = scene_model.predict(frame_bgr, tta=3)
```

## Face Expert: EmoNetAdapter
- Location: `models/face/emonet_adapter.py`
- Input: primary face crop `np.ndarray` (BGR, shape `[H, W, 3]`); adapter performs alignment, resize (256x256), and normalization
- Output scale: reference space `[-1, 1]`
- Output: `(valence: float, arousal: float, variance: (float, float))`
- Behavior:
  - Uses vendored EmoNet weights from `models/emonet/pretrained/`
  - Supports deterministic + TTA-based variance; invalid inputs return neutral zeros
  - Optional calibration via `models.calibration.CrossDomainCalibration`

```python
from models.face import EmoNetAdapter

face_expert = EmoNetAdapter(tta=5, tta_seed_mode="content")
v, a, (v_var, a_var) = face_expert.predict(face_bgr, tta=5)
```

## Single Face Processor: EmoNetSingleFaceProcessor
- Location: `utils/emonet_single_face_processor.py`
- Input: frame `np.ndarray` in BGR format `[H, W, 3]`
- Output: tuple `(face_crop: Optional[np.ndarray], bbox: Optional[Tuple[int, int, int, int]], score: float)`
- Behavior:
  - MediaPipe long-range detector with optional OpenCV cascade fallback
  - Applies aspect-ratio padding and optional resize (default 256x256)
  - Returns `(None, None, 0.0)` when no face is found or dependencies are unavailable

```python
from utils.emonet_single_face_processor import EmoNetSingleFaceProcessor

face_processor = EmoNetSingleFaceProcessor()
face, bbox, score = face_processor.extract_primary_face(frame_bgr)
```

## SceneFaceFusion
- Location: `models/fusion.py`
- Input: raw BGR frame `np.ndarray`
- Output: `FusionResult` with fields:
  - `scene: Optional[EmotionPrediction]`
  - `face: Optional[EmotionPrediction]`
  - `fused: EmotionPrediction` (fields: `valence`, `arousal`, `var_valence`, `var_arousal`)
  - Optional metadata: `face_bbox`, `face_score`, stability metrics, sampled faces
- Behavior:
  - Calls `scene_predictor.predict(frame_bgr, tta=scene_mc_samples)` when provided
  - Runs face detection + EmoNetAdapter on up to `face_mc_samples` faces with configurable sampling
  - Performs inverse-variance fusion by default; falls back to fixed weights (`scene_weight=0.6`, `face_weight=0.4`)
  - Supports gating (face score, variance, brightness) and an EMA stabilizer (`enable_stabilizer=True`)

```python
from models.fusion import SceneFaceFusion
from models.scene import SceneCLIPAdapter
from models.face import EmoNetAdapter
from utils.emonet_single_face_processor import EmoNetSingleFaceProcessor

fusion = SceneFaceFusion(
    scene_predictor=SceneCLIPAdapter(),
    face_expert=EmoNetAdapter(),
    face_processor=EmoNetSingleFaceProcessor(),
    use_variance_weighting=True,
)

result = fusion.perceive_and_fuse(frame_bgr)
v, a = result.fused.valence, result.fused.arousal
```

## Stabilizer (Adaptive EMA)
- Implemented as `_AdaptiveStabilizer` inside `models/fusion.py`
- Enabled via `SceneFaceFusion(..., enable_stabilizer=True, stabilizer_alpha=0.7, uncertainty_threshold=0.4)`
- Input per frame: `valence: float`, `arousal: float`, optional `(var_valence, var_arousal)`
- Output: stabilized `(valence, arousal)` and optional metrics via `get_stability_metrics()`

```python
sv, sa = fusion._stabilizer.update(v, a, variance=(v_var, a_var))  # internal use
metrics = fusion._stabilizer.get_stability_metrics()
```

## EmotionPipeline (Unified Interface)
- Location: `utils/emotion_pipeline.py`
- Input: raw EmoNet outputs in their native scale (≈`[-1, 1]`)
- Output: calibrated predictions in reference space or converted scales (FindingEmo, DEAM static)
- Pipeline: `raw -> scale_alignment -> domain_calibration -> final`

```python
from utils.emotion_pipeline import EmotionPipeline
from models.calibration import CrossDomainCalibration

calibration = CrossDomainCalibration()
pipeline = EmotionPipeline(calibration_layer=calibration, enable_calibration=True)

v_ref, a_ref = pipeline.emonet_to_reference(v_emonet, a_emonet)
v_fe, a_fe = pipeline.emonet_to_findingemo(v_emonet, a_emonet)
```

## Notes
- All adapters operate in the shared reference space `[-1, 1]`.
- Batched execution is supported where applicable (scene adapter accepts CLIP batches internally; runtime path uses batch size `1`).
- Scale alignment (e.g., FindingEmo ↔ reference) and domain calibration live at the pipeline boundaries and do not alter the adapter interfaces.
- Variance-aware fusion requires both scene and face adapters to return finite variances; otherwise fusion falls back to fixed weights.
