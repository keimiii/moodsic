# Face Expert: EmoNet Adapter

Status: design and integration guide for the face pathway

## Purpose
A small adapter that exposes EmoNet as a drop‑in "face expert" to the runtime pipeline. It hides model loading, preprocessing, calibration, and uncertainty so PERCEIVE/FUSION can use a simple, stable API.

## Responsibilities
- Model management
  - Load EmoNet checkpoints from `models/emonet/pretrained/` (5‑class or 8‑class model).
  - Select device (CPU/GPU), set `eval()` and `no_grad()`.
- Preprocessing
  - Input: a BGR face crop sampled from the detection pool (MediaPipe primary detections plus optional Haar-cascade fallback—only non-overlapping faces are added, so we never fabricate crops when none exist).
  - Face alignment via MediaPipe eye keypoints (inter-ocular rotation) to approximately level faces before resizing/normalization (no separate `face-alignment` dependency).
  - Resize and apply EmoNet’s normalization exactly as in upstream demos.
- Inference
  - Run forward pass to get continuous valence and arousal (optionally expression class if needed).
- Uncertainty (optional, recommended)
  - Test‑time augmentation (horizontal flip; minor crop/scale jitter) for N passes; return mean and per‑dimension variance for V and A. `predict(..., seed=...)` allows the runtime to control TTA randomness per sampled face (e.g., new seed for each Monte Carlo draw). `tta_seed_mode` (`"content"` or `"random"`) controls the default seeding strategy when a seed is not supplied.
- Calibration
  - Apply learned CrossDomainCalibration layer to correct face→scene domain bias in the reference space.
  - Keep outputs in reference space `[-1, 1]`; use `EmotionScaleAligner` for any conversion to dataset/consumer scales (e.g., FindingEmo, DEAM) at boundaries.
- Output contract
  - `(valence: float, arousal: float, variance: tuple[float, float])` in reference space `[-1, 1]`

## File locations
- Vendored upstream code and weights (unmodified):
  - `models/emonet/` (see [project_overview.md](file:///Users/desmondchoy/Projects/emo-rec/docs/project_overview.md))
- Adapter (our code):
  - Suggested: `models/emonet_adapter.py` (or `models/face/emonet_adapter.py`)
  - Loads vendored package from `models/emonet/` and weights from `models/emonet/pretrained/`

## Minimal interface
```python
class EmoNetAdapter:
    def __init__(self,
                 ckpt_dir: str = "models/emonet/pretrained",
                 n_classes: int = 8,
                 device: str = "auto",
                 tta: int = 5,
                 tta_seed_mode: str = "content",
                 calibration_checkpoint: str | None = None):
        # Load trained CrossDomainCalibration layer
        from models.calibration import CrossDomainCalibration
        self.calibration = CrossDomainCalibration() if calibration_checkpoint else None
        if calibration_checkpoint:
            self.calibration.load_state_dict(torch.load(calibration_checkpoint))
        ...

    def predict(self,
                face_bgr: np.ndarray,
                tta: int | None = None,
                seed: int | None = None) -> tuple[float, float, tuple[float, float]]:
        """
        Returns (valence_ref, arousal_ref, (v_var, a_var)) in reference space [-1, 1].
        - Applies alignment, normalization, inference, TTA variance, and calibration (in reference space).
        - `seed` controls deterministic vs. stochastic TTA; runtime can pass a new seed per face sample.
        """
        ...
```

## Integration points
- PERCEIVE stage: after the face processor surfaces candidate crops (MediaPipe + Haar fallback), call `adapter.predict(face_crop, seed=...)` for each sampled face.
- FUSION: use returned mean and variance for inverse‑variance weighting with the scene model.
- STABILIZE: use variance for uncertainty gating.
- MATCH (POC): song-level retrieval with FE/DEAM conversions via `EmotionScaleAligner`;
  optional GMM station gating before simple k-NN.

## Calibration details
- Uses trained CrossDomainCalibration PyTorch module for learnable domain bias correction
- Training: 4-parameter affine transform learned on FindingEmo validation subset
  - `v_out = scale_v * v_in + shift_v`
  - `a_out = scale_a * a_in + shift_a`
- Load from checkpoint: `models/calibration/cross_domain_emonet_to_findingemo.pth`
- Statistical validation ensures calibration improves performance before deployment
- Outputs remain in reference space `[-1, 1]`; convert to FindingEmo/DEAM only where required via `EmotionScaleAligner`.

## Dependencies
- `mediapipe` (detection + eye keypoints for alignment)
- `torch`, `opencv-python`, `numpy`

## Notes on licensing
- EmoNet is CC BY‑NC‑ND 4.0. We vendor code and (optionally) unmodified checkpoints with attribution. No fine‑tuning or redistribution of modified weights.
- Keep `LICENSE.txt` and `NOTICE.txt` under `models/emonet/`.
