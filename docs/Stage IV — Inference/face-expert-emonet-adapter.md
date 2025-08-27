# Face Expert: EmoNet Adapter

Status: design and integration guide for the face pathway

## Purpose
A small adapter that exposes EmoNet as a drop‑in "face expert" to the runtime pipeline. It hides model loading, preprocessing, calibration, and uncertainty so PERCEIVE/FUSION can use a simple, stable API.

## Responsibilities
- Model management
  - Load EmoNet checkpoints from `models/emonet/pretrained/` (5‑class or 8‑class model).
  - Select device (CPU/GPU), set `eval()` and `no_grad()`.
- Preprocessing
  - Input: an RGB/BGR face crop (from the primary face selection).
  - Face alignment (via the `face-alignment` library) to match EmoNet’s expected input.
  - Resize and apply EmoNet’s normalization exactly as in upstream demos.
- Inference
  - Run forward pass to get continuous valence and arousal (optionally expression class if needed).
- Uncertainty (optional, recommended)
  - Test‑time augmentation (horizontal flip; minor crop/scale jitter) for N passes; return mean and per‑dimension variance for V and A.
- Calibration
  - Apply learned CrossDomainCalibration layer to correct face→scene domain bias.
  - Convert EmoNet outputs → FindingEmo-aligned space for downstream FE→DEAM scaling.
- Output contract
  - `(valence_fe: float, arousal_fe: float, variance: tuple[float, float])`

## File locations
- Vendored upstream code and weights (unmodified):
  - `models/emonet/` (see [project_overview.md](file:///Users/desmondchoy/Projects/emo-rec/docs/project_overview.md))
- Adapter (our code):
  - Suggested: `models/emonet_adapter.py` (or `models/face/emonet_adapter.py`)
  - Loads vendored package from `models/emonet/emonet/` and weights from `models/emonet/pretrained/`

## Minimal interface
```python
class EmoNetAdapter:
    def __init__(self,
                 ckpt_dir: str = "models/emonet/pretrained",
                 n_classes: int = 8,
                 device: str = "auto",
                 tta: int = 5,
                 calibration_checkpoint: str | None = None):
        # Load trained CrossDomainCalibration layer
        from models.calibration import CrossDomainCalibration
        self.calibration = CrossDomainCalibration() if calibration_checkpoint else None
        if calibration_checkpoint:
            self.calibration.load_state_dict(torch.load(calibration_checkpoint))
        ...

    def predict(self, face_bgr: np.ndarray) -> tuple[float, float, tuple[float, float]]:
        """
        Returns (valence_fe, arousal_fe, (v_var, a_var)) in FindingEmo ranges.
        - Applies alignment, normalization, inference, TTA variance, and calibration.
        """
        ...
```

## Integration points
- PERCEIVE stage: after MediaPipe selects the primary face, call `adapter.predict(face_crop)`.
- FUSION: use returned mean and variance for inverse‑variance weighting with the scene model.
- STABILIZE: use variance for uncertainty gating.
- MATCH: unchanged; FE→DEAM scaling already implemented.

## Calibration details
- Uses trained CrossDomainCalibration PyTorch module for learnable domain bias correction
- Training: 4-parameter affine transform learned on FindingEmo validation subset
  - `v_out = scale_v * v_in + shift_v`
  - `a_out = scale_a * a_in + shift_a`
- Load from checkpoint: `models/calibration/cross_domain_emonet_to_findingemo.pth`
- Statistical validation ensures calibration improves performance before deployment
- Outputs aligned to FindingEmo ranges for downstream FE→DEAM scaling

## Dependencies
- `face-alignment` (alignment)
- `torch`, `opencv-python`, `numpy`

## Notes on licensing
- EmoNet is CC BY‑NC‑ND 4.0. We vendor code and (optionally) unmodified checkpoints with attribution. No fine‑tuning or redistribution of modified weights.
- Keep `LICENSE.txt` and `NOTICE.txt` under `models/emonet/`.
