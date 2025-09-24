"""
EmoNetAdapter — usage & quickstart

Environment setup (fish shell):
- Activate virtualenv: `source .venv/bin/activate.fish`
- Install deps: `uv pip install torch mediapipe opencv-python numpy`

Checkpoints:
- Place EmoNet weights in `models/emonet/pretrained/` (e.g., `emonet_8.pth`).
- The adapter picks a file matching `_{n_classes}.pth`, else the first `*.pth`.

Quick test in Python:
    >>> import cv2
    >>> from models.face import EmoNetAdapter
    >>> from utils.emonet_single_face_processor import EmoNetSingleFaceProcessor
    >>> frame = cv2.imread("data/Run_1/some_image.jpg")
    >>> face_proc = EmoNetSingleFaceProcessor()
    >>> face, bbox, score = face_proc.extract_primary_face(frame)
    >>> if face is not None:
    ...     adapter = EmoNetAdapter(n_classes=8, device="auto")
    ...     v, a, (vv, va) = adapter.predict(face)
    ...     print(f"v={v:.3f}, a={a:.3f}, var=({vv:.4f},{va:.4f})")
    ... else:
    ...     print("No face detected")

Calibration (optional):
- Provide `calibration_checkpoint` trained for reference-space alignment, e.g.:
  `models/emonet/evaluation/results/calibration_emonet2findingemo.pt`

Outputs:
- `predict()` returns calibrated (valence, arousal) in reference space [-1, 1]
  and per-dimension variance. Task 3 returns zero variance; Task 4 adds TTA.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover - torch may be missing in some envs
    torch = None  # type: ignore
    logging.getLogger(__name__).warning("PyTorch not available: %s", e)

# Local utilities
from utils.emotion_scale_aligner import EmotionScaleAligner


class EmoNetAdapter:
    """
    Production adapter exposing EmoNet as a face expert.

    Responsibilities:
    - Load vendored EmoNet weights from `models/emonet/pretrained/`.
    - Select device (CUDA > MPS > CPU) and set model to eval.
    - Preprocess BGR face crops: optional eye-level alignment via MediaPipe; resize; normalize.
    - Predict continuous valence/arousal in reference space [-1, 1].
    - Optional cross-domain calibration in reference space.

    Notes on licensing:
    - Uses vendored EmoNet under CC BY-NC-ND 4.0 (see `models/emonet/`).

    Output contract:
    - predict() returns (valence_ref, arousal_ref, (var_v, var_a)) in [-1, 1].
      Variances are ~0 when TTA is effectively disabled. TTA-based uncertainty
      beyond the identity pass is implemented in Task 4.
    """

    def __init__(
        self,
        ckpt_dir: str = "models/emonet/pretrained",
        n_classes: int = 8,
        device: str = "auto",
        tta: int = 5,
        calibration_checkpoint: Optional[str] = None,
    ) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required for EmoNetAdapter. Please install torch."
            )

        self.n_classes = int(n_classes)
        self.tta_default = int(tta)
        self.device = self._select_device(device)
        self.scale_aligner = EmotionScaleAligner(strict=False)

        # Prepare vendored EmoNet import from models/emonet
        self._emonet = self._load_emonet_model(ckpt_dir)
        self._emonet.eval()
        self._emonet.to(self.device)

        # Optional calibration layer (applied in reference space)
        self.calibration = None
        if calibration_checkpoint:
            try:
                from models.calibration import CrossDomainCalibration

                self.calibration = CrossDomainCalibration()
                state = torch.load(calibration_checkpoint, map_location="cpu")
                self.calibration.load_state_dict(state)
                self.calibration.eval()
                self.calibration.to(self.device)
            except Exception as e:  # pragma: no cover - env dependent
                logging.getLogger(__name__).warning(
                    "Failed to load calibration from %s: %s",
                    calibration_checkpoint,
                    e,
                )
                self.calibration = None

        # Lazy MediaPipe detector for alignment
        self._mp = None
        self._mp_detector = None

    # ---------- Public API ----------
    def predict(
        self, face_bgr: np.ndarray, tta: Optional[int] = None
    ) -> Tuple[float, float, Tuple[float, float]]:
        """
        Run EmoNet inference on a single BGR face crop.

        Args:
            face_bgr: Face image in BGR format (H, W, 3).
            tta: Optional override for test-time augmentation passes. For Task 3,
                 TTA beyond the identity pass is ignored and variance is ~0.

        Returns:
            (valence_ref, arousal_ref, (var_v, var_a)) in reference space [-1, 1].
        """
        if (
            face_bgr is None
            or not isinstance(face_bgr, np.ndarray)
            or face_bgr.ndim != 3
            or face_bgr.shape[2] != 3
        ):
            # Invalid input: return neutral with zero variance
            return 0.0, 0.0, (0.0, 0.0)

        # Effective TTA count
        n_tta = self.tta_default if tta is None else int(tta)
        if n_tta <= 1:
            n_tta = 1

        # Preprocess base: eye-level alignment then canonical 256x256
        aligned_bgr = self._align_face_by_eyes_mediapipe(face_bgr)
        base_bgr_256 = cv2.resize(aligned_bgr, (256, 256))

        # Build augmented variants (always include original; include flip; add scale/crop jitter)
        batch_chw = self._build_tta_batch(base_bgr_256, n_tta)

        # Convert to torch batch
        tensor = torch.from_numpy(batch_chw).to(self.device)

        with torch.no_grad():
            out = self._emonet(tensor)
            # Model outputs (N,) in raw EmoNet space (already ~[-1,1])
            v_raw = out["valence"].view(-1)
            a_raw = out["arousal"].view(-1)

            # Reference-space clamp [-1, 1]
            v_ref_t = torch.clamp(v_raw, -1.0, 1.0)
            a_ref_t = torch.clamp(a_raw, -1.0, 1.0)

            # Optional calibration in reference space
            if self.calibration is not None:
                v_ref_t, a_ref_t = self.calibration(v_ref_t, a_ref_t)
                # Ensure final bounds
                v_ref_t = torch.clamp(v_ref_t, -1.0, 1.0)
                a_ref_t = torch.clamp(a_ref_t, -1.0, 1.0)

        # Move to CPU numpy for aggregation
        v_np = v_ref_t.detach().cpu().numpy().astype(np.float32)
        a_np = a_ref_t.detach().cpu().numpy().astype(np.float32)

        # Aggregate mean and unbiased variance
        mean_v = float(np.mean(v_np))
        mean_a = float(np.mean(a_np))
        if v_np.size > 1:
            var_v = float(np.var(v_np, ddof=1))
            var_a = float(np.var(a_np, ddof=1))
        else:
            var_v, var_a = 0.0, 0.0

        return mean_v, mean_a, (var_v, var_a)

    # ---------- Internals ----------
    def _select_device(self, device: str):
        d = (device or "auto").lower()
        if d == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        try:
            return torch.device(d)
        except Exception:
            logging.getLogger(__name__).warning("Unknown device '%s', falling back to CPU", device)
            return torch.device("cpu")

    def _load_emonet_model(self, ckpt_dir: str):
        """Import vendored EmoNet and load weights from ckpt_dir."""
        # Ensure the vendored package is importable: add models/ to sys.path
        project_root = Path(__file__).resolve().parents[2]
        models_root = project_root / "models"
        if str(models_root) not in sys.path:
            sys.path.insert(0, str(models_root))

        try:
            from emonet.models.emonet import EmoNet  # type: ignore
        except Exception as e:
            raise ImportError(
                f"Failed to import vendored EmoNet from {models_root}: {e}"
            )

        model = EmoNet(n_expression=self.n_classes)
        model.eval()

        # Load checkpoint — resolve relative paths from callers (e.g., notebooks/)
        ckpt_dir_path = Path(ckpt_dir)
        if not ckpt_dir_path.is_absolute() and not ckpt_dir_path.exists():
            resolved = (project_root / ckpt_dir_path).resolve()
            if resolved.exists():
                ckpt_dir_path = resolved

        ckpt_path = self._select_checkpoint(ckpt_dir_path)
        if ckpt_path is not None and ckpt_path.exists():
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Failed to load EmoNet weights from %s: %s (using random init)",
                    ckpt_path,
                    e,
                )
        else:
            logging.getLogger(__name__).warning(
                "No EmoNet checkpoint found in %s; using random initialization.",
                ckpt_dir_path,
            )

        return model

    def _select_checkpoint(self, ckpt_dir: Path) -> Optional[Path]:
        """
        Select a checkpoint file from directory.
        Preference: contains f"_{n_classes}.pth", else first *.pth by name.
        """
        if not ckpt_dir.exists():
            return None
        candidates = sorted(ckpt_dir.glob("*.pth"))
        if not candidates:
            return None
        # Prefer file containing _{n_classes}.pth
        preferred = [p for p in candidates if f"_{self.n_classes}.pth" in p.name]
        return preferred[0] if preferred else candidates[0]

    # --------- Alignment via MediaPipe eye keypoints ---------
    def _ensure_mediapipe(self) -> bool:
        if self._mp_detector is not None:
            return True
        try:
            import mediapipe as mp  # type: ignore

            self._mp = mp
            # Lightweight face detection with keypoints
            self._mp_detector = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=0.5
            )
            return True
        except Exception as e:  # pragma: no cover - optional dependency
            logging.getLogger(__name__).debug(
                "MediaPipe not available for alignment: %s", e
            )
            self._mp = None
            self._mp_detector = None
            return False

    def _align_face_by_eyes_mediapipe(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Rotate the face to level the inter-ocular line using MediaPipe keypoints.
        Falls back to the original crop when keypoints are not available.
        Always returns a valid BGR image (resized to 256x256 downstream).
        """
        h, w = face_bgr.shape[:2]
        if h == 0 or w == 0:
            return face_bgr

        if not self._ensure_mediapipe():
            # No alignment if MediaPipe is unavailable
            return face_bgr

        try:
            # MediaPipe expects RGB
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            results = self._mp_detector.process(rgb)
            detections = getattr(results, "detections", None)
            if not detections:
                return face_bgr

            det = detections[0]
            # Relative keypoints for left/right eyes
            rel_kps = getattr(det.location_data, "relative_keypoints", [])
            # MediaPipe FaceDetection returns 6 keypoints in a fixed order
            # 0: RIGHT_EYE, 1: LEFT_EYE, 2: NOSE_TIP, 3: MOUTH_CENTER, 4: RIGHT_EAR, 5: LEFT_EAR
            if len(rel_kps) < 2:
                return face_bgr

            right_eye = rel_kps[0]
            left_eye = rel_kps[1]
            rx = float(getattr(right_eye, "x", 0.5)) * w
            ry = float(getattr(right_eye, "y", 0.5)) * h
            lx = float(getattr(left_eye, "x", 0.5)) * w
            ly = float(getattr(left_eye, "y", 0.5)) * h

            dx = lx - rx
            dy = ly - ry
            if abs(dx) < 1e-3 and abs(dy) < 1e-3:
                return face_bgr

            angle = math.degrees(math.atan2(dy, dx))
            # Rotate around image center so that the eye line is horizontal
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                face_bgr,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            return rotated
        except Exception as e:  # pragma: no cover - robustness
            logging.getLogger(__name__).debug(
                "Alignment failed; returning unaligned crop: %s", e
            )
            return face_bgr

    # --------- TTA helpers ---------
    def _build_tta_batch(self, base_bgr_256: np.ndarray, n_tta: int) -> np.ndarray:
        """
        Build a batch of CHW images in [0,1] with TTA variants:
        - Always include original
        - Include horizontal flip of the original
        - Fill remaining with small scale/crop jitter variants

        Returns: np.ndarray of shape (N, 3, 256, 256), dtype float32
        """
        # Ensure base is 256x256 BGR
        if base_bgr_256.shape[:2] != (256, 256):
            base_bgr_256 = cv2.resize(base_bgr_256, (256, 256))

        variants: list[np.ndarray] = []
        variants.append(base_bgr_256)

        if n_tta >= 2:
            variants.append(cv2.flip(base_bgr_256, 1))  # horizontal flip

        # Deterministic RNG seed per call, derived from image content
        # Using uint32 sum keeps it fast and stable across runs
        seed = int(np.uint32(base_bgr_256.sum()))
        rng = np.random.RandomState(seed)

        # Fill remaining slots with scale/crop jitter
        needed = max(0, n_tta - len(variants))
        for _ in range(needed):
            # Scale factor in [0.96, 1.04]
            scale = 1.0 + (rng.rand() * 0.08 - 0.04)
            jittered = self._scale_jitter_center(base_bgr_256, scale)
            variants.append(jittered)

        # Convert to RGB, normalize to [0,1], CHW, and stack
        batch = []
        for bgr in variants[:n_tta]:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            arr = rgb.astype(np.float32) / 255.0
            chw = np.transpose(arr, (2, 0, 1))
            batch.append(chw)
        return np.stack(batch, axis=0).astype(np.float32)

    def _scale_jitter_center(self, img_256: np.ndarray, scale: float) -> np.ndarray:
        """
        Apply scale jitter about the center, returning a 256x256 BGR image.
        - If scale > 1: zoom in by cropping a smaller center region and resizing back.
        - If scale < 1: zoom out by shrinking and padding reflectively back to 256.
        """
        h, w = img_256.shape[:2]
        assert h == 256 and w == 256, "_scale_jitter_center expects 256x256 input"

        # Guard extreme cases
        scale = float(max(0.90, min(1.10, scale)))

        new_size = max(1, int(round(256 * scale)))
        resized = cv2.resize(img_256, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
        if new_size == 256:
            return resized
        if new_size > 256:
            # Center crop to 256
            start = (new_size - 256) // 2
            end = start + 256
            return resized[start:end, start:end]
        else:
            # Pad to 256 with reflection, then center crop (in case of off-by-one)
            pad_total = 256 - new_size
            top = pad_total // 2
            bottom = pad_total - top
            left = pad_total // 2
            right = pad_total - left
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101
            )
            if padded.shape[0] != 256 or padded.shape[1] != 256:
                # Final safety crop
                y0 = max(0, (padded.shape[0] - 256) // 2)
                x0 = max(0, (padded.shape[1] - 256) // 2)
                padded = padded[y0 : y0 + 256, x0 : x0 + 256]
            return padded


__all__ = ["EmoNetAdapter"]


def _cli() -> int:
    """Lightweight CLI for single-image inference with TTA."""
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description="Run EmoNetAdapter on an image")
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--tta", type=int, default=5, help="Number of TTA samples")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device selection",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Optional path to CrossDomainCalibration checkpoint",
    )
    parser.add_argument(
        "--min-det-conf",
        type=float,
        default=0.5,
        help="MediaPipe min detection confidence",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.2,
        help="Padding ratio around detected bbox",
    )
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Failed to read image: {args.image}")
        return 1

    try:
        from utils.emonet_single_face_processor import EmoNetSingleFaceProcessor
    except Exception as e:
        print(f"Failed to import face processor: {e}")
        return 1

    face_proc = EmoNetSingleFaceProcessor(
        min_detection_confidence=args.min_det_conf, padding_ratio=args.padding
    )
    face, bbox, score = face_proc.extract_primary_face(img)
    if face is None:
        print("No face detected")
        return 2

    adapter = EmoNetAdapter(
        # default head size is 8 in the adapter; prefer 8-class weights
        device=args.device,
        tta=args.tta,
        calibration_checkpoint=args.calibration,
    )
    v, a, (v_var, a_var) = adapter.predict(face)
    print(
        f"valence={v:.3f} arousal={a:.3f} v_var={v_var:.4f} a_var={a_var:.4f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
