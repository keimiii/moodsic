# Stage IV — EmoNet Face Expert: Missing Tasks Plan

This document lists all remaining implementation tasks for the EmoNet face-expert adapter and related runtime pieces. For each task, it states what needs to be done, why it matters, and how to implement it.

References:
- docs/Stage IV — Inference/face-expert-emonet-adapter.md
- docs/Stage II — Modeling/face-model.md
- docs/Stage IV — Inference/runtime-pipeline.md
- utils/emotion_scale_aligner.py
- models/calibration/

---

## 1) Single-Face Processor (MediaPipe)

- What: Implement a reusable component to detect faces and extract the primary face crop from a BGR frame using MediaPipe.
- Why: Centralizes detection logic for runtime inference (PERCEIVE); enables consistent selection of the main face and graceful no-face handling. Current detection lives only in an evaluation script.
- How:
  - Create `utils/emonet_single_face_processor.py` with class `EmoNetSingleFaceProcessor`.
  - Use `mediapipe.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)`.
  - Strategy to pick primary face: score × sqrt(area) × (0.7 + 0.3 × center_proximity) as documented in face-model.md.
  - Return: `(face_crop_bgr: np.ndarray | None, bbox: tuple[int,int,int,int] | None, score: float)`.
  - Conversions: input frame is BGR; convert to RGB for MediaPipe; crop from original BGR; resize crop to 256×256 (pre-alignment) or leave original for alignment step.
  - Handle no detections by returning `(None, None, 0.0)`.
  - Optional: small padding around the bbox (e.g., 20%).

Acceptance criteria:
- A single `extract_primary_face(frame_bgr)` method returns the crop or `None` without exceptions across varied inputs.

---

## 2) Face Alignment (face-alignment library)

- What: Add a face alignment step to normalize the detected face prior to EmoNet.
- Why: EmoNet upstream assumes aligned faces; alignment improves robustness and matches expected input distribution.
- How:
  - Add dependency: `face-alignment` (1adrianb) and its prerequisites (PyTorch present elsewhere).
  - Implement `align_face(face_bgr) -> np.ndarray` in a helper (either inside the adapter or a new `utils/emonet_face_alignment_utils.py`).
  - Instantiate `face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)` once and reuse.
  - Detect landmarks on the 256×256 (or original) crop; compute similarity transform to align eyes horizontally; warp to 256×256.
  - Fallback: if landmarks not found, return the resized crop (no alignment) and log a debug warning.

Acceptance criteria:
- For valid faces, produces an aligned 256×256 BGR image; degrades gracefully when landmarks fail.

---

## 3) EmoNetAdapter (loading, preprocessing, inference, calibration)

- What: Implement a production adapter that exposes EmoNet as a face expert with a stable API.
- Why: Decouples model details from the runtime pipeline; ensures consistent preprocessing, calibration, and output scaling.
- How:
  - File: `models/face/emonet_adapter.py` (or `models/emonet_adapter.py`; prefer the former).
  - Class signature:
    - `__init__(ckpt_dir="models/emonet/pretrained", n_classes=8, device="auto", tta=5, calibration_checkpoint: str | None = None)`
    - `predict(face_bgr: np.ndarray, tta: int | None = None) -> tuple[float, float, tuple[float, float]]`
  - Loading:
    - Import vendored `emonet` package from `models/emonet/emonet`.
    - Construct `EmoNet(n_expression=n_classes)`, load checkpoint from `ckpt_dir` (first `*.pth` if multiple), set `eval()`.
    - Device selection: prefer CUDA, else MPS (Apple), else CPU.
  - Preprocessing:
    - Input: face crop BGR from `EmoNetSingleFaceProcessor`.
    - Convert BGR→RGB; apply alignment (see Task 2); resize to 256×256; convert to float in [0,1]; CHW; add batch dim; torch tensor on device.
  - Inference I/O:
    - Forward pass returns dict with `valence` and `arousal`; cast to `float`.
    - Keep outputs in the common reference space [-1, 1] by default.
    - Optionally support `output_scale` (e.g., `'reference'|'findingemo'|'deam_static'`; default `'reference'`) to convert via `utils/emotion_scale_aligner.EmotionScaleAligner` when needed.
  - Calibration:
    - If `calibration_checkpoint` provided, load `models.calibration.CrossDomainCalibration` and apply in reference space. No conversion is applied by default.

Acceptance criteria:
- `predict()` returns `(v_ref, a_ref, (var_v, var_a))` in reference space [-1, 1]. When TTA is disabled, `var_*` should be near zero.

---

## 4) TTA-Based Uncertainty (flip + crop/scale jitter)

- What: Add lightweight test-time augmentation to produce mean and variance for valence/arousal predictions.
- Why: Provides uncertainty estimates for inverse-variance fusion and gating; required by runtime pipeline.
- How:
  - In `EmoNetAdapter.predict()`, run N stochastic passes (default N=5) over deterministic EmoNet by augmenting inputs:
    - Always include the original crop.
    - Horizontal flip variant (mirror); invert back if needed.
    - Minor scale/crop jitter (e.g., random 0–4% scale, center crop to 256) with fixed RNG seed per call for reproducibility.
  - Aggregate: compute per-dimension mean and unbiased variance over reference-space outputs ([-1, 1]).
  - Return `(mean_v, mean_a, (var_v, var_a))`.

Acceptance criteria:
- With `tta=1`, variance is ~0. With `tta>1`, variance reflects input perturbations and remains finite and stable.

---

## 5) Runtime Integration (PERCEIVE/FUSION)

- What: Wire the face expert into the inference pipeline and optional UI.
- Why: Enable end-to-end usage: scene + face fusion with uncertainty and graceful no-face handling.
- How:
  - Implement `models/fusion.py` with `SceneFaceFusion` from docs (inverse-variance weighting; fallback to fixed weights when variances unavailable).
  - Integrate in runtime pipeline driver (if applicable) and/or `app.py` as a debug overlay:
    - Instantiate `EmoNetSingleFaceProcessor`, `EmoNetAdapter`, and scene model.
    - Ensure both scene and face predictions are in reference space [-1, 1] prior to fusion.
    - On each frame: get scene mean/var via MC Dropout; get face `(v,a,var)` via adapter; fuse; optionally draw `(v,a)` and uncertainties on the video stream; convert to FE/DEAM only for reporting or downstream matching as required.
  - Ensure no-face path falls back to scene-only.

Acceptance criteria:
- Fusion returns stable predictions; no-face path works; optional UI overlay shows numbers without blocking the stream.

---

## 6) Dependencies & Environment

- What: Add and verify required packages for the adapter path.
- Why: Ensure reproducible installs and working runtime.
- How:
  - Add to `requirements.txt` (or project’s preferred lock):
    - `mediapipe` (face detection)
    - `face-alignment` (alignment)
    - `torch`, `torchvision` (if not already managed elsewhere)
  - Confirm guidance in docs to use `uv pip install <package>` and to activate the virtualenv via `source .venv/bin/activate.fish`.

Acceptance criteria:
- Fresh environment can install and import all dependencies used by Tasks 1–5.

---

## Notes on Licensing (FYI)

- EmoNet licensing and attribution are already satisfied in `models/emonet/` (LICENSE.txt, NOTICE.txt). No additional changes required for these tasks; ensure the adapter docstring notes EmoNet usage and points to that directory.

---

## Suggested File Layout

 - utils/emonet_single_face_processor.py
- utils/emonet_face_alignment_utils.py (optional if alignment is not in the adapter)
- models/face/emonet_adapter.py
- models/fusion.py

---

## Quick Implementation Checklist

- [ ] EmoNetSingleFaceProcessor (MediaPipe) in utils/
- [ ] Face alignment helper or inline adapter integration
- [ ] EmoNetAdapter with calibration + scale alignment
- [ ] TTA (flip + crop/scale jitter) and variance aggregation
- [ ] Fusion module and runtime integration
- [ ] Update dependencies; confirm environment notes
