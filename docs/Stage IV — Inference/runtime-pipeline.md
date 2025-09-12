# Runtime Pipeline

- [ ] Implement PERCEIVE → STABILIZE → MATCH end-to-end
- [ ] Enable MC Dropout for uncertainty in PERCEIVE
- [ ] Integrate scene–face fusion with variance-weighted averaging
- [ ] Wire stabilized outputs into k-NN music retrieval with dwell-time

Extracted from [project_overview.md](file:///Users/desmondchoy/Projects/emo-rec/docs/project_overview.md).

## Overview

Three-stage runtime pipeline that converts video frames into song recommendations.

```
[RUNTIME INFERENCE PIPELINE]

[Input Video]
     |
     v
+------------------------------------------+
| PERCEIVE: Extract V-A per frame         |
| Phase 0: Scene model predictions        |
| Phase 1: + Face detection, alignment & EmoNet (via adapter) |
| Phase 2: + Fusion of both paths         |
| + MC Dropout uncertainty estimation     |
+------------------------------------------+
     |
     v
+------------------------------------------+
| STABILIZE: Temporal smoothing           |
| - EMA (α-tuned, 3-5s window)           |
| - Uncertainty gating (hold if σ > τ)    |
| - Per-frame processing                  |
+------------------------------------------+
     |
     v
+------------------------------------------+
| MATCH: Song-level retrieval (POC)      |
| - Query per stabilized frame           |
| - GMM station gating (predict_proba)   |
|   - If top posterior < 0.55 → top-2    |
| - Linear-scan k-NN over DEAM static    |
| - Scale alignment (FE→DEAM static [1, 9]) |
| - Minimum dwell time (20-30s)          |
+------------------------------------------+
     |
     v
[Recommended Songs]
```

## Stage details

- PERCEIVE
  - Scene model: CLIP/ViT backbone, regression heads with dropout; MC Dropout for mean/variance.
  - Face path: single-face detection (MediaPipe), face alignment, and EmoNet inference via an adapter that handles preprocessing, calibration (EmoNet→FindingEmo), and optional TTA-based uncertainty.
  - Fusion: variance-weighted averaging when both paths available; fall back to scene-only when no face.

- STABILIZE
  - Exponential Moving Average (EMA) over valence/arousal.
  - Uncertainty gating: if variance exceeds threshold, hold last stable values.

- MATCH (POC)
  - Linear-scan k-NN over DEAM songs using static [1, 9] annotations.
  - Optional: apply GMM station gating from the DEAM clustering notebook
    (StandardScaler + GaussianMixture) before k-NN.
  - Enforce minimum dwell time and recent-song avoidance.
  - Use explicit FE→DEAM static [1, 9] scaling for queries.

## Runtime Driver (PERCEIVE Orchestrator)

Purpose: Single place that coordinates PERCEIVE per frame, returning fused
valence/arousal and uncertainties to any frontend or the next stages.

- Location (planned): `utils/runtime_driver.py`
- Depends on:
  - `utils/emonet_single_face_processor.EmoNetSingleFaceProcessor`
  - `models.face.emonet_adapter.EmoNetAdapter`
  - `models.fusion.SceneFaceFusion`
  - `utils/fusion_overlay.draw_fusion_overlay` (optional for annotation)

API (proposed):

```python
class PerceiveFusionDriver:
    def __init__(
        self,
        scene_predictor: Optional[object] = None,
        face_processor: Optional[EmoNetSingleFaceProcessor] = None,
        face_expert: Optional[EmoNetAdapter] = None,
        *,
        scene_mc_samples: int = 10,
        face_tta: int = 5,
        use_variance_weighting: bool = True,
        scene_weight: float = 0.6,
        face_weight: float = 0.4,
        max_hz: float = 4.0,  # throttle to avoid blocking UIs
    ):
        ...  # assemble SceneFaceFusion internally

    def step(self, frame_bgr: np.ndarray) -> FusionResult:
        """Run PERCEIVE on a single BGR frame and fuse outputs."""

    def overlay(self, frame_bgr: np.ndarray, result: FusionResult) -> np.ndarray:
        """Draw optional debug overlay for UIs (calls draw_fusion_overlay)."""

    def reset(self) -> None: ...
```

Behavior:
- If `scene_predictor` is None, driver runs face-only and returns face results.
- If face detection fails on a frame, falls back to scene-only when available.
- All outputs in reference space `[-1, 1]`. Variances reflect TTA/MC sampling.
- Throttling via `max_hz` controls how often PERCEIVE is executed in UIs.

Scene model integration (later):
- Provide a `scene_predictor` implementing
  `predict(frame_bgr, tta:int) -> (v,a,(var_v,var_a))` in reference space.
- No changes to the driver or fusion core are required.

TODO:
- [ ] Add `utils/runtime_driver.py` with `PerceiveFusionDriver` skeleton.
- [ ] Connect STABILIZE (EMA + uncertainty gating) after `step()`.
- [ ] Connect MATCH (DEAM scaling + linear-scan k-NN; optional GMM gating) after stabilization.
