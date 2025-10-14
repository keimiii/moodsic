# Runtime Pipeline

- [✅] Enable MC Dropout/TT sampling for uncertainty in PERCEIVE
- [✅] Integrate scene–face fusion with variance-weighted averaging
- [✅] Serve MATCH recommendations via backend helpers (`backend/helpers/process_video.py`, `backend/helpers/song_recommendation.py`) using the VEATIC parquet exports
- [ ] Optional: Wire the runtime driver’s stabilized outputs directly into MATCH for live inference (current demo uses pre-generated VEATIC runs)

Extracted from [project_overview.md](file:///Users/desmondchoy/Projects/emo-rec/docs/project_overview.md).

## Overview

Three-stage runtime pipeline that converts video frames into song recommendations.

Current status (code):
- PERCEIVE implemented for scene and face experts; both provide uncertainty via sampling.
  - Scene: CLIP/ViT adapter with MC Dropout heads — `models/scene/clip_vit_scene_adapter.py`
  - Face: EmoNet adapter with TTA-based variance — `models/face/emonet_adapter.py`
- Fusion implemented with inverse-variance weighting and fallbacks — `models/fusion.py`
- Post-fusion STABILIZE (EMA + uncertainty gating) integrated as an optional component inside fusion — `models/fusion.py`
- Overlay/debug utilities available — `utils/fusion_overlay.py`
- Tests cover fusion math, gating, stabilizer behavior, overlay, and retrieval logic — `tests/test_fusion.py`, `tests/test_fusion_overlay.py`, `tests/test_song_matcher.py`
- MATCH retrieval surfaces in two layers:
  - `utils/song_matcher.py` retains the canonical GMM/dwell implementation for live inference.
  - `backend/helpers/process_video.py` and `backend/helpers/song_recommendation.py` load the VEATIC parquet exports (`results/inference/pipeline_results_*.parquet`) plus clustered DEAM catalog to return fused metrics, pathway MAE, comments, and songs to the React frontend.

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
  - Face path: MediaPipe drives detection; when fewer than `face_mc_samples` faces are found, an OpenCV Haar cascade fallback proposes additional candidates. Only genuinely new, non-overlapping faces are added—the fallback does not fabricate extra crops when none exist. We sample up to `face_mc_samples` crops (score-weighted), align them, and run EmoNet with stochastic TTA seeding so the returned variance captures both crop-level noise and inter-face disagreement.
  - Fusion: variance-weighted averaging when both paths available; fall back to scene-only when no face candidates remain. Per-face samples are stored for overlays/debugging.
  - Defaults: `scene_mc_samples=5`, `face_mc_samples=5`, `face_sampling="weighted"`, `face_tta=5`; outputs in reference space `[-1, 1]`.

- STABILIZE
  - Exponential Moving Average (EMA) over valence/arousal.
  - Uncertainty gating: if variance exceeds threshold, hold last stable values.
  - Implemented inside `SceneFaceFusion` (enable with `enable_stabilizer=True`). Defaults: `alpha=0.7`, `τ=0.4`, window `60`.
  - Note: `window_size` is for metrics history (variance/jitter) only; EMA smoothing/latency depends solely on α.

- MATCH (POC)
  - Implemented in `utils/song_matcher.py` as `SongMatcher` with GMM station gating, dwell-time enforcement, and recent-song memory.
  - Flask backend path: `backend/helpers/process_video.py` loads aggregated VEATIC metrics, infers clusters, and delegates to the same matching logic (`_pick_song`) so the React UI receives fused scores, per-pathway MAE, and song metadata without recomputing inference live.
  - Consumes valence/arousal already in reference space `[-1, 1]`; convert upstream when working directly with FE or DEAM static scales.
  - Optional: widen the candidate set to the top-2 GMM clusters when posterior confidence falls below the configured threshold.
  - Tests: `tests/test_song_matcher.py` covers gating, dwell enforcement, recent-history filtering, and artifact loading.

## Runtime Driver (PERCEIVE Orchestrator)

Purpose: Single place that coordinates PERCEIVE per frame, returning fused
valence/arousal and uncertainties to any frontend or the next stages.

- Location: `utils/runtime_driver.py`
- Provides `PerceiveFusionDriver` plus functional helpers `perceive_once` and `perceive_video` that wrap `SceneFaceFusion`.
- Depends on:
  - `utils/emonet_single_face_processor.EmoNetSingleFaceProcessor`
  - `models.face.emonet_adapter.EmoNetAdapter`
  - `models.fusion.SceneFaceFusion`
  - `utils/fusion_overlay.draw_fusion_overlay` (optional for annotation)

API:

```python
class PerceiveFusionDriver:
    def __init__(
        self,
        scene_predictor: Optional[object] = None,
        face_processor: Optional[EmoNetSingleFaceProcessor] = None,
        face_expert: Optional[EmoNetAdapter] = None,
        *,
        scene_mc_samples: int = 5,
        face_tta: int = 5,
        face_mc_samples: int = 5,
        face_sampling: str = "weighted",
        face_sampling_temperature: float = 1.0,
        face_sampling_seed: Optional[int] = None,
        face_tta_mode: str = "auto",
        use_variance_weighting: bool = True,
        scene_weight: float = 0.6,
        face_weight: float = 0.4,
        face_score_threshold: Optional[float] = None,
        face_max_sigma: Optional[float] = None,
        brightness_threshold: Optional[float] = None,
        enable_stabilizer: bool = False,
        stabilizer_alpha: float = 0.7,
        uncertainty_threshold: float = 0.4,
        stabilizer_window: int = 60,
        max_hz: float = 4.0,
        variance_floor: Optional[float] = 1e-3,
        max_weight_ratio: Optional[float] = None,
    ):
        ...  # assembles SceneFaceFusion internally

    def step(self, frame_bgr: np.ndarray) -> FusionResult:
        """Run PERCEIVE on a single BGR frame and fuse outputs."""

    def overlay(self, frame_bgr: np.ndarray, result: FusionResult) -> np.ndarray:
        """Draw optional debug overlay for UIs (calls draw_fusion_overlay)."""

    def reset(self) -> None: ...
```

- Behavior:
  - If `scene_predictor` is None, driver runs face-only and returns face results.
  - If face detection fails on a frame, falls back to scene-only when available.
  - All outputs in reference space `[-1, 1]`. Variances reflect TTA/MC sampling.
  - Throttling via `max_hz` controls how often PERCEIVE is executed; when throttled the driver reuses the last result.
  - Helper functions `perceive_once` and `perceive_video` expose the same wiring for single frames and full videos.

Scene model integration (later):
- Provide a `scene_predictor` implementing
  `predict(frame_bgr, tta:int) -> (v,a,(var_v,var_a))` in reference space.
- No changes to the driver or fusion core are required.
- A reference adapter is available: `models/scene/clip_vit_scene_adapter.py`.

Frontend note:
- The shipped Flask + React demo does not call the runtime driver live; instead it serves cached VEATIC perception runs from disk for deterministic playback. Use the driver when running new videos or benchmarking on-device.

TODO:
- [x] Add `utils/runtime_driver.py` with `PerceiveFusionDriver`.
- [x] Wire `SceneFaceFusion` inside driver (`step()`), pass sampling counts, and expose stabilizer controls.
- [ ] Optional: connect live MATCH inference (`SongMatcher`) to the driver so the backend can stream real-time recommendations instead of cached VEATIC exports.

References:
- Fusion core and stabilizer: `models/fusion.py`
- Face expert: `models/face/emonet_adapter.py`
- Face detection/cropping: `utils/emonet_single_face_processor.py`
- Scene adapter (CLIP/ViT with MC Dropout): `models/scene/clip_vit_scene_adapter.py`
- Overlay utility: `utils/fusion_overlay.py`
- Scale alignment utilities: `utils/emotion_scale_aligner.py`
- Retrieval core: `utils/song_matcher.py`
- Backend helpers serving the demo: `backend/helpers/process_video.py`, `backend/helpers/song_recommendation.py`
- Tests: `tests/test_fusion.py`, `tests/test_fusion_overlay.py`, `tests/test_song_matcher.py`
