"""
PerceiveFusionDriver — runtime orchestrator thin wrapper over models.fusion.

Purpose:
- Coordinate PERCEIVE per frame: face detection + EmoNet adapter (and later
  scene predictor) and fuse outputs.
- Provide a small, UI-agnostic API that any frontend can call.

Documentation:
- See docs/Stage IV — Inference/runtime-pipeline.md ("Runtime Driver" section)
  for behavior, integration notes, and TODOs for STABILIZE and MATCH stages.

Notes:
- This module intentionally delegates fusion logic to models.fusion.SceneFaceFusion
  to avoid code duplication. It exposes a small driver and a one-shot helper.
"""

from __future__ import annotations

from typing import Optional

# Import types with a soft fallback to keep import-time light in constrained envs
try:  # pragma: no cover - optional typing convenience
    from models.fusion import FusionResult, SceneFaceFusion  # type: ignore
except Exception:  # pragma: no cover
    FusionResult = object  # type: ignore
    SceneFaceFusion = object  # type: ignore


class PerceiveFusionDriver:
    """Thin wrapper that owns a SceneFaceFusion and optional throttling.

    This driver is UI-agnostic. It forwards configuration to
    models.fusion.SceneFaceFusion and provides:
    - step(frame): returns FusionResult from perceive_and_fuse.
    - overlay(frame, result): draws a debug overlay when OpenCV is available.
    - reset(): resets internal throttling and reinitializes fusion if needed.
    """

    def __init__(
        self,
        scene_predictor: Optional[object] = None,
        face_processor: Optional[object] = None,
        face_expert: Optional[object] = None,
        *,
        scene_mc_samples: int = 5,
        face_tta: int = 5,
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
        # Variance-weighted fusion guardrails
        variance_floor: Optional[float] = 1e-3,
        max_weight_ratio: Optional[float] = None,
    ) -> None:
        # Provide sensible defaults if not supplied
        scene_predictor, face_processor, face_expert = _ensure_default_components(
            scene_predictor, face_processor, face_expert
        )

        # Store dependencies and config for possible reset/rebuild
        self._deps = dict(
            scene_predictor=scene_predictor,
            face_processor=face_processor,
            face_expert=face_expert,
        )
        self._cfg = dict(
            scene_mc_samples=int(scene_mc_samples),
            face_tta=int(face_tta),
            use_variance_weighting=bool(use_variance_weighting),
            scene_weight=float(scene_weight),
            face_weight=float(face_weight),
            face_score_threshold=face_score_threshold,
            face_max_sigma=face_max_sigma,
            brightness_threshold=brightness_threshold,
            enable_stabilizer=bool(enable_stabilizer),
            stabilizer_alpha=float(stabilizer_alpha),
            uncertainty_threshold=float(uncertainty_threshold),
            stabilizer_window=int(stabilizer_window),
            variance_floor=variance_floor,
            max_weight_ratio=max_weight_ratio,
        )
        self._interval = 0.0 if max_hz <= 0 else (1.0 / float(max_hz))
        self._last_ts = 0.0
        self._last_res: Optional[FusionResult] = None

        # Build fusion
        self._fusion = SceneFaceFusion(
            scene_predictor=scene_predictor,
            face_expert=face_expert,
            face_processor=face_processor,
            scene_mc_samples=self._cfg["scene_mc_samples"],
            face_tta=self._cfg["face_tta"],
            use_variance_weighting=self._cfg["use_variance_weighting"],
            scene_weight=self._cfg["scene_weight"],
            face_weight=self._cfg["face_weight"],
            face_score_threshold=self._cfg["face_score_threshold"],
            face_max_sigma=self._cfg["face_max_sigma"],
            brightness_threshold=self._cfg["brightness_threshold"],
            enable_stabilizer=self._cfg["enable_stabilizer"],
            stabilizer_alpha=self._cfg["stabilizer_alpha"],
            uncertainty_threshold=self._cfg["uncertainty_threshold"],
            stabilizer_window=self._cfg["stabilizer_window"],
            variance_floor=self._cfg["variance_floor"],
            max_weight_ratio=self._cfg["max_weight_ratio"],
        )

    def step(self, frame_bgr: np.ndarray) -> FusionResult:
        """Run PERCEIVE on one BGR frame and return FusionResult.

        Respects basic throttling via max_hz by returning the last result if
        called too frequently.
        """
        import time

        now = time.monotonic()
        if self._interval > 0 and (now - self._last_ts) < self._interval and self._last_res is not None:
            return self._last_res
        res: FusionResult = self._fusion.perceive_and_fuse(frame_bgr)
        self._last_res = res
        self._last_ts = now
        return res

    def overlay(self, frame_bgr: np.ndarray, result: FusionResult) -> np.ndarray:
        """Return a copy of frame with a debug overlay, if available."""
        try:
            from utils.fusion_overlay import draw_fusion_overlay  # type: ignore

            return draw_fusion_overlay(frame_bgr, result)
        except Exception:
            # Best-effort: return original on environments without OpenCV
            return frame_bgr

    def reset(self) -> None:
        """Reset throttling and rebuild the fusion module with the same config."""
        self._last_ts = 0.0
        self._last_res = None
        # Re-resolve defaults in case environment changed (e.g., mediapipe installed later)
        sp, fp, fe = _ensure_default_components(
            self._deps["scene_predictor"], self._deps["face_processor"], self._deps["face_expert"]
        )
        self._deps.update(scene_predictor=sp, face_processor=fp, face_expert=fe)
        self._fusion = SceneFaceFusion(
            scene_predictor=sp,
            face_expert=fe,
            face_processor=fp,
            **self._cfg,
        )


__all__ = ["PerceiveFusionDriver"]


# ---------------- Minimal functional helper ----------------

# One-shot helper that delegates to SceneFaceFusion and returns FusionResult


def perceive_once(
    frame_bgr: np.ndarray,
    *,
    scene_predictor: Optional[object] = None,
    face_processor: Optional[object] = None,
    face_expert: Optional[object] = None,
    scene_tta: int = 5,
    face_tta: int = 5,
    # Fusion behavior
    use_variance_weighting: bool = True,
    scene_weight: float = 0.6,
    face_weight: float = 0.4,
    # Optional guardrails for the face path
    face_score_threshold: Optional[float] = None,
    face_max_sigma: Optional[float] = None,
    brightness_threshold: Optional[float] = None,
    # Optional post-fusion stabilizer
    enable_stabilizer: bool = False,
    stabilizer_alpha: float = 0.7,
    uncertainty_threshold: float = 0.4,
    stabilizer_window: int = 60,
    # Variance-weighted fusion guardrails
    variance_floor: Optional[float] = 1e-3,
    max_weight_ratio: Optional[float] = None,
) -> FusionResult:
    """
    Run a PERCEIVE flow on a single BGR frame via models.fusion.SceneFaceFusion.

    - Delegates to SceneFaceFusion for scene/face inference and fusion.
    - Returns fused V/A and per-path readings (when available) in reference
      space [-1, 1].
    """
    # Import locally to avoid heavy import at module import time
    try:
        from models.fusion import SceneFaceFusion, FusionResult  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("models.fusion is unavailable; cannot run perceive_once") from e

    # Default components if not provided
    scene_predictor, face_processor, face_expert = _ensure_default_components(
        scene_predictor, face_processor, face_expert
    )

    fusion = SceneFaceFusion(
        scene_predictor=scene_predictor,
        face_expert=face_expert,
        face_processor=face_processor,
        scene_mc_samples=scene_tta,
        face_tta=face_tta,
        use_variance_weighting=use_variance_weighting,
        scene_weight=scene_weight,
        face_weight=face_weight,
        face_score_threshold=face_score_threshold,
        face_max_sigma=face_max_sigma,
        brightness_threshold=brightness_threshold,
        enable_stabilizer=enable_stabilizer,
        stabilizer_alpha=stabilizer_alpha,
        uncertainty_threshold=uncertainty_threshold,
        stabilizer_window=stabilizer_window,
        variance_floor=variance_floor,
        max_weight_ratio=max_weight_ratio,
    )

    return fusion.perceive_and_fuse(frame_bgr)


# ----------- Internals: default components -----------------

def _ensure_default_components(
    scene_predictor: Optional[object],
    face_processor: Optional[object],
    face_expert: Optional[object],
):
    """
    Instantiate default adapters when not provided:
    - Scene: models.scene.clip_vit_scene_adapter.SceneCLIPAdapter
    - Face processor: utils.emonet_single_face_processor.EmoNetSingleFaceProcessor
    - Face expert: models.face.emonet_adapter.EmoNetAdapter

    Any failures (e.g., missing deps, no weights, no network) are swallowed and
    the corresponding component remains None so the system can still run in
    degraded mode (e.g., scene-only).
    """
    # Scene predictor
    if scene_predictor is None:
        try:
            from models.scene.clip_vit_scene_adapter import SceneCLIPAdapter  # type: ignore

            scene_predictor = SceneCLIPAdapter()
        except Exception:
            pass

    # Face processor
    if face_processor is None:
        try:
            from utils.emonet_single_face_processor import (  # type: ignore
                EmoNetSingleFaceProcessor,
            )

            face_processor = EmoNetSingleFaceProcessor()
        except Exception:
            pass

    # Face expert
    if face_expert is None:
        try:
            from models.face.emonet_adapter import EmoNetAdapter  # type: ignore

            face_expert = EmoNetAdapter(n_classes=8)
        except Exception:
            pass

    return scene_predictor, face_processor, face_expert


__all__.extend(["perceive_once"])  # type: ignore
