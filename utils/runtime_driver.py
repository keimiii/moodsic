"""
PerceiveFusionDriver — runtime orchestrator skeleton (no logic).

Purpose:
- Coordinate PERCEIVE per frame: face detection + EmoNet adapter (and later
  scene predictor) and fuse outputs using SceneFaceFusion.
- Provide a small, UI-agnostic API that any frontend can call.

Documentation:
- See docs/Stage IV — Inference/runtime-pipeline.md ("Runtime Driver" section)
  for behavior, integration notes, and TODOs for STABILIZE and MATCH stages.

Notes:
- This is a skeleton only. Implementations should wire
  utils.emonet_single_face_processor.EmoNetSingleFaceProcessor,
  models.face.emonet_adapter.EmoNetAdapter, models.fusion.SceneFaceFusion,
  and optionally utils.fusion_overlay.draw_fusion_overlay.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Type-only imports to avoid heavy deps at import time
try:  # pragma: no cover - optional typing convenience
    from models.fusion import FusionResult  # type: ignore
except Exception:  # pragma: no cover
    FusionResult = object  # type: ignore


class PerceiveFusionDriver:
    """Skeleton API for the PERCEIVE orchestrator (no implementation)."""

    def __init__(
        self,
        scene_predictor: Optional[object] = None,
        face_processor: Optional[object] = None,
        face_expert: Optional[object] = None,
        *,
        scene_mc_samples: int = 10,
        face_tta: int = 5,
        use_variance_weighting: bool = True,
        scene_weight: float = 0.6,
        face_weight: float = 0.4,
        max_hz: float = 4.0,
    ) -> None:
        """
        Initialize the driver with optional scene and face components.

        Args mirror the driver spec in the docs; concrete implementations should
        assemble models.fusion.SceneFaceFusion internally and apply throttling
        according to max_hz when used in UIs.
        """
        pass

    def step(self, frame_bgr: np.ndarray) -> FusionResult:
        """Run PERCEIVE on one BGR frame and return a FusionResult."""
        raise NotImplementedError

    def overlay(self, frame_bgr: np.ndarray, result: FusionResult) -> np.ndarray:
        """Return a copy of frame with an optional debug overlay drawn."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state or throttling timers."""
        raise NotImplementedError


__all__ = ["PerceiveFusionDriver"]

