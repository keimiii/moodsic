from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np


@dataclass
class EmotionPrediction:
    """
    Container for a single model's emotion prediction.

    Values are in the common reference space [-1, 1]. Variances are per-dimension
    (valence, arousal). If variance is unknown, set to None.
    """

    valence: float
    arousal: float
    var_valence: Optional[float] = None
    var_arousal: Optional[float] = None
    valid: bool = True


@dataclass
class FusionResult:
    scene: Optional[EmotionPrediction]
    face: Optional[EmotionPrediction]
    fused: EmotionPrediction
    # Optional face metadata for overlays
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    face_score: float = 0.0


class SceneFaceFusion:
    """
    Variance-weighted fusion between scene and face predictions.

    - If both predictions have finite, positive variances, uses inverse-variance
      weighting per dimension.
    - Otherwise falls back to fixed linear weights (scene_weight, face_weight).
    - If only one path is available/valid, returns that path.

    All inputs/outputs are in the reference space [-1, 1].
    """

    def __init__(
        self,
        scene_predictor: Optional[object] = None,
        face_expert: Optional[object] = None,
        face_processor: Optional[object] = None,
        *,
        scene_weight: float = 0.6,
        face_weight: float = 0.4,
        use_variance_weighting: bool = True,
        scene_mc_samples: int = 10,
        face_tta: int = 5,
    ) -> None:
        self.scene_predictor = scene_predictor
        self.face_expert = face_expert
        self.face_processor = face_processor
        self.scene_weight = float(scene_weight)
        self.face_weight = float(face_weight)
        self.use_variance_weighting = bool(use_variance_weighting)
        self.scene_mc_samples = int(scene_mc_samples)
        self.face_tta = int(face_tta)

    # ---- Public API ----
    def perceive_and_fuse(self, frame_bgr: np.ndarray) -> FusionResult:
        """
        Run PERCEIVE for scene and face on a BGR frame, then fuse.

        Returns FusionResult with per-path predictions and fused output. The
        method degrades gracefully when a path is unavailable.
        """
        # Scene path (optional)
        scene_pred = None
        if self.scene_predictor is not None:
            try:
                sv, sa, (svv, sva) = self.scene_predictor.predict(
                    frame_bgr, tta=self.scene_mc_samples
                )
                scene_pred = EmotionPrediction(
                    valence=float(sv),
                    arousal=float(sa),
                    var_valence=float(svv) if self._finite_pos(svv) else None,
                    var_arousal=float(sva) if self._finite_pos(sva) else None,
                    valid=True,
                )
            except Exception:
                scene_pred = EmotionPrediction(0.0, 0.0, None, None, valid=False)

        # Face path (optional)
        face_pred = None
        face_bbox = None
        face_score = 0.0
        if self.face_processor is not None and self.face_expert is not None:
            try:
                face_crop, bbox, score = self.face_processor.extract_primary_face(
                    frame_bgr
                )
                face_bbox = bbox
                face_score = float(score or 0.0)
                if face_crop is not None and face_crop.size > 0:
                    fv, fa, (fvv, fva) = self.face_expert.predict(
                        face_crop, tta=self.face_tta
                    )
                    face_pred = EmotionPrediction(
                        valence=float(fv),
                        arousal=float(fa),
                        var_valence=float(fvv) if self._finite_pos(fvv) else None,
                        var_arousal=float(fva) if self._finite_pos(fva) else None,
                        valid=True,
                    )
                else:
                    face_pred = EmotionPrediction(0.0, 0.0, None, None, valid=False)
            except Exception:
                face_pred = EmotionPrediction(0.0, 0.0, None, None, valid=False)

        # Fuse
        fused = self._fuse(scene_pred, face_pred)

        return FusionResult(
            scene=scene_pred,
            face=face_pred,
            fused=fused,
            face_bbox=face_bbox,
            face_score=face_score,
        )

    # ---- Internals ----
    def _fuse(
        self,
        scene: Optional[EmotionPrediction],
        face: Optional[EmotionPrediction],
    ) -> EmotionPrediction:
        # If neither path is valid, return neutral
        if not (scene and scene.valid) and not (face and face.valid):
            return EmotionPrediction(0.0, 0.0, 1.0, 1.0, valid=False)

        # If one path is missing/invalid, use the other
        if scene is None or not scene.valid:
            return EmotionPrediction(
                face.valence,
                face.arousal,
                face.var_valence,
                face.var_arousal,
                valid=True,
            )
        if face is None or not face.valid:
            return EmotionPrediction(
                scene.valence,
                scene.arousal,
                scene.var_valence,
                scene.var_arousal,
                valid=True,
            )

        # Both are valid: compute fused per dimension
        v, vv = self._fuse_scalar(
            scene.valence, scene.var_valence, face.valence, face.var_valence
        )
        a, av = self._fuse_scalar(
            scene.arousal, scene.var_arousal, face.arousal, face.var_arousal
        )
        return EmotionPrediction(v, a, vv, av, valid=True)

    def _fuse_scalar(
        self,
        p1: float,
        v1: Optional[float],
        p2: float,
        v2: Optional[float],
    ) -> Tuple[float, Optional[float]]:
        # Variance-weighted fusion when both variances are finite and positive
        if (
            self.use_variance_weighting
            and self._finite_pos(v1)
            and self._finite_pos(v2)
        ):
            w1 = 1.0 / (float(v1) + 1e-6)
            w2 = 1.0 / (float(v2) + 1e-6)
            total = w1 + w2
            if total <= 0.0 or not math.isfinite(total):
                # Safety fallback to fixed weights
                w1, w2 = self.scene_weight, self.face_weight
                total = w1 + w2
            p = (w1 * p1 + w2 * p2) / total
            v = 1.0 / total if total > 0 else None
            return float(np.clip(p, -1.0, 1.0)), v

        # Fixed weights fallback
        w1, w2 = self.scene_weight, self.face_weight
        total = w1 + w2
        if total <= 0.0:
            # Degenerate case; pick equal weights
            w1 = w2 = 0.5
            total = 1.0
        p = (w1 * p1 + w2 * p2) / total
        return float(np.clip(p, -1.0, 1.0)), None

    @staticmethod
    def _finite_pos(x: Optional[float]) -> bool:
        try:
            return x is not None and math.isfinite(float(x)) and float(x) > 0.0
        except Exception:
            return False


__all__ = [
    "EmotionPrediction",
    "FusionResult",
    "SceneFaceFusion",
]

