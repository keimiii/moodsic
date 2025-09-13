from __future__ import annotations

import math
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple, Dict, Any

import numpy as np


@dataclass
class EmotionPrediction:
    """
    A single model's "opinion" about emotion for one frame.

    Plain-English guide:
    - valence (V) and arousal (A) are numbers between -1 and 1.
      Think of valence as negative↔positive and arousal as calm↔excited.
    - var_valence/var_arousal measure uncertainty (higher = less confident).
      If a variance is unknown, it is set to None.
    - valid indicates whether this opinion should be trusted for fusion.
    """

    valence: float
    arousal: float
    var_valence: Optional[float] = None
    var_arousal: Optional[float] = None
    valid: bool = True


@dataclass
class FusionResult:
    """Outputs of the fusion step for one frame.

    - scene: Scene (whole image) opinion, if available.
    - face: Face opinion, if a face was detected and evaluated.
    - fused: The combined result we actually report/use downstream.
    - face_bbox/face_score: Optional extras for drawing overlays or debugging.
    """

    scene: Optional[EmotionPrediction]
    face: Optional[EmotionPrediction]
    fused: EmotionPrediction
    # Optional face metadata for overlays
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    face_score: float = 0.0
    # Optional stability metrics (post-fusion EMA with uncertainty gating)
    stability_variance: Optional[Tuple[float, float]] = None
    stability_jitter: Optional[Tuple[float, float]] = None


class SceneFaceFusion:
    """
    Combine two opinions (scene and face) into one stable output per frame.

    Easy version:
    - If both opinions include uncertainty, trust the more confident one more
      (inverse-variance weighting).
    - If uncertainties are missing, use a simple fixed blend (defaults to
      60% scene, 40% face).
    - If only one opinion is available, use it directly.
    - Everything is clipped to [-1, 1].

    Optional guardrails for stability (off by default):
    - Guardrails are simple thresholds that temporarily ignore the face opinion
      for a frame when it looks unreliable. This prevents jitter when the face
      is occluded or the lighting is poor.
      • face_score_threshold (0..1): if face_score < threshold, ignore face.
      • face_max_sigma (≈0.2..1.0): if sqrt(variance) for V or A > threshold,
        ignore face for that frame.
      • brightness_threshold (0..255): if average frame brightness < threshold,
        ignore face (dark frames often yield noisy face predictions).
      All thresholds default to None (disabled). See __init__ for details.

    All inputs/outputs live in the same reference space [-1, 1].
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
        scene_mc_samples: int = 5,
        face_tta: int = 5,
        # Optional gating to improve stability under occlusion/low light
        face_score_threshold: Optional[float] = None,
        face_max_sigma: Optional[float] = None,
        brightness_threshold: Optional[float] = None,
        # Optional post-fusion stabilizer (EMA with uncertainty gating)
        enable_stabilizer: bool = False,
        stabilizer_alpha: float = 0.7,
        uncertainty_threshold: float = 0.4,
        stabilizer_window: int = 60,
    ) -> None:
        """
        Initialize the fusion module and (optionally) enable stability guardrails.

        Parameters (selected):
        - scene_weight, face_weight: fixed fallback blend when variances are
          missing/invalid (defaults: 0.6 scene / 0.4 face).
        - use_variance_weighting: if True and both variances are valid, use
          inverse-variance fusion (more confident opinion gets more weight).
        - scene_mc_samples, face_tta: how many stochastic samples to request
          from scene and face predictors (if they support uncertainty). Defaults
          are 5 for both.

        Guardrail thresholds (optional; disabled by default):
        - face_score_threshold (float in [0, 1])
        - face_max_sigma (float)  # sigma = sqrt(variance)
        - brightness_threshold (float in [0, 255])  # luma
          See docs/Stage II — Modeling/uncertainty-and-gating.md for
          recommended starting ranges and rationale.

        Post-fusion stabilizer (optional; disabled by default):
        - enable_stabilizer (bool): If True, apply an exponential moving average
          (EMA) to fused outputs and hold updates when fused variance exceeds
          `uncertainty_threshold` per dimension. This follows the design in
          docs/Stage II — Modeling/uncertainty-and-gating.md.
        - stabilizer_alpha (float): EMA coefficient (default 0.7).
        - uncertainty_threshold (float): Variance threshold τ; when fused
          variance for V or A exceeds τ, the last stable value is held for that
          dimension (default 0.4). Note: τ is on variance, not sigma.
        - stabilizer_window (int): Window for basic stability metrics (variance
          and mean absolute first-difference jitter) reported in FusionResult.

        Notes:
        - All three thresholds default to None (feature off). Enable and tune
          per your data. See scripts/fusion_threshold_tuning.py to sweep values
          and minimize jitter on a validation slice. For guidance on typical
          ranges and the underlying rationale, refer to the docs mentioned above.
        """
        self.scene_predictor = scene_predictor
        self.face_expert = face_expert
        self.face_processor = face_processor
        self.scene_weight = float(scene_weight)
        self.face_weight = float(face_weight)
        self.use_variance_weighting = bool(use_variance_weighting)
        self.scene_mc_samples = int(scene_mc_samples)
        self.face_tta = int(face_tta)
        # Gating thresholds (disabled by default)
        self.face_score_threshold = (
            float(face_score_threshold) if face_score_threshold is not None else None
        )
        self.face_max_sigma = float(face_max_sigma) if face_max_sigma is not None else None
        # Brightness threshold on frame luma (0-255); if set and frame is darker,
        # gate off face to prefer the scene path for stability.
        self.brightness_threshold = (
            float(brightness_threshold) if brightness_threshold is not None else None
        )
        # Post-fusion stabilizer
        self._stabilizer = (
            _AdaptiveStabilizer(
                alpha=float(stabilizer_alpha),
                uncertainty_threshold=float(uncertainty_threshold),
                window_size=int(stabilizer_window),
            )
            if bool(enable_stabilizer)
            else None
        )

    # ---- Public API ----
    def perceive_and_fuse(self, frame_bgr: np.ndarray) -> FusionResult:
        """
        Run PERCEIVE for scene and face on a BGR frame, then fuse.

        Returns FusionResult with per-path predictions and the final fused
        output. This method is resilient: if one path fails (e.g., no face
        detected), it uses the other; if both fail, it returns a neutral value.
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
                    # Apply optional gating for robustness
                    if self.face_score_threshold is not None and face_score < self.face_score_threshold:
                        face_pred.valid = False
                    if self.face_max_sigma is not None:
                        sig_v = math.sqrt(max(face_pred.var_valence, 0.0)) if face_pred.var_valence is not None else float("inf")
                        sig_a = math.sqrt(max(face_pred.var_arousal, 0.0)) if face_pred.var_arousal is not None else float("inf")
                        if sig_v > self.face_max_sigma or sig_a > self.face_max_sigma:
                            face_pred.valid = False
                    if self.brightness_threshold is not None:
                        luma = self._estimate_luma(frame_bgr)
                        if luma < self.brightness_threshold:
                            face_pred.valid = False
                else:
                    face_pred = EmotionPrediction(0.0, 0.0, None, None, valid=False)
            except Exception:
                face_pred = EmotionPrediction(0.0, 0.0, None, None, valid=False)

        # Fuse
        fused = self._fuse(scene_pred, face_pred)

        # Optional post-fusion stabilization (EMA + uncertainty gating)
        stability_variance: Optional[Tuple[float, float]] = None
        stability_jitter: Optional[Tuple[float, float]] = None
        if self._stabilizer is not None:
            v_var = fused.var_valence if self._finite_pos(fused.var_valence) else None
            a_var = fused.var_arousal if self._finite_pos(fused.var_arousal) else None
            out_v, out_a = self._stabilizer.update(
                fused.valence,
                fused.arousal,
                variance=(v_var, a_var) if (v_var is not None or a_var is not None) else None,
            )
            fused = EmotionPrediction(
                valence=float(out_v),
                arousal=float(out_a),
                var_valence=fused.var_valence,
                var_arousal=fused.var_arousal,
                valid=fused.valid,
            )
            metrics = self._stabilizer.get_stability_metrics()
            if metrics is not None:
                sv = metrics.get("variance")
                sj = metrics.get("jitter")
                if isinstance(sv, tuple):
                    stability_variance = sv
                if isinstance(sj, tuple):
                    stability_jitter = sj

        return FusionResult(
            scene=scene_pred,
            face=face_pred,
            fused=fused,
            face_bbox=face_bbox,
            face_score=face_score,
            stability_variance=stability_variance,
            stability_jitter=stability_jitter,
        )

    # ---- Internals ----
    def _fuse(
        self,
        scene: Optional[EmotionPrediction],
        face: Optional[EmotionPrediction],
    ) -> EmotionPrediction:
        """Combine scene and face opinions, handling edge cases simply.

        - Neither valid: return neutral (0, 0) with unit variance.
        - Only one valid: passthrough that opinion as-is.
        - Both valid: fuse valence and arousal independently via _fuse_scalar.
        """
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
        """Fuse one number (e.g., valence) from two sources.

        - Preferred: inverse-variance weighting (more confident gets more say).
          Returns the fused value and its fused variance (= 1 / sum(precisions)).
        - Fallback: fixed weights (scene_weight, face_weight) when any variance
          is missing/invalid; in this case the returned variance is None.
        - Output is always clipped to [-1, 1].
        """
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
        """True if x is a real number greater than 0 (used for variances)."""
        try:
            return x is not None and math.isfinite(float(x)) and float(x) > 0.0
        except Exception:
            return False

    @staticmethod
    def _estimate_luma(frame_bgr: np.ndarray) -> float:
        """Estimate frame brightness (0=dark … 255=bright) without OpenCV.

        Uses a standard formula (BT.601) on B, G, R channels to compute luma:
        Y ≈ 0.114*B + 0.587*G + 0.299*R. We only need an average to decide if a
        frame is "too dark" for reliable faces.
        """
        try:
            if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
                return 0.0
            b = frame_bgr[..., 0].astype(np.float32)
            g = frame_bgr[..., 1].astype(np.float32)
            r = frame_bgr[..., 2].astype(np.float32)
            y = 0.114 * b + 0.587 * g + 0.299 * r
            return float(np.mean(y))
        except Exception:
            return 0.0


class _AdaptiveStabilizer:
    """Simple EMA-based stabilizer with uncertainty gating per dimension.

    - Applies EMA to valence and arousal using coefficient `alpha`.
    - If `variance` is provided and exceeds `uncertainty_threshold` for a
      dimension, the last stable value is held for that dimension.
    - Tracks a short history to expose basic stability metrics.
    """

    def __init__(self, *, alpha: float = 0.7, uncertainty_threshold: float = 0.4, window_size: int = 60) -> None:
        self.alpha = float(alpha)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.window_size = int(window_size)
        self.ema_v: Optional[float] = None
        self.ema_a: Optional[float] = None
        self.last_stable_v: float = 0.0
        self.last_stable_a: float = 0.0
        self.history: "deque[Tuple[float, float]]" = deque(maxlen=self.window_size)

    def update(self, valence: float, arousal: float, variance: Optional[Tuple[Optional[float], Optional[float]]] = None) -> Tuple[float, float]:
        if self.ema_v is None or self.ema_a is None:
            self.ema_v, self.ema_a = float(valence), float(arousal)
            self.last_stable_v, self.last_stable_a = float(valence), float(arousal)
            self.history.append((self.ema_v, self.ema_a))
            return self.ema_v, self.ema_a

        # EMA update
        self.ema_v = self.alpha * float(valence) + (1.0 - self.alpha) * self.ema_v
        self.ema_a = self.alpha * float(arousal) + (1.0 - self.alpha) * self.ema_a

        out_v, out_a = self.ema_v, self.ema_a
        if variance is not None:
            v_var, a_var = variance
            if v_var is not None and v_var > self.uncertainty_threshold:
                out_v = self.last_stable_v
            else:
                self.last_stable_v = out_v
            if a_var is not None and a_var > self.uncertainty_threshold:
                out_a = self.last_stable_a
            else:
                self.last_stable_a = out_a
        else:
            self.last_stable_v, self.last_stable_a = out_v, out_a

        self.history.append((out_v, out_a))
        return out_v, out_a

    def get_stability_metrics(self) -> Optional[Dict[str, Tuple[float, float]]]:
        if len(self.history) < 2:
            return None
        arr = np.array(list(self.history), dtype=np.float32)
        var_v, var_a = float(np.var(arr[:, 0])), float(np.var(arr[:, 1]))
        jit_v = float(np.mean(np.abs(np.diff(arr[:, 0]))))
        jit_a = float(np.mean(np.abs(np.diff(arr[:, 1]))))
        return {"variance": (var_v, var_a), "jitter": (jit_v, jit_a)}

__all__ = [
    "EmotionPrediction",
    "FusionResult",
    "SceneFaceFusion",
]
