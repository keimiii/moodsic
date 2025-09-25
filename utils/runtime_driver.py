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

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union, List

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
            face_mc_samples=int(face_mc_samples),
            face_sampling=face_sampling,
            face_sampling_temperature=float(face_sampling_temperature),
            face_sampling_seed=face_sampling_seed,
            face_tta_mode=face_tta_mode,
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
            face_mc_samples=self._cfg["face_mc_samples"],
            face_sampling=self._cfg["face_sampling"],
            face_sampling_temperature=self._cfg["face_sampling_temperature"],
            face_sampling_seed=self._cfg["face_sampling_seed"],
            face_tta_mode=self._cfg["face_tta_mode"],
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


@dataclass
class VideoPerceptionResult:
    """Container for fused valence/arousal time series extracted from a video."""

    video_path: Path
    frame_indices: List[int]
    valence: List[float]
    arousal: List[float]
    var_valence: List[float]
    var_arousal: List[float]
    fps: float
    width: int
    height: int
    frame_count: Optional[int]
    processed_frames: int
    first_overlay: Optional["np.ndarray"] = None
    last_overlay: Optional["np.ndarray"] = None
    fusion_results: Optional[List[FusionResult]] = None
    overlay_path: Optional[Path] = None


__all__ = ["PerceiveFusionDriver", "VideoPerceptionResult"]


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
        face_mc_samples: int = 5,
        face_sampling: str = "weighted",
        face_sampling_temperature: float = 1.0,
        face_sampling_seed: Optional[int] = None,
        face_tta_mode: str = "auto",
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
        face_mc_samples=face_mc_samples,
        face_sampling=face_sampling,
        face_sampling_temperature=face_sampling_temperature,
        face_sampling_seed=face_sampling_seed,
        face_tta_mode=face_tta_mode,
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


def _resolve_video_path(video_path: Union[str, Path], search_roots: Optional[Sequence[Path]]) -> Path:
    """Resolve a video path, mirroring notebook fallback to the repo root."""

    candidate = Path(video_path)
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"video not found: {candidate}")
        return candidate

    if candidate.exists():
        return candidate.resolve()

    roots: List[Path] = []
    if search_roots:
        for root in search_roots:
            p = Path(root)
            if p not in roots:
                roots.append(p)

    cwd = Path.cwd()
    if cwd not in roots:
        roots.append(cwd)
    if cwd.name == "notebooks":
        parent = cwd.parent
        if parent not in roots:
            roots.append(parent)

    for root in roots:
        hopeful = (root / candidate).resolve()
        if hopeful.exists():
            return hopeful

    raise FileNotFoundError(f"video not found: {video_path}")


def _finite_float(value: Optional[float]) -> bool:
    """Return True when value is a finite float-like number."""

    return isinstance(value, (int, float)) and math.isfinite(float(value))


def perceive_video(
    video_path: Union[str, Path],
    *,
    scene_predictor: Optional[object] = None,
    face_processor: Optional[object] = None,
    face_expert: Optional[object] = None,
    scene_tta: int = 5,
    face_tta: int = 5,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
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
    variance_floor: Optional[float] = 1e-3,
    max_weight_ratio: Optional[float] = None,
    max_hz: float = 0.0,
    capture_overlays: bool = True,
    save_overlay_to: Optional[Union[str, Path]] = None,
    overlay_codec: str = "mp4v",
    return_fusion: bool = False,
    search_roots: Optional[Sequence[Path]] = None,
) -> VideoPerceptionResult:
    """Run PERCEIVE over a full video and collect fused valence/arousal series.

    Parameters mirror the notebook defaults while keeping the API frontend-agnostic.
    Set ``capture_overlays`` to False to skip overlay generation entirely, and
    ``save_overlay_to`` to export an annotated video alongside numeric results.
    """

    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("OpenCV (cv2) is required to read videos") from exc

    src = _resolve_video_path(video_path, search_roots)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {src}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        fps = 25.0  # reasonable fallback

    driver = PerceiveFusionDriver(
        scene_predictor=scene_predictor,
        face_processor=face_processor,
        face_expert=face_expert,
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
        max_hz=max_hz,
    )

    writer = None
    out_path: Optional[Path] = None
    if save_overlay_to is not None:
        out_path = Path(save_overlay_to)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*overlay_codec)
        writer = cv2.VideoWriter(str(out_path), fourcc, max(fps, 1.0), (width, height))
        if not writer.isOpened():
            writer.release()
            writer = None

    frame_indices: List[int] = []
    vals: List[float] = []
    aros: List[float] = []
    vvars: List[float] = []
    avars: List[float] = []
    fusion_results: Optional[List[FusionResult]] = [] if return_fusion else None

    first_overlay = None
    last_overlay = None

    processed = 0
    idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx < start_frame:
                idx += 1
                continue

            if ((idx - start_frame) % frame_stride) != 0:
                idx += 1
                continue

            res = driver.step(frame)

            vals.append(float(res.fused.valence))
            aros.append(float(res.fused.arousal))
            vvars.append(float(res.fused.var_valence) if _finite_float(res.fused.var_valence) else math.nan)
            avars.append(float(res.fused.var_arousal) if _finite_float(res.fused.var_arousal) else math.nan)
            frame_indices.append(idx)

            if return_fusion and fusion_results is not None:
                fusion_results.append(res)

            need_overlay = capture_overlays or writer is not None
            if need_overlay:
                overlay_frame = driver.overlay(frame, res)
                if capture_overlays:
                    if first_overlay is None:
                        first_overlay = overlay_frame.copy()
                    last_overlay = overlay_frame
                if writer is not None:
                    writer.write(overlay_frame)

            processed += 1
            idx += 1

            if max_frames is not None and processed >= int(max_frames):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    result = VideoPerceptionResult(
        video_path=src,
        frame_indices=frame_indices,
        valence=vals,
        arousal=aros,
        var_valence=vvars,
        var_arousal=avars,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count if frame_count > 0 else None,
        processed_frames=processed,
        first_overlay=first_overlay if capture_overlays else None,
        last_overlay=last_overlay if capture_overlays else None,
        fusion_results=fusion_results if return_fusion else None,
        overlay_path=out_path if writer is not None else None,
    )

    return result


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


__all__.extend(["perceive_once", "perceive_video"])  # type: ignore
