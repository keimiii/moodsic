from __future__ import annotations

"""Batch export VEATIC-style perception payloads via CLI.

This script mirrors the behavior of ``notebooks/Inference/batch_video_fusion_export.ipynb``
while exposing its configuration knobs as command-line switches. It enumerates
videos under the target directory, runs ``utils.runtime_driver.perceive_video``
with the requested adapters, and stores both per-video payloads and summary
statistics inside a single Parquet artifact named
``results/inference/pipeline_results_<timestamp>.parquet``.

Typical usage (after activating the project virtualenv)::

    source .venv/bin/activate.fish
    python scripts/evaluation/run_inference_pipeline.py \
        --video-dir data/VEATIC/videos \
        --output-root results/inference \
        --stabilizer-mode both

Sweeping fusion and gating settings is done by passing flags such as
``--scene-weight``, ``--face-weight``, ``--face-score-threshold``, and
``--face-max-sigma``. The defaults match the Round I VEATIC evaluation from
``docs/Stage VI - Evaluation/eval_res_round_i.md``; override them to search for
alternative optima or dataset-specific configurations.

Examples for adjusting pathway priors::

    # Give the face stream more influence when variance weighting is disabled
    python scripts/evaluation/run_inference_pipeline.py \
        --no-variance-weighting \
        --scene-weight 0.3 \
        --face-weight 0.7

    # Tilt fusion toward scene predictions while keeping variance weighting on
    python scripts/evaluation/run_inference_pipeline.py \
        --scene-weight 0.7 \
        --face-weight 0.3
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - ensure Parquet support is available
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("pyarrow is required to write Parquet summaries. Install it first.") from exc

try:  # Align with notebook guard that registers fastai learner for torch
    import torch.serialization
    from fastai.learner import Learner  # type: ignore
    from fastai.data.core import DataLoaders  # type: ignore
except Exception as exc:  # pragma: no cover - informative failure path
    raise RuntimeError(
        "fastai and torch must be importable before running the export script"
    ) from exc
else:  # pragma: no cover - harmless registration side-effect
    torch.serialization.add_safe_globals([Learner, DataLoaders])  # Over-engineering check: single registration keeps pickle loads simple for both CLI and notebooks.


@dataclass
class ExportConfig:
    """Dense bundle of runtime settings shared across export modes."""
    video_dir: Path
    output_root: Path
    video_extensions: Tuple[str, ...]
    video_limit: Optional[int]
    scene_tta: int
    face_tta: int
    target_sample_fps: float
    max_frames: Optional[int]
    use_variance_weighting: bool
    scene_weight: float
    face_weight: float
    face_score_threshold: Optional[float]
    face_max_sigma: Optional[float]
    brightness_threshold: Optional[float]
    uncertainty_threshold: float
    stabilizer_window: int
    stabilizer_alpha: float
    capture_overlays: bool


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Build and parse the CLI interface for the batch exporter.

    All parameters default to the VEATIC Round I configuration, so invoking the
    script without flags reproduces the notebook run. Override specific
    arguments for parameter sweeps—for example, ``--scene-weight 0.5`` to try a
    balanced fusion prior or ``--stabilizer-mode off`` to skip the EMA pass.
    """
    repo_root = Path(__file__).resolve().parents[2]  # Over-engineering check: direct parent climb keeps defaults simple; packaging the repo would be heavier than needed.
    parser = argparse.ArgumentParser(
        description="Batch export fused valence/arousal JSON payloads from videos.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=repo_root / "data" / "VEATIC" / "videos",
        help="Directory containing source videos.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "results" / "inference",
        help="Destination root for JSON payloads.",
    )
    parser.add_argument(
        "--video-extensions",
        nargs="*",
        default=(".mp4", ".mov", ".mkv", ".avi"),
        help="File extensions to include (prefix with dot, e.g. .mp4).",
    )
    parser.add_argument("--scene-tta", type=int, default=3, help="Scene TTA passes.")
    parser.add_argument("--face-tta", type=int, default=3, help="Face TTA passes.")
    parser.add_argument(
        "--target-sample-fps",
        type=float,
        default=1.0,
        help="Approximate sampling rate when frame stride is auto-computed.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on processed frames per video.",
    )
    parser.add_argument(
        "--scene-weight",
        type=float,
        default=0.6,
        help="Fixed scene weight when variance weighting is disabled or unavailable.",
    )
    parser.add_argument(
        "--face-weight",
        type=float,
        default=0.4,
        help="Fixed face weight when variance weighting is disabled or unavailable.",
    )
    parser.add_argument(
        "--face-score-threshold",
        type=str,
        default=None,
        help="Optional minimum detection score for face predictions (None to disable).",
    )
    parser.add_argument(
        "--face-max-sigma",
        type=str,
        default=None,
        help="Optional sigma cap for face pathway variance gating (None to disable).",
    )
    parser.add_argument(
        "--brightness-threshold",
        type=str,
        default=None,
        help="Optional brightness floor for face pathway gating (None to disable).",
    )
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.4,
        help="Uncertainty ceiling that enables stabilizer smoothing.",
    )
    parser.add_argument(
        "--stabilizer-window",
        type=int,
        default=60,
        help="EMA window (frames) for stabilized exports.",
    )
    parser.add_argument(
        "--stabilizer-alpha",
        type=float,
        default=0.7,
        help="EMA alpha coefficient used when stabilizer is enabled.",
    )
    parser.add_argument(
        "--stabilizer-mode",
        choices=("both", "on", "off"),
        default="both",
        help="Export stabilized only, unstabilized only, or both (default).",
    )
    parser.add_argument(
        "--capture-overlays",
        action="store_true",
        help="Request overlay frames from the runtime driver (costs extra compute).",
    )
    parser.add_argument(
        "--no-variance-weighting",
        action="store_true",
        help="Disable variance-weighted fusion (fallback to fixed weights).",
    )
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=None,
        help="Optional override for adapter weight paths (defaults to repo layout).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit processing to the first N videos (sorted) for faster debugging.",
    )
    parsed = parser.parse_args(argv)
    if parsed.n is not None and parsed.n <= 0:
        parser.error("--n must be a positive integer")
    return parsed


def _ensure_repo_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def _parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return float(text)


def _prepare_extensions(exts: Iterable[str]) -> Tuple[str, ...]:
    cleaned: List[str] = []
    for ext in exts:
        token = ext.strip()
        if not token:
            continue
        if not token.startswith("."):
            token = f".{token}"
        cleaned.append(token.lower())
    return tuple(dict.fromkeys(cleaned))


def load_adapters(
    *,
    scene_tta: int,
    face_tta: int,
    repo_root: Path,
    weights_root: Optional[Path] = None,
) -> Tuple[Optional[object], Optional[object], Optional[object]]:
    """Instantiate scene and face adapters with the requested TTA counts.

    ``weights_root`` can be pointed at alternate checkpoints when searching for
    better-performing models; omit it to fall back to the repository's default
    layout.
    """
    scene_adapter: Optional[object] = None
    face_processor: Optional[object] = None
    face_adapter: Optional[object] = None

    try:
        from models.scene.clip_vit_scene_adapter import SceneCLIPAdapter  # type: ignore

        weights_dir = weights_root or repo_root
        scene_weights = weights_dir / "scene" / "checkpoints" / "clip_vit-b32_improved_fixed.pkl"  # Over-engineering check: force learner pickle to avoid silent fallback paths.
        scene_adapter = SceneCLIPAdapter(
            model_name="openai/clip-vit-base-patch32",
            dropout_rate=0.3,
            device="auto",
            tta=scene_tta,
            weights_path=str(scene_weights),
            auto_load_best=False,
        )
        print("SceneCLIPAdapter: OK")
    except Exception as exc:
        print(f"SceneCLIPAdapter unavailable: {exc}")

    try:
        from utils.emonet_single_face_processor import EmoNetSingleFaceProcessor  # type: ignore

        face_processor = EmoNetSingleFaceProcessor(
            min_detection_confidence=0.5,
            padding_ratio=0.2,
        )
        if getattr(face_processor, "available", False):
            print("EmoNetSingleFaceProcessor: OK")
        else:
            print("EmoNetSingleFaceProcessor: MediaPipe detector unavailable")
    except Exception as exc:
        print(f"Face processor unavailable: {exc}")

    try:
        from models.face.emonet_adapter import EmoNetAdapter  # type: ignore

        ckpt_root = (weights_root or repo_root) / "models" / "emonet" / "pretrained"
        face_adapter = EmoNetAdapter(
            ckpt_dir=str(ckpt_root),
            n_classes=8,
            device="auto",
            tta=face_tta,
        )
        print("EmoNetAdapter: OK")
    except Exception as exc:
        print(f"EmoNetAdapter unavailable: {exc}")

    return scene_adapter, face_processor, face_adapter


def list_videos(video_dir: Path, extensions: Sequence[str]) -> List[Path]:
    if not video_dir.exists():
        return []
    videos: List[Path] = []
    for ext in extensions:
        videos.extend(video_dir.glob(f"*{ext}"))
    return sorted(videos)


def _clean_sequence(values: Iterable[float]) -> List[Any]:
    cleaned: List[Any] = []
    for value in values:
        if isinstance(value, float):
            if math.isfinite(value):
                cleaned.append(float(value))
            else:
                cleaned.append(None)
        else:
            try:
                casted = float(value)
            except (TypeError, ValueError):
                cleaned.append(None)
            else:
                cleaned.append(casted if math.isfinite(casted) else None)
    return cleaned


def _coverage(series) -> float:
    total = len(series.valence)
    if total == 0:
        return 0.0
    valid = sum(1 for v in series.valence if isinstance(v, float) and math.isfinite(v))
    return valid / total


def series_to_dict(series, *, include_coverage: bool = False) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "valence": _clean_sequence(series.valence),
        "arousal": _clean_sequence(series.arousal),
        "var_valence": _clean_sequence(series.var_valence),
        "var_arousal": _clean_sequence(series.var_arousal),
        "mean_valence": series.mean_valence,
        "mean_arousal": series.mean_arousal,
        "median_valence": series.median_valence,
        "median_arousal": series.median_arousal,
        "mode_valence": series.mode_valence,
        "mode_arousal": series.mode_arousal,
    }
    if include_coverage:
        data["coverage"] = _coverage(series)
    return data


def build_payload(result, *, enable_stabilizer: bool, config: ExportConfig) -> Dict[str, Any]:
    if result.fps and result.fps > 0:
        timestamps = [idx / result.fps for idx in result.frame_indices]
    else:
        timestamps = [None] * len(result.frame_indices)

    return {
        "video": {
            "path": str(result.video_path),
            "width": result.width,
            "height": result.height,
            "fps": result.fps,
            "frame_count": result.frame_count,
        },
        "processing": {
            "processed_frames": result.processed_frames,
            "sample_stride": result.sample_stride,
            "effective_fps": result.effective_fps,
            "enable_stabilizer": enable_stabilizer,
            "scene_tta": config.scene_tta,
            "face_tta": config.face_tta,
            "target_sample_fps": config.target_sample_fps,
            "max_frames": config.max_frames,
            "use_variance_weighting": config.use_variance_weighting,
            "scene_weight": config.scene_weight,
            "face_weight": config.face_weight,
            "uncertainty_threshold": config.uncertainty_threshold
            if enable_stabilizer
            else None,
            "stabilizer_window": config.stabilizer_window if enable_stabilizer else None,
            "stabilizer_alpha": config.stabilizer_alpha if enable_stabilizer else None,
        },
        "samples": {
            "frame_indices": list(result.frame_indices),
            "timestamps_sec": timestamps,
            "scene": series_to_dict(result.scene_result),
            "face": series_to_dict(result.face_result, include_coverage=True),
            "fusion": series_to_dict(result.fusion_result),
        },
    }


def _build_summary_row(
    video_path: Path,
    *,
    enable_stabilizer: bool,
    result,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a flattened per-video record embedding the full payload."""

    scene_series = result.scene_result
    face_series = result.face_result
    fusion_series = result.fusion_result

    return {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "stabilizer_enabled": enable_stabilizer,
        "processed_frames": result.processed_frames,
        "sample_stride": result.sample_stride,
        "effective_fps": result.effective_fps,
        "frame_count": result.frame_count,
        "scene_mean_valence": scene_series.mean_valence,
        "scene_mean_arousal": scene_series.mean_arousal,
        "scene_median_valence": scene_series.median_valence,
        "scene_median_arousal": scene_series.median_arousal,
        "face_mean_valence": face_series.mean_valence,
        "face_mean_arousal": face_series.mean_arousal,
        "face_median_valence": face_series.median_valence,
        "face_median_arousal": face_series.median_arousal,
        "face_coverage": _coverage(face_series),
        "fusion_mean_valence": fusion_series.mean_valence,
        "fusion_mean_arousal": fusion_series.mean_arousal,
        "fusion_median_valence": fusion_series.median_valence,
        "fusion_median_arousal": fusion_series.median_arousal,
        "payload_json": json.dumps(payload, ensure_ascii=False),
    }


def _write_parquet(
    rows: List[Dict[str, Any]],
    *,
    metadata: Dict[str, Any],
    output_path: Path,
) -> None:
    """Write collected rows to Parquet with run metadata."""

    if not rows:
        return

    columns: Dict[str, List[Any]] = {}
    keys = rows[0].keys()
    for key in keys:
        columns[key] = [row.get(key) for row in rows]

    table = pa.Table.from_pydict(columns)
    schema_meta = table.schema.metadata or {}
    run_meta = {str(k).encode(): str(v).encode() for k, v in metadata.items()}
    table = table.replace_schema_metadata({**schema_meta, **run_meta})

    pq.write_table(table, output_path)

def export_batch(
    *,
    enable_stabilizer: bool,
    config: ExportConfig,
    adapters: Tuple[Optional[object], Optional[object], Optional[object]],
) -> List[Dict[str, Any]]:
    """Run the notebook-equivalent export loop for one stabilizer setting."""
    from utils.runtime_driver import perceive_video  # Local import to trim startup

    videos = list_videos(config.video_dir, config.video_extensions)
    if config.video_limit is not None and config.video_limit > 0:
        videos = videos[: config.video_limit]  # Over-engineering check: simple slice gives us fast PoC iterations without bespoke samplers.
    label = "stabilized" if enable_stabilizer else "unstabilized"

    if not videos:
        print(f"No videos found in {config.video_dir}")
        return []

    scene_adapter, face_processor, face_adapter = adapters

    rows: List[Dict[str, Any]] = []
    for video_path in videos:
        print(f"[{label}] Processing {video_path.name} ...", end=" ")
        try:
            result = perceive_video(
                video_path,
                scene_predictor=scene_adapter,
                face_processor=face_processor,
                face_expert=face_adapter,
                scene_tta=config.scene_tta,
                face_tta=config.face_tta,
                target_sample_fps=config.target_sample_fps,
                max_frames=config.max_frames,
                use_variance_weighting=config.use_variance_weighting,
                scene_weight=config.scene_weight,
                face_weight=config.face_weight,
                face_score_threshold=config.face_score_threshold,
                face_max_sigma=config.face_max_sigma,
                brightness_threshold=config.brightness_threshold,
                enable_stabilizer=enable_stabilizer,
                stabilizer_alpha=config.stabilizer_alpha,
                uncertainty_threshold=config.uncertainty_threshold,
                stabilizer_window=config.stabilizer_window,
                capture_overlays=config.capture_overlays,
                return_fusion=False,
            )
        except Exception as exc:
            print(f"failed ({exc})")
            continue

        payload = build_payload(result, enable_stabilizer=enable_stabilizer, config=config)
        rows.append(
            _build_summary_row(
                video_path,
                enable_stabilizer=enable_stabilizer,
                result=result,
                payload=payload,
            )
        )
        print("done")

    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for CLI execution; see module docstring for examples."""
    start_time = time.perf_counter()
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]  # Over-engineering check: explicit repo root ensures imports work without requiring install; adding a helper would be marginal gain.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    video_dir = _ensure_repo_path(args.video_dir, repo_root)
    output_root = _ensure_repo_path(args.output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    video_extensions = _prepare_extensions(args.video_extensions)
    face_score_threshold = _parse_optional_float(args.face_score_threshold)
    face_max_sigma = _parse_optional_float(args.face_max_sigma)
    brightness_threshold = _parse_optional_float(args.brightness_threshold)

    weights_root = (
        _ensure_repo_path(args.weights_root, repo_root)
        if args.weights_root is not None
        else None
    )

    scene_adapter, face_processor, face_adapter = load_adapters(
        scene_tta=args.scene_tta,
        face_tta=args.face_tta,
        repo_root=repo_root,
        weights_root=weights_root,
    )

    config = ExportConfig(
        video_dir=video_dir,
        output_root=output_root,
        video_extensions=video_extensions,
        video_limit=args.n,
        scene_tta=args.scene_tta,
        face_tta=args.face_tta,
        target_sample_fps=args.target_sample_fps,
        max_frames=args.max_frames,
        use_variance_weighting=not args.no_variance_weighting,
        scene_weight=args.scene_weight,
        face_weight=args.face_weight,
        face_score_threshold=face_score_threshold,
        face_max_sigma=face_max_sigma,
        brightness_threshold=brightness_threshold,
        uncertainty_threshold=args.uncertainty_threshold,
        stabilizer_window=args.stabilizer_window,
        stabilizer_alpha=args.stabilizer_alpha,
        capture_overlays=args.capture_overlays,
    )

    adapters = (scene_adapter, face_processor, face_adapter)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parquet_path = output_root / f"pipeline_results_{run_timestamp}.parquet"

    modes = []
    if args.stabilizer_mode in {"both", "off"}:
        modes.append(False)
    if args.stabilizer_mode in {"both", "on"}:
        modes.append(True)

    if not modes:
        print("Nothing to do (no stabilizer mode selected).")
        return 1

    all_rows: List[Dict[str, Any]] = []
    for enabled in modes:
        rows = export_batch(
            enable_stabilizer=enabled,
            config=config,
            adapters=adapters,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("No results to write; exiting without Parquet output.")
        return 1

    run_metadata = {
        "timestamp": run_timestamp,
        "scene_weight": args.scene_weight,
        "face_weight": args.face_weight,
        "use_variance_weighting": not args.no_variance_weighting,
        "video_limit": args.n,
        "stabilizer_mode": args.stabilizer_mode,
        "scene_tta": args.scene_tta,
        "face_tta": args.face_tta,
        "target_sample_fps": args.target_sample_fps,
        "max_frames": args.max_frames,
        "face_score_threshold": args.face_score_threshold,
        "face_max_sigma": args.face_max_sigma,
        "brightness_threshold": args.brightness_threshold,
        "command": " ".join(sys.argv),
    }

    _write_parquet(all_rows, metadata=run_metadata, output_path=parquet_path)

    print(f"Wrote Parquet summary: {parquet_path}")
    elapsed = time.perf_counter() - start_time
    print(f"Total duration: {elapsed:.2f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Over-engineering check: CLI mirrors notebook with lean tweaks; explicit parent climb avoids packaging overhead while keeping flexibility intact.
