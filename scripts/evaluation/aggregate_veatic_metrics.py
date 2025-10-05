#!/usr/bin/env python3
"""Aggregate VEATIC inference runs into evaluation-ready tables."""

import argparse
import csv
import json
import math
import random
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

PATHWAYS = ("scene", "face", "fusion")
METRICS = ("valence", "arousal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stabilized-dir",
        type=Path,
        default=Path("results/inference/stabilized"),
        help="Directory containing stabilized inference JSON files.",
    )
    parser.add_argument(
        "--unstabilized-dir",
        type=Path,
        default=Path("results/inference/unstabilized"),
        help="Directory containing unstabilized inference JSON files.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/VEATIC/rating_averaged"),
        help="Directory containing VEATIC averaged rating CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evaluation"),
        help="Directory where aggregate CSV outputs will be written.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples for confidence intervals.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence interval level (e.g., 0.95 for 95%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20240209,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Override timestamp portion of output filenames.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_ground_truth_means(labels_dir: Path, video_id: str) -> Dict[str, float]:
    means: Dict[str, float] = {}
    for metric in METRICS:
        label_path = labels_dir / f"{video_id}_{metric}.csv"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")
        total = 0.0
        count = 0
        with label_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                try:
                    total += float(row[1])
                    count += 1
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"Malformed label row in {label_path}: {row}") from exc
        if count == 0:
            raise ValueError(f"No label rows found in {label_path}")
        means[metric] = total / count
    return means


def finite_values(values: Iterable[Optional[float]]) -> List[float]:
    cleaned: List[float] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (int, float)):
            value = float(value)
            if math.isnan(value):
                continue
            cleaned.append(value)
    return cleaned


def load_pathway_summary(samples: Mapping[str, object], pathway: str) -> Dict[str, Optional[float]]:
    if pathway not in samples:
        raise KeyError(f"Pathway '{pathway}' missing in inference JSON")
    block = samples[pathway]
    summary: Dict[str, Optional[float]] = {}
    for metric in METRICS:
        summary[f"mean_{metric}"] = float(block.get(f"mean_{metric}"))
        summary[f"var_{metric}"] = _safe_float(block.get(f"var_{metric}"))
    summary["coverage"] = _safe_float(block.get("coverage"))
    return summary


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_run(
    video_id: str,
    inference: Mapping[str, object],
    labels: Mapping[str, float],
    stabilized: bool,
) -> Dict[str, object]:
    samples = inference.get("samples")
    if not isinstance(samples, Mapping):
        raise ValueError(f"Inference JSON for {video_id} missing 'samples' block")
    run_summary: Dict[str, object] = {
        "video_id": video_id,
        "Stablization": stabilized,
        "gt": dict(labels),
        "metrics": {},
        "coverage": {},
        "variances": {},
        "means": {},
    }
    for pathway in PATHWAYS:
        pathway_summary = load_pathway_summary(samples, pathway)
        run_summary["coverage"][pathway] = pathway_summary.get("coverage")
        run_summary["metrics"].setdefault(pathway, {})
        run_summary["variances"].setdefault(pathway, {})
        run_summary["means"].setdefault(pathway, {})
        for metric in METRICS:
            mean_key = f"mean_{metric}"
            mean_value = pathway_summary[mean_key]
            run_summary["means"][pathway][metric] = mean_value
            run_summary["metrics"][pathway][metric] = abs(mean_value - labels[metric])
            run_summary["variances"][pathway][metric] = pathway_summary.get(f"var_{metric}")
    return run_summary


def build_per_video_rows(
    video_data: Mapping[str, MutableMapping[bool, Dict[str, object]]],
    export_timestamp: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for video_id in sorted(video_data):
        runs = video_data[video_id]
        if True not in runs or False not in runs:
            raise ValueError(f"Video {video_id} missing stabilized or unstabilized run")
        run_rows: Dict[bool, Dict[str, object]] = {}
        for stabilized_flag, run_summary in runs.items():
            row: Dict[str, object] = {
                "video_id": video_id,
                "Stablization": stabilized_flag,
                "gt_mean_valence": run_summary["gt"]["valence"],
                "gt_mean_arousal": run_summary["gt"]["arousal"],
            }
            for pathway in PATHWAYS:
                row[f"coverage_{pathway}"] = run_summary["coverage"].get(pathway)
                for metric in METRICS:
                    row[f"mean_{pathway}_{metric}"] = run_summary["means"][pathway][metric]
                    row[f"mae_{pathway}_{metric}"] = run_summary["metrics"][pathway][metric]
                    row[f"var_{pathway}_{metric}"] = run_summary["variances"][pathway][metric]
            row["export_timestamp"] = export_timestamp
            run_rows[stabilized_flag] = row
        for metric in METRICS:
            face_key = f"mae_face_{metric}"
            scene_key = f"mae_scene_{metric}"
            fusion_key = f"mae_fusion_{metric}"
            for stabilized_flag, row in run_rows.items():
                row[f"face_beats_fusion_{metric}"] = row[face_key] < row[fusion_key]
                row[f"scene_beats_fusion_{metric}"] = row[scene_key] < row[fusion_key]
        for pathway in PATHWAYS:
            for metric in METRICS:
                delta = (
                    run_rows[True][f"mae_{pathway}_{metric}"]
                    - run_rows[False][f"mae_{pathway}_{metric}"]
                )
                run_rows[True][f"delta_mae_{pathway}_{metric}"] = delta
                run_rows[False][f"delta_mae_{pathway}_{metric}"] = delta
        rows.append(run_rows[False])
        rows.append(run_rows[True])
    return rows


def bootstrap_ci(
    values: List[float],
    rng: random.Random,
    bootstrap_samples: int,
    ci: float,
) -> List[float]:
    if not values:
        return [math.nan, math.nan]
    if len(values) == 1:
        return [values[0], values[0]]
    stats: List[float] = []
    sample_size = len(values)
    for _ in range(bootstrap_samples):
        sample = [values[rng.randrange(sample_size)] for _ in range(sample_size)]
        stats.append(sum(sample) / sample_size)
    stats.sort()
    alpha = (1.0 - ci) / 2.0
    lower_index = max(0, int(math.floor(alpha * (len(stats) - 1))))
    upper_index = max(0, int(math.ceil((1.0 - alpha) * (len(stats) - 1))))
    upper_index = min(upper_index, len(stats) - 1)
    return [stats[lower_index], stats[upper_index]]


def aggregate_dataset_rows(
    video_data: Mapping[str, MutableMapping[bool, Dict[str, object]]],
    rng: random.Random,
    bootstrap_samples: int,
    ci: float,
    export_timestamp: str,
) -> List[Dict[str, object]]:
    aggregates: List[Dict[str, object]] = []
    for pathway in PATHWAYS:
        for metric in METRICS:
            stabilized_values: List[float] = []
            raw_values: List[float] = []
            coverage_stabilized: List[float] = []
            coverage_raw: List[float] = []
            variance_stabilized: List[float] = []
            variance_raw: List[float] = []
            delta_values: List[float] = []
            for video_id in video_data:
                runs = video_data[video_id]
                if True not in runs or False not in runs:
                    continue
                stabilized_run = runs[True]
                raw_run = runs[False]
                stabilized_values.append(
                    stabilized_run["metrics"][pathway][metric]
                )
                raw_values.append(raw_run["metrics"][pathway][metric])
                coverage_stabilized.append(
                    stabilized_run["coverage"].get(pathway)
                )
                coverage_raw.append(raw_run["coverage"].get(pathway))
                variance_stabilized.append(
                    stabilized_run["variances"][pathway][metric]
                )
                variance_raw.append(
                    raw_run["variances"][pathway][metric]
                )
                delta_values.append(
                    stabilized_run["metrics"][pathway][metric]
                    - raw_run["metrics"][pathway][metric]
                )
            cleaned_stabilized = finite_values(stabilized_values)
            cleaned_raw = finite_values(raw_values)
            cleaned_delta = finite_values(delta_values)
            coverage_stabilized = finite_values(coverage_stabilized)
            coverage_raw = finite_values(coverage_raw)
            variance_stabilized = finite_values(variance_stabilized)
            variance_raw = finite_values(variance_raw)
            row: Dict[str, object] = {
                "pathway": pathway,
                "metric": metric,
                "count": len(cleaned_stabilized),
                "mae_mean_stabilized": _mean(cleaned_stabilized),
                "mae_mean_unstabilized": _mean(cleaned_raw),
                "mae_median_stabilized": _median(cleaned_stabilized),
                "mae_median_unstabilized": _median(cleaned_raw),
                "mae_std_stabilized": _stdev(cleaned_stabilized),
                "mae_std_unstabilized": _stdev(cleaned_raw),
                "mae_ci_low_stabilized": math.nan,
                "mae_ci_high_stabilized": math.nan,
                "mae_ci_low_unstabilized": math.nan,
                "mae_ci_high_unstabilized": math.nan,
                "delta_mae_mean": _mean(cleaned_delta),
                "delta_mae_median": _median(cleaned_delta),
                "delta_mae_ci_low": math.nan,
                "delta_mae_ci_high": math.nan,
                "coverage_mean_stabilized": _mean(coverage_stabilized),
                "coverage_mean_unstabilized": _mean(coverage_raw),
                "coverage_std_stabilized": _stdev(coverage_stabilized),
                "coverage_std_unstabilized": _stdev(coverage_raw),
                "variance_mean_stabilized": _mean(variance_stabilized),
                "variance_mean_unstabilized": _mean(variance_raw),
                "export_timestamp": export_timestamp,
            }
            row["mae_ci_low_stabilized"], row["mae_ci_high_stabilized"] = bootstrap_ci(
                cleaned_stabilized, rng, bootstrap_samples, ci
            )
            row["mae_ci_low_unstabilized"], row["mae_ci_high_unstabilized"] = bootstrap_ci(
                cleaned_raw, rng, bootstrap_samples, ci
            )
            row["delta_mae_ci_low"], row["delta_mae_ci_high"] = bootstrap_ci(
                cleaned_delta, rng, bootstrap_samples, ci
            )
            aggregates.append(row)
    return aggregates


def _mean(values: List[float]) -> float:
    if not values:
        return math.nan
    return statistics.fmean(values)


def _median(values: List[float]) -> float:
    if not values:
        return math.nan
    return statistics.median(values)


def _stdev(values: List[float]) -> float:
    if len(values) <= 1:
        return math.nan
    return statistics.stdev(values)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    stabilized_dir = args.stabilized_dir
    unstabilized_dir = args.unstabilized_dir
    labels_dir = args.labels_dir
    for path in (stabilized_dir, unstabilized_dir, labels_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required directory does not exist: {path}")
    stabilized_files = {path.stem: path for path in stabilized_dir.glob("*.json")}
    unstabilized_files = {path.stem: path for path in unstabilized_dir.glob("*.json")}
    if set(stabilized_files) != set(unstabilized_files):
        missing_in_unstabilized = sorted(set(stabilized_files) - set(unstabilized_files))
        missing_in_stabilized = sorted(set(unstabilized_files) - set(stabilized_files))
        message_parts: List[str] = []
        if missing_in_unstabilized:
            message_parts.append(
                f"Missing unstabilized runs for: {', '.join(missing_in_unstabilized)}"
            )
        if missing_in_stabilized:
            message_parts.append(
                f"Missing stabilized runs for: {', '.join(missing_in_stabilized)}"
            )
        raise ValueError("; ".join(message_parts))
    export_timestamp = args.timestamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    rng = random.Random(args.seed)
    video_data: Dict[str, MutableMapping[bool, Dict[str, object]]] = {}
    for video_id in sorted(stabilized_files):
        labels = load_ground_truth_means(labels_dir, video_id)
        stabilized_summary = summarize_run(
            video_id, load_json(stabilized_files[video_id]), labels, True
        )
        unstabilized_summary = summarize_run(
            video_id, load_json(unstabilized_files[video_id]), labels, False
        )
        video_data[video_id] = {
            True: stabilized_summary,
            False: unstabilized_summary,
        }
    per_video_rows = build_per_video_rows(video_data, export_timestamp)
    per_video_columns: List[str] = [
        "video_id",
        "Stablization",
        "gt_mean_valence",
        "gt_mean_arousal",
    ]
    for pathway in PATHWAYS:
        for metric in METRICS:
            per_video_columns.append(f"mean_{pathway}_{metric}")
        per_video_columns.append(f"coverage_{pathway}")
        for metric in METRICS:
            per_video_columns.append(f"mae_{pathway}_{metric}")
        for metric in METRICS:
            per_video_columns.append(f"var_{pathway}_{metric}")
    for metric in METRICS:
        per_video_columns.append(f"face_beats_fusion_{metric}")
        per_video_columns.append(f"scene_beats_fusion_{metric}")
    for pathway in PATHWAYS:
        for metric in METRICS:
            per_video_columns.append(f"delta_mae_{pathway}_{metric}")
    per_video_columns.append("export_timestamp")
    aggregates = aggregate_dataset_rows(
        video_data,
        rng=rng,
        bootstrap_samples=args.bootstrap_samples,
        ci=args.ci,
        export_timestamp=export_timestamp,
    )
    aggregate_columns: List[str] = [
        "pathway",
        "metric",
        "count",
        "mae_mean_stabilized",
        "mae_mean_unstabilized",
        "mae_median_stabilized",
        "mae_median_unstabilized",
        "mae_std_stabilized",
        "mae_std_unstabilized",
        "mae_ci_low_stabilized",
        "mae_ci_high_stabilized",
        "mae_ci_low_unstabilized",
        "mae_ci_high_unstabilized",
        "delta_mae_mean",
        "delta_mae_median",
        "delta_mae_ci_low",
        "delta_mae_ci_high",
        "coverage_mean_stabilized",
        "coverage_mean_unstabilized",
        "coverage_std_stabilized",
        "coverage_std_unstabilized",
        "variance_mean_stabilized",
        "variance_mean_unstabilized",
        "export_timestamp",
    ]
    output_dir = args.output_dir
    per_video_path = output_dir / f"veatic_per_video_{export_timestamp}.csv"
    aggregate_path = output_dir / f"veatic_aggregate_{export_timestamp}.csv"
    write_csv(per_video_path, per_video_rows, per_video_columns)
    write_csv(aggregate_path, aggregates, aggregate_columns)
    print(f"Wrote per-video metrics to {per_video_path}")
    print(f"Wrote aggregate metrics to {aggregate_path}")


if __name__ == "__main__":
    main()
# Over-engineering check: Script balances rigor and scope; bootstrap CI can be simplified to mean-only if schedule tight.
