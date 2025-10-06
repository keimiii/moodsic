#!/usr/bin/env python3
"""Export the DEAM Gaussian mixture parameters into a versioned bundle.

The bundle follows the structure proposed in ``docs/Stage VI - Evaluation``:

* ``clusters_params.npz`` holds the raw GMM arrays (weights, means, covariances)
  plus convenience copies of the component means in reference space.
* ``clusters_meta.json`` captures quadrant labels, scaler details, and
  provenance so downstream jobs can reason about bundle lineage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

QUADRANT_NAMES: Dict[Tuple[int, int], str] = {
    (1, 1): "q1_happy",
    (-1, 1): "q2_angry",
    (-1, -1): "q3_sad",
    (1, -1): "q4_calm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("notebooks/Dataset - DEAM/artifacts/deam_gmm"),
        help="Directory containing gmm.pkl and scaler.pkl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/clustering"),
        help="Root directory where the versioned bundle will be written",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        help="Optional timestamp override for the bundle directory",
    )
    parser.add_argument(
        "--gmm-name",
        type=str,
        default="gmm.pkl",
        help="Filename of the pickled GaussianMixture",
    )
    parser.add_argument(
        "--scaler-name",
        type=str,
        default="scaler.pkl",
        help="Filename of the pickled StandardScaler",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing bundle directory",
    )
    return parser.parse_args()


def resolve_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def ensure_output_dir(output_root: Path, timestamp: str, overwrite: bool) -> Path:
    bundle_dir = output_root / f"deam_gmm_{timestamp}"
    if bundle_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Bundle directory '{bundle_dir}' already exists. "
                "Use --overwrite or provide a different --timestamp."
            )
    else:
        bundle_dir.mkdir(parents=True, exist_ok=True)
    return bundle_dir


def compute_quadrant(valence: float, arousal: float) -> str:
    sign_v = 1 if valence >= 0 else -1
    sign_a = 1 if arousal >= 0 else -1
    return QUADRANT_NAMES[(sign_v, sign_a)]


def export_bundle(args: argparse.Namespace) -> Path:
    source_dir = args.source_dir
    gmm_path = source_dir / args.gmm_name
    scaler_path = source_dir / args.scaler_name

    if not gmm_path.exists():
        raise FileNotFoundError(f"Missing GaussianMixture pickle at '{gmm_path}'")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing StandardScaler pickle at '{scaler_path}'")

    gmm = joblib.load(gmm_path)
    scaler = joblib.load(scaler_path)

    means_ref = gmm.means_ * scaler.scale_ + scaler.mean_

    component_meta: List[Dict[str, float | str | int]] = []
    for idx, (weight, mean_scaled, mean_ref) in enumerate(
        zip(gmm.weights_, gmm.means_, means_ref)
    ):
        valence_ref = float(mean_ref[0])
        arousal_ref = float(mean_ref[1])
        quadrant = compute_quadrant(valence_ref, arousal_ref)
        component_meta.append(
            {
                "component": idx,
                "weight": float(weight),
                "quadrant": quadrant,
                "mean_ref": {
                    "valence": valence_ref,
                    "arousal": arousal_ref,
                },
                "mean_scaled": {
                    "valence": float(mean_scaled[0]),
                    "arousal": float(mean_scaled[1]),
                },
            }
        )

    bundle_dir = ensure_output_dir(args.output_root, args.timestamp, args.overwrite)

    params_payload = {
        "weights": gmm.weights_,
        "means": gmm.means_,
        "covariances": gmm.covariances_,
        "means_ref": means_ref,
    }
    np.savez(bundle_dir / "clusters_params.npz", **params_payload)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(bundle_dir),
        "gmm": {
            "n_components": int(gmm.n_components),
            "covariance_type": getattr(gmm, "covariance_type", "unknown"),
            "converged": bool(getattr(gmm, "converged_", False)),
            "n_iter": int(getattr(gmm, "n_iter_", 0)),
            "tol": float(getattr(gmm, "tol", 0.0)),
        },
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "components": component_meta,
        "reference_space": {
            "valence": [-1.0, 1.0],
            "arousal": [-1.0, 1.0],
        },
        "provenance": {
            "git_commit": resolve_git_commit(),
            "source_dir": str(source_dir.resolve()),
            "export_script": Path(__file__).name,
        },
        "artifacts": {
            "params": "clusters_params.npz",
            "metadata": "clusters_meta.json",
        },
    }

    with (bundle_dir / "clusters_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)

    return bundle_dir


def main() -> None:
    args = parse_args()
    bundle_dir = export_bundle(args)
    print(f"Exported DEAM GMM bundle to: {bundle_dir}")


if __name__ == "__main__":
    main()
