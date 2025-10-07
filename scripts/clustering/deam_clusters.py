"""Utilities to interact with persisted DEAM Gaussian mixture bundles.

The export script under ``scripts/clustering/export_deam_gmm.py`` produces a
bundle with ``clusters_meta.json`` (metadata, scaler, quadrants) and
``clusters_params.npz`` (weights, means, covariances). This module loads that
bundle and exposes helpers needed by the evaluator and inference jobs:

* :class:`DEAMClusterBundle` wraps the arrays and metadata.
* :meth:`DEAMClusterBundle.predict_proba` mirrors scikit-learn's API on the
  stored parameters without requiring the original ``GaussianMixture`` object.
* :meth:`DEAMClusterBundle.quadrant_for_component` provides the component-to
  quadrant lookup captured during export.
* :func:`annotate_parquet_with_clusters` consumes an inference parquet export
  and writes an annotated copy with DEAM component/quadrant columns.

Example
-------
Link fused valence/arousal predictions to their most likely DEAM station and
quadrant by passing the pipeline parquet through the helper:

```
python scripts/clustering/deam_clusters.py \
    --bundle-dir results/clustering/deam_gmm_20251006_151857 \
    --parquet results/inference/pipeline_results_20251006_144126.parquet \
    --output results/inference/pipeline_results_20251006_144126_clusters.parquet
```

Programmatic usage is also available:

```
from scripts.clustering.deam_clusters import (
    annotate_parquet_with_clusters,
    load_deam_cluster_bundle,
)

annotated_path = annotate_parquet_with_clusters(
    parquet_path="results/inference/pipeline_results_20251006_144126.parquet",
    bundle_dir="results/clustering/deam_gmm_20251006_151857",
)
print(f"Annotated parquet written to {annotated_path}")
```
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd

BUNDLE_META_NAME = "clusters_meta.json"
BUNDLE_PARAMS_NAME = "clusters_params.npz"


@dataclass(frozen=True)
class ComponentMetadata:
    """Minimal metadata tracked per component in the bundle."""

    component: int
    quadrant: str
    weight: float
    mean_ref: tuple[float, float]
    mean_scaled: tuple[float, float]


class DEAMClusterBundle:
    """In-memory representation of the exported DEAM GMM bundle."""

    def __init__(
        self,
        *,
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        components: Mapping[int, ComponentMetadata],
        metadata: Mapping[str, object],
    ) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.means = np.asarray(means, dtype=np.float64)
        self.covariances = np.asarray(covariances, dtype=np.float64)
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float64)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
        self._components = dict(components)
        self.metadata = dict(metadata)

        if self.weights.ndim != 1:
            raise ValueError("weights must be 1-dimensional")
        if self.means.shape != self.covariances.shape:
            raise ValueError("means and covariances must share the same shape")
        if self.means.ndim != 2 or self.means.shape[1] != 2:
            raise ValueError("only 2D valence/arousal GMMs are supported")
        if self.scaler_mean.shape != (2,) or self.scaler_scale.shape != (2,):
            raise ValueError("scaler parameters must be length-2 for VA space")
        if len(self._components) != self.weights.shape[0]:
            raise ValueError("component metadata length does not match weights")
        if np.any(self.scaler_scale == 0):
            raise ValueError("scaler scale entries must be non-zero")
        if np.any(self.weights <= 0):
            raise ValueError("mixture weights must be strictly positive")
        if np.any(self.covariances <= 0):
            raise ValueError("covariance entries must be strictly positive")

        self._log_weight_norm = np.log(self.weights)
        # Pre-compute terms used in the Gaussian likelihood evaluation.
        self._inv_covariances = 1.0 / self.covariances
        self._log_norm_consts = -0.5 * (
            np.log(2.0 * np.pi * self.covariances).sum(axis=1)
        )

    @property
    def n_components(self) -> int:
        return self.weights.shape[0]

    @property
    def components(self) -> Dict[int, ComponentMetadata]:
        return dict(self._components)

    def predict_proba(
        self, valence: float | Iterable[float], arousal: float | Iterable[float]
    ) -> np.ndarray:
        """Return mixture posteriors for the provided valence/arousal inputs."""

        valence_arr = np.asarray(valence, dtype=np.float64)
        arousal_arr = np.asarray(arousal, dtype=np.float64)

        if valence_arr.shape != arousal_arr.shape:
            raise ValueError("valence and arousal must share the same shape")

        single_input = valence_arr.ndim == 0
        if single_input:
            features = np.array([[float(valence_arr), float(arousal_arr)]])
        else:
            features = np.stack([valence_arr, arousal_arr], axis=1)

        transformed = (features - self.scaler_mean) / self.scaler_scale
        log_resp = self._log_prob(transformed)
        resp = np.exp(log_resp - log_resp.max(axis=1, keepdims=True))
        resp /= resp.sum(axis=1, keepdims=True)

        if single_input:
            return resp[0]
        return resp

    def top_component(
        self, valence: float | Iterable[float], arousal: float | Iterable[float]
    ) -> int | np.ndarray:
        """Return the index of the highest-probability component."""

        posteriors = self.predict_proba(valence, arousal)
        if posteriors.ndim == 1:
            return int(np.argmax(posteriors))
        return np.argmax(posteriors, axis=1)

    def quadrant_for_point(
        self, valence: float | Iterable[float], arousal: float | Iterable[float]
    ) -> str | list[str]:
        """Return the quadrant label associated with the top component."""

        top_components = self.top_component(valence, arousal)
        if np.isscalar(top_components):
            return self.quadrant_for_component(int(top_components))
        return [
            self.quadrant_for_component(int(component))
            for component in np.asarray(top_components).tolist()
        ]

    def quadrant_for_component(self, component: int) -> str:
        try:
            return self._components[int(component)].quadrant
        except KeyError as exc:
            raise KeyError(f"Unknown component id: {component}") from exc

    def component_metadata(self, component: int) -> ComponentMetadata:
        try:
            return self._components[int(component)]
        except KeyError as exc:
            raise KeyError(f"Unknown component id: {component}") from exc

    def _log_prob(self, transformed: np.ndarray) -> np.ndarray:
        if transformed.ndim != 2 or transformed.shape[1] != 2:
            raise ValueError("transformed inputs must have shape (n_samples, 2)")

        diff = transformed[:, None, :] - self.means[None, :, :]
        mahal = -0.5 * np.sum(diff * diff * self._inv_covariances, axis=2)
        return mahal + self._log_norm_consts + self._log_weight_norm


def load_deam_cluster_bundle(bundle_dir: str | Path) -> DEAMClusterBundle:
    """Load a DEAM GMM bundle exported by ``export_deam_gmm.py``."""

    bundle_path = Path(bundle_dir)
    meta_path = bundle_path / BUNDLE_META_NAME
    params_path = bundle_path / BUNDLE_PARAMS_NAME

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"Missing parameter file: {params_path}")

    with meta_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    components_meta = metadata.get("components")
    if not isinstance(components_meta, list):
        raise ValueError("metadata missing 'components' list")

    component_map: Dict[int, ComponentMetadata] = {}
    for entry in components_meta:
        try:
            idx = int(entry["component"])
            quadrant = str(entry["quadrant"])
            weight = float(entry["weight"])
            mean_ref = entry.get("mean_ref", {})
            mean_scaled = entry.get("mean_scaled", {})
            mean_ref_tuple = (
                float(mean_ref.get("valence")),
                float(mean_ref.get("arousal")),
            )
            mean_scaled_tuple = (
                float(mean_scaled.get("valence")),
                float(mean_scaled.get("arousal")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid component metadata entry: {entry!r}"
            ) from exc
        component_map[idx] = ComponentMetadata(
            component=idx,
            quadrant=quadrant,
            weight=weight,
            mean_ref=mean_ref_tuple,
            mean_scaled=mean_scaled_tuple,
        )

    scaler_meta = metadata.get("scaler")
    if not isinstance(scaler_meta, Mapping):
        raise ValueError("metadata missing 'scaler' mapping")
    try:
        scaler_mean = np.asarray(scaler_meta["mean"], dtype=np.float64)
        scaler_scale = np.asarray(scaler_meta["scale"], dtype=np.float64)
    except KeyError as exc:
        raise ValueError("scaler metadata missing required keys") from exc

    with np.load(params_path) as npz:
        try:
            weights = npz["weights"]
            means = npz["means"]
            covariances = npz["covariances"]
        except KeyError as exc:
            raise ValueError("parameter file missing expected arrays") from exc

    return DEAMClusterBundle(
        weights=weights,
        means=means,
        covariances=covariances,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        components=component_map,
        metadata=metadata,
    )


def annotate_parquet_with_clusters(
    *,
    parquet_path: str | Path,
    bundle_dir: str | Path,
    valence_column: str = "fusion_mean_valence",
    arousal_column: str = "fusion_mean_arousal",
    output_path: str | Path | None = None,
) -> Path:
    """Annotate a pipeline parquet with DEAM component and quadrant columns."""

    parquet_path = Path(parquet_path)
    bundle = load_deam_cluster_bundle(bundle_dir)
    df = pd.read_parquet(parquet_path)

    if valence_column not in df.columns:
        raise KeyError(f"Missing valence column '{valence_column}' in parquet")
    if arousal_column not in df.columns:
        raise KeyError(f"Missing arousal column '{arousal_column}' in parquet")

    valence = df[valence_column].to_numpy(dtype=np.float64)
    arousal = df[arousal_column].to_numpy(dtype=np.float64)

    posteriors = bundle.predict_proba(valence, arousal)
    top_components = np.asarray(bundle.top_component(valence, arousal), dtype=int)
    quadrants = bundle.quadrant_for_point(valence, arousal)

    annotated = df.copy()
    annotated["deam_component"] = top_components
    annotated["deam_quadrant"] = quadrants

    for idx in range(bundle.n_components):
        annotated[f"deam_proba_component_{idx}"] = posteriors[:, idx]

    if output_path is None:
        output_path = parquet_path.with_name(
            f"{parquet_path.stem}_with_clusters{parquet_path.suffix}"
        )
    else:
        output_path = Path(output_path)

    annotated.to_parquet(output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        required=True,
        help="Directory containing clusters_meta.json and clusters_params.npz",
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Inference parquet file with fused valence/arousal columns",
    )
    parser.add_argument(
        "--output",
        help="Optional output parquet path; defaults to <input>_with_clusters",
    )
    parser.add_argument(
        "--valence-column",
        default="fusion_mean_valence",
        help="Column containing stabilized valence values",
    )
    parser.add_argument(
        "--arousal-column",
        default="fusion_mean_arousal",
        help="Column containing stabilized arousal values",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = annotate_parquet_with_clusters(
        parquet_path=args.parquet,
        bundle_dir=args.bundle_dir,
        valence_column=args.valence_column,
        arousal_column=args.arousal_column,
        output_path=args.output,
    )
    print(f"Annotated parquet written to: {output_path}")


if __name__ == "__main__":
    main()
