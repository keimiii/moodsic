from __future__ import annotations

import logging
from functools import lru_cache
from math import sqrt
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import polars as pl

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_RESULTS_PATH = REPO_ROOT / "results/inference/pipeline_results_20251006_144126.parquet"
SONG_CLUSTER_PATH = REPO_ROOT / "notebooks/Dataset - DEAM/artifacts/deam_gmm/deam_gmm_clusters.csv"
ENRICHED_PARQUET_PATH = PIPELINE_RESULTS_PATH.with_name(
    f"{PIPELINE_RESULTS_PATH.stem}_enriched{PIPELINE_RESULTS_PATH.suffix}"
)
VEATIC_PER_VIDEO_PATH = REPO_ROOT / "results/evaluation/veatic_per_video_20251006_144126.csv"
DEFAULT_STABILIZER = False

_REQUIRED_PIPELINE_COLUMNS = [
    "video_name",
    "stabilizer_enabled",
    "fusion_mean_valence",
    "fusion_mean_arousal",
]
_SONG_DISTANCE_KEY = "distance"
_MAE_COLUMNS = {
    "scene": ("mae_scene_valence", "mae_scene_arousal"),
    "face": ("mae_face_valence", "mae_face_arousal"),
    "fusion": ("mae_fusion_valence", "mae_fusion_arousal"),
}
_MEAN_COLUMNS = {
    "scene": ("mean_scene_valence", "mean_scene_arousal"),
    "face": ("mean_face_valence", "mean_face_arousal"),
    "fusion": ("mean_fusion_valence", "mean_fusion_arousal"),
}


@lru_cache(maxsize=1)
def _load_pipeline_records() -> pl.DataFrame:
    if not PIPELINE_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Pipeline results parquet not found at {PIPELINE_RESULTS_PATH}"
        )
    return pl.read_parquet(PIPELINE_RESULTS_PATH, columns=_REQUIRED_PIPELINE_COLUMNS)


@lru_cache(maxsize=1)
def _load_song_catalog() -> pl.DataFrame:
    if not SONG_CLUSTER_PATH.exists():
        raise FileNotFoundError(
            f"Clustered DEAM catalog not found at {SONG_CLUSTER_PATH}"
        )
    return pl.read_csv(SONG_CLUSTER_PATH)


@lru_cache(maxsize=1)
def _cluster_centroids() -> Dict[int, Tuple[float, float]]:
    catalog = _load_song_catalog()
    grouped = catalog.group_by("cluster").agg(
        [
            pl.col("valence_ref").mean().alias("centroid_valence"),
            pl.col("arousal_ref").mean().alias("centroid_arousal"),
        ]
    )
    return {
        int(row["cluster"]): (float(row["centroid_valence"]), float(row["centroid_arousal"]))
        for row in grouped.iter_rows(named=True)
    }


_SONGS_BY_CLUSTER: Dict[int, pl.DataFrame] = {}


def _songs_for_cluster(cluster_id: int) -> pl.DataFrame:
    if cluster_id not in _SONGS_BY_CLUSTER:
        catalog = _load_song_catalog()
        _SONGS_BY_CLUSTER[cluster_id] = catalog.filter(pl.col("cluster") == cluster_id)
    return _SONGS_BY_CLUSTER[cluster_id]


def _predict_cluster(valence: float, arousal: float) -> int:
    centroids = _cluster_centroids()
    best_cluster = None
    best_distance = float("inf")
    for cluster_id, (centroid_valence, centroid_arousal) in centroids.items():
        distance = sqrt((valence - centroid_valence) ** 2 + (arousal - centroid_arousal) ** 2)
        if distance < best_distance:
            best_cluster = cluster_id
            best_distance = distance
    if best_cluster is None:
        raise ValueError("Unable to infer cluster: centroid table empty")
    return best_cluster


def _pick_song(cluster_id: int, valence: float, arousal: float) -> Dict[str, Any]:
    songs = _songs_for_cluster(cluster_id)
    if songs.is_empty():
        return {}

    scored = songs.with_columns(
        (
            (pl.col("valence_ref") - valence) ** 2
            + (pl.col("arousal_ref") - arousal) ** 2
        )
        .sqrt()
        .alias(_SONG_DISTANCE_KEY)
    ).sort(by=[_SONG_DISTANCE_KEY, "cluster_conf"], descending=[False, True])

    return scored.row(0, named=True)


def _maybe_write_enriched_index() -> None:
    if ENRICHED_PARQUET_PATH.exists():
        return

    try:
        records = _load_pipeline_records()
    except FileNotFoundError as exc:
        LOGGER.warning("Skipping enrichment because parquet missing: %s", exc)
        return

    enrichment_rows = []

    for row in records.iter_rows(named=True):
        valence = float(row["fusion_mean_valence"])
        arousal = float(row["fusion_mean_arousal"])
        cluster_id = _predict_cluster(valence, arousal)
        song = _pick_song(cluster_id, valence, arousal)

        if song:
            song_id = str(song.get("song_id"))
            song_title = song.get("song_title")
            song_artist = song.get("artist")
            song_conf = song.get("cluster_conf")
            song_distance = song.get(_SONG_DISTANCE_KEY)
        else:
            song_id = None
            song_title = None
            song_artist = None
            song_conf = None
            song_distance = None

        enrichment_rows.append({
            "video_name": row["video_name"],
            "stabilizer_enabled": row["stabilizer_enabled"],
            "fusion_mean_valence": valence,
            "fusion_mean_arousal": arousal,
            "cluster_id": cluster_id,
            "song_id": song_id,
            "song_title": song_title,
            "song_artist": song_artist,
            "song_cluster_conf": song_conf,
            "song_distance": song_distance,
        })

    enriched = pl.DataFrame(enrichment_rows)
    enriched.write_parquet(ENRICHED_PARQUET_PATH)
    LOGGER.info("Wrote enriched pipeline index to %s", ENRICHED_PARQUET_PATH)


@lru_cache(maxsize=1)
def _load_per_video_metrics() -> pl.DataFrame:
    if not VEATIC_PER_VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Per-video evaluation CSV not found at {VEATIC_PER_VIDEO_PATH}"
        )
    return pl.read_csv(VEATIC_PER_VIDEO_PATH)


def _pathway_metrics_for_video(
    video_id: str,
    stabilizer_enabled: bool,
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    try:
        metrics = _load_per_video_metrics()
    except FileNotFoundError:
        return {"mae": {}, "means": {}}

    video_matches = metrics.filter(pl.col("video_id").cast(pl.Utf8) == str(video_id))
    if video_matches.is_empty():
        return {"mae": {}, "means": {}}

    stabilizer_key = str(stabilizer_enabled).lower()
    preferred = video_matches.filter(
        pl.col("Stablization").cast(pl.Utf8).str.to_lowercase() == stabilizer_key
    )
    if preferred.is_empty():
        preferred = video_matches

    row = preferred.row(0, named=True)

    def _to_float(value: Any) -> Optional[float]:
        if value in ("", None):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    mae_payload: Dict[str, Dict[str, Optional[float]]] = {}
    mean_payload: Dict[str, Dict[str, Optional[float]]] = {}
    for pathway, (valence_key, arousal_key) in _MAE_COLUMNS.items():
        mae_payload[pathway] = {
            "valence": _to_float(row.get(valence_key)),
            "arousal": _to_float(row.get(arousal_key)),
        }
        mean_valence_key, mean_arousal_key = _MEAN_COLUMNS[pathway]
        mean_payload[pathway] = {
            "valence": _to_float(row.get(mean_valence_key)),
            "arousal": _to_float(row.get(mean_arousal_key)),
        }

    return {"mae": mae_payload, "means": mean_payload}


def process_video_for_emotion(video_id: str) -> Dict[str, Any]:
    """
    Look up fused emotion scores for the provided VEATIC video.

    The inference pipeline already produced per-video fused valence/arousal scores
    in ``results/inference/pipeline_results_20251006_144126.parquet``. We reuse
    those aggregates, approximate the DEAM cluster via centroid distance, and pick
    the closest song within that cluster.
    """

    _maybe_write_enriched_index()

    video_name = f"{video_id}.mp4"
    records = _load_pipeline_records()
    matches = records.filter(pl.col("video_name") == video_name)

    if matches.is_empty():
        raise ValueError(f"Unknown video id: {video_id}")

    preferred = matches.filter(pl.col("stabilizer_enabled") == DEFAULT_STABILIZER)
    if preferred.is_empty():
        preferred = matches

    row = preferred.row(0, named=True)
    valence = float(row["fusion_mean_valence"])
    arousal = float(row["fusion_mean_arousal"])
    stabilizer_enabled = bool(row["stabilizer_enabled"])

    cluster_id = _predict_cluster(valence, arousal)
    song = _pick_song(cluster_id, valence, arousal)
    pathway_metrics = _pathway_metrics_for_video(video_id, stabilizer_enabled)

    result: Dict[str, Any] = {
        "video_name": video_name,
        "stabilizer_enabled": stabilizer_enabled,
        "valence": valence,
        "arousal": arousal,
        "cluster_id": cluster_id,
    }
    if pathway_metrics["mae"]:
        result["mae"] = pathway_metrics["mae"]
    if pathway_metrics["means"]:
        result["pathway_means"] = pathway_metrics["means"]

    if song:
        result.update(
            {
                "recommended_song": {
                    "song_id": str(song.get("song_id")),
                    "title": song.get("song_title"),
                    "artist": song.get("artist"),
                    "cluster_conf": song.get("cluster_conf"),
                    "distance": song.get(_SONG_DISTANCE_KEY),
                }
            }
        )

    return result
