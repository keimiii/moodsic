from __future__ import annotations

import logging
from typing import Any, Dict

from .process_video import _pick_song

LOGGER = logging.getLogger(__name__)


def recommend_song(valence: float, arousal: float, cluster_id: int) -> Dict[str, Any]:
    """Recommend the closest DEAM track for the provided emotion estimate."""

    try:
        song = _pick_song(int(cluster_id), float(valence), float(arousal))
    except Exception as exc:  # pragma: no cover - defensive fallback only
        LOGGER.warning("Falling back to empty recommendation: %s", exc)
        song = {}

    if not song:
        return {}

    return {
        "song_id": str(song.get("song_id")),
        "title": song.get("song_title"),
        "artist": song.get("artist"),
        "genre": song.get("genre"),
        "cluster": int(song.get("cluster", cluster_id)),
        "cluster_conf": song.get("cluster_conf"),
        "valence_ref": song.get("valence_ref"),
        "arousal_ref": song.get("arousal_ref"),
        "distance": song.get("distance"),
    }
