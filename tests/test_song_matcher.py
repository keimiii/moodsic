from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from utils.song_matcher import MatchResult, SongMatcher


class IdentityScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)


class ConstantGMM:
    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        return np.tile(self._probs, (X.shape[0], 1))


def sample_songs():
    return pd.DataFrame(
        [
            {"song_id": "a", "valence_ref": 0.1, "arousal_ref": 0.0, "cluster": 0},
            {"song_id": "b", "valence_ref": 0.0, "arousal_ref": 0.0, "cluster": 1},
            {"song_id": "c", "valence_ref": 0.2, "arousal_ref": 0.0, "cluster": 0},
        ]
    )


def test_gmm_gating_prefers_top1_cluster():
    songs = sample_songs()
    matcher = SongMatcher(
        songs,
        IdentityScaler(),
        ConstantGMM([0.9, 0.1]),
        min_dwell_time=0,
        recent_k=5,
        top2_threshold=0.55,
    )

    res = matcher.recommend(0.0, 0.0, now=0.0)
    assert isinstance(res, MatchResult)
    assert res.song is not None
    assert res.song["song_id"] == "a"


def test_gmm_gating_widens_to_top2_when_uncertain():
    songs = sample_songs()
    matcher = SongMatcher(
        songs,
        IdentityScaler(),
        ConstantGMM([0.45, 0.35, 0.20]),
        min_dwell_time=0,
        recent_k=5,
        top2_threshold=0.55,
    )

    res = matcher.recommend(0.0, 0.0, now=0.0)
    assert res.song is not None
    assert res.song["song_id"] == "b"


def test_dwell_time_blocks_switch_until_elapsed():
    songs = sample_songs()
    matcher = SongMatcher(
        songs,
        IdentityScaler(),
        ConstantGMM([0.9, 0.1]),
        min_dwell_time=25.0,
        recent_k=5,
    )

    first = matcher.recommend(0.0, 0.0, now=0.0)
    second = matcher.recommend(0.5, 0.0, now=10.0)
    assert second.song is first.song
    assert second.switch is False

    third = matcher.recommend(0.5, 0.0, now=30.0)
    assert third.switch is True


def test_recent_memory_skips_recent_when_alternative_exists():
    songs = sample_songs()
    matcher = SongMatcher(
        songs,
        IdentityScaler(),
        ConstantGMM([0.9, 0.1]),
        min_dwell_time=0,
        recent_k=2,
    )

    first = matcher.recommend(0.0, 0.0, now=0.0)
    second = matcher.recommend(0.0, 0.0, now=1.0)
    assert first.song is not None
    assert second.song is not None
    assert first.song["song_id"] == "a"
    assert second.song["song_id"] == "c"


def test_from_artifacts_loads_scaler_and_gmm(tmp_path: Path):
    songs = sample_songs()
    scaler = IdentityScaler()
    gmm = ConstantGMM([0.8, 0.2])

    joblib.dump(scaler, tmp_path / "scaler.pkl")
    joblib.dump(gmm, tmp_path / "gmm.pkl")

    matcher = SongMatcher.from_artifacts(songs, tmp_path)
    res = matcher.recommend(0.0, 0.0, now=0.0)
    assert res.song is not None
    assert res.song["song_id"] == "a"
