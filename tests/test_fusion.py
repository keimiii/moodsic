"""
Tests for SceneFaceFusion and its public entrypoint `perceive_and_fuse`.

What this suite covers (high level):
- Inverse-variance fusion math: fused mean uses precision weights (1/var),
  fused variance equals 1 / (sum of precisions).
- Safety clamps: outputs are clipped to the reference scale [-1, 1].
- Fallbacks: when any variance is missing/invalid, use fixed weights; when the
  sum of fixed weights is degenerate (0), use equal weights.
- Edge-case routing: passthrough when only one path is valid; neutral output
  when neither path is valid.
- Public API integration: `perceive_and_fuse` behavior for scene-only, face-only,
  and both paths present; includes a real-image fallback test (auto-skips if the
  sample image is unavailable).

Why these assertions matter:
- They document the intended fusion behavior without needing to read the model
  implementations.
- They provide quick feedback if fusion math or edge handling regresses during
  refactors.

Notes for newcomers:
- Valence (V) and Arousal (A) are scalar predictions in [-1, 1].
- Variance reflects uncertainty per dimension; lower variance means higher
  confidence and thus higher weight in inverse-variance fusion.
- We use simple mocks so tests are fast and do not require ML dependencies.

New in this suite (stability guardrails):
- Optional "gating" that temporarily ignores the face path for a frame to avoid
  jitter when the face is unreliable (occluded/low light):
  - Gate by face score (low detector confidence)
  - Gate by sigma (uncertainty too high)
  - Gate by brightness (frame too dark)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pytest

from models.fusion import EmotionPrediction, SceneFaceFusion


# ---- Helpers and Mocks ----------------------------------------------------

class MockScenePredictor:
    """Minimal scene predictor that returns fixed values for testing."""

    def __init__(self, v: float, a: float, var_v: Optional[float], var_a: Optional[float]):
        self.v = float(v)
        self.a = float(a)
        self.var_v = var_v
        self.var_a = var_a

    def predict(self, frame_bgr: np.ndarray, tta: int = 10):
        return self.v, self.a, (self.var_v, self.var_a)


class MockFaceProcessor:
    """Returns either no face or a dummy crop + bbox for testing."""

    def __init__(self, mode: str = "none"):
        self.mode = mode

    def extract_primary_face(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float]:
        if self.mode == "none":
            return None, None, 0.0
        # Return a dummy non-empty crop and bbox
        crop = np.zeros((224, 224, 3), dtype=np.uint8)
        bbox = (10, 10, 50, 50)
        return crop, bbox, 0.9


class MockFaceExpert:
    """Minimal face expert (EmoNet adapter stand-in) with fixed outputs."""

    def __init__(self, v: float, a: float, var_v: Optional[float], var_a: Optional[float]):
        self.v = float(v)
        self.a = float(a)
        self.var_v = var_v
        self.var_a = var_a

    def predict(self, crop: np.ndarray, tta: int = 5):
        return self.v, self.a, (self.var_v, self.var_a)


def iv_fuse_expected(p1: float, v1: float, p2: float, v2: float) -> Tuple[float, float]:
    """Compute expected inverse-variance fused mean and variance.

    The fused mean is weighted by precision (1/variance).
    The fused variance is 1 / (sum of precisions).
    """

    w1 = 1.0 / (v1 + 1e-6)
    w2 = 1.0 / (v2 + 1e-6)
    total = w1 + w2
    mean = (w1 * p1 + w2 * p2) / total
    variance = 1.0 / total
    return mean, variance


# ---- Unit tests: fusion math ---------------------------------------------

def test_fuse_scalar_inverse_variance_math():
    """Inverse-variance fusion matches the closed-form solution."""

    fusion = SceneFaceFusion(use_variance_weighting=True)
    scene_mean, scene_var = 0.6, 0.09  # σ≈0.30
    face_mean, face_var = -0.2, 0.25   # σ=0.50

    fused_mean, fused_var = fusion._fuse_scalar(scene_mean, scene_var, face_mean, face_var)
    exp_mean, exp_var = iv_fuse_expected(scene_mean, scene_var, face_mean, face_var)

    assert pytest.approx(fused_mean, rel=1e-5, abs=1e-5) == exp_mean
    assert pytest.approx(fused_var, rel=1e-5, abs=1e-5) == exp_var
    # Lower variance path (scene) should influence the result more.
    assert abs(fused_mean - scene_mean) < abs(fused_mean - face_mean)


def test_fuse_scalar_clips_to_reference_range():
    """Outputs are clamped to [-1, 1] even with extreme inputs."""

    fusion = SceneFaceFusion(use_variance_weighting=False, scene_weight=1.0, face_weight=0.0)
    fused, var = fusion._fuse_scalar(2.0, None, 2.0, None)
    assert fused == 1.0
    assert var is None


def test_fixed_weights_used_when_any_variance_missing():
    """If a variance is missing/invalid, fusion falls back to fixed weights."""

    fusion = SceneFaceFusion(use_variance_weighting=True, scene_weight=0.6, face_weight=0.4)
    scene_mean, face_mean = 0.5, -0.5
    fused, var = fusion._fuse_scalar(scene_mean, None, face_mean, 0.1)
    expected = (0.6 * scene_mean + 0.4 * face_mean) / (0.6 + 0.4)
    assert pytest.approx(fused, rel=1e-6) == expected
    assert var is None


def test_fixed_weights_degenerate_sum_replaced_with_equal_weights():
    """If scene_weight + face_weight == 0, use equal weights (0.5, 0.5)."""

    fusion = SceneFaceFusion(use_variance_weighting=False, scene_weight=0.0, face_weight=0.0)
    fused, var = fusion._fuse_scalar(0.8, None, -0.2, None)
    assert pytest.approx(fused, rel=1e-6) == (0.8 - 0.2) / 2.0
    assert var is None


# ---- Unit tests: edge-case routing ---------------------------------------

def test_fuse_passthrough_when_only_one_path_valid():
    """When only one path is valid, its prediction is returned unchanged."""

    fusion = SceneFaceFusion()
    scene = EmotionPrediction(0.1, -0.2, 0.04, 0.09, valid=True)
    face = EmotionPrediction(0.9, 0.9, 0.01, 0.01, valid=False)
    fused = fusion._fuse(scene, face)
    assert fused.valence == scene.valence
    assert fused.arousal == scene.arousal
    assert fused.var_valence == scene.var_valence
    assert fused.var_arousal == scene.var_arousal


def test_fuse_returns_neutral_when_both_invalid():
    """Neutral output with unit variance when neither path is valid."""

    fusion = SceneFaceFusion()
    scene = EmotionPrediction(0.0, 0.0, None, None, valid=False)
    face = EmotionPrediction(0.0, 0.0, None, None, valid=False)
    fused = fusion._fuse(scene, face)
    assert fused.valence == 0.0 and fused.arousal == 0.0
    assert fused.var_valence == 1.0 and fused.var_arousal == 1.0
    assert fused.valid is False


# ---- Integration tests: public API ---------------------------------------

def test_perceive_and_fuse_scene_only_path():
    """Scene-only configuration: fused result should equal scene prediction."""

    scene = MockScenePredictor(0.0, 0.5, 0.04, 0.09)
    fusion = SceneFaceFusion(scene_predictor=scene)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    res = fusion.perceive_and_fuse(frame)
    assert res.scene is not None and res.scene.valid
    assert res.face is None or not res.face.valid
    assert pytest.approx(res.fused.valence) == 0.0
    assert pytest.approx(res.fused.arousal) == 0.5
    assert pytest.approx(res.fused.var_valence) == 0.04
    assert pytest.approx(res.fused.var_arousal) == 0.09


def test_perceive_and_fuse_face_only_path():
    """Face-only configuration: fused result should equal face prediction."""

    face_proc = MockFaceProcessor(mode="valid")
    face_exp = MockFaceExpert(0.3, -0.4, 0.01, 0.02)
    fusion = SceneFaceFusion(face_expert=face_exp, face_processor=face_proc)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    res = fusion.perceive_and_fuse(frame)
    assert res.face is not None and res.face.valid
    assert res.scene is None or not res.scene or not getattr(res.scene, "valid", False)
    assert pytest.approx(res.fused.valence) == 0.3
    assert pytest.approx(res.fused.arousal) == -0.4
    assert pytest.approx(res.fused.var_valence) == 0.01
    assert pytest.approx(res.fused.var_arousal) == 0.02
    assert res.face_bbox == (10, 10, 50, 50)
    assert res.face_score > 0


def test_perceive_and_fuse_both_paths_inverse_variance():
    """Both paths present: fused value follows inverse-variance weighting per dim."""

    scene = MockScenePredictor(0.6, 0.2, 0.09, 0.25)  # σ≈0.30, 0.50
    face_exp = MockFaceExpert(-0.2, 0.8, 0.16, 0.04)  # σ=0.40, 0.20
    face_proc = MockFaceProcessor(mode="valid")
    fusion = SceneFaceFusion(
        scene_predictor=scene,
        face_expert=face_exp,
        face_processor=face_proc,
        use_variance_weighting=True,
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    res = fusion.perceive_and_fuse(frame)

    exp_v, _ = iv_fuse_expected(0.6, 0.09, -0.2, 0.16)
    exp_a, _ = iv_fuse_expected(0.2, 0.25, 0.8, 0.04)
    assert pytest.approx(res.fused.valence, rel=1e-5) == exp_v
    assert pytest.approx(res.fused.arousal, rel=1e-5) == exp_a


def test_fallback_to_scene_when_no_face_detected_with_real_image():
    """Uses a real image (if present) to exercise the face-missing fallback path."""

    cv2 = pytest.importorskip("cv2")

    img_path = os.path.join(
        "data",
        "Run_2",
        "Affectionate retiree competition",
        "Elderly-group-in-museum-1200x800.jpg",
    )
    if not os.path.exists(img_path):
        pytest.skip("sample image not found; skipping real-image fallback test")

    frame = cv2.imread(img_path)
    assert frame is not None, "failed to read test image"

    # Force face path to return None to trigger fallback
    scene = MockScenePredictor(0.2, -0.1, 0.04, 0.09)
    face_proc = MockFaceProcessor(mode="none")
    face_exp = MockFaceExpert(0.0, 0.0, None, None)  # won't be used

    fusion = SceneFaceFusion(
        scene_predictor=scene,
        face_expert=face_exp,
        face_processor=face_proc,
        use_variance_weighting=True,
    )
    res = fusion.perceive_and_fuse(frame)
    assert pytest.approx(res.fused.valence) == 0.2
    assert pytest.approx(res.fused.arousal) == -0.1
    assert pytest.approx(res.fused.var_valence) == 0.04
    assert pytest.approx(res.fused.var_arousal) == 0.09


# ---- New unit tests: stability guardrails (gating) -------------------------

class MockFaceProcessorWithScore:
    """Face processor that always returns a face crop with a configurable score."""

    def __init__(self, score: float = 0.9):
        self.score = float(score)

    def extract_primary_face(self, frame_bgr: np.ndarray):
        crop = np.zeros((224, 224, 3), dtype=np.uint8) + 255
        bbox = (5, 5, 40, 40)
        return crop, bbox, self.score


def test_gating_by_face_score_disables_unreliable_face():
    """When face score is below threshold, fusion falls back to scene output."""

    scene = MockScenePredictor(0.4, -0.2, 0.04, 0.09)
    # Face would otherwise pull result away from scene
    face_exp = MockFaceExpert(-0.8, 0.9, 0.01, 0.02)
    face_proc = MockFaceProcessorWithScore(score=0.2)

    fusion = SceneFaceFusion(
        scene_predictor=scene,
        face_expert=face_exp,
        face_processor=face_proc,
        use_variance_weighting=True,
        face_score_threshold=0.5,  # gate off low-score faces
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    res = fusion.perceive_and_fuse(frame)

    # Fused should equal scene since face was gated invalid
    assert pytest.approx(res.fused.valence) == 0.4
    assert pytest.approx(res.fused.arousal) == -0.2


# ---- New unit tests: post-fusion EMA stabilizer ---------------------------

class MockScenePredictorSeq:
    """Scene predictor that yields a sequence of (v,a,var_v,var_a) per call."""

    def __init__(self, seq):
        self.seq = list(seq)

    def predict(self, frame_bgr: np.ndarray, tta: int = 10):
        if not self.seq:
            # repeat last if exhausted
            v, a, vv, va = 0.0, 0.0, 0.01, 0.01
        else:
            v, a, vv, va = self.seq.pop(0)
        return float(v), float(a), (vv, va)


def test_stabilizer_holds_on_high_fused_variance():
    """EMA stabilizer should hold last stable values when variance spikes."""

    # Sequence: stable low-var → spike high-var → back to low-var
    seq = [
        (0.0, 0.0, 0.01, 0.01),  # initialize EMA and last_stable
        (0.8, 0.8, 1.0, 1.0),     # high variance → hold last_stable (0.0, 0.0)
        (0.4, 0.4, 0.01, 0.01),  # low variance → EMA update from previous EMA
    ]
    scene = MockScenePredictorSeq(seq)
    fusion = SceneFaceFusion(
        scene_predictor=scene,
        enable_stabilizer=True,
        stabilizer_alpha=0.5,
        uncertainty_threshold=0.4,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    res1 = fusion.perceive_and_fuse(frame)
    assert pytest.approx(res1.fused.valence) == 0.0
    assert pytest.approx(res1.fused.arousal) == 0.0

    res2 = fusion.perceive_and_fuse(frame)
    # Despite input 0.8 with high variance, output should hold at last stable (0.0)
    assert pytest.approx(res2.fused.valence) == 0.0
    assert pytest.approx(res2.fused.arousal) == 0.0

    res3 = fusion.perceive_and_fuse(frame)
    # Now low variance; EMA updates were applied even during the gated step,
    # so prior EMA is 0.4. With input 0.4 and alpha=0.5, EMA stays at 0.4.
    assert pytest.approx(res3.fused.valence, rel=1e-6) == 0.4
    assert pytest.approx(res3.fused.arousal, rel=1e-6) == 0.4


def test_stability_metrics_present_when_enabled():
    """FusionResult should include stability metrics when stabilizer is enabled."""

    scene = MockScenePredictorSeq([
        (0.0, 0.0, 0.01, 0.01),
        (0.1, 0.1, 0.01, 0.01),
        (0.2, 0.2, 0.01, 0.01),
    ])
    fusion = SceneFaceFusion(scene_predictor=scene, enable_stabilizer=True)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    _ = fusion.perceive_and_fuse(frame)
    res = fusion.perceive_and_fuse(frame)
    # After two frames, metrics should be available
    assert res.stability_variance is None or (
        isinstance(res.stability_variance, tuple) and len(res.stability_variance) == 2
    )
    assert res.stability_jitter is None or (
        isinstance(res.stability_jitter, tuple) and len(res.stability_jitter) == 2
    )


def test_gating_by_sigma_disables_high_uncertainty_face():
    """When face sigma is too large, face is ignored in favor of scene."""

    scene = MockScenePredictor(0.1, 0.1, 0.04, 0.04)
    # Large variance → sigma ~ 1.0 > threshold below
    face_exp = MockFaceExpert(-1.0, 1.0, 1.0, 1.0)
    face_proc = MockFaceProcessorWithScore(score=0.9)

    fusion = SceneFaceFusion(
        scene_predictor=scene,
        face_expert=face_exp,
        face_processor=face_proc,
        use_variance_weighting=True,
        face_max_sigma=0.5,  # gate off faces with sigma > 0.5
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    res = fusion.perceive_and_fuse(frame)

    assert pytest.approx(res.fused.valence) == 0.1
    assert pytest.approx(res.fused.arousal) == 0.1


def test_gating_by_brightness_disables_face_on_dark_frames():
    """On dark frames (low luma), face is ignored for stability."""

    scene = MockScenePredictor(0.2, -0.2, 0.04, 0.04)
    face_exp = MockFaceExpert(-0.9, 0.9, 0.01, 0.01)
    face_proc = MockFaceProcessorWithScore(score=0.9)

    fusion = SceneFaceFusion(
        scene_predictor=scene,
        face_expert=face_exp,
        face_processor=face_proc,
        use_variance_weighting=True,
        brightness_threshold=10.0,  # black frame has luma=0 < 10 → gate face
    )
    # Pure black image
    dark = np.zeros((64, 64, 3), dtype=np.uint8)
    res = fusion.perceive_and_fuse(dark)

    assert pytest.approx(res.fused.valence) == 0.2
    assert pytest.approx(res.fused.arousal) == -0.2
