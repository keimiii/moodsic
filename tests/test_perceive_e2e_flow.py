"""
End-to-end PERCEIVE flow smoke tests using real adapters and real images.

Covers two cases:
- Frame without a face: validates scene adapter path returns V/A in [-1, 1].
- Frame with a face: runs face detection + EmoNet adapter and fuses with scene.

Notes:
- Uses real models: SceneCLIPAdapter (CLIP ViT) and EmoNetAdapter.
- Requires transformers to have access to pretrained weights for CLIP; if not
  available (e.g., offline), the test will be skipped.
- EmoNetAdapter loads weights from models/emonet/pretrained if present; if not
  found, it still runs with random init (results are not meaningful, but this
  test only checks for valid outputs in [-1, 1]).
- MediaPipe must be installed for face detection; otherwise the face test
  skips. Paths are provided by the user and the test skips if the files are
  missing.

This suite is structured to be easily extended to STABILIZE and MATCH stages.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np
import pytest


cv2 = pytest.importorskip("cv2")

# Use the runtime one-shot orchestrator
from utils.runtime_driver import perceive_once


# --- Constants (real data paths provided by user) ---
NO_FACE_IMG = (
    "data/Run_2/Affectionate retiree competition/"
    "Elderly-group-in-museum-1200x800.jpg"
)

WITH_FACE_IMG = (
    "data/Run_2/Accepting adults meeting/"
    "videoblocks-diverse-group-of-multinational-business-people-taking-part-in-"
    "corporate-meeting-bored-african-american-man-yawning-while-other-members-"
    "of-seminar-attentively-listening-speech-and-writing-notes_"
    "rsz1toto3e_thumbnail-1080_01.png"
)

# --- Helpers ---
def _load_image_or_skip(path: str) -> np.ndarray:
    if not os.path.exists(path):
        pytest.skip(f"input image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        pytest.skip(f"failed to read image: {path}")
    if img.ndim != 3 or img.shape[2] != 3:
        pytest.skip("expecting a 3-channel BGR image")
    return img


# --- Fixtures: instantiate adapters with robust skipping ---
@pytest.fixture(scope="session")
def scene_adapter():
    """Real CLIP ViT scene adapter. Skips if transformers/weights unavailable."""
    try:
        from models.scene.clip_vit_scene_adapter import SceneCLIPAdapter

        adapter = SceneCLIPAdapter(
            model_name="openai/clip-vit-base-patch32",
            dropout_rate=0.3,
            device="auto",
            tta=3,
        )
        return adapter
    except Exception as e:
        pytest.skip(
            f"SceneCLIPAdapter unavailable (transformers/weights/device): {e}"
        )


@pytest.fixture(scope="session")
def face_processor():
    """MediaPipe-based single-face selector."""
    try:
        from utils.emonet_single_face_processor import EmoNetSingleFaceProcessor

        proc = EmoNetSingleFaceProcessor(min_detection_confidence=0.5, padding_ratio=0.2)
        if not proc.available:
            pytest.skip("MediaPipe not initialized; face detector unavailable")
        return proc
    except Exception as e:
        pytest.skip(f"Face processor unavailable (mediapipe missing?): {e}")


@pytest.fixture(scope="session")
def face_adapter():
    """Real EmoNet face adapter. Runs without weights if missing."""
    try:
        from models.face.emonet_adapter import EmoNetAdapter

        return EmoNetAdapter(
            ckpt_dir="models/emonet/pretrained",  # optional; may be empty
            n_classes=8,
            device="auto",
            tta=3,
            calibration_checkpoint=None,
        )
    except Exception as e:
        pytest.skip(f"EmoNetAdapter unavailable (torch/weights/device): {e}")


# --- Tests ---
def test_perceive_returns_va_no_face(scene_adapter):
    """
    Given a frame with no clear face, PERCEIVE still returns a valid V/A
    via the scene adapter. Face path, if any, is ignored in the final pick.
    """
    img = _load_image_or_skip(NO_FACE_IMG)

    # Use runtime driver one-shot in scene-only mode
    res = perceive_once(
        img,
        scene_predictor=scene_adapter,
        face_processor=face_processor,
        face_expert=face_adapter,
        scene_tta=3,
        face_tta=1,
    )
    # Expect scene path to be valid and fused == scene-only
    assert res.scene is not None and res.scene.valid
    assert res.face is None or not res.face.valid
    v, a = res.fused.valence, res.fused.arousal
    assert math.isfinite(v) and math.isfinite(a)
    assert -1.0 <= v <= 1.0 and -1.0 <= a <= 1.0


def test_perceive_returns_va_with_face(scene_adapter, face_processor, face_adapter):
    """
    Given a frame with a face, run face detection + EmoNet and fuse with scene.
    Returns a fused V/A in reference space [-1, 1].
    """
    img = _load_image_or_skip(WITH_FACE_IMG)

    # Use runtime driver one-shot with both paths
    res = perceive_once(
        img,
        scene_predictor=scene_adapter,
        face_processor=face_processor,
        face_expert=face_adapter,
        scene_tta=3,
        face_tta=3,
    )

    # If face not detected or unusable in this image, skip to match original behavior
    if res.face is None or not res.face.valid:
        pytest.skip("face not detected/usable in WITH_FACE image; skipping")

    # Validate fused outputs are in bounds
    v, a = res.fused.valence, res.fused.arousal
    assert math.isfinite(v) and math.isfinite(a)
    assert -1.0 <= v <= 1.0 and -1.0 <= a <= 1.0


# The structure above is intentionally minimal and ready to extend with:
# - STABILIZE: add EMA + uncertainty gating over sequences of frames
# - MATCH: scale to DEAM [1..9] and run k-NN retrieval with dwell-time
# Keep this file focused on the PERCEIVE stage to avoid heavy dependencies.

# 1s take 1 frame, take the first frame.