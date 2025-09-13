"""
Tests for the debug overlay utility.

What this suite covers:
- Overlay draws a face bbox and text (including σ when variances are present).
- Overlay draws a text panel even when variances or bbox are missing.
- Output image shape matches the input; the function draws onto a copy so the
  original is unchanged.

Practical choice: we do not assert exact text strings or pixel-by-pixel content
to avoid flakiness across platforms, fonts, and OpenCV builds. Instead we assert
that some pixels changed, which indicates the overlay rendered.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from models.fusion import EmotionPrediction, FusionResult
from utils.fusion_overlay import draw_fusion_overlay


# Skip overlay tests if OpenCV is unavailable in the environment.
cv2 = pytest.importorskip("cv2")


def make_pred(v, a, vv=None, va=None, valid=True):
    return EmotionPrediction(float(v), float(a), vv, va, valid=valid)


def test_overlay_draws_bbox_and_variance_text():
    """Overlay with bbox and variances should alter some pixels and keep shape."""

    img = np.zeros((120, 160, 3), dtype=np.uint8)
    scene = make_pred(0.1, -0.2, 0.04, 0.09)
    face = make_pred(-0.3, 0.7, 0.16, 0.04)
    fused = make_pred(0.0, 0.1, 0.02, 0.03)
    res = FusionResult(
        scene=scene,
        face=face,
        fused=fused,
        face_bbox=(20, 15, 40, 30),
        face_score=0.85,
    )

    out = draw_fusion_overlay(img, res)
    assert out.shape == img.shape
    assert np.any(out != img)  # some pixels should change


def test_overlay_draws_without_bbox_or_variances():
    """Overlay still draws a text panel when no bbox/variances are present."""

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    scene = make_pred(0.2, 0.3, None, None)
    face = make_pred(-0.1, 0.0, None, None)
    fused = make_pred(0.05, 0.15, None, None)
    res = FusionResult(scene=scene, face=face, fused=fused, face_bbox=None, face_score=0.0)

    out = draw_fusion_overlay(img, res)
    assert out.shape == img.shape
    assert np.any(out != img)


def test_overlay_real_image_and_export_artifact():
    """Draw overlay on a real image and save an artifact for manual inspection.

    - Uses a sample from the repo (if present).
    - Writes output to tests/_artifacts/overlay_real.png.
    - Skips if the input image is not available to keep CI green.
    """

    # Locate real image relative to repository root
    img_path = os.path.join(
        "data",
        "Run_2",
        "Affectionate babies holiday",
        "f3c7071d4ad36a5b6a5148635ba291f7.png",
    )
    if not os.path.exists(img_path):
        pytest.skip("real overlay sample not found; skipping export artifact test")

    img = cv2.imread(img_path)
    assert img is not None, "failed to read real test image"

    # Compose a plausible FusionResult; we don't depend on actual predictions here.
    h, w = img.shape[:2]
    scene = make_pred(0.10, -0.05, 0.04, 0.06)
    face = make_pred(-0.20, 0.35, 0.09, 0.02)
    fused = make_pred(0.00, 0.10, 0.03, 0.03)
    # Draw a bbox somewhere near the center for visibility
    bbox_w, bbox_h = max(40, w // 8), max(40, h // 8)
    bbox_x, bbox_y = max(5, w // 3), max(5, h // 3)
    bbox = (
        int(min(w - 1, bbox_x)),
        int(min(h - 1, bbox_y)),
        int(min(w - bbox_x - 1, bbox_w)),
        int(min(h - bbox_y - 1, bbox_h)),
    )

    res = FusionResult(scene=scene, face=face, fused=fused, face_bbox=bbox, face_score=0.9)
    out = draw_fusion_overlay(img, res)
    assert out.shape == img.shape
    assert np.any(out != img)

    # Save artifact to tests/_artifacts
    artifacts_dir = Path(__file__).resolve().parent / "_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "overlay_real.png"
    ok = cv2.imwrite(str(out_path), out)
    assert ok is True
    assert out_path.exists()
