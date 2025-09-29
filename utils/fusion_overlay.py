from __future__ import annotations

import math
from typing import Optional

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - handle environments without OpenCV
    cv2 = None  # type: ignore
import numpy as np

try:
    # Typed import if available
    from models.fusion import FusionResult
except Exception:  # pragma: no cover - type only
    FusionResult = object  # type: ignore


def draw_fusion_overlay(img: np.ndarray, result: FusionResult) -> np.ndarray:
    """
    Draw a lightweight overlay of fused and per-path predictions on a BGR image.

    - Does not perform inference; expects a FusionResult from SceneFaceFusion.
    - Draws face bbox if present, plus text lines with valence/arousal and
      uncertainties when available. We show σ (sigma) as the square root of
      variance — larger σ means less confidence.
    - Returns a new image; original is not modified.
    """
    # If OpenCV is unavailable or the image is not a standard BGR frame, return input
    if cv2 is None or img is None or img.ndim != 3 or img.shape[2] != 3:
        return img

    out = img.copy()
    overlay = img.copy()

    # Draw sampled face bboxes if present
    face_samples = getattr(result, "face_samples", None)
    if face_samples:
        for idx, sample in enumerate(face_samples):
            try:
                x, y, bw, bh = sample.bbox
            except Exception:
                continue
            color = (0, 255, 0) if idx == 0 else (0, 255, 255)
            thickness = 2 if idx == 0 else 1
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, thickness)
            if idx < 3:  # avoid clutter
                label = f"#{idx+1}"
                cv2.putText(
                    overlay,
                    label,
                    (max(0, x), max(15, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    elif getattr(result, "face_bbox", None):
        x, y, bw, bh = result.face_bbox
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    # Compose text lines
    lines = []
    fused = result.fused
    if getattr(fused, "var_valence", None) is not None and getattr(fused, "var_arousal", None) is not None:
        lines.append(
            f"Fused V={fused.valence:+.2f} A={fused.arousal:+.2f} "
            f"(σv={math.sqrt(max(fused.var_valence, 0.0)):.2f}, "
            f"σa={math.sqrt(max(fused.var_arousal, 0.0)):.2f})"
        )
    else:
        lines.append(f"Fused V={fused.valence:+.2f} A={fused.arousal:+.2f}")

    if result.face and result.face.valid:
        face = result.face
        if face.var_valence is not None and face.var_arousal is not None:
            face_sig = (
                f"σv={math.sqrt(max(face.var_valence, 0.0)):.2f}, "
                f"σa={math.sqrt(max(face.var_arousal, 0.0)):.2f}"
            )
        else:
            face_sig = "σv=?, σa=?"
        lines.append(f"Face  V={face.valence:+.2f} A={face.arousal:+.2f} ({face_sig})")
    else:
        lines.append("Face  unavailable")

    if result.scene and result.scene.valid:
        scene = result.scene
        if scene.var_valence is not None and scene.var_arousal is not None:
            scene_sig = (
                f"σv={math.sqrt(max(scene.var_valence, 0.0)):.2f}, "
                f"σa={math.sqrt(max(scene.var_arousal, 0.0)):.2f}"
            )
        else:
            scene_sig = "σv=?, σa=?"
        lines.append(f"Scene V={scene.valence:+.2f} A={scene.arousal:+.2f} ({scene_sig})")
    else:
        lines.append("Scene not configured")

    # Draw backdrop rectangle
    pad = 6
    line_h = 22
    box_w = max(240, max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] for l in lines) + 2 * pad)
    box_h = line_h * len(lines) + 2 * pad
    x0, y0 = 10, 10
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    alpha = 0.5
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    # Put text
    y = y0 + pad + 16
    for l in lines:
        cv2.putText(out, l, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_h
    return out


__all__ = ["draw_fusion_overlay"]
