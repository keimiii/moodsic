from __future__ import annotations

import math
import logging
from typing import Optional, Tuple

import cv2
import numpy as np


class EmoNetSingleFaceProcessor:
    """
    MediaPipe-based single face selector and cropper for EmoNet pipeline.

    - Input frames are expected in BGR format (OpenCV convention).
    - Internally converts to RGB for MediaPipe detection.
    - Selects the primary face per frame using: score * sqrt(area) * (0.7 + 0.3 * center_proximity).
    - Returns a padded BGR crop and its bbox with a combined score.
    - Uses MediaPipe's long-range face detector by default (model_selection=1)
      for better recall on distant faces.

    Notes:
    - When MediaPipe is unavailable, the processor remains importable and
      `extract_primary_face` will return (None, None, 0.0).
    - `bbox` is returned as (x, y, w, h) in absolute integer pixels, clamped to frame bounds.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        padding_ratio: float = 0.2,
        output_size: Optional[Tuple[int, int]] = (256, 256),
        model_selection: int = 1,
    ) -> None:
        self.min_detection_confidence = float(min_detection_confidence)
        self.padding_ratio = float(padding_ratio)
        self.output_size = output_size
        self.model_selection = int(model_selection)

        # Lazy / safe import for environments without mediapipe installed yet.
        self._mp = None
        self._detector = None
        try:
            import mediapipe as mp  # type: ignore

            self._mp = mp
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=self.model_selection,
                min_detection_confidence=self.min_detection_confidence,
            )
        except Exception as e:  # pragma: no cover - environment dependent
            logging.getLogger(__name__).debug(
                "EmoNetSingleFaceProcessor: MediaPipe not available (%s). "
                "Face extraction will return None.",
                e,
            )

    @property
    def available(self) -> bool:
        """Whether MediaPipe detector is available."""
        return self._detector is not None

    def extract_primary_face(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float]:
        """
        Detect faces and return the primary face crop.

        Args:
            frame_bgr: Input image in BGR format of shape (H, W, 3).

        Returns:
            - face_crop_bgr: Cropped face in BGR format (optionally resized to `output_size`), or None
            - bbox: (x, y, w, h) in absolute pixels within the original frame, or None
            - score: Combined selection score for the chosen face (0.0 if none)
        """
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return None, None, 0.0

        if not self.available:
            return None, None, 0.0

        try:
            h, w = frame_bgr.shape[:2]
            # MediaPipe expects RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._detector.process(frame_rgb)

            detections = getattr(results, "detections", None)
            if not detections:
                return None, None, 0.0

            # Select best detection
            best_det, best_score = self._select_best_detection(detections, (h, w))
            if best_det is None:
                return None, None, 0.0

            # Convert relative bbox to absolute with padding
            bbox = self._relative_bbox_to_abs(best_det, (h, w), self.padding_ratio)
            x, y, bw, bh = bbox

            # Crop from original BGR frame
            crop = frame_bgr[y : y + bh, x : x + bw]

            # Optionally resize for downstream alignment step
            if self.output_size is not None and crop.size > 0:
                ow, oh = self.output_size[0], self.output_size[1]
                crop = cv2.resize(crop, (ow, oh))

            # Return crop, bbox (w.r.t original image), and score
            return crop if crop.size > 0 else None, bbox, float(best_score)

        except Exception as e:  # pragma: no cover - robustness
            logging.getLogger(__name__).debug(
                "EmoNetSingleFaceProcessor: exception during extraction: %s", e
            )
            return None, None, 0.0

    # ---------- Internals ----------
    def _select_best_detection(self, detections, hw: Tuple[int, int]):
        """
        Choose the primary face using:
        score = conf * sqrt(area) * (0.7 + 0.3 * center_proximity)

        - conf: MediaPipe detection.score[0]
        - area: relative bbox area (width * height)
        - center_proximity: 1 - (distance(face_center, image_center) / max_distance)
        """
        h, w = hw
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        max_distance = float(np.linalg.norm(center)) + 1e-6

        best = None
        best_score = -np.inf

        for det in detections:
            try:
                rel_bb = det.location_data.relative_bounding_box
                conf = float(det.score[0]) if hasattr(det, "score") and det.score else 0.0

                # Relative bbox metrics
                rel_w = float(getattr(rel_bb, "width", 0.0))
                rel_h = float(getattr(rel_bb, "height", 0.0))
                rel_x = float(getattr(rel_bb, "xmin", 0.0))
                rel_y = float(getattr(rel_bb, "ymin", 0.0))

                # Skip extremely small or invalid boxes
                if rel_w <= 0 or rel_h <= 0:
                    continue

                area = max(rel_w * rel_h, 0.0)
                face_center_px = np.array(
                    [
                        (rel_x + rel_w / 2.0) * w,
                        (rel_y + rel_h / 2.0) * h,
                    ],
                    dtype=np.float32,
                )
                dist = float(np.linalg.norm(face_center_px - center))
                proximity = 1.0 - (dist / max_distance)
                proximity = float(np.clip(proximity, 0.0, 1.0))

                score = conf * math.sqrt(area) * (0.7 + 0.3 * proximity)

                if score > best_score:
                    best = rel_bb
                    best_score = score
            except Exception:
                continue

        if best is None or not np.isfinite(best_score):
            return None, 0.0
        return best, float(best_score)

    @staticmethod
    def _relative_bbox_to_abs(rel_bb, hw: Tuple[int, int], padding_ratio: float) -> Tuple[int, int, int, int]:
        """
        Convert MediaPipe relative bbox to absolute (x, y, w, h) with padding.
        Values are clamped to image bounds.
        """
        h, w = hw

        rel_x = float(getattr(rel_bb, "xmin", 0.0))
        rel_y = float(getattr(rel_bb, "ymin", 0.0))
        rel_w = float(getattr(rel_bb, "width", 0.0))
        rel_h = float(getattr(rel_bb, "height", 0.0))

        # Apply padding on all sides proportionally to bbox size
        pad_w = padding_ratio * rel_w
        pad_h = padding_ratio * rel_h

        x0 = (rel_x - pad_w) * w
        y0 = (rel_y - pad_h) * h
        x1 = (rel_x + rel_w + pad_w) * w
        y1 = (rel_y + rel_h + pad_h) * h

        x0 = int(max(0, math.floor(x0)))
        y0 = int(max(0, math.floor(y0)))
        x1 = int(min(w, math.ceil(x1)))
        y1 = int(min(h, math.ceil(y1)))

        abs_w = max(0, x1 - x0)
        abs_h = max(0, y1 - y0)

        return x0, y0, abs_w, abs_h
