from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple
from types import SimpleNamespace

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
        self._cascade = None
        self._cascade_available = False
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

        # Prepare optional OpenCV cascade fallback for multi-face harvesting
        try:
            cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                raise ValueError("Cascade classifier failed to load")
        except Exception:
            self._cascade = None
            self._cascade_available = False
        else:
            self._cascade = cascade
            self._cascade_available = True

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
        faces = self.extract_faces(frame_bgr, max_faces=1, sampling="topk")
        if not faces:
            return None, None, 0.0

        crop, bbox, score = faces[0]
        return crop, bbox, score

    def extract_faces(
        self,
        frame_bgr: np.ndarray,
        *,
        max_faces: int = 1,
        sampling: str = "topk",
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Detect faces and return up to `max_faces` crops according to `sampling`.

        sampling:
            - "topk": deterministic top-k by score (original behavior).
            - "weighted": sample without replacement proportionally to score,
              using `temperature` as a softmax temperature (>0).
        """
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return []

        if max_faces <= 0:
            return []


        scored_faces: List[Tuple[float, Tuple[int, int, int, int], np.ndarray]] = []

        try:
            h, w = frame_bgr.shape[:2]
            if self.available:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self._detector.process(frame_rgb)
                detections = getattr(results, "detections", None)
                if detections:
                    for det in detections:
                        info = self._face_from_detection(det, frame_bgr, (h, w))
                        if info is None:
                            continue
                        scored_faces.append(info)

            # Optional OpenCV fallback when MediaPipe returns too few faces
            if (
                len(scored_faces) < max_faces
                and self._cascade_available
                and self._cascade is not None
            ):
                needed = max_faces - len(scored_faces)
                fallback = self._detect_faces_cv2(
                    frame_bgr,
                    (h, w),
                    [bbox for _, bbox, _ in scored_faces],
                    limit=needed,
                )
                scored_faces.extend(fallback)

            if not scored_faces:
                return []

            limit = min(max_faces, len(scored_faces))
            if sampling == "weighted" and limit > 0 and len(scored_faces) > 1:
                temp = max(1e-3, float(temperature))
                scores = np.array([sf[0] for sf in scored_faces], dtype=np.float64)
                logits = scores / temp
                logits -= float(np.max(logits))
                weights = np.exp(logits)
                if not np.isfinite(weights).all() or weights.sum() <= 0:
                    weights = np.ones_like(weights)
                weights /= weights.sum()
                rng = np.random.default_rng(seed)
                indices = rng.choice(len(scored_faces), size=limit, replace=False, p=weights)
                indices = list(indices)
                rng.shuffle(indices)
                selected = [scored_faces[i] for i in indices]
            else:
                selected = sorted(scored_faces, key=lambda item: item[0], reverse=True)[:limit]

            result: List[Tuple[np.ndarray, Tuple[int, int, int, int], float]] = []
            for score, bbox, crop in selected:
                if crop is None or crop.size == 0:
                    continue
                result.append((crop, bbox, float(score)))
            return result

        except Exception as e:  # pragma: no cover - robustness
            logging.getLogger(__name__).debug(
                "EmoNetSingleFaceProcessor: exception during extraction: %s", e
            )
            return []

    # ---------- Internals ----------
    def _face_from_detection(self, det, frame_bgr: np.ndarray, hw: Tuple[int, int]):
        """
        Compute score, bbox, and crop for a MediaPipe detection.

        Returns None when the detection is invalid or produces an empty crop.
        """
        h, w = hw
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        max_distance = float(np.linalg.norm(center)) + 1e-6

        try:
            rel_bb = det.location_data.relative_bounding_box
            conf = float(det.score[0]) if hasattr(det, "score") and det.score else 0.0

            rel_w = float(getattr(rel_bb, "width", 0.0))
            rel_h = float(getattr(rel_bb, "height", 0.0))
            rel_x = float(getattr(rel_bb, "xmin", 0.0))
            rel_y = float(getattr(rel_bb, "ymin", 0.0))

            if rel_w <= 0 or rel_h <= 0:
                return None

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

            bbox = self._relative_bbox_to_abs(rel_bb, hw, self.padding_ratio)
            x, y, bw, bh = bbox
            crop = frame_bgr[y : y + bh, x : x + bw]
            if crop.size == 0:
                return None

            if self.output_size is not None:
                ow, oh = self.output_size[0], self.output_size[1]
                crop = cv2.resize(crop, (ow, oh))

            return float(score), bbox, crop
        except Exception:
            return None

    def _detect_faces_cv2(
        self,
        frame_bgr: np.ndarray,
        hw: Tuple[int, int],
        existing_bboxes: List[Tuple[int, int, int, int]],
        limit: int,
    ) -> List[Tuple[float, Tuple[int, int, int, int], np.ndarray]]:
        """Fallback using OpenCV Haar cascade to harvest additional faces."""
        if self._cascade is None or limit <= 0:
            return []

        h, w = hw
        if h <= 0 or w <= 0:
            return []

        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
        except Exception:
            return []

        try:
            detections = self._cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(24, 24),
            )
        except Exception:
            return []

        if detections is None or len(detections) == 0:
            return []

        results: List[Tuple[float, Tuple[int, int, int, int], np.ndarray]] = []
        existing = list(existing_bboxes) if existing_bboxes else []

        for (x, y, bw, bh) in detections:
            if bw <= 0 or bh <= 0:
                continue
            rel = SimpleNamespace(
                xmin=float(x) / float(w),
                ymin=float(y) / float(h),
                width=float(bw) / float(w),
                height=float(bh) / float(h),
            )
            bbox = self._relative_bbox_to_abs(rel, hw, self.padding_ratio)
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            if any(self._bbox_iou(bbox, eb) > 0.45 for eb in existing):
                continue

            x0, y0, ww, hh = bbox
            crop = frame_bgr[y0 : y0 + hh, x0 : x0 + ww]
            if crop.size == 0:
                continue

            if self.output_size is not None:
                ow, oh = self.output_size[0], self.output_size[1]
                crop = cv2.resize(crop, (ow, oh))

            area = (ww * hh) / float(max(w * h, 1))
            score = 0.01 + math.sqrt(max(area, 0.0))
            results.append((score, bbox, crop))
            existing.append(bbox)
            if len(results) >= limit:
                break

        return results

    @staticmethod
    def _bbox_iou(
        bbox_a: Tuple[int, int, int, int],
        bbox_b: Tuple[int, int, int, int],
    ) -> float:
        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b
        if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
            return 0.0

        ax2 = ax + aw
        ay2 = ay + ah
        bx2 = bx + bw
        by2 = by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

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
