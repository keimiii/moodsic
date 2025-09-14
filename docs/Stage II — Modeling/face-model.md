# Face Path (EmoNet)

- [✅] Implement `SingleFaceProcessor` (MediaPipe) to extract primary face → `utils/emonet_single_face_processor.py`
- [✅] Apply face alignment prior to inference via MediaPipe eye keypoints (inter-ocular rotation; no separate face-alignment dependency) → `models/face/emonet_adapter.py`
- [✅] Integrate EmoNet via an adapter that handles alignment, normalization, inference, TTA uncertainty, and calibration (EmoNet→FindingEmo) → `models/face/emonet_adapter.py` + `models/calibration/cross_domain.py`
- [✅] Evaluate face detection rate and impact → Completed (see findings below)

## Findings on FindingEmo (Aug 25, 2025)

- Coverage: Face detection succeeds on ≈38% of images (7,453/19,606).
- Baseline (faces-found subset, FindingEmo scale):
  - Valence CCC ≈ 0.167, Arousal CCC ≈ 0.016 (mean ≈ 0.092)
  - Pearson/Spearman: V r≈0.33/ρ≈0.31, A r≈0.08/ρ≈0.08
- After calibration (fixed method; trained in reference space, no tanh clamp):
  - Valence CCC ≈ 0.295, Arousal CCC ≈ 0.064 (mean ≈ 0.180)
  - r/ρ essentially unchanged → calibration mainly corrects bias/scale
- Holdout ceiling (best affine on a validation split):
  - Valence CCC ≈ 0.205, Arousal CCC ≈ 0.017

Conclusion: EmoNet is a weak, coverage-limited expert on FindingEmo. Use it in ablations (face-only) and as one input to fusion; the headline metric should come from fusion + gating (fallback to scene when no face).

Note on calibration evaluation: EmoNet weights remain frozen. Any train/validation split (or k-fold CV) refers to the calibration layer only. After confirming out-of-sample gains, refit the calibration on 100% of the faces-found subset for deployment.

## Face Detection and Cropping
Single-face pipeline selects the best face per frame (confidence × area × center proximity), then extracts a padded crop resized to 224×224.

```python
import cv2
import mediapipe as mp
import numpy as np

class SingleFaceProcessor:
    def __init__(self, confidence_threshold=0.5):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(min_detection_confidence=confidence_threshold)

    def extract_primary_face(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        if not results.detections:
            return None
        bbox = self._select_best_face(results.detections, image.shape)
        if bbox is None:
            return None
        return self._extract_face_crop(image, bbox)

    def _select_best_face(self, detections, image_shape):
        h, w = image_shape[:2]
        center = np.array([w/2, h/2])
        best, best_score = None, -1
        for det in detections:
            bb = det.location_data.relative_bounding_box
            conf = det.score[0]
            area = bb.width * bb.height
            face_center = np.array([bb.xmin + bb.width/2, bb.ymin + bb.height/2]) * np.array([w, h])
            dist = np.linalg.norm(face_center - center)
            maxd = np.linalg.norm(center)
            prox = 1 - (dist / maxd)
            score = conf * np.sqrt(area) * (0.7 + 0.3 * prox)
            if score > best_score:
                best, best_score = bb, score
        return best

    def _extract_face_crop(self, image, bb, padding=0.2):
        h, w = image.shape[:2]
        x0 = int(max(0, (bb.xmin - padding * bb.width) * w))
        y0 = int(max(0, (bb.ymin - padding * bb.height) * h))
        x1 = int(min(w, (bb.xmin + bb.width * (1 + padding)) * w))
        y1 = int(min(h, (bb.ymin + bb.height * (1 + padding)) * h))
        crop = image[y0:y1, x0:x1]
        return cv2.resize(crop, (224, 224))
```

## EmoNet Integration (Adapter)
The adapter wraps EmoNet for inference:
- Align face crop → resize → EmoNet normalization
- Forward pass → continuous V/A
- Optional TTA (flip/jitter) → mean/variance
- Apply calibration (EmoNet→FindingEmo) only if it improves holdout V/A MAE; default off. Train/operate the calibrator in reference space [-1, 1]; avoid clamping during optimization; convert back to FindingEmo after calibration.

```python
v, a, (v_var, a_var) = face_expert.predict(face_crop, tta=5)
```

Training a separate face regressor is not required; EmoNet serves as the face expert (frozen). Its role is for ablation and fusion rather than standalone performance.
