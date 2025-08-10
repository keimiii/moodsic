# Face Model

- [ ] Implement `SingleFaceProcessor` (MediaPipe) to extract primary face
- [ ] Prepare face-crops dataset from FindingEmo
- [ ] Implement face emotion regressor (light ResNet18 backbone)
- [ ] Evaluate face detection rate and impact

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

## Face Emotion Regressor
Lightweight ResNet18 backbone with separate V/A heads and dropout; supports MC Dropout.

```python
import torch
import torch.nn as nn

class FaceEmotionRegressor(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        for p in list(self.backbone.parameters())[:-10]:
            p.requires_grad = False
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)
        self.valence_head = nn.Sequential(nn.Linear(num_features, 128), nn.ReLU(), self.dropout, nn.Linear(128, 1))
        self.arousal_head = nn.Sequential(nn.Linear(num_features, 128), nn.ReLU(), self.dropout, nn.Linear(128, 1))

    def forward(self, face_images, n_samples=1):
        if n_samples > 1:
            return self._mc_forward(face_images, n_samples)
        feats = self.backbone(face_images)
        feats = self.dropout(feats)
        v = self.valence_head(feats).squeeze()
        a = self.arousal_head(feats).squeeze()
        return v, a

    def _mc_forward(self, images, n_samples):
        preds = []
        for _ in range(n_samples):
            v, a = self.forward(images, n_samples=1)
            preds.append(torch.stack([v, a]))
        preds = torch.stack(preds)
        mean = preds.mean(dim=0)
        var = preds.var(dim=0)
        return mean, var
```

## Dataset Preparation (for training)
- Extract face crops from FindingEmo, preserving V/A labels
- Save metadata to `face_annotations.csv` for dataloaders

```python
class FaceDatasetPreparer:
    # See implementation in overview for details
    ...
```
