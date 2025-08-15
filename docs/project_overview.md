# Emotion-Aware Music Recommendation from Facial Video — Technical Architecture

**Revision:** Aug 15, 2025  
**Team:** 4 members (≈40 man-days total)  
**Course:** NUS MTech in AI Systems

---

## 1. Executive Summary

Modern organizations face a critical challenge in creating environments that respond dynamically to the emotional interplay between people and spaces. Static background music fails to address the complex relationship between environmental design and human emotional states, missing opportunities to enhance experiences and drive business outcomes.

Our emotion-aware music system employs dual-pathway detection to understand both the designed intent of spaces and actual human emotional responses within them. By discriminating between environmental ambiance and genuine human emotions, the system delivers contextually appropriate music that measurably improves user experiences. The technology recognizes that identical emotional expressions require different musical interventions depending on environmental context, enabling intelligent atmospheric orchestration across diverse industry applications.

At runtime, the system still follows the perceive → stabilize → match pipeline, with uncertainty-aware fusion and temporal smoothing to ensure enterprise-grade reliability and smooth transitions.

---

### The Problem We're Solving

Static background music does not account for the dynamic interplay between environmental design and human emotion. Identical facial expressions can require different musical interventions depending on the surrounding context (e.g., clinical waiting room vs. luxury retail boutique), leading to mismatched ambiance and suboptimal outcomes.

Single-source emotion detection also suffers from a fundamental attribution problem: systems conflate environmental cues with genuine human emotional state. Our approach explicitly separates and fuses scene context with facial signals using inverse-variance weighting, and stabilizes outputs with uncertainty-aware temporal smoothing. This yields explainable, context-appropriate recommendations that improve user experience and business metrics.

- Inverse-variance fusion (intuitive): We trust the signal that’s more confident. Clear faces → face signal leads. Occlusions or no faces → scene signal leads. When both are confident, we blend them so neither dominates.
- Uncertainty-aware smoothing (intuitive): We smooth frame-to-frame changes. If the system is unsure, we briefly hold the last stable value instead of reacting. When confidence returns, updates resume—preventing jitter and sudden music flips.
- Example: Dramatic store lighting suggests “tense,” but customers’ faces show calm curiosity. A scene-only system picks intense tracks; ours selects gentle, sophisticated instrumentals that fit the context and keep shoppers comfortable.

---
### TLDR

**Why These Specific Datasets**

We use two carefully selected datasets that share a common emotional measurement system:
- **FindingEmo**: 25,000 images labeled with valence (happy vs. sad) and arousal (excited vs. calm) values
- **DEAM**: 1,802 songs with the same valence-arousal annotations

This shared measurement system enables direct emotion mapping between video and music. When video shows specific valence-arousal values, we can query music with matching emotional signatures.

**Why We Need Two Vision Models**

**The Scene Model** examines entire frames - lighting, colors, environment - providing context about overall mood. However, it can be misled by environmental factors.

**The Face Expert (EmoNet)** focuses exclusively on facial expressions, providing direct emotional signals. This grounds predictions in actual human expressions rather than just environmental cues.

Training these separately allows each model to specialize, learning distinct patterns relevant to their focus area.

**The Need for Fusion**

Neither model alone is sufficient. Scene-only can miss actual human emotion; face-only loses context and fails when faces aren't visible. **Fusion** intelligently combines both predictions using variance-weighted averaging - trusting confident predictions more than uncertain ones.

**The Runtime Pipeline: PERCEIVE → STABILIZE → MATCH**

**PERCEIVE**: Analyzes each video frame using both models, extracting valence-arousal values with uncertainty estimates through Monte Carlo Dropout (multiple predictions with slight variations).

**STABILIZE**: Applies exponential moving average (EMA) smoothing to prevent jarring jumps. When uncertainty exceeds threshold, holds previous stable values rather than updating.

**MATCH**: Uses k-nearest neighbor (k-NN) search over pre-processed DEAM music segments:
- DEAM songs divided into 10-second segments with 50% overlap
- Each segment labeled with average valence-arousal values
- KD-Tree structure enables fast similarity search
- Retrieves k=20 nearest segments, selecting best match while avoiding recent repeats
- Enforces 20-30 second minimum play time before switching

## 2. Course Requirements Coverage

The system fulfills three of four course requirements through its technical implementation:

**Supervised Learning** is demonstrated through fine-tuning pre-trained vision models on the FindingEmo dataset, training regression heads to predict continuous valence and arousal values from both scene context and facial features.

**Deep Learning** forms the core of the system through transformer-based architectures including CLIP and vision transformers, which provide robust feature extraction for emotion recognition tasks.

**Hybrid/Ensemble Methods** are implemented through the fusion of scene-based and face-based emotion predictions, combining complementary perspectives to improve accuracy and reduce context overfitting.

---

Update: EmoNet integration for face pathway (Aug 15, 2025)

- Decision: Replace the Phase 1 face model with the pretrained EmoNet face-affect estimator as the face expert. No face-model training is needed for the MVP.
- Preprocessing: Use single-face detection (MediaPipe) to select the primary face, then apply face alignment (face-alignment library) to match EmoNet's expected input. Resize/normalize exactly as per EmoNet demo.
- Uncertainty: Approximate with test-time augmentation (TTA; e.g., flip and small scale/crop jitter) and use the prediction variance with the existing uncertainty gating.
- Calibration: Fit a per-dimension affine mapping on a small FindingEmo validation split to transform EmoNet valence/arousal to FindingEmo space; then apply the existing FE→DEAM mapping in MATCH.
- Fusion: Keep the current inverse-variance fusion between scene and face experts. EmoNet serves as the face expert; optional student model can be added later as a distilled replacement or fallback.
- Licensing: EmoNet is CC BY-NC-ND 4.0. We will not fine-tune or modify the weights. For distribution, prefer loading checkpoints from upstream or a download script with attribution; if vendoring unmodified weights, preserve the original license and attribution and keep the project non-commercial.

## 3. System Architecture

Face path: EmoNet (pretrained) serves as the face expert; scene path remains CLIP/ViT fine-tuned on FindingEmo. Fusion uses inverse-variance weighting.

### Complete Processing Pipeline with Phased Implementation

```
[OFFLINE TRAINING PIPELINE]
============================

FindingEmo Dataset (25k images with V-A labels)
                |
                v
    +------------------------+
    | Data Processing        |
    | - Download & validate  |
    | - Train/val/test split |
    | - Face detection cache |
    | - Augmentation setup    |
    +------------------------+
                |
         ---------------
         |             |
         v             v
    [Phase 0]      [Phase 1]
    Scene Model    Face Expert: EmoNet
    (CLIP/ViT)     (pretrained)
         |             |
         ---------------
                |
                v
    +------------------------+
    | [Phase 2] Fusion      |
    | - Linear combination  |
    | - Variance weighting  |
    +------------------------+
                |
                v
    [Trained Emotion Models]

DEAM Dataset (1802 songs with dynamic V-A)
                |
                v
    +------------------------+
    | Music Preprocessing    |
    | - 10s segments         |
    | - 50% overlap          |
    | - KD-Tree indexing     |
    +------------------------+
                |
                v
    [Indexed Music Segments]

============================
[RUNTIME INFERENCE PIPELINE]
============================

[Input Video]
     |
     v
+------------------------------------------+
| PERCEIVE: Extract V-A per frame         |
| Phase 0: Scene model predictions        |
| Phase 1: + Face detection, alignment & EmoNet inference  |
| Phase 2: + Fusion of both paths         |
| + MC Dropout uncertainty estimation     |
+------------------------------------------+
     |
     v
+------------------------------------------+
| STABILIZE: Temporal smoothing           |
| - EMA (α-tuned, 3-5s window)           |
| - Uncertainty gating (hold if σ > τ)    |
| - Per-frame processing                  |
+------------------------------------------+
     |
     v
+------------------------------------------+
| MATCH: Segment-level retrieval          |
| - Query per stabilized frame            |
| - k-NN over 10s DEAM segments          |
| - Scale alignment (FE→DEAM)            |
| - Minimum dwell time (20-30s)          |
+------------------------------------------+
     |
     v
[Recommended Music Segments]
```

---

## 4. Phased Implementation Strategy

### Phase 0: Core Baseline (Days 1-3)

The initial phase establishes a functional system addressing the most critical requirements while maintaining simplicity. This phase focuses on scene-based emotion recognition with proper temporal handling and stability mechanisms.

#### Scene-based Emotion Regressor

The scene model leverages pre-trained CLIP or Vision Transformer architectures with minimal modifications. The implementation maintains the transfer learning approach while adding uncertainty estimation capabilities essential for stable recommendations.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

class SceneEmotionRegressor(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", dropout_rate=0.3):
        super().__init__()
        self.backbone = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.feature_dim = self.backbone.config.projection_dim
        
        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Regression heads with dropout for uncertainty
        self.dropout = nn.Dropout(dropout_rate)
        self.valence_head = self._create_head()
        self.arousal_head = self._create_head()
        
    def _create_head(self):
        return nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 1)
        )
    
    def preprocess_batch(self, pil_images, device):
        # CLIP-normalized pixel_values expected by get_image_features
        batch = self.processor(images=pil_images, return_tensors="pt")
        return batch["pixel_values"].to(device)
    
    def forward(self, pixel_values, n_samples=1):
        if n_samples > 1:
            # MC Dropout for uncertainty estimation
            return self._mc_forward(pixel_values, n_samples)
        
        features = self.backbone.get_image_features(pixel_values)
        features = self.dropout(features)
        valence = self.valence_head(features)
        arousal = self.arousal_head(features)
        
        return valence.squeeze(), arousal.squeeze()
    
    def _mc_forward(self, pixel_values, n_samples):
        # Enable dropout during multiple stochastic forward passes
        was_training = self.training
        self.train(True)
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                v, a = self.forward(pixel_values, n_samples=1)
                predictions.append(torch.stack([v, a]))
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        self.train(was_training)
        return mean, variance
```

#### Segment-level DEAM Processing

The music database requires temporal segmentation to enable dynamic matching with changing emotions. Each song is divided into overlapping segments with aggregated emotion values for efficient retrieval.

```python
import time
import pandas as pd
import numpy as np
from collections import deque
from sklearn.neighbors import KDTree

# FE ranges (confirm in config): V in [-3, 3], A in [0, 6]
# DEAM dynamic ranges: both in [-10, 10]

def fe_to_deam(v_fe: float, a_fe: float):
    v_deam = (10.0 / 3.0) * v_fe
    a_deam = -10.0 + (20.0 / 6.0) * a_fe
    return v_deam, a_deam

class DEAMSegmentProcessor:
    def __init__(self, deam_path, window_size=10, overlap=0.5):
        self.deam_path = deam_path
        self.window_size = window_size  # seconds
        self.overlap = overlap
        self.sampling_rate = 2  # Hz
        
    def process_dataset(self):
        annotations = pd.read_csv(f"{self.deam_path}/annotations_dynamic.csv")
        segments = []
        
        for song_id, song_data in annotations.groupby('song_id'):
            segments.extend(self._extract_segments(song_id, song_data))
        
        segments_df = pd.DataFrame(segments)
        
        # Build KD-Tree for fast k-NN in DEAM space
        self.kd_tree = KDTree(
            segments_df[['valence', 'arousal']].values
        )
        self.segments_metadata = segments_df
        
        return segments_df
    
    def _extract_segments(self, song_id, song_data):
        segments = []
        samples_per_window = self.window_size * self.sampling_rate
        step_size = int(samples_per_window * (1 - self.overlap))
        
        for start_idx in range(0, len(song_data) - samples_per_window, step_size):
            end_idx = start_idx + samples_per_window
            window_data = song_data.iloc[start_idx:end_idx]
            
            segment = {
                'song_id': song_id,
                'start_time': start_idx / self.sampling_rate,
                'end_time': end_idx / self.sampling_rate,
                # DEAM dynamic means already in [-10, 10]
                'valence': window_data['valence'].mean(),
                'arousal': window_data['arousal'].mean(),
            }
            segments.append(segment)
        
        return segments

class SegmentMatcher:
    """
    Segment-level retriever with FE→DEAM scaling, minimum dwell-time, and
    simple recent-song avoidance to prevent thrashing.
    """
    def __init__(self, segments_df: pd.DataFrame, min_dwell_time: float = 25.0, recent_k: int = 5):
        self.segments = segments_df.reset_index(drop=True)
        self.kd_tree = KDTree(self.segments[["valence", "arousal"]].values)
        self.min_dwell = min_dwell_time
        self.recent_songs = deque(maxlen=recent_k)
        self.current = None
        self.current_start = None
    
    def _can_switch(self, now: float) -> bool:
        if self.current is None or self.current_start is None:
            return True
        return (now - self.current_start) >= self.min_dwell
    
    def _choose_candidate(self, indices, distances):
        # Prefer a candidate not in recent_songs
        for idx, dist in zip(indices[0], distances[0]):
            row = self.segments.iloc[idx]
            if row["song_id"] not in self.recent_songs:
                return row
        # Fallback: allow repeats if all blocked
        return self.segments.iloc[indices[0][0]]
    
    def recommend(self, v_fe: float, a_fe: float, now: float = None, k: int = 20):
        now = time.time() if now is None else now
        # Respect dwell-time
        if not self._can_switch(now):
            return self.current
        
        # Scale FE→DEAM for query
        vq, aq = fe_to_deam(v_fe, a_fe)
        distances, indices = self.kd_tree.query(np.array([[vq, aq]]), k=k)
        best = self._choose_candidate(indices, distances)
        
        # If new song, start dwell period and track recent songs
        if self.current is None or best["song_id"] != self.current["song_id"]:
            self.current = best
            self.current_start = now
            self.recent_songs.append(best["song_id"])
        return self.current
```

#### EMA with Uncertainty Gating

The stabilization module combines exponential moving average smoothing with uncertainty-based gating to prevent erratic recommendations while maintaining responsiveness to genuine emotional changes. Defaults for MVP: `alpha = 0.7`, `uncertainty_threshold (τ) = 0.4`, and `n_mc_samples = 5` in prediction. Tune only if jitter or sluggishness is observed.

```python
from collections import deque
import numpy as np

class AdaptiveStabilizer:
    def __init__(self, alpha=0.7, uncertainty_threshold=0.4, window_size=60):
        self.alpha = alpha  # EMA weight
        self.uncertainty_threshold = uncertainty_threshold
        self.window_size = window_size
        
        # State tracking
        self.ema_valence = None
        self.ema_arousal = None
        self.last_stable_valence = 0.0
        self.last_stable_arousal = 0.0
        self.history = deque(maxlen=window_size)
        
    def update(self, valence, arousal, variance=None):
        # Initialize on first frame
        if self.ema_valence is None:
            self.ema_valence = valence
            self.ema_arousal = arousal
            self.last_stable_valence = valence
            self.last_stable_arousal = arousal
            return valence, arousal
        
        # Apply EMA
        self.ema_valence = self.alpha * valence + (1 - self.alpha) * self.ema_valence
        self.ema_arousal = self.alpha * arousal + (1 - self.alpha) * self.ema_arousal
        
        # Uncertainty gating: hold last stable if variance exceeds threshold
        if variance is not None:
            v_var, a_var = variance
            
            if v_var > self.uncertainty_threshold:
                output_valence = self.last_stable_valence
            else:
                output_valence = self.ema_valence
                self.last_stable_valence = output_valence
            
            if a_var > self.uncertainty_threshold:
                output_arousal = self.last_stable_arousal
            else:
                output_arousal = self.ema_arousal
                self.last_stable_arousal = output_arousal
        else:
            output_valence = self.ema_valence
            output_arousal = self.ema_arousal
            self.last_stable_valence = output_valence
            self.last_stable_arousal = output_arousal
        
        self.history.append((output_valence, output_arousal))
        return output_valence, output_arousal
    
    def get_stability_metrics(self):
        if len(self.history) < 2:
            return {'variance': 0, 'jitter': 0}
        
        history_array = np.array(self.history)
        variance = np.var(history_array, axis=0)
        jitter = np.mean(np.abs(np.diff(history_array, axis=0)), axis=0)
        
        return {
            'variance': variance,
            'jitter': jitter
        }
```

### Phase 1: Face-Aware Enhancement with EmoNet (Days 4-6)

The second phase replaces an in-house face regressor with the pretrained EmoNet model as the face expert. We detect the primary face per frame (MediaPipe), apply face alignment to match EmoNet’s expected input, and run EmoNet to obtain continuous valence/arousal. We estimate uncertainty via lightweight test-time augmentation (e.g., flip/jitter) and feed both means and variances into the existing fusion and stabilization modules.

#### Single-Face Detection and Processing

The face processing pipeline identifies and extracts the most prominent face per frame, providing a direct signal for emotion recognition that complements scene-based predictions.

```python
import cv2
import mediapipe as mp
import numpy as np

class SingleFaceProcessor:
    def __init__(self, confidence_threshold=0.5):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            min_detection_confidence=confidence_threshold
        )
        
    def extract_primary_face(self, image):
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_image)
        
        if not results.detections:
            return None
        
        # Select face with highest score × area × center proximity
        best_face = self._select_best_face(results.detections, image.shape)
        
        if best_face is None:
            return None
        
        # Extract face crop with padding
        face_crop = self._extract_face_crop(image, best_face)
        return face_crop
    
    def _select_best_face(self, detections, image_shape):
        h, w = image_shape[:2]
        image_center = np.array([w/2, h/2])
        
        best_score = -1
        best_face = None
        
        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            
            # Calculate metrics
            confidence = detection.score[0]
            area = bbox.width * bbox.height
            
            # Center proximity (0 to 1, higher is closer to center)
            face_center = np.array([
                bbox.xmin + bbox.width/2,
                bbox.ymin + bbox.height/2
            ]) * np.array([w, h])
            
            distance_to_center = np.linalg.norm(face_center - image_center)
            max_distance = np.linalg.norm(image_center)
            center_proximity = 1 - (distance_to_center / max_distance)
            
            # Combined score
            score = confidence * np.sqrt(area) * (0.7 + 0.3 * center_proximity)
            
            if score > best_score:
                best_score = score
                best_face = bbox
        
        return best_face
    
    def _extract_face_crop(self, image, bbox, padding=0.2):
        h, w = image.shape[:2]
        
        # Convert relative to absolute coordinates
        x_min = int(max(0, (bbox.xmin - padding * bbox.width) * w))
        y_min = int(max(0, (bbox.ymin - padding * bbox.height) * h))
        x_max = int(min(w, (bbox.xmin + bbox.width * (1 + padding)) * w))
        y_max = int(min(h, (bbox.ymin + bbox.height * (1 + padding)) * h))
        
        face_crop = image[y_min:y_max, x_min:x_max]
        
        # Resize to standard size
        face_crop = cv2.resize(face_crop, (224, 224))
        
        return face_crop
```

#### Face-Specific Emotion Model

The face emotion model uses a lightweight architecture optimized for facial expression analysis, trained on face crops extracted from FindingEmo images.

```python
class FaceEmotionRegressor(nn.Module):
    def __init__(self, backbone="microsoft/resnet-18", dropout_rate=0.3):
        super().__init__()
        # Use lighter backbone for face-specific features
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Emotion-specific heads
        self.valence_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 1)
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 1)
        )
    
    def forward(self, face_images, n_samples=1):
        if n_samples > 1:
            return self._mc_forward(face_images, n_samples)
        
        features = self.backbone(face_images)
        features = self.dropout(features)
        
        valence = self.valence_head(features).squeeze()
        arousal = self.arousal_head(features).squeeze()
        
        return valence, arousal
    
    def _mc_forward(self, images, n_samples):
        predictions = []
        for _ in range(n_samples):
            v, a = self.forward(images, n_samples=1)
            predictions.append(torch.stack([v, a]))
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        return mean, variance
```

### Phase 2: Scene-Face Fusion (Day 7)

The fusion phase combines scene and face predictions through variance-weighted averaging, leveraging the complementary strengths of both approaches.

```python
class SceneFaceFusion:
    def __init__(self, scene_model, face_model, face_processor):
        self.scene_model = scene_model
        self.face_model = face_model
        self.face_processor = face_processor
        
        # Fusion weights (can be learned on validation set)
        self.scene_weight = 0.6
        self.face_weight = 0.4
        
    def predict(self, frame, use_variance_weighting=True, n_mc_samples=5):
        # Scene prediction with uncertainty
        scene_tensor = self._preprocess_frame(frame)
        scene_mean, scene_var = self.scene_model(scene_tensor, n_samples=n_mc_samples)
        
        # Face prediction (if face detected)
        face_crop = self.face_processor.extract_primary_face(frame)
        
        if face_crop is not None:
            face_tensor = self._preprocess_frame(face_crop)
            face_mean, face_var = self.face_model(face_tensor, n_samples=n_mc_samples)
            
            # Variance-weighted fusion
            if use_variance_weighting:
                final_pred, final_var = self._variance_weighted_fusion(
                    scene_mean, scene_var,
                    face_mean, face_var
                )
            else:
                # Simple weighted average
                final_pred = (self.scene_weight * scene_mean + 
                             self.face_weight * face_mean)
                final_var = (self.scene_weight * scene_var + 
                            self.face_weight * face_var)
        else:
            # No face detected, use scene only
            final_pred = scene_mean
            final_var = scene_var
        
        valence = final_pred[0].item()
        arousal = final_pred[1].item()
        variance = (final_var[0].item(), final_var[1].item())
        
        return valence, arousal, variance
    
    def _variance_weighted_fusion(self, pred1, var1, pred2, var2):
        # Weight by inverse variance
        weight1 = 1 / (var1 + 1e-6)
        weight2 = 1 / (var2 + 1e-6)
        
        # Normalize weights
        total_weight = weight1 + weight2
        weight1 = weight1 / total_weight
        weight2 = weight2 / total_weight
        
        # Fused prediction
        fused_pred = weight1 * pred1 + weight2 * pred2
        
        # Fused variance (approximation)
        fused_var = 1 / total_weight
        
        return fused_pred, fused_var
    
    def _preprocess_frame(self, frame):
        # Convert to tensor and normalize
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = cv2.resize(frame, (224, 224))
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame_tensor = (frame_tensor - mean) / std
        
        return frame_tensor.unsqueeze(0)
```

---

## 5. Music Matching with Proper Scaling

The matching stage performs frame-level retrieval over DEAM segments with explicit scale alignment between FindingEmo and DEAM emotion spaces.

```python
class SegmentLevelMusicMatcher:
    def __init__(self, deam_processor):
        self.processor = deam_processor
        self.segments = deam_processor.segments_metadata
        self.kd_tree = deam_processor.kd_tree
        
        # Track recently played to ensure variety
        self.recent_songs = deque(maxlen=10)
        self.current_segment = None
        self.segment_start_time = None
        self.min_dwell_time = 20  # seconds
        
    def get_music_for_frame(self, valence, arousal, current_time):
        # Scale from FindingEmo to DEAM space
        v_deam = self._scale_fe_to_deam(valence, 'valence')
        a_deam = self._scale_fe_to_deam(arousal, 'arousal')
        
        # Check if we should switch segments
        if self._should_switch_segment(current_time):
            # Find new segment
            new_segment = self._find_best_segment(v_deam, a_deam)
            
            if new_segment is not None:
                self.current_segment = new_segment
                self.segment_start_time = current_time
                self.recent_songs.append(new_segment['song_id'])
        
        return self.current_segment
    
    def _scale_fe_to_deam(self, value, dimension):
        # Explicit mapping from FindingEmo to DEAM scales
        if dimension == 'valence':
            # FindingEmo: [-3, 3] → DEAM: [-10, 10]
            return (10/3) * value
        else:  # arousal
            # FindingEmo: [0, 6] → DEAM: [-10, 10]
            return -10 + (20/6) * value
    
    def _should_switch_segment(self, current_time):
        # Switch only after minimum dwell time
        if self.current_segment is None:
            return True
        
        time_in_segment = current_time - self.segment_start_time
        return time_in_segment >= self.min_dwell_time
    
    def _find_best_segment(self, valence, arousal, k=20):
        # Query k nearest segments
        query = np.array([[valence, arousal]])
        distances, indices = self.kd_tree.query(query, k=k)
        
        # Filter out recently played songs
        candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            segment = self.segments.iloc[idx]
            if segment['song_id'] not in self.recent_songs:
                candidates.append((segment, dist))
        
        if not candidates:
            # All songs recently played, allow repetition but pick furthest
            candidates = [(self.segments.iloc[idx], dist) 
                         for idx, dist in zip(indices[0], distances[0])]
        
        # Return best candidate
        best_segment, _ = min(candidates, key=lambda x: x[1])
        return best_segment.to_dict()
```

---

## 6. Training and Optimization

### 6.1 Scene Model Training (Face path uses EmoNet; no face training)

We use EmoNet as the face expert with no face dataset or training required. This section is intentionally simplified to focus on scene model training; the face path is plug-and-play via EmoNet.



### 6.2 Scene Model Fine-tuning Protocol (face path uses EmoNet; no face fine-tuning)

The training covers only the scene model. The face path uses EmoNet as a fixed pretrained expert and is not trained or fine-tuned.

```python
from fastai.vision.all import *
from fastai.callback.all import *

class PhaseTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def train_phase_0(self):
        # Scene model training
        scene_dls = self._prepare_scene_dataloaders()
        scene_model = SceneEmotionRegressor()
        
        learn = Learner(
            scene_dls,
            scene_model,
            loss_func=self._combined_loss,
            metrics=[self._ccc_metric, mae],
            cbs=[EarlyStoppingCallback(patience=5)]
        )
        
        # Find optimal LR
        lr = learn.lr_find().valley
        
        # Train with frozen backbone (10 epochs)
        learn.fit_one_cycle(10, lr_max=lr)
        
        # Fine-tune with unfrozen layers (5 epochs)
        scene_model.backbone.requires_grad_(True)
        learn.fit_one_cycle(5, lr_max=slice(lr/100, lr/10))
        
        return learn
    
    # Face training is not used: EmoNet is the face expert and remains fixed (no training).
    
    def optimize_fusion_weights(self, scene_model, face_model, val_data):
        # Grid search for optimal fusion weights
        best_weights = None
        best_score = float('inf')
        
        for scene_w in np.arange(0.3, 0.8, 0.1):
            face_w = 1 - scene_w
            
            fusion = SceneFaceFusion(scene_model, face_model, SingleFaceProcessor())
            fusion.scene_weight = scene_w
            fusion.face_weight = face_w
            
            # Evaluate on validation set
            val_loss = self._evaluate_fusion(fusion, val_data)
            
            if val_loss < best_score:
                best_score = val_loss
                best_weights = (scene_w, face_w)
        
        return best_weights
    
    def _combined_loss(self, pred, target):
        # 70% CCC + 30% MSE
        ccc_loss = 2 - self._ccc(pred[:, 0], target[:, 0]) - self._ccc(pred[:, 1], target[:, 1])
        mse_loss = F.mse_loss(pred, target)
        return 0.7 * ccc_loss + 0.3 * mse_loss
    
    def _ccc(self, pred, true):
        pred_mean = pred.mean()
        true_mean = true.mean()
        covariance = ((pred - pred_mean) * (true - true_mean)).mean()
        pred_var = pred.var()
        true_var = true.var()
        
        ccc = (2 * covariance) / (pred_var + true_var + (pred_mean - true_mean)**2 + 1e-8)
        return ccc
```

---



### 7.1 Performance Metrics

The evaluation framework employs carefully selected metrics for each pipeline stage, with particular attention to the risks of context overfitting and prediction instability.

#### Perceive Stage Metrics

**Concordance Correlation Coefficient (CCC)** serves as the primary metric, measuring both correlation and absolute agreement between predictions and ground truth. The CCC is particularly critical for evaluating whether face-based predictions improve upon scene-only baselines, indicating successful mitigation of context overfitting.

**Face Detection Rate** measures the percentage of frames where a face is successfully detected and processed, providing insight into the robustness of the face pathway across diverse video content.

**Scene-Face Divergence** quantifies the difference between scene and face predictions, where high divergence may indicate context overfitting in the scene model. This metric is calculated as the mean Euclidean distance between scene and face predictions when both are available.

#### Stabilize Stage Metrics

**Jitter Reduction Rate** compares frame-to-frame variation before and after stabilization, directly measuring the effectiveness of the EMA and uncertainty gating mechanisms. A target reduction of 40-60% indicates successful smoothing without over-dampening.

**Gating Activation Frequency** tracks how often uncertainty gating holds predictions stable, providing insight into model confidence and the effectiveness of MC Dropout uncertainty estimation.

**Response Time to Emotional Shifts** measures the delay between significant emotional changes in video content and stabilized output updates, ensuring the system maintains responsiveness despite smoothing.

#### Match Stage Metrics

**Segment-Level Emotional Distance** calculates the mean distance between query emotions and retrieved segments at each time point, providing a more granular assessment than whole-video averaging.

**Switching Frequency** measures how often the system changes music segments, where excessive switching indicates poor stability while insufficient switching suggests unresponsiveness.

**Dwell Time Distribution** analyzes how long each segment plays before switching, ensuring the minimum dwell time prevents jarring transitions while allowing appropriate variety.

### 7.2 Ablation Studies

#### Study 1: Scene-Only vs EmoNet-Enhanced Performance

This study quantifies the improvement from incorporating EmoNet by comparing Phase 0 (scene-only) against Phase 1 (scene + EmoNet face expert). The evaluation measures CCC improvement, reduction in context-dependent errors, and performance on multi-person scenes where facial grounding is critical.

#### Study 2: Impact of Uncertainty Gating

This experiment evaluates the stabilization improvements from uncertainty-based gating by comparing simple EMA against EMA with MC Dropout uncertainty gating. Metrics include jitter reduction, false positive gating events, and user experience ratings on recommendation stability.

#### Study 3: Segment-Level vs Whole-Video Retrieval

This study validates the importance of temporal granularity by comparing segment-level k-NN retrieval against whole-video averaging. The evaluation measures emotional alignment accuracy, music variety, and user preference ratings for dynamic versus static matching.

---

## 8. Implementation Timeline

### Week 1: Foundation (Phase 0)
- **Days 1-2:** Dataset preparation, DEAM segmentation with 10s windows
- **Day 3:** Scene model training with CLIP/ViT backbone
- **Day 3:** Implement EMA with uncertainty gating

### Week 2: Face Enhancement (Phase 1)
- **Days 4-5:** Face detection pipeline and dataset preparation
- **Day 6:** Train single-face emotion model
- **Day 7:** Implement scene-face fusion (Phase 2)

### Week 3: Integration and Optimization
- **Days 8-9:** End-to-end pipeline integration
- **Day 10:** Hyperparameter tuning (α, τ, fusion weights)
- **Days 11-12:** Performance optimization and testing

### Week 4: Evaluation and Delivery
- **Days 13-14:** Ablation studies and metrics collection
- **Day 15:** User study (N=10)
- **Days 16-17:** Documentation and demo preparation
- **Days 18-20:** Buffer for refinements

---

## 9. Risk Mitigation

### Addressed Risks

**Context Overfitting** is mitigated through the single-face pathway that grounds predictions in facial features, with scene-face fusion providing balanced emotion detection that leverages both facial expressions and contextual cues.

**Prediction Instability** is addressed through MC Dropout uncertainty estimation combined with adaptive gating, ensuring smooth music recommendations while maintaining responsiveness to genuine emotional changes.

**Scale Misalignment** is resolved through explicit FindingEmo to DEAM scale mapping functions, preventing retrieval errors from incompatible emotion spaces.

### Contingency Plans

If face detection fails frequently, the system gracefully degrades to scene-only predictions with increased smoothing to compensate for potential context artifacts. If training time exceeds estimates, Phase 2 fusion can use fixed weights (0.6 scene, 0.4 face) instead of optimization. If MC Dropout proves computationally expensive, a simpler threshold-based gating using prediction confidence can substitute.

---

## 10. Demonstration Application

```python
import gradio as gr
import cv2
import time

class EmotionMusicDemo:
    def __init__(self):
        # Initialize all components
        self.scene_model = self._load_model('checkpoints/scene_model.pth')
        self.face_model = self._load_model('checkpoints/face_model.pth')
        self.face_processor = SingleFaceProcessor()
        self.fusion = SceneFaceFusion(self.scene_model, self.face_model, self.face_processor)
        self.stabilizer = AdaptiveStabilizer(alpha=0.7, uncertainty_threshold=0.5)
        self.matcher = SegmentLevelMusicMatcher(DEAMSegmentProcessor('./deam_data'))
        
    def process_video(self, video_path, show_face_crops=True):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        results = {
            'emotion_timeline': [],
            'music_segments': [],
            'face_crops': [] if show_face_crops else None,
            'metrics': {
                'face_detection_rate': 0,
                'gating_activations': 0,
                'scene_face_divergence': []
            }
        }
        
        frame_count = 0
        faces_detected = 0
        current_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # Process frame (every 3rd frame for efficiency)
            if frame_count % 3 == 0:
                # Get emotions with uncertainty
                valence, arousal, variance = self.fusion.predict(frame, n_mc_samples=10)
                
                # Track if face was detected
                face_crop = self.face_processor.extract_primary_face(frame)
                if face_crop is not None:
                    faces_detected += 1
                    if show_face_crops and frame_count % 30 == 0:  # Sample face crops
                        results['face_crops'].append({
                            'time': current_time,
                            'image': face_crop
                        })
                
                # Stabilize
                v_stable, a_stable = self.stabilizer.update(valence, arousal, variance)
                
                # Get music recommendation
                segment = self.matcher.get_music_for_frame(v_stable, a_stable, current_time)
                
                # Record results
                results['emotion_timeline'].append({
                    'time': current_time,
                    'valence_raw': valence,
                    'arousal_raw': arousal,
                    'valence_stable': v_stable,
                    'arousal_stable': a_stable,
                    'variance': variance
                })
                
                if segment and (not results['music_segments'] or 
                              results['music_segments'][-1]['song_id'] != segment['song_id']):
                    results['music_segments'].append({
                        'start_time': current_time,
                        'song_id': segment['song_id'],
                        'segment_info': segment
                    })
            
            frame_count += 1
        
        cap.release()
        
        # Calculate metrics
        results['metrics']['face_detection_rate'] = faces_detected / (frame_count // 3)
        results['metrics']['stability'] = self.stabilizer.get_stability_metrics()
        
        return results
    
    def create_visualization(self, results):
        # Create emotion trajectory plot
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        times = [r['time'] for r in results['emotion_timeline']]
        valence_raw = [r['valence_raw'] for r in results['emotion_timeline']]
        valence_stable = [r['valence_stable'] for r in results['emotion_timeline']]
        arousal_raw = [r['arousal_raw'] for r in results['emotion_timeline']]
        arousal_stable = [r['arousal_stable'] for r in results['emotion_timeline']]
        
        # Valence plot
        axes[0].plot(times, valence_raw, 'b-', alpha=0.3, label='Raw')
        axes[0].plot(times, valence_stable, 'b-', label='Stabilized')
        axes[0].set_ylabel('Valence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Arousal plot
        axes[1].plot(times, arousal_raw, 'r-', alpha=0.3, label='Raw')
        axes[1].plot(times, arousal_stable, 'r-', label='Stabilized')
        axes[1].set_ylabel('Arousal')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Music segments
        for segment in results['music_segments']:
            axes[2].axvline(x=segment['start_time'], color='g', linestyle='--', alpha=0.5)
            axes[2].text(segment['start_time'], 0.5, segment['song_id'][:10], 
                        rotation=45, fontsize=8)
        
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Music Changes')
        axes[2].set_ylim([0, 1])
        
        plt.tight_layout()
        return fig
    
    def launch(self):
        def process_and_visualize(video):
            results = self.process_video(video, show_face_crops=True)
            fig = self.create_visualization(results)
            
            metrics_text = f"""
            Face Detection Rate: {results['metrics']['face_detection_rate']:.2%}
            Emotion Stability: V={results['metrics']['stability']['variance'][0]:.3f}, 
                             A={results['metrics']['stability']['variance'][1]:.3f}
            Music Segments: {len(results['music_segments'])}
            """
            
            return fig, metrics_text, results
        
        interface = gr.Interface(
            fn=process_and_visualize,
            inputs=gr.Video(label="Upload Video"),
            outputs=[
                gr.Plot(label="Emotion Timeline & Music Segments"),
                gr.Textbox(label="Performance Metrics"),
                gr.JSON(label="Detailed Results")
            ],
            title="Emotion-Aware Music Recommendation System",
            description="""
            Upload a video to receive music recommendations based on detected emotions.
            The system uses both scene context and facial expressions for robust emotion detection,
            with temporal smoothing to ensure stable recommendations.
            """
        )
        
        interface.launch(share=True)

if __name__ == "__main__":
    demo = EmotionMusicDemo()
    demo.launch()
```

---

## 11. Technical Dependencies

```python
# requirements.txt
transformers==4.35.0     # Pre-trained models
fastai==2.7.13          # Training framework
torch==2.0.1            # Deep learning backend
torchvision==0.15.2     # Vision utilities
opencv-python==4.8.0    # Video processing
mediapipe==0.10.0       # Face detection
scikit-learn==1.3.0     # KD-Tree and metrics
numpy==1.24.3           # Numerical operations
pandas==2.0.3           # Data manipulation
gradio==3.50.0          # Demo interface
matplotlib==3.7.1       # Visualization
tqdm==4.65.0           # Progress tracking
```

### 11.1 EmoNet Integration and Licensing

- Face expert: EmoNet (pretrained), integrated via an adapter that aligns inputs (face-alignment), matches EmoNet normalization, and outputs valence/arousal for fusion and stabilization.
- Calibration: Fit per-dimension affine transforms on a small FindingEmo validation split to map EmoNet outputs into FindingEmo’s V/A space; then apply FE→DEAM mapping in MATCH as defined earlier. Store calibration parameters with the model artifacts.
- Uncertainty: Use test-time augmentation (e.g., horizontal flip and minor crop/scale jitter) to obtain a prediction variance for inverse-variance fusion and uncertainty gating.
- Distribution and license:
  - EmoNet is released under CC BY-NC-ND 4.0. We will not modify or fine-tune the weights for distribution.
  - Recommended practice: do not vendor weights; provide an automated download step (or load from upstream) with proper attribution and a clear non-commercial notice. If vendoring unmodified checkpoints becomes necessary for offline use, preserve the original license file and attribution and keep the repository non-commercial.

### 11.2 Files to vendor from EmoNet (unmodified)

Copy the following from https://github.com/face-analysis/emonet into `models/emonet/` in this repo. Keep directory names and filenames unchanged; do not modify the contents. Preserve the license and attribution.

- LICENSE.txt → models/emonet/LICENSE.txt
- README.md → models/emonet/README.md
- emonet/ → models/emonet/emonet/ (entire package directory)
- pretrained/ → models/emonet/pretrained/ (all files; includes 5-class and 8-class checkpoints)

Optional (for reference/testing, not required by our pipeline):
- demo.py, demo_video.py

Notes:
- We use EmoNet as an unmodified face expert under CC BY-NC-ND 4.0; no fine-tuning or weight modification will be distributed.
- If we decide not to vendor checkpoints, replace `pretrained/` with a small download script that fetches from upstream at setup time and stores under `models/emonet/pretrained/`.

---

## Conclusion

This architecture provides a robust emotion-aware music recommendation system that addresses the critical risks of context overfitting and prediction instability through a carefully phased implementation. The system leverages transfer learning for efficiency while incorporating targeted enhancements that ensure predictions are grounded in facial expressions and temporally stable.

The phased approach enables incremental development with clear validation at each stage, allowing the team to achieve a functional baseline quickly while progressively adding sophistication. The explicit handling of scale alignment, uncertainty estimation, and segment-level retrieval ensures the system delivers meaningful music recommendations that respond appropriately to emotional changes in video content.