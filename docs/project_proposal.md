# 1. Project Background

## 1.1 Problem Statement

Conventional emotion recognition systems predominantly rely on facial expression analysis of a single subject, neglecting multimodal cues such as environmental context and scene dynamics. This limitation becomes critical in applications where emotion is shaped not only by an individual’s expression but also by the surrounding context, such as group interactions or scene atmosphere.

In parallel, music recommendation systems have traditionally relied on collaborative filtering (CF) and content-based models (CBM). While effective for personalized music streaming, these approaches are optimized for individual listening preferences derived from historical user data. They are less suited to contexts where music must be selected to match or modulate a shared emotional environment, such as enhancing narrative coherence in videos. In these cases, contextual fit and affective alignment are more important than individual user taste.

## 1.2 Proposed Solution

We propose **Moodsic**, an AI-powered tool that combines pattern recognition and computer vision techniques for mood recognition and music matching. The system integrates human emotions (facial expressions/signals) and scene-level features to generate a richer emotional profile, including both emotional states and intensity levels. This profile is then mapped to music tracks characterized by affective features such as valence and arousal.

As our training and inference datasets are largely drawn from movies, the model is currently best suited for video media applications, such as automatic soundtrack recommendation for short-form and AI-generated content. Nonetheless, the framework also has future potential in physical spaces (cafés, retail, luxury stores), where adaptive background music selection can be optimized to enhance customer experience and use ambience to influence desired behavioural & business outcomes.

---

# 2. Datasets

We use three complementary datasets that share a common **valence-arousal (V-A)** emotional representation system:

* **Training Dataset**: *FindingEmo* has ≈25k images of scenes depicting people in various naturalistic social settings labeled with V-A values for training of vision models. In current experiments, 19,606 images were used after dataset filtering; see Modeling docs for faces-found coverage.
* **Inference Dataset**: *VEATIC (Video-based Emotion and Affect Tracking in Context)* contains 124 videos labeled with frame-by-frame V-A values for inference of the vision models.
* **Music Retrieval Database**: *DEAM* has 1,802 songs annotated with V-A values, genres, artists, and song titles. This enables direct mapping between detected emotions in video and music segments.

---

# 3. System Architecture

Moodsic employs a **dual-pathway mixture-of-experts (MoE) architecture** that combines scene-level and facial emotion recognition to generate robust emotion estimates for music matching. The system processes video input through two specialized models, fuses their outputs using uncertainty-aware weighting, and applies temporal stabilization before mapping emotions to music recommendations.

## 3.1 Emotion Recognition

Our approach addresses the fundamental attribution problem in single-source emotion detection, where systems cannot distinguish between environmental emotional cues and genuine human emotional states. We deploy two complementary experts:

* **Scene Expert**: A CLIP/DINOv3 backbone with a lightweight MLP regressor analyzes full-frame contextual information to estimate scene-level emotional content.
* **Face Expert**: *EmoNet*, a pre-trained model, processes facial expressions detected and aligned using MediaPipe, focusing on individual emotions from facial cues.

Both experts output valence and arousal values in a unified `[-1, 1]` coordinate space. **Uncertainty estimation** is implemented through *Monte Carlo Dropout* for the scene model and *test-time augmentation* for the face model.

### Expert Fusion and Temporal Stabilization

* **Inverse-Variance Weighting**: The system dynamically weights expert contributions based on prediction confidence. When facial detection is clear and confident, facial signals dominate. When faces are occluded or absent, scene context takes precedence. When both experts exhibit high confidence, their outputs are blended proportionally.
* **Temporal Smoothing**: Frame-to-frame emotion estimates are stabilized using *exponential moving average (EMA)* smoothing with uncertainty gating. During periods of high uncertainty, the system maintains the last stable estimate rather than propagating unreliable predictions. This prevents abrupt music transitions caused by momentary detection errors.

## 3.2 Music Retrieval and Matching

The temporally-stabilized emotion output undergoes a two-stage retrieval process:

1. **Emotion-to-Mood Translation**: Output is mapped onto *DEAM* dataset clusters that represent certain moods (e.g., *“Downcast & Low-energy”*, *“Bright & Engaging”*, *“Calm & Neutral”*).
2. **Music Selection**: Within the identified mood cluster, *k-nearest neighbors* (`k=1`) retrieval selects the music with the most similar valence-arousal profile to the detected emotional state.

This approach ensures that music recommendations maintain both emotional coherence with the visual content and temporal stability throughout video playback.

---

# 4. Evaluation

- Where:
  - FindingEmo (scene‑path ablations in reference space [-1, 1])
  - VEATIC (inference‑time video with per‑frame V/A labels)

- How:
  - Scene ablations: freeze backbones and train regression heads under a uniform protocol; select the best model by lowest Average MAE (mean of V‑MAE and A‑MAE) on the test split; cross reference Spearman’s ρ. Scale conversions use `utils/emotion_scale_aligner.py`.
  - VEATIC: For each video, run the inference pipeline to produce a time series of valence/arousal. Align predictions to the dataset’s annotated frames, compare them point‑by‑point, and compute V/A MAE, Average MAE, and Spearman’s ρ. Report results per video and overall. Optionally show raw vs smoothed scores and face‑detection coverage for context.

- Why these metrics:
  - MAE (per‑dimension) is interpretable in dataset units and robust to outliers.
  - Average MAE provides a single scalar that balances valence and arousal.
  - Spearman’s ρ measures monotonic agreement, important when calibration/scale differences exist and for assessing trend tracking in videos.

---

# 5. Course Requirements Coverage

* **Supervised Learning**: Demonstrated by training lightweight regression heads atop multiple pretrained CV backbones to predict continuous valence–arousal (V/A) on the *FindingEmo* dataset. The ablation study is confined to the scene pathway, comparing several backbones under a uniform training protocol for these heads. 
* **Deep Learning**: The system is anchored in transformer-based visual encoders (e.g., CLIP and Vision Transformers) that serve as high-capacity feature extractors for emotion recognition.
* **Hybrid/Ensemble Models**: Uses a *Mixture-of-Experts system* to adaptively activate a “scene” and “face” expert model, potentially improving overall accuracy and mitigating context overfitting.
---
