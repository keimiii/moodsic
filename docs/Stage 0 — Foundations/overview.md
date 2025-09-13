# Foundations Overview

## Executive Summary

This system maps facial emotions from video to personalized music. It uses a three-stage runtime pipeline — PERCEIVE → STABILIZE → MATCH — with transfer learning to produce robust recommendations aligned to detected emotional states.

Training uses FindingEmo (25k images with valence–arousal) for emotion recognition and DEAM (1,802 songs with dynamic valence–arousal) for retrieval. The phased plan targets two risks: context-only learning (ignoring faces) and unstable, over-switching recommendations. Pre-trained models are leveraged with targeted enhancements to ground predictions in facial features and keep music stable while responsive.

## The Problem We're Solving

By analyzing facial expressions over time (not just single moments), the system reflects users emotional journeys and recommends music with matching valence–arousal signatures. It can validate the current state or gently guide mood (e.g., calming music for high arousal/negative valence; energizing music for low mood).

## TLDR

- Why these datasets: FindingEmo (images, valence–arousal) and DEAM (songs, same space) share a measurement system, enabling direct mapping from video emotions to music.
- Two vision models: Scene model captures overall context but can be misled; Face model focuses on expressions. Training separately lets each specialize.
- Fusion: Combine scene and face via variance-weighted averaging, trusting more confident predictions.
- Runtime pipeline:
  - PERCEIVE: Per-frame valence–arousal from both models with MC Dropout uncertainty.
  - STABILIZE: EMA smoothing; hold values if uncertainty exceeds a threshold.
  - MATCH (POC default): Song-level matching over DEAM static [1, 9] via simple k-NN
    (linear scan), with GMM “station” gating from the DEAM clustering notebook;
    enforce 20–30s minimum dwell and recent-song memory.

## Course Requirements Coverage

- Supervised Learning: Fine-tune pre-trained vision models on FindingEmo to regress valence and arousal.
- Deep Learning: Transformer-based backbones (CLIP/ViT) for robust feature extraction.
- Hybrid/Ensemble: Fuse scene-based and face-based predictions to reduce context overfitting.

## System Architecture

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
    Scene Model    Face Model
    (CLIP/ViT)     (Single-face)
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

DEAM Dataset (1802 songs; static V-A used for POC)
                |
                v
    +------------------------+
    | Music Preprocessing    |
    | - Song-level V-A (static [1, 9]) |
    +------------------------+
                |
                v
    [Indexed Songs (simple table)]

============================
[RUNTIME INFERENCE PIPELINE]
============================

[Input Video]
     |
     v
+------------------------------------------+
| PERCEIVE: Extract V-A per frame         |
| Phase 0: Scene model predictions        |
| Phase 1: + Face detection & prediction  |
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
| MATCH: Song-level retrieval (POC)      |
| - Query per stabilized frame           |
| - GMM station gating (predict_proba)   |
|   - If top posterior < 0.55 → top-2    |
| - k-NN over DEAM songs (linear scan)   |
| - Scale alignment (FE→DEAM)            |
| - Minimum dwell time (20-30s)          |
+------------------------------------------+
     |
     v
[Recommended Songs]
```

## Phased Implementation Snapshot

- Phase 0 (Core baseline): Scene-only emotion recognition with temporal handling and stability (EMA + uncertainty gating).
- Phase 1 (Face-aware): Single-face detection/processing to ground predictions in expressions.
- Phase 2 (Fusion): Linear/variance-weighted fusion of scene and face predictions.

## Risk Mitigation (Extracted)

- Context overfitting: Add single-face pathway; fuse with scene to balance cues.
- Prediction instability: MC Dropout uncertainty + adaptive gating to smooth while staying responsive.
- Scale misalignment: Explicit FEDEAM mapping to avoid retrieval errors.

### Contingencies

- If face detection fails: degrade to scene-only with increased smoothing.
- If training time is tight: fixed fusion weights (e.g., 0.6 scene / 0.4 face).
- If MC Dropout is costly: replace with simpler confidence-based gating.
