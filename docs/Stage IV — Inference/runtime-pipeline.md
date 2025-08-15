# Runtime Pipeline

- [ ] Implement PERCEIVE → STABILIZE → MATCH end-to-end
- [ ] Enable MC Dropout for uncertainty in PERCEIVE
- [ ] Integrate scene–face fusion with variance-weighted averaging
- [ ] Wire stabilized outputs into k-NN music retrieval with dwell-time

Extracted from [project_overview.md](file:///Users/desmondchoy/Projects/emo-rec/docs/project_overview.md).

## Overview

Three-stage runtime pipeline that converts video frames into music segment recommendations.

```
[RUNTIME INFERENCE PIPELINE]

[Input Video]
     |
     v
+------------------------------------------+
| PERCEIVE: Extract V-A per frame         |
| Phase 0: Scene model predictions        |
| Phase 1: + Face detection, alignment & EmoNet (via adapter) |
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

## Stage details

- PERCEIVE
  - Scene model: CLIP/ViT backbone, regression heads with dropout; MC Dropout for mean/variance.
  - Face path: single-face detection (MediaPipe), face alignment, and EmoNet inference via an adapter that handles preprocessing, calibration (EmoNet→FindingEmo), and optional TTA-based uncertainty.
  - Fusion: variance-weighted averaging when both paths available; fall back to scene-only when no face.

- STABILIZE
  - Exponential Moving Average (EMA) over valence/arousal.
  - Uncertainty gating: if variance exceeds threshold, hold last stable values.

- MATCH
  - Query k-NN over DEAM 10s segments (50% overlap) indexed with KD-Tree.
  - Enforce minimum dwell time and recent-song avoidance.
  - Use explicit FE→DEAM scaling for queries.
