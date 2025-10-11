# Repo Conventions (extracted)

- Phase naming:
  - Phase 0: Scene model baseline (CLIP/ViT), EMA + uncertainty gating.
  - Phase 1: Single-face detection and prediction.
  - Phase 2: Fusion (linear / variance-weighted) of scene and face outputs.

- Runtime stage naming:
  - PERCEIVE → STABILIZE → MATCH

- Retrieval parameters (POC defaults):
  - Song-level retrieval using DEAM static `[1, 9]`
  - Simple k-NN via linear scan; `k = 20` shortlist within selected cluster (if used)
  - Optional: GMM “station” gating (K≈5) to bias selection
  - Minimum dwell time: 20–30 seconds; maintain recent-song memory to avoid repeats

- Scale alignment:
  - `EmotionScaleAligner` is the canonical helper for converting between FindingEmo (`v∈[-3,3], a∈[0,6]`), reference space `[-1,1]`, and DEAM static `[1,9]`.
  - Retrieval/matching pipelines operate on reference-space values; conversions to/from dataset scales should use the aligner at pipeline boundaries.
  - Some legacy utilities still perform manual `/3` and `+/-3` normalization; track migration to the aligner when touching those paths.
