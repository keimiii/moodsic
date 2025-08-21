# Repo Conventions (extracted)

- Phase naming:
  - Phase 0: Scene model baseline (CLIP/ViT), EMA + uncertainty gating.
  - Phase 1: Single-face detection and prediction.
  - Phase 2: Fusion (linear / variance-weighted) of scene and face outputs.

- Runtime stage naming:
  - PERCEIVE → STABILIZE → MATCH

- Retrieval parameters (defaults from overview):
  - DEAM segmentation: 10s windows, 50% overlap
  - k-NN over KD-Tree with `k = 20`
  - Minimum dwell time: 20–30 seconds; maintain recent-song memory to avoid repeats

- Scale alignment:
  - POC uses DEAM static annotations `[1, 9]` (dynamic `[-10, 10]` also available).
  - FE→DEAM mappings used during retrieval (static [1, 9]):
    - `v_deam = 1 + (8/6) * (v_fe + 3)`
    - `a_deam = 1 + (8/6) * a_fe`
