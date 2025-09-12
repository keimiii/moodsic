# DEAM Dataset

- [ ] Document FE→DEAM scaling usage in queries

## Summary
- 1,802 songs. For this academic POC we use the static song-level SAM
  annotations `[1, 9]` for retrieval.

## Dynamic Annotations (reference)
- Both valence and arousal in range `[-10, 10]` (per-frame). Not used in POC.
- Example input file: `annotations_dynamic.csv`.
- Typical sampling rate: `2 Hz`.

## Static Annotations (POC default)
- Both valence and arousal on a nine-point scale `[1, 9]` (whole 45s excerpt).
- We retrieve at the song level using these static annotations.

## Song-Level Policy (POC)
- Retrieval operates at the song level using static V/A.
- Metadata per song: `song_id`, `valence`, `arousal`, and optional `gmm_cluster`.

## Indexing and Selection
- Keep a simple table of songs with static `[valence, arousal]` and do a
  linear-scan k-NN.
- GMM station gating: assign the stabilized query V/A to the most likely
  cluster via `predict_proba`; if the top posterior is low (e.g., < 0.55),
  widen to the top-2 clusters and search within those.
-
  Rank within the selected cluster set by Euclidean distance between the
  stabilized V/A and each song’s V/A; pick top-1 (or top-N for variety).

## FE→DEAM Scaling (for queries)
- Valence: `v_deam = 1 + (8/6) * (v_fe + 3)`
- Arousal: `a_deam = 1 + (8/6) * a_fe`
