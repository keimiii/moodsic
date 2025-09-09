# DEAM Dataset

- [ ] Confirm dynamic annotation schema and sampling rate
- [ ] Define 10s segment length and 50% overlap policy
- [ ] Build KD-Tree index for segment retrieval
- [ ] Document FE→DEAM scaling usage in queries

## Summary
- 1,802 songs. For this academic POC we use the static song-level SAM
  annotations `[1, 9]` for retrieval.

## Dynamic Annotations (optional/future)
- Both valence and arousal in range `[-10, 10]` (per-frame).
- Example input file: `annotations_dynamic.csv`.
- Typical sampling rate: `2 Hz`.

## Static Annotations (POC default)
- Both valence and arousal on a nine-point scale `[1, 9]` (whole 45s excerpt).
- We retrieve at the song level using these static annotations.

## Segmentation Policy (optional/future)
- Segment length: `10 seconds`.
- Overlap: `50%`.
- Metadata per segment: `song_id`, `start_time`, `end_time`, `valence`, `arousal`.

## Indexing
- POC default: keep a simple table of songs with static `[valence, arousal]` and do
  a linear-scan k-NN.
- Optional: Build KD-Tree over `[valence, arousal]` for fast k-NN when scaling up.

## FE→DEAM Scaling (for queries)
- Valence: `v_deam = 1 + (8/6) * (v_fe + 3)`
- Arousal: `a_deam = 1 + (8/6) * a_fe`
