# DEAM Dataset

- [ ] Confirm dynamic annotation schema and sampling rate
- [ ] Define 10s segment length and 50% overlap policy
- [ ] Build KD-Tree index for segment retrieval
- [ ] Document FE→DEAM scaling usage in queries

## Summary
- 1,802 songs with dynamic valence–arousal annotations.
- Used for segment-level retrieval during matching.

## Dynamic Annotations
- Both valence and arousal in range `[-10, 10]`.
- Example input file referenced: `annotations_dynamic.csv`.
- Sampling rate used in processing: `2 Hz`.

## Static Annotations
- Both valence and arousal on a nine-point scale `[1, 9]` (whole 45s excerpt).
- For this academic POC, we use static annotations `[1, 9]` for retrieval.

## Segmentation Policy
- Segment length: `10 seconds`.
- Overlap: `50%`.
- Metadata stored per segment: `song_id`, `start_time`, `end_time`, `valence`, `arousal`.

## Indexing
- Build KD-Tree over `[valence, arousal]` for fast k-NN.

## FE→DEAM Scaling (for queries)
- Valence: `v_deam = 1 + (8/6) * (v_fe + 3)`
- Arousal: `a_deam = 1 + (8/6) * a_fe`
