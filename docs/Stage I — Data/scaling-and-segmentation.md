# Scaling and Segmentation

- [✅] Define FE→DEAM scaling formula
- [ ] Verify mapping with plots
- [ ] DEAM 10s segments with 50% overlap
- [ ] Persist segment metadata

## FE→DEAM Scaling
- Note: For this academic POC we use DEAM static annotations `[1, 9]` (dynamic `[-10, 10]` also available).
- Valence: `v_deam = 1 + (8/6) * (v_fe + 3)`
- Arousal: `a_deam = 1 + (8/6) * a_fe`
- FE ranges: Valence `[-3, 3]`, Arousal `[0, 6]` → DEAM static ranges `[1, 9]`.

## DEAM Segmentation
- Window size: `10s`
- Overlap: `50%`
- Sampling rate used: `2 Hz`
- Build KD-Tree over segment `[valence, arousal]` means for retrieval.
- Persist segment metadata: `song_id`, `start_time`, `end_time`, `valence`, `arousal`.
