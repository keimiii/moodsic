# Scaling and Segmentation

- [✅] Define FE→DEAM scaling formula
- [ ] Verify mapping with plots
- [ ] DEAM 10s segments with 50% overlap
- [ ] Persist segment metadata

## FE→DEAM Scaling
- Valence: `v_deam = (10/3) * v_fe`
- Arousal: `a_deam = -10 + (20/6) * a_fe`
- FE ranges: Valence `[-3, 3]`, Arousal `[0, 6]` → DEAM ranges `[-10, 10]`.

## DEAM Segmentation
- Window size: `10s`
- Overlap: `50%`
- Sampling rate used: `2 Hz`
- Build KD-Tree over segment `[valence, arousal]` means for retrieval.
- Persist segment metadata: `song_id`, `start_time`, `end_time`, `valence`, `arousal`.
