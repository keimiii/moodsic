# Scaling and Segmentation

- [✅] Implemented unified EmotionScaleAligner for all scale conversions
- [ ] Verify mapping with plots
- [ ] DEAM 10s segments with 50% overlap
- [ ] Persist segment metadata

## Unified Scale Alignment

All scale conversions are now handled by the EmotionScaleAligner class:

```python
from utils.emotion_scale_aligner import EmotionScaleAligner

# Initialize aligner
aligner = EmotionScaleAligner()

# FindingEmo → DEAM static conversion
v_deam, a_deam = aligner.findingemo_to_deam_static(v_fe, a_fe)
```

- Note: For this academic POC we use DEAM static annotations `[1, 9]` (dynamic `[-10, 10]` also available).
- FE ranges: Valence `[-3, 3]`, Arousal `[0, 6]` → DEAM static ranges `[1, 9]`.
- All conversions maintain consistency and handle edge cases automatically.

## DEAM Segmentation
- Window size: `10s`
- Overlap: `50%`
- Sampling rate used: `2 Hz`
- Build KD-Tree over segment `[valence, arousal]` means for retrieval.
- Persist segment metadata: `song_id`, `start_time`, `end_time`, `valence`, `arousal`.
