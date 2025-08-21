# Scaling in Inference

- [ ] Implement FE→DEAM mapping for both V and A
- [ ] Verify FE ranges: V ∈ [-3, 3], A ∈ [0, 6]; DEAM static in [1, 9] (POC)
- [ ] Use the same scaling in all inference paths (matching, diagnostics)

Extracted from [project_overview.md](file:///Users/desmondchoy/Projects/emo-rec/docs/project_overview.md).

## Mapping (FindingEmo → DEAM)

Explicit mapping from FindingEmo to DEAM emotion space used for queries:

```python
# Valence: [-3, 3] → [1, 9]
v_deam = 1.0 + (8.0 / 6.0) * (v_fe + 3.0)
# Arousal: [0, 6] → [1, 9]
a_deam = 1.0 + (8.0 / 6.0) * a_fe
```

These formulas are used in both `SegmentMatcher.recommend` and `SegmentLevelMusicMatcher.get_music_for_frame`.

## Calibration (EmoNet → FindingEmo)

When using EmoNet as the face expert, apply an affine calibration per dimension to map EmoNet outputs into FindingEmo ranges before the FE→DEAM mapping above:

```
v_fe ≈ a_v * v_emonet + b_v
a_fe ≈ a_a * a_emonet + b_a
```

- Learn `(a_v, b_v, a_a, b_a)` on a small FindingEmo validation split.
- Clamp to FE ranges after calibration: `v∈[-3, 3]`, `a∈[0, 6]`.
- Store parameters in `models/emonet/calibration.json` and load in the EmoNet adapter.
