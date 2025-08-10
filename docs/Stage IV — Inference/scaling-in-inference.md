# Scaling in Inference

- [ ] Implement FE→DEAM mapping for both V and A
- [ ] Verify FE ranges: V ∈ [-3, 3], A ∈ [0, 6]; DEAM in [-10, 10]
- [ ] Use the same scaling in all inference paths (matching, diagnostics)

Extracted from [project_overview.md](file:///Users/desmondchoy/Projects/emo-rec/docs/project_overview.md).

## Mapping

Explicit mapping from FindingEmo to DEAM emotion space used for queries:

```python
# Valence: [-3, 3] → [-10, 10]
v_deam = (10.0 / 3.0) * v_fe
# Arousal: [0, 6] → [-10, 10]
a_deam = -10.0 + (20.0 / 6.0) * a_fe
```

These formulas are used in both `SegmentMatcher.recommend` and `SegmentLevelMusicMatcher.get_music_for_frame`.
