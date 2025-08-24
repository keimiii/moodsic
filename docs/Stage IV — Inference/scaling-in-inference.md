# Scaling in Inference

- [✅] Implement unified scale alignment with EmotionScaleAligner
- [✅] Verified FE ranges: V ∈ [-3, 3], A ∈ [0, 6]; DEAM static in [1, 9] (POC)
- [✅] Unified scaling used across all inference paths (matching, diagnostics)

All scale conversions now use the unified EmotionScaleAligner class for consistency and maintainability.

## Unified Scale Alignment

All emotion scale conversions are handled by the EmotionScaleAligner class:

```python
from utils.emotion_scale_aligner import EmotionScaleAligner

# Initialize aligner
aligner = EmotionScaleAligner()

# FindingEmo → DEAM static conversion
v_deam, a_deam = aligner.findingemo_to_deam_static(v_fe, a_fe)

# EmoNet → FindingEmo conversion (for face expert)
v_fe, a_fe = aligner.emonet_to_findingemo(v_emonet, a_emonet)

# Direct EmoNet → DEAM conversion
v_deam, a_deam = aligner.emonet_to_deam_static(v_emonet, a_emonet)
```

The EmotionScaleAligner provides all necessary conversions between FindingEmo, DEAM static, and EmoNet scales through a unified reference space [-1, 1].

## EmoNet Integration

When using EmoNet as the face expert, the aligner handles the conversion automatically:

```python
# EmoNet outputs [-1, 1] can be used directly or converted to target scales
v_fe, a_fe = aligner.emonet_to_findingemo(emonet_valence, emonet_arousal)
v_deam, a_deam = aligner.emonet_to_deam_static(emonet_valence, emonet_arousal)
```

- No manual calibration needed - the aligner handles scale alignment
- All conversions maintain numerical precision and handle edge cases
- Strict mode available for validation during development
