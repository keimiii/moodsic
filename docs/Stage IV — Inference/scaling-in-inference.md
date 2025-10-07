# Scaling in Inference

- [✅] EmotionScaleAligner available for inference boundary conversions
- [✅] Verified FE ranges: V ∈ [-3, 3], A ∈ [0, 6]; DEAM static in [1, 9] (POC)
- [ ] Migrate remaining inference/diagnostic utilities to use the aligner instead of inline math

`EmotionScaleAligner` centralizes the conversion formulas, but several inference-adjacent helpers (diagnostics, dataset loaders) still replicate the math manually. Those callers need to be migrated before we can claim full adoption.

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

The EmotionScaleAligner provides the authoritative conversions between FindingEmo, DEAM static, and EmoNet scales through a unified reference space `[-1, 1]`. New inference code should call into it; existing helpers that still divide by `3`/subtract `3` are pending cleanup (`src/data/datasets.py`, `src/evaluation/evaluator.py`, `src/utils/metrics.py`, …).

## EmoNet Integration

When using EmoNet as the face expert, outputs stay in reference space by default. Callers can opt into the aligner when they need downstream scales:

```python
# EmoNet outputs [-1, 1] can be used directly or converted via the aligner
v_fe, a_fe = aligner.emonet_to_findingemo(emonet_valence, emonet_arousal)
v_deam, a_deam = aligner.emonet_to_deam_static(emonet_valence, emonet_arousal)
```

`EmoNetAdapter.predict()` (`models/face/emonet_adapter.py`) returns clamped reference-space values; any conversion to FindingEmo or DEAM must still be done explicitly by the caller via the aligner. Optional domain-calibration layers (e.g., `CrossDomainCalibration`) operate in reference space before any conversion.

- The aligner handles range checking, clipping, and precision.
- Strict mode remains available for validation during development.
