# Uncertainty and Gating

- ✅ Implement MC Dropout for scene model; use TTA-based uncertainty for EmoNet face expert
- ✅ Set and tune `n_mc_samples`, `alpha`, and uncertainty threshold `τ`
- ✅ Apply EMA smoothing with uncertainty gating
- ✅ Expose stability metrics (variance, jitter)

## MC Dropout
Perform multiple stochastic forward passes to estimate predictive variance for valence and arousal.

- Default `n_mc_samples = 5`
- Variance used for fusion weighting and stabilization gating
 - Stabilizer is optional and disabled by default in code
   (`enable_stabilizer=False`).

## Stabilization: EMA with Uncertainty Gating
Maintain smooth yet responsive outputs using an exponential moving average (EMA) and hold updates when uncertainty is high.

Defaults for MVP:
- `alpha = 0.7`
- `uncertainty_threshold (τ) = 0.4`

```python
from collections import deque
import numpy as np

class AdaptiveStabilizer:
    def __init__(self, alpha=0.7, uncertainty_threshold=0.4, window_size=60):
        self.alpha = alpha
        self.uncertainty_threshold = uncertainty_threshold
        self.window_size = window_size
        self.ema_valence = None
        self.ema_arousal = None
        self.last_stable_valence = 0.0
        self.last_stable_arousal = 0.0
        self.history = deque(maxlen=window_size)

    def update(self, valence, arousal, variance=None):
        if self.ema_valence is None:
            self.ema_valence, self.ema_arousal = valence, arousal
            self.last_stable_valence, self.last_stable_arousal = valence, arousal
            return valence, arousal
        self.ema_valence = self.alpha * valence + (1 - self.alpha) * self.ema_valence
        self.ema_arousal = self.alpha * arousal + (1 - self.alpha) * self.ema_arousal
        if variance is not None:
            v_var, a_var = variance
            out_v = self.last_stable_valence if v_var > self.uncertainty_threshold else self.ema_valence
            out_a = self.last_stable_arousal if a_var > self.uncertainty_threshold else self.ema_arousal
            if v_var <= self.uncertainty_threshold:
                self.last_stable_valence = out_v
            if a_var <= self.uncertainty_threshold:
                self.last_stable_arousal = out_a
        else:
            out_v, out_a = self.ema_valence, self.ema_arousal
            self.last_stable_valence, self.last_stable_arousal = out_v, out_a
        self.history.append((out_v, out_a))
        return out_v, out_a

    def get_stability_metrics(self):
        if len(self.history) < 2:
            return {"variance": 0, "jitter": 0}
        arr = np.array(self.history)
        return {"variance": np.var(arr, axis=0), "jitter": np.mean(np.abs(np.diff(arr, axis=0)), axis=0)}
```

## Integration Points
- Use variance from MC Dropout in fusion weighting and in gating
- Apply stabilizer to per-frame fused predictions before matching

Notes:
- In fixed-weight fusion fallback (when uncertainties are missing/invalid), the
  fused variance may be absent; gating uses provided variances when available.
- Additional guardrails in fusion can gate the face path for stability (low
  face score, high sigma, or low frame brightness). These are optional and
  disabled by default.

## Face Path Guardrails (Recommended Ranges and Rationale)

These optional thresholds help suppress jitter from unreliable face readings
due to occlusion, low light, or weak detections. Tune on a validation slice.

- Face score threshold: ~0.4–0.6
  - Many detectors output scores in [0, 1] with usable faces clustered above
    ~0.5. Starting near the midpoint avoids over-gating while filtering weak
    detections.

- Face sigma threshold (`face_max_sigma`): ~0.6–0.8
  - Sigma is the square root of variance on the same [-1, 1] scale as
    valence/arousal. Well-behaved frames often show σ ≈ 0.2–0.5 from TTA/MC.
    Values above ~0.6–0.8 indicate high uncertainty; gating reduces large swings.

- Brightness threshold: ~40–60 (on 8-bit luma 0–255)
  - Average frame brightness uses standard luma (approx. Y ≈ 0.114B + 0.587G +
    0.299R). Below ~50 is visibly dim; face quality degrades, so gating the face
    path reduces noise and jitter.

Implementation details: these guardrails are implemented in `SceneFaceFusion`
and are disabled by default. See the fusion module’s constructor for how to
enable and pass custom thresholds.
