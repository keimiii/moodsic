# Stabilization: EMA and Gating

- [ ] Implement EMA smoothing over V and A
- [ ] Add uncertainty gating with MC Dropout variance
- [ ] Set defaults: `alpha=0.7`, `τ=0.4`, `n_mc_samples=5`
- [ ] Track jitter reduction and response time

Extracted from [project_overview.md](file:///Users/desmondchoy/Projects/emo-rec/docs/project_overview.md).

## Mechanism

- EMA smooths frame-to-frame predictions with an α tuned for a 3–5s effective window.
- Uncertainty gating: when per-dimension variance `σ²` exceeds threshold `τ`, hold the last stable EMA output instead of updating.
- Targets (from evaluation plan): jitter reduction of 40–60% without over-dampening, and acceptable response time to genuine shifts.

## Reference implementation

```python
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
            self.ema_valence = valence
            self.ema_arousal = arousal
            self.last_stable_valence = valence
            self.last_stable_arousal = arousal
            return valence, arousal
        self.ema_valence = self.alpha * valence + (1 - self.alpha) * self.ema_valence
        self.ema_arousal = self.alpha * arousal + (1 - self.alpha) * self.ema_arousal
        if variance is not None:
            v_var, a_var = variance
            if v_var > self.uncertainty_threshold:
                out_v = self.last_stable_valence
            else:
                out_v = self.ema_valence
                self.last_stable_valence = out_v
            if a_var > self.uncertainty_threshold:
                out_a = self.last_stable_arousal
            else:
                out_a = self.ema_arousal
                self.last_stable_arousal = out_a
        else:
            out_v = self.ema_valence
            out_a = self.ema_arousal
            self.last_stable_valence = out_v
            self.last_stable_arousal = out_a
        self.history.append((out_v, out_a))
        return out_v, out_a
```
