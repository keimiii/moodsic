# Stabilization: EMA and Gating

Status (cross-referenced with implementation):

- ✅ EMA smoothing over V and A implemented inside fusion stabilizer
  - `_AdaptiveStabilizer` with EMA and last-stable holding is integrated in `SceneFaceFusion`.
  - Code: `models/fusion.py` (class `_AdaptiveStabilizer`; used in `SceneFaceFusion.__init__`).
- ✅ Uncertainty gating using variance implemented
  - Post-fusion: stabilizer holds last stable values when fused variance exceeds `τ` per dimension.
  - Pre-fusion guardrails: optional gating of the face path by `face_score_threshold`, `face_max_sigma` (σ = √variance), and `brightness_threshold` (frame luma).
  - Code: `models/fusion.py`.
- ✅ Defaults set in code
  - `alpha=0.7`, `τ=0.4`, `scene_mc_samples=5`, `face_tta=5`.
  - Code: `models/fusion.py` (constructor defaults); scene MC Dropout in `models/scene/clip_vit_scene_adapter.py`; face TTA in `models/face/emonet_adapter.py`.
- ⏳ Metrics: jitter exposed; response time tracking pending
  - Jitter and variance over a short window are available when stabilizer is enabled.
  - No explicit response-time metric is recorded yet.

Extracted from [project_overview.md](../project_overview.md).

## Mechanism

- EMA smooths frame-to-frame predictions with coefficient α. EMA itself has no fixed window; smoothing and latency depend only on α (recent frames have exponentially higher weight).
- Metrics window: `window_size` controls only the history length used to compute stability metrics (variance/jitter). Changing `window_size` does not change EMA smoothing behavior or latency.
- Uncertainty gating: when per-dimension variance `σ²` exceeds threshold `τ`, hold the last stable EMA output instead of updating.
- Additional pre-fusion guardrails can ignore the face path when it looks unreliable (low detector score, high σ, or low frame brightness).
- Targets: reduce jitter by ~40–60% without over-dampening and keep acceptable response to genuine shifts.

See also: `docs/Stage II — Modeling/uncertainty-and-gating.md` for deeper rationale and ranges.

## EMA Walkthrough

The stabilizer keeps one EMA value per fused dimension. With default `α = 0.7` and `τ = 0.4`, the loop below shows how a single dimension (e.g., valence) evolves frame by frame. Numbers are illustrative but reflect the implementation:

| Frame | Raw fused score | Variance σ² | Decision | Stabilized output |
| --- | --- | --- | --- | --- |
| 0 | 0.50 | 0.12 | seed EMA | 0.50 |
| 1 | 0.60 | 0.10 | update | 0.57 = 0.7·0.60 + 0.3·0.50 |
| 2 | 0.75 | 0.25 | update | 0.70 = 0.7·0.75 + 0.3·0.57 |
| 3 | 0.20 | 0.55 | **hold** (σ² > τ) | 0.70 (unchanged) |
| 4 | 0.65 | 0.15 | update | 0.66 = 0.700·0.65 + 0.210·0.75 + 0.063·0.60 + 0.027·0.50 |
| 5 | 0.80 | 0.30 | update | 0.76 = 0.700·0.80 + 0.210·0.65 + 0.063·0.75 + 0.019·0.60 + 0.008·0.50 |

Key points illustrated above:

- A high-variance frame (Frame 3) is dropped; the EMA sticks to the last stable value until the signal looks trustworthy again.
- When an update is allowed, the EMA uses only the held value and the current raw score; past frames already live inside the held value.
- The effective weights decay exponentially: the latest frame keeps weight `α`, the previous one keeps `α(1-α)`, and so on.

```
weight(t-0) = 0.70
weight(t-1) = 0.70 × 0.30 ≈ 0.21
weight(t-2) = 0.70 × 0.30² ≈ 0.06
```

That decay explains why the stabilizer reacts quickly to sustained shifts while still damping isolated spikes. Frames skipped by gating (Frame 3 above) feed no new weight; the EMA simply reuses the last stable mixture until a trustworthy update arrives.

## Usage

Enable stabilization and (optionally) guardrails via `SceneFaceFusion`:

```python
from models.fusion import SceneFaceFusion

fusion = SceneFaceFusion(
    scene_predictor=...,            # optional
    face_expert=...,               # optional
    face_processor=...,            # optional
    # Sampling defaults (forwarded to adapters)
    scene_mc_samples=5,
    face_tta=5,
    # Post-fusion stabilizer
    enable_stabilizer=True,
    stabilizer_alpha=0.7,
    uncertainty_threshold=0.4,
    # Optional pre-fusion guardrails for the face path
    face_score_threshold=0.5,
    face_max_sigma=0.6,
    brightness_threshold=50.0,
)

res = fusion.perceive_and_fuse(frame)
# When enabled, stability metrics are attached to the result
print(res.stability_variance, res.stability_jitter)
```

Notes:
- Variance comes from stochastic inference:
  - Scene: MC Dropout with multiple passes → `models/scene/clip_vit_scene_adapter.py`.
  - Face: TTA-based uncertainty in `models/face/emonet_adapter.py`.
- If variances are missing, fusion falls back to fixed-weight blending and stabilizer gating uses whatever variances are available.

## Tests and Coverage

- Fusion math, guardrails, and stabilizer behavior: `tests/test_fusion.py`.
- Overlay helper that renders σ and per-path values: `tests/test_fusion_overlay.py` and `utils/fusion_overlay.py`.

## Open Items

- Response time metric: add a simple estimate (e.g., time-to-90% of a step change) to stabilizer metrics or pipeline logs.
