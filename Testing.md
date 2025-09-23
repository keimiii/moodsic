# Testing Guide

This project includes a small pytest suite for the fusion math, overlay utilities, and a PERCEIVE-stage end-to-end smoke test using real adapters.

## Quick Start

1) Activate the virtual environment:

2) Install pytest (and OpenCV if missing):

```bash
uv pip install pytest opencv-python-headless
```

3) Run all tests from the repository root:

```bash
pytest -q
```

## Selective Runs

- Fusion tests only:

```bash
pytest -q tests/test_fusion.py
```

- Overlay tests only:

```bash
pytest -q tests/test_fusion_overlay.py
```

- PERCEIVE (real adapters, real images):

```bash
pytest -q tests/test_perceive_e2e_flow.py
```

- Single test function:

```bash
pytest -q tests/test_fusion.py::test_inverse_variance_fusion_correctness
```

- Keyword filter:

```bash
pytest -q -k "variance or overlay"
```

## Test Files

- `tests/test_fusion.py` — unit and integration tests for `SceneFaceFusion`.
- `tests/test_fusion_overlay.py` — overlay drawing checks.
- `tests/test_perceive_e2e_flow.py` — PERCEIVE smoke test using real models
  (SceneCLIPAdapter + EmoNetAdapter) and real images. Skips gracefully when
  prerequisites (transformers cache, mediapipe, image files) are unavailable.

## Coverage Overview (What tests aim to prove)

- Fusion math is correct:
  - Inverse-variance weighting produces the expected fused mean and variance.
  - Outputs are clamped to the reference range [-1, 1].
  - Fixed-weight fallback is used when variances are missing/invalid.
  - Degenerate fixed-weight sum is handled by using equal weights.
- Edge-case routing behaves as designed:
  - Passthrough when only scene or face is valid.
  - Neutral output (0,0) with unit variance when neither path is valid.
- Public API integration works with simple mocks:
  - Scene-only and face-only configurations return the expected fused result.
  - With both paths, per-dimension fusion respects inverse-variance logic.
  - Optional real-image test exercises the no-face-detected fallback.
- Overlay utility draws without mutating input:
  - Preserves image shape and visibly changes pixels on the output.
  - Renders both with and without bbox/variance information.

## Dataset Note (Optional)

One integration test will use a real image if available to verify fallback behavior when no face is detected:

```
data/Run_2/Affectionate retiree competition/Elderly-group-in-museum-1200x800.jpg
```

If the file is missing, the test auto-skips. Place the dataset under `data/` to enable it.

## Troubleshooting

- Import errors: ensure you run `pytest` from the repository root so relative imports resolve.
- OpenCV missing: `uv pip install opencv-python-headless`.
- Non-fish shells: adapt the venv activation command for your shell, but this repo standardizes on fish (`source .venv/bin/activate.fish`).
- CLIP weights not found/offline: the PERCEIVE test will skip if it cannot
  instantiate the CLIP model (requires access to a local HF cache or network).
- EmoNet weights missing: the face adapter runs with random init; results are
  not meaningful but the test only validates output ranges.
