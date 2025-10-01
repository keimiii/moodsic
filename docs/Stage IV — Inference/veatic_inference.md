# VEATIC Inference Notes

This document captures the current VEATIC-specific runtime behavior for the PERCEIVE stage and outlines the assumptions behind the demo notebooks and `utils/runtime_driver.py` helper.

## Scope

- Applies to one-shot video evaluation (e.g., `notebooks/e2e_video_to_fusion.ipynb`).
- Uses the fused scene + face pipeline implemented in `models/fusion.py` and orchestrated by `utils/runtime_driver.py`.
- Focuses on preprocessing, sampling cadence, and configuration that align VEATIC clips with the rest of the stack.

## Video Sampling / Preprocessing

- **Default cadence:** `perceive_video(..., target_sample_fps=1.0)` rounds the native FPS to the nearest stride so we process ≈1 frame per second. This mirrors the notebook default (`TARGET_SAMPLE_FPS = 1.0`) and keeps the valence/arousal timeseries manageable while tracking the VEATIC label cadence.
- **Stride calculation:** When no explicit `frame_stride` is supplied, the helper computes `stride = round(native_fps / desired_fps)` (minimum 1). Non-positive or ≥ native target FPS collapses to `stride = 1` (full-rate).
- **Frame selection controls:**
  - `start_frame` skips an initial portion of the video.
  - `max_frames` hard-caps the number of processed samples (after stride is applied).
  - `max_hz` throttles repeated calls to `PerceiveFusionDriver.step` (defaults to no throttling for offline processing).
- **Overlays:** Set `capture_overlays=True/False` and `save_overlay_to='outputs/overlay_fusion.mp4'` to generate annotated video; disabled when storage or OpenCV codecs are unavailable.

## Face Crops & Scene Frames

- The face pathway uses `utils/emonet_single_face_processor.EmoNetSingleFaceProcessor`:
  - Converts BGR → RGB for MediaPipe long-range face detection.
  - Scores candidates by detection confidence × sqrt(area) × center proximity.
  - Adds a Haar cascade fallback when MediaPipe returns fewer than `max_faces` unique boxes.
  - Applies configurable padding (`padding_ratio=0.2`) and resizes crops to 256×256 by default before EmoNet inference.
- Scene frames are fed directly to the CLIP/ViT adapter (`models/scene/clip_vit_scene_adapter.py`), which performs MC Dropout sampling for uncertainty.

## Fusion & Stabilization Defaults

- Fusion is handled by `models/fusion.SceneFaceFusion` with inverse-variance weighting and guardrails:
  - Scene / face prior weights: 0.6 / 0.4 when variances are unavailable.
  - Optional gating on face score, brightness, and variance floor.
- Stabilization:
  - Disabled in the first notebook example, enabled in the second (`ENABLE_STABILIZER = True`).
  - EMA α defaults to `0.7`; uncertainty gating holds the previous value when σ > 0.4 and maintains a history window of 60 samples for jitter metrics.

## Outputs & Diagnostics

- `VideoPerceptionResult` returns:
  - Fused V/A sequences in reference space `[-1, 1]` with per-sample variances.
  - Effective sampling FPS, frame indices, average V/A, and optional overlays.
- The notebook plots fused valence/arousal and (when available) the associated variances for quick QA.
- Downstream evaluation should align these outputs with VEATIC’s frame-level labels (e.g., by matching timestamps or frame indices after stride/downsampling).

## Proposed Music Cue Segmentation (PoC)

- Motivation: the scene adapter is trained on still images while inference runs on full clips. Processing one frame per second keeps latency low but flattens emotional build-ups; we want to identify the specific spans where music should enter instead of scoring the entire clip.
- Plan: run offline change-point detection (PELT via `ruptures`) over the fused valence–arousal time series returned by `perceive_video`. Smooth VA with a short rolling mean, optionally transform to an energy envelope (e.g., `sqrt(v**2 + a**2)`), and feed the sequence to the detector.
- Output: change points split each clip into emotion regimes. Post-process segments (minimum duration, mean energy thresholds, uncertainty gating) to mark candidate windows for music cues while leaving low-intensity lead-ins untouched.
- Benefits over current flow:
  - Without change points we either (a) score the whole clip uniformly or (b) hand-trim sections, both ignoring VEATIC’s continuous labels.
  - With change points the pipeline can automatically isolate the high-arousal climax of a tense scene or a brief emotional twist inside an otherwise calm dialogue, enabling tighter cue timing.
- PELT (Pruned Exact Linear Time) in plain language: think of it as an algorithm that watches the emotional curve and decides where to “cut” the video into chapters. It tries every possible split, but quickly prunes the ones that would not improve the story, so we keep only the points where the mood truly changes while keeping computation fast.
- Illustrative scenarios:
  - **Slow-burn to climax:** a VEATIC clip spends 40 seconds in low-energy dialogue before a sudden confrontation. Today the 1 fps samples all look similar, so the MATCH stage would score the entire minute. PELT spots the jump in the smoothed VA envelope, cuts the clip into “build-up” and “climax” segments, and we can launch the soundtrack only from the inflection point onward.

    | Time (s) | Valence | Arousal | VA Energy | Action without PELT | Action with PELT |
    | --- | --- | --- | --- | --- | --- |
    | 0–39 | –0.1 | 0.05 | 0.11 | Score entire clip | Mark silence (setup) |
    | 40–60 | 0.3 | 0.75 | 0.81 | Score entire clip | Trigger music cue |

  - **Blink-and-you-miss-it twist:** another clip stays calm until a 6-second surge of fear, then returns to baseline. Without segmentation that surge is averaged away; with PELT the brief spike becomes its own segment even though it is short, so we can insert a dramatic sting exactly where the emotional swing happens and keep the rest silent.

    | Time (s) | Valence | Arousal | VA Energy | Action without PELT | Action with PELT |
    | --- | --- | --- | --- | --- | --- |
    | 0–24 | 0.05 | 0.10 | 0.11 | Subtle pad (false positive) | Hold silence |
    | 25–31 | –0.45 | 0.85 | 0.96 | Same pad (missed impact) | Insert sting |
    | 32–50 | 0.00 | 0.08 | 0.08 | Pad continues | Return to silence |
- Next steps: add a notebook experiment inside `docs/Stage IV — Inference` that loads a `VideoPerceptionResult`, runs the segmentation routine, and overlays detected regions on the VA plots for review before wiring outputs into MATCH.

## Integration Pointers

- Use `search_roots` in `perceive_video` when videos live outside the notebook directory.
- Enabling `return_fusion=True` exposes the raw `FusionResult` objects for deeper inspection (path-specific V/A, σ, gating flags).
- MATCH stage wiring (DEAM retrieval) is tracked separately; see `docs/Stage IV — Inference/runtime-pipeline.md` for the broader PERCEIVE → STABILIZE → MATCH plan.

## References

- Notebook: `notebooks/e2e_video_to_fusion.ipynb`
- Runtime helper: `utils/runtime_driver.py`
- Fusion core: `models/fusion.py`
- Face processor: `utils/emonet_single_face_processor.py`
- Scene adapter example: `models/scene/clip_vit_scene_adapter.py`

Add future VEATIC-specific evaluation notes here (e.g., label alignment scripts, metrics, or comparative baselines) to keep the runtime documentation centralized.
