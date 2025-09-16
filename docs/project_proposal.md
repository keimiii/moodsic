# 1. Summary

Moodsic is an emotion-aware music recommendation system that detects the
emotional state of video content and selects music that matches or gently
modulates it. Unlike single-source emotion systems that rely only on facial
expressions or scene cues, Moodsic fuses both sources with uncertainty-aware
weighting and temporal smoothing to produce stable, context-appropriate
recommendations. The solution targets two primary settings: (1) soundtrack
selection for short-form or AI-generated videos, and (2) future deployment in
physical spaces where adaptive background music can shape customer experience.

Deliverables include a working demo app (Streamlit/WebRTC), trained models,
and a compact evaluation suite demonstrating end-to-end performance.

---

# 2. Problem & Pain Points

- Context–person ambiguity: Scene-only or face-only approaches conflate
  environmental mood with genuine human affect.
- Instability: Per-frame predictions can jitter, causing jarring music flips.
- Personalization mismatch: Classic CF/CB recommenders optimize for individual taste, not affective fit to the moment or group setting.
- Scale mismatches: Datasets label emotions in different numeric ranges,
  complicating training, evaluation, and retrieval.

---

# 3. Proposed Approach (High Level)

Pipeline: Perceive → Stabilize → Match

- Perceive: Dual-pathway mixture-of-experts
  - Scene expert: CLIP/DINOv3 backbone + lightweight regressor predicts valence
    and arousal from full-frame context.
  - Face expert: Pretrained EmoNet on primary face crops (MediaPipe), aligned
    and normalized to EmoNet’s expected input.
- Stabilize: Inverse-variance fusion and EMA smoothing with uncertainty gating
  hold the last stable value when confidence is low to prevent flicker.
- Match: Map stabilized valence–arousal to songs in DEAM via scale alignment
  and k-NN retrieval; optional mood clusters ensure coherent soundtrack
  regions and minimum dwell time prevents rapid switches.

Why this approach: It addresses the fundamental attribution problem by
disentangling scene ambiance from human affect, then ensures practical runtime
stability suitable for real applications.

---

# 4. Datasets

All datasets use the valence–arousal (V/A) affect space, enabling direct cross-domain mapping:

- FindingEmo (images; training/ablation): ≈25k images of people in scenes with
  V/A labels. Current filtered split uses 19,606 images.
- VEATIC (video; inference evaluation): 124 videos with frame-level V/A labels
  for temporal evaluation and stability analysis.
- DEAM (music; retrieval): 1,802 songs annotated with V/A and metadata for
  emotion-aligned matching.

Scale alignment is handled by a common reference space `[-1, 1]` with dataset
conversions applied only at boundaries (training/eval, retrieval, viz).

---

# 5. Evaluation & Success Criteria

Protocol

- FindingEmo: Train scene regressors on frozen backbones; select by lowest
  Average MAE (mean of V-MAE, A-MAE) on test split; report Spearman’s ρ.
- VEATIC: Run the full inference pipeline; align to annotated frames and
  report per-video and overall V/A MAE, Average MAE, and Spearman’s ρ. Compare
  raw vs. stabilized outputs and face-detection coverage.

System-level metrics (runtime quality)

- Jitter reduction: 40–60% reduction in frame-to-frame variation after
  stabilization.
- Switching behavior: Minimum dwell time attained; reasonable variety without
  churning.
- Scene–face divergence: Reduced divergence after fusion vs. scene-only.

Targets (POC)

- Scene baseline: Establish Average MAE on FindingEmo; use as reference.
- Fusion uplift: Demonstrate improvement vs. scene-only on VEATIC (MAE and ρ).
- Stability: Achieve jitter reduction within the 40–60% band without excessive
  lag on real videos.

---

# 6. Scope & Team Complexity (4 Members)

Subsystems and responsibilities

- Experimentation & Modeling Infrastructure: Standardized scene‑model runner
  supporting HF Transformers (e.g., DINO/CLIP/ViT) and timm backbones via a
  common head/metrics pipeline; CSV logging, LR finder, and reproducible
  splits for ablations across architectures.
- Face Pathway Engineering: MediaPipe‑based single‑face selection/cropping and
  a production‑style EmoNet adapter with optional eye‑level alignment,
  TTA‑based uncertainty, and an optional cross‑domain calibration layer with
  trainer/evaluator for holdout validation.
- Fusion & Stabilization: Variance‑weighted mixture‑of‑experts combining scene
  and face predictions with safety guardrails (score/σ/brightness gating) and
  an optional EMA stabilizer with uncertainty gating, plus unit tests for math
  and edge cases.
- Data Ops: High‑throughput parallel downloader with archival fallback,
  dataset filtering/validation, and unified scale conversion utilities across
  FindingEmo/EmoNet/DEAM.
- Retrieval & Music (POC): DEAM indexing and GMM‑gated k‑NN design with dwell
  and recent‑song memory, mapped via the unified scale aligner.
- Runtime & UI: Streamlit/WebRTC app scaffold, driver skeleton for the PERCEIVE
  stage, and a lightweight overlay utility for debugging fused outputs and
  uncertainties.

Hidden complexity already built (justifying team distribution)

- Uncertainty plumbing: MC Dropout in the scene adapter and TTA-derived
  variance in the face adapter feed inverse-variance fusion, with guardrails
  (score/σ/brightness) and an EMA stabilizer that exposes stability metrics
  (variance, jitter) for analysis.
- Calibration rigor: A lightweight cross-domain affine calibrator is trained
  with a CCC+MSE composite objective, early-stopping, and significance tests;
  it defaults to identity when improvements are not statistically supported.
- Data resilience: Downloader falls back to Wayback (API + "if_" rewriting) and
  enforces concurrency limits; CSV filtering validates existence/size and image
  decodability to keep splits clean.
- Scale discipline: A single reference space [-1, 1] and one aligner utility
  prevent drift across FE/DEAM/EmoNet; conversions only at boundaries reduce
  subtle evaluation/retrieval mismatches.
- QA & observability: Unit tests verify closed‑form inverse‑variance fusion and
  edge cases; an overlay renders per‑path/fused values and σ for manual QA and
  artifact export.

Integration complexity

- Multimodal fusion with confidence weighting, guardrails, and temporal logic
  balancing responsiveness vs. stability.
- Cross‑domain calibration and scale alignment across three domains
  (FindingEmo/EmoNet/DEAM) with consistent reference space handling.
- Experimentation to production: exporting scene heads, wiring adapters, and
  preserving metric parity between notebook runs and runtime adapters.
- Retrieval policy design (GMM gating, k‑NN shortlist, dwell/repeat logic)
  with clear scale conversions and safety constraints.
- Planned end‑to‑end evaluation (image → video → retrieval) including ablations
  and system‑level stability criteria; VEATIC evaluation plan and EDA are in
  place.

---

# 7. Curriculum Alignment

This section references the attached “Practice Module: Project Work —
Requirements” image. The proposal explicitly addresses each requirement:

- Team size: 4 members (≤ 5) enrolled as a group.
- Practical application: Pattern recognition + machine learning system that
  demonstrates clear advantage over naive/static music selection.
- Real-world problem: Design and build an AI/ML pattern recognition system to
  solve emotion-aware soundtrack selection for videos (and future physical
  spaces), applying skills from the 3 modules.

Demonstrates at least three aspects (we cover all four):

- Supervised learning / unsupervised learning scenarios: Supervised regression
  heads for V/A on FindingEmo; unsupervised mood clustering with GMM for
  station gating and retrieval coherence.
- Machine learning / Deep learning techniques: Transformer/ViT backbones,
  transfer learning, MC Dropout/TTA for uncertainty; scale alignment and
  calibration utilities.
- Hybrid machine learning / Ensemble approach: Mixture-of-experts fusion (face
  + scene) with inverse-variance weighting and temporal smoothing.
- Intelligent sensing / sense making techniques: WebRTC camera/video as
  sensing; MediaPipe face detection; fusion-based sense-making to infer affect
  and decide music via V/A retrieval with minimum dwell logic.

Additional considerations

- Information retrieval: k-NN over DEAM V/A space with optional mood stations.
- Ethics & licensing: Face privacy considerations and EmoNet used unmodified
  under CC BY-NC-ND 4.0 (non‑commercial).
