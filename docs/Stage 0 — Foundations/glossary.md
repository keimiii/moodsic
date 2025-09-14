# Glossary

- Valence/Arousal scales:
  - FindingEmo valence: `[-3, 3]`
  - FindingEmo arousal: `[0, 6]`
  - DEAM dynamic annotations (per-frame): `[-10, 10]`
  - DEAM static annotations (per 45s excerpt): `[1, 9]`
  - For this academic POC, we use DEAM static annotations `[1, 9]`.
  - FE→DEAM mapping (POC: to static [1, 9]):
    - Valence: `v_deam = 1 + (8/6) * (v_fe + 3)`
    - Arousal: `a_deam = 1 + (8/6) * a_fe`

- PERCEIVE: Per-frame extraction of valence–arousal using scene and face models with MC Dropout for uncertainty.

- STABILIZE: Exponential Moving Average smoothing with uncertainty gating (hold last stable output when variance exceeds a threshold). Defaults cited for MVP: `alpha = 0.7`, `uncertainty_threshold τ ≈ 0.4`, `n_mc_samples ≈ 5`.

- MATCH (POC): Song-level retrieval over DEAM static [1, 9] using simple k-NN (linear scan) plus minimum dwell time (20–30s) and recent-song memory.  
GMM “station” gating assigns the stabilized V/A to the most likely cluster via `predict_proba`. If the top posterior is low (e.g., < 0.55), widen the gate to the top-2 clusters. Rank within the selected cluster set by Euclidean distance between the stabilized V/A and each song’s V/A; pick top-1 (or top-N for variety).

- MC Dropout: Multiple stochastic forward passes with dropout active to estimate prediction uncertainty (mean and variance across samples).

- EMA (Exponential Moving Average): Temporal smoothing where `ema_t = α * x_t + (1-α) * ema_{t-1}`.

- MAE (Mean Absolute Error): Primary evaluation metric for valence and arousal; lower is better and easy to interpret.
- CCC (Concordance Correlation Coefficient): Deprecated due to instability when predictions or targets have near-zero variance (torchmetrics may return NaN). Not used.

- Scene–Face Divergence: Mean Euclidean distance between scene and face predictions when both available; high divergence may indicate context overfitting in the scene model.

- Dwell time: Minimum duration (≈20–30s) to keep the current song before allowing a switch.

- GMM Stations: Gaussian Mixture clusters (K≈5, diag covariance) trained on
  DEAM song-level valence–arousal (in reference space with StandardScaler).
  Used to gate or bias retrieval toward the active “mood station.”

- Fusion (variance-weighted): Combine predictions by weighting each by inverse variance; normalized to produce fused prediction and approximate fused variance.

- Affine Calibration: Cross-domain bias correction using learnable linear transformation `output = scale * input + shift`. Corrects systematic differences between emotion datasets (e.g., EmoNet face predictions vs FindingEmo scene labels) with 4 parameters: scale_v, scale_a, shift_v, shift_a.

- Domain Shift: Systematic biases between emotion recognition contexts (facial expressions vs scene emotions) due to different annotation protocols, human labelers, and semantic gaps between modalities.
