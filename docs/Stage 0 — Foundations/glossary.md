# Glossary

- Valence/Arousal scales:
  - FindingEmo valence: `[-3, 3]`
  - FindingEmo arousal: `[0, 6]`
  - DEAM valence/arousal: `[-10, 10]`
  - FE→DEAM mapping:
    - Valence: `v_deam = (10/3) * v_fe`
    - Arousal: `a_deam = -10 + (20/6) * a_fe`

- PERCEIVE: Per-frame extraction of valence–arousal using scene and face models with MC Dropout for uncertainty.

- STABILIZE: Exponential Moving Average smoothing with uncertainty gating (hold last stable output when variance exceeds a threshold). Defaults cited for MVP: `alpha = 0.7`, `uncertainty_threshold τ ≈ 0.4`, `n_mc_samples ≈ 5`.

- MATCH: Segment-level retrieval over DEAM with KD-Tree, 10s windows, 50% overlap, k=20; minimum dwell time 20–30s before switching; track recently played songs to ensure variety.

- MC Dropout: Multiple stochastic forward passes with dropout active to estimate prediction uncertainty (mean and variance across samples).

- EMA (Exponential Moving Average): Temporal smoothing where `ema_t = α * x_t + (1-α) * ema_{t-1}`.

- CCC (Concordance Correlation Coefficient): Primary evaluation metric combining correlation and agreement between predictions and ground truth.

- Scene–Face Divergence: Mean Euclidean distance between scene and face predictions when both available; high divergence may indicate context overfitting in the scene model.

- Dwell time: Minimum duration (≈20–30s) to keep the current music segment before allowing a switch.

- KD-Tree / k-NN: Data structure and method used for fast nearest-neighbor search over DEAM segment valence–arousal space.

- Fusion (variance-weighted): Combine predictions by weighting each by inverse variance; normalized to produce fused prediction and approximate fused variance.
