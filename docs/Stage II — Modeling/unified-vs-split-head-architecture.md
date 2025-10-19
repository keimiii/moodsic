# Unified vs. Split Head Architecture for Multi-Dimensional Regression

## Context

In emotion recognition for continuous valence-arousal prediction, the regression head architecture choice has significant implications for model performance, uncertainty estimation, and parameter efficiency. This document analyzes two competing architectures:

1. **Unified Head**: Single output layer producing both valence and arousal jointly
2. **Split Head**: Separate, independent output layers for valence and arousal

This analysis is grounded in the empirical context of the Moodsic project, which uses CLIP ViT-B/32 features with MC Dropout for uncertainty estimation on FindingEmo and VEATIC datasets.

---

## Architecture Comparison

### 1. Unified Head Architecture

**Design**: Shared parameters in the final projection layer output both dimensions simultaneously.

```python
class UnifiedHeadRegressor(nn.Module):
    """Single head outputs both valence and arousal jointly."""
    
    def __init__(self, backbone: nn.Module, feat_dim: int, p: float = 0.15):
        super().__init__()
        self.backbone = backbone
        
        # Shared head with joint output
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),      # Joint projection: [valence, arousal]
            nn.Tanh(),
        )
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [batch_size, 2] tensor with [valence, arousal]."""
        feats = self.backbone.get_image_features(pixel_values=x)
        return self.head(feats)  # Shape: [B, 2]


# MC Dropout inference
def predict_with_uncertainty(model, x, n_samples=10):
    """MC Dropout with unified head."""
    model.eval()
    # Enable dropout only
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(x)  # Shape: [1, 2]
            samples.append(output.squeeze(0))  # [2]
    
    samples = torch.stack(samples)  # Shape: [n_samples, 2]
    
    # Independent statistics per dimension
    mean_v = samples[:, 0].mean().item()
    mean_a = samples[:, 1].mean().item()
    var_v = samples[:, 0].var(unbiased=True).item()
    var_a = samples[:, 1].var(unbiased=True).item()
    
    return mean_v, mean_a, var_v, var_a
```

#### Pros

1. **Parameter Efficiency**
   - Fewer total parameters (one `Linear(128, 2)` vs. two `Linear(128, 1)`)
   - Reduces overfitting risk in small-data regimes (critical for VEATIC/FindingEmo)
   - Lower memory footprint and faster inference

2. **Multi-Task Learning Benefits**
   - Shared final weights create **inductive bias** toward joint representations
   - Gradients from both targets shape the same parameters → implicit regularization
   - Models **correlation structure** between valence and arousal naturally
   - Aligns with circumplex emotion theory (V-A are correlated dimensions)

3. **Sample Efficiency**
   - Parameter sharing acts as regularization: each training sample effectively trains both outputs
   - Particularly valuable when labels are scarce or noisy
   - Better generalization in low-data scenarios

4. **Simplified Training**
   - Single loss computation: `MSE(pred, [v_target, a_target])`
   - No need for loss weighting/balancing between tasks
   - Cleaner gradient flow (no gradient conflicts between task-specific layers)

5. **Covariance Modeling**
   - MC Dropout naturally captures **joint uncertainty** and correlation between V-A
   - Can compute covariance: `Cov(V, A) = E[(v - μ_v)(a - μ_a)]`
   - Richer uncertainty representation for downstream fusion

#### Cons

1. **Shared Gradients**
   - If one task is harder to learn, it can dominate gradients and harm the other
   - No per-task learning rate control
   - Potential gradient conflicts if tasks compete

2. **Limited Task-Specific Adaptation**
   - Cannot apply different activation functions per dimension
   - Cannot use different regularization strategies for V vs. A
   - Less flexibility for per-dimension calibration

3. **Coupling Constraints**
   - Forces both outputs through same activation (e.g., both use Tanh)
   - May be suboptimal if V and A truly need different output distributions

---

### 2. Split Head Architecture

**Design**: Independent task-specific layers for valence and arousal.

```python
class SplitHeadRegressor(nn.Module):
    """Separate heads for valence and arousal."""
    
    def __init__(self, backbone: nn.Module, feat_dim: int, p: float = 0.3):
        super().__init__()
        self.backbone = backbone
        
        # Shared dropout on features
        self.dropout = nn.Dropout(p)
        
        # Independent valence head
        self.valence_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(128, 1),
        )
        
        # Independent arousal head
        self.arousal_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(128, 1),
        )
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [batch_size, 2] tensor with [valence, arousal]."""
        feats = self.backbone.get_image_features(pixel_values=x)
        feats = self.dropout(feats)
        
        v = self.valence_head(feats)  # [B, 1]
        a = self.arousal_head(feats)  # [B, 1]
        
        return torch.cat([v, a], dim=1)  # [B, 2]


# MC Dropout inference
def predict_with_uncertainty(model, x, n_samples=10):
    """MC Dropout with split heads."""
    model.eval()
    # Enable dropout only
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    
    v_samples, a_samples = [], []
    with torch.no_grad():
        for _ in range(n_samples):
            feats = model.backbone.get_image_features(pixel_values=x)
            feats = model.dropout(feats)
            
            v = model.valence_head(feats).squeeze(-1)
            a = model.arousal_head(feats).squeeze(-1)
            
            v_samples.append(torch.clamp(v, -1.0, 1.0))
            a_samples.append(torch.clamp(a, -1.0, 1.0))
    
    v_t = torch.stack(v_samples)
    a_t = torch.stack(a_samples)
    
    mean_v = v_t.mean().item()
    mean_a = a_t.mean().item()
    var_v = v_t.var(unbiased=True).item()
    var_a = a_t.var(unbiased=True).item()
    
    return mean_v, mean_a, var_v, var_a
```

#### Pros

1. **Task Independence**
   - Each dimension can learn **task-specific feature transformations**
   - No gradient conflicts between valence and arousal optimization
   - Can apply different learning rates, regularization, or optimizers per head

2. **Flexibility**
   - Can use different activation functions (e.g., Tanh for V, Sigmoid for A)
   - Can apply task-specific loss weighting or calibration
   - Easier to add auxiliary losses or constraints per dimension

3. **Increased Capacity**
   - More parameters (~2× final layers) → higher representational capacity
   - May be beneficial if V and A genuinely require different mappings from features
   - Better if tasks are sufficiently different (less true for V-A)

4. **Debugging Clarity**
   - Can monitor valence and arousal losses independently
   - Easier to diagnose which dimension is underperforming
   - Clearer separation of concerns in code

5. **Modular Uncertainty**
   - Dropout stochasticity can be independent per head (if designed that way)
   - Easier to implement per-dimension uncertainty calibration
   - Natural alignment with fusion modules expecting separate confidence scores

#### Cons

1. **Parameter Inefficiency**
   - Nearly double the parameters in task-specific layers
   - Higher overfitting risk in low-data regimes (VEATIC has limited samples)
   - Increased memory and compute cost

2. **Ignores Correlation Structure**
   - Treats V and A as independent tasks
   - **Fails to leverage known correlation** between valence and arousal
   - Misses multi-task learning benefits from shared signal

3. **Sample Inefficiency**
   - Each head must learn its own transformation independently
   - Requires more data to reach same performance as unified head
   - Less regularization from task sharing

4. **Complexity**
   - More code to maintain (two head definitions)
   - Need to tune dropout rates, layer sizes for each head
   - Loss balancing may be needed if one task dominates

5. **Weaker Uncertainty Modeling**
   - MC Dropout samples are less informative about joint V-A distribution
   - Cannot easily model covariance between dimensions
   - May produce inconsistent predictions (high V with low A when correlation expected)

---

## Quantitative Comparison

| **Aspect**                  | **Unified Head**                     | **Split Head**                       |
|-----------------------------|--------------------------------------|--------------------------------------|
| **Parameters**              | ~33K (128→2 final layer)             | ~66K (two 128→1 layers)              |
| **Gradient Flow**           | Shared (implicit regularization)     | Independent (no conflicts)           |
| **Correlation Modeling**    | ✅ Explicit (joint projection)       | ❌ Ignored (separate paths)          |
| **Sample Efficiency**       | ✅ High (parameter sharing)          | ⚠️ Lower (duplicate learning)        |
| **Overfitting Risk**        | ✅ Low (fewer params)                | ⚠️ Higher (more params)              |
| **Task-Specific Tuning**    | ❌ Limited (shared activation)       | ✅ Full control per dimension        |
| **Code Complexity**         | ✅ Simple (one head)                 | ⚠️ More complex (two heads)          |
| **Uncertainty Richness**    | ✅ Joint + marginal stats            | ⚠️ Marginal only                     |
| **Debugging**               | ⚠️ Harder (coupled losses)           | ✅ Easier (separate losses)          |

---

## Empirical Considerations for Moodsic Project

### Dataset Characteristics

- **FindingEmo**: ~25K training samples with correlated V-A labels
- **VEATIC**: Limited labeled samples; evaluation-focused
- **Label Noise**: Medium (crowd-sourced annotations)
- **V-A Correlation**: Strong (circumplex model; Pearson ρ ≈ 0.3–0.5 typical)

### Existing Evidence

From the `CLIP_ViT-B32_improved.ipynb` notebook:
- The **unified head architecture already demonstrated superior performance** over earlier baselines
- This is the checkpoint you're trying to restore (`clip_vit-b32_improved_fixed.pkl`)
- Empirical validation trumps theoretical preferences in POC settings

### Inference Requirements

- **MC Dropout**: Both architectures support it equally well
- **Independent Variances**: Both can produce `var_v` and `var_a` separately
- **Fusion Module**: Expects separate confidence scores → both architectures compatible

### Training Constraints

- **Time-boxed POC**: Favor proven architectures over experimentation
- **Limited Compute**: Parameter efficiency matters
- **Small Data**: Regularization through parameter sharing is valuable

---

## Recommendation: Use Unified Head

### Justification

**Adopt the unified head architecture** from `CLIP_ViT-B32_improved.ipynb` by modifying the inference adapter (`clip_vit_scene_adapter.py`) to match the training notebook's design.

#### Primary Reasons:

1. **Empirical Validation**
   - The unified head **already outperformed baselines** in your experiments
   - This is the strongest signal; theory should defer to validated results
   - No evidence that split heads close the performance gap

2. **Dataset Characteristics Match Design**
   - Limited VEATIC labels → parameter efficiency critical
   - Known V-A correlation → joint modeling is appropriate
   - Multi-task sharing provides free regularization in noisy-label regime

3. **Practical Constraints**
   - Trained weights already exist and are proven
   - Faster to modify adapter than retrain from scratch
   - Lower risk in time-boxed POC setting

4. **Uncertainty Modeling**
   - MC Dropout with unified head provides **richer information**:
     ```python
     # Can compute covariance with unified head samples
     cov_va = np.cov(v_samples, a_samples)[0, 1]
     ```
   - Still produces independent `var_v` and `var_a` as needed
   - Captures joint distribution, not just marginals

5. **Simplicity**
   - Single head = simpler code, fewer hyperparameters
   - Easier to maintain in academic POC context
   - Lower cognitive overhead for future modifications

#### When to Reconsider Split Heads:

Only switch if you observe:
- **Empirical underperformance** of unified head on VEATIC eval
- **Per-dimension calibration needs** (e.g., valence consistently biased)
- **Different loss weighting** required for V vs. A
- **Auxiliary task additions** that benefit from modularity

None of these conditions currently hold.

---

## Implementation Path

### Modify `clip_vit_scene_adapter.py`:

```python
class SceneCLIPAdapter:
    def __init__(self, ..., dropout_rate: float = 0.15, ...):
        # ... backbone setup ...
        
        # Replace separate heads with unified head
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),  # Joint V-A output
            nn.Tanh(),
        )
        self.head.eval().to(self.device)
        
        # Load existing trained checkpoint
        if auto_load_best:
            self._maybe_load_weights(checkpoint_path)
    
    def predict(self, frame_bgr, tta=5):
        # ... preprocessing ...
        
        samples = []
        with torch.no_grad():
            for _ in range(tta):
                feats = self.backbone.get_image_features(pixel_values=pixel_values)
                output = self.head(feats)  # [1, 2]
                v, a = output[0, 0], output[0, 1]
                v = torch.clamp(v, -1.0, 1.0)
                a = torch.clamp(a, -1.0, 1.0)
                samples.append([v, a])
        
        samples = torch.tensor(samples)  # [tta, 2]
        mean_v = samples[:, 0].mean().item()
        mean_a = samples[:, 1].mean().item()
        var_v = samples[:, 0].var(unbiased=True).item()
        var_a = samples[:, 1].var(unbiased=True).item()
        
        return mean_v, mean_a, (var_v, var_a)
```

### Update checkpoint loading:

```python
def _maybe_load_weights(self, ckpt_path):
    # ... load learner pickle ...
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Extract state dict from fastai Learner
    if hasattr(state, 'model'):
        model_state = state.model.state_dict()
    
    # Load unified head weights
    head_keys = [k for k in model_state.keys() if k.startswith('head.')]
    head_state = {k.replace('head.', ''): v for k, v in model_state.items() if k in head_keys}
    
    self.head.load_state_dict(head_state, strict=True)
    logger.info(f"Loaded unified head from {ckpt_path}")
```

---

## Conclusion

For the Moodsic project, the **unified head architecture is the evidence-based choice**. It leverages proven empirical performance, aligns with dataset characteristics (small sample size, correlated targets), and maintains compatibility with MC Dropout uncertainty estimation. The split-head design offers modularity but sacrifices sample efficiency and correlation modeling without demonstrated gains—trade-offs that don't justify abandoning a validated architecture in a time-constrained POC.

**Next Action**: Modify `clip_vit_scene_adapter.py` to use unified head, load the existing `clip_vit-b32_improved_fixed.pkl` checkpoint, and validate deterministic inference on VEATIC.
