# Cross-Domain Calibration

- [✅] Implement 4-parameter affine transformation for domain bias correction
- [✅] Support for EmoNet (face) → FindingEmo (scene) alignment
- [✅] Learnable parameters with L2 regularization and identity initialization
- [✅] Train calibration layer on FindingEmo (reference space [-1,1])
- [✅] Run ablation study to validate effectiveness

## Overview

CrossDomainCalibration corrects systematic biases between emotion recognition domains after mathematical scale alignment. While `emotion_scale_aligner.py` handles dataset-specific ranges, this layer learns domain-specific shifts.

**Problem**: EmoNet (trained on facial expressions) may systematically differ from scene-based emotions even after scale normalization due to:
- Annotation differences (face crops vs full scenes)
- Semantic gaps (facial muscle patterns ≠ scene emotions)
- Different human labeler pools and guidelines

**Solution**: 4-parameter affine transform that learns bias correction from validation data.

## Architecture

```python
class CrossDomainCalibration(nn.Module):
    def __init__(self, use_tanh: bool = False):
        # Learnable parameters (initialized to identity: no change)
        self.scale_v = nn.Parameter(torch.ones(1))   # Valence scaling
        self.scale_a = nn.Parameter(torch.ones(1))   # Arousal scaling  
        self.shift_v = nn.Parameter(torch.zeros(1))  # Valence offset
        self.shift_a = nn.Parameter(torch.zeros(1))  # Arousal offset
        self.use_tanh = use_tanh
        
    def forward(self, v, a):
        v_out = v * self.scale_v + self.shift_v
        a_out = a * self.scale_a + self.shift_a
        if self.use_tanh:
            # Optional smooth clamping; for training we keep it disabled to avoid range shrinkage
            return torch.tanh(v_out), torch.tanh(a_out)
        return v_out, a_out
```

**Why 4 parameters?**
- **Scale factors**: Correct amplitude differences (e.g., EmoNet arousal 20% lower)
- **Shift factors**: Correct center-point bias (e.g., EmoNet 0.1 units more positive)
- **Minimal design**: Prevents overfitting with limited validation data
- **Identity initialization**: No change unless training finds systematic bias

## Training

Use MSE loss with L2 regularization:

```python
# Loss matching project conventions
total_loss = F.mse_loss(pred, target) + l2_reg

# L2 regularization prevents drift from identity
l2_reg = λ * [(scale_v - 1)² + (scale_a - 1)² + shift_v² + shift_a²]
```

**Training procedure (fixed):**
1. Use faces-found subset of FindingEmo with EmoNet predictions and GT (EmoNet remains frozen)
2. Convert both predictions and GT to reference space [-1,1]
3. Evaluate the calibration layer out-of-sample (e.g., 80/20 val split or k-fold CV); train with MSE; disable tanh clamping during optimization
4. Early stopping when validation loss plateaus; monitor parameters — if near identity, skip calibration
5. Deployment: if calibration shows out-of-sample gains, refit the 4 parameters on 100% of the faces-found subset for the runtime model

## Integration Pipeline

```python
# Complete emotion processing pipeline
EmoNet raw → emotion_scale_aligner → CrossDomainCalibration → final (v,a)

# Usage
from utils.emotion_pipeline import EmotionPipeline
pipeline = EmotionPipeline(calibration_layer=trained_calibration)

# With calibration
v_final, a_final = pipeline.emonet_to_findingemo(v_emonet, a_emonet)

# Without calibration (for comparison)
v_baseline, a_baseline = pipeline.emonet_to_findingemo(v_emonet, a_emonet, apply_calibration=False)
```

## Validation Strategy

**1. Ablation Study**
- Train multiple runs with different random seeds
- Compare "with" vs "without" calibration on held-out test set
- Use paired t-tests for statistical significance
- Require p < 0.05 and meaningful effect size

**2. Generalization Test**
- Test on completely different dataset (EMOTIC, ArtPhoto)
- Ensure calibration doesn't overfit to FindingEmo specifics
- Monitor for performance degradation on unseen domains

**3. Parameter Monitoring**
- Track learned parameters during training
- If params stay near (1,1,0,0), no systematic bias exists → remove layer
- Log parameter evolution and convergence

**4. Bias Visualization**
- Bland-Altman plots showing bias reduction
- Residual analysis before/after calibration
- Distribution plots of prediction errors

## Usage Guidelines

**When to use:**
- Clear domain shift between training data (AffectNet faces) and target application (scene emotions)
- Sufficient paired validation data (≥500 samples recommended)
- Statistical validation shows significant improvement

**When NOT to use:**
- Parameters converge to identity transform
- No improvement in ablation study
- Degraded performance on generalization test
- Insufficient validation data for reliable parameter estimation

**Implementation flags:**
```python
# Always test with calibration disabled for comparison
pipeline = EmotionPipeline(enable_calibration=False)  # Baseline
pipeline = EmotionPipeline(enable_calibration=True)   # With calibration
```

## Results: EmoNet → FindingEmo (Aug 25, 2025)

- Setup: Trained on faces-found subset in reference space [-1,1]; no tanh clamp during training; 4-parameter affine.
- Baseline (faces-found): Valence CCC ≈ 0.167, Arousal CCC ≈ 0.016 (mean ≈ 0.092)
- After calibration (A/B on same faces-found set): Valence CCC ≈ 0.295, Arousal CCC ≈ 0.064 (mean ≈ 0.180)
- Ranking (r, ρ) unchanged → calibration corrects bias/scale, not ordering.
- Holdout ceiling (best affine on validation split): Valence CCC ≈ 0.205, Arousal CCC ≈ 0.017 → simple linear mapping cannot close the gap.

Conclusion: Calibration can modestly improve valence on FindingEmo but remains well below acceptable thresholds; arousal gains are small and often not significant. Use calibration only if it improves holdout V/A MAE; otherwise omit. The headline POC metric should rely on fusion + gating rather than face-only.
