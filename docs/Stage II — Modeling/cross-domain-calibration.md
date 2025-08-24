# Cross-Domain Calibration

- [✅] Implement 4-parameter affine transformation for domain bias correction
- [✅] Support for EmoNet (face) → FindingEmo (scene) alignment
- [✅] Learnable parameters with L2 regularization and identity initialization
- [ ] Train calibration layer on validation subset of FindingEmo
- [ ] Run ablation study to validate effectiveness

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
    def __init__(self):
        # Learnable parameters (initialized to identity: no change)
        self.scale_v = nn.Parameter(torch.ones(1))   # Valence scaling
        self.scale_a = nn.Parameter(torch.ones(1))   # Arousal scaling  
        self.shift_v = nn.Parameter(torch.zeros(1))  # Valence offset
        self.shift_a = nn.Parameter(torch.zeros(1))  # Arousal offset
        
    def forward(self, v, a):
        v_out = v * self.scale_v + self.shift_v
        a_out = a * self.scale_a + self.shift_a
        return torch.tanh(v_out), torch.tanh(a_out)  # Smooth clamping
```

**Why 4 parameters?**
- **Scale factors**: Correct amplitude differences (e.g., EmoNet arousal 20% lower)
- **Shift factors**: Correct center-point bias (e.g., EmoNet 0.1 units more positive)
- **Minimal design**: Prevents overfitting with limited validation data
- **Identity initialization**: No change unless training finds systematic bias

## Training

Uses project's standard CCC + MSE loss (70/30 split) with L2 regularization:

```python
# Combined loss matching project conventions
ccc_loss = 2 - ccc(pred_v, true_v) - ccc(pred_a, true_a)
mse_loss = F.mse_loss(pred, target)
total_loss = 0.7 * ccc_loss + 0.3 * mse_loss + l2_reg

# L2 regularization prevents drift from identity
l2_reg = λ * [(scale_v - 1)² + (scale_a - 1)² + shift_v² + shift_a²]
```

**Training procedure:**
1. Use subset of FindingEmo where you have both EmoNet predictions and ground truth
2. Split into train/validation (80/20)
3. Early stopping when validation loss plateaus
4. Monitor parameters - if they stay near identity (1,1,0,0), remove layer

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
