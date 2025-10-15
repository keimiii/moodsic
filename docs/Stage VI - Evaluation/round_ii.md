# Evaluation Round II: Cluster Distribution Analysis

**Date**: 2025-10-15  
**Status**: 🔴 Issue Identified - Predictions Collapsed to 2/5 Clusters  
**Priority**: High - Requires immediate attention before deployment

---

## Executive Summary

Analysis of the VEATIC inference results reveals that **only 2 out of 5 music clusters** are being utilized. 78% of videos are assigned to cluster 3 (negative valence, neutral arousal), while 22% go to cluster 2 (slightly positive valence, low arousal). Clusters 0, 1, and 4 receive no assignments, indicating a systematic bias in the emotion prediction pipeline towards negative/neutral emotional states.

---

## Key Findings

### 1. Cluster Assignment Distribution

**Dataset**: [`results/inference/pipeline_results_20251006_144126_enriched.parquet`](../../results/inference/pipeline_results_20251006_144126_enriched.parquet)

| Cluster ID | Inference Entries | Percentage | Valence | Arousal | Quadrant |
|------------|-------------------|------------|---------|---------|----------|
| **0** | 0 | 0% | +0.367 | +0.335 | Happy/Energetic |
| **1** | 0 | 0% | -0.309 | -0.434 | Sad/Low Energy |
| **2** | 54 | 21.77% | +0.113 | +0.142 | Neutral/Slightly Positive ✅ |
| **3** | 194 | 78.23% | -0.202 | -0.006 | Neutral/Sad ✅ |
| **4** | 0 | 0% | +0.007 | -0.242 | Calm/Neutral |

**Total inference rows**: 248 (124 VEATIC videos exported twice: unstabilized + stabilized)

**Cluster metadata**: [`results/clustering/deam_gmm_20251006_151857/clusters_meta.json`](../../results/clustering/deam_gmm_20251006_151857/clusters_meta.json)

### 2. Fused Prediction Characteristics

**Valence-Arousal Statistics from Inference Results**:

```
Valence Range:   [-0.389, +0.396]
Arousal Range:   [-0.068, +0.309]
Mean Valence:    -0.113 (negative skew)
Mean Arousal:    +0.064 (slightly positive)
```

**By Cluster**:

| Cluster | Mean Valence | Min Valence | Max Valence | Mean Arousal | Min Arousal | Max Arousal |
|---------|--------------|-------------|-------------|--------------|-------------|-------------|
| 2 | +0.050 | -0.103 | +0.396 | +0.142 | -0.068 | +0.228 |
| 3 | -0.158 | -0.389 | +0.395 | -0.006 | -0.068 | +0.309 |

---

## Root Cause Analysis

### 1. Face Path Drives Negative Valence

The face stream outputs are **strongly negative** (mean valence ≈ -0.19, only 2.4% > 0), while the scene stream is mildly positive on average (mean valence ≈ +0.03). Because both passes (stabilized + unstabilized) agree on cluster IDs, the face predictions are pulling the fusion into clusters 2 and 3.

### 2. Fusion Weighting Amplifies the Bias

**Current settings** ([`results/evaluation/veatic_run_params_20251006_144126.json`](../../results/evaluation/veatic_run_params_20251006_144126.json)):
```json
{
  "SCENE_WEIGHT": 0.6,
  "FACE_WEIGHT": 0.4,
  "USE_VARIANCE_WEIGHTING": true,
  "ENABLE_STABILIZER": false
}
```

Inverse-variance fusion gives extra emphasis to the (over-confident) face stream, so the effective weights skew even more negative than 60/40.

### 3. Compressed Valence Range vs. Ground Truth

**VEATIC Ground Truth Distribution** (from [`notebooks/Dataset - VEATIC/VEATIC_eda.ipynb`](../../notebooks/Dataset%20-%20VEATIC/VEATIC_eda.ipynb#L2363-L2364)):
- Overall mean valence: **-0.1273** (slightly negative, but broad coverage)
- Overall mean arousal: **+0.1406**
- Videos in "Excited" quadrant (high V, high A): **28/124 (22.6%)**
- Valence range: **[-0.915, 0.886]** (much broader than predictions)
- Arousal range: **[-0.646, 0.889]** (much broader than predictions)

**Implication**:
- **Ground truth**: Broad emotional range with mild negative skew
- **Model predictions**: Compressed towards neutral/negative with only 14% positive valence (≈ 50% shortfall vs. VEATIC)

### 4. Potential Contributing Factors

#### Scene Pathway
- **Dropout rate**: 0.15 (already reduced vs. base config; further lowering may trade robustness for range)
- **Training loss weights**: Valence weight = 1.0 (equal to arousal; no prioritization)
- **Ambiguity weighting**: Factor 0.5 down-weights uncertain samples (may bias towards neutral)

#### Face Pathway
- **EmoNet pre-trained model**: No domain adaptation to VEATIC-style videos
- **Coverage**: High (≈ 0.86), so systematic negative bias is unlikely to be due to gating

#### Fusion Layer
- **No calibration**: No learned bias correction to adjust for domain shift
- **Variance weighting**: Currently over-trusting the biased face pathway

---

## Proposed Solutions

### Tier 1: Immediate Fixes (No Retraining Required)

#### A. Rebalance Fusion Toward Scene Path

**Implementation**: Adjust fusion configuration to reduce face dominance.

**Files**:
- [`models/fusion.py`](../../models/fusion.py#L94-L205)
- [`scripts/evaluation/run_inference_pipeline.py`](../../scripts/evaluation/run_inference_pipeline.py)

**Parameters to test**:
- Disable variance weighting (`use_variance_weighting=False`)
- Increase scene weight to 0.7–0.8 and/or drop face weight to 0.2–0.3

**Command examples**:
```bash
python scripts/evaluation/run_inference_pipeline.py \
  --scene-weight 0.8 \
  --face-weight 0.2 \
  --no-variance-weighting \
  --output-dir results/inference/round_ii_scene80_face20
```

**Expected impact**: Allows the already less-biased scene stream to seed clusters beyond 2/3 without rerunning the expensive scene/face inference.

#### B. Apply VEATIC-Specific Bias Offset (if Needed)

**Context**: Past attempts to align EmoNet to full-scene annotations with a learnable affine layer (see [`models/emonet/evaluation/train_calibration.py`](../../models/emonet/evaluation/train_calibration.py)) degraded performance even after aggressive parameter shifts (`results/calibration_emonet2findingemo.pt`). This ruled out cross-domain unification.

**Implementation**: Limit calibration to a simple bias correction on fused VEATIC predictions (e.g., grid search over `valence_shift` in `[-0.1, +0.2]`), avoiding full affine re-training.

**Expected impact**: Provides a reversible tweak for the current dataset while respecting prior evidence that more complex calibration harms generalization.

#### C. Inspect Face Path Predictions Directly

**Implementation**: Use cached payloads in `results/inference/pipeline_results_20251006_144126.parquet` to chart per-frame face outputs.

**Goal**:
- Identify whether camera angles, lighting, or domain mismatch drive negative predictions.
- Use findings to prioritize fine-tuning or adjust face weighting per scene type.

---

### Tier 2: Medium-Term Fixes (Requires Retraining)

#### A. Retrain Scene Model with Valence Prioritization

**File**: [`src/training/losses.py`](../../src/training/losses.py)

**Current loss weights**:
```python
loss_weights.valence = 1.0
loss_weights.arousal = 1.0
```

**Proposed**:
```python
loss_weights.valence = 1.5 to 2.0  # Prioritize valence accuracy
loss_weights.arousal = 1.0
```

**Training script**: [`scripts/train_scene_model.py`](../../scripts/train_scene_model.py)

#### B. Adjust Scene Model Dropout

**File**: [`configs/scene_models/scene_model_clip_vit_b32_frozen_auto_lr_config.yaml`](../../configs/scene_models/scene_model_clip_vit_b32_frozen_auto_lr_config.yaml)

**Current**: `dropout_rate: 0.15`  
**Proposed**: Experiment with `0.05 – 0.1` to expand the valence range while monitoring overfitting

#### C. Fine-Tune Face Expert on VEATIC-Like Clips

**Files**:
- [`models/emonet`](../../models/emonet)
- [`scripts/training`](../../scripts)

**Goals**:
1. Collect VEATIC face crops with balanced quadrants.
2. Fine-tune the face regression head to reduce systematic negative bias.
3. Re-evaluate fusion once the face path no longer drags valence downward.

---

### Tier 3: Long-Term Improvements (Research-Level)

1. **Domain-adaptive fine-tuning**: Fine-tune scene model on VEATIC training split
2. **Learned fusion weights**: Train fusion weights end-to-end on VEATIC
3. **Multi-modal attention**: Replace fixed fusion with learned attention mechanism
4. **Temporal modeling**: Add LSTM/Transformer to capture temporal emotion dynamics

---

## Next Steps

### Top 3 Short-Term Actions

1. **Re-run fusion with reduced face weight / no variance weighting**  
   Lets the less-biased scene stream steer valence without redoing model inference; quickest path to unlocking clusters 0 and 4.

2. **Audit face pathway outputs from cached payloads**  
   Confirms whether the bias is global or tied to specific lighting/pose subsets, informing whether to fine-tune or selectively down-weight faces.

3. **Test a lightweight VEATIC bias offset only if weighting changes fall short**  
   Respects the earlier negative calibration results while giving us a last-mile lever to nudge valence without touching the core models.

### Success Metrics

**Target cluster distribution** (based on VEATIC ground truth):
- Cluster 0 (Happy): 15-25% (currently 0%)
- Cluster 1 (Sad/Low): 10-20% (currently 0%)
- Cluster 2 (Neutral+): 20-30% (currently 22%)
- Cluster 3 (Neutral-): 30-40% (currently 78%)
- Cluster 4 (Calm): 10-20% (currently 0%)

**Minimum acceptable**: At least 4 out of 5 clusters should have >5% representation

---

## References

### Datasets
- **VEATIC inference results**: [`results/inference/pipeline_results_20251006_144126_enriched.parquet`](../../results/inference/pipeline_results_20251006_144126_enriched.parquet)
- **DEAM cluster metadata**: [`results/clustering/deam_gmm_20251006_151857/clusters_meta.json`](../../results/clustering/deam_gmm_20251006_151857/clusters_meta.json)
- **VEATIC EDA**: [`notebooks/Dataset - VEATIC/VEATIC_eda.ipynb`](../../notebooks/Dataset%20-%20VEATIC/VEATIC_eda.ipynb)

### Code Files
- **Fusion module**: [`models/fusion.py`](../../models/fusion.py)
- **Calibration layer**: [`models/calibration/cross_domain.py`](../../models/calibration/cross_domain.py)
- **Training losses**: [`src/training/losses.py`](../../src/training/losses.py)
- **Inference pipeline**: [`scripts/evaluation/run_inference_pipeline.py`](../../scripts/evaluation/run_inference_pipeline.py)
- **Fusion tuning**: [`scripts/fusion_threshold_tuning.py`](../../scripts/fusion_threshold_tuning.py)

### Documentation
- **Round I evaluation**: [`docs/Stage VI - Evaluation/eval_res_round_i.md`](eval_res_round_i.md)
- **Fusion strategy**: [`docs/Stage II — Modeling/fusion-strategy.md`](../Stage%20II%20%E2%80%94%20Modeling/fusion-strategy.md)
- **Uncertainty & gating**: [`docs/Stage II — Modeling/uncertainty-and-gating.md`](../Stage%20II%20%E2%80%94%20Modeling/uncertainty-and-gating.md)
- **Hyperparameters**: [`docs/Stage III — Training/hyperparameters.md`](../Stage%20III%20%E2%80%94%20Training/hyperparameters.md)

### External Resources
- **VEATIC paper**: [arXiv:2309.06745](https://arxiv.org/abs/2309.06745)
- **VEATIC project page**: [veatic.github.io](https://veatic.github.io/)
- **VEATIC GitHub**: [github.com/AlbusPeter/VEATIC](https://github.com/AlbusPeter/VEATIC)

---

**Last Updated**: 2025-10-15  
**Author**: Evaluation Team  
**Reviewers**: TBD
