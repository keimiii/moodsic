"""
Example usage of CrossDomainCalibration for EmoNet → FindingEmo alignment.
"""

import torch
import numpy as np
from models.calibration import CrossDomainCalibration, CalibrationTrainer, CalibrationEvaluator
from utils.emotion_pipeline import EmotionPipeline

def example_calibration_workflow():
    """Demonstrate complete calibration workflow."""
    
    # 1. Generate sample data (replace with real EmoNet outputs + FindingEmo labels)
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate EmoNet predictions (already in [-1, 1] after scale alignment)
    emonet_pred = np.random.randn(n_samples, 2) * 0.5  # Centered around 0
    
    # Simulate FindingEmo ground truth with systematic bias
    # (this is what calibration should learn to correct)
    bias_v, bias_a = 0.2, -0.1  # Systematic shifts
    scale_v, scale_a = 0.8, 1.2  # Scale differences
    
    findingemo_truth = np.column_stack([
        np.clip(emonet_pred[:, 0] * scale_v + bias_v + np.random.normal(0, 0.1, n_samples), -1, 1),
        np.clip(emonet_pred[:, 1] * scale_a + bias_a + np.random.normal(0, 0.1, n_samples), -1, 1)
    ])
    
    print("=== CrossDomainCalibration Example ===")
    print(f"Data: {n_samples} samples")
    print(f"True bias: v={bias_v:.2f}, a={bias_a:.2f}")
    print(f"True scale: v={scale_v:.2f}, a={scale_a:.2f}")
    
    # 2. Create and train calibration layer
    calibration = CrossDomainCalibration(l2_reg=1e-4, use_tanh=True)
    trainer = CalibrationTrainer(calibration, lr=0.01, patience=15)
    
    print("\n--- Training Calibration ---")
    results = trainer.fit(emonet_pred, findingemo_truth, val_split=0.2, max_epochs=100)
    
    print(f"\nTraining completed at epoch {results['converged_epoch']}")
    print(f"Final parameters: {results['final_params']}")
    
    # 3. Run ablation study
    evaluator = CalibrationEvaluator()
    
    print("\n--- Ablation Study ---")
    ablation_results = evaluator.ablation_study(emonet_pred, findingemo_truth, n_runs=3)
    
    for metric, stats in ablation_results.items():
        if stats['significant']:
            print(f"{metric}: {stats['improvement']:+.4f} "
                  f"(p={stats['p_value']:.3f}, d={stats['effect_size']:.2f}) ✓")
        else:
            print(f"{metric}: {stats['improvement']:+.4f} "
                  f"(p={stats['p_value']:.3f}, d={stats['effect_size']:.2f}) ✗")
    
    # 4. Test unified pipeline
    print("\n--- Pipeline Integration ---")
    pipeline = EmotionPipeline(calibration_layer=calibration, enable_calibration=True)
    
    # Test on a small batch
    test_v, test_a = emonet_pred[:10, 0], emonet_pred[:10, 1]
    
    # Without calibration
    v_ref_raw, a_ref_raw = pipeline.emonet_to_reference(test_v, test_a, apply_calibration=False)
    
    # With calibration
    v_ref_cal, a_ref_cal = pipeline.emonet_to_reference(test_v, test_a, apply_calibration=True)
    
    print(f"Sample predictions:")
    print(f"  Raw:        v={v_ref_raw[0]:.3f}, a={a_ref_raw[0]:.3f}")
    print(f"  Calibrated: v={v_ref_cal[0]:.3f}, a={a_ref_cal[0]:.3f}")
    print(f"  Target:     v={findingemo_truth[0, 0]:.3f}, a={findingemo_truth[0, 1]:.3f}")
    
    # 5. Check if calibration learned identity (should be removed if so)
    if calibration.is_near_identity(tolerance=0.05):
        print("\n⚠️  Calibration parameters near identity - consider removing this layer")
    else:
        print("\n✓ Calibration learned meaningful transformation")
    
    return pipeline, results, ablation_results

def test_scale_conversions():
    """Test that scale conversions work correctly with calibration."""
    pipeline = EmotionPipeline(enable_calibration=False)  # Test without calibration first
    
    # Test round-trip conversions
    v_original, a_original = 0.5, -0.3
    
    # EmoNet → FindingEmo → EmoNet (should be close to original)
    v_fe, a_fe = pipeline.scale_aligner.reference_to_findingemo(v_original, a_original)
    v_back, a_back = pipeline.scale_aligner.findingemo_to_reference(v_fe, a_fe)
    
    print(f"Round-trip test:")
    print(f"  Original: v={v_original:.3f}, a={a_original:.3f}")
    print(f"  FindingEmo: v={v_fe:.3f}, a={a_fe:.3f}")
    print(f"  Back: v={v_back:.3f}, a={a_back:.3f}")
    print(f"  Error: v={abs(v_original - v_back):.6f}, a={abs(a_original - a_back):.6f}")

if __name__ == "__main__":
    print("Testing scale conversions...")
    test_scale_conversions()
    
    print("\n" + "="*50)
    print("Running calibration example...")
    example_calibration_workflow()
