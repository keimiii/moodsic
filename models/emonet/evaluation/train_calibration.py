"""
Train cross-domain calibration for EmoNet → FindingEmo alignment.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from models.calibration import CrossDomainCalibration, CalibrationTrainer, CalibrationEvaluator
from utils.emotion_scale_aligner import EmotionScaleAligner


def load_emonet_predictions(results_dir: Path):
    """Load EmoNet predictions and ground truth from evaluation results."""
    pred_file = results_dir / "per_image_preds.csv"
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    
    df = pd.read_csv(pred_file)
    print(f"Loaded {len(df)} total predictions")
    
    # Filter to face-detected samples only
    face_df = df[df["has_face"] == True].copy()
    print(f"Face-detected samples: {len(face_df)} ({len(face_df)/len(df)*100:.1f}%)")
    
    if len(face_df) == 0:
        raise ValueError("No face-detected samples found for calibration training")
    
    # Extract predictions and ground truth
    emonet_preds = face_df[["v_pred", "a_pred"]].values
    findingemo_gt = face_df[["v_gt", "a_gt"]].values
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(emonet_preds).any(axis=1) | np.isnan(findingemo_gt).any(axis=1))
    emonet_preds = emonet_preds[valid_mask]
    findingemo_gt = findingemo_gt[valid_mask]
    
    print(f"Valid samples after NaN removal: {len(emonet_preds)}")
    
    # CRITICAL FIX: Convert FindingEmo ground truth to reference space [-1, 1]
    # so both input and target are in the same scale during calibration training
    aligner = EmotionScaleAligner()
    v_ref_gt, a_ref_gt = aligner.findingemo_to_reference(findingemo_gt[:, 0], findingemo_gt[:, 1])
    findingemo_ref = np.stack([v_ref_gt, a_ref_gt], axis=1)
    
    print(f"Ground truth converted from FindingEmo to reference space [-1, 1]")
    
    return emonet_preds, findingemo_ref


def train_calibration():
    """Train calibration model on EmoNet predictions."""
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load evaluation results
    results_dir = Path(__file__).parent / "results"
    emonet_preds, findingemo_ref = load_emonet_predictions(results_dir)
    
    print("\n=== Data Summary ===")
    print(f"EmoNet predictions shape: {emonet_preds.shape}")
    print(f"FindingEmo reference ground truth shape: {findingemo_ref.shape}")
    print(f"EmoNet - Valence range: [{emonet_preds[:, 0].min():.3f}, {emonet_preds[:, 0].max():.3f}]")
    print(f"EmoNet - Arousal range: [{emonet_preds[:, 1].min():.3f}, {emonet_preds[:, 1].max():.3f}]")
    print(f"Reference GT - Valence range: [{findingemo_ref[:, 0].min():.3f}, {findingemo_ref[:, 0].max():.3f}]")
    print(f"Reference GT - Arousal range: [{findingemo_ref[:, 1].min():.3f}, {findingemo_ref[:, 1].max():.3f}]")
    
    # Create calibration model
    calibration = CrossDomainCalibration(l2_reg=1e-4, use_tanh=True)
    trainer = CalibrationTrainer(calibration, lr=0.01, patience=20)
    
    print("\n=== Training Calibration ===")
    results = trainer.fit(emonet_preds, findingemo_ref, val_split=0.2, max_epochs=100)
    
    print(f"Training completed at epoch {results['converged_epoch']}")
    print(f"Final parameters: {results['final_params']}")
    
    # Save trained calibration
    calibration_path = results_dir / "calibration_emonet2findingemo.pt"
    torch.save(calibration.state_dict(), calibration_path)
    print(f"Calibration saved to: {calibration_path}")
    
    # Run ablation study
    print("\n=== Ablation Study ===")
    evaluator = CalibrationEvaluator()
    ablation_results = evaluator.ablation_study(emonet_preds, findingemo_ref, n_runs=3)
    
    print("Performance improvements:")
    for metric, stats in ablation_results.items():
        significance = "✓" if stats['significant'] else "✗"
        print(f"  {metric}: {stats['improvement']:+.4f} "
              f"(p={stats['p_value']:.3f}, d={stats['effect_size']:.2f}) {significance}")
    
    # Check if calibration learned meaningful transformation
    if calibration.is_near_identity(tolerance=0.05):
        print("\n⚠️  Calibration parameters near identity - minimal effect expected")
    else:
        print("\n✓ Calibration learned meaningful transformation")
    
    return calibration_path, results, ablation_results


if __name__ == "__main__":
    try:
        calibration_path, results, ablation_results = train_calibration()
        print(f"\n🎯 Calibration training complete!")
        print(f"Next step: Run evaluation with --use-calibration flag")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
