import torch
import numpy as np
from typing import Tuple, Optional, Union
from .emotion_scale_aligner import EmotionScaleAligner
from ..models.calibration import CrossDomainCalibration

ArrayLike = Union[float, int, np.ndarray, torch.Tensor]

class EmotionPipeline:
    """
    Unified interface for emotion prediction pipeline:
    Raw model output → Scale alignment → Domain calibration → Final (v, a)
    """
    
    def __init__(
        self,
        calibration_layer: Optional[CrossDomainCalibration] = None,
        enable_calibration: bool = True,
        strict_ranges: bool = False
    ):
        self.scale_aligner = EmotionScaleAligner(strict=strict_ranges)
        self.calibration = calibration_layer if enable_calibration else None
        self.enable_calibration = enable_calibration
        
    def emonet_to_reference(
        self, 
        v_emonet: ArrayLike, 
        a_emonet: ArrayLike,
        apply_calibration: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete pipeline: EmoNet → reference space → optional calibration.
        
        Args:
            v_emonet: EmoNet valence outputs (already in [-1, 1])
            a_emonet: EmoNet arousal outputs (already in [-1, 1])
            apply_calibration: Whether to apply domain calibration
            
        Returns:
            Final (valence, arousal) in reference space [-1, 1]
        """
        # Step 1: Scale alignment (EmoNet is already in [-1, 1])
        v_ref, a_ref = self.scale_aligner.emonet_to_reference(v_emonet, a_emonet)
        
        # Step 2: Optional domain calibration
        if apply_calibration and self.calibration is not None:
            v_tensor = torch.tensor(v_ref, dtype=torch.float32)
            a_tensor = torch.tensor(a_ref, dtype=torch.float32)
            
            self.calibration.eval()
            with torch.no_grad():
                v_cal, a_cal = self.calibration(v_tensor, a_tensor)
                v_ref = v_cal.numpy()
                a_ref = a_cal.numpy()
        
        return v_ref, a_ref
    
    def emonet_to_findingemo(
        self,
        v_emonet: ArrayLike,
        a_emonet: ArrayLike,
        apply_calibration: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete pipeline: EmoNet → calibrated reference → FindingEmo scale.
        """
        # Get calibrated reference values
        v_ref, a_ref = self.emonet_to_reference(v_emonet, a_emonet, apply_calibration)
        
        # Convert to FindingEmo scale
        return self.scale_aligner.reference_to_findingemo(v_ref, a_ref)
    
    def emonet_to_deam_static(
        self,
        v_emonet: ArrayLike,
        a_emonet: ArrayLike,
        apply_calibration: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete pipeline: EmoNet → calibrated reference → DEAM static scale.
        """
        # Get calibrated reference values
        v_ref, a_ref = self.emonet_to_reference(v_emonet, a_emonet, apply_calibration)
        
        # Convert to DEAM static scale
        return self.scale_aligner.reference_to_deam_static(v_ref, a_ref)
    
    def load_calibration(self, checkpoint_path: str):
        """Load trained calibration parameters from checkpoint."""
        if self.calibration is None:
            self.calibration = CrossDomainCalibration()
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.calibration.load_state_dict(checkpoint)
        print(f"Loaded calibration from {checkpoint_path}")
        
        params = self.calibration.get_params_summary()
        print(f"Calibration params: {params}")
    
    def save_calibration(self, checkpoint_path: str):
        """Save trained calibration parameters."""
        if self.calibration is None:
            raise ValueError("No calibration layer to save")
        
        torch.save(self.calibration.state_dict(), checkpoint_path)
        print(f"Saved calibration to {checkpoint_path}")
    
    def compare_predictions(
        self,
        source_pred: np.ndarray,
        target_labels: np.ndarray,
        output_scale: str = 'reference'
    ) -> Dict:
        """
        Compare predictions with and without calibration.
        
        Args:
            source_pred: EmoNet predictions after scale alignment
            target_labels: Ground truth labels in target scale
            output_scale: 'reference', 'findingemo', or 'deam_static'
            
        Returns:
            Comparison metrics and visualizations
        """
        if self.calibration is None:
            raise ValueError("No calibration layer loaded")
        
        # Predictions without calibration
        if output_scale == 'reference':
            pred_without = source_pred.copy()
        elif output_scale == 'findingemo':
            pred_without = np.column_stack(
                self.scale_aligner.reference_to_findingemo(source_pred[:, 0], source_pred[:, 1])
            )
        elif output_scale == 'deam_static':
            pred_without = np.column_stack(
                self.scale_aligner.reference_to_deam_static(source_pred[:, 0], source_pred[:, 1])
            )
        
        # Predictions with calibration
        if output_scale == 'reference':
            pred_with = np.column_stack(
                self.emonet_to_reference(source_pred[:, 0], source_pred[:, 1])
            )
        elif output_scale == 'findingemo':
            pred_with = np.column_stack(
                self.emonet_to_findingemo(source_pred[:, 0], source_pred[:, 1])
            )
        elif output_scale == 'deam_static':
            pred_with = np.column_stack(
                self.emonet_to_deam_static(source_pred[:, 0], source_pred[:, 1])
            )
        
        # Evaluate both
        from .calibration.evaluation import CalibrationEvaluator
        evaluator = CalibrationEvaluator()
        
        metrics_without = evaluator._evaluate_predictions(pred_without, target_labels)
        metrics_with = evaluator._evaluate_predictions(pred_with, target_labels)
        
        return {
            'without_calibration': metrics_without,
            'with_calibration': metrics_with,
            'improvement': {
                k: metrics_with[k] - metrics_without[k] 
                for k in metrics_without.keys()
            },
            'calibration_params': self.calibration.get_params_summary() if self.calibration else None
        }
