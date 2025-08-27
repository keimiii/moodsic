import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy import stats
import matplotlib.pyplot as plt
from .cross_domain import CrossDomainCalibration
from .trainer import CalibrationTrainer

class CalibrationEvaluator:
    """
    Comprehensive evaluation and ablation testing for CrossDomainCalibration.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def ablation_study(
        self,
        source_pred: np.ndarray,
        target_labels: np.ndarray,
        n_runs: int = 5,
        val_split: float = 0.2
    ) -> Dict:
        """
        Run ablation study comparing with/without calibration.
        
        Args:
            source_pred: (N, 2) EmoNet predictions after scale alignment
            target_labels: (N, 2) FindingEmo ground truth
            n_runs: Number of random seed runs for statistical significance
            val_split: Validation split fraction
            
        Returns:
            Statistical comparison results
        """
        print("Running ablation study: WITH vs WITHOUT calibration")
        
        results_with = []
        results_without = []
        
        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}")
            
            # Set random seed for reproducible splits
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)
            
            # Split data
            indices = np.random.permutation(len(source_pred))
            n_val = int(len(indices) * val_split)
            train_idx, test_idx = indices[n_val:], indices[:n_val]
            
            # Test set for evaluation
            test_pred = source_pred[test_idx]
            test_labels = target_labels[test_idx]
            
            # WITHOUT calibration (baseline)
            metrics_without = self._evaluate_predictions(test_pred, test_labels)
            results_without.append(metrics_without)
            
            # WITH calibration
            # Train on train split
            train_pred = source_pred[train_idx]
            train_labels = target_labels[train_idx]
            
            calibration = CrossDomainCalibration().to(self.device)
            trainer = CalibrationTrainer(calibration)
            
            # Fit calibration parameters
            trainer.fit(train_pred, train_labels, val_split=0.2, max_epochs=50)
            
            # Test calibrated predictions
            calibration.eval()
            with torch.no_grad():
                test_pred_tensor = torch.tensor(test_pred, dtype=torch.float32)
                v_cal, a_cal = calibration(test_pred_tensor[:, 0], test_pred_tensor[:, 1])
                calibrated_pred = torch.stack([v_cal, a_cal], dim=1).numpy()
            
            metrics_with = self._evaluate_predictions(calibrated_pred, test_labels)
            results_with.append(metrics_with)
        
        return self._compute_statistical_comparison(results_without, results_with)
    
    def _evaluate_predictions(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute metrics for prediction array."""
        # CCC using existing project implementation
        def ccc(y_true, y_pred):
            mean_pred = np.mean(y_pred)
            mean_true = np.mean(y_true)
            covariance = np.mean((y_pred - mean_pred) * (y_true - mean_true))
            pred_var = np.var(y_pred)
            true_var = np.var(y_true)
            return (2 * covariance) / (pred_var + true_var + (mean_pred - mean_true)**2 + 1e-8)
        
        ccc_v = ccc(target[:, 0], pred[:, 0])
        ccc_a = ccc(target[:, 1], pred[:, 1])
        mae_v = np.mean(np.abs(pred[:, 0] - target[:, 0]))
        mae_a = np.mean(np.abs(pred[:, 1] - target[:, 1]))
        
        return {
            'ccc_v': ccc_v,
            'ccc_a': ccc_a,
            'ccc_avg': (ccc_v + ccc_a) / 2,
            'mae_v': mae_v,
            'mae_a': mae_a,
            'mae_avg': (mae_v + mae_a) / 2
        }
    
    def _compute_statistical_comparison(
        self, 
        results_without: List[Dict], 
        results_with: List[Dict]
    ) -> Dict:
        """Compute statistical significance of improvements."""
        metrics = ['ccc_v', 'ccc_a', 'ccc_avg', 'mae_v', 'mae_a', 'mae_avg']
        comparison = {}
        
        for metric in metrics:
            without_vals = [r[metric] for r in results_without]
            with_vals = [r[metric] for r in results_with]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(with_vals, without_vals)
            
            # Effect size (Cohen's d for paired samples)
            diff = np.array(with_vals) - np.array(without_vals)
            effect_size = np.mean(diff) / np.std(diff)
            
            comparison[metric] = {
                'without_mean': np.mean(without_vals),
                'without_std': np.std(without_vals),
                'with_mean': np.mean(with_vals),
                'with_std': np.std(with_vals),
                'improvement': np.mean(with_vals) - np.mean(without_vals),
                'p_value': p_value,
                'significant': p_value < 0.05,
                'effect_size': effect_size
            }
        
        return comparison
    
    def generalization_test(
        self,
        calibration: CrossDomainCalibration,
        unseen_pred: np.ndarray,
        unseen_labels: np.ndarray,
        dataset_name: str = "unseen"
    ) -> Dict:
        """
        Test calibration generalization on unseen dataset.
        
        Args:
            calibration: Trained calibration layer
            unseen_pred: Predictions on unseen dataset
            unseen_labels: Labels for unseen dataset
            dataset_name: Name for logging
            
        Returns:
            Generalization metrics
        """
        print(f"Testing generalization on {dataset_name} dataset")
        
        calibration.eval()
        
        # Baseline (no calibration)
        baseline_metrics = self._evaluate_predictions(unseen_pred, unseen_labels)
        
        # With calibration
        with torch.no_grad():
            pred_tensor = torch.tensor(unseen_pred, dtype=torch.float32)
            v_cal, a_cal = calibration(pred_tensor[:, 0], pred_tensor[:, 1])
            calibrated_pred = torch.stack([v_cal, a_cal], dim=1).numpy()
        
        calibrated_metrics = self._evaluate_predictions(calibrated_pred, unseen_labels)
        
        return {
            'dataset': dataset_name,
            'baseline': baseline_metrics,
            'calibrated': calibrated_metrics,
            'improvement': {
                k: calibrated_metrics[k] - baseline_metrics[k] 
                for k in baseline_metrics.keys()
            }
        }
    
    def plot_bland_altman(
        self,
        pred_before: np.ndarray,
        pred_after: np.ndarray,
        target: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Generate Bland-Altman plots to visualize bias reduction."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        dimensions = ['Valence', 'Arousal']
        
        for i, dim in enumerate(dimensions):
            # Before calibration
            diff_before = pred_before[:, i] - target[:, i]
            mean_before = (pred_before[:, i] + target[:, i]) / 2
            
            axes[i, 0].scatter(mean_before, diff_before, alpha=0.6, s=20)
            axes[i, 0].axhline(0, color='r', linestyle='--')
            axes[i, 0].axhline(np.mean(diff_before), color='b', linestyle='-', 
                              label=f'Bias: {np.mean(diff_before):.3f}')
            axes[i, 0].set_title(f'{dim} - Before Calibration')
            axes[i, 0].set_xlabel('Mean of Prediction and Target')
            axes[i, 0].set_ylabel('Difference (Pred - Target)')
            axes[i, 0].legend()
            
            # After calibration
            diff_after = pred_after[:, i] - target[:, i]
            mean_after = (pred_after[:, i] + target[:, i]) / 2
            
            axes[i, 1].scatter(mean_after, diff_after, alpha=0.6, s=20)
            axes[i, 1].axhline(0, color='r', linestyle='--')
            axes[i, 1].axhline(np.mean(diff_after), color='b', linestyle='-',
                              label=f'Bias: {np.mean(diff_after):.3f}')
            axes[i, 1].set_title(f'{dim} - After Calibration')
            axes[i, 1].set_xlabel('Mean of Prediction and Target')
            axes[i, 1].set_ylabel('Difference (Pred - Target)')
            axes[i, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
