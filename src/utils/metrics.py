"""
Evaluation metrics for continuous Valence-Arousal prediction and Emo8 classification.
Implements CCC, RMSE, MAE, Pearson/Spearman correlation for V-A evaluation,
and Weighted-F1, AP, macro-F1 for Emo8 classification as per research findings.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    f1_score, average_precision_score, classification_report,
    confusion_matrix, accuracy_score, precision_recall_fscore_support
)
import logging

logger = logging.getLogger(__name__)


def concordance_correlation_coefficient(y_true: Union[torch.Tensor, np.ndarray], 
                                     y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Concordance Correlation Coefficient (CCC).
    
    CCC measures the agreement between two continuous variables.
    It combines measures of both precision and accuracy to determine
    how far the observed data deviates from the line of perfect concordance.
    
    Formula: CCC = (2 * ρ * σ_x * σ_y) / (σ_x^2 + σ_y^2 + (μ_x - μ_y)^2)
    
    Where:
    - ρ is the Pearson correlation coefficient
    - σ_x, σ_y are standard deviations
    - μ_x, μ_y are means
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        CCC value between -1 and 1 (1 = perfect agreement)
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        logger.warning("No valid values for CCC calculation")
        return 0.0
    
    # Calculate means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    # Calculate variances and covariance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    # Calculate CCC
    numerator = 2 * cov
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    
    if denominator == 0:
        return 0.0
    
    ccc = numerator / denominator
    return float(ccc)


def mean_square_error(y_true: Union[torch.Tensor, np.ndarray], 
                      y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Mean Square Error (MSE).
    
    MSE = mean((y_true - y_pred)^2)
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MSE value (lower is better, 0 = perfect)
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        logger.warning("No valid values for MSE calculation")
        return float('inf')
    
    mse = np.mean((y_true - y_pred) ** 2)
    return float(mse)


def root_mean_square_error(y_true: Union[torch.Tensor, np.ndarray], 
                           y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Root Mean Square Error (RMSE).
    
    RMSE = sqrt(mean((y_true - y_pred)^2))
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        RMSE value (lower is better, 0 = perfect)
    """
    # Use MSE function for consistency
    mse = mean_square_error(y_true, y_pred)
    if mse == float('inf'):
        return float('inf')
    
    rmse = np.sqrt(mse)
    return float(rmse)


def mean_absolute_error(y_true: Union[torch.Tensor, np.ndarray], 
                       y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    MAE = mean(|y_true - y_pred|)
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MAE value (lower is better, 0 = perfect)
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        logger.warning("No valid values for MAE calculation")
        return float('inf')
    
    mae = np.mean(np.abs(y_true - y_pred))
    return float(mae)


def pearson_correlation(y_true: Union[torch.Tensor, np.ndarray], 
                       y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Pearson correlation coefficient.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Pearson correlation coefficient between -1 and 1
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < 2:
        logger.warning("Insufficient valid values for Pearson correlation")
        return 0.0
    
    # Check for constant arrays (zero variance)
    if np.var(y_true) == 0 or np.var(y_pred) == 0:
        logger.warning("Constant input detected - correlation undefined")
        return 0.0
    
    try:
        corr, _ = pearsonr(y_true, y_pred)
        return float(corr) if not np.isnan(corr) else 0.0
    except Exception as e:
        logger.warning(f"Error calculating Pearson correlation: {e}")
        return 0.0


def spearman_correlation(y_true: Union[torch.Tensor, np.ndarray], 
                        y_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Spearman rank correlation coefficient.
    
    Research findings show Spearman-r is important for V-A evaluation
    as it captures monotonic relationships better than Pearson.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Spearman correlation coefficient between -1 and 1
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < 2:
        logger.warning("Insufficient valid values for Spearman correlation")
        return 0.0
    
    # Check for constant arrays (zero variance)
    if np.var(y_true) == 0 or np.var(y_pred) == 0:
        logger.warning("Constant input detected - correlation undefined")
        return 0.0
    
    try:
        corr, _ = spearmanr(y_true, y_pred)
        return float(corr) if not np.isnan(corr) else 0.0
    except Exception as e:
        logger.warning(f"Error calculating Spearman correlation: {e}")
        return 0.0


def emo8_classification_metrics(y_true: Union[torch.Tensor, np.ndarray],
                               y_pred: Union[torch.Tensor, np.ndarray],
                               class_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics for Emo8.
    
    Research findings emphasize:
    - Weighted-F1 score (handles class imbalance)
    - Average Precision (AP) 
    - Macro-F1 (treats all classes equally)
    - Per-class analysis for skewed classes (Joy/Anticipation vs Surprise/Disgust)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (not probabilities)
        class_names: Optional class names for detailed reporting
        
    Returns:
        Dictionary of classification metrics
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Flatten arrays
    y_true = y_true.flatten().astype(int)
    y_pred = y_pred.flatten().astype(int)
    
    # Default Emo8 class names
    if class_names is None:
        class_names = ['joy', 'anticipation', 'anger', 'fear', 
                      'sadness', 'disgust', 'trust', 'surprise']
    
    results = {}
    
    try:
        # Basic accuracy
        results['accuracy'] = accuracy_score(y_true, y_pred)
        
        # F1 scores
        results['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
        results['macro_f1'] = f1_score(y_true, y_pred, average='macro')
        results['micro_f1'] = f1_score(y_true, y_pred, average='micro')
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        for i, class_name in enumerate(class_names):
            if i < len(precision):
                results[f'{class_name}_precision'] = float(precision[i])
                results[f'{class_name}_recall'] = float(recall[i])
                results[f'{class_name}_f1'] = float(f1[i])
                results[f'{class_name}_support'] = int(support[i])
        
        # Class imbalance analysis (research findings highlight this)
        unique_classes, class_counts = np.unique(y_true, return_counts=True)
        total_samples = len(y_true)
        
        for i, class_idx in enumerate(unique_classes):
            if class_idx < len(class_names):
                class_name = class_names[class_idx]
                results[f'{class_name}_frequency'] = float(class_counts[i] / total_samples)
        
        # Confusion matrix based metrics
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class accuracy from confusion matrix
        for i, class_name in enumerate(class_names):
            if i < cm.shape[0]:
                if cm[i, :].sum() > 0:  # Avoid division by zero
                    class_accuracy = cm[i, i] / cm[i, :].sum()
                    results[f'{class_name}_accuracy'] = float(class_accuracy)
        
        logger.debug(f"📊 Computed Emo8 metrics: accuracy={results['accuracy']:.3f}, "
                    f"weighted_f1={results['weighted_f1']:.3f}, macro_f1={results['macro_f1']:.3f}")
        
    except Exception as e:
        logger.warning(f"Error computing Emo8 metrics: {e}")
        # Return default values
        results = {
            'accuracy': 0.0,
            'weighted_f1': 0.0,
            'macro_f1': 0.0,
            'micro_f1': 0.0
        }
    
    return results


def emo8_average_precision(y_true: Union[torch.Tensor, np.ndarray],
                          y_probs: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """
    Calculate Average Precision (AP) for Emo8 classification.
    
    Research findings emphasize AP as key metric for imbalanced classes.
    
    Args:
        y_true: Ground truth labels [batch_size]
        y_probs: Predicted probabilities [batch_size, num_classes]
        
    Returns:
        Dictionary with AP scores per class and mean AP
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()
    
    y_true = y_true.flatten().astype(int)
    
    # Convert to one-hot for AP calculation
    num_classes = y_probs.shape[1] if y_probs.ndim > 1 else 8
    y_true_onehot = np.eye(num_classes)[y_true]
    
    class_names = ['joy', 'anticipation', 'anger', 'fear', 
                   'sadness', 'disgust', 'trust', 'surprise']
    
    results = {}
    ap_scores = []
    
    try:
        for i, class_name in enumerate(class_names):
            if i < num_classes and i < y_true_onehot.shape[1]:
                ap = average_precision_score(y_true_onehot[:, i], y_probs[:, i])
                results[f'{class_name}_ap'] = float(ap)
                ap_scores.append(ap)
        
        # Mean AP (mAP)
        results['mean_ap'] = float(np.mean(ap_scores)) if ap_scores else 0.0
        
    except Exception as e:
        logger.warning(f"Error computing AP scores: {e}")
        results = {'mean_ap': 0.0}
    
    return results


class VAMetrics:
    """
    Comprehensive metrics calculator for Valence-Arousal prediction.
    
    Computes multiple metrics and supports per-quadrant analysis.
    """
    
    def __init__(self, metrics: List[str] = None, compute_per_quadrant: bool = True):
        """
        Initialize metrics calculator.
        
        Args:
            metrics: List of metrics to compute ['ccc', 'mse', 'rmse', 'mae', 'pearson', 'spearman']
            compute_per_quadrant: Whether to compute metrics per V-A quadrant
        """
        if metrics is None:
            # Research findings emphasize MAE and Spearman-r for V-A evaluation
            # Include MSE for comprehensive error analysis
            metrics = ['ccc', 'mse', 'rmse', 'mae', 'pearson', 'spearman']
        
        self.metrics = metrics
        self.compute_per_quadrant = compute_per_quadrant
        
        # Available metric functions
        self.metric_functions = {
            'ccc': concordance_correlation_coefficient,
            'mse': mean_square_error,
            'rmse': root_mean_square_error,
            'mae': mean_absolute_error,
            'pearson': pearson_correlation,
            'spearman': spearman_correlation
        }
        
        # Validate metrics
        for metric in self.metrics:
            if metric not in self.metric_functions:
                raise ValueError(f"Unknown metric: {metric}")
    
    def compute_metrics(self, 
                       valence_true: Union[torch.Tensor, np.ndarray],
                       valence_pred: Union[torch.Tensor, np.ndarray],
                       arousal_true: Union[torch.Tensor, np.ndarray],
                       arousal_pred: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
        """
        Compute all specified metrics for valence and arousal.
        
        Args:
            valence_true: Ground truth valence values
            valence_pred: Predicted valence values
            arousal_true: Ground truth arousal values
            arousal_pred: Predicted arousal values
            
        Returns:
            Dictionary containing all computed metrics
        """
        results = {}
        
        # Compute metrics for valence
        for metric_name in self.metrics:
            metric_fn = self.metric_functions[metric_name]
            results[f'valence_{metric_name}'] = metric_fn(valence_true, valence_pred)
        
        # Compute metrics for arousal
        for metric_name in self.metrics:
            metric_fn = self.metric_functions[metric_name]
            results[f'arousal_{metric_name}'] = metric_fn(arousal_true, arousal_pred)
        
        # Compute average metrics
        for metric_name in self.metrics:
            val_metric = results[f'valence_{metric_name}']
            ar_metric = results[f'arousal_{metric_name}']
            
            # For correlation metrics, average directly
            if metric_name in ['ccc', 'pearson']:
                results[f'{metric_name}_avg'] = (val_metric + ar_metric) / 2
            # For error metrics, use quadratic mean (RMS of errors)
            else:  # mse, rmse, mae
                results[f'{metric_name}_avg'] = np.sqrt((val_metric**2 + ar_metric**2) / 2)
        
        # Compute per-quadrant metrics if requested
        if self.compute_per_quadrant:
            quadrant_metrics = self._compute_quadrant_metrics(
                valence_true, valence_pred, arousal_true, arousal_pred
            )
            results.update(quadrant_metrics)
        
        return results
    
    def _compute_quadrant_metrics(self,
                                 valence_true: Union[torch.Tensor, np.ndarray],
                                 valence_pred: Union[torch.Tensor, np.ndarray],
                                 arousal_true: Union[torch.Tensor, np.ndarray],
                                 arousal_pred: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
        """
        Compute metrics per V-A quadrant.
        
        Quadrants:
        - Q1: High Valence, High Arousal (Happy/Excited)
        - Q2: Low Valence, High Arousal (Angry/Stressed)  
        - Q3: Low Valence, Low Arousal (Sad/Depressed)
        - Q4: High Valence, Low Arousal (Calm/Content)
        """
        # Convert to numpy
        if isinstance(valence_true, torch.Tensor):
            valence_true = valence_true.detach().cpu().numpy()
        if isinstance(valence_pred, torch.Tensor):
            valence_pred = valence_pred.detach().cpu().numpy()
        if isinstance(arousal_true, torch.Tensor):
            arousal_true = arousal_true.detach().cpu().numpy()
        if isinstance(arousal_pred, torch.Tensor):
            arousal_pred = arousal_pred.detach().cpu().numpy()
        
        # Flatten arrays
        valence_true = valence_true.flatten()
        valence_pred = valence_pred.flatten()
        arousal_true = arousal_true.flatten()
        arousal_pred = arousal_pred.flatten()
        
        # Define quadrants based on ground truth
        q1_mask = (valence_true > 0) & (arousal_true > 0)  # Happy/Excited
        q2_mask = (valence_true <= 0) & (arousal_true > 0)  # Angry/Stressed
        q3_mask = (valence_true <= 0) & (arousal_true <= 0)  # Sad/Depressed
        q4_mask = (valence_true > 0) & (arousal_true <= 0)  # Calm/Content
        
        quadrants = {
            'q1_happy': q1_mask,
            'q2_angry': q2_mask,
            'q3_sad': q3_mask,
            'q4_calm': q4_mask
        }
        
        results = {}
        
        for quad_name, mask in quadrants.items():
            if np.sum(mask) == 0:
                # No samples in this quadrant
                for metric_name in self.metrics:
                    results[f'{quad_name}_{metric_name}_avg'] = 0.0
                results[f'{quad_name}_count'] = 0
                continue
            
            # Extract samples for this quadrant
            v_true_quad = valence_true[mask]
            v_pred_quad = valence_pred[mask]
            a_true_quad = arousal_true[mask]
            a_pred_quad = arousal_pred[mask]
            
            # Compute metrics for this quadrant
            quad_results = {}
            for metric_name in self.metrics:
                metric_fn = self.metric_functions[metric_name]
                v_metric = metric_fn(v_true_quad, v_pred_quad)
                a_metric = metric_fn(a_true_quad, a_pred_quad)
                
                # Average valence and arousal metrics
                if metric_name in ['ccc', 'pearson']:
                    quad_results[f'{quad_name}_{metric_name}_avg'] = (v_metric + a_metric) / 2
                else:
                    quad_results[f'{quad_name}_{metric_name}_avg'] = np.sqrt((v_metric**2 + a_metric**2) / 2)
            
            quad_results[f'{quad_name}_count'] = int(np.sum(mask))
            results.update(quad_results)
        
        return results
    
    def format_results(self, results: Dict[str, float], precision: int = 4) -> str:
        """
        Format metrics results for display.
        
        Args:
            results: Dictionary of computed metrics
            precision: Number of decimal places
            
        Returns:
            Formatted string representation
        """
        lines = []
        lines.append("📊 Valence-Arousal Metrics")
        lines.append("=" * 40)
        
        # Overall metrics
        lines.append("\n🎯 Overall Performance:")
        for metric in self.metrics:
            if f'{metric}_avg' in results:
                lines.append(f"  {metric.upper()}: {results[f'{metric}_avg']:.{precision}f}")
        
        # Individual dimension metrics
        lines.append("\n📈 Valence Metrics:")
        for metric in self.metrics:
            key = f'valence_{metric}'
            if key in results:
                lines.append(f"  {metric.upper()}: {results[key]:.{precision}f}")
        
        lines.append("\n📉 Arousal Metrics:")
        for metric in self.metrics:
            key = f'arousal_{metric}'
            if key in results:
                lines.append(f"  {metric.upper()}: {results[key]:.{precision}f}")
        
        # Quadrant metrics
        if self.compute_per_quadrant:
            lines.append("\n🎭 Per-Quadrant Performance:")
            quadrants = [
                ('q1_happy', 'Happy/Excited'),
                ('q2_angry', 'Angry/Stressed'),
                ('q3_sad', 'Sad/Depressed'),
                ('q4_calm', 'Calm/Content')
            ]
            
            for quad_key, quad_name in quadrants:
                count_key = f'{quad_key}_count'
                if count_key in results and results[count_key] > 0:
                    lines.append(f"\n  {quad_name} (n={results[count_key]}):")
                    for metric in self.metrics:
                        key = f'{quad_key}_{metric}_avg'
                        if key in results:
                            lines.append(f"    {metric.upper()}: {results[key]:.{precision}f}")
        
        return "\n".join(lines)


def evaluate_model_predictions(valence_true: Union[torch.Tensor, np.ndarray],
                             valence_pred: Union[torch.Tensor, np.ndarray],
                             arousal_true: Union[torch.Tensor, np.ndarray],
                             arousal_pred: Union[torch.Tensor, np.ndarray],
                             metrics: List[str] = None,
                             compute_per_quadrant: bool = True,
                             verbose: bool = True) -> Dict[str, float]:
    """
    Convenience function to evaluate V-A model predictions.
    
    Args:
        valence_true: Ground truth valence values
        valence_pred: Predicted valence values
        arousal_true: Ground truth arousal values
        arousal_pred: Predicted arousal values
        metrics: List of metrics to compute
        compute_per_quadrant: Whether to compute per-quadrant metrics
        verbose: Whether to print formatted results
        
    Returns:
        Dictionary containing all computed metrics
    """
    # Align ranges: clip both predictions and targets to [-1, 1]
    def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float64)
        return np.asarray(x, dtype=np.float64)

    v_t = _to_numpy(valence_true)
    v_p = _to_numpy(valence_pred)
    a_t = _to_numpy(arousal_true)
    a_p = _to_numpy(arousal_pred)

    # Clip to expected V–A range
    v_t = np.clip(v_t, -1.0, 1.0)
    v_p = np.clip(v_p, -1.0, 1.0)
    a_t = np.clip(a_t, -1.0, 1.0)
    a_p = np.clip(a_p, -1.0, 1.0)

    evaluator = VAMetrics(metrics=metrics, compute_per_quadrant=compute_per_quadrant)
    results = evaluator.compute_metrics(v_t, v_p, a_t, a_p)
    
    if verbose:
        print(evaluator.format_results(results))
    
    return results


def evaluate_scene_model_predictions(predictions: Dict[str, torch.Tensor],
                                   targets: Dict[str, torch.Tensor],
                                   metrics_config: Optional[Dict[str, any]] = None,
                                   verbose: bool = True) -> Dict[str, float]:
    """
    Comprehensive evaluation for Scene Model predictions (V-A + Emo8).
    
    Research findings require evaluation of:
    - V-A regression: MAE, Spearman-r, CCC, RMSE
    - Emo8 classification: Weighted-F1, AP, macro-F1
    - Cross-group analysis: replicate ethnicity/region consistency
    
    Args:
        predictions: Model predictions dict
        targets: Ground truth targets dict  
        metrics_config: Configuration for metrics computation
        verbose: Whether to print detailed results
        
    Returns:
        Combined metrics dictionary
    """
    if metrics_config is None:
        metrics_config = {}
    
    all_results = {}
    
    # V-A regression metrics
    if all(key in predictions for key in ['valence', 'arousal']) and \
       all(key in targets for key in ['valence', 'arousal']):
        
        va_metrics = metrics_config.get('va_metrics', ['mae', 'spearman', 'ccc', 'rmse', 'pearson'])
        compute_quadrants = metrics_config.get('compute_per_quadrant', True)
        
        va_results = evaluate_model_predictions(
            valence_true=targets['valence'],
            valence_pred=predictions['valence'],
            arousal_true=targets['arousal'],
            arousal_pred=predictions['arousal'],
            metrics=va_metrics,
            compute_per_quadrant=compute_quadrants,
            verbose=False
        )
        
        # Prefix with 'va_' for clarity
        all_results.update({f'va_{k}': v for k, v in va_results.items()})
    
    # Emo8 classification metrics
    if 'emo8_logits' in predictions and 'emo8_label' in targets:
        # Convert logits to predictions
        emo8_pred = torch.argmax(predictions['emo8_logits'], dim=1)
        emo8_probs = torch.softmax(predictions['emo8_logits'], dim=1)
        
        # Classification metrics
        emo8_results = emo8_classification_metrics(
            y_true=targets['emo8_label'],
            y_pred=emo8_pred
        )
        all_results.update({f'emo8_{k}': v for k, v in emo8_results.items()})
        
        # Average Precision
        ap_results = emo8_average_precision(
            y_true=targets['emo8_label'],
            y_probs=emo8_probs
        )
        all_results.update({f'emo8_{k}': v for k, v in ap_results.items()})
    
    # Research findings: Separate Valence vs Arousal analysis
    # (Arousal is typically harder to predict)
    if 'va_valence_mae' in all_results and 'va_arousal_mae' in all_results:
        all_results['va_mae_ratio'] = all_results['va_arousal_mae'] / (all_results['va_valence_mae'] + 1e-8)
        all_results['va_difficulty_analysis'] = {
            'valence_mae': all_results['va_valence_mae'],
            'arousal_mae': all_results['va_arousal_mae'],
            'arousal_harder': all_results['va_arousal_mae'] > all_results['va_valence_mae']
        }
    
    if verbose:
        print_scene_model_results(all_results)
    
    return all_results


def print_scene_model_results(results: Dict[str, float], precision: int = 4) -> None:
    """
    Print formatted Scene Model evaluation results.
    
    Args:
        results: Dictionary of computed metrics
        precision: Number of decimal places
    """
    print("🏞️  Scene Model Evaluation Results")
    print("=" * 60)
    
    # V-A Regression Results
    va_keys = [k for k in results.keys() if k.startswith('va_') and not k.startswith('va_q')]
    if va_keys:
        print("\n📈 Valence-Arousal Regression:")
        print("-" * 30)
        
        # Key metrics (research findings)
        key_metrics = ['va_mae_avg', 'va_mse_avg', 'va_spearman_avg', 'va_ccc_avg', 'va_rmse_avg']
        for metric in key_metrics:
            if metric in results:
                metric_name = metric.replace('va_', '').replace('_avg', '').upper()
                print(f"  {metric_name:12s}: {results[metric]:.{precision}f}")
        
        # Individual dimensions
        print(f"\n  Valence MAE:  {results.get('va_valence_mae', 0):.{precision}f}")
        print(f"  Arousal MAE:  {results.get('va_arousal_mae', 0):.{precision}f}")
        
        if 'va_difficulty_analysis' in results:
            analysis = results['va_difficulty_analysis']
            harder_dim = "Arousal" if analysis['arousal_harder'] else "Valence"
            print(f"  Harder dimension: {harder_dim}")
    
    # Emo8 Classification Results
    emo8_keys = [k for k in results.keys() if k.startswith('emo8_')]
    if emo8_keys:
        print("\n🎭 Emo8 Classification:")
        print("-" * 30)
        
        # Key metrics (research findings)
        key_metrics = ['emo8_weighted_f1', 'emo8_mean_ap', 'emo8_macro_f1', 'emo8_accuracy']
        for metric in key_metrics:
            if metric in results:
                metric_name = metric.replace('emo8_', '').replace('_', ' ').title()
                print(f"  {metric_name:15s}: {results[metric]:.{precision}f}")
        
        # Class imbalance analysis
        print(f"\n  Class Balance Analysis:")
        class_names = ['joy', 'anticipation', 'anger', 'fear', 'sadness', 'disgust', 'trust', 'surprise']
        for class_name in class_names:
            freq_key = f'emo8_{class_name}_frequency'
            f1_key = f'emo8_{class_name}_f1'
            if freq_key in results and f1_key in results:
                print(f"    {class_name:12s}: freq={results[freq_key]:.3f}, f1={results[f1_key]:.3f}")
    
    # Quadrant Analysis (if available)
    quadrant_keys = [k for k in results.keys() if 'q1_' in k or 'q2_' in k or 'q3_' in k or 'q4_' in k]
    if quadrant_keys:
        print(f"\n🎯 V-A Quadrant Analysis:")
        print("-" * 30)
        quadrants = [
            ('q1_happy', 'Happy/Excited'),
            ('q2_angry', 'Angry/Stressed'),
            ('q3_sad', 'Sad/Depressed'),
            ('q4_calm', 'Calm/Content')
        ]
        
        for quad_key, quad_name in quadrants:
            count_key = f'va_{quad_key}_count'
            mae_key = f'va_{quad_key}_mae_avg'
            if count_key in results and mae_key in results:
                print(f"  {quad_name:15s}: n={results[count_key]:3.0f}, MAE={results[mae_key]:.{precision}f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test metrics with synthetic data
    print("🧪 Testing V-A Evaluation Metrics")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Ground truth
    valence_true = np.random.uniform(-1, 1, n_samples)
    arousal_true = np.random.uniform(-1, 1, n_samples)
    
    # Add some noise to create predictions
    noise_level = 0.3
    valence_pred = valence_true + np.random.normal(0, noise_level, n_samples)
    arousal_pred = arousal_true + np.random.normal(0, noise_level, n_samples)
    
    # Clip predictions to valid range
    valence_pred = np.clip(valence_pred, -1, 1)
    arousal_pred = np.clip(arousal_pred, -1, 1)
    
    # Evaluate
    results = evaluate_model_predictions(
        valence_true, valence_pred,
        arousal_true, arousal_pred,
        verbose=True
    )
    
    print(f"\n✅ Metrics computed successfully!")
    print(f"📊 Total metrics: {len(results)}")
