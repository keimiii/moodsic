#!/usr/bin/env python3
"""
Universal V-A Model Evaluator and Visualizer.
Works with any model and dataset for valence-arousal prediction evaluation.
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from datetime import datetime
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from src.utils.metrics import evaluate_scene_model_predictions

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class VAEvaluator:
    """Universal evaluator for Valence-Arousal prediction models."""
    
    def __init__(self, 
                 model_name: str,
                 dataset_name: str,
                 output_dir: str = "./evaluation_results",
                 device: Optional[torch.device] = None,
                 flat_output: bool = False):
        """
        Initialize the evaluator.
        
        Args:
            model_name: Name of the model being evaluated (e.g., "dinov3_baseline", "dinov3_trained")
            dataset_name: Name of the dataset (e.g., "affectnet", "findingemo")
            output_dir: Base directory for saving results
            device: Device to run evaluation on
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.device = device or self._get_device()
        # Context placeholders (can be enriched by caller)
        self.experiment_name: Optional[str] = None
        self.training_config: Dict[str, Any] = {}
        self.training_summary: Dict[str, Any] = {}
        self.checkpoint_info: Dict[str, Any] = {}
        self.extra_metadata: Dict[str, Any] = {}
        
        # Create output directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(output_dir)
        if flat_output:
            # Use the provided output_dir directly (no nested run folder)
            self.output_dir = base_dir
        else:
            # Default behavior: create a timestamped subfolder
            self.output_dir = base_dir / f"{model_name}_{dataset_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories (only plots and data; avoid nested 'logs' directory)
        self.plots_dir = self.output_dir / "plots"
        self.data_dir = self.output_dir / "data"
        for dir_path in [self.plots_dir, self.data_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Results storage
        self.evaluation_data = []
        self.metrics = {}
        self.metadata = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "timestamp": timestamp,
            "device": str(self.device)
        }

    def attach_context(self,
                       experiment_name: Optional[str] = None,
                       training_config: Optional[Dict[str, Any]] = None,
                       training_summary: Optional[Dict[str, Any]] = None,
                       checkpoint_info: Optional[Dict[str, Any]] = None,
                       extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Attach optional context to be included in saved summaries.

        - experiment_name: Human-readable experiment/run name
        - training_config: Dict of training hyperparameters/config used
        - training_summary: Dict of training outcomes (best metric, epochs, converged)
        - checkpoint_info: Dict with best checkpoint path/epoch, etc.
        - extra_metadata: Any additional metadata to merge
        """
        if experiment_name is not None:
            self.experiment_name = experiment_name
        if training_config:
            self.training_config.update(training_config)
        if training_summary:
            self.training_summary.update(training_summary)
        if checkpoint_info:
            self.checkpoint_info.update(checkpoint_info)
        if extra_metadata:
            self.extra_metadata.update(extra_metadata)
    
    def _get_device(self) -> torch.device:
        """Auto-detect the best available device."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def evaluate_model(self,
                      model: nn.Module,
                      dataloader: DataLoader,
                      max_samples: Optional[int] = None,
                      prediction_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Evaluate a model on a dataset.
        
        Args:
            model: The model to evaluate
            dataloader: DataLoader for the dataset
            max_samples: Maximum number of samples to evaluate (None for all)
            prediction_fn: Custom function to get predictions from model
                         Should take (model, batch) and return (valence, arousal) predictions
        
        Returns:
            Dictionary containing evaluation results and metrics
        """
        model.eval()
        model.to(self.device)
        
        print(f"\n🔄 Evaluating {self.model_name} on {self.dataset_name}")
        print(f"📱 Device: {self.device}")
        print(f"📁 Results will be saved to: {self.output_dir}")
        
        start_time = time.time()
        processed_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_samples and processed_samples >= max_samples:
                    break
                
                try:
                    # Extract data from batch (flexible format)
                    if isinstance(batch, (list, tuple)):
                        if len(batch) >= 3:  # images, valence, arousal format
                            images, valence_true, arousal_true = batch[:3]
                        elif len(batch) == 2:  # images, targets format
                            images, targets = batch
                            if targets.shape[-1] >= 2:
                                valence_true, arousal_true = targets[:, 0], targets[:, 1]
                            else:
                                raise ValueError("Targets must have at least 2 dimensions for V-A")
                    elif isinstance(batch, dict):
                        images = batch['image'] if 'image' in batch else batch['images']
                        valence_true = batch['valence']
                        arousal_true = batch['arousal']
                    else:
                        raise ValueError(f"Unsupported batch format: {type(batch)}")
                    
                    images = images.to(self.device)
                    # Convert targets to numpy
                    valence_true = valence_true.cpu().numpy() if torch.is_tensor(valence_true) else valence_true
                    arousal_true = arousal_true.cpu().numpy() if torch.is_tensor(arousal_true) else arousal_true
                    
                    # Get predictions
                    if prediction_fn:
                        valence_pred, arousal_pred = prediction_fn(model, images)
                    else:
                        # Default prediction method
                        outputs = model(images)
                        if isinstance(outputs, (list, tuple)):
                            valence_pred, arousal_pred = outputs
                        else:
                            valence_pred, arousal_pred = outputs[:, 0], outputs[:, 1]
                    
                    # Convert predictions to numpy
                    valence_pred = valence_pred.cpu().numpy() if torch.is_tensor(valence_pred) else valence_pred
                    arousal_pred = arousal_pred.cpu().numpy() if torch.is_tensor(arousal_pred) else arousal_pred

                    # Auto-map all values to reference [-1,1] range if needed
                    import numpy as _np
                    def _to_ref_v(v):
                        v = _np.asarray(v, dtype=_np.float64)
                        vmax = _np.nanmax(v)
                        vmin = _np.nanmin(v)
                        return v / 3.0 if (vmax > 1.2 or vmin < -1.2) else v
                    def _to_ref_a(a):
                        a = _np.asarray(a, dtype=_np.float64)
                        amax = _np.nanmax(a)
                        amin = _np.nanmin(a)
                        return (a - 3.0) / 3.0 if (amax > 1.2 or amin < -1.2) else a

                    v_t_ref = _to_ref_v(valence_true)
                    a_t_ref = _to_ref_a(arousal_true)
                    v_p_ref = _to_ref_v(valence_pred)
                    a_p_ref = _to_ref_a(arousal_pred)

                    # Clip to [-1,1] to be safe
                    v_t_ref = _np.clip(v_t_ref, -1.0, 1.0)
                    a_t_ref = _np.clip(a_t_ref, -1.0, 1.0)
                    v_p_ref = _np.clip(v_p_ref, -1.0, 1.0)
                    a_p_ref = _np.clip(a_p_ref, -1.0, 1.0)
                    
                    # Store results
                    batch_size = len(v_t_ref)
                    for i in range(batch_size):
                        self.evaluation_data.append({
                            'valence_true': float(v_t_ref[i]),
                            'arousal_true': float(a_t_ref[i]),
                            'valence_pred': float(v_p_ref[i]),
                            'arousal_pred': float(a_p_ref[i]),
                            'quadrant_true': self._get_quadrant(v_t_ref[i], a_t_ref[i]),
                            'quadrant_pred': self._get_quadrant(v_p_ref[i], a_p_ref[i])
                        })
                    
                    processed_samples += batch_size
                    
                    if processed_samples % 100 == 0:
                        print(f"📊 Processed {processed_samples} samples...")
                
                except Exception as e:
                    print(f"⚠️ Error processing batch {batch_idx}: {e}")
                    continue
        
        # Calculate timing
        total_time = time.time() - start_time
        samples_per_sec = processed_samples / total_time if total_time > 0 else 0
        
        print(f"✅ Evaluation completed: {processed_samples} samples in {total_time:.2f}s")
        print(f"⚡ Processing speed: {samples_per_sec:.1f} samples/sec")
        
        # Convert to DataFrame and calculate metrics using the same utility
        # used by the training notebook (ensures naming and math match)
        df = pd.DataFrame(self.evaluation_data)
        preds = {
            'valence': df['valence_pred'].to_numpy(),
            'arousal': df['arousal_pred'].to_numpy(),
        }
        targs = {
            'valence': df['valence_true'].to_numpy(),
            'arousal': df['arousal_true'].to_numpy(),
        }
        self.metrics = evaluate_scene_model_predictions(
            predictions=preds,
            targets=targs,
            metrics_config={'va_metrics': ['mae', 'mse', 'rmse', 'ccc', 'pearson', 'spearman'],
                           'compute_per_quadrant': True},
            verbose=False,
        )
        self.metadata.update({
            "num_samples": processed_samples,
            "processing_time": total_time,
            "samples_per_sec": samples_per_sec
        })
        
        # Save results
        self._save_results(df)
        
        return {
            "dataframe": df,
            "metrics": self.metrics,
            "metadata": self.metadata
        }
    
    def _get_quadrant(self, valence: float, arousal: float) -> str:
        """Map V-A coordinates to emotion quadrant."""
        if valence >= 0 and arousal >= 0:
            return "happy"
        elif valence < 0 and arousal >= 0:
            return "angry"
        elif valence < 0 and arousal < 0:
            return "sad"
        else:
            return "calm"
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        # Add error columns
        df['valence_error'] = abs(df['valence_pred'] - df['valence_true'])
        df['arousal_error'] = abs(df['arousal_pred'] - df['arousal_true'])
        df['total_error'] = np.sqrt(df['valence_error']**2 + df['arousal_error']**2)
        
        metrics = {}
        
        # Basic metrics
        metrics['valence_mae'] = df['valence_error'].mean()
        metrics['arousal_mae'] = df['arousal_error'].mean()
        metrics['valence_rmse'] = np.sqrt((df['valence_error']**2).mean())
        metrics['arousal_rmse'] = np.sqrt((df['arousal_error']**2).mean())
        metrics['total_mae'] = (metrics['valence_mae'] + metrics['arousal_mae']) / 2
        metrics['total_rmse'] = np.sqrt((df['total_error']**2).mean())
        
        # Correlations
        metrics['valence_correlation'] = df['valence_true'].corr(df['valence_pred'])
        metrics['arousal_correlation'] = df['arousal_true'].corr(df['arousal_pred'])
        
        # CCC (Concordance Correlation Coefficient)
        metrics['valence_ccc'] = self._calculate_ccc(df['valence_true'], df['valence_pred'])
        metrics['arousal_ccc'] = self._calculate_ccc(df['arousal_true'], df['arousal_pred'])
        metrics['average_ccc'] = (metrics['valence_ccc'] + metrics['arousal_ccc']) / 2
        
        # Quadrant accuracy
        quadrant_accuracy = {}
        overall_correct = 0
        for quadrant in ['angry', 'sad', 'calm', 'happy']:
            quad_data = df[df['quadrant_true'] == quadrant]
            if len(quad_data) > 0:
                accuracy = (quad_data['quadrant_true'] == quad_data['quadrant_pred']).mean()
                quadrant_accuracy[f'{quadrant}_accuracy'] = accuracy
                quadrant_accuracy[f'{quadrant}_count'] = len(quad_data)
                overall_correct += (quad_data['quadrant_true'] == quad_data['quadrant_pred']).sum()
        
        metrics.update(quadrant_accuracy)
        metrics['overall_quadrant_accuracy'] = overall_correct / len(df) if len(df) > 0 else 0
        
        # Range compression
        val_range_true = df['valence_true'].max() - df['valence_true'].min()
        val_range_pred = df['valence_pred'].max() - df['valence_pred'].min()
        ar_range_true = df['arousal_true'].max() - df['arousal_true'].min()
        ar_range_pred = df['arousal_pred'].max() - df['arousal_pred'].min()
        
        metrics['valence_range_compression'] = val_range_pred / val_range_true if val_range_true > 0 else 0
        metrics['arousal_range_compression'] = ar_range_pred / ar_range_true if ar_range_true > 0 else 0
        
        return metrics
    
    def _calculate_ccc(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate Concordance Correlation Coefficient."""
        if len(y_true) == 0 or y_true.var() == 0 or y_pred.var() == 0:
            return 0.0
        
        mean_true = y_true.mean()
        mean_pred = y_pred.mean()
        var_true = y_true.var()
        var_pred = y_pred.var()
        correlation = y_true.corr(y_pred)
        
        if pd.isna(correlation):
            return 0.0
        
        numerator = 2 * correlation * np.sqrt(var_true) * np.sqrt(var_pred)
        denominator = var_true + var_pred + (mean_true - mean_pred)**2
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _save_results(self, df: pd.DataFrame):
        """Save evaluation results to files."""
        # Save raw data
        data_file = self.data_dir / "evaluation_data.csv"
        df.to_csv(data_file, index=False)
        
        # Save metrics
        metrics_file = self.data_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Save metadata
        metadata_file = self.data_dir / "metadata.json"
        # Merge attached context into metadata for completeness
        merged_metadata = {
            **self.metadata,
            **({"experiment_name": self.experiment_name} if self.experiment_name else {}),
            **({"training_config": self.training_config} if self.training_config else {}),
            **({"training_summary": self.training_summary} if self.training_summary else {}),
            **({"checkpoint_info": self.checkpoint_info} if self.checkpoint_info else {}),
            **self.extra_metadata,
        }
        with open(metadata_file, 'w') as f:
            json.dump(merged_metadata, f, indent=2, default=str)

        # Save comprehensive evaluation summary (single file consumers can read)
        summary = {
            "experiment_info": {
                "model_name": self.model_name,
                "dataset_name": self.dataset_name,
                "experiment_name": self.experiment_name,
                "timestamp": self.metadata.get("timestamp"),
                "device": self.metadata.get("device"),
            },
            "evaluation_info": {
                "num_samples": self.metadata.get("num_samples"),
                "processing_time": self.metadata.get("processing_time"),
                "samples_per_sec": self.metadata.get("samples_per_sec"),
                "results_dir": str(self.output_dir),
            },
            # Training context may be partially empty if not provided
            "training_context": {
                **({"best_metric": self.training_summary.get("best_metric")} if self.training_summary else {}),
                **({"total_epochs": self.training_summary.get("total_epochs")} if self.training_summary else {}),
                **({"converged": self.training_summary.get("converged")} if self.training_summary else {}),
                **({"monitor_metric": self.training_config.get("monitor_metric")} if self.training_config else {}),
                **({"best_checkpoint_path": self.checkpoint_info.get("path")} if self.checkpoint_info else {}),
                **({"best_checkpoint_epoch": self.checkpoint_info.get("epoch")} if self.checkpoint_info else {}),
            },
            # Put all computed metrics under one key
            "metrics": self.metrics,
        }
        summary_file = self.data_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"💾 Data saved to: {self.data_dir}")
    
    def create_comprehensive_visualization(self, df: Optional[pd.DataFrame] = None) -> plt.Figure:
        """Create comprehensive visualization of evaluation results."""
        if df is None:
            if not self.evaluation_data:
                raise ValueError("No evaluation data available. Run evaluate_model first.")
            df = pd.DataFrame(self.evaluation_data)
        
        # Add error columns if not present
        if 'valence_error' not in df.columns:
            df['valence_error'] = abs(df['valence_pred'] - df['valence_true'])
            df['arousal_error'] = abs(df['arousal_pred'] - df['arousal_true'])
            df['total_error'] = np.sqrt(df['valence_error']**2 + df['arousal_error']**2)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(22, 18))
        fig.suptitle(f'{self.model_name} on {self.dataset_name} - Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Scatter plot: Predictions vs Ground Truth (Valence)
        ax1 = plt.subplot(3, 4, 1)
        ax1.scatter(df['valence_true'], df['valence_pred'], alpha=0.6, s=30, c='blue', label='Predictions')
        ax1.plot([-1, 1], [-1, 1], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        ax1.set_xlabel('True Valence')
        ax1.set_ylabel('Predicted Valence')
        ax1.set_title(
            f'Valence: Predicted vs True\nMAE: {self.metrics.get("va_valence_mae", self.metrics.get("valence_mae", 0)):.3f}, '
            f'CCC: {self.metrics.get("va_valence_ccc", self.metrics.get("valence_ccc", 0)):.3f}'
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        
        # 2. Scatter plot: Predictions vs Ground Truth (Arousal)
        ax2 = plt.subplot(3, 4, 2)
        ax2.scatter(df['arousal_true'], df['arousal_pred'], alpha=0.6, s=30, c='orange', label='Predictions')
        ax2.plot([-1, 1], [-1, 1], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('True Arousal')
        ax2.set_ylabel('Predicted Arousal')
        ax2.set_title(
            f'Arousal: Predicted vs True\nMAE: {self.metrics.get("va_arousal_mae", self.metrics.get("arousal_mae", 0)):.3f}, '
            f'CCC: {self.metrics.get("va_arousal_ccc", self.metrics.get("arousal_ccc", 0)):.3f}'
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        
        # 3. V-A Space visualization
        ax3 = plt.subplot(3, 4, 3)
        ax3.scatter(df['valence_true'], df['arousal_true'], alpha=0.6, s=30, c='blue', label='Ground Truth')
        ax3.scatter(df['valence_pred'], df['arousal_pred'], alpha=0.6, s=30, c='red', marker='x', label='Predictions')
        
        # Add quadrant lines and labels
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax3.text(0.7, 0.7, 'Happy/Excited\n(+V, +A)', ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax3.text(-0.7, 0.7, 'Angry/Stressed\n(-V, +A)', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        ax3.text(-0.7, -0.7, 'Sad/Depressed\n(-V, -A)', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax3.text(0.7, -0.7, 'Calm/Content\n(+V, -A)', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        ax3.set_xlabel('Valence')
        ax3.set_ylabel('Arousal')
        ax3.set_title('Valence-Arousal Space')
        ax3.legend()
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-1, 1)
        ax3.grid(True, alpha=0.3)
        
        # 4. Distribution comparison (Valence)
        ax4 = plt.subplot(3, 4, 4)
        ax4.hist(df['valence_true'], bins=30, alpha=0.6, label='True Valence', color='blue', density=True)
        ax4.hist(df['valence_pred'], bins=30, alpha=0.6, label='Pred Valence', color='red', density=True)
        ax4.set_xlabel('Valence')
        ax4.set_ylabel('Density')
        ax4.set_title('Valence Distribution Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Distribution comparison (Arousal)
        ax5 = plt.subplot(3, 4, 5)
        ax5.hist(df['arousal_true'], bins=30, alpha=0.6, label='True Arousal', color='blue', density=True)
        ax5.hist(df['arousal_pred'], bins=30, alpha=0.6, label='Pred Arousal', color='orange', density=True)
        ax5.set_xlabel('Arousal')
        ax5.set_ylabel('Density')
        ax5.set_title('Arousal Distribution Comparison')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Error analysis
        ax6 = plt.subplot(3, 4, 6)
        ax6.hist(df['valence_error'], bins=30, alpha=0.7, color='red', 
                label=f'Valence MAE: {df["valence_error"].mean():.3f}')
        ax6.hist(df['arousal_error'], bins=30, alpha=0.7, color='orange', 
                label=f'Arousal MAE: {df["arousal_error"].mean():.3f}')
        ax6.set_xlabel('Absolute Error')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Error Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Quadrant distribution
        ax7 = plt.subplot(3, 4, 7)
        quadrant_counts = df['quadrant_true'].value_counts()
        colors = ['lightcoral', 'lightblue', 'lightyellow', 'lightgreen']
        wedges, texts, autotexts = ax7.pie(quadrant_counts.values, labels=quadrant_counts.index, 
                                          autopct='%1.1f%%', colors=colors)
        ax7.set_title('True Quadrant Distribution')
        
        # 8. Quadrant accuracy
        ax8 = plt.subplot(3, 4, 8)
        accuracy_data = []
        for quadrant in ['angry', 'sad', 'calm', 'happy']:
            if f'{quadrant}_accuracy' in self.metrics:
                accuracy_data.append({
                    'Quadrant': quadrant.title(), 
                    'Accuracy': self.metrics[f'{quadrant}_accuracy'] * 100,
                    'Count': self.metrics[f'{quadrant}_count']
                })
        
        if accuracy_data:
            acc_df = pd.DataFrame(accuracy_data)
            bars = ax8.bar(acc_df['Quadrant'], acc_df['Accuracy'], 
                          color=['lightcoral', 'lightblue', 'lightyellow', 'lightgreen'])
            ax8.set_ylabel('Accuracy (%)')
            ax8.set_title(f'Quadrant Prediction Accuracy\nOverall: {self.metrics.get("overall_quadrant_accuracy", 0)*100:.1f}%')
            ax8.set_ylim(0, 100)
            
            # Add count labels
            for bar, count in zip(bars, acc_df['Count']):
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'n={count}', ha='center', va='bottom', fontsize=8)
            ax8.grid(True, alpha=0.3)
        
        # 9. V-A error heatmap (mean absolute error over true V-A space)
        ax9 = plt.subplot(3, 4, 9)
        try:
            import numpy as _np
            v_true = df['valence_true'].to_numpy()
            a_true = df['arousal_true'].to_numpy()
            v_err = _np.abs(df['valence_pred'].to_numpy() - v_true)
            a_err = _np.abs(df['arousal_pred'].to_numpy() - a_true)
            mean_err = (v_err + a_err) / 2.0
            # Bin grid
            N = 25
            edges = _np.linspace(-1, 1, N + 1)
            vi = _np.clip(_np.digitize(v_true, edges) - 1, 0, N - 1)
            ai = _np.clip(_np.digitize(a_true, edges) - 1, 0, N - 1)
            heat = _np.zeros((N, N), dtype=float)
            cnt = _np.zeros((N, N), dtype=int)
            for i in range(len(mean_err)):
                heat[ai[i], vi[i]] += mean_err[i]
                cnt[ai[i], vi[i]] += 1
            with _np.errstate(invalid='ignore'):
                heat = _np.divide(heat, cnt, out=_np.zeros_like(heat), where=cnt>0)
            im = ax9.imshow(heat, origin='lower', extent=[-1,1,-1,1], cmap='magma')
            ax9.set_title('Mean Absolute Error across V-A Space')
            ax9.set_xlabel('Valence (true)')
            ax9.set_ylabel('Arousal (true)')
            ax9.axhline(0, color='w', ls='--', lw=1, alpha=0.6)
            ax9.axvline(0, color='w', ls='--', lw=1, alpha=0.6)
            cbar = plt.colorbar(im, ax=ax9)
            cbar.set_label('Mean Abs Error')
        except Exception as _e:
            ax9.axis('off')
            ax9.text(0.5, 0.5, f'Heatmap error: {_e}', ha='center', va='center', transform=ax9.transAxes)

        # 10. Residuals vs True (Valence)
        ax10 = plt.subplot(3, 4, 10)
        v_res = df['valence_pred'] - df['valence_true']
        ax10.scatter(df['valence_true'], v_res, s=10, alpha=0.4, color='tab:blue')
        ax10.axhline(0, color='r', ls='--', lw=1)
        # binned mean residual
        try:
            import numpy as _np
            bins = _np.linspace(-1, 1, 21)
            idx = _np.digitize(df['valence_true'], bins) - 1
            m = _np.full(len(bins)-1, _np.nan)
            for i in range(len(m)):
                mask = idx == i
                if mask.any():
                    m[i] = v_res[mask].mean()
            centers = (bins[:-1] + bins[1:]) / 2
            ax10.plot(centers, m, color='k', lw=2, label='Binned mean residual')
            ax10.legend()
        except Exception:
            pass
        ax10.set_xlabel('True Valence')
        ax10.set_ylabel('Residual (pred - true)')
        ax10.set_title('Residuals vs True (Valence)')
        ax10.grid(True, alpha=0.3)

        # 11. Residuals vs True (Arousal)
        ax11 = plt.subplot(3, 4, 11)
        a_res = df['arousal_pred'] - df['arousal_true']
        ax11.scatter(df['arousal_true'], a_res, s=10, alpha=0.4, color='tab:orange')
        ax11.axhline(0, color='r', ls='--', lw=1)
        try:
            import numpy as _np
            bins = _np.linspace(-1, 1, 21)
            idx = _np.digitize(df['arousal_true'], bins) - 1
            m = _np.full(len(bins)-1, _np.nan)
            for i in range(len(m)):
                mask = idx == i
                if mask.any():
                    m[i] = a_res[mask].mean()
            centers = (bins[:-1] + bins[1:]) / 2
            ax11.plot(centers, m, color='k', lw=2, label='Binned mean residual')
            ax11.legend()
        except Exception:
            pass
        ax11.set_xlabel('True Arousal')
        ax11.set_ylabel('Residual (pred - true)')
        ax11.set_title('Residuals vs True (Arousal)')
        ax11.grid(True, alpha=0.3)

        # 12. Performance summary text
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        summary_text = f"""
Performance Summary (V-A Regression):
• Total MAE: {self.metrics.get('total_mae', 0):.3f}
• Average CCC: {self.metrics.get('average_ccc', 0):.3f}
• Valence: MAE {self.metrics.get('valence_mae', 0):.3f}, CCC {self.metrics.get('valence_ccc', 0):.3f}
• Arousal: MAE {self.metrics.get('arousal_mae', 0):.3f}, CCC {self.metrics.get('arousal_ccc', 0):.3f}
• Range Compression (V/A): {self.metrics.get('valence_range_compression', 0)*100:.1f}% / {self.metrics.get('arousal_range_compression', 0)*100:.1f}%
• Samples: {self.metadata.get('num_samples', 0):,}  |  Speed: {self.metadata.get('samples_per_sec', 0):.1f}/s
• Device: {self.metadata.get('device', 'Unknown')}
        """
        ax12.text(0.02, 0.98, summary_text, transform=ax12.transAxes, fontsize=10,
                  va='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax12.set_title('Performance Summary')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = self.plots_dir / "comprehensive_evaluation.png"
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        plot_file_pdf = self.plots_dir / "comprehensive_evaluation.pdf"
        fig.savefig(plot_file_pdf, bbox_inches='tight')
        
        print(f"🎨 Visualization saved to: {self.plots_dir}")
        # --- Calibration analysis (V-A regression) ---
        try:
            import numpy as _np
            import pandas as _pd

            def _calibration(df, true_col, pred_col, bins=21):
                edges = _np.linspace(-1, 1, bins)
                idx = _np.clip(_np.digitize(df[true_col], edges) - 1, 0, bins - 2)
                rows = []
                for i in range(bins - 1):
                    mask = (idx == i)
                    if mask.any():
                        tvals = df.loc[mask, true_col].to_numpy()
                        pvals = df.loc[mask, pred_col].to_numpy()
                        rows.append({
                            'bin_center': float((edges[i] + edges[i+1]) / 2),
                            'count': int(mask.sum()),
                            'true_mean': float(tvals.mean()),
                            'pred_mean': float(pvals.mean()),
                            'mae': float(_np.mean(_np.abs(pvals - tvals))),
                        })
                    else:
                        rows.append({
                            'bin_center': float((edges[i] + edges[i+1]) / 2),
                            'count': 0,
                            'true_mean': _np.nan,
                            'pred_mean': _np.nan,
                            'mae': _np.nan,
                        })
                return _pd.DataFrame(rows)

            cal_v = _calibration(df, 'valence_true', 'valence_pred', bins=21)
            cal_v.insert(0, 'dimension', 'valence')
            cal_a = _calibration(df, 'arousal_true', 'arousal_pred', bins=21)
            cal_a.insert(0, 'dimension', 'arousal')
            cal_df = _pd.concat([cal_v, cal_a], ignore_index=True)

            # Save calibration CSV
            cal_csv = self.data_dir / 'calibration_bins.csv'
            cal_df.to_csv(cal_csv, index=False)

            # Create calibration figure (two subplots)
            fig_cal = plt.figure(figsize=(12, 5))
            axc1 = plt.subplot(1, 2, 1)
            axc2 = plt.subplot(1, 2, 2)
            # Valence calibration
            axc1.plot([-1, 1], [-1, 1], 'k--', lw=1, alpha=0.7, label='Ideal')
            axc1.plot(cal_v['true_mean'], cal_v['pred_mean'], marker='o', lw=2, color='tab:blue', label='Binned mean')
            axc1.set_title('Calibration (Valence)')
            axc1.set_xlabel('True (binned mean)')
            axc1.set_ylabel('Predicted (binned mean)')
            axc1.set_xlim([-1, 1])
            axc1.set_ylim([-1, 1])
            axc1.grid(True, alpha=0.3)
            axc1.legend()
            # Arousal calibration
            axc2.plot([-1, 1], [-1, 1], 'k--', lw=1, alpha=0.7, label='Ideal')
            axc2.plot(cal_a['true_mean'], cal_a['pred_mean'], marker='o', lw=2, color='tab:orange', label='Binned mean')
            axc2.set_title('Calibration (Arousal)')
            axc2.set_xlabel('True (binned mean)')
            axc2.set_ylabel('Predicted (binned mean)')
            axc2.set_xlim([-1, 1])
            axc2.set_ylim([-1, 1])
            axc2.grid(True, alpha=0.3)
            axc2.legend()
            plt.tight_layout()
            cal_png = self.plots_dir / 'calibration.png'
            fig_cal.savefig(cal_png, dpi=300, bbox_inches='tight')
        except Exception as _e:
            print(f"⚠️ Calibration analysis failed: {_e}")

        return fig
    
    def print_summary(self):
        """Print a comprehensive summary of evaluation results."""
        print("\n" + "="*80)
        print(f"📊 EVALUATION SUMMARY: {self.model_name} on {self.dataset_name}")
        print("="*80)
        
        if not self.metrics:
            print("❌ No evaluation results available. Run evaluate_model first.")
            return
        # Map new-style 'va_' metrics to summary values, fallback to legacy keys
        val_mae = self.metrics.get('va_valence_mae', self.metrics.get('valence_mae', 0.0))
        aro_mae = self.metrics.get('va_arousal_mae', self.metrics.get('arousal_mae', 0.0))
        total_mae = self.metrics.get('va_mae_avg', self.metrics.get('total_mae', (val_mae + aro_mae) / 2 if (val_mae or aro_mae) else 0.0))

        val_rmse = self.metrics.get('va_valence_rmse', self.metrics.get('valence_rmse', 0.0))
        aro_rmse = self.metrics.get('va_arousal_rmse', self.metrics.get('arousal_rmse', 0.0))
        total_rmse = self.metrics.get('va_rmse_avg', self.metrics.get('total_rmse', (val_rmse + aro_rmse) / 2 if (val_rmse or aro_rmse) else 0.0))

        avg_ccc = self.metrics.get('va_ccc_avg', self.metrics.get('average_ccc', 0.0))

        # Correlations: prefer Pearson
        val_corr = self.metrics.get('va_valence_pearson', self.metrics.get('valence_correlation', 0.0))
        aro_corr = self.metrics.get('va_arousal_pearson', self.metrics.get('arousal_correlation', 0.0))

        # Quadrant accuracies: compute from evaluation_data if not present
        overall_acc = self.metrics.get('overall_quadrant_accuracy', None)
        per_quad = {}
        if overall_acc is None and self.evaluation_data:
            try:
                import pandas as _pd
                _df = _pd.DataFrame(self.evaluation_data)
                if {'quadrant_true', 'quadrant_pred'}.issubset(_df.columns):
                    overall_acc = float((_df['quadrant_true'] == _df['quadrant_pred']).mean())
                    for quad in ['angry', 'sad', 'calm', 'happy']:
                        _qd = _df[_df['quadrant_true'] == quad]
                        if len(_qd) > 0:
                            per_quad[quad] = {
                                'acc': float((_qd['quadrant_true'] == _qd['quadrant_pred']).mean()),
                                'count': int(len(_qd))
                            }
            except Exception:
                overall_acc = overall_acc or 0.0
        overall_acc = overall_acc if overall_acc is not None else 0.0

        print(f"\n📈 Overall Performance:")
        print(f"  Total MAE: {total_mae:.4f}")
        print(f"  Total RMSE: {total_rmse:.4f}")
        print(f"  Average CCC: {avg_ccc:.4f}")
        print(f"  Overall Quadrant Accuracy: {overall_acc*100:.1f}%")

        print(f"\n📊 Valence Performance:")
        print(f"  MAE: {val_mae:.4f}")
        print(f"  RMSE: {val_rmse:.4f}")
        print(f"  CCC: {self.metrics.get('va_valence_ccc', self.metrics.get('valence_ccc', 0.0)):.4f}")
        print(f"  Correlation (Pearson): {val_corr:.4f}")

        print(f"\n📊 Arousal Performance:")
        print(f"  MAE: {aro_mae:.4f}")
        print(f"  RMSE: {aro_rmse:.4f}")
        print(f"  CCC: {self.metrics.get('va_arousal_ccc', self.metrics.get('arousal_ccc', 0.0)):.4f}")
        print(f"  Correlation (Pearson): {aro_corr:.4f}")

        print(f"\n🎭 Quadrant Performance:")
        if per_quad:
            for quadrant in ['angry', 'sad', 'calm', 'happy']:
                if quadrant in per_quad:
                    acc = per_quad[quadrant]['acc'] * 100
                    count = per_quad[quadrant]['count']
                    print(f"  {quadrant.title()}: {acc:.1f}% accuracy ({count:,} samples)")
        else:
            for quadrant in ['angry', 'sad', 'calm', 'happy']:
                if f'{quadrant}_accuracy' in self.metrics:
                    acc = self.metrics[f'{quadrant}_accuracy'] * 100
                    count = self.metrics.get(f'{quadrant}_count', 0)
                    print(f"  {quadrant.title()}: {acc:.1f}% accuracy ({count:,} samples)")
        
        print(f"\n⚡ Processing Information:")
        print(f"  Samples: {self.metadata.get('num_samples', 0):,}")
        print(f"  Processing Time: {self.metadata.get('processing_time', 0):.2f}s")
        print(f"  Speed: {self.metadata.get('samples_per_sec', 0):.1f} samples/sec")
        print(f"  Device: {self.metadata.get('device', 'Unknown')}")
        
        print(f"\n📁 Results Location: {self.output_dir}")
        print("="*80)


def quick_evaluate(model: nn.Module,
                  dataloader: DataLoader,
                  model_name: str,
                  dataset_name: str,
                  max_samples: Optional[int] = None,
                  prediction_fn: Optional[Callable] = None,
                  output_dir: str = "./evaluation_results") -> Dict[str, Any]:
    """
    Quick evaluation function for simple use cases.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for dataset
        model_name: Name of the model
        dataset_name: Name of the dataset
        max_samples: Maximum samples to evaluate
        prediction_fn: Custom prediction function
        output_dir: Output directory for results
    
    Returns:
        Dictionary with evaluation results
    """
    evaluator = VAEvaluator(model_name, dataset_name, output_dir)
    results = evaluator.evaluate_model(model, dataloader, max_samples, prediction_fn)
    evaluator.create_comprehensive_visualization()
    evaluator.print_summary()
    
    return results
