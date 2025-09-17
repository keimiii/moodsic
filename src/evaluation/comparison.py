#!/usr/bin/env python3
"""
Model comparison utilities for V-A prediction evaluation.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


class ModelComparison:
    """Compare multiple model evaluation results."""
    
    def __init__(self, comparison_name: str = "model_comparison"):
        self.comparison_name = comparison_name
        self.models = []
        self.results = []
    
    def add_model_results(self, result_dir: str, model_alias: str = None):
        """
        Add model results from an evaluation directory.
        
        Args:
            result_dir: Path to evaluation results directory
            model_alias: Optional alias for the model (defaults to model_name from metadata)
        """
        result_path = Path(result_dir)
        
        # Prefer comprehensive summary if available
        summary_file = result_path / "data" / "evaluation_summary.json"
        metadata_file = result_path / "data" / "metadata.json"
        metrics_file = result_path / "data" / "metrics.json"
        data_file = result_path / "data" / "evaluation_data.csv"

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            # Extract comparable structures
            exp = summary.get('experiment_info', {})
            eval_info = summary.get('evaluation_info', {})
            metrics = summary.get('metrics', {})
            # Flattened metadata for downstream usage
            metadata = {
                'model_name': exp.get('model_name', 'Unknown'),
                'dataset_name': exp.get('dataset_name', 'Unknown'),
                'experiment_name': exp.get('experiment_name', None),
                'num_samples': eval_info.get('num_samples', 0),
                'processing_time': eval_info.get('processing_time', None),
                'samples_per_sec': eval_info.get('samples_per_sec', None),
            }
        else:
            # Fall back to legacy files (must exist)
            if not all(f.exists() for f in [metadata_file, metrics_file, data_file]):
                raise FileNotFoundError(f"Required files not found in {result_dir}")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

        # Load data CSV if present
        if data_file.exists():
            data = pd.read_csv(data_file)
        else:
            # Provide empty DataFrame if missing (keeps API stable)
            data = pd.DataFrame()

        # Model naming: prefer alias, then experiment name, then model_name
        model_name = model_alias or metadata.get('experiment_name') or metadata.get('model_name', 'Unknown')
        
        self.models.append(model_name)
        self.results.append({
            'name': model_name,
            'metadata': metadata,
            'metrics': metrics,
            'data': data,
            'result_dir': str(result_path)
        })
        
        print(f"✅ Added {model_name} to comparison")
    
    def create_comparison_report(self, output_dir: str = "./comparison_results"):
        """Create comprehensive comparison report."""
        if len(self.results) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create comparison metrics table
        self._create_metrics_table(output_path)
        
        # Create comparison visualizations
        self._create_comparison_plots(output_path)
        
        # Create detailed report
        self._create_detailed_report(output_path)
        
        print(f"📊 Comparison report saved to: {output_path}")
    
    def _create_metrics_table(self, output_path: Path):
        """Create a table comparing key metrics across models."""
        metrics_data = []
        
        for result in self.results:
            metrics = result['metrics']
            metadata = result['metadata']

            def getm(*keys, default=0.0):
                for k in keys:
                    if k in metrics:
                        return metrics[k]
                return default

            row = {
                'Model': result['name'],
                'Dataset': metadata.get('dataset_name', 'Unknown'),
                'Samples': metadata.get('num_samples', 0),
                # Prefer new 'va_' keys, fallback to legacy keys
                'Valence MAE': getm('va_valence_mae', 'valence_mae', default=0),
                'Arousal MAE': getm('va_arousal_mae', 'arousal_mae', default=0),
                'Total MAE': getm('va_mae_avg', 'total_mae', default=0),
                'Valence CCC': getm('va_valence_ccc', 'valence_ccc', default=0),
                'Arousal CCC': getm('va_arousal_ccc', 'arousal_ccc', default=0),
                'Average CCC': getm('va_ccc_avg', 'average_ccc', default=0),
                'Valence Correlation': getm('va_valence_pearson', 'valence_correlation', default=0),
                'Arousal Correlation': getm('va_arousal_pearson', 'arousal_correlation', default=0),
                'Quadrant Accuracy': metrics.get('overall_quadrant_accuracy', 0),
                'Processing Speed (samples/sec)': metadata.get('samples_per_sec', 0)
            }
            
            # Add quadrant-specific accuracies
            for quadrant in ['angry', 'sad', 'calm', 'happy']:
                key = f'{quadrant}_accuracy'
                if key in metrics:
                    row[f'{quadrant.title()} Accuracy'] = metrics[key]
            
            metrics_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(metrics_data)
        
        # Save as CSV
        df.to_csv(output_path / "metrics_comparison.csv", index=False)
        
        # Save as formatted text table
        with open(output_path / "metrics_comparison.txt", 'w') as f:
            f.write("MODEL COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(df.to_string(index=False, float_format='%.4f'))
            f.write("\n\n")
            
            # Add ranking by key metrics
            f.write("RANKINGS BY KEY METRICS:\n")
            f.write("-" * 40 + "\n")
            
            for metric in ['Total MAE', 'Average CCC', 'Quadrant Accuracy']:
                if metric in df.columns:
                    ascending = metric == 'Total MAE'  # Lower is better for MAE
                    ranking = df.nlargest(len(df), metric) if not ascending else df.nsmallest(len(df), metric)
                    f.write(f"\n{metric} ({'Lower is better' if ascending else 'Higher is better'}):\n")
                    for i, (_, row) in enumerate(ranking.iterrows(), 1):
                        f.write(f"  {i}. {row['Model']}: {row[metric]:.4f}\n")
        
        print(f"📊 Metrics comparison saved to: {output_path / 'metrics_comparison.csv'}")
    
    def _create_comparison_plots(self, output_path: Path):
        """Create comparison visualizations."""
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Metrics comparison bar chart
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Model Performance Comparison - {self.comparison_name}', fontsize=16, fontweight='bold')
        
        metrics_to_plot = [
            ('valence_mae', 'Valence MAE', 'Lower is Better'),
            ('arousal_mae', 'Arousal MAE', 'Lower is Better'),
            ('average_ccc', 'Average CCC', 'Higher is Better'),
            ('overall_quadrant_accuracy', 'Quadrant Accuracy', 'Higher is Better'),
            ('valence_correlation', 'Valence Correlation', 'Higher is Better'),
            ('arousal_correlation', 'Arousal Correlation', 'Higher is Better')
        ]
        
        for i, (metric_key, title, direction) in enumerate(metrics_to_plot):
            ax = axes[i // 3, i % 3]
            
            model_names = [r['name'] for r in self.results]
            metric_values = [r['metrics'].get(metric_key, 0) for r in self.results]
            
            bars = ax.bar(model_names, metric_values)
            ax.set_title(f'{title}\n({direction})')
            ax.set_ylabel(title)
            
            # Color bars based on performance
            if 'Lower is Better' in direction:
                best_idx = np.argmin(metric_values)
            else:
                best_idx = np.argmax(metric_values)
            
            for j, bar in enumerate(bars):
                if j == best_idx:
                    bar.set_color('green')
                    bar.set_alpha(0.8)
                else:
                    bar.set_alpha(0.6)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(output_path / "metrics_comparison.pdf", bbox_inches='tight')
        plt.close()
        
        # 2. V-A space comparison
        fig, axes = plt.subplots(1, len(self.results), figsize=(6*len(self.results), 6))
        if len(self.results) == 1:
            axes = [axes]
        
        fig.suptitle('Valence-Arousal Space Comparison', fontsize=16, fontweight='bold')
        
        for i, result in enumerate(self.results):
            ax = axes[i]
            data = result['data']
            
            # Plot ground truth and predictions
            ax.scatter(data['valence_true'], data['arousal_true'], 
                      alpha=0.6, s=30, c='blue', label='Ground Truth')
            ax.scatter(data['valence_pred'], data['arousal_pred'], 
                      alpha=0.6, s=30, c='red', marker='x', label='Predictions')
            
            # Add quadrant lines
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Valence')
            ax.set_ylabel('Arousal')
            ax.set_title(f'{result["name"]}\nCCC: {result["metrics"].get("average_ccc", 0):.3f}')
            ax.legend()
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "va_space_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(output_path / "va_space_comparison.pdf", bbox_inches='tight')
        plt.close()
        
        # 3. Error distribution comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Error Distribution Comparison', fontsize=16, fontweight='bold')
        
        # Valence errors
        ax1 = axes[0]
        for result in self.results:
            data = result['data']
            valence_errors = abs(data['valence_pred'] - data['valence_true'])
            ax1.hist(valence_errors, bins=30, alpha=0.6, label=result['name'], density=True)
        
        ax1.set_xlabel('Valence Absolute Error')
        ax1.set_ylabel('Density')
        ax1.set_title('Valence Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Arousal errors
        ax2 = axes[1]
        for result in self.results:
            data = result['data']
            arousal_errors = abs(data['arousal_pred'] - data['arousal_true'])
            ax2.hist(arousal_errors, bins=30, alpha=0.6, label=result['name'], density=True)
        
        ax2.set_xlabel('Arousal Absolute Error')
        ax2.set_ylabel('Density')
        ax2.set_title('Arousal Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "error_distribution_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(output_path / "error_distribution_comparison.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"🎨 Comparison plots saved to: {output_path}")
    
    def _create_detailed_report(self, output_path: Path):
        """Create a detailed comparison report."""
        report_file = output_path / "detailed_comparison_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Model Comparison Report: {self.comparison_name}\n\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Models Compared\n\n")
            for i, result in enumerate(self.results, 1):
                metadata = result['metadata']
                f.write(f"{i}. **{result['name']}**\n")
                f.write(f"   - Dataset: {metadata.get('dataset_name', 'Unknown')}\n")
                f.write(f"   - Samples: {metadata.get('num_samples', 0):,}\n")
                f.write(f"   - Device: {metadata.get('device', 'Unknown')}\n")
                f.write(f"   - Processing Speed: {metadata.get('samples_per_sec', 0):.1f} samples/sec\n\n")
            
            f.write("## Performance Summary\n\n")
            
            # Find best performing model for each metric
            best_models = {}
            for metric in ['total_mae', 'average_ccc', 'overall_quadrant_accuracy']:
                if metric == 'total_mae':
                    # Lower is better
                    best_idx = min(range(len(self.results)), 
                                 key=lambda i: self.results[i]['metrics'].get(metric, float('inf')))
                else:
                    # Higher is better
                    best_idx = max(range(len(self.results)), 
                                 key=lambda i: self.results[i]['metrics'].get(metric, 0))
                best_models[metric] = self.results[best_idx]['name']
            
            f.write(f"- **Best Total MAE**: {best_models['total_mae']}\n")
            f.write(f"- **Best Average CCC**: {best_models['average_ccc']}\n")
            f.write(f"- **Best Quadrant Accuracy**: {best_models['overall_quadrant_accuracy']}\n\n")
            
            f.write("## Detailed Metrics\n\n")
            
            # Create detailed metrics table in markdown
            metrics_df = pd.DataFrame([
                {
                    'Model': r['name'],
                    'Total MAE': r['metrics'].get('total_mae', 0),
                    'Average CCC': r['metrics'].get('average_ccc', 0),
                    'Quadrant Acc': r['metrics'].get('overall_quadrant_accuracy', 0),
                    'Val MAE': r['metrics'].get('valence_mae', 0),
                    'Ar MAE': r['metrics'].get('arousal_mae', 0),
                    'Val CCC': r['metrics'].get('valence_ccc', 0),
                    'Ar CCC': r['metrics'].get('arousal_ccc', 0)
                }
                for r in self.results
            ])
            
            f.write(metrics_df.to_markdown(index=False, floatfmt='.4f'))
            f.write("\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- `metrics_comparison.csv` - Detailed metrics comparison table\n")
            f.write("- `metrics_comparison.png/pdf` - Performance comparison charts\n")
            f.write("- `va_space_comparison.png/pdf` - V-A space visualization\n")
            f.write("- `error_distribution_comparison.png/pdf` - Error distribution plots\n")
            f.write("- `detailed_comparison_report.md` - This report\n\n")
        
        print(f"📝 Detailed report saved to: {report_file}")


def compare_models_from_dirs(result_dirs: List[str], 
                           model_aliases: List[str] = None,
                           comparison_name: str = "model_comparison",
                           output_dir: str = "./comparison_results"):
    """
    Convenience function to compare models from result directories.
    
    Args:
        result_dirs: List of paths to evaluation result directories
        model_aliases: Optional list of aliases for the models
        comparison_name: Name for the comparison
        output_dir: Output directory for comparison results
    """
    comparison = ModelComparison(comparison_name)
    
    aliases = model_aliases or [None] * len(result_dirs)
    
    for result_dir, alias in zip(result_dirs, aliases):
        comparison.add_model_results(result_dir, alias)
    
    comparison.create_comparison_report(output_dir)
    
    return comparison
