"""
Universal evaluation module for Valence-Arousal prediction models.
Provides comprehensive evaluation and visualization capabilities.
"""

from .evaluator import VAEvaluator, quick_evaluate
from .comparison import ModelComparison, compare_models_from_dirs

__all__ = ['VAEvaluator', 'quick_evaluate', 'ModelComparison', 'compare_models_from_dirs']
