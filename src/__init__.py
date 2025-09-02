"""
Continuous Valence-Arousal Emotion Prediction Framework

A production-ready system for training and deploying continuous V-A emotion 
prediction models using DINOv3 backbones. Supports both Scene Model (FindingEmo) 
and Face Mood Model (AffectNet) with fusion capabilities.

Features:
- DINOv3-based backbone with frozen/fine-tunable parameters
- Continuous V-A prediction with CCC and RMSE metrics
- Scene emotion recognition (FindingEmo dataset)
- Facial emotion recognition (AffectNet dataset)
- Model fusion for multi-modal emotion prediction
- Cross-platform support (CPU/GPU/Metal)
- Configuration-driven training and inference
- Comprehensive logging and checkpointing
"""

__version__ = "1.0.0"
__author__ = "Staff FAANG ML Engineer"
__email__ = "ml.engineer@company.com"

# Core imports for easy access (simplified for training)
# from .models.va_models import SceneEmotionModel, FaceMoodModel, EnsembleVAModel  
# from .inference.predictor import ScenePredictor, FacePredictor, EnsemblePredictor, VAResult
# from .utils.config import Config, load_config
# from .utils.metrics import evaluate_model_predictions, VAMetrics
# from .utils.device import DeviceManager

__all__ = [
    # Metadata
    '__version__',
    '__author__',
    '__email__'
]
