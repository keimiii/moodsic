"""Model architectures for V-A emotion prediction."""

from .backbone import DINOv3Backbone, create_dinov3_backbone
from .emotion_head import EmotionHead, MultiHeadEmotionHead, AttentionEmotionHead, create_emotion_head
from .va_models import BaseVAModel, SceneEmotionModel, FaceMoodModel, EnsembleVAModel, create_va_model

__all__ = [
    # Backbone
    'DINOv3Backbone',
    'create_dinov3_backbone',
    
    # Emotion heads
    'EmotionHead',
    'MultiHeadEmotionHead', 
    'AttentionEmotionHead',
    'create_emotion_head',
    
    # Complete models
    'BaseVAModel',
    'SceneEmotionModel',
    'FaceMoodModel',
    'EnsembleVAModel',
    'create_va_model'
]
