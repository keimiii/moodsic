"""Top-level package for model components.

Exports common fusion types for convenience:

from models.fusion import EmotionPrediction, FusionResult, SceneFaceFusion
"""

from .fusion import EmotionPrediction, FusionResult, SceneFaceFusion

__all__ = [
    "EmotionPrediction",
    "FusionResult",
    "SceneFaceFusion",
]

