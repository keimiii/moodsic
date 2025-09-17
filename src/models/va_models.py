"""
Complete Valence-Arousal prediction models.
Combines DINOv3 backbone with emotion heads for Scene and Face models.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

from .backbone import (
    DINOv3Backbone,
    create_dinov3_backbone,
    CLIPViTBackbone,
    create_clip_backbone,
    ResNetBackbone,
)
from .emotion_head import EmotionHead, create_emotion_head

logger = logging.getLogger(__name__)


class BaseVAModel(nn.Module):
    """
    Base class for Valence-Arousal prediction models.
    Combines frozen backbone with trainable emotion head.
    """
    
    def __init__(self,
                 backbone_path: Union[str, Path] = None,
                 model_name: str = "va_model",
                 feature_dim: int = 768,
                 head_config: Dict = None,
                 freeze_backbone: bool = True,
                 backbone_type: str = "dinov3",
                 clip_model_name: str = "ViT-B/32",
                 imagenet_backbone_name: str = "resnet50"):
        """
        Initialize base V-A model.
        
        Args:
            backbone_path: Path to DINOv3 model (required for DINOv3)
            model_name: Name identifier for the model
            feature_dim: Expected feature dimension from backbone
            head_config: Configuration for emotion head
            freeze_backbone: Whether to freeze backbone parameters
            backbone_type: Type of backbone ("dinov3" or "clip")
            clip_model_name: Name of CLIP model if using CLIP backbone
        """
        super().__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.backbone_type = backbone_type
        
        # Initialize backbone based on type
        if backbone_type == "clip":
            self.backbone = create_clip_backbone(
                model_name=clip_model_name,
                freeze=freeze_backbone
            )
        elif backbone_type == "dinov3":
            if backbone_path is None:
                raise ValueError("backbone_path is required for DINOv3 backbone")
            self.backbone = create_dinov3_backbone(
                model_path=backbone_path,
                freeze=freeze_backbone
            )
        elif backbone_type == "imagenet":
            # Use torchvision ResNet family
            self.backbone = ResNetBackbone(model_name=imagenet_backbone_name, freeze=freeze_backbone, pretrained=True)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}. Use 'dinov3', 'clip', or 'imagenet'.")
        
        # Verify feature dimension
        if self.backbone.feature_dim != feature_dim:
            logger.warning(f"Feature dim mismatch: expected {feature_dim}, got {self.backbone.feature_dim}")
            feature_dim = self.backbone.feature_dim
        
        # Initialize emotion head
        head_config = head_config or {}
        self.emotion_head = create_emotion_head(
            input_dim=feature_dim,
            **head_config
        )
        
        # Store configuration
        self.config = {
            'model_name': model_name,
            'backbone_path': str(backbone_path) if backbone_path else None,
            'backbone_type': backbone_type,
            'clip_model_name': clip_model_name if backbone_type == "clip" else None,
            'feature_dim': feature_dim,
            'head_config': head_config,
            'freeze_backbone': freeze_backbone,
            'imagenet_backbone_name': imagenet_backbone_name if backbone_type == "imagenet" else None
        }
        
        logger.info(f"🎭 {model_name} initialized")
        if backbone_type == "clip":
            logger.info(f"  Backbone: CLIP {clip_model_name}")
        elif backbone_type == "imagenet":
            logger.info("  Backbone: ImageNet (torchvision)")
        else:
            logger.info(f"  Backbone: {Path(str(backbone_path)).name}")
        logger.info(f"  Feature dim: {feature_dim}")
        logger.info(f"  Backbone frozen: {freeze_backbone}")
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model.
        
        Args:
            images: Input images [batch_size, 3, height, width]
            
        Returns:
            Dictionary containing predictions and intermediate results
        """
        # Extract features from backbone
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)
        
        # Predict emotions
        emotion_outputs = self.emotion_head(features)
        
        # Handle different head types
        if isinstance(emotion_outputs, dict):
            # Combined head returns dictionary
            result = {
                'valence': emotion_outputs.get('valence'),
                'arousal': emotion_outputs.get('arousal'),
                'features': features if not self.freeze_backbone else features.detach(),
                'va_vector': emotion_outputs.get('va_vector', torch.stack([emotion_outputs['valence'], emotion_outputs['arousal']], dim=1))
            }
            
            # Add Emo8 outputs if available
            if 'emo8_logits' in emotion_outputs:
                result['emo8_logits'] = emotion_outputs['emo8_logits']
                result['emo8_probs'] = emotion_outputs['emo8_probs']
            
            return result
        else:
            # Standard head returns tuple
            valence, arousal = emotion_outputs
            
            return {
                'valence': valence,
                'arousal': arousal,
                'features': features if not self.freeze_backbone else features.detach(),
                'va_vector': torch.stack([valence, arousal], dim=1)
            }
    
    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple prediction interface.
        
        Args:
            images: Input images
            
        Returns:
            Tuple of (valence, arousal) predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images)
            return outputs['valence'], outputs['arousal']
    
    def predict_single(self, image: torch.Tensor) -> Dict[str, float]:
        """
        Predict for a single image.
        
        Args:
            image: Single image tensor [3, height, width]
            
        Returns:
            Dictionary with scalar predictions
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        valence, arousal = self.predict(image)
        
        return {
            'valence': valence.item(),
            'arousal': arousal.item(),
            'quadrant': self._get_quadrant(valence.item(), arousal.item())
        }
    
    def _get_quadrant(self, valence: float, arousal: float) -> str:
        """Get emotion quadrant from V-A values."""
        if valence > 0 and arousal > 0:
            return "happy"  # High V, High A
        elif valence <= 0 and arousal > 0:
            return "angry"  # Low V, High A
        elif valence <= 0 and arousal <= 0:
            return "sad"    # Low V, Low A
        else:
            return "calm"   # High V, Low A
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of trainable parameters."""
        if self.freeze_backbone:
            return list(self.emotion_head.parameters())
        else:
            return list(self.parameters())
    
    def get_parameter_groups(self, backbone_lr: float, head_lr: float) -> List[Dict]:
        """
        Get parameter groups for differential learning rates.
        
        Args:
            backbone_lr: Learning rate for backbone parameters
            head_lr: Learning rate for head parameters
            
        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []
        
        if not self.freeze_backbone:
            # Add backbone parameters with lower learning rate
            backbone_params = list(self.backbone.parameters())
            if backbone_params:  # Only add if backbone has parameters
                param_groups.append({
                    'params': backbone_params,
                    'lr': backbone_lr,
                    'name': 'backbone'
                })
        
        # Add head parameters with higher learning rate
        head_params = list(self.emotion_head.parameters())
        param_groups.append({
            'params': head_params,
            'lr': head_lr,
            'name': 'head'
        })
        
        logger.info(f"📊 Parameter groups: backbone_lr={backbone_lr}, head_lr={head_lr}")
        return param_groups
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        logger.info("🔓 Backbone unfrozen for fine-tuning")
    
    def freeze_backbone_layers(self, layers_to_freeze: int) -> None:
        """Freeze specific number of backbone layers."""
        # This would need to be implemented based on the specific backbone architecture
        logger.info(f"🧊 Freezing {layers_to_freeze} backbone layers")
    
    def save_model(self, save_path: Union[str, Path]) -> None:
        """Save model state and configuration."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save state dict and config
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"💾 Model saved to: {save_path}")
    
    @classmethod
    def load_model(cls, load_path: Union[str, Path]) -> 'BaseVAModel':
        """Load model from checkpoint."""
        load_path = Path(load_path)
        checkpoint = torch.load(load_path, map_location='cpu')
        
        config = checkpoint['config']
        
        # Create model instance
        model = cls(
            backbone_path=config['backbone_path'],
            model_name=config['model_name'],
            feature_dim=config['feature_dim'],
            head_config=config['head_config'],
            freeze_backbone=config['freeze_backbone']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"📂 Model loaded from: {load_path}")
        return model


class SceneEmotionModel(BaseVAModel):
    """
    Scene emotion model for FindingEmo dataset.
    Optimized for scene-level emotion recognition.
    """
    
    def __init__(self,
                 backbone_path: Union[str, Path] = None,
                 feature_dim: int = 768,
                 head_config: Dict = None,
                 freeze_backbone: bool = True,
                 backbone_type: str = "dinov3",
                 clip_model_name: str = "ViT-B/32",
                 imagenet_backbone_name: str = "resnet50"):
        """
        Initialize scene emotion model.
        
        Args:
            backbone_path: Path to DINOv3 model
            feature_dim: Feature dimension
            head_config: Emotion head configuration
            freeze_backbone: Whether to freeze backbone
        """
        # Default scene-specific head configuration
        default_head_config = {
            'head_type': 'standard',
            'hidden_dims': [256, 128],
            'dropout_rate': 0.1,
            'activation': 'relu',
            'output_activation': 'tanh',
            'batch_norm': True
        }
        
        if head_config:
            default_head_config.update(head_config)
        
        super().__init__(
            backbone_path=backbone_path,
            model_name="scene_emotion_model",
            feature_dim=feature_dim,
            head_config=default_head_config,
            freeze_backbone=freeze_backbone,
            backbone_type=backbone_type,
            clip_model_name=clip_model_name,
            imagenet_backbone_name=imagenet_backbone_name
        )
        
        logger.info("🏞️  Scene emotion model ready for FindingEmo dataset")


class FaceMoodModel(BaseVAModel):
    """
    Face mood model for AffectNet dataset.
    Optimized for facial emotion recognition.
    """
    
    def __init__(self,
                 backbone_path: Union[str, Path] = None,
                 feature_dim: int = 768,
                 head_config: Dict = None,
                 freeze_backbone: bool = True,
                 backbone_type: str = "dinov3",
                 clip_model_name: str = "ViT-B/32"):
        """
        Initialize face mood model.
        
        Args:
            backbone_path: Path to DINOv3 model
            feature_dim: Feature dimension
            head_config: Emotion head configuration
            freeze_backbone: Whether to freeze backbone
        """
        # Default face-specific head configuration
        default_head_config = {
            'head_type': 'standard',
            'hidden_dims': [256, 128],
            'dropout_rate': 0.15,  # Slightly higher dropout for faces
            'activation': 'relu',
            'output_activation': 'tanh',
            'batch_norm': True
        }
        
        if head_config:
            default_head_config.update(head_config)
        
        super().__init__(
            backbone_path=backbone_path,
            model_name="face_mood_model",
            feature_dim=feature_dim,
            head_config=default_head_config,
            freeze_backbone=freeze_backbone,
            backbone_type=backbone_type,
            clip_model_name=clip_model_name
        )
        
        logger.info("😊 Face mood model ready for AffectNet dataset")


class EnsembleVAModel(nn.Module):
    """
    Ensemble model that can combine Scene and Face models for fusion.
    """
    
    def __init__(self,
                 scene_model: Optional[SceneEmotionModel] = None,
                 face_model: Optional[FaceMoodModel] = None,
                 fusion_strategy: str = "average",
                 fusion_weights: Optional[Tuple[float, float]] = None):
        """
        Initialize ensemble model.
        
        Args:
            scene_model: Scene emotion model
            face_model: Face mood model
            fusion_strategy: How to combine predictions ("average", "weighted", "learned")
            fusion_weights: Weights for weighted fusion (scene_weight, face_weight)
        """
        super().__init__()
        
        self.scene_model = scene_model
        self.face_model = face_model
        self.fusion_strategy = fusion_strategy
        self.fusion_weights = fusion_weights or (0.5, 0.5)
        
        # Learned fusion network (if using learned strategy)
        if fusion_strategy == "learned":
            input_dim = 4  # 2 predictions from each model
            self.fusion_net = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
                nn.Tanh()
            )
        
        logger.info(f"🤝 Ensemble model created with {fusion_strategy} fusion")
    
    def forward(self, scene_images: Optional[torch.Tensor] = None,
               face_images: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Args:
            scene_images: Scene images for scene model
            face_images: Face images for face model
            
        Returns:
            Dictionary with individual and fused predictions
        """
        results = {}
        predictions = []
        
        # Scene model predictions
        if self.scene_model is not None and scene_images is not None:
            scene_out = self.scene_model(scene_images)
            results['scene_valence'] = scene_out['valence']
            results['scene_arousal'] = scene_out['arousal']
            predictions.append(scene_out['va_vector'])
        
        # Face model predictions
        if self.face_model is not None and face_images is not None:
            face_out = self.face_model(face_images)
            results['face_valence'] = face_out['valence']
            results['face_arousal'] = face_out['arousal']
            predictions.append(face_out['va_vector'])
        
        # Fusion
        if len(predictions) > 1:
            fused_va = self._fuse_predictions(predictions)
            results['valence'] = fused_va[:, 0]
            results['arousal'] = fused_va[:, 1]
            results['va_vector'] = fused_va
        elif len(predictions) == 1:
            # Only one model available
            results['valence'] = predictions[0][:, 0]
            results['arousal'] = predictions[0][:, 1]
            results['va_vector'] = predictions[0]
        
        return results
    
    def _fuse_predictions(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Fuse predictions from multiple models."""
        if self.fusion_strategy == "average":
            return torch.mean(torch.stack(predictions), dim=0)
        
        elif self.fusion_strategy == "weighted":
            weights = torch.tensor(self.fusion_weights, device=predictions[0].device)
            weighted_preds = [w * pred for w, pred in zip(weights, predictions)]
            return torch.sum(torch.stack(weighted_preds), dim=0)
        
        elif self.fusion_strategy == "learned":
            # Concatenate predictions and pass through fusion network
            concat_preds = torch.cat(predictions, dim=1)
            return self.fusion_net(concat_preds)
        
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")


def create_va_model(model_type: str,
                   backbone_path: Union[str, Path] = None,
                   **kwargs) -> BaseVAModel:
    """
    Factory function to create V-A models.
    
    Args:
        model_type: Type of model ("scene", "face")
        backbone_path: Path to DINOv3 backbone
        **kwargs: Additional model arguments
        
    Returns:
        V-A model instance
    """
    model_classes = {
        'scene': SceneEmotionModel,
        'face': FaceMoodModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = model_classes[model_type]
    return model_class(backbone_path=backbone_path, **kwargs)


if __name__ == "__main__":
    # Test V-A models
    print("🧪 Testing V-A Models")
    print("=" * 40)
    
    # Test model paths (update these to your actual paths)
    backbone_path = "<PATH-HERE>/dinov3_convnext_tiny"
    
    if not Path(backbone_path).exists():
        print("⚠️  Backbone path not found, using mock model")
        backbone_path = "/tmp/mock_backbone"  # Will create mock model
    
    # Test configurations
    test_configs = [
        ("Scene Model", "scene", {}),
        ("Face Model", "face", {}),
        ("Scene with Custom Head", "scene", {
            'head_config': {
                'head_type': 'multi_head',
                'hidden_dims': [512, 256]
            }
        })
    ]
    
    batch_size = 4
    test_images = torch.randn(batch_size, 3, 224, 224)
    
    for name, model_type, kwargs in test_configs:
        print(f"\n🔧 Testing {name}:")
        print("-" * 25)
        
        try:
            # Create model
            model = create_va_model(
                model_type=model_type,
                backbone_path=backbone_path,
                **kwargs
            )
            
            # Test forward pass
            with torch.no_grad():
                outputs = model(test_images)
            
            print(f"  Input shape: {test_images.shape}")
            print(f"  Valence shape: {outputs['valence'].shape}")
            print(f"  Arousal shape: {outputs['arousal'].shape}")
            print(f"  Valence range: [{outputs['valence'].min():.3f}, {outputs['valence'].max():.3f}]")
            print(f"  Arousal range: [{outputs['arousal'].min():.3f}, {outputs['arousal'].max():.3f}]")
            
            # Test single prediction
            single_pred = model.predict_single(test_images[0])
            print(f"  Single prediction: V={single_pred['valence']:.3f}, A={single_pred['arousal']:.3f}, Q={single_pred['quadrant']}")
            
            # Count parameters
            trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Trainable params: {trainable_params:,}/{total_params:,}")
            
            print(f"  ✅ {name} passed")
            
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
    
    print(f"\n🎯 V-A model testing completed!")
