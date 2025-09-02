"""
Unified inference interface for continuous V-A emotion prediction.
Supports both Scene and Face models with consistent output format for fusion.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from dataclasses import dataclass
import time

from ..models.va_models import BaseVAModel, SceneEmotionModel, FaceMoodModel, EnsembleVAModel
from ..data.transforms import create_transforms_from_config, DenormalizeTransform
from ..utils.device import DeviceManager
from ..utils.metrics import VAMetrics

logger = logging.getLogger(__name__)


@dataclass
class VAResult:
    """
    Standardized result object for V-A predictions.
    
    Provides consistent interface for both individual and ensemble models.
    """
    valence: float
    arousal: float
    confidence: Optional[float] = None
    quadrant: Optional[str] = None
    model_type: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # Auto-compute quadrant if not provided
        if self.quadrant is None:
            self.quadrant = self._compute_quadrant()
    
    def _compute_quadrant(self) -> str:
        """Compute emotion quadrant from V-A values."""
        if self.valence > 0 and self.arousal > 0:
            return "happy"      # High V, High A (Happy/Excited)
        elif self.valence <= 0 and self.arousal > 0:
            return "angry"      # Low V, High A (Angry/Stressed)
        elif self.valence <= 0 and self.arousal <= 0:
            return "sad"        # Low V, Low A (Sad/Depressed)
        else:
            return "calm"       # High V, Low A (Calm/Content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            'valence': self.valence,
            'arousal': self.arousal,
            'quadrant': self.quadrant
        }
        
        if self.confidence is not None:
            result['confidence'] = self.confidence
        if self.model_type is not None:
            result['model_type'] = self.model_type
        if self.processing_time is not None:
            result['processing_time'] = self.processing_time
        if self.metadata is not None:
            result['metadata'] = self.metadata
            
        return result
    
    def get_va_vector(self) -> np.ndarray:
        """Get V-A as numpy array."""
        return np.array([self.valence, self.arousal])


@dataclass
class BatchVAResult:
    """Result object for batch predictions."""
    valence: np.ndarray
    arousal: np.ndarray
    quadrants: List[str]
    confidence: Optional[np.ndarray] = None
    model_type: Optional[str] = None
    processing_time: Optional[float] = None
    batch_size: Optional[int] = None
    
    def __post_init__(self):
        self.batch_size = len(self.valence)
    
    def get_result(self, index: int) -> VAResult:
        """Get single result from batch."""
        return VAResult(
            valence=float(self.valence[index]),
            arousal=float(self.arousal[index]),
            quadrant=self.quadrants[index],
            confidence=float(self.confidence[index]) if self.confidence is not None else None,
            model_type=self.model_type
        )
    
    def to_list(self) -> List[VAResult]:
        """Convert to list of VAResult objects."""
        return [self.get_result(i) for i in range(self.batch_size)]


class VAPredictor:
    """
    Unified predictor for V-A emotion prediction models.
    
    Provides consistent interface for inference with Scene and Face models.
    Supports single image and batch prediction with optional preprocessing.
    """
    
    def __init__(self,
                 model: BaseVAModel,
                 device_manager: Optional[DeviceManager] = None,
                 transform_config: Optional[Dict] = None,
                 model_type: str = "unknown"):
        """
        Initialize predictor.
        
        Args:
            model: Trained V-A prediction model
            device_manager: Device management (auto-created if None)
            transform_config: Transform configuration for preprocessing
            model_type: Type identifier for the model
        """
        self.model = model
        self.model_type = model_type
        
        # Setup device management
        if device_manager is None:
            self.device_manager = DeviceManager(device="auto", verbose=False)
        else:
            self.device_manager = device_manager
        
        # Move model to device and set to eval mode
        self.model = self.device_manager.to_device(self.model)
        self.model.eval()
        
        # Setup transforms
        self.transform = self._create_transform(transform_config)
        
        # Setup denormalization for visualization
        self.denorm_transform = DenormalizeTransform(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        logger.info(f"🔮 VAPredictor initialized for {model_type} model")
        logger.info(f"🔧 Device: {self.device_manager.device}")
    
    def _create_transform(self, transform_config: Optional[Dict]):
        """Create preprocessing transform."""
        if transform_config is None:
            # Default transform configuration
            transform_config = {
                'image_size': 224,
                'normalize_mean': [0.485, 0.456, 0.406],
                'normalize_std': [0.229, 0.224, 0.225]
            }
        
        from ..data.transforms import create_base_transforms
        return create_base_transforms(
            image_size=transform_config.get('image_size', 224),
            mean=transform_config.get('normalize_mean'),
            std=transform_config.get('normalize_std')
        )
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image in various formats
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Convert to PIL Image if needed
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            # If already a tensor, assume it's preprocessed
            if image.dim() == 3:
                image = image.unsqueeze(0)  # Add batch dimension
            return self.device_manager.to_device(image)
        
        # Apply preprocessing transform
        if self.transform:
            tensor = self.transform(image)
        else:
            # Fallback: basic preprocessing
            from torchvision.transforms import ToTensor, Normalize, Resize
            tensor = ToTensor()(image)
            tensor = Resize((224, 224))(tensor)
            tensor = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tensor)
        
        # Add batch dimension and move to device
        tensor = tensor.unsqueeze(0)
        return self.device_manager.to_device(tensor)
    
    def predict_single(self, 
                      image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
                      return_confidence: bool = False) -> VAResult:
        """
        Predict V-A values for a single image.
        
        Args:
            image: Input image
            return_confidence: Whether to compute prediction confidence
            
        Returns:
            VAResult with predictions
        """
        start_time = time.time()
        
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Extract predictions
        valence = float(outputs['valence'].cpu().item())
        arousal = float(outputs['arousal'].cpu().item())
        
        # Compute confidence if requested
        confidence = None
        if return_confidence:
            confidence = self._compute_confidence(outputs)
        
        processing_time = time.time() - start_time
        
        return VAResult(
            valence=valence,
            arousal=arousal,
            confidence=confidence,
            model_type=self.model_type,
            processing_time=processing_time
        )
    
    def predict_batch(self,
                     images: List[Union[str, Path, Image.Image, np.ndarray]],
                     batch_size: int = 32,
                     return_confidence: bool = False) -> BatchVAResult:
        """
        Predict V-A values for a batch of images.
        
        Args:
            images: List of input images
            batch_size: Processing batch size
            return_confidence: Whether to compute prediction confidence
            
        Returns:
            BatchVAResult with predictions
        """
        start_time = time.time()
        
        all_valence = []
        all_arousal = []
        all_confidence = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                tensor = self.preprocess_image(img)
                batch_tensors.append(tensor.squeeze(0))  # Remove individual batch dim
            
            # Stack into batch
            batch_tensor = torch.stack(batch_tensors)
            batch_tensor = self.device_manager.to_device(batch_tensor)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
            
            # Extract predictions
            all_valence.extend(outputs['valence'].cpu().numpy())
            all_arousal.extend(outputs['arousal'].cpu().numpy())
            
            if return_confidence:
                batch_confidence = self._compute_batch_confidence(outputs)
                all_confidence.extend(batch_confidence)
        
        # Convert to numpy arrays
        valence_array = np.array(all_valence)
        arousal_array = np.array(all_arousal)
        confidence_array = np.array(all_confidence) if all_confidence else None
        
        # Compute quadrants
        quadrants = [
            VAResult(v, a).quadrant 
            for v, a in zip(valence_array, arousal_array)
        ]
        
        processing_time = time.time() - start_time
        
        return BatchVAResult(
            valence=valence_array,
            arousal=arousal_array,
            quadrants=quadrants,
            confidence=confidence_array,
            model_type=self.model_type,
            processing_time=processing_time
        )
    
    def _compute_confidence(self, outputs: Dict[str, torch.Tensor]) -> float:
        """
        Compute prediction confidence (placeholder implementation).
        
        In a real implementation, this could use:
        - Prediction variance from multiple forward passes
        - Model uncertainty estimation
        - Distance from training distribution
        """
        # Simple confidence based on distance from center
        va_vector = outputs['va_vector'].cpu().numpy().flatten()
        distance_from_center = np.linalg.norm(va_vector)
        
        # Normalize to [0, 1] range (higher distance = lower confidence)
        confidence = 1.0 / (1.0 + distance_from_center)
        return float(confidence)
    
    def _compute_batch_confidence(self, outputs: Dict[str, torch.Tensor]) -> List[float]:
        """Compute confidence for batch predictions."""
        va_vectors = outputs['va_vector'].cpu().numpy()
        confidences = []
        
        for va_vector in va_vectors:
            distance_from_center = np.linalg.norm(va_vector)
            confidence = 1.0 / (1.0 + distance_from_center)
            confidences.append(confidence)
        
        return confidences
    
    def predict_with_visualization(self, 
                                  image: Union[str, Path, Image.Image],
                                  save_path: Optional[Union[str, Path]] = None) -> Tuple[VAResult, Image.Image]:
        """
        Predict with visualization overlay.
        
        Args:
            image: Input image
            save_path: Optional path to save visualization
            
        Returns:
            Tuple of (VAResult, visualization_image)
        """
        # Get prediction
        result = self.predict_single(image, return_confidence=True)
        
        # Create visualization
        if isinstance(image, (str, Path)):
            orig_image = Image.open(image).convert('RGB')
        else:
            orig_image = image
        
        vis_image = self._create_visualization(orig_image, result)
        
        if save_path:
            vis_image.save(save_path)
            logger.info(f"💾 Visualization saved to: {save_path}")
        
        return result, vis_image
    
    def _create_visualization(self, image: Image.Image, result: VAResult) -> Image.Image:
        """Create visualization with V-A overlay."""
        # This is a placeholder implementation
        # In practice, you would overlay the V-A predictions on the image
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from io import BytesIO
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title("Input Image")
        ax1.axis('off')
        
        # V-A plot
        ax2.scatter(result.valence, result.arousal, s=200, c='red', alpha=0.8)
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.set_xlabel('Valence')
        ax2.set_ylabel('Arousal')
        ax2.set_title(f'Prediction: {result.quadrant.title()}\nV: {result.valence:.3f}, A: {result.arousal:.3f}')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add quadrant labels
        ax2.text(0.5, 0.5, 'Happy', ha='center', va='center', alpha=0.5)
        ax2.text(-0.5, 0.5, 'Angry', ha='center', va='center', alpha=0.5)
        ax2.text(-0.5, -0.5, 'Sad', ha='center', va='center', alpha=0.5)
        ax2.text(0.5, -0.5, 'Calm', ha='center', va='center', alpha=0.5)
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        vis_image = Image.open(buf)
        plt.close()
        
        return vis_image


class ScenePredictor(VAPredictor):
    """Specialized predictor for scene emotion models."""
    
    def __init__(self, model: SceneEmotionModel, device_manager: Optional[DeviceManager] = None):
        # Scene-specific transform configuration
        transform_config = {
            'image_size': 224,
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225]
        }
        
        super().__init__(
            model=model,
            device_manager=device_manager,
            transform_config=transform_config,
            model_type="scene"
        )


class FacePredictor(VAPredictor):
    """Specialized predictor for face emotion models."""
    
    def __init__(self, model: FaceMoodModel, device_manager: Optional[DeviceManager] = None):
        # Face-specific transform configuration
        transform_config = {
            'image_size': 224,
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225]
        }
        
        super().__init__(
            model=model,
            device_manager=device_manager,
            transform_config=transform_config,
            model_type="face"
        )


class EnsemblePredictor:
    """
    Ensemble predictor that combines Scene and Face models.
    
    Provides fusion-ready interface for multi-modal emotion prediction.
    """
    
    def __init__(self,
                 scene_predictor: Optional[ScenePredictor] = None,
                 face_predictor: Optional[FacePredictor] = None,
                 fusion_strategy: str = "average",
                 fusion_weights: Optional[Tuple[float, float]] = None):
        """
        Initialize ensemble predictor.
        
        Args:
            scene_predictor: Scene emotion predictor
            face_predictor: Face emotion predictor
            fusion_strategy: Fusion method ("average", "weighted", "max_confidence")
            fusion_weights: Weights for weighted fusion (scene_weight, face_weight)
        """
        self.scene_predictor = scene_predictor
        self.face_predictor = face_predictor
        self.fusion_strategy = fusion_strategy
        self.fusion_weights = fusion_weights or (0.5, 0.5)
        
        if not scene_predictor and not face_predictor:
            raise ValueError("At least one predictor must be provided")
        
        logger.info(f"🤝 EnsemblePredictor initialized with {fusion_strategy} fusion")
    
    def predict_multimodal(self,
                          scene_image: Optional[Union[str, Path, Image.Image]] = None,
                          face_image: Optional[Union[str, Path, Image.Image]] = None,
                          return_individual: bool = True) -> Dict[str, VAResult]:
        """
        Predict using multiple modalities.
        
        Args:
            scene_image: Scene image for context emotion
            face_image: Face image for facial emotion
            return_individual: Whether to return individual predictions
            
        Returns:
            Dictionary with individual and fused predictions
        """
        results = {}
        predictions = []
        confidences = []
        
        # Scene prediction
        if self.scene_predictor and scene_image is not None:
            scene_result = self.scene_predictor.predict_single(scene_image, return_confidence=True)
            if return_individual:
                results['scene'] = scene_result
            predictions.append(scene_result)
            confidences.append(scene_result.confidence or 1.0)
        
        # Face prediction
        if self.face_predictor and face_image is not None:
            face_result = self.face_predictor.predict_single(face_image, return_confidence=True)
            if return_individual:
                results['face'] = face_result
            predictions.append(face_result)
            confidences.append(face_result.confidence or 1.0)
        
        # Fusion
        if len(predictions) > 1:
            fused_result = self._fuse_predictions(predictions, confidences)
            results['fused'] = fused_result
        elif len(predictions) == 1:
            # Only one prediction available
            results['fused'] = predictions[0]
        else:
            raise ValueError("No valid predictions available")
        
        return results
    
    def _fuse_predictions(self, predictions: List[VAResult], confidences: List[float]) -> VAResult:
        """Fuse multiple predictions into single result."""
        if self.fusion_strategy == "average":
            # Simple average
            valence = np.mean([p.valence for p in predictions])
            arousal = np.mean([p.arousal for p in predictions])
            confidence = np.mean(confidences)
        
        elif self.fusion_strategy == "weighted":
            # Weighted average using fusion weights
            weights = np.array(self.fusion_weights[:len(predictions)])
            weights = weights / weights.sum()  # Normalize
            
            valence = np.sum([w * p.valence for w, p in zip(weights, predictions)])
            arousal = np.sum([w * p.arousal for w, p in zip(weights, predictions)])
            confidence = np.sum([w * c for w, c in zip(weights, confidences)])
        
        elif self.fusion_strategy == "max_confidence":
            # Use prediction with highest confidence
            max_idx = np.argmax(confidences)
            return predictions[max_idx]
        
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        return VAResult(
            valence=float(valence),
            arousal=float(arousal),
            confidence=float(confidence),
            model_type="ensemble"
        )


def load_predictor(model_path: Union[str, Path],
                  model_type: str,
                  device: str = "auto") -> VAPredictor:
    """
    Load predictor from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model ("scene" or "face")
        device: Device for inference
        
    Returns:
        Configured predictor
    """
    device_manager = DeviceManager(device=device, verbose=False)
    
    # Load model
    if model_type == "scene":
        model = SceneEmotionModel.load_model(model_path)
        return ScenePredictor(model, device_manager)
    elif model_type == "face":
        model = FaceMoodModel.load_model(model_path)
        return FacePredictor(model, device_manager)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test inference interface
    print("🧪 Testing Inference Interface")
    print("=" * 50)
    
    # This would require actual trained models to test
    print("ℹ️  This module provides the VAPredictor, ScenePredictor, FacePredictor, and EnsemblePredictor classes")
    print("📚 Usage: Load trained models and create predictors for inference")
    print("🔮 Features: Single/batch prediction, visualization, confidence estimation, multi-modal fusion")
    
    # Test VAResult
    print("\n🧪 Testing VAResult:")
    result = VAResult(valence=0.5, arousal=-0.3, confidence=0.8, model_type="test")
    print(f"  Result: {result.to_dict()}")
    print(f"  Quadrant: {result.quadrant}")
    print(f"  V-A Vector: {result.get_va_vector()}")
    
    print("\n✅ Inference interface test completed!")
