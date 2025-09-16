"""
Image transformation pipelines for V-A emotion prediction.
Supports dataset-specific augmentations for face and scene images.
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image, ImageFilter
import random
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Backbone-specific normalization constants
BACKBONE_NORMALIZATION = {
    'imagenet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'places365': {
        'mean': [0.485, 0.456, 0.406], 
        'std': [0.229, 0.224, 0.225]
    },
    'clip': {
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711]
    },
    'dinov3': {
        # DINOv3 uses ImageNet normalization
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}


class AspectRatioPreservingResize:
    """
    Resize image to target size while preserving aspect ratio and padding with black borders.
    Based on research specifications for ImageNet and DINOv3 models.
    """
    
    def __init__(self, target_size: Tuple[int, int], fill: int = 0):
        """
        Initialize aspect ratio preserving resize.
        
        Args:
            target_size: Target (width, height) size
            fill: Fill value for padding (0 for black)
        """
        self.target_width, self.target_height = target_size
        self.fill = fill
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Resize image preserving aspect ratio with padding.
        
        Args:
            image: Input PIL image
            
        Returns:
            Resized and padded image
        """
        # Calculate scaling factor to fit within target size
        img_width, img_height = image.size
        scale_w = self.target_width / img_width
        scale_h = self.target_height / img_height
        scale = min(scale_w, scale_h)  # Use smaller scale to fit within bounds
        
        # Calculate new size
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with target size and fill color
        result = Image.new(image.mode, (self.target_width, self.target_height), self.fill)
        
        # Calculate position to center the resized image
        x = (self.target_width - new_width) // 2
        y = (self.target_height - new_height) // 2
        
        # Paste resized image onto result
        result.paste(resized, (x, y))
        
        return result


class DenormalizeTransform:
    """Denormalize images for visualization."""
    
    def __init__(self, mean: List[float], std: List[float]):
        """
        Initialize denormalization transform.
        
        Args:
            mean: Normalization mean values
            std: Normalization std values
        """
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize tensor.
        
        Args:
            tensor: Normalized tensor
            
        Returns:
            Denormalized tensor
        """
        return tensor * self.std + self.mean


class RandomGaussianBlur:
    """Apply Gaussian blur with specified probability."""
    
    def __init__(self, probability: float = 0.1, radius_range: Tuple[float, float] = (0.1, 2.0)):
        """
        Initialize Gaussian blur transform.
        
        Args:
            probability: Probability of applying blur
            radius_range: Range of blur radius
        """
        self.probability = probability
        self.radius_range = radius_range
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply Gaussian blur with probability.
        
        Args:
            image: Input PIL image
            
        Returns:
            Potentially blurred image
        """
        if random.random() < self.probability:
            radius = random.uniform(*self.radius_range)
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image


class RandomPerspective:
    """Apply random perspective transformation."""
    
    def __init__(self, distortion_scale: float = 0.2, p: float = 0.2):
        """
        Initialize perspective transform.
        
        Args:
            distortion_scale: Distortion scale factor
            p: Probability of applying transform
        """
        self.distortion_scale = distortion_scale
        self.p = p
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply random perspective with probability.
        
        Args:
            image: Input PIL image
            
        Returns:
            Potentially transformed image
        """
        if random.random() < self.p:
            width, height = image.size
            
            # Calculate perspective points
            half_height = height // 2
            half_width = width // 2
            topleft = (
                random.randint(0, int(self.distortion_scale * half_width)),
                random.randint(0, int(self.distortion_scale * half_height))
            )
            topright = (
                random.randint(width - int(self.distortion_scale * half_width) - 1, width - 1),
                random.randint(0, int(self.distortion_scale * half_height))
            )
            botright = (
                random.randint(width - int(self.distortion_scale * half_width) - 1, width - 1),
                random.randint(height - int(self.distortion_scale * half_height) - 1, height - 1)
            )
            botleft = (
                random.randint(0, int(self.distortion_scale * half_width)),
                random.randint(height - int(self.distortion_scale * half_height) - 1, height - 1)
            )
            
            startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
            endpoints = [topleft, topright, botright, botleft]
            
            return F.perspective(image, startpoints, endpoints)
        return image


class FaceCropTransform:
    """Crop face region with margin."""
    
    def __init__(self, margin: float = 0.1):
        """
        Initialize face crop transform.
        
        Args:
            margin: Margin around face region (as fraction of face size)
        """
        self.margin = margin
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply face cropping (placeholder - in practice would use face detection).
        
        Args:
            image: Input PIL image
            
        Returns:
            Cropped image (currently returns center crop as placeholder)
        """
        # Placeholder implementation - center crop
        # In practice, this would use face detection to find face bbox
        width, height = image.size
        min_dim = min(width, height)
        
        # Calculate crop region with margin
        crop_size = int(min_dim * (1 + self.margin))
        crop_size = min(crop_size, min(width, height))
        
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        return image.crop((left, top, right, bottom))


def get_backbone_normalization(backbone_type: str) -> Dict[str, List[float]]:
    """
    Get normalization parameters for specific backbone.
    
    Args:
        backbone_type: Type of backbone ('imagenet', 'places365', 'clip', 'dinov3')
        
    Returns:
        Dictionary with 'mean' and 'std' normalization parameters
    """
    if backbone_type not in BACKBONE_NORMALIZATION:
        logger.warning(f"Unknown backbone type '{backbone_type}', using ImageNet normalization")
        backbone_type = 'imagenet'
    
    return BACKBONE_NORMALIZATION[backbone_type]


def create_backbone_transforms(backbone_type: str = 'imagenet',
                              dataset_type: str = 'scene') -> transforms.Compose:
    """
    Create backbone-specific preprocessing transforms based on research specifications.
    
    Args:
        backbone_type: Type of backbone ('imagenet', 'places365', 'clip', 'dinov3')
        dataset_type: Type of dataset ('scene' or 'face') for appropriate resizing
        
    Returns:
        Backbone-specific transforms following research paper specifications
    """
    norm_params = get_backbone_normalization(backbone_type)
    
    if backbone_type == 'clip':
        # CLIP: Use default CLIP preprocessing chain
        # CLIP typically expects square images, so we'll use standard square resize
        transform_list = [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_params['mean'], std=norm_params['std'])
        ]
    
    elif backbone_type == 'dinov3':
        # DINOv3: Same as ImageNet but with 798x602 rescaling
        if dataset_type == 'scene':
            transform_list = [
                AspectRatioPreservingResize((798, 602)),  # DINOv3 specific size
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_params['mean'], std=norm_params['std'])
            ]
        else:
            # For face images, use standard square resize
            transform_list = [
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_params['mean'], std=norm_params['std'])
            ]
    
    elif backbone_type in ['imagenet', 'places365']:
        # ImageNet/Places365: 800x600 resolution, keep aspect ratio, center and pad with black borders
        if dataset_type == 'scene':
            transform_list = [
                AspectRatioPreservingResize((800, 600)),  # ImageNet specification
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_params['mean'], std=norm_params['std'])
            ]
        else:
            # For face images, use standard square resize
            transform_list = [
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_params['mean'], std=norm_params['std'])
            ]
    
    else:
        # Default fallback
        logger.warning(f"Unknown backbone type '{backbone_type}', using default preprocessing")
        transform_list = [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_params['mean'], std=norm_params['std'])
        ]
    
    return transforms.Compose(transform_list)


def create_base_transforms(image_size: int = 224,
                          mean: Optional[List[float]] = None,
                          std: Optional[List[float]] = None,
                          backbone_type: str = 'imagenet') -> transforms.Compose:
    """
    Create basic preprocessing transforms (resize, normalize).
    
    Args:
        image_size: Target image size
        mean: Normalization mean (backbone default if None)
        std: Normalization std (backbone default if None)
        backbone_type: Backbone type for default normalization
        
    Returns:
        Composed transforms
    """
    if mean is None or std is None:
        norm_params = get_backbone_normalization(backbone_type)
        mean = mean or norm_params['mean']
        std = std or norm_params['std']
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def create_face_transforms(image_size: int = 224,
                          is_training: bool = True,
                          augmentation_config: Optional[Dict[str, Any]] = None,
                          backbone_type: str = 'imagenet') -> transforms.Compose:
    """
    Create transforms for face emotion recognition with backbone-specific normalization.
    
    Args:
        image_size: Target image size
        is_training: Whether to apply training augmentations
        augmentation_config: Augmentation configuration
        backbone_type: Backbone type for normalization
        
    Returns:
        Composed transforms for faces
    """
    if augmentation_config is None:
        augmentation_config = {}
    
    transform_list = []
    
    # Base resize - for faces, we typically use square images
    transform_list.append(transforms.Resize((image_size, image_size)))
    
    if is_training:
        # Face-specific augmentations (conservative to preserve facial structure)
        
        # Horizontal flip
        if augmentation_config.get('horizontal_flip', 0.5) > 0:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=augmentation_config['horizontal_flip'])
            )
        
        # Rotation (conservative for faces)
        rotation_degrees = augmentation_config.get('rotation', 10)
        if rotation_degrees > 0:
            transform_list.append(
                transforms.RandomRotation(degrees=rotation_degrees, fill=0)
            )
        
        # Color jitter (moderate for faces)
        color_jitter = augmentation_config.get('color_jitter', {})
        if color_jitter:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=color_jitter.get('brightness', 0.25),
                    contrast=color_jitter.get('contrast', 0.25),
                    saturation=color_jitter.get('saturation', 0.15),
                    hue=color_jitter.get('hue', 0.05)
                )
            )
        
        # Gaussian blur (subtle for faces)
        blur_prob = augmentation_config.get('gaussian_blur', 0.1)
        if blur_prob > 0:
            transform_list.append(RandomGaussianBlur(probability=blur_prob))
        
        # Face crop margin (if specified)
        face_crop_margin = augmentation_config.get('face_crop_margin', 0)
        if face_crop_margin > 0:
            transform_list.append(FaceCropTransform(margin=face_crop_margin))
    
    # Convert to tensor and normalize with backbone-specific parameters
    norm_params = get_backbone_normalization(backbone_type)
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=norm_params['mean'],
            std=norm_params['std']
        )
    ])
    
    return transforms.Compose(transform_list)


def create_scene_transforms(image_size: int = 224,
                           is_training: bool = True,
                           augmentation_config: Optional[Dict[str, Any]] = None,
                           backbone_type: str = 'imagenet') -> transforms.Compose:
    """
    Create transforms for scene emotion recognition with backbone-specific normalization.
    
    Args:
        image_size: Target image size
        is_training: Whether to apply training augmentations
        augmentation_config: Augmentation configuration
        backbone_type: Backbone type for normalization
        
    Returns:
        Composed transforms for scenes
    """
    if augmentation_config is None:
        augmentation_config = {}
    
    transform_list = []
    
    # Apply backbone-specific preprocessing for scene images
    if backbone_type == 'dinov3':
        # DINOv3: 798x602 with aspect ratio preservation
        transform_list.append(AspectRatioPreservingResize((798, 602)))
    elif backbone_type in ['imagenet', 'places365']:
        # ImageNet/Places365: 800x600 with aspect ratio preservation
        transform_list.append(AspectRatioPreservingResize((800, 600)))
    elif backbone_type == 'clip':
        # CLIP: Use default CLIP preprocessing (square images)
        transform_list.append(transforms.Resize((224, 224), antialias=True))
    else:
        # Default fallback
        transform_list.append(transforms.Resize((image_size, image_size), antialias=True))
    
    if is_training:
        # Scene-specific augmentations (more aggressive since scenes are more varied)
        
        # Random crop (for scene context variation) - only if not using aspect ratio preservation
        if backbone_type not in ['imagenet', 'places365', 'dinov3']:
            crop_scale = augmentation_config.get('crop_scale', [0.8, 1.0])
            if crop_scale and len(crop_scale) == 2:
                transform_list.append(
                    transforms.RandomResizedCrop(
                        image_size, 
                        scale=tuple(crop_scale),
                        ratio=(0.75, 1.33)  # Aspect ratio range
                    )
                )
        
        # Horizontal flip
        if augmentation_config.get('horizontal_flip', 0.5) > 0:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=augmentation_config['horizontal_flip'])
            )
        
        # Rotation (more aggressive for scenes)
        rotation_degrees = augmentation_config.get('rotation', 20)
        if rotation_degrees > 0:
            transform_list.append(
                transforms.RandomRotation(degrees=rotation_degrees, fill=0)
            )
        
        # Color jitter (more aggressive for scenes)
        color_jitter = augmentation_config.get('color_jitter', {})
        if color_jitter:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=color_jitter.get('brightness', 0.3),
                    contrast=color_jitter.get('contrast', 0.3),
                    saturation=color_jitter.get('saturation', 0.2),
                    hue=color_jitter.get('hue', 0.1)
                )
            )
        
        # Perspective transform (good for scene variety)
        perspective_scale = augmentation_config.get('perspective_transform', 0.2)
        if perspective_scale > 0:
            transform_list.append(RandomPerspective(distortion_scale=perspective_scale))
        
        # Gaussian blur
        blur_prob = augmentation_config.get('gaussian_blur', 0.1)
        if blur_prob > 0:
            transform_list.append(RandomGaussianBlur(probability=blur_prob))
    
    # Convert to tensor and normalize with backbone-specific parameters
    norm_params = get_backbone_normalization(backbone_type)
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=norm_params['mean'],
            std=norm_params['std']
        )
    ])
    
    return transforms.Compose(transform_list)


def create_transforms_from_config(data_config: Dict[str, Any],
                                 dataset_type: str,
                                 is_training: bool = True,
                                 backbone_type: str = 'imagenet') -> transforms.Compose:
    """
    Create transforms from configuration with backbone-specific preprocessing.
    
    Args:
        data_config: Data configuration dictionary
        dataset_type: Type of dataset ("affectnet" or "findingemo")
        is_training: Whether to apply training augmentations
        backbone_type: Type of backbone for normalization ('imagenet', 'places365', 'clip', 'dinov3')
        
    Returns:
        Composed transforms with backbone-specific preprocessing
    """
    image_size = data_config.get('image_size', 224)
    augmentation_config = data_config.get('augmentation', {}) if is_training else {}
    
    # Use backbone-specific transforms if specified
    use_backbone_specific = data_config.get('use_backbone_specific', True)
    
    if use_backbone_specific and not is_training:
        # For inference, use backbone-specific transforms
        return create_backbone_transforms(
            backbone_type=backbone_type,
            dataset_type=dataset_type
        )
    
    # For training or when backbone-specific is disabled, use dataset-specific transforms
    if dataset_type == "affectnet":
        # Face transforms with backbone-specific normalization
        return create_face_transforms(
            image_size=image_size,
            is_training=is_training,
            augmentation_config=augmentation_config,
            backbone_type=backbone_type
        )
    elif dataset_type == "findingemo":
        # Scene transforms with backbone-specific normalization
        return create_scene_transforms(
            image_size=image_size,
            is_training=is_training,
            augmentation_config=augmentation_config,
            backbone_type=backbone_type
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_inference_transforms(backbone_type: str = 'imagenet') -> transforms.Compose:
    """
    Create transforms for inference (no augmentation) with backbone-specific preprocessing.
    
    Args:
        backbone_type: Backbone type for appropriate preprocessing
        
    Returns:
        Composed transforms for inference
    """
    norm_params = get_backbone_normalization(backbone_type)
    
    if backbone_type == 'dinov3':
        # DINOv3: 798x602 with aspect ratio preservation
        return transforms.Compose([
            AspectRatioPreservingResize((798, 602)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_params['mean'], std=norm_params['std'])
        ])
    elif backbone_type in ['imagenet', 'places365']:
        # ImageNet/Places365: 800x600 with aspect ratio preservation
        return transforms.Compose([
            AspectRatioPreservingResize((800, 600)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_params['mean'], std=norm_params['std'])
        ])
    elif backbone_type == 'clip':
        # CLIP: Standard square resize
        return transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_params['mean'], std=norm_params['std'])
        ])
    else:
        # Default fallback
        return transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_params['mean'], std=norm_params['std'])
        ])


def create_visualization_transforms(backbone_type: str = 'imagenet') -> DenormalizeTransform:
    """
    Create transforms for visualization (denormalization) with backbone-specific parameters.
    
    Args:
        backbone_type: Backbone type for appropriate denormalization
        
    Returns:
        Denormalization transform
    """
    norm_params = get_backbone_normalization(backbone_type)
    return DenormalizeTransform(
        mean=norm_params['mean'],
        std=norm_params['std']
    )


def visualize_augmentations(image_path: str,
                           dataset_type: str,
                           backbone_type: str = 'imagenet',
                           num_samples: int = 8,
                           save_path: Optional[str] = None):
    """
    Visualize augmentations for debugging.
    
    Args:
        image_path: Path to test image
        dataset_type: Type of dataset
        backbone_type: Backbone type for appropriate preprocessing
        num_samples: Number of augmented samples to show
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    
    # Create transforms
    transform = create_transforms_from_config(
        data_config={'image_size': 224, 'augmentation': {'enabled': True}},
        dataset_type=dataset_type,
        is_training=True,
        backbone_type=backbone_type
    )
    
    # Create denormalization transform
    denorm = create_visualization_transforms(backbone_type)
    
    # Generate augmented samples
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_samples):
        # Apply transform
        transformed = transform(original_image)
        
        # Denormalize for visualization
        denormalized = denorm(transformed)
        denormalized = torch.clamp(denormalized, 0, 1)
        
        # Convert to numpy for plotting
        img_np = denormalized.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np)
        axes[i].set_title(f'Augmentation {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"🎨 Augmentation visualization saved: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test transform creation
    print("🧪 Testing Image Transforms")
    print("=" * 50)
    
    # Test face transforms
    print("👤 Testing face transforms:")
    face_config = {
        'image_size': 224,
        'augmentation': {
            'horizontal_flip': 0.5,
            'rotation': 10,
            'color_jitter': {
                'brightness': 0.25,
                'contrast': 0.25,
                'saturation': 0.15,
                'hue': 0.05
            },
            'gaussian_blur': 0.1
        }
    }
    
    face_transforms = create_transforms_from_config(
        face_config, 
        dataset_type="affectnet", 
        is_training=True
    )
    print(f"  ✅ Face transforms created: {len(face_transforms.transforms)} steps")
    
    # Test scene transforms
    print("\n🏞️  Testing scene transforms:")
    scene_config = {
        'image_size': 224,
        'augmentation': {
            'horizontal_flip': 0.5,
            'rotation': 20,
            'crop_scale': [0.8, 1.0],
            'perspective_transform': 0.2,
            'color_jitter': {
                'brightness': 0.3,
                'contrast': 0.3,
                'saturation': 0.2,
                'hue': 0.1
            }
        }
    }
    
    scene_transforms = create_transforms_from_config(
        scene_config,
        dataset_type="findingemo",
        is_training=True
    )
    print(f"  ✅ Scene transforms created: {len(scene_transforms.transforms)} steps")
    
    # Test backbone-specific transforms
    print("\n🔧 Testing backbone-specific transforms:")
    for backbone in ['imagenet', 'dinov3', 'clip', 'places365']:
        backbone_transforms = create_backbone_transforms(backbone, 'scene')
        print(f"  ✅ {backbone} transforms created: {len(backbone_transforms.transforms)} steps")
    
    # Test inference transforms
    print("\n🔮 Testing inference transforms:")
    for backbone in ['imagenet', 'dinov3', 'clip']:
        inference_transforms = create_inference_transforms(backbone)
        print(f"  ✅ {backbone} inference transforms created: {len(inference_transforms.transforms)} steps")
    
    # Test denormalization
    print("\n🎨 Testing denormalization:")
    for backbone in ['imagenet', 'dinov3', 'clip']:
        denorm = create_visualization_transforms(backbone)
        test_tensor = torch.randn(3, 224, 224)
        denormalized = denorm(test_tensor)
        print(f"  ✅ {backbone} denormalization working: {test_tensor.shape} -> {denormalized.shape}")
    
    print(f"\n🎯 Transform system ready!")
    print(f"📊 Supports research paper specifications for all backbones")
    print(f"🔧 ImageNet/DINOv3: Aspect ratio preservation with padding")
    print(f"🎨 CLIP: Default preprocessing chain")
    print(f"📐 Places365/EmoNet: Paper-specific preprocessing")