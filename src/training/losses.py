"""
Loss functions for Scene Model V-A prediction and Emo8 classification.
Implements UCE (Unbalanced Cross Entropy) and WMSE (Weighted MSE) as per research findings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class UnbalancedCrossEntropyLoss(nn.Module):
    """
    Unbalanced Cross Entropy (UCE) loss for handling class imbalance in Emo8 classification.
    
    Research findings show that Joy/Anticipation are overrepresented while 
    Surprise/Disgust are underrepresented in FindingEmo dataset.
    """
    
    def __init__(self, 
                 class_weights: Optional[Union[torch.Tensor, List[float]]] = None,
                 weight_mode: str = 'inverse_freq',
                 smooth_factor: float = 0.1,
                 ignore_index: int = -100):
        """
        Initialize UCE loss.
        
        Args:
            class_weights: Explicit class weights (optional)
            weight_mode: How to compute weights ('inverse_freq', 'balanced', 'focal')
            smooth_factor: Smoothing factor for weights
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        
        self.weight_mode = weight_mode
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        
        logger.info(f"🎯 UCE Loss initialized with weight_mode: {weight_mode}")
    
    def compute_class_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute class weights based on target distribution.
        
        Args:
            targets: Target labels [batch_size]
            
        Returns:
            Class weights [num_classes]
        """
        device = targets.device
        num_classes = 8  # Emo8 classes
        
        # Count class frequencies
        class_counts = torch.zeros(num_classes, device=device)
        for i in range(num_classes):
            class_counts[i] = (targets == i).sum().float()
        
        # Add smoothing to avoid division by zero
        class_counts = class_counts + self.smooth_factor
        total_samples = class_counts.sum()
        
        if self.weight_mode == 'inverse_freq':
            # Inverse frequency weighting
            weights = total_samples / (num_classes * class_counts)
        elif self.weight_mode == 'balanced':
            # Sklearn-style balanced weighting
            weights = total_samples / (num_classes * class_counts)
        elif self.weight_mode == 'focal':
            # Focal loss style weighting (less aggressive)
            freqs = class_counts / total_samples
            weights = 1.0 / (freqs + 1e-8)
            weights = torch.pow(weights, 0.5)  # Square root for gentler weighting
        else:
            weights = torch.ones(num_classes, device=device)
        
        # Normalize weights
        weights = weights / weights.mean()
        
        return weights
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            logits: Model logits [batch_size, num_classes]
            targets: Target labels [batch_size]
            
        Returns:
            UCE loss value
        """
        if self.class_weights is None:
            # Compute weights dynamically
            weights = self.compute_class_weights(targets)
        else:
            weights = self.class_weights.to(logits.device)
        
        # Apply weighted cross entropy
        loss = F.cross_entropy(
            logits, 
            targets, 
            weight=weights, 
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        
        return loss


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE (WMSE) loss for V-A regression with ambiguity and age-based weighting.
    
    Research findings suggest down-weighting ambiguous samples and considering
    age-group variations in emotional expression.
    """
    
    def __init__(self,
                 use_ambiguity_weights: bool = True,
                 use_age_weights: bool = True,
                 ambiguity_weight_factor: float = 0.5,
                 age_weight_factors: Optional[Dict[str, float]] = None):
        """
        Initialize WMSE loss.
        
        Args:
            use_ambiguity_weights: Whether to down-weight ambiguous samples
            use_age_weights: Whether to apply age-based weighting
            ambiguity_weight_factor: Weight factor for ambiguous samples (0-1)
            age_weight_factors: Weight factors for different age groups
        """
        super().__init__()
        
        self.use_ambiguity_weights = use_ambiguity_weights
        self.use_age_weights = use_age_weights
        self.ambiguity_weight_factor = ambiguity_weight_factor
        
        # Default age weights (research shows age-dependent emotion expression)
        self.age_weight_factors = age_weight_factors or {
            'child': 1.2,      # Higher weight - clearer emotional expressions
            'young_adult': 1.0,  # Baseline
            'adult': 1.0,      # Baseline  
            'elderly': 0.9     # Lower weight - more subtle expressions
        }
        
        logger.info(f"📊 WMSE Loss initialized:")
        logger.info(f"   Ambiguity weighting: {use_ambiguity_weights}")
        logger.info(f"   Age weighting: {use_age_weights}")
    
    def compute_sample_weights(self, 
                              batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute per-sample weights based on metadata.
        
        Args:
            batch: Batch dictionary containing metadata
            
        Returns:
            Sample weights [batch_size]
        """
        batch_size = len(batch['valence'])
        device = batch['valence'].device
        weights = torch.ones(batch_size, device=device)
        
        metadata = batch.get('metadata', {})
        
        # Ambiguity-based weighting
        if self.use_ambiguity_weights and 'ambiguity' in metadata:
            ambiguity_scores = metadata['ambiguity']
            if isinstance(ambiguity_scores, (list, tuple)):
                ambiguity_scores = torch.tensor(ambiguity_scores, device=device)
            elif torch.is_tensor(ambiguity_scores):
                ambiguity_scores = ambiguity_scores.to(device)
            
            # Down-weight highly ambiguous samples
            # Ambiguity typically ranges 0-1, with 1 being most ambiguous
            ambiguity_weights = 1.0 - (ambiguity_scores * (1.0 - self.ambiguity_weight_factor))
            weights *= ambiguity_weights
        
        # Age-based weighting
        if self.use_age_weights and 'age_group' in metadata:
            age_groups = metadata['age_group']
            for i, age_group in enumerate(age_groups):
                if age_group in self.age_weight_factors:
                    weights[i] *= self.age_weight_factors[age_group]
        
        return weights
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                batch: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for WMSE loss.
        
        Args:
            predictions: Model predictions {'valence': tensor, 'arousal': tensor}
            targets: Ground truth targets {'valence': tensor, 'arousal': tensor}
            batch: Full batch for metadata (optional)
            
        Returns:
            Dictionary with loss components
        """
        pred_valence = predictions['valence']
        pred_arousal = predictions['arousal']
        true_valence = targets['valence']
        true_arousal = targets['arousal']
        
        # Compute base MSE losses
        valence_mse = F.mse_loss(pred_valence, true_valence, reduction='none')
        arousal_mse = F.mse_loss(pred_arousal, true_arousal, reduction='none')
        
        # Compute sample weights if batch metadata is provided
        if batch is not None:
            weights = self.compute_sample_weights(batch)
            valence_mse = valence_mse * weights
            arousal_mse = arousal_mse * weights
        
        # Compute weighted losses
        valence_loss = valence_mse.mean()
        arousal_loss = arousal_mse.mean()
        total_loss = valence_loss + arousal_loss
        
        return {
            'total_loss': total_loss,
            'valence_loss': valence_loss,
            'arousal_loss': arousal_loss,
            'valence_mse': valence_mse.mean(),
            'arousal_mse': arousal_mse.mean()
        }


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance (alternative to UCE).
    
    Focal Loss = -alpha * (1-p)^gamma * log(p)
    where p is the predicted probability for the true class.
    """
    
    def __init__(self, 
                 alpha: Optional[Union[float, torch.Tensor]] = 1.0,
                 gamma: float = 2.0,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
            class_weights: Per-class weights
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            logits: Model logits [batch_size, num_classes]
            targets: Target labels [batch_size]
            
        Returns:
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        
        # Compute p_t (predicted probability for true class)
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha.gather(0, targets)
        else:
            alpha_t = self.alpha
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        return focal_loss.mean()


class CombinedVALoss(nn.Module):
    """
    Combined loss for multi-task learning: V-A regression + Emo8 classification.
    """
    
    def __init__(self,
                 va_loss_weight: float = 1.0,
                 emo8_loss_weight: float = 0.1,
                 use_focal_loss: bool = False):
        """
        Initialize combined loss.
        
        Args:
            va_loss_weight: Weight for V-A regression loss
            emo8_loss_weight: Weight for Emo8 classification loss
            use_focal_loss: Whether to use focal loss instead of UCE
        """
        super().__init__()
        
        self.va_loss_weight = va_loss_weight
        self.emo8_loss_weight = emo8_loss_weight
        
        # V-A regression loss
        self.va_loss = WeightedMSELoss()
        
        # Emo8 classification loss
        if use_focal_loss:
            self.emo8_loss = FocalLoss()
        else:
            self.emo8_loss = UnbalancedCrossEntropyLoss()
        
        logger.info(f"🎯 Combined VA Loss initialized:")
        logger.info(f"   V-A weight: {va_loss_weight}")
        logger.info(f"   Emo8 weight: {emo8_loss_weight}")
        logger.info(f"   Using focal loss: {use_focal_loss}")
    
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                batch: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for combined loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets  
            batch: Full batch for metadata
            
        Returns:
            Dictionary with loss components
        """
        results = {}
        
        # V-A regression loss
        va_targets = {
            'valence': targets['valence'],
            'arousal': targets['arousal']
        }
        va_preds = {
            'valence': predictions['valence'],
            'arousal': predictions['arousal']
        }
        
        va_loss_dict = self.va_loss(va_preds, va_targets, batch)
        results.update({f'va_{k}': v for k, v in va_loss_dict.items()})
        
        # Emo8 classification loss (if available)
        if 'emo8_logits' in predictions and 'emo8_label' in targets:
            emo8_loss = self.emo8_loss(predictions['emo8_logits'], targets['emo8_label'])
            results['emo8_loss'] = emo8_loss
        else:
            results['emo8_loss'] = torch.tensor(0.0, device=predictions['valence'].device)
        
        # Combined loss
        total_loss = (self.va_loss_weight * va_loss_dict['total_loss'] + 
                     self.emo8_loss_weight * results['emo8_loss'])
        results['total_loss'] = total_loss
        
        return results


def create_loss_function(loss_config: Dict[str, any]) -> nn.Module:
    """
    Factory function to create loss functions from configuration.
    
    Args:
        loss_config: Loss configuration dictionary
        
    Returns:
        Configured loss function
    """
    loss_type = loss_config.get('type', 'combined')
    
    if loss_type == 'combined':
        return CombinedVALoss(
            va_loss_weight=loss_config.get('va_loss_weight', 1.0),
            emo8_loss_weight=loss_config.get('emo8_loss_weight', 0.1),
            use_focal_loss=loss_config.get('use_focal_loss', False)
        )
    elif loss_type == 'wmse':
        return WeightedMSELoss(
            use_ambiguity_weights=loss_config.get('use_ambiguity_weights', True),
            use_age_weights=loss_config.get('use_age_weights', True)
        )
    elif loss_type == 'uce':
        return UnbalancedCrossEntropyLoss(
            weight_mode=loss_config.get('weight_mode', 'inverse_freq')
        )
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=loss_config.get('alpha', 1.0),
            gamma=loss_config.get('gamma', 2.0)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("🧪 Testing Loss Functions")
    print("=" * 50)
    
    batch_size = 16
    num_classes = 8
    device = torch.device('cpu')
    
    # Create synthetic data
    predictions = {
        'valence': torch.randn(batch_size, device=device),
        'arousal': torch.randn(batch_size, device=device),
        'emo8_logits': torch.randn(batch_size, num_classes, device=device)
    }
    
    targets = {
        'valence': torch.randn(batch_size, device=device),
        'arousal': torch.randn(batch_size, device=device),
        'emo8_label': torch.randint(0, num_classes, (batch_size,), device=device)
    }
    
    batch = {
        'valence': targets['valence'],
        'arousal': targets['arousal'],
        'metadata': {
            'ambiguity': torch.rand(batch_size),
            'age_group': ['young_adult'] * batch_size
        }
    }
    
    # Test UCE Loss
    print("🎯 Testing UCE Loss:")
    uce_loss = UnbalancedCrossEntropyLoss()
    uce_value = uce_loss(predictions['emo8_logits'], targets['emo8_label'])
    print(f"  UCE Loss: {uce_value.item():.4f}")
    
    # Test WMSE Loss
    print("\n📊 Testing WMSE Loss:")
    wmse_loss = WeightedMSELoss()
    wmse_dict = wmse_loss(predictions, targets, batch)
    print(f"  Total Loss: {wmse_dict['total_loss'].item():.4f}")
    print(f"  Valence Loss: {wmse_dict['valence_loss'].item():.4f}")
    print(f"  Arousal Loss: {wmse_dict['arousal_loss'].item():.4f}")
    
    # Test Combined Loss
    print("\n🎭 Testing Combined Loss:")
    combined_loss = CombinedVALoss()
    combined_dict = combined_loss(predictions, targets, batch)
    print(f"  Total Loss: {combined_dict['total_loss'].item():.4f}")
    print(f"  V-A Loss: {combined_dict['va_total_loss'].item():.4f}")
    print(f"  Emo8 Loss: {combined_dict['emo8_loss'].item():.4f}")
    
    # Test Focal Loss
    print("\n🔥 Testing Focal Loss:")
    focal_loss = FocalLoss()
    focal_value = focal_loss(predictions['emo8_logits'], targets['emo8_label'])
    print(f"  Focal Loss: {focal_value.item():.4f}")
    
    print("\n✅ All loss functions working correctly!")
