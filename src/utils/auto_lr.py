"""
Automatic Learning Rate Detection for V-A Models.
Production-ready implementation for finding optimal learning rates.
"""

import torch
import torch.nn as nn
from typing import Union, Dict, Optional, Tuple
import logging
from pathlib import Path
import numpy as np

from .lr_finder import LRFinder

logger = logging.getLogger(__name__)


class AutoLRDetector:
    """
    Automatic learning rate detection based on model and training configuration.
    Uses empirical rules and optional LR finder for optimal learning rate selection.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 criterion: nn.Module,
                 device: torch.device):
        """
        Initialize auto LR detector.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            device: Training device
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        
        # Empirical learning rate rules based on research and best practices
        self.empirical_rules = {
            'frozen_backbone': {
                'clip': {
                    'base_lr': 1e-4,
                    'range': (5e-5, 5e-4),
                    'batch_size_factor': 1.0
                },
                'dinov3': {
                    'base_lr': 1e-4,
                    'range': (5e-5, 5e-4),
                    'batch_size_factor': 1.0
                },
                'imagenet': {
                    # Linear probe head on frozen CNN features typically tolerates higher LR
                    'base_lr': 3e-3,
                    'range': (1e-4, 1e-2),
                    'batch_size_factor': 1.0
                }
            },
            'unfrozen_backbone': {
                'clip': {
                    'base_lr': 1e-5,
                    'range': (5e-6, 5e-5),
                    'batch_size_factor': 0.5
                },
                'dinov3': {
                    'base_lr': 1e-5,
                    'range': (5e-6, 5e-5),
                    'batch_size_factor': 0.5
                },
                'imagenet': {
                    # When unfreezing CNN, use conservative LR
                    'base_lr': 1e-4,
                    'range': (5e-5, 5e-4),
                    'batch_size_factor': 0.75
                }
            }
        }
    
    def detect_optimal_lr(self,
                         backbone_type: str,
                         freeze_backbone: bool,
                         batch_size: int = 32,
                         train_loader: Optional[torch.utils.data.DataLoader] = None,
                         use_lr_finder: bool = False,
                         lr_finder_kwargs: Optional[Dict] = None) -> Dict[str, Union[float, str]]:
        """
        Detect optimal learning rate using empirical rules or LR finder.
        
        Args:
            backbone_type: Type of backbone ('clip', 'dinov3')
            freeze_backbone: Whether backbone is frozen
            batch_size: Training batch size
            train_loader: Data loader for LR finder (optional)
            use_lr_finder: Whether to use LR finder instead of empirical rules
            lr_finder_kwargs: Additional arguments for LR finder
            
        Returns:
            Dictionary with optimal learning rate and method used
        """
        # Determine training mode
        training_mode = 'frozen_backbone' if freeze_backbone else 'unfrozen_backbone'
        
        # Get empirical recommendation
        empirical_lr = self._get_empirical_lr(backbone_type, training_mode, batch_size)
        
        if use_lr_finder and train_loader is not None:
            # Use LR finder for more precise detection
            logger.info("🔍 Using LR Finder for automatic learning rate detection...")
            finder_lr = self._run_lr_finder(train_loader, lr_finder_kwargs or {})
            
            # Validate LR finder result against empirical bounds
            empirical_range = self.empirical_rules[training_mode][backbone_type]['range']
            
            if empirical_range[0] <= finder_lr <= empirical_range[1]:
                optimal_lr = finder_lr
                method = "lr_finder"
                logger.info(f"✅ LR Finder result ({finder_lr:.2e}) within empirical range")
            else:
                optimal_lr = empirical_lr
                method = "empirical_fallback"
                logger.warning(f"⚠️ LR Finder result ({finder_lr:.2e}) outside safe range, using empirical value")
        else:
            # Use empirical rules
            optimal_lr = empirical_lr
            method = "empirical"
            logger.info(f"📊 Using empirical learning rate: {optimal_lr:.2e}")
        
        return {
            'learning_rate': optimal_lr,
            'method': method,
            'backbone_type': backbone_type,
            'frozen': freeze_backbone,
            'batch_size': batch_size,
            'empirical_range': self.empirical_rules[training_mode][backbone_type]['range']
        }
    
    def _get_empirical_lr(self, backbone_type: str, training_mode: str, batch_size: int) -> float:
        """Get empirical learning rate based on configuration."""
        try:
            rule = self.empirical_rules[training_mode][backbone_type]
            base_lr = rule['base_lr']
            batch_factor = rule['batch_size_factor']
            
            # Adjust for batch size (linear scaling rule)
            if batch_size != 32:  # Reference batch size
                batch_scale = (batch_size / 32) ** 0.5  # Square root scaling for stability
                adjusted_lr = base_lr * batch_scale * batch_factor
            else:
                adjusted_lr = base_lr
            
            # Clamp to safe range
            min_lr, max_lr = rule['range']
            adjusted_lr = max(min_lr, min(max_lr, adjusted_lr))
            
            logger.info(f"📈 Empirical LR calculation:")
            logger.info(f"   Base LR: {base_lr:.2e}")
            logger.info(f"   Batch size: {batch_size} (factor: {batch_factor})")
            logger.info(f"   Final LR: {adjusted_lr:.2e}")
            
            return adjusted_lr
            
        except KeyError:
            logger.error(f"Unknown backbone type or training mode: {backbone_type}, {training_mode}")
            # Fallback to conservative value
            return 1e-5 if training_mode == 'unfrozen_backbone' else 1e-4
    
    def _run_lr_finder(self, 
                      train_loader: torch.utils.data.DataLoader,
                      kwargs: Dict) -> float:
        """Run LR finder and return optimal learning rate."""
        # Create temporary optimizer for LR finding
        temp_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=kwargs.get('weight_decay', 1e-4)
        )
        
        # Initialize LR finder
        lr_finder = LRFinder(self.model, temp_optimizer, self.criterion, self.device)
        
        # Run range test with conservative settings
        start_lr = kwargs.get('start_lr', 1e-7)
        end_lr = kwargs.get('end_lr', 1e-2)
        num_iter = kwargs.get('num_iter', 30)
        
        try:
            lr_finder.range_test(
                train_loader,
                start_lr=start_lr,
                end_lr=end_lr,
                num_iter=num_iter,
                smooth_f=0.05,
                diverge_th=4
            )
            
            # Get suggestion using steepest descent method
            suggested_lr = lr_finder.suggest_lr(method='steepest')
            logger.info(f"🎯 LR Finder suggested: {suggested_lr:.2e}")
            
            return suggested_lr
            
        except Exception as e:
            logger.error(f"❌ LR Finder failed: {e}")
            logger.info("🔄 Falling back to empirical method")
            raise
    
    def get_lr_schedule_config(self, 
                             optimal_lr: float,
                             freeze_backbone: bool,
                             num_epochs: int = 100) -> Dict:
        """
        Get recommended learning rate schedule configuration.
        
        Args:
            optimal_lr: Optimal learning rate from detection
            freeze_backbone: Whether backbone is frozen
            num_epochs: Total training epochs
            
        Returns:
            Dictionary with schedule configuration
        """
        if freeze_backbone:
            # Frozen backbone - can use more aggressive scheduling
            config = {
                'scheduler': 'cosine',
                'warmup_epochs': 2,
                'min_lr': optimal_lr / 100,
                'eta_min': optimal_lr / 100,
                'T_max': num_epochs - 2  # Exclude warmup epochs
            }
        else:
            # Unfrozen backbone - more conservative scheduling
            config = {
                'scheduler': 'cosine',
                'warmup_epochs': 5,
                'min_lr': optimal_lr / 1000,
                'eta_min': optimal_lr / 1000,
                'T_max': num_epochs - 5  # Exclude warmup epochs
            }
        
        logger.info(f"📈 Recommended LR schedule:")
        logger.info(f"   Scheduler: {config['scheduler']}")
        logger.info(f"   Warmup epochs: {config['warmup_epochs']}")
        logger.info(f"   Min LR: {config['min_lr']:.2e}")
        
        return config


def auto_detect_learning_rate(model: nn.Module,
                            backbone_type: str,
                            freeze_backbone: bool,
                            batch_size: int,
                            device: torch.device,
                            criterion: Optional[nn.Module] = None,
                            train_loader: Optional[torch.utils.data.DataLoader] = None,
                            use_lr_finder: bool = False,
                            **kwargs) -> Tuple[float, Dict]:
    """
    Convenience function for automatic learning rate detection.
    
    Args:
        model: PyTorch model
        backbone_type: Type of backbone ('clip', 'dinov3')
        freeze_backbone: Whether backbone is frozen
        batch_size: Training batch size
        device: Training device
        criterion: Loss function (defaults to MSELoss)
        train_loader: Data loader for LR finder (optional)
        use_lr_finder: Whether to use LR finder
        **kwargs: Additional arguments for LR finder
        
    Returns:
        Tuple of (optimal_learning_rate, detection_info)
    """
    if criterion is None:
        criterion = nn.MSELoss()
    
    detector = AutoLRDetector(model, criterion, device)
    
    result = detector.detect_optimal_lr(
        backbone_type=backbone_type,
        freeze_backbone=freeze_backbone,
        batch_size=batch_size,
        train_loader=train_loader,
        use_lr_finder=use_lr_finder,
        lr_finder_kwargs=kwargs
    )
    
    # Also get schedule recommendation
    schedule_config = detector.get_lr_schedule_config(
        optimal_lr=result['learning_rate'],
        freeze_backbone=freeze_backbone,
        num_epochs=kwargs.get('num_epochs', 100)
    )
    
    result['schedule_config'] = schedule_config
    
    return result['learning_rate'], result


if __name__ == "__main__":
    print("Auto LR Detection utility - import and use with your training script")
