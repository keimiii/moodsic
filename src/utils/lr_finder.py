"""
Learning Rate Finder for optimal learning rate discovery.
Implementation based on the fastai LR finder approach.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List, Dict
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class LRFinder:
    """
    Learning Rate Finder using the range test method.
    
    This implementation sweeps through learning rates exponentially from
    start_lr to end_lr and tracks the loss. The optimal learning rate
    is typically found at the steepest descent before the loss explodes.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 criterion: nn.Module,
                 device: torch.device):
        """
        Initialize LR Finder.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device to run on (CPU/GPU/MPS)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Results storage
        self.losses = []
        self.lrs = []
        self.best_loss = float('inf')
        
        # Store initial state
        self.initial_state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]['lr']
        }
    
    def range_test(self, 
                   train_loader: torch.utils.data.DataLoader,
                   start_lr: float = 1e-7,
                   end_lr: float = 10.0,
                   num_iter: int = 100,
                   smooth_f: float = 0.05,
                   diverge_th: float = 5) -> Tuple[List[float], List[float]]:
        """
        Perform the learning rate range test.
        
        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations to test
            smooth_f: Smoothing factor for loss (0 = no smoothing, 1 = only current loss)
            diverge_th: Threshold for stopping if loss diverges (multiplier of best loss)
            
        Returns:
            Tuple of (learning_rates, losses)
        """
        logger.info(f"🔍 Starting LR range test: {start_lr:.1e} to {end_lr:.1e} over {num_iter} iterations")
        
        # Reset results
        self.losses = []
        self.lrs = []
        self.best_loss = float('inf')
        
        # Set initial learning rate
        self._set_lr(start_lr)
        
        # Calculate exponential multiplier
        gamma = (end_lr / start_lr) ** (1 / num_iter)
        
        # Set model to training mode
        self.model.train()
        
        # Track smoothed loss
        avg_loss = 0.0
        beta = 1 - smooth_f
        
        for i, batch in enumerate(train_loader):
            if i >= num_iter:
                break
                
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lrs.append(current_lr)
            
            # Forward pass - handle different batch formats
            try:
                if isinstance(batch, dict):
                    inputs = batch['image']
                    # Combine valence and arousal into targets
                    valence = batch['valence'].float()
                    arousal = batch['arousal'].float()
                    targets = torch.stack([valence, arousal], dim=1)
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).float()
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                logger.error(f"Batch type: {type(batch)}")
                if isinstance(batch, dict):
                    logger.error(f"Batch keys: {batch.keys()}")
                    logger.error(f"Valence shape: {batch['valence'].shape if 'valence' in batch else 'N/A'}")
                    logger.error(f"Arousal shape: {batch['arousal'].shape if 'arousal' in batch else 'N/A'}")
                raise
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Handle model outputs (some models return dict)
            if isinstance(outputs, dict):
                if 'predictions' in outputs:
                    outputs = outputs['predictions']
                elif 'output' in outputs:
                    outputs = outputs['output']
                elif 'va_vector' in outputs:
                    # For V-A models that return separate valence/arousal
                    outputs = outputs['va_vector']
                elif 'valence' in outputs and 'arousal' in outputs:
                    # Combine valence and arousal into V-A vector
                    outputs = torch.stack([outputs['valence'], outputs['arousal']], dim=1)
                else:
                    # Try to find the tensor output
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor) and value.dim() >= 2:
                            outputs = value
                            break
            
            # Debug: check types and shapes
            if i == 0:  # Only log for first iteration
                logger.info(f"Debug - inputs type: {type(inputs)}, shape: {inputs.shape}")
                logger.info(f"Debug - targets type: {type(targets)}, shape: {targets.shape if hasattr(targets, 'shape') else 'no shape'}")
                logger.info(f"Debug - outputs type: {type(outputs)}, shape: {outputs.shape if hasattr(outputs, 'shape') else 'no shape'}")
            
            loss = self.criterion(outputs, targets)
            
            # Compute smoothed loss
            if i == 0:
                avg_loss = loss.item()
            else:
                avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            
            # Apply bias correction
            smoothed_loss = avg_loss / (1 - beta ** (i + 1))
            self.losses.append(smoothed_loss)
            
            # Check for divergence
            if smoothed_loss < self.best_loss:
                self.best_loss = smoothed_loss
            
            if smoothed_loss > diverge_th * self.best_loss:
                logger.info(f"🛑 Stopping early at iteration {i}: loss diverged")
                break
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update learning rate exponentially
            self._set_lr(current_lr * gamma)
            
            # Progress logging
            if i % max(1, num_iter // 10) == 0:
                logger.info(f"📊 Iteration {i}/{num_iter} | LR: {current_lr:.2e} | Loss: {smoothed_loss:.4f}")
        
        # Restore initial state
        self._restore_state()
        
        logger.info(f"✅ LR range test completed with {len(self.losses)} iterations")
        return self.lrs, self.losses
    
    def plot(self, 
             skip_start: int = 10, 
             skip_end: int = 5,
             log_lr: bool = True,
             show_suggested: bool = True,
             save_path: Optional[str] = None) -> None:
        """
        Plot the learning rate vs loss curve.
        
        Args:
            skip_start: Number of points to skip at the beginning
            skip_end: Number of points to skip at the end
            log_lr: Whether to use log scale for learning rate axis
            show_suggested: Whether to mark the suggested learning rate
            save_path: Path to save the plot (optional)
        """
        if not self.losses:
            logger.error("No results to plot. Run range_test() first.")
            return
        
        # Prepare data
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]
        
        plt.figure(figsize=(12, 8))
        plt.plot(lrs, losses, 'b-', linewidth=2, label='Loss')
        
        if log_lr:
            plt.xscale('log')
        
        plt.xlabel('Learning Rate', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Learning Rate Finder', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Mark suggested learning rate
        if show_suggested:
            try:
                suggested_lr = self.suggest_lr(skip_start, skip_end)
                plt.axvline(x=suggested_lr, color='red', linestyle='--', 
                           linewidth=2, label=f'Suggested LR: {suggested_lr:.2e}')
                logger.info(f"💡 Suggested learning rate: {suggested_lr:.2e}")
            except Exception as e:
                logger.warning(f"Could not calculate suggested LR: {e}")
        
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Plot saved to {save_path}")
        
        plt.show()
    
    def suggest_lr(self, 
                   skip_start: int = 10, 
                   skip_end: int = 5,
                   method: str = 'steepest') -> float:
        """
        Suggest optimal learning rate based on the loss curve.
        
        Args:
            skip_start: Number of points to skip at the beginning
            skip_end: Number of points to skip at the end
            method: Method to use ('steepest', 'minimum', 'valley')
            
        Returns:
            Suggested learning rate
        """
        if not self.losses:
            raise ValueError("No results available. Run range_test() first.")
        
        # Prepare data
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]
        
        if method == 'minimum':
            # Simple minimum loss approach
            min_idx = np.argmin(losses)
            return lrs[min_idx]
        
        elif method == 'steepest':
            # Find steepest negative gradient
            try:
                # Calculate gradients
                gradients = np.gradient(losses)
                
                # Find the steepest descent (most negative gradient)
                min_gradient_idx = np.argmin(gradients)
                
                # Use learning rate at steepest descent
                return lrs[min_gradient_idx]
            
            except Exception:
                # Fallback to numerical gradient
                gradients = []
                for i in range(1, len(losses)):
                    grad = (losses[i] - losses[i-1]) / (np.log10(lrs[i]) - np.log10(lrs[i-1]))
                    gradients.append(grad)
                
                min_gradient_idx = np.argmin(gradients)
                return lrs[min_gradient_idx + 1]
        
        elif method == 'valley':
            # Find the learning rate before the loss starts increasing significantly
            try:
                # Smooth the curve
                from scipy.signal import savgol_filter
                smoothed = savgol_filter(losses, min(11, len(losses)//4 + 1), 3)
                
                # Find where the derivative changes from negative to positive
                gradients = np.gradient(smoothed)
                
                # Find last significant decrease before increase
                for i in range(len(gradients) - 1, 0, -1):
                    if gradients[i] < -0.01:  # Significant decrease
                        return lrs[i]
                
                # Fallback to steepest method
                return self.suggest_lr(skip_start, skip_end, 'steepest')
            
            except ImportError:
                logger.warning("scipy not available, falling back to steepest method")
                return self.suggest_lr(skip_start, skip_end, 'steepest')
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def save_results(self, save_path: str) -> None:
        """Save LR finder results to JSON file."""
        results = {
            'learning_rates': self.lrs,
            'losses': self.losses,
            'suggested_lr_steepest': self.suggest_lr(method='steepest'),
            'suggested_lr_minimum': self.suggest_lr(method='minimum')
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Results saved to {save_path}")
    
    def _set_lr(self, lr: float) -> None:
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _restore_state(self) -> None:
        """Restore model and optimizer to initial state."""
        self.model.load_state_dict(self.initial_state['model'])
        self.optimizer.load_state_dict(self.initial_state['optimizer'])
        self._set_lr(self.initial_state['lr'])


def find_optimal_lr(model: nn.Module,
                   train_loader: torch.utils.data.DataLoader,
                   criterion: nn.Module,
                   device: torch.device,
                   optimizer_class: type = torch.optim.AdamW,
                   optimizer_kwargs: Optional[Dict] = None,
                   **lr_finder_kwargs) -> float:
    """
    Convenience function to find optimal learning rate.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        device: Device to run on
        optimizer_class: Optimizer class to use
        optimizer_kwargs: Additional arguments for optimizer
        **lr_finder_kwargs: Arguments for LRFinder.range_test()
        
    Returns:
        Suggested optimal learning rate
    """
    # Create temporary optimizer
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    
    temp_optimizer = optimizer_class(model.parameters(), lr=1e-3, **optimizer_kwargs)
    
    # Create LR finder
    lr_finder = LRFinder(model, temp_optimizer, criterion, device)
    
    # Run range test
    lr_finder.range_test(train_loader, **lr_finder_kwargs)
    
    # Plot results
    lr_finder.plot()
    
    # Get suggestion
    suggested_lr = lr_finder.suggest_lr()
    
    return suggested_lr


if __name__ == "__main__":
    # Example usage
    print("LR Finder utility - import and use with your model and data loader")
