"""
Complete training pipeline for continuous V-A emotion prediction.
Supports checkpointing, early stopping, learning rate scheduling, and comprehensive evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass

from ..utils.metrics import VAMetrics, evaluate_model_predictions
from ..utils.logging_utils import StructuredLogger, TrainingProgressLogger
from ..utils.device import DeviceManager
from ..models.va_models import BaseVAModel


@dataclass
class TrainingState:
    """Training state for checkpointing and resuming."""
    epoch: int = 0
    step: int = 0
    best_metric: float = float('-inf')
    best_epoch: int = 0
    early_stopping_counter: int = 0
    train_losses: List[float] = None
    val_metrics: List[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.train_losses is None:
            self.train_losses = []
        if self.val_metrics is None:
            self.val_metrics = []


class VATrainer:
    """
    Comprehensive trainer for Valence-Arousal emotion prediction models.
    
    Features:
    - Mixed precision training
    - Checkpointing and resuming
    - Early stopping
    - Learning rate scheduling
    - Comprehensive metrics evaluation
    - Progress tracking and logging
    """
    
    def __init__(self,
                 model: BaseVAModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config,
                 device_manager: DeviceManager,
                 logger: StructuredLogger,
                 test_loader: Optional[DataLoader] = None):
        """
        Initialize trainer.
        
        Args:
            model: V-A prediction model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device_manager: Device management
            logger: Structured logger
            test_loader: Optional test data loader
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device_manager = device_manager
        self.logger = logger
        
        # Move model to device
        self.model = self.device_manager.to_device(self.model)
        
        # Checkpointing (set early since scheduler needs monitor_metric)
        self.checkpoint_dir = Path(config.checkpointing.save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_metric = config.checkpointing.monitor_metric
        self.save_best_only = config.checkpointing.save_best_only
        
        # Setup training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.loss_fn = self._create_loss_function()
        self.metrics_calculator = VAMetrics(
            metrics=config.evaluation.metrics,
            compute_per_quadrant=config.evaluation.compute_per_quadrant
        )
        
        # Training state
        self.state = TrainingState()
        
        # Early stopping
        self.early_stopping_patience = config.training.early_stopping_patience
        self.early_stopping_enabled = self.early_stopping_patience > 0
        
        # Progress logging
        self.progress_logger = TrainingProgressLogger(
            logger, 
            log_every_n_steps=config.logging.log_every_n_steps
        )
        
        # Log training setup
        self._log_training_setup()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from configuration."""
        optimizer_name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay
        
        # Check if differential learning rates are configured
        backbone_lr = getattr(self.config.training, 'backbone_lr', None)
        head_lr = getattr(self.config.training, 'head_lr', lr)
        
        if backbone_lr is not None and hasattr(self.model, 'get_parameter_groups'):
            # Use differential learning rates
            param_groups = self.model.get_parameter_groups(backbone_lr, head_lr)
            self.logger.info(f"🔧 Using differential learning rates: backbone={backbone_lr}, head={head_lr}")
        else:
            # Use traditional single learning rate
            param_groups = self.model.get_trainable_parameters()
            self.logger.info(f"🔧 Using single learning rate: {lr}")
        
        if optimizer_name == "adam":
            optimizer = optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(param_groups, lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        self.logger.info(f"🔧 Optimizer: {optimizer.__class__.__name__} (wd={weight_decay})")
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from configuration."""
        scheduler_name = self.config.training.scheduler.lower()
        
        if scheduler_name == "none":
            return None
        
        scheduler_params = self.config.training.scheduler_params
        
        if scheduler_name == "reduce_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max' if 'ccc' in self.monitor_metric else 'min',
                factor=scheduler_params.get('factor', 0.5),
                patience=scheduler_params.get('patience', 5),
                min_lr=scheduler_params.get('min_lr', 1e-7)
            )
        elif scheduler_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=scheduler_params.get('min_lr', 1e-7)
            )
        elif scheduler_name == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_params.get('step_size', 30),
                gamma=scheduler_params.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        self.logger.info(f"📈 Scheduler: {scheduler.__class__.__name__}")
        return scheduler
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function from configuration."""
        loss_name = self.config.training.loss_function.lower()
        
        if loss_name == "mse":
            loss_fn = nn.MSELoss()
        elif loss_name == "huber":
            loss_fn = nn.HuberLoss()
        elif loss_name == "smooth_l1":
            loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
        
        self.logger.info(f"📉 Loss function: {loss_fn.__class__.__name__}")
        return loss_fn
    
    def _log_training_setup(self) -> None:
        """Log training setup information."""
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.get_trainable_parameters())
        self.logger.log_model_info(self.model, total_params, trainable_params)
        
        # Dataset info
        dataset_info = {
            'train_samples': len(self.train_loader.dataset),
            'val_samples': len(self.val_loader.dataset),
            'batch_size': self.train_loader.batch_size,
            'train_batches': len(self.train_loader),
            'val_batches': len(self.val_loader)
        }
        if self.test_loader:
            dataset_info['test_samples'] = len(self.test_loader.dataset)
            dataset_info['test_batches'] = len(self.test_loader)
        
        self.logger.log_dataset_info(dataset_info)
        
        # Training hyperparameters
        hparams = {
            'num_epochs': self.config.training.num_epochs,
            'learning_rate': self.config.training.learning_rate,
            'weight_decay': self.config.training.weight_decay,
            'optimizer': self.config.training.optimizer,
            'scheduler': self.config.training.scheduler,
            'loss_function': self.config.training.loss_function,
            'early_stopping_patience': self.early_stopping_patience,
            'monitor_metric': self.monitor_metric,
            'device': str(self.device_manager.device),
            'mixed_precision': self.device_manager.mixed_precision
        }
        self.logger.log_hyperparameters(hparams)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        all_valence_true = []
        all_valence_pred = []
        all_arousal_true = []
        all_arousal_pred = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self.device_manager.to_device(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with self.device_manager.autocast_context():
                outputs = self.model(batch['image'])
                
                # Calculate loss
                valence_loss = self.loss_fn(outputs['valence'], batch['valence'])
                arousal_loss = self.loss_fn(outputs['arousal'], batch['arousal'])
                
                # Weighted loss
                loss_weights = self.config.training.loss_weights
                total_loss_batch = (loss_weights['valence'] * valence_loss + 
                                  loss_weights['arousal'] * arousal_loss)
            
            # Backward pass
            self.device_manager.backward(total_loss_batch)
            
            # Optimizer step
            self.device_manager.step_optimizer(self.optimizer)
            
            # Accumulate metrics
            total_loss += total_loss_batch.item()
            
            # Store predictions for metric calculation
            all_valence_true.extend(batch['valence'].cpu().numpy())
            all_valence_pred.extend(outputs['valence'].detach().cpu().numpy())
            all_arousal_true.extend(batch['arousal'].cpu().numpy())
            all_arousal_pred.extend(outputs['arousal'].detach().cpu().numpy())
            
            # Progress logging
            self.progress_logger.log_step(
                epoch, batch_idx + 1, len(self.train_loader), 
                total_loss_batch.item()
            )
            
            # Update step counter
            self.state.step += 1
            self.logger.increment_step()
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        train_metrics = self.metrics_calculator.compute_metrics(
            np.array(all_valence_true), np.array(all_valence_pred),
            np.array(all_arousal_true), np.array(all_arousal_pred)
        )
        train_metrics['loss'] = avg_loss
        
        # Store training loss
        self.state.train_losses.append(avg_loss)
        
        return train_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_valence_true = []
        all_valence_pred = []
        all_arousal_true = []
        all_arousal_pred = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = self.device_manager.to_device(batch)
                
                # Forward pass
                outputs = self.model(batch['image'])
                
                # Calculate loss
                valence_loss = self.loss_fn(outputs['valence'], batch['valence'])
                arousal_loss = self.loss_fn(outputs['arousal'], batch['arousal'])
                
                loss_weights = self.config.training.loss_weights
                total_loss_batch = (loss_weights['valence'] * valence_loss + 
                                  loss_weights['arousal'] * arousal_loss)
                
                total_loss += total_loss_batch.item()
                
                # Store predictions
                all_valence_true.extend(batch['valence'].cpu().numpy())
                all_valence_pred.extend(outputs['valence'].cpu().numpy())
                all_arousal_true.extend(batch['arousal'].cpu().numpy())
                all_arousal_pred.extend(outputs['arousal'].cpu().numpy())
        
        # Calculate validation metrics
        avg_loss = total_loss / len(self.val_loader)
        val_metrics = self.metrics_calculator.compute_metrics(
            np.array(all_valence_true), np.array(all_valence_pred),
            np.array(all_arousal_true), np.array(all_arousal_pred)
        )
        val_metrics['loss'] = avg_loss
        
        # Store validation metrics
        self.state.val_metrics.append(val_metrics)
        
        return val_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_state': self.state,
            'metrics': metrics,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 Best model saved to: {best_path}")
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device_manager.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if self.config.checkpointing.save_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and self.config.checkpointing.save_scheduler:
            if checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.state = checkpoint['training_state']
        
        self.logger.info(f"📂 Checkpoint loaded from: {checkpoint_path}")
        self.logger.info(f"  Resuming from epoch {self.state.epoch}")
    
    def check_early_stopping(self, current_metric: float) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            current_metric: Current validation metric value
            
        Returns:
            True if training should stop
        """
        if not self.early_stopping_enabled:
            return False
        
        # Determine if metric is improving
        is_better = False
        if 'ccc' in self.monitor_metric or 'pearson' in self.monitor_metric:
            # Higher is better for correlation metrics
            is_better = current_metric > self.state.best_metric
        else:
            # Lower is better for error metrics
            is_better = current_metric < self.state.best_metric
        
        if is_better:
            self.state.best_metric = current_metric
            self.state.best_epoch = self.state.epoch
            self.state.early_stopping_counter = 0
            return False
        else:
            self.state.early_stopping_counter += 1
            
            if self.state.early_stopping_counter >= self.early_stopping_patience:
                self.logger.info(f"🛑 Early stopping triggered after {self.early_stopping_patience} epochs without improvement")
                self.logger.info(f"📈 Best {self.monitor_metric}: {self.state.best_metric:.4f} at epoch {self.state.best_epoch}")
                return True
            
            return False
    
    def train(self) -> Dict[str, Any]:
        """
        Run complete training loop.
        
        Returns:
            Training results and final metrics
        """
        self.logger.info("🚀 Starting training...")
        
        start_time = time.time()
        start_epoch = self.state.epoch
        
        for epoch in range(start_epoch, self.config.training.num_epochs):
            epoch_start_time = time.time()
            
            # Update epoch counter
            self.state.epoch = epoch
            self.logger.increment_epoch()
            
            # Training phase
            train_metrics = self.train_epoch(epoch)
            
            # Validation phase
            val_metrics = self.validate_epoch(epoch)
            
            # Log metrics
            self.logger.log_metrics(train_metrics, epoch=epoch, prefix="train/")
            self.logger.log_metrics(val_metrics, epoch=epoch, prefix="val/")
            
            # Learning rate scheduling
            if self.scheduler:
                # Extract metric name without 'val_' prefix for scheduler
                metric_key = self.monitor_metric.replace('val_', '') if self.monitor_metric.startswith('val_') else self.monitor_metric
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[metric_key])
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_learning_rate(current_lr)
            
            # Check for best model
            metric_key = self.monitor_metric.replace('val_', '') if self.monitor_metric.startswith('val_') else self.monitor_metric
            current_metric = val_metrics[metric_key]
            is_best = False
            if 'ccc' in self.monitor_metric or 'pearson' in self.monitor_metric:
                is_best = current_metric > self.state.best_metric
            else:
                is_best = current_metric < self.state.best_metric
            
            if is_best:
                self.state.best_metric = current_metric
                self.state.best_epoch = epoch
            
            # Save checkpoint
            if not self.save_best_only or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            self.logger.log_epoch_summary(epoch, train_metrics, val_metrics, epoch_time)
            
            # Early stopping check
            if self.check_early_stopping(current_metric):
                break
        
        # Training completed
        total_time = time.time() - start_time
        
        self.logger.info("🏁 Training completed!")
        self.logger.info(f"⏱️  Total training time: {total_time:.1f}s")
        self.logger.info(f"📈 Best {self.monitor_metric}: {self.state.best_metric:.4f} at epoch {self.state.best_epoch}")
        
        # Final evaluation on test set if available
        test_results = None
        if self.test_loader:
            test_results = self.evaluate_test_set()
        
        return {
            'best_metric': self.state.best_metric,
            'best_epoch': self.state.best_epoch,
            'total_epochs': self.state.epoch + 1,
            'total_time': total_time,
            'train_losses': self.state.train_losses,
            'val_metrics': self.state.val_metrics,
            'test_results': test_results
        }
    
    def evaluate_test_set(self) -> Dict[str, float]:
        """Evaluate model on test set."""
        if not self.test_loader:
            return {}
        
        self.logger.info("🧪 Evaluating on test set...")
        
        # Load best model
        best_model_path = self.checkpoint_dir / "best_model.pth"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device_manager.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("📂 Loaded best model for test evaluation")
        
        self.model.eval()
        
        all_valence_true = []
        all_valence_pred = []
        all_arousal_true = []
        all_arousal_pred = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = self.device_manager.to_device(batch)
                outputs = self.model(batch['image'])
                
                all_valence_true.extend(batch['valence'].cpu().numpy())
                all_valence_pred.extend(outputs['valence'].cpu().numpy())
                all_arousal_true.extend(batch['arousal'].cpu().numpy())
                all_arousal_pred.extend(outputs['arousal'].cpu().numpy())
        
        # Calculate test metrics
        test_metrics = self.metrics_calculator.compute_metrics(
            np.array(all_valence_true), np.array(all_valence_pred),
            np.array(all_arousal_true), np.array(all_arousal_pred)
        )
        
        self.logger.log_metrics(test_metrics, prefix="test/")
        self.logger.info("✅ Test evaluation completed")
        
        return test_metrics


if __name__ == "__main__":
    print("🧪 Trainer module loaded successfully")
    print("ℹ️  This module provides the VATrainer class for training V-A emotion prediction models")
    print("📚 Usage: Import VATrainer and use with appropriate model, data loaders, and configuration")
