"""
Structured logging utilities for V-A emotion prediction training.
Supports file logging, console output, metrics tracking, and TensorBoard integration.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import json
import time
from dataclasses import dataclass, field

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


@dataclass
class LoggingConfig:
    """Configuration for logging setup."""
    level: str = "INFO"
    log_dir: str = "./logs"
    experiment_name: str = ""
    log_every_n_steps: int = 50
    eval_every_n_epochs: int = 1
    save_every_n_epochs: int = 5
    tensorboard_enabled: bool = False
    tensorboard_log_dir: str = "./tensorboard_logs"
    log_images: bool = True
    log_histograms: bool = False


class StructuredLogger:
    """
    Comprehensive logging system for ML training.
    
    Features:
    - File and console logging
    - Metrics tracking and export
    - Training progress monitoring
    - Optional TensorBoard integration
    - Experiment organization
    """
    
    def __init__(self, config: Union[LoggingConfig, Dict[str, Any]]):
        """
        Initialize structured logger.
        
        Args:
            config: Logging configuration
        """
        if isinstance(config, dict):
            config = LoggingConfig(**config)
        
        self.config = config
        self.experiment_name = self._generate_experiment_name()
        self.log_dir = Path(self.config.log_dir) / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize metrics tracking
        self.metrics_history = []
        self.current_epoch = 0
        self.current_step = 0
        self.start_time = time.time()
        
        # TensorBoard setup
        self.tensorboard_writer = None
        if self.config.tensorboard_enabled and TENSORBOARD_AVAILABLE:
            self._setup_tensorboard()
        
        self.logger.info(f"🚀 Structured logging initialized: {self.experiment_name}")
        self.logger.info(f"📁 Log directory: {self.log_dir}")
    
    def _generate_experiment_name(self) -> str:
        """Generate experiment name with timestamp."""
        if self.config.experiment_name:
            base_name = self.config.experiment_name
        else:
            base_name = "va_training"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}"
    
    def _setup_logger(self) -> logging.Logger:
        """Setup file and console logging."""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.level.upper()))
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        if not TENSORBOARD_AVAILABLE:
            self.logger.warning("TensorBoard not available, skipping TensorBoard logging")
            return
        
        tb_dir = Path(self.config.tensorboard_log_dir) / self.experiment_name
        tb_dir.mkdir(parents=True, exist_ok=True)
        
        self.tensorboard_writer = SummaryWriter(log_dir=str(tb_dir))
        self.logger.info(f"📊 TensorBoard logging enabled: {tb_dir}")
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def increment_epoch(self):
        """Increment epoch counter."""
        self.current_epoch += 1
    
    def increment_step(self):
        """Increment step counter."""
        self.current_step += 1
    
    def log_metrics(self, 
                   metrics: Dict[str, float], 
                   epoch: Optional[int] = None,
                   step: Optional[int] = None,
                   prefix: str = ""):
        """
        Log metrics to all configured outputs.
        
        Args:
            metrics: Dictionary of metric names and values
            epoch: Epoch number (uses current if None)
            step: Step number (uses current if None)
            prefix: Prefix for metric names
        """
        epoch = epoch if epoch is not None else self.current_epoch
        step = step if step is not None else self.current_step
        
        # Store metrics
        metric_entry = {
            'epoch': epoch,
            'step': step,
            'timestamp': time.time(),
            'metrics': metrics
        }
        self.metrics_history.append(metric_entry)
        
        # Log to console/file
        metrics_str = ", ".join([f"{prefix}{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"📊 Epoch {epoch} | {metrics_str}")
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            for name, value in metrics.items():
                self.tensorboard_writer.add_scalar(f"{prefix}{name}", value, step)
    
    def log_learning_rate(self, lr: float, step: Optional[int] = None):
        """Log learning rate."""
        step = step if step is not None else self.current_step
        
        self.logger.info(f"📈 Learning rate: {lr:.2e}")
        
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("learning_rate", lr, step)
    
    def log_model_info(self, model, total_params: int, trainable_params: int):
        """Log model architecture information."""
        self.logger.info(f"🧠 Model Information:")
        self.logger.info(f"  Architecture: {model.__class__.__name__}")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Frozen parameters: {total_params - trainable_params:,}")
        
        if self.tensorboard_writer:
            self.tensorboard_writer.add_text("model/architecture", str(model))
            self.tensorboard_writer.add_scalar("model/total_params", total_params, 0)
            self.tensorboard_writer.add_scalar("model/trainable_params", trainable_params, 0)
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information."""
        self.logger.info(f"📊 Dataset Information:")
        for key, value in dataset_info.items():
            self.logger.info(f"  {key}: {value}")
        
        if self.tensorboard_writer:
            for key, value in dataset_info.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(f"dataset/{key}", value, 0)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        self.logger.info(f"⚙️  Hyperparameters:")
        for key, value in hparams.items():
            self.logger.info(f"  {key}: {value}")
        
        # Save hyperparameters to file
        hparams_file = self.log_dir / "hyperparameters.json"
        with open(hparams_file, 'w') as f:
            json.dump(hparams, f, indent=2, default=str)
        
        if self.tensorboard_writer:
            # Convert all values to strings for TensorBoard
            hparams_tb = {k: str(v) for k, v in hparams.items()}
            self.tensorboard_writer.add_hparams(hparams_tb, {})
    
    def log_epoch_summary(self, 
                         epoch: int,
                         train_metrics: Dict[str, float],
                         val_metrics: Dict[str, float],
                         epoch_time: float):
        """Log epoch summary."""
        self.logger.info(f"📋 Epoch {epoch} Summary:")
        self.logger.info(f"  ⏱️  Time: {epoch_time:.1f}s")
        self.logger.info(f"  🏋️ Train Loss: {train_metrics.get('loss', 0):.4f}")
        self.logger.info(f"  🔍 Val CCC: {val_metrics.get('ccc_avg', 0):.4f}")
        self.logger.info(f"  📊 Val MAE: {val_metrics.get('mae_avg', 0):.4f}")
        
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("epoch/time", epoch_time, epoch)
    
    def save_metrics_history(self):
        """Save complete metrics history to file."""
        metrics_file = self.log_dir / "metrics_history.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.logger.info(f"💾 Metrics history saved: {metrics_file}")
    
    def close(self):
        """Close logging and save final metrics."""
        total_time = time.time() - self.start_time
        self.logger.info(f"🏁 Training completed in {total_time:.1f}s")
        
        # Save metrics history
        self.save_metrics_history()
        
        # Close TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
            self.logger.info("📊 TensorBoard logging closed")
        
        # Close logging handlers
        for handler in self.logger.handlers:
            handler.close()


class TrainingProgressLogger:
    """Progress logger for training steps."""
    
    def __init__(self, logger: StructuredLogger, log_every_n_steps: int = 50):
        """
        Initialize progress logger.
        
        Args:
            logger: Main structured logger
            log_every_n_steps: How often to log progress
        """
        self.logger = logger
        self.log_every_n_steps = log_every_n_steps
        self.last_log_time = time.time()
    
    def log_step(self, 
                epoch: int,
                batch_idx: int,
                total_batches: int,
                loss: float):
        """
        Log training step progress.
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            total_batches: Total number of batches
            loss: Current loss value
        """
        if batch_idx % self.log_every_n_steps == 0:
            current_time = time.time()
            time_per_batch = (current_time - self.last_log_time) / self.log_every_n_steps
            samples_per_sec = (self.log_every_n_steps * 32) / (current_time - self.last_log_time)  # Assuming batch_size=32
            
            progress = (batch_idx / total_batches) * 100
            self.logger.info(f"🔄 Epoch {epoch} [{batch_idx:3d}/{total_batches:3d}] "
                           f"({progress:5.1f}%) | Loss: {loss:.4f} | "
                           f"Time: {time_per_batch:.3f}s/batch | "
                           f"Speed: {samples_per_sec:.1f} samples/s")
            
            self.last_log_time = current_time


def setup_logging(config) -> StructuredLogger:
    """
    Setup logging from configuration.
    
    Args:
        config: Configuration object with logging section
        
    Returns:
        Configured StructuredLogger
    """
    # Extract logging config
    if hasattr(config, 'logging'):
        logging_dict = config.logging.to_dict() if hasattr(config.logging, 'to_dict') else config.logging.__dict__
    else:
        logging_dict = {}
    
    # Handle nested tensorboard config
    if 'tensorboard' in logging_dict:
        tb_config = logging_dict.pop('tensorboard')
        if isinstance(tb_config, dict):
            logging_dict['tensorboard_enabled'] = tb_config.get('enabled', False)
            logging_dict['tensorboard_log_dir'] = tb_config.get('log_dir', './tensorboard_logs')
            logging_dict['log_images'] = tb_config.get('log_images', True)
            logging_dict['log_histograms'] = tb_config.get('log_histograms', False)
    
    # Create logging config
    logging_config = LoggingConfig(**logging_dict)
    
    # Create and return logger
    return StructuredLogger(logging_config)


if __name__ == "__main__":
    # Test logging system
    print("🧪 Testing Logging System")
    print("=" * 50)
    
    # Create test configuration
    config = LoggingConfig(
        level="INFO",
        experiment_name="test_logging",
        tensorboard_enabled=False  # Disable for testing
    )
    
    # Initialize logger
    logger = StructuredLogger(config)
    
    # Test basic logging
    logger.info("Testing basic logging functionality")
    logger.warning("Testing warning message")
    
    # Test metrics logging
    test_metrics = {
        'loss': 0.5432,
        'ccc_avg': 0.3456,
        'mae_avg': 0.2345
    }
    logger.log_metrics(test_metrics, epoch=1, prefix="train/")
    
    # Test progress logger
    progress_logger = TrainingProgressLogger(logger, log_every_n_steps=10)
    for i in range(0, 50, 10):
        progress_logger.log_step(epoch=1, batch_idx=i, total_batches=50, loss=0.5 - i*0.01)
    
    # Test model info logging
    import torch.nn as nn
    test_model = nn.Linear(10, 2)
    total_params = sum(p.numel() for p in test_model.parameters())
    logger.log_model_info(test_model, total_params, total_params)
    
    # Test hyperparameter logging
    test_hparams = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'optimizer': 'adam'
    }
    logger.log_hyperparameters(test_hparams)
    
    # Close logger
    logger.close()
    
    print(f"\n✅ Logging test completed!")
    print(f"📁 Check logs in: {logger.log_dir}")
    print(f"🎯 Logging utilities ready for training!")