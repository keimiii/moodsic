import os
import sys

# Ensure project root is on sys.path so running scripts works without PYTHONPATH
_CURRENT_FILE = os.path.abspath(__file__)
_SCRIPTS_DIR = os.path.dirname(_CURRENT_FILE)
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

#!/usr/bin/env python3
"""
Scene Model Training Script for FindingEmo Dataset
Implements research findings for V-A prediction + Emo8 classification baseline.

Usage:
    # Combined V-A + Emo8 with DINOv3
    python scripts/train_scene_model.py --config configs/scene_models/scene_model_dinov3_combined_config.yaml --data.findingemo_path /path/to/findingemo
    
    # CLIP backbone
    python scripts/train_scene_model.py --config configs/scene_models/scene_model_clip_combined_config.yaml --data.findingemo_path /path/to/findingemo
    
    # Baseline comparison
    python scripts/train_scene_model.py --config configs/scene_models/scene_model_baseline_config.yaml --data.findingemo_path /path/to/findingemo
"""

import sys
import argparse
from pathlib import Path
import traceback
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import create_arg_parser, load_config
from src.utils.device import DeviceManager
from src.utils.logging_utils import setup_logging
from src.utils.auto_lr import auto_detect_learning_rate
from src.data.datasets import create_dataset, create_dataloader
from src.data.transforms import create_transforms_from_config
from src.models.va_models import create_va_model
from src.training.losses import create_loss_function
from src.utils.metrics import evaluate_scene_model_predictions
from src.evaluation.evaluator import VAEvaluator
import logging

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

logger = logging.getLogger(__name__)


class SceneModelTrainer:
    """
    Comprehensive trainer for Scene Model with research-based configuration.
    
    Implements:
    - V-A regression with WMSE loss
    - Emo8 classification with UCE loss  
    - Stratified data splits
    - Backbone-specific preprocessing
    - Research-compliant evaluation metrics
    """
    
    def __init__(self, config, device_manager, experiment_logger):
        """
        Initialize Scene Model trainer.
        
        Args:
            config: Training configuration
            device_manager: Device management utility
            experiment_logger: Logging utility
        """
        self.config = config
        self.device_manager = device_manager
        self.logger = experiment_logger
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.patience_counter = 0
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_training()
        
        self.logger.info("🏞️  Scene Model Trainer initialized")
        self.logger.info(f"📊 Dataset: {len(self.train_loader.dataset)} train, {len(self.val_loader.dataset)} val")
        self.logger.info(f"🧠 Model: {self.config.model.model_name}")
        self.logger.info(f"🎯 Monitor metric: {self.config.training.early_stopping.monitor_metric}")
    
    def _setup_data(self):
        """Setup data loaders with research-compliant preprocessing."""
        self.logger.info("📂 Setting up FindingEmo dataset...")
        
        # Determine backbone type for transforms
        backbone_type = self.config.model.backbone_type
        if backbone_type == "clip":
            backbone_type = "clip"
        elif backbone_type == "dinov3":
            backbone_type = "dinov3"
        else:
            backbone_type = "imagenet"
        
        # Create transforms
        train_transform = create_transforms_from_config(
            data_config=self.config.data.to_dict(),
            dataset_type=self.config.data.dataset_type,
            is_training=True,
            backbone_type=backbone_type
        )
        
        val_transform = create_transforms_from_config(
            data_config=self.config.data.to_dict(),
            dataset_type=self.config.data.dataset_type,
            is_training=False,
            backbone_type=backbone_type
        )
        
        # Optional: subset each split by a fraction or fixed count for quick dev runs
        def _coerce_optional_number(x, int_only=False):
            if x is None:
                return None
            # Handle strings like 'null', 'none', ''
            if isinstance(x, str):
                xs = x.strip().lower()
                if xs in ('', 'none', 'null', 'nan'):
                    return None
                try:
                    val = float(xs)
                    return int(val) if int_only else val
                except Exception:
                    return None
            # Already numeric
            if isinstance(x, (int, float)):
                return int(x) if int_only else x
            return None

        subset_fraction = _coerce_optional_number(getattr(self.config.data, 'subset_fraction', None))
        max_samples_per_split = _coerce_optional_number(getattr(self.config.data, 'max_samples_per_split', None), int_only=True)
        max_by_split = {s: None for s in ['train', 'val', 'test']}
        if isinstance(max_samples_per_split, (int, float)) and max_samples_per_split > 0:
            # Same cap for all splits
            max_by_split = {s: int(max_samples_per_split) for s in max_by_split}
            self.logger.info(f"🔢 Using max_samples_per_split={int(max_samples_per_split)} for all splits")
        elif isinstance(subset_fraction, (int, float)) and 0 < subset_fraction <= 1:
            # Try to derive per-split sizes from saved split indices
            try:
                split_file = Path(self.config.data.findingemo_path) / 'split_indices.json'
                if split_file.exists():
                    import json as _json
                    with open(split_file, 'r') as _f:
                        _splits = _json.load(_f)['splits']
                    for s in ['train', 'val', 'test']:
                        max_by_split[s] = max(1, int(len(_splits.get(s, [])) * subset_fraction))
                    self.logger.info(f"🔢 Using subset_fraction={subset_fraction:.2f} -> per-split caps: {max_by_split}")
                else:
                    self.logger.info(f"ℹ️ subset_fraction={subset_fraction:.2f} provided, but no split_indices.json found; caps will be applied after splits are created")
            except Exception as _e:
                self.logger.warning(f"⚠️ Failed to derive subset sizes: {_e}")
        
        # Create datasets with stratified splits
        self.train_dataset = create_dataset(
            dataset_type=self.config.data.dataset_type,
            root_path=self.config.data.findingemo_path,
            split="train",
            transform=train_transform,
            max_samples=max_by_split.get('train'),
            stratify_on=self.config.data.stratify_on,
            save_split_indices=self.config.data.save_split_indices,
            load_split_indices=self.config.data.load_split_indices
        )
        
        self.val_dataset = create_dataset(
            dataset_type=self.config.data.dataset_type,
            root_path=self.config.data.findingemo_path,
            split="val",
            transform=val_transform,
            max_samples=max_by_split.get('val'),
            stratify_on=self.config.data.stratify_on,
            save_split_indices=self.config.data.save_split_indices,
            load_split_indices=self.config.data.load_split_indices
        )
        
        self.test_dataset = create_dataset(
            dataset_type=self.config.data.dataset_type,
            root_path=self.config.data.findingemo_path,
            split="test",
            transform=val_transform,
            max_samples=max_by_split.get('test'),
            stratify_on=self.config.data.stratify_on,
            save_split_indices=self.config.data.save_split_indices,
            load_split_indices=self.config.data.load_split_indices
        )
        
        # Create data loaders
        self.train_loader = create_dataloader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.hardware.num_workers,
            pin_memory=self.config.hardware.pin_memory
        )
        
        self.val_loader = create_dataloader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.hardware.num_workers,
            pin_memory=self.config.hardware.pin_memory
        )
        
        self.test_loader = create_dataloader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.hardware.num_workers,
            pin_memory=self.config.hardware.pin_memory
        )
        
        self.logger.info(f"✅ Datasets created with {backbone_type} preprocessing")
    
    def _setup_model(self):
        """Setup Scene Model with configurable backbone and head."""
        self.logger.info("🧠 Creating Scene Model...")
        
        # Determine model type
        model_type = "scene"
        backbone_type = getattr(self.config.model, 'backbone_type', 'dinov3')

        # Handle different backbone types
        if backbone_type == "clip":
            self.model = create_va_model(
                model_type=model_type,
                backbone_type="clip",
                clip_model_name=getattr(self.config.model, 'clip_model_name', 'ViT-B/32'),
                feature_dim=self.config.model.feature_dim,
                head_config=self.config.model.head_config.to_dict(),
                freeze_backbone=self.config.model.freeze_backbone
            )
        elif backbone_type == "imagenet":
            # Torchvision ResNet-style backbone
            imagenet_backbone_name = getattr(self.config.model, 'backbone', 'resnet50')
            self.model = create_va_model(
                model_type=model_type,
                backbone_type="imagenet",
                feature_dim=self.config.model.feature_dim,
                head_config=self.config.model.head_config.to_dict(),
                freeze_backbone=self.config.model.freeze_backbone,
                imagenet_backbone_name=imagenet_backbone_name
            )
        else:
            # DINOv3 backbone
            backbone_path = self._find_backbone_path()
            self.model = create_va_model(
                model_type=model_type,
                backbone_path=backbone_path,
                backbone_type="dinov3",
                feature_dim=self.config.model.feature_dim,
                head_config=self.config.model.head_config.to_dict(),
                freeze_backbone=self.config.model.freeze_backbone
            )
        
        # Move to device
        self.model = self.model.to(self.device_manager.device)
        
        # Mixed precision
        if self.config.hardware.mixed_precision and self.device_manager.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        self.logger.info(f"✅ Scene Model created and moved to {self.device_manager.device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.get_trainable_parameters())
        self.logger.info(f"📊 Parameters: {trainable_params:,}/{total_params:,} trainable")
    
    def _find_backbone_path(self):
        """Find available DINOv3 backbone path."""
        backbone_type = self.config.model.backbone_type
        
        # Available backbones
        available_backbones = {
            'dinov3': [
                "<PATH-HERE>/dinov3_convnext_tiny",
                "<PATH-HERE>/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
                "<PATH-HERE>/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
                "<PATH-HERE>/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
            ]
        }
        
        # Try to find backbone
        backbone_path = None
        for path in available_backbones.get(backbone_type, []):
            if Path(path).exists():
                backbone_path = path
                break
        
        if not backbone_path:
            raise FileNotFoundError(f"No {backbone_type} backbone found. Please check paths.")
        
        self.logger.info(f"📂 Using backbone: {Path(backbone_path).name}")
        return backbone_path
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and loss function."""
        self.logger.info("🏋️ Setting up training components...")
        
        # Create loss function
        self.criterion = create_loss_function(self.config.training.loss.to_dict())
        
        # Create optimizer
        opt_name = self.config.training.optimizer.lower()
        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.get_trainable_parameters(),
                lr=self.config.training.learning_rate,
                **self.config.training.optimizer_config.to_dict()
            )
        elif opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.get_trainable_parameters(),
                lr=self.config.training.learning_rate,
                **self.config.training.optimizer_config.to_dict()
            )
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.get_trainable_parameters(),
                lr=self.config.training.learning_rate,
                **self.config.training.optimizer_config.to_dict()
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        # Create scheduler
        if self.config.training.scheduler.lower() == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                **self.config.training.scheduler_config.to_dict()
            )
        elif self.config.training.scheduler.lower() == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.scheduler_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"✅ Training setup complete")
        self.logger.info(f"   Optimizer: {self.config.training.optimizer}")
        self.logger.info(f"   Scheduler: {self.config.training.scheduler}")
        self.logger.info(f"   Loss: {self.config.training.loss.type}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = {k: v.to(self.device_manager.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if self.scaler and self.device_manager.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    predictions = self.model(batch['image'])
                    
                    # Prepare targets
                    targets = {
                        'valence': batch['valence'],
                        'arousal': batch['arousal']
                    }
                    if 'emo8_label' in batch:
                        targets['emo8_label'] = batch['emo8_label']
                    
                    # Compute loss
                    loss_dict = self.criterion(predictions, targets, batch)
                    loss = loss_dict['total_loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if hasattr(self.config.training, 'grad_clip_norm'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.grad_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training without mixed precision
                predictions = self.model(batch['image'])
                
                targets = {
                    'valence': batch['valence'],
                    'arousal': batch['arousal']
                }
                if 'emo8_label' in batch:
                    targets['emo8_label'] = batch['emo8_label']
                
                loss_dict = self.criterion(predictions, targets, batch)
                loss = loss_dict['total_loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if hasattr(self.config.training, 'grad_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip_norm
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch:3d} [{batch_idx:4d}/{num_batches:4d}] "
                    f"Loss: {loss.item():.4f} LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                batch = {k: v.to(self.device_manager.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                predictions = self.model(batch['image'])
                
                # Collect predictions and targets
                all_predictions.append({
                    k: v.cpu() for k, v in predictions.items() if torch.is_tensor(v)
                })
                
                targets = {
                    'valence': batch['valence'].cpu(),
                    'arousal': batch['arousal'].cpu()
                }
                if 'emo8_label' in batch:
                    targets['emo8_label'] = batch['emo8_label'].cpu()
                
                all_targets.append(targets)
        
        # Concatenate all predictions and targets
        final_predictions = {}
        final_targets = {}
        
        for key in all_predictions[0].keys():
            final_predictions[key] = torch.cat([p[key] for p in all_predictions], dim=0)
        
        for key in all_targets[0].keys():
            final_targets[key] = torch.cat([t[key] for t in all_targets], dim=0)
        
        # Evaluate
        metrics = evaluate_scene_model_predictions(
            predictions=final_predictions,
            targets=final_targets,
            verbose=False
        )
        
        return metrics
    
    def train(self):
        """Main training loop."""
        self.logger.info("🚀 Starting Scene Model training...")
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Get monitor metric
            monitor_metric = self.config.training.early_stopping.monitor_metric
            current_metric = val_metrics.get(monitor_metric, float('inf'))
            
            # Log epoch results
            self.logger.info(f"📊 Epoch {epoch:3d} Results:")
            self.logger.info(f"   Train Loss: {train_loss:.4f}")
            self.logger.info(f"   Val {monitor_metric}: {current_metric:.4f}")
            if 'va_mae_avg' in val_metrics:
                self.logger.info(f"   Val MAE: {val_metrics['va_mae_avg']:.4f}")
            if 'emo8_weighted_f1' in val_metrics:
                self.logger.info(f"   Val Emo8 F1: {val_metrics['emo8_weighted_f1']:.4f}")
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Check for improvement
            mode = self.config.training.early_stopping.mode
            improved = (
                (mode == "min" and current_metric < self.best_metric) or
                (mode == "max" and current_metric > self.best_metric)
            )
            
            if improved:
                self.best_metric = current_metric
                self.patience_counter = 0
                self.logger.info(f"🎯 New best {monitor_metric}: {current_metric:.4f}")
                
                # Save best model
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Early stopping check
            if (self.config.training.early_stopping.enabled and 
                self.patience_counter >= self.config.training.early_stopping.patience):
                self.logger.info(f"🛑 Early stopping after {epoch + 1} epochs")
                break
            
            # Save regular checkpoint
            if epoch % self.config.checkpointing.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, is_best=False)
        
        # Final evaluation on test set using BEST checkpoint
        # Load best model weights if available before evaluating on test set
        best_checkpoint_path = None
        best_checkpoint_epoch = None
        try:
            from pathlib import Path as _Path
            best_ckpt = _Path(self.config.checkpointing.save_dir) / "best_model.pth"
            if best_ckpt.exists():
                _ckpt = torch.load(best_ckpt, map_location=self.device_manager.device)
                if isinstance(_ckpt, dict) and 'model_state_dict' in _ckpt:
                    self.model.load_state_dict(_ckpt['model_state_dict'])
                    best_checkpoint_path = str(best_ckpt)
                    best_checkpoint_epoch = int(_ckpt.get('epoch')) if 'epoch' in _ckpt else None
                    self.logger.info(f"📂 Loaded best model weights from: {best_ckpt}")
                else:
                    self.logger.warning(f"⚠️ Best checkpoint missing model_state_dict: {best_ckpt}")
            else:
                self.logger.warning(f"⚠️ Best checkpoint not found at {best_ckpt}; using current model state")
        except Exception as _e:
            self.logger.warning(f"⚠️ Failed to load best checkpoint for test eval: {_e}")

        test_metrics = self._evaluate_test_set()
        
        self.logger.info("🏁 Training completed!")
        self.logger.info(f"📈 Best {monitor_metric}: {self.best_metric:.4f}")
        
        return {
            'best_metric': self.best_metric,
            'test_metrics': test_metrics,
            'total_epochs': self.current_epoch + 1,
            'best_checkpoint_path': best_checkpoint_path,
            'best_checkpoint_epoch': best_checkpoint_epoch,
        }
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint as .pth (full) and .pkl (weights-only)."""
        from src.utils.checkpoints import save_checkpoint_bundle
        checkpoint_dir = Path(self.config.checkpointing.save_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        paths = save_checkpoint_bundle(
            checkpoint_dir=checkpoint_dir,
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            best_metric=self.best_metric,
            config=self.config,
            scheduler=self.scheduler,
            is_best=is_best,
        )
        if is_best:
            self.logger.info(f"💾 Saved best model: {paths['pth']} and {paths['pkl']}")
        else:
            self.logger.debug(f"💾 Saved checkpoint: {paths['pth']} and {paths['pkl']}")
    
    def _evaluate_test_set(self):
        """Evaluate on test set."""
        self.logger.info("🧪 Evaluating on test set...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = {k: v.to(self.device_manager.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                predictions = self.model(batch['image'])
                
                all_predictions.append({
                    k: v.cpu() for k, v in predictions.items() if torch.is_tensor(v)
                })
                
                targets = {
                    'valence': batch['valence'].cpu(),
                    'arousal': batch['arousal'].cpu()
                }
                if 'emo8_label' in batch:
                    targets['emo8_label'] = batch['emo8_label'].cpu()
                
                all_targets.append(targets)
        
        # Concatenate results
        final_predictions = {}
        final_targets = {}
        
        for key in all_predictions[0].keys():
            final_predictions[key] = torch.cat([p[key] for p in all_predictions], dim=0)
        
        for key in all_targets[0].keys():
            final_targets[key] = torch.cat([t[key] for t in all_targets], dim=0)
        
        # Evaluate with detailed output
        test_metrics = evaluate_scene_model_predictions(
            predictions=final_predictions,
            targets=final_targets,
            verbose=True
        )
        
        return test_metrics


def main():
    """Main training function."""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        if not args.config:
            print("❌ Error: Configuration file is required")
            print("Usage: python train_scene_model.py --config path/to/config.yaml")
            return 1
        
        config = load_config(args.config, args)

        # Force checkpoints to notebooks/checkpoints/<model_name>
        try:
            from pathlib import Path as _Path
            _model_name = getattr(config.model, 'model_name', 'model')
            _ckpt_root = _Path('notebooks') / 'checkpoints'
            _ckpt_root.mkdir(parents=True, exist_ok=True)
            config.checkpointing.save_dir = str(_ckpt_root / _model_name)
        except Exception:
            pass
        
        # Validate FindingEmo path
        if not config.data.findingemo_path:
            print("❌ Error: FindingEmo dataset path is required")
            print("Specify with: --data.findingemo_path /path/to/findingemo")
            return 1
        
        if not Path(config.data.findingemo_path).exists():
            print(f"❌ Error: FindingEmo path does not exist: {config.data.findingemo_path}")
            return 1
        
        # Setup reproducibility
        torch.manual_seed(config.reproducibility.seed)
        torch.cuda.manual_seed_all(config.reproducibility.seed)
        
        # Setup logging
        experiment_logger = setup_logging(config)
        
        experiment_logger.info("🏞️  Starting Scene Model Training")
        experiment_logger.info(f"📋 Experiment: {experiment_logger.experiment_name}")
        experiment_logger.info(f"🎯 Model: {config.model.model_name}")
        experiment_logger.info(f"📊 Dataset: {config.data.dataset_type}")
        
        # Setup device management
        device_manager = DeviceManager(
            device=config.hardware.device,
            mixed_precision=config.hardware.mixed_precision,
            verbose=True
        )
        
        # Auto-detect learning rate if needed
        if hasattr(config.training, 'learning_rate') and (
            config.training.learning_rate == 'auto' or 
            str(config.training.learning_rate).lower() == 'auto'
        ):
            experiment_logger.info("🤖 Auto-detecting optimal learning rate...")
            
            # Temporarily set a default learning rate for trainer creation
            original_lr = config.training.learning_rate
            config.training.learning_rate = 1e-4  # Default for auto detection
            
            # Create temporary trainer to get data loaders
            temp_trainer = SceneModelTrainer(config, device_manager, experiment_logger)
            
            # Determine backbone configuration
            backbone_type = getattr(config.model, 'backbone_type', 'dinov3')
            freeze_backbone = getattr(config.model, 'freeze_backbone', True)
            batch_size = config.training.batch_size
            
            # Run auto LR detection
            optimal_lr, lr_info = auto_detect_learning_rate(
                model=temp_trainer.model,
                backbone_type=backbone_type,
                freeze_backbone=freeze_backbone,
                batch_size=batch_size,
                device=device_manager.device,
                train_loader=temp_trainer.train_loader,
                use_lr_finder=getattr(config.training, 'use_lr_finder', False),
                num_epochs=getattr(config.training, 'num_epochs', 100)
            )
            
            # Update config with detected learning rate
            config.training.learning_rate = optimal_lr
            
            # Clean up temporary trainer
            del temp_trainer
            
            # Log the detection results
            experiment_logger.info(f"✅ Auto-detected learning rate: {optimal_lr:.2e}")
            experiment_logger.info(f"📊 Detection method: {lr_info['method']}")
            experiment_logger.info(f"🎯 Backbone: {lr_info['backbone_type']} ({'frozen' if lr_info['frozen'] else 'unfrozen'})")
            experiment_logger.info(f"📦 Batch size: {lr_info['batch_size']}")
            experiment_logger.info(f"📈 Empirical range: {lr_info['empirical_range'][0]:.2e} - {lr_info['empirical_range'][1]:.2e}")
            
            # Optionally update scheduler config
            if 'schedule_config' in lr_info and hasattr(config.training, 'scheduler'):
                schedule_config = lr_info['schedule_config']
                if hasattr(config.training, 'warmup_epochs'):
                    config.training.warmup_epochs = schedule_config['warmup_epochs']
                if hasattr(config.training, 'min_lr'):
                    config.training.min_lr = schedule_config['min_lr']
                
                experiment_logger.info(f"📈 Updated scheduler config:")
                experiment_logger.info(f"   Warmup epochs: {schedule_config['warmup_epochs']}")
                experiment_logger.info(f"   Min LR: {schedule_config['min_lr']:.2e}")
        
        # Create trainer
        trainer = SceneModelTrainer(config, device_manager, experiment_logger)
        
        # Start training
        results = trainer.train()

        # After training: generate evaluation visualization using the universal evaluator
        try:
            experiment_logger.info("🎨 Generating evaluation visualization...")

            # Ensure best model weights are loaded
            from pathlib import Path as _Path
            import torch as _torch
            best_ckpt = _Path(config.checkpointing.save_dir) / "best_model.pth"
            if best_ckpt.exists():
                _ckpt = _torch.load(best_ckpt, map_location=device_manager.device)
                if 'model_state_dict' in _ckpt:
                    trainer.model.load_state_dict(_ckpt['model_state_dict'])
                    experiment_logger.info(f"📂 Loaded best model weights from: {best_ckpt}")
            else:
                experiment_logger.warning(f"⚠️ Best checkpoint not found at {best_ckpt}; using current model state")

            # Build evaluator
            model_name = experiment_logger.experiment_name
            dataset_name = f"{config.data.dataset_type}_test"
            # Put visualization under this run's log folder
            from pathlib import Path as _P
            _run_log_dir = _P(experiment_logger.log_dir)
            evaluator = VAEvaluator(
                model_name=model_name,
                dataset_name=dataset_name,
                output_dir=str(_run_log_dir),
                device=device_manager.device,
                flat_output=True,
            )

            # Attach comprehensive context so evaluator writes a full summary
            try:
                _train_cfg = {
                    'learning_rate': float(config.training.learning_rate) if isinstance(config.training.learning_rate, (int, float)) else str(config.training.learning_rate),
                    'batch_size': int(config.training.batch_size),
                    'num_epochs': int(config.training.num_epochs),
                    'optimizer': getattr(config.training, 'optimizer', None),
                    'scheduler': getattr(config.training, 'scheduler', None),
                    'early_stopping_patience': getattr(config.training.early_stopping, 'patience', None),
                    'monitor_metric': getattr(config.training.early_stopping, 'monitor_metric', None),
                }
            except Exception:
                _train_cfg = {}

            _train_summary = {
                'best_metric': float(results.get('best_metric', trainer.best_metric if hasattr(trainer, 'best_metric') else float('nan'))),
                'total_epochs': int(results.get('total_epochs', trainer.current_epoch + 1 if hasattr(trainer, 'current_epoch') else 0)),
                'converged': bool(getattr(trainer, 'patience_counter', 0) < getattr(config.training.early_stopping, 'patience', 0)) if hasattr(config.training, 'early_stopping') else None,
            }
            _ckpt_info = {
                'path': str(best_ckpt) if 'best_ckpt' in locals() and best_ckpt else None,
                'epoch': int(_ckpt.get('epoch')) if '_ckpt' in locals() and isinstance(_ckpt, dict) and 'epoch' in _ckpt else None,
            }
            evaluator.attach_context(
                experiment_name=model_name,
                training_config=_train_cfg,
                training_summary=_train_summary,
                checkpoint_info=_ckpt_info,
            )

            # Adapter: our models return dict with 'valence'/'arousal'
            def _prediction_fn(m, images):
                out = m(images)
                if isinstance(out, dict) and 'valence' in out and 'arousal' in out:
                    return out['valence'], out['arousal']
                if isinstance(out, (list, tuple)) and len(out) >= 2:
                    return out[0], out[1]
                return out[:, 0], out[:, 1]

            evaluator.evaluate_model(
                model=trainer.model,
                dataloader=trainer.test_loader,
                prediction_fn=_prediction_fn,
            )
            evaluator.create_comprehensive_visualization()
            evaluator.print_summary()
            # Copy all generated PNG/PDF plots into logs/<run>/plots/
            import shutil as _shutil
            _target_plots = _run_log_dir / "plots"
            _target_plots.mkdir(parents=True, exist_ok=True)
            for _src in list(evaluator.plots_dir.glob("*.png")) + list(evaluator.plots_dir.glob("*.pdf")):
                _dst = _target_plots / _src.name
                try:
                    _shutil.copy2(_src, _dst)
                except Exception:
                    pass
            experiment_logger.info(f"🎯 Visualizations copied to: {_target_plots}")
        except Exception as viz_err:
            experiment_logger.warning(f"⚠️ Failed to generate visualization: {viz_err}")
        
        # Log final results
        experiment_logger.info("✅ Scene Model training completed successfully!")
        experiment_logger.info(f"📈 Final results: {results}")
        
        experiment_logger.close()
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
