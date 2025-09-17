"""
Configuration management utilities for V-A emotion prediction training.
Supports YAML configs with CLI overrides and inheritance.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from dataclasses import dataclass, field
from copy import deepcopy

logger = logging.getLogger(__name__)


class DictConfig:
    """
    Configuration object that allows both dict-style and attribute-style access.
    Supports nested configurations and CLI overrides.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        self._config = {}
        self._load_dict(config_dict)
    
    def _load_dict(self, config_dict: Dict[str, Any], prefix: str = ""):
        """Recursively load dictionary into config object."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Create nested DictConfig for dictionaries
                setattr(self, key, DictConfig(value))
                self._config[key] = getattr(self, key)
            else:
                setattr(self, key, value)
                self._config[key] = value
    
    def __getitem__(self, key):
        """Dict-style access."""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """Dict-style assignment."""
        setattr(self, key, value)
        self._config[key] = value
    
    def __contains__(self, key):
        """Check if key exists."""
        return hasattr(self, key)
    
    def get(self, key, default=None):
        """Dict-style get with default."""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self._config.items():
            if isinstance(value, DictConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def update(self, other: Union[Dict[str, Any], 'DictConfig']):
        """Update config with values from another config or dict."""
        if isinstance(other, dict):
            other_dict = other
        else:
            other_dict = other.to_dict()
        
        self._update_recursive(self._config, other_dict)
        self._load_dict(self._config)
    
    def _update_recursive(self, base: Dict, update: Dict):
        """Recursively update nested dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_recursive(base[key], value)
            else:
                base[key] = value
    
    def _set_nested_value(self, keys: list, value: Any):
        """Set a nested value using dot notation keys while preserving existing values."""
        current = self._config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # If it's not a dict, preserve existing dict structure if it exists in the object
                if hasattr(self, key) and hasattr(getattr(self, key), 'to_dict'):
                    current[key] = getattr(self, key).to_dict()
                else:
                    current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to YAML file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        
        logger.info(f"💾 Configuration saved to: {file_path}")


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration with support for base config inheritance.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing loaded configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle base config inheritance
    if 'base' in config:
        base_config_name = config.pop('base')
        base_config_path = config_path.parent / base_config_name
        
        if not base_config_path.exists():
            raise FileNotFoundError(f"Base configuration not found: {base_config_path}")
        
        # Recursively load base config
        base_config = load_yaml_config(base_config_path)
        
        # Merge base config with current config (current overrides base)
        merged_config = deepcopy(base_config)
        _merge_configs(merged_config, config)
        config = merged_config
    
    logger.info(f"📋 Configuration loaded from: {config_path}")
    return config


def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]):
    """Recursively merge configuration dictionaries."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _merge_configs(base[key], value)
        else:
            base[key] = value


def apply_cli_overrides(config: DictConfig, args: argparse.Namespace) -> DictConfig:
    """
    Apply CLI argument overrides to configuration.
    
    Supports dot notation like: --training.learning_rate 0.001
    
    Args:
        config: Configuration object
        args: Parsed CLI arguments
        
    Returns:
        Updated configuration object
    """
    # Convert args to dict, filtering out None values and non-override args
    args_dict = vars(args)
    
    # Apply each nested override individually to preserve existing structure
    applied_overrides = {}
    for arg_name, arg_value in args_dict.items():
        if arg_value is not None and '.' in arg_name:
            # Split nested key and apply directly
            keys = arg_name.split('.')
            config._set_nested_value(keys, arg_value)
            applied_overrides[arg_name] = arg_value
    
    # Reload the config structure after manual updates
    if applied_overrides:
        config._load_dict(config._config)
        logger.info(f"🔧 Applied CLI overrides: {applied_overrides}")
    
    return config


def create_arg_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for training script.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Train continuous Valence-Arousal emotion prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    
    # Optional training overrides
    parser.add_argument('--training.batch_size', type=int,
                       help='Override batch size')
    parser.add_argument('--training.learning_rate', type=float,
                       help='Override learning rate')
    parser.add_argument('--training.num_epochs', type=int,
                       help='Override number of epochs')
    parser.add_argument('--training.early_stopping_patience', type=int,
                       help='Override early stopping patience')
    
    # Data path overrides
    parser.add_argument('--data.affectnet_path', type=str,
                       help='Path to AffectNet dataset')
    parser.add_argument('--data.findingemo_path', type=str,
                       help='Path to FindingEmo dataset')
    
    # Model overrides
    parser.add_argument('--model.freeze_backbone', type=bool,
                       help='Whether to freeze backbone')
    parser.add_argument('--model.dropout_rate', type=float,
                       help='Override dropout rate')
    
    # Hardware overrides
    parser.add_argument('--hardware.device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Override device selection')
    parser.add_argument('--hardware.mixed_precision', type=bool,
                       help='Override mixed precision setting')
    parser.add_argument('--hardware.num_workers', type=int,
                       help='Override number of DataLoader workers')
    
    # Logging overrides
    parser.add_argument('--logging.experiment_name', type=str,
                       help='Override experiment name')
    parser.add_argument('--logging.level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Override logging level')
    
    # Checkpointing overrides
    parser.add_argument('--checkpointing.save_dir', type=str,
                       help='Override checkpoint save directory')
    parser.add_argument('--checkpointing.monitor_metric', type=str,
                       help='Override metric to monitor for best model')
    
    # Resume training
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')

    # Data overrides
    parser.add_argument('--data.subset_fraction', type=float,
                       help='Use only a fraction of each split (0<frac<=1)')
    parser.add_argument('--data.max_samples_per_split', type=int,
                       help='Cap number of samples per split (train/val/test)')
    
    # Additional flags
    parser.add_argument('--dry_run', action='store_true',
                       help='Run without actual training (config validation only)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser


def load_config(config_path: Union[str, Path], 
                args: Optional[argparse.Namespace] = None) -> DictConfig:
    """
    Load configuration from YAML file with optional CLI overrides.
    
    Args:
        config_path: Path to YAML configuration file
        args: Optional parsed CLI arguments for overrides
        
    Returns:
        Loaded and processed configuration object
    """
    # Load base configuration
    config_dict = load_yaml_config(config_path)
    config = DictConfig(config_dict)
    
    # Apply CLI overrides if provided
    if args is not None:
        config = apply_cli_overrides(config, args)
    
    # Validate required sections
    _validate_config(config)
    
    return config


def _validate_config(config: DictConfig):
    """Validate that required configuration sections exist."""
    required_sections = ['model', 'training', 'data', 'hardware', 'logging', 'checkpointing']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate data section based on dataset type
    if hasattr(config.data, 'dataset_type'):
        if config.data.dataset_type == 'affectnet' and not hasattr(config.data, 'affectnet_path'):
            logger.warning("AffectNet dataset specified but no affectnet_path provided")
        elif config.data.dataset_type == 'findingemo' and not hasattr(config.data, 'findingemo_path'):
            logger.warning("FindingEmo dataset specified but no findingemo_path provided")
    
    logger.info("✅ Configuration validation passed")


def print_config(config: DictConfig, title: str = "Configuration"):
    """Print configuration in a readable format."""
    print(f"\n🔧 {title}")
    print("=" * 50)
    _print_config_recursive(config.to_dict(), indent=0)
    print("=" * 50)


def _print_config_recursive(config_dict: Dict[str, Any], indent: int = 0):
    """Recursively print configuration dictionary."""
    for key, value in config_dict.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            _print_config_recursive(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")


if __name__ == "__main__":
    # Test configuration loading
    print("🧪 Testing Configuration Management")
    print("=" * 50)
    
    # Test with face model config
    try:
        config_path = Path(__file__).parent.parent.parent / "configs" / "face_model_config.yaml"
        if config_path.exists():
            config = load_config(config_path)
            print_config(config, "Face Model Configuration")
            
            # Test CLI override simulation
            class MockArgs:
                def __init__(self):
                    setattr(self, 'training.learning_rate', 0.0005)
                    setattr(self, 'data.affectnet_path', '/test/path')
                    setattr(self, 'config', str(config_path))
            
            args = MockArgs()
            config_with_overrides = load_config(config_path, args)
            print(f"\n✅ CLI Override Test:")
            print(f"  Original LR: 0.001")
            print(f"  Override LR: {config_with_overrides.training.learning_rate}")
            print(f"  AffectNet path: {config_with_overrides.data.affectnet_path}")
        else:
            print("⚠️  Face model config not found, skipping test")
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
    
    print("\n🎯 Configuration utilities ready!")
