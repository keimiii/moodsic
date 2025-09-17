"""
Device management utilities for cross-platform ML training.
Supports CPU, CUDA, and Apple Silicon MPS with automatic detection.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import logging
from typing import Optional, Union, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Comprehensive device management for ML training.
    
    Features:
    - Automatic device detection (CPU/CUDA/MPS)
    - Mixed precision training support
    - Memory management utilities
    - Cross-platform compatibility
    """
    
    def __init__(self, 
                 device: str = "auto",
                 mixed_precision: bool = True,
                 verbose: bool = False):
        """
        Initialize device manager.
        
        Args:
            device: Device selection ("auto", "cpu", "cuda", "mps")
            mixed_precision: Whether to enable automatic mixed precision
            verbose: Whether to print detailed device information
        """
        self.verbose = verbose
        self.device = self._setup_device(device)
        self.mixed_precision = mixed_precision and self._supports_amp()
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler() if self.mixed_precision else None
        
        if self.verbose:
            self._print_device_info()
    
    def _setup_device(self, device_spec: str) -> torch.device:
        """Setup and validate device."""
        if device_spec == "auto":
            return self._auto_detect_device()
        elif device_spec == "cpu":
            return torch.device("cpu")
        elif device_spec == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")
            return torch.device("cuda")
        elif device_spec == "mps":
            if not torch.backends.mps.is_available():
                logger.warning("MPS requested but not available, falling back to CPU")
                return torch.device("cpu")
            return torch.device("mps")
        else:
            raise ValueError(f"Unknown device specification: {device_spec}")
    
    def _auto_detect_device(self) -> torch.device:
        """Automatically detect the best available device."""
        if torch.backends.mps.is_available():
            # Apple Silicon MPS
            return torch.device("mps")
        elif torch.cuda.is_available():
            # NVIDIA CUDA
            return torch.device("cuda")
        else:
            # CPU fallback
            return torch.device("cpu")
    
    def _supports_amp(self) -> bool:
        """Check if automatic mixed precision is supported."""
        if self.device.type == "cuda":
            return True
        elif self.device.type == "cpu":
            return False  # AMP not beneficial on CPU
        elif self.device.type == "mps":
            # MPS supports some AMP operations but can be unstable
            return False  # Disable for stability
        return False
    
    def _print_device_info(self):
        """Print detailed device information."""
        print(f"\n🔧 Device Manager Information")
        print(f"=" * 40)
        print(f"Selected Device: {self.device}")
        print(f"Mixed Precision: {self.mixed_precision}")
        
        if self.device.type == "cuda":
            print(f"CUDA Device: {torch.cuda.get_device_name()}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        elif self.device.type == "mps":
            print(f"Apple Silicon: MPS backend available")
        
        print(f"PyTorch Version: {torch.__version__}")
        print(f"=" * 40)
    
    def to_device(self, obj: Union[torch.Tensor, nn.Module, Dict[str, torch.Tensor]]):
        """
        Move tensor, model, or dictionary of tensors to device.
        
        Args:
            obj: Object to move to device
            
        Returns:
            Object moved to device
        """
        if isinstance(obj, dict):
            return {key: self.to_device(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.to_device(item) for item in obj]
        elif hasattr(obj, 'to'):
            return obj.to(self.device)
        else:
            # Return as-is for non-tensor objects (strings, numbers, etc.)
            return obj
    
    @contextmanager
    def autocast_context(self):
        """
        Context manager for automatic mixed precision.
        
        Usage:
            with device_manager.autocast_context():
                outputs = model(inputs)
        """
        if self.mixed_precision and self.device.type == "cuda":
            with autocast():
                yield
        else:
            yield
    
    def backward(self, loss: torch.Tensor):
        """
        Backward pass with optional gradient scaling.
        
        Args:
            loss: Loss tensor to backpropagate
        """
        if self.mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """
        Optimizer step with optional gradient scaling.
        
        Args:
            optimizer: Optimizer to step
        """
        if self.mixed_precision and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory usage information.
        
        Returns:
            Dictionary with memory statistics
        """
        info = {"device": str(self.device)}
        
        if self.device.type == "cuda":
            info.update({
                "allocated": torch.cuda.memory_allocated(),
                "cached": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated(),
                "total": torch.cuda.get_device_properties(0).total_memory
            })
        elif self.device.type == "mps":
            # MPS doesn't have detailed memory info yet
            info.update({
                "allocated": "N/A (MPS)",
                "cached": "N/A (MPS)",
                "max_allocated": "N/A (MPS)",
                "total": "N/A (MPS)"
            })
        else:
            info.update({
                "allocated": "N/A (CPU)",
                "cached": "N/A (CPU)",
                "max_allocated": "N/A (CPU)",
                "total": "N/A (CPU)"
            })
        
        return info
    
    def clear_cache(self):
        """Clear device memory cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            if self.verbose:
                logger.info("🧹 CUDA cache cleared")
        elif self.device.type == "mps":
            # MPS doesn't have explicit cache clearing
            if self.verbose:
                logger.info("🧹 MPS memory management handled automatically")
    
    def synchronize(self):
        """Synchronize device operations."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            # MPS synchronization
            torch.mps.synchronize()
    
    def set_deterministic(self, deterministic: bool = True):
        """
        Set deterministic behavior for reproducibility.
        
        Args:
            deterministic: Whether to use deterministic algorithms
        """
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # For newer PyTorch versions
            try:
                torch.use_deterministic_algorithms(True)
            except AttributeError:
                # Older PyTorch versions
                pass
            
            if self.verbose:
                logger.info("🎯 Deterministic mode enabled")
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            try:
                torch.use_deterministic_algorithms(False)
            except AttributeError:
                pass
            
            if self.verbose:
                logger.info("🚀 Performance mode enabled (non-deterministic)")
    
    def optimize_for_inference(self):
        """Optimize device settings for inference."""
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        
        # Disable gradient computation globally for inference
        torch.set_grad_enabled(False)
        
        if self.verbose:
            logger.info("⚡ Device optimized for inference")
    
    def get_batch_size_recommendation(self, 
                                    model_size_mb: float,
                                    target_memory_usage: float = 0.8) -> int:
        """
        Recommend batch size based on available memory.
        
        Args:
            model_size_mb: Model size in megabytes
            target_memory_usage: Target memory usage ratio (0.0-1.0)
            
        Returns:
            Recommended batch size
        """
        if self.device.type == "cuda":
            total_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = total_memory * target_memory_usage
            
            # Rough estimation: each sample needs ~4x model size in memory
            memory_per_sample = model_size_mb * 1024 * 1024 * 4
            recommended_batch_size = max(1, int(available_memory / memory_per_sample))
            
            if self.verbose:
                logger.info(f"💡 Recommended batch size: {recommended_batch_size}")
            
            return recommended_batch_size
        else:
            # Conservative estimate for CPU/MPS
            return 16 if model_size_mb < 100 else 8
    
    def profile_memory_usage(self, operation_name: str = "operation"):
        """
        Context manager for profiling memory usage.
        
        Usage:
            with device_manager.profile_memory_usage("forward_pass"):
                outputs = model(inputs)
        """
        return MemoryProfiler(self, operation_name)


class MemoryProfiler:
    """Context manager for memory profiling."""
    
    def __init__(self, device_manager: DeviceManager, operation_name: str):
        self.device_manager = device_manager
        self.operation_name = operation_name
        self.start_memory = None
    
    def __enter__(self):
        if self.device_manager.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device_manager.device.type == "cuda":
            end_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            memory_used = (end_memory - self.start_memory) / 1024**2  # MB
            peak_memory_mb = peak_memory / 1024**2  # MB
            
            logger.info(f"📊 {self.operation_name} memory usage:")
            logger.info(f"  Used: {memory_used:.1f} MB")
            logger.info(f"  Peak: {peak_memory_mb:.1f} MB")


def get_device_manager(device: str = "auto", 
                      mixed_precision: bool = True,
                      verbose: bool = False) -> DeviceManager:
    """
    Factory function to create device manager.
    
    Args:
        device: Device selection
        mixed_precision: Whether to enable mixed precision
        verbose: Whether to print device information
        
    Returns:
        Configured DeviceManager instance
    """
    return DeviceManager(device=device, mixed_precision=mixed_precision, verbose=verbose)


if __name__ == "__main__":
    # Test device management
    print("🧪 Testing Device Management")
    print("=" * 50)
    
    # Test auto-detection
    dm = DeviceManager(device="auto", verbose=True)
    
    # Test tensor operations
    print(f"\n🔧 Testing tensor operations:")
    test_tensor = torch.randn(2, 3, 224, 224)
    device_tensor = dm.to_device(test_tensor)
    print(f"  Original device: {test_tensor.device}")
    print(f"  Moved to: {device_tensor.device}")
    
    # Test mixed precision context
    print(f"\n⚡ Testing mixed precision context:")
    with dm.autocast_context():
        result = device_tensor * 2
        print(f"  Mixed precision operation completed")
    
    # Test memory info
    print(f"\n📊 Memory information:")
    memory_info = dm.get_memory_info()
    for key, value in memory_info.items():
        print(f"  {key}: {value}")
    
    # Test batch size recommendation
    recommended_batch = dm.get_batch_size_recommendation(model_size_mb=100)
    print(f"\n💡 Recommended batch size for 100MB model: {recommended_batch}")
    
    print(f"\n✅ Device management test completed!")
    print(f"🎯 Device utilities ready for training!")