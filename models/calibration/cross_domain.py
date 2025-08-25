import torch
import torch.nn as nn
from typing import Tuple, Union, Optional

ArrayLike = Union[float, int, torch.Tensor]

class CrossDomainCalibration(nn.Module):
    """
    Learns affine transformation to align emotion spaces
    across different domains (face vs scene).
    
    Applies: v_out = v_in * scale_v + shift_v
             a_out = a_in * scale_a + shift_a
    
    Parameters are initialized to identity transform (no change).
    """
    
    def __init__(self, l2_reg: float = 1e-4, use_tanh: bool = True):
        super().__init__()
        
        # Learnable scale and shift parameters (initialized to identity)
        self.scale_v = nn.Parameter(torch.ones(1))
        self.scale_a = nn.Parameter(torch.ones(1))
        self.shift_v = nn.Parameter(torch.zeros(1))
        self.shift_a = nn.Parameter(torch.zeros(1))
        
        self.l2_reg = l2_reg
        self.use_tanh = use_tanh
        
    def forward(self, v: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply learned calibration to valence and arousal.
        
        Args:
            v: Valence tensor in [-1, 1]
            a: Arousal tensor in [-1, 1]
            
        Returns:
            Calibrated (v, a) tensors in [-1, 1]
        """
        v_calibrated = v * self.scale_v + self.shift_v
        a_calibrated = a * self.scale_a + self.shift_a
        
        if self.use_tanh:
            # Smooth clamping to avoid gradient saturation
            v_calibrated = torch.tanh(v_calibrated)
            a_calibrated = torch.tanh(a_calibrated)
        else:
            # Hard clamping
            v_calibrated = torch.clamp(v_calibrated, -1.0, 1.0)
            a_calibrated = torch.clamp(a_calibrated, -1.0, 1.0)
            
        return v_calibrated, a_calibrated
    
    def get_regularization_loss(self) -> torch.Tensor:
        """L2 regularization on parameters to prevent drift from identity."""
        reg_loss = (
            (self.scale_v - 1.0) ** 2 +
            (self.scale_a - 1.0) ** 2 +
            self.shift_v ** 2 +
            self.shift_a ** 2
        )
        return self.l2_reg * reg_loss
    
    def get_params_summary(self) -> dict:
        """Get current parameter values for monitoring."""
        return {
            'scale_v': self.scale_v.item(),
            'scale_a': self.scale_a.item(), 
            'shift_v': self.shift_v.item(),
            'shift_a': self.shift_a.item()
        }
    
    def is_near_identity(self, tolerance: float = 0.1) -> bool:
        """Check if parameters are close to identity transform."""
        params = self.get_params_summary()
        return (
            abs(params['scale_v'] - 1.0) < tolerance and
            abs(params['scale_a'] - 1.0) < tolerance and
            abs(params['shift_v']) < tolerance and
            abs(params['shift_a']) < tolerance
        )
