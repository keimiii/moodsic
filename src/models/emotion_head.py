"""
Emotion prediction heads for continuous Valence-Arousal prediction.
Maps visual features to V-A coordinates with configurable architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import logging

logger = logging.getLogger(__name__)


class EmotionHead(nn.Module):
    """
    Neural network head for predicting continuous valence and arousal values.
    
    Takes visual features and outputs V-A coordinates in specified range.
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dims: List[int] = None,
                 output_range: Tuple[float, float] = (-1.0, 1.0),
                 dropout_rate: float = 0.1,
                 activation: str = "relu",
                 output_activation: str = "tanh",
                 batch_norm: bool = True,
                 residual_connections: bool = False):
        """
        Initialize emotion prediction head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_range: Output value range (min, max)
            dropout_rate: Dropout probability
            activation: Hidden layer activation function
            output_activation: Output layer activation
            batch_norm: Whether to use batch normalization
            residual_connections: Whether to use residual connections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [256, 128]
        self.output_range = output_range
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.residual_connections = residual_connections
        
        # Build network layers
        self.layers = self._build_network(activation)
        
        # Output projection to 2D (valence, arousal)
        self.output_proj = nn.Linear(self.hidden_dims[-1], 2)
        
        # Output activation
        self.output_activation = self._get_activation(output_activation)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"🧠 EmotionHead created: {input_dim} -> {self.hidden_dims} -> 2")
        logger.info(f"  Output range: {self.output_range}")
        logger.info(f"  Dropout: {self.dropout_rate}")
        logger.info(f"  Batch norm: {self.batch_norm}")
    
    def _build_network(self, activation: str) -> nn.ModuleList:
        """Build the main network layers."""
        layers = nn.ModuleList()
        
        # Input dimension
        prev_dim = self.input_dim
        
        # Hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Linear layer
            linear = nn.Linear(prev_dim, hidden_dim)
            layers.append(linear)
            
            # Batch normalization
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self._get_activation(activation))
            
            # Dropout
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            prev_dim = hidden_dim
        
        return layers
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU()
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        return activations[activation]
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through emotion head.
        
        Args:
            features: Input features [batch_size, input_dim]
            
        Returns:
            Tuple of (valence, arousal) tensors [batch_size]
        """
        x = features
        
        # Pass through network layers
        for layer in self.layers:
            if self.residual_connections and isinstance(layer, nn.Linear):
                # Simple residual connection (only if dimensions match)
                if x.shape[-1] == layer.out_features:
                    x = x + layer(x)
                else:
                    x = layer(x)
            else:
                x = layer(x)
        
        # Output projection
        va_output = self.output_proj(x)  # [batch_size, 2]
        
        # Apply output activation
        if self.output_activation is not None:
            va_output = self.output_activation(va_output)
        
        # Scale to output range
        if self.output_range != (-1.0, 1.0):
            # Scale from [-1, 1] to [min, max]
            min_val, max_val = self.output_range
            va_output = va_output * (max_val - min_val) / 2.0 + (max_val + min_val) / 2.0
        
        # Split into valence and arousal
        valence = va_output[:, 0]
        arousal = va_output[:, 1]
        
        return valence, arousal
    
    def predict_va(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict V-A values and return as dictionary.
        
        Args:
            features: Input features
            
        Returns:
            Dictionary with valence, arousal, and combined predictions
        """
        valence, arousal = self.forward(features)
        
        return {
            'valence': valence,
            'arousal': arousal,
            'va_vector': torch.stack([valence, arousal], dim=1)
        }


class MultiHeadEmotionHead(nn.Module):
    """
    Multi-head emotion prediction with separate heads for valence and arousal.
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dims: List[int] = None,
                 output_range: Tuple[float, float] = (-1.0, 1.0),
                 dropout_rate: float = 0.1,
                 activation: str = "relu"):
        """
        Initialize multi-head emotion predictor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions for each head
            output_range: Output value range
            dropout_rate: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [256, 128]
        self.output_range = output_range
        
        # Shared feature processing
        self.shared_layers = self._build_shared_layers(activation, dropout_rate)
        
        # Separate heads for valence and arousal
        head_input_dim = self.hidden_dims[-1] if self.hidden_dims else input_dim
        
        self.valence_head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_input_dim // 2, 1),
            nn.Tanh()
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_input_dim // 2, 1),
            nn.Tanh()
        )
        
        logger.info(f"🔄 MultiHeadEmotionHead created with separate V/A heads")
    
    def _build_shared_layers(self, activation: str, dropout_rate: float) -> nn.Sequential:
        """Build shared feature processing layers."""
        if not self.hidden_dims:
            return nn.Identity()
        
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if activation == "relu" else nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through multi-head architecture."""
        # Shared processing
        shared_features = self.shared_layers(features)
        
        # Separate heads
        valence = self.valence_head(shared_features).squeeze(1)
        arousal = self.arousal_head(shared_features).squeeze(1)
        
        # Scale to output range if needed
        if self.output_range != (-1.0, 1.0):
            min_val, max_val = self.output_range
            scale = (max_val - min_val) / 2.0
            offset = (max_val + min_val) / 2.0
            valence = valence * scale + offset
            arousal = arousal * scale + offset
        
        return valence, arousal


class AttentionEmotionHead(nn.Module):
    """
    Emotion head with attention mechanism for focusing on relevant features.
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 256,
                 num_attention_heads: int = 8,
                 output_range: Tuple[float, float] = (-1.0, 1.0),
                 dropout_rate: float = 0.1):
        """
        Initialize attention-based emotion head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for attention
            num_attention_heads: Number of attention heads
            output_range: Output value range
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_range = output_range
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feature processing
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim // 2, 2)
        self.output_activation = nn.Tanh()
        
        logger.info(f"🎯 AttentionEmotionHead created with {num_attention_heads} heads")
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention mechanism."""
        # Add sequence dimension if needed
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # Self-attention
        attended_features, _ = self.attention(features, features, features)
        attended_features = attended_features.squeeze(1)  # Remove sequence dim
        
        # Feature processing
        processed_features = self.feature_proj(attended_features)
        
        # Output projection
        va_output = self.output_activation(self.output_proj(processed_features))
        
        # Scale to output range
        if self.output_range != (-1.0, 1.0):
            min_val, max_val = self.output_range
            scale = (max_val - min_val) / 2.0
            offset = (max_val + min_val) / 2.0
            va_output = va_output * scale + offset
        
        valence = va_output[:, 0]
        arousal = va_output[:, 1]
        
        return valence, arousal


class CombinedEmotionHead(nn.Module):
    """
    Combined emotion head for both V-A regression and Emo8 classification.
    
    Research findings require multi-task learning with:
    - Primary task: V-A regression 
    - Auxiliary task: Emo8 classification for comparability
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 va_hidden_dims: List[int] = None,
                 emo8_hidden_dims: List[int] = None,
                 output_range: Tuple[float, float] = (-1.0, 1.0),
                 dropout_rate: float = 0.1,
                 activation: str = "relu",
                 shared_layers: bool = True,
                 emo8_num_classes: int = 8):
        """
        Initialize combined emotion head.
        
        Args:
            input_dim: Input feature dimension
            va_hidden_dims: Hidden dimensions for V-A regression head
            emo8_hidden_dims: Hidden dimensions for Emo8 classification head
            output_range: V-A output range
            dropout_rate: Dropout rate
            activation: Activation function
            shared_layers: Whether to use shared feature processing
            emo8_num_classes: Number of Emo8 classes
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.va_hidden_dims = va_hidden_dims or [256, 128]
        self.emo8_hidden_dims = emo8_hidden_dims or [256, 128]
        self.output_range = output_range
        self.dropout_rate = dropout_rate
        self.shared_layers = shared_layers
        self.emo8_num_classes = emo8_num_classes
        
        # Shared feature processing (if enabled)
        if shared_layers:
            shared_hidden_dim = max(self.va_hidden_dims[0], self.emo8_hidden_dims[0])
            self.shared_network = nn.Sequential(
                nn.Linear(input_dim, shared_hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(shared_hidden_dim)
            )
            va_input_dim = shared_hidden_dim
            emo8_input_dim = shared_hidden_dim
        else:
            self.shared_network = nn.Identity()
            va_input_dim = input_dim
            emo8_input_dim = input_dim
        
        # V-A regression head
        self.va_head = self._build_va_head(va_input_dim, activation)
        
        # Emo8 classification head
        self.emo8_head = self._build_emo8_head(emo8_input_dim, activation)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"🎭 CombinedEmotionHead created:")
        logger.info(f"  V-A head: {va_input_dim} -> {self.va_hidden_dims} -> 2")
        logger.info(f"  Emo8 head: {emo8_input_dim} -> {self.emo8_hidden_dims} -> {emo8_num_classes}")
        logger.info(f"  Shared layers: {shared_layers}")
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU()
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        return activations[activation]
    
    def _build_va_head(self, input_dim: int, activation: str) -> nn.Module:
        """Build V-A regression head."""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.va_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(self.dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer with tanh activation for [-1, 1] range
        layers.extend([
            nn.Linear(prev_dim, 2),
            nn.Tanh()
        ])
        
        return nn.Sequential(*layers)
    
    def _build_emo8_head(self, input_dim: int, activation: str) -> nn.Module:
        """Build Emo8 classification head."""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.emo8_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(self.dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation - will use softmax in loss)
        layers.append(nn.Linear(prev_dim, self.emo8_num_classes))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through combined head.
        
        Args:
            features: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary with V-A and Emo8 predictions
        """
        # Shared feature processing
        shared_features = self.shared_network(features)
        
        # V-A regression
        va_output = self.va_head(shared_features)  # [batch_size, 2]
        
        # Scale V-A to output range if needed
        if self.output_range != (-1.0, 1.0):
            min_val, max_val = self.output_range
            scale = (max_val - min_val) / 2.0
            offset = (max_val + min_val) / 2.0
            va_output = va_output * scale + offset
        
        valence = va_output[:, 0]
        arousal = va_output[:, 1]
        
        # Emo8 classification
        emo8_logits = self.emo8_head(shared_features)  # [batch_size, num_classes]
        
        return {
            'valence': valence,
            'arousal': arousal,
            'va_vector': va_output,
            'emo8_logits': emo8_logits,
            'emo8_probs': torch.softmax(emo8_logits, dim=1)
        }
    
    def predict_va_only(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict only V-A values (for compatibility)."""
        outputs = self.forward(features)
        return outputs['valence'], outputs['arousal']


def create_emotion_head(head_type: str = "standard",
                       input_dim: int = 768,
                       **kwargs) -> nn.Module:
    """
    Factory function to create emotion heads.
    
    Args:
        head_type: Type of head ("standard", "multi_head", "attention", "combined")
        input_dim: Input feature dimension
        **kwargs: Additional head-specific arguments
        
    Returns:
        Emotion head instance
    """
    head_classes = {
        'standard': EmotionHead,
        'multi_head': MultiHeadEmotionHead,
        'attention': AttentionEmotionHead,
        'combined': CombinedEmotionHead
    }
    
    if head_type not in head_classes:
        raise ValueError(f"Unknown head type: {head_type}")
    
    head_class = head_classes[head_type]
    return head_class(input_dim=input_dim, **kwargs)


if __name__ == "__main__":
    # Test emotion heads
    print("🧪 Testing Emotion Heads")
    print("=" * 40)
    
    # Test configurations
    test_configs = [
        ("Standard Head", "standard", {}),
        ("Multi-Head", "multi_head", {}),
        ("Attention Head", "attention", {"num_attention_heads": 4}),
        ("Custom Standard", "standard", {
            "hidden_dims": [512, 256, 128],
            "dropout_rate": 0.2,
            "activation": "gelu"
        })
    ]
    
    input_dim = 768
    batch_size = 8
    test_features = torch.randn(batch_size, input_dim)
    
    for name, head_type, kwargs in test_configs:
        print(f"\n🔧 Testing {name}:")
        print("-" * 25)
        
        try:
            # Create head
            head = create_emotion_head(
                head_type=head_type,
                input_dim=input_dim,
                **kwargs
            )
            
            # Test forward pass
            with torch.no_grad():
                valence, arousal = head(test_features)
            
            print(f"  Input shape: {test_features.shape}")
            print(f"  Valence shape: {valence.shape}")
            print(f"  Arousal shape: {arousal.shape}")
            print(f"  Valence range: [{valence.min():.3f}, {valence.max():.3f}]")
            print(f"  Arousal range: [{arousal.min():.3f}, {arousal.max():.3f}]")
            
            # Count parameters
            num_params = sum(p.numel() for p in head.parameters())
            print(f"  Parameters: {num_params:,}")
            
            print(f"  ✅ {name} passed")
            
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
    
    print(f"\n🎯 Emotion head testing completed!")
