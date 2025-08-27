import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict, Optional
from .cross_domain import CrossDomainCalibration

class CalibrationTrainer:
    """
    Trainer for CrossDomainCalibration layer using CCC + MSE loss.
    """
    
    def __init__(
        self,
        calibration_layer: CrossDomainCalibration,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        ccc_weight: float = 0.7,
        mse_weight: float = 0.3,
        patience: int = 20,
        min_delta: float = 1e-4
    ):
        self.calibration = calibration_layer
        self.optimizer = optim.Adam(
            self.calibration.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.ccc_weight = ccc_weight
        self.mse_weight = mse_weight
        self.patience = patience
        self.min_delta = min_delta
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_ccc_v': [], 'val_ccc_a': []}
    
    def _ccc_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        Concordance Correlation Coefficient loss (1 - CCC).
        Based on existing project CCC implementation.
        """
        pred_mean = pred.mean()
        true_mean = true.mean()
        
        covariance = ((pred - pred_mean) * (true - true_mean)).mean()
        pred_var = pred.var()
        true_var = true.var()
        
        ccc = (2 * covariance) / (pred_var + true_var + (pred_mean - true_mean)**2 + 1e-8)
        return 1.0 - ccc  # Convert to loss (lower is better)
    
    def _combined_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Combined CCC + MSE loss matching project convention (70% CCC + 30% MSE).
        """
        # CCC loss for valence and arousal
        ccc_loss_v = self._ccc_loss(pred[:, 0], target[:, 0])
        ccc_loss_a = self._ccc_loss(pred[:, 1], target[:, 1])
        total_ccc_loss = ccc_loss_v + ccc_loss_a
        
        # MSE loss
        mse_loss = nn.functional.mse_loss(pred, target)
        
        # Combined loss
        combined = self.ccc_weight * total_ccc_loss + self.mse_weight * mse_loss
        
        # Add regularization
        reg_loss = self.calibration.get_regularization_loss()
        
        return combined + reg_loss
    
    def _evaluate_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics (CCC, MAE) for monitoring."""
        with torch.no_grad():
            # CCC for each dimension
            pred_v, pred_a = pred[:, 0], pred[:, 1]
            true_v, true_a = target[:, 0], target[:, 1]
            
            ccc_v = 1.0 - self._ccc_loss(pred_v, true_v).item()
            ccc_a = 1.0 - self._ccc_loss(pred_a, true_a).item()
            
            # MAE
            mae_v = torch.abs(pred_v - true_v).mean().item()
            mae_a = torch.abs(pred_a - true_a).mean().item()
            
            return {
                'ccc_v': ccc_v,
                'ccc_a': ccc_a,
                'mae_v': mae_v,
                'mae_a': mae_a,
                'ccc_avg': (ccc_v + ccc_a) / 2
            }
    
    def fit(
        self, 
        source_pred: np.ndarray,  # EmoNet predictions after scale alignment
        target_labels: np.ndarray,  # FindingEmo ground truth
        val_split: float = 0.2,
        max_epochs: int = 100,
        batch_size: int = 32
    ) -> Dict:
        """
        Train calibration parameters on validation data.
        
        Args:
            source_pred: (N, 2) array of (valence, arousal) predictions
            target_labels: (N, 2) array of ground truth labels
            val_split: Fraction for validation split
            max_epochs: Maximum training epochs
            batch_size: Training batch size
            
        Returns:
            Training history and final metrics
        """
        # Convert to tensors
        source_tensor = torch.tensor(source_pred, dtype=torch.float32)
        target_tensor = torch.tensor(target_labels, dtype=torch.float32)
        
        # Train/val split
        n_val = int(len(source_tensor) * val_split)
        indices = torch.randperm(len(source_tensor))
        
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        train_data = TensorDataset(source_tensor[train_idx], target_tensor[train_idx])
        val_data = TensorDataset(source_tensor[val_idx], target_tensor[val_idx])
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        print(f"Training calibration: {len(train_data)} train, {len(val_data)} val samples")
        
        for epoch in range(max_epochs):
            # Training phase
            self.calibration.train()
            train_loss = 0.0
            
            for batch_pred, batch_target in train_loader:
                self.optimizer.zero_grad()
                
                # Apply calibration
                v_cal, a_cal = self.calibration(batch_pred[:, 0], batch_pred[:, 1])
                calibrated_pred = torch.stack([v_cal, a_cal], dim=1)
                
                # Compute loss
                loss = self._combined_loss(calibrated_pred, batch_target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            val_loss, val_metrics = self._validate(val_loader)
            
            # Logging
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_ccc_v'].append(val_metrics['ccc_v'])
            self.history['val_ccc_a'].append(val_metrics['ccc_a'])
            
            if epoch % 10 == 0:
                params = self.calibration.get_params_summary()
                print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"CCC_v={val_metrics['ccc_v']:.3f}, CCC_a={val_metrics['ccc_a']:.3f}")
                print(f"  Params: scale=({params['scale_v']:.3f}, {params['scale_a']:.3f}), "
                      f"shift=({params['shift_v']:.3f}, {params['shift_a']:.3f})")
            
            # Early stopping
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.patience_counter = 0
                self.best_state = self.calibration.state_dict().copy()
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Restore best parameters
        self.calibration.load_state_dict(self.best_state)
        
        return {
            'history': self.history,
            'final_params': self.calibration.get_params_summary(),
            'best_val_loss': self.best_loss,
            'converged_epoch': epoch - self.patience_counter
        }
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Run validation and return loss + metrics."""
        self.calibration.eval()
        val_loss = 0.0
        all_pred, all_target = [], []
        
        with torch.no_grad():
            for batch_pred, batch_target in val_loader:
                v_cal, a_cal = self.calibration(batch_pred[:, 0], batch_pred[:, 1])
                calibrated_pred = torch.stack([v_cal, a_cal], dim=1)
                
                loss = self._combined_loss(calibrated_pred, batch_target)
                val_loss += loss.item()
                
                all_pred.append(calibrated_pred)
                all_target.append(batch_target)
        
        val_loss /= len(val_loader)
        all_pred = torch.cat(all_pred, dim=0)
        all_target = torch.cat(all_target, dim=0)
        
        metrics = self._evaluate_metrics(all_pred, all_target)
        
        return val_loss, metrics
