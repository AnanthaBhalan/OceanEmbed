"""
Physics-Informed Loss Function
Combines reconstruction loss with physical oceanography constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import numpy as np


class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss for ocean temperature reconstruction."""
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        stratification_weight: float = 0.3,
        gradient_smoothness_weight: float = 0.1,
        depth_levels: list = None
    ):
        """Initialize physics-informed loss.
        
        Args:
            mse_weight: Weight for MSE reconstruction loss
            stratification_weight: Weight for thermal stratification penalty
            gradient_smoothness_weight: Weight for spatial gradient smoothness
            depth_levels: List of depth levels in meters
        """
        super().__init__()
        
        self.mse_weight = mse_weight
        self.stratification_weight = stratification_weight
        self.gradient_smoothness_weight = gradient_smoothness_weight
        
        if depth_levels is None:
            self.depth_levels = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000])
        else:
            self.depth_levels = np.array(depth_levels)
    
    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE loss with NaN/land masking.
        
        Args:
            pred: Predicted temperature (B, 15, H, W)
            target: Ground truth temperature (B, 15, H, W)
            mask: Valid ocean mask (B, 1, H, W)
            
        Returns:
            MSE loss
        """
        # Expand mask to all depth levels
        mask = mask.expand_as(pred)
        
        # Compute MSE only for valid ocean points
        mse = (pred - target) ** 2
        mse_masked = mse * mask
        
        # Average over valid points
        loss = mse_masked.sum() / (mask.sum() + 1e-6)
        
        return loss
    
    def stratification_loss(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Penalize physically impossible temperature inversions.
        
        In the ocean, temperature should generally decrease with depth (positive stratification).
        Strong temperature inversions (temperature increasing with depth) are rare and
        should be penalized.
        
        Args:
            pred: Predicted temperature (B, 15, H, W)
            mask: Valid ocean mask (B, 1, H, W)
            
        Returns:
            Stratification penalty
        """
        B, D, H, W = pred.shape
        
        # Compute temperature differences between adjacent depth levels
        # dT/dz should be <= 0 (temperature decreases or stays constant with depth)
        temp_diffs = []
        
        for i in range(D - 1):
            # Temperature difference (deeper - shallower)
            dt = pred[:, i+1] - pred[:, i]  # (B, H, W)
            
            # Depth difference
            dz = self.depth_levels[i+1] - self.depth_levels[i]
            
            # Vertical gradient (°C/m)
            gradient = dt / dz
            
            # Penalize positive gradients (temperature increasing with depth)
            # Use ReLU to only penalize positive inversions
            inversion_penalty = F.relu(gradient)  # Only positive values
            
            temp_diffs.append(inversion_penalty)
        
        # Stack and apply mask
        inversions = torch.stack(temp_diffs, dim=1)  # (B, 14, H, W)
        mask_reduced = mask[:, 0:1, :, :].expand_as(inversions)
        
        inversions_masked = inversions * mask_reduced
        
        # Average penalty
        loss = inversions_masked.sum() / (mask_reduced.sum() + 1e-6)
        
        return loss
    
    def gradient_smoothness_loss(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encourage spatial smoothness in temperature fields.
        
        Ocean temperature fields should have smooth spatial gradients without
        sharp discontinuities (except at fronts, which are rare).
        
        Args:
            pred: Predicted temperature (B, 15, H, W)
            mask: Valid ocean mask (B, 1, H, W)
            
        Returns:
            Gradient smoothness loss
        """
        # Compute spatial gradients using Sobel-like filters
        # Gradient in x-direction
        dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        
        # Gradient in y-direction
        dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        # Compute second-order gradients (smoothness)
        d2x = dx[:, :, :, 1:] - dx[:, :, :, :-1]
        d2y = dy[:, :, 1:, :] - dy[:, :, :-1, :]
        
        # Apply mask (reduced due to gradient operations)
        mask_x = mask[:, :, :, 2:].expand_as(d2x)
        mask_y = mask[:, :, 2:, :].expand_as(d2y)
        
        # L2 norm of second derivatives
        smoothness_x = (d2x ** 2 * mask_x).sum() / (mask_x.sum() + 1e-6)
        smoothness_y = (d2y ** 2 * mask_y).sum() / (mask_y.sum() + 1e-6)
        
        loss = (smoothness_x + smoothness_y) / 2
        
        return loss
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute total physics-informed loss.
        
        Args:
            pred: Predicted temperature (B, 15, H, W)
            target: Ground truth temperature (B, 15, H, W)
            mask: Valid ocean mask (B, 1, H, W) or (B, 8, H, W) - will extract last channel
            
        Returns:
            Dictionary with total loss and individual components
        """
        # Extract mask if it has multiple channels (use last channel - the ocean mask)
        if mask.shape[1] > 1:
            mask = mask[:, -1:, :, :]  # Extract mask channel (B, 1, H, W)
        
        # Compute individual loss components
        mse_loss = self.reconstruction_loss(pred, target, mask)
        strat_loss = self.stratification_loss(pred, mask)
        smooth_loss = self.gradient_smoothness_loss(pred, mask)
        
        # Weighted combination
        total_loss = (
            self.mse_weight * mse_loss +
            self.stratification_weight * strat_loss +
            self.gradient_smoothness_weight * smooth_loss
        )
        
        return {
            'loss': total_loss,
            'mse_loss': mse_loss.detach(),
            'stratification_loss': strat_loss.detach(),
            'smoothness_loss': smooth_loss.detach()
        }


class DepthWeightedMSELoss(nn.Module):
    """MSE loss with depth-dependent weighting."""
    
    def __init__(self, depth_levels: list = None):
        """Initialize depth-weighted loss.
        
        Args:
            depth_levels: List of depth levels in meters
        """
        super().__init__()
        
        if depth_levels is None:
            self.depth_levels = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000])
        else:
            self.depth_levels = np.array(depth_levels)
        
        # Compute depth weights (higher weight for shallow layers)
        weights = 1.0 / np.log1p(self.depth_levels + 1)
        weights = weights / weights.sum()
        
        self.register_buffer('depth_weights', torch.from_numpy(weights).float())
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute depth-weighted MSE loss.
        
        Args:
            pred: Predicted temperature (B, 15, H, W)
            target: Ground truth temperature (B, 15, H, W)
            mask: Valid ocean mask (B, 1, H, W)
            
        Returns:
            Weighted MSE loss
        """
        B, D, H, W = pred.shape
        
        # Expand mask
        mask = mask.expand_as(pred)
        
        # Compute per-depth MSE
        mse_per_depth = []
        for d in range(D):
            mse_d = ((pred[:, d] - target[:, d]) ** 2 * mask[:, d]).sum() / (mask[:, d].sum() + 1e-6)
            mse_per_depth.append(mse_d)
        
        mse_per_depth = torch.stack(mse_per_depth)
        
        # Apply depth weights
        loss = (mse_per_depth * self.depth_weights).sum()
        
        return loss


if __name__ == "__main__":
    # Test loss functions
    B, D, H, W = 2, 15, 101, 241
    
    pred = torch.randn(B, D, H, W)
    target = torch.randn(B, D, H, W)
    mask = torch.ones(B, 1, H, W)
    
    # Physics-informed loss
    criterion = PhysicsInformedLoss()
    loss_dict = criterion(pred, target, mask)
    
    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Depth-weighted loss
    depth_criterion = DepthWeightedMSELoss()
    depth_loss = depth_criterion(pred, target, mask)
    print(f"\nDepth-weighted MSE: {depth_loss.item():.4f}")
