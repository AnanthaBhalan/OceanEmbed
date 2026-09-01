"""
Evaluation Metrics for Ocean Temperature Reconstruction
Depth-wise RMSE, Pearson correlation, MAE, and Bias calculations.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yaml


class OceanMetrics:
    """Compute evaluation metrics for ocean temperature reconstruction."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize metrics calculator.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.depth_levels = np.array(self.config['depth_levels'])
        self.regions = self.config['validation']['regions']
    
    def compute_rmse(self, pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
        """Compute Root Mean Squared Error.
        
        Args:
            pred: Predicted values
            target: Target values
            mask: Valid points mask
            
        Returns:
            RMSE value
        """
        if mask is not None:
            mask = np.asarray(mask).astype(bool)
            pred = pred[mask]
            target = target[mask]
        
        # Remove NaN values
        valid = ~(np.isnan(pred) | np.isnan(target))
        pred = pred[valid]
        target = target[valid]
        
        if len(pred) == 0:
            return np.nan
        
        return np.sqrt(mean_squared_error(target, pred))
    
    def compute_mae(self, pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
        """Compute Mean Absolute Error.
        
        Args:
            pred: Predicted values
            target: Target values
            mask: Valid points mask
            
        Returns:
            MAE value
        """
        if mask is not None:
            mask = np.asarray(mask).astype(bool)
            pred = pred[mask]
            target = target[mask]
        
        # Remove NaN values
        valid = ~(np.isnan(pred) | np.isnan(target))
        pred = pred[valid]
        target = target[valid]
        
        if len(pred) == 0:
            return np.nan
        
        return mean_absolute_error(target, pred)
    
    def compute_bias(self, pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
        """Compute Bias (mean error).
        
        Args:
            pred: Predicted values
            target: Target values
            mask: Valid points mask
            
        Returns:
            Bias value
        """
        if mask is not None:
            mask = np.asarray(mask).astype(bool)
            pred = pred[mask]
            target = target[mask]
        
        # Remove NaN values
        valid = ~(np.isnan(pred) | np.isnan(target))
        pred = pred[valid]
        target = target[valid]
        
        if len(pred) == 0:
            return np.nan
        
        return np.mean(pred - target)
    
    def compute_correlation(self, pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
        """Compute Pearson correlation coefficient.
        
        Args:
            pred: Predicted values
            target: Target values
            mask: Valid points mask
            
        Returns:
            Correlation coefficient
        """
        if mask is not None:
            mask = np.asarray(mask).astype(bool)
            pred = pred[mask]
            target = target[mask]
        
        # Remove NaN values
        valid = ~(np.isnan(pred) | np.isnan(target))
        pred = pred[valid]
        target = target[valid]
        
        if len(pred) < 2:
            return np.nan
        
        try:
            r, _ = pearsonr(pred, target)
            return r
        except:
            return np.nan
    
    def compute_depth_wise_metrics(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray = None
    ) -> Dict[str, List[float]]:
        """Compute metrics for each depth level.
        
        Args:
            pred: Predicted temperature (D, H, W) or (B, D, H, W)
            target: Target temperature (D, H, W) or (B, D, H, W)
            mask: Valid ocean mask (H, W) or (B, H, W)
            
        Returns:
            Dictionary with depth-wise metrics
        """
        # Handle batch dimension
        if pred.ndim == 4:
            # Average over batch
            pred = pred.reshape(-1, pred.shape[-3], pred.shape[-2], pred.shape[-1])
            target = target.reshape(-1, target.shape[-3], target.shape[-2], target.shape[-1])
            if mask is not None:
                mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])
        
        if pred.ndim == 3:
            pred = pred[np.newaxis, ...]
            target = target[np.newaxis, ...]
            if mask is not None:
                mask = mask[np.newaxis, ...]
        
        B, D, H, W = pred.shape
        
        metrics = {
            'depth': self.depth_levels.tolist(),
            'rmse': [],
            'mae': [],
            'bias': [],
            'correlation': []
        }
        
        for d in range(D):
            pred_d = pred[:, d, :, :].flatten()
            target_d = target[:, d, :, :].flatten()
            
            if mask is not None:
                # mask is (1, H, W) [single shared mask -> broadcast to batch]
                # or (B, H, W) [already per-batch]. Tile only in the shared case.
                if mask.shape[0] == 1:
                    mask_d = np.tile(mask, (B, 1, 1)).flatten()
                else:
                    mask_d = mask.flatten()
            else:
                mask_d = None
            
            # Compute metrics
            metrics['rmse'].append(self.compute_rmse(pred_d, target_d, mask_d))
            metrics['mae'].append(self.compute_mae(pred_d, target_d, mask_d))
            metrics['bias'].append(self.compute_bias(pred_d, target_d, mask_d))
            metrics['correlation'].append(self.compute_correlation(pred_d, target_d, mask_d))
        
        return metrics
    
    def compute_regional_metrics(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        region_name: str
    ) -> Dict[str, List[float]]:
        """Compute metrics for a specific region.
        
        Args:
            pred: Predicted temperature (D, H, W)
            target: Target temperature (D, H, W)
            lat: Latitude array (H,)
            lon: Longitude array (W,)
            region_name: Name of the region ('arabian_sea' or 'bay_of_bengal')
            
        Returns:
            Dictionary with regional metrics
        """
        if region_name not in self.regions:
            raise ValueError(f"Unknown region: {region_name}")
        
        # Get region bounds
        region_bounds = self.regions[region_name]
        lat_range = region_bounds[0]
        lon_range = region_bounds[1]
        
        # Create regional mask
        lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
        lon_mask = (lon >= lon_range[0]) & (lon <= lon_range[1])
        
        # Create 2D mask
        regional_mask = np.outer(lat_mask, lon_mask)
        
        # Compute depth-wise metrics for this region
        metrics = self.compute_depth_wise_metrics(pred, target, regional_mask)
        
        return metrics
    
    def print_metrics_summary(self, metrics: Dict[str, List[float]], title: str = "Metrics Summary"):
        """Print formatted metrics summary.
        
        Args:
            metrics: Dictionary with metrics
            title: Title for the summary
        """
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        print(f"{'Depth (m)':>10} {'RMSE':>10} {'MAE':>10} {'Bias':>10} {'Corr':>10}")
        print(f"{'-'*60}")
        
        for i, depth in enumerate(metrics['depth']):
            print(f"{depth:10.0f} "
                  f"{metrics['rmse'][i]:10.4f} "
                  f"{metrics['mae'][i]:10.4f} "
                  f"{metrics['bias'][i]:10.4f} "
                  f"{metrics['correlation'][i]:10.4f}")
        
        # Summary statistics
        print(f"{'-'*60}")
        print(f"{'Mean':>10} "
              f"{np.nanmean(metrics['rmse']):10.4f} "
              f"{np.nanmean(metrics['mae']):10.4f} "
              f"{np.nanmean(metrics['bias']):10.4f} "
              f"{np.nanmean(metrics['correlation']):10.4f}")
        print(f"{'='*60}\n")


def compute_depth_wise_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
    depth_levels: List[float] = None
) -> Dict[str, np.ndarray]:
    """Convenience function to compute depth-wise metrics from PyTorch tensors.
    
    Args:
        pred: Predicted temperature tensor (B, D, H, W)
        target: Target temperature tensor (B, D, H, W)
        mask: Valid ocean mask (B, 1, H, W) or (B, H, W)
        depth_levels: List of depth levels
        
    Returns:
        Dictionary with depth-wise metrics
    """
    # Convert to numpy
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    if mask is not None:
        mask_np = mask.detach().cpu().numpy()
        if mask_np.ndim == 4:
            mask_np = mask_np[:, 0, :, :]  # Extract first channel
    else:
        mask_np = None
    
    # Create metrics calculator
    if depth_levels is not None:
        # Temporary config
        import tempfile
        import json
        temp_config = {
            'depth_levels': depth_levels,
            'validation': {'regions': {}}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(temp_config, f)
            config_path = f.name
        
        metrics_calc = OceanMetrics(config_path)
    else:
        metrics_calc = OceanMetrics()
    
    # Compute metrics
    metrics = metrics_calc.compute_depth_wise_metrics(pred_np, target_np, mask_np)
    
    return metrics


if __name__ == "__main__":
    # Test metrics
    print("Testing OceanMetrics...")
    
    # Create dummy data
    B, D, H, W = 4, 15, 101, 241
    pred = np.random.randn(B, D, H, W) * 5 + 20
    target = pred + np.random.randn(B, D, H, W) * 1
    mask = np.ones((B, H, W))
    
    # Compute metrics
    metrics_calc = OceanMetrics("config.yaml")
    metrics = metrics_calc.compute_depth_wise_metrics(pred, target, mask)
    
    # Print summary
    metrics_calc.print_metrics_summary(metrics, "Test Metrics")
    
    print("Metrics test complete!")
