"""
ARGO Float Validation
Independent validation against in-situ ARGO float measurements.
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

from .metrics import OceanMetrics


class ArgoValidator:
    """Validator for comparing model predictions with ARGO float data."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize ARGO validator.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.depth_levels = np.array(self.config['depth_levels'])
        self.metrics_calc = OceanMetrics(config_path)
    
    def load_argo_data(self, argo_file: str) -> xr.Dataset:
        """Load ARGO float data from NetCDF file.
        
        Args:
            argo_file: Path to ARGO NetCDF file
            
        Returns:
            xarray Dataset with ARGO data
        """
        try:
            ds = xr.open_dataset(argo_file)
            return ds
        except Exception as e:
            print(f"Error loading ARGO data: {e}")
            return None
    
    def match_depth_levels(
        self,
        argo_temp: np.ndarray,
        argo_depths: np.ndarray,
        target_depths: np.ndarray = None
    ) -> np.ndarray:
        """Interpolate ARGO measurements to model depth levels.
        
        Args:
            argo_temp: ARGO temperature profile
            argo_depths: ARGO depth levels
            target_depths: Target depth levels for interpolation
            
        Returns:
            Interpolated temperature at target depths
        """
        if target_depths is None:
            target_depths = self.depth_levels
        
        # Remove NaN values
        valid = ~np.isnan(argo_temp)
        if np.sum(valid) < 2:
            return np.full_like(target_depths, np.nan, dtype=np.float32)
        
        argo_temp_valid = argo_temp[valid]
        argo_depths_valid = argo_depths[valid]
        
        # Interpolate to target depths
        interpolated = np.interp(
            target_depths,
            argo_depths_valid,
            argo_temp_valid,
            left=np.nan,
            right=np.nan
        )
        
        return interpolated
    
    def spatiotemporal_match(
        self,
        model_data: xr.Dataset,
        argo_data: xr.Dataset,
        max_dist_km: float = 50.0,
        max_time_hours: float = 24.0
    ) -> List[Tuple[int, int]]:
        """Find spatiotemporal matches between model and ARGO data.
        
        Args:
            model_data: Model prediction dataset
            argo_data: ARGO observation dataset
            max_dist_km: Maximum spatial distance for matching (km)
            max_time_hours: Maximum temporal distance for matching (hours)
            
        Returns:
            List of (model_idx, argo_idx) matched pairs
        """
        matches = []
        
        # This is a simplified matching algorithm
        # In practice, you would implement more sophisticated matching
        # considering ocean dynamics and float trajectories
        
        # For mock implementation, return empty list
        # Real implementation would involve:
        # 1. Haversine distance calculation
        # 2. Temporal alignment
        # 3. Quality control of ARGO profiles
        
        return matches
    
    def validate_against_argo(
        self,
        predictions: np.ndarray,
        argo_profiles: np.ndarray,
        argo_depths: np.ndarray
    ) -> Dict[str, float]:
        """Validate model predictions against ARGO profiles.
        
        Args:
            predictions: Model predictions (N, D)
            argo_profiles: ARGO temperature profiles (N, D_argo)
            argo_depths: ARGO depth levels (D_argo,)
            
        Returns:
            Dictionary with validation metrics
        """
        N = predictions.shape[0]
        
        # Interpolate ARGO data to model depth levels
        argo_interpolated = np.zeros((N, len(self.depth_levels)))
        for i in range(N):
            argo_interpolated[i] = self.match_depth_levels(
                argo_profiles[i],
                argo_depths,
                self.depth_levels
            )
        
        # Compute metrics
        metrics = {
            'rmse': [],
            'mae': [],
            'bias': [],
            'correlation': []
        }
        
        for d in range(len(self.depth_levels)):
            pred_d = predictions[:, d]
            argo_d = argo_interpolated[:, d]
            
            # Remove NaN values
            valid = ~(np.isnan(pred_d) | np.isnan(argo_d))
            if np.sum(valid) > 0:
                metrics['rmse'].append(
                    self.metrics_calc.compute_rmse(pred_d[valid], argo_d[valid])
                )
                metrics['mae'].append(
                    self.metrics_calc.compute_mae(pred_d[valid], argo_d[valid])
                )
                metrics['bias'].append(
                    self.metrics_calc.compute_bias(pred_d[valid], argo_d[valid])
                )
                metrics['correlation'].append(
                    self.metrics_calc.compute_correlation(pred_d[valid], argo_d[valid])
                )
            else:
                metrics['rmse'].append(np.nan)
                metrics['mae'].append(np.nan)
                metrics['bias'].append(np.nan)
                metrics['correlation'].append(np.nan)
        
        # Add depth information
        metrics['depth'] = self.depth_levels.tolist()
        
        return metrics
    
    def generate_validation_report(
        self,
        metrics: Dict[str, List[float]],
        output_file: str = "argo_validation_report.txt"
    ):
        """Generate and save ARGO validation report.
        
        Args:
            metrics: Dictionary with validation metrics
            output_file: Path to save report
        """
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ARGO Float Independent Validation Report\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"{'Depth (m)':>12} {'RMSE':>12} {'MAE':>12} {'Bias':>12} {'Corr':>12}\n")
            f.write("-"*70 + "\n")
            
            for i, depth in enumerate(metrics['depth']):
                f.write(f"{depth:12.0f} "
                       f"{metrics['rmse'][i]:12.4f} "
                       f"{metrics['mae'][i]:12.4f} "
                       f"{metrics['bias'][i]:12.4f} "
                       f"{metrics['correlation'][i]:12.4f}\n")
            
            f.write("-"*70 + "\n")
            f.write(f"{'Mean':>12} "
                   f"{np.nanmean(metrics['rmse']):12.4f} "
                   f"{np.nanmean(metrics['mae']):12.4f} "
                   f"{np.nanmean(metrics['bias']):12.4f} "
                   f"{np.nanmean(metrics['correlation']):12.4f}\n")
            f.write("="*70 + "\n\n")
            
            # Summary statistics
            f.write("Summary:\n")
            f.write(f"  Average RMSE: {np.nanmean(metrics['rmse']):.4f} °C\n")
            f.write(f"  Average MAE: {np.nanmean(metrics['mae']):.4f} °C\n")
            f.write(f"  Average Bias: {np.nanmean(metrics['bias']):.4f} °C\n")
            f.write(f"  Average Correlation: {np.nanmean(metrics['correlation']):.4f}\n")
            
            # Depth-stratified analysis
            f.write("\nDepth-Stratified Performance:\n")
            f.write(f"  Surface (0-100m) RMSE: {np.nanmean(metrics['rmse'][:9]):.4f} °C\n")
            f.write(f"  Thermocline (100-300m) RMSE: {np.nanmean(metrics['rmse'][9:12]):.4f} °C\n")
            f.write(f"  Deep (>300m) RMSE: {np.nanmean(metrics['rmse'][12:]):.4f} °C\n")
        
        print(f"✓ ARGO validation report saved to {output_file}")


if __name__ == "__main__":
    # Test ARGO validator
    print("Testing ArgoValidator...")
    
    validator = ArgoValidator("config.yaml")
    
    # Create mock data
    N = 100  # Number of profiles
    D = len(validator.depth_levels)
    
    predictions = np.random.randn(N, D) * 5 + 15
    argo_profiles = predictions + np.random.randn(N, D) * 1
    argo_depths = validator.depth_levels
    
    # Validate
    metrics = validator.validate_against_argo(predictions, argo_profiles, argo_depths)
    
    # Generate report
    validator.generate_validation_report(metrics, "test_argo_report.txt")
    
    print("✓ ARGO validator test complete!")
