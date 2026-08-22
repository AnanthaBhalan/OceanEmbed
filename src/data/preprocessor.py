"""
Data Preprocessor for OceanEmbed
Handles normalization, regridding, and NaN masking for multi-source ocean data.
"""

import numpy as np
import xarray as xr
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import pickle
import yaml


class OceanDataPreprocessor:
    """Preprocess ocean data for deep learning."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize preprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.normalization_method = self.config['normalization']['method']
        self.input_channels = self.config['input_channels']
        self.depth_levels = np.array(self.config['depth_levels'])
        
        # Normalization statistics (will be computed from training data)
        self.stats: Dict[str, Dict[str, float]] = {}
        self.fitted = False
    
    def compute_statistics(self, data_dir: str, split: str = 'train'):
        """Compute normalization statistics from training data.
        
        Args:
            data_dir: Directory containing NetCDF files
            split: Data split to use for computing statistics
        """
        print(f"Computing normalization statistics from {split} data...")
        
        surface_dir = Path(data_dir) / "surface"
        subsurface_dir = Path(data_dir) / "subsurface"
        
        # Initialize accumulators for online statistics
        stats = {}
        for channel in self.input_channels:
            stats[channel] = {
                'sum': 0.0,
                'sum_sq': 0.0,
                'count': 0,
                'min': np.inf,
                'max': -np.inf
            }
        
        stats['temperature'] = {
            'sum': 0.0,
            'sum_sq': 0.0,
            'count': 0,
            'min': np.inf,
            'max': -np.inf
        }
        
        # Process all files
        surface_files = sorted(surface_dir.glob("*.nc"))
        subsurface_files = sorted(subsurface_dir.glob("*.nc"))
        
        for surf_file, subsrf_file in zip(surface_files, subsurface_files):
            # Load data
            surf_ds = xr.open_dataset(surf_file)
            subsrf_ds = xr.open_dataset(subsrf_file)
            
            # Process surface channels
            for channel in self.input_channels:
                if channel == 'mask':
                    continue  # Skip mask channel
                
                data = surf_ds[channel].values
                valid_data = data[~np.isnan(data)]
                
                if len(valid_data) > 0:
                    stats[channel]['sum'] += np.sum(valid_data)
                    stats[channel]['sum_sq'] += np.sum(valid_data ** 2)
                    stats[channel]['count'] += len(valid_data)
                    stats[channel]['min'] = min(stats[channel]['min'], np.min(valid_data))
                    stats[channel]['max'] = max(stats[channel]['max'], np.max(valid_data))
            
            # Process temperature
            temp_data = subsrf_ds['temperature'].values
            valid_temp = temp_data[~np.isnan(temp_data)]
            
            if len(valid_temp) > 0:
                stats['temperature']['sum'] += np.sum(valid_temp)
                stats['temperature']['sum_sq'] += np.sum(valid_temp ** 2)
                stats['temperature']['count'] += len(valid_temp)
                stats['temperature']['min'] = min(stats['temperature']['min'], np.min(valid_temp))
                stats['temperature']['max'] = max(stats['temperature']['max'], np.max(valid_temp))
            
            surf_ds.close()
            subsrf_ds.close()
        
        # Compute final statistics
        for key in stats:
            if stats[key]['count'] > 0:
                mean = stats[key]['sum'] / stats[key]['count']
                variance = (stats[key]['sum_sq'] / stats[key]['count']) - (mean ** 2)
                std = np.sqrt(max(variance, 0))
                
                self.stats[key] = {
                    'mean': float(mean),
                    'std': float(std),
                    'min': float(stats[key]['min']),
                    'max': float(stats[key]['max'])
                }
        
        # For mask channel, use binary values
        self.stats['mask'] = {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0}
        
        self.fitted = True
        print("✓ Statistics computation complete")
        
        # Print summary
        print("\nNormalization Statistics:")
        for key, vals in self.stats.items():
            print(f"  {key:12s}: mean={vals['mean']:8.3f}, std={vals['std']:7.3f}, "
                  f"range=[{vals['min']:7.2f}, {vals['max']:7.2f}]")
    
    def normalize(self, data: np.ndarray, channel: str) -> np.ndarray:
        """Normalize data using precomputed statistics.
        
        Args:
            data: Input data array
            channel: Channel name for statistics lookup
            
        Returns:
            Normalized data
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call compute_statistics first.")
        
        if channel not in self.stats:
            raise ValueError(f"No statistics found for channel: {channel}")
        
        stats = self.stats[channel]
        
        if self.normalization_method == 'zscore':
            # Z-score normalization
            if stats['std'] > 1e-6:
                return (data - stats['mean']) / stats['std']
            else:
                return data - stats['mean']
        
        elif self.normalization_method == 'minmax':
            # Min-max normalization to [0, 1]
            range_val = stats['max'] - stats['min']
            if range_val > 1e-6:
                return (data - stats['min']) / range_val
            else:
                return np.zeros_like(data)
        
        elif self.normalization_method == 'robust':
            # Robust scaling (using mean and std as proxies)
            return (data - stats['mean']) / (stats['std'] + 1e-6)
        
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
    
    def denormalize(self, data: np.ndarray, channel: str) -> np.ndarray:
        """Denormalize data back to original scale.
        
        Args:
            data: Normalized data array
            channel: Channel name for statistics lookup
            
        Returns:
            Denormalized data
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted.")
        
        stats = self.stats[channel]
        
        if self.normalization_method == 'zscore':
            return data * stats['std'] + stats['mean']
        
        elif self.normalization_method == 'minmax':
            range_val = stats['max'] - stats['min']
            return data * range_val + stats['min']
        
        elif self.normalization_method == 'robust':
            return data * stats['std'] + stats['mean']
        
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
    
    def preprocess_surface(self, surface_ds: xr.Dataset) -> np.ndarray:
        """Preprocess surface data into normalized tensor.
        
        Args:
            surface_ds: xarray Dataset with surface variables
            
        Returns:
            Normalized tensor of shape (8, H, W)
        """
        channels = []
        
        for channel_name in self.input_channels:
            data = surface_ds[channel_name].values
            
            # Normalize (except mask)
            if channel_name != 'mask':
                data_norm = self.normalize(data, channel_name)
            else:
                data_norm = data
            
            channels.append(data_norm)
        
        # Stack channels
        surface_tensor = np.stack(channels, axis=0).astype(np.float32)
        
        return surface_tensor
    
    def preprocess_subsurface(self, subsurface_ds: xr.Dataset) -> np.ndarray:
        """Preprocess subsurface temperature into normalized tensor.
        
        Args:
            subsurface_ds: xarray Dataset with temperature profile
            
        Returns:
            Normalized tensor of shape (15, H, W)
        """
        temp_data = subsurface_ds['temperature'].values  # (depth, lat, lon)
        
        # Normalize
        temp_norm = self.normalize(temp_data, 'temperature')
        
        return temp_norm.astype(np.float32)
    
    def save_statistics(self, filepath: str):
        """Save normalization statistics to file.
        
        Args:
            filepath: Path to save statistics
        """
        if not self.fitted:
            raise ValueError("No statistics to save. Fit the preprocessor first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'stats': self.stats,
                'normalization_method': self.normalization_method,
                'input_channels': self.input_channels,
                'depth_levels': self.depth_levels.tolist()
            }, f)
        
        print(f"✓ Statistics saved to {filepath}")
    
    def load_statistics(self, filepath: str):
        """Load normalization statistics from file.
        
        Args:
            filepath: Path to load statistics from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.stats = data['stats']
        self.normalization_method = data['normalization_method']
        self.fitted = True
        
        print(f"✓ Statistics loaded from {filepath}")
    
    def handle_nan_mask(self, surface: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Handle NaN values by filling with zeros and updating mask channel.
        
        Args:
            surface: Surface tensor (8, H, W)
            target: Target tensor (15, H, W)
            
        Returns:
            surface_filled, target_filled: Tensors with NaN replaced by 0
        """
        # Mask channel is last (index 7)
        valid_mask = surface[7]  # Ocean mask (1 = ocean, 0 = land)
        
        # Fill NaN with 0
        surface_filled = np.nan_to_num(surface, nan=0.0)
        target_filled = np.nan_to_num(target, nan=0.0)
        
        # Ensure mask is applied consistently
        for i in range(surface.shape[0]):
            surface_filled[i] = surface_filled[i] * valid_mask
        
        for i in range(target.shape[0]):
            target_filled[i] = target_filled[i] * valid_mask
        
        return surface_filled, target_filled


if __name__ == "__main__":
    # Test preprocessor
    preprocessor = OceanDataPreprocessor("config.yaml")
    preprocessor.compute_statistics("mock_data", split='train')
    preprocessor.save_statistics("preprocessor_stats.pkl")
