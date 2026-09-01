"""
PyTorch Dataset for OceanEmbed
Efficient data loading for ocean temperature reconstruction.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Tuple, Optional, List
import yaml

from .preprocessor import OceanDataPreprocessor


class OceanDataset(Dataset):
    """PyTorch Dataset for ocean surface-to-subsurface learning with temporal sequences."""
    
    def __init__(
        self,
        data_dir: str,
        preprocessor: OceanDataPreprocessor,
        split: str = 'train',
        sequence_length: int = 7,
        transform: Optional[callable] = None
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing NetCDF files
            preprocessor: Fitted OceanDataPreprocessor instance
            split: Data split ('train', 'val', 'test')
            sequence_length: Number of consecutive days to use as input (default: 7)
            transform: Optional data augmentation transform
        """
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.split = split
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Get file lists
        self.surface_dir = self.data_dir / "surface"
        self.subsurface_dir = self.data_dir / "subsurface"
        
        self.surface_files = sorted(self.surface_dir.glob("*.nc"))
        self.subsurface_files = sorted(self.subsurface_dir.glob("*.nc"))
        
        assert len(self.surface_files) == len(self.subsurface_files), \
            "Mismatch between surface and subsurface file counts"
        
        # Split data first
        self._split_data()
        
        # Create valid sequence indices (sliding window)
        # Each sample i uses files [i, i+1, ..., i+sequence_length-1] as input
        # and file i+sequence_length-1 as target
        self.valid_indices = []
        max_idx = len(self.surface_files) - self.sequence_length + 1
        if max_idx > 0:
            self.valid_indices = list(range(max_idx))
        
        # Optional CPU-PoC subsampling (config: data.window_stride / data.grid_stride).
        # Defaults of 1 preserve full-resolution behaviour.
        with open('config.yaml', 'r') as f:
            _cfg = yaml.safe_load(f)
        self.window_stride = int(_cfg.get('data', {}).get('window_stride', 1))
        self.grid_stride = int(_cfg.get('data', {}).get('grid_stride', 1))
        if self.window_stride > 1:
            self.valid_indices = self.valid_indices[::self.window_stride]
        
        print(f"Loaded {len(self.valid_indices)} valid sequences (length={sequence_length}, "
              f"window_stride={self.window_stride}, grid_stride={self.grid_stride}) for {split} split")
    
    def _split_data(self):
        """Split data into train/val/test sets."""
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        train_ratio = config['data']['train_split']
        val_ratio = config['data']['val_split']
        
        n_total = len(self.surface_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        if self.split == 'train':
            self.surface_files = self.surface_files[:n_train]
            self.subsurface_files = self.subsurface_files[:n_train]
        elif self.split == 'val':
            self.surface_files = self.surface_files[n_train:n_train + n_val]
            self.subsurface_files = self.subsurface_files[n_train:n_train + n_val]
        elif self.split == 'test':
            self.surface_files = self.surface_files[n_train + n_val:]
            self.subsurface_files = self.subsurface_files[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def __len__(self) -> int:
        """Return dataset size (number of valid sequences)."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Get a single sample with temporal sequence.
        
        Args:
            idx: Sample index (into valid_indices)
            
        Returns:
            surface: Surface tensor (T=7, C=8, H=101, W=241)
            target: Subsurface temperature tensor (D=15, H=101, W=241) for final day
            metadata: Dictionary with additional information
        """
        # Get the starting index for this sequence
        start_idx = self.valid_indices[idx]
        
        # Load sequence of surface data (7 consecutive days)
        surface_sequence = []
        for t in range(self.sequence_length):
            file_idx = start_idx + t
            surface_ds = xr.open_dataset(self.surface_files[file_idx])
            surface = self.preprocessor.preprocess_surface(surface_ds)
            surface_sequence.append(surface)
            surface_ds.close()
        
        # Stack into (T, C, H, W)
        surface_sequence = np.stack(surface_sequence, axis=0)
        
        # Load target (subsurface for final day in sequence)
        target_idx = start_idx + self.sequence_length - 1
        subsurface_ds = xr.open_dataset(self.subsurface_files[target_idx])
        target = self.preprocessor.preprocess_subsurface(subsurface_ds)
        
        # Handle NaN values (apply to each timestep)
        for t in range(self.sequence_length):
            surface_sequence[t], target = self.preprocessor.handle_nan_mask(
                surface_sequence[t], target
            )
        
        # Optional spatial sub-sampling (CPU PoC mode; grid_stride=1 keeps full res)
        if self.grid_stride > 1:
            _s = self.grid_stride
            surface_sequence = surface_sequence[..., ::_s, ::_s].copy()
            target = np.ascontiguousarray(target[..., ::_s, ::_s])
        
        # Apply transforms (data augmentation)
        if self.transform is not None:
            surface_sequence, target = self.transform(surface_sequence, target)
        
        # Convert to torch tensors
        surface_tensor = torch.from_numpy(surface_sequence).float()  # (T, C, H, W)
        target_tensor = torch.from_numpy(target).float()  # (D, H, W)
        
        # Metadata (from final day)
        final_surface_ds = xr.open_dataset(self.surface_files[target_idx])
        metadata = {
            'filename': self.surface_files[target_idx].name,
            'sequence_start': self.surface_files[start_idx].name,
            'sequence_end': self.surface_files[target_idx].name,
            'time': str(final_surface_ds.time.values),
            'lat_min': float(final_surface_ds.lat.min()),
            'lat_max': float(final_surface_ds.lat.max()),
            'lon_min': float(final_surface_ds.lon.min()),
            'lon_max': float(final_surface_ds.lon.max()),
            'sequence_length': self.sequence_length,
        }
        
        final_surface_ds.close()
        subsurface_ds.close()
        
        return surface_tensor, target_tensor, metadata


class OceanDataAugmentation:
    """Data augmentation for ocean data with temporal sequences."""
    
    def __init__(self, flip_prob: float = 0.5, noise_std: float = 0.01):
        """Initialize augmentation.
        
        Args:
            flip_prob: Probability of horizontal flip
            noise_std: Standard deviation of Gaussian noise
        """
        self.flip_prob = flip_prob
        self.noise_std = noise_std
    
    def __call__(self, surface: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation.
        
        Args:
            surface: Surface tensor (T, C=8, H, W)
            target: Target tensor (D=15, H, W)
            
        Returns:
            Augmented surface and target
        """
        # Horizontal flip (along longitude)
        if np.random.rand() < self.flip_prob:
            surface = np.flip(surface, axis=3).copy()  # Flip width dimension
            target = np.flip(target, axis=2).copy()
            
            # Flip U-components (indices 3 and 5) across all timesteps
            surface[:, 3] = -surface[:, 3]  # U_curr
            surface[:, 5] = -surface[:, 5]  # U_wind
        
        # Add small Gaussian noise (only to non-mask channels)
        if self.noise_std > 0:
            for t in range(surface.shape[0]):  # For each timestep
                for i in range(7):  # Skip mask channel (index 7)
                    noise = np.random.randn(*surface[t, i].shape) * self.noise_std
                    surface[t, i] = surface[t, i] + noise
            
            # Add noise to target
            for i in range(target.shape[0]):
                noise = np.random.randn(*target[i].shape) * self.noise_std
                target[i] = target[i] + noise
        
        return surface, target


def create_dataloaders(
    data_dir: str,
    preprocessor: OceanDataPreprocessor,
    batch_size: int = 8,
    num_workers: int = 4,
    use_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Directory containing NetCDF files
        preprocessor: Fitted OceanDataPreprocessor
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        use_augmentation: Whether to use data augmentation
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Data augmentation for training
    transform = OceanDataAugmentation() if use_augmentation else None
    
    # Create datasets
    train_dataset = OceanDataset(data_dir, preprocessor, split='train', transform=transform)
    val_dataset = OceanDataset(data_dir, preprocessor, split='val', transform=None)
    test_dataset = OceanDataset(data_dir, preprocessor, split='test', transform=None)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    preprocessor = OceanDataPreprocessor("config.yaml")
    preprocessor.compute_statistics("mock_data")
    
    dataset = OceanDataset("mock_data", preprocessor, split='train')
    print(f"Dataset size: {len(dataset)}")
    
    surface, target, metadata = dataset[0]
    print(f"Surface shape: {surface.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Metadata: {metadata}")
