"""
OceanEmbedNet: Complete Model
Combines encoder and decoder for end-to-end ocean temperature reconstruction.
Supports both single-day (2D) and spatiotemporal (7-day) modes.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import yaml
import numpy as np

from .encoder import (
    SatelliteEmbeddingEncoder,
    AlternativeResNetEncoder,
    SpatiotemporalSatelliteEncoder,
    LightweightSpatiotemporalEncoder
)
from .decoder import DepthReconstructionDecoder
from .loss import PhysicsInformedLoss


class OceanEmbedNet(nn.Module):
    """Complete OceanEmbed model for subsurface temperature reconstruction."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize OceanEmbedNet.
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        model_config = self.config['model']
        encoder_config = model_config['encoder']
        decoder_config = model_config['decoder']
        
        # Initialize encoder
        if encoder_config['type'] == 'convnext':
            self.encoder = SatelliteEmbeddingEncoder(
                in_channels=8,
                embed_dim=encoder_config['embed_dim'],
                depths=encoder_config['depths'],
                dims=encoder_config['dims']
            )
            encoder_dims = encoder_config['dims']
        elif encoder_config['type'] == 'resnet':
            self.encoder = AlternativeResNetEncoder(
                in_channels=8,
                embed_dim=encoder_config['embed_dim']
            )
            encoder_dims = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown encoder type: {encoder_config['type']}")
        
        # Initialize decoder
        self.decoder = DepthReconstructionDecoder(
            embed_dim=encoder_config['embed_dim'],
            encoder_dims=encoder_dims,
            hidden_dims=decoder_config['hidden_dims'],
            num_depths=len(self.config['depth_levels']),
            use_attention=decoder_config['use_attention'],
            target_shape=tuple(self.config['domain']['grid_shape'])
        )
        
        # Initialize loss function
        loss_config = self.config['loss']
        self.criterion = PhysicsInformedLoss(
            mse_weight=loss_config['mse_weight'],
            stratification_weight=loss_config['stratification_weight'],
            gradient_smoothness_weight=loss_config['gradient_smoothness_weight'],
            depth_levels=self.config['depth_levels']
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Surface observations (B, 8, 101, 241)
            
        Returns:
            Predicted subsurface temperature (B, 15, 101, 241)
        """
        # Encode surface observations
        embedding, features = self.encoder(x)
        
        # Decode to subsurface temperature
        temperature = self.decoder(embedding, features)
        
        return temperature
    
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        surface: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute physics-informed loss.
        
        Args:
            pred: Predicted temperature (B, 15, H, W)
            target: Ground truth temperature (B, 15, H, W)
            surface: Surface observations containing mask (B, 8, H, W)
            
        Returns:
            Dictionary with loss components
        """
        # Extract mask from surface observations (last channel)
        mask = surface[:, -1:, :, :]
        
        # Compute loss
        loss_dict = self.criterion(pred, target, mask)
        
        return loss_dict
    
    def predict(self, surface: torch.Tensor, denormalize: bool = False, preprocessor=None) -> torch.Tensor:
        """Make prediction with optional denormalization.
        
        Args:
            surface: Surface observations (B, 8, H, W)
            denormalize: Whether to denormalize predictions
            preprocessor: Preprocessor for denormalization
            
        Returns:
            Predicted temperature in original scale if denormalize=True
        """
        self.eval()
        with torch.no_grad():
            pred = self.forward(surface)
            
            if denormalize and preprocessor is not None:
                # Convert to numpy for denormalization
                pred_np = pred.cpu().numpy()
                
                # Denormalize each depth level
                pred_denorm = []
                for d in range(pred_np.shape[1]):
                    temp_d = preprocessor.denormalize(pred_np[:, d], 'temperature')
                    pred_denorm.append(temp_d)
                
                pred = torch.from_numpy(np.stack(pred_denorm, axis=1))
        
        return pred
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params,
            'trainable': trainable_params
        }


class OceanEmbedLightningModule(nn.Module):
    """PyTorch Lightning wrapper for OceanEmbedNet (optional)."""
    
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__()
        self.model = OceanEmbedNet(config_path)
        self.config_path = config_path
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        surface, target, metadata = batch
        pred = self.model(surface)
        loss_dict = self.model.compute_loss(pred, target, surface)
        
        # Log metrics
        for key, value in loss_dict.items():
            self.log(f'train_{key}', value, on_step=True, on_epoch=True)
        
        return loss_dict['loss']
    
    def validation_step(self, batch, batch_idx):
        surface, target, metadata = batch
        pred = self.model(surface)
        loss_dict = self.model.compute_loss(pred, target, surface)
        
        # Log metrics
        for key, value in loss_dict.items():
            self.log(f'val_{key}', value, on_epoch=True)
        
        return loss_dict['loss']
    
    def configure_optimizers(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        train_config = config['training']
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config['num_epochs']
        )
        
        return [optimizer], [scheduler]


if __name__ == "__main__":
    import numpy as np
    
    # Test OceanEmbedNet
    print("Initializing OceanEmbedNet...")
    model = OceanEmbedNet("config.yaml")
    
    # Print parameter counts
    params = model.count_parameters()
    print("\nModel Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    B = 2
    surface = torch.randn(B, 8, 101, 241)
    target = torch.randn(B, 15, 101, 241)
    
    # Forward
    pred = model(surface)
    print(f"Input shape: {surface.shape}")
    print(f"Output shape: {pred.shape}")
    
    # Compute loss
    loss_dict = model.compute_loss(pred, target, surface)
    print("\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\nModel test complete!")



class OceanSpatiotemporalNet(nn.Module):
    """
    Spatiotemporal OceanEmbed model for 7-day sequence processing.
    Input: (B, T=7, C=8, H=101, W=241)
    Output: (B, D=15, H=101, W=241)
    """
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        encoder_type: str = "full",  # "full" or "lightweight"
        use_lstm_cell_state: bool = True
    ):
        """Initialize OceanSpatiotemporalNet.
        
        Args:
            config_path: Path to configuration file
            encoder_type: Type of encoder ("full" uses ConvNeXt+LSTM, "lightweight" uses LSTM only)
            use_lstm_cell_state: Whether to use LSTM cell state in decoder
        """
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        model_config = self.config['model']
        encoder_config = model_config['encoder']
        decoder_config = model_config['decoder']
        
        self.encoder_type = encoder_type
        self.use_lstm_cell_state = use_lstm_cell_state
        
        # Initialize spatiotemporal encoder
        if encoder_type == "full":
            self.encoder = SpatiotemporalSatelliteEncoder(
                in_channels=8,
                embed_dim=encoder_config['embed_dim'],
                spatial_dims=encoder_config['dims'],
                spatial_depths=encoder_config['depths'],
                lstm_hidden_channels=[64, 128, 256],
                lstm_num_layers=3,
                drop_path_rate=0.1
            )
            encoder_dims = encoder_config['dims']
        elif encoder_type == "lightweight":
            self.encoder = LightweightSpatiotemporalEncoder(
                in_channels=8,
                embed_dim=encoder_config['embed_dim'],
                lstm_hidden_channels=[64, 128, 256, 512],
                lstm_num_layers=4
            )
            # For lightweight, features are LSTM hidden states
            encoder_dims = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Initialize decoder
        self.decoder = DepthReconstructionDecoder(
            embed_dim=encoder_config['embed_dim'],
            encoder_dims=encoder_dims,
            hidden_dims=decoder_config['hidden_dims'],
            num_depths=len(self.config['depth_levels']),
            use_attention=decoder_config['use_attention'],
            target_shape=tuple(self.config['domain']['grid_shape'])
        )
        
        # Optional: Fuse LSTM cell state into embedding
        if use_lstm_cell_state:
            lstm_hidden_dim = 256 if encoder_type == "full" else 512
            self.cell_fusion = nn.Sequential(
                nn.Conv2d(encoder_config['embed_dim'] + lstm_hidden_dim,
                         encoder_config['embed_dim'],
                         kernel_size=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.cell_fusion = None
        
        # Initialize loss function
        loss_config = self.config['loss']
        self.criterion = PhysicsInformedLoss(
            mse_weight=loss_config['mse_weight'],
            stratification_weight=loss_config['stratification_weight'],
            gradient_smoothness_weight=loss_config['gradient_smoothness_weight'],
            depth_levels=self.config['depth_levels']
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Surface observations (B, T=7, C=8, H=101, W=241)
            
        Returns:
            Predicted subsurface temperature (B, D=15, H=101, W=241)
        """
        # Encode 7-day sequence
        embedding, lstm_cell, features = self.encoder(x)
        
        # Optionally fuse LSTM cell state with embedding
        if self.cell_fusion is not None:
            # Concatenate hidden and cell states
            combined = torch.cat([embedding, lstm_cell], dim=1)
            embedding = self.cell_fusion(combined)
        
        # Decode to subsurface temperature
        temperature = self.decoder(embedding, features)
        
        return temperature
    
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        surface: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute physics-informed loss.
        
        Args:
            pred: Predicted temperature (B, D=15, H, W)
            target: Ground truth temperature (B, D=15, H, W)
            surface: Surface observations from final timestep (B, T, C=8, H, W)
                     Mask is extracted from last channel of final timestep
            
        Returns:
            Dictionary with loss components
        """
        # Extract mask from last timestep, last channel
        if surface.dim() == 5:  # (B, T, C, H, W)
            mask = surface[:, -1, -1:, :, :]  # Take final day, mask channel
        else:  # (B, C, H, W) - fallback for single timestep
            mask = surface[:, -1:, :, :]
        
        # Compute loss
        loss_dict = self.criterion(pred, target, mask)
        
        return loss_dict
    
    def predict(
        self,
        surface: torch.Tensor,
        denormalize: bool = False,
        preprocessor=None
    ) -> torch.Tensor:
        """Make prediction with optional denormalization.
        
        Args:
            surface: Surface observations (B, T=7, C=8, H, W)
            denormalize: Whether to denormalize predictions
            preprocessor: Preprocessor for denormalization
            
        Returns:
            Predicted temperature in original scale if denormalize=True
        """
        self.eval()
        with torch.no_grad():
            pred = self.forward(surface)
            
            if denormalize and preprocessor is not None:
                # Convert to numpy for denormalization
                pred_np = pred.cpu().numpy()
                
                # Denormalize each depth level
                pred_denorm = []
                for d in range(pred_np.shape[1]):
                    temp_d = preprocessor.denormalize(pred_np[:, d], 'temperature')
                    pred_denorm.append(temp_d)
                
                pred = torch.from_numpy(np.stack(pred_denorm, axis=1))
        
        return pred
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params,
            'trainable': trainable_params
        }


if __name__ == "__main__":
    print("=" * 70)
    print("Testing OceanSpatiotemporalNet (7-Day ConvLSTM Framework)")
    print("=" * 70)
    
    # Test spatiotemporal model
    print("\n1. Initializing OceanSpatiotemporalNet (full encoder)...")
    model_full = OceanSpatiotemporalNet("config.yaml", encoder_type="full")
    
    # Print parameter counts
    params = model_full.count_parameters()
    print("\nModel Parameters (Full Encoder):")
    for key, value in params.items():
        print(f"  {key}: {value:,}")
    
    # Test forward pass with 7-day sequence
    print("\n2. Testing forward pass with 7-day sequence...")
    B, T, C, H, W = 2, 7, 8, 101, 241
    surface_seq = torch.randn(B, T, C, H, W)
    target = torch.randn(B, 15, H, W)
    
    print(f"   Input shape: {surface_seq.shape}")
    
    # Forward
    pred = model_full(surface_seq)
    print(f"   Output shape: {pred.shape}")
    assert pred.shape == (B, 15, H, W), f"Expected (2, 15, 101, 241), got {pred.shape}"
    
    # Compute loss
    print("\n3. Computing physics-informed loss...")
    loss_dict = model_full.compute_loss(pred, target, surface_seq)
    print("   Loss components:")
    for key, value in loss_dict.items():
        print(f"     {key}: {value.item():.4f}")
    
    # Test lightweight encoder
    print("\n4. Testing lightweight encoder...")
    model_light = OceanSpatiotemporalNet("config.yaml", encoder_type="lightweight")
    
    params_light = model_light.count_parameters()
    print("   Model Parameters (Lightweight):")
    for key, value in params_light.items():
        print(f"     {key}: {value:,}")
    
    pred_light = model_light(surface_seq)
    print(f"   Output shape: {pred_light.shape}")
    
    print("\n" + "=" * 70)
    print("All spatiotemporal tests passed!")
    print("=" * 70)
