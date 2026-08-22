"""
Satellite Embedding Encoder
High-capacity ConvNeXt-based encoder for surface ocean observations.
Spatiotemporal variant with ConvLSTM for 7-day sequence processing.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import timm
from einops import rearrange

from .convlstm import SpatiotemporalEncoder as ConvLSTMEncoder


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with depthwise convolution and inverted bottleneck."""
    
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        """Initialize ConvNeXt block.
        
        Args:
            dim: Number of input/output channels
            drop_path: Drop path rate
            layer_scale_init_value: Initial value for layer scale
        """
        super().__init__()
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, C, H, W)
        """
        input_x = x
        
        # Depthwise convolution
        x = self.dwconv(x)
        
        # Permute to (B, H, W, C) for LayerNorm
        x = x.permute(0, 2, 3, 1)
        
        # Feedforward network
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Layer scaling
        if self.gamma is not None:
            x = self.gamma * x
        
        # Permute back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Residual connection
        x = input_x + self.drop_path(x)
        
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        
        return output


class SatelliteEmbeddingEncoder(nn.Module):
    """ConvNeXt-based encoder for satellite surface observations."""
    
    def __init__(
        self,
        in_channels: int = 8,
        embed_dim: int = 512,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.1
    ):
        """Initialize encoder.
        
        Args:
            in_channels: Number of input channels (8 surface parameters)
            embed_dim: Final embedding dimension
            depths: Number of blocks at each stage
            dims: Channel dimensions at each stage
            drop_path_rate: Stochastic depth rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Stem: patchify with 4x4 conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm([dims[0], 25, 60], eps=1e-6)  # Adjusted for 101x241 -> 25x60
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(4):
            # Downsampling between stages (except first)
            if i > 0:
                downsample = nn.Sequential(
                    nn.LayerNorm([dims[i-1], 25 // (2**(i-1)), 60 // (2**(i-1))], eps=1e-6),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2)
                )
            else:
                downsample = nn.Identity()
            
            # ConvNeXt blocks
            blocks = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j]
                )
                for j in range(depths[i])
            ])
            
            self.stages.append(nn.Sequential(downsample, blocks))
            cur += depths[i]
        
        # Final normalization
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        # Projection to embedding dimension
        self.proj = nn.Conv2d(dims[-1], embed_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input surface observations (B, 8, 101, 241)
            
        Returns:
            embedding: Final embedding (B, 512, H', W')
            features: List of intermediate features for skip connections
        """
        # Stem
        x = self.stem(x)  # (B, 96, 25, 60)
        
        # Stages with feature extraction
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
        
        # Final normalization (B, 768, H'', W'')
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Project to embedding dimension
        embedding = self.proj(x)  # (B, 512, H'', W'')
        
        return embedding, features


class AlternativeResNetEncoder(nn.Module):
    """ResNet-based alternative encoder using pretrained weights."""
    
    def __init__(self, in_channels: int = 8, embed_dim: int = 512):
        """Initialize ResNet encoder.
        
        Args:
            in_channels: Number of input channels
            embed_dim: Output embedding dimension
        """
        super().__init__()
        
        # Load pretrained ResNet50
        resnet = timm.create_model('resnet50', pretrained=True, features_only=True)
        
        # Modify first conv for 8 channels
        original_conv = resnet.conv1
        self.conv1 = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Initialize with average of RGB weights
        with torch.no_grad():
            self.conv1.weight[:, :3] = original_conv.weight
            self.conv1.weight[:, 3:] = original_conv.weight.mean(dim=1, keepdim=True).repeat(1, 5, 1, 1)
        
        # Get ResNet stages
        self.bn1 = resnet.bn1
        self.act1 = resnet.act1
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Projection to embedding dimension
        self.proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input tensor (B, 8, H, W)
            
        Returns:
            embedding, features
        """
        features = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        features.append(x)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)
        
        embedding = self.proj(x)
        
        return embedding, features


if __name__ == "__main__":
    # Test encoder
    encoder = SatelliteEmbeddingEncoder(
        in_channels=8,
        embed_dim=512,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768]
    )
    
    # Test input
    x = torch.randn(2, 8, 101, 241)
    embedding, features = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Number of feature maps: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  Feature {i}: {feat.shape}")



class SpatiotemporalSatelliteEncoder(nn.Module):
    """
    Spatiotemporal encoder combining ConvNeXt spatial features with ConvLSTM temporal processing.
    Processes 7-day sequences: (B, T=7, C=8, H=101, W=241) -> (B, C_embed, H', W')
    """
    
    def __init__(
        self,
        in_channels: int = 8,
        embed_dim: int = 512,
        spatial_dims: List[int] = [96, 192, 384, 768],
        spatial_depths: List[int] = [3, 3, 9, 3],
        lstm_hidden_channels: List[int] = [64, 128, 256],
        lstm_num_layers: int = 3,
        drop_path_rate: float = 0.1
    ):
        """Initialize spatiotemporal encoder.
        
        Args:
            in_channels: Number of input channels per timestep (8 surface params)
            embed_dim: Final embedding dimension
            spatial_dims: Channel dimensions for ConvNeXt stages
            spatial_depths: Depth of each ConvNeXt stage
            lstm_hidden_channels: Hidden channels for ConvLSTM layers
            lstm_num_layers: Number of ConvLSTM layers
            drop_path_rate: Stochastic depth rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Spatial feature extractor (applied per timestep)
        self.spatial_encoder = SatelliteEmbeddingEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=spatial_depths,
            dims=spatial_dims,
            drop_path_rate=drop_path_rate
        )
        
        # Temporal encoder (ConvLSTM on spatial features)
        # Input to LSTM will be the embedding dimension from spatial encoder
        self.temporal_encoder = ConvLSTMEncoder(
            input_channels=embed_dim,
            hidden_channels=lstm_hidden_channels,
            kernel_size=3,
            num_layers=lstm_num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        
        # Final projection (optional, can be identity if lstm output = embed_dim)
        final_lstm_dim = lstm_hidden_channels[-1]
        if final_lstm_dim != embed_dim:
            self.final_proj = nn.Conv2d(final_lstm_dim, embed_dim, kernel_size=1)
        else:
            self.final_proj = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass through spatiotemporal encoder.
        
        Args:
            x: Input sequence (B, T=7, C=8, H=101, W=241)
            
        Returns:
            final_embedding: Final spatiotemporal embedding (B, embed_dim, H', W')
            lstm_cell_state: Final cell state from ConvLSTM (B, C_hidden, H', W')
            spatial_features: List of spatial features from final timestep (for skip connections)
        """
        B, T, C, H, W = x.shape
        
        # Process each timestep through spatial encoder
        spatial_embeddings = []
        spatial_features_list = []
        
        for t in range(T):
            # Extract spatial features for this timestep
            embedding_t, features_t = self.spatial_encoder(x[:, t, :, :, :])
            spatial_embeddings.append(embedding_t)
            spatial_features_list.append(features_t)
        
        # Stack spatial embeddings: (B, T, C_embed, H', W')
        spatial_embeddings = torch.stack(spatial_embeddings, dim=1)
        
        # Process through temporal encoder (ConvLSTM)
        temporal_output, last_states = self.temporal_encoder(spatial_embeddings)
        
        # Get final hidden and cell states
        final_h, final_c = last_states[-1]  # From last ConvLSTM layer
        
        # Project to embedding dimension
        final_embedding = self.final_proj(final_h)
        
        # Use spatial features from the final timestep for skip connections
        spatial_features = spatial_features_list[-1]
        
        return final_embedding, final_c, spatial_features


class LightweightSpatiotemporalEncoder(nn.Module):
    """
    Lightweight spatiotemporal encoder that directly processes sequences.
    Alternative to the two-stage approach for faster inference.
    """
    
    def __init__(
        self,
        in_channels: int = 8,
        embed_dim: int = 512,
        lstm_hidden_channels: List[int] = [64, 128, 256, 512],
        lstm_num_layers: int = 4
    ):
        """Initialize lightweight encoder.
        
        Args:
            in_channels: Number of input channels per timestep
            embed_dim: Final embedding dimension
            lstm_hidden_channels: Hidden channels for each LSTM layer
            lstm_num_layers: Number of LSTM layers
        """
        super().__init__()
        
        # Direct ConvLSTM on input (no spatial encoder preprocessing)
        self.temporal_encoder = ConvLSTMEncoder(
            input_channels=in_channels,
            hidden_channels=lstm_hidden_channels,
            kernel_size=3,
            num_layers=lstm_num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=True  # Return all for skip connections
        )
        
        # Project to embedding dimension
        final_lstm_dim = lstm_hidden_channels[-1]
        if final_lstm_dim != embed_dim:
            self.final_proj = nn.Conv2d(final_lstm_dim, embed_dim, kernel_size=1)
        else:
            self.final_proj = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input sequence (B, T=7, C=8, H=101, W=241)
            
        Returns:
            final_embedding: Final embedding (B, embed_dim, H, W)
            lstm_cell_state: Final cell state (B, C_hidden, H, W)
            features: List of hidden states from all LSTM layers (for skip connections)
        """
        # Process through ConvLSTM
        temporal_output, last_states = self.temporal_encoder(x)
        
        # Get final hidden and cell states
        final_h, final_c = last_states[-1]
        
        # Project to embedding dimension
        final_embedding = self.final_proj(final_h)
        
        # Collect hidden states from all layers as "features"
        features = [h for h, c in last_states]
        
        return final_embedding, final_c, features
