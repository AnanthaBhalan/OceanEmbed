"""
Depth Reconstruction Decoder
Multi-scale decoder for reconstructing 15 depth-level temperature fields.
"""

import torch
import torch.nn as nn
from typing import List
from einops import rearrange


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv(x)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention module."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class UpsampleBlock(nn.Module):
    """Upsampling block with attention and skip connections."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )
        
        # Fusion of upsampled and skip features
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.use_attention = use_attention
        if use_attention:
            self.spatial_attn = SpatialAttention(out_channels)
            self.channel_attn = ChannelAttention(out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, C_in, H, W)
            skip: Skip connection tensor (B, C_skip, H*2, W*2)
            
        Returns:
            Output tensor (B, C_out, H*2, W*2)
        """
        # Upsample
        x = self.upsample(x)
        
        # Ensure spatial dimensions match
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Fusion
        x = self.fusion(x)
        
        # Attention
        if self.use_attention:
            x = self.spatial_attn(x)
            x = self.channel_attn(x)
        
        return x


class DepthReconstructionDecoder(nn.Module):
    """Decoder for reconstructing 15 depth-level temperature fields."""
    
    def __init__(
        self,
        embed_dim: int = 512,
        encoder_dims: List[int] = [96, 192, 384, 768],
        hidden_dims: List[int] = [256, 128, 64],
        num_depths: int = 15,
        use_attention: bool = True,
        target_shape: tuple = (101, 241)
    ):
        """Initialize decoder.
        
        Args:
            embed_dim: Encoder embedding dimension
            encoder_dims: Channel dimensions from encoder stages
            hidden_dims: Hidden dimensions for decoder stages
            num_depths: Number of depth levels to reconstruct (15)
            use_attention: Whether to use attention mechanisms
            target_shape: Target spatial shape (H, W)
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_depths = num_depths
        self.target_shape = target_shape
        
        # Initial projection
        self.init_proj = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling blocks with skip connections
        self.up_blocks = nn.ModuleList()
        
        # Stage 1: 256 channels
        self.up_blocks.append(
            UpsampleBlock(
                in_channels=hidden_dims[0],
                skip_channels=encoder_dims[3],  # 768
                out_channels=hidden_dims[0],
                use_attention=use_attention
            )
        )
        
        # Stage 2: 128 channels
        self.up_blocks.append(
            UpsampleBlock(
                in_channels=hidden_dims[0],
                skip_channels=encoder_dims[2],  # 384
                out_channels=hidden_dims[1],
                use_attention=use_attention
            )
        )
        
        # Stage 3: 64 channels
        self.up_blocks.append(
            UpsampleBlock(
                in_channels=hidden_dims[1],
                skip_channels=encoder_dims[1],  # 192
                out_channels=hidden_dims[2],
                use_attention=use_attention
            )
        )
        
        # Stage 4: Further upsampling
        self.up_blocks.append(
            UpsampleBlock(
                in_channels=hidden_dims[2],
                skip_channels=encoder_dims[0],  # 96
                out_channels=hidden_dims[2],
                use_attention=use_attention
            )
        )
        
        # Final upsampling to target resolution
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[2], 32, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Depth-wise prediction heads (separate head for each depth level)
        self.depth_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1)
            )
            for _ in range(num_depths)
        ])
        
        # Depth-aware feature modulation
        self.depth_embedding = nn.Embedding(num_depths, 32)
        self.depth_modulation = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.Sigmoid()
        )
    
    def forward(self, embedding: torch.Tensor, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass.
        
        Args:
            embedding: Encoder embedding (B, 512, H', W')
            encoder_features: List of encoder features for skip connections
            
        Returns:
            Temperature predictions (B, 15, 101, 241)
        """
        B = embedding.shape[0]
        
        # Initial projection
        x = self.init_proj(embedding)  # (B, 256, H', W')
        
        # Reverse encoder features for skip connections (coarse to fine)
        skip_features = encoder_features[::-1]
        
        # Upsampling with skip connections
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, skip_features[i])
        
        # Final upsampling to target resolution
        x = self.final_upsample(x)  # (B, 32, 101, 241)
        
        # Ensure exact target shape
        if x.shape[2:] != self.target_shape:
            x = nn.functional.interpolate(x, size=self.target_shape, mode='bilinear', align_corners=False)
        
        # Depth-wise predictions
        depth_predictions = []
        
        for depth_idx in range(self.num_depths):
            # Get depth embedding
            depth_emb = self.depth_embedding(torch.tensor([depth_idx], device=x.device))
            depth_emb = depth_emb.expand(B, -1)  # (B, 32)
            
            # Depth modulation
            depth_mod = self.depth_modulation(depth_emb)  # (B, 32)
            depth_mod = depth_mod.view(B, 32, 1, 1)  # (B, 32, 1, 1)
            
            # Modulate features
            x_modulated = x * depth_mod
            
            # Predict temperature at this depth
            temp = self.depth_heads[depth_idx](x_modulated)  # (B, 1, 101, 241)
            depth_predictions.append(temp)
        
        # Stack depth predictions
        output = torch.cat(depth_predictions, dim=1)  # (B, 15, 101, 241)
        
        return output


if __name__ == "__main__":
    # Test decoder
    decoder = DepthReconstructionDecoder(
        embed_dim=512,
        encoder_dims=[96, 192, 384, 768],
        hidden_dims=[256, 128, 64],
        num_depths=15
    )
    
    # Create mock encoder features
    B = 2
    features = [
        torch.randn(B, 96, 25, 60),
        torch.randn(B, 192, 13, 30),
        torch.randn(B, 384, 7, 15),
        torch.randn(B, 768, 4, 8)
    ]
    embedding = torch.randn(B, 512, 4, 8)
    
    output = decoder(embedding, features)
    print(f"Output shape: {output.shape}")
    assert output.shape == (B, 15, 101, 241), f"Expected (2, 15, 101, 241), got {output.shape}"
