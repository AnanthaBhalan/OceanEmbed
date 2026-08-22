"""
ConvLSTM Cell and Spatiotemporal Encoder for Ocean Dynamics
Captures mesoscale eddies, kinetic momentum, and thermal memory across 7-day sequences.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell that preserves spatial structure.
    Processes (H, W) grids while maintaining temporal dependencies.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True
    ):
        """Initialize ConvLSTM Cell.
        
        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden state channels
            kernel_size: Convolution kernel size (default: 3)
            bias: Whether to use bias (default: True)
        """
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Combined convolution for input and forget gates, cell and output gates
        # Total: 4 gates (input, forget, cell, output)
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
    
    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ConvLSTM cell.
        
        Args:
            x: Input tensor (B, C_in, H, W)
            h_prev: Previous hidden state (B, C_hidden, H, W)
            c_prev: Previous cell state (B, C_hidden, H, W)
            
        Returns:
            h_next: Next hidden state (B, C_hidden, H, W)
            c_next: Next cell state (B, C_hidden, H, W)
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)  # (B, C_in + C_hidden, H, W)
        
        # Compute all gates in one convolution
        combined_conv = self.conv(combined)  # (B, 4 * C_hidden, H, W)
        
        # Split into 4 gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        
        # Apply activations
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell gate (candidate)
        
        # Update cell state
        c_next = f * c_prev + i * g
        
        # Update hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size: int, spatial_size: Tuple[int, int], device: torch.device):
        """Initialize hidden and cell states.
        
        Args:
            batch_size: Batch size
            spatial_size: (height, width) of the spatial grid
            device: Device to create tensors on
            
        Returns:
            h: Initial hidden state (B, C_hidden, H, W)
            c: Initial cell state (B, C_hidden, H, W)
        """
        height, width = spatial_size
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c


class SpatiotemporalEncoder(nn.Module):
    """
    Multi-layer Spatiotemporal Encoder using stacked ConvLSTM cells.
    Processes 7-day sequences to capture ocean dynamics:
    - Mesoscale eddies (50-500 km)
    - Kinetic momentum transfer
    - Thermal memory and stratification
    """
    
    def __init__(
        self,
        input_channels: int = 8,
        hidden_channels: List[int] = [64, 128, 256],
        kernel_size: int = 3,
        num_layers: int = 3,
        batch_first: bool = True,
        bias: bool = True,
        return_all_layers: bool = False
    ):
        """Initialize Spatiotemporal Encoder.
        
        Args:
            input_channels: Number of input channels (default: 8 for ocean variables)
            hidden_channels: List of hidden channels for each layer
            kernel_size: Convolution kernel size (default: 3)
            num_layers: Number of ConvLSTM layers (default: 3)
            batch_first: If True, input shape is (B, T, C, H, W), else (T, B, C, H, W)
            bias: Whether to use bias in convolutions
            return_all_layers: Whether to return hidden states from all layers
        """
        super(SpatiotemporalEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        # Validate hidden_channels
        if len(hidden_channels) != num_layers:
            raise ValueError(f"len(hidden_channels) must equal num_layers")
        
        # Create ConvLSTM cells for each layer
        cell_list = []
        for i in range(num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_channels=cur_input_channels,
                    hidden_channels=hidden_channels[i],
                    kernel_size=kernel_size,
                    bias=bias
                )
            )
        
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through spatiotemporal encoder.
        
        Args:
            x: Input tensor (B, T, C, H, W) if batch_first=True
            hidden_state: Initial hidden states for all layers (optional)
            
        Returns:
            layer_output: Output from final layer (B, T, C_hidden[-1], H, W)
            last_state_list: List of (h, c) tuples for each layer at final timestep
        """
        if not self.batch_first:
            # Convert (T, B, C, H, W) -> (B, T, C, H, W)
            x = x.permute(1, 0, 2, 3, 4)
        
        batch_size, seq_len, _, height, width = x.size()
        device = x.device
        
        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = []
            for i in range(self.num_layers):
                h, c = self.cell_list[i].init_hidden(
                    batch_size=batch_size,
                    spatial_size=(height, width),
                    device=device
                )
                hidden_state.append((h, c))
        
        # Process sequence through layers
        layer_output_list = []
        layer_state_list = []
        
        cur_layer_input = x
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # Process each timestep
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    x=cur_layer_input[:, t, :, :, :],
                    h_prev=h,
                    c_prev=c
                )
                output_inner.append(h)
            
            # Stack outputs across time
            layer_output = torch.stack(output_inner, dim=1)  # (B, T, C_hidden, H, W)
            
            # Store final states
            layer_state_list.append((h, c))
            layer_output_list.append(layer_output)
            
            # Input for next layer is output of current layer
            cur_layer_input = layer_output
        
        if not self.return_all_layers:
            # Return only the last layer's output
            layer_output_list = layer_output_list[-1:]
        
        return layer_output_list[-1], layer_state_list
    
    def get_final_states(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get final hidden and cell states (for decoder).
        
        Args:
            x: Input tensor (B, T, C, H, W)
            
        Returns:
            final_h: Final hidden state from last layer (B, C_hidden[-1], H, W)
            final_c: Final cell state from last layer (B, C_hidden[-1], H, W)
        """
        _, last_state_list = self.forward(x)
        final_h, final_c = last_state_list[-1]
        return final_h, final_c


if __name__ == "__main__":
    # Test ConvLSTM components
    print("Testing ConvLSTMCell...")
    cell = ConvLSTMCell(input_channels=8, hidden_channels=64, kernel_size=3)
    
    # Create dummy input
    B, C, H, W = 2, 8, 101, 241
    x = torch.randn(B, C, H, W)
    h_prev, c_prev = cell.init_hidden(B, (H, W), x.device)
    
    h_next, c_next = cell(x, h_prev, c_prev)
    print(f"  Input: {x.shape}")
    print(f"  Hidden: {h_next.shape}")
    print(f"  Cell: {c_next.shape}")
    
    print("\nTesting SpatiotemporalEncoder...")
    encoder = SpatiotemporalEncoder(
        input_channels=8,
        hidden_channels=[64, 128, 256],
        num_layers=3
    )
    
    # Create 7-day sequence
    B, T, C, H, W = 2, 7, 8, 101, 241
    x_seq = torch.randn(B, T, C, H, W)
    
    layer_output, last_states = encoder(x_seq)
    print(f"  Input: {x_seq.shape}")
    print(f"  Output: {layer_output.shape}")
    print(f"  Number of layers: {len(last_states)}")
    
    for i, (h, c) in enumerate(last_states):
        print(f"  Layer {i} - h: {h.shape}, c: {c.shape}")
    
    # Test final state extraction
    final_h, final_c = encoder.get_final_states(x_seq)
    print(f"\n  Final hidden: {final_h.shape}")
    print(f"  Final cell: {final_c.shape}")
    
    print("\n✓ ConvLSTM tests passed!")
