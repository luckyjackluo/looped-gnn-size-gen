"""
Fully Convolutional Network (FCN) U-Net for handling different image sizes.

This module implements a pure FCN U-Net that can handle input images of different sizes
from sample to sample, maintaining the same input-output size relationship.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class _ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for (N, C, H, W) tensors."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W) -> (N, H, W, C) -> norm -> (N, C, H, W)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


def _create_fcn_conv_block(in_channels: int, out_channels: int, dropout: float = 0.1, use_layer_norm: bool = True) -> nn.Sequential:
    """
    Create a FCN convolution block with two convolutions.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        dropout: Dropout rate
        use_layer_norm: Whether to use layer normalization (over channels)
        
    Returns:
        nn.Sequential: Convolution block
    """
    layers = []
    
    # First convolution
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    if use_layer_norm:
        layers.append(_ChannelLayerNorm2d(out_channels))
    layers.append(nn.SiLU())
    layers.append(nn.Dropout2d(dropout))
    
    # Second convolution
    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    if use_layer_norm:
        layers.append(_ChannelLayerNorm2d(out_channels))
    layers.append(nn.SiLU())
    layers.append(nn.Dropout2d(dropout))
    
    return nn.Sequential(*layers)


class FCNUNet(nn.Module):
    """
    Fully Convolutional Network (FCN) U-Net for handling different image sizes.
    
    This U-Net implementation can handle input images of different sizes from sample to sample,
    maintaining the same input-output size relationship. It uses adaptive pooling and 
    interpolation to ensure consistent processing regardless of input size.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        base_channels (int): Base number of channels (doubled at each level)
        depth (int): Number of encoder/decoder levels
        dropout (float): Dropout rate
        use_layer_norm (bool): Whether to use layer normalization (over channels)
        activation (str): Final activation function ('sigmoid', 'softmax', 'none')
        min_size (int): Minimum input size to handle (default: 32)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        activation: str = 'none',
        min_size: int = 32
    ):
        super().__init__()
        
        self.depth = depth
        self.activation = activation
        self.min_size = min_size
        self.base_channels = base_channels
        
        # Input convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            _ChannelLayerNorm2d(base_channels) if use_layer_norm else nn.Identity(),
            nn.SiLU()
        )
        
        # Encoder blocks
        self.encoders = nn.ModuleList()
        in_ch = base_channels
        
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(_create_fcn_conv_block(in_ch, out_ch, dropout, use_layer_norm))
            in_ch = out_ch
        
        # Bottleneck
        bottleneck_channels = base_channels * (2 ** depth)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, bottleneck_channels, kernel_size=3, padding=1),
            _ChannelLayerNorm2d(bottleneck_channels) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            _ChannelLayerNorm2d(bottleneck_channels) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Dropout2d(dropout)
        )
        
        # Decoder blocks
        self.decoders = nn.ModuleList()
        
        for i in range(depth):
            level = depth - 1 - i  # Reverse order
            
            # Input channels from previous decoder (or bottleneck)
            if i == 0:
                dec_in_channels = bottleneck_channels
            else:
                dec_in_channels = base_channels * (2 ** (level + 1))
            
            # Skip connection channels from corresponding encoder
            skip_channels = base_channels * (2 ** level)
            
            # Output channels
            dec_out_channels = base_channels * (2 ** level)
            
            # Create decoder block with adaptive upsampling
            decoder = FCNDecoderBlock(
                dec_in_channels, 
                skip_channels, 
                dec_out_channels, 
                dropout, 
                use_layer_norm
            )
            self.decoders.append(decoder)
        
        # Final output convolution
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Final activation
        if activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif activation == 'tanh':
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FCN U-Net.
        
        Args:
            x: Input tensor [B, C, H, W] where H, W can vary between samples
            
        Returns:
            torch.Tensor: Output predictions [B, out_channels, H, W] (same size as input)
        """
        input_size = x.shape[2:]  # Store original input size (H, W)
        
        # Initial convolution
        x = self.input_conv(x)
        
        # Encoder path with adaptive pooling
        skip_connections = []
        current = x
        
        for i, encoder in enumerate(self.encoders):
            # Apply convolution block
            current = encoder(current)
            # Store features before pooling for skip connection
            skip_connections.append(current)
            # Apply adaptive pooling to ensure consistent downsampling
            current = self._adaptive_downsample(current)
        
        # Bottleneck
        current = self.bottleneck(current)
        
        # Decoder path with adaptive upsampling
        for i, decoder in enumerate(self.decoders):
            # Get corresponding skip connection (in reverse order)
            skip_idx = self.depth - 1 - i
            skip = skip_connections[skip_idx]
            
            # Apply decoder with adaptive upsampling
            current = decoder(current, skip)
        
        # Final output convolution
        output = self.output_conv(current)
        
        # Ensure output matches input size exactly
        if output.shape[2:] != input_size:
            output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        
        output = self.final_activation(output)
        
        return output
    
    def _adaptive_downsample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adaptive downsampling that works with different input sizes.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Downsampled tensor [B, C, H//2, W//2]
        """
        # Use adaptive average pooling for consistent downsampling
        h, w = x.shape[2], x.shape[3]
        target_h, target_w = max(h // 2, 1), max(w // 2, 1)
        
        if h > 2 and w > 2:
            # Use max pooling for better feature preservation
            return F.max_pool2d(x, kernel_size=2, stride=2)
        else:
            # Use adaptive pooling for very small sizes
            return F.adaptive_avg_pool2d(x, (target_h, target_w))


class FCNDecoderBlock(nn.Module):
    """
    FCN Decoder block with adaptive upsampling and skip connections.
    
    Args:
        in_channels: Input channels from previous decoder level
        skip_channels: Channels from skip connection
        out_channels: Output channels
        dropout: Dropout rate
        use_layer_norm: Whether to use layer normalization (over channels)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        skip_channels: int, 
        out_channels: int, 
        dropout: float = 0.1, 
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        
        # Upsampling projection
        self.upsample_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        
        # Convolution block after concatenation
        conv_in_channels = (in_channels // 2) + skip_channels
        self.conv_block = _create_fcn_conv_block(conv_in_channels, out_channels, dropout, use_layer_norm)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder block.
        
        Args:
            x: Input from previous decoder level [B, in_channels, H_in, W_in]
            skip: Skip connection [B, skip_channels, H_skip, W_skip]
            
        Returns:
            torch.Tensor: Decoded features [B, out_channels, H_skip, W_skip]
        """
        # Adaptive upsampling to match skip connection size
        skip_size = skip.shape[2:]
        
        # Upsample and project channels
        x = F.interpolate(x, size=skip_size, mode='bilinear', align_corners=True)
        x = self.upsample_conv(x)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Apply convolution block
        x = self.conv_block(x)
        
        return x
    
    def get_output_size(self, input_size: tuple, skip_size: tuple) -> tuple:
        """
        Get output size given input and skip connection sizes.
        
        Args:
            input_size: Size of input tensor (H, W)
            skip_size: Size of skip connection (H, W)
            
        Returns:
            tuple: Output size (H, W) - same as skip_size
        """
        return skip_size


class AdaptiveFCNUNet(FCNUNet):
    """
    Adaptive FCN U-Net that automatically adjusts depth based on input size.
    
    This version automatically reduces the depth for smaller input images to prevent
    over-downsampling and ensures stable training across different image sizes.
    
    Args:
        Same as FCNUNet, but depth is automatically adjusted based on input size
    """
    
    def __init__(self, *args, **kwargs):
        # Store original depth for reference
        self.original_depth = kwargs.get('depth', 4)
        super().__init__(*args, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive depth based on input size.
        
        Args:
            x: Input tensor [B, C, H, W] where H, W can vary between samples
            
        Returns:
            torch.Tensor: Output predictions [B, out_channels, H, W] (same size as input)
        """
        input_size = x.shape[2:]  # Store original input size (H, W)
        min_dim = min(input_size)
        
        # Automatically adjust effective depth based on input size
        # Ensure we don't downsample below 4x4
        max_depth = max(1, int(torch.log2(torch.tensor(min_dim / 4.0)).item()))
        effective_depth = min(self.depth, max_depth)
        
        # Initial convolution
        x = self.input_conv(x)
        
        # Encoder path with adaptive depth
        skip_connections = []
        current = x
        
        for i in range(effective_depth):
            if i < len(self.encoders):
                # Apply convolution block
                current = self.encoders[i](current)
                # Store features before pooling for skip connection
                skip_connections.append(current)
                # Apply adaptive pooling
                current = self._adaptive_downsample(current)
        
        # Bottleneck
        current = self.bottleneck(current)
        
        # Decoder path with adaptive depth
        for i in range(effective_depth):
            if i < len(self.decoders):
                # Get corresponding skip connection (in reverse order)
                skip_idx = effective_depth - 1 - i
                skip = skip_connections[skip_idx]
                
                # Apply decoder
                current = self.decoders[i](current, skip)
        
        # Final output convolution
        output = self.output_conv(current)
        
        # Ensure output matches input size exactly
        if output.shape[2:] != input_size:
            output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        
        output = self.final_activation(output)
        
        return output
