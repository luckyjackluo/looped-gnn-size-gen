"""MLP Encoders and Decoders - unified from ChipGen and TopoGeoNet."""

import torch
import torch.nn as nn
from typing import Optional


class MLPEncoder(nn.Module):
    """
    Multi-layer perceptron encoder for feature processing.
    Merged from ChipGen and TopoGeoNet implementations.

    Supports:
    - Optional positional encoding for coordinates (ChipDiffusion-style)
    - Dual-branch encoding: separate paths for physical and non-physical features
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        activation: str = 'relu',
        use_positional_encoding: bool = False,
        positional_encoding_dim: int = 32,
        positional_encoding_max_freq: float = 100.0,
        physical_feature_dims: int = None,  # NEW: number of physical features (coordinates + sizes)
    ):
        super().__init__()

        # Store encoding parameters
        self.use_positional_encoding = use_positional_encoding
        self.input_dim = input_dim
        self.physical_feature_dims = physical_feature_dims  # NEW
        self.use_dual_branch = physical_feature_dims is not None  # NEW

        # Dual-branch encoding: separate physical and non-physical features
        if self.use_dual_branch:
            # Physical features (coordinates + sizes): apply positional encoding
            physical_dim = physical_feature_dims
            non_physical_dim = input_dim - physical_feature_dims
            
            if use_positional_encoding:
                from .positional import SinusoidalPositionalEncoding
                # Apply positional encoding to first 2 dims of physical features (coordinates)
                self.pos_encoder = SinusoidalPositionalEncoding(
                    input_dim=2,  # Encode x, y coordinates only
                    encoding_dim=positional_encoding_dim,
                    max_freq=positional_encoding_max_freq
                )
                # Physical branch input: coords(2) + pos_encoding + rest_physical_features
                physical_input_dim = 2 + self.pos_encoder.output_dim + (physical_dim - 2)
            else:
                self.pos_encoder = None
                physical_input_dim = physical_dim
            
            # Create separate MLPs for physical and non-physical features
            self.physical_mlp = self._build_mlp(
                physical_input_dim, hidden_dim, hidden_dim, num_layers, 
                dropout, use_layer_norm, activation
            )
            self.non_physical_mlp = self._build_mlp(
                non_physical_dim, hidden_dim, hidden_dim, num_layers,
                dropout, use_layer_norm, activation, clamp_input=True  # Clamp non-physical features
            )
            
            # Final projection to combine branches
            self.final_proj = nn.Linear(hidden_dim * 2, output_dim)
            nn.init.xavier_uniform_(self.final_proj.weight, gain=1.0)
            if self.final_proj.bias is not None:
                nn.init.zeros_(self.final_proj.bias)
            
            total_input_dim = None  # Not used in dual-branch mode
        else:
            # Original single-branch encoding
            # Positional encoding for coordinates (optional, ChipDiffusion-style)
            # Encodes first 2 dimensions (positions) with sinusoidal features
            if use_positional_encoding:
                from .positional import SinusoidalPositionalEncoding
                self.pos_encoder = SinusoidalPositionalEncoding(
                    input_dim=2,  # Always encode 2D positions (x, y)
                    encoding_dim=positional_encoding_dim,
                    max_freq=positional_encoding_max_freq
                )
                # Adjust input dimension: original features + positional encoding
                # Concatenate: [coords, pos_encoding, rest_features]
                # Total = 2 + pos_encoding_output_dim + (input_dim - 2)
                total_input_dim = input_dim + self.pos_encoder.output_dim
            else:
                self.pos_encoder = None
                total_input_dim = input_dim

        # Build single-branch MLP if not using dual-branch
        if not self.use_dual_branch:
            if activation == 'relu':
                act_fn = nn.ReLU()
            elif activation == 'silu':
                act_fn = nn.SiLU()
            elif activation == 'gelu':
                act_fn = nn.GELU()
            else:
                act_fn = nn.ReLU()

            layers = []
            dim = total_input_dim  # Start with total input dimension after encoding

            for i in range(num_layers):
                linear_layer = nn.Linear(dim, hidden_dim if i < num_layers - 1 else output_dim)
                layers.append(linear_layer)
                
                # Initialize weights properly for numerical stability
                # When positional encoding is used, input_dim increases significantly (12 -> 76)
                # Use smaller initialization to prevent large activations
                if use_positional_encoding and i == 0:
                    # First layer receives positional encoding - use smaller initialization
                    nn.init.xavier_uniform_(linear_layer.weight, gain=0.5)
                else:
                    # Standard initialization for other layers
                    nn.init.xavier_uniform_(linear_layer.weight, gain=1.0)
                
                if linear_layer.bias is not None:
                    nn.init.zeros_(linear_layer.bias)

                if i < num_layers - 1:
                    if use_layer_norm:
                        layers.append(nn.LayerNorm(hidden_dim))
                    layers.append(act_fn)
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    dim = hidden_dim

            self.mlp = nn.Sequential(*layers)
    
    def _build_mlp(self, input_dim, hidden_dim, output_dim, num_layers, dropout, use_layer_norm, activation, clamp_input=False):
        """Helper to build MLP layers."""
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU()

        layers = []
        dim = input_dim

        for i in range(num_layers):
            linear_layer = nn.Linear(dim, hidden_dim if i < num_layers - 1 else output_dim)
            
            # Smaller initialization for first layer if positional encoding or clamped input
            if i == 0 and (self.use_positional_encoding or clamp_input):
                nn.init.xavier_uniform_(linear_layer.weight, gain=0.5)
            else:
                nn.init.xavier_uniform_(linear_layer.weight, gain=1.0)
            
            if linear_layer.bias is not None:
                nn.init.zeros_(linear_layer.bias)
            
            layers.append(linear_layer)

            if i < num_layers - 1:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(act_fn)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                dim = hidden_dim

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional positional encoding and dual-branch processing.

        Args:
            x: [N, input_dim] or [B, N, input_dim] input features
                First 2 dimensions are assumed to be coordinates (x, y)
                If use_dual_branch: first physical_feature_dims are physical features,
                                     rest are non-physical features

        Returns:
            encoded: [N, output_dim] or [B, N, output_dim] encoded features
        """
        if self.use_dual_branch:
            # Dual-branch mode: separate physical and non-physical features
            physical_feats = x[:, :self.physical_feature_dims]  # [N, physical_dims]
            non_physical_feats = x[:, self.physical_feature_dims:]  # [N, non_physical_dims]
            
            # Process physical features (with optional positional encoding)
            if self.use_positional_encoding:
                coords = physical_feats[:, :2]  # [N, 2]
                rest_physical = physical_feats[:, 2:]  # [N, physical_dims-2]
                
                # Apply positional encoding to coordinates
                pos_encoding = self.pos_encoder(coords)  # [N, pos_encoding_dim]
                
                # Concatenate: [coords, pos_encoding, rest_physical]
                physical_input = torch.cat([coords, pos_encoding, rest_physical], dim=1)
            else:
                physical_input = physical_feats
            
            # No clamp on physical branch to preserve real size/position info; use gradient clipping if needed
            # Non-physical branch: do not clamp (preserves feature meaning; normalization is in data)
            non_physical_input = non_physical_feats
            
            # Process through separate branches
            physical_out = self.physical_mlp(physical_input)  # [N, hidden_dim]
            non_physical_out = self.non_physical_mlp(non_physical_input)  # [N, hidden_dim]
            
            # Combine branches directly (no normalization)
            combined = torch.cat([physical_out, non_physical_out], dim=1)  # [N, 2*hidden_dim]
            output = self.final_proj(combined)  # [N, output_dim]
            return output
            
        elif self.use_positional_encoding:
            # Extract coordinates (first 2 dims) and rest of features
            if x.dim() == 3:
                # Batched: [B, N, F]
                B, N, F = x.shape
                coords = x[:, :, :2].reshape(-1, 2)  # [B*N, 2]
                rest_feats = x[:, :, 2:].reshape(-1, F - 2)  # [B*N, F-2]
            else:
                # Flat: [N, F]
                coords = x[:, :2]  # [N, 2]
                rest_feats = x[:, 2:]  # [N, F-2]
            
            # No clamp on rest_feats to preserve feature information; use gradient clipping if needed
            pos_encoding = self.pos_encoder(coords)  # [N or B*N, pos_encoding_dim]
            x_with_encoding = torch.cat([coords, pos_encoding, rest_feats], dim=1)
            output = self.mlp(x_with_encoding)

            # Reshape back if batched
            if x.dim() == 3:
                output = output.view(B, N, -1)

            return output
        else:
            # No positional encoding, direct pass through
            return self.mlp(x)


class MLPDecoder(nn.Module):
    """MLP decoder - same as encoder but conceptually separate."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'relu',
        zero_init: bool = False
    ):
        super().__init__()

        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU()

        layers = []
        dim = input_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        final_layer = nn.Linear(dim, output_dim)
        if zero_init:
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

        layers.append(final_layer)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
