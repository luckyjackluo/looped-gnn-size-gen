"""
Residual Gated Adapter for safely integrating transformer outputs with GNN features.

This module allows adding a transformer (global module) to a pretrained GNN model
by learning to gate/scale the transformer output to prevent disruption of learned
GNN representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualGatedAdapter(nn.Module):
    """
    Residual Gated Adapter that learns to combine GNN and transformer outputs.
    
    Architecture:
        output = gnn_output + gate * transformer_output
        
    The gate is learned via a small MLP that takes both GNN and transformer outputs
    and produces a per-feature gating signal. This allows the model to learn when
    to use transformer information vs. rely on GNN features.
    
    Args:
        hidden_dim: Feature dimension
        gate_type: Type of gating mechanism ('simple', 'mlp', 'sigmoid')
        init_gate_scale: Initial scale for gate (default: 0.1, small to start)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        gate_type: str = 'mlp',
        init_gate_scale: float = 0.1,
        use_time_gate: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.gate_type = gate_type
        self.init_gate_scale = init_gate_scale
        self.use_time_gate = use_time_gate
        
        if gate_type == 'simple':
            # Simple learnable scalar gate
            self.gate_weight = nn.Parameter(torch.tensor(init_gate_scale))
            
        elif gate_type == 'sigmoid':
            # Per-feature sigmoid gate (learnable bias)
            self.gate_bias = nn.Parameter(torch.zeros(hidden_dim))
            self.gate_weight = nn.Parameter(torch.ones(hidden_dim) * init_gate_scale)
            
        elif gate_type == 'mlp':
            # MLP-based gate that considers both GNN and transformer outputs
            # This allows adaptive gating based on feature content
            gate_hidden_dim = max(32, hidden_dim // 4)
            self.gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(gate_hidden_dim, hidden_dim),
                nn.Sigmoid()  # Output gate values in [0, 1]
            )
            # Initialize MLP to produce small gate values initially
            with torch.no_grad():
                for layer in self.gate_mlp:
                    if isinstance(layer, nn.Linear):
                        layer.weight.data *= init_gate_scale
                        if layer.bias is not None:
                            layer.bias.data.fill_(0.0)
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}. Must be 'simple', 'sigmoid', or 'mlp'")

        # Optional time-dependent gate scaling (diffusion-aware residual strength).
        if self.use_time_gate:
            gate_hidden_dim = max(32, hidden_dim // 4)
            self.time_gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim, gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(gate_hidden_dim, 1),
                nn.Sigmoid(),
            )
            with torch.no_grad():
                for layer in self.time_gate_mlp:
                    if isinstance(layer, nn.Linear):
                        layer.weight.data *= init_gate_scale
                        if layer.bias is not None:
                            layer.bias.data.fill_(0.0)
        else:
            self.time_gate_mlp = None
    
    def forward(
        self,
        gnn_output: torch.Tensor,
        transformer_output: torch.Tensor,
        time_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Combine GNN and transformer outputs using learned gating.
        
        Args:
            gnn_output: [N, hidden_dim] GNN block output
            transformer_output: [N, hidden_dim] Transformer (global module) output
            
        Returns:
            [N, hidden_dim] Combined output
        """
        time_scale = 1.0
        if self.time_gate_mlp is not None and time_embed is not None:
            if time_embed.shape[0] != gnn_output.shape[0]:
                # Fallback for per-graph embeddings: broadcast if singleton.
                if time_embed.shape[0] == 1:
                    time_embed = time_embed.expand(gnn_output.shape[0], -1)
                else:
                    # If shape mismatch is unexpected, ignore time gate to avoid crashing.
                    time_embed = None
            if time_embed is not None:
                time_scale = self.time_gate_mlp(time_embed)  # [N, 1]

        if self.gate_type == 'simple':
            # Simple scalar gate
            gate = self.gate_weight
            output = gnn_output + (gate * time_scale) * transformer_output
            
        elif self.gate_type == 'sigmoid':
            # Per-feature sigmoid gate
            gate = torch.sigmoid(self.gate_bias) * self.gate_weight
            output = gnn_output + (gate * time_scale) * transformer_output
            
        elif self.gate_type == 'mlp':
            # MLP-based adaptive gate
            # Concatenate GNN and transformer outputs
            combined = torch.cat([gnn_output, transformer_output], dim=-1)  # [N, 2*hidden_dim]
            gate = self.gate_mlp(combined)  # [N, hidden_dim], values in [0, 1]
            output = gnn_output + (gate * time_scale) * transformer_output
            
        return output
    
    def get_gate_statistics(self) -> dict:
        """Get statistics about gate values for monitoring."""
        if self.gate_type == 'simple':
            return {
                'gate_mean': self.gate_weight.item(),
                'gate_std': 0.0,
            }
        elif self.gate_type == 'sigmoid':
            gate_vals = torch.sigmoid(self.gate_bias) * self.gate_weight
            return {
                'gate_mean': gate_vals.mean().item(),
                'gate_std': gate_vals.std().item(),
                'gate_min': gate_vals.min().item(),
                'gate_max': gate_vals.max().item(),
            }
        else:  # mlp
            # Return None for MLP (would need forward pass to compute)
            return {'gate_type': 'mlp', 'note': 'requires forward pass to compute'}
