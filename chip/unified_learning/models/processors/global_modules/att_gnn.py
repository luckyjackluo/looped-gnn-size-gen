"""AttGNN (Attention-based GNN) implementation for chipgen.

This module ports the AttGNN architecture from chipdiffusion to chipgen,
adapting it to work with chipgen's code style and data formats.
"""

from typing import Optional, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgn
import numpy as np
from torch_geometric.data import Data


# ============================================================================
# Helper Components
# ============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron with optional skip connection and layer norm."""
    
    def __init__(self, num_layers: int, model_width: int, in_size: int, out_size: int, 
                 skip: bool = False, layernorm: bool = False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.model_width = model_width
        self.in_size = in_size
        self.out_size = out_size
        self.skip = skip
        self.layernorm = layernorm
        
        layers = []
        for i in range(num_layers):
            inputs = in_size if i == 0 else model_width
            outputs = out_size if i == (num_layers - 1) else model_width
            layers.append(nn.Linear(inputs, outputs))
            if i < (num_layers - 1):
                layers.append(nn.ReLU())
        
        self._nn = nn.Sequential(*layers)
        if layernorm:
            self._ln = nn.LayerNorm(in_size)
        else:
            self._ln = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.layernorm:
            in_shape = x.shape
            x_flat = x.view(-1, in_shape[-1])
            x_norm = self._ln(x_flat)
            x_norm = x_norm.view(*in_shape)
            output = self._nn(x_norm)
        else:
            output = self._nn(x)
        
        return x + output if self.skip else output


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer for conditioning."""
    
    def __init__(self, cond_dim: int, input_dim: int, channel_axis: int = -1):
        """
        Args:
            cond_dim: Dimension of conditioning vector
            input_dim: Dimension of input features
            channel_axis: Axis along which to apply modulation (excluding batch dim)
        """
        super().__init__()
        self._mult_proj = nn.Linear(cond_dim, input_dim)
        self._add_proj = nn.Linear(cond_dim, input_dim)
        self.channel_axis = channel_axis
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, ..., input_dim) input features
            cond: (B, cond_dim) conditioning vector
            
        Returns:
            Modulated features: mult * x + add
        """
        mult = self._mult_proj(cond)
        add = self._add_proj(cond)
        
        # Reshape for broadcasting
        if len(x.shape) > 2:
            B = mult.shape[0]
            expand_dims = [1] * (len(x.shape) - 1)
            # Adjust channel_axis to account for batch dimension
            if self.channel_axis < 0:
                channel_idx = len(x.shape) + self.channel_axis
            else:
                channel_idx = self.channel_axis + 1  # +1 for batch dim
            expand_dims[channel_idx - 1] = mult.shape[1]  # -1 because batch is already accounted
            mult = mult.view(B, *expand_dims)
            add = add.view(B, *expand_dims)
        
        return mult * x + add


class BatchWrapper(nn.Module):
    """Wrapper to handle batched operations for PyG layers that don't support batching."""
    
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Args:
            x: (B, V, F) batched node features
            edge_index: (2, E) edge index for single graph
            edge_attr: (E, F_edge) optional edge attributes
            **kwargs: Additional arguments
            
        Returns:
            (B, V, F_out) batched output features
        """
        B, V, F = x.shape
        _, E = edge_index.shape
        
        # Unbatch: reshape to (B*V, F)
        x_unbatched = x.reshape(B * V, F)
        
        # Handle empty edges
        if E == 0:
            edge_index_unbatched = torch.zeros((2, 0), dtype=edge_index.dtype, device=edge_index.device)
            if edge_attr is not None:
                edge_attr_feat_dim = edge_attr.shape[-1] if len(edge_attr.shape) > 1 else 1
                edge_attr_unbatched = torch.zeros((0, edge_attr_feat_dim), dtype=edge_attr.dtype, device=edge_attr.device)
            else:
                edge_attr_unbatched = None
        else:
            # Replicate edges for each batch
            edge_index_unbatched = edge_index.unsqueeze(0).expand(B, 2, E)  # (B, 2, E)
            edge_index_offset = torch.arange(0, V * B, V, device=edge_index.device, dtype=edge_index.dtype).view(B, 1, 1)
            edge_index_unbatched = edge_index_unbatched + edge_index_offset
            edge_index_unbatched = edge_index_unbatched.reshape(2, B * E)
            
            if edge_attr is not None:
                if edge_attr.shape[0] != E:
                    raise ValueError(f"edge_attr shape mismatch: expected {E} edges, got {edge_attr.shape[0]}")
                feat_dim = edge_attr.shape[-1] if len(edge_attr.shape) > 1 else 1
                edge_attr_unbatched = edge_attr.unsqueeze(0).expand(B, E, feat_dim).reshape(B * E, feat_dim)
            else:
                edge_attr_unbatched = None
        
        # Apply layer
        if edge_attr_unbatched is not None:
            output_unbatched = self.net(x_unbatched, edge_index_unbatched, edge_attr=edge_attr_unbatched, **kwargs)
        else:
            output_unbatched = self.net(x_unbatched, edge_index_unbatched, **kwargs)
        
        # Get output feature dimension
        if len(output_unbatched.shape) == 1:
            output_feat_dim = output_unbatched.numel() // (B * V) if (B * V > 0) else F
        else:
            output_feat_dim = output_unbatched.shape[-1]
        
        # Reshape back to batched format
        if output_unbatched.numel() == 0:
            output = torch.zeros((B, V, output_feat_dim), dtype=output_unbatched.dtype, device=output_unbatched.device)
        else:
            output = output_unbatched.reshape(B, V, output_feat_dim)
        
        return output


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, num_heads: int, key_dim: int, value_dim: int, 
                 in_dim: int, out_dim: int, mask: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.in_dim = in_dim
        self.mask = mask  # Causal mask flag (deprecated)
        
        self._k_linear = nn.Linear(in_dim, num_heads * key_dim, bias=False)
        self._q_linear = nn.Linear(in_dim, num_heads * key_dim, bias=False)
        self._v_linear = nn.Linear(in_dim, num_heads * value_dim, bias=False)
        self._out_linear = nn.Linear(num_heads * value_dim, out_dim, bias=False)
        self._tril = None
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) input tensor
            attn_mask: Optional (B, T, T) or (B, num_heads, T, T) boolean mask
                      where True = keep, False = mask out (set to -inf)
                      
        Returns:
            (B, T, out_dim) output tensor
        """
        B, T, _ = x.shape
        
        # Prepare K, Q, V
        k = self._k_linear(x).view(B, T, self.num_heads, self.key_dim)  # (B, T, n_h, d_k)
        q = self._q_linear(x).view(B, T, self.num_heads, self.key_dim)  # (B, T, n_h, d_k)
        v = self._v_linear(x).view(B, T, self.num_heads, self.value_dim)  # (B, T, n_h, d_v)
        
        # Reshape for matrix multiplication
        k = torch.movedim(k, 1, -1).reshape(B * self.num_heads, self.key_dim, T)
        q = torch.movedim(q, 1, -2).reshape(B * self.num_heads, T, self.key_dim)
        
        # Compute attention logits
        attn_logits = (torch.matmul(q, k) / np.sqrt(self.key_dim)).view(B, self.num_heads, T, T)

        # CRITICAL FIX: Clip attention logits to prevent overflow in softmax
        # Without clipping, large logits (>88) cause exp() overflow → Inf → NaN in softmax
        # This can happen after many training epochs when weights grow large
        # Safe range: [-50, 50] allows full dynamic range while preventing numerical issues
        attn_logits = torch.clamp(attn_logits, min=-50.0, max=50.0)

        # Apply causal mask (backward compatibility)
        if self.mask:
            if self._tril is None or self._tril.shape != (T, T):
                self._tril = torch.tril(torch.ones(T, T, device=attn_logits.device), diagonal=0) == 0
            attn_logits = attn_logits.masked_fill(self._tril, float('-inf'))
        
        # Apply provided attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 3:  # (B, T, T)
                attn_mask = attn_mask.unsqueeze(1).expand(B, self.num_heads, T, T)
            # Convert boolean mask to float mask
            attn_mask_float = (~attn_mask).float() * float('-inf')
            attn_logits = attn_logits + attn_mask_float
        
        # Compute attention
        attn_maps = F.softmax(attn_logits, dim=-1)  # (B, n_h, T_q, T_k)
        v = torch.movedim(v, 1, 2)  # (B, n_h, T, d_v)
        attn_output = torch.matmul(attn_maps, v)  # (B, n_h, T_q, d_v)
        attn_output = torch.movedim(attn_output, -2, 1).reshape(B, T, self.num_heads * self.value_dim)
        
        out = self._out_linear(attn_output)
        return out


class AttentionBlock(nn.Module):
    """Transformer attention block with feedforward network."""
    
    def __init__(self, num_heads: int, model_dim: int, ff_num_layers: int, 
                 ff_size_factor: int, dropout: float, att_implementation: str = "default"):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self._attn_dropout = nn.Dropout(p=dropout)
        self._ff_dropout = nn.Dropout(p=dropout)
        
        if att_implementation == "performer":
            try:
                from performer_pytorch import SelfAttention
                self._attn = SelfAttention(dim=model_dim, heads=num_heads, causal=False, dim_head=None)
            except ImportError:
                raise ImportError("performer_pytorch not available. Install with: pip install performer-pytorch")
        elif att_implementation == "default":
            self._attn = MultiHeadAttention(num_heads, model_dim // num_heads, model_dim // num_heads, 
                                           model_dim, model_dim)
        elif att_implementation == "flash":
            # Use PyTorch's scaled_dot_product_attention
            self._attn = MultiHeadFlashAttention(num_heads, model_dim // num_heads, model_dim // num_heads,
                                                 model_dim, model_dim)
        else:
            raise ValueError(f"Unknown att_implementation: {att_implementation}")
        
        # Feedforward network
        ff_layers = []
        for i in range(ff_num_layers):
            in_dim = model_dim if i == 0 else ff_size_factor * model_dim
            out_dim = model_dim if i == (ff_num_layers - 1) else ff_size_factor * model_dim
            ff_layers.append(nn.Linear(in_dim, out_dim))
            if i < (ff_num_layers - 1):
                ff_layers.append(nn.ReLU())
        self._ff = nn.Sequential(*ff_layers)
        
        self._ln1 = nn.LayerNorm(model_dim)
        self._ln2 = nn.LayerNorm(model_dim)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) input tensor
            attn_mask: Optional attention mask
            
        Returns:
            (B, T, C) output tensor
        """
        x = x + self._attn_dropout(self._attn(self._ln1(x), attn_mask=attn_mask))
        x = x + self._ff_dropout(self._ff(self._ln2(x)))
        return x


class MultiHeadFlashAttention(nn.Module):
    """Multi-head attention using PyTorch's flash attention."""
    
    def __init__(self, num_heads: int, key_dim: int, value_dim: int, 
                 in_dim: int, out_dim: int, mask: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.in_dim = in_dim
        self.mask = mask
        
        self._k_linear = nn.Linear(in_dim, num_heads * key_dim, bias=False)
        self._q_linear = nn.Linear(in_dim, num_heads * key_dim, bias=False)
        self._v_linear = nn.Linear(in_dim, num_heads * value_dim, bias=False)
        self._out_linear = nn.Linear(num_heads * value_dim, out_dim, bias=False)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass using scaled_dot_product_attention."""
        B, T, _ = x.shape
        
        # Prepare K, Q, V
        k = self._k_linear(x).view(B, T, self.num_heads, self.key_dim)  # (B, T, n_h, d_k)
        q = self._q_linear(x).view(B, T, self.num_heads, self.key_dim)  # (B, T, n_h, d_k)
        v = self._v_linear(x).view(B, T, self.num_heads, self.value_dim)  # (B, T, n_h, d_v)
        
        k = torch.movedim(k, 1, -2)  # (B, n_h, T, d_k)
        q = torch.movedim(q, 1, -2)  # (B, n_h, T, d_k)
        v = torch.movedim(v, 1, -2)  # (B, n_h, T, d_v)
        
        # Prepare attention mask
        sdp_attn_mask = None
        if attn_mask is not None:
            if attn_mask.dim() == 3:  # (B, T, T)
                sdp_attn_mask = attn_mask.unsqueeze(1)  # (B, 1, T, T)
            elif attn_mask.dim() == 4:  # (B, num_heads, T, T)
                sdp_attn_mask = attn_mask
            else:
                raise ValueError(f"attn_mask must be 3D or 4D, got {attn_mask.dim()}D")
        
        is_causal = self.mask if attn_mask is None else False
        
        # OPTIMIZATION: Use flash attention with proper kernel selection
        # Flash attention is fastest but has constraints (no custom masks, specific shapes)
        # Memory-efficient is good fallback, math is slowest but most compatible
        # For batched processing with masks, we prioritize flash > mem_efficient > math
        try:
            # Try flash attention first (fastest, but may not support all mask types)
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
                attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=sdp_attn_mask, is_causal=is_causal)
        except RuntimeError:
            # Fallback to memory-efficient attention (good performance, supports masks)
            try:
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False):
                    attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=sdp_attn_mask, is_causal=is_causal)
            except RuntimeError:
                # Final fallback to math kernel (slowest but most compatible)
                attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=sdp_attn_mask, is_causal=is_causal)
        
        attn_output = torch.movedim(attn_output, -2, 1)  # (B, n_h, T, d_v)
        attn_output = attn_output.reshape(B, T, self.num_heads * self.value_dim)
        
        out = self._out_linear(attn_output)
        return out


# ============================================================================
# GNN Components
# ============================================================================

def get_conv_layer(layer_type: str, in_channels: int, out_channels: int, 
                   edge_features: int, **layer_kwargs) -> nn.Module:
    """Get a GNN convolution layer."""
    layer_fns = {
        "gcn": tgn.GCNConv,
        "sage": tgn.SAGEConv,
        "gin": tgn.GINConv,
        "transformer": tgn.TransformerConv,
        "gat": tgn.GATv2Conv,
    }
    
    if layer_type not in layer_fns:
        raise ValueError(f"Unknown layer_type: {layer_type}. Supported: {list(layer_fns.keys())}")
    
    if layer_type == "gin":
        # Create a simple MLP for GIN
        class GINMLP(nn.Module):
            def __init__(self, num_layers, model_width, in_size, out_size):
                super().__init__()
                layers = []
                for i in range(num_layers):
                    inputs = in_size if i == 0 else model_width
                    outputs = out_size if i == (num_layers - 1) else model_width
                    layers.append(nn.Linear(inputs, outputs))
                    if i < (num_layers - 1):
                        layers.append(nn.ReLU())
                self.net = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.net(x)
        
        layer_params = {
            "nn": GINMLP(
                num_layers=layer_kwargs.get("nn_num_layer", 2),
                model_width=layer_kwargs.get("nn_hidden_width", out_channels),
                in_size=in_channels,
                out_size=out_channels,
            ),
            **{k: v for k, v in layer_kwargs.items() if k not in ["nn_num_layer", "nn_hidden_width"]}
        }
    elif layer_type in ["gat", "transformer"]:
        if layer_kwargs.get("concat", True):
            channel_divisor = layer_kwargs.get("heads", 1)
        else:
            channel_divisor = 1
        layer_params = {
            "in_channels": in_channels,
            "out_channels": out_channels // channel_divisor,
            "edge_dim": edge_features,
            "add_self_loops": layer_kwargs.get("add_self_loops", True),
            **{k: v for k, v in layer_kwargs.items() if k not in ["concat", "add_self_loops"]}
        }
    else:
        layer_params = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            **layer_kwargs
        }
    
    layer = layer_fns[layer_type](**layer_params)
    
    # Wrap layers that need batching support
    if layer_type in ["gat", "transformer"]:
        layer = BatchWrapper(layer)
    
    return layer


def accepts_edge_attr(layer: nn.Module) -> bool:
    """Check if layer accepts edge attributes."""
    return isinstance(layer, BatchWrapper)


class GConvLayer(nn.Module):
    """Simple GCN convolution layer."""
    
    def __init__(self, in_node_features: int, out_node_features: int):
        super().__init__()
        self._layer = tgn.GCNConv(in_node_features, out_node_features)
    
    def forward(self, x_in):
        """Forward pass."""
        x, data, _ = x_in
        edge_index = data.edge_index
        return self._layer(x, edge_index)


class LinearEncoderLayer(nn.Module):
    """Linear encoder layer with optional positional encoding."""
    
    def __init__(self, in_node_features: int, out_node_features: int, 
                 input_encoding_dim: int = 0, mask_key: Optional[str] = None, device: str = "cpu"):
        super().__init__()
        MAX_FREQ = 100
        mask_features = 1 if mask_key is not None else 0
        self._layer = nn.Linear(in_node_features + mask_features, out_node_features)
        self._encoding_layer = nn.Linear(in_node_features * input_encoding_dim, out_node_features) if input_encoding_dim > 0 else None
        self.input_encoding_dim = input_encoding_dim
        input_encoding_freqs = torch.exp(
            np.log(MAX_FREQ) * torch.arange(0, self.input_encoding_dim // 2, dtype=torch.float32, device=device) / (self.input_encoding_dim // 2)
        ).view(1, 1, 1, self.input_encoding_dim // 2)
        self.register_buffer('input_encoding_freqs', input_encoding_freqs)
        self.mask_key = mask_key
        # Add LayerNorm to normalize concatenated features (original + normalized) to prevent scale imbalance
        # This normalizes the concatenated vector before the first linear layer
        self._input_norm = nn.LayerNorm(in_node_features + mask_features)
    
    def forward(self, x: torch.Tensor, cond_data: Data, t_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, V, F) batched node features
            cond_data: Data object with node features and optional mask
            t_embed: Optional time embedding (not used here)
            
        Returns:
            (B, V, out_node_features) encoded features
        """
        B, V, F = x.shape
        node_data = cond_data.x
        
        # Handle batched graphs
        if hasattr(cond_data, 'ptr') and hasattr(cond_data, 'num_graphs') and cond_data.num_graphs > 1:
            ptr = cond_data.ptr
            node_data_list = []
            for i in range(B):
                start_idx = int(ptr[i])
                end_idx = int(ptr[i+1])
                graph_node_data = node_data[start_idx:end_idx]
                V_i = graph_node_data.shape[0]
                
                if V_i < V:
                    padding = torch.zeros(V - V_i, graph_node_data.shape[1], 
                                        dtype=graph_node_data.dtype, device=graph_node_data.device)
                    graph_node_data_padded = torch.cat([graph_node_data, padding], dim=0)
                else:
                    graph_node_data_padded = graph_node_data[:V]
                
                node_data_list.append(graph_node_data_padded)
            
            node_data = torch.stack(node_data_list, dim=0)
        else:
            node_data = node_data.view(1, *node_data.shape).expand(B, -1, -1)
        
        spatial_input = torch.cat((x, node_data), dim=-1)
        
        if self.mask_key is not None:
            node_mask = cond_data[self.mask_key]
            if hasattr(cond_data, 'ptr') and hasattr(cond_data, 'num_graphs') and cond_data.num_graphs > 1:
                ptr = cond_data.ptr
                mask_list = []
                for i in range(B):
                    start_idx = int(ptr[i])
                    end_idx = int(ptr[i+1])
                    graph_mask = node_mask[start_idx:end_idx].float()
                    V_i = graph_mask.shape[0]
                    if V_i < V:
                        padding = torch.zeros(V - V_i, dtype=graph_mask.dtype, device=graph_mask.device)
                        graph_mask_padded = torch.cat([graph_mask, padding], dim=0)
                    else:
                        graph_mask_padded = graph_mask[:V]
                    mask_list.append(graph_mask_padded)
                node_mask = torch.stack(mask_list, dim=0).unsqueeze(-1)
            else:
                node_mask = node_mask.float().view(1, *node_mask.shape, 1).expand(B, -1, 1)
            proj_input = torch.cat((spatial_input, node_mask), dim=-1)
        else:
            proj_input = spatial_input
        
        # Normalize concatenated features to prevent scale imbalance between original and normalized values
        # This ensures gradients and learning rates are balanced across feature channels
        proj_input = self._input_norm(proj_input)
        
        output = self._layer(proj_input)
        
        if self._encoding_layer is not None:
            input_encodings = self.get_input_encoding(spatial_input)
            input_encodings_proj = self._encoding_layer(input_encodings)
            output = output + input_encodings_proj
        
        return output
    
    def get_input_encoding(self, spatial_input: torch.Tensor) -> torch.Tensor:
        """Get positional encoding for spatial input."""
        B, V, D = spatial_input.shape
        theta = spatial_input.unsqueeze(dim=-1) * self.input_encoding_freqs
        embedding = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        embedding = embedding.view(B, V, D * self.input_encoding_dim)
        return embedding


class LinearDecoderLayer(nn.Module):
    """Simple linear decoder layer."""
    
    def __init__(self, in_node_features: int, out_node_features: int):
        super().__init__()
        self._layer = nn.Linear(in_node_features, out_node_features)
    
    def forward(self, x: torch.Tensor, cond_data: Data, t_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        return self._layer(x)


def _hash_coords(cell_coords: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Stable integer hash for 2D grid coordinates with batch offset."""
    prime1 = 73856093
    prime2 = 19349663
    prime3 = 83492791
    return cell_coords[..., 0] * prime1 + cell_coords[..., 1] * prime2 + batch * prime3


class HashGridTokenizer(nn.Module):
    """Multi-level hash grid tokenization with mean pooling."""
    
    def __init__(self, in_dim: int, grid_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, grid_dim)
        self.gate = nn.Linear(in_dim, grid_dim)
        # Per-bin statistics encoder:
        # [num_instances, total_area, avg_area, density]
        self.stats_mlp = nn.Sequential(
            nn.Linear(4, grid_dim),
            nn.SiLU(),
            nn.Linear(grid_dim, grid_dim),
        )
        self.stats_scale = 0.1
    
    def forward(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        batch: torch.Tensor,
        cell_size: float,
        node_areas: Optional[torch.Tensor] = None,
        coord_min: float = -1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            node_feats: (B, V, F) node embeddings
            coords: (B, V, 2) node coordinates
            batch: (B, V) batch ids
            cell_size: grid cell size
        Returns:
            grid_feats: (G, D) pooled grid embeddings
            grid_coords: (G, 2) real-valued grid coordinates (cell centers)
            grid_cell_coords: (G, 2) integer grid coordinates (cell indices)
            grid_batch: (G,) batch ids for grids
            inverse: (B*V,) mapping from node -> grid index
        """
        # Support both batched dense inputs:
        #   node_feats: (B, V, F), coords: (B, V, 2), batch: (B, V)
        # and flat inputs:
        #   node_feats: (N, F), coords: (N, 2), batch: (N,)
        if node_feats.dim() == 3:
            coords_flat = coords.reshape(-1, 2)
            batch_flat = batch.reshape(-1)
            node_flat = node_feats.reshape(-1, node_feats.shape[-1])
        elif node_feats.dim() == 2:
            coords_flat = coords
            batch_flat = batch
            node_flat = node_feats
        else:
            raise ValueError(f"Unsupported node_feats shape: {tuple(node_feats.shape)}")
        
        cell_coords = torch.floor((coords_flat - coord_min) / cell_size).long()
        hash_ids = _hash_coords(cell_coords, batch_flat)
        
        # Stable unique to recover one coordinate per cell
        sorted_hash, sorted_idx = torch.sort(hash_ids)
        if sorted_hash.numel() == 0:
            empty = torch.zeros(0, self.proj.out_features, device=node_feats.device, dtype=node_feats.dtype)
            empty_coords = torch.zeros(0, 2, device=node_feats.device, dtype=coords_flat.dtype)
            empty_cell_coords = torch.zeros(0, 2, device=node_feats.device, dtype=torch.long)
            empty_batch = torch.zeros(0, device=node_feats.device, dtype=torch.long)
            return empty, empty_coords, empty_cell_coords, empty_batch, torch.zeros_like(hash_ids)
        
        # Get unique consecutive values and inverse mapping
        # Note: torch.unique_consecutive doesn't have return_indices, so we compute first indices manually
        unique_hashes, inverse_sorted = torch.unique_consecutive(sorted_hash, return_inverse=True)
        
        # Find first occurrence index of each unique value in sorted array
        # First index is where inverse_sorted changes value (or index 0 for the first unique)
        num_unique = unique_hashes.numel()
        first_idx = torch.zeros(num_unique, dtype=torch.long, device=sorted_hash.device)
        first_idx[0] = 0
        if num_unique > 1:
            # Find where inverse changes (these are the first occurrences of new values)
            changes = (inverse_sorted[1:] != inverse_sorted[:-1]).nonzero(as_tuple=True)[0] + 1
            first_idx[1:] = changes
        
        orig_idx = sorted_idx[first_idx]
        grid_cell_coords = cell_coords[orig_idx]
        grid_coords = (grid_cell_coords.to(coords_flat.dtype) + 0.5) * cell_size + coord_min
        grid_batch = batch_flat[orig_idx]
        
        # Pool features with gating
        gated = self.proj(node_flat) * torch.sigmoid(self.gate(node_flat))
        num_grids = unique_hashes.numel()
        grid_feats = torch.zeros(num_grids, gated.shape[-1], device=gated.device, dtype=gated.dtype)
        counts = torch.zeros(num_grids, device=gated.device, dtype=gated.dtype)
        
        inverse = torch.zeros_like(hash_ids)
        # inverse: position of each hash in unique_hashes.
        # sorted_hash is already grouped by unique_hashes, so inverse_sorted is exact and cheaper
        # than an extra bucketize() call.
        inverse[sorted_idx] = inverse_sorted
        
        grid_feats.index_add_(0, inverse, gated)
        counts.index_add_(0, inverse, torch.ones_like(hash_ids, dtype=gated.dtype))
        grid_feats = grid_feats / counts.clamp(min=1.0).unsqueeze(-1)

        # Compute vectorized on-the-fly bin statistics and fuse them into bin features.
        if node_areas is not None:
            node_areas_flat = node_areas.reshape(-1).to(gated.dtype)
        else:
            node_areas_flat = torch.ones(coords_flat.shape[0], device=gated.device, dtype=gated.dtype)

        # 1) Number of instances in bin (hard assignment)
        count_per_bin = counts

        # 2) Total area in bin (hard assignment)
        total_area_per_bin = torch.zeros(num_grids, device=gated.device, dtype=gated.dtype)
        total_area_per_bin.index_add_(0, inverse, node_areas_flat)

        # 3) Average area in bin
        avg_area_per_bin = total_area_per_bin / count_per_bin.clamp(min=1.0)

        # 4) DreamPlace/ePlace-style density via bilinear splat:
        #    rho_b = (1 / A_b) * sum_i a_i * psi_b(x_i, y_i)
        # where A_b = cell_size^2, and psi_b is a separable tent kernel.
        coords_cell = (coords_flat - coord_min) / cell_size
        base_x = torch.floor(coords_cell[:, 0]).long()
        base_y = torch.floor(coords_cell[:, 1]).long()
        frac_x = (coords_cell[:, 0] - base_x.to(coords_cell.dtype)).clamp(min=0.0, max=1.0)
        frac_y = (coords_cell[:, 1] - base_y.to(coords_cell.dtype)).clamp(min=0.0, max=1.0)

        # 4 neighbors: (0,0), (1,0), (0,1), (1,1)
        nb_cell = torch.stack([
            torch.stack([base_x, base_y], dim=-1),
            torch.stack([base_x + 1, base_y], dim=-1),
            torch.stack([base_x, base_y + 1], dim=-1),
            torch.stack([base_x + 1, base_y + 1], dim=-1),
        ], dim=1)  # (N, 4, 2)

        w00 = (1.0 - frac_x) * (1.0 - frac_y)
        w10 = frac_x * (1.0 - frac_y)
        w01 = (1.0 - frac_x) * frac_y
        w11 = frac_x * frac_y
        bilinear_w = torch.stack([w00, w10, w01, w11], dim=1)  # (N, 4)

        nb_batch = batch_flat.unsqueeze(1).expand(-1, 4)
        nb_hash = _hash_coords(nb_cell.reshape(-1, 2), nb_batch.reshape(-1))
        nb_weighted_area = (node_areas_flat.unsqueeze(1) * bilinear_w).reshape(-1)

        # Map neighbor hashes to existing bins (sparse hash map via bucketize).
        bucket = torch.bucketize(nb_hash, unique_hashes)
        valid = bucket < unique_hashes.numel()
        bucket_clamped = bucket.clamp(max=max(unique_hashes.numel() - 1, 0))
        gathered_hash = unique_hashes.take(bucket_clamped)
        valid = valid & (gathered_hash == nb_hash)

        density_mass = torch.zeros(num_grids, device=gated.device, dtype=gated.dtype)
        if valid.any():
            density_mass.index_add_(0, bucket_clamped[valid], nb_weighted_area[valid])
        bin_area = max(cell_size * cell_size, 1e-8)
        density_per_bin = density_mass / bin_area

        stats = torch.stack(
            [count_per_bin, total_area_per_bin, avg_area_per_bin, density_per_bin],
            dim=-1,
        )
        # Log-compress dynamic range for numerical stability.
        stats_enc = self.stats_mlp(torch.log1p(torch.clamp(stats, min=0.0)))
        grid_feats = grid_feats + self.stats_scale * stats_enc
        
        return grid_feats, grid_coords, grid_cell_coords, grid_batch, inverse


class GridWindowAttention(nn.Module):
    """Windowed/shifted attention over grid tokens.
    
    Supports multiple stacked attention layers per scale for curriculum learning.
    Special window_size values:
      -1: force single-window full attention over all grid tokens at this level
       0: disable attention at this level (identity)
    """
    
    def __init__(self, dim: int, num_heads: int, ff_layers: int, ff_factor: int,
                 dropout: float, window_size: int, shifted: bool, att_implementation: str,
                 num_attention_layers: int = 1):
        super().__init__()
        self.window_size = window_size
        self.shifted = shifted
        self.num_attention_layers = num_attention_layers
        
        # Support multiple stacked attention layers for curriculum learning
        # Each layer is an independent AttentionBlock
        self.att_blocks = nn.ModuleList([
            AttentionBlock(num_heads, dim, ff_layers, ff_factor, dropout, att_implementation)
            for _ in range(num_attention_layers)
        ])
    
    def _run_windows(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        batch: torch.Tensor,
        cell_size: Optional[float] = None,
    ) -> torch.Tensor:
        if feats.numel() == 0:
            return feats
        
        # Check if we have fewer cells than window_size - if so, use one window
        num_cells = feats.shape[0]
        if num_cells <= self.window_size * self.window_size:
            # Fewer cells than window_size^2, treat everything as one window
            tokens = feats.unsqueeze(0)  # (1, T, D)
            # Apply all attention layers sequentially
            for att_block in self.att_blocks:
                tokens = att_block(tokens)
            return tokens.squeeze(0)
        
        if cell_size is None:
            win_coords = torch.div(coords, self.window_size, rounding_mode='floor')
        else:
            win_coords = torch.div(coords, self.window_size * cell_size, rounding_mode='floor')
        win_coords = win_coords.to(torch.long)
        win_hash = _hash_coords(win_coords, batch)
        unique_win, inverse = torch.unique(win_hash, return_inverse=True)

        # CUDA-friendly grouping:
        # Sort by window id once, then process contiguous segments.
        order = torch.argsort(inverse)
        inv_sorted = inverse[order]
        feats_sorted = feats[order]
        counts = torch.bincount(inv_sorted, minlength=unique_win.numel())

        out_sorted = torch.empty_like(feats_sorted)
        start = 0
        for c in counts.tolist():
            if c == 0:
                continue
            end = start + c
            tokens = feats_sorted[start:end].unsqueeze(0)  # (1, T, D)
            for att_block in self.att_blocks:
                tokens = att_block(tokens)
            out_sorted[start:end] = tokens.squeeze(0)
            start = end

        # Unsort back to original token order.
        out = torch.empty_like(feats)
        out[order] = out_sorted
        return out
    
    def forward(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        batch: torch.Tensor,
        cell_size: Optional[float] = None,
    ) -> torch.Tensor:
        # Special mode: -1 means single-window full attention (no window partitioning).
        if self.window_size == -1:
            if feats.numel() == 0:
                return feats
            tokens = feats.unsqueeze(0)  # (1, T, D)
            for att_block in self.att_blocks:
                tokens = att_block(tokens)
            return tokens.squeeze(0)
        # window_size == 0 disables this level's attention.
        if self.window_size == 0:
            return feats
        base = self._run_windows(feats, coords, batch, cell_size=cell_size)
        if not self.shifted:
            return base
        shift = self.window_size // 2
        if cell_size is not None:
            shift = shift * cell_size
        shifted_coords = coords + coords.new_tensor(shift)
        shifted = self._run_windows(feats, shifted_coords, batch, cell_size=cell_size)
        return 0.5 * (base + shifted)


class GridBroadcast(nn.Module):
    """Broadcast grid context back to points with neighbor pooling."""
    
    def __init__(self, grid_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(grid_dim, out_dim)
        # Gate combines node_feats (out_dim) and context (out_dim) -> 2*out_dim
        self.gate = nn.Linear(out_dim + out_dim, out_dim)
        # Precompute Moore neighborhood (including self)
        offsets = torch.tensor([[-1, -1], [-1, 0], [-1, 1],
                                [0, -1], [0, 0], [0, 1],
                                [1, -1], [1, 0], [1, 1]])
        self.register_buffer("neighbor_offsets", offsets, persistent=False)
    
    def forward(
        self,
        node_feats: torch.Tensor,
        node_coords: torch.Tensor,
        node_batch: torch.Tensor,
        grid_feats: torch.Tensor,
        grid_cell_coords: torch.Tensor,
        grid_batch: torch.Tensor,
        grid_hashes: torch.Tensor,
        cell_size: float,
        coord_min: float = -1.0,
    ) -> torch.Tensor:
        if grid_feats.numel() == 0:
            # Return zeros but maintain gradient connection to node_feats for downstream trainable layers
            zeros = torch.zeros_like(node_feats)
            if node_feats.requires_grad:
                zeros = zeros + node_feats * 0.0
            return zeros
        
        N = node_coords.shape[0]
        device = node_coords.device
        neighbor_offsets = self.neighbor_offsets.to(device)
        
        node_cells = torch.floor((node_coords - coord_min) / cell_size).long()  # (N, 2)
        neighbors = node_cells.unsqueeze(1) + neighbor_offsets.view(1, -1, 2)
        neighbor_batch = node_batch.unsqueeze(1).expand(-1, neighbor_offsets.shape[0])
        neighbor_hashes = _hash_coords(neighbors.reshape(-1, 2), neighbor_batch.reshape(-1))
        neighbor_hashes = neighbor_hashes.view(N, -1)
        
        sorted_hashes, perm = torch.sort(grid_hashes)
        bucket = torch.bucketize(neighbor_hashes, sorted_hashes)
        # torch.gather requires index to have same dims as input; use take() on flattened indices.
        bucket_clamped = bucket.clamp(max=sorted_hashes.numel() - 1)
        gathered = sorted_hashes.take(bucket_clamped.reshape(-1)).reshape(bucket_clamped.shape)
        valid = (bucket < sorted_hashes.numel()) & (gathered == neighbor_hashes)
        bucket = bucket_clamped
        grid_indices = perm.take(bucket.reshape(-1)).reshape(bucket.shape)
        grid_indices = torch.where(valid, grid_indices, torch.full_like(grid_indices, -1))
        
        # Build neighbor_feats using scatter to maintain gradients
        flat_indices = grid_indices.view(-1)
        valid_flat = flat_indices >= 0
        
        # Initialize neighbor features.
        neighbor_feats = torch.zeros(
            N, neighbor_offsets.shape[0], grid_feats.shape[-1], device=device, dtype=grid_feats.dtype
        )
        
        if valid_flat.any():
            valid_indices_flat = torch.where(valid_flat)[0]
            valid_grid_indices = flat_indices[valid_flat]
            valid_grid_feats = grid_feats[valid_grid_indices]
            
            neighbor_idx_2d = valid_indices_flat // neighbor_offsets.shape[0]
            neighbor_idx_1d = valid_indices_flat % neighbor_offsets.shape[0]
            neighbor_feats_flat = neighbor_feats.view(-1, grid_feats.shape[-1])
            flat_positions = neighbor_idx_2d * neighbor_offsets.shape[0] + neighbor_idx_1d
            # Direct indexed assignment is faster here (unique positions by construction).
            neighbor_feats_flat[flat_positions] = valid_grid_feats
            neighbor_feats = neighbor_feats_flat.view(N, neighbor_offsets.shape[0], grid_feats.shape[-1])
        
        counts = valid.sum(dim=-1, keepdim=True).clamp(min=1)
        pooled = neighbor_feats.sum(dim=1) / counts
        
        context = self.proj(pooled)
        gate = torch.sigmoid(self.gate(torch.cat([node_feats, context], dim=-1)))
        return gate * context


class HashedGridHierarchy(nn.Module):
    """Multi-level hash-grid hierarchy with windowed attention and broadcast.
    
    Supports density-controlled cell size (adaptive per design):
    - Level 0 (finest): Computed from target occupancy (default: 16 nodes per cell)
    - Levels 1+: Computed to ensure window attention can cover the whole graph
    """
    
    def __init__(
        self,
        hidden_dim: int,
        levels: int = 2,
        base_cell_size: Optional[float] = None,  # If None, use density-controlled
        cell_size_multiplier: Optional[float] = None,  # If None, compute from levels
        grid_dim: int = 64,
        window_sizes: Optional[List[int]] = None,
        shifted: bool = True,
        ff_num_layers: int = 1,
        ff_size_factor: int = 2,
        dropout: float = 0.0,
        att_implementation: str = "default",
        rehash_interval: int = 1,
        target_occupancy: float = 4.0,  # Target nodes per cell for finest level
        target_graph_size: int = 1000,  # Target graph size for coarser level computation
        num_attention_layers: int = 1,  # Number of attention layers per scale (for curriculum learning)
        coord_min: float = -1.0,
        coord_max: float = 1.0,
        clamp_coords_to_canvas: bool = True,
        use_fixed_canvas_range: bool = True,
        use_time_level_mixing: bool = True,
        level_mixing_power: float = 1.0,
        min_level_weight: float = 0.05,
    ):
        super().__init__()
        self.levels = levels
        self.target_occupancy = target_occupancy
        self.target_graph_size = target_graph_size
        self.rehash_interval = max(1, rehash_interval)
        self.num_attention_layers = num_attention_layers
        self.coord_min = coord_min
        self.coord_max = coord_max
        self.clamp_coords_to_canvas = clamp_coords_to_canvas
        self.use_fixed_canvas_range = use_fixed_canvas_range
        self.use_time_level_mixing = use_time_level_mixing
        self.level_mixing_power = level_mixing_power
        self.min_level_weight = min_level_weight
        
        # If base_cell_size is provided, use fixed cell sizes (legacy mode)
        # Otherwise, use density-controlled mode (computed dynamically in forward)
        self.use_density_control = (base_cell_size is None)
        if self.use_density_control:
            # Store None - will be computed per graph in forward
            self.base_cell_size = None
            self.cell_size_multiplier = None
        else:
            # Legacy fixed mode
            self.base_cell_size = base_cell_size
            self.cell_size_multiplier = cell_size_multiplier if cell_size_multiplier is not None else 2.0
        
        window_sizes = window_sizes or [9, 9]
        if len(window_sizes) < levels:
            # Pad with last value
            window_sizes = window_sizes + [window_sizes[-1]] * (levels - len(window_sizes))
        
        self.tokenizers = nn.ModuleList()
        self.window_attn = nn.ModuleList()
        self.broadcasts = nn.ModuleList()
        self.grid_dim = grid_dim
        for i in range(levels):
            self.tokenizers.append(HashGridTokenizer(hidden_dim, grid_dim))
            self.window_attn.append(
                GridWindowAttention(
                    dim=grid_dim,
                    num_heads=max(1, grid_dim // 16),
                    ff_layers=ff_num_layers,
                    ff_factor=ff_size_factor,
                    dropout=dropout,
                    window_size=window_sizes[i],
                    shifted=shifted,
                    att_implementation=att_implementation,
                    num_attention_layers=num_attention_layers,
                )
            )
            self.broadcasts.append(GridBroadcast(grid_dim, hidden_dim))

    def _compute_level_weights(
        self,
        level_indices: List[int],
        t_scalar: Optional[float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Compute time-dependent coarse-to-fine weights.
        level 0 = fine, level (L-1) = coarse.
        Early t -> emphasize coarse levels; late t -> emphasize fine levels.
        """
        n = len(level_indices)
        if n == 0:
            return torch.zeros(0, device=device, dtype=dtype)
        if (not self.use_time_level_mixing) or self.levels <= 1 or t_scalar is None:
            return torch.ones(n, device=device, dtype=dtype) / float(n)

        t = float(max(0.0, min(1.0, t_scalar)))
        # normalized level coordinate in [0,1], 0=fine, 1=coarse
        if self.levels == 1:
            lvl = torch.zeros(n, device=device, dtype=dtype)
        else:
            lvl = torch.tensor(level_indices, device=device, dtype=dtype) / float(self.levels - 1)
        coarse_pref = lvl
        fine_pref = 1.0 - lvl
        w = (1.0 - t) * coarse_pref + t * fine_pref
        w = torch.clamp(w, min=self.min_level_weight)
        if self.level_mixing_power != 1.0:
            w = torch.pow(w, self.level_mixing_power)
        return w / w.sum().clamp(min=1e-8)
    
    def _compute_cell_sizes(self, num_nodes: int, coord_range: Optional[Tuple[float, float]] = None) -> List[float]:
        """
        Compute cell sizes for each level using density-controlled approach.
        
        Level 0 (finest): Density-controlled based on target occupancy
        - N nodes, target ρ nodes per cell → M ≈ N/ρ cells needed
        - For square grid: cells_per_side ≈ √M = √(N/ρ)
        - Cell size: s_fine = coord_width / cells_per_side = coord_width * √(ρ/N)
        
        Levels 1+: Coarser scales based on finest cell size
        - Level 1: cell size = 2 * s_fine (≈ 4x fewer cells)
        - Level 2+: geometric progression from s_level1
        
        Args:
            num_nodes: Number of nodes in the graph
            coord_range: Optional (min, max) tuple for coordinate range. 
                        If None, assumes [-1, 1] (width = 2)
            
        Returns:
            List of cell sizes for each level [s0, s1, s2, ...]
        """
        # Determine coordinate width
        if coord_range is None:
            # Default: assume normalized coordinates [-1, 1]
            coord_width = 2.0
        else:
            coord_min, coord_max = coord_range
            coord_width = coord_max - coord_min
            coord_width = max(coord_width, 1e-8)  # Avoid division by zero
        
        # Level 0: Density-controlled finest level
        # s_fine = coord_width * sqrt(ρ / N) where ρ = target_occupancy
        # For target occupancy ρ, we need M ≈ N/ρ cells
        # For square grid: cells_per_side ≈ √M = √(N/ρ)
        # Cell size: s_fine = coord_width / cells_per_side = coord_width * √(ρ/N)
        s_fine = coord_width * torch.sqrt(torch.tensor(self.target_occupancy / max(num_nodes, 1.0))).item()
        s_fine = max(s_fine, 0.01)  # Minimum cell size to avoid numerical issues
        
        # Levels 1+: Coarser by fixed ratio relative to finest level
        # Level 1: 2x cell size -> ~4x fewer cells
        s_level1 = s_fine * 2.0
        s_level2 = s_level1 * 2.0
        
        # Ensure monotonicity: s_fine < s_level1 < s_level2
        if s_level1 <= s_fine:
            s_level1 = s_fine * 2.0
        if s_level2 <= s_level1:
            s_level2 = s_level1 * 2.0
        
        if self.levels == 1:
            return [s_fine]
        elif self.levels == 2:
            return [s_fine, s_level1]
        elif self.levels == 3:
            return [s_fine, s_level1, s_level2]
        else:
            # For more levels, use geometric progression from level 1 to level 2
            cell_sizes = [s_fine, s_level1]
            ratio = s_level2 / s_level1
            for i in range(2, self.levels):
                cell_sizes.append(s_level1 * (ratio ** (i - 1)))
            return cell_sizes
    
    def forward(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        node_areas: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        step_idx: Optional[int] = None,
        t_continuous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_feats: (B, V, F)
            coords: (B, V, 2)
            batch: (B, V) batch ids (default zeros)
            step_idx: optional refinement step index for rehash scheduling
        Returns:
            context: (B, V, F)
        """
        if step_idx is not None and (step_idx % self.rehash_interval) != 0:
            # Return zeros connected to node_feats to maintain gradient flow
            if node_feats.requires_grad:
                return torch.zeros_like(node_feats) + node_feats * 0.0
            return torch.zeros_like(node_feats)
        
        if node_feats.dim() not in (2, 3):
            raise ValueError(f"node_feats must be 2D or 3D, got shape {tuple(node_feats.shape)}")
        if coords.dim() not in (2, 3):
            raise ValueError(f"coords must be 2D or 3D, got shape {tuple(coords.shape)}")

        is_flat = (node_feats.dim() == 2)

        if batch is None:
            if is_flat:
                batch = torch.zeros(coords.shape[0], device=coords.device, dtype=torch.long)
            else:
                batch = torch.zeros(coords.shape[:2], device=coords.device, dtype=torch.long)
        if self.clamp_coords_to_canvas:
            coords = coords.clamp(min=self.coord_min, max=self.coord_max)
        
        if is_flat:
            batch_flat = batch
            coords_flat = coords
            node_flat = node_feats
            B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            V = None
        else:
            B, V, _ = coords.shape
            batch_flat = batch.reshape(-1)
            coords_flat = coords.reshape(-1, 2)
            node_flat = node_feats.reshape(-1, node_feats.shape[-1])
        
        # Derive scalar diffusion time for coarse->fine level mixing.
        t_scalar = None
        if t_continuous is not None and isinstance(t_continuous, torch.Tensor) and t_continuous.numel() > 0:
            t_scalar = float(t_continuous.mean().detach().item())

        # Accumulate contexts in a list to ensure proper gradient flow
        context_list = []
        context_level_indices = []
        
        # Compute cell sizes (per graph if batched, or globally)
        if self.use_density_control:
            if self.use_fixed_canvas_range:
                coord_range = (self.coord_min, self.coord_max)
            else:
                # Compute coordinate range from actual data
                coord_min = coords_flat.min(dim=0)[0].min().item()
                coord_max = coords_flat.max(dim=0)[0].max().item()
                coord_range = (coord_min, coord_max)
            
            # For batched graphs, we need to compute per graph
            # For now, use the number of nodes in the batch (assuming similar sizes)
            # In the future, we could compute per-graph cell sizes
            if B == 1:
                # Single graph
                num_nodes = int(coords_flat.shape[0]) if is_flat else V
            else:
                # Batched: use average nodes per graph
                # If batch info is available, compute per graph; otherwise use V/B
                if batch is not None:
                    # Count nodes per graph using bincount (GPU-friendly).
                    if is_flat:
                        counts = torch.bincount(batch_flat, minlength=B)
                    else:
                        counts = torch.bincount(batch_flat, minlength=B)
                    num_nodes = int(counts.float().mean().item()) if counts.numel() > 0 else (coords_flat.shape[0] // max(B, 1))
                else:
                    total_nodes = coords_flat.shape[0]
                    num_nodes = total_nodes // B if B > 1 else total_nodes
            
            cell_sizes = self._compute_cell_sizes(num_nodes, coord_range=coord_range)
        else:
            # Legacy fixed mode
            cell_sizes = [self.base_cell_size * (self.cell_size_multiplier ** i) for i in range(self.levels)]
        
        for level_idx in range(self.levels):
            cell_size = cell_sizes[level_idx]
            tokenizer = self.tokenizers[level_idx]
            windower = self.window_attn[level_idx]
            broadcaster = self.broadcasts[level_idx]
            
            grid_feats, grid_coords, grid_cell_coords, grid_batch, inverse = tokenizer(
                node_feats, coords, batch, cell_size, node_areas=node_areas, coord_min=self.coord_min
            )
            if grid_feats.numel() == 0:
                continue
            
            grid_hashes = _hash_coords(grid_cell_coords, grid_batch)
            grid_feats = windower(grid_feats, grid_coords, grid_batch, cell_size=cell_size)
            
            context_flat = broadcaster(
                node_feats=node_flat,
                node_coords=coords_flat,
                node_batch=batch_flat,
                grid_feats=grid_feats,
                grid_cell_coords=grid_cell_coords,
                grid_batch=grid_batch,
                grid_hashes=grid_hashes,
                cell_size=cell_size,
                coord_min=self.coord_min,
            )
            
            if is_flat:
                context_list.append(context_flat)
            else:
                context = context_flat.view(B, V, -1)
                context_list.append(context)
            context_level_indices.append(level_idx)
        
        # Sum all contexts - this ensures gradient flow
        # Even when module is frozen, we need to connect to inputs that require gradients
        # to maintain the computation graph for downstream trainable layers
        if context_list:
            weights = self._compute_level_weights(
                level_indices=context_level_indices,
                t_scalar=t_scalar,
                device=context_list[0].device,
                dtype=context_list[0].dtype,
            )
            agg_context = sum(weights[i] * context_list[i] for i in range(len(context_list)))
            # If aggregated context doesn't require gradients but node_feats does,
            # connect them to maintain gradient flow (important for frozen modules)
            if not agg_context.requires_grad and node_feats.requires_grad:
                agg_context = agg_context + node_feats * 0.0
            return agg_context
        else:
            # Return zeros connected to node_feats if it requires gradients
            if node_feats.requires_grad:
                return torch.zeros_like(node_feats) + node_feats * 0.0
            return torch.zeros_like(node_feats)


class ResGNNBlock(nn.Module):
    """Residual GNN block with multiple GNN layers."""
    
    def __init__(self, in_node_features: int, out_node_features: int, 
                 hidden_node_features: int, cond_node_features: int, 
                 edge_features: int, num_layers: int, encoding_dim: int,
                 residual: bool = True, norm: bool = True, dropout: float = 0.0,
                 conv_params: dict = None, device: str = "cpu", **kwargs):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.edge_features = edge_features
        self.residual = residual
        
        self._gconv_layers = nn.ModuleList()
        self._lnorm_layers = nn.ModuleList()
        self._linear_layers = nn.ModuleList()
        
        # Conditioning dimension (no lap_pe)
        self.cond_node_features = cond_node_features
        
        # Add LayerNorm to normalize concatenated features (x + cond_x) before first GNN layer
        # This prevents scale imbalance when original and normalized features are concatenated
        if cond_node_features > 0:
            self._input_norm = nn.LayerNorm(in_node_features + cond_node_features)
        else:
            self._input_norm = None
        
        self._cond_layer = FiLM(encoding_dim, hidden_node_features, channel_axis=-1) if encoding_dim > 0 else None
        
        conv_params = conv_params or {"layer_type": "gcn"}
        
        for i in range(num_layers):
            in_features = in_node_features + cond_node_features if i == 0 else hidden_node_features
            out_features = hidden_node_features if i < (num_layers - 1) else out_node_features
            
            self._gconv_layers.append(get_conv_layer(
                in_channels=in_features,
                out_channels=hidden_node_features,
                edge_features=edge_features,
                **conv_params
            ))
            self._lnorm_layers.append(nn.LayerNorm(hidden_node_features))
            self._linear_layers.append(nn.Linear(hidden_node_features, out_features))
        
        self.use_edge_attr = accepts_edge_attr(self._gconv_layers[0])
        
        if norm:
            self._norm = nn.GroupNorm(1, hidden_node_features)
        else:
            self._norm = None
        
        self._nonlinear = nn.ReLU()
        self._dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor, data: Data, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, V, F) batched node features
            data: Data object with edge_index, edge_attr, x (conditioning), etc.
            t: Optional time embedding
            
        Returns:
            (B, V, out_node_features) processed features
        """
        B, V, F = x.shape
        edge_index, edge_attr = data.edge_index, getattr(data, 'edge_attr', None)
        
        # Skip conditioning concatenation if not expected (cond_node_features == 0)
        if self.cond_node_features == 0:
            cond_x = None
        else:
            cond_x = data.x
        
        # Handle batched graphs - need to extract correct edge_index regardless of conditioning
        if hasattr(data, 'ptr') and hasattr(data, 'num_graphs') and data.num_graphs > 1:
            ptr = data.ptr
            batch_idx = data.batch
            edge_mask = (batch_idx[edge_index[0]] == 0) & (batch_idx[edge_index[1]] == 0)
            edge_index_original = edge_index[:, edge_mask]
            start_idx = int(ptr[0])
            edge_index = edge_index_original - start_idx
            
            if edge_attr is not None:
                edge_attr = edge_attr[edge_mask]
        
        # Process conditioning if expected
        if cond_x is not None:
            if hasattr(data, 'ptr') and hasattr(data, 'num_graphs') and data.num_graphs > 1:
                ptr = data.ptr
                cond_x_list = []
                for i in range(B):
                    start_idx = int(ptr[i])
                    end_idx = int(ptr[i+1])
                    graph_cond_x = cond_x[start_idx:end_idx]
                    V_i = graph_cond_x.shape[0]
                    
                    if V_i < V:
                        padding = torch.zeros(V - V_i, graph_cond_x.shape[1],
                                            dtype=graph_cond_x.dtype, device=graph_cond_x.device)
                        graph_cond_x_padded = torch.cat([graph_cond_x, padding], dim=0)
                    else:
                        graph_cond_x_padded = graph_cond_x[:V]
                    
                    cond_x_list.append(graph_cond_x_padded)
                
                cond_x = torch.stack(cond_x_list, dim=0)
            else:
                cond_x = cond_x.view(1, *cond_x.shape).expand(B, -1, -1)
        
        x_skip = x
        # Only concatenate conditioning if expected
        if cond_x is not None:
            x = torch.cat((x, cond_x), dim=-1)
            # Normalize concatenated features to prevent scale imbalance
            if self._input_norm is not None:
                x = self._input_norm(x)
        
        for i, (lnorm, linear, conv) in enumerate(zip(self._lnorm_layers[:-1], self._linear_layers[:-1], self._gconv_layers[:-1])):
            if self._norm is not None and x.shape[-1] == self.hidden_node_features:
                x = torch.movedim(x, -1, 1)
                x = self._norm(x)
                x = torch.movedim(x, 1, -1)
            
            x = conv(x, edge_index, edge_attr=edge_attr) if self.use_edge_attr else conv(x, edge_index)
            x = self._nonlinear(x)
            x = lnorm(x)
            x = linear(x)
            x = self._nonlinear(x)
            x = self._dropout(x)
        
        x = self._gconv_layers[-1](x, edge_index, edge_attr=edge_attr) if self.use_edge_attr else self._gconv_layers[-1](x, edge_index)
        
        if self._cond_layer is not None and t is not None:
            x = self._cond_layer(x, t)
        
        x = self._nonlinear(x)
        x = self._lnorm_layers[-1](x)
        x = self._linear_layers[-1](x)
        
        if self.residual:
            x = x + x_skip
        
        return x


class AttGNNBlock(nn.Module):
    """Attention-based GNN block combining GNN layers with self-attention."""
    
    def __init__(self, in_node_features: int, out_node_features: int,
                 hidden_node_features: int, cond_node_features: int,
                 attention_extra_features: int, edge_features: int, num_layers: int,
                 encoding_dim: int, residual: bool = True, norm: bool = True,
                 dropout: float = 0.0, conv_params: dict = None, device: str = "cpu", **kwargs):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.attention_extra_features = attention_extra_features
        self.edge_features = edge_features
        self.residual = residual
        
        self._gconv_layers = nn.ModuleList()
        self._att_extra_input_embed_layers = nn.ModuleList()
        self._attention_layers = nn.ModuleList()
        self._lnorm_layers = nn.ModuleList()
        self._linear_layers = nn.ModuleList()
        
        # Conditioning dimension (no lap_pe)
        self.cond_node_features = cond_node_features
        
        # Add LayerNorm to normalize concatenated features (x + cond_x) before first GNN layer
        # This prevents scale imbalance when original and normalized features are concatenated
        if cond_node_features > 0:
            self._input_norm = nn.LayerNorm(in_node_features + cond_node_features)
        else:
            self._input_norm = None
        
        self._cond_layer = FiLM(encoding_dim, hidden_node_features, channel_axis=-1) if encoding_dim > 0 else None
        
        conv_params = conv_params or {"layer_type": "gcn"}
        
        # Required attention parameters
        required_kwargs = ["num_heads", "ff_num_layers", "ff_size_factor", "att_implementation"]
        missing_kwargs = [k for k in required_kwargs if k not in kwargs]
        if missing_kwargs:
            raise ValueError(f"Missing required attention parameters: {missing_kwargs}")
        
        for i in range(num_layers):
            in_features = in_node_features + cond_node_features if i == 0 else hidden_node_features
            out_features = hidden_node_features if i < (num_layers - 1) else out_node_features
            
            self._gconv_layers.append(get_conv_layer(
                in_channels=in_features,
                out_channels=hidden_node_features,
                edge_features=edge_features,
                **conv_params
            ))
            
            att_extra_input_dim = cond_node_features + attention_extra_features
            self._att_extra_input_embed_layers.append(nn.Linear(att_extra_input_dim, hidden_node_features))
            
            self._attention_layers.append(AttentionBlock(
                kwargs["num_heads"],
                hidden_node_features,
                kwargs["ff_num_layers"],
                kwargs["ff_size_factor"],
                dropout,
                att_implementation=kwargs["att_implementation"]
            ))
            
            self._lnorm_layers.append(nn.LayerNorm(hidden_node_features))
            self._linear_layers.append(nn.Linear(hidden_node_features, out_features))
        
        self.use_edge_attr = accepts_edge_attr(self._gconv_layers[0])
        
        if norm:
            self._norm = nn.GroupNorm(1, hidden_node_features)
        else:
            self._norm = None
        
        self._nonlinear = nn.ReLU()
        self._dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor, data: Data, t: Optional[torch.Tensor] = None,
                att_extra_input: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, V, F) batched node features
            data: Data object with edge_index, edge_attr, x (conditioning), etc.
            t: Optional time embedding
            att_extra_input: Optional extra input for attention (B, V, attention_extra_features)
            attn_mask: Optional attention mask
            
        Returns:
            (B, V, out_node_features) processed features
        """
        B, V, F = x.shape
        if att_extra_input is not None and att_extra_input.shape[-1] != self.attention_extra_features:
            raise AssertionError(
                f"extra attention features must have right shape. "
                f"Expected shape[-1]={self.attention_extra_features}, got {att_extra_input.shape[-1]}. "
                f"att_extra_input.shape={att_extra_input.shape}"
            )
        
        edge_index, edge_attr = data.edge_index, getattr(data, 'edge_attr', None)
        
        # Skip conditioning concatenation if not expected (cond_node_features == 0)
        if self.cond_node_features == 0:
            cond_x = None
        else:
            cond_x = data.x
        
        # Handle batched graphs - extract edge_index regardless of conditioning
        if hasattr(data, 'ptr') and hasattr(data, 'num_graphs') and data.num_graphs > 1:
            ptr = data.ptr
            batch_idx = data.batch
            edge_mask = (batch_idx[edge_index[0]] == 0) & (batch_idx[edge_index[1]] == 0)
            edge_index_original = edge_index[:, edge_mask]
            start_idx = int(ptr[0])
            edge_index = edge_index_original - start_idx
            
            if edge_attr is not None:
                edge_attr = edge_attr[edge_mask]
        
        # Process conditioning if expected
        if cond_x is not None:
            if hasattr(data, 'ptr') and hasattr(data, 'num_graphs') and data.num_graphs > 1:
                ptr = data.ptr
                cond_x_list = []
                for i in range(B):
                    start_idx = int(ptr[i])
                    end_idx = int(ptr[i+1])
                    graph_cond_x = cond_x[start_idx:end_idx]
                    V_i = graph_cond_x.shape[0]
                    
                    if V_i < V:
                        padding = torch.zeros(V - V_i, graph_cond_x.shape[1],
                                            dtype=graph_cond_x.dtype, device=graph_cond_x.device)
                        graph_cond_x_padded = torch.cat([graph_cond_x, padding], dim=0)
                    else:
                        graph_cond_x_padded = graph_cond_x[:V]
                    
                    cond_x_list.append(graph_cond_x_padded)
                
                cond_x = torch.stack(cond_x_list, dim=0)
            else:
                cond_x = cond_x.view(1, *cond_x.shape).expand(B, -1, -1)
        
        x_skip = x
        # Only concatenate conditioning if expected
        if cond_x is not None:
            x = torch.cat((x, cond_x), dim=-1)
            # Normalize concatenated features to prevent scale imbalance
            if self._input_norm is not None:
                x = self._input_norm(x)
            x_att_features = torch.cat((cond_x, att_extra_input), dim=-1) if att_extra_input is not None else cond_x
        else:
            # When no conditioning, use att_extra_input directly (or skip attention features)
            x_att_features = att_extra_input
        
        for i, (lnorm, linear, conv, attention, att_input_embed_layer) in enumerate(
            zip(self._lnorm_layers[:-1], self._linear_layers[:-1], self._gconv_layers[:-1],
                self._attention_layers[:-1], self._att_extra_input_embed_layers[:-1])):
            
            if self._norm is not None and x.shape[-1] == self.hidden_node_features:
                x = torch.movedim(x, -1, 1)
                x = self._norm(x)
                x = torch.movedim(x, 1, -1)
            
            x = conv(x, edge_index, edge_attr=edge_attr) if self.use_edge_attr else conv(x, edge_index)
            x = self._nonlinear(x)
            # Only add attention features if available
            if x_att_features is not None:
                att_extra_embedded = att_input_embed_layer(x_att_features)
                x = x + att_extra_embedded
            x = attention(x, attn_mask=attn_mask)
            x = lnorm(x)
            x = linear(x)
            x = self._nonlinear(x)
            x = self._dropout(x)
        
        x = self._gconv_layers[-1](x, edge_index, edge_attr=edge_attr) if self.use_edge_attr else self._gconv_layers[-1](x, edge_index)
        
        if self._cond_layer is not None and t is not None:
            x = self._cond_layer(x, t)
        
        x = self._nonlinear(x)
        # Only add attention features if available
        if x_att_features is not None:
            att_extra_embedded = self._att_extra_input_embed_layers[-1](x_att_features)
            x = x + att_extra_embedded
        x = self._attention_layers[-1](x, attn_mask=attn_mask)
        x = self._lnorm_layers[-1](x)
        x = self._linear_layers[-1](x)
        
        if self.residual:
            x = x + x_skip
        
        return x


# ============================================================================
# Main AttGNN Model
# ============================================================================

class AttGNN(nn.Module):
    """Attention-based GNN model for graph processing."""
    
    def __init__(self, in_node_features: int, out_node_features: int,
                 hidden_size: int, hidden_node_features: list, attention_node_features: list,
                 cond_node_features: int, edge_features: int, layers_per_block: int,
                 t_encoding_dim: int, conv_params: dict, mlp_num_layers: int,
                 mlp_size_factor: int, input_encoding_dim: int = 0, dir_att_input: bool = False,
                 mask_key: Optional[str] = None, dropout: float = 0.0, device: str = "cpu",
                 res_gnn_layers_per_block: Optional[Union[int, List[int]]] = None,
                 att_gnn_layers_per_block: Optional[Union[int, List[int]]] = None,
                 **kwargs):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.attention_node_features = attention_node_features
        num_blocks = len(hidden_node_features)
        # ResGNN and AttGNN layers per block: int = same for all, list = per block
        self._res_layers = res_gnn_layers_per_block if res_gnn_layers_per_block is not None else layers_per_block
        self._att_layers = att_gnn_layers_per_block if att_gnn_layers_per_block is not None else 1
        if isinstance(self._res_layers, int):
            self._res_layers = [self._res_layers] * num_blocks
        if isinstance(self._att_layers, int):
            self._att_layers = [self._att_layers] * num_blocks
        # Extra features fed into attention blocks.
        # CRITICAL FIX: Use raw position features (2D) instead of hidden embeddings (256D)
        # This allows attention to directly reason about spatial relationships
        # Following ChipDiffusion's design: attention gets 2D position coordinates
        # Note: in_node_features here is hidden_dim (256), not the position dim
        self.attention_extra_features = 2  # 2D positions (x, y)
        self.edge_features = edge_features
        self.dir_att_input = dir_att_input
        self.mask_key = mask_key
        self.device = device
        
        # Hash-grid hierarchical attention config
        hash_grid_cfg = kwargs.pop("hash_grid", kwargs.pop("hash_grid_config", None))
        hash_grid_defaults = {
            "enabled": False,
            "levels": 2,
            "base_cell_size": 8.0,
            "cell_size_multiplier": 2.0,
            "grid_dim": 64,
            "window_sizes": [9, 9],
            "shifted": True,
            "ff_num_layers": 1,
            "ff_size_factor": 2,
            "rehash_interval": 1,
            "target_occupancy": 4.0,
            "num_attention_layers": 1,  # For curriculum learning
        }
        self.hash_grid_cfg = hash_grid_defaults if hash_grid_cfg is None else {**hash_grid_defaults, **hash_grid_cfg}
        self.hash_grid_enabled = self.hash_grid_cfg.get("enabled", False)
        if self.hash_grid_enabled:
            self.hash_grid_module = HashedGridHierarchy(
                hidden_dim=hidden_size,
                levels=self.hash_grid_cfg["levels"],
                base_cell_size=self.hash_grid_cfg["base_cell_size"],
                cell_size_multiplier=self.hash_grid_cfg["cell_size_multiplier"],
                grid_dim=self.hash_grid_cfg["grid_dim"],
                window_sizes=self.hash_grid_cfg["window_sizes"],
                shifted=self.hash_grid_cfg["shifted"],
                ff_num_layers=self.hash_grid_cfg["ff_num_layers"],
                ff_size_factor=self.hash_grid_cfg["ff_size_factor"],
                dropout=dropout,
                att_implementation=kwargs.get("att_implementation", "default"),
                rehash_interval=self.hash_grid_cfg["rehash_interval"],
                target_occupancy=self.hash_grid_cfg.get("target_occupancy", 4.0),
                num_attention_layers=self.hash_grid_cfg.get("num_attention_layers", 1),
            )
        else:
            self.hash_grid_module = None
        
        gnn_blocks = []
        self.use_enc = not (hidden_size == in_node_features == out_node_features)
        
        if self.use_enc:
            # Input: x (in_node_features=2) + cond.x (cond_node_features=7) = 9
            encoder_input_dim = in_node_features + cond_node_features
            gnn_blocks.append(LinearEncoderLayer(
                encoder_input_dim,
                hidden_size,
                mask_key=mask_key,
                input_encoding_dim=input_encoding_dim,
                device=device,
            ))
        
        for i, (hidden_node_size, attention_node_size) in enumerate(zip(hidden_node_features, attention_node_features)):
            res_layers_i = self._res_layers[i]
            att_layers_i = self._att_layers[i]
            gnn_blocks.append(ResGNNBlock(
                in_node_features=hidden_size,
                out_node_features=hidden_size,
                hidden_node_features=hidden_node_size,
                cond_node_features=cond_node_features,
                edge_features=edge_features,
                num_layers=res_layers_i,
                encoding_dim=t_encoding_dim,
                conv_params=conv_params,
                residual=True,
                norm=True,
                dropout=dropout,
                device=device,
                **kwargs,
            ))
            
            if mlp_num_layers > 0 and mlp_size_factor > 0:
                gnn_blocks.append(MLP(
                    mlp_num_layers,
                    mlp_size_factor * hidden_size,
                    hidden_size,
                    hidden_size,
                    skip=True,
                    layernorm=True,
                ))
            
            if attention_node_size > 0:
                gnn_blocks.append(AttGNNBlock(
                    in_node_features=hidden_size,
                    out_node_features=hidden_size,
                    hidden_node_features=attention_node_size,
                    cond_node_features=cond_node_features,
                    attention_extra_features=self.attention_extra_features,
                    edge_features=edge_features,
                    num_layers=att_layers_i,
                    encoding_dim=t_encoding_dim,
                    conv_params=conv_params,
                    residual=True,
                    norm=True,
                    dropout=dropout,
                    device=device,
                    **kwargs,
                ))
            else:
                gnn_blocks.append(ResGNNBlock(
                    in_node_features=hidden_size,
                    out_node_features=hidden_size,
                    hidden_node_features=hidden_node_size,
                    cond_node_features=cond_node_features,
                    edge_features=edge_features,
                    num_layers=att_layers_i,
                    encoding_dim=t_encoding_dim,
                    conv_params=conv_params,
                    residual=True,
                    norm=True,
                    dropout=dropout,
                    device=device,
                    **kwargs,
                ))
            
            if mlp_num_layers > 0 and mlp_size_factor > 0:
                gnn_blocks.append(MLP(
                    mlp_num_layers,
                    mlp_size_factor * hidden_size,
                    hidden_size,
                    hidden_size,
                    skip=True,
                    layernorm=True,
                ))
        
        if self.use_enc:
            gnn_blocks.append(LinearDecoderLayer(hidden_size, out_node_features))
            if self.in_node_features != self.out_node_features:
                self._skip_linear = nn.Linear(in_node_features, self.out_node_features)
        
        self._gnn_blocks = nn.ModuleList(gnn_blocks)
    
    def forward(self, x: torch.Tensor, cond: Data, t_embed: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None, step_idx: Optional[int] = None,
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, V, F) batched node features
            cond: Data object with graph structure (edge_index, edge_attr, etc.)
            t_embed: Optional (B, t_dim) time embeddings
            attn_mask: Optional attention mask
            step_idx: Optional refinement step index for controlling rehash
            positions: Optional (B, V, 2) position coordinates for attention input

        Returns:
            (B, V, out_node_features) processed features
        """
        x_skip = x
        # CRITICAL FIX: Use provided position coordinates instead of extracting from x
        # x has been encoded to hidden_dim (256), so x[..., :2] would NOT be positions
        # positions should be passed from model.py (x_t parameter)
        if positions is not None:
            coords = positions  # (B, V, 2) - actual position coordinates
        else:
            # Fallback: assume first two dims are coords (for backward compatibility)
            coords = x[..., :2]
        batch_idx = getattr(cond, "batch", None)
        
        for block in self._gnn_blocks:
            if self.hash_grid_enabled and isinstance(block, (ResGNNBlock, AttGNNBlock)):
                grid_context = self.hash_grid_module(
                    node_feats=x,
                    coords=coords,
                    batch=batch_idx if batch_idx is not None else None,
                    step_idx=step_idx,
                )
                # Ensure gradient flow: even if both x and grid_context don't require gradients
                # (because layers are frozen), we need to maintain the computation graph
                # for downstream trainable layers. Use addition that preserves graph structure.
                # Critical: Ensure gradient flow even when module is frozen
                # If grid_context doesn't require gradients (frozen module), we still need to
                # maintain the computation graph for downstream trainable layers
                # The addition should work, but we ensure x remains connected to the graph
                x = x + grid_context
            if isinstance(block, AttGNNBlock):
                # CRITICAL FIX: Use raw position coordinates for attention input
                # Instead of full hidden embeddings (256D), use just x, y positions (2D)
                # This allows attention to directly reason about spatial relationships
                att_input = coords  # Use positions (B, V, 2), NOT embeddings (B, V, 256)
                x = block(x, cond, t_embed, att_extra_input=att_input, attn_mask=attn_mask)
            elif isinstance(block, MLP):
                x = block(x)
            else:
                x = block(x, cond, t_embed)
        
        if self.use_enc:
            if self.in_node_features != self.out_node_features:
                x_skip = self._skip_linear(x_skip)
            x = x + x_skip
        
        # CRITICAL: Ensure output requires gradients even when most layers are frozen
        # This is necessary for finetuning scenarios where only some layers (e.g., decoders) are trainable
        # Even if x doesn't require gradients (from frozen layers), we need to maintain the graph
        # for downstream trainable layers. Check if any parameters in this module require gradients.
        has_trainable_params = any(p.requires_grad for p in self.parameters())
        if not x.requires_grad and has_trainable_params:
            # Connect to a trainable parameter to maintain gradient flow
            # Find any trainable parameter and create a connection
            for param in self.parameters():
                if param.requires_grad:
                    # Create a minimal connection: x + param * 0.0
                    # This maintains the graph without changing the value
                    x = x + param.sum() * 0.0
                    break
        
        return x
