"""
Unified Global Module System

This module provides all global processing modules in one place:
1. Identity - No-op baseline
2. Attention - Transformer/self-attention (with ModuleList for fine-grained layer control)
3. Hash Grid - Multi-scale hierarchical attention
4. Gated Residual - Wrapper for progressive architecture extension

All modules implement the BaseGlobalModule interface and support fine-grained
layer freezing/training for curriculum learning.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from .att_gnn import HashedGridHierarchy


# ============================================================================
# Base Interface
# ============================================================================

class BaseGlobalModule(nn.Module, ABC):
    """
    Base class for all global modules that can be plugged into FullInstNetModel.
    
    Global modules process features after GNN blocks to add global context.
    They can be attention-based (transformer), hash-grid based, or identity.
    
    All global modules must implement the forward method with this signature.
    """
    
    def __init__(self, hidden_dim: int, **kwargs):
        """
        Initialize the global module.
        
        Args:
            hidden_dim: Hidden dimension (must match inst_x and net_x dimensions)
            **kwargs: Additional module-specific parameters
        """
        super().__init__()
        self.hidden_dim = hidden_dim
    
    @abstractmethod
    def forward(
        self,
        inst_x: torch.Tensor,
        net_x: torch.Tensor,
        inst_batch: Optional[torch.Tensor] = None,
        net_batch: Optional[torch.Tensor] = None,
        inst_rel_pos_bias: Optional[torch.Tensor] = None,
        net_rel_pos_bias: Optional[torch.Tensor] = None,
        inst_pos: Optional[torch.Tensor] = None,
        net_pos: Optional[torch.Tensor] = None,
        data: Optional[HeteroData] = None,
        time_embed: Optional[torch.Tensor] = None,
        step_idx: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process instance and net features to add global context.
        
        Args:
            inst_x: [N_inst_total, hidden_dim] concatenated instance features across batch
            net_x: [N_net_total, hidden_dim] concatenated net features across batch
            inst_batch: [N_inst_total] batch assignment for each instance (graph index)
            net_batch: [N_net_total] batch assignment for each net (graph index)
            inst_rel_pos_bias: Optional relative position bias for instances
            net_rel_pos_bias: Optional relative position bias for nets
            inst_pos: [N_inst_total, 2] instance coordinates (optional)
            net_pos: [N_net_total, 2] net coordinates (optional)
            data: Optional HeteroData object with full graph information
            time_embed: [N_inst_total, time_embed_dim] time embeddings (optional, for diffusion)
            step_idx: Optional integer timestep index (for hash grid rehashing)
            **kwargs: Additional arguments for module-specific processing
            
        Returns:
            inst_output: [N_inst_total, hidden_dim] updated instance features
            net_output: [N_net_total, hidden_dim] updated net features
        """
        pass


# ============================================================================
# Global Module Registry
# ============================================================================

_GLOBAL_MODULE_REGISTRY: Dict[str, type] = {}


def register_global_module(name: str):
    """
    Decorator to register a global module class.
    
    Usage:
        @register_global_module("my_module")
        class MyGlobalModule(BaseGlobalModule):
            ...
    """
    def decorator(cls: type):
        if not issubclass(cls, BaseGlobalModule):
            raise TypeError(f"{cls.__name__} must inherit from BaseGlobalModule")
        _GLOBAL_MODULE_REGISTRY[name] = cls
        return cls
    return decorator


def get_global_module(name: str) -> type:
    """Get a registered global module class by name."""
    if name not in _GLOBAL_MODULE_REGISTRY:
        raise ValueError(
            f"Unknown global module: '{name}'. "
            f"Available modules: {list(_GLOBAL_MODULE_REGISTRY.keys())}"
        )
    return _GLOBAL_MODULE_REGISTRY[name]


def list_global_modules() -> list:
    """List all registered global module names."""
    return list(_GLOBAL_MODULE_REGISTRY.keys())


# ============================================================================
# Attention Building Blocks (from global_attention.py)
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module with Flash Attention.
    
    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_layer_norm: Whether to use layer normalization
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self._init_attention_weights()
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
    
    def _init_attention_weights(self):
        """Initialize attention weights for pre-norm transformer stability."""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.5)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rel_pos_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with Flash Attention.
        
        Args:
            x: [B, N, hidden_dim] or [N, hidden_dim] input features
            mask: [B, N] or [N] attention mask where True=valid, False=padded (optional)
            rel_pos_bias: [B, N, N] or [N, N] relative position bias (optional).
                          If provided, uses manual attention computation instead of Flash Attention.
            
        Returns:
            output: [B, N, hidden_dim] or [N, hidden_dim] output features
        """
        input_shape = x.shape
        if len(input_shape) == 2:
            x = x.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        B, N, _ = x.shape
        residual = x
        
        # Apply LayerNorm then zero padded positions — single fused CUDA kernels,
        # no index-gather/scatter overhead.
        x = self.layer_norm(x)
        if mask is not None and not mask.all():
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # FORCE Flash Attention: relative position bias is NOT allowed (triggers manual fallback).
        if rel_pos_bias is not None:
            raise RuntimeError(
                "Flash attention is required but rel_pos_bias was provided. "
                "Set use_rel_pos_bias: false in config to use flash attention."
            )

        # Convert mask to SDPA attn_mask format for Flash path. SDPA supports bool mask: True=keep.
        # Key padding: mask [B, N] True=valid key -> attn_mask [B, 1, 1, N] broadcasts to [B, H, L, S]
        attn_mask_sdpa = None
        if mask is not None:
            key_mask = mask.to(dtype=torch.bool)  # [B, N], True=valid
            attn_mask_sdpa = key_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]

        # Always use SDPA (Flash Attention preferred). rel_pos_bias would force manual path - forbidden above.
        # When attn_mask is used (padding), Flash may fall back to mem_efficient/math; that is acceptable.
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask_sdpa,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )

        # Zero outputs for padded queries when mask was provided (SDPA may not handle query padding)
        if mask is not None:
            key_mask = mask.to(dtype=torch.bool)
            attn_output = attn_output.masked_fill(~key_mask.unsqueeze(1).unsqueeze(-1), 0.0)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        output = output + residual
        
        if len(input_shape) == 2:
            output = output.squeeze(0)
        
        return output


class TopKGatedMultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with learnable top-K gating.

    Each query attends to at most *top_k* keys per graph, preventing attention
    dilution on large OOD graphs.

    Two-phase forward:
      1. **Gate** — lightweight projections (gate_q, gate_k) produce per-pair
         relevance scores; the top-K keys per query are selected.
      2. **Attend** — standard multi-head softmax attention computed only over
         the selected K keys (others masked to -inf).

    The gate shares a single top-K mask across all heads; attention Q/K/V are
    per-head as usual.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        top_k: int = 64,
        gate_dim: int = 32,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.top_k = top_k
        self.gate_dim = gate_dim

        # Gate projections (shared across heads, lightweight)
        self.gate_q = nn.Linear(hidden_dim, gate_dim)
        self.gate_k = nn.Linear(hidden_dim, gate_dim)
        self.gate_scale = gate_dim ** -0.5

        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

    def _init_weights(self):
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(proj.weight, gain=0.5)
            nn.init.zeros_(proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        nn.init.zeros_(self.out_proj.bias)
        for proj in (self.gate_q, self.gate_k):
            nn.init.xavier_uniform_(proj.weight, gain=1.0)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rel_pos_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] or [N, D]
            mask: [B, N] or [N], True = valid token
            rel_pos_bias: ignored (kept for interface compat)
        Returns:
            [B, N, D] or [N, D]
        """
        input_shape = x.shape
        if len(input_shape) == 2:
            x = x.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        B, N, _ = x.shape
        residual = x

        x = self.layer_norm(x)
        if mask is not None and not mask.all():
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        # --- Phase 1: learnable gate → top-K selection ---
        gq = self.gate_q(x)  # [B, N, gate_dim]
        gk = self.gate_k(x)  # [B, N, gate_dim]
        gate_logits = torch.bmm(gq, gk.transpose(1, 2)) * self.gate_scale  # [B, N, N]

        # Mask out padding keys before top-K selection
        if mask is not None and not mask.all():
            gate_logits = gate_logits.masked_fill(~mask.unsqueeze(1), float("-inf"))

        effective_k = min(self.top_k, N)
        _, topk_idx = gate_logits.topk(effective_k, dim=-1)  # [B, N, K]

        # Build a boolean mask [B, N, N] with True only at top-K positions
        topk_mask = torch.zeros(B, N, N, device=x.device, dtype=torch.bool)
        topk_mask.scatter_(2, topk_idx, True)

        # --- Phase 2: standard multi-head attention with top-K mask ---
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)

        # Apply top-K mask (shared across heads): [B, 1, N, N]
        attn_logits = attn_logits.masked_fill(~topk_mask.unsqueeze(1), float("-inf"))

        # Apply padding mask on keys: [B, 1, 1, N]
        if mask is not None and not mask.all():
            attn_logits = attn_logits.masked_fill(~mask[:, None, None, :], float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = attn_weights.nan_to_num(0.0)  # all-masked rows → 0
        attn_weights = self.dropout(attn_weights) if self.training else attn_weights

        attn_out = torch.matmul(attn_weights, v)  # [B, H, N, head_dim]

        if mask is not None:
            attn_out = attn_out.masked_fill(~mask[:, None, :, None], 0.0)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)
        output = self.out_proj(attn_out)
        output = self.dropout(output)
        output = output + residual

        if len(input_shape) == 2:
            output = output.squeeze(0)
        return output


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention module with Flash Attention.
    
    Cross-attention: queries come from one sequence, keys/values from another.
    This is used for Perceiver-style modules where we attend from queries to keys/values.
    
    Args:
        hidden_dim: Hidden dimension (must match query_dim and kv_dim)
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_layer_norm: Whether to use layer normalization
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query projections (from query sequence)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # Key/Value projections (from key/value sequence)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self._init_attention_weights()
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
    
    def _init_attention_weights(self):
        """Initialize attention weights for pre-norm transformer stability."""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.5)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with Flash Attention.
        
        Args:
            query: [B, N_q, hidden_dim] or [N_q, hidden_dim] query features
            key_value: [B, N_kv, hidden_dim] or [N_kv, hidden_dim] key/value features
            query_mask: [B, N_q] or [N_q] attention mask for queries (optional)
            kv_mask: [B, N_kv] or [N_kv] attention mask for keys/values (optional)
            attn_bias: [B, N_q, N_kv] or [B, 1, N_q, N_kv] additive bias on attention scores (optional).
                When provided, uses manual attention path (no Flash) so bias is applied correctly.
            
        Returns:
            output: [B, N_q, hidden_dim] or [N_q, hidden_dim] output features
        """
        query_shape = query.shape
        kv_shape = key_value.shape
        
        # Handle 2D inputs
        if len(query_shape) == 2:
            query = query.unsqueeze(0)
            if query_mask is not None:
                query_mask = query_mask.unsqueeze(0)
        if len(kv_shape) == 2:
            key_value = key_value.unsqueeze(0)
            if kv_mask is not None:
                kv_mask = kv_mask.unsqueeze(0)
        
        B, N_q, _ = query.shape
        _, N_kv, _ = key_value.shape
        residual = query
        
        # Apply LayerNorm then zero padded query positions — single fused CUDA kernels,
        # no index-gather/scatter overhead.
        query = self.layer_norm(query)
        if query_mask is not None and not query_mask.all():
            query = query.masked_fill(~query_mask.unsqueeze(-1), 0.0)
        
        # Project queries, keys, values
        q = self.q_proj(query)  # [B, N_q, hidden_dim]
        k = self.k_proj(key_value)  # [B, N_kv, hidden_dim]
        v = self.v_proj(key_value)  # [B, N_kv, hidden_dim]
        
        # Reshape for multi-head
        q = q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N_q, head_dim]
        k = k.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N_kv, head_dim]
        v = v.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N_kv, head_dim]

        # FORCE Flash Attention: attn_bias is NOT allowed (triggers manual fallback).
        if attn_bias is not None:
            raise RuntimeError(
                "Flash attention is required but attn_bias was provided. "
                "Set use_distance_penalty: false in config to use flash attention."
            )

        # Convert kv_mask to SDPA attn_mask: [B, N_kv] True=valid -> [B, 1, 1, N_kv]
        attn_mask_sdpa = None
        if kv_mask is not None:
            attn_mask_sdpa = kv_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(2)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask_sdpa,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )

        # Zero outputs for padded queries when query_mask was provided
        if query_mask is not None:
            q_mask = query_mask.to(dtype=torch.bool)
            if len(q_mask.shape) == 2:
                attn_output = attn_output.masked_fill(~q_mask.unsqueeze(1).unsqueeze(-1), 0.0)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_q, self.hidden_dim)
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        output = output + residual
        
        if len(query_shape) == 2:
            output = output.squeeze(0)
        
        return output


class AttGNNStyleBlock(nn.Module):
    """
    Attention block matching AttGNNBlock style: Attention → LayerNorm → Linear
    
    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_layer_norm: Whether to use layer normalization
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.self_attn = MultiHeadSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=True,
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rel_pos_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: Attention → LayerNorm → Linear"""
        # Pass rel_pos_bias to self-attention
        x = self.self_attn(x, mask=mask, rel_pos_bias=rel_pos_bias)
        x = self.layer_norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and optional feed-forward network.
    
    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension (if None, defaults to 4 * hidden_dim)
        dropout: Dropout rate
        activation: Activation function ("gelu" or "relu")
        use_layer_norm: Whether to use layer normalization
        att_gnn_style: If True, use AttGNNStyleBlock (Attention → LayerNorm → Linear)
                       If False, use standard transformer (pre-norm style)
        top_k: If > 0, use TopKGatedMultiHeadSelfAttention with learnable
               gate that restricts each query to attend to at most top_k keys.
        gate_dim: Dimension for the lightweight gate projections (only used when top_k > 0).
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layer_norm: bool = True,
        att_gnn_style: bool = False,
        top_k: int = 0,
        gate_dim: int = 32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.att_gnn_style = att_gnn_style
        
        if att_gnn_style:
            self.block = AttGNNStyleBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
        else:
            self.ff_dim = ff_dim if ff_dim is not None else 4 * hidden_dim
            self.use_ffn = self.ff_dim > 0
            
            if top_k > 0:
                self.self_attn = TopKGatedMultiHeadSelfAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                    top_k=top_k,
                    gate_dim=gate_dim,
                )
            else:
                self.self_attn = MultiHeadSelfAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
            
            if self.use_ffn:
                self.ff = nn.Sequential(
                    nn.Linear(hidden_dim, self.ff_dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.ff_dim, hidden_dim),
                    nn.Dropout(dropout),
                )
                self._init_ffn_weights()
            else:
                self.ff = nn.Identity()
            
            self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
    
    def _init_ffn_weights(self):
        """Initialize FFN weights for pre-norm transformer stability."""
        if not self.use_ffn:
            return
        
        nn.init.xavier_uniform_(self.ff[0].weight, gain=1.0)
        nn.init.zeros_(self.ff[0].bias)
        nn.init.xavier_uniform_(self.ff[3].weight, gain=0.5)
        nn.init.zeros_(self.ff[3].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rel_pos_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        if self.att_gnn_style:
            return self.block(x, mask=mask, rel_pos_bias=rel_pos_bias)
        
        # Standard transformer style
        # Pass rel_pos_bias to self-attention
        x = self.self_attn(x, mask=mask, rel_pos_bias=rel_pos_bias)
        
        if self.use_ffn:
            residual = x
            
            if mask is not None and len(x.shape) == 3 and not mask.all():
                B, N, _ = x.shape
                x_flat = x.reshape(-1, self.hidden_dim)
                mask_flat = mask.reshape(-1)
                
                x_normalized_flat = torch.zeros_like(x_flat)
                if mask_flat.any():
                    valid_indices = torch.where(mask_flat)[0]
                    # LayerNorm runs in fp32 under AMP; cast back for indexed writes.
                    x_normalized_flat[valid_indices] = self.layer_norm(
                        x_flat[valid_indices]
                    ).to(dtype=x_flat.dtype)
                
                x = x_normalized_flat.reshape(B, N, self.hidden_dim)
            else:
                x = self.layer_norm(x)
            
            x = self.ff(x)
            x = x + residual
        
        return x


class LocalWindowsTransformer(nn.Module):
    """
    Local-windows transformer that divides nodes into spatial windows.
    
    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        ff_dim: Feed-forward dimension (0 to disable FFN)
        dropout: Dropout rate
        activation: Activation function
        use_layer_norm: Whether to use layer normalization
        window_size: Maximum nodes per window (default: 128)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layer_norm: bool = True,
        window_size: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.window_size = window_size
        self.use_ffn = ff_dim != 0
        
        if self.use_ffn:
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    activation=activation,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_layers)
            ])
        else:
            self.attention_blocks = nn.ModuleList([
                MultiHeadSelfAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_layers)
            ])
    
    def _determine_grid_size(self, num_nodes: int) -> int:
        """Determine grid size based on total node count."""
        if num_nodes <= self.window_size:
            return 1
        elif num_nodes <= self.window_size * 4:
            return 2
        elif num_nodes <= self.window_size * 9:
            return 3
        else:
            num_windows = max(4, math.ceil(num_nodes / self.window_size))
            grid_size = math.ceil(math.sqrt(num_windows))
            return grid_size
    
    def _assign_to_windows_batched(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        grid_size: int,
    ) -> torch.Tensor:
        """Assign nodes to windows with batch offsets (vectorized)."""
        device = pos.device
        N = pos.shape[0]
        batch_size = batch.max().item() + 1
        num_windows_per_graph = grid_size * grid_size
        
        # Compute min/max per batch
        pos_min = torch.zeros(batch_size, 2, device=device, dtype=pos.dtype)
        pos_max = torch.zeros(batch_size, 2, device=device, dtype=pos.dtype)
        
        if hasattr(torch.Tensor, 'scatter_reduce'):
            pos_min_init = torch.full((batch_size, 2), float('inf'), device=device, dtype=pos.dtype)
            pos_max_init = torch.full((batch_size, 2), float('-inf'), device=device, dtype=pos.dtype)
            batch_expanded = batch.unsqueeze(1).expand(-1, 2)
            pos_min = pos_min_init.scatter_reduce(0, batch_expanded, pos, reduce='amin', include_self=False)
            pos_max = pos_max_init.scatter_reduce(0, batch_expanded, pos, reduce='amax', include_self=False)
        else:
            for b in range(batch_size):
                mask = (batch == b)
                pos_b = pos[mask]
                if pos_b.shape[0] > 0:
                    pos_min[b] = pos_b.min(dim=0)[0]
                    pos_max[b] = pos_b.max(dim=0)[0]
        
        # Normalize positions per graph
        pos_min_expanded = pos_min[batch]
        pos_max_expanded = pos_max[batch]
        pos_range = pos_max_expanded - pos_min_expanded
        pos_range = torch.where(pos_range > 0, pos_range, torch.ones_like(pos_range))
        pos_norm = (pos - pos_min_expanded) / pos_range
        
        # Discretize to grid cells
        grid_x = torch.clamp((pos_norm[:, 0] * grid_size).long(), 0, grid_size - 1)
        grid_y = torch.clamp((pos_norm[:, 1] * grid_size).long(), 0, grid_size - 1)
        
        window_ids_local = grid_y * grid_size + grid_x
        window_ids = window_ids_local + batch * num_windows_per_graph
        
        return window_ids
    
    def _prepare_window_batches(
        self,
        x: torch.Tensor,
        window_ids: torch.Tensor,
        num_windows: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare windowed batches with padding (vectorized)."""
        device = x.device
        N = x.shape[0]
        
        window_counts = torch.bincount(window_ids, minlength=num_windows)
        sorted_indices = torch.argsort(window_ids)
        sorted_window_ids = window_ids[sorted_indices]
        
        window_offsets = torch.cat([torch.tensor([0], device=device), window_counts.cumsum(0)[:-1]])
        positions_in_window = torch.arange(N, device=device) - window_offsets[sorted_window_ids]
        valid_mask = positions_in_window < self.window_size
        
        window_x = torch.zeros(num_windows, self.window_size, self.hidden_dim, 
                              device=device, dtype=x.dtype)
        window_mask = torch.zeros(num_windows, self.window_size, 
                                  device=device, dtype=torch.bool)
        
        valid_sorted_indices = sorted_indices[valid_mask]
        valid_positions = positions_in_window[valid_mask]
        valid_window_ids = sorted_window_ids[valid_mask]
        
        window_x[valid_window_ids, valid_positions] = x[valid_sorted_indices]
        window_mask[valid_window_ids, valid_positions] = True
        
        node_to_window_slot = torch.full((N,), -1, dtype=torch.long, device=device)
        node_to_window_slot[valid_sorted_indices] = valid_positions
        
        return window_x, window_mask, node_to_window_slot
    
    def _scatter_back(
        self,
        window_x: torch.Tensor,
        window_ids: torch.Tensor,
        node_to_window_slot: torch.Tensor,
        original_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Scatter windowed features back to original node ordering (vectorized)."""
        N, hidden_dim = original_shape
        device = window_x.device
        
        x_out = torch.zeros(N, hidden_dim, device=device, dtype=window_x.dtype)
        valid_mask = node_to_window_slot >= 0
        valid_node_indices = torch.where(valid_mask)[0]
        
        valid_window_ids = window_ids[valid_node_indices]
        valid_slots = node_to_window_slot[valid_node_indices]
        
        x_out[valid_node_indices] = window_x[valid_window_ids, valid_slots]
        
        return x_out
    
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through local windows."""
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        batch_size = batch.max().item() + 1
        counts_per_graph = torch.bincount(batch, minlength=batch_size)
        max_nodes_per_graph = counts_per_graph.max().item()
        
        grid_size = self._determine_grid_size(max_nodes_per_graph)
        num_windows_per_graph = grid_size * grid_size
        total_windows = batch_size * num_windows_per_graph
        
        window_ids = self._assign_to_windows_batched(pos, batch, grid_size)
        window_x, window_mask, node_to_window_slot = self._prepare_window_batches(
            x, window_ids, total_windows
        )
        
        # Apply transformer to all windows
        if self.use_ffn:
            for transformer_block in self.transformer_blocks:
                window_x = transformer_block(window_x, mask=window_mask)
        else:
            for attention_block in self.attention_blocks:
                window_x = attention_block(window_x, mask=window_mask)
        
        x_out = self._scatter_back(window_x, window_ids, node_to_window_slot, 
                                   (x.shape[0], self.hidden_dim))
        
        return x_out


class StableLocalAttention(nn.Module):
    """k-NN local self-attention."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        k_neighbors: int = 32,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.k_neighbors = k_neighbors
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

    def _compute_knn_indices(
        self, pos: torch.Tensor, batch: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Returns indices of k nearest neighbors per node."""
        N = pos.shape[0]
        if N == 0:
            return pos.new_zeros((0, self.k_neighbors), dtype=torch.long)

        if batch is None:
            dists = torch.cdist(pos, pos)
            _, knn = torch.topk(dists, k=min(self.k_neighbors, N), largest=False)
            if knn.shape[1] < self.k_neighbors:
                pad = self.k_neighbors - knn.shape[1]
                knn = torch.cat([knn, knn[:, -1:].expand(-1, pad)], dim=1)
            return knn

        device = pos.device
        batch_size = int(batch.max().item()) + 1
        
        # OPTIMIZATION: Pre-compute masks and indices to reduce overhead
        graph_masks = [batch == b for b in range(batch_size)]
        graph_indices = [torch.where(mask)[0] for mask in graph_masks]
        
        # Process each graph (k-NN computation is O(N^2) per graph, requires per-graph processing)
        knn_list = []
        for b in range(batch_size):
            idx = graph_indices[b]
            if idx.numel() == 0:
                continue
            pos_b = pos[idx]
            dists = torch.cdist(pos_b, pos_b)
            k = min(self.k_neighbors, pos_b.shape[0])
            _, local_knn = torch.topk(dists, k=k, largest=False)
            if local_knn.shape[1] < self.k_neighbors:
                pad = self.k_neighbors - local_knn.shape[1]
                local_knn = torch.cat([local_knn, local_knn[:, -1:].expand(-1, pad)], dim=1)
            global_knn = idx[local_knn]
            knn_list.append((idx, global_knn))

        # Vectorized assignment
        knn = torch.zeros((N, self.k_neighbors), device=device, dtype=torch.long)
        for idx, global_knn in knn_list:
            knn[idx] = global_knn
        return knn

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        k_neighbors: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass with k-NN attention."""
        if x.numel() == 0:
            return x

        if k_neighbors is not None and k_neighbors != self.k_neighbors:
            k_backup = self.k_neighbors
            self.k_neighbors = k_neighbors
            knn_idx = self._compute_knn_indices(pos, batch)
            self.k_neighbors = k_backup
        else:
            knn_idx = self._compute_knn_indices(pos, batch)

        residual = x
        x_norm = self.layer_norm(x)

        q = self.q_proj(x_norm).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x_norm).view(-1, self.num_heads, self.head_dim)

        knn_exp = knn_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_heads, self.head_dim)
        k_neighbors = torch.gather(k, 0, knn_exp)
        v_neighbors = torch.gather(v, 0, knn_exp)

        q_exp = q.unsqueeze(1)
        attn_scores = (q_exp * k_neighbors).sum(-1) * self.scale
        # CRITICAL FIX: Clip attention scores to prevent overflow in softmax
        attn_scores = torch.clamp(attn_scores, min=-50.0, max=50.0)
        attn_probs = torch.softmax(attn_scores, dim=1)
        attn_probs = self.dropout(attn_probs)

        attn_out = (attn_probs.unsqueeze(-1) * v_neighbors).sum(1)
        attn_out = attn_out.reshape(-1, self.hidden_dim)
        out = self.out_proj(attn_out)
        out = self.dropout(out)
        return residual + out


# ============================================================================
# Core Global Modules (Identity, Attention, Hash Grid)
# ============================================================================

@register_global_module("identity")
class IdentityGlobalModule(BaseGlobalModule):
    """
    Identity module that returns inputs unchanged.
    Useful for ablation studies to disable global processing.
    """
    
    def __init__(self, hidden_dim: int, **kwargs):
        super().__init__(hidden_dim, **kwargs)
    
    def forward(
        self,
        inst_x: torch.Tensor,
        net_x: torch.Tensor,
        inst_batch: Optional[torch.Tensor] = None,
        net_batch: Optional[torch.Tensor] = None,
        inst_rel_pos_bias: Optional[torch.Tensor] = None,
        net_rel_pos_bias: Optional[torch.Tensor] = None,
        inst_pos: Optional[torch.Tensor] = None,
        net_pos: Optional[torch.Tensor] = None,
        data: Optional[HeteroData] = None,
        time_embed: Optional[torch.Tensor] = None,
        step_idx: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Identity module returns zero deltas (FullInstNetModel applies residual add)
        inst_delta = torch.zeros_like(inst_x)
        net_delta = torch.zeros_like(net_x) if net_x is not None else inst_delta
        return inst_delta, net_delta


@register_global_module("attention")
class AttentionGlobalModule(BaseGlobalModule):
    """
    Global attention module with direct ModuleList access for fine-grained layer control.
    
    Supports:
    - transformer: Stack of TransformerBlock layers (self-attention + FFN)
    - self_attention: Single MultiHeadSelfAttention layer
    - local_windows: LocalWindowsTransformer (spatial windowing)
    - knn_local: StableLocalAttention (k-NN based)
    
    The inst_transformer and net_transformer are ModuleLists, allowing direct access
    to individual layers for fine-grained freezing (e.g., freeze inst_transformer.0, train inst_transformer.1).
    
    Args:
        hidden_dim: Hidden dimension
        attention_type: "transformer", "self_attention", "local_windows", or "knn_local"
        num_heads: Number of attention heads
        num_layers: Number of transformer layers (for transformer type)
        ff_dim: Feed-forward dimension (None = 4*hidden_dim, 0 = no FFN)
        dropout: Dropout rate
        activation: Activation function ("gelu" or "relu")
        use_layer_norm: Whether to use layer normalization
        apply_to_nets: If True, also process nets (memory intensive)
        window_size: Window size for local_windows/knn_local
        att_gnn_style: Use AttGNNBlock style (Attention → LayerNorm → Linear)
        top_k: If > 0, transformer blocks use TopKGatedMultiHeadSelfAttention
               where each query attends to at most top_k keys.
        gate_dim: Dimension of the lightweight gate projections (only used when top_k > 0).
    """
    
    def __init__(
        self,
        hidden_dim: int,
        attention_type: str = "transformer",
        num_heads: int = 8,
        num_layers: int = 1,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layer_norm: bool = True,
        apply_to_nets: bool = False,
        window_size: int = 128,
        global_layers: int = 1,  # Unused, kept for backward compatibility
        att_gnn_style: bool = True,
        use_pos_encoding: bool = True,  # NEW: Enable positional encoding for inst_pos
        time_embed_dim: Optional[int] = None,  # NEW: Time embedding dimension for proper initialization
        top_k: int = 0,
        gate_dim: int = 32,
        **kwargs
    ):
        super().__init__(hidden_dim, **kwargs)
        self.attention_type = attention_type
        self.num_layers = num_layers
        self.apply_to_nets = apply_to_nets
        self.att_gnn_style = att_gnn_style
        self.use_pos_encoding = use_pos_encoding
        
        # Positional encoder for inst_pos (noisy coordinates during diffusion)
        # This helps the attention module understand spatial relationships
        if use_pos_encoding:
            self.pos_encoder = nn.Sequential(
                nn.Linear(2, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
            )
        
        # Relative position bias configuration
        self.use_rel_pos_bias = kwargs.get('use_rel_pos_bias', False)
        self.rel_pos_bias_type = kwargs.get('rel_pos_bias_type', 'l2')  # 'l1' or 'l2'
        self.rel_pos_bias_learnable = kwargs.get('rel_pos_bias_learnable', True)
        
        if self.use_rel_pos_bias:
            # Learnable projection from distance to bias value
            # Input: distance (scalar) -> Output: bias value (scalar)
            if self.rel_pos_bias_learnable:
                self.rel_pos_bias_proj = nn.Sequential(
                    nn.Linear(1, hidden_dim // 4),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 4, 1),
                )
            else:
                self.rel_pos_bias_proj = None
        
        # Initialize FiLM for time conditioning if time_embed_dim is provided
        # FiLM provides more expressive conditioning than additive (per-feature scaling/shifting)
        if time_embed_dim is not None:
            from ....training.diffusion_modules import FiLM
            self.time_film = FiLM(cond_dim=time_embed_dim, feature_dim=hidden_dim)
        else:
            self.time_film = None
        
        if attention_type == "transformer":
            # CRITICAL: Use ModuleList for fine-grained layer access
            # This allows freezing/training individual layers: inst_transformer.0, inst_transformer.1, etc.
            self.inst_transformer = nn.ModuleList([
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    activation=activation,
                    use_layer_norm=use_layer_norm,
                    att_gnn_style=att_gnn_style,
                    top_k=top_k,
                    gate_dim=gate_dim,
                )
                for _ in range(num_layers)
            ])
            
            if apply_to_nets:
                self.net_transformer = nn.ModuleList([
                    TransformerBlock(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        ff_dim=ff_dim,
                        dropout=dropout,
                        activation=activation,
                        use_layer_norm=use_layer_norm,
                        att_gnn_style=att_gnn_style,
                        top_k=top_k,
                        gate_dim=gate_dim,
                    )
                    for _ in range(num_layers)
                ])
            else:
                self.net_transformer = None
                
        elif attention_type == "self_attention":
            self.inst_attention = MultiHeadSelfAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
            if apply_to_nets:
                self.net_attention = MultiHeadSelfAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
            else:
                self.net_attention = None
                
        elif attention_type == "local_windows":
            self.inst_local_windows = LocalWindowsTransformer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                ff_dim=ff_dim,
                dropout=dropout,
                activation=activation,
                use_layer_norm=use_layer_norm,
                window_size=window_size,
            )
            if apply_to_nets:
                self.net_local_windows = LocalWindowsTransformer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    activation=activation,
                    use_layer_norm=use_layer_norm,
                    window_size=window_size,
                )
            else:
                self.net_local_windows = None
                
        elif attention_type == "knn_local":
            self.inst_knn_attn = StableLocalAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                k_neighbors=window_size,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
            if apply_to_nets:
                self.net_knn_attn = StableLocalAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    k_neighbors=window_size,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
            else:
                self.net_knn_attn = None
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
        
        # Stability: LayerNorm after unbatch (scatter-back) to prevent residual accumulation
        # and decoder blow-up. Use 0.1--0.3 residual scale in caller (model.py uses 0.1).
        self.ln_after_unbatch = nn.LayerNorm(hidden_dim)
    
    def _pad_and_batch(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Pad variable-length sequences into a batch tensor (fully vectorized).
        
        CRITICAL: Output shape [B, max_nodes, D] ensures self-attention is STRICTLY
        WITHIN each graph only. Each batch index b holds one graph's nodes (padded).
        Attention computes [B, N, N] - no cross-graph attention; graphs are independent.
        
        Uses advanced indexing to avoid for-loops and preserve gradients.
        This is ~25× faster than the sequential approach for typical batch sizes.
        
        OPTIMIZATION: Minimizes CPU-GPU synchronization by computing batch_size
        and max_nodes together, reducing .item() calls.
        """
        device = x.device
        # OPTIMIZATION: Compute batch_size and max_nodes together to minimize CPU-GPU sync
        # Use a single .item() call by computing max_nodes from counts_tensor
        batch_max = batch.max()
        batch_size = int(batch_max.item()) + 1
        
        # Compute node counts per graph
        counts_tensor = torch.bincount(batch, minlength=batch_size)
        counts = counts_tensor.tolist()
        # OPTIMIZATION: max_nodes can be computed from counts_tensor without .item() if we keep it as tensor
        # But we need it as int for tensor creation, so we compute it here
        max_nodes = int(counts_tensor.max().item())
        
        # Compute cumulative counts to get start indices for each graph
        cumsum = torch.cat([torch.tensor([0], device=device), counts_tensor[:-1].cumsum(0)])
        
        # Compute position of each node within its graph (vectorized)
        # arange_all - cumsum[batch_idx] gives the within-graph position
        arange_all = torch.arange(x.shape[0], device=device)
        batch_positions = arange_all - cumsum[batch]
        
        # Create padded output tensor (zeros for padding)
        batched_x = torch.zeros(batch_size, max_nodes, x.shape[1], dtype=x.dtype, device=device)
        
        # Fill in real values using advanced indexing (preserves gradients)
        batched_x[batch, batch_positions] = x  # [batch_size, max_nodes, hidden_dim]
        
        # Create attention mask (True for real nodes, False for padding)
        mask = torch.zeros(batch_size, max_nodes, device=device, dtype=torch.bool)
        mask[batch, batch_positions] = True  # [batch_size, max_nodes]
        
        # INVARIANT: total valid tokens must equal flat node count (no padding in scatter source)
        num_valid = mask.sum().item()
        assert num_valid == x.shape[0], (
            f"_pad_and_batch: valid_mask.sum()={num_valid} != node_x.size(0)={x.shape[0]}; "
            "pack/unpack mapping may be wrong."
        )
        # Padding fill is 0 (never use large negative sentinel before residual add)
        return batched_x, mask, counts
    
    def _unbatch(
        self,
        batched_x: torch.Tensor,
        batch: torch.Tensor,
        counts: List[int],
    ) -> torch.Tensor:
        """Unbatch and concatenate back to original format (fully vectorized).
        Only valid (real) positions are read; padded positions are never scattered into output.
        """
        device = batched_x.device
        batch_size = len(counts)
        total_nodes = batch.shape[0]
        
        # INVARIANT: batch length must match sum of per-graph counts (total valid tokens)
        sum_counts = sum(counts)
        assert total_nodes == sum_counts, (
            f"_unbatch: batch.size(0)={total_nodes} != sum(counts)={sum_counts}; "
            "batch/counts mismatch."
        )
        assert batched_x.shape[0] == batch_size and batched_x.shape[1] >= max(counts), (
            f"_unbatch: batched_x shape {tuple(batched_x.shape)} inconsistent with counts (max={max(counts)})."
        )
        
        # Compute cumulative counts to get start indices
        counts_tensor = torch.tensor(counts, device=device)
        cumsum = torch.cat([torch.tensor([0], device=device), counts_tensor[:-1].cumsum(0)])
        
        # Compute position of each node within its graph (vectorized)
        arange_all = torch.arange(total_nodes, device=device)
        batch_positions = arange_all - cumsum[batch]
        
        # Extract real values using advanced indexing (only valid positions; no padded tokens)
        x = batched_x[batch, batch_positions]  # [total_nodes, hidden_dim]
        
        return x
    
    def _compute_sparse_rel_pos_bias(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        distance_type: str = 'l2',
    ) -> torch.Tensor:
        """
        Efficiently compute sparse relative position bias only for edges that exist.
        
        Complexity: O(E) instead of O(N²) - much cheaper for sparse graphs!
        
        Args:
            pos: [N_total, 2] node positions (normalized coordinates)
            edge_index: [2, E] edge indices (src, dst)
            batch: [N_total] batch assignment (optional, for batched graphs)
            distance_type: 'l1' or 'l2' distance metric
            
        Returns:
            edge_bias: [E] bias values for each edge
        """
        if edge_index.shape[1] == 0:
            # No edges - return empty bias
            return torch.zeros(0, device=pos.device, dtype=pos.dtype)
        
        # Extract source and destination positions
        src_pos = pos[edge_index[0]]  # [E, 2]
        dst_pos = pos[edge_index[1]]  # [E, 2]
        
        # Compute relative position (difference)
        rel_pos = src_pos - dst_pos  # [E, 2]
        
        # Compute distance (L1 or L2)
        if distance_type == 'l1':
            distances = torch.abs(rel_pos).sum(dim=-1)  # [E] L1 distance
        elif distance_type == 'l2':
            distances = torch.norm(rel_pos, p=2, dim=-1)  # [E] L2 distance
        else:
            raise ValueError(f"Unknown distance_type: {distance_type}. Use 'l1' or 'l2'")
        
        # Normalize distances (optional - helps with training stability)
        # Use log(1 + distance) to compress large distances
        distances_normalized = torch.log(1.0 + distances)
        
        # Project through learnable MLP if enabled
        if self.rel_pos_bias_learnable and self.rel_pos_bias_proj is not None:
            # [E, 1] -> [E, 1] -> [E]
            bias_values = self.rel_pos_bias_proj(distances_normalized.unsqueeze(-1)).squeeze(-1)
        else:
            # Simple negative distance (closer = higher bias)
            bias_values = -distances_normalized
        
        return bias_values
    
    def _create_dense_bias_from_sparse(
        self,
        edge_index: torch.Tensor,
        edge_bias: torch.Tensor,
        num_nodes: int,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert sparse edge bias to dense [N, N] bias matrix.
        Fully vectorized for batched case (no per-graph loops).

        Args:
            edge_index: [2, E] edge indices (global indices)
            edge_bias: [E] bias values for edges
            num_nodes: Number of nodes N (for single graph) or max_nodes_per_graph (for batched)
            batch: [N_total] batch assignment (optional, for batched graphs)

        Returns:
            bias_matrix: [N, N] dense bias matrix (single graph) or [B, max_nodes, max_nodes] (batched)
        """
        device = edge_bias.device
        dtype = edge_bias.dtype

        if batch is None:
            # Single graph: [N, N]
            bias_matrix = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
            if edge_index.shape[1] > 0:
                bias_matrix[edge_index[0], edge_index[1]] = edge_bias
        else:
            # Batched: fully vectorized (no for-loop over batch)
            batch_size = int(batch.max().item()) + 1
            max_nodes = num_nodes

            # Compute node counts and cumulative offsets per graph
            counts_tensor = torch.bincount(batch, minlength=batch_size)
            cumsum = torch.cat([torch.tensor([0], device=device, dtype=torch.long), counts_tensor[:-1].cumsum(0)])

            # Get batch index for each edge's src/dst nodes
            edge_batch = batch[edge_index[0]]  # [E] - all edges in same graph have same batch
            # Local indices: global_idx - cumsum[batch_of_node]
            local_src = edge_index[0] - cumsum[edge_batch]
            local_dst = edge_index[1] - cumsum[edge_batch]

            # Clamp to valid range (edges should be intra-graph, but safety check)
            local_src = torch.clamp(local_src, 0, max_nodes - 1)
            local_dst = torch.clamp(local_dst, 0, max_nodes - 1)

            # Linear index into [B, max_nodes, max_nodes]: b * max_nodes^2 + src * max_nodes + dst
            linear_idx = edge_batch * (max_nodes * max_nodes) + local_src * max_nodes + local_dst

            bias_matrix = torch.zeros(batch_size * max_nodes * max_nodes, device=device, dtype=dtype)
            bias_matrix.scatter_(0, linear_idx, edge_bias)
            bias_matrix = bias_matrix.view(batch_size, max_nodes, max_nodes)

        return bias_matrix
    
    def forward(
        self,
        inst_x: torch.Tensor,
        net_x: torch.Tensor,
        inst_batch: Optional[torch.Tensor] = None,
        net_batch: Optional[torch.Tensor] = None,
        inst_rel_pos_bias: Optional[torch.Tensor] = None,
        net_rel_pos_bias: Optional[torch.Tensor] = None,
        inst_pos: Optional[torch.Tensor] = None,
        net_pos: Optional[torch.Tensor] = None,
        data: Optional[HeteroData] = None,
        time_embed: Optional[torch.Tensor] = None,
        step_idx: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply per-graph global attention with Flash Attention (fully parallelized).
        
        Args:
            time_embed: [N_inst_total, time_embed_dim] time embeddings (optional, for diffusion)
            step_idx: Optional integer timestep index (unused for attention, kept for interface consistency)
        
        Returns:
            inst_output: [N_inst_total, hidden_dim] updated instance features
            net_output: [N_net_total, hidden_dim] updated net features (unchanged if apply_to_nets=False)
        """
        # Save inputs for delta computation
        inst_input = inst_x
        net_input = net_x

        # CRITICAL: Inject time embeddings into inst_x if provided
        # This ensures attention modules have access to diffusion timestep information
        # Use FiLM for more expressive time conditioning (per-feature scaling/shifting)
        if self.time_film is not None and time_embed is not None:
            inst_x = self.time_film(inst_x, time_embed)
        
        # CRITICAL: Encode positional information (inst_pos) if provided
        # inst_pos contains noisy coordinates during diffusion - critical spatial information
        # This allows the attention module to understand spatial relationships between nodes
        if self.use_pos_encoding and inst_pos is not None:
            # inst_pos: [N_inst_total, 2] - normalized coordinates in [-1, 1] range
            pos_encoding = self.pos_encoder(inst_pos)  # [N_inst_total, hidden_dim]
            inst_x = inst_x + pos_encoding
        
        # Compute sparse relative position bias if enabled
        inst_rel_pos_bias_dense = None
        if self.use_rel_pos_bias and inst_pos is not None:
            # Extract edge_index from data
            if data is not None:
                if hasattr(data, 'edge_index'):
                    edge_index = data.edge_index
                elif hasattr(data, '__contains__') and 'inst' in data:
                    # Heterogeneous data
                    if hasattr(data['inst', 'to', 'inst'], 'edge_index'):
                        edge_index = data['inst', 'to', 'inst'].edge_index
                    else:
                        edge_index = None
                else:
                    edge_index = None
            else:
                edge_index = None
            
            if edge_index is not None and edge_index.shape[1] > 0:
                # Compute sparse bias: O(E) complexity
                edge_bias = self._compute_sparse_rel_pos_bias(
                    pos=inst_pos,
                    edge_index=edge_index,
                    batch=inst_batch,
                    distance_type=self.rel_pos_bias_type,
                )
                
                # Convert to dense bias matrix for attention
                # For batched case, we'll create bias per graph after padding
                if inst_batch is not None:
                    # Will create batched bias matrix in _pad_and_batch_with_bias
                    inst_rel_pos_bias_dense = (edge_index, edge_bias)  # Store sparse for later
                else:
                    # Single graph: create dense [N, N] matrix
                    num_nodes = inst_x.shape[0]
                    inst_rel_pos_bias_dense = self._create_dense_bias_from_sparse(
                        edge_index=edge_index,
                        edge_bias=edge_bias,
                        num_nodes=num_nodes,
                        batch=None,
                    )
        
        # Process instances
        if self.attention_type == "local_windows":
            if inst_pos is None:
                raise ValueError("inst_pos is required for local_windows attention")
            
            if inst_batch is not None:
                batch_size = inst_batch.max().item() + 1
                N_total = inst_x.shape[0]
                
                counts_per_graph = torch.bincount(inst_batch, minlength=batch_size)
                max_nodes_per_graph = counts_per_graph.max().item()
                grid_size = self.inst_local_windows._determine_grid_size(max_nodes_per_graph)
                num_windows_per_graph = grid_size * grid_size
                total_windows = batch_size * num_windows_per_graph
                
                window_ids = self.inst_local_windows._assign_to_windows_batched(
                    inst_pos, inst_batch, grid_size
                )
                
                window_x, window_mask, node_to_window_slot = self.inst_local_windows._prepare_window_batches(
                    inst_x, window_ids, total_windows
                )
                
                if self.inst_local_windows.use_ffn:
                    for transformer_block in self.inst_local_windows.transformer_blocks:
                        window_x = transformer_block(window_x, mask=window_mask)
                else:
                    for attention_block in self.inst_local_windows.attention_blocks:
                        window_x = attention_block(window_x, mask=window_mask)
                
                inst_x = self.inst_local_windows._scatter_back(
                    window_x, window_ids, node_to_window_slot, 
                    (N_total, self.hidden_dim)
                )
            else:
                inst_x = self.inst_local_windows(inst_x, inst_pos)
                
        elif self.attention_type == "knn_local":
            if inst_pos is None:
                raise ValueError("inst_pos is required for knn_local attention")
            inst_x = self.inst_knn_attn(
                inst_x,
                inst_pos,
                batch=inst_batch,
                k_neighbors=self.inst_knn_attn.k_neighbors,
            )
        else:
            # Standard transformer or self_attention
            if inst_batch is not None:
                # OPTIMIZATION: Use batched processing with padding for better GPU utilization
                # The vectorized _pad_and_batch method is ~25× faster than sequential loops
                # Even with padding overhead, batched attention is more efficient for typical batch sizes
                # OPTIMIZATION: Avoid .item() call by computing batch_size in _pad_and_batch
                # batch_size is computed inside _pad_and_batch to avoid CPU-GPU sync
                
                # Per-graph self-attention only (no cross-graph). _pad_and_batch produces
                # [B, N, D] so attention [B, N, N] is strictly within each graph.
                inst_batched, inst_mask, inst_counts = self._pad_and_batch(inst_x, inst_batch)
                
                # Convert sparse bias to batched dense bias if needed
                batched_bias = None
                if inst_rel_pos_bias_dense is not None and isinstance(inst_rel_pos_bias_dense, tuple):
                    # Sparse bias needs to be converted to batched dense
                    edge_index, edge_bias = inst_rel_pos_bias_dense
                    max_nodes = inst_batched.shape[1]  # max_nodes_per_graph after padding
                    batched_bias = self._create_dense_bias_from_sparse(
                        edge_index=edge_index,
                        edge_bias=edge_bias,
                        num_nodes=max_nodes,  # max_nodes_per_graph
                        batch=inst_batch,
                    )
                elif inst_rel_pos_bias_dense is not None:
                    # Already dense (single graph case) - shouldn't happen in batched path
                    # But handle it gracefully by expanding to batch dimension
                    if len(inst_rel_pos_bias_dense.shape) == 2:
                        # [N, N] -> [1, N, N]
                        batched_bias = inst_rel_pos_bias_dense.unsqueeze(0)
                    else:
                        batched_bias = inst_rel_pos_bias_dense
                
                # Apply attention with proper masking and bias
                if self.attention_type == "transformer":
                    for inst_block in self.inst_transformer:
                        inst_batched = inst_block(inst_batched, mask=inst_mask, rel_pos_bias=batched_bias)
                else:  # self_attention
                    inst_batched = self.inst_attention(inst_batched, mask=inst_mask, rel_pos_bias=batched_bias)
                
                # Unbatch back to original format (only valid tokens; no padded positions)
                inst_x = self._unbatch(inst_batched, inst_batch, inst_counts)
                # Stability: LayerNorm after scatter-back to bound magnitude before residual
                inst_x = self.ln_after_unbatch(inst_x)
            else:
                # Single graph (no batching)
                if self.attention_type == "transformer":
                    for inst_block in self.inst_transformer:
                        inst_x = inst_block(inst_x, rel_pos_bias=inst_rel_pos_bias_dense)
                else:  # self_attention
                    inst_x = self.inst_attention(inst_x, rel_pos_bias=inst_rel_pos_bias_dense)
                inst_x = self.ln_after_unbatch(inst_x)
        
        # Process nets if enabled
        if self.apply_to_nets:
            if self.attention_type == "local_windows":
                if net_pos is None:
                    raise ValueError("net_pos is required for local_windows attention with apply_to_nets=True")
                
                if net_batch is not None:
                    batch_size = net_batch.max().item() + 1
                    N_total = net_x.shape[0]
                    
                    counts_per_graph = torch.bincount(net_batch, minlength=batch_size)
                    max_nodes_per_graph = counts_per_graph.max().item()
                    grid_size = self.net_local_windows._determine_grid_size(max_nodes_per_graph)
                    num_windows_per_graph = grid_size * grid_size
                    total_windows = batch_size * num_windows_per_graph
                    
                    window_ids = self.net_local_windows._assign_to_windows_batched(
                        net_pos, net_batch, grid_size
                    )
                    
                    window_x, window_mask, node_to_window_slot = self.net_local_windows._prepare_window_batches(
                        net_x, window_ids, total_windows
                    )
                    
                    if self.net_local_windows.use_ffn:
                        for transformer_block in self.net_local_windows.transformer_blocks:
                            window_x = transformer_block(window_x, mask=window_mask)
                    else:
                        for attention_block in self.net_local_windows.attention_blocks:
                            window_x = attention_block(window_x, mask=window_mask)
                    
                    net_x = self.net_local_windows._scatter_back(
                        window_x, window_ids, node_to_window_slot, 
                        (N_total, self.hidden_dim)
                    )
                else:
                    net_x = self.net_local_windows(net_x, net_pos)
                    
            elif self.attention_type == "knn_local":
                if net_pos is None:
                    raise ValueError("net_pos is required for knn_local attention with apply_to_nets=True")
                net_x = self.net_knn_attn(
                    net_x,
                    net_pos,
                    batch=net_batch,
                    k_neighbors=self.net_knn_attn.k_neighbors,
                )
            else:
                if net_batch is not None:
                    net_batched, net_mask, net_counts = self._pad_and_batch(net_x, net_batch)
                    
                    if self.attention_type == "transformer":
                        for net_block in self.net_transformer:
                            net_batched = net_block(net_batched, mask=net_mask)
                    else:  # self_attention
                        net_batched = self.net_attention(net_batched, mask=net_mask)
                    
                    net_x = self._unbatch(net_batched, net_batch, net_counts)
                    net_x = self.ln_after_unbatch(net_x)
                else:
                    if self.attention_type == "transformer":
                        for net_block in self.net_transformer:
                            net_x = net_block(net_x)
                    else:  # self_attention
                        net_x = self.net_attention(net_x)
        
        # Return deltas (FullInstNetModel applies residual add)
        inst_delta = inst_x - inst_input
        net_delta = net_x - net_input if net_input is not None else inst_delta
        return inst_delta, net_delta


@register_global_module("hash_grid")
class HashGridGlobalModule(BaseGlobalModule):
    """
    Hash-grid hierarchical multi-scale attention global module.
    
    Uses multi-level hash grids with windowed attention to provide
    efficient multi-scale global context.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        levels: int = 2,
        base_cell_size: Optional[float] = None,
        cell_size_multiplier: Optional[float] = None,
        grid_dim: int = 64,
        window_sizes: Optional[list] = None,
        shifted: bool = True,
        ff_num_layers: int = 1,
        ff_size_factor: int = 2,
        dropout: float = 0.0,
        att_implementation: str = "default",
        rehash_interval: int = 1,
        target_occupancy: float = 4.0,
        target_graph_size: int = 1000,
        num_attention_layers: int = 1,
        coord_min: float = -1.0,
        coord_max: float = 1.0,
        clamp_coords_to_canvas: bool = True,
        use_fixed_canvas_range: bool = True,
        use_time_level_mixing: bool = True,
        level_mixing_power: float = 1.0,
        min_level_weight: float = 0.05,
        **kwargs
    ):
        super().__init__(hidden_dim, **kwargs)
        
        window_sizes = window_sizes or [9, 9]
        
        # Create hash grid hierarchy for instances
        self.inst_hash_grid = HashedGridHierarchy(
            hidden_dim=hidden_dim,
            levels=levels,
            base_cell_size=base_cell_size,
            cell_size_multiplier=cell_size_multiplier,
            grid_dim=grid_dim,
            window_sizes=window_sizes,
            shifted=shifted,
            ff_num_layers=ff_num_layers,
            ff_size_factor=ff_size_factor,
            dropout=dropout,
            att_implementation=att_implementation,
            rehash_interval=rehash_interval,
            target_occupancy=target_occupancy,
            target_graph_size=target_graph_size,
            num_attention_layers=num_attention_layers,
            coord_min=coord_min,
            coord_max=coord_max,
            clamp_coords_to_canvas=clamp_coords_to_canvas,
            use_fixed_canvas_range=use_fixed_canvas_range,
            use_time_level_mixing=use_time_level_mixing,
            level_mixing_power=level_mixing_power,
            min_level_weight=min_level_weight,
        )
        
        # Create hash grid hierarchy for nets
        self.net_hash_grid = HashedGridHierarchy(
            hidden_dim=hidden_dim,
            levels=levels,
            base_cell_size=base_cell_size,
            cell_size_multiplier=cell_size_multiplier,
            grid_dim=grid_dim,
            window_sizes=window_sizes,
            shifted=shifted,
            ff_num_layers=ff_num_layers,
            ff_size_factor=ff_size_factor,
            dropout=dropout,
            att_implementation=att_implementation,
            rehash_interval=rehash_interval,
            target_occupancy=target_occupancy,
            target_graph_size=target_graph_size,
            num_attention_layers=num_attention_layers,
            coord_min=coord_min,
            coord_max=coord_max,
            clamp_coords_to_canvas=clamp_coords_to_canvas,
            use_fixed_canvas_range=use_fixed_canvas_range,
            use_time_level_mixing=use_time_level_mixing,
            level_mixing_power=level_mixing_power,
            min_level_weight=min_level_weight,
        )

    @staticmethod
    def _sizes_to_areas(sizes: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Convert [N,2] sizes to [N] areas with validation."""
        if sizes is None or not isinstance(sizes, torch.Tensor):
            return None
        if sizes.numel() == 0 or sizes.dim() < 2 or sizes.shape[-1] < 2:
            return None
        sizes = sizes.to(device=device, dtype=dtype)
        if not torch.isfinite(sizes).all():
            return None
        sizes = sizes.clamp(min=1e-8)
        return (sizes[:, 0] * sizes[:, 1]).clamp(min=0.0)

    @staticmethod
    def _extract_node_areas(
        data_obj,
        fallback_num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Extract per-node areas from runtime data, on-the-fly.
        Falls back to ones when explicit sizes are unavailable.
        """
        # Preferred: explicit sizes tensors.
        for attr in ("size", "instance_sizes", "sizes"):
            if hasattr(data_obj, attr):
                areas = HashGridGlobalModule._sizes_to_areas(getattr(data_obj, attr), device, dtype)
                if areas is not None:
                    return areas

        # Fallback from node features:
        # For chipgen diffusion features, x[:,2:4] are often log-relative w/h.
        if hasattr(data_obj, "x") and isinstance(data_obj.x, torch.Tensor) and data_obj.x.shape[1] >= 4:
            x = data_obj.x.to(device=device, dtype=dtype)
            w_rel = x[:, 2]
            h_rel = x[:, 3]
            # Area proxy in physical-ish scale (row-height squared factor omitted intentionally;
            # relative area is sufficient for bin-stat conditioning).
            areas = torch.exp(w_rel + h_rel).clamp(min=1e-8, max=1e8)
            if torch.isfinite(areas).all():
                return areas

        return torch.ones(fallback_num_nodes, device=device, dtype=dtype)

    @staticmethod
    def _assert_normalized_coords(
        coords: Optional[torch.Tensor],
        name: str,
        tol: float = 1.1,
    ) -> None:
        """Fail fast if coordinates are not normalized to model space."""
        if coords is None or not isinstance(coords, torch.Tensor) or coords.numel() == 0:
            return
        if not torch.isfinite(coords).all():
            raise ValueError(f"{name} contains NaN/Inf; expected normalized finite coordinates in [-1,1].")
        cmin = coords.min().item()
        cmax = coords.max().item()
        if cmin < -tol or cmax > tol:
            raise ValueError(
                f"{name} appears unnormalized (range [{cmin:.4f}, {cmax:.4f}] exceeds [-{tol}, {tol}]). "
                "Hash-grid global module expects normalized coordinates only."
            )
    
    def forward(
        self,
        inst_x: torch.Tensor,
        net_x: torch.Tensor,
        inst_batch: Optional[torch.Tensor] = None,
        net_batch: Optional[torch.Tensor] = None,
        inst_rel_pos_bias: Optional[torch.Tensor] = None,
        net_rel_pos_bias: Optional[torch.Tensor] = None,
        inst_pos: Optional[torch.Tensor] = None,
        net_pos: Optional[torch.Tensor] = None,
        data: Optional[HeteroData] = None,
        step_idx: Optional[int] = None,
        t_continuous: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process instance and net features using hash-grid multi-scale attention."""
        # Use runtime positions if provided (e.g., diffusion x_t); fallback to stored graph pos.
        inst_pos_for_grid = inst_pos
        if inst_pos_for_grid is None and data is not None and "inst" in data and hasattr(data["inst"], "pos"):
            inst_pos_for_grid = data["inst"].pos
        if inst_pos_for_grid is None:
            raise ValueError("HashGridGlobalModule requires inst_pos or data['inst'].pos")
        self._assert_normalized_coords(inst_pos_for_grid, name="inst_pos_for_grid", tol=1.1)
        
        net_pos_for_grid = net_pos
        if net_pos_for_grid is None and data is not None and "net" in data and hasattr(data["net"], "pos"):
            net_pos_for_grid = data["net"].pos
        self._assert_normalized_coords(net_pos_for_grid, name="net_pos_for_grid", tol=1.1)

        is_homogeneous = (net_x is None or net_pos_for_grid is None)

        # Build per-node area signals for bin statistics (on-the-fly, no precompute).
        inst_areas = None
        if data is not None and "inst" in data:
            inst_areas = self._extract_node_areas(
                data["inst"],
                fallback_num_nodes=inst_x.shape[0],
                device=inst_x.device,
                dtype=inst_x.dtype,
            )
        elif data is not None:
            inst_areas = self._extract_node_areas(
                data,
                fallback_num_nodes=inst_x.shape[0],
                device=inst_x.device,
                dtype=inst_x.dtype,
            )
        if inst_areas is None:
            inst_areas = torch.ones(inst_x.shape[0], device=inst_x.device, dtype=inst_x.dtype)
        
        # Process instances in one flat batched pass (avoids per-graph Python loop).
        inst_context = self.inst_hash_grid(
            node_feats=inst_x,
            coords=inst_pos_for_grid,
            node_areas=inst_areas,
            batch=inst_batch,
            step_idx=step_idx,
            t_continuous=t_continuous,
        )

        if is_homogeneous:
            return inst_context, inst_context
        
        # Process nets in one flat batched pass (avoids per-graph Python loop).
        net_areas = torch.ones(net_x.shape[0], device=net_x.device, dtype=net_x.dtype)
        net_context = self.net_hash_grid(
            node_feats=net_x,
            coords=net_pos_for_grid,
            node_areas=net_areas,
            batch=net_batch,
            step_idx=step_idx,
            t_continuous=t_continuous,
        )
        
        return inst_context, net_context


@register_global_module("perceiver")
class PerceiverGlobalModule(BaseGlobalModule):
    """
    Perceiver-based global module with efficient cross-attention pattern.
    
    Architecture:
    1. Cross-attention: N instances -> K global tokens (cost: NK)
    2. Self-attention: K global tokens (cost: K^2)
    3. Cross-attention: K global tokens -> N instances (cost: NK)
    
    Total cost: O(K^2 + NK) instead of O(N^2) for full self-attention.
    This is efficient when K << N.
    
    Args:
        hidden_dim: Hidden dimension
        num_global_tokens: Number of global tokens K (default: 64)
        num_heads: Number of attention heads
        num_self_attn_layers: Number of self-attention layers on global tokens (default: 1)
        dropout: Dropout rate
        use_layer_norm: Whether to use layer normalization
        apply_to_nets: If True, also process nets (memory intensive)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_global_tokens: int = 64,
        num_heads: int = 8,
        num_self_attn_layers: int = 1,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        apply_to_nets: bool = False,
        **kwargs
    ):
        super().__init__(hidden_dim, **kwargs)
        self.num_global_tokens = num_global_tokens
        self.num_self_attn_layers = num_self_attn_layers
        self.apply_to_nets = apply_to_nets
        
        # Learnable global tokens (latent queries)
        # Shape: [1, num_global_tokens, hidden_dim] - will be broadcasted per batch
        self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, hidden_dim))
        nn.init.normal_(self.global_tokens, std=0.02)
        
        # Cross-attention: instances -> global tokens
        self.inst_to_global_cross_attn = MultiHeadCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        
        # Self-attention on global tokens (can have multiple layers)
        if num_self_attn_layers > 0:
            self.global_self_attn = nn.ModuleList([
                MultiHeadSelfAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_self_attn_layers)
            ])
        else:
            self.global_self_attn = None
        
        # Cross-attention: global tokens -> instances
        self.global_to_inst_cross_attn = MultiHeadCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        
        # Optional: process nets as well
        if apply_to_nets:
            self.net_to_global_cross_attn = MultiHeadCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
            self.global_to_net_cross_attn = MultiHeadCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
        else:
            self.net_to_global_cross_attn = None
            self.global_to_net_cross_attn = None
    
    def forward(
        self,
        inst_x: torch.Tensor,
        net_x: torch.Tensor,
        inst_batch: Optional[torch.Tensor] = None,
        net_batch: Optional[torch.Tensor] = None,
        inst_rel_pos_bias: Optional[torch.Tensor] = None,
        net_rel_pos_bias: Optional[torch.Tensor] = None,
        inst_pos: Optional[torch.Tensor] = None,
        net_pos: Optional[torch.Tensor] = None,
        data: Optional[HeteroData] = None,
        time_embed: Optional[torch.Tensor] = None,
        step_idx: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Perceiver architecture.
        
        Args:
            inst_x: [N_inst_total, hidden_dim] instance features
            net_x: [N_net_total, hidden_dim] net features
            inst_batch: [N_inst_total] batch assignment for instances
            net_batch: [N_net_total] batch assignment for nets
            time_embed: Optional time embeddings (for diffusion)
            step_idx: Optional step index (unused)
            
        Returns:
            inst_delta: [N_inst_total, hidden_dim] instance feature deltas
            net_delta: [N_net_total, hidden_dim] net feature deltas
        """
        inst_input = inst_x
        net_input = net_x
        
        # Determine batch size
        if inst_batch is not None:
            batch_size = inst_batch.max().item() + 1
        else:
            batch_size = 1
        
        # Expand global tokens for batch: [1, K, D] -> [B, K, D]
        global_tokens = self.global_tokens.expand(batch_size, -1, -1)
        
        # Process instances
        if inst_batch is not None:
            # OPTIMIZATION: Batch cross-attention operations for better GPU utilization
            # Pad instances to max_nodes_per_graph and process all graphs in parallel
            device = inst_x.device
            counts_tensor = torch.bincount(inst_batch, minlength=batch_size)
            max_nodes = counts_tensor.max().item()
            
            # Pad and batch instances
            cumsum = torch.cat([torch.tensor([0], device=device), counts_tensor[:-1].cumsum(0)])
            arange_all = torch.arange(inst_x.shape[0], device=device)
            batch_positions = arange_all - cumsum[inst_batch]
            
            inst_x_batched = torch.zeros(batch_size, max_nodes, inst_x.shape[1], 
                                         dtype=inst_x.dtype, device=device)
            inst_mask = torch.zeros(batch_size, max_nodes, device=device, dtype=torch.bool)
            inst_x_batched[inst_batch, batch_positions] = inst_x
            inst_mask[inst_batch, batch_positions] = True
            
            # Step 1: Cross-attention: instances -> global tokens (batched)
            # Query: global_tokens [B, K, D], Key/Value: inst_x_batched [B, N_max, D]
            global_tokens = self.inst_to_global_cross_attn(
                query=global_tokens,
                key_value=inst_x_batched,
                kv_mask=inst_mask,
            )  # [B, K, hidden_dim]
            
            # Step 2: Self-attention on global tokens (batched)
            if self.global_self_attn is not None:
                for self_attn_layer in self.global_self_attn:
                    global_tokens = self_attn_layer(global_tokens)  # [B, K, hidden_dim]
            
            # Step 3: Cross-attention: global tokens -> instances (batched)
            # Query: inst_x_batched [B, N_max, D], Key/Value: global_tokens [B, K, D]
            # No query_mask: padded node outputs are discarded in the unbatch step below.
            inst_x_batched = self.global_to_inst_cross_attn(
                query=inst_x_batched,
                key_value=global_tokens,
            )  # [B, N_max, hidden_dim]

            # Unbatch: extract real values using advanced indexing
            inst_x = inst_x_batched[inst_batch, batch_positions]  # [total_nodes, hidden_dim]
        else:
            # Single graph, no batching needed
            inst_x_batched = inst_x.unsqueeze(0) if len(inst_x.shape) == 2 else inst_x
            
            # Step 1: Cross-attention: instances -> global tokens
            global_tokens = self.inst_to_global_cross_attn(
                query=global_tokens,
                key_value=inst_x_batched,
            )
            
            # Step 2: Self-attention on global tokens
            if self.global_self_attn is not None:
                for self_attn_layer in self.global_self_attn:
                    global_tokens = self_attn_layer(global_tokens)
            
            # Step 3: Cross-attention: global tokens -> instances
            inst_x = self.global_to_inst_cross_attn(
                query=inst_x_batched,
                key_value=global_tokens,
            )
            
            if len(inst_x.shape) == 3:
                inst_x = inst_x.squeeze(0)
        
        # Process nets if enabled
        if self.apply_to_nets and net_x is not None:
            if net_batch is not None:
                # OPTIMIZATION: Batch cross-attention operations for nets
                device = net_x.device
                net_counts_tensor = torch.bincount(net_batch, minlength=batch_size)
                max_nets = net_counts_tensor.max().item()
                
                # Pad and batch nets
                net_cumsum = torch.cat([torch.tensor([0], device=device), net_counts_tensor[:-1].cumsum(0)])
                net_arange_all = torch.arange(net_x.shape[0], device=device)
                net_batch_positions = net_arange_all - net_cumsum[net_batch]
                
                net_x_batched = torch.zeros(batch_size, max_nets, net_x.shape[1], 
                                           dtype=net_x.dtype, device=device)
                net_mask = torch.zeros(batch_size, max_nets, device=device, dtype=torch.bool)
                net_x_batched[net_batch, net_batch_positions] = net_x
                net_mask[net_batch, net_batch_positions] = True
                
                # Cross-attention: nets -> global tokens (batched)
                global_tokens = self.net_to_global_cross_attn(
                    query=global_tokens,
                    key_value=net_x_batched,
                    kv_mask=net_mask,
                )
                
                # Self-attention on global tokens (batched)
                if self.global_self_attn is not None:
                    for self_attn_layer in self.global_self_attn:
                        global_tokens = self_attn_layer(global_tokens)
                
                # Cross-attention: global tokens -> nets (batched)
                net_x_batched = self.global_to_net_cross_attn(
                    query=net_x_batched,
                    key_value=global_tokens,
                    query_mask=net_mask,
                )
                
                # Unbatch: extract real values
                net_x = net_x_batched[net_batch, net_batch_positions]
            else:
                net_x_batched = net_x.unsqueeze(0) if len(net_x.shape) == 2 else net_x
                
                # Cross-attention: nets -> global tokens
                global_tokens = self.net_to_global_cross_attn(
                    query=global_tokens,
                    key_value=net_x_batched,
                )
                
                # Self-attention on global tokens
                if self.global_self_attn is not None:
                    for self_attn_layer in self.global_self_attn:
                        global_tokens = self_attn_layer(global_tokens)
                
                # Cross-attention: global tokens -> nets
                net_x = self.global_to_net_cross_attn(
                    query=net_x_batched,
                    key_value=global_tokens,
                )
                
                if len(net_x.shape) == 3:
                    net_x = net_x.squeeze(0)
        
        # Return deltas (FullInstNetModel applies residual add)
        inst_delta = inst_x - inst_input
        net_delta = net_x - net_input if net_input is not None else inst_delta

        return inst_delta, net_delta


# ============================================================================
# Dynamic-Token Perceiver — plain perceiver with graph-size-dependent K
# ============================================================================

@register_global_module("dynamic_token_perceiver")
class DynamicTokenPerceiverGlobalModule(BaseGlobalModule):
    """
    Plain perceiver variant that scales the latent token count with graph size,
    without using any METIS-derived features or partition assignments.

    Token schedule:
        K = num_persistent_tokens + floor(N / nodes_per_scaling_token)

    Design:
      - The first ``num_persistent_tokens`` are always present and act as
        stable global collectors across all graph sizes.
      - The extra ``floor(N / nodes_per_scaling_token)`` tokens grow with the
        number of nodes. They are initialized from a shared learnable base plus
        a small learned projection of normalized token index to break symmetry,
        avoiding METIS-specific structure and avoiding untrained embedding rows.

    This isolates the "more tokens as N grows" hypothesis from the "METIS
    partition / topology features" hypothesis.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_persistent_tokens: int = 5,
        nodes_per_scaling_token: int = 5,
        num_heads: int = 8,
        num_self_attn_layers: int = 1,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        apply_to_nets: bool = False,
        **kwargs
    ):
        super().__init__(hidden_dim, **kwargs)
        if num_persistent_tokens <= 0:
            raise ValueError("num_persistent_tokens must be positive")
        if nodes_per_scaling_token <= 0:
            raise ValueError("nodes_per_scaling_token must be positive")

        self.num_persistent_tokens = num_persistent_tokens
        self.nodes_per_scaling_token = nodes_per_scaling_token
        self.num_self_attn_layers = num_self_attn_layers
        self.apply_to_nets = apply_to_nets

        self.persistent_tokens = nn.Parameter(
            torch.randn(1, num_persistent_tokens, hidden_dim)
        )
        self.scaling_token_base = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.normal_(self.persistent_tokens, std=0.02)
        nn.init.normal_(self.scaling_token_base, std=0.02)

        # Learned projection from normalized token index in [0, 1] to token
        # embedding. This breaks symmetry among replicated scaling tokens while
        # staying agnostic to graph topology or METIS annotations.
        self.scaling_token_pos_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.inst_to_global_cross_attn = MultiHeadCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

        if num_self_attn_layers > 0:
            self.global_self_attn = nn.ModuleList([
                MultiHeadSelfAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_self_attn_layers)
            ])
        else:
            self.global_self_attn = None

        self.global_to_inst_cross_attn = MultiHeadCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

        if apply_to_nets:
            self.net_to_global_cross_attn = MultiHeadCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
            self.global_to_net_cross_attn = MultiHeadCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
        else:
            self.net_to_global_cross_attn = None
            self.global_to_net_cross_attn = None

    def _build_global_tokens(
        self,
        node_counts: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build padded latent tokens and token validity mask.

        Returns:
            tokens_batched: [B, max_K, D]
            token_mask: [B, max_K] with True for valid tokens
        """
        batch_size = int(node_counts.shape[0])
        extra_tokens_per_graph = torch.div(
            node_counts,
            self.nodes_per_scaling_token,
            rounding_mode="floor",
        )
        K_per_graph = extra_tokens_per_graph + self.num_persistent_tokens
        max_K = int(K_per_graph.max().item())

        tokens_batched = torch.zeros(
            batch_size,
            max_K,
            self.hidden_dim,
            dtype=dtype,
            device=device,
        )

        persistent = self.persistent_tokens.to(device=device, dtype=dtype)
        tokens_batched[:, :self.num_persistent_tokens, :] = persistent.expand(
            batch_size, -1, -1
        )

        token_mask = (
            torch.arange(max_K, device=device).unsqueeze(0)
            < K_per_graph.unsqueeze(1)
        )

        extra_slot_idx = (
            torch.arange(max_K, device=device) - self.num_persistent_tokens
        )
        valid_extra_mask = (
            extra_slot_idx.unsqueeze(0) >= 0
        ) & (
            extra_slot_idx.unsqueeze(0) < extra_tokens_per_graph.unsqueeze(1)
        )

        if bool(valid_extra_mask.any()):
            extra_slot_idx = extra_slot_idx.clamp(min=0)
            denom = (extra_tokens_per_graph - 1).clamp(min=1).to(dtype=dtype)
            norm_pos = extra_slot_idx.unsqueeze(0).to(dtype=dtype) / denom.unsqueeze(1)
            norm_pos = norm_pos * valid_extra_mask.to(dtype=dtype)

            extra_tokens = self.scaling_token_base.to(device=device, dtype=dtype)
            extra_tokens = extra_tokens.expand(batch_size, max_K, -1)
            extra_tokens = extra_tokens + self.scaling_token_pos_proj(norm_pos.unsqueeze(-1))
            extra_tokens = extra_tokens * valid_extra_mask.unsqueeze(-1).to(dtype=dtype)
            tokens_batched = tokens_batched + extra_tokens

        return tokens_batched, token_mask

    def forward(
        self,
        inst_x: torch.Tensor,
        net_x: torch.Tensor,
        inst_batch: Optional[torch.Tensor] = None,
        net_batch: Optional[torch.Tensor] = None,
        inst_rel_pos_bias: Optional[torch.Tensor] = None,
        net_rel_pos_bias: Optional[torch.Tensor] = None,
        inst_pos: Optional[torch.Tensor] = None,
        net_pos: Optional[torch.Tensor] = None,
        data: Optional[HeteroData] = None,
        time_embed: Optional[torch.Tensor] = None,
        step_idx: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inst_input = inst_x
        net_input = net_x
        device = inst_x.device
        dtype = inst_x.dtype

        if inst_batch is not None:
            batch_size = int(inst_batch.max().item()) + 1
        else:
            batch_size = 1
            inst_batch = torch.zeros(inst_x.shape[0], dtype=torch.long, device=device)

        node_counts = torch.bincount(inst_batch, minlength=batch_size)
        max_nodes = int(node_counts.max().item())
        cumsum = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            node_counts[:-1].cumsum(0),
        ])
        arange_all = torch.arange(inst_x.shape[0], device=device)
        batch_positions = arange_all - cumsum[inst_batch]

        inst_x_batched = torch.zeros(
            batch_size,
            max_nodes,
            inst_x.shape[1],
            dtype=dtype,
            device=device,
        )
        inst_mask = torch.zeros(batch_size, max_nodes, device=device, dtype=torch.bool)
        inst_x_batched[inst_batch, batch_positions] = inst_x
        inst_mask[inst_batch, batch_positions] = True

        global_tokens, token_mask = self._build_global_tokens(node_counts, device, dtype)
        has_token_pad = not bool(token_mask.all())
        inv_token_mask = (~token_mask).unsqueeze(-1) if has_token_pad else None

        global_tokens = self.inst_to_global_cross_attn(
            query=global_tokens,
            key_value=inst_x_batched,
            kv_mask=inst_mask,
            query_mask=token_mask,
        )

        if self.global_self_attn is not None:
            for self_attn_layer in self.global_self_attn:
                if inv_token_mask is not None:
                    global_tokens = global_tokens.masked_fill(inv_token_mask, 0.0)
                global_tokens = self_attn_layer(global_tokens, mask=token_mask)
                if inv_token_mask is not None:
                    global_tokens = global_tokens.masked_fill(inv_token_mask, 0.0)

        inst_x_batched = self.global_to_inst_cross_attn(
            query=inst_x_batched,
            key_value=global_tokens,
            kv_mask=token_mask,
        )
        inst_x = inst_x_batched[inst_batch, batch_positions]

        if self.apply_to_nets and net_x is not None:
            if net_batch is not None:
                net_counts_tensor = torch.bincount(net_batch, minlength=batch_size)
                max_nets = int(net_counts_tensor.max().item())
                net_cumsum = torch.cat([
                    torch.zeros(1, dtype=torch.long, device=device),
                    net_counts_tensor[:-1].cumsum(0),
                ])
                net_arange_all = torch.arange(net_x.shape[0], device=device)
                net_batch_positions = net_arange_all - net_cumsum[net_batch]

                net_x_batched = torch.zeros(
                    batch_size,
                    max_nets,
                    net_x.shape[1],
                    dtype=net_x.dtype,
                    device=device,
                )
                net_mask = torch.zeros(batch_size, max_nets, device=device, dtype=torch.bool)
                net_x_batched[net_batch, net_batch_positions] = net_x
                net_mask[net_batch, net_batch_positions] = True

                global_tokens = self.net_to_global_cross_attn(
                    query=global_tokens,
                    key_value=net_x_batched,
                    kv_mask=net_mask,
                    query_mask=token_mask,
                )

                if self.global_self_attn is not None:
                    for self_attn_layer in self.global_self_attn:
                        if inv_token_mask is not None:
                            global_tokens = global_tokens.masked_fill(inv_token_mask, 0.0)
                        global_tokens = self_attn_layer(global_tokens, mask=token_mask)
                        if inv_token_mask is not None:
                            global_tokens = global_tokens.masked_fill(inv_token_mask, 0.0)

                net_x_batched = self.global_to_net_cross_attn(
                    query=net_x_batched,
                    key_value=global_tokens,
                    query_mask=net_mask,
                    kv_mask=token_mask,
                )
                net_x = net_x_batched[net_batch, net_batch_positions]
            else:
                net_x_batched = net_x.unsqueeze(0) if len(net_x.shape) == 2 else net_x
                net_mask = torch.ones(
                    net_x_batched.shape[:2],
                    device=net_x_batched.device,
                    dtype=torch.bool,
                )

                global_tokens = self.net_to_global_cross_attn(
                    query=global_tokens,
                    key_value=net_x_batched,
                    kv_mask=net_mask,
                    query_mask=token_mask,
                )

                if self.global_self_attn is not None:
                    for self_attn_layer in self.global_self_attn:
                        if inv_token_mask is not None:
                            global_tokens = global_tokens.masked_fill(inv_token_mask, 0.0)
                        global_tokens = self_attn_layer(global_tokens, mask=token_mask)
                        if inv_token_mask is not None:
                            global_tokens = global_tokens.masked_fill(inv_token_mask, 0.0)

                net_x = self.global_to_net_cross_attn(
                    query=net_x_batched,
                    key_value=global_tokens,
                    kv_mask=token_mask,
                )

                if len(net_x.shape) == 3:
                    net_x = net_x.squeeze(0)

        inst_delta = inst_x - inst_input
        net_delta = net_x - net_input if net_input is not None else inst_delta
        return inst_delta, net_delta




# ============================================================================
# Bank Dynamic-Token Perceiver — dynamic K with an independent K_max-sized bank
# ============================================================================

@register_global_module("bank_dynamic_token_perceiver")
class BankDynamicTokenPerceiverGlobalModule(DynamicTokenPerceiverGlobalModule):
    """
    Dynamic-K perceiver whose K latent tokens are a prefix of an INDEPENDENT
    learnable bank of size ``max_global_tokens``.

    Token schedule (same as ``DynamicTokenPerceiverGlobalModule``):
        K(N) = num_persistent_tokens + floor(N / nodes_per_scaling_token),
        clipped to ``max_global_tokens``.

    Unlike ``DynamicTokenPerceiverGlobalModule``, every latent slot is an
    independent learnable vector — identical parameterization to the fixed-K
    ``PerceiverGlobalModule`` with ``num_global_tokens = max_global_tokens``.
    This isolates the "more tokens when N is larger" hypothesis from the
    "compressed 1-D manifold token parameterization" hypothesis, giving a
    clean A/B against the fixed-K perceiver:
      * fixed-K perceiver: always uses all K_max independent tokens.
      * bank dynamic-K: uses only the first K(N) of the same-style bank.
    Any performance gap is therefore attributable to dynamic K itself, not
    to the token parameterization.

    When a graph's required K exceeds ``max_global_tokens`` (e.g. N outside
    the training range), the module clips to ``max_global_tokens``; capacity
    is then identical to the fixed-K bank.

    All cross/self-attention layers and the batched-forward / masking logic
    are inherited from ``DynamicTokenPerceiverGlobalModule``; only
    ``_build_global_tokens`` differs.

    Args:
        hidden_dim: D (hidden dimension).
        num_persistent_tokens: minimum K used for the smallest graphs.
        nodes_per_scaling_token: K grows by one per this many nodes.
        max_global_tokens: K_max; size of the independent learnable bank.
        num_heads: attention heads.
        num_self_attn_layers: self-attention layers on latent tokens.
        dropout: dropout rate.
        use_layer_norm: layer-norm toggle inside attention blocks.
        apply_to_nets: also route the latent tokens through net features.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_persistent_tokens: int = 26,
        nodes_per_scaling_token: int = 13,
        max_global_tokens: int = 102,
        num_heads: int = 8,
        num_self_attn_layers: int = 1,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        apply_to_nets: bool = False,
        **kwargs,
    ):
        # Intentionally skip DynamicTokenPerceiverGlobalModule.__init__ so we
        # don't instantiate its compressed base+pos_proj parameters. We
        # rebuild the attention stack here and keep its inherited ``forward``
        # (which only references self._build_global_tokens and the attention
        # modules created below).
        BaseGlobalModule.__init__(self, hidden_dim, **kwargs)
        if num_persistent_tokens <= 0:
            raise ValueError("num_persistent_tokens must be positive")
        if nodes_per_scaling_token <= 0:
            raise ValueError("nodes_per_scaling_token must be positive")
        if max_global_tokens < num_persistent_tokens:
            raise ValueError(
                f"max_global_tokens ({max_global_tokens}) must be >= "
                f"num_persistent_tokens ({num_persistent_tokens})"
            )

        self.num_persistent_tokens = num_persistent_tokens
        self.nodes_per_scaling_token = nodes_per_scaling_token
        self.max_global_tokens = max_global_tokens
        self.num_self_attn_layers = num_self_attn_layers
        self.apply_to_nets = apply_to_nets

        # Independent bank of K_max learnable tokens. Same parameterization
        # as PerceiverGlobalModule.global_tokens.
        self.global_token_bank = nn.Parameter(
            torch.randn(1, max_global_tokens, hidden_dim)
        )
        nn.init.normal_(self.global_token_bank, std=0.02)

        self.inst_to_global_cross_attn = MultiHeadCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

        if num_self_attn_layers > 0:
            self.global_self_attn = nn.ModuleList([
                MultiHeadSelfAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_self_attn_layers)
            ])
        else:
            self.global_self_attn = None

        self.global_to_inst_cross_attn = MultiHeadCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

        if apply_to_nets:
            self.net_to_global_cross_attn = MultiHeadCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
            self.global_to_net_cross_attn = MultiHeadCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
        else:
            self.net_to_global_cross_attn = None
            self.global_to_net_cross_attn = None

    def _build_global_tokens(
        self,
        node_counts: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Slice the first K_i entries of the shared bank for each graph and
        pad to max_K across the batch.

        Returns:
            tokens_batched: [B, max_K, D]  (broadcast of bank[:, :max_K, :])
            token_mask:     [B, max_K]     True where j < K_i for graph i
        """
        batch_size = int(node_counts.shape[0])
        extra_tokens_per_graph = torch.div(
            node_counts,
            self.nodes_per_scaling_token,
            rounding_mode="floor",
        )
        K_per_graph = (extra_tokens_per_graph + self.num_persistent_tokens).clamp(
            max=self.max_global_tokens
        )
        max_K = int(K_per_graph.max().item())

        bank = self.global_token_bank.to(device=device, dtype=dtype)  # [1, K_max, D]
        tokens_batched = bank[:, :max_K, :].expand(batch_size, -1, -1).contiguous()

        token_mask = (
            torch.arange(max_K, device=device).unsqueeze(0)
            < K_per_graph.unsqueeze(1)
        )
        return tokens_batched, token_mask


# ============================================================================
# Spatial Perceiver (K_w x K_h grid, physical scale, optional distance penalty)
# ============================================================================

@register_global_module("spatial_perceiver")
class SpatialPerceiverGlobalModule(BaseGlobalModule):
    """
    Perceiver with K_w x K_h spatial grid tokens; grid size from physical scale (L* = 10 µm).
    No self-attention over N nodes; only cross-attn latents<-nodes, self-attn on latents, cross-attn nodes<-latents.
    Optional distance penalty (bias) in cross-attention; when disabled, Flash path is used.
    V1: instances only (net_delta = zero).
    """

    def __init__(
        self,
        hidden_dim: int,
        L_star_um: float = 10.0,
        num_heads: int = 8,
        num_self_attn_layers: int = 1,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_distance_penalty: bool = True,
        distance_penalty_lambda: float = 0.1,
        **kwargs
    ):
        super().__init__(hidden_dim, **kwargs)
        self.L_star_um = L_star_um
        self.use_distance_penalty = use_distance_penalty and (distance_penalty_lambda > 0)
        self.distance_penalty_lambda = distance_penalty_lambda if self.use_distance_penalty else 0.0

        # Single learnable latent embedding; replicated to K_w*K_h per graph
        self.latent_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.normal_(self.latent_embed, std=0.02)

        self.inst_to_global_cross_attn = MultiHeadCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        if num_self_attn_layers > 0:
            self.global_self_attn = nn.ModuleList([
                MultiHeadSelfAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_self_attn_layers)
            ])
        else:
            self.global_self_attn = None
        self.global_to_inst_cross_attn = MultiHeadCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    def _get_chip_size_per_graph(
        self,
        data: Optional[HeteroData],
        inst_batch: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Return chip_size [num_graphs, 2] (W, H in µm) or None."""
        if data is None:
            return None
        cs = getattr(data, "chip_size", None)
        if cs is None:
            return None
        cs = cs.to(device=device, dtype=dtype)
        if cs.dim() == 1 and cs.numel() >= 2:
            return cs.unsqueeze(0)  # single graph
        return cs  # [num_graphs, 2]

    def _grid_centers_normalized(self, K_w: int, K_h: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Cell centers in [-1, 1]^2. Returns [K_w*K_h, 2]."""
        i = torch.arange(K_w, device=device, dtype=dtype)
        j = torch.arange(K_h, device=device, dtype=dtype)
        cx = -1.0 + (i + 0.5) * (2.0 / max(K_w, 1))
        cy = -1.0 + (j + 0.5) * (2.0 / max(K_h, 1))
        grid = torch.stack(torch.meshgrid(cx, cy, indexing="ij"), dim=-1)  # [K_w, K_h, 2]
        return grid.reshape(-1, 2)  # [K_w*K_h, 2]

    def _d_phys_bias(
        self,
        node_pos: torch.Tensor,
        grid_centers: torch.Tensor,
        s: torch.Tensor,
        lambda_val: float,
    ) -> torch.Tensor:
        """Physical distance bias: d_phys(i,k) = |s * (node_i - center_k)|; bias = -lambda * d_phys.
        node_pos [N, 2], grid_centers [K, 2], s [2] or [1,2]. Returns [N, K] (for Q=nodes, KV=latents)."""
        # diff [N, K, 2]
        diff = node_pos.unsqueeze(1) - grid_centers.unsqueeze(0)
        if s.dim() == 1:
            s = s.view(1, 1, 2)
        d_phys = (s * diff).norm(dim=-1)  # [N, K]
        return -lambda_val * d_phys

    def forward(
        self,
        inst_x: torch.Tensor,
        net_x: torch.Tensor,
        inst_batch: Optional[torch.Tensor] = None,
        net_batch: Optional[torch.Tensor] = None,
        inst_rel_pos_bias: Optional[torch.Tensor] = None,
        net_rel_pos_bias: Optional[torch.Tensor] = None,
        inst_pos: Optional[torch.Tensor] = None,
        net_pos: Optional[torch.Tensor] = None,
        data: Optional[HeteroData] = None,
        time_embed: Optional[torch.Tensor] = None,
        step_idx: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # V1: instances only
        net_delta = torch.zeros_like(net_x) if net_x is not None else inst_x.new_zeros(inst_x.shape[0], inst_x.shape[1])

        device = inst_x.device
        dtype = inst_x.dtype
        chip_sizes = self._get_chip_size_per_graph(data, inst_batch, device, dtype)
        pos = inst_pos if inst_pos is not None else (data["inst"].pos if data is not None and "inst" in data and hasattr(data["inst"], "pos") else None)
        if pos is None:
            # No positions: fallback to no bias, Flash path
            chip_sizes = None

        if inst_batch is None or inst_batch.numel() == 0:
            batch_size = 1
            inst_batch = torch.zeros(inst_x.shape[0], device=device, dtype=torch.long)
        else:
            batch_size = int(inst_batch.max().item()) + 1

        if chip_sizes is None or chip_sizes.shape[0] != batch_size:
            use_penalty_this_batch = False
        else:
            use_penalty_this_batch = self.use_distance_penalty

        inst_input = inst_x
        out_list = []
        index_list = []

        for g in range(batch_size):
            mask_g = inst_batch == g
            if not mask_g.any():
                continue
            inst_x_g = inst_x[mask_g]  # [N_g, D]
            pos_g = pos[mask_g] if pos is not None else None  # [N_g, 2]
            N_g = inst_x_g.shape[0]

            if chip_sizes is not None and g < chip_sizes.shape[0]:
                W, H = chip_sizes[g, 0].item(), chip_sizes[g, 1].item()
                K_w = max(1, int(math.ceil(W / self.L_star_um)))
                K_h = max(1, int(math.ceil(H / self.L_star_um)))
            else:
                K_w = max(1, 8)
                K_h = max(1, 8)

            K = K_w * K_h
            grid_centers = self._grid_centers_normalized(K_w, K_h, device, dtype)  # [K, 2]
            latents = self.latent_embed.expand(1, K, self.hidden_dim)  # [1, K, D]

            # Scale s = [W/2, H/2] for physical distance (same units as W, H)
            if chip_sizes is not None and g < chip_sizes.shape[0]:
                s_g = chip_sizes[g] * 0.5  # [2]
            else:
                s_g = torch.tensor([1.0, 1.0], device=device, dtype=dtype)

            if use_penalty_this_batch and pos_g is not None and self.distance_penalty_lambda != 0:
                bias_latents_query = self._d_phys_bias(pos_g, grid_centers, s_g, self.distance_penalty_lambda)  # [N_g, K]
                bias_latents_query = bias_latents_query.unsqueeze(0)  # [1, N_g, K]
                bias_kv_latents = bias_latents_query.transpose(1, 2)  # [1, K, N_g] for first cross-attn (Q=latents, KV=nodes)
            else:
                bias_latents_query = None
                bias_kv_latents = None

            # (1) Cross-attn: latents attend to nodes. Q [1,K,D], KV [1,N_g,D]
            latents = self.inst_to_global_cross_attn(
                query=latents,
                key_value=inst_x_g.unsqueeze(0),
                attn_bias=bias_kv_latents,
            )
            # (2) Self-attn on latents
            if self.global_self_attn is not None:
                for layer in self.global_self_attn:
                    latents = layer(latents)
            # (3) Cross-attn: nodes attend to latents. Q [1,N_g,D], KV [1,K,D]
            inst_out_g = self.global_to_inst_cross_attn(
                query=inst_x_g.unsqueeze(0),
                key_value=latents,
                attn_bias=bias_latents_query,
            )
            out_list.append(inst_out_g.squeeze(0))
            index_list.append(torch.where(mask_g)[0])

        if not out_list:
            return inst_input - inst_input, net_delta  # zero deltas

        all_out = torch.cat(out_list, dim=0)
        all_idx = torch.cat(index_list, dim=0)
        inst_delta = torch.zeros_like(inst_x)
        inst_delta[all_idx] = all_out - inst_input[all_idx]
        return inst_delta, net_delta




@register_global_module("recurrent_slot_memory")
class RecurrentSlotMemoryGlobalModule(BaseGlobalModule):
    """
    Fixed-width global memory with loop-carried slot state.

    This module is designed for iterative models such as `LoopedPlacementModel`.
    It keeps a constant number of memory slots regardless of graph size and
    updates those slots through repeated read/write passes.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_slots: int = 32,
        num_heads: int = 4,
        num_self_attn_layers: int = 1,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        slot_update: str = "gru",
        competitive_slots: bool = False,
        assignment_temperature: float = 1.0,
        slot_mlp_expansion: float = 2.0,
        **kwargs,
    ):
        super().__init__(hidden_dim, **kwargs)
        self.num_slots = int(num_slots)
        self.num_self_attn_layers = int(num_self_attn_layers)
        self.slot_update = str(slot_update).lower()
        self.competitive_slots = bool(competitive_slots)
        self.assignment_temperature = float(max(assignment_temperature, 1e-4))
        self.slot_mlp_hidden_dim = max(int(hidden_dim * float(slot_mlp_expansion)), hidden_dim)

        if self.slot_update not in {"gru", "residual"}:
            raise ValueError(
                f"slot_update must be 'gru' or 'residual', got: {slot_update}"
            )

        self.initial_slots = nn.Parameter(torch.randn(1, self.num_slots, hidden_dim))
        nn.init.normal_(self.initial_slots, std=0.02)

        self.slot_read_attn = MultiHeadCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        self.slot_to_node_attn = MultiHeadCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

        if self.num_self_attn_layers > 0:
            self.slot_self_attn = nn.ModuleList(
                [
                    MultiHeadSelfAttention(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        use_layer_norm=use_layer_norm,
                    )
                    for _ in range(self.num_self_attn_layers)
                ]
            )
        else:
            self.slot_self_attn = None

        if self.competitive_slots:
            self.slot_assign_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            self.node_assign_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            self.slot_query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.node_key_proj = nn.Linear(hidden_dim, hidden_dim)
            self.node_value_proj = nn.Linear(hidden_dim, hidden_dim)
            self.competitive_out_proj = nn.Linear(hidden_dim, hidden_dim)
            self.assignment_scale = hidden_dim ** -0.5
        else:
            self.slot_assign_norm = None
            self.node_assign_norm = None
            self.slot_query_proj = None
            self.node_key_proj = None
            self.node_value_proj = None
            self.competitive_out_proj = None
            self.assignment_scale = None

        if self.slot_update == "gru":
            self.slot_update_cell = nn.GRUCell(hidden_dim, hidden_dim)
            self.slot_update_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            self.slot_update_mlp = None
        else:
            self.slot_update_cell = None
            self.slot_update_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            self.slot_update_mlp = nn.Sequential(
                nn.Linear(hidden_dim, self.slot_mlp_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.slot_mlp_hidden_dim, hidden_dim),
            )

        self.slot_ffn_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.slot_ffn = nn.Sequential(
            nn.Linear(hidden_dim, self.slot_mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.slot_mlp_hidden_dim, hidden_dim),
        )
        self.latest_slot_stats: Dict[str, Any] = {}

    def init_loop_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create per-graph slot memory for the outer loop."""
        return self.initial_slots.to(device=device, dtype=dtype).expand(batch_size, -1, -1).clone()

    def _pack_nodes(
        self,
        inst_x: torch.Tensor,
        inst_batch: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if inst_batch is None:
            batched = inst_x.unsqueeze(0)
            mask = torch.ones(
                1,
                inst_x.shape[0],
                device=inst_x.device,
                dtype=torch.bool,
            )
            return batched, mask, None

        batch_size = int(inst_batch.max().item()) + 1
        counts = torch.bincount(inst_batch, minlength=batch_size)
        max_nodes = int(counts.max().item())
        device = inst_x.device
        cumsum = torch.cat(
            [
                torch.zeros(1, device=device, dtype=counts.dtype),
                counts[:-1].cumsum(0),
            ]
        )
        positions = torch.arange(inst_x.shape[0], device=device) - cumsum[inst_batch]

        batched = torch.zeros(
            batch_size,
            max_nodes,
            inst_x.shape[1],
            device=device,
            dtype=inst_x.dtype,
        )
        mask = torch.zeros(batch_size, max_nodes, device=device, dtype=torch.bool)
        batched[inst_batch, positions] = inst_x
        mask[inst_batch, positions] = True
        return batched, mask, positions

    def _unpack_nodes(
        self,
        batched_x: torch.Tensor,
        inst_batch: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if inst_batch is None:
            return batched_x.squeeze(0)
        return batched_x[inst_batch, positions]

    def _competitive_slot_read(
        self,
        slots: torch.Tensor,
        node_batched: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        slot_queries = self.slot_query_proj(self.slot_assign_norm(slots))
        node_keys = self.node_key_proj(self.node_assign_norm(node_batched))
        node_values = self.node_value_proj(node_batched)

        logits = torch.einsum("bkd,bnd->bnk", slot_queries, node_keys)
        logits = logits * self.assignment_scale / self.assignment_temperature
        logits = logits.masked_fill(~node_mask.unsqueeze(-1), -1e4)

        slot_weights = F.softmax(logits, dim=-1)
        slot_weights = slot_weights * node_mask.unsqueeze(-1)

        denom = slot_weights.sum(dim=1).unsqueeze(-1)
        denom = denom.clamp(min=1e-6)
        aggregated = torch.einsum("bnk,bnd->bkd", slot_weights, node_values) / denom
        entropy = -(slot_weights.clamp(min=1e-8) * slot_weights.clamp(min=1e-8).log()).sum(dim=-1)
        entropy = entropy.masked_fill(~node_mask, 0.0)
        entropy_mean = entropy.sum() / node_mask.sum().clamp(min=1)
        return slots + self.competitive_out_proj(aggregated), float(entropy_mean.item())

    def _apply_slot_update(
        self,
        previous_slots: torch.Tensor,
        candidate_slots: torch.Tensor,
    ) -> torch.Tensor:
        if self.slot_self_attn is not None:
            for self_attn in self.slot_self_attn:
                candidate_slots = self_attn(candidate_slots)

        if self.slot_update == "gru":
            flat_candidate = candidate_slots.reshape(-1, self.hidden_dim)
            flat_previous = previous_slots.reshape(-1, self.hidden_dim)
            updated_slots = self.slot_update_cell(flat_candidate, flat_previous)
            updated_slots = updated_slots.view_as(previous_slots)
        else:
            slot_delta = self.slot_update_mlp(self.slot_update_norm(candidate_slots))
            updated_slots = previous_slots + slot_delta

        updated_slots = updated_slots + self.slot_ffn(self.slot_ffn_norm(updated_slots))
        return updated_slots

    def forward_with_memory(
        self,
        inst_x: torch.Tensor,
        net_x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        inst_batch: Optional[torch.Tensor] = None,
        net_batch: Optional[torch.Tensor] = None,
        inst_rel_pos_bias: Optional[torch.Tensor] = None,
        net_rel_pos_bias: Optional[torch.Tensor] = None,
        inst_pos: Optional[torch.Tensor] = None,
        net_pos: Optional[torch.Tensor] = None,
        data: Optional[HeteroData] = None,
        time_embed: Optional[torch.Tensor] = None,
        step_idx: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del net_batch, inst_rel_pos_bias, net_rel_pos_bias, inst_pos, net_pos, data, time_embed, step_idx, kwargs

        node_batched, node_mask, node_positions = self._pack_nodes(inst_x, inst_batch)
        batch_size = node_batched.shape[0]
        if memory_state is None:
            memory_state = self.init_loop_state(
                batch_size=batch_size,
                device=inst_x.device,
                dtype=inst_x.dtype,
            )
        elif memory_state.shape[0] != batch_size:
            raise ValueError(
                f"memory_state batch size {memory_state.shape[0]} does not match "
                f"input batch size {batch_size}"
            )

        slots = memory_state
        if self.competitive_slots:
            candidate_slots, slot_assignment = self._competitive_slot_read(
                slots=slots,
                node_batched=node_batched,
                node_mask=node_mask,
            )
        else:
            candidate_slots = self.slot_read_attn(
                query=slots,
                key_value=node_batched,
                kv_mask=node_mask,
            )
            slot_assignment = None

        updated_slots = self._apply_slot_update(
            previous_slots=slots,
            candidate_slots=candidate_slots,
        )

        node_context = self.slot_to_node_attn(
            query=node_batched,
            key_value=updated_slots,
            query_mask=node_mask,
        )
        node_delta = self._unpack_nodes(node_context, inst_batch, node_positions) - inst_x
        net_delta = torch.zeros_like(net_x) if net_x is not None else torch.zeros_like(node_delta)

        with torch.no_grad():
            self.latest_slot_stats = {
                "slot_norm_mean": float(updated_slots.norm(dim=-1).mean().item()),
                "slot_norm_max": float(updated_slots.norm(dim=-1).max().item()),
            }
            if slot_assignment is not None:
                self.latest_slot_stats["slot_assignment_entropy"] = float(slot_assignment)

        return node_delta, net_delta, updated_slots

    def forward(
        self,
        inst_x: torch.Tensor,
        net_x: torch.Tensor,
        inst_batch: Optional[torch.Tensor] = None,
        net_batch: Optional[torch.Tensor] = None,
        inst_rel_pos_bias: Optional[torch.Tensor] = None,
        net_rel_pos_bias: Optional[torch.Tensor] = None,
        inst_pos: Optional[torch.Tensor] = None,
        net_pos: Optional[torch.Tensor] = None,
        data: Optional[HeteroData] = None,
        time_embed: Optional[torch.Tensor] = None,
        step_idx: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inst_delta, net_delta, _ = self.forward_with_memory(
            inst_x=inst_x,
            net_x=net_x,
            memory_state=None,
            inst_batch=inst_batch,
            net_batch=net_batch,
            inst_rel_pos_bias=inst_rel_pos_bias,
            net_rel_pos_bias=net_rel_pos_bias,
            inst_pos=inst_pos,
            net_pos=net_pos,
            data=data,
            time_embed=time_embed,
            step_idx=step_idx,
            **kwargs,
        )
        return inst_delta, net_delta


def create_global_module(
    module_type: str,
    hidden_dim: int,
    **module_params
) -> BaseGlobalModule:
    """
    Factory function to create a global module by name.
    
    Args:
        module_type: Module name (e.g. "identity", "attention", "perceiver",
                     "dynamic_token_perceiver", "hybrid_metis_perceiver", etc.)
        hidden_dim: Hidden dimension
        **module_params: Module-specific parameters
        
    Returns:
        BaseGlobalModule instance
    """
    module_class = get_global_module(module_type)
    return module_class(hidden_dim=hidden_dim, **module_params)


# ============================================================================
# Hybrid Metis Perceiver — global tokens + partition-local metis tokens
# ============================================================================

@register_global_module("hybrid_metis_perceiver")
class HybridMetisPerceiverGlobalModule(BaseGlobalModule):
    """
    Perceiver with two kinds of latent tokens:

      1. **Global tokens** (fixed count ``num_global_tokens``):
         Learnable embeddings that cross-attend to *every* node in the graph,
         providing whole-graph context.

      2. **Metis tokens** (one per Metis partition, count K varies per graph):
         Content-initialised via ``scatter_mean`` of member-node hidden states.
         Each metis token cross-attends *only* to nodes in its partition,
         enforcing locality.

    A learned **token-type embedding** is added so the model can distinguish
    the two token populations.

    Attention masks (built per batch element):

      * **nodes → tokens cross-attn** (Q = all tokens, KV = nodes):
        - Global token rows: may attend to every real node (full row True).
        - Metis  token rows: may attend only to nodes in the corresponding
          partition (sparse row).

      * **token self-attn**: all (global + metis) tokens attend freely.
        Information flows between partitions through the global tokens.

      * **tokens → nodes cross-attn** (Q = nodes, KV = all tokens):
        - For each node: may attend to all global tokens + the single metis
          token of its own partition.

    Spectral PE from the full-graph Laplacian can optionally be added (same
    scheme as ``MetisPerceiverGlobalModule``).

    Required data fields (same as ``MetisPerceiverGlobalModule``):
        data.metis_partition_id : LongTensor[N_total]               0..K-1
        data.num_partitions     : LongTensor[N_total]               K broadcast
        data.lap_eigenvectors   : FloatTensor[N_total, num_pe_dims] (optional)

    Args:
        hidden_dim           : int  — model hidden dimension
        num_global_tokens    : int  — fixed count of global tokens (default 64)
        max_k                : int  — upper bound on partition count (default 64)
        num_pe_dims          : int  — Laplacian eigenvector dims (default 5)
        num_heads            : int  — attention heads (default 4)
        num_self_attn_layers : int  — self-attn layers on combined tokens (default 2)
        dropout              : float
        use_layer_norm       : bool
    """

    def __init__(
        self,
        hidden_dim: int,
        num_global_tokens: int = 64,
        max_k: int = 64,
        num_pe_dims: int = 5,
        num_heads: int = 4,
        num_self_attn_layers: int = 2,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        **kwargs,
    ):
        super().__init__(hidden_dim, **kwargs)
        self.num_global_tokens = num_global_tokens
        self.max_k = max_k
        self.num_pe_dims = num_pe_dims

        # Learnable global tokens  [1, G, D]
        self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, hidden_dim))
        nn.init.normal_(self.global_tokens, std=0.02)

        # Token-type embeddings: 0 = global, 1 = metis
        self.token_type_embed = nn.Embedding(2, hidden_dim)

        # Spectral PE projections
        self.node_pe_proj = nn.Sequential(
            nn.Linear(num_pe_dims, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.token_pe_proj = nn.Sequential(
            nn.Linear(num_pe_dims, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # nodes → combined tokens   (Q = tokens, KV = nodes)
        self.nodes_to_tokens_cross_attn = MultiHeadCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

        # Self-attention among all tokens (global + metis)
        if num_self_attn_layers > 0:
            self.token_self_attn = nn.ModuleList([
                MultiHeadSelfAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_self_attn_layers)
            ])
        else:
            self.token_self_attn = None

        # combined tokens → nodes   (Q = nodes, KV = tokens)
        self.tokens_to_nodes_cross_attn = MultiHeadCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    # ------------------------------------------------------------------
    # mask builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_nodes_to_tokens_mask(
        batch_size: int,
        num_global: int,
        max_K: int,
        max_N: int,
        K_per_graph: torch.Tensor,
        node_counts: torch.Tensor,
        partition_ids_batched: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Build [B, G+max_K, max_N] bool mask for nodes→tokens cross-attn.

        Rows 0..G-1 (global tokens): True for every real node.
        Rows G..G+k-1 (metis token j): True only for nodes with partition_id == j.
        Rows G+k..G+max_K-1 (padding): False everywhere.
        """
        T = num_global + max_K
        mask = torch.zeros(batch_size, T, max_N, dtype=torch.bool, device=device)

        node_range = torch.arange(max_N, device=device).unsqueeze(0)  # [1, max_N]
        real_node_mask = node_range < node_counts.unsqueeze(1)        # [B, max_N]

        # Global rows: attend to all real nodes
        mask[:, :num_global, :] = real_node_mask.unsqueeze(1)

        # Metis rows: each metis token j attends only to nodes with pid == j
        metis_range = torch.arange(max_K, device=device)              # [max_K]
        # partition_ids_batched: [B, max_N] with -1 for padding
        # For each (b, j, n): True iff pid[b,n] == j AND n < node_count[b] AND j < K[b]
        pid_match = (
            partition_ids_batched.unsqueeze(1)                        # [B, 1, max_N]
            == metis_range.view(1, -1, 1)                             # [1, max_K, 1]
        )
        metis_valid = metis_range.unsqueeze(0) < K_per_graph.unsqueeze(1)  # [B, max_K]
        pid_match = pid_match & metis_valid.unsqueeze(-1) & real_node_mask.unsqueeze(1)
        mask[:, num_global:num_global + max_K, :] = pid_match

        return mask

    @staticmethod
    def _build_tokens_to_nodes_mask(
        batch_size: int,
        num_global: int,
        max_K: int,
        max_N: int,
        K_per_graph: torch.Tensor,
        node_counts: torch.Tensor,
        partition_ids_batched: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Build [B, max_N, G+max_K] bool mask for tokens→nodes cross-attn.

        For each node n in partition j:
          - columns 0..G-1 (global tokens): True (always visible)
          - column  G+j (own metis token):  True
          - other metis columns:            False
        Padding node rows: all False.
        Padding token columns: all False.
        """
        T = num_global + max_K
        mask = torch.zeros(batch_size, max_N, T, dtype=torch.bool, device=device)

        node_range = torch.arange(max_N, device=device).unsqueeze(0)
        real_node_mask = node_range < node_counts.unsqueeze(1)        # [B, max_N]

        # Global columns: every real node may attend to every global token
        mask[:, :, :num_global] = real_node_mask.unsqueeze(-1)

        # Metis columns: node n attends to metis token j iff pid[n] == j
        metis_range = torch.arange(max_K, device=device)
        pid_match = (
            partition_ids_batched.unsqueeze(-1)                       # [B, max_N, 1]
            == metis_range.view(1, 1, -1)                             # [1, 1, max_K]
        )
        metis_valid = metis_range.unsqueeze(0) < K_per_graph.unsqueeze(1)  # [B, max_K]
        pid_match = pid_match & metis_valid.unsqueeze(1) & real_node_mask.unsqueeze(-1)
        mask[:, :, num_global:num_global + max_K] = pid_match

        return mask

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        inst_x: torch.Tensor,
        net_x: torch.Tensor,
        inst_batch: Optional[torch.Tensor] = None,
        net_batch: Optional[torch.Tensor] = None,
        data=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = inst_x.device

        if (
            data is None
            or not hasattr(data, "metis_partition_id")
            or data.metis_partition_id is None
        ):
            raise ValueError(
                "HybridMetisPerceiverGlobalModule requires data.metis_partition_id. "
                "Annotate with scripts/dataset_generation/add_metis_partition_to_dataset.py."
            )

        if inst_batch is None:
            inst_batch = torch.zeros(inst_x.shape[0], dtype=torch.long, device=device)

        batch_size = int(inst_batch.max().item()) + 1
        D = inst_x.shape[1]
        G = self.num_global_tokens

        partition_ids_all = data.metis_partition_id.to(device)
        num_parts_all = data.num_partitions.to(device)

        # ----------------------------------------------------------------
        # Step 1: batch nodes → [B, max_N, D], build partition-id matrix
        # ----------------------------------------------------------------
        node_counts = torch.bincount(inst_batch, minlength=batch_size)
        max_N = int(node_counts.max().item())
        node_cumsum = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            node_counts[:-1].cumsum(0),
        ])
        node_pos = torch.arange(inst_x.shape[0], device=device) - node_cumsum[inst_batch]

        # Spectral PE on nodes
        has_lap_pe = (
            hasattr(data, "lap_eigenvectors")
            and data.lap_eigenvectors is not None
        )
        if has_lap_pe:
            lap_pe_all = data.lap_eigenvectors.to(device=device, dtype=inst_x.dtype)
            pe_cols = min(self.num_pe_dims, lap_pe_all.shape[1])
            if pe_cols < self.num_pe_dims:
                pad = torch.zeros(
                    lap_pe_all.shape[0], self.num_pe_dims - pe_cols,
                    dtype=inst_x.dtype, device=device,
                )
                lap_pe_all = torch.cat([lap_pe_all[:, :pe_cols], pad], dim=1)
            else:
                lap_pe_all = lap_pe_all[:, :self.num_pe_dims]
            inst_x_pe = inst_x + self.node_pe_proj(lap_pe_all)
        else:
            lap_pe_all = None
            inst_x_pe = inst_x

        inst_batched = torch.zeros(batch_size, max_N, D, dtype=inst_x.dtype, device=device)
        inst_mask = torch.zeros(batch_size, max_N, dtype=torch.bool, device=device)
        inst_batched[inst_batch, node_pos] = inst_x_pe
        inst_mask[inst_batch, node_pos] = True

        # Partition IDs batched: [B, max_N], padded with -1
        partition_ids_batched = torch.full(
            (batch_size, max_N), -1, dtype=torch.long, device=device,
        )
        partition_ids_batched[inst_batch, node_pos] = partition_ids_all

        # ----------------------------------------------------------------
        # Step 2: build metis tokens via scatter_mean + spectral PE
        # ----------------------------------------------------------------
        K_per_graph = num_parts_all[node_cumsum]
        K_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            K_per_graph.cumsum(0)[:-1],
        ])
        max_K = int(K_per_graph.max().item())
        if max_K > self.max_k:
            raise ValueError(
                f"HybridMetisPerceiverGlobalModule: batch needs K={max_K} metis tokens "
                f"but max_k={self.max_k}."
            )

        token_batch_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=device), K_per_graph,
        )
        total_K = token_batch_ids.shape[0]
        token_local_pos = (
            torch.arange(total_K, device=device) - K_offsets[token_batch_ids]
        )

        global_pids = partition_ids_all + K_offsets[inst_batch]

        # scatter_mean per partition
        counts = torch.bincount(global_pids, minlength=total_K).to(
            dtype=inst_x.dtype,
        ).clamp(min=1.0)
        token_sums = torch.zeros(total_K, D, dtype=inst_x.dtype, device=device)
        token_sums.scatter_add_(
            0,
            global_pids.unsqueeze(-1).expand_as(inst_x),
            inst_x,
        )
        metis_tokens_flat = token_sums / counts.unsqueeze(-1)

        # Spectral PE for metis tokens
        if has_lap_pe and lap_pe_all is not None:
            pe_sums = torch.zeros(
                total_K, self.num_pe_dims, dtype=inst_x.dtype, device=device,
            )
            pe_sums.scatter_add_(
                0,
                global_pids.unsqueeze(-1).expand(-1, self.num_pe_dims),
                lap_pe_all,
            )
            token_pe = pe_sums / counts.unsqueeze(-1)
            metis_tokens_flat = metis_tokens_flat + self.token_pe_proj(token_pe)

        # Pad metis tokens → [B, max_K, D]
        metis_batched = torch.zeros(
            batch_size, max_K, D, dtype=inst_x.dtype, device=device,
        )
        metis_batched[token_batch_ids, token_local_pos] = metis_tokens_flat

        metis_mask = (
            torch.arange(max_K, device=device).unsqueeze(0)
            < K_per_graph.unsqueeze(1)
        )  # [B, max_K]

        # ----------------------------------------------------------------
        # Step 3: concatenate global + metis tokens → [B, G+max_K, D]
        # ----------------------------------------------------------------
        global_tok = self.global_tokens.expand(batch_size, -1, -1)  # [B, G, D]

        # Add token-type embeddings
        global_type_id = torch.zeros(batch_size, G, dtype=torch.long, device=device)
        metis_type_id = torch.ones(batch_size, max_K, dtype=torch.long, device=device)
        global_tok = global_tok + self.token_type_embed(global_type_id)
        metis_batched = metis_batched + self.token_type_embed(metis_type_id)

        all_tokens = torch.cat([global_tok, metis_batched], dim=1)  # [B, G+max_K, D]

        # Combined token mask for self-attn: globals always valid, metis per K
        all_token_mask = torch.cat([
            torch.ones(batch_size, G, dtype=torch.bool, device=device),
            metis_mask,
        ], dim=1)  # [B, G+max_K]

        # ----------------------------------------------------------------
        # Step 4: build 2-D attention masks for cross-attn
        # ----------------------------------------------------------------
        # nodes→tokens: [B, G+max_K, max_N]  (query=tokens, kv=nodes)
        n2t_mask = self._build_nodes_to_tokens_mask(
            batch_size, G, max_K, max_N,
            K_per_graph, node_counts, partition_ids_batched, device,
        )

        # tokens→nodes: [B, max_N, G+max_K]  (query=nodes, kv=tokens)
        t2n_mask = self._build_tokens_to_nodes_mask(
            batch_size, G, max_K, max_N,
            K_per_graph, node_counts, partition_ids_batched, device,
        )

        # ----------------------------------------------------------------
        # Step 5: attention pipeline
        # ----------------------------------------------------------------
        # 5a. nodes → tokens cross-attn  (Q=tokens, KV=nodes)
        #     Need 2-D mask [B, T, N] → SDPA wants [B, 1, T, N] or [B, H, T, N]
        all_tokens = self._masked_cross_attn(
            self.nodes_to_tokens_cross_attn,
            query=all_tokens,
            key_value=inst_batched,
            mask_2d=n2t_mask,
        )

        # 5b. Self-attention over all tokens
        if self.token_self_attn is not None:
            has_pad = not bool(all_token_mask.all())
            inv_mask = (~all_token_mask).unsqueeze(-1) if has_pad else None
            for layer in self.token_self_attn:
                if inv_mask is not None:
                    all_tokens = all_tokens.masked_fill(inv_mask, 0.0)
                all_tokens = layer(all_tokens, mask=all_token_mask)
                if inv_mask is not None:
                    all_tokens = all_tokens.masked_fill(inv_mask, 0.0)

        # 5c. tokens → nodes cross-attn  (Q=nodes, KV=tokens)
        inst_batched = self._masked_cross_attn(
            self.tokens_to_nodes_cross_attn,
            query=inst_batched,
            key_value=all_tokens,
            mask_2d=t2n_mask,
        )

        # ----------------------------------------------------------------
        # Step 6: unbatch → [N_total, D], return residual delta
        # ----------------------------------------------------------------
        inst_x_out = inst_batched[inst_batch, node_pos]
        return inst_x_out - inst_x, None

    @staticmethod
    def _masked_cross_attn(
        module: MultiHeadCrossAttention,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Run cross-attention with a full 2-D boolean mask.

        ``module`` is a ``MultiHeadCrossAttention`` that normally only
        accepts a 1-D ``kv_mask``.  We bypass its internal SDPA call
        and manually construct the mask in ``[B, 1, N_q, N_kv]`` form
        so that different query rows see different key subsets.

        The residual connection and output projection follow the same
        pattern as the original module.

        Important: SDPA with an all-False row in ``attn_mask`` produces
        NaN (``softmax([-inf,...]) = NaN``) in both the forward output
        **and** the backward gradient, which under AMP causes
        ``GradScaler`` to skip every optimizer step and freezes training.
        Padding query rows (padding metis-token slots; padding node rows
        in tokens→nodes) are genuinely all-False, so we route them to a
        dummy key (column 0 forced True), then zero-out their output
        after SDPA so they contribute nothing downstream.
        """
        B, N_q, _ = query.shape
        _, N_kv, _ = key_value.shape

        residual = query
        query_normed = module.layer_norm(query)

        q = module.q_proj(query_normed)
        k = module.k_proj(key_value)
        v = module.v_proj(key_value)

        q = q.view(B, N_q, module.num_heads, module.head_dim).transpose(1, 2)
        k = k.view(B, N_kv, module.num_heads, module.head_dim).transpose(1, 2)
        v = v.view(B, N_kv, module.num_heads, module.head_dim).transpose(1, 2)

        # Detect rows with no valid key; force them to attend to col 0 so
        # SDPA does not emit NaN, then zero their outputs afterwards.
        row_valid = mask_2d.any(dim=-1, keepdim=True)          # [B, N_q, 1]
        safe_mask = mask_2d.clone()
        if not bool(row_valid.all()):
            # For rows with no valid key, allow the dummy key at column 0.
            safe_mask[..., 0] = safe_mask[..., 0] | (~row_valid.squeeze(-1))

        # safe_mask: [B, N_q, N_kv] bool → [B, 1, N_q, N_kv] for SDPA broadcast
        attn_mask = safe_mask.unsqueeze(1)

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=True, enable_math=True,
        ):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=module.dropout.p if module.training else 0.0,
                is_causal=False,
            )

        # Zero out outputs for padding query rows so they contribute nothing
        # downstream (belt-and-braces; also kills any residual NaN).
        if not bool(row_valid.all()):
            out = out.masked_fill(~row_valid.unsqueeze(1), 0.0)

        out = out.transpose(1, 2).contiguous().view(B, N_q, module.hidden_dim)
        out = module.out_proj(out)
        out = module.dropout(out)
        return out + residual
