"""
Deep Incidence Transformer processor.

Architecture per layer l:

    Step A  [V→E]   z_E ← z_E + Attn_{V→E}^l(LN(z_E), LN(z_V))     (message construction)
                    z_E ← z_E + FFN_E^l(LN(z_E))

    Step B  [E→V]   z_V ← z_V + Attn_{E→V}^l(LN(z_V), LN(z_E))     (aggregation)
                    z_V ← z_V + FFN_V^l(LN(z_V))

    Step C  [Global] z_V_glob ← GlobalBranch^l(z_V)                   (optional, every N-th layer)
                    z_V ← Router^l(z_V_local, z_V_glob)

Each layer has its OWN independent parameters (no weight sharing).
"""

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Dict, Any

from .config import IncidenceTransformerConfig
from .tokenizer import GraphTokenizer
from .attention import NodeToEdgeAttention, EdgeToNodeAttention, PreNormFFN
from .global_branch import GlobalMemoryBranch, DenseGlobalBranch, RouterGate


def _apply_deep_init(layers: nn.ModuleList, num_layers: int):
    """
    GPT-2-style scaled initialization for deep transformers.

    Residual-path output projections (W_o, FFN linear2) are scaled by
    1/sqrt(2*num_layers) so the variance of the residual stream stays
    bounded as depth grows.
    """
    scale = 1.0 / math.sqrt(2 * max(num_layers, 1))

    for layer in layers:
        for name, param in layer.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("linear2.weight"):
                param.data.mul_(scale)


class IncidenceBlock(nn.Module):
    """
    One layer of the Incidence Bipartite Transformer.

    Contains:
        - V→E attention + edge FFN  (message construction)
        - E→V attention + node FFN  (aggregation / update)
        - Optional global branch + gated routing
    """

    def __init__(self, config: IncidenceTransformerConfig, use_global: bool = False):
        super().__init__()
        d = config.d_model
        s = config.residual_scale

        self.v2e_attn = NodeToEdgeAttention(config)
        self.edge_ffn = PreNormFFN(
            d, config.ffn_multiplier, config.dropout, config.ffn_activation,
            output_scale=s,
        )

        self.e2v_attn = EdgeToNodeAttention(config)
        self.node_ffn = PreNormFFN(
            d, config.ffn_multiplier, config.dropout, config.ffn_activation,
            output_scale=s,
        )

        self.use_global = use_global
        if self.use_global:
            if config.global_branch_type == "memory":
                self.global_branch = GlobalMemoryBranch(config)
            elif config.global_branch_type == "dense":
                self.global_branch = DenseGlobalBranch(config)
            else:
                raise ValueError(f"Unknown global_branch_type: {config.global_branch_type}")
            self.router = RouterGate(d)

        self.residual_scale = s

    def forward(
        self,
        z_V: torch.Tensor,
        z_E: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.residual_scale

        # Step A: V→E message construction
        z_E = z_E + s * self.v2e_attn(z_V, z_E, edge_index)
        z_E = self.edge_ffn(z_E)

        # Step B: E→V aggregation
        z_V_local = z_V + s * self.e2v_attn(z_E, z_V, edge_index)
        z_V_local = self.node_ffn(z_V_local)

        # Step C: optional global branch + routing
        if self.use_global:
            z_V_glob = self.global_branch(z_V, batch=batch)
            z_V = self.router(z_V_local, z_V_glob)
        else:
            z_V = z_V_local

        return z_V, z_E


class IncidenceTransformerProcessor(nn.Module):
    """
    Full pipeline: Tokenizer → L independent IncidenceBlocks → final LN.

    Each layer has its own parameters. Global branch is optionally added
    every `global_branch_every_n` layers (e.g. every 4th layer to save memory).

    Supports gradient checkpointing via config.use_activation_checkpointing.
    """

    def __init__(self, config: IncidenceTransformerConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers

        self.tokenizer = GraphTokenizer(config)

        layers = []
        for i in range(config.num_layers):
            has_global = (
                config.use_global_branch
                and ((i + 1) % config.global_branch_every_n == 0)
            )
            layers.append(IncidenceBlock(config, use_global=has_global))
        self.layers = nn.ModuleList(layers)

        self.final_norm_v = nn.LayerNorm(config.d_model)
        self.final_norm_e = nn.LayerNorm(config.d_model)

        _apply_deep_init(self.layers, config.num_layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            x: (N, d_node_in) node features from encoder
            edge_index: (2, M) directed edge indices
            edge_attr: (M, d_edge_in) edge features, optional
            batch: (N,) graph batch indices

        Returns:
            dict with z_V: (N, d) and z_E: (M, d)
        """
        use_ckpt = self.config.use_activation_checkpointing and self.training

        z_V, z_E = self.tokenizer(x, edge_index, edge_attr, batch)

        for layer in self.layers:
            if use_ckpt:
                def _run(mod, _zV, _zE, _ei, _b):
                    return mod(_zV, _zE, _ei, _b)

                z_V, z_E = checkpoint(
                    _run, layer, z_V, z_E, edge_index, batch,
                    use_reentrant=False,
                )
            else:
                z_V, z_E = layer(z_V, z_E, edge_index, batch=batch)

        z_V = self.final_norm_v(z_V)
        z_E = self.final_norm_e(z_E)

        return {"z_V": z_V, "z_E": z_E}

    @torch.no_grad()
    def get_state_norms(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, list]:
        """Monitor hidden-state norms across layers (stability diagnosis)."""
        z_V, z_E = self.tokenizer(x, edge_index, edge_attr, batch)

        node_norms = [z_V.norm(dim=-1).mean().item()]
        edge_norms = [z_E.norm(dim=-1).mean().item()]

        for layer in self.layers:
            z_V, z_E = layer(z_V, z_E, edge_index, batch=batch)
            node_norms.append(z_V.norm(dim=-1).mean().item())
            edge_norms.append(z_E.norm(dim=-1).mean().item())

        return {"node_norms": node_norms, "edge_norms": edge_norms}
