"""
Graph tokenizer: converts a PyG graph into node tokens and directed edge tokens.

Node token:
    z_i^V = W_V(x_i) + degree_enc(i) + id_enc(i) + t_V

Directed edge token:
    z_{j→i}^E = W_E(e_{ji}) + W_src(u_j) + W_tgt(u_i) + t_E
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import IncidenceTransformerConfig


def _sinusoidal_ids(n: int, d: int, device: torch.device) -> torch.Tensor:
    """Generate sinusoidal positional IDs of shape (n, d)."""
    position = torch.arange(n, device=device, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d, 2, device=device, dtype=torch.float)
        * (-math.log(10000.0) / d)
    )
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: d // 2])
    return pe


class GraphTokenizer(nn.Module):
    """
    Tokenize a graph into node tokens and directed edge tokens.

    Handles batched PyG graphs (flat tensors with batch indices).
    """

    def __init__(self, config: IncidenceTransformerConfig):
        super().__init__()
        d = config.d_model

        # --- node projection ---
        self.node_proj = nn.Linear(config.d_node_in, d)

        # --- edge projection ---
        self.has_edge_features = config.d_edge_in > 0
        if self.has_edge_features:
            self.edge_proj = nn.Linear(config.d_edge_in, d)
        else:
            self.edge_proj = None

        # --- type embeddings ---
        self.node_type_embed = nn.Parameter(torch.zeros(d))
        self.edge_type_embed = nn.Parameter(torch.zeros(d))

        # --- degree encodings (Graphormer-style) ---
        self.use_degree_encoding = config.use_degree_encoding
        if self.use_degree_encoding:
            self.in_degree_embed = nn.Embedding(config.max_in_degree + 1, d)
            self.out_degree_embed = nn.Embedding(config.max_out_degree + 1, d)
            nn.init.normal_(self.in_degree_embed.weight, std=0.02)
            nn.init.normal_(self.out_degree_embed.weight, std=0.02)

        # --- node identifiers ---
        self.use_node_ids = config.use_node_ids
        self.id_type = config.id_type
        self.id_noise_scale = config.id_noise_scale
        self.d_node_id = config.d_node_id

        if self.use_node_ids:
            if config.id_type == "learned":
                self.id_embed = nn.Embedding(config.max_nodes, config.d_node_id)
                nn.init.normal_(self.id_embed.weight, std=0.02)
            elif config.id_type == "random":
                rand_ids = torch.randn(config.max_nodes, config.d_node_id)
                rand_ids = nn.functional.normalize(rand_ids, dim=-1) * math.sqrt(config.d_node_id)
                self.register_buffer("random_ids", rand_ids)
            # sinusoidal IDs are computed on-the-fly

            self.id_to_node = nn.Linear(config.d_node_id, d, bias=False)
            self.id_src_proj = nn.Linear(config.d_node_id, d, bias=False)
            self.id_tgt_proj = nn.Linear(config.d_node_id, d, bias=False)

    def _get_local_indices(
        self, batch: Optional[torch.Tensor], num_nodes: int, device: torch.device
    ) -> torch.Tensor:
        """Compute per-graph local node indices (0-based within each graph)."""
        if batch is None:
            return torch.arange(num_nodes, device=device)
        counts = torch.bincount(batch)
        graph_start = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.long), counts.cumsum(0)[:-1]]
        )
        return torch.arange(num_nodes, device=device) - graph_start[batch]

    def _get_node_ids(
        self,
        local_idx: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Return node identifier vectors (N, d_node_id)."""
        max_idx = (
            self.id_embed.num_embeddings - 1
            if hasattr(self, "id_embed")
            else self.random_ids.size(0) - 1
            if hasattr(self, "random_ids")
            else 9999
        )
        clamped = local_idx.clamp(max=max_idx)

        if self.id_type == "learned":
            u = self.id_embed(clamped)
        elif self.id_type == "random":
            u = self.random_ids[clamped]
        else:  # sinusoidal
            u = _sinusoidal_ids(int(clamped.max().item()) + 1, self.d_node_id, device)
            u = u[clamped]

        if self.training and self.id_noise_scale > 0:
            u = u + torch.randn_like(u) * self.id_noise_scale
        return u

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N, d_node_in) node features
            edge_index: (2, M) directed edge indices
            edge_attr: (M, d_edge_in) edge features, optional
            batch: (N,) graph-level batch indices

        Returns:
            z_V: (N, d_model) node tokens
            z_E: (M, d_model) directed edge tokens
        """
        N = x.size(0)
        M = edge_index.size(1)
        device = x.device

        src, tgt = edge_index[0], edge_index[1]

        # ---- node tokens ----
        z_V = self.node_proj(x) + self.node_type_embed

        if self.use_degree_encoding:
            in_deg = torch.zeros(N, dtype=torch.long, device=device)
            out_deg = torch.zeros(N, dtype=torch.long, device=device)
            ones = torch.ones(M, dtype=torch.long, device=device)
            in_deg.scatter_add_(0, tgt, ones)
            out_deg.scatter_add_(0, src, ones)
            max_in = self.in_degree_embed.num_embeddings - 1
            max_out = self.out_degree_embed.num_embeddings - 1
            z_V = z_V + self.in_degree_embed(in_deg.clamp(max=max_in))
            z_V = z_V + self.out_degree_embed(out_deg.clamp(max=max_out))

        # ---- node IDs ----
        u = None
        if self.use_node_ids:
            local_idx = self._get_local_indices(batch, N, device)
            u = self._get_node_ids(local_idx, device)
            z_V = z_V + self.id_to_node(u)

        # ---- edge tokens ----
        if self.has_edge_features and edge_attr is not None:
            z_E = self.edge_proj(edge_attr) + self.edge_type_embed
        else:
            z_E = self.edge_type_embed.unsqueeze(0).expand(M, -1).clone()

        if self.use_node_ids and u is not None:
            z_E = z_E + self.id_src_proj(u[src]) + self.id_tgt_proj(u[tgt])

        return z_V, z_E
