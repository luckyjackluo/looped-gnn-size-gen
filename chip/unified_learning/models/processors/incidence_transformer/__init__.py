"""
Incidence Bipartite Transformer — a deep graph processor that combines:

    - Node tokens + directed edge tokens (TokenGT-style)
    - Incidence bipartite sparse attention (V→E / E→V)
    - Optional gated global branch (memory tokens or dense self-attention)
    - L independent layers, each with its own parameters

Usage:
    from unified_learning.models.processors.incidence_transformer import (
        IncidenceTransformerConfig,
        IncidenceTransformerProcessor,
    )

    config = IncidenceTransformerConfig(d_node_in=7, d_edge_in=4, d_model=256, num_layers=12)
    processor = IncidenceTransformerProcessor(config)

    result = processor(x=data.x, edge_index=data.edge_index,
                       edge_attr=data.edge_attr, batch=data.batch)
    z_V, z_E = result["z_V"], result["z_E"]
"""

from .config import IncidenceTransformerConfig
from .tokenizer import GraphTokenizer
from .attention import NodeToEdgeAttention, EdgeToNodeAttention, PreNormFFN
from .global_branch import GlobalMemoryBranch, DenseGlobalBranch, RouterGate
from .processor import IncidenceBlock, IncidenceTransformerProcessor

__all__ = [
    "IncidenceTransformerConfig",
    "GraphTokenizer",
    "NodeToEdgeAttention",
    "EdgeToNodeAttention",
    "PreNormFFN",
    "GlobalMemoryBranch",
    "DenseGlobalBranch",
    "RouterGate",
    "IncidenceBlock",
    "IncidenceTransformerProcessor",
]
