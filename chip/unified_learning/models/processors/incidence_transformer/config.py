"""
Configuration for the Incidence Bipartite Transformer.

Architecture:
    Deep stack of independent transformer layers
    + node tokens
    + directed edge tokens
    + incidence bipartite sparse attention (V→E / E→V)
    + optional gated global branch

References:
    - TokenGT (Kim et al., NeurIPS 2022): node/edge tokenization
    - EGT (Hussain et al., 2021): explicit edge-state handling
    - Graphormer (Ying et al., NeurIPS 2021): structural bias
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class IncidenceTransformerConfig:
    """Full configuration for the Incidence Bipartite Transformer processor."""

    # ---- core dimensions ----
    d_model: int = 256
    d_node_in: int = 7
    d_edge_in: int = 4

    # ---- depth ----
    num_layers: int = 12

    # ---- attention ----
    num_heads_local: int = 8
    num_heads_global: int = 4
    dropout: float = 0.1

    # ---- feed-forward ----
    ffn_multiplier: float = 4.0
    ffn_activation: Literal["gelu", "silu", "relu"] = "gelu"

    # ---- global branch ----
    use_global_branch: bool = True
    global_branch_type: Literal["memory", "dense"] = "memory"
    num_memory_tokens: int = 16
    global_branch_every_n: int = 1

    # ---- node identifiers ----
    use_node_ids: bool = True
    d_node_id: int = 32
    max_nodes: int = 10000
    id_type: Literal["learned", "random", "sinusoidal"] = "learned"
    id_noise_scale: float = 0.0

    # ---- structural encodings ----
    use_degree_encoding: bool = True
    max_in_degree: int = 128
    max_out_degree: int = 128

    # ---- edge attention bias (Graphormer-style) ----
    use_edge_bias: bool = False
    edge_bias_dim: int = 0

    # ---- stability ----
    residual_scale: float = 1.0
    use_activation_checkpointing: bool = False

    def __post_init__(self):
        if self.d_model % self.num_heads_local != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads_local ({self.num_heads_local})"
            )
        if self.use_global_branch and self.d_model % self.num_heads_global != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads_global ({self.num_heads_global})"
            )
