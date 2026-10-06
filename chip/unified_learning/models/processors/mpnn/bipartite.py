from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import scatter
from torch_geometric.data import HeteroData


class BipartiteGCNConv(nn.Module):
    """
    Simple bipartite GCN convolution.
    
    For bipartite graphs (src -> dst), this applies:
    - Linear transformation on source features
    - Aggregation to destination nodes
    """
    
    def __init__(self, src_dim: int, dst_dim: int, out_dim: int):
        super().__init__()
        self.src_dim = src_dim
        self.dst_dim = dst_dim
        self.out_dim = out_dim
        
        # Linear transformation for source features
        self.lin_src = nn.Linear(src_dim, out_dim, bias=False)
        
        # Optional: linear transformation for destination features (for residual)
        if dst_dim == out_dim:
            self.lin_dst = nn.Identity()
        else:
            self.lin_dst = nn.Linear(dst_dim, out_dim, bias=False)
    
    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_src: [N_src, src_dim] source node features
            x_dst: [N_dst, dst_dim] destination node features
            edge_index: [2, E] edge index where edge_index[0] are source indices,
                       edge_index[1] are destination indices
        
        Returns:
            output: [N_dst, out_dim] output features
        """
        row, col = edge_index
        
        # Transform source features
        x_src_transformed = self.lin_src(x_src)  # [N_src, out_dim]
        
        # Aggregate messages: for each destination node, sum incoming source features
        out = scatter(x_src_transformed[row], col, dim=0, dim_size=x_dst.shape[0], reduce='mean')
        
        # Add transformed destination features (residual-like connection)
        out = out + self.lin_dst(x_dst)
        
        return out


class InstNetBlock(nn.Module):
    """
    Bipartite message-passing block:
      - inst -> net (GCN + GATv2)
      - net  -> inst (GCN + GATv2)

    This block assumes:
      - inst_x: [N_inst, inst_dim]
      - net_x:  [N_net,  net_dim]
      - edge_index_inst_to_net: [2, E], edges from inst (source) to net (target)
      - edge_attr: [E, edge_dim]
    """

    def __init__(
        self,
        inst_dim: int,
        net_dim: int,
        edge_dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        act: Optional[nn.Module] = None,
        use_gcn: bool = True,  # Option to disable GCN (use only GAT)
    ) -> None:
        super().__init__()
        self.act = act if act is not None else nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.use_gcn = use_gcn

        # inst -> net
        if use_gcn:
            self.gcn_inst_to_net = BipartiteGCNConv(inst_dim, net_dim, net_dim)
        self.gat_inst_to_net = GATv2Conv(
            (inst_dim, net_dim),
            net_dim,
            heads=heads,
            edge_dim=edge_dim,
            concat=False,  # keep output dim = net_dim
            dropout=dropout,
            add_self_loops=False,  # No self-loops for bipartite graphs
        )

        # net -> inst
        if use_gcn:
            self.gcn_net_to_inst = BipartiteGCNConv(net_dim, inst_dim, inst_dim)
        self.gat_net_to_inst = GATv2Conv(
            (net_dim, inst_dim),
            inst_dim,
            heads=heads,
            edge_dim=edge_dim,
            concat=False,
            dropout=dropout,
            add_self_loops=False,  # No self-loops for bipartite graphs
        )

    def forward(
        self,
        inst_x: torch.Tensor,
        net_x: torch.Tensor,
        edge_index_inst_to_net: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # -------- inst -> net --------
        # Compute message (reuse variable to save memory)
        if self.use_gcn:
            net_msg = self.gcn_inst_to_net(inst_x, net_x, edge_index_inst_to_net)
            # Add GAT output in-place (safe because net_msg is newly created)
            net_msg.add_(self.gat_inst_to_net(
                (inst_x, net_x),
                edge_index_inst_to_net,
                edge_attr,
            ))
        else:
            # GAT only
            net_msg = self.gat_inst_to_net(
                (inst_x, net_x),
                edge_index_inst_to_net,
                edge_attr,
            )
        
        # Apply activation and dropout
        net_msg = self.act(net_msg)
        net_msg = self.dropout(net_msg)
        # Residual connection: DO NOT use in-place on input tensor (breaks gradients)
        net_x = net_x + net_msg

        # -------- net -> inst --------
        edge_index_net_to_inst = edge_index_inst_to_net.flip(0)

        # Reuse net_msg variable for inst_msg (original net_msg no longer needed)
        if self.use_gcn:
            inst_msg = self.gcn_net_to_inst(net_x, inst_x, edge_index_net_to_inst)
            # In-place add is safe here because inst_msg is newly created
            inst_msg.add_(self.gat_net_to_inst(
                (net_x, inst_x),
                edge_index_net_to_inst,
                edge_attr,
            ))
        else:
            # GAT only
            inst_msg = self.gat_net_to_inst(
                (net_x, inst_x),
                edge_index_net_to_inst,
                edge_attr,
            )
        
        inst_msg = self.act(inst_msg)
        inst_msg = self.dropout(inst_msg)
        # Residual connection
        inst_x = inst_x + inst_msg

        return inst_x, net_x

