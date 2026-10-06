"""Masked Geometry Modeling (MGM) modules for self-supervised pretraining.

MGM masks geometric information (coordinates, sizes, edge attributes) and forces reconstruction.
This makes embeddings encode geometric neighborhoods, not just topology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torch_geometric.data import Data


class GeometryMasker:
    """
    Masks geometric information for MGM pretraining.
    
    Masking strategies:
    - Node coordinates (x, y)
    - Node sizes (w, h) / shape descriptors
    - Edge attributes (distances, angles)
    - Local frame features (relative vectors)
    """
    
    def __init__(
        self,
        mask_coord_prob: float = 0.15,
        mask_size_prob: float = 0.15,
        mask_edge_prob: float = 0.15,
        coord_mask_value: float = 0.0,
        size_mask_value: float = 0.0,
        edge_mask_value: float = 0.0,
    ):
        """
        Args:
            mask_coord_prob: Probability of masking node coordinates
            mask_size_prob: Probability of masking node sizes
            mask_edge_prob: Probability of masking edge attributes
            coord_mask_value: Value to use for masked coordinates
            size_mask_value: Value to use for masked sizes
            edge_mask_value: Value to use for masked edges
        """
        self.mask_coord_prob = mask_coord_prob
        self.mask_size_prob = mask_size_prob
        self.mask_edge_prob = mask_edge_prob
        self.coord_mask_value = coord_mask_value
        self.size_mask_value = size_mask_value
        self.edge_mask_value = edge_mask_value
    
    def mask_data(
        self,
        data: Data,
        coord_indices: Tuple[int, int] = (0, 2),  # Slice of coordinate features
        size_indices: Tuple[int, int] = (2, 3),  # Slice of size features (elevation in pos[:, 2])
        use_pos: bool = True,  # If True, use data.pos instead of data.x
    ) -> Tuple[Data, Dict[str, torch.Tensor]]:
        """
        Mask geometric information in the data.
        
        Args:
            data: PyG Data object
            coord_indices: (start, end) indices for coordinate features
                          If use_pos=True: (0, 2) for pos[:, 0:2] (x, y)
                          If use_pos=False: indices in data.x
            size_indices: (start, end) indices for size features
                         If use_pos=True: (2, 3) for pos[:, 2] (elevation/z)
                         If use_pos=False: indices in data.x
            use_pos: If True, mask data.pos; if False, mask data.x
        
        Returns:
            masked_data: Data with masked features
            targets: Dictionary of reconstruction targets
        """
        # Determine feature source
        if use_pos:
            if not hasattr(data, 'pos') or data.pos is None:
                raise ValueError("use_pos=True but data.pos is None")
            features = data.pos  # [num_nodes, 3] for (x, y, elevation)
            num_nodes = features.shape[0]
            device = features.device
        else:
            if not hasattr(data, 'x') or data.x is None:
                raise ValueError("use_pos=False but data.x is None")
            features = data.x
            num_nodes = features.shape[0]
            device = features.device
        
        # Clone data to avoid in-place modifications
        masked_data = data.clone()
        targets = {}
        
        # 1. Mask node coordinates (x, y from pos[:, 0:2])
        coord_mask = torch.rand(num_nodes, device=device) < self.mask_coord_prob
        if coord_mask.any():
            # Store original coordinates for reconstruction
            coord_start, coord_end = coord_indices
            if use_pos:
                targets['coords'] = masked_data.pos[coord_mask, coord_start:coord_end].clone()
            else:
                targets['coords'] = masked_data.x[coord_mask, coord_start:coord_end].clone()
            targets['coord_mask'] = coord_mask
            
            # Mask coordinates
            if use_pos:
                masked_data.pos[coord_mask, coord_start:coord_end] = self.coord_mask_value
            else:
                masked_data.x[coord_mask, coord_start:coord_end] = self.coord_mask_value
        
        # 2. Mask node sizes/elevation (elevation from pos[:, 2])
        # For Perlin terrain: elevation comes from raw DEM values during graph conversion
        # DEM (rows, cols) → mesh vertices V = [x=j, y=i, z=dem[i,j]] → data.pos[:, 2] = elevation
        size_mask = torch.rand(num_nodes, device=device) < self.mask_size_prob
        if size_mask.any():
            # Store original sizes for reconstruction
            size_start, size_end = size_indices
            if use_pos:
                # Extract elevation from pos[:, 2] (or pos[:, size_start:size_end] for multi-dim)
                targets['sizes'] = masked_data.pos[size_mask, size_start:size_end].clone()
            else:
                # Extract from data.x if not using pos
                targets['sizes'] = masked_data.x[size_mask, size_start:size_end].clone()
            targets['size_mask'] = size_mask
            
            # Mask sizes (set to mask_value, typically 0.0)
            if use_pos:
                masked_data.pos[size_mask, size_start:size_end] = self.size_mask_value
            else:
                masked_data.x[size_mask, size_start:size_end] = self.size_mask_value
        
        # 3. Mask edge attributes
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            num_edges = data.edge_attr.shape[0]
            edge_mask = torch.rand(num_edges, device=device) < self.mask_edge_prob
            if edge_mask.any():
                # Store original edge attributes
                targets['edge_attrs'] = data.edge_attr[edge_mask].clone()
                targets['edge_mask'] = edge_mask
                
                # Mask edge attributes
                masked_data.edge_attr[edge_mask] = self.edge_mask_value
        
        # 4. Compute pairwise distances for relative reconstruction (more stable)
        if 'coords' in targets:
            # Get neighbors for masked nodes
            edge_index = data.edge_index
            src, dst = edge_index
            
            # Find edges involving masked nodes
            masked_node_edges = coord_mask[src] | coord_mask[dst]
            
            if masked_node_edges.any():
                # Extract coordinates (x, y only, not elevation)
                coord_start, coord_end = coord_indices
                if use_pos:
                    coords_orig = data.pos[:, coord_start:coord_end]  # [num_nodes, 2] for (x, y)
                else:
                    coords_orig = data.x[:, coord_start:coord_end]
                
                # Compute pairwise distances and directions for masked node edges
                edge_vecs = coords_orig[dst] - coords_orig[src]  # [num_edges, 2]
                edge_dists = torch.norm(edge_vecs, dim=1, keepdim=True)  # [num_edges, 1]
                edge_dirs = edge_vecs / (edge_dists + 1e-8)  # [num_edges, 2] - unit vectors
                
                # Store for reconstruction
                targets['pairwise_distances'] = edge_dists[masked_node_edges]
                targets['pairwise_directions'] = edge_dirs[masked_node_edges]
                targets['pairwise_mask'] = masked_node_edges
                targets['pairwise_src'] = src[masked_node_edges]
                targets['pairwise_dst'] = dst[masked_node_edges]
        
        return masked_data, targets


class MGMDecoder(nn.Module):
    """
    Decoder for Masked Geometry Modeling.
    
    Reconstructs masked geometric information from node embeddings.
    Predicts relative geometry (distances, directions) for stability.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        coord_dim: int = 2,
        size_dim: int = 8,
        edge_dim: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Dimension of node embeddings
            coord_dim: Dimension of coordinate outputs
            size_dim: Dimension of size outputs
            edge_dim: Dimension of edge attribute outputs
            num_layers: Number of MLP layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.size_dim = size_dim
        self.edge_dim = edge_dim
        
        # Coordinate decoder (reconstructs coordinates)
        self.coord_decoder = self._make_mlp(hidden_dim, coord_dim, num_layers, dropout)
        
        # Size decoder (reconstructs sizes/shape descriptors)
        self.size_decoder = self._make_mlp(hidden_dim, size_dim, num_layers, dropout)
        
        # Edge decoder (reconstructs edge attributes)
        # Takes concatenated src and dst node embeddings
        self.edge_decoder = self._make_mlp(hidden_dim * 2, edge_dim, num_layers, dropout)
        
        # Pairwise distance decoder (more stable than absolute coordinates)
        self.distance_decoder = self._make_mlp(hidden_dim * 2, 1, num_layers, dropout)
        
        # Pairwise direction decoder (unit vectors to neighbors)
        self.direction_decoder = self._make_mlp(hidden_dim * 2, 2, num_layers, dropout)
    
    def _make_mlp(self, input_dim: int, output_dim: int, num_layers: int, dropout: float) -> nn.Module:
        """Create MLP with specified architecture."""
        layers = []
        dim = input_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(dim, self.hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = self.hidden_dim
        
        layers.append(nn.Linear(dim, output_dim))
        return nn.Sequential(*layers)
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        data: Data,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode masked geometry from node embeddings.
        
        Args:
            node_embeddings: [num_nodes, hidden_dim] node embeddings from encoder
            targets: Dictionary of reconstruction targets from GeometryMasker
            data: Original PyG Data object (for edge_index)
        
        Returns:
            predictions: Dictionary of predictions for each masked element
        """
        predictions = {}
        
        # 1. Reconstruct coordinates
        if 'coord_mask' in targets:
            coord_mask = targets['coord_mask']
            masked_embeddings = node_embeddings[coord_mask]
            pred_coords = self.coord_decoder(masked_embeddings)
            predictions['coords'] = pred_coords
        
        # 2. Reconstruct sizes
        if 'size_mask' in targets:
            size_mask = targets['size_mask']
            masked_embeddings = node_embeddings[size_mask]
            pred_sizes = self.size_decoder(masked_embeddings)
            predictions['sizes'] = pred_sizes
        
        # 3. Reconstruct edge attributes
        if 'edge_mask' in targets:
            edge_mask = targets['edge_mask']
            edge_index = data.edge_index
            
            # Get src and dst embeddings for masked edges
            src_emb = node_embeddings[edge_index[0, edge_mask]]
            dst_emb = node_embeddings[edge_index[1, edge_mask]]
            
            # Concatenate and decode
            edge_emb = torch.cat([src_emb, dst_emb], dim=1)
            pred_edge_attrs = self.edge_decoder(edge_emb)
            predictions['edge_attrs'] = pred_edge_attrs
        
        # 4. Reconstruct pairwise distances (more stable than absolute coords)
        if 'pairwise_mask' in targets:
            pairwise_mask = targets['pairwise_mask']
            src_nodes = targets['pairwise_src']
            dst_nodes = targets['pairwise_dst']
            
            # Get embeddings
            src_emb = node_embeddings[src_nodes]
            dst_emb = node_embeddings[dst_nodes]
            
            # Decode distance and direction
            pair_emb = torch.cat([src_emb, dst_emb], dim=1)
            pred_distances = self.distance_decoder(pair_emb)
            pred_directions_raw = self.direction_decoder(pair_emb)
            
            # Normalize directions to unit vectors
            pred_directions = F.normalize(pred_directions_raw, p=2, dim=1)
            
            predictions['pairwise_distances'] = pred_distances
            predictions['pairwise_directions'] = pred_directions
        
        return predictions


def compute_mgm_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    coord_weight: float = 1.0,
    size_weight: float = 1.0,
    edge_weight: float = 1.0,
    distance_weight: float = 2.0,  # Higher weight for relative geometry (more important)
    direction_weight: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute MGM reconstruction loss.
    
    Args:
        predictions: Dictionary of predictions from MGMDecoder
        targets: Dictionary of targets from GeometryMasker
        coord_weight: Weight for coordinate reconstruction loss
        size_weight: Weight for size reconstruction loss
        edge_weight: Weight for edge attribute reconstruction loss
        distance_weight: Weight for pairwise distance loss (more stable)
        direction_weight: Weight for pairwise direction loss
    
    Returns:
        loss: Total weighted loss
        loss_dict: Dictionary of individual loss components
    """
    losses = {}
    total_loss = 0.0
    
    # 1. Coordinate reconstruction loss
    if 'coords' in predictions:
        coord_loss = F.mse_loss(predictions['coords'], targets['coords'])
        losses['coord_loss'] = coord_loss.item()
        total_loss = total_loss + coord_weight * coord_loss
    
    # 2. Size reconstruction loss
    if 'sizes' in predictions:
        size_loss = F.mse_loss(predictions['sizes'], targets['sizes'])
        losses['size_loss'] = size_loss.item()
        total_loss = total_loss + size_weight * size_loss
    
    # 3. Edge attribute reconstruction loss
    if 'edge_attrs' in predictions:
        edge_loss = F.mse_loss(predictions['edge_attrs'], targets['edge_attrs'])
        losses['edge_loss'] = edge_loss.item()
        total_loss = total_loss + edge_weight * edge_loss
    
    # 4. Pairwise distance loss (more stable than absolute coordinates)
    if 'pairwise_distances' in predictions:
        dist_loss = F.mse_loss(predictions['pairwise_distances'], targets['pairwise_distances'])
        losses['distance_loss'] = dist_loss.item()
        total_loss = total_loss + distance_weight * dist_loss
    
    # 5. Pairwise direction loss (unit vectors)
    if 'pairwise_directions' in predictions:
        # Cosine similarity loss (1 - cosine_sim)
        cosine_sim = F.cosine_similarity(
            predictions['pairwise_directions'],
            targets['pairwise_directions'],
            dim=1
        )
        direction_loss = (1.0 - cosine_sim).mean()
        losses['direction_loss'] = direction_loss.item()
        total_loss = total_loss + direction_weight * direction_loss
    
    losses['total_loss'] = total_loss.item()
    
    return total_loss, losses
