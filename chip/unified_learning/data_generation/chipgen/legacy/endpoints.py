"""Endpoint sampling with hierarchical locality."""

import torch
from typing import Tuple

from .config import HierarchyConfig, DegreeConfig


def sample_endpoints(
    perm: torch.Tensor,
    invperm: torch.Tensor,
    deg: torch.Tensor,
    level: torch.Tensor,
    N_inst: int,
    hier_cfg: HierarchyConfig,
    deg_cfg: DegreeConfig,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample endpoints for all nets using hierarchical clustering.

    For normal nets: sample from contiguous cluster at chosen hierarchy level
    For huge nets: sample globally to avoid clustering artifacts

    Args:
        perm: [N_inst] random permutation defining clusters
        invperm: [N_inst] inverse permutation
        deg: [N_net] degree for each net
        level: [N_net] hierarchy level for each net
        N_inst: total number of instances
        hier_cfg: hierarchy configuration
        deg_cfg: degree configuration (for huge net threshold)
        device: device for tensors

    Returns:
        (inst_id_per_edge, net_id_per_edge):
        - inst_id_per_edge: [E] instance ID for each edge
        - net_id_per_edge: [E] net ID for each edge
    """
    N_net = len(deg)
    E = deg.sum().item()
    k = hier_cfg.k

    # Identify huge nets (different sampling strategy)
    huge_threshold = deg_cfg.medium_deg_range[1]
    is_huge = deg > huge_threshold

    # Create edge-level tensors
    net_id_per_edge = torch.repeat_interleave(torch.arange(N_net, device=device), deg)

    # Gather level per edge
    level_per_edge = level[net_id_per_edge]

    # Gather huge flag per edge
    is_huge_per_edge = is_huge[net_id_per_edge]

    # Initialize instance IDs
    inst_id_per_edge = torch.zeros(E, dtype=torch.int64, device=device)

    # --- Sample normal nets (hierarchical clustering) ---
    normal_mask = ~is_huge_per_edge

    if normal_mask.any():
        n_normal = normal_mask.sum().item()

        # For each edge, sample a cluster at its level
        # First, sample a random "seed" instance to determine cluster
        # (Alternative: sample cluster ID directly, but this is simpler)

        # Sample random positions in permutation for each edge
        level_normal = level_per_edge[normal_mask]
        group_size = k ** (level_normal + 1)

        # Compute number of clusters at each level
        n_clusters = (N_inst + group_size - 1) // group_size  # ceiling division

        # Sample cluster ID for each edge
        # Use randint with proper broadcasting
        cluster_ids = []
        for i in range(n_normal):
            gs = group_size[i].item()
            nc = (N_inst + gs - 1) // gs
            cid = torch.randint(0, nc, (1,), device=device).item()
            cluster_ids.append(cid)

        cluster_id_per_edge = torch.tensor(cluster_ids, dtype=torch.int64, device=device)

        # Compute cluster ranges
        start = cluster_id_per_edge * group_size
        end = torch.minimum((cluster_id_per_edge + 1) * group_size, torch.tensor(N_inst, device=device))
        cluster_size = end - start

        # Sample offset within cluster
        # offset = randint(0, cluster_size) for each edge
        offsets = []
        for i in range(n_normal):
            cs = cluster_size[i].item()
            offset = torch.randint(0, cs, (1,), device=device).item()
            offsets.append(offset)

        offset_per_edge = torch.tensor(offsets, dtype=torch.int64, device=device)

        # Map to instance ID via permutation
        pos_in_perm = start + offset_per_edge
        pos_in_perm = torch.clamp(pos_in_perm, 0, N_inst - 1)
        inst_id_normal = perm[pos_in_perm]

        inst_id_per_edge[normal_mask] = inst_id_normal

    # --- Sample huge nets (global, with uniqueness) ---
    if is_huge_per_edge.any():
        n_huge = is_huge_per_edge.sum().item()

        # For huge nets, sample globally
        # To ensure some uniqueness per net, we oversample and deduplicate per net
        # But for simplicity in vectorized code, just sample globally

        # Get net IDs for huge edges
        huge_net_ids = net_id_per_edge[is_huge_per_edge]

        # Sample globally for each huge edge
        inst_id_huge = torch.randint(0, N_inst, (n_huge,), device=device)

        inst_id_per_edge[is_huge_per_edge] = inst_id_huge

    return inst_id_per_edge, net_id_per_edge


def sample_endpoints_optimized(
    perm: torch.Tensor,
    invperm: torch.Tensor,
    deg: torch.Tensor,
    level: torch.Tensor,
    N_inst: int,
    hier_cfg: HierarchyConfig,
    deg_cfg: DegreeConfig,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized vectorized endpoint sampling (fully GPU, no loops).

    This version is faster but slightly more complex.

    Args:
        perm: [N_inst] random permutation defining clusters
        invperm: [N_inst] inverse permutation
        deg: [N_net] degree for each net
        level: [N_net] hierarchy level for each net
        N_inst: total number of instances
        hier_cfg: hierarchy configuration
        deg_cfg: degree configuration
        device: device for tensors

    Returns:
        (inst_id_per_edge, net_id_per_edge)
    """
    N_net = len(deg)
    E = deg.sum().item()
    k = hier_cfg.k

    # Edge-level net IDs
    net_id_per_edge = torch.repeat_interleave(torch.arange(N_net, device=device), deg)

    # Edge-level level
    level_per_edge = level[net_id_per_edge]

    # Huge net mask
    huge_threshold = deg_cfg.medium_deg_range[1]
    is_huge = deg > huge_threshold
    is_huge_per_edge = is_huge[net_id_per_edge]

    # --- Determine which edges use global vs hierarchical sampling ---
    # Sample a fraction of edges globally to improve Rent's exponent
    use_global = torch.rand(E, device=device) < hier_cfg.global_edge_fraction
    use_hierarchical = ~use_global & ~is_huge_per_edge

    # --- Normal nets (hierarchical sampling) ---
    # Compute group size for each edge
    group_size = k ** (level_per_edge + 1)

    # Sample cluster ID per edge
    n_clusters_per_edge = (N_inst + group_size - 1) // group_size

    # Sample uniform [0, 1) and scale to cluster ID
    cluster_id_per_edge = (torch.rand(E, device=device) * n_clusters_per_edge.float()).long()
    cluster_id_per_edge = torch.minimum(cluster_id_per_edge, n_clusters_per_edge - 1)
    cluster_id_per_edge = torch.clamp(cluster_id_per_edge, min=0)

    # Compute cluster boundaries
    start = cluster_id_per_edge * group_size
    end = torch.minimum((cluster_id_per_edge + 1) * group_size, torch.tensor(N_inst, device=device))
    cluster_size = end - start

    # Sample offset within cluster
    offset_per_edge = (torch.rand(E, device=device) * cluster_size.float()).long()
    offset_per_edge = torch.minimum(offset_per_edge, cluster_size - 1)
    offset_per_edge = torch.clamp(offset_per_edge, min=0)

    # Map to instance ID (for hierarchical edges)
    pos_in_perm = start + offset_per_edge
    pos_in_perm = torch.clamp(pos_in_perm, 0, N_inst - 1)
    inst_id_per_edge = perm[pos_in_perm]

    # --- Override with global sampling for designated edges ---
    # This includes both use_global edges and huge nets
    sample_globally = use_global | is_huge_per_edge
    if sample_globally.any():
        inst_id_global = torch.randint(0, N_inst, (E,), device=device)
        inst_id_per_edge = torch.where(sample_globally, inst_id_global, inst_id_per_edge)

    # --- Fix isolated instances: ensure every instance appears at least once ---
    inst_id_per_edge, net_id_per_edge = ensure_no_isolated_instances(
        inst_id_per_edge, net_id_per_edge, N_inst, N_net, device
    )

    return inst_id_per_edge, net_id_per_edge


def ensure_no_isolated_instances(
    inst_id_per_edge: torch.Tensor,
    net_id_per_edge: torch.Tensor,
    N_inst: int,
    N_net: int,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ensure no instances are isolated by adding them to random nets if needed.

    Args:
        inst_id_per_edge: [E] instance IDs
        net_id_per_edge: [E] net IDs
        N_inst: total number of instances
        N_net: total number of nets
        device: device

    Returns:
        Updated (inst_id_per_edge, net_id_per_edge) with no isolated instances
    """
    # Find which instances are connected
    connected_insts = torch.unique(inst_id_per_edge)
    all_insts = torch.arange(N_inst, device=device)

    # Find isolated instances
    connected_mask = torch.zeros(N_inst, dtype=torch.bool, device=device)
    connected_mask[connected_insts] = True
    isolated_mask = ~connected_mask
    isolated_insts = all_insts[isolated_mask]

    if len(isolated_insts) == 0:
        return inst_id_per_edge, net_id_per_edge

    # For each isolated instance, add it to a random net
    n_isolated = len(isolated_insts)
    random_nets = torch.randint(0, N_net, (n_isolated,), device=device)

    # Create new edges
    new_inst_ids = isolated_insts
    new_net_ids = random_nets

    # Concatenate with existing edges
    inst_id_per_edge = torch.cat([inst_id_per_edge, new_inst_ids], dim=0)
    net_id_per_edge = torch.cat([net_id_per_edge, new_net_ids], dim=0)

    return inst_id_per_edge, net_id_per_edge
