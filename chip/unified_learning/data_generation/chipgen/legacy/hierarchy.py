"""Hierarchical clustering via contiguous permutation."""

import torch
from typing import Tuple

from .config import HierarchyConfig


def make_perm(N_inst: int, seed: int, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a random permutation and its inverse for hierarchical clustering.

    The permutation defines contiguous clusters at each hierarchy level:
    - At level ℓ, cluster size = k^(ℓ+1)
    - Cluster c at level ℓ occupies indices [c*group, (c+1)*group) in perm

    Args:
        N_inst: Number of instances
        seed: Random seed for reproducibility
        device: Device for tensors

    Returns:
        (perm, invperm):
        - perm: [N_inst] random permutation of instance IDs
        - invperm: [N_inst] inverse permutation (invperm[perm[i]] = i)
    """
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        perm = torch.randperm(N_inst, generator=generator, device=device)
    else:
        perm = torch.randperm(N_inst, device=device)

    # Compute inverse permutation
    invperm = torch.empty_like(perm)
    invperm[perm] = torch.arange(N_inst, device=device)

    return perm, invperm


def sample_levels(N_net: int, hier_cfg: HierarchyConfig, device: str = "cuda") -> torch.Tensor:
    """
    Sample hierarchy level for each net using geometric distribution.

    Lower levels (0, 1) are local nets, higher levels are more global.
    Distribution: p(level=ℓ) ∝ decay^ℓ (normalized)

    Args:
        N_net: Number of nets
        hier_cfg: Hierarchy configuration
        device: Device for tensors

    Returns:
        level: [N_net] hierarchy level for each net (int64)
    """
    # Compute probabilities for each level (geometric decay)
    max_level = hier_cfg.max_level
    decay = hier_cfg.level_decay

    # p(level=l) = decay^l / Z where Z = sum_l decay^l
    levels_range = torch.arange(max_level + 1, dtype=torch.float32, device=device)
    probs = torch.pow(decay, levels_range)
    probs = probs / probs.sum()  # normalize

    # Sample levels
    level = torch.multinomial(probs, num_samples=N_net, replacement=True)

    return level


def get_cluster_id(inst_id: torch.Tensor, invperm: torch.Tensor, level: int, k: int) -> torch.Tensor:
    """
    Get cluster ID for given instance(s) at specified hierarchy level.

    Args:
        inst_id: [*] instance ID(s)
        invperm: [N_inst] inverse permutation
        level: hierarchy level
        k: branching factor

    Returns:
        cluster_id: [*] cluster ID(s) at given level
    """
    group_size = k ** (level + 1)
    pos_in_perm = invperm[inst_id]
    cluster_id = pos_in_perm // group_size
    return cluster_id


def get_cluster_range(cluster_id: torch.Tensor, level: int, k: int, N_inst: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the [start, end) range of instance positions for a cluster.

    Args:
        cluster_id: [*] cluster ID(s)
        level: hierarchy level
        k: branching factor
        N_inst: total number of instances

    Returns:
        (start, end): [*] start and end positions in permutation
    """
    group_size = k ** (level + 1)
    start = cluster_id * group_size
    end = torch.minimum((cluster_id + 1) * group_size, torch.tensor(N_inst, device=cluster_id.device))
    return start, end
