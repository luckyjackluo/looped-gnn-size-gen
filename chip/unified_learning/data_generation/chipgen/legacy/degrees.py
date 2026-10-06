"""Net degree sampling with heavy-tail mixture."""

import torch
from typing import Dict

from .config import DegreeConfig


def sample_degrees_to_target(
    N_inst: int,
    E_target: int,
    deg_cfg: DegreeConfig,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Sample net degrees until total endpoints reaches E_target.

    Uses a mixture of small, medium, and huge degree distributions.
    Ensures total endpoints = E_target by adjusting the last degree.

    Args:
        N_inst: Number of instances (for capping huge degrees)
        E_target: Target total number of endpoints
        deg_cfg: Degree configuration
        device: Device for tensors

    Returns:
        deg: [N_net] degree for each net (all >= 2)
    """
    # Cap huge degree based on N_inst
    max_huge_deg = min(
        deg_cfg.huge_deg_range[1],
        int(deg_cfg.huge_deg_cap_factor * N_inst)
    )
    max_huge_deg = max(max_huge_deg, deg_cfg.huge_deg_range[0])

    degrees_list = []
    total_endpoints = 0

    # Batch size for sampling (for efficiency)
    batch_size = 1000

    while total_endpoints < E_target:
        # Sample mixture components
        n_remaining = max(batch_size, E_target - total_endpoints)
        batch_size_actual = min(batch_size, (E_target - total_endpoints) // 2 + 100)

        mixture = torch.rand(batch_size_actual, device=device)

        # Allocate to small/medium/huge
        is_small = mixture < deg_cfg.p_small
        is_medium = (mixture >= deg_cfg.p_small) & (mixture < deg_cfg.p_small + deg_cfg.p_medium)
        is_huge = mixture >= (deg_cfg.p_small + deg_cfg.p_medium)

        deg_batch = torch.zeros(batch_size_actual, dtype=torch.int64, device=device)

        # Sample small degrees
        n_small = is_small.sum().item()
        if n_small > 0:
            deg_batch[is_small] = torch.randint(
                deg_cfg.small_deg_range[0],
                deg_cfg.small_deg_range[1] + 1,
                (n_small,),
                device=device
            )

        # Sample medium degrees (log-uniform)
        n_medium = is_medium.sum().item()
        if n_medium > 0:
            log_min = torch.log(torch.tensor(float(deg_cfg.medium_deg_range[0]), device=device))
            log_max = torch.log(torch.tensor(float(deg_cfg.medium_deg_range[1]), device=device))
            log_deg = torch.rand(n_medium, device=device) * (log_max - log_min) + log_min
            deg_batch[is_medium] = torch.exp(log_deg).long().clamp(
                deg_cfg.medium_deg_range[0],
                deg_cfg.medium_deg_range[1]
            )

        # Sample huge degrees (log-uniform)
        n_huge = is_huge.sum().item()
        if n_huge > 0:
            log_min = torch.log(torch.tensor(float(deg_cfg.huge_deg_range[0]), device=device))
            log_max = torch.log(torch.tensor(float(max_huge_deg), device=device))
            log_deg = torch.rand(n_huge, device=device) * (log_max - log_min) + log_min
            deg_batch[is_huge] = torch.exp(log_deg).long().clamp(
                deg_cfg.huge_deg_range[0],
                max_huge_deg
            )

        degrees_list.append(deg_batch)
        total_endpoints += deg_batch.sum().item()

    # Concatenate all batches
    deg = torch.cat(degrees_list, dim=0)

    # Trim to exactly E_target
    cumsum = torch.cumsum(deg, dim=0)
    n_nets = torch.searchsorted(cumsum, E_target, right=False).item() + 1
    deg = deg[:n_nets]

    # Adjust last degree to hit E_target exactly
    current_sum = deg.sum().item()
    if current_sum > E_target:
        diff = current_sum - E_target
        deg[-1] = max(2, deg[-1].item() - diff)
    elif current_sum < E_target:
        diff = E_target - current_sum
        deg[-1] = deg[-1] + diff

    return deg


def degrees_summary(deg: torch.Tensor, deg_cfg: DegreeConfig) -> Dict[str, int]:
    """
    Compute summary statistics of degree distribution.

    Args:
        deg: [N_net] degrees
        deg_cfg: Degree configuration (for thresholds)

    Returns:
        Dictionary with counts of small/medium/huge nets
    """
    small_mask = deg <= deg_cfg.small_deg_range[1]
    medium_mask = (deg > deg_cfg.small_deg_range[1]) & (deg <= deg_cfg.medium_deg_range[1])
    huge_mask = deg > deg_cfg.medium_deg_range[1]

    return {
        "n_small": small_mask.sum().item(),
        "n_medium": medium_mask.sum().item(),
        "n_huge": huge_mask.sum().item(),
        "deg_min": deg.min().item(),
        "deg_max": deg.max().item(),
        "deg_mean": deg.float().mean().item(),
    }
