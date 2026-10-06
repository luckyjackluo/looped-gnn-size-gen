"""Size bin sampling for graph generation."""

import torch
from typing import Tuple

from .config import SizeConfig


def sample_N(size_cfg: SizeConfig, device: str = "cuda") -> Tuple[int, int]:
    """
    Sample N_inst from configured size bins with exact target proportions.

    Args:
        size_cfg: Size configuration with bins and weights
        device: Device for sampling (though result is returned as int)

    Returns:
        (N_inst, bin_id): Number of instances and which bin it came from
    """
    # Sample bin index according to weights
    weights = torch.tensor(size_cfg.weights, dtype=torch.float32, device=device)
    bin_id = torch.multinomial(weights, num_samples=1).item()

    # Sample uniform integer within chosen bin
    min_size, max_size = size_cfg.bins[bin_id]
    N_inst = torch.randint(min_size, max_size, (1,), device=device).item()

    return N_inst, bin_id


def get_bin_name(size_cfg: SizeConfig, bin_id: int) -> str:
    """Get human-readable name for bin."""
    return size_cfg.bin_names[bin_id]
