"""Distribution helpers for V5 algorithm."""

import torch
import torch.distributions as dist


def get_distribution(dist_type: str, dist_params: dict):
    """
    Get a PyTorch distribution object.
    
    Args:
        dist_type: Type of distribution ("uniform", "normal", "bernoulli", "cond_binomial")
        dist_params: Parameters for the distribution
        
    Returns:
        Distribution object
    """
    if dist_type == "uniform":
        # Handle both scalar and tensor bounds
        low = dist_params.get("low", 0.0)
        high = dist_params.get("high", 1.0)
        if isinstance(low, torch.Tensor) and isinstance(high, torch.Tensor):
            # Multi-dimensional uniform (for 2D positions)
            return dist.Independent(dist.Uniform(low, high), 1)
        else:
            return dist.Uniform(low=low, high=high)
    elif dist_type == "normal":
        # PyTorch Normal uses 'loc' and 'scale', not 'mean' and 'std'
        if "mean" in dist_params:
            dist_params = dist_params.copy()
            dist_params["loc"] = dist_params.pop("mean")
        if "std" in dist_params:
            dist_params = dist_params.copy()
            dist_params["scale"] = dist_params.pop("std")
        return dist.Normal(**dist_params)
    elif dist_type == "bernoulli":
        return dist.Bernoulli(**dist_params)
    elif dist_type == "cond_binomial":
        return ConditionalBinomial(**dist_params)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


class ConditionalBinomial:
    """
    Conditional binomial distribution for Rent's Rule terminal assignment.
    
    Samples num_terminals based on instance area: num_terminals ∝ area^p
    """
    
    def __init__(self, binom_p: float, binom_min_n: int, t: float, p: float):
        """
        Args:
            binom_p: Base binomial probability
            binom_min_n: Minimum n for binomial
            t: Scaling factor
            p: Rent exponent (typically 0.65)
        """
        self.binom_p = binom_p
        self.binom_min_n = binom_min_n
        self.t = t
        self.p = p
    
    def sample(self, instance_area: torch.Tensor) -> torch.Tensor:
        """
        Sample number of terminals conditioned on instance area.
        
        Args:
            instance_area: (V,) tensor of instance areas
            
        Returns:
            (V,) tensor of number of terminals
        """
        # Rent's Rule: num_terminals ∝ area^p
        # Scale area to get n for binomial
        scaled_area = (instance_area / self.t) ** self.p
        n = torch.clamp(scaled_area, min=self.binom_min_n)
        
        # Round to nearest integer but keep as float (required for CUDA Binomial)
        # PyTorch Binomial requires integer values but float tensor type on CUDA
        n_rounded = torch.round(n).float()
        # Ensure minimum value
        n_rounded = torch.clamp(n_rounded, min=float(self.binom_min_n))
        
        # Sample from binomial with n and p
        # Note: Binomial on CUDA requires float tensor with integer values
        binomial = dist.Binomial(total_count=n_rounded, probs=self.binom_p)
        num_terminals = binomial.sample()
        
        return num_terminals.int()
