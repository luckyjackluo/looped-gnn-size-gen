"""Fourier feature encoders - unified from ChipGen and TopoGeoNet."""

import math
import torch
import torch.nn as nn


class RandomFourierEncoding(nn.Module):
    """
    Random Fourier Features (RFF) encoding for coordinates.
    
    Optimized for coordinates in [-0.5, 0.5] range (domain width = 1.0).
    This is the "Goldilocks" range for Flow Matching in VLSI placement:
    - Zero-centered: good for weight initialization and GNN aggregations
    - Unit domain width: simplifies frequency math (freq=1.0 = one cycle across chip)
    - Symmetric: optimal transport paths stay centered, preventing drift
    
    Uses log-uniform frequency distribution to cover all scales:
    - Global (0.1-1.0): Floorplanning and macro-grouping
    - Regional (1.0-10.0): Organizing logic clusters and power domains
    - Local (10.0-1000.0): Resolving cell-to-cell overlaps and site alignment
    
    For a chip with ~1000 cells, max_freq=1000 provides cell-level resolution.
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int = 64,
        scale: float = 1.0,
        learnable_scale: bool = False,
        use_log_uniform: bool = True,
        min_freq: float = 0.1,
        max_freq: float = 1000.0,
        seed: int = None,
    ):
        """
        Args:
            input_dim: Input dimension (2 for 2D coordinates)
            num_features: Number of random Fourier features (output_dim = 2 * num_features)
            scale: Legacy parameter (ignored if use_log_uniform=True)
            learnable_scale: Whether to make scale learnable (ignored if use_log_uniform=True)
            use_log_uniform: If True, use log-uniform frequency distribution (recommended for [-0.5, 0.5])
            min_freq: Minimum frequency for log-uniform distribution (global scale)
            max_freq: Maximum frequency for log-uniform distribution (local scale)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.output_dim = 2 * num_features
        self.use_log_uniform = use_log_uniform
        self.min_freq = min_freq
        self.max_freq = max_freq

        if use_log_uniform:
            # Log-uniform frequency distribution: frequencies sampled from [min_freq, max_freq] in log space
            # This ensures equal coverage of all scales (geometric progression)
            # 
            # For coordinates in [-0.5, 0.5] (domain width = 1.0):
            # - freq = 0.1: ~0.1 cycles across chip (global floorplanning)
            # - freq = 1.0: 1 full cycle across chip (regional organization)
            # - freq = 10.0: 10 cycles across chip (logic clusters)
            # - freq = 1000.0: 1000 cycles across chip (cell-level resolution for ~1000 cells)
            rng = torch.Generator()
            if seed is not None:
                rng.manual_seed(seed)
            
            # Sample frequencies uniformly in log space (geometric progression)
            # We sample num_features frequencies total, each with a random 2D direction
            log_min = math.log(min_freq)
            log_max = math.log(max_freq)
            log_freqs = torch.rand(num_features, generator=rng) * (log_max - log_min) + log_min
            freqs = torch.exp(log_freqs)  # [num_features] - one frequency per feature
            
            # Sample random directions (unit vectors) for each frequency
            # Each frequency gets a random 2D direction vector to define wave orientation
            directions = torch.randn(input_dim, num_features, generator=rng)
            directions = directions / (directions.norm(dim=0, keepdim=True) + 1e-8)  # Normalize to unit vectors [input_dim, num_features]
            
            # B matrix: frequency * direction (each column is a frequency vector)
            # B[i, j] = frequency[j] * direction[i, j]
            # When we compute x @ B, we get: sum_i x[i] * B[i, j] = sum_i x[i] * freq[j] * dir[i, j]
            # This gives us the projection of coordinates onto each frequency vector
            B = freqs.unsqueeze(0) * directions  # [input_dim, num_features]
            self.register_buffer("B", B)
            self.scale_param = None
        else:
            # Legacy: simple Gaussian scaling
            B = torch.randn(input_dim, num_features) * scale
            self.register_buffer("B", B)
            
            if learnable_scale:
                self.scale_param = nn.Parameter(torch.tensor(scale))
            else:
                self.register_buffer("scale_param", torch.tensor(scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input coordinates [..., input_dim], expected to be in [-0.5, 0.5]
        Returns:
            Fourier features [..., 2 * num_features]
        """
        if self.use_log_uniform:
            B = self.B
        else:
            if isinstance(self.scale_param, nn.Parameter):
                # CRITICAL FIX: Add epsilon to prevent division by zero
                # If B has zero variance (all values similar), std() → 0 → Inf → NaN
                B = self.B * (self.scale_param / (self.B.std() + 1e-8))
            else:
                B = self.B

        # Project coordinates onto frequency vectors
        # For domain width = 1.0 ([-0.5, 0.5]), we multiply by 2π to get proper phase
        proj = torch.matmul(x, B)  # [..., num_features]
        proj = 2 * math.pi * proj

        sin_features = torch.sin(proj)
        cos_features = torch.cos(proj)

        return torch.cat([sin_features, cos_features], dim=-1)


class FourierFeatureEncoder(nn.Module):
    """
    Fourier Feature Encoder with multi-scale bands.
    Unified from TopoGeoNet implementation.
    """

    def __init__(
        self,
        input_dim: int = 3,
        mapping_size: int = 128,
        passthrough: bool = True,
        normalize_coords: bool = True,
        multi_scale: bool = True,
        sigmas: tuple = (1/(4*math.pi), 1/(2*math.pi), 1/math.pi, 2/math.pi, 4/math.pi),
        seed: int = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.passthrough = passthrough
        self.mapping_size = mapping_size
        self.normalize_coords = normalize_coords
        self.multi_scale = multi_scale

        if multi_scale:
            self.sigmas = tuple(sigmas)
            half = mapping_size // 2

            per_band = max(1, half // len(self.sigmas))
            counts = [per_band] * len(self.sigmas)
            counts[-1] = half - per_band * (len(self.sigmas) - 1)

            rng = torch.Generator()
            if seed is not None:
                rng.manual_seed(seed)

            Bs = []
            for sigma_band, c in zip(self.sigmas, counts):
                B = torch.randn(input_dim, c, generator=rng) * sigma_band
                Bs.append(B)
            B = torch.cat(Bs, dim=1)
        else:
            rng = torch.Generator()
            if seed is not None:
                rng.manual_seed(seed)
            default_sigma = 1/math.pi
            B = torch.randn(input_dim, mapping_size // 2, generator=rng) * default_sigma

        self.register_buffer("B", B)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.normalize_coords:
            coords_min = coords.min(dim=0, keepdim=True)[0]
            coords_max = coords.max(dim=0, keepdim=True)[0]
            coords_range = coords_max - coords_min
            coords_range = torch.where(coords_range < 1e-8, torch.ones_like(coords_range), coords_range)
            coords_normalized = (coords - coords_min) / coords_range
        else:
            coords_normalized = coords

        B = self.B.to(dtype=coords_normalized.dtype, device=coords_normalized.device)
        x = 2 * math.pi * coords_normalized @ B
        fourier_features = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

        if self.passthrough:
            return torch.cat([coords_normalized, fourier_features], dim=-1)
        else:
            return fourier_features
