"""Positional encoding modules for coordinate inputs."""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh, lobpcg
    from scipy.sparse.linalg import ArpackNoConvergence
    from scipy.linalg import qr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class RandomFourierEncoding(nn.Module):
    """
    Random Fourier Features (RFF) encoding for coordinates.
    
    Encodes coordinates x_t using random Fourier features:
    [sin(2π * B * x_t), cos(2π * B * x_t)]
    where B is a random matrix sampled from N(0, scale^2)
    
    Args:
        input_dim: Dimension of input coordinates (typically 2 for x, y)
        num_features: Number of random features (output dim = 2 * num_features)
        scale: Scale parameter (bandwidth) for RFF
        learnable_scale: Whether to make scale learnable
    """
    
    def __init__(
        self,
        input_dim: int,
        num_features: int = 64,
        scale: float = 1.0,
        learnable_scale: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.output_dim = 2 * num_features
        
        # Random projection matrix B
        # Sample from N(0, scale^2)
        B = torch.randn(input_dim, num_features) * scale
        self.register_buffer("B", B)
        
        # Learnable scale parameter
        if learnable_scale:
            self.scale_param = nn.Parameter(torch.tensor(scale))
        else:
            self.register_buffer("scale_param", torch.tensor(scale))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates using RFF.
        
        Args:
            x: [N, input_dim] coordinate tensor
            
        Returns:
            encoded: [N, 2 * num_features] encoded features
        """
        # Apply scale if learnable
        if isinstance(self.scale_param, nn.Parameter):
            # CRITICAL FIX: Add epsilon to prevent division by zero
            # If B has zero variance (all values similar), std() → 0 → Inf → NaN
            B = self.B * (self.scale_param / (self.B.std() + 1e-8))
        else:
            B = self.B
        
        # Project: [N, input_dim] @ [input_dim, num_features] -> [N, num_features]
        proj = torch.matmul(x, B)
        
        # Apply 2π scaling
        proj = 2 * math.pi * proj
        
        # Compute sin and cos
        sin_features = torch.sin(proj)  # [N, num_features]
        cos_features = torch.cos(proj)  # [N, num_features]
        
        # Concatenate
        encoded = torch.cat([sin_features, cos_features], dim=-1)  # [N, 2*num_features]
        
        return encoded


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding based on coordinate differences.
    
    Computes relative positions between all pairs of instances and encodes them.
    Can use either learnable embeddings or fixed sinusoidal encoding.
    
    Args:
        encoding_dim: Dimension of positional encoding
        max_distance: Maximum distance for normalization (if None, auto-computed)
        learnable: Whether to use learnable embeddings
    """
    
    def __init__(
        self,
        encoding_dim: int = 64,
        max_distance: Optional[float] = None,
        learnable: bool = True,
    ):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.max_distance = max_distance
        self.learnable = learnable
        
        if learnable:
            # Learnable embedding for relative positions
            # We'll use a simple MLP to encode distance and direction
            self.embedding = nn.Sequential(
                nn.Linear(2, encoding_dim),  # 2D relative position
                nn.LayerNorm(encoding_dim),
                nn.GELU(),
                nn.Linear(encoding_dim, encoding_dim),
            )
        else:
            # Fixed sinusoidal encoding
            # Not implemented here, but could be added if needed
            raise NotImplementedError(
                "Fixed sinusoidal encoding not yet implemented. "
                "Use learnable=True for now."
            )
    
    def forward(
        self, x: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute relative positional encodings.
        
        Args:
            x: [N, 2] coordinate tensor (x, y positions)
            normalize: Whether to normalize distances
            
        Returns:
            rel_pos_enc: [N, N, encoding_dim] relative positional encodings
        """
        N = x.shape[0]
        
        # Compute pairwise differences: [N, 1, 2] - [1, N, 2] -> [N, N, 2]
        x_i = x.unsqueeze(1)  # [N, 1, 2]
        x_j = x.unsqueeze(0)  # [1, N, 2]
        rel_pos = x_i - x_j  # [N, N, 2]
        
        # Normalize distances if requested
        if normalize and self.max_distance is not None:
            rel_pos = rel_pos / self.max_distance
        elif normalize:
            # Auto-normalize based on max distance in current batch
            max_dist = torch.abs(rel_pos).max()
            if max_dist > 0:
                rel_pos = rel_pos / (max_dist + 1e-8)
        
        # Reshape for MLP: [N*N, 2]
        rel_pos_flat = rel_pos.view(-1, 2)
        
        # Encode: [N*N, encoding_dim]
        encoded_flat = self.embedding(rel_pos_flat)
        
        # Reshape back: [N, N, encoding_dim]
        encoded = encoded_flat.view(N, N, self.encoding_dim)
        
        return encoded
    
    def get_pairwise_encoding(
        self, x_i: torch.Tensor, x_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Get relative encoding between two sets of coordinates.
        
        Args:
            x_i: [N_i, 2] first set of coordinates
            x_j: [N_j, 2] second set of coordinates
            
        Returns:
            rel_pos_enc: [N_i, N_j, encoding_dim] relative encodings
        """
        # Compute pairwise differences: [N_i, 1, 2] - [1, N_j, 2] -> [N_i, N_j, 2]
        x_i_expanded = x_i.unsqueeze(1)  # [N_i, 1, 2]
        x_j_expanded = x_j.unsqueeze(0)  # [1, N_j, 2]
        rel_pos = x_i_expanded - x_j_expanded  # [N_i, N_j, 2]
        
        # Normalize if needed
        if self.max_distance is not None:
            rel_pos = rel_pos / self.max_distance
        else:
            max_dist = torch.abs(rel_pos).max()
            if max_dist > 0:
                rel_pos = rel_pos / (max_dist + 1e-8)
        
        # Encode: [N_i*N_j, 2] -> [N_i*N_j, encoding_dim]
        rel_pos_flat = rel_pos.view(-1, 2)
        encoded_flat = self.embedding(rel_pos_flat)
        
        # Reshape: [N_i, N_j, encoding_dim]
        encoded = encoded_flat.view(x_i.shape[0], x_j.shape[0], self.encoding_dim)
        
        return encoded


class SinusoidalPositionalEncoding(nn.Module):
    """
    ChipDiffusion-style sinusoidal positional encoding for 2D coordinates.

    Uses logarithmically-spaced frequencies from 1 to MAX_FREQ (default 100).
    This provides rich multi-scale spatial information for the model.

    Given coordinates x in R^2 (typically normalized to [-1, 1]), we compute:
        encoding_i = [sin(freq_k * x_i), cos(freq_k * x_i)] for k = 0..K-1, i = 0..D-1
    where freq_k = MAX_FREQ^(k / (K-1))

    Output dimension = input_dim * encoding_dim
    For 2D coordinates with encoding_dim=32: output = 2 * 32 = 64
    Features are normalized by 1/sqrt(encoding_dim) to prevent input domination

    Args:
        input_dim: Dimension of input coordinates (typically 2 for x, y)
        encoding_dim: Number of frequency bands (must be even)
        max_freq: Maximum frequency (default 100)
    """

    def __init__(
        self,
        input_dim: int = 2,
        encoding_dim: int = 32,
        max_freq: float = 100.0,
    ):
        super().__init__()
        assert encoding_dim % 2 == 0, "encoding_dim must be even"
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.max_freq = max_freq

        # Compute logarithmically-spaced frequencies: [1, ..., MAX_FREQ]
        # freq_k = MAX_FREQ^(k / (K/2 - 1)) for k = 0..K/2-1
        freq_indices = torch.arange(0, encoding_dim // 2, dtype=torch.float32)
        freqs = torch.exp(
            math.log(max_freq) * freq_indices / (encoding_dim // 2)
        )  # Shape: [encoding_dim // 2]

        # Register as buffer (non-trainable, moved with model)
        self.register_buffer("freqs", freqs.view(1, 1, -1))  # [1, 1, K/2]

        # Output dimension: input_dim * encoding_dim
        # For 2D coords with encoding_dim=32: 2 * 32 = 64
        # Each coordinate dimension gets encoding_dim features (sin/cos interleaved)
        self.output_dim = input_dim * encoding_dim
        
        # Normalization factor to prevent positional encoding from dominating input
        # Scale by 1/sqrt(encoding_dim) so total "energy" is comparable to original features
        # This ensures the model doesn't need to learn to downweight positional features
        # Additional factor of 0.5 for extra safety (prevents extreme values that could cause NaN)
        self.normalization_scale = 0.5 / math.sqrt(encoding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates using sinusoidal features.

        Args:
            x: [N, input_dim] coordinate tensor (e.g., [N, 2] for 2D positions)
               Not clamped; phase is clamped for sin/cos numerical stability only

        Returns:
            encoded: [N, input_dim * encoding_dim] encoded features
                     e.g., [N, 64] for 2D coords with encoding_dim=32
                     Normalized by 1/sqrt(encoding_dim) to prevent input domination
        """
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input with {self.input_dim} dimensions, got {x.shape[1]}"
            )

        # No coordinate clamp to preserve position info; phase is clamped below for sin/cos stability
        x_work = x.float()
        freqs = self.freqs.float()

        # x: [N, D] -> [N, D, 1]
        x_expanded = x_work.unsqueeze(-1)  # [N, D, 1]

        # Compute phase: [N, D, K/2]
        # freqs: [1, 1, K/2] broadcasts to [N, D, K/2]
        phase = x_expanded * freqs  # [N, D, K/2]
        
        # Clamp phase to prevent overflow/NaN in sin/cos for extreme max_freq.
        phase = torch.clamp(phase, min=-1.0e4, max=1.0e4)
        
        # Compute sin and cos features
        sin_features = torch.sin(phase)  # [N, D, K/2]
        cos_features = torch.cos(phase)  # [N, D, K/2]

        # Interleave sin and cos: [N, D, K/2, 2] -> [N, D, K]
        enc = torch.stack([sin_features, cos_features], dim=-1)  # [N, D, K/2, 2]
        enc = enc.view(x.shape[0], self.input_dim, self.encoding_dim)  # [N, D, K]

        # Flatten to [N, D*K]
        enc = enc.view(x.shape[0], -1)  # [N, D*K]
        
        # Normalize to prevent positional encoding from dominating the input
        # This ensures the total "signal strength" from positional encoding is
        # comparable to the original features, preventing initialization issues
        enc = enc * self.normalization_scale
        
        # Cast back to input dtype if needed
        if enc.dtype != x.dtype:
            enc = enc.to(dtype=x.dtype)
        
        return enc


class RobustCoordinateEncoding(nn.Module):
    """
    Robust coordinate encoding for 2D positions.

    Deterministic Fourier-style encoding compatible with `RobustEncodeConfig`.

    Given coordinates x in R^2 (typically normalized to [-1, 1]), we compute:
        [sin(2^k * pi * x), cos(2^k * pi * x)] for k = 0..K-1
    concatenated over both dimensions.

    Output dimension = 4 * num_freq_bands (2 dims * 2 trig * K bands).

    Args:
        num_freq_bands: Number of frequency bands K.
        clamp_norm: If True, clamp coordinates to [-1, 1] before encoding (default False to preserve info).
    """

    def __init__(
        self,
        num_freq_bands: int = 8,
        clamp_norm: bool = False,
    ):
        super().__init__()
        self.num_freq_bands = num_freq_bands
        self.clamp_norm = clamp_norm

        # Frequencies: [1, 2, 4, ..., 2^{K-1}]
        freq_exponents = torch.arange(num_freq_bands, dtype=torch.float32)
        freqs = 2.0 ** freq_exponents  # [K]
        self.register_buffer("freqs", freqs)

        # Output dimension: 2 dims * 2 (sin, cos) * num_freq_bands
        self.output_dim = 4 * num_freq_bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode 2D coordinates.

        Args:
            x: [N, 2] coordinate tensor

        Returns:
            encoded: [N, 4 * num_freq_bands] encoded features
        """
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"RobustCoordinateEncoding expects [N, 2] input, got {tuple(x.shape)}")

        if self.clamp_norm:
            x = torch.clamp(x, -1.0, 1.0)

        # x: [N, 2] -> [N, 2, 1]
        x_expanded = x.unsqueeze(-1)  # [N, 2, 1]

        # freqs: [K] -> [1, 1, K]
        freqs = self.freqs.view(1, 1, -1)  # [1, 1, K]

        # Compute phase: [N, 2, K]
        phase = math.pi * x_expanded * freqs

        sin_features = torch.sin(phase)  # [N, 2, K]
        cos_features = torch.cos(phase)  # [N, 2, K]

        # Concatenate over trig and flatten dims: [N, 2, 2K] -> [N, 4K]
        enc = torch.cat([sin_features, cos_features], dim=-1)  # [N, 2, 2K]
        enc = enc.view(x.shape[0], -1)  # [N, 4K]

        return enc


def compute_laplacian_eigenvectors(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_eigenvectors: int = 16,
    edge_weight: Optional[torch.Tensor] = None,
    normalized: bool = True,
) -> torch.Tensor:
    """
    Compute Laplacian eigenvectors for graph positional encoding.
    
    Computes the k smallest eigenvalues and eigenvectors of the graph Laplacian.
    The eigenvectors provide a natural positional encoding that captures the graph structure.
    
    Args:
        edge_index: [2, E] edge index tensor
        num_nodes: Number of nodes in the graph
        num_eigenvectors: Number of eigenvectors to compute (k)
        edge_weight: Optional [E] edge weights (default: uniform weights)
        normalized: If True, use normalized Laplacian (L = I - D^(-1/2) A D^(-1/2))
                   If False, use unnormalized Laplacian (L = D - A)
    
    Returns:
        eigenvectors: [num_nodes, num_eigenvectors] tensor of eigenvectors
                     (excluding the constant eigenvector corresponding to eigenvalue 0)
    """
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for Laplacian eigenvector computation. "
            "Install with: pip install scipy"
        )
    
    # Convert to numpy for scipy
    edge_index_np = edge_index.cpu().numpy()
    
    # Build adjacency matrix
    if edge_weight is not None:
        edge_weight_np = edge_weight.cpu().numpy()
    else:
        edge_weight_np = None
    
    # Create sparse adjacency matrix
    row = edge_index_np[0]
    col = edge_index_np[1]
    if edge_weight_np is not None:
        data = edge_weight_np
    else:
        data = np.ones(len(row), dtype=np.float32)
    
    # Create symmetric adjacency matrix (undirected graph)
    # More efficient: create symmetric matrix directly
    # Combine (i,j) and (j,i) edges upfront
    row_sym = np.concatenate([row, col])
    col_sym = np.concatenate([col, row])
    data_sym = np.concatenate([data, data])
    
    A = csr_matrix(
        (data_sym, (row_sym, col_sym)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    )
    # Eliminate duplicates by summing (in case of self-loops or duplicate edges)
    A.sum_duplicates()
    
    if normalized:
        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        # Compute degree matrix
        degrees = np.array(A.sum(axis=1)).flatten()
        degrees = np.maximum(degrees, 1e-10)  # Avoid division by zero
        D_inv_sqrt = csr_matrix(
            (1.0 / np.sqrt(degrees), (np.arange(num_nodes), np.arange(num_nodes))),
            shape=(num_nodes, num_nodes)
        )
        # L = I - D^(-1/2) A D^(-1/2)
        L = csr_matrix(np.eye(num_nodes)) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        # Unnormalized Laplacian: L = D - A
        degrees = np.array(A.sum(axis=1)).flatten()
        D = csr_matrix(
            (degrees, (np.arange(num_nodes), np.arange(num_nodes))),
            shape=(num_nodes, num_nodes)
        )
        L = D - A
    
    # Compute k+1 smallest eigenvalues/eigenvectors (k+1 because we'll exclude the first one)
    # The smallest eigenvalue is 0 with constant eigenvector (all nodes have same value)
    # We want the k smallest NON-ZERO eigenvalues for positional encoding
    k = min(num_eigenvectors + 1, num_nodes)
    
    # For large graphs, use lobpcg which is often faster than eigsh
    # Threshold: use lobpcg for graphs with > 5000 nodes
    use_lobpcg = num_nodes > 5000
    
    L64 = L.astype(np.float64)
    
    if use_lobpcg:
        # Use LOBPCG for large graphs (often faster)
        try:
            # Initialize with random vectors
            X = np.random.randn(num_nodes, k).astype(np.float64)
            # Orthonormalize
            X, _ = qr(X, mode='economic')
            
            # LOBPCG settings
            eigenvalues, eigenvectors = lobpcg(
                L64,
                X,
                largest=False,  # Find smallest eigenvalues
                maxiter=200,    # Usually converges faster
                tol=1e-4        # Slightly looser tolerance for speed
            )
        except Exception:
            # Fallback to eigsh if lobpcg fails
            use_lobpcg = False
    
    if not use_lobpcg:
        # Use eigsh for smaller graphs or as fallback
        # For Laplacians (PSD), 'SA' (smallest algebraic) is typically more stable than 'SM'
        # Increase maxiter and loosen tolerance for better convergence on large graphs
        eig_kwargs = dict(
            which="SA",
            maxiter=max(5000, num_nodes * 2),  # Scale maxiter with graph size
            tol=1e-4  # Looser tolerance for faster convergence
        )
        try:
            eigenvalues, eigenvectors = eigsh(L64, k=k, **eig_kwargs)
        except ArpackNoConvergence as e:
            # Partial convergence: keep what we got and pad later.
            eigenvalues = getattr(e, "eigenvalues", None)
            eigenvectors = getattr(e, "eigenvectors", None)
            if eigenvalues is None or eigenvectors is None:
                # Try with even looser tolerance
                eig_kwargs['tol'] = 1e-3
                eig_kwargs['maxiter'] = max(10000, num_nodes * 3)
                try:
                    eigenvalues, eigenvectors = eigsh(L64, k=k, **eig_kwargs)
                except ArpackNoConvergence as e2:
                    eigenvalues = getattr(e2, "eigenvalues", None)
                    eigenvectors = getattr(e2, "eigenvectors", None)
                    if eigenvalues is None or eigenvectors is None:
                        raise
        except Exception:
            # Fallback: reduce k slightly and try again.
            k2 = min(num_eigenvectors + 1, max(2, num_nodes - 1))
            try:
                eigenvalues, eigenvectors = eigsh(L64, k=k2, **eig_kwargs)
            except ArpackNoConvergence as e:
                eigenvalues = getattr(e, "eigenvalues", None)
                eigenvectors = getattr(e, "eigenvectors", None)
                if eigenvalues is None or eigenvectors is None:
                    raise
    
    # If nothing converged, return zeros.
    if eigenvectors is None or getattr(eigenvectors, "shape", (0, 0))[1] == 0:
        return torch.zeros((num_nodes, num_eigenvectors), dtype=torch.float32)

    # Sort by eigenvalue (ascending) - smallest eigenvalues first
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Exclude the first eigenvector (constant, eigenvalue ≈ 0)
    # Take the next k eigenvectors corresponding to the k smallest NON-ZERO eigenvalues
    # These capture the graph structure: 2nd smallest (Fiedler value) captures connectivity,
    # larger ones capture more local structure
    # Exclude the first eigenvector (constant, eigenvalue ≈ 0).
    # If partial convergence returned fewer vectors than needed, pad with zeros.
    if eigenvectors.shape[1] <= 1:
        eigenvectors = np.zeros((num_nodes, num_eigenvectors), dtype=np.float32)
    else:
        ev = eigenvectors[:, 1:]  # drop constant
        if ev.shape[1] >= num_eigenvectors:
            eigenvectors = ev[:, :num_eigenvectors]
        else:
            pad = np.zeros((num_nodes, num_eigenvectors - ev.shape[1]), dtype=ev.dtype)
            eigenvectors = np.concatenate([ev, pad], axis=1)
    
    # Column-normalize eigenvectors (safe for padded zeros)
    col_norms = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
    col_norms = np.maximum(col_norms, 1e-10)
    eigenvectors = eigenvectors / col_norms

    # Row-normalize: project each node's k-dim spectral coordinate onto the
    # unit sphere.  This makes magnitudes O(1) regardless of graph size N,
    # eliminating the 1/sqrt(N) scaling that causes OOD shift on larger graphs.
    # Nodes near spectral boundaries (small row norm) are clamped via epsilon.
    row_norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-10)
    eigenvectors = eigenvectors / row_norms
    
    # Convert back to torch tensor
    eigenvectors_torch = torch.from_numpy(eigenvectors.astype(np.float32))
    
    return eigenvectors_torch


class LaplacianEigenvectorPositionalEncoding(nn.Module):
    """
    Laplacian eigenvector positional encoding for graphs.
    
    Uses the eigenvectors of the graph Laplacian as positional encodings.
    These eigenvectors naturally encode the graph structure and provide
    a multi-scale representation of node positions.
    
    This module can work in two modes:
    1. Precomputed: eigenvectors are provided as input (recommended for efficiency)
    2. On-the-fly: eigenvectors are computed from edge_index during forward pass
    
    Args:
        num_eigenvectors: Number of eigenvectors to use (output dimension)
        precomputed_eigenvectors: Optional precomputed eigenvectors [N, num_eigenvectors]
        normalized_laplacian: Whether to use normalized Laplacian (default: True)
    """
    
    def __init__(
        self,
        num_eigenvectors: int = 16,
        precomputed_eigenvectors: Optional[torch.Tensor] = None,
        normalized_laplacian: bool = True,
    ):
        super().__init__()
        self.num_eigenvectors = num_eigenvectors
        self.output_dim = num_eigenvectors
        self.normalized_laplacian = normalized_laplacian
        
        if precomputed_eigenvectors is not None:
            # Register as buffer (non-trainable)
            self.register_buffer("eigenvectors", precomputed_eigenvectors)
            self.use_precomputed = True
        else:
            self.register_buffer("eigenvectors", None)
            self.use_precomputed = False
    
    def forward(
        self,
        edge_index: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute or retrieve Laplacian eigenvector positional encoding.
        
        Args:
            edge_index: [2, E] edge index (required if not using precomputed)
            num_nodes: Number of nodes (required if not using precomputed)
            edge_weight: Optional [E] edge weights
        
        Returns:
            encoding: [num_nodes, num_eigenvectors] positional encoding
        """
        if self.use_precomputed:
            if self.eigenvectors is None:
                raise ValueError("Precomputed eigenvectors not set")
            return self.eigenvectors
        
        if edge_index is None or num_nodes is None:
            raise ValueError(
                "edge_index and num_nodes required when not using precomputed eigenvectors"
            )
        
        # Compute eigenvectors on-the-fly
        eigenvectors = compute_laplacian_eigenvectors(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_eigenvectors=self.num_eigenvectors,
            edge_weight=edge_weight,
            normalized=self.normalized_laplacian,
        )
        
        return eigenvectors.to(edge_index.device)
