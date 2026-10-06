from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Union, List, Any

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from tqdm import tqdm

from unified_learning.training.test_time_techniques import (
    FlowMatchingTestTimeController,
    SamplingTechniqueConfig,
)

logger = logging.getLogger(__name__)


class NaNDetectedError(RuntimeError):
    """Raised when NaN/Inf is detected during training step. Carries the batch data for saving to .pt."""
    def __init__(
        self,
        message: str,
        data: Any,
        context: str = "",
        *,
        solver_state: Optional[dict] = None,
    ):
        super().__init__(message)
        self.data = data
        self.context = context
        # Optional: exact t, z_t, z0 (and similar) that led to NaN; trainer saves to nan_solver_state_*.pt
        self.solver_state = solver_state


class RejectBatchError(Exception):
    """Raised when sampled t or coordinates are out of safe range; trainer should skip this batch."""
    def __init__(self, message: str, data: Any = None):
        super().__init__(message)
        self.data = data


# Safe ranges for solver outputs: reject batch if t or coords fall outside (skip to next batch)
T_SAFE_MIN = 0.0
T_SAFE_MAX = 1.0
COORD_SAFE_MIN = -10.0  # normalized coords (z)
COORD_SAFE_MAX = 10.0


def _check_safe_range_coords(z_t: torch.Tensor, z0: torch.Tensor, data: Any) -> None:
    """Raise RejectBatchError if coordinates are out of safe range or non-finite."""
    if not torch.isfinite(z_t).all():
        raise RejectBatchError("z_t (sampled coords) has non-finite values. Rejecting batch.", data=data)
    z_flat = z_t.detach().float().flatten()
    if (z_flat < COORD_SAFE_MIN).any() or (z_flat > COORD_SAFE_MAX).any():
        raise RejectBatchError(
            f"z_t out of safe range [{COORD_SAFE_MIN}, {COORD_SAFE_MAX}]: min={z_flat.min().item():.6g}, max={z_flat.max().item():.6g}. Rejecting batch.",
            data=data,
        )
    if not torch.isfinite(z0).all():
        raise RejectBatchError("z0 (normalized target coords) has non-finite values. Rejecting batch.", data=data)
    z0_flat = z0.detach().float().flatten()
    if (z0_flat < COORD_SAFE_MIN).any() or (z0_flat > COORD_SAFE_MAX).any():
        raise RejectBatchError(
            f"z0 out of safe range [{COORD_SAFE_MIN}, {COORD_SAFE_MAX}]: min={z0_flat.min().item():.6g}, max={z0_flat.max().item():.6g}. Rejecting batch.",
            data=data,
        )


def _check_safe_range(
    t_cont: torch.Tensor,
    z_t: torch.Tensor,
    z0: torch.Tensor,
    data: Any,
) -> None:
    """Raise RejectBatchError if t or coordinates are out of safe range or non-finite (for flow_matching / DDPM)."""
    if not torch.isfinite(t_cont).all():
        raise RejectBatchError("t_cont has non-finite values (nan/inf). Rejecting batch.", data=data)
    t_flat = t_cont.detach().float().flatten()
    if (t_flat < T_SAFE_MIN).any() or (t_flat > T_SAFE_MAX).any():
        raise RejectBatchError(
            f"t_cont out of safe range [{T_SAFE_MIN}, {T_SAFE_MAX}]: min={t_flat.min().item():.6g}, max={t_flat.max().item():.6g}. Rejecting batch.",
            data=data,
        )
    _check_safe_range_coords(z_t, z0, data)


def _check_tensor_finite(tensor: torch.Tensor, name: str) -> Tuple[bool, str]:
    """
    Check if a tensor has only finite values (no NaN/Inf).
    Returns (is_finite, description). Description is empty if finite.
    """
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return True, ""
    flat = tensor.detach().float().flatten()
    n = flat.numel()
    nan_count = torch.isnan(flat).sum().item()
    inf_count = torch.isinf(flat).sum().item()
    if nan_count == 0 and inf_count == 0:
        return True, ""
    finite = flat[torch.isfinite(flat)]
    msg = (
        f"{name}: shape={tuple(tensor.shape)}, nan_count={nan_count}, inf_count={inf_count}"
    )
    if finite.numel() > 0:
        msg += f", finite_min={finite.min().item():.6g}, finite_max={finite.max().item():.6g}"
    return False, msg


def _find_first_nan_layer(
    denoiser: nn.Module,
    data: Any,
    z_t_conditioned: torch.Tensor,
    t_cont: torch.Tensor,
    context: str = "",
) -> Optional[str]:
    """
    Run denoiser forward with hooks to find the first module whose output contains NaN/Inf.
    denoiser may be a wrapper with .model (UnifiedModel). Returns the module path (e.g. encoder.0, gnn_blocks.2).
    """
    model = getattr(denoiser, "model", denoiser)
    if not isinstance(model, nn.Module):
        return None

    first_bad_name: List[Optional[str]] = [None]
    handles: List[Any] = []

    def make_hook(mod_name: str):
        def _hook(_module: nn.Module, _args: Any, output: Any) -> None:
            if first_bad_name[0] is not None:
                return
            tensors = []
            if isinstance(output, torch.Tensor):
                tensors.append(output)
            elif isinstance(output, (tuple, list)):
                for o in output:
                    if isinstance(o, torch.Tensor):
                        tensors.append(o)
            for t in tensors:
                if not torch.isfinite(t).all():
                    first_bad_name[0] = mod_name
                    return
        return _hook

    for name, child in model.named_modules():
        if child is model:
            continue
        try:
            h = child.register_forward_hook(make_hook(name))
            handles.append(h)
        except Exception:
            pass

    try:
        with torch.no_grad():
            _ = denoiser(data, z_t_conditioned, t_cont)
    except Exception as e:
        logger.warning("NaN diagnostic forward failed: %s", e)
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    return first_bad_name[0]


def _log_nan_diagnostics(
    ddpm: "GraphDDPM",
    data: Any,
    z0: torch.Tensor,
    z_t_conditioned: torch.Tensor,
    t_cont: torch.Tensor,
    mu: torch.Tensor,
    s: torch.Tensor,
    pred_out: torch.Tensor,
    loss_value: float,
    context: str = "",
) -> None:
    """When NaN/Inf is detected, log inputs and find first layer producing non-finite values."""
    logger.warning(
        "[NaN/Inf detected] %s pred has non-finite values; loss=%s. Running diagnostics.",
        context,
        loss_value,
    )
    ok_z0, msg_z0 = _check_tensor_finite(z0, "z0")
    if not ok_z0:
        logger.warning("  Input: %s", msg_z0)
    ok_zt, msg_zt = _check_tensor_finite(z_t_conditioned, "z_t_conditioned")
    if not ok_zt:
        logger.warning("  Input: %s", msg_zt)
    ok_t, msg_t = _check_tensor_finite(t_cont, "t_cont")
    if not ok_t:
        logger.warning("  Input: %s", msg_t)
    ok_mu, msg_mu = _check_tensor_finite(mu, "mu")
    if not ok_mu:
        logger.warning("  Input: %s", msg_mu)
    ok_s, msg_s = _check_tensor_finite(s, "s")
    if not ok_s:
        logger.warning("  Input: %s", msg_s)
    ok_pred, msg_pred = _check_tensor_finite(pred_out, "pred (v_pred/eps_pred)")
    if not ok_pred:
        logger.warning("  Output: %s", msg_pred)
    if hasattr(data, "x") and data.x is not None:
        ok_x, msg_x = _check_tensor_finite(data.x, "data.x")
        if not ok_x:
            logger.warning("  Data: %s", msg_x)

    first_bad = _find_first_nan_layer(
        ddpm.denoiser, data, z_t_conditioned, t_cont, context=context
    )
    if first_bad:
        logger.warning("  First layer with NaN/Inf in output: %s", first_bad)
    else:
        logger.warning("  Could not identify layer (hooks did not catch a non-finite output).")


@dataclass
class DiffusionConfig:
    """Configuration for DDPM-style diffusion on inst positions."""

    num_steps: int = 1000
    beta_schedule: Literal["linear", "cosine", "simple_cosine", "polynomial", "power_law", "force_field"] = "cosine"
    loss_type: Literal["eps"] = "eps"
    parameterization: Literal["ddpm", "edm_x0_precond", "flow_matching", "flow_matching_raw"] = "ddpm"
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 1.0
    # Refinement flow matching: if > 0, start from z_init + σ*ε instead of pure noise N(0,I)
    # This reduces stiffness and improves convergence for dense placement problems.
    # Training: z_0 = z_init + init_noise_std * ε (when pos_init available)
    # Sampling: z_t = z_init + init_noise_std * ε (when pos_init available)
    # Recommended: 0.1-0.3 for dense placement. Dataset must provide pos_init (initial layout).
    init_noise_std: float = 0.0
    time_embedding_dim: int = 128
    time_mlp_hidden_dim: int = 256
    film_in_inst_encoder: bool = True
    film_in_blocks: bool = False
    use_timestep_weighting: bool = True
    timestep_weight_strength: float = 0.1
    use_predictor_corrector: bool = True
    # Curriculum learning: progressively increase max timestep during training
    use_curriculum: bool = False
    curriculum_schedule: Literal["linear", "exponential"] = "linear"
    curriculum_start_steps: int = 100
    curriculum_end_epoch: int = 50
    # Schedule parameters for polynomial/power_law/force_field
    schedule_power: float = 2.0  # Power parameter for polynomial/power_law schedules

    # Local refinement optimization for flow matching
    local_refinement_bias: float = 1.0  # Beta(α,1): mean=α/(α+1). Use 2.0–3.0 for stability; <2 gives more early-t → noisier z_t → can trigger NaN (encoder/pos_encoding)
    use_refinement_weighting: bool = False  # Enable timestep-dependent loss weighting (emphasizes late t)
    refinement_weight_strength: float = 2.0  # Weight multiplier for late timesteps when use_refinement_weighting=True
    time_film_scale: float = 0.25  # Time conditioning strength in model (0.25=weak, 0.5=balanced, 0.75=strong)
    
    # ODE solver for flow matching sampling
    ode_solver: Literal["euler", "heun"] = "euler"  # ODE solver: "euler" (1st order, faster) or "heun" (2nd order, more accurate)

    # Flow-matching sampling time grid (non-uniform time improves late-stage local detail)
    # - "none": uniform t grid in [0,1]
    # - "late_power": concentrate more steps near t -> 1 using t = 1 - (1-u)^p, p>1
    fm_time_warp: Literal["none", "late_power"] = "none"
    fm_time_warp_power: float = 1.0

    # Sigma rescaling for size generalization (inference): t' = t/(t + sigma_scale*(1-t))
    # When 1.0, no remapping. When >1, model receives earlier conditioning -> stronger early denoising on large graphs.
    # Only affects flow_matching sampling. s>1 recommended when trained on small graphs, applied to large-N.
    sigma_scale: float = 1.0
    
    # N-dependent, time-dependent sigma scaling for over-repulsion correction
    # When enabled, replaces sigma_scale with s(N,t) that depends on graph size N and time t
    use_adaptive_sigma_scale: bool = False  # Enable N-dependent, time-dependent scaling
    adaptive_sigma_n0: float = 300.0  # Reference graph size N0
    adaptive_sigma_beta: float = 0.1  # Log scaling coefficient β
    adaptive_sigma_s_min: float = 0.7  # Minimum scaling factor s_min
    adaptive_sigma_t_start: float = 0.6  # Time gate start (only apply scaling after this t)
    
    # Overlap-aware training and guided sampling
    use_overlap_loss: bool = False  # Add overlap penalty to training loss
    overlap_loss_weight: float = 0.1  # Weight for overlap penalty in training loss
    use_guided_sampling: bool = False  # Use guided diffusion with overlap penalty during sampling
    guidance_scale: float = 1.0  # Scale for guidance signal (higher = stronger overlap avoidance)
    overlap_penalty_threshold: float = 0.0  # Only penalize overlaps above this threshold (0.0 = penalize all overlaps)
    guidance_every_steps: int = 1  # Apply overlap guidance every this many ODE steps (1=every step; >1 reduces cost, gradient is slow)

    # Graph-level weighted loss: give larger weight to larger graphs (each graph's loss weighted by n_nodes^power)
    use_graph_weighted_loss: bool = False  # If True, loss = weighted sum of per-graph losses, weight_g = n_g^graph_weight_power
    graph_weight_power: float = 1.0  # weight_g = (num nodes in graph g)^graph_weight_power; 1.0 = linear in size
    graph_weight_mode: Literal["power", "normalized"] = "power"  # "power": weight_g = N_g^power; "normalized": weight_g = N_G / N_B where N_B = mean(N_g) in batch

    # Two-scale velocity head: v_phys = L * v_die + s_c * v_cell; loss in physical space (improves scale generalization)
    two_scale_velocity: bool = False

    # Sample-time coordinate clamp (normalized z): prevents explosion over ODE/DDPM steps.
    # For DDPM with normal init noise, coords can go outside [-1,1]; use a large clip (e.g. [-10, 10]).
    sample_coord_clip_min: float = -5.0
    sample_coord_clip_max: float = 5.0
    sampling_techniques: SamplingTechniqueConfig = field(default_factory=SamplingTechniqueConfig)


class TimeEmbedding(nn.Module):
    """Pure MLP: t (scalar) -> output_dim. Single module added to initial node embedding (no separate proj)."""

    def __init__(self, mlp_hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [...,] -> [..., output_dim] (ready to add to node features)."""
        t = t.float().unsqueeze(-1)  # [..., 1]
        return self.mlp(t)


class FiLM(nn.Module):
    """Feature-wise linear modulation block."""

    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2),
        )
        # Initialize FiLM network weights for numerical stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize FiLM network weights. Use gain=0.5 to avoid extreme modulation while allowing velocity scale ~1."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, C] features to modulate.
            cond: [N, cond_dim] conditioning vectors.
        """
        gamma_beta = self.net(cond)  # [N, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        # No clamp on gamma/beta to preserve full modulation range; use gradient clipping if needed
        return (1.0 + gamma) * x + beta


def make_beta_schedule(
    num_steps: int,
    schedule: Literal["linear", "cosine", "simple_cosine", "polynomial", "power_law", "force_field"] = "linear",
    device: Optional[torch.device] = None,
    schedule_power: float = 2.0,  # Power for polynomial/power_law schedules
) -> torch.Tensor:
    """
    Create a beta schedule.
    
    Args:
        num_steps: Number of diffusion steps
        schedule: Schedule type
        device: Device for tensors
        schedule_power: Power parameter for polynomial/power_law schedules (default: 2.0)
    """
    if schedule == "linear":
        betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
    elif schedule == "cosine":
        # Improved DDPM cosine schedule (approximation)
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps, device=device)
        alphas_cumprod = torch.cos(((x / num_steps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(1e-4, 0.999)
    elif schedule == "simple_cosine":
        # Simple cosine schedule (matches chipdiffusion reference)
        # alpha(t) = cos(pi/2 * t) where t in [0, 1]
        t = torch.linspace(0, 1, num_steps + 1, device=device)
        alphas = torch.cos(t * torch.pi / 2) ** 2
        alphas_cumprod = alphas[1:]  # Skip first (1.0)
        # Compute betas from alphas_cumprod
        alphas_cumprod_prev = torch.cat([alphas[0:1], alphas_cumprod[:-1]])
        betas = 1 - (alphas_cumprod / alphas_cumprod_prev)
        betas = betas.clamp(1e-4, 0.999)
    elif schedule == "polynomial":
        # Polynomial schedule: alpha_bar(t) = (1 - t)^p
        # More gradual than cosine, emphasizes early global structure learning
        # Higher p = more gradual change, more noise early
        t = torch.linspace(0, 1, num_steps + 1, device=device)
        alphas_cumprod = (1 - t) ** schedule_power
        alphas_cumprod = alphas_cumprod[1:]  # Skip first (1.0)
        alphas_cumprod_prev = torch.cat([alphas_cumprod[0:1], alphas_cumprod[:-1]])
        betas = 1 - (alphas_cumprod / alphas_cumprod_prev)
        betas = betas.clamp(1e-4, 0.999)
    elif schedule == "power_law":
        # Power law schedule: alpha_bar(t) = (1 - t^p)
        # Similar to polynomial but different curvature
        # More gradual at the end (t close to 1)
        t = torch.linspace(0, 1, num_steps + 1, device=device)
        alphas_cumprod = 1 - (t ** schedule_power)
        alphas_cumprod = alphas_cumprod[1:]  # Skip first (1.0)
        alphas_cumprod_prev = torch.cat([alphas_cumprod[0:1], alphas_cumprod[:-1]])
        betas = 1 - (alphas_cumprod / alphas_cumprod_prev)
        betas = betas.clamp(1e-4, 0.999)
    elif schedule == "force_field":
        # Force-field inspired schedule: emphasizes global structure early
        # Uses exponential decay with slower transition at high noise levels
        # alpha_bar(t) = exp(-k * t^p) where k controls decay rate
        # This keeps high noise longer, forcing model to learn global structure first
        t = torch.linspace(0, 1, num_steps + 1, device=device)
        # Use power < 1 to make it more gradual at the end
        # schedule_power < 1: more gradual, more noise early
        # schedule_power > 1: sharper transition
        k = 3.0  # Decay rate (higher = faster decay)
        alphas_cumprod = torch.exp(-k * (t ** schedule_power))
        alphas_cumprod = alphas_cumprod[1:]  # Skip first (1.0)
        alphas_cumprod_prev = torch.cat([alphas_cumprod[0:1], alphas_cumprod[:-1]])
        betas = 1 - (alphas_cumprod / alphas_cumprod_prev)
        betas = betas.clamp(1e-4, 0.999)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}. Supported: linear, cosine, simple_cosine, polynomial, power_law, force_field")
    return betas


def compute_overlap_penalty(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    threshold: float = 0.0,
    use_spatial_hash: bool = True,
    cell_size: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute overlap penalty for a set of positions and sizes.
    
    Uses spatial hashing for O(N*k) complexity where k is average neighbors per cell,
    falling back to O(N²) vectorized operations for small N or when spatial_hash=False.
    
    Only penalizes overlaps above threshold (useful for allowing small overlaps).
    
    Args:
        positions: (N, 2) center positions
        sizes: (N, 2) widths and heights
        mask: Optional (N,) boolean mask - only compute penalty for masked instances
        threshold: Only penalize overlaps above this threshold (0.0 = penalize all overlaps)
        use_spatial_hash: If True, use spatial hashing for efficiency (default: True)
        cell_size: Grid cell size for spatial hashing (auto-computed if None)
    
    Returns:
        overlap_penalty: Scalar tensor with total overlap penalty
    """
    N = positions.shape[0]
    device = positions.device
    
    # Validate inputs for NaN/Inf - return 0 penalty if invalid
    if not torch.isfinite(positions).all() or not torch.isfinite(sizes).all():
        return torch.tensor(0.0, device=device)
    
    # Validate sizes are positive
    if torch.any(sizes <= 0):
        return torch.tensor(0.0, device=device)
    
    # Apply mask if provided
    if mask is not None:
        positions = positions[mask]
        sizes = sizes[mask]
        N = positions.shape[0]
        if N == 0:
            return torch.tensor(0.0, device=device)
    
    return _compute_overlap_penalty_spatial_hash(positions, sizes, threshold, cell_size)


def _compute_overlap_penalty_spatial_hash(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    threshold: float = 0.0,
    cell_size: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute overlap penalty using spatial hashing for O(N*k) complexity.
    
    Uses a uniform grid to partition space. Each instance checks only nearby
    instances in overlapping grid cells, reducing from O(N²) to O(N*k) where
    k is the average number of neighbors per cell.
    """
    N = positions.shape[0]
    device = positions.device
    dtype = positions.dtype
    
    # Convert to box bounds
    x_min = positions[:, 0] - sizes[:, 0] / 2
    y_min = positions[:, 1] - sizes[:, 1] / 2
    x_max = positions[:, 0] + sizes[:, 0] / 2
    y_max = positions[:, 1] + sizes[:, 1] / 2
    
    # Compute bounding box of all instances
    bbox_x_min = x_min.min().item()
    bbox_y_min = y_min.min().item()
    bbox_x_max = x_max.max().item()
    bbox_y_max = y_max.max().item()
    
    # Auto-compute cell size if not provided (use median instance size)
    if cell_size is None:
        median_size = torch.median(sizes.mean(dim=1))
        cell_size = max(median_size.item() * 2.0, 1.0)  # At least 2x median size
    
    # Grid dimensions
    grid_w = int(torch.ceil(torch.tensor((bbox_x_max - bbox_x_min) / cell_size, device=device)).item()) + 1
    grid_h = int(torch.ceil(torch.tensor((bbox_y_max - bbox_y_min) / cell_size, device=device)).item()) + 1
    
    # Build spatial hash: grid[cell_y][cell_x] contains list of instance indices
    # Use lists instead of sets for PyTorch compatibility
    grid = [[[] for _ in range(grid_w)] for _ in range(grid_h)]
    
    # Insert instances into grid cells
    for i in range(N):
        # Get grid cells this instance overlaps
        cell_x_min = int((x_min[i].item() - bbox_x_min) / cell_size)
        cell_x_max = int((x_max[i].item() - bbox_x_min) / cell_size) + 1
        cell_y_min = int((y_min[i].item() - bbox_y_min) / cell_size)
        cell_y_max = int((y_max[i].item() - bbox_y_min) / cell_size) + 1
        
        # Clamp to grid bounds
        cell_x_min = max(0, min(cell_x_min, grid_w - 1))
        cell_x_max = max(0, min(cell_x_max, grid_w))
        cell_y_min = max(0, min(cell_y_min, grid_h - 1))
        cell_y_max = max(0, min(cell_y_max, grid_h))
        
        # Insert into all overlapping cells
        for cy in range(cell_y_min, cell_y_max):
            for cx in range(cell_x_min, cell_x_max):
                grid[cy][cx].append(i)
    
    # Compute overlap penalty by checking only nearby instances
    total_overlap_area = torch.tensor(0.0, device=device, dtype=dtype)
    
    for i in range(N):
        # Get grid cells this instance overlaps
        cell_x_min = int((x_min[i].item() - bbox_x_min) / cell_size)
        cell_x_max = int((x_max[i].item() - bbox_x_min) / cell_size) + 1
        cell_y_min = int((y_min[i].item() - bbox_y_min) / cell_size)
        cell_y_max = int((y_max[i].item() - bbox_y_min) / cell_size) + 1
        
        # Clamp to grid bounds
        cell_x_min = max(0, min(cell_x_min, grid_w - 1))
        cell_x_max = max(0, min(cell_x_max, grid_w))
        cell_y_min = max(0, min(cell_y_min, grid_h - 1))
        cell_y_max = max(0, min(cell_y_max, grid_h))
        
        # Collect candidate neighbors from overlapping cells
        candidates = set()
        for cy in range(cell_y_min, cell_y_max):
            for cx in range(cell_x_min, cell_x_max):
                candidates.update(grid[cy][cx])
        
        # Remove self
        candidates.discard(i)
        
        # Check overlap with each candidate (only check j > i to avoid double counting)
        for j in candidates:
            if j <= i:
                continue
            
            # Check if boxes overlap
            if (x_min[i] < x_max[j] and x_min[j] < x_max[i] and
                y_min[i] < y_max[j] and y_min[j] < y_max[i]):
                
                # Compute overlap area
                overlap_x_min = torch.maximum(x_min[i], x_min[j])
                overlap_x_max = torch.minimum(x_max[i], x_max[j])
                overlap_y_min = torch.maximum(y_min[i], y_min[j])
                overlap_y_max = torch.minimum(y_max[i], y_max[j])
                
                overlap_w = torch.clamp(overlap_x_max - overlap_x_min, min=0.0)
                overlap_h = torch.clamp(overlap_y_max - overlap_y_min, min=0.0)
                overlap_area = overlap_w * overlap_h
                
                if overlap_area > threshold:
                    total_overlap_area = total_overlap_area + overlap_area
    
    return total_overlap_area


def compute_overlap_gradient(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    threshold: float = 0.0,
    use_spatial_hash: bool = True,
    cell_size: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute gradient of overlap penalty with respect to positions.
    
    This is used for guided diffusion - the gradient points in the direction
    that reduces overlaps. Uses spatial hashing for efficiency.
    
    Runs on CPU to avoid O(N) GPU syncs from .item() in Python loops; only 2
    device transfers (to CPU, back to original device) per call.
    
    Args:
        positions: (N, 2) center positions
        sizes: (N, 2) widths and heights
        mask: Optional (N,) boolean mask - only compute gradient for masked instances
        threshold: Only penalize overlaps above this threshold
        use_spatial_hash: If True, use spatial hashing (default: True)
        cell_size: Grid cell size for spatial hashing (auto-computed if None)
    
    Returns:
        gradient: (N, 2) gradient tensor pointing away from overlaps
    """
    N = positions.shape[0]
    device = positions.device
    dtype = positions.dtype
    
    # Run on CPU to avoid many GPU->CPU syncs from .item() in spatial-hash loops
    positions = positions.detach().cpu()
    sizes = sizes.detach().cpu()
    if mask is not None:
        mask = mask.cpu()
        positions_masked = positions[mask]
        sizes_masked = sizes[mask]
        N_masked = positions_masked.shape[0]
        if N_masked == 0:
            return torch.zeros(N, 2, device=device, dtype=dtype)
    else:
        positions_masked = positions
        sizes_masked = sizes
        N_masked = N
    
    grad = _compute_overlap_gradient_spatial_hash(positions_masked, sizes_masked, threshold, cell_size)
    
    # Map back to full size if mask was used
    if mask is not None:
        full_grad = torch.zeros(N, 2, device=grad.device, dtype=dtype)
        full_grad[mask] = grad
        grad = full_grad
    
    return grad.to(device=device)


def _compute_overlap_gradient_spatial_hash(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    threshold: float = 0.0,
    cell_size: Optional[float] = None,
) -> torch.Tensor:
    """Compute overlap gradient using spatial hashing for O(N*k) complexity."""
    N = positions.shape[0]
    device = positions.device
    dtype = positions.dtype
    
    # Convert to box bounds
    x_min = positions[:, 0] - sizes[:, 0] / 2
    y_min = positions[:, 1] - sizes[:, 1] / 2
    x_max = positions[:, 0] + sizes[:, 0] / 2
    y_max = positions[:, 1] + sizes[:, 1] / 2
    
    # Compute bounding box
    bbox_x_min = x_min.min().item()
    bbox_y_min = y_min.min().item()
    bbox_x_max = x_max.max().item()
    bbox_y_max = y_max.max().item()
    
    # Auto-compute cell size
    if cell_size is None:
        median_size = torch.median(sizes.mean(dim=1))
        cell_size = max(median_size.item() * 2.0, 1.0)
    
    # Grid dimensions
    grid_w = int(torch.ceil(torch.tensor((bbox_x_max - bbox_x_min) / cell_size, device=device)).item()) + 1
    grid_h = int(torch.ceil(torch.tensor((bbox_y_max - bbox_y_min) / cell_size, device=device)).item()) + 1
    
    # Build spatial hash
    grid = [[[] for _ in range(grid_w)] for _ in range(grid_h)]
    
    for i in range(N):
        cell_x_min = int((x_min[i].item() - bbox_x_min) / cell_size)
        cell_x_max = int((x_max[i].item() - bbox_x_min) / cell_size) + 1
        cell_y_min = int((y_min[i].item() - bbox_y_min) / cell_size)
        cell_y_max = int((y_max[i].item() - bbox_y_min) / cell_size) + 1
        
        cell_x_min = max(0, min(cell_x_min, grid_w - 1))
        cell_x_max = max(0, min(cell_x_max, grid_w))
        cell_y_min = max(0, min(cell_y_min, grid_h - 1))
        cell_y_max = max(0, min(cell_y_max, grid_h))
        
        for cy in range(cell_y_min, cell_y_max):
            for cx in range(cell_x_min, cell_x_max):
                grid[cy][cx].append(i)
    
    # Initialize gradient
    grad = torch.zeros(N, 2, device=device, dtype=dtype)
    
    # Compute gradients by checking only nearby instances
    for i in range(N):
        cell_x_min = int((x_min[i].item() - bbox_x_min) / cell_size)
        cell_x_max = int((x_max[i].item() - bbox_x_min) / cell_size) + 1
        cell_y_min = int((y_min[i].item() - bbox_y_min) / cell_size)
        cell_y_max = int((y_max[i].item() - bbox_y_min) / cell_size) + 1
        
        cell_x_min = max(0, min(cell_x_min, grid_w - 1))
        cell_x_max = max(0, min(cell_x_max, grid_w))
        cell_y_min = max(0, min(cell_y_min, grid_h - 1))
        cell_y_max = max(0, min(cell_y_max, grid_h))
        
        candidates = set()
        for cy in range(cell_y_min, cell_y_max):
            for cx in range(cell_x_min, cell_x_max):
                candidates.update(grid[cy][cx])
        
        candidates.discard(i)
        
        for j in candidates:
            if j <= i:
                continue
            
            if (x_min[i] < x_max[j] and x_min[j] < x_max[i] and
                y_min[i] < y_max[j] and y_min[j] < y_max[i]):
                
                overlap_x_min = torch.maximum(x_min[i], x_min[j])
                overlap_x_max = torch.minimum(x_max[i], x_max[j])
                overlap_y_min = torch.maximum(y_min[i], y_min[j])
                overlap_y_max = torch.minimum(y_max[i], y_max[j])
                
                overlap_w = torch.clamp(overlap_x_max - overlap_x_min, min=0.0)
                overlap_h = torch.clamp(overlap_y_max - overlap_y_min, min=0.0)
                overlap_area = overlap_w * overlap_h
                
                if overlap_area > threshold:
                    overlap_center_x = (overlap_x_min + overlap_x_max) / 2.0
                    overlap_center_y = (overlap_y_min + overlap_y_max) / 2.0
                    
                    dx_i = positions[i, 0] - overlap_center_x
                    dy_i = positions[i, 1] - overlap_center_y
                    dist_i = torch.clamp(torch.sqrt(dx_i**2 + dy_i**2), min=1e-6)
                    
                    dx_j = positions[j, 0] - overlap_center_x
                    dy_j = positions[j, 1] - overlap_center_y
                    dist_j = torch.clamp(torch.sqrt(dx_j**2 + dy_j**2), min=1e-6)
                    
                    grad_magnitude = overlap_area * 0.1
                    
                    grad[i, 0] += grad_magnitude * (dx_i / dist_i)
                    grad[i, 1] += grad_magnitude * (dy_i / dist_i)
                    grad[j, 0] += grad_magnitude * (dx_j / dist_j)
                    grad[j, 1] += grad_magnitude * (dy_j / dist_j)
    
    return grad


class GraphDDPM(nn.Module):
    """DDPM for inst positions on batched HeteroData graphs."""

    def __init__(
        self,
        denoiser: DiffusionDenoiser,
        diffusion_cfg: DiffusionConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.cfg = diffusion_cfg

        betas = make_beta_schedule(
            num_steps=diffusion_cfg.num_steps,
            schedule=diffusion_cfg.beta_schedule,
            device=device,
            schedule_power=getattr(diffusion_cfg, 'schedule_power', 2.0),
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    @property
    def num_steps(self) -> int:
        return self.cfg.num_steps

    def _two_scale_velocity_to_normalized(self, v_pred: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Convert two-scale velocity (N, 4) to normalized (N, 2): v_phys = L*v_die + s_c*v_cell, v_norm = v_phys/s."""
        ROW_HEIGHT = 0.9898
        v_die = v_pred[:, :2]
        v_cell = v_pred[:, 2:]
        W = 2.0 * s[:, 0]
        H = 2.0 * s[:, 1]
        L = torch.maximum(W, H).unsqueeze(-1)
        v_phys = L * v_die + ROW_HEIGHT * v_cell
        return v_phys / s

    def _get_conditioning_features(self, original_feats: torch.Tensor) -> torch.Tensor:
        """
        Return features to concatenate with z_t: [z_t, feats] -> model input.
        - feat_dim == expected_input_dim - 2: features only (no coords), use as-is.
        - feat_dim == expected_input_dim: already has coords in first 2, strip them.
        """
        expected_input_dim = None
        if hasattr(self.denoiser, 'model') and hasattr(self.denoiser.model, 'input_dim'):
            expected_input_dim = self.denoiser.model.input_dim
        feat_dim = original_feats.shape[1]
        if expected_input_dim is not None:
            if feat_dim == expected_input_dim - 2:
                return original_feats  # features only
            if feat_dim == expected_input_dim:
                return original_feats[:, 2:]  # has coords in first 2
        # Legacy: strip when data has coords in first 2 (12, 15, 18, 24, 31); 16 is eigen features-only -> don't strip
        if feat_dim in (12, 15, 18, 24, 31):
            return original_feats[:, 2:]
        return original_feats

    def _sample_timesteps(self, num_samples: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample discrete step indices and corresponding continuous times.

        Returns:
            t_idxs: [N] integer indices in [0, T-1]
            t_cont: [N] continuous times in (0, 1]
        """
        t_idxs = torch.randint(
            low=0,
            high=self.num_steps,
            size=(num_samples,),
            device=device,
            dtype=torch.long,
        )
        # Map to (0,1]; add 1 so last step maps to 1.0
        t_cont = (t_idxs + 1).float() / float(self.num_steps)
        return t_idxs, t_cont

    def q_sample(
        self,
        z0: torch.Tensor,
        t_idxs: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion q(z_t | z_0) in normalized gauge space.

        CRITICAL: All inputs/outputs are in normalized coordinate space:
        - z0: normalized clean positions (z0 = (x0 - mu) / s)
        - noise: sampled from N(0, I) in normalized space
        - z_t: noisy positions in normalized space
        - Returns: (z_t, noise) both in normalized space

        Args:
            z0: [N, 2] normalized clean positions ( (x0-mu)/s )
            t_idxs: [N] integer step indices
            noise: optional [N, 2] noise in normalized space (if None, sampled from N(0, I))
        Returns:
            z_t: [N, 2] noisy positions in normalized space
            noise: [N, 2] noise in normalized space
        """
        if noise is None:
            noise = torch.randn_like(z0)  # Sample noise in normalized space
        alphas_cumprod = self.alphas_cumprod[t_idxs].unsqueeze(-1)  # [N,1]
        mean = torch.sqrt(alphas_cumprod) * z0
        var = 1.0 - alphas_cumprod
        std = torch.sqrt(var)
        z_t = mean + std * noise  # z_t is in normalized space
        return z_t, noise  # Both in normalized space
    
    def _extract_sizes_from_data(self, data, device: torch.device) -> Optional[torch.Tensor]:
        """
        Extract instance sizes from data (handles both HeteroData and homogeneous Data).
        
        Returns:
            sizes: (N, 2) tensor with [width, height] or None if not available
        """
        ROW_HEIGHT = 0.9898  # μm (must match data_loading_homogeneous.py)
        
        def _sizes_from_feats(x, s, n, dev):
            # Use w_norm_d, h_norm_d and scale s (same scale as coords)
            if x.shape[1] >= 4 and s is not None and s.numel() >= 2:
                if x.shape[1] >= 12:
                    w_nd, h_nd = x[:, 2], x[:, 3]   # 12 dims with z_t
                elif x.shape[1] >= 10:
                    w_nd, h_nd = x[:, 0], x[:, 1]   # 10 dims without z_t
                elif x.shape[1] >= 16:
                    w_nd, h_nd = x[:, 6], x[:, 7]   # legacy
                elif x.shape[1] >= 14:
                    w_nd, h_nd = x[:, 4], x[:, 5]   # legacy
                else:
                    w_nd = h_nd = None
                if w_nd is not None:
                    # Validate s for NaN/Inf before using
                    if not torch.isfinite(s).all():
                        # If s contains NaN/Inf, return default sizes
                        return torch.ones((n, 2), device=dev) * 1.0
                    if s.dim() >= 2:
                        scale = s[:, 0].to(dev)
                        scale_y = s[:, 1].to(dev)
                    else:
                        scale = s.flatten()[0].expand(n).to(dev)
                        scale_y = s.flatten()[1].expand(n).to(dev)
                    # Validate scale values
                    if torch.any(scale <= 0) or torch.any(scale_y <= 0):
                        return torch.ones((n, 2), device=dev) * 1.0
                    sizes = torch.stack([w_nd * scale, h_nd * scale_y], dim=1)
                    # Validate result for NaN/Inf
                    if not torch.isfinite(sizes).all():
                        return torch.ones((n, 2), device=dev) * 1.0
                    return sizes
            # Fallback: legacy w_rel, h_rel -> exp(*)*ROW_HEIGHT
            if x.shape[1] >= 4:
                if x.shape[1] >= 16:
                    w_rel, h_rel = x[:, 2], x[:, 3]
                elif x.shape[1] >= 14:
                    w_rel, h_rel = x[:, 0], x[:, 1]
                else:
                    w_rel = h_rel = None
                if w_rel is not None:
                    sizes = torch.stack([
                        torch.exp(w_rel) * ROW_HEIGHT,
                        torch.exp(h_rel) * ROW_HEIGHT
                    ], dim=1)
                    # Validate result for NaN/Inf
                    if not torch.isfinite(sizes).all():
                        return torch.ones((n, 2), device=dev) * 1.0
                    return sizes
            return torch.ones((n, 2), device=dev) * 1.0
        
        if hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data):
            if getattr(data['inst'], 'size', None) is not None and isinstance(data['inst'].size, torch.Tensor):
                sizes = data['inst'].size
                # Validate sizes before returning
                if not torch.isfinite(sizes).all() or torch.any(sizes <= 0):
                    N = sizes.shape[0]
                    return torch.ones((N, 2), device=device) * 1.0
                return sizes
            if hasattr(data['inst'], 'x'):
                N = data['inst'].x.shape[0]
                s = data['inst'].s if hasattr(data['inst'], 's') and data['inst'].s is not None else None
                if s is not None:
                    s = s.to(device) if hasattr(s, 'to') else torch.tensor(s, device=device)
                sizes = _sizes_from_feats(data['inst'].x, s, N, device)
                # Validate sizes before returning
                if not torch.isfinite(sizes).all() or torch.any(sizes <= 0):
                    return torch.ones((N, 2), device=device) * 1.0
                return sizes
            N = data['inst'].pos.shape[0] if hasattr(data['inst'], 'pos') else 0
            if N == 0:
                return None
            return torch.ones((N, 2), device=device) * 1.0
        else:
            if getattr(data, 'instance_sizes', None) is not None:
                sizes = data.instance_sizes
                # Validate sizes before returning
                if not torch.isfinite(sizes).all() or torch.any(sizes <= 0):
                    N = sizes.shape[0]
                    return torch.ones((N, 2), device=device) * 1.0
                return sizes
            if hasattr(data, 'x') and data.x is not None:
                N = data.x.shape[0]
                s = data.s if hasattr(data, 's') and data.s is not None else None
                if s is not None:
                    s = s.to(device) if hasattr(s, 'to') else torch.tensor(s, device=device)
                sizes = _sizes_from_feats(data.x, s, N, device)
                # Validate sizes before returning
                if not torch.isfinite(sizes).all() or torch.any(sizes <= 0):
                    return torch.ones((N, 2), device=device) * 1.0
                return sizes
            if hasattr(data, 'pos'):
                N = data.pos.shape[0]
            else:
                N = 0
            if N == 0:
                return None
            return torch.ones((N, 2), device=device) * 1.0

    def training_step(self, data, z_reg: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Perform one diffusion training step on a batch of graphs.
        
        Supports both HeteroData and homogeneous Data formats.
        When loss or prediction is NaN/Inf, diagnostics run automatically (input checks + first layer with non-finite output).

        Returns:
            loss (scalar tensor), metrics dict
        """
        device = self.betas.device
        
        # Handle both HeteroData and homogeneous Data formats.
        # For homogeneous Data loaded via chipdiffusion pickle loader, prefer pos_target if present.
        is_hetero = hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data)
        inst_data = data["inst"] if is_hetero else data
        if is_hetero:
            x0 = inst_data.pos  # [N_inst, 2]
        else:
            x0 = inst_data.pos_target if hasattr(inst_data, "pos_target") and inst_data.pos_target is not None else inst_data.pos  # [N_inst, 2]
        mask = inst_data.pos_mask.bool() if hasattr(inst_data, "pos_mask") else torch.ones(x0.shape[0], dtype=torch.bool, device=device)

        # Pull fixed gauge parameters (mu, s) computed from chip_size (global normalization)
        # mu: (N, 2) global center [W/2, H/2] per node (same for all)
        # s: (N, 2) global anisotropic scale [W/2, H/2] per node (same for all)
        mu = inst_data.mu if hasattr(inst_data, "mu") else torch.zeros_like(x0)
        s = inst_data.s if hasattr(inst_data, "s") else torch.ones(x0.shape[0], 2, device=device, dtype=x0.dtype)
        
        # Ensure s has shape (N, 2) for anisotropic scaling
        # Handle backward compatibility: if s is (N, 1), expand to (N, 2)
        if s.ndim == 1:
            s = s.unsqueeze(-1)
        if s.shape[1] == 1:
            # Old format: (N, 1) -> expand to (N, 2) for isotropic scaling
            s = s.expand(-1, 2)
        elif s.shape[1] != 2:
            raise ValueError(f"s must have shape (N, 1) or (N, 2), got {s.shape}")
        
        # Ensure mu has shape (N, 2)
        if mu.ndim == 1:
            mu = mu.unsqueeze(-1)
        if mu.shape[1] == 1:
            mu = mu.expand(-1, 2)
        elif mu.shape[1] != 2:
            raise ValueError(f"mu must have shape (N, 1) or (N, 2), got {mu.shape}")
        
        mu = mu.to(device=device, dtype=x0.dtype)
        s = s.to(device=device, dtype=x0.dtype)

        # Refinement diffusion: start the forward process from X_init (if available), otherwise from X0.
        # For refinement flow matching: use z_init + small noise instead of pure noise
        x_init = inst_data.pos_init.to(device=device) if hasattr(inst_data, "pos_init") and inst_data.pos_init is not None else x0

        # Normalize to gauge space (normalized coordinate space) — skip for flow_matching_raw
        # Min/max normalization to [-1, 1]:
        #   Step 1: Centering: x~_i = x_i - W/2, y~_i = y_i - H/2
        #   Step 2: Min/max scale: z = x~ / (W/2, H/2)
        #   Result: Coordinates in [-1, 1]
        # z0 is in normalized space: z = (x - mu) / s where mu = [W/2, H/2], s = [W/2, H/2]
        if self.cfg.parameterization == "flow_matching_raw":
            z0 = x0  # (N, 2) raw physical positions
            z_init = x_init  # (N, 2) raw physical initial positions
        else:
            z0 = (x0 - mu) / s  # (N, 2) normalized clean positions
            z_init = (x_init - mu) / s  # (N, 2) normalized initial positions
        
        # Optional: log if inputs are already non-finite (data/gauge issue); then raise and save batch
        ok_z0, msg_z0 = _check_tensor_finite(z0, "z0")
        if not ok_z0:
            logger.warning("[training_step] Inputs before denoiser: %s", msg_z0)
            raise NaNDetectedError("NaN/Inf in input z0. Training stopped.", data=data, context="input (z0)")
        ok_s, msg_s = _check_tensor_finite(s, "s")
        if not ok_s:
            logger.warning("[training_step] Inputs before denoiser: %s", msg_s)
            raise NaNDetectedError("NaN/Inf in input s. Training stopped.", data=data, context="input (s)")
        
        # Initialize variables that may be used in metrics
        v_target = None

        if self.cfg.parameterization in ("flow_matching", "flow_matching_raw"):
            # Flow matching: linear path z_t = (1-t)*z_0 + t*z_1, v = z_1 - z_0.
            # Train to predict velocity; sample by ODE integration from t=0 (noise) to t=1 (data).
            # flow_matching: normalized space [-1,1]. flow_matching_raw: raw physical scale [0, W]x[0, H].
            # Refinement FM: if init_noise_std > 0, start from z_init + σ*ε instead of pure noise.
            if hasattr(data, 'batch') and data.batch is not None:
                batch_idx = data.batch
                num_graphs = batch_idx.max().item() + 1
            else:
                batch_idx = torch.zeros(x0.shape[0], dtype=torch.long, device=device)
                num_graphs = 1

            # Refinement flow matching: start from z_init + small noise instead of pure noise
            if self.cfg.init_noise_std > 0.0 and not torch.allclose(z_init, z0):
                # Use refinement: z_0 = z_init + σ*ε (initial layout + small noise)
                noise_std = self.cfg.init_noise_std
                eps = torch.randn_like(z_init)
                z_0 = z_init + noise_std * eps
            else:
                if z_reg is not None and self.cfg.parameterization == "flow_matching":
                    # Warm-start FM: use regression prediction as the noise endpoint z_0.
                    # Velocity target becomes v = z_1 - z_reg (correction from regression to truth).
                    # t is sampled over [0,1] as usual; the model learns corrections at all noise levels.
                    z_0 = z_reg.to(device=device, dtype=z0.dtype)
                elif self.cfg.parameterization == "flow_matching_raw":
                    # Raw physical scale: z_0 uniform in [0, W] x [0, H]; s = [W/2, H/2] so 2*s = [W, H]
                    physical_size = 2.0 * s
                    z_0 = torch.rand_like(z0, device=device, dtype=z0.dtype) * physical_size
                else:
                    # Standard flow matching: z_0 = uniform noise in [-1, 1] to match normalized coordinate space
                    z_0 = torch.rand_like(z0) * 2.0 - 1.0  # Uniform in [-1, 1]
            
            z_1 = z0
            if mask.any():
                z_0 = z_0.clone()
                z_0[~mask] = z_1[~mask]
            v = z_1 - z_0
            
            # Sample timestep
            if self.cfg.local_refinement_bias > 1.0:
                # Beta(α, β) with α > β biases toward 1 (late time)
                # Beta(local_refinement_bias, 1) gives mean = bias/(bias+1), e.g., Beta(3,1) → mean ≈ 0.75
                beta_dist = torch.distributions.Beta(
                    torch.tensor(self.cfg.local_refinement_bias, device=device),
                    torch.tensor(1.0, device=device)
                )
                t_cont_per_graph = beta_dist.sample((num_graphs,)).to(dtype=x0.dtype)
            else:
                t_cont_per_graph = torch.rand(num_graphs, device=device, dtype=x0.dtype)
            t_cont = t_cont_per_graph[batch_idx]
            
            z_t = (1.0 - t_cont).unsqueeze(-1) * z_0 + t_cont.unsqueeze(-1) * z_1
            z_t_conditioned = z_t.clone()
            if mask.any():
                z_t_conditioned[~mask] = z0[~mask]
            # Reject batch if t or coords out of safe range (skip to next); skip for raw (physical scale)
            if self.cfg.parameterization != "flow_matching_raw":
                _check_safe_range(t_cont, z_t_conditioned, z0, data)
            
            # Update features: prepend z_t to node features for encoder input.
            # Data from data_loading_homogeneous has features ONLY (no coords) -> use all.
            # Legacy formats may have coords in first 2 cols -> strip before prepending z_t.
            # Use model's expected input_dim when available to decide correctly.
            expected_input_dim = None
            if hasattr(self.denoiser, 'model') and hasattr(self.denoiser.model, 'input_dim'):
                expected_input_dim = self.denoiser.model.input_dim
            feat_dim = inst_data.x.shape[1]
            if expected_input_dim is not None:
                if feat_dim == expected_input_dim - 2:
                    orig_feats = inst_data.x  # features only (no coords)
                elif feat_dim == expected_input_dim:
                    orig_feats = inst_data.x[:, 2:]  # already has coords in first 2
                else:
                    orig_feats = inst_data.x
            else:
                orig_feats = inst_data.x[:, 2:] if feat_dim in (12, 15, 18, 24) else inst_data.x
            inst_data.x = torch.cat([z_t_conditioned, orig_feats], dim=1)
            
            v_pred, _ = self.denoiser(data, z_t_conditioned, t_cont)
            
            # Two-scale velocity: v_pred is (N, 4) -> v_phys = L*v_die + s_c*v_cell; loss in physical space
            ROW_HEIGHT = 0.9898  # μm (must match data_loading_homogeneous.py)
            if getattr(self.denoiser, "loop_gated_output_enabled", False) and not getattr(
                self.denoiser, "loop_gated_decode_once", True
            ):
                if getattr(self.cfg, 'two_scale_velocity', False) and v_pred.shape[-1] == 4:
                    raise NotImplementedError(
                        "Gated intermediate loss is not supported with two_scale_velocity yet."
                    )
                from unified_learning.models.placement.looped_fm_denoiser import (
                    compute_fm_gated_intermediate_loss,
                )
                loss = compute_fm_gated_intermediate_loss(
                    self.denoiser,
                    v,
                    mask=mask if mask.any() else None,
                    t_cont=t_cont,
                    use_refinement_weighting=self.cfg.use_refinement_weighting,
                    refinement_weight_strength=self.cfg.refinement_weight_strength,
                )
                v_target = v
            elif getattr(self.denoiser, "loop_pondernet_enabled", False):
                if getattr(self.cfg, 'two_scale_velocity', False) and v_pred.shape[-1] == 4:
                    raise NotImplementedError(
                        "PonderNet halting is not supported with two_scale_velocity yet."
                    )
                from unified_learning.models.placement.looped_fm_denoiser import (
                    compute_fm_ponder_loss,
                )
                loss = compute_fm_ponder_loss(
                    self.denoiser,
                    v,
                    mask=mask if mask.any() else None,
                    t_cont=t_cont,
                    use_refinement_weighting=self.cfg.use_refinement_weighting,
                    refinement_weight_strength=self.cfg.refinement_weight_strength,
                )
                v_target = v
            elif getattr(self.cfg, 'two_scale_velocity', False) and v_pred.shape[-1] == 4:
                v_die = v_pred[:, :2]
                v_cell = v_pred[:, 2:]
                # L = max(W, H) per node; s is (N, 2) with s = [W/2, H/2]
                W = 2.0 * s[:, 0]
                H = 2.0 * s[:, 1]
                L = torch.maximum(W, H).unsqueeze(-1)  # (N, 1)
                v_pred_phys = L * v_die + ROW_HEIGHT * v_cell
                v_target_phys = v * s
                # Loss in physical space; then convert v_pred to normalized for overlap/metrics
                if mask.any():
                    v_pred_masked = v_pred_phys[mask]
                    v_target_masked = v_target_phys[mask]
                else:
                    v_pred_masked = v_pred_phys
                    v_target_masked = v_target_phys
                if self.cfg.use_refinement_weighting:
                    t_weights = (1.0 + self.cfg.refinement_weight_strength * t_cont[mask]).unsqueeze(-1) if mask.any() else (1.0 + self.cfg.refinement_weight_strength * t_cont).unsqueeze(-1)
                else:
                    t_weights = 1.0
                sq_err = (t_weights * (v_pred_masked - v_target_masked) ** 2)
                if self.cfg.use_graph_weighted_loss and num_graphs > 1:
                    batch_idx_masked = batch_idx[mask] if mask.any() else batch_idx
                    se_per_node = sq_err.sum(dim=-1)
                    sum_se_per_graph = torch.zeros(num_graphs, device=device, dtype=sq_err.dtype)
                    sum_se_per_graph.scatter_add_(0, batch_idx_masked, se_per_node)
                    count_per_graph = torch.zeros(num_graphs, device=device, dtype=torch.long)
                    count_per_graph.scatter_add_(0, batch_idx_masked, torch.ones(batch_idx_masked.shape[0], device=device, dtype=torch.long))
                    loss_per_graph = sum_se_per_graph / count_per_graph.clamp(min=1).to(sq_err.dtype)
                    n_b = count_per_graph.float().mean()
                    graph_weights = count_per_graph.float() / n_b.clamp(min=1e-8) if self.cfg.graph_weight_mode == "normalized" else count_per_graph.float().pow(self.cfg.graph_weight_power)
                    loss = (graph_weights * loss_per_graph).sum() / graph_weights.sum().clamp(min=1e-8)
                else:
                    loss = sq_err.mean()
                v_target = v  # keep for metrics
                v_pred = v_pred_phys / s  # normalized for overlap and downstream (z1_hat, etc.)
            else:
                # Compute loss (single-scale velocity in normalized space)
                if mask.any():
                    v_pred_masked = v_pred[mask]
                    v_target_masked = v[mask]
                else:
                    v_pred_masked = v_pred
                    v_target_masked = v
                if self.cfg.use_refinement_weighting:
                    t_weights = (1.0 + self.cfg.refinement_weight_strength * t_cont[mask]).unsqueeze(-1) if mask.any() else (1.0 + self.cfg.refinement_weight_strength * t_cont).unsqueeze(-1)
                else:
                    t_weights = 1.0
                sq_err = (t_weights * (v_pred_masked - v_target_masked) ** 2)
                if self.cfg.use_graph_weighted_loss and num_graphs > 1:
                    batch_idx_masked = batch_idx[mask] if mask.any() else batch_idx
                    se_per_node = sq_err.sum(dim=-1)
                    sum_se_per_graph = torch.zeros(num_graphs, device=device, dtype=sq_err.dtype)
                    sum_se_per_graph.scatter_add_(0, batch_idx_masked, se_per_node)
                    count_per_graph = torch.zeros(num_graphs, device=device, dtype=torch.long)
                    count_per_graph.scatter_add_(0, batch_idx_masked, torch.ones(batch_idx_masked.shape[0], device=device, dtype=torch.long))
                    loss_per_graph = sum_se_per_graph / count_per_graph.clamp(min=1).to(sq_err.dtype)
                    n_b = count_per_graph.float().mean()
                    graph_weights = count_per_graph.float() / n_b.clamp(min=1e-8) if self.cfg.graph_weight_mode == "normalized" else count_per_graph.float().pow(self.cfg.graph_weight_power)
                    loss = (graph_weights * loss_per_graph).sum() / graph_weights.sum().clamp(min=1e-8)
                else:
                    loss = sq_err.mean()
                v_target = v
            
            # NaN/Inf diagnostics: log which layer produced non-finite output (always when NaN detected)
            if not torch.isfinite(v_pred).all():
                _log_nan_diagnostics(
                    self, data, z0, z_t_conditioned, t_cont, mu, s,
                    v_pred, float("nan"), "flow_matching (v_pred)" if self.cfg.parameterization == "flow_matching" else "flow_matching_raw (v_pred)"
                )
                solver_state = {
                    "parameterization": self.cfg.parameterization,
                    "t_cont": t_cont.detach().cpu(),
                    "z_t": z_t_conditioned.detach().cpu(),
                    "z0": z0.detach().cpu(),
                }
                raise NaNDetectedError(
                    "NaN/Inf in flow_matching (v_pred). Training stopped." if self.cfg.parameterization == "flow_matching" else "NaN/Inf in flow_matching_raw (v_pred). Training stopped.",
                    data=data,
                    context="flow_matching (v_pred)" if self.cfg.parameterization == "flow_matching" else "flow_matching_raw (v_pred)",
                    solver_state=solver_state,
                )
            
            # Add overlap-aware loss if enabled (for flow matching)
            if self.cfg.use_overlap_loss:
                sizes = self._extract_sizes_from_data(data, device)
                if sizes is not None and sizes.shape[0] == x0.shape[0]:
                    # Validate v_pred for NaN/Inf before computing overlap penalty
                    if torch.isnan(v_pred).any() or torch.isinf(v_pred).any():
                        # Skip overlap penalty if v_pred contains NaN/Inf
                        # This prevents NaN propagation into loss
                        pass
                    else:
                        # Compute predicted positions in physical space
                        # For flow matching: z1_hat = z_t + (1-t) * v_pred
                        t_exp = t_cont.unsqueeze(-1)
                        z1_hat = z_t_conditioned + (1.0 - t_exp) * v_pred
                        if self.cfg.parameterization == "flow_matching_raw":
                            x0_pred_physical = z1_hat  # already in physical space
                        else:
                            x0_pred_physical = z1_hat * s + mu
                        
                        # Validate sizes and positions for NaN/Inf
                        if (torch.isfinite(sizes).all() and torch.isfinite(x0_pred_physical).all() and
                            torch.isfinite(s).all() and torch.isfinite(mu).all()):
                            overlap_penalty = compute_overlap_penalty(
                                x0_pred_physical,
                                sizes,
                                mask=mask if mask.any() else None,
                                threshold=self.cfg.overlap_penalty_threshold,
                            )
                            # Validate overlap_penalty before adding to loss
                            if torch.isfinite(overlap_penalty):
                                loss = loss + self.cfg.overlap_loss_weight * overlap_penalty

            # NaN/Inf diagnostics when loss is non-finite (after all loss terms)
            if not torch.isfinite(loss).all():
                _log_nan_diagnostics(
                    self, data, z0, z_t_conditioned, t_cont, mu, s,
                    v_pred, loss.item(), "flow_matching (loss)" if self.cfg.parameterization == "flow_matching" else "flow_matching_raw (loss)"
                )
                solver_state = {
                    "parameterization": self.cfg.parameterization,
                    "t_cont": t_cont.detach().cpu(),
                    "z_t": z_t_conditioned.detach().cpu(),
                    "z0": z0.detach().cpu(),
                }
                raise NaNDetectedError(
                    "NaN/Inf in flow_matching (loss). Training stopped." if self.cfg.parameterization == "flow_matching" else "NaN/Inf in flow_matching_raw (loss). Training stopped.",
                    data=data,
                    context="flow_matching (loss)" if self.cfg.parameterization == "flow_matching" else "flow_matching_raw (loss)",
                    solver_state=solver_state,
                )

            t_idxs = None
            eps_pred = v_pred
            noise = z_0
        elif self.cfg.parameterization == "edm_x0_precond":
            # EDM-style continuous sigma training with preconditioning in normalized z space.
            u = torch.rand((x0.shape[0], 1), device=device, dtype=x0.dtype)
            sigma_min = torch.tensor(float(self.cfg.sigma_min), device=device, dtype=x0.dtype)
            sigma_max = torch.tensor(float(self.cfg.sigma_max), device=device, dtype=x0.dtype)
            sigma = sigma_min * (sigma_max / sigma_min) ** u  # (N,1)

            eps = torch.randn_like(z0)  # unit noise in normalized space
            z_sigma = z_init + sigma * eps
            # Reject batch if coords out of safe range (EDM t is log-sigma, not in [0,1])
            _check_safe_range_coords(z_sigma, z0, data)

            sigma_data = torch.tensor(float(self.cfg.sigma_data), device=device, dtype=x0.dtype)
            c_skip = (sigma_data ** 2) / (sigma ** 2 + sigma_data ** 2)
            c_out = (sigma * sigma_data) / torch.sqrt(sigma ** 2 + sigma_data ** 2)
            c_in = 1.0 / torch.sqrt(sigma ** 2 + sigma_data ** 2)
            c_noise = 0.25 * torch.log(torch.clamp(sigma, min=1e-12))  # (N,1)

            z_in = c_in * z_sigma
            t_cont = c_noise.squeeze(-1)

            # For EDM: The model receives z_sigma (unscaled noisy positions) for positional encoding
            # Even though z_sigma can be large (std ~18 when sigma is large), the positional encoder
            # will clamp it to [-1, 1] which is acceptable - it's just for encoding rough position
            # The model predicts z0 (clean positions in normalized space, same as z_sigma space)
            # Note: EDM preconditioning (c_skip, c_out) is handled here, not in the model
            # CRITICAL: Model must predict in z_sigma space (normalized), not z_in space (scaled)
            f_pred, _ = self.denoiser(data, z_sigma, t_cont)
            
            # Verify output shape
            if f_pred.shape != z0.shape:
                raise ValueError(f"Model output shape {f_pred.shape} doesn't match z0 shape {z0.shape}. "
                               f"Expected {z0.shape}, got {f_pred.shape}")
            
            # EDM preconditioning: z0_hat = c_skip * z_sigma + c_out * f_pred
            # f_pred should be in normalized space (same as z_sigma)
            # c_skip and c_out handle the scaling
            z0_hat = c_skip * z_sigma + c_out * f_pred
            
            # Debug: Check model inputs, outputs, and loss computation (only first batch, first epoch)
            if not hasattr(self, '_debug_printed'):
                self._debug_printed = True
                print(f"\n{'='*80}")
                print(f"DEBUG: EDM Training Step Analysis")
                print(f"{'='*80}")
                print(f"Input shapes:")
                print(f"  z_sigma (x_t): {z_sigma.shape}, mean={z_sigma.mean().item():.6f}, std={z_sigma.std().item():.6f}")
                print(f"  t_cont: {t_cont.shape}, mean={t_cont.mean().item():.6f}, std={t_cont.std().item():.6f}, "
                      f"min={t_cont.min().item():.6f}, max={t_cont.max().item():.6f}")
                print(f"  data.x (node features): {data.x.shape if hasattr(data, 'x') else 'N/A'}")
                print(f"\nModel output:")
                print(f"  f_pred: {f_pred.shape}, mean={f_pred.mean().item():.6f}, std={f_pred.std().item():.6f}, "
                      f"min={f_pred.min().item():.6f}, max={f_pred.max().item():.6f}")
                print(f"\nTarget:")
                print(f"  z0: {z0.shape}, mean={z0.mean().item():.6f}, std={z0.std().item():.6f}, "
                      f"min={z0.min().item():.6f}, max={z0.max().item():.6f}")
                print(f"\nEDM coefficients:")
                print(f"  c_skip: {c_skip.mean().item():.6f} (shape: {c_skip.shape})")
                print(f"  c_out: {c_out.mean().item():.6f} (shape: {c_out.shape})")
                print(f"  c_in: {c_in.mean().item():.6f} (shape: {c_in.shape})")
                print(f"  sigma: {sigma.mean().item():.6f} (shape: {sigma.shape})")
                print(f"\nComputed values:")
                print(f"  z0_hat: mean={z0_hat.mean().item():.6f}, std={z0_hat.std().item():.6f}")
                print(f"  z0_hat - z0: mean={(z0_hat - z0).mean().item():.6f}, std={(z0_hat - z0).std().item():.6f}")
                print(f"  f_pred - z0: mean={(f_pred - z0).mean().item():.6f}, std={(f_pred - z0).std().item():.6f}")
                w = (sigma ** 2 + sigma_data ** 2) / torch.clamp((sigma * sigma_data) ** 2, min=1e-12)
                loss_val = (w * (z0_hat - z0) ** 2).mean()
                print(f"\nLoss:")
                print(f"  w (weight): mean={w.mean().item():.6f}, min={w.min().item():.6f}, max={w.max().item():.6f}")
                print(f"  loss: {loss_val.item():.6f}")
                print(f"{'='*80}\n")
            
            # Store f_pred for eps statistics (prediction error in normalized space)
            eps_pred = (f_pred - z0)  # Prediction error: should be close to 0 mean

            # Loss weighting from EDM: (sigma^2 + sigma_data^2) / (sigma*sigma_data)^2
            w = (sigma ** 2 + sigma_data ** 2) / torch.clamp((sigma * sigma_data) ** 2, min=1e-12)

            if mask.any():
                loss = (w[mask] * (z0_hat[mask] - z0[mask]) ** 2).mean()
            else:
                loss = (w * (z0_hat - z0) ** 2).mean()
            
            # Add overlap-aware loss if enabled
            if self.cfg.use_overlap_loss:
                sizes = self._extract_sizes_from_data(data, device)
                if sizes is not None and sizes.shape[0] == x0.shape[0]:
                    # Validate z0_hat for NaN/Inf before computing overlap penalty
                    if torch.isnan(z0_hat).any() or torch.isinf(z0_hat).any():
                        # Skip overlap penalty if z0_hat contains NaN/Inf
                        # This prevents NaN propagation into loss
                        pass
                    else:
                        # Compute predicted positions in physical space
                        x0_pred_physical = z0_hat * s + mu
                        
                        # Validate sizes and positions for NaN/Inf
                        if (torch.isfinite(sizes).all() and torch.isfinite(x0_pred_physical).all() and
                            torch.isfinite(s).all() and torch.isfinite(mu).all()):
                            overlap_penalty = compute_overlap_penalty(
                                x0_pred_physical,
                                sizes,
                                mask=mask if mask.any() else None,
                                threshold=self.cfg.overlap_penalty_threshold,
                            )
                            # Validate overlap_penalty before adding to loss
                            if torch.isfinite(overlap_penalty):
                                loss = loss + self.cfg.overlap_loss_weight * overlap_penalty

            # NaN/Inf diagnostics when loss is non-finite
            if not torch.isfinite(loss).all():
                _log_nan_diagnostics(
                    self, data, z0, z_sigma, t_cont, mu, s,
                    f_pred, loss.item(), "edm_x0_precond (loss)"
                )
                solver_state = {
                    "parameterization": "edm_x0_precond",
                    "t_cont": t_cont.detach().cpu(),
                    "z_t": z_sigma.detach().cpu(),
                    "z0": z0.detach().cpu(),
                }
                raise NaNDetectedError(
                    "NaN/Inf in edm_x0_precond (loss). Training stopped.",
                    data=data,
                    context="edm_x0_precond (loss)",
                    solver_state=solver_state,
                )

            # eps_pred is now the prediction error (f_pred - z0) for EDM
            noise = eps
            z_t = z_sigma
            t_idxs = None
        else:
            # Classic discrete DDPM epsilon-prediction in normalized z space.
            # CRITICAL: All operations are in normalized coordinate space:
            # - z0: normalized clean positions (z0 = (x0 - mu) / s)
            # - noise: sampled from N(0, I) in normalized space
            # - z_t: noisy positions in normalized space (z_t = sqrt(alpha_bar) * z0 + sqrt(1-alpha_bar) * noise)
            # - eps_pred: predicted noise in normalized space (model output)
            # - Loss: MSE between eps_pred and noise (both in normalized space)
            
            # Sample timesteps per GRAPH (not per node!)
            # In PyG batched graphs, data.batch indicates which graph each node belongs to
            # We want one timestep per graph, expanded to all nodes in that graph
            if hasattr(data, 'batch') and data.batch is not None:
                batch_idx = data.batch  # (N,) tensor with graph index for each node
                num_graphs = batch_idx.max().item() + 1
            else:
                # Single graph (not batched)
                batch_idx = torch.zeros(x0.shape[0], dtype=torch.long, device=device)
                num_graphs = 1
            
            # Sample one timestep per graph
            t_idxs_per_graph, t_cont_per_graph = self._sample_timesteps(num_graphs, device=device)
            
            # Expand to per-node timesteps
            t_idxs = t_idxs_per_graph[batch_idx]  # (N,) - same timestep for all nodes in same graph
            t_cont = t_cont_per_graph[batch_idx]  # (N,) - same continuous time for all nodes in same graph

            # Forward diffusion: add noise to z0 (both in normalized space)
            # CRITICAL: Macros and ports are FIXED - do NOT add noise to them!
            # Only std-cells should have noisy positions during diffusion
            # Macro/port clean positions provide valuable conditioning for std-cell placement
            z_t, noise = self.q_sample(z0, t_idxs)  # noise is in normalized space
            
            # Mask macro/port positions: keep their clean positions (z0) as conditioning
            # This is critical: fixed macro/port positions guide std-cell placement
            if mask.any():
                # mask indicates which nodes should be diffused (mask=True means diffuse)
                # For macros/ports (mask=False), use clean positions z0 instead of noisy z_t
                z_t_conditioned = z_t.clone()
                z_t_conditioned[~mask] = z0[~mask]  # Keep macro/port positions clean
                noise_masked = noise.clone()
                noise_masked[~mask] = 0.0  # No noise for macros/ports
            else:
                z_t_conditioned = z_t
                noise_masked = noise
            # Reject batch if t or coords out of safe range (skip to next)
            _check_safe_range(t_cont, z_t_conditioned, z0, data)

            # Update node features with z_t_conditioned (noisy for std-cells, clean for macros/ports)
            # Node features: [z_tx, z_ty, w_rel, h_rel, r, area_rel, w_norm_d, h_norm_d, w_raw, h_raw,
            #                 chip_w_raw, chip_h_raw, chip_w_rel, chip_h_rel, num_stdcells, num_macros,
            #                 num_ports, num_edges, degree, log_degree, neighbor_deg_mean, is_stdcell, is_macro, is_port] (24 dims)
            # Original node features are [w_rel, h_rel, ...] (22 dims without z_t)
            # We need to prepend z_t_conditioned to make [z_tx, z_ty, ...] (24 dims)
            # CRITICAL: z_t_conditioned contains noisy positions for std-cells and CLEAN positions for macros/ports
            if hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data):
                feats = self._get_conditioning_features(data['inst'].x)
                data['inst'].x = torch.cat([z_t_conditioned, feats], dim=1)
            else:
                feats = self._get_conditioning_features(data.x)
                data.x = torch.cat([z_t_conditioned, feats], dim=1)

            # Denoise (predict epsilon)
            # Model predicts noise in normalized coordinate space
            # eps_pred should be in the same space as noise (normalized space)
            # CRITICAL: Model receives z_t_conditioned (with clean macro/port positions as conditioning)
            eps_pred, _ = self.denoiser(data, z_t_conditioned, t_cont)  # eps_pred is in normalized space

            # Compute loss with v-prediction and SNR weighting
            # v-prediction parameterization: v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x0
            # This provides better gradient flow than pure epsilon prediction
            # Loss compares predictions to targets (actual noise/velocity, 0 for macros/ports)
            # IMPORTANT: mask ensures we only compute loss on std-cells, not macros/ports
            
            alphas_cumprod_t = self.alphas_cumprod[t_idxs].unsqueeze(-1)  # [N, 1]
            sqrt_alpha_bar = torch.sqrt(alphas_cumprod_t + 1e-8)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alphas_cumprod_t + 1e-8)
            
            # Compute v-target (velocity parameterization)
            v_target = sqrt_alpha_bar * noise_masked - sqrt_one_minus_alpha_bar * z0
            
            # Convert epsilon prediction to v-prediction
            # For epsilon prediction: x0_pred = (z_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
            # Then: v_pred = sqrt(alpha_bar) * eps - sqrt(1-alpha_bar) * x0_pred
            # Simplified: v_pred = sqrt(alpha_bar) * eps - (z_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar) * sqrt(1-alpha_bar)
            # Further: v_pred = eps * sqrt(alpha_bar) - z_t * sqrt(1-alpha_bar) / sqrt(alpha_bar) + eps * (1-alpha_bar) / sqrt(alpha_bar)
            # Final: v_pred = eps - z_t * sqrt(1-alpha_bar) / sqrt(alpha_bar)
            v_pred = sqrt_alpha_bar * eps_pred - sqrt_one_minus_alpha_bar * ((z_t_conditioned - sqrt_one_minus_alpha_bar * eps_pred) / (sqrt_alpha_bar + 1e-8))
            
            # SNR-based weighting (Signal-to-Noise Ratio)
            # SNR = alpha_bar / (1 - alpha_bar)
            # Higher SNR (low noise) → lower weight (easier to predict)
            # Lower SNR (high noise) → higher weight (harder to predict, need more focus)
            snr = alphas_cumprod_t / (1.0 - alphas_cumprod_t + 1e-8)
            # Clamp weights to avoid extreme values
            # snr / (snr + 1) maps SNR to [0, 1]: high SNR → 1, low SNR → 0
            # We want inverse: high noise (low SNR) → high weight
            # Use 1 / (snr + 1); only clamp snr for numerical stability, not the weights
            snr_weights = 1.0 / (1.0 + snr.clamp(min=1e-4, max=1e4))
            
            # Compute loss
            if mask.any():
                loss_v = (v_pred[mask] - v_target[mask]) ** 2
                loss = (snr_weights[mask] * loss_v).mean()
            else:
                loss_v = (v_pred - v_target) ** 2
                loss = (snr_weights * loss_v).mean()
            
            # Auxiliary loss: MSE on reconstructed x0 (helps stabilize training)
            # x0_pred = (z_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
            x0_pred = (z_t_conditioned - sqrt_one_minus_alpha_bar * eps_pred) / (sqrt_alpha_bar + 1e-8)
            if mask.any():
                loss_x0 = torch.nn.functional.mse_loss(x0_pred[mask], z0[mask])
            else:
                loss_x0 = torch.nn.functional.mse_loss(x0_pred, z0)
            
            # Combine losses: v-prediction (main) + x0 reconstruction (auxiliary)
            loss = loss + 0.1 * loss_x0
            
            # Add overlap-aware loss if enabled
            if self.cfg.use_overlap_loss:
                # Extract sizes from data
                sizes = self._extract_sizes_from_data(data, device)
                if sizes is not None and sizes.shape[0] == x0.shape[0]:
                    # Validate x0_pred for NaN/Inf before computing overlap penalty
                    if torch.isnan(x0_pred).any() or torch.isinf(x0_pred).any():
                        # Skip overlap penalty if x0_pred contains NaN/Inf
                        # This prevents NaN propagation into loss
                        pass
                    else:
                        # Compute predicted positions in physical space
                        x0_pred_physical = x0_pred * s + mu
                        # Compute overlap penalty on predicted positions
                        # Only penalize overlaps for diffused instances (std-cells)
                        
                        # Validate sizes and positions for NaN/Inf
                        if (torch.isfinite(sizes).all() and torch.isfinite(x0_pred_physical).all() and
                            torch.isfinite(s).all() and torch.isfinite(mu).all()):
                            overlap_penalty = compute_overlap_penalty(
                                x0_pred_physical,
                                sizes,
                                mask=mask if mask.any() else None,
                                threshold=self.cfg.overlap_penalty_threshold,
                            )
                            # Validate overlap_penalty before adding to loss
                            if torch.isfinite(overlap_penalty):
                                # Add overlap loss to main loss
                                loss = loss + self.cfg.overlap_loss_weight * overlap_penalty

            # NaN/Inf diagnostics when loss is non-finite
            if not torch.isfinite(loss).all():
                _log_nan_diagnostics(
                    self, data, z0, z_t_conditioned, t_cont, mu, s,
                    eps_pred, loss.item(), "ddpm (loss)"
                )
                solver_state = {
                    "parameterization": "ddpm",
                    "t_cont": t_cont.detach().cpu(),
                    "z_t": z_t_conditioned.detach().cpu(),
                    "z0": z0.detach().cpu(),
                }
                raise NaNDetectedError(
                    "NaN/Inf in ddpm (loss). Training stopped.",
                    data=data,
                    context="ddpm (loss)",
                    solver_state=solver_state,
                )

        # Simple RMSE on reconstructed x0_hat as a sanity metric
        # NOTE: This is averaged across instances with DIFFERENT timesteps
        # High-noise timesteps (t=75-99) dominate due to amplification, so rmse_x0
        # will stay high even as eps_pred improves, until eps_pred_std reaches ~1.0
        rmse_x0_unnorm = None  # Initialize to avoid UnboundLocalError
        eps_mean = None
        eps_variance = None
        coord_error_pct = None
        # Flow matching velocity statistics (initialized here, set in metrics block)
        v_pred_mean = None
        v_pred_std = None
        v_target_mean = None
        v_target_std = None
        
        with torch.no_grad():
            if self.cfg.parameterization == "edm_x0_precond":
                x0_hat = z0_hat * s + mu  # back to raw units
                if mask.any():
                    mse_x0 = torch.nn.functional.mse_loss(x0_hat[mask], x0[mask])
                    x0_hat_masked = x0_hat[mask]
                    x0_masked = x0[mask]
                else:
                    mse_x0 = torch.nn.functional.mse_loss(x0_hat, x0)
                    x0_hat_masked = x0_hat
                    x0_masked = x0
                rmse_x0 = torch.sqrt(mse_x0)
                rmse_x0_low, rmse_x0_mid, rmse_x0_high = 0.0, 0.0, 0.0
                
                # For EDM, compute eps statistics from prediction error (f_pred - z0)
                # This measures how well the model predicts z0
                if eps_pred is not None:
                    if mask.any():
                        eps_pred_masked = eps_pred[mask]
                    else:
                        eps_pred_masked = eps_pred
                    eps_pred_flat = eps_pred_masked.flatten()
                    eps_mean = eps_pred_flat.mean().item()
                    eps_variance = eps_pred_flat.var().item()
                else:
                    # Fallback: use z0_hat - z0
                    if mask.any():
                        eps_like = (z0_hat[mask] - z0[mask]).flatten()
                    else:
                        eps_like = (z0_hat - z0).flatten()
                    eps_mean = eps_like.mean().item()
                    eps_variance = eps_like.var().item()
            elif self.cfg.parameterization in ("flow_matching", "flow_matching_raw"):
                # Flow matching: z_t = (1-t) * z_0 + t * z_1 = z_0 + t * v, where v = z_1 - z_0
                # To recover z_1 (data): z_1 = z_t + (1-t) * v
                t_exp = t_cont.unsqueeze(-1)
                v_pred = eps_pred  # eps_pred stores v_pred for flow_matching
                z1_hat = z_t + (1.0 - t_exp) * v_pred  # Recover z_1 (data), not z_0
                if self.cfg.parameterization == "flow_matching_raw":
                    x0_hat = z1_hat  # already physical
                else:
                    x0_hat = z1_hat * s + mu
                if mask.any():
                    mse_x0 = torch.nn.functional.mse_loss(x0_hat[mask], x0[mask])
                    x0_hat_masked = x0_hat[mask]
                    x0_masked = x0[mask]
                    v_pred_masked = v_pred[mask]
                    v_target_masked = v_target[mask] if v_target is not None else None
                else:
                    mse_x0 = torch.nn.functional.mse_loss(x0_hat, x0)
                    x0_hat_masked = x0_hat
                    x0_masked = x0
                    v_pred_masked = v_pred
                    v_target_masked = v_target if v_target is not None else None
                rmse_x0 = torch.sqrt(mse_x0)
                rmse_x0_unnorm = rmse_x0
                # Flow matching metrics: velocity statistics instead of epsilon
                if v_pred_masked is not None:
                    v_pred_flat = v_pred_masked.flatten()
                    v_mean = v_pred_flat.mean().item()
                    v_std = v_pred_flat.std().item()
                    if v_target_masked is not None:
                        v_target_flat = v_target_masked.flatten()
                        v_tgt_mean = v_target_flat.mean().item()
                        v_tgt_std = v_target_flat.std().item()
                    else:
                        v_tgt_mean = None
                        v_tgt_std = None
                else:
                    v_mean = None
                    v_std = None
                    v_tgt_mean = None
                    v_tgt_std = None
                eps_mean = v_mean  # Reuse eps_mean for velocity mean (backward compat)
                eps_variance = v_std ** 2 if v_std is not None else None  # Reuse eps_variance for velocity variance
                # Store velocity stats for metrics dict (assign from local vars to outer scope)
                v_pred_mean = v_mean
                v_pred_std = v_std
                v_target_mean = v_tgt_mean
                v_target_std = v_tgt_std
                valid_mask = mask if mask.any() else torch.ones(x0.shape[0], dtype=torch.bool, device=device)
                low_noise = (t_cont < 0.25) & valid_mask
                mid_noise = ((t_cont >= 0.25) & (t_cont < 0.75)) & valid_mask
                high_noise = (t_cont >= 0.75) & valid_mask
                rmse_x0_low = torch.sqrt(torch.nn.functional.mse_loss(x0_hat[low_noise], x0[low_noise])).item() if low_noise.any() else 0.0
                rmse_x0_mid = torch.sqrt(torch.nn.functional.mse_loss(x0_hat[mid_noise], x0[mid_noise])).item() if mid_noise.any() else 0.0
                rmse_x0_high = torch.sqrt(torch.nn.functional.mse_loss(x0_hat[high_noise], x0[high_noise])).item() if high_noise.any() else 0.0
            else:
                alphas_cumprod = self.alphas_cumprod[t_idxs].unsqueeze(-1)
                std = torch.sqrt(1.0 - alphas_cumprod)
                sqrt_alpha = torch.sqrt(alphas_cumprod.clamp(min=1e-8))
                z0_hat = (z_t - std * eps_pred) / sqrt_alpha
                x0_hat = z0_hat * s + mu
                if mask.any():
                    mse_x0 = torch.nn.functional.mse_loss(x0_hat[mask], x0[mask])
                    x0_hat_masked = x0_hat[mask]
                    x0_masked = x0[mask]
                    eps_pred_masked = eps_pred[mask]
                else:
                    mse_x0 = torch.nn.functional.mse_loss(x0_hat, x0)
                    x0_hat_masked = x0_hat
                    x0_masked = x0
                    eps_pred_masked = eps_pred
                rmse_x0 = torch.sqrt(mse_x0)

                # New recommended normalization keeps positions in raw units (no min-max / [-1,1] denormalization).
                # So rmse_x0 is already in the correct unnormalized space.
                rmse_x0_unnorm = rmse_x0
                
                # Compute eps statistics (should be mean ~0, variance ~1)
                eps_pred_flat = eps_pred_masked.flatten()
                eps_mean = eps_pred_flat.mean().item()
                eps_variance = eps_pred_flat.var().item()
            
            # Compute average absolute coordinate error in percentage
            # Error percentage = mean(|x0_hat - x0|) / chip_size * 100
            # Use true chip size [W, H] from data.chip_size when available (s is diagonal scale [D, D], not [W/2, H/2])
            abs_error = torch.abs(x0_hat_masked - x0_masked)  # (N, 2)
            chip_size_per_dim = None
            if hasattr(data, "chip_size") and data.chip_size is not None:
                cs = data.chip_size.to(device=device, dtype=abs_error.dtype)
                if cs.numel() == 2:
                    # Single graph: reshape to (2,) first, then expand to (N_masked, 2) or (N, 2)
                    cs = cs.view(2)  # Ensure shape is (2,) regardless of input shape
                    if mask.any():
                        chip_size_per_dim = cs.unsqueeze(0).expand(mask.sum().item(), 2)
                    else:
                        chip_size_per_dim = cs.unsqueeze(0).expand(x0.shape[0], 2)
                elif cs.dim() == 2 and cs.shape[0] == x0.shape[0]:
                    # Per-node chip_size (N, 2)
                    chip_size_per_dim = cs[mask] if mask.any() else cs
                elif cs.dim() == 2 and hasattr(data, "batch") and data.batch is not None:
                    # Batched: chip_size (num_graphs, 2), expand by data.batch
                    chip_size_per_dim = cs[data.batch]  # (N, 2)
                    if mask.any():
                        chip_size_per_dim = chip_size_per_dim[mask]
            if chip_size_per_dim is None:
                # Fallback: infer from s (diagonal norm gives s = [D, D]; true [W,H] not available)
                if mask.any():
                    chip_size_per_dim = s[mask] * 2.0  # 2*D per dim, approximate
                else:
                    chip_size_per_dim = s * 2.0
            chip_size_per_dim = torch.clamp(chip_size_per_dim, min=1e-8)
            error_pct_per_dim = abs_error / chip_size_per_dim * 100.0  # (N, 2) percentage error per dimension
            coord_error_pct = error_pct_per_dim.mean().item()  # Average across all instances and dimensions

            if self.cfg.parameterization not in ("edm_x0_precond", "flow_matching", "flow_matching_raw"):
                # DDPM only: compute rmse_x0 separately for different noise levels (flow_matching does this in its branch)
                # Group by noise level: low (t < num_steps//4), mid (num_steps//4 <= t < num_steps*3//4), high (t >= num_steps*3//4)
                if mask.any():
                    valid_mask = mask
                else:
                    valid_mask = torch.ones(x0.shape[0], dtype=torch.bool, device=device)
                num_steps = self.num_steps
                low_threshold = num_steps // 4
                mid_threshold = num_steps * 3 // 4
                low_noise_mask = (t_idxs < low_threshold) & valid_mask
                mid_noise_mask = ((t_idxs >= low_threshold) & (t_idxs < mid_threshold)) & valid_mask
                high_noise_mask = (t_idxs >= mid_threshold) & valid_mask
                if low_noise_mask.any():
                    rmse_x0_low = torch.sqrt(torch.nn.functional.mse_loss(x0_hat[low_noise_mask], x0[low_noise_mask])).item()
                else:
                    rmse_x0_low = 0.0
                if mid_noise_mask.any():
                    rmse_x0_mid = torch.sqrt(torch.nn.functional.mse_loss(x0_hat[mid_noise_mask], x0[mid_noise_mask])).item()
                else:
                    rmse_x0_mid = 0.0
                if high_noise_mask.any():
                    rmse_x0_high = torch.sqrt(torch.nn.functional.mse_loss(x0_hat[high_noise_mask], x0[high_noise_mask])).item()
                else:
                    rmse_x0_high = 0.0

        metrics = {
            "loss": loss.item(),
            "rmse_x0": rmse_x0.item(),
            "rmse_x0_low": rmse_x0_low,   # Low noise (t < num_steps//4) or low t (<0.25) for flow_matching
            "rmse_x0_mid": rmse_x0_mid,   # Mid noise (num_steps//4 <= t < num_steps*3//4) or mid t (0.25-0.75) for flow_matching
            "rmse_x0_high": rmse_x0_high, # High noise (t >= num_steps*3//4) or high t (>=0.75) for flow_matching
        }
        if rmse_x0_unnorm is not None:
            metrics["rmse_x0_unnorm"] = rmse_x0_unnorm.item()
        if eps_mean is not None:
            # For flow_matching: eps_mean is v_mean (velocity mean), eps_variance is v_variance
            metrics["eps_mean"] = eps_mean
        if eps_variance is not None:
            metrics["eps_variance"] = eps_variance
        if coord_error_pct is not None:
            metrics["coord_error_pct"] = coord_error_pct
        # Flow matching specific metrics: velocity statistics
        # NOTE: If loss decreases but coord_error doesn't, check v_pred_std vs v_target_std
        # A scale mismatch (v_pred too small/large) can cause low loss but high coord_error
        if self.cfg.parameterization in ("flow_matching", "flow_matching_raw"):
            if v_pred_mean is not None:
                metrics["v_pred_mean"] = v_pred_mean
            if v_pred_std is not None:
                metrics["v_pred_std"] = v_pred_std
            if v_target_mean is not None:
                metrics["v_target_mean"] = v_target_mean
            if v_target_std is not None:
                metrics["v_target_std"] = v_target_std
        return loss, metrics

    def _build_fm_test_time_controller(
        self,
        *,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> FlowMatchingTestTimeController:
        return FlowMatchingTestTimeController(
            self.cfg.sampling_techniques,
            num_nodes=num_nodes,
            device=device,
            dtype=dtype,
        )

    def _prepare_fm_conditioned_state(
        self,
        data,
        z_state: torch.Tensor,
        original_feats: torch.Tensor,
        z0_clean: Optional[torch.Tensor],
        mask: torch.Tensor,
        clip_stats: dict,
    ) -> torch.Tensor:
        z_conditioned = z_state.clone()
        if z0_clean is not None:
            z_conditioned[~mask] = z0_clean[~mask]

        if self.cfg.parameterization == "flow_matching":
            fm_input_clip = clip_stats["input_clip"]
            if mask.any():
                pre_abs = float(z_conditioned[mask].abs().max().item())
                clip_stats["max_abs_pre_clip"] = max(clip_stats["max_abs_pre_clip"], pre_abs)
                if pre_abs > fm_input_clip:
                    clip_stats["clip_count"] += 1
                    z_conditioned[mask] = z_conditioned[mask].clamp(-fm_input_clip, fm_input_clip)
            else:
                pre_abs = float(z_conditioned.abs().max().item())
                clip_stats["max_abs_pre_clip"] = max(clip_stats["max_abs_pre_clip"], pre_abs)
                if pre_abs > fm_input_clip:
                    clip_stats["clip_count"] += 1
                    z_conditioned = z_conditioned.clamp(-fm_input_clip, fm_input_clip)

        if hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data):
            data['inst'].x = torch.cat([z_conditioned, original_feats], dim=1)
        else:
            data.x = torch.cat([z_conditioned, original_feats], dim=1)
        return z_conditioned

    def _evaluate_fm_velocity(
        self,
        data,
        z_state: torch.Tensor,
        t_value: float,
        original_feats: torch.Tensor,
        z0_clean: Optional[torch.Tensor],
        mask: torch.Tensor,
        s: torch.Tensor,
        clip_stats: dict,
        controller: FlowMatchingTestTimeController,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_conditioned = self._prepare_fm_conditioned_state(
            data=data,
            z_state=z_state,
            original_feats=original_feats,
            z0_clean=z0_clean,
            mask=mask,
            clip_stats=clip_stats,
        )
        t_remapped = controller.remap_time(t_value)
        t_cont = torch.full((z_state.shape[0],), t_remapped, device=z_state.device, dtype=z_state.dtype)
        v_pred, _ = self.denoiser(data, z_conditioned, t_cont)
        if getattr(self.cfg, 'two_scale_velocity', False) and v_pred.shape[-1] == 4:
            v_pred = self._two_scale_velocity_to_normalized(v_pred, s)
        v_pred = controller.apply_velocity_scale(v_pred, t_value)
        return v_pred, z_conditioned

    def _apply_fm_post_step(
        self,
        data,
        z_state: torch.Tensor,
        *,
        t_value: float,
        dt_value: float,
        mask: torch.Tensor,
        z0_clean: Optional[torch.Tensor],
        s: torch.Tensor,
        mu: torch.Tensor,
        controller: FlowMatchingTestTimeController,
    ) -> torch.Tensor:
        guidance_every = getattr(self.cfg, "guidance_every_steps", 1)
        step_idx = int(max(round(t_value * max(self.num_steps, 1)), 0))
        do_guidance = (
            self.cfg.use_guided_sampling
            and t_value > 0.3
            and (step_idx % max(guidance_every, 1) == 0)
        )
        if do_guidance:
            sizes = self._extract_sizes_from_data(data, z_state.device)
            if sizes is not None and sizes.shape[0] == z_state.shape[0]:
                if self.cfg.parameterization == "flow_matching_raw":
                    x_t_physical = z_state
                    overlap_grad_norm = compute_overlap_gradient(
                        x_t_physical,
                        sizes,
                        mask=mask if mask.any() else None,
                        threshold=self.cfg.overlap_penalty_threshold,
                    )
                else:
                    x_t_physical = z_state * s + mu
                    overlap_grad = compute_overlap_gradient(
                        x_t_physical,
                        sizes,
                        mask=mask if mask.any() else None,
                        threshold=self.cfg.overlap_penalty_threshold,
                    )
                    overlap_grad_norm = overlap_grad / s
                guidance_strength = self.cfg.guidance_scale * (1.0 - t_value)
                z_state = z_state - dt_value * guidance_strength * overlap_grad_norm

        z_state = controller.apply_sde_noise(
            z_state,
            t_value=t_value,
            dt_value=dt_value,
            mask=mask if mask.any() else None,
        )

        if self.cfg.parameterization == "flow_matching_raw":
            phys_max = (2.0 * s).to(z_state.dtype)
            if mask.any():
                z_state[mask] = z_state[mask].clamp(0.0, phys_max[mask])
            else:
                z_state = z_state.clamp(0.0, phys_max)
        else:
            clip_lo = getattr(self.cfg, 'sample_coord_clip_min', -5.0)
            clip_hi = getattr(self.cfg, 'sample_coord_clip_max', 5.0)
            if mask.any():
                z_state[mask] = z_state[mask].clamp(clip_lo, clip_hi)
            else:
                z_state = z_state.clamp(clip_lo, clip_hi)

        if z0_clean is not None:
            z_state[~mask] = z0_clean[~mask]
        return z_state

    def _adaptive_error_ratio(
        self,
        z_base: torch.Tensor,
        z_euler: torch.Tensor,
        z_heun: torch.Tensor,
        mask: torch.Tensor,
        *,
        atol: float,
        rtol: float,
    ) -> float:
        if mask.any():
            z_ref = z_base[mask]
            z_e = z_euler[mask]
            z_h = z_heun[mask]
        else:
            z_ref = z_base
            z_e = z_euler
            z_h = z_heun
        scale = atol + rtol * torch.maximum(z_ref.abs(), z_h.abs())
        scale = scale.clamp(min=1e-8)
        err = (z_h - z_e) / scale
        return float(torch.sqrt(torch.mean(err ** 2)).item())

    def _integrate_fm_interval(
        self,
        data,
        z_t: torch.Tensor,
        *,
        t_start: float,
        t_end: float,
        original_feats: torch.Tensor,
        z0_clean: Optional[torch.Tensor],
        mask: torch.Tensor,
        s: torch.Tensor,
        mu: torch.Tensor,
        clip_stats: dict,
        controller: FlowMatchingTestTimeController,
        solver: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        adaptive_cfg = controller.cfg.adaptive_step
        adaptive_enabled = adaptive_cfg.enabled and solver == "heun"

        def single_step(z_state: torch.Tensor, start_t: float, dt_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
            v_pred_start, _ = self._evaluate_fm_velocity(
                data=data,
                z_state=z_state,
                t_value=start_t,
                original_feats=original_feats,
                z0_clean=z0_clean,
                mask=mask,
                s=s,
                clip_stats=clip_stats,
                controller=controller,
            )
            if not torch.isfinite(v_pred_start).all():
                raise RuntimeError("GraphDDPM.sample: NaN/Inf in flow-matching predictor velocity.")

            if solver == "heun":
                z_pred = z_state + dt_value * v_pred_start
                if z0_clean is not None:
                    z_pred[~mask] = z0_clean[~mask]
                v_pred_end, _ = self._evaluate_fm_velocity(
                    data=data,
                    z_state=z_pred,
                    t_value=start_t + dt_value,
                    original_feats=original_feats,
                    z0_clean=z0_clean,
                    mask=mask,
                    s=s,
                    clip_stats=clip_stats,
                    controller=controller,
                )
                if not torch.isfinite(v_pred_end).all():
                    raise RuntimeError("GraphDDPM.sample: NaN/Inf in flow-matching corrector velocity.")
                v_used = 0.5 * (v_pred_start + v_pred_end)
            else:
                v_used = v_pred_start

            z_next = z_state + dt_value * v_used
            z_next = self._apply_fm_post_step(
                data=data,
                z_state=z_next,
                t_value=start_t,
                dt_value=dt_value,
                mask=mask,
                z0_clean=z0_clean,
                s=s,
                mu=mu,
                controller=controller,
            )
            return z_next, v_used

        if not adaptive_enabled:
            return single_step(z_t, t_start, t_end - t_start)

        direction = 1.0 if t_end >= t_start else -1.0
        dt_mag = min(adaptive_cfg.max_step_size, abs(t_end - t_start))
        t_cur = t_start
        z_cur = z_t
        last_v = torch.zeros_like(z_t)
        substeps = 0

        while direction * (t_end - t_cur) > 1e-8:
            if substeps >= adaptive_cfg.max_substeps_per_interval:
                z_cur, last_v = single_step(z_cur, t_cur, t_end - t_cur)
                t_cur = t_end
                break

            dt_mag = min(max(dt_mag, adaptive_cfg.min_step_size), adaptive_cfg.max_step_size)
            dt_value = direction * min(dt_mag, abs(t_end - t_cur))

            v_pred_start, _ = self._evaluate_fm_velocity(
                data=data,
                z_state=z_cur,
                t_value=t_cur,
                original_feats=original_feats,
                z0_clean=z0_clean,
                mask=mask,
                s=s,
                clip_stats=clip_stats,
                controller=controller,
            )
            z_euler = z_cur + dt_value * v_pred_start
            if z0_clean is not None:
                z_euler[~mask] = z0_clean[~mask]
            v_pred_end, _ = self._evaluate_fm_velocity(
                data=data,
                z_state=z_euler,
                t_value=t_cur + dt_value,
                original_feats=original_feats,
                z0_clean=z0_clean,
                mask=mask,
                s=s,
                clip_stats=clip_stats,
                controller=controller,
            )
            v_used = 0.5 * (v_pred_start + v_pred_end)
            z_heun = z_cur + dt_value * v_used
            error_ratio = self._adaptive_error_ratio(
                z_cur,
                z_euler,
                z_heun,
                mask,
                atol=adaptive_cfg.atol,
                rtol=adaptive_cfg.rtol,
            )

            if error_ratio <= 1.0 or abs(dt_value) <= adaptive_cfg.min_step_size + 1e-12:
                z_cur = self._apply_fm_post_step(
                    data=data,
                    z_state=z_heun,
                    t_value=t_cur,
                    dt_value=dt_value,
                    mask=mask,
                    z0_clean=z0_clean,
                    s=s,
                    mu=mu,
                    controller=controller,
                )
                t_cur += dt_value
                last_v = v_used
                if error_ratio <= 1e-12:
                    grow = 2.0
                else:
                    grow = adaptive_cfg.safety * (error_ratio ** -0.5)
                dt_mag = min(adaptive_cfg.max_step_size, max(adaptive_cfg.min_step_size, dt_mag * min(grow, 2.0)))
            else:
                shrink = adaptive_cfg.safety * (error_ratio ** -0.5)
                dt_mag = max(adaptive_cfg.min_step_size, dt_mag * max(0.25, min(shrink, 0.9)))
            substeps += 1

        return z_cur, last_v

    def _run_fm_restart_sampling(
        self,
        data,
        z_t: torch.Tensor,
        *,
        original_feats: torch.Tensor,
        z0_clean: Optional[torch.Tensor],
        mask: torch.Tensor,
        s: torch.Tensor,
        mu: torch.Tensor,
        clip_stats: dict,
        controller: FlowMatchingTestTimeController,
        solver: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        restart_cfg = controller.cfg.restart
        if not restart_cfg.enabled or restart_cfg.num_restarts <= 0:
            return z_t, torch.zeros_like(z_t)

        start_t = max(0.0, min(1.0, restart_cfg.t_start))
        end_t = max(start_t, min(1.0, restart_cfg.t_end))
        if end_t <= start_t:
            return z_t, torch.zeros_like(z_t)

        inner_steps = max(2, restart_cfg.inner_steps)
        subgrid = torch.linspace(start_t, end_t, inner_steps + 1, device=z_t.device, dtype=z_t.dtype)
        z_cur = z_t
        last_v = torch.zeros_like(z_t)

        for _ in range(restart_cfg.num_restarts):
            for idx in range(inner_steps, 0, -1):
                z_cur, last_v = self._integrate_fm_interval(
                    data=data,
                    z_t=z_cur,
                    t_start=float(subgrid[idx].item()),
                    t_end=float(subgrid[idx - 1].item()),
                    original_feats=original_feats,
                    z0_clean=z0_clean,
                    mask=mask,
                    s=s,
                    mu=mu,
                    clip_stats=clip_stats,
                    controller=controller,
                    solver=solver,
                )

            if restart_cfg.noise_std > 0.0:
                restart_noise = torch.randn_like(z_cur) * restart_cfg.noise_std
                if mask.any():
                    z_cur[mask] = z_cur[mask] + restart_noise[mask]
                else:
                    z_cur = z_cur + restart_noise
                if z0_clean is not None:
                    z_cur[~mask] = z0_clean[~mask]

            for idx in range(inner_steps):
                z_cur, last_v = self._integrate_fm_interval(
                    data=data,
                    z_t=z_cur,
                    t_start=float(subgrid[idx].item()),
                    t_end=float(subgrid[idx + 1].item()),
                    original_feats=original_feats,
                    z0_clean=z0_clean,
                    mask=mask,
                    s=s,
                    mu=mu,
                    clip_stats=clip_stats,
                    controller=controller,
                    solver=solver,
                )

        return z_cur, last_v

    @torch.no_grad()
    def validation_step(self, data) -> Tuple[torch.Tensor, dict]:
        """
        Validation step - same as training but with additional metrics.
        Supports both HeteroData and homogeneous Data formats.
        """
        device = self.betas.device

        # Match training_step: support HeteroData and homogeneous Data
        if hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data):
            # HeteroData format
            x0 = data["inst"].pos  # [N_inst, 2]
            if "pos_mask" in data["inst"]:
                mask = data["inst"].pos_mask.bool()
            else:
                mask = torch.ones(x0.shape[0], dtype=torch.bool, device=device)
        else:
            # Homogeneous Data format
            x0 = data.pos  # [N_inst, 2]
            if hasattr(data, 'pos_mask'):
                mask = data.pos_mask.bool()
            else:
                mask = torch.ones(x0.shape[0], dtype=torch.bool, device=device)

        # Pull fixed gauge parameters (mu, s) computed from chip_size (global normalization)
        # mu: (N, 2) global center [W/2, H/2] per node (same for all)
        # s: (N, 2) global anisotropic scale [W/2, H/2] per node (same for all)
        if hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data):
            mu = data["inst"].mu if hasattr(data["inst"], "mu") else torch.zeros_like(x0)
            s = data["inst"].s if hasattr(data["inst"], "s") else torch.ones(x0.shape[0], 2, device=device, dtype=x0.dtype)
        else:
            mu = data.mu if hasattr(data, "mu") else torch.zeros_like(x0)
            s = data.s if hasattr(data, "s") else torch.ones(x0.shape[0], 2, device=device, dtype=x0.dtype)
        
        # Ensure s has shape (N, 2) for anisotropic scaling
        # Handle backward compatibility: if s is (N, 1), expand to (N, 2)
        if s.ndim == 1:
            s = s.unsqueeze(-1)
        if s.shape[1] == 1:
            # Old format: (N, 1) -> expand to (N, 2) for isotropic scaling
            s = s.expand(-1, 2)
        elif s.shape[1] != 2:
            raise ValueError(f"s must have shape (N, 1) or (N, 2), got {s.shape}")
        
        # Ensure mu has shape (N, 2)
        if mu.ndim == 1:
            mu = mu.unsqueeze(-1)
        if mu.shape[1] == 1:
            mu = mu.expand(-1, 2)
        elif mu.shape[1] != 2:
            raise ValueError(f"mu must have shape (N, 1) or (N, 2), got {mu.shape}")
        
        mu = mu.to(device=device, dtype=x0.dtype)
        s = s.to(device=device, dtype=x0.dtype)

        # Normalize to gauge space (normalized coordinate space) — skip for flow_matching_raw
        # z0 is in normalized space: z = (x - c) / S where c = (W/2, H/2), S = diag(W/2, H/2)
        if self.cfg.parameterization == "flow_matching_raw":
            z0 = x0  # (N, 2) raw physical positions
        else:
            z0 = (x0 - mu) / s  # (N, 2) normalized clean positions

        if self.cfg.parameterization in ("flow_matching", "flow_matching_raw"):
            # Flow matching validation: Run full ODE integration (like sample()) to get accurate metrics
            # This matches test-time behavior and gives realistic validation metrics
            
            # Store original node features (without z_t) - use same logic as sample()
            if hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data):
                original_feats = self._get_conditioning_features(data['inst'].x.clone())
            else:
                original_feats = self._get_conditioning_features(data.x.clone())
            
            # Start from noise (like sample()) - normalized: [-1, 1]; raw: [0, 2*s]
            if self.cfg.parameterization == "flow_matching_raw":
                z_t = torch.rand_like(z0, device=device, dtype=z0.dtype) * (2.0 * s)
            else:
                z_t = torch.rand_like(z0) * 2.0 - 1.0  # Uniform in [-1, 1]
            z0_clean = z0.clone()
            if mask.any() and not mask.all():
                # For macros/ports (mask=False), use clean positions
                z_t[~mask] = z0_clean[~mask]
            
            # Run full ODE integration from t=0 to t=1 (matching sample() behavior)
            dt = 1.0 / self.num_steps
            N = z_t.shape[0]
            
            # Also compute training loss on a random timestep (for backward compatibility)
            if hasattr(data, 'batch') and data.batch is not None:
                batch_idx = data.batch
                num_graphs = batch_idx.max().item() + 1
            else:
                batch_idx = torch.zeros(N, dtype=torch.long, device=device)
                num_graphs = 1
            t_cont_per_graph = torch.rand(num_graphs, device=device, dtype=x0.dtype)
            t_cont_train = t_cont_per_graph[batch_idx]
            # Match training: z_0 uniform (normalized: [-1,1]; raw: [0, 2*s])
            if self.cfg.parameterization == "flow_matching_raw":
                z_0_train = torch.rand_like(z0, device=device, dtype=z0.dtype) * (2.0 * s)
            else:
                z_0_train = torch.rand_like(z0) * 2.0 - 1.0
            z_1_train = z0
            if mask.any():
                z_0_train = z_0_train.clone()
                z_0_train[~mask] = z_1_train[~mask]
            z_t_train = (1.0 - t_cont_train).unsqueeze(-1) * z_0_train + t_cont_train.unsqueeze(-1) * z_1_train
            v_train = z_1_train - z_0_train
            if mask.any():
                v_masked_train = v_train.clone()
                v_masked_train[~mask] = 0.0
            else:
                v_masked_train = v_train
            z_t_conditioned_train = z_t_train.clone()
            if mask.any():
                z_t_conditioned_train[~mask] = z0[~mask]
            if hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data):
                data["inst"].x = torch.cat([z_t_conditioned_train, original_feats], dim=1)
            else:
                data.x = torch.cat([z_t_conditioned_train, original_feats], dim=1)
            v_pred_train, _ = self.denoiser(data, z_t_conditioned_train, t_cont_train)
            if getattr(self.cfg, 'two_scale_velocity', False) and v_pred_train.shape[-1] == 4:
                v_pred_train = self._two_scale_velocity_to_normalized(v_pred_train, s)
            if mask.any():
                loss = torch.nn.functional.mse_loss(v_pred_train[mask], v_masked_train[mask])
            else:
                loss = torch.nn.functional.mse_loss(v_pred_train, v_masked_train)
            
            # Full ODE integration for accurate validation metrics (matches sample() including sigma_scale)
            sigma_scale = getattr(self.cfg, 'sigma_scale', 1.0)
            use_adaptive = getattr(self.cfg, 'use_adaptive_sigma_scale', False)
            # Get graph size N for adaptive scaling
            if hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data):
                num_nodes = data["inst"].x.shape[0]
            else:
                num_nodes = data.x.shape[0]
            fm_state_clip = 1.2
            fm_input_clip = 1.2
            fm_val_clip_count = 0
            fm_val_max_abs_pre_clip = 0.0
            solver = getattr(self.cfg, 'ode_solver', 'euler')
            for step in range(self.num_steps):
                t_val = step * dt
                if use_adaptive:
                    # N-dependent, time-dependent scaling
                    n0 = getattr(self.cfg, 'adaptive_sigma_n0', 300.0)
                    beta = getattr(self.cfg, 'adaptive_sigma_beta', 0.1)
                    s_min = getattr(self.cfg, 'adaptive_sigma_s_min', 0.7)
                    t_start = getattr(self.cfg, 'adaptive_sigma_t_start', 0.6)
                    log_ratio = torch.log(torch.clamp(torch.tensor(num_nodes / n0, device=device, dtype=torch.float32), min=1e-6))
                    s_N = torch.clamp(1.0 - beta * log_ratio, min=s_min, max=1.0)
                    g_t_val = (t_val - t_start) / max(1.0 - t_start, 1e-6)
                    g_t = max(0.0, min(1.0, g_t_val))
                    s_N_t = 1.0 - g_t * (1.0 - s_N)
                    denom_val = t_val + s_N_t * (1.0 - t_val)
                    t_val_remapped = t_val / denom_val if denom_val > 1e-9 else t_val
                else:
                    denom_val = t_val + sigma_scale * (1.0 - t_val)
                    t_val_remapped = t_val / denom_val if denom_val > 1e-9 else t_val
                t_cont = torch.full((N,), t_val_remapped, device=device, dtype=z_t.dtype)
                z_t_conditioned = z_t.clone()
                if mask.any() and not mask.all():
                    z_t_conditioned[~mask] = z0_clean[~mask]
                if self.cfg.parameterization == "flow_matching":
                    if mask.any():
                        pre_abs = float(z_t_conditioned[mask].abs().max().item())
                        fm_val_max_abs_pre_clip = max(fm_val_max_abs_pre_clip, pre_abs)
                        if pre_abs > fm_input_clip:
                            fm_val_clip_count += 1
                            z_t_conditioned[mask] = z_t_conditioned[mask].clamp(-fm_input_clip, fm_input_clip)
                    else:
                        pre_abs = float(z_t_conditioned.abs().max().item())
                        fm_val_max_abs_pre_clip = max(fm_val_max_abs_pre_clip, pre_abs)
                        if pre_abs > fm_input_clip:
                            fm_val_clip_count += 1
                            z_t_conditioned = z_t_conditioned.clamp(-fm_input_clip, fm_input_clip)
                if hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data):
                    data['inst'].x = torch.cat([z_t_conditioned, original_feats], dim=1)
                else:
                    data.x = torch.cat([z_t_conditioned, original_feats], dim=1)
                if solver == "heun":
                    v_pred_start, _ = self.denoiser(data, z_t_conditioned, t_cont)
                    if getattr(self.cfg, 'two_scale_velocity', False) and v_pred_start.shape[-1] == 4:
                        v_pred_start = self._two_scale_velocity_to_normalized(v_pred_start, s)
                    z_pred = z_t + dt * v_pred_start
                    if mask.any() and not mask.all():
                        z_pred[~mask] = z0_clean[~mask]
                    z_pred_conditioned = z_pred.clone()
                    if mask.any() and not mask.all():
                        z_pred_conditioned[~mask] = z0_clean[~mask]
                    if self.cfg.parameterization == "flow_matching":
                        if mask.any():
                            pre_abs = float(z_pred_conditioned[mask].abs().max().item())
                            fm_val_max_abs_pre_clip = max(fm_val_max_abs_pre_clip, pre_abs)
                            if pre_abs > fm_input_clip:
                                fm_val_clip_count += 1
                                z_pred_conditioned[mask] = z_pred_conditioned[mask].clamp(-fm_input_clip, fm_input_clip)
                        else:
                            pre_abs = float(z_pred_conditioned.abs().max().item())
                            fm_val_max_abs_pre_clip = max(fm_val_max_abs_pre_clip, pre_abs)
                            if pre_abs > fm_input_clip:
                                fm_val_clip_count += 1
                                z_pred_conditioned = z_pred_conditioned.clamp(-fm_input_clip, fm_input_clip)
                    t_next = (step + 1) * dt
                    if use_adaptive:
                        # N-dependent, time-dependent scaling for t_next
                        n0 = getattr(self.cfg, 'adaptive_sigma_n0', 300.0)
                        beta = getattr(self.cfg, 'adaptive_sigma_beta', 0.1)
                        s_min = getattr(self.cfg, 'adaptive_sigma_s_min', 0.7)
                        t_start = getattr(self.cfg, 'adaptive_sigma_t_start', 0.6)
                        log_ratio = torch.log(torch.clamp(torch.tensor(num_nodes / n0, device=device, dtype=torch.float32), min=1e-6))
                        s_N = torch.clamp(1.0 - beta * log_ratio, min=s_min, max=1.0)
                        g_t_next_val = (t_next - t_start) / max(1.0 - t_start, 1e-6)
                        g_t_next = max(0.0, min(1.0, g_t_next_val))
                        s_N_t_next = 1.0 - g_t_next * (1.0 - s_N)
                        denom_next = t_next + s_N_t_next * (1.0 - t_next)
                        t_next_remapped = t_next / denom_next if denom_next > 1e-9 else t_next
                    else:
                        denom_next = t_next + sigma_scale * (1.0 - t_next)
                        t_next_remapped = t_next / denom_next if denom_next > 1e-9 else t_next
                    t_cont_next = torch.full((N,), t_next_remapped, device=device, dtype=z_t.dtype)
                    if hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data):
                        data['inst'].x = torch.cat([z_pred_conditioned, original_feats], dim=1)
                    else:
                        data.x = torch.cat([z_pred_conditioned, original_feats], dim=1)
                    v_pred_end, _ = self.denoiser(data, z_pred_conditioned, t_cont_next)
                    if getattr(self.cfg, 'two_scale_velocity', False) and v_pred_end.shape[-1] == 4:
                        v_pred_end = self._two_scale_velocity_to_normalized(v_pred_end, s)
                    v_pred = 0.5 * (v_pred_start + v_pred_end)
                    z_t = z_t + dt * v_pred
                else:
                    v_pred, _ = self.denoiser(data, z_t_conditioned, t_cont)
                    if getattr(self.cfg, 'two_scale_velocity', False) and v_pred.shape[-1] == 4:
                        v_pred = self._two_scale_velocity_to_normalized(v_pred, s)
                    z_t = z_t + dt * v_pred
                # Clamp z_t to prevent explosion over many ODE steps (e.g. num_steps=200)
                if self.cfg.parameterization == "flow_matching_raw":
                    # Physical scale: clamp to [0, W] x [0, H]
                    phys_max = (2.0 * s).to(z_t.dtype)
                    if mask.any():
                        z_t[mask] = z_t[mask].clamp(0.0, phys_max[mask])
                    else:
                        z_t = z_t.clamp(0.0, phys_max)
                else:
                    if mask.any():
                        z_t[mask] = z_t[mask].clamp(-fm_state_clip, fm_state_clip)
                    else:
                        z_t = z_t.clamp(-fm_state_clip, fm_state_clip)
                if mask.any() and not mask.all():
                    z_t[~mask] = z0_clean[~mask]
            if self.cfg.parameterization == "flow_matching" and fm_val_clip_count > 0:
                logger.warning(
                    "[validation_step] flow_matching clipping activated %d times "
                    "(max |z| before clip=%.4f, input_clip=%.2f, state_clip=%.2f).",
                    fm_val_clip_count, fm_val_max_abs_pre_clip, fm_input_clip, fm_state_clip
                )
            
            # Final positions: denormalize for flow_matching; already physical for flow_matching_raw
            if self.cfg.parameterization == "flow_matching_raw":
                x0_hat = z_t
            else:
                x0_hat = z_t * s + mu
            
            # Compute metrics from full ODE integration (matching test-time)
            if mask.any():
                mse_x0 = torch.nn.functional.mse_loss(x0_hat[mask], x0[mask])
            else:
                mse_x0 = torch.nn.functional.mse_loss(x0_hat, x0)
            rmse_x0 = torch.sqrt(mse_x0)
            
            # For noise level metrics, use the training timestep distribution
            valid_mask = mask if mask.any() else torch.ones(x0.shape[0], dtype=torch.bool, device=device)
            low_noise = (t_cont_train < 0.25) & valid_mask
            mid_noise = ((t_cont_train >= 0.25) & (t_cont_train < 0.75)) & valid_mask
            high_noise = (t_cont_train >= 0.75) & valid_mask
            # Use single-step recovery for noise level metrics (for backward compatibility)
            t_exp_train = t_cont_train.unsqueeze(-1)
            z1_hat_train = z_t_train + (1.0 - t_exp_train) * v_pred_train
            if self.cfg.parameterization == "flow_matching_raw":
                x0_hat_train = z1_hat_train
            else:
                x0_hat_train = z1_hat_train * s + mu
            rmse_x0_low = torch.sqrt(torch.nn.functional.mse_loss(x0_hat_train[low_noise], x0[low_noise])).item() if low_noise.any() else 0.0
            rmse_x0_mid = torch.sqrt(torch.nn.functional.mse_loss(x0_hat_train[mid_noise], x0[mid_noise])).item() if mid_noise.any() else 0.0
            rmse_x0_high = torch.sqrt(torch.nn.functional.mse_loss(x0_hat_train[high_noise], x0[high_noise])).item() if high_noise.any() else 0.0
            
            # Compute coordinate error percentage from full ODE integration
            abs_err = torch.abs((x0_hat[mask] - x0[mask]) if mask.any() else (x0_hat - x0))
            # Use true chip size [W, H] from data.chip_size when available
            if hasattr(data, "chip_size") and data.chip_size is not None:
                cs = data.chip_size.to(device=device, dtype=abs_err.dtype)
                if cs.numel() == 2:
                    n_pts = abs_err.shape[0]
                    chip = cs.unsqueeze(0).expand(n_pts, 2)
                elif cs.dim() == 2 and hasattr(data, "batch") and data.batch is not None:
                    chip = cs[data.batch]
                    if mask.any():
                        chip = chip[mask]
                else:
                    chip = cs[mask] if mask.any() else cs
            else:
                chip = (s[mask] * 2.0) if mask.any() else (s * 2.0)
            chip = torch.clamp(chip, min=1e-8)
            err_pct = (abs_err / chip * 100.0).mean().item()
            
            # Flow matching: velocity statistics from training timestep
            v_pred_masked = v_pred_train[mask] if mask.any() else v_pred_train
            v_target_masked = v_masked_train[mask] if mask.any() else v_masked_train
            v_pred_mean = v_pred_masked.mean().item()
            v_pred_std = v_pred_masked.std().item()
            v_target_mean = v_target_masked.mean().item()
            v_target_std = v_target_masked.std().item()
            
            metrics = {
                "loss": loss.item(),
                "rmse_x0": rmse_x0.item(),  # From full ODE integration (matches test-time)
                "rmse_x0_low": rmse_x0_low,
                "rmse_x0_mid": rmse_x0_mid,
                "rmse_x0_high": rmse_x0_high,
                "coord_error_pct": err_pct,  # From full ODE integration (matches test-time)
                "v_pred_mean": v_pred_mean,
                "v_pred_std": v_pred_std,
                "v_target_mean": v_target_mean,
                "v_target_std": v_target_std,
            }
            return loss, metrics

        # Sample timesteps per GRAPH (not per node!)
        # Get batch assignment (which graph each node belongs to)
        if hasattr(data, 'batch') and data.batch is not None:
            batch_idx = data.batch  # (N,) tensor with graph index for each node
            num_graphs = batch_idx.max().item() + 1
        else:
            # Single graph (not batched)
            batch_idx = torch.zeros(x0.shape[0], dtype=torch.long, device=device)
            num_graphs = 1
        
        # Sample one timestep per graph
        t_idxs_per_graph, t_cont_per_graph = self._sample_timesteps(num_graphs, device=device)
        
        # Expand to per-node timesteps
        t_idxs = t_idxs_per_graph[batch_idx]  # (N,) - same timestep for all nodes in same graph
        t_cont = t_cont_per_graph[batch_idx]  # (N,) - same continuous time for all nodes in same graph

        # Forward diffusion in normalized z space
        # noise is sampled from N(0, I) in normalized space
        # CRITICAL: Macros and ports are FIXED - do NOT add noise to them!
        z_t, noise = self.q_sample(z0, t_idxs)  # noise is in normalized space
        
        # Mask macro/port positions: keep their clean positions (z0) as conditioning
        if mask.any():
            z_t_conditioned = z_t.clone()
            z_t_conditioned[~mask] = z0[~mask]  # Keep macro/port positions clean
            noise_masked = noise.clone()
            noise_masked[~mask] = 0.0  # No noise for macros/ports
        else:
            z_t_conditioned = z_t
            noise_masked = noise

        # Update node features with z_t_conditioned (noisy for std-cells, clean for macros/ports)
        if hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data):
            feats = self._get_conditioning_features(data['inst'].x)
            data['inst'].x = torch.cat([z_t_conditioned, feats], dim=1)
        else:
            feats = self._get_conditioning_features(data.x)
            data.x = torch.cat([z_t_conditioned, feats], dim=1)

        # Denoise (predict epsilon)
        eps_pred, _ = self.denoiser(data, z_t_conditioned, t_cont)  # eps_pred is in normalized space

        # Compute loss
        # Loss compares eps_pred (predicted noise) to noise_masked (actual noise, 0 for macros/ports)
        # Both are in normalized coordinate space
        if mask.any():
            loss = torch.nn.functional.mse_loss(eps_pred[mask], noise_masked[mask])
        else:
            loss = torch.nn.functional.mse_loss(eps_pred, noise_masked)

        # Compute metrics with eps_pred statistics
        alphas_cumprod = self.alphas_cumprod[t_idxs].unsqueeze(-1)
        std = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_alpha = torch.sqrt(alphas_cumprod.clamp(min=1e-8))
        z0_hat = (z_t_conditioned - std * eps_pred) / sqrt_alpha
        x0_hat = z0_hat * s + mu
        if mask.any():
            mse_x0 = torch.nn.functional.mse_loss(x0_hat[mask], x0[mask])
        else:
            mse_x0 = torch.nn.functional.mse_loss(x0_hat, x0)
        rmse_x0 = torch.sqrt(mse_x0)
        
        # eps_pred statistics (should be mean ~0, std ~1 for correct scale)
        eps_pred_mean = eps_pred[mask].mean().item() if mask.any() else eps_pred.mean().item()
        eps_pred_std = eps_pred[mask].std().item() if mask.any() else eps_pred.std().item()
        noise_mean = noise_masked[mask].mean().item() if mask.any() else noise_masked.mean().item()
        noise_std = noise_masked[mask].std().item() if mask.any() else noise_masked.std().item()
        
        # Also compute rmse_x0 separately for different noise levels
        if mask.any():
            valid_mask = mask
        else:
            valid_mask = torch.ones(x0.shape[0], dtype=torch.bool, device=device)
        
        num_steps = self.num_steps
        low_threshold = num_steps // 4
        mid_threshold = num_steps * 3 // 4
        
        low_noise_mask = (t_idxs < low_threshold) & valid_mask
        mid_noise_mask = ((t_idxs >= low_threshold) & (t_idxs < mid_threshold)) & valid_mask
        high_noise_mask = (t_idxs >= mid_threshold) & valid_mask
        
        rmse_x0_low = torch.sqrt(torch.nn.functional.mse_loss(x0_hat[low_noise_mask], x0[low_noise_mask])).item() if low_noise_mask.any() else 0.0
        rmse_x0_mid = torch.sqrt(torch.nn.functional.mse_loss(x0_hat[mid_noise_mask], x0[mid_noise_mask])).item() if mid_noise_mask.any() else 0.0
        rmse_x0_high = torch.sqrt(torch.nn.functional.mse_loss(x0_hat[high_noise_mask], x0[high_noise_mask])).item() if high_noise_mask.any() else 0.0

        # New recommended normalization keeps positions in raw units (no min-max / [-1,1] denormalization).
        # So rmse_x0 is already in the correct unnormalized space.
        rmse_x0_unnorm = rmse_x0

        metrics = {
            "loss": loss.item(),
            "rmse_x0": rmse_x0.item(),
            "rmse_x0_low": rmse_x0_low,
            "rmse_x0_mid": rmse_x0_mid,
            "rmse_x0_high": rmse_x0_high,
            "eps_pred_mean": eps_pred_mean,
            "eps_pred_std": eps_pred_std,
            "noise_mean": noise_mean,
            "noise_std": noise_std,
        }
        if rmse_x0_unnorm is not None:
            metrics["rmse_x0_unnorm"] = rmse_x0_unnorm.item()
        return loss, metrics

    @torch.no_grad()
    def sample(self, data: HeteroData, show_progress: bool = False, track_errors: bool = False, track_trajectory: bool = False, x0_true: Optional[torch.Tensor] = None, chip_size: Optional[torch.Tensor] = None, capture_intermediate: Optional[int] = None, capture_t_values: Optional[List[float]] = None, return_groundtruth_trajectory: bool = False, z_init_norm: Optional[torch.Tensor] = None, t_start: float = 0.0, num_steps_override: Optional[int] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, List[float]], Tuple[torch.Tensor, List[float], List[torch.Tensor]], Tuple[torch.Tensor, List[float], List[dict]], Tuple[torch.Tensor, List[float], List[dict], List[torch.Tensor]], Tuple[torch.Tensor, List[float], List[dict], List[torch.Tensor], List[torch.Tensor]], Tuple[torch.Tensor, List[float], List[dict], List[torch.Tensor], List[torch.Tensor], List[float]]]:
        """
        Generate positions for a batch of graphs using ancestral sampling.

        Uses predictor-corrector approach if enabled (default), which is more
        numerically stable than direct DDPM sampling.

        Args:
            data: HeteroData with node/edge features (pos will be ignored).
            show_progress: If True, show a progress bar for the diffusion sampling steps.
            track_errors: If True, track coordinate errors at each step (requires x0_true).
            track_trajectory: If True (and track_errors=True), record per-step trajectory metrics for comparison.
            x0_true: Ground truth positions for error tracking [N_inst, 2] (required if track_errors=True).
            chip_size: Chip size [W, H] (optional, for error %; step_errors are de-normalized MAE in physical units).
            capture_intermediate: If set to K, capture K intermediate positions during sampling (for visualization).
            capture_t_values: If set (e.g. [0.0, 0.5, 0.8, 0.9, 1.0]), capture at these exact t values (flow matching only).
                When provided, overrides capture_intermediate for flow matching and returns captured_t_values as 6th element.
            return_groundtruth_trajectory: If True (and flow matching, capture_intermediate or capture_t_values, x0_true), also return
                groundtruth intermediate positions following v_true = z_1 - z_0 (expected to recover x0_true at end).
        Returns:
            If track_errors=False and capture_intermediate=None: x0_samples [N_inst, 2]
            If track_errors=True and capture_intermediate=None: (x0_samples [N_inst, 2], step_errors List[float])
            If track_trajectory=True (and track_errors): adds step_trajectory List[dict] with per-step:
                step, t, coord_mae, coord_mse, coord_error_pct, vel_mae, vel_mse (flow matching), displacement_norm
            If capture_intermediate=K (or capture_t_values): adds intermediate_positions List[torch.Tensor]
            If return_groundtruth_trajectory=True (flow matching only): adds groundtruth_intermediate_positions
                as 5th element (same t grid, v_true = z_1 - z_0; final step recovers groundtruth).
            If capture_t_values was provided: adds captured_t_values List[float] as 6th element (parallel to intermediate_positions).
        """
        device = self.betas.device
        # Support both HeteroData and homogeneous Data.
        if hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data):
            N = data["inst"].x.shape[0]
            num_nodes = N  # Graph size for adaptive sigma scaling
            mu = data["inst"].mu if hasattr(data["inst"], "mu") else torch.zeros((N, 2), device=device)
            s = data["inst"].s.to(device=device) if hasattr(data["inst"], "s") and data["inst"].s is not None else None
        else:
            N = data.x.shape[0]
            num_nodes = N  # Graph size for adaptive sigma scaling
            mu = data.mu if hasattr(data, "mu") else torch.zeros((N, 2), device=device)
            s = data.s.to(device=device) if hasattr(data, "s") and data.s is not None else None

        if s is None:
            s = torch.ones((N, 2), device=device)
        
        # Ensure s has shape (N, 2) for anisotropic scaling
        # Handle backward compatibility: if s is (N, 1), expand to (N, 2)
        if s.ndim == 1:
            s = s.unsqueeze(-1)
        if s.shape[1] == 1:
            # Old format: (N, 1) -> expand to (N, 2) for isotropic scaling
            s = s.expand(-1, 2)
        elif s.shape[1] != 2:
            raise ValueError(f"s must have shape (N, 1) or (N, 2), got {s.shape}")
        
        # Ensure mu has shape (N, 2)
        if mu.ndim == 1:
            mu = mu.unsqueeze(-1)
        if mu.shape[1] == 1:
            mu = mu.expand(-1, 2)
        elif mu.shape[1] != 2:
            raise ValueError(f"mu must have shape (N, 1) or (N, 2), got {mu.shape}")
        
        mu = mu.to(device=device, dtype=s.dtype)

        # Get mask for which instances should be diffused (True = diffuse, False = keep fixed)
        # CRITICAL: Macros and ports should be FIXED (not diffused)
        if hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data):
            if "pos_mask" in data["inst"]:
                mask = data["inst"].pos_mask.bool()
            else:
                mask = torch.ones(N, dtype=torch.bool, device=device)
            # Get clean positions for macros/ports (they should not be diffused)
            x0_clean = data["inst"].pos if hasattr(data["inst"], "pos") else None
        else:
            # Homogeneous Data format
            if hasattr(data, 'pos_mask'):
                mask = data.pos_mask.bool()
            else:
                mask = torch.ones(N, dtype=torch.bool, device=device)
            # Get clean positions for macros/ports
            x0_clean = data.pos_target if hasattr(data, "pos_target") and data.pos_target is not None else (data.pos if hasattr(data, "pos") else None)
        
        # Clean positions for macros/ports: normalized (flow_matching) or raw (flow_matching_raw)
        z0_clean = None
        if x0_clean is not None and mask.any() and not mask.all():
            if self.cfg.parameterization == "flow_matching_raw":
                z0_clean = x0_clean.to(device=device, dtype=s.dtype)
            else:
                z0_clean = (x0_clean - mu) / s  # Normalize clean positions
        
        # Get initial positions for refinement flow matching
        x_init = None
        if hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data):
            # HeteroData format
            if "pos_init" in data["inst"] and data["inst"].pos_init is not None:
                x_init = data["inst"].pos_init.to(device=device)
        else:
            # Homogeneous Data format
            if hasattr(data, "pos_init") and data.pos_init is not None:
                x_init = data.pos_init.to(device=device)
        
        # Resolve effective step count and warm-start time offset
        _n_steps = num_steps_override if num_steps_override is not None else self.num_steps
        _t_start = t_start if (z_init_norm is not None and t_start > 0.0) else 0.0

        # Initialize noise based on parameterization and refinement settings
        if _t_start > 0.0 and z_init_norm is not None:
            # FM warm-start: construct valid FM interpolant z_{t_start} = (1-t)*z_rand + t*z_reg.
            # z_reg ≈ z_1 (regression prediction in normalized space), z_rand ~ Uniform[-1,1].
            # This is a proper FM state at time t_start — fully consistent with the trained velocity field.
            z_rand = torch.rand(N, 2, device=device, dtype=s.dtype) * 2.0 - 1.0
            z_init_n = z_init_norm.to(device=device, dtype=s.dtype)
            z_t = (1.0 - _t_start) * z_rand + _t_start * z_init_n
        elif self.cfg.init_noise_std > 0.0 and x_init is not None:
            # Refinement: start from z_init + small noise instead of pure noise
            if self.cfg.parameterization == "flow_matching_raw":
                z_init = x_init.to(device=device, dtype=s.dtype)
            else:
                z_init = (x_init - mu) / s  # Normalize initial positions
            noise_std = self.cfg.init_noise_std
            eps = torch.randn(N, 2, device=device, dtype=s.dtype)
            z_t = z_init + noise_std * eps
        else:
            # Standard initialization: choose noise type based on parameterization
            if self.cfg.parameterization == "flow_matching":
                # Flow matching: start from uniform noise in [-1, 1] to match normalized coordinate space
                z_t = torch.rand(N, 2, device=device, dtype=s.dtype) * 2.0 - 1.0  # Uniform in [-1, 1]
            elif self.cfg.parameterization == "flow_matching_raw":
                # Raw physical scale: uniform in [0, W] x [0, H]
                z_t = torch.rand(N, 2, device=device, dtype=s.dtype) * (2.0 * s)
            else:
                # DDPM and EDM: use Gaussian noise N(0,1) in normalized space
                z_t = torch.randn(N, 2, device=device, dtype=s.dtype)  # Gaussian N(0,1)
        
        # CRITICAL: For macros/ports (mask=False), use their clean positions instead of noise
        if z0_clean is not None:
            z_t[~mask] = z0_clean[~mask]  # Use clean positions for macros/ports
        
        # Store original node features (without z_t) - use same logic as training
        if hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data):
            original_feats = self._get_conditioning_features(data['inst'].x.clone())
        else:
            original_feats = self._get_conditioning_features(data.x.clone())

        if self.cfg.parameterization in ("flow_matching", "flow_matching_raw"):
            # ODE integration: dz/dt = v_θ(z, t), t in [0, 1]
            # Solver options:
            #   - Euler: z_{i+1} = z_i + dt * v_θ(z_i, t_i) (1st order, 1 model call per step)
            #   - Heun: predictor z_pred = z_i + dt * v_θ(z_i, t_i), then corrector z_{i+1} = z_i + (dt/2) * [v_θ(z_i, t_i) + v_θ(z_pred, t_{i+1})] (2nd order, 2 model calls per step)
            controller = self._build_fm_test_time_controller(
                num_nodes=num_nodes,
                device=device,
                dtype=z_t.dtype,
            )
            solver = controller.resolve_solver(getattr(self.cfg, 'ode_solver', 'euler'))
            step_range_fm = range(_n_steps)
            solver_name = solver.upper() if solver == "heun" else "Euler"
            if show_progress:
                step_range_fm = tqdm(step_range_fm, desc=f"Flow-matching ODE ({solver_name})", total=_n_steps, unit="step")

            # Warm-start: integrate only over [t_start, 1.0] with _n_steps uniform steps.
            # Full-noise start: integrate [0, 1] as before (build_time_grid may apply time-warping).
            if _t_start > 0.0:
                t_grid = torch.linspace(_t_start, 1.0, _n_steps + 1, device=device, dtype=z_t.dtype)
            else:
                t_grid = controller.build_time_grid(_n_steps)
            # Flow-matching trajectories are expected to remain near [-1, 1].
            # Keep inputs/states bounded during ODE integration and log drift.
            clip_stats = {
                "state_clip": 1.2,
                "input_clip": 1.2,
                "clip_count": 0,
                "max_abs_pre_clip": 0.0,
            }
            
            step_errors = [] if track_errors else None
            step_trajectory = [] if (track_errors and track_trajectory) else None
            if track_errors and x0_true is None:
                raise ValueError("x0_true must be provided when track_errors=True")

            # For flow matching trajectory: v_target = z_1 - z_0 (rectilinear flow)
            # Also needed for return_groundtruth_trajectory
            z_1_norm = None
            v_target_norm = None
            chip_size_for_pct = None
            if (step_trajectory is not None or return_groundtruth_trajectory) and x0_true is not None:
                if self.cfg.parameterization == "flow_matching_raw":
                    z_1_norm = x0_true.to(device=device, dtype=z_t.dtype)  # target already physical
                else:
                    z_1_norm = (x0_true - mu) / s  # target in normalized space
                z_0_init = z_t.clone()  # initial noise (before any ODE step)
                v_target_norm = z_1_norm - z_0_init  # constant velocity (rectilinear flow)
                if chip_size is not None and chip_size.numel() >= 2:
                    chip_size_for_pct = chip_size.flatten()[:2].to(device)  # (2,)
                else:
                    chip_size_for_pct = (2.0 * s[0]) if s.shape[0] > 0 else None  # approximate

            # Capture intermediate positions if requested (by count or by exact t values)
            use_capture_by_t = capture_t_values is not None and len(capture_t_values) > 0
            use_capture_count = capture_intermediate is not None and not use_capture_by_t
            intermediate_positions = [] if (use_capture_by_t or use_capture_count) else None
            groundtruth_intermediate_positions = [] if ((use_capture_by_t or use_capture_count) and return_groundtruth_trajectory and x0_true is not None) else None
            # Also capture velocities for radial drift analysis
            intermediate_velocities = [] if (use_capture_by_t or use_capture_count) else None
            groundtruth_intermediate_velocities = [] if ((use_capture_by_t or use_capture_count) and return_groundtruth_trajectory and x0_true is not None and v_target_norm is not None) else None
            captured_t_values: List[float] = []  # parallel to intermediate_positions when use_capture_by_t
            if use_capture_by_t:
                capture_t_sorted = sorted(set(float(t) for t in capture_t_values))
            if use_capture_count:
                # Calculate which steps to capture (evenly spaced)
                capture_steps = []
                if capture_intermediate > 0:
                    step_interval = max(1, self.num_steps // capture_intermediate)
                    capture_steps = list(range(0, self.num_steps, step_interval))[:capture_intermediate]
                    # Always include the last step
                    if capture_steps[-1] != self.num_steps - 1:
                        capture_steps.append(self.num_steps - 1)
            else:
                capture_steps = []  # not used when use_capture_by_t
            
            # Track initial error (before any steps) — physical units
            if track_errors:
                if self.cfg.parameterization == "flow_matching_raw":
                    x_t_init = z_t
                else:
                    x_t_init = z_t * s + mu  # Denormalize initial positions
                abs_error = torch.abs(x_t_init - x0_true)  # (N, 2)
                mae = abs_error.mean().item()
                step_errors.append(mae)
                if step_trajectory is not None:
                    mse = ((x_t_init - x0_true) ** 2).mean().item()
                    err_pct = float('nan')
                    if chip_size_for_pct is not None:
                        err_per_dim = abs_error / chip_size_for_pct.unsqueeze(0).clamp(min=1e-8)
                        err_pct = err_per_dim.mean().item() * 100.0
                    step_trajectory.append({
                        "step": 0,
                        "t": 0.0,
                        "coord_mae": mae,
                        "coord_mse": mse,
                        "coord_error_pct": err_pct,
                        "vel_mae": float('nan'),
                        "vel_mse": float('nan'),
                        "displacement_norm": 0.0,
                    })

            # Capture at t=0 when using capture_t_values (before any ODE step)
            if use_capture_by_t and capture_t_sorted and capture_t_sorted[0] <= 0.0:
                intermediate_positions.append(z_t.clone())
                # At t=0, velocity hasn't been computed yet, so we'll compute it
                if intermediate_velocities is not None:
                    v_pred_0, _ = self._evaluate_fm_velocity(
                        data=data,
                        z_state=z_t,
                        t_value=0.0,
                        original_feats=original_feats,
                        z0_clean=z0_clean,
                        mask=mask,
                        s=s,
                        clip_stats=clip_stats,
                        controller=controller,
                    )
                    intermediate_velocities.append(v_pred_0.clone())
                if groundtruth_intermediate_positions is not None and z_1_norm is not None:
                    z_gt_0 = z_0_init.clone()
                    if z0_clean is not None:
                        z_gt_0 = z_gt_0.clone()
                        z_gt_0[~mask] = z0_clean[~mask]
                    groundtruth_intermediate_positions.append(z_gt_0)
                    if groundtruth_intermediate_velocities is not None and v_target_norm is not None:
                        groundtruth_intermediate_velocities.append(v_target_norm.clone())
                captured_t_values.append(0.0)

            next_capture_t_idx = 1 if (use_capture_by_t and capture_t_sorted and capture_t_sorted[0] <= 0.0) else 0

            v_used = torch.zeros_like(z_t)
            for step in step_range_fm:
                # Time at start/end of this step (non-uniform dt allowed)
                t_val = float(t_grid[step].item())
                t_next = float(t_grid[step + 1].item())
                dt_step = t_next - t_val
                z_t, v_used = self._integrate_fm_interval(
                    data=data,
                    z_t=z_t,
                    t_start=t_val,
                    t_end=t_next,
                    original_feats=original_feats,
                    z0_clean=z0_clean,
                    mask=mask,
                    s=s,
                    mu=mu,
                    clip_stats=clip_stats,
                    controller=controller,
                    solver=solver,
                )

                if not torch.isfinite(v_used).all():
                    step_1based = step + 1
                    logger.error(
                        "[sample] NaN/Inf in v_pred at step %d (1-based) of %d (flow_matching, solver=%s). "
                        "First non-finite occurrence during ODE integration.",
                        step_1based, self.num_steps, solver,
                    )
                    raise RuntimeError(
                        f"GraphDDPM.sample: NaN/Inf in v_pred at step {step_1based} (of {self.num_steps}). "
                        f"Solver={solver}. Check model and batch for instability."
                    )
                if not torch.isfinite(z_t).all():
                    step_1based = step + 1
                    logger.error(
                        "[sample] NaN/Inf in z_t at step %d (1-based) of %d (flow_matching, solver=%s). "
                        "First non-finite occurrence during ODE integration.",
                        step_1based, self.num_steps, solver,
                    )
                    raise RuntimeError(
                        f"GraphDDPM.sample: NaN/Inf in z_t at step {step_1based} (of {self.num_steps}). "
                        f"Solver={solver}. Check model and batch for instability."
                    )
                
                # Track error at this step — physical units
                if track_errors:
                    if self.cfg.parameterization == "flow_matching_raw":
                        x_t = z_t
                    else:
                        x_t = z_t * s + mu  # Denormalize current positions
                    abs_error = torch.abs(x_t - x0_true)  # (N, 2)
                    mae = abs_error.mean().item()
                    step_errors.append(mae)
                    if step_trajectory is not None:
                        mse = ((x_t - x0_true) ** 2).mean().item()
                        err_pct = float('nan')
                        if chip_size_for_pct is not None:
                            err_per_dim = abs_error / chip_size_for_pct.unsqueeze(0).clamp(min=1e-8)
                            err_pct = err_per_dim.mean().item() * 100.0
                        # Velocity error (flow matching): v_pred vs v_target over diffused nodes
                        vel_mae = float('nan')
                        vel_mse = float('nan')
                        if v_target_norm is not None and mask.any():
                            v_err = (v_used - v_target_norm)[mask]  # (N_diff, 2)
                            vel_mae = v_err.abs().mean().item()
                            vel_mse = (v_err ** 2).mean().item()
                        # Displacement this step (physical units)
                        if self.cfg.parameterization == "flow_matching_raw":
                            disp_phys = dt_step * v_used  # (N, 2) already physical
                        else:
                            disp_phys = dt_step * v_used * s  # (N, 2)
                        disp_norm = (disp_phys ** 2).sum(dim=1).sqrt().mean().item()
                        t_next_val = float(t_grid[step + 1].item())
                        step_trajectory.append({
                            "step": step + 1,
                            "t": t_next_val,
                            "coord_mae": mae,
                            "coord_mse": mse,
                            "coord_error_pct": err_pct,
                            "vel_mae": vel_mae,
                            "vel_mse": vel_mse,
                            "displacement_norm": disp_norm,
                        })

                # Capture intermediate position if requested (in normalized space)
                t_next_val = float(t_grid[step + 1].item())
                if use_capture_by_t:
                    while next_capture_t_idx < len(capture_t_sorted) and capture_t_sorted[next_capture_t_idx] <= t_next_val:
                        t_target = capture_t_sorted[next_capture_t_idx]
                        intermediate_positions.append(z_t.clone())
                        # Capture velocity used in this step
                        if intermediate_velocities is not None:
                            intermediate_velocities.append(v_used.clone())
                        if groundtruth_intermediate_positions is not None and z_1_norm is not None:
                            z_gt = (1.0 - t_target) * z_0_init + t_target * z_1_norm
                            if z0_clean is not None:
                                z_gt = z_gt.clone()
                                z_gt[~mask] = z0_clean[~mask]
                            groundtruth_intermediate_positions.append(z_gt)
                            if groundtruth_intermediate_velocities is not None and v_target_norm is not None:
                                groundtruth_intermediate_velocities.append(v_target_norm.clone())
                        captured_t_values.append(t_target)
                        next_capture_t_idx += 1
                elif intermediate_positions is not None and step in capture_steps:
                    intermediate_positions.append(z_t.clone())  # Keep in normalized space
                    # Capture velocity used in this step
                    if intermediate_velocities is not None:
                        intermediate_velocities.append(v_used.clone())
                # Groundtruth trajectory (when using capture_steps, not capture_t_values)
                if groundtruth_intermediate_positions is not None and not use_capture_by_t and step in capture_steps and z_1_norm is not None:
                    t_val = t_next_val
                    z_gt = (1.0 - t_val) * z_0_init + t_val * z_1_norm
                    if z0_clean is not None:
                        z_gt = z_gt.clone()
                        z_gt[~mask] = z0_clean[~mask]
                    groundtruth_intermediate_positions.append(z_gt)
                    if groundtruth_intermediate_velocities is not None and v_target_norm is not None:
                        groundtruth_intermediate_velocities.append(v_target_norm.clone())

            if controller.cfg.restart.enabled:
                z_t, v_used = self._run_fm_restart_sampling(
                    data=data,
                    z_t=z_t,
                    original_feats=original_feats,
                    z0_clean=z0_clean,
                    mask=mask,
                    s=s,
                    mu=mu,
                    clip_stats=clip_stats,
                    controller=controller,
                    solver=solver,
                )
                if track_errors:
                    if self.cfg.parameterization == "flow_matching_raw":
                        x_t = z_t
                    else:
                        x_t = z_t * s + mu
                    abs_error = torch.abs(x_t - x0_true)
                    mae = abs_error.mean().item()
                    if step_errors:
                        step_errors[-1] = mae
                    if step_trajectory:
                        mse = ((x_t - x0_true) ** 2).mean().item()
                        err_pct = float('nan')
                        if chip_size_for_pct is not None:
                            err_per_dim = abs_error / chip_size_for_pct.unsqueeze(0).clamp(min=1e-8)
                            err_pct = err_per_dim.mean().item() * 100.0
                        step_trajectory[-1]["coord_mae"] = mae
                        step_trajectory[-1]["coord_mse"] = mse
                        step_trajectory[-1]["coord_error_pct"] = err_pct

                if use_capture_by_t and captured_t_values and abs(captured_t_values[-1] - 1.0) <= 1e-6:
                    if intermediate_positions is not None:
                        intermediate_positions[-1] = z_t.clone()
                    if intermediate_velocities is not None:
                        intermediate_velocities[-1] = v_used.clone()
            
            if self.cfg.parameterization == "flow_matching_raw":
                x0 = z_t  # already in physical space
            else:
                x0 = z_t * s + mu
            if self.cfg.parameterization == "flow_matching" and clip_stats["clip_count"] > 0:
                logger.warning(
                    "[sample] flow_matching clipping activated %d times "
                    "(max |z| before clip=%.4f, input_clip=%.2f, state_clip=%.2f).",
                    clip_stats["clip_count"],
                    clip_stats["max_abs_pre_clip"],
                    clip_stats["input_clip"],
                    clip_stats["state_clip"],
                )
            # Always capture final position when using step-based capture (not when using capture_t_values; we already captured t=1 in loop)
            if intermediate_positions is not None and use_capture_count:
                intermediate_positions.append(z_t.clone())  # Keep in normalized space
                # Capture final velocity (use the last computed velocity)
                if intermediate_velocities is not None:
                    intermediate_velocities.append(v_used.clone())
            if groundtruth_intermediate_positions is not None and z_1_norm is not None and use_capture_count:
                z_final_gt = z_1_norm.clone()
                if z0_clean is not None:
                    z_final_gt[~mask] = z0_clean[~mask]
                groundtruth_intermediate_positions.append(z_final_gt)
                if groundtruth_intermediate_velocities is not None and v_target_norm is not None:
                    groundtruth_intermediate_velocities.append(v_target_norm.clone())

            if intermediate_positions is not None:
                if return_groundtruth_trajectory and groundtruth_intermediate_positions is not None:
                    if use_capture_by_t:
                        if track_errors:
                            if step_trajectory is not None:
                                return x0, step_errors, step_trajectory, intermediate_positions, groundtruth_intermediate_positions, captured_t_values, intermediate_velocities, groundtruth_intermediate_velocities
                            return x0, step_errors, intermediate_positions, groundtruth_intermediate_positions, captured_t_values, intermediate_velocities, groundtruth_intermediate_velocities
                        return x0, None, intermediate_positions, groundtruth_intermediate_positions, captured_t_values, intermediate_velocities, groundtruth_intermediate_velocities
                    if track_errors:
                        if step_trajectory is not None:
                            return x0, step_errors, step_trajectory, intermediate_positions, groundtruth_intermediate_positions, intermediate_velocities, groundtruth_intermediate_velocities
                        return x0, step_errors, intermediate_positions, groundtruth_intermediate_positions, intermediate_velocities, groundtruth_intermediate_velocities
                    return x0, None, intermediate_positions, groundtruth_intermediate_positions, intermediate_velocities, groundtruth_intermediate_velocities
                if track_errors:
                    if step_trajectory is not None:
                        return x0, step_errors, step_trajectory, intermediate_positions, intermediate_velocities
                    return x0, step_errors, intermediate_positions, intermediate_velocities
                return x0, None, intermediate_positions, intermediate_velocities
            elif track_errors:
                if step_trajectory is not None:
                    return x0, step_errors, step_trajectory
                return x0, step_errors
            return x0

        # Create progress bar if requested
        step_range = reversed(range(self.num_steps))
        if show_progress:
            step_range = tqdm(step_range, desc="Diffusion sampling", total=self.num_steps, unit="step")
        
        step_errors = [] if track_errors else None
        if track_errors and x0_true is None:
            raise ValueError("x0_true must be provided when track_errors=True")
        
        # Capture intermediate positions if requested
        intermediate_positions = [] if capture_intermediate is not None else None
        if capture_intermediate is not None:
            # Calculate which steps to capture (evenly spaced)
            capture_steps = []
            if capture_intermediate > 0:
                step_interval = max(1, self.num_steps // capture_intermediate)
                capture_steps = list(range(self.num_steps - 1, -1, -step_interval))[:capture_intermediate]
                # Always include the first step (t=num_steps-1)
                if capture_steps[-1] != self.num_steps - 1:
                    capture_steps.append(self.num_steps - 1)
                capture_steps = sorted(set(capture_steps), reverse=True)  # Reverse order for DDPM
        
        # Track initial error (before any steps) — de-normalized MAE (physical units)
        if track_errors:
            x_t_init = z_t * s + mu  # Denormalize initial positions
            abs_error = torch.abs(x_t_init - x0_true)  # (N, 2)
            mae = abs_error.mean().item()
            step_errors.append(mae)
        
        # Capture initial position if requested (for DDPM, this is the starting noise, in normalized space)
        if intermediate_positions is not None and self.num_steps - 1 in capture_steps:
            intermediate_positions.append(z_t.clone())  # Keep in normalized space
        
        for step in step_range:
            t_idx = torch.full((N,), step, device=device, dtype=torch.long)
            t_cont = (t_idx + 1).float() / float(self.num_steps)
            
            # CRITICAL: Create z_t_conditioned (same as training)
            # For macros/ports (mask=False), use clean positions z0_clean as conditioning
            # For std-cells (mask=True), use noisy z_t
            # This matches training exactly: macros/ports provide fixed conditioning
            if z0_clean is not None and mask.any() and not mask.all():
                z_t_conditioned = z_t.clone()
                z_t_conditioned[~mask] = z0_clean[~mask]  # Keep macro/port positions clean as conditioning
            else:
                z_t_conditioned = z_t
            
            # Update node features with z_t_conditioned (noisy for std-cells, clean for macros/ports)
            # Node features: [z_tx, z_ty, w_rel, h_rel, r, area_rel, w_norm_d, h_norm_d, w_raw, h_raw,
            #                 chip_w_raw, chip_h_raw, chip_w_rel, chip_h_rel, num_stdcells, num_macros,
            #                 num_ports, num_edges, degree, log_degree, neighbor_deg_mean, is_stdcell, is_macro, is_port] (24 dims)
            # CRITICAL: z_t_conditioned contains noisy positions for std-cells and CLEAN positions for macros/ports
            # This matches training exactly - macros/ports positions are fed as conditioning in the features
            if hasattr(data, 'node_types') or (hasattr(data, '__contains__') and 'inst' in data):
                # HeteroData: update inst.x - prepend z_t_conditioned to original_feats (22 dims) -> 24 dims
                data['inst'].x = torch.cat([z_t_conditioned, original_feats], dim=1)
            else:
                # Homogeneous Data: update data.x - prepend z_t_conditioned to original_feats (22 dims) -> 24 dims
                data.x = torch.cat([z_t_conditioned, original_feats], dim=1)

            # CRITICAL: Pass z_t_conditioned to denoiser (same as training)
            # Model receives clean macro/port positions as conditioning in both features and z_t input
            eps_pred, _ = self.denoiser(data, z_t_conditioned, t_cont)

            # NaN detection during DDPM/EDM sampling: report step (t_idx) where eps_pred becomes non-finite
            if not torch.isfinite(eps_pred).all():
                logger.error(
                    "[sample] NaN/Inf in eps_pred at DDPM step (t_idx) %d of %d. "
                    "First non-finite occurrence during ancestral sampling.",
                    step, self.num_steps,
                )
                raise RuntimeError(
                    f"GraphDDPM.sample: NaN/Inf in eps_pred at step (t_idx) {step} (of {self.num_steps}). "
                    "Check model and batch for instability."
                )

            alpha_t = self.alphas[step]
            alpha_bar_t = self.alphas_cumprod[step]

            if self.cfg.use_predictor_corrector:
                # Predictor-Corrector sampling (like chipdiffusion)
                # Step 1: Predict clean z0 from noisy z_t
                sigma_t = torch.sqrt(1.0 - alpha_bar_t + 1e-8)
                z0_pred = (z_t - sigma_t * eps_pred) / torch.sqrt(alpha_bar_t + 1e-8)

                # CRITICAL: For macros/ports, keep their clean positions fixed
                if z0_clean is not None:
                    z0_pred[~mask] = z0_clean[~mask]

                # Step 2: Compute x_{t-1} from predicted x0
                # Use the DDPM posterior p(z_{t-1} | z_t, z0_pred).
                # NOTE: The previous implementation ignored the z_t term in the posterior mean,
                # which can over-randomize transitions and hurt sample quality.
                if step > 0:
                    alpha_bar_prev = self.alphas_cumprod[step - 1]
                    beta_t = self.betas[step]
                    denom = (1.0 - alpha_bar_t + 1e-8)
                    coef_z0 = (torch.sqrt(alpha_bar_prev) * beta_t) / denom
                    coef_zt = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / denom
                    mean = coef_z0 * z0_pred + coef_zt * z_t

                    # Add noise for stochasticity (only for instances that should be diffused)
                    noise = torch.randn_like(z_t)
                    var = beta_t * (1.0 - alpha_bar_prev) / denom
                    var = torch.clamp(var, min=1e-20, max=1.0)
                    z_t = mean + torch.sqrt(var) * noise

                    # Clamp to configurable range (for normal init noise, [-10, 10] is fine)
                    clip_lo = getattr(self.cfg, 'sample_coord_clip_min', -5.0)
                    clip_hi = getattr(self.cfg, 'sample_coord_clip_max', 5.0)
                    if mask.any():
                        z_t[mask] = z_t[mask].clamp(clip_lo, clip_hi)
                    else:
                        z_t = z_t.clamp(clip_lo, clip_hi)
                    
                    # CRITICAL: Keep macros/ports fixed at their clean positions
                    if z0_clean is not None:
                        z_t[~mask] = z0_clean[~mask]
                    
                    # Track error at this step — de-normalized MAE (physical units)
                    if track_errors:
                        x_t = z_t * s + mu  # Denormalize current positions
                        abs_error = torch.abs(x_t - x0_true)  # (N, 2)
                        mae = abs_error.mean().item()
                        step_errors.append(mae)
                    
                    # Capture intermediate position if requested (in normalized space)
                    if intermediate_positions is not None and step in capture_steps:
                        intermediate_positions.append(z_t.clone())  # Keep in normalized space
                else:
                    # Final step: return predicted z0
                    z_t = z0_pred
                    clip_lo = getattr(self.cfg, 'sample_coord_clip_min', -5.0)
                    clip_hi = getattr(self.cfg, 'sample_coord_clip_max', 5.0)
                    if mask.any():
                        z_t[mask] = z_t[mask].clamp(clip_lo, clip_hi)
                    else:
                        z_t = z_t.clamp(clip_lo, clip_hi)
                    if z0_clean is not None:
                        z_t[~mask] = z0_clean[~mask]
                    
                    # Track error at final step — de-normalized MAE (physical units)
                    if track_errors:
                        x_t = z_t * s + mu  # Denormalize current positions
                        abs_error = torch.abs(x_t - x0_true)  # (N, 2)
                        mae = abs_error.mean().item()
                        step_errors.append(mae)
            else:
                # Standard DDPM sampling (direct formula)
                if step > 0:
                    alpha_bar_prev = self.alphas_cumprod[step - 1]
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=device, dtype=z_t.dtype)

                beta_t = self.betas[step]

                # Coefficients for DDPM update
                sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t + 1e-8)
                sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t + 1e-8)
                coef_eps = beta_t / sqrt_one_minus_alpha_bar_t

                # Compute mean
                mean = sqrt_recip_alpha_t * (z_t - coef_eps * eps_pred)
                
                # Guided sampling: add overlap penalty gradient (guidance_every_steps reduces cost)
                guidance_every_ddpm = getattr(self.cfg, "guidance_every_steps", 1)
                do_guidance_ddpm = (
                    self.cfg.use_guided_sampling
                    and step < self.num_steps * 0.7
                    and (step % guidance_every_ddpm == 0)
                )
                if do_guidance_ddpm:
                    sizes = self._extract_sizes_from_data(data, device)
                    if sizes is not None and sizes.shape[0] == N:
                        # Compute current positions in physical space
                        x_t_physical = z_t * s + mu
                        # Compute overlap gradient (points away from overlaps)
                        overlap_grad = compute_overlap_gradient(
                            x_t_physical,
                            sizes,
                            mask=mask if mask.any() else None,
                            threshold=self.cfg.overlap_penalty_threshold,
                        )
                        # Convert gradient to normalized space
                        overlap_grad_norm = overlap_grad / s  # Normalize gradient
                        # Scale guidance by step (stronger at earlier steps) and guidance_scale
                        t_ratio = step / self.num_steps  # 0 at start, 1 at end
                        guidance_strength = self.cfg.guidance_scale * (1.0 - t_ratio)  # Stronger at early steps
                        # Apply guidance to mean (before adding noise)
                        mean = mean - guidance_strength * overlap_grad_norm * sqrt_recip_alpha_t  # Scale by sqrt_recip_alpha_t for proper scaling

                # Add variance for standard DDPM
                if step > 0:
                    noise = torch.randn_like(z_t)
                    # Variance from original DDPM formulation
                    var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-8)
                    var = torch.clamp(var, min=1e-20, max=1.0)
                    z_t = mean + torch.sqrt(var) * noise
                    # Clamp to configurable range (for normal init noise, [-10, 10] is fine)
                    clip_lo = getattr(self.cfg, 'sample_coord_clip_min', -5.0)
                    clip_hi = getattr(self.cfg, 'sample_coord_clip_max', 5.0)
                    if mask.any():
                        z_t[mask] = z_t[mask].clamp(clip_lo, clip_hi)
                    else:
                        z_t = z_t.clamp(clip_lo, clip_hi)
                    
                    # CRITICAL: Keep macros/ports fixed at their clean positions
                    if z0_clean is not None:
                        z_t[~mask] = z0_clean[~mask]
                    
                    # Track error at this step — de-normalized MAE (physical units)
                    if track_errors:
                        x_t = z_t * s + mu  # Denormalize current positions
                        abs_error = torch.abs(x_t - x0_true)  # (N, 2)
                        mae = abs_error.mean().item()
                        step_errors.append(mae)
                    
                    # Capture intermediate position if requested (in normalized space)
                    if intermediate_positions is not None and step in capture_steps:
                        intermediate_positions.append(z_t.clone())  # Keep in normalized space
                else:
                    z_t = mean
                    # Clamp to configurable range (same as noisy steps)
                    clip_lo = getattr(self.cfg, 'sample_coord_clip_min', -5.0)
                    clip_hi = getattr(self.cfg, 'sample_coord_clip_max', 5.0)
                    if mask.any():
                        z_t[mask] = z_t[mask].clamp(clip_lo, clip_hi)
                    else:
                        z_t = z_t.clamp(clip_lo, clip_hi)
                    
                    # CRITICAL: Keep macros/ports fixed at their clean positions
                    if z0_clean is not None:
                        z_t[~mask] = z0_clean[~mask]
                    
                    # Track error at final step — de-normalized MAE (physical units)
                    if track_errors:
                        x_t = z_t * s + mu  # Denormalize current positions
                        abs_error = torch.abs(x_t - x0_true)  # (N, 2)
                        mae = abs_error.mean().item()
                        step_errors.append(mae)

        # Map back to raw coordinates
        x0 = z_t * s + mu
        # Always capture final position (in normalized space)
        if intermediate_positions is not None:
            intermediate_positions.append(z_t.clone())  # Keep in normalized space
        
        if capture_intermediate is not None:
            if track_errors:
                return x0, step_errors, intermediate_positions
            else:
                return x0, None, intermediate_positions
        elif track_errors:
            return x0, step_errors
        return x0


def build_diffusion_denoiser(
    model_cfg: dict,
    diffusion_cfg: DiffusionConfig,
    ablation_config: Optional[ModelAblationConfig] = None,
) -> DiffusionDenoiser:
    """
    Factory to build a DiffusionDenoiser from a model config dict.

    The expected keys largely mirror those used for FullInstNetModel.
    """
    # Build ablation_config from model_cfg if not provided
    if ablation_config is None:
        ablation_cfg_dict = model_cfg.get("ablation_config")
        if isinstance(ablation_cfg_dict, dict):
            from .model_config import RobustEncodeConfig, GlobalAttentionConfig
            robust_encode_dict = ablation_cfg_dict.get("robust_encode", {})
            robust_encode_config = None
            if robust_encode_dict.get("enabled", False):
                robust_encode_config = RobustEncodeConfig(
                    enabled=True,
                    num_freq_bands=robust_encode_dict.get("num_freq_bands", 8),
                    clamp_norm=robust_encode_dict.get("clamp_norm", False),
                )
                ablation_config = ModelAblationConfig(
                    hidden_dim=model_cfg.get("hidden_dim", 256),
                    use_x_init=True,  # robust_encode requires x_t
                    robust_encode=robust_encode_config,
                )
    
    return DiffusionDenoiser(
        inst_input_dim=model_cfg.get("inst_input_dim", 7),
        net_input_dim=model_cfg.get("net_input_dim", 3),
        edge_input_dim=model_cfg.get("edge_input_dim", 4),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        inst_output_dim=model_cfg.get("inst_output_dim", 2),
        net_output_dim=model_cfg.get("net_output_dim", 2),
        num_gnn_blocks=model_cfg.get("num_gnn_blocks", 4),
        gnn_heads=model_cfg.get("gnn_heads", 4),
        num_encoding_layers=model_cfg.get("num_encoding_layers", 2),
        num_output_layers=model_cfg.get("num_output_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "relu"),
        edge_hidden_dim=model_cfg.get("edge_hidden_dim"),
        use_gcn=model_cfg.get("use_gcn", True),
        gnn_type=model_cfg.get("gnn_type", "bipartite"),  # Infer from data_format if not specified
        diffusion_cfg=diffusion_cfg,
        ablation_config=ablation_config,
    )


