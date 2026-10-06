"""DreamPlace-style stdcell gradient optimizer — pure PyTorch, no external dependencies.

Post-processing step applied to (positions, data) returned by V5.sample().
Macros and ports remain fixed; only stdcell positions are optimized via gradient descent.

Optimization objective (per iteration):
  L = WA-HPWL(pos_all)  +  λ_d(t) * density_overflow(pos_stdcells)
                         +  λ_b   * boundary_penalty(pos_stdcells)

WA-HPWL (Weighted-Average — smooth proxy for true HPWL):
  For edge e = (i, j) with pin offsets [ox_i, oy_i, ox_j, oy_j] from edge_attr:
    pin_i = pos[i] + [ox_i, oy_i]
    pin_j = pos[j] + [ox_j, oy_j]
    dx = (pin_i_x - pin_j_x) / γ
    hpwl_e = γ * (logaddexp(dx, -dx) + logaddexp(dy, -dy))
           = γ * log(2·cosh(dx)) + γ * log(2·cosh(dy))
  As γ → 0 this converges to true |Δx| + |Δy|. For 2-pin nets (our case) WA is exact.

Density penalty (DreamPlace bell/triangle kernel):
  Canvas is divided into Bx × By bins.
  Each stdcell i spreads into nearby bins via a triangle kernel:
    overlap_x[i, b] = relu(1 - |pos_x[i] - cx_b| / (w_i/2 + bw/2)) * (w_i + bw) / (2*bw)
  density[bx, by] = (overlap_x.T @ overlap_y) / (bw * bh)
  penalty = Σ_bins relu(density - target)²

Legalization (post-optimization):
  Regenerates the same row-by-row grid used in V5Placer._generate_structured_grids,
  filters slots that overlap macros/ports, then uses a greedy k-d tree assignment to
  map each stdcell to its nearest available non-overlapping slot.
"""

import math
import warnings
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


@dataclass
class DreamPlaceConfig:
    """Hyper-parameters for the stdcell gradient optimizer."""

    # ── Gradient optimizer ─────────────────────────────────────────────────
    num_iterations: int = 300
    learning_rate: float = 0.01
    optimizer_type: str = "adam"   # "adam" | "sgd"

    # ── WA-HPWL ────────────────────────────────────────────────────────────
    wa_gamma: float = 1.0          # smoothing factor; smaller → closer to true HPWL

    # ── Density penalty ────────────────────────────────────────────────────
    num_bins_x: int = 32
    num_bins_y: int = 32
    target_density: float = 0.7
    lambda_density_max: float = 10.0
    lambda_growth_rate: float = 0.02   # exponential growth per iter after warmup
    warmup_iters: int = 50             # HPWL-only iterations before density kicks in

    # ── Boundary penalty ───────────────────────────────────────────────────
    lambda_boundary: float = 1.0

    # ── Gradient clipping ──────────────────────────────────────────────────
    grad_clip_norm: float = 5.0        # 0 = disabled

    # ── Legalization ───────────────────────────────────────────────────────
    legalize: bool = True
    stdcell_grid_spacing_factor_w: float = 0.97
    stdcell_grid_spacing_factor_h: float = 0.98
    stdcell_macro_keepout: float = 0.2

    # ── Fallback ───────────────────────────────────────────────────────────
    fallback_on_nan: bool = True
    fallback_hpwl_ratio: float = 2.0   # revert if final HPWL > ratio * original

    # ── Device ─────────────────────────────────────────────────────────────
    device: str = "auto"               # "auto" | "cpu" | "cuda"


class StdcellOptimizer:
    """
    Gradient-based stdcell placement optimizer.

    Usage::

        cfg = DreamPlaceConfig(num_iterations=300, legalize=True)
        opt = StdcellOptimizer(cfg)
        new_positions, new_data = opt.optimize(positions, data)

    The returned ``(new_positions, new_data)`` are always valid: on any failure
    the optimizer reverts to the original ``(positions, data)`` unchanged.
    """

    def __init__(self, cfg: DreamPlaceConfig) -> None:
        self.cfg = cfg
        self.device = self._resolve_device()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def optimize(
        self,
        positions: torch.Tensor,   # [V, 2]  CPU, microns
        data: Data,
    ) -> Tuple[torch.Tensor, Data]:
        """Run WA-HPWL + density gradient optimization, then legalize.

        Args:
            positions: [V, 2] placement returned by V5.sample() (CPU tensor).
            data:      PyG Data from V5.sample() with edge_index, edge_attr,
                       is_stdcell, chip_size, placer_seed, x (sizes).

        Returns:
            (new_positions, new_data)  — always on CPU, always valid.
        """
        cfg = self.cfg
        dev = self.device

        is_stdcell = data.is_stdcell.bool()          # [V]
        n_stdcells = int(is_stdcell.sum())
        if n_stdcells == 0:
            return positions, data

        canvas_W = float(data.chip_size[0])
        canvas_H = float(data.chip_size[1])

        # ── Move to device ──────────────────────────────────────────────────
        pos_all      = positions.to(dev)                   # [V, 2]
        sizes_all    = data.x.to(dev)                      # [V, 2]
        edge_index   = data.edge_index.to(dev)             # [2, E]
        edge_attr    = data.edge_attr.to(dev)              # [E, 4]
        is_std_dev   = is_stdcell.to(dev)                  # [V]
        stdcell_idx  = is_std_dev.nonzero(as_tuple=True)[0]  # [S]

        # ── Baseline HPWL for fallback comparison ───────────────────────────
        with torch.no_grad():
            hpwl_orig = self._wa_hpwl(pos_all, edge_index, edge_attr, cfg.wa_gamma).item()
        if hpwl_orig <= 0.0:
            return positions, data

        # ── Free variables: stdcell positions ───────────────────────────────
        pos_free   = pos_all[stdcell_idx].clone().detach().requires_grad_(True)   # [S, 2]
        pos_fixed  = pos_all.detach().clone()   # [V, 2] — obstacle positions (no grad)
        sizes_free = sizes_all[stdcell_idx].detach()   # [S, 2]

        # Adaptive bin count to keep memory manageable
        S = n_stdcells
        bx = min(cfg.num_bins_x, max(8, int(math.sqrt(S / 10))))
        by = min(cfg.num_bins_y, max(8, int(math.sqrt(S / 10))))

        # ── Optimizer ───────────────────────────────────────────────────────
        if cfg.optimizer_type == "adam":
            opt = torch.optim.Adam([pos_free], lr=cfg.learning_rate)
        else:
            opt = torch.optim.SGD([pos_free], lr=cfg.learning_rate, momentum=0.9)

        # ── Gradient loop ───────────────────────────────────────────────────
        for t in range(cfg.num_iterations):
            opt.zero_grad()

            # Assemble full [V, 2] positions with current stdcell estimate
            pos_t = pos_fixed.index_put((stdcell_idx,), pos_free)

            hpwl = self._wa_hpwl(pos_t, edge_index, edge_attr, cfg.wa_gamma)
            lam_d = self._lambda_density(t)
            dens  = self._density_penalty(
                pos_free, sizes_free, canvas_W, canvas_H, bx, by, cfg.target_density
            )
            bnd   = self._boundary_penalty(pos_free, sizes_free, canvas_W, canvas_H)
            loss  = hpwl + lam_d * dens + cfg.lambda_boundary * bnd

            if not torch.isfinite(loss):
                if cfg.fallback_on_nan:
                    return positions, data
                break

            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([pos_free], cfg.grad_clip_norm)
            opt.step()

            # Hard per-cell clamp: ensure each stdcell stays within canvas
            with torch.no_grad():
                hw = sizes_free[:, 0] / 2   # [S]
                hh = sizes_free[:, 1] / 2
                pos_free.data[:, 0] = torch.max(hw, torch.min(canvas_W - hw, pos_free.data[:, 0]))
                pos_free.data[:, 1] = torch.max(hh, torch.min(canvas_H - hh, pos_free.data[:, 1]))

        # ── Assemble final positions ─────────────────────────────────────────
        with torch.no_grad():
            pos_final = pos_fixed.index_put((stdcell_idx,), pos_free)
            hpwl_final = self._wa_hpwl(pos_final, edge_index, edge_attr, cfg.wa_gamma).item()

        # Revert if HPWL regressed badly
        if hpwl_final > cfg.fallback_hpwl_ratio * hpwl_orig:
            return positions, data

        pos_final_cpu = pos_final.cpu()

        # ── Legalization ────────────────────────────────────────────────────
        if cfg.legalize:
            try:
                new_pos, new_sizes = self._legalize(pos_final_cpu, data, canvas_W, canvas_H)
                new_data = _clone_data(data)
                new_data.x = new_sizes.to(data.x.device)
                return new_pos, new_data
            except Exception as exc:
                warnings.warn(
                    f"[DreamPlace] Legalization failed, using raw optimized positions: {exc}"
                )

        return pos_final_cpu, data

    # ─────────────────────────────────────────────────────────────────────────
    # Loss components
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _wa_hpwl(
        pos: torch.Tensor,         # [V, 2]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: torch.Tensor,   # [E, 4]
        gamma: float,
    ) -> torch.Tensor:
        """Weighted-Average HPWL using numerically stable logaddexp.

        For a 2-pin edge e = (i, j):
            hpwl_e = γ·logaddexp(Δx/γ, −Δx/γ) + γ·logaddexp(Δy/γ, −Δy/γ)
                   = γ·log(2·cosh(Δx/γ)) + γ·log(2·cosh(Δy/γ))

        ``torch.logaddexp`` is numerically stable: it computes
        ``max(a,b) + log(1 + exp(−|a−b|))``, avoiding overflow for large |Δ|/γ.
        """
        if edge_index.shape[1] == 0:
            return pos.new_zeros(())

        src, dst = edge_index[0], edge_index[1]
        pin_src = pos[src] + edge_attr[:, 0:2]   # [E, 2]
        pin_dst = pos[dst] + edge_attr[:, 2:4]   # [E, 2]

        dx = (pin_src[:, 0] - pin_dst[:, 0]) / gamma   # [E]
        dy = (pin_src[:, 1] - pin_dst[:, 1]) / gamma

        per_edge = gamma * (torch.logaddexp(dx, -dx) + torch.logaddexp(dy, -dy))
        return per_edge.mean()

    @staticmethod
    def _density_penalty(
        pos_free: torch.Tensor,    # [S, 2] — stdcell positions only
        sizes_free: torch.Tensor,  # [S, 2]
        canvas_W: float,
        canvas_H: float,
        num_bins_x: int,
        num_bins_y: int,
        target_density: float,
    ) -> torch.Tensor:
        """Bell-function (triangle kernel) density overflow penalty.

        density[bx, by] = Σ_i  overlap_x[i, bx] · overlap_y[i, by] / (bw · bh)

        where the triangle kernel in x is:
            overlap_x[i, b] = relu(1 − |px_i − cx_b| / (w_i/2 + bw/2))
                               × (w_i + bw) / (2·bw)

        penalty = Σ_bins relu(density − target)²
        """
        dev = pos_free.device
        bw  = canvas_W / num_bins_x
        bh  = canvas_H / num_bins_y

        cx = torch.linspace(bw / 2, canvas_W - bw / 2, num_bins_x, device=dev)  # [Bx]
        cy = torch.linspace(bh / 2, canvas_H - bh / 2, num_bins_y, device=dev)  # [By]

        px = pos_free[:, 0]   # [S]
        py = pos_free[:, 1]
        sw = sizes_free[:, 0] # [S]
        sh = sizes_free[:, 1]

        # Triangle kernel — [S, Bx]
        dx          = (px.unsqueeze(1) - cx.unsqueeze(0)).abs()
        half_ext_x  = sw.unsqueeze(1) / 2 + bw / 2
        overlap_x   = F.relu(1.0 - dx / half_ext_x.clamp(min=1e-6)) \
                      * (sw.unsqueeze(1) + bw) / (2.0 * bw)

        # Triangle kernel — [S, By]
        dy          = (py.unsqueeze(1) - cy.unsqueeze(0)).abs()
        half_ext_y  = sh.unsqueeze(1) / 2 + bh / 2
        overlap_y   = F.relu(1.0 - dy / half_ext_y.clamp(min=1e-6)) \
                      * (sh.unsqueeze(1) + bh) / (2.0 * bh)

        # density[bx, by] via batched outer product: (Bx × S) @ (S × By) → [Bx, By]
        density_grid = (overlap_x.T @ overlap_y) / (bw * bh)
        return F.relu(density_grid - target_density).pow(2).sum()

    @staticmethod
    def _boundary_penalty(
        pos_free: torch.Tensor,    # [S, 2]
        sizes_free: torch.Tensor,  # [S, 2]
        canvas_W: float,
        canvas_H: float,
    ) -> torch.Tensor:
        """Squared penalty for stdcells extending outside the canvas."""
        hw = sizes_free[:, 0] / 2
        hh = sizes_free[:, 1] / 2
        px, py = pos_free[:, 0], pos_free[:, 1]
        return (
            F.relu(hw - px).pow(2)              # left wall
            + F.relu(px + hw - canvas_W).pow(2) # right wall
            + F.relu(hh - py).pow(2)            # bottom wall
            + F.relu(py + hh - canvas_H).pow(2) # top wall
        ).sum()

    # ─────────────────────────────────────────────────────────────────────────
    # Lambda schedule
    # ─────────────────────────────────────────────────────────────────────────

    def _lambda_density(self, t: int) -> float:
        """Exponential ramp from 0 → lambda_density_max starting after warmup_iters."""
        cfg = self.cfg
        if t < cfg.warmup_iters:
            return 0.0
        growth = (t - cfg.warmup_iters) * cfg.lambda_growth_rate
        return min(cfg.lambda_density_max, 0.001 * math.exp(growth))

    # ─────────────────────────────────────────────────────────────────────────
    # Legalization
    # ─────────────────────────────────────────────────────────────────────────

    def _legalize(
        self,
        opt_pos: torch.Tensor,  # [V, 2] CPU — optimized, stdcells may overlap
        data: Data,
        canvas_W: float,
        canvas_H: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy k-d tree legalization.

        1. Regenerate row-by-row grid (matching V5Placer._generate_structured_grids).
        2. Remove macro/port-overlapping slots (vectorized AABB with keepout margin).
        3. Sort stdcells by distance to nearest slot (descending = hardest first).
        4. Assign each stdcell to its nearest unoccupied slot.

        Returns:
            new_positions: [V, 2] tensor (CPU)
            new_sizes:     [V, 2] tensor (CPU) — stdcell entries updated to slot sizes
        """
        try:
            from scipy.spatial import KDTree
        except ImportError:
            raise RuntimeError(
                "scipy is required for legalization. Install it with: pip install scipy"
            )

        cfg = self.cfg

        is_stdcell  = data.is_stdcell.bool().cpu()
        stdcell_idx = is_stdcell.nonzero(as_tuple=True)[0].numpy()   # [S]
        sizes_np    = data.x.cpu().numpy()                            # [V, 2]
        sc_sizes    = sizes_np[stdcell_idx]                           # [S, 2]
        placer_seed = int(data.placer_seed.item())

        # Obstacle positions and sizes (macros + ports — they haven't moved)
        obs_mask = (~is_stdcell).numpy()
        obs_pos  = opt_pos.numpy()[obs_mask]    # [M+P, 2]
        obs_sz   = sizes_np[obs_mask]           # [M+P, 2]

        # ── Step 1: Generate grid slots ──────────────────────────────────────
        all_slots = _generate_grid_slots(sc_sizes, canvas_W, canvas_H, seed=placer_seed)
        if not all_slots:
            return opt_pos, data.x.cpu()

        # ── Step 2: Remove macro/port-overlapping slots (vectorised) ────────
        k = cfg.stdcell_macro_keepout
        slot_arr  = np.array(all_slots, dtype=np.float32)     # [N, 4] (cx, cy, sw, sh)
        s_cx, s_cy, s_sw, s_sh = slot_arr[:, 0], slot_arr[:, 1], slot_arr[:, 2], slot_arr[:, 3]
        s_xmin = s_cx - s_sw / 2 - k
        s_ymin = s_cy - s_sh / 2 - k
        s_xmax = s_cx + s_sw / 2 + k
        s_ymax = s_cy + s_sh / 2 + k

        if len(obs_pos) > 0:
            o_xmin = (obs_pos[:, 0] - obs_sz[:, 0] / 2)[None, :]  # [1, M+P]
            o_ymin = (obs_pos[:, 1] - obs_sz[:, 1] / 2)[None, :]
            o_xmax = (obs_pos[:, 0] + obs_sz[:, 0] / 2)[None, :]
            o_ymax = (obs_pos[:, 1] + obs_sz[:, 1] / 2)[None, :]

            overlap_x = (s_xmin[:, None] < o_xmax) & (s_xmax[:, None] > o_xmin)
            overlap_y = (s_ymin[:, None] < o_ymax) & (s_ymax[:, None] > o_ymin)
            blocked   = (overlap_x & overlap_y).any(axis=1)   # [N]
            avail_arr = slot_arr[~blocked]
        else:
            avail_arr = slot_arr

        if len(avail_arr) == 0:
            return opt_pos, data.x.cpu()

        # ── Step 3: k-d tree over available slot centres ─────────────────────
        slot_centers = avail_arr[:, :2]                # [A, 2]
        kdtree       = KDTree(slot_centers)

        # ── Step 4: Greedy assignment ─────────────────────────────────────────
        opt_sc = opt_pos.numpy()[stdcell_idx]          # [S, 2]

        # Sort by distance to nearest slot — largest first (hardest-to-place first)
        dists_to_nearest, _ = kdtree.query(opt_sc, k=1)
        order = np.argsort(-dists_to_nearest)

        max_k   = min(32, len(avail_arr))
        sf_w    = cfg.stdcell_grid_spacing_factor_w
        sf_h    = cfg.stdcell_grid_spacing_factor_h
        occupied = set()

        assigned_pos   = opt_sc.copy()
        assigned_sizes = sc_sizes.copy()

        for i in order:
            _, nn_idx = kdtree.query(opt_sc[i:i+1], k=max_k)
            nn_idx = nn_idx.flatten()
            for slot_i in nn_idx:
                if slot_i not in occupied:
                    occupied.add(int(slot_i))
                    cx, cy, sw, sh = avail_arr[slot_i]
                    assigned_pos[i]   = [cx, cy]
                    assigned_sizes[i] = [sw * sf_w, sh * sf_h]
                    break
            # else: keep optimized position (should not happen with adequate slot pool)

        # ── Rebuild full tensors ─────────────────────────────────────────────
        new_positions = opt_pos.clone()
        new_positions[torch.from_numpy(stdcell_idx)] = torch.from_numpy(
            assigned_pos.astype(np.float32)
        )

        new_sizes = data.x.cpu().clone()
        new_sizes[torch.from_numpy(stdcell_idx)] = torch.from_numpy(
            assigned_sizes.astype(np.float32)
        )
        return new_positions, new_sizes

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_device(self) -> torch.device:
        if self.cfg.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.cfg.device)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate_grid_slots(
    sc_sizes: np.ndarray,   # [S, 2]  stdcell (w, h) — from data.x after placement
    canvas_W: float,
    canvas_H: float,
    seed: int,
) -> List[Tuple[float, float, float, float]]:
    """Replicate V5Placer._generate_structured_grids to produce legalization slots.

    Uses the same row-by-row packing algorithm and the same RNG type
    (``random.Random``), seeded deterministically from ``placer_seed``.
    Because the original grid was generated mid-way through the placer's RNG
    sequence (after macro and port placement), the regenerated grid will
    differ in the specific slots but match the size statistics and canvas coverage.
    For legalization purposes, statistical compatibility is all that matters.

    Returns:
        List of (cx, cy, w, h) — shuffled slot centres and sizes.
    """
    if len(sc_sizes) == 0:
        return []

    widths  = sc_sizes[:, 0]
    heights = sc_sizes[:, 1]
    w_mean, w_std = float(np.mean(widths)),  float(np.std(widths))
    w_min,  w_max = float(np.min(widths)),   float(np.max(widths))
    h_mean, h_std = float(np.mean(heights)), float(np.std(heights))
    h_min,  h_max = float(np.min(heights)),  float(np.max(heights))

    rng   = random.Random(seed)
    slots: List[Tuple[float, float, float, float]] = []
    y = 0.0

    while y < canvas_H:
        row_h = rng.gauss(h_mean, h_std * 0.5)
        row_h = max(h_min, min(h_max * 1.5, row_h))
        if y + row_h > canvas_H:
            if canvas_H - y < h_min:
                break
            row_h = canvas_H - y
        if row_h <= 0:
            break
        row_cy = y + row_h / 2
        x = 0.0
        while x < canvas_W:
            cell_w = rng.gauss(w_mean, w_std * 0.5)
            cell_w = max(w_min, min(w_max * 1.5, cell_w))
            if x + cell_w > canvas_W:
                if canvas_W - x < w_min:
                    break
                cell_w = canvas_W - x
            if cell_w <= 0:
                break
            slots.append((x + cell_w / 2, row_cy, cell_w, row_h))
            x += cell_w
        y += row_h

    rng.shuffle(slots)
    return slots


def _clone_data(data: Data) -> Data:
    """Shallow-clone a PyG Data object (tensors are cloned, non-tensors shared)."""
    try:
        return data.clone()
    except AttributeError:
        kwargs = {}
        for key, val in data:
            kwargs[key] = val.clone() if isinstance(val, torch.Tensor) else val
        return Data(**kwargs)
