"""Pin offset and driver generation."""

import torch
from typing import Dict

from .config import PinConfig, DegreeConfig


def sample_pin_attrs(
    inst_feats: Dict[str, torch.Tensor],
    inst_id_per_edge: torch.Tensor,
    net_id_per_edge: torch.Tensor,
    deg: torch.Tensor,
    pin_cfg: PinConfig,
    deg_cfg: DegreeConfig,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Sample pin attributes for each edge: offset (dx, dy) + driver flag.

    Pin offsets are relative to instance position (future use).
    For now, they represent logical pin positions on cell boundaries.

    Args:
        inst_feats: Dictionary with instance features (w, h, is_macro, is_port)
        inst_id_per_edge: [E] instance ID for each edge
        net_id_per_edge: [E] net ID for each edge
        deg: [N_net] degree for each net
        pin_cfg: Pin configuration
        deg_cfg: Degree configuration (for huge net threshold)
        device: Device

    Returns:
        edge_attr: [E, 4] with [pin_dx, pin_dy, is_driver, pin_role]
    """
    E = len(inst_id_per_edge)

    # Gather instance features per edge
    w_per_edge = inst_feats["w"][inst_id_per_edge]
    h_per_edge = inst_feats["h"][inst_id_per_edge]
    is_macro_per_edge = inst_feats["is_macro"][inst_id_per_edge]
    is_port_per_edge = inst_feats["is_port"][inst_id_per_edge]

    # --- Sample pin offsets ---
    pin_dx = torch.zeros(E, dtype=torch.float32, device=device)
    pin_dy = torch.zeros(E, dtype=torch.float32, device=device)

    # Standard cells: prefer left/right edges
    stdcell_mask = (is_macro_per_edge == 0) & (is_port_per_edge == 0)
    n_stdcells = stdcell_mask.sum().item()

    if n_stdcells > 0:
        # Choose edge: left/right with probability stdcell_lr_prob
        is_lr = torch.rand(n_stdcells, device=device) < pin_cfg.stdcell_lr_prob

        w_std = w_per_edge[stdcell_mask]
        h_std = h_per_edge[stdcell_mask]

        # Left/Right edges
        dx_lr = torch.where(
            torch.rand(n_stdcells, device=device) < 0.5,
            torch.zeros_like(w_std),  # left edge
            w_std  # right edge
        )
        dy_lr = torch.rand(n_stdcells, device=device) * h_std

        # Top/Bottom edges
        dx_tb = torch.rand(n_stdcells, device=device) * w_std
        dy_tb = torch.where(
            torch.rand(n_stdcells, device=device) < 0.5,
            torch.zeros_like(h_std),  # bottom edge
            h_std  # top edge
        )

        dx_std = torch.where(is_lr, dx_lr, dx_tb)
        dy_std = torch.where(is_lr, dy_lr, dy_tb)

        pin_dx[stdcell_mask] = dx_std
        pin_dy[stdcell_mask] = dy_std

    # Macros: more uniform over all 4 edges
    macro_mask_bool = is_macro_per_edge.bool()
    n_macros = macro_mask_bool.sum().item()

    if n_macros > 0:
        edge_choice = torch.randint(0, 4, (n_macros,), device=device)

        w_macro = w_per_edge[macro_mask_bool]
        h_macro = h_per_edge[macro_mask_bool]

        dx_macro = torch.zeros(n_macros, device=device)
        dy_macro = torch.zeros(n_macros, device=device)

        # Edge 0: left
        mask_0 = edge_choice == 0
        dx_macro[mask_0] = 0
        dy_macro[mask_0] = torch.rand(mask_0.sum().item(), device=device) * h_macro[mask_0]

        # Edge 1: right
        mask_1 = edge_choice == 1
        dx_macro[mask_1] = w_macro[mask_1]
        dy_macro[mask_1] = torch.rand(mask_1.sum().item(), device=device) * h_macro[mask_1]

        # Edge 2: bottom
        mask_2 = edge_choice == 2
        dx_macro[mask_2] = torch.rand(mask_2.sum().item(), device=device) * w_macro[mask_2]
        dy_macro[mask_2] = 0

        # Edge 3: top
        mask_3 = edge_choice == 3
        dx_macro[mask_3] = torch.rand(mask_3.sum().item(), device=device) * w_macro[mask_3]
        dy_macro[mask_3] = h_macro[mask_3]

        pin_dx[macro_mask_bool] = dx_macro
        pin_dy[macro_mask_bool] = dy_macro

    # Ports: pin at origin (simplified)
    port_mask_bool = is_port_per_edge.bool()
    if port_mask_bool.any():
        pin_dx[port_mask_bool] = 0.0
        pin_dy[port_mask_bool] = 0.0

    # --- Sample driver for each net (VECTORIZED) ---
    is_driver = torch.zeros(E, dtype=torch.float32, device=device)

    # For each net, choose one edge as driver
    N_net = len(deg)

    # Compute start index for each net
    net_start = torch.cat([torch.tensor([0], device=device), deg.cumsum(0)[:-1]])

    # Identify huge nets
    huge_threshold = deg_cfg.medium_deg_range[1]
    is_huge_net = deg > huge_threshold

    # Sample random offset within each net [N_net]
    driver_offset_per_net = (torch.rand(N_net, device=device) * deg.float()).long()
    driver_offset_per_net = torch.minimum(driver_offset_per_net, deg - 1)
    driver_offset_per_net = torch.clamp(driver_offset_per_net, min=0)

    # For huge nets with ports, bias toward port drivers
    # This optimization handles the special case but only loops over huge nets (rare)
    if is_huge_net.any():
        huge_net_indices = torch.where(is_huge_net)[0]

        # Batch process huge nets to minimize loop overhead
        for net_idx in huge_net_indices:
            start_idx = net_start[net_idx]
            end_idx = start_idx + deg[net_idx]

            # Check if this net has ports
            ports_in_net_mask = is_port_per_edge[start_idx:end_idx]
            has_ports = ports_in_net_mask.any()
            use_port_bias = torch.rand(1, device=device).item() < pin_cfg.huge_net_driver_port_bias

            if has_ports and use_port_bias:
                # Choose from port edges
                port_offsets = torch.where(ports_in_net_mask)[0]
                chosen_port_offset = port_offsets[torch.randint(0, len(port_offsets), (1,), device=device)]
                driver_offset_per_net[net_idx] = chosen_port_offset

    # Compute absolute driver indices [N_net]
    driver_indices = net_start + driver_offset_per_net

    # Set driver flags
    is_driver[driver_indices] = 1.0

    # Pin role (placeholder, can be extended)
    pin_role = torch.zeros(E, dtype=torch.float32, device=device)

    # Stack into edge attributes
    edge_attr = torch.stack([pin_dx, pin_dy, is_driver, pin_role], dim=1)

    return edge_attr
