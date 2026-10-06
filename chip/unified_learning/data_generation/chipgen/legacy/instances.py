"""Instance feature generation (stdcells, macros, ports)."""

import torch
from typing import Dict

from .config import InstanceConfig


def sample_instance_features(
    N_inst: int,
    inst_cfg: InstanceConfig,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Generate instance features for all instances.

    Instances include:
    - Ports (is_port=1): boundary I/O
    - Macros (is_macro=1): large blocks
    - Standard cells: rest

    Args:
        N_inst: Number of instances to generate
        inst_cfg: Instance configuration
        device: Device for tensors

    Returns:
        Dictionary containing:
        - cell_type: [N_inst] discrete cell type ID (0=PORT, 1=MACRO, 2-30=stdcell types)
        - w: [N_inst] widths
        - h: [N_inst] heights
        - area: [N_inst] areas
        - is_macro: [N_inst] binary flag
        - is_port: [N_inst] binary flag
        - pin_cap: [N_inst] pin capacity (avg pins)
        - pos: [N_inst, 2] positions (only for ports if enabled)
        - pos_mask: [N_inst] binary mask for valid positions
    """
    # Sample port mask
    n_ports = max(1, int(N_inst * inst_cfg.port_fraction))
    port_indices = torch.randperm(N_inst, device=device)[:n_ports]
    is_port = torch.zeros(N_inst, dtype=torch.float32, device=device)
    is_port[port_indices] = 1.0

    # Sample macro mask (among non-ports)
    non_port_mask = (is_port == 0)
    n_non_ports = non_port_mask.sum().item()
    
    # For larger designs, ensure we have a few (1-8) very large macros
    # Estimate canvas size from N_inst: assume avg stdcell area ~6.0, utilization 0.6
    avg_stdcell_area = 6.0  # avg width ~3.0 * avg height ~2.0
    target_utilization = 0.6
    estimated_canvas_area = (N_inst * avg_stdcell_area) / target_utilization
    estimated_canvas_size = (estimated_canvas_area ** 0.5).item()
    
    # For larger designs (>= 5000 instances), ensure we have a few large macros
    large_design_threshold = 5000
    if N_inst >= large_design_threshold:
        # Always have at least 1, at most 8 very large macros
        n_large_macros = min(8, max(1, int(N_inst / 10000)))  # Scale with design size
        # Calculate minimum macros needed: large macros + some regular ones
        n_macros_base = max(0, int(n_non_ports * inst_cfg.macro_fraction))
        n_macros = max(n_large_macros, n_macros_base)
    else:
        n_macros = max(0, int(n_non_ports * inst_cfg.macro_fraction))

    non_port_indices = torch.where(non_port_mask)[0]
    if n_macros > 0 and len(non_port_indices) > 0:
        macro_among_non_ports = torch.randperm(len(non_port_indices), device=device)[:n_macros]
        macro_indices = non_port_indices[macro_among_non_ports]
        is_macro = torch.zeros(N_inst, dtype=torch.float32, device=device)
        is_macro[macro_indices] = 1.0
    else:
        is_macro = torch.zeros(N_inst, dtype=torch.float32, device=device)

    # Sample sizes
    # Stdcells
    w = torch.empty(N_inst, dtype=torch.float32, device=device)
    h = torch.empty(N_inst, dtype=torch.float32, device=device)

    stdcell_mask = (is_port == 0) & (is_macro == 0)
    n_stdcells = stdcell_mask.sum().item()

    if n_stdcells > 0:
        w[stdcell_mask] = torch.rand(n_stdcells, device=device) * (
            inst_cfg.stdcell_w_range[1] - inst_cfg.stdcell_w_range[0]
        ) + inst_cfg.stdcell_w_range[0]
        h[stdcell_mask] = torch.rand(n_stdcells, device=device) * (
            inst_cfg.stdcell_h_range[1] - inst_cfg.stdcell_h_range[0]
        ) + inst_cfg.stdcell_h_range[0]

    # Macros
    n_macros_actual = is_macro.sum().item()
    if n_macros_actual > 0:
        macro_mask_bool = is_macro.bool()
        macro_indices_list = torch.where(macro_mask_bool)[0]
        
        # For larger designs, assign very large sizes to first few macros
        if N_inst >= large_design_threshold and estimated_canvas_size > 0:
            n_large_macros = min(8, max(1, int(N_inst / 10000)))
            n_large_macros = min(n_large_macros, n_macros_actual)
            
            # Very large macros: 1/10 to 2/10 of canvas size
            large_macro_size_min = estimated_canvas_size * 0.1
            large_macro_size_max = estimated_canvas_size * 0.2
            
            if n_large_macros > 0:
                large_macro_indices = macro_indices_list[:n_large_macros]
                w[large_macro_indices] = torch.rand(n_large_macros, device=device) * (
                    large_macro_size_max - large_macro_size_min
                ) + large_macro_size_min
                h[large_macro_indices] = torch.rand(n_large_macros, device=device) * (
                    large_macro_size_max - large_macro_size_min
                ) + large_macro_size_min
            
            # Regular macros: use scaled ranges based on canvas size, but smaller than large macros
            if n_large_macros < n_macros_actual:
                regular_macro_indices = macro_indices_list[n_large_macros:]
                n_regular = len(regular_macro_indices)
                # Regular macros should be smaller than large macros but scale with design
                # Use a range from base config up to ~5% of canvas (much smaller than large macros)
                regular_max_from_canvas = estimated_canvas_size * 0.05  # Max 5% of canvas for regular macros
                regular_min = inst_cfg.macro_w_range[0]
                # Use the larger of canvas-based max and original config max, but ensure it's reasonable
                regular_max = max(inst_cfg.macro_w_range[1], min(regular_max_from_canvas, estimated_canvas_size * 0.08))
                # Ensure max > min
                regular_max = max(regular_max, regular_min * 1.5)
                
                w[regular_macro_indices] = torch.rand(n_regular, device=device) * (
                    regular_max - regular_min
                ) + regular_min
                h[regular_macro_indices] = torch.rand(n_regular, device=device) * (
                    regular_max - regular_min
                ) + regular_min
        else:
            # For smaller designs, use fixed ranges
            w[macro_mask_bool] = torch.rand(int(n_macros_actual), device=device) * (
                inst_cfg.macro_w_range[1] - inst_cfg.macro_w_range[0]
            ) + inst_cfg.macro_w_range[0]
            h[macro_mask_bool] = torch.rand(int(n_macros_actual), device=device) * (
                inst_cfg.macro_h_range[1] - inst_cfg.macro_h_range[0]
            ) + inst_cfg.macro_h_range[0]

    # Ports (small size)
    n_ports_actual = is_port.sum().item()
    if n_ports_actual > 0:
        port_mask_bool = is_port.bool()
        w[port_mask_bool] = torch.ones(int(n_ports_actual), device=device) * 1.0
        h[port_mask_bool] = torch.ones(int(n_ports_actual), device=device) * 1.0

    # Area
    area = w * h

    # Pin capacity
    pin_cap = torch.rand(N_inst, device=device) * (
        inst_cfg.pin_cap_range[1] - inst_cfg.pin_cap_range[0]
    ) + inst_cfg.pin_cap_range[0]

    # Discrete cell types
    # 0 = PORT, 1 = MACRO, 2-30 = standard cell types (e.g., AND2, OR2, NAND2, etc.)
    cell_type = torch.zeros(N_inst, dtype=torch.int64, device=device)

    # Assign PORT type (type 0)
    cell_type[is_port.bool()] = 0

    # Assign MACRO type (type 1)
    cell_type[is_macro.bool()] = 1

    # Assign standard cell types (types 2-30)
    # Sample from ~29 different standard cell types with varying frequencies
    stdcell_mask = (is_port == 0) & (is_macro == 0)
    n_stdcells = stdcell_mask.sum().item()

    if n_stdcells > 0:
        # Use power-law distribution: common cells appear more frequently
        # Types 2-10 are most common (70%), 11-20 medium (25%), 21-30 rare (5%)
        type_probs = torch.ones(29, device=device)
        type_probs[:9] = 7.0   # Types 2-10: common
        type_probs[9:19] = 2.5  # Types 11-20: medium
        type_probs[19:] = 0.5   # Types 21-30: rare
        type_probs = type_probs / type_probs.sum()

        sampled_types = torch.multinomial(type_probs, n_stdcells, replacement=True) + 2
        cell_type[stdcell_mask] = sampled_types

    result = {
        "cell_type": cell_type,
        "w": w,
        "h": h,
        "area": area,
        "is_macro": is_macro,
        "is_port": is_port,
        "pin_cap": pin_cap,
    }

    # Optional: positions for ports
    if inst_cfg.generate_port_positions:
        pos = torch.zeros(N_inst, 2, dtype=torch.float32, device=device)
        pos_mask = torch.zeros(N_inst, dtype=torch.float32, device=device)

        if n_ports_actual > 0:
            # Place ports on boundary (simplified: random positions)
            # In reality, you'd have a canvas size and place on edges
            port_mask_bool = is_port.bool()
            pos[port_mask_bool] = torch.randn(int(n_ports_actual), 2, device=device) * 100.0
            pos_mask[port_mask_bool] = 1.0

        result["pos"] = pos
        result["pos_mask"] = pos_mask

    return result
