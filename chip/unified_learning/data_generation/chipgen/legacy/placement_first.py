"""Placement-first generation: place instances first, then create edges based on spatial proximity."""

import torch
import torch.distributions as dist
from typing import Dict, Tuple, Optional, Any

from .placement_config import PlacementConfig
from .placement_engine import GPUPlacement
from .export_pyg import to_heterodata


def get_distribution(dist_type: str, params: Dict[str, Any], device: str = "cuda") -> dist.Distribution:
    """
    Create a PyTorch distribution from config.
    
    Args:
        dist_type: Type of distribution ("uniform", "normal", "lognormal", "exponential", "poisson", "bernoulli")
        params: Distribution parameters
        device: Device for tensors
        
    Returns:
        PyTorch Distribution object
    """
    if dist_type == "uniform":
        low = params.get("low", 0.0)
        high = params.get("high", 1.0)
        if isinstance(low, torch.Tensor):
            return dist.Uniform(low.to(device), high.to(device))
        return dist.Uniform(torch.tensor(low, device=device), torch.tensor(high, device=device))
    
    elif dist_type == "normal":
        mean = params.get("mean", 0.0)
        std = params.get("std", 1.0)
        if isinstance(mean, torch.Tensor):
            return dist.Normal(mean.to(device), std.to(device))
        return dist.Normal(torch.tensor(mean, device=device), torch.tensor(std, device=device))
    
    elif dist_type == "lognormal":
        mean = params.get("mean", 0.0)
        std = params.get("std", 1.0)
        if isinstance(mean, torch.Tensor):
            return dist.LogNormal(mean.to(device), std.to(device))
        return dist.LogNormal(torch.tensor(mean, device=device), torch.tensor(std, device=device))
    
    elif dist_type == "exponential":
        rate = params.get("rate", 1.0)
        scale = params.get("scale", 1.0 / rate if rate else 1.0)
        if isinstance(scale, torch.Tensor):
            return dist.Exponential(1.0 / scale.to(device))
        return dist.Exponential(torch.tensor(1.0 / scale, device=device))
    
    elif dist_type == "poisson":
        rate = params.get("rate", 1.0)
        if isinstance(rate, torch.Tensor):
            return dist.Poisson(rate.to(device))
        return dist.Poisson(torch.tensor(rate, device=device))
    
    elif dist_type == "bernoulli":
        probs = params.get("probs", 0.5)
        if isinstance(probs, torch.Tensor):
            return dist.Bernoulli(probs.to(device))
        return dist.Bernoulli(torch.tensor(probs, device=device))
    
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


class ExponentialEdgeDist:
    """Exponential distance-based edge distribution with configurable parameters."""
    
    def __init__(self, scale: float, prob_multiplier_factor: float = 1.0,
                 prob_multiplier_exp: float = 0.0, prob_clip: float = 1.0,
                 global_scale: bool = True, device: str = "cuda"):
        self.scale = torch.tensor(scale, device=device)
        self.prob_multiplier_factor = prob_multiplier_factor
        self.prob_multiplier_exp = prob_multiplier_exp
        self.prob_clip = prob_clip
        self.global_scale = global_scale
        self.device = device
    
    def sample(self, distances: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
        """
        Sample edges based on distances.
        
        Args:
            distances: (V, T, V, T) tensor of terminal distances
            device: Optional device override
            
        Returns:
            (V, T, V, T) binary tensor of edge existence
        """
        if device is None:
            device = self.device
        
        # Compute probabilities: P(edge) = prob_multiplier * exp(-distance/scale)
        if self.global_scale:
            scale = self.scale.to(device)
        else:
            # Per-terminal scale (not implemented in basic version)
            scale = self.scale.to(device)
        
        prob_multiplier = self.prob_multiplier_factor * (scale ** self.prob_multiplier_exp)
        rate = distances / scale
        probs = prob_multiplier * torch.exp(-rate)
        probs = torch.clamp(probs, max=self.prob_clip)
        
        # Sample from Bernoulli
        edge_dist = dist.Bernoulli(probs=probs)
        return edge_dist.sample()


def generate_placement_first(
    N_inst: int,
    placement_cfg: PlacementConfig,
    inst_cfg: Any,  # InstanceConfig from config.py
    device: str = "cuda"
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Generate a placement-first netlist.
    
    Args:
        N_inst: Target number of instances
        placement_cfg: Placement configuration
        inst_cfg: Instance configuration (for cell types, ports, macros)
        device: Device for computation
        
    Returns:
        (positions, inst_feats, net_data) where:
        - positions: (V, 2) instance center positions
        - inst_feats: Dict with instance features
        - net_data: Dict with net information (edge_index, edge_attr, deg, level)
    """
    # Determine stop density
    if placement_cfg.stop_density_dist is not None:
        stop_density_dist = get_distribution(**placement_cfg.stop_density_dist, device=device)
        stop_density = stop_density_dist.sample().item()
    else:
        stop_density = placement_cfg.stop_density
    
    # Initialize placement engine
    placement_device = placement_cfg.placement_device if placement_cfg.placement_device else device
    placement = GPUPlacement(
        device=placement_device,
        chip_min=placement_cfg.chip_min,
        chip_max=placement_cfg.chip_max
    )
    
    # Generate instance sizes
    aspect_ratio_dist = get_distribution(**placement_cfg.aspect_ratio_dist, device=device)
    instance_size_dist = get_distribution(**placement_cfg.instance_size_dist, device=device)
    
    aspect_ratio = aspect_ratio_dist.sample((N_inst,))
    long_size = instance_size_dist.sample((N_inst,))
    short_size = aspect_ratio * long_size
    long_x = dist.Bernoulli(torch.tensor(0.5, device=device)).sample((N_inst,))
    
    x_sizes = long_x * long_size + (1 - long_x) * short_size
    y_sizes = (1 - long_x) * long_size + long_x * short_size
    
    # Sort by area (descending) - place largest first
    areas = x_sizes * y_sizes
    _, indices = torch.sort(areas, descending=True)
    x_sizes = x_sizes[indices]
    y_sizes = y_sizes[indices]
    
    # Place instances
    for i, (x_size, y_size) in enumerate(zip(x_sizes, y_sizes)):
        x_size = float(x_size.item())
        y_size = float(y_size.item())
        
        # Sample candidate position
        dist_params = {
            "low": torch.tensor([(x_size/2) + placement_cfg.chip_min, (y_size/2) + placement_cfg.chip_min], device=device),
            "high": torch.tensor([placement_cfg.chip_max - (x_size/2), placement_cfg.chip_max - (y_size/2)], device=device)
        }
        candidate_dist = get_distribution("uniform", dist_params, device=device)
        
        placed = False
        for attempt in range(placement_cfg.max_attempts_per_instance):
            candidate_pos = candidate_dist.sample()
            x_pos = candidate_pos[0].item()
            y_pos = candidate_pos[1].item()
            
            if placement.check_legality(x_pos, y_pos, x_size, y_size):
                placement.commit_instance(x_pos, y_pos, x_size, y_size, is_port=False)
                placed = True
                break
        
        # Check density
        density = placement.get_density()
        if density >= stop_density:
            break
    
    # Get placed instances
    positions = placement.get_positions()
    sizes = placement.get_sizes()
    num_instances = positions.shape[0]
    
    # Assign cell types (ports, macros, stdcells) based on inst_cfg
    # This is a simplified version - you may want to enhance it
    n_ports = max(1, int(num_instances * inst_cfg.port_fraction))
    port_indices = torch.randperm(num_instances, device=device)[:n_ports]
    is_port = torch.zeros(num_instances, dtype=torch.float32, device=device)
    is_port[port_indices] = 1.0
    
    non_port_mask = (is_port == 0)
    n_non_ports = non_port_mask.sum().item()
    n_macros = max(0, int(n_non_ports * inst_cfg.macro_fraction))
    
    is_macro = torch.zeros(num_instances, dtype=torch.float32, device=device)
    if n_macros > 0:
        non_port_indices = torch.where(non_port_mask)[0]
        macro_among_non_ports = torch.randperm(len(non_port_indices), device=device)[:n_macros]
        macro_indices = non_port_indices[macro_among_non_ports]
        is_macro[macro_indices] = 1.0
    
    # Assign cell types
    cell_type = torch.zeros(num_instances, dtype=torch.long, device=device)
    cell_type[is_port.bool()] = 0  # Ports
    cell_type[is_macro.bool()] = 1  # Macros
    # Standard cells get types 2-30 (simplified)
    stdcell_mask = (is_port == 0) & (is_macro == 0)
    n_stdcells = stdcell_mask.sum().item()
    if n_stdcells > 0:
        stdcell_types = torch.randint(2, 31, (n_stdcells,), device=device)
        cell_type[stdcell_mask] = stdcell_types
    
    # Generate terminals
    instance_area = sizes[:, 0] * sizes[:, 1]
    num_terminals_dist = get_distribution(**placement_cfg.num_terminals_dist, device=device)
    num_terminals = num_terminals_dist.sample(instance_area).int()
    num_terminals = torch.clamp(num_terminals, min=1, max=256)
    max_num_terminals = torch.max(num_terminals)
    
    # Generate terminal offsets
    terminal_offsets = _get_terminal_offsets(
        sizes[:, 0], sizes[:, 1], max_num_terminals,
        placement_cfg, device=device
    )
    
    # Generate edges
    terminal_positions = positions.unsqueeze(dim=1) + terminal_offsets  # (V, T, 2)
    
    edge_device = placement_cfg.edge_device if placement_cfg.edge_device else device
    if edge_device != device:
        terminal_positions = terminal_positions.to(edge_device)
        num_terminals_gpu = num_terminals.to(edge_device)
    else:
        num_terminals_gpu = num_terminals
    
    # Compute terminal distances
    terminal_distances = _get_terminal_distances(
        terminal_positions, placement_cfg.distance_norm_order
    )
    
    # Sample edges
    edge_dist_obj = ExponentialEdgeDist(
        scale=placement_cfg.edge_dist.get("scale", 0.1),
        prob_multiplier_factor=placement_cfg.edge_dist.get("prob_multiplier_factor", 1.0),
        prob_multiplier_exp=placement_cfg.edge_dist.get("prob_multiplier_exp", 0.0),
        prob_clip=placement_cfg.edge_dist.get("prob_clip", 1.0),
        global_scale=placement_cfg.edge_dist.get("global_scale", True),
        device=edge_device
    )
    edge_exists = edge_dist_obj.sample(terminal_distances, device=edge_device)
    
    # Source terminal distribution
    source_terminal_dist = get_distribution(**placement_cfg.source_terminal_dist, device=edge_device)
    is_source = source_terminal_dist.sample((num_instances, max_num_terminals))
    
    # Process edge matrix
    edge_exists = _process_edge_matrix(edge_exists, is_source, num_terminals_gpu)
    
    # Connect isolated instances
    _connect_isolated_instances(edge_exists, terminal_positions, positions, device=edge_device)
    
    # Move back to CPU if needed
    if edge_device != device:
        edge_exists = edge_exists.cpu()
        terminal_offsets = terminal_offsets.cpu()
    
    # Convert to edge list
    edge_index, edge_attr = _generate_edge_list(edge_exists, terminal_offsets)
    
    if placement_cfg.zero_edge_attr:
        edge_attr = 0 * edge_attr
    
    # Convert to net-based representation (for compatibility with existing pipeline)
    # Group edges by net (each source terminal defines a net)
    net_id_per_edge, inst_id_per_edge, deg = _convert_to_net_format(
        edge_exists, num_instances, max_num_terminals, device=device
    )
    
    # Now create edge_index and edge_attr from edge_exists
    # edge_index format: [inst_id, net_id] (not [src_v, dst_v])
    edges = torch.nonzero(edge_exists)  # (E, 4) [src_v, src_t, dst_v, dst_t]
    
    if edges.shape[0] == 0:
        # No edges - return empty tensors
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 4), dtype=torch.float32, device=device)
        net_id_per_edge = torch.empty((0,), dtype=torch.long, device=device)
        inst_id_per_edge = torch.empty((0,), dtype=torch.long, device=device)
    else:
        # Get net IDs for each edge (from source terminal)
        src_vt = edges[:, 0] * max_num_terminals + edges[:, 1]  # Unique ID for (src_v, src_t)
        unique_nets, net_ids = torch.unique(src_vt, return_inverse=True)
        
        # Each edge connects src_v to dst_v via a net
        # Create edges: [src_v -> net] and [dst_v -> net]
        src_inst = edges[:, 0]  # Source instances
        dst_inst = edges[:, 2]  # Destination instances
        net_ids_for_edges = net_ids  # Net ID for each edge
        
        # Create edge_index: [inst_id, net_id]
        # Forward: src_v -> net
        edge_index_src = torch.stack([src_inst, net_ids_for_edges], dim=0)  # (2, E)
        # Reverse: dst_v -> net (same net)
        edge_index_dst = torch.stack([dst_inst, net_ids_for_edges], dim=0)  # (2, E)
        
        edge_index = torch.cat([edge_index_src, edge_index_dst], dim=1)  # (2, 2E)
        
        # Create edge attributes: [pin_dx, pin_dy, is_driver, pin_role]
        edge_attr_source = terminal_offsets[edges[:, 0], edges[:, 1], :]  # (E, 2)
        edge_attr_sink = terminal_offsets[edges[:, 2], edges[:, 3], :]  # (E, 2)
        
        # For src->net edges: source terminal is driver
        is_driver_src = torch.ones(edges.shape[0], 1, device=device)
        pin_role_src = torch.zeros(edges.shape[0], 1, device=device)
        edge_attr_src = torch.cat([edge_attr_source, is_driver_src, pin_role_src], dim=-1)  # (E, 4)
        
        # For dst->net edges: sink terminal is not driver
        is_driver_dst = torch.zeros(edges.shape[0], 1, device=device)
        pin_role_dst = torch.ones(edges.shape[0], 1, device=device)
        edge_attr_dst = torch.cat([edge_attr_sink, is_driver_dst, pin_role_dst], dim=-1)  # (E, 4)
        
        edge_attr = torch.cat([edge_attr_src, edge_attr_dst], dim=0)  # (2E, 4)
        
        # Create inst_id_per_edge and net_id_per_edge for compatibility
        inst_id_per_edge = edge_index[0]  # (2E,)
        net_id_per_edge = edge_index[1]  # (2E,)
    
    # Create dummy level tensor (not used in placement-first, but needed for HeteroData)
    N_net = len(deg)
    level = torch.zeros(N_net, dtype=torch.long, device=device)
    
    # Prepare instance features
    pin_cap = torch.rand(num_instances, device=device) * (
        inst_cfg.pin_cap_range[1] - inst_cfg.pin_cap_range[0]
    ) + inst_cfg.pin_cap_range[0]
    
    inst_feats = {
        "cell_type": cell_type,
        "w": sizes[:, 0],
        "h": sizes[:, 1],
        "area": instance_area,
        "is_macro": is_macro,
        "is_port": is_port,
        "pin_cap": pin_cap,
        "pos": positions,  # Include positions!
        "pos_mask": torch.ones(num_instances, dtype=torch.float32, device=device)
    }
    
    net_data = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "inst_id_per_edge": inst_id_per_edge,
        "net_id_per_edge": net_id_per_edge,
        "deg": deg,
        "level": level
    }
    
    return positions, inst_feats, net_data


def _get_terminal_offsets(
    x_sizes: torch.Tensor,
    y_sizes: torch.Tensor,
    max_num_terminals: int,
    placement_cfg: PlacementConfig,
    reference: str = "center",
    device: str = "cuda"
) -> torch.Tensor:
    """Generate terminal offsets (relative to instance center)."""
    half_perim = (x_sizes + y_sizes)
    
    terminal_locations = get_distribution(
        "uniform", {"low": 0, "high": half_perim}, device=device
    ).sample((max_num_terminals,))  # (max_term, num_instances)
    
    terminal_flip = get_distribution(
        "bernoulli", {"probs": 0.5}, device=device
    ).sample((max_num_terminals, x_sizes.shape[0]))  # (max_term, num_instances)
    terminal_flip = (2 * terminal_flip) - 1
    
    x_sizes = x_sizes.unsqueeze(dim=0)
    y_sizes = y_sizes.unsqueeze(dim=0)
    boundary_offset_x = torch.clamp(terminal_locations, torch.zeros_like(x_sizes), x_sizes) - (x_sizes/2)
    boundary_offset_y = torch.clamp(terminal_locations - x_sizes, torch.zeros_like(y_sizes), y_sizes) - (y_sizes/2)
    
    boundary_offset_x = terminal_flip * boundary_offset_x
    boundary_offset_y = terminal_flip * boundary_offset_y
    
    boundary_offset = torch.stack((boundary_offset_x, boundary_offset_y), dim=-1).movedim(1, 0)
    
    # Interior terminals (if configured)
    if placement_cfg.interior_terminals_dist is not None:
        sizes = torch.stack((x_sizes, y_sizes), dim=-1).squeeze(dim=0)
        gm_size = torch.sqrt(x_sizes * y_sizes).squeeze(dim=0)
        is_terminal_interior = get_distribution(
            **placement_cfg.interior_terminals_dist, device=device
        ).sample(gm_size).view(gm_size.shape[0], 1, 1)
        
        if placement_cfg.interior_terminals_loc == "uniform":
            interior_offset = get_distribution(
                "uniform", {"low": -sizes/2, "high": sizes/2}, device=device
            ).sample((max_num_terminals,))
            interior_offset = interior_offset.moveaxis(0, 1)
        elif placement_cfg.interior_terminals_loc == "center":
            interior_offset = torch.zeros_like(boundary_offset)
        else:
            raise NotImplementedError
        
        terminal_offset = is_terminal_interior * interior_offset + (1 - is_terminal_interior) * boundary_offset
    else:
        terminal_offset = boundary_offset
    
    if reference == "bottom_left":
        terminal_offset[:, :, 0] += x_sizes.squeeze(0) / 2
        terminal_offset[:, :, 1] += y_sizes.squeeze(0) / 2
    
    return terminal_offset


def _get_terminal_distances(terminal_positions: torch.Tensor, norm_order: int = 1) -> torch.Tensor:
    """Compute pairwise terminal distances."""
    if norm_order == "inf":
        norm_order = float("inf")
    
    V, T, _ = terminal_positions.shape
    t_pos_1 = terminal_positions.view(V, T, 1, 1, 2)
    t_pos_2 = terminal_positions.view(1, 1, V, T, 2)
    delta_pos = t_pos_1 - t_pos_2
    distance = torch.norm(delta_pos, p=norm_order, dim=-1)
    return distance


def _process_edge_matrix(
    edge_exists: torch.Tensor,
    is_source: torch.Tensor,
    num_terminals: torch.Tensor
) -> torch.Tensor:
    """Process edge matrix: apply source/sink filters, remove self-edges."""
    V, T, _, _ = edge_exists.shape
    device = edge_exists.device
    
    # Terminal filter
    terminal_filter = torch.zeros((V, T), device=device)
    for i, num_terminal in enumerate(num_terminals):
        terminal_filter[i, :num_terminal] = 1
    
    source_filter = (terminal_filter * is_source).view(V, T, 1, 1)
    sink_filter = (terminal_filter * (1 - is_source)).view(1, 1, V, T)
    self_edge_filter = (1 - torch.eye(V, device=device)).view(V, 1, V, 1)
    
    edges = edge_exists * source_filter
    edges = edges * sink_filter
    edges = edges * self_edge_filter
    
    return edges


def _connect_isolated_instances(
    edge_matrix: torch.Tensor,
    terminal_positions: torch.Tensor,
    instance_positions: torch.Tensor,
    device: str = "cuda"
):
    """Connect isolated instances to nearest neighbors."""
    V, T, _, _ = edge_matrix.shape
    out_degree = edge_matrix.sum(dim=(2, 3))
    in_degree = edge_matrix.sum(dim=(0, 1, 3))
    degree = out_degree.sum(dim=-1) + in_degree
    
    isolated_mask = (degree == 0)
    isolated_instances = torch.nonzero(isolated_mask, as_tuple=False).squeeze(-1).tolist()
    
    if not isolated_instances:
        return
    
    for i in isolated_instances:
        pos_i = instance_positions[i, :]
        instance_distances = torch.norm(instance_positions - pos_i, p=1, dim=-1)
        max_dist = 10.0 + instance_distances.max().item()
        instance_distances[i] = max_dist
        
        target_instance = torch.argmin(instance_distances).item()
        target_positions = terminal_positions[target_instance, :, :]
        terminal_distances = torch.norm(target_positions - pos_i, p=1, dim=-1)
        
        terminal_has_outdegree = (out_degree[target_instance, :] > 0).float()
        terminal_distances = terminal_distances - (terminal_has_outdegree * terminal_distances * 0.3)
        
        terminal_idx = torch.argmin(terminal_distances).item()
        edge_matrix[target_instance, terminal_idx, i, 0] = 1


def _generate_edge_list(
    edge_exists: torch.Tensor,
    terminal_offsets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert edge matrix to edge_index and edge_attr."""
    V, T, _, _ = edge_exists.shape
    edges = torch.nonzero(edge_exists)  # (E, 4: v, t, v, t)
    
    edge_index_forward = edges[:, [0, 2]]
    edge_index_reverse = edges[:, [2, 0]]
    
    edge_attr_source = terminal_offsets[edges[:, 0], edges[:, 1], :]  # (E, 2)
    edge_attr_sink = terminal_offsets[edges[:, 2], edges[:, 3], :]  # (E, 2)
    edge_attr_forward = torch.cat((edge_attr_source, edge_attr_sink), dim=-1)
    edge_attr_reverse = torch.cat((edge_attr_sink, edge_attr_source), dim=-1)
    
    edge_index = torch.cat((edge_index_forward, edge_index_reverse), dim=0).T
    edge_attr = torch.cat((edge_attr_forward, edge_attr_reverse), dim=0)
    
    return edge_index.clone(), edge_attr.clone()


def _convert_to_net_format(
    edge_exists: torch.Tensor,
    num_instances: int,
    max_num_terminals: int,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert terminal-based edge matrix to net-based format.
    
    Each source terminal defines a net. All edges from that terminal belong to the same net.
    
    Args:
        edge_exists: (V, T, V, T) binary tensor
        num_instances: Number of instances
        max_num_terminals: Maximum terminals per instance
        device: Device
        
    Returns:
        (net_id_per_edge, inst_id_per_edge, deg) where:
        - net_id_per_edge: (E,) net ID for each edge
        - inst_id_per_edge: (E,) instance ID for each edge
        - deg: (N_net,) degree of each net
    """
    V, T, _, _ = edge_exists.shape
    
    # Find all edges: (src_v, src_t, dst_v, dst_t)
    edges = torch.nonzero(edge_exists)  # (E, 4)
    
    if edges.shape[0] == 0:
        # No edges
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device)
        )
    
    # Group edges by source terminal: each (src_v, src_t) pair defines a net
    # Create unique net IDs for each (src_v, src_t) pair
    src_vt = edges[:, 0] * max_num_terminals + edges[:, 1]  # Unique ID for (src_v, src_t)
    unique_nets, net_ids = torch.unique(src_vt, return_inverse=True)
    
    # net_ids now maps each edge to its net ID
    net_id_per_edge = net_ids  # (E,)
    inst_id_per_edge = edges[:, 0]  # Source instance IDs
    
    # Compute net degrees
    N_net = len(unique_nets)
    deg = torch.zeros(N_net, dtype=torch.long, device=device)
    for net_id in range(N_net):
        deg[net_id] = (net_id_per_edge == net_id).sum().item()
    
    return net_id_per_edge, inst_id_per_edge, deg

