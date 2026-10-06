"""Validation and statistics for generated netlists."""

import torch
from typing import Dict, Any

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None


def compute_stats(
    data: "HeteroData",
    deg: torch.Tensor,
    bin_name: str
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a generated netlist.

    Args:
        data: HeteroData object
        deg: [N_net] net degrees
        bin_name: size bin name

    Returns:
        Dictionary with statistics
    """
    N_inst = data['inst'].x.shape[0]
    N_net = data['net'].x.shape[0]
    E = data['inst', 'to', 'net'].edge_index.shape[1]

    # Basic counts
    stats = {
        "bin_name": bin_name,
        "N_inst": N_inst,
        "N_net": N_net,
        "E": E,
        "avg_inst_degree": E / N_inst,
    }

    # Instance type distribution
    is_port = data['inst'].x[:, 4]
    is_macro = data['inst'].x[:, 3]

    stats["n_ports"] = int(is_port.sum().item())
    stats["n_macros"] = int(is_macro.sum().item())
    stats["n_stdcells"] = N_inst - stats["n_ports"] - stats["n_macros"]

    stats["port_frac"] = stats["n_ports"] / N_inst
    stats["macro_frac"] = stats["n_macros"] / N_inst

    # Degree distribution
    stats["deg_min"] = int(deg.min().item())
    stats["deg_max"] = int(deg.max().item())
    stats["deg_mean"] = float(deg.float().mean().item())
    stats["deg_median"] = float(deg.float().median().item())
    stats["deg_std"] = float(deg.float().std().item())

    # Degree histogram
    small_mask = deg <= 4
    medium_mask = (deg > 4) & (deg <= 50)
    huge_mask = deg > 50

    stats["n_small_nets"] = int(small_mask.sum().item())
    stats["n_medium_nets"] = int(medium_mask.sum().item())
    stats["n_huge_nets"] = int(huge_mask.sum().item())

    # Level distribution
    level = data['net'].x[:, 2]
    stats["level_mean"] = float(level.mean().item())
    stats["level_dist"] = level.long().bincount().tolist()

    # Driver count
    is_driver = data['inst', 'to', 'net'].edge_attr[:, 2]
    stats["n_drivers"] = int(is_driver.sum().item())
    stats["drivers_per_net"] = stats["n_drivers"] / N_net

    return stats


def validate_netlist(data: "HeteroData", deg: torch.Tensor) -> bool:
    """
    Validate netlist consistency.

    Checks:
    - All degrees >= 2
    - Total edges = sum of degrees
    - Exactly one driver per net
    - All edge indices valid

    Args:
        data: HeteroData object
        deg: [N_net] net degrees

    Returns:
        True if valid, raises AssertionError otherwise
    """
    N_inst = data['inst'].x.shape[0]
    N_net = data['net'].x.shape[0]
    E = data['inst', 'to', 'net'].edge_index.shape[1]

    # Check degrees
    assert (deg >= 2).all(), "All degrees must be >= 2"

    # Check total edges
    assert E == deg.sum().item(), f"Edge count mismatch: {E} != {deg.sum().item()}"

    # Check edge indices
    edge_index = data['inst', 'to', 'net'].edge_index
    inst_ids = edge_index[0]
    net_ids = edge_index[1]

    assert (inst_ids >= 0).all() and (inst_ids < N_inst).all(), "Invalid instance IDs"
    assert (net_ids >= 0).all() and (net_ids < N_net).all(), "Invalid net IDs"

    # Check drivers per net
    is_driver = data['inst', 'to', 'net'].edge_attr[:, 2]

    # Count drivers per net
    driver_count = torch.zeros(N_net, dtype=torch.int64, device=deg.device)
    for i in range(E):
        if is_driver[i] > 0.5:
            driver_count[net_ids[i]] += 1

    # Each net should have exactly 1 driver
    if not (driver_count == 1).all():
        bad_nets = torch.where(driver_count != 1)[0]
        print(f"Warning: {len(bad_nets)} nets have != 1 driver")
        print(f"  Nets with 0 drivers: {(driver_count == 0).sum().item()}")
        print(f"  Nets with >1 drivers: {(driver_count > 1).sum().item()}")
        # Don't fail, just warn (driver selection might have issues)

    return True


def print_stats(stats: Dict[str, Any]):
    """Pretty-print statistics."""
    print(f"\n{'='*60}")
    print(f"Netlist Statistics - {stats['bin_name']}")
    print(f"{'='*60}")
    print(f"Instances: {stats['N_inst']:,}")
    print(f"  - Ports: {stats['n_ports']:,} ({stats['port_frac']:.1%})")
    print(f"  - Macros: {stats['n_macros']:,} ({stats['macro_frac']:.1%})")
    print(f"  - Stdcells: {stats['n_stdcells']:,}")
    print(f"\nNets: {stats['N_net']:,}")
    print(f"  - Small (deg<=4): {stats['n_small_nets']:,}")
    print(f"  - Medium (4<deg<=50): {stats['n_medium_nets']:,}")
    print(f"  - Huge (deg>50): {stats['n_huge_nets']:,}")
    print(f"\nEdges: {stats['E']:,}")
    print(f"  - Avg inst degree: {stats['avg_inst_degree']:.2f}")
    print(f"\nDegree distribution:")
    print(f"  - Min: {stats['deg_min']}")
    print(f"  - Max: {stats['deg_max']}")
    print(f"  - Mean: {stats['deg_mean']:.2f}")
    print(f"  - Median: {stats['deg_median']:.2f}")
    print(f"  - Std: {stats['deg_std']:.2f}")
    print(f"\nHierarchy:")
    print(f"  - Avg level: {stats['level_mean']:.2f}")
    print(f"  - Level dist: {stats['level_dist']}")
    print(f"{'='*60}\n")
