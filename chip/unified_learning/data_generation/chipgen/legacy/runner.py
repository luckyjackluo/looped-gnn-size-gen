"""Main runner for continuous netlist generation."""

import torch
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

from .config import NetlistConfig
from .sampler_sizes import sample_N, get_bin_name
from .instances import sample_instance_features
from .hierarchy import make_perm, sample_levels
from .degrees import sample_degrees_to_target, degrees_summary
from .endpoints import sample_endpoints_optimized
from .pins import sample_pin_attrs
from .export_pyg import to_heterodata, save_heterodata
from .validate import compute_stats, validate_netlist, print_stats

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None


def generate_extra_edge_attrs(
    inst_feats: Dict[str, torch.Tensor],
    inst_ids: torch.Tensor,
    n_edges: int,
    device: str
) -> torch.Tensor:
    """
    Generate edge attributes for extra edges added to fix isolated instances.

    Args:
        inst_feats: Instance features
        inst_ids: [n_edges] instance IDs for the new edges
        n_edges: Number of extra edges
        device: Device

    Returns:
        edge_attr: [n_edges, 4] with [pin_dx, pin_dy, is_driver=0, pin_role=0]
    """
    w = inst_feats["w"][inst_ids]
    h = inst_feats["h"][inst_ids]

    # Simple pin placement: random on boundary
    pin_dx = torch.rand(n_edges, device=device) * w
    pin_dy = torch.rand(n_edges, device=device) * h

    # Not drivers (these are just fix-up connections)
    is_driver = torch.zeros(n_edges, device=device)
    pin_role = torch.zeros(n_edges, device=device)

    edge_attr = torch.stack([pin_dx, pin_dy, is_driver, pin_role], dim=1)
    return edge_attr


def generate_one(cfg: NetlistConfig, graph_id: int = 0) -> Tuple["HeteroData", Dict[str, Any]]:
    """
    Generate a single synthetic netlist.

    Args:
        cfg: Configuration
        graph_id: ID for this graph (for logging)

    Returns:
        (data, stats): HeteroData object and statistics dictionary
    """
    device = cfg.device

    # Check generation mode
    if cfg.generation_mode == "placement_first":
        return _generate_one_placement_first(cfg, graph_id, device)
    else:
        return _generate_one_netlist_first(cfg, graph_id, device)


def _generate_one_netlist_first(cfg: NetlistConfig, graph_id: int, device: str) -> Tuple["HeteroData", Dict[str, Any]]:
    """Original netlist-first generation."""
    # 1. Sample size
    N_inst, bin_id = sample_N(cfg.size, device=device)
    bin_name = get_bin_name(cfg.size, bin_id)

    # 2. Generate instance features
    inst_feats = sample_instance_features(N_inst, cfg.instance, device=device)

    # 3. Create hierarchy permutation
    perm, invperm = make_perm(N_inst, cfg.hierarchy.perm_seed, device=device)

    # 4. Sample net degrees
    E_target = round(cfg.degree.avg_inst_degree * N_inst)
    deg = sample_degrees_to_target(N_inst, E_target, cfg.degree, device=device)
    N_net = len(deg)

    # 5. Sample hierarchy levels for nets
    level = sample_levels(N_net, cfg.hierarchy, device=device)

    # 6. Sample endpoints (vectorized)
    inst_id_per_edge, net_id_per_edge = sample_endpoints_optimized(
        perm, invperm, deg, level, N_inst,
        cfg.hierarchy, cfg.degree, device=device
    )

    # Note: sample_endpoints_optimized may add extra edges to ensure no isolated instances
    # We need to handle the new edges by adding dummy edge attributes

    # 7. Sample pin attributes
    original_E = deg.sum().item()
    actual_E = len(inst_id_per_edge)

    edge_attr = sample_pin_attrs(
        inst_feats, inst_id_per_edge[:original_E], net_id_per_edge[:original_E], deg,
        cfg.pin, cfg.degree, device=device
    )

    # If extra edges were added (for isolated instances), generate edge attrs for them
    if actual_E > original_E:
        n_extra = actual_E - original_E
        extra_edge_attr = generate_extra_edge_attrs(
            inst_feats, inst_id_per_edge[original_E:], n_extra, device
        )
        edge_attr = torch.cat([edge_attr, extra_edge_attr], dim=0)

    # 8. Convert to HeteroData
    data = to_heterodata(
        inst_feats, deg, level,
        inst_id_per_edge, net_id_per_edge, edge_attr,
        device=device
    )

    # 9. Compute statistics
    stats = compute_stats(data, deg, bin_name)
    stats["graph_id"] = graph_id
    stats["bin_id"] = bin_id

    # 10. Validate (optional, can be expensive)
    # validate_netlist(data, deg)

    return data, stats


def _generate_one_placement_first(cfg: NetlistConfig, graph_id: int, device: str) -> Tuple["HeteroData", Dict[str, Any]]:
    """Placement-first generation."""
    from .placement_first import generate_placement_first
    from .sampler_sizes import sample_N, get_bin_name
    
    # 1. Sample size
    N_inst, bin_id = sample_N(cfg.size, device=device)
    bin_name = get_bin_name(cfg.size, bin_id)
    
    # 2. Generate placement-first netlist
    positions, inst_feats, net_data = generate_placement_first(
        N_inst, cfg.placement, cfg.instance, device=device
    )
    
    # 3. Convert to HeteroData
    data = to_heterodata(
        inst_feats, net_data["deg"], net_data["level"],
        net_data["inst_id_per_edge"], net_data["net_id_per_edge"], net_data["edge_attr"],
        device=device
    )
    
    # 4. Compute statistics
    stats = compute_stats(data, net_data["deg"], bin_name)
    stats["graph_id"] = graph_id
    stats["bin_id"] = bin_id
    stats["has_placement"] = True  # Mark that this has placement
    
    return data, stats


def generate_forever(
    cfg: NetlistConfig,
    out_dir: str,
    log_path: str = "generation.jsonl",
    print_every: int = 10,
    validate_every: int = 200,
    max_graphs: int = None,
    save_graphs: bool = True,
    graphs_per_shard: int = 1000,
    group_by_bin: bool = True,
):
    """
    Continuously generate synthetic netlists.

    Args:
        cfg: Configuration
        out_dir: Output directory for saving graphs
        log_path: Path to JSON Lines log file
        print_every: Print progress every N graphs
        validate_every: Run validation every N graphs
        max_graphs: Maximum graphs to generate (None = infinite)
        save_graphs: Whether to save graphs to disk
        graphs_per_shard: Number of graphs to save per .pt file (default: 1000)
        group_by_bin: If True, organize shards by size bin (default: True)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(log_path)
    log_file = open(log_path, "a")

    # Bin counters
    bin_counters = {name: 0 for name in cfg.size.bin_names}
    total_generated = 0

    # Shard buffers (for batch saving)
    if group_by_bin:
        shard_buffers = {name: [] for name in cfg.size.bin_names}
        shard_counters = {name: 0 for name in cfg.size.bin_names}
    else:
        shard_buffer = []
        shard_counter = 0

    print("="*60)
    print("Synthetic Netlist Generation")
    print("="*60)
    print(f"Output directory: {out_dir}")
    print(f"Log file: {log_path}")
    print(f"Device: {cfg.device}")
    print(f"Target size bins: {cfg.size.bin_names}")
    print(f"Target weights: {cfg.size.weights}")
    print(f"Graphs per shard: {graphs_per_shard}")
    print(f"Group by bin: {group_by_bin}")
    print("="*60)
    print("\nStarting generation...\n")

    start_time = time.time()

    try:
        graph_id = 0
        while True:
            if max_graphs is not None and graph_id >= max_graphs:
                break

            # Generate one graph
            try:
                data, stats = generate_one(cfg, graph_id=graph_id)

                # Update counters
                bin_name = stats["bin_name"]
                bin_counters[bin_name] += 1
                total_generated += 1

                # Move to CPU if needed
                if save_graphs and cfg.device == "cuda":
                    data = data.cpu()

                # Add to shard buffer
                if save_graphs:
                    if group_by_bin:
                        shard_buffers[bin_name].append(data)

                        # Save shard if buffer is full
                        if len(shard_buffers[bin_name]) >= graphs_per_shard:
                            shard_id = shard_counters[bin_name]
                            shard_path = out_dir / f"{bin_name}_shard_{shard_id:04d}.pt"
                            torch.save(shard_buffers[bin_name], str(shard_path))
                            print(f"  → Saved {len(shard_buffers[bin_name])} graphs to {shard_path.name}")
                            shard_buffers[bin_name] = []
                            shard_counters[bin_name] += 1
                    else:
                        shard_buffer.append(data)

                        # Save shard if buffer is full
                        if len(shard_buffer) >= graphs_per_shard:
                            shard_path = out_dir / f"shard_{shard_counter:04d}.pt"
                            torch.save(shard_buffer, str(shard_path))
                            print(f"  → Saved {len(shard_buffer)} graphs to {shard_path.name}")
                            shard_buffer = []
                            shard_counter += 1

                # Log to file
                log_file.write(json.dumps(stats) + "\n")
                log_file.flush()

                # Print progress
                if (graph_id + 1) % print_every == 0:
                    elapsed = time.time() - start_time
                    rate = total_generated / elapsed
                    print(f"[{graph_id+1:6d}] Generated {total_generated} graphs "
                          f"({rate:.2f} graphs/s)")
                    print(f"          Bin distribution: {bin_counters}")

                    # Print realized proportions
                    realized_props = {k: v/total_generated for k, v in bin_counters.items()}
                    print(f"          Realized props: {realized_props}")

                # Validate periodically
                if validate_every > 0 and (graph_id + 1) % validate_every == 0:
                    print(f"\n[Validation at graph {graph_id+1}]")
                    try:
                        validate_netlist(data, torch.tensor([stats["deg_mean"]]))
                        print("✓ Validation passed")
                    except AssertionError as e:
                        print(f"✗ Validation failed: {e}")
                    print()

                graph_id += 1

            except Exception as e:
                print(f"Error generating graph {graph_id}: {e}")
                import traceback
                traceback.print_exc()
                graph_id += 1
                continue

    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")

    finally:
        # Save any remaining graphs in buffers
        if save_graphs:
            print("\nSaving remaining graphs in buffers...")
            if group_by_bin:
                for bin_name, buffer in shard_buffers.items():
                    if len(buffer) > 0:
                        shard_id = shard_counters[bin_name]
                        shard_path = out_dir / f"{bin_name}_shard_{shard_id:04d}.pt"
                        torch.save(buffer, str(shard_path))
                        print(f"  → Saved {len(buffer)} graphs to {shard_path.name}")
            else:
                if len(shard_buffer) > 0:
                    shard_path = out_dir / f"shard_{shard_counter:04d}.pt"
                    torch.save(shard_buffer, str(shard_path))
                    print(f"  → Saved {len(shard_buffer)} graphs to {shard_path.name}")

        log_file.close()
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print("Generation Summary")
        print("="*60)
        print(f"Total graphs generated: {total_generated}")
        print(f"Elapsed time: {elapsed:.2f}s")
        print(f"Average rate: {total_generated/elapsed:.2f} graphs/s")
        print(f"\nFinal bin distribution:")
        for bin_name, count in bin_counters.items():
            prop = count / total_generated if total_generated > 0 else 0
            print(f"  {bin_name}: {count} ({prop:.1%})")
        print("="*60)


def generate_batch(
    cfg: NetlistConfig,
    n_graphs: int,
    out_dir: str = None,
    verbose: bool = True
) -> list:
    """
    Generate a batch of graphs and return them as a list.

    Useful for training pipelines that consume graphs directly.

    Args:
        cfg: Configuration
        n_graphs: Number of graphs to generate
        out_dir: Optional output directory (if None, don't save)
        verbose: Print progress

    Returns:
        List of HeteroData objects
    """
    graphs = []

    for i in range(n_graphs):
        data, stats = generate_one(cfg, graph_id=i)

        if cfg.device == "cuda":
            data = data.cpu()

        graphs.append(data)

        if verbose and (i + 1) % 10 == 0:
            print(f"Generated {i+1}/{n_graphs} graphs")

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / f"graph_{i:06d}.pt"
            save_heterodata(data, str(save_path))

    return graphs
