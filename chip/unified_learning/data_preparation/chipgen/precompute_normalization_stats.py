"""Precompute dataset-level normalization statistics for ChipGen diffusion.

This ensures all samples use the same normalization, which is critical for stable training.
Statistics computed:
- Size statistics: median and MAD of log(width), log(height) across entire dataset
- Edge offset statistics: mean and std of edge pin offsets
"""

import pickle
from pathlib import Path
from typing import List, Dict, Union
import torch
from tqdm import tqdm


def compute_dataset_normalization_stats(
    data_dirs: List[Union[str, Path]],
    pickle_files: List[str],
    output_path: Union[str, Path],
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Compute dataset-level normalization statistics across all training data.
    
    Args:
        data_dirs: List of directories containing pickle files
        pickle_files: List of pickle file names to process
        output_path: Path to save statistics
        device: Device for computation
        
    Returns:
        Dictionary with normalization statistics
    """
    print("Computing dataset-level normalization statistics...")
    
    all_log_widths = []
    all_log_heights = []
    all_edge_offsets = []
    all_chip_sizes = []
    
    total_nodes = 0
    total_edges = 0
    total_graphs = 0
    
    # Collect statistics from all files
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        for pickle_file in pickle_files:
            pickle_path = data_dir / pickle_file
            if not pickle_path.exists():
                print(f"Warning: {pickle_path} not found, skipping...")
                continue
                
            print(f"Processing {pickle_path}...")
            with open(pickle_path, 'rb') as f:
                data_list = pickle.load(f)
            
            for x, cond in tqdm(data_list, desc=f"Processing {pickle_file}"):
                # Extract sizes
                if hasattr(cond, 'sizes') and cond.sizes is not None:
                    sizes = cond.sizes
                elif hasattr(cond, 'x') and cond.x is not None and cond.x.shape[1] >= 2:
                    sizes = cond.x[:, :2]
                else:
                    print(f"Warning: No size information found in sample, skipping...")
                    continue
                
                widths = sizes[:, 0]
                heights = sizes[:, 1]
                
                # Collect log sizes
                log_widths = torch.log(torch.clamp(widths, min=1e-6))
                log_heights = torch.log(torch.clamp(heights, min=1e-6))
                all_log_widths.append(log_widths)
                all_log_heights.append(log_heights)
                
                total_nodes += widths.shape[0]
                
                # Collect edge attributes (pin offsets)
                if hasattr(cond, 'edge_attr') and cond.edge_attr is not None:
                    # Edge attr: [src_pin_x, src_pin_y, dst_pin_x, dst_pin_y]
                    edge_attr = cond.edge_attr
                    all_edge_offsets.append(edge_attr.flatten())
                    total_edges += edge_attr.shape[0]
                
                # Collect chip sizes
                if hasattr(cond, 'chip_size') and cond.chip_size is not None:
                    all_chip_sizes.append(cond.chip_size)
                
                total_graphs += 1
    
    if len(all_log_widths) == 0:
        raise ValueError("No valid data found! Check your data paths.")
    
    # Concatenate all statistics
    print(f"\nCollected statistics from {total_graphs} graphs, {total_nodes} nodes, {total_edges} edges")
    
    all_log_widths = torch.cat(all_log_widths, dim=0).to(device)
    all_log_heights = torch.cat(all_log_heights, dim=0).to(device)
    
    # Compute size statistics (robust median and MAD)
    print("Computing size statistics...")
    m_w = torch.median(all_log_widths)
    m_h = torch.median(all_log_heights)
    
    # MAD (Median Absolute Deviation) - more robust than std
    mad_w = torch.median(torch.abs(all_log_widths - m_w))
    mad_h = torch.median(torch.abs(all_log_heights - m_h))
    
    # Fallback to std if MAD is too small, clamp to reasonable minimum
    s_w = torch.clamp(mad_w, min=0.1) if mad_w > 1e-3 else torch.clamp(all_log_widths.std(), min=0.1)
    s_h = torch.clamp(mad_h, min=0.1) if mad_h > 1e-3 else torch.clamp(all_log_heights.std(), min=0.1)
    
    print(f"  Width:  median(log(w)) = {m_w:.4f}, MAD = {s_w:.4f}")
    print(f"  Height: median(log(h)) = {m_h:.4f}, MAD = {s_h:.4f}")
    
    # Compute edge offset statistics
    stats = {
        'size_log_median_w': m_w.cpu(),
        'size_log_mad_w': s_w.cpu(),
        'size_log_median_h': m_h.cpu(),
        'size_log_mad_h': s_h.cpu(),
        'total_graphs': total_graphs,
        'total_nodes': total_nodes,
        'total_edges': total_edges,
    }
    
    if len(all_edge_offsets) > 0:
        print("Computing edge offset statistics...")
        all_edge_offsets = torch.cat(all_edge_offsets, dim=0).to(device)
        
        # Use median and MAD for robustness (outliers are common in pin offsets)
        edge_median = torch.median(all_edge_offsets)
        edge_mad = torch.median(torch.abs(all_edge_offsets - edge_median))
        edge_std = all_edge_offsets.std()
        
        # Use MAD if significant, otherwise std
        edge_scale = torch.clamp(edge_mad, min=0.01) if edge_mad > 1e-3 else torch.clamp(edge_std, min=0.01)
        
        print(f"  Edge offsets: median = {edge_median:.4f}, MAD = {edge_scale:.4f}, std = {edge_std:.4f}")
        
        stats['edge_offset_median'] = edge_median.cpu()
        stats['edge_offset_scale'] = edge_scale.cpu()
    
    if len(all_chip_sizes) > 0:
        all_chip_sizes = torch.stack(all_chip_sizes, dim=0).to(device)
        chip_size_median = torch.median(all_chip_sizes, dim=0).values
        print(f"  Chip size median: {chip_size_median.tolist()}")
        stats['chip_size_median'] = chip_size_median.cpu()
    
    # Save statistics
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, output_path)
    print(f"\n✅ Statistics saved to: {output_path}")
    
    return stats


if __name__ == "__main__":
    # Example usage for test_v5 dataset
    import sys
    
    # Default paths
    data_dirs = [
        Path("data/chipgen/raw/test_v5"),
    ]
    
    train_files = [
        "v5_1000_5000.pickle",
        "v5_2000_2500.pickle",
        "v5_5000_1000.pickle",
    ]
    
    output_path = Path("data/chipgen/raw/test_v5/normalization_stats.pt")
    
    stats = compute_dataset_normalization_stats(
        data_dirs=data_dirs,
        pickle_files=train_files,
        output_path=output_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("\n" + "="*60)
    print("Dataset Statistics Summary:")
    print("="*60)
    for key, value in stats.items():
        if isinstance(value, torch.Tensor):
            print(f"{key:30s}: {value.item() if value.numel() == 1 else value.tolist()}")
        else:
            print(f"{key:30s}: {value}")
