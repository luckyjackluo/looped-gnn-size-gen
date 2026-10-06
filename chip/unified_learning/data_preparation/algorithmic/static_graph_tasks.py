"""Static CLRS-style graph task generation and loading utilities."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable, List, Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm


ALGORITHMIC_TASK_NAMES = frozenset(
    {
        "sssp",
        "bfs_reachability",
        "articulation_points",
    }
)

# Graph models that carry meaningful spatial coordinates directly (no need for
# a spring-layout post-hoc embedding). For these models the node positions
# produced by the generator are used as-is; for all other models we fall back
# to nx.spring_layout.
_SPATIAL_GRAPH_MODELS = frozenset({"random_geometric_graph", "rgg_s1"})


def _build_graph(
    *,
    num_nodes: int,
    graph_model: str,
    edge_prob: float,
    ba_num_edges: int,
    ws_k: int,
    ws_rewire_prob: float,
    rgg_target_avg_deg: float,
    seed: int,
) -> nx.Graph:
    if num_nodes <= 1:
        return nx.empty_graph(num_nodes)

    rng = random.Random(seed)
    graph_model = graph_model.lower()
    if graph_model == "erdos_renyi":
        graph = nx.gnp_random_graph(num_nodes, edge_prob, seed=seed)
    elif graph_model == "barabasi_albert":
        m = max(1, min(ba_num_edges, num_nodes - 1))
        graph = nx.barabasi_albert_graph(num_nodes, m, seed=seed)
    elif graph_model == "watts_strogatz":
        k = max(2, min(ws_k if ws_k % 2 == 0 else ws_k + 1, num_nodes - 1))
        if k % 2 == 1:
            k -= 1
        k = max(2, k)
        graph = nx.watts_strogatz_graph(num_nodes, k, ws_rewire_prob, seed=seed)
    elif graph_model == "path":
        graph = nx.path_graph(num_nodes)
    elif graph_model in ("random_geometric_graph", "rgg_s1"):
        if graph_model == "random_geometric_graph":
            # S2 graph: fixed [0,1]² bounding box, r ∝ 1/√N.
            # avg_deg ≈ rgg_target_avg_deg at every N.
            # Edge weights shrink as N grows — denser mesh, same physical space.
            r = math.sqrt(rgg_target_avg_deg / (math.pi * max(num_nodes, 1)))
            graph = nx.random_geometric_graph(num_nodes, r, seed=seed)
        else:
            # S1 graph: physical world scales as [0, √(N/N_ref)]², constant r.
            # avg_deg ≈ rgg_target_avg_deg at every N (same as S2).
            # Edge weights stay CONSTANT at all N — local structure is identical.
            # We generate in [0,1]² with r_eff = r_ref / √(N/N_ref) so that
            # after rescaling positions to the larger world, each edge has
            # length ≈ r_ref = sqrt(avg_deg / (π * N_ref)) = constant.
            N_ref = 32.0
            r_ref = math.sqrt(rgg_target_avg_deg / (math.pi * N_ref))
            scale = math.sqrt(max(num_nodes, 1) / N_ref)   # [0,1]² → [0, scale]²
            r_eff = r_ref / scale                           # effective r in [0,1]²
            graph = nx.random_geometric_graph(num_nodes, r_eff, seed=seed)
            # Rescale positions to [0, scale]² so edge weights equal true distances
            for node in graph.nodes():
                x, y = graph.nodes[node]["pos"]
                graph.nodes[node]["pos"] = (x * scale, y * scale)

        # Ensure connectivity by bridging isolated components with shortest edge.
        while not nx.is_connected(graph):
            components = sorted(nx.connected_components(graph), key=len, reverse=True)
            pos_dict = nx.get_node_attributes(graph, "pos")
            # Find the closest inter-component node pair by Euclidean distance.
            best_u, best_v, best_dist = None, None, float("inf")
            main_comp = components[0]
            for other_comp in components[1:]:
                for u in main_comp:
                    pu = np.array(pos_dict[u])
                    for v in other_comp:
                        pv = np.array(pos_dict[v])
                        d = float(np.linalg.norm(pu - pv))
                        if d < best_dist:
                            best_dist, best_u, best_v = d, u, v
                main_comp = main_comp | other_comp
            graph.add_edge(best_u, best_v)
    else:
        raise ValueError(f"Unsupported graph_model '{graph_model}'")

    if graph.number_of_edges() == 0 and num_nodes >= 2:
        graph.add_edge(rng.randrange(num_nodes), rng.randrange(num_nodes))
    return graph


def _node_layout(graph: nx.Graph, graph_model: str, seed: int) -> torch.Tensor:
    """Return node positions.  For spatial models use the embedded coordinates
    directly; for all others fall back to spring layout.

    For ``rgg_s1`` the raw positions live in [0, √(N/N_ref)]²; we normalise them
    to [0, 1]² so that the spatial node features are on the same scale as all
    other graph models.  The un-normalised coordinates are used only for computing
    edge weights and ``geo_dist_from_source`` (see ``build_algorithmic_graph``).
    """
    n = graph.number_of_nodes()
    if n == 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    if n == 1:
        return torch.zeros((1, 2), dtype=torch.float32)

    if graph_model in _SPATIAL_GRAPH_MODELS:
        pos_dict = nx.get_node_attributes(graph, "pos")
        coords = np.asarray(
            [[pos_dict[i][0], pos_dict[i][1]] for i in range(n)], dtype=np.float32
        )
        if graph_model == "rgg_s1":
            # Normalise [0, scale]² → [0, 1]² for consistent node features.
            max_val = coords.max()
            if max_val > 0:
                coords = coords / max_val
        return torch.from_numpy(coords)

    layout = nx.spring_layout(graph, seed=seed, dim=2)
    coords = torch.from_numpy(
        np.asarray([layout[i] for i in range(n)], dtype=np.float32)
    )
    return coords


def _edge_index_and_attr(graph: nx.Graph, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if graph.number_of_edges() == 0:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 1), dtype=torch.float32),
        )

    edges = []
    edge_attr = []
    for u, v in graph.edges():
        diff = pos[v] - pos[u]
        dist = float(torch.linalg.norm(diff).item())
        weight = max(dist, 1e-3)
        edges.append((u, v))
        edges.append((v, u))
        edge_attr.append([weight])
        edge_attr.append([weight])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    return edge_index, edge_attr_tensor


def _base_features(
    graph: nx.Graph,
    pos: torch.Tensor,
    *,
    source_idx: Optional[int],
) -> torch.Tensor:
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        return torch.zeros((0, 5), dtype=torch.float32)

    degrees = torch.tensor([graph.degree(i) for i in range(num_nodes)], dtype=torch.float32)
    max_degree = float(max(degrees.max().item(), 1.0))
    degree_norm = (degrees / max_degree).unsqueeze(-1)

    clustering = torch.tensor(
        [nx.clustering(graph, i) for i in range(num_nodes)],
        dtype=torch.float32,
    ).unsqueeze(-1)
    source_flag = torch.zeros((num_nodes, 1), dtype=torch.float32)
    if source_idx is not None:
        source_flag[source_idx, 0] = 1.0

    num_nodes_feat = torch.full(
        (num_nodes, 1),
        math.log1p(num_nodes) / 9.21,
        dtype=torch.float32,
    )
    return torch.cat([pos, degree_norm, clustering, source_flag, num_nodes_feat], dim=1)


def _annotate_sssp(
    graph: nx.Graph, *, source_idx: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (y_node [N,1], target_mask [N], hop_dist_from_source [N]).

    ``hop_dist_from_source`` holds the unweighted BFS hop count from the
    source to each reachable node, and -1 for unreachable nodes.  It is
    stored on graph data objects so the evaluation harness can restrict
    the target mask to nodes within ``h`` hops for the S1 ablation.
    """
    distances = nx.single_source_dijkstra_path_length(graph, source_idx, weight="weight")
    hop_lengths = nx.single_source_shortest_path_length(graph, source_idx)
    n = graph.number_of_nodes()
    y = torch.zeros((n, 1), dtype=torch.float32)
    mask = torch.zeros(n, dtype=torch.bool)
    hop_dist = torch.full((n,), fill_value=-1, dtype=torch.long)
    for node_idx, dist in distances.items():
        y[node_idx, 0] = float(dist)
        mask[node_idx] = True
    for node_idx, hops in hop_lengths.items():
        hop_dist[node_idx] = int(hops)
    return y, mask, hop_dist


def _annotate_bfs_reachability(graph: nx.Graph, *, source_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    reachable = nx.single_source_shortest_path_length(graph, source_idx)
    y = torch.zeros(graph.number_of_nodes(), dtype=torch.long)
    mask = torch.ones(graph.number_of_nodes(), dtype=torch.bool)
    for node_idx in reachable:
        y[node_idx] = 1
    return y, mask


def _annotate_articulation_points(graph: nx.Graph) -> tuple[torch.Tensor, torch.Tensor]:
    points = set(nx.articulation_points(graph))
    y = torch.tensor(
        [1 if idx in points else 0 for idx in range(graph.number_of_nodes())],
        dtype=torch.long,
    )
    mask = torch.ones(graph.number_of_nodes(), dtype=torch.bool)
    return y, mask


def build_algorithmic_graph(
    *,
    task_name: str,
    num_nodes: int,
    graph_model: str = "erdos_renyi",
    edge_prob: float = 0.15,
    ba_num_edges: int = 2,
    ws_k: int = 4,
    ws_rewire_prob: float = 0.2,
    rgg_target_avg_deg: float = 6.0,
    seed: int = 0,
) -> Data:
    """Build one static algorithmic graph with CLRS-style supervision.

    For ``graph_model='random_geometric_graph'`` two extra fields are stored
    on the returned ``Data`` object to enable the S1 / S2 evaluation ablation:

    - ``hop_dist_from_source`` [N, long]: unweighted BFS hop count from the
      source node.  -1 for unreachable nodes.  Used to restrict evaluation
      to nodes within h hops (S1 condition: fixed local neighbourhood).

    - ``geo_dist_from_source`` [N, float32]: Euclidean distance from the
      source's position in the [0, 1]² embedding to every other node.
      Used to restrict evaluation to nodes within a fixed geometric radius.
    """
    if task_name not in ALGORITHMIC_TASK_NAMES:
        raise ValueError(f"Unknown task_name '{task_name}'")

    graph = _build_graph(
        num_nodes=num_nodes,
        graph_model=graph_model,
        edge_prob=edge_prob,
        ba_num_edges=ba_num_edges,
        ws_k=ws_k,
        ws_rewire_prob=ws_rewire_prob,
        rgg_target_avg_deg=rgg_target_avg_deg,
        seed=seed,
    )
    # pos used for node features — normalised to [0,1]² for rgg_s1.
    pos = _node_layout(graph, graph_model=graph_model, seed=seed)
    source_idx = seed % max(num_nodes, 1) if task_name in {"sssp", "bfs_reachability"} else None
    x = _base_features(graph, pos, source_idx=source_idx)

    # Edge weights use raw (un-normalised) spatial coordinates so that rgg_s1
    # produces constant edge weights regardless of N.
    if graph_model == "rgg_s1":
        n = graph.number_of_nodes()
        pos_dict = nx.get_node_attributes(graph, "pos")
        raw_coords = np.asarray(
            [[pos_dict[i][0], pos_dict[i][1]] for i in range(n)], dtype=np.float32
        )
        pos_for_edges = torch.from_numpy(raw_coords)
    else:
        pos_for_edges = pos
    edge_index, edge_attr = _edge_index_and_attr(graph, pos_for_edges)

    hop_dist: Optional[torch.Tensor] = None
    if task_name == "sssp":
        assert source_idx is not None
        y_node, target_mask, hop_dist = _annotate_sssp(graph, source_idx=source_idx)
        supervision_type = "regression"
    elif task_name == "bfs_reachability":
        assert source_idx is not None
        y_node, target_mask = _annotate_bfs_reachability(graph, source_idx=source_idx)
        supervision_type = "classification"
    else:
        y_node, target_mask = _annotate_articulation_points(graph)
        supervision_type = "classification"

    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_node=y_node,
        target_mask=target_mask,
        task_name=task_name,
        supervision_level="node",
        supervision_type=supervision_type,
        graph_num_nodes=torch.tensor([graph.number_of_nodes()], dtype=torch.long),
        graph_num_edges=torch.tensor([graph.number_of_edges()], dtype=torch.long),
    )
    if source_idx is not None:
        data.source_idx = torch.tensor([source_idx], dtype=torch.long)

    if hop_dist is not None:
        data.hop_dist_from_source = hop_dist

    # For spatial graphs, store Euclidean distance from source.
    # rgg_s1: use raw (physical) coordinates so geo_eval_radius is meaningful
    #   in constant physical units — the correct quantity for the strict S1 eval.
    # rgg_s2: use the [0,1]² normalised coordinates.
    if graph_model in _SPATIAL_GRAPH_MODELS and source_idx is not None:
        pos_geo = pos_for_edges.numpy() if graph_model == "rgg_s1" else pos.numpy()
        src_pos = pos_geo[source_idx]
        geo_dist = torch.from_numpy(
            np.linalg.norm(pos_geo - src_pos, axis=1).astype(np.float32)
        )
        data.geo_dist_from_source = geo_dist

    return data


def generate_algorithmic_dataset(
    *,
    task_name: str,
    num_graphs: int,
    min_nodes: int,
    max_nodes: int,
    graph_model: str = "erdos_renyi",
    edge_prob: float = 0.15,
    ba_num_edges: int = 2,
    ws_k: int = 4,
    ws_rewire_prob: float = 0.2,
    rgg_target_avg_deg: float = 6.0,
    seed: int = 0,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
) -> List[Data]:
    """Generate a list of static algorithmic graphs."""
    if min_nodes > max_nodes:
        raise ValueError("min_nodes must be <= max_nodes")
    rng = random.Random(seed)
    graphs = []
    desc = progress_desc or f"gen[{task_name} N={min_nodes}-{max_nodes}]"
    iterator = range(num_graphs)
    if show_progress:
        iterator = tqdm(iterator, total=num_graphs, desc=desc, dynamic_ncols=True)
    for idx in iterator:
        num_nodes = rng.randint(min_nodes, max_nodes)
        graphs.append(
            build_algorithmic_graph(
                task_name=task_name,
                num_nodes=num_nodes,
                graph_model=graph_model,
                edge_prob=edge_prob,
                ba_num_edges=ba_num_edges,
                ws_k=ws_k,
                ws_rewire_prob=ws_rewire_prob,
                rgg_target_avg_deg=rgg_target_avg_deg,
                seed=seed + idx,
            )
        )
    return graphs


def generate_and_save_algorithmic_dataset(
    *,
    task_name: str,
    num_graphs: int,
    min_nodes: int,
    max_nodes: int,
    output_dir: str | Path,
    shard_prefix: str,
    graphs_per_file: int = 256,
    graph_model: str = "erdos_renyi",
    edge_prob: float = 0.15,
    ba_num_edges: int = 2,
    ws_k: int = 4,
    ws_rewire_prob: float = 0.2,
    rgg_target_avg_deg: float = 6.0,
    seed: int = 0,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
) -> List[Path]:
    """Generate algorithmic graphs and stream them to ``.pt`` shards as soon as
    each shard fills up. Surfaces a single tqdm bar over all ``num_graphs``.

    Returns the list of written shard paths.
    """
    if min_nodes > max_nodes:
        raise ValueError("min_nodes must be <= max_nodes")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    desc = progress_desc or f"gen[{task_name} N={min_nodes}-{max_nodes}]"
    iterator = range(num_graphs)
    if show_progress:
        iterator = tqdm(iterator, total=num_graphs, desc=desc, dynamic_ncols=True)

    written: List[Path] = []
    buffer: List[Data] = []
    shard_idx = 0
    for idx in iterator:
        num_nodes = rng.randint(min_nodes, max_nodes)
        buffer.append(
            build_algorithmic_graph(
                task_name=task_name,
                num_nodes=num_nodes,
                graph_model=graph_model,
                edge_prob=edge_prob,
                ba_num_edges=ba_num_edges,
                ws_k=ws_k,
                ws_rewire_prob=ws_rewire_prob,
                rgg_target_avg_deg=rgg_target_avg_deg,
                seed=seed + idx,
            )
        )
        if len(buffer) >= graphs_per_file:
            shard_path = output_path / f"{shard_prefix}_batch{shard_idx:03d}.pt"
            torch.save(buffer, shard_path)
            written.append(shard_path)
            buffer = []
            shard_idx += 1

    if buffer:
        shard_path = output_path / f"{shard_prefix}_batch{shard_idx:03d}.pt"
        torch.save(buffer, shard_path)
        written.append(shard_path)

    return written


def save_dataset_shards(
    graphs: Iterable[Data],
    output_dir: str | Path,
    *,
    shard_prefix: str,
    graphs_per_file: int = 256,
) -> List[Path]:
    """Save graphs into `.pt` shards and return written paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    graphs = list(graphs)
    written = []
    for shard_idx in range(0, len(graphs), graphs_per_file):
        shard_graphs = graphs[shard_idx: shard_idx + graphs_per_file]
        shard_name = f"{shard_prefix}_batch{shard_idx // graphs_per_file:03d}.pt"
        shard_path = output_path / shard_name
        torch.save(shard_graphs, shard_path)
        written.append(shard_path)
    return written


def load_algorithmic_dataset(
    *,
    data_dir: str | Path,
    specific_files: Optional[Iterable[str]] = None,
    max_samples: Optional[int] = None,
) -> List[Data]:
    """Load algorithmic graphs from `.pt` shards in a directory or explicit list."""
    data_dir = Path(data_dir)
    if specific_files:
        files = [data_dir / f if not Path(f).is_absolute() else Path(f) for f in specific_files]
    else:
        files = sorted(data_dir.glob("*.pt"))

    graphs: List[Data] = []
    for file_path in files:
        loaded = torch.load(file_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, list):
            graphs.extend(loaded)
        else:
            graphs.append(loaded)
        if max_samples is not None and len(graphs) >= max_samples:
            return graphs[:max_samples]
    return graphs[:max_samples] if max_samples is not None else graphs
