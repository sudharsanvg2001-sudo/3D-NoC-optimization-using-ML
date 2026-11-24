"""
data_utils.py  (fixed)
──────────────────────
Build PyTorch-Geometric Data objects for the two input graphs.

Fixes applied vs previous version:
  • load_traffic_file()  — full file loading pipeline matching the previous
                           code's replace_inf_in_txt + txt_to_pt + cr_trDic
  • create_mesh()        — matches previous code's create_mesh, builds nw_dict
                           that Env_tsp expects
  • make_env() moved to train.py now correctly passes nw_dict + traff_dict
  • build_core_graph()   — unchanged, but now also called from load_traffic_file
  • build_noc_topology() — unchanged, uses id2coor from env.py
  • mapping_to_tensor()  — unchanged
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple

from env import id2coor, coor2id


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1 — File loading pipeline
# Replaces: replace_inf_in_txt + txt_to_pt + cr_trDic from previous code
# ─────────────────────────────────────────────────────────────────────────────

def load_traffic_file(
    txt_file : str,
    inf_value: float = 1_000_000.0,
) -> Tuple[Data, Dict, float]:
    """
    Full file loading pipeline — matches the previous code exactly:

        replace_inf_in_txt  →  txt_to_pt  →  cr_trDic

    Reads a raw traffic .txt file (adjacency matrix with INF for no-link),
    replaces INF with inf_value, builds a PyG Data object for the core graph,
    and builds the traff_dict expected by Env_tsp.

    Args:
        txt_file  : path to the traffic .txt file
        inf_value : replacement value for INF strings (default 1_000_000.0)

    Returns:
        core_data  : PyG Data — IP core graph
                     .x          [N, N]   comm profile node features
                     .edge_index [2, E]   non-INF directed edges
                     .edge_attr  [E, 1]   communication volumes
        traff_dict : dict — traffic links in Env_tsp format
                     e.g. {'l0t1': {'val': 0.42, 'src': 0, 'dst': 1}, ...}
        max_val    : float — max communication volume (used for normalisation)
    """
    # ── Step 1: read file and replace INF ────────────────────────────────
    # Matches replace_inf_in_txt from previous code, but in memory (no disk write)
    with open(txt_file, 'r') as f:
        content = f.read().replace('INF', str(inf_value))

    # ── Step 2: parse adjacency matrix ───────────────────────────────────
    # Matches txt_to_pt matrix parsing
    matrix = []
    for line in content.strip().split('\n'):
        row = line.strip().split()
        if row:
            matrix.append([float(x) for x in row])

    row_lengths = set(len(r) for r in matrix)
    if len(row_lengths) > 1:
        raise ValueError(
            f"Inconsistent row lengths in {txt_file}: {row_lengths}"
        )

    adj = torch.tensor(matrix, dtype=torch.float32)   # [N, N]
    N   = adj.size(0)

    # ── Step 3: build PyG Data — matches txt_to_pt logic ─────────────────
    # Edges: all (i,j) where value != inf_value  (i.e. real communication)
    edge_index = (adj != inf_value).nonzero(as_tuple=False).t().contiguous()
    edge_attr  = adj[edge_index[0], edge_index[1]].view(-1, 1)

    # Node features: full communication profile row (same as build_core_graph)
    x = adj.float().clone()   # [N, N]

    # Guard against empty graphs
    if edge_index.size(1) == 0:
        idx        = torch.arange(N)
        edge_index = torch.stack([idx, idx], dim=0)
        edge_attr  = torch.zeros(N, 1)

    core_data = Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=N
    )

    # ── Step 4: build traff_dict — matches cr_trDic logic ─────────────────
    # max_val computed from non-INF, non-zero, non-diagonal values
    real_vals = [
        matrix[i][j]
        for i in range(N)
        for j in range(N)
        if i != j and matrix[i][j] > 0 and matrix[i][j] < inf_value
    ]
    max_val = max(real_vals) if real_vals else 1.0

    traff_dict: Dict = {}
    for i in range(N):
        for j in range(N):
            val = matrix[i][j]
            # Matches cr_trDic: skip diagonal, skip zero, skip INF
            if i != j and val > 0 and val < inf_value:
                traff_dict[f'l{i}t{j}'] = {
                    'val': val / max_val,
                    'src': i,
                    'dst': j,
                }

    return core_data, traff_dict, max_val


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2 — create_mesh
# Matches create_mesh from previous code exactly — builds nw_dict for Env_tsp
# ─────────────────────────────────────────────────────────────────────────────

def create_mesh(dimX: int, dimY: int, dimZ: int) -> Dict:
    """
    Create a mesh network dictionary for Env_tsp.

    Matches previous code's create_mesh() exactly:
        nw_dict = {'0': {'cor': (x,y,z)}, '1': {'cor': (x,y,z)}, ...}

    Router ID ordering:
        id = z*(dimX*dimY) + y*dimX + x   (consistent with env.py id2coor)

    Args:
        dimX, dimY, dimZ : mesh dimensions

    Returns:
        nw_dict : dict keyed by string router ID
    """
    nw_dict = {}
    idx = 0
    for z in range(dimZ):
        for y in range(dimY):
            for x in range(dimX):
                nw_dict[str(idx)] = {'cor': (x, y, z)}
                idx += 1
    return nw_dict


# ─────────────────────────────────────────────────────────────────────────────
# IP Core Communication Graph  (from comm_matrix tensor directly)
# Used when comm_matrix is already available as a tensor (e.g. random instances)
# For real data from .txt files use load_traffic_file() above
# ─────────────────────────────────────────────────────────────────────────────

def build_core_graph(comm_matrix: torch.Tensor) -> Data:
    """
    Build an IP core graph from a communication-volume matrix tensor.

    Args:
        comm_matrix : [N, N] float tensor
                      comm_matrix[i, j] = data volume from core i to core j

    Returns:
        PyG Data with:
            .x          [N, N]   each core's full communication row profile
            .edge_index [2, E]   directed edges for non-zero communication
            .edge_attr  [E, 1]   communication volumes
    """
    N = comm_matrix.size(0)
    x = comm_matrix.float().clone()

    row_idx, col_idx = (comm_matrix > 0).nonzero(as_tuple=True)
    edge_index = torch.stack([row_idx, col_idx], dim=0).long()
    edge_attr  = comm_matrix[row_idx, col_idx].unsqueeze(1).float()

    # Guard: GAT needs at least one edge — add self-loops if graph is empty
    if edge_index.size(1) == 0:
        idx        = torch.arange(N)
        edge_index = torch.stack([idx, idx], dim=0)
        edge_attr  = torch.zeros(N, 1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ─────────────────────────────────────────────────────────────────────────────
# 3D NoC Topology Graph
# ─────────────────────────────────────────────────────────────────────────────

def _compute_tsv_dist_per_router(
    coords : torch.Tensor,          # [R, 3] integer coords
    tsv_ids: Optional[torch.Tensor], # [K] TSV router IDs
    dimX: int, dimY: int, dimZ: int,
) -> torch.Tensor:
    """
    For each router, compute XY Manhattan distance to the nearest TSV.
    Returns [R] float tensor. If no TSVs, returns zeros.
    """
    R = coords.size(0)
    if tsv_ids is None or tsv_ids.numel() == 0:
        return torch.zeros(R, dtype=torch.float32)

    # TSV coordinates — only use z=0 layer XY position
    tsv_coords = id2coor(tsv_ids.long(), dimX, dimY, dimZ)   # [K, 3]
    tsv_xy     = tsv_coords[:, :2].float()                    # [K, 2]
    router_xy  = coords[:, :2].float()                        # [R, 2]

    # Pairwise XY Manhattan distance [R, K]
    dist = (router_xy.unsqueeze(1) - tsv_xy.unsqueeze(0)).abs().sum(dim=-1)
    # Nearest TSV per router
    return dist.min(dim=1).values   # [R]


def _compute_cross_layer_cost(
    src_coord  : torch.Tensor,   # [3] int  (x, y, z)
    dst_coord  : torch.Tensor,   # [3] int  (x, y, z)
    tsv_dist_all: torch.Tensor,  # [R] nearest-TSV dist per router
    src_id     : int,
) -> float:
    """
    Elevator-first cross-layer routing cost for a ±z edge.
    cost = dist(src→nearest_TSV) + |z1-z2| + dist(TSV→dst)
    Since TSV is at the same XY as src's nearest TSV:
        tsv_to_dst ≈ dist_to_nearest_tsv(dst)
    So: cost = tsv_dist(src) + |z1-z2| + tsv_dist(dst)
    This is an approximation — assumes same TSV is best for both ends.
    """
    src_tsv = float(tsv_dist_all[src_id].item())
    z_diff  = abs(int(src_coord[2].item()) - int(dst_coord[2].item()))
    return src_tsv + z_diff   # conservative — TSV→dst handled by dst embedding


def build_noc_topology(
    dimX   : int,
    dimY   : int,
    dimZ   : int   = 1,
    tsv_ids: Optional[torch.Tensor] = None,
    device : str   = 'cpu',
) -> Data:
    """
    Build a 3D mesh topology graph for a dimX × dimY × dimZ NoC.

    Router IDs follow env.py convention:
        id = z*(dimX*dimY) + y*dimX + x

    Node features (dim = 7)  — Fix 2: richer global node features:
        [norm_x, norm_y, norm_z, is_tsv,
         norm_tsv_dist,              nearest TSV XY distance (normalised)
         norm_mean_cross_layer_cost, avg elevator-first cost to other layers
         norm_min_cross_layer_cost]  min elevator-first cost to other layers

    Edge features (dim = 1)  — Fix 1: routing-cost-aware edges:
        Same-layer edges:   XY Manhattan distance (= 1 for adjacent)
        Cross-layer ±z edges: elevator-first cost via nearest TSV

    Args:
        dimX, dimY, dimZ : mesh dimensions
        tsv_ids          : 1-D long tensor of TSV router IDs (partial placement ok)
        device           : torch device string

    Returns:
        PyG Data with:
            .x            [R, 7]   enriched node features
            .edge_index   [2, E]   bidirectional mesh links
            .edge_attr    [E, 1]   routing cost per edge
            .positions    [R, 3]   integer (x, y, z)
            .tsv_ids      Tensor
            .dimX/Y/Z     ints
            .topo_feat_dim int     = 7  (so model can read it)
            .topo_edge_dim int     = 1
    """
    R = dimX * dimY * dimZ

    # ── Router IDs → integer (x, y, z) ───────────────────────────────────
    all_ids = torch.arange(R)
    coords  = id2coor(all_ids, dimX, dimY, dimZ)   # [R, 3] int  (on CPU)

    # ── Fix 2a: normalised position ───────────────────────────────────────
    norm_x = coords[:, 0].float() / max(dimX - 1, 1)
    norm_y = coords[:, 1].float() / max(dimY - 1, 1)
    norm_z = coords[:, 2].float() / max(dimZ - 1, 1)

    # ── Fix 2b: TSV binary flag ───────────────────────────────────────────
    is_tsv = torch.zeros(R, dtype=torch.float32)
    if tsv_ids is not None and tsv_ids.numel() > 0:
        valid        = tsv_ids.long()
        valid        = valid[(valid >= 0) & (valid < R)]
        is_tsv[valid] = 1.0

    # ── Fix 2c: distance to nearest TSV per router ────────────────────────
    tsv_dist = _compute_tsv_dist_per_router(coords, tsv_ids, dimX, dimY, dimZ)
    # normalise by max possible XY distance
    max_xy_dist    = float(dimX - 1 + dimY - 1) if (dimX > 1 or dimY > 1) else 1.0
    norm_tsv_dist  = tsv_dist / max_xy_dist      # [R]

    # ── Fix 2d: cross-layer cost summary per router ───────────────────────
    # For each router on layer z, compute elevator-first cost to all routers
    # on other layers. Summarise as mean and min.
    if dimZ > 1:
        # For each router: cost to reach any cross-layer router
        # = tsv_dist(self) + z_diff + tsv_dist(target)
        # We approximate mean/min over all cross-layer targets
        cross_costs = torch.zeros(R, dtype=torch.float32)
        cross_min   = torch.full((R,), float('inf'))
        for r in range(R):
            r_z = int(coords[r, 2].item())
            d_self = float(tsv_dist[r].item())
            total, count, mn = 0.0, 0, float('inf')
            for t in range(R):
                t_z = int(coords[t, 2].item())
                if t_z == r_z:
                    continue   # same layer — not cross-layer
                z_diff = abs(r_z - t_z)
                cost   = d_self + z_diff + float(tsv_dist[t].item())
                total += cost
                count += 1
                mn = min(mn, cost)
            cross_costs[r] = total / count if count > 0 else 0.0
            cross_min[r]   = mn if mn < float('inf') else 0.0

        max_cross = float(cross_costs.max().item()) or 1.0
        norm_mean_cross = cross_costs / max_cross
        norm_min_cross  = cross_min  / max_cross
    else:
        # Single layer — no cross-layer routing
        norm_mean_cross = torch.zeros(R, dtype=torch.float32)
        norm_min_cross  = torch.zeros(R, dtype=torch.float32)

    # ── Assemble node feature matrix [R, 7] ──────────────────────────────
    x = torch.stack([
        norm_x,           # position
        norm_y,
        norm_z,
        is_tsv,           # structural
        norm_tsv_dist,    # Fix 2c: how far to nearest TSV
        norm_mean_cross,  # Fix 2d: avg cross-layer routing cost
        norm_min_cross,   # Fix 2d: min cross-layer routing cost
    ], dim=1)             # [R, 7]

    # ── Fix 1: routing-cost-aware edge features ───────────────────────────
    src_list, dst_list, edge_cost_list = [], [], []
    for z in range(dimZ):
        for y in range(dimY):
            for xi in range(dimX):
                node = z * dimX * dimY + y * dimX + xi
                for dz, dy, dx in [
                    ( 0,  0,  1), ( 0,  0, -1),   # ±x
                    ( 0,  1,  0), ( 0, -1,  0),   # ±y
                    ( 1,  0,  0), (-1,  0,  0),   # ±z cross-layer
                ]:
                    nz, ny, nx = z + dz, y + dy, xi + dx
                    if 0 <= nx < dimX and 0 <= ny < dimY and 0 <= nz < dimZ:
                        nbr = nz * dimX * dimY + ny * dimX + nx
                        src_list.append(node)
                        dst_list.append(nbr)

                        if dz != 0:
                            # Fix 1: cross-layer edge — elevator-first cost
                            # = tsv_dist(src) + |z_diff|
                            # GAT will propagate this asymmetric cost
                            cost = float(tsv_dist[node].item()) + abs(dz)
                        else:
                            # Same-layer edge — unit Manhattan cost = 1
                            cost = 1.0
                        edge_cost_list.append(cost)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    # Normalise edge costs
    edge_costs_raw = torch.tensor(edge_cost_list, dtype=torch.float32)
    max_edge_cost  = float(edge_costs_raw.max().item()) or 1.0
    edge_attr      = (edge_costs_raw / max_edge_cost).unsqueeze(1)   # [E, 1]

    # ── Build Data object ─────────────────────────────────────────────────
    data                = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.positions      = coords.float()
    data.tsv_ids        = tsv_ids.long() if tsv_ids is not None else torch.tensor([], dtype=torch.long)
    data.dimX, data.dimY, data.dimZ = dimX, dimY, dimZ
    data.topo_feat_dim  = 7   # model reads this to set topo_feat_dim correctly
    data.topo_edge_dim  = 1   # model reads this to set edge_dim in GAT

    # Move to target device
    return data.to(device) if device != 'cpu' else data


# ─────────────────────────────────────────────────────────────────────────────
# Mapping format helper
# ─────────────────────────────────────────────────────────────────────────────

def mapping_to_tensor(
    mapping  : List[Tuple[int, int]],
    num_cores: int,
    device   : str = 'cpu',
) -> torch.Tensor:
    """
    Convert a list of (core_idx, router_idx) pairs to a dense tensor
    of shape [num_cores] where out[core_i] = router_i.

    This is the format expected by Env_tsp.test_cost2().
    """
    out = torch.zeros(num_cores, dtype=torch.long, device=device)
    for core_idx, router_idx in mapping:
        out[core_idx] = router_idx
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Random instance generators  (for training / testing without real data files)
# ─────────────────────────────────────────────────────────────────────────────

def load_tsv_placements(tsv_file: str) -> List[torch.Tensor]:
    """
    Load all TSV placements from a text file.
    Each line is one placement — space-separated router IDs.

    File format:
        7 9 14 12 23 25 30 28
        0 3 5 15 16 19 21 31
        2 6 8 13 18 22 24 29
        ...

    Args:
        tsv_file : path to TSV placements text file

    Returns:
        List of 1-D long tensors, one per placement
    """
    placements = []
    with open(tsv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue   # skip empty lines and comments
            ids = [int(x) for x in line.split()]
            placements.append(torch.tensor(ids, dtype=torch.long))
    print(f"✅ Loaded {len(placements)} TSV placements from {tsv_file}")
    return placements


def load_all_traffic_files(data_dir: str) -> List[Tuple[Data, torch.Tensor, Dict]]:
    """
    Load all .txt traffic files from a directory.
    Returns a list of (core_data, comm_matrix, traff_dict) tuples.

    Args:
        data_dir : directory containing .txt traffic files

    Returns:
        dataset : list of (core_data, comm_matrix, traff_dict)
    """
    import glob as _glob
    txt_files = sorted(_glob.glob(f"{data_dir}/*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    dataset = []
    for txt_file in txt_files:
        core_data, traff_dict, max_val = load_traffic_file(txt_file)
        comm_matrix = core_data.x   # [N, N] comm profile
        dataset.append((core_data, comm_matrix, traff_dict))
        

    print(f"✅ Dataset: {len(dataset)} traffic graphs loaded from {data_dir}")
    return dataset


def random_comm_matrix(
    num_cores  : int,
    density    : float = 0.4,
    max_volume : float = 10.0,
    symmetric  : bool  = True,
) -> torch.Tensor:
    """Random [num_cores, num_cores] communication-volume matrix."""
    comm = torch.rand(num_cores, num_cores) * max_volume
    mask = (torch.rand(num_cores, num_cores) < density).float()
    comm = comm * mask
    if symmetric:
        comm = (comm + comm.t()) / 2
    comm.fill_diagonal_(0)
    return comm


def generate_tsv_ids(
    dimX    : int,
    dimY    : int,
    dimZ    : int   = 1,
    density : float = 0.25,
    max_attempts: int = 10000,
    seed    : Optional[int] = None,
) -> torch.Tensor:
    """
    Generate TSV router IDs per layer satisfying three constraints:

        1. No horizontal neighbours  — TSV at (x,y) blocks (x-1,y) and (x+1,y)
        2. No vertical neighbours    — TSV at (x,y) blocks (x,y-1) and (x,y+1)
        3. Target density            — exactly 25% (or nearest feasible) of
                                       each layer's grid filled with TSVs

    TSVs are generated per z-layer independently and share the same
    placement pattern (same (x,y) positions replicated across all layers).
    Router IDs follow env.py convention:
        id = z*(dimX*dimY) + y*dimX + x

    Args:
        dimX, dimY   : grid dimensions per layer
        dimZ         : number of layers (TSV pattern replicated across all)
        density      : target fraction of grid cells to fill  (default 0.25)
        max_attempts : retry limit for random placement
        seed         : optional random seed for reproducibility

    Returns:
        tsv_ids : 1-D long tensor of router IDs across ALL layers

    Raises:
        RuntimeError : if no valid placement found within max_attempts
    """
    if seed is not None:
        torch.manual_seed(seed)

    target = int(round(dimX * dimY * density))
    target = max(1, target)   # at least one TSV

    # ── Build candidate positions ──────────────────────────────────────────
    # All (x, y) positions in one layer
    all_positions = [
        (x, y) for y in range(dimY) for x in range(dimX)
    ]

    def neighbours(x, y):
        """Return immediate horizontal and vertical neighbours."""
        nbrs = set()
        if x > 0:       nbrs.add((x - 1, y))
        if x < dimX-1:  nbrs.add((x + 1, y))
        if y > 0:       nbrs.add((x, y - 1))
        if y < dimY-1:  nbrs.add((x, y + 1))
        return nbrs

    # ── Greedy random placement with backtracking ─────────────────────────
    # We shuffle candidate positions and greedily place TSVs,
    # skipping positions blocked by neighbours.
    # Retry up to max_attempts times with different shuffles.

    best_placement = []   # best found so far (closest to target)

    for attempt in range(max_attempts):
        perm      = torch.randperm(len(all_positions)).tolist()
        shuffled  = [all_positions[i] for i in perm]
        placed    = []
        blocked   = set()

        for (x, y) in shuffled:
            if (x, y) in blocked:
                continue
            # Place TSV here
            placed.append((x, y))
            # Block all horizontal and vertical neighbours
            blocked.update(neighbours(x, y))
            # Stop once target is reached
            if len(placed) == target:
                break

        # Track best placement found
        if len(placed) > len(best_placement):
            best_placement = placed

        # Accept if target reached exactly
        if len(placed) == target:
            break
    else:
        if not best_placement:
            raise RuntimeError(
                f"Could not place any TSVs in {dimX}×{dimY} grid "
                f"after {max_attempts} attempts."
            )
        import warnings
        warnings.warn(
            f"Target density {density:.0%} ({target} TSVs) not achievable "
            f"with no-neighbour constraint in {dimX}×{dimY} grid. "
            f"Placed {len(best_placement)} TSVs instead "
            f"({len(best_placement)/(dimX*dimY):.1%} density).",
            UserWarning,
        )

    # ── Replicate pattern across all z layers ─────────────────────────────
    tsv_ids = []
    for z in range(dimZ):
        for (x, y) in best_placement:
            router_id = z * dimX * dimY + y * dimX + x
            tsv_ids.append(router_id)

    return torch.tensor(tsv_ids, dtype=torch.long)


def random_tsv_ids(
    dimX    : int,
    dimY    : int,
    num_tsvs: int,
) -> torch.Tensor:
    """
    Legacy helper — sample `num_tsvs` random router IDs from z=0 layer.
    For constrained placement use generate_tsv_ids() instead.
    """
    layer0_ids = torch.arange(dimX * dimY)
    perm       = torch.randperm(dimX * dimY)[:num_tsvs]
    return layer0_ids[perm]


def random_instance(
    num_cores: int,
    density  : float = 0.4,
) -> Tuple[Data, torch.Tensor]:
    """Quick (core_graph, comm_matrix) pair for smoke tests."""
    comm      = random_comm_matrix(num_cores, density)
    core_data = build_core_graph(comm)
    return core_data, comm