from cgi import test
import torch
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from operator import itemgetter
import operator
import random
from itertools import chain
from torch import tensor
from typing import List, Tuple, Optional, Dict

def dist_eval(ids1, ids2, dimX, dimY, dimZ):
    t1 = id2coor(ids1, dimX, dimY, dimZ)
    t2 = id2coor(ids2, dimX, dimY, dimZ)
    dist = abs(t1[:,0]-t2[:,0]) + abs(t1[:,1]-t2[:,1])
    return dist

def node_nebrs(id, dimX, dimY):
    id_crds = id2coor_batch(id, dimX, dimY, 2)
    nebrs = torch.zeros(4*id.shape[0], 3)
    i = 0
    nebrs[i:i+id.shape[0], 0] = id_crds[:,0] + 1
    nebrs[i:i+id.shape[0], 1] = id_crds[:,1]
    i += id.shape[0]
    nebrs[i:i+id.shape[0], 0] = id_crds[:,0] - 1
    nebrs[i:i+id.shape[0], 1] = id_crds[:,1]
    i += id.shape[0]
    nebrs[i:i+id.shape[0], 1] = id_crds[:,1] + 1
    nebrs[i:i+id.shape[0], 0] = id_crds[:,0]
    i += id.shape[0]
    nebrs[i:i+id.shape[0], 1] = id_crds[:,1] - 1
    nebrs[i:i+id.shape[0], 0] = id_crds[:,0]
    nebrs[nebrs == dimX] = dimX - 1
    nebrs[nebrs == -1] = 1
    return coor2id(nebrs.to('cuda'), dimX, dimY, 2)


# Inside env.py

def hop_cnt(TSV_id, src, dst, nw_dict, dimX, dimY, dimZ, tsv_assign, stage):
    device_used = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    r1 = id2coor(src, dimX, dimY, dimZ)
    r2 = id2coor(dst, dimX, dimY, dimZ)
    
    # Base same-layer cost (Manhattan in x,y)
    same_layer_cost = abs(r1[:, 0] - r2[:, 0]) + abs(r1[:, 1] - r2[:, 1])
    hop_dst = same_layer_cost.float()
    
    d = torch.where(r1[:, 2] != r2[:, 2])  # Identify cross-layer flows
    TSV_usage_dict = {'up': {}, 'down': {}, 'total': {}}
    used_tsv_id = None
    
    if d[0].shape[0] > 0:
        if stage == 3:  # Stage 3: Use the fixed symmetric TSV assignment
            for i in range(d[0].shape[0]):
                idx = d[0][i].item()
                src_coords = r1[idx]
                dst_coords = r2[idx]
                src_layer = src_coords[2].item()
                dst_layer = dst_coords[2].item()
                
                # In our symmetric case, we use the TSV assignment for the (x,y) coordinate.
                # Assume that tsv_assign is a 1D tensor where each entry corresponds to a router (by index)
                # and that routers with the same (x,y) across layers have the same tsv.
                # We can simply take the tsv assigned to the source router.
                # print(tsv_assign)
                tsv_id = tsv_assign[src[idx].item()].item()
                tsv_id = tsv_assign[src[idx].item()].item()
                used_tsv_id = tsv_id
                tsv_coords = id2coor(torch.tensor([tsv_id], device=device_used), dimX, dimY, dimZ).squeeze(0)
                
                # Calculate cost components:
                src_to_tsv = abs(src_coords[0] - tsv_coords[0]) + abs(src_coords[1] - tsv_coords[1])
                tsv_to_dst = abs(tsv_coords[0] - dst_coords[0]) + abs(tsv_coords[1] - dst_coords[1])
                z_diff = abs(src_coords[2] - dst_coords[2])
                
                total_cost = src_to_tsv + tsv_to_dst + z_diff
                hop_dst[idx] = total_cost
                
                direction = 'up' if src_layer < dst_layer else 'down'
                TSV_usage_dict[direction][tsv_id] = TSV_usage_dict[direction].get(tsv_id, 0.0) + total_cost
                TSV_usage_dict['total'][tsv_id] = TSV_usage_dict['total'].get(tsv_id, 0.0) + total_cost

        elif stage == 2:  # Stage 2: Nearest TSV selection
            tsv_candidates = TSV_id.squeeze(0) if TSV_id.dim() > 1 else TSV_id
            r1_d = r1[d]
            
            Vlink_dsts = torch.zeros(d[0].shape[0], len(tsv_candidates), device=device_used)
            for pos in range(len(tsv_candidates)):
                candidate_coords = id2coor(torch.tensor([tsv_candidates[pos]], device=device_used), dimX, dimY, dimZ).squeeze(0)
                Vlink_dsts[:, pos] = abs(r1_d[:, 0] - candidate_coords[0]) + abs(r1_d[:, 1] - candidate_coords[1])
            
            min_dists, min_indices = torch.min(Vlink_dsts, 1)
            best_tsvs = tsv_candidates[min_indices]
            tsv_coords = id2coor(best_tsvs, dimX, dimY, dimZ)
            
            tsv_to_dst = abs(tsv_coords[:, 0] - r2[d][:, 0]) + abs(tsv_coords[:, 1] - r2[d][:, 1])
            z_diff = abs(r1[d][:, 2] - r2[d][:, 2]).float()
            
            hop_dst[d] = min_dists + tsv_to_dst + z_diff
            
            for i, idx in enumerate(d[0]):
                tsv_used = best_tsvs[i].item()
                used_tsv_id = tsv_used  # record which TSV handled this cross-layer flow
                src_layer = r1[idx, 2].item()
                dst_layer = r2[idx, 2].item()
                direction = 'up' if src_layer < dst_layer else 'down'
                cost_val = hop_dst[idx].item()
                TSV_usage_dict[direction][tsv_used] = TSV_usage_dict[direction].get(tsv_used, 0) + cost_val
                TSV_usage_dict['total'][tsv_used] = TSV_usage_dict['total'].get(tsv_used, 0) + cost_val

        else: 
            manhattan_distance = abs(r1[d][:, 0] - r2[d][:, 0]) + abs(r1[d][:, 1] - r2[d][:, 1])
            z_diff = abs(r1[d][:, 2] - r2[d][:, 2]).float()
            hop_dst[d] = manhattan_distance.float() + z_diff
            for idx in d[0]:
                router_id = src[idx].item()
                cost_val = hop_dst[idx].item()
                TSV_usage_dict['total'][router_id] = TSV_usage_dict['total'].get(router_id, 0) + cost_val

    return hop_dst.unsqueeze(0), TSV_usage_dict, used_tsv_id

def TSV_loc(tsv_info, nw_dict, dimX, dimY):
    TSV_ids = {}
    for loc, val in enumerate(tsv_info[0]): 	
        if(val == 1):
            TSV_ids[str(loc)] = {'cor': nw_dict[str(loc)]['cor']}
            dst_vlk = loc + (dimX * dimY)
            TSV_ids[str(dst_vlk)] = {'cor': nw_dict[str(dst_vlk)]['cor']}
    return TSV_ids

def coor2id(tsv_cords, dimX, dimY, dimZ):
    ids = tsv_cords[:,2] * dimX * dimY + tsv_cords[:,1] * dimX + tsv_cords[:,0]
    return ids

def id2coor(id, dimX, dimY, dimZ):
    device_used = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert scalar tensors to 1D tensors
    if isinstance(id, torch.Tensor) and id.dim() == 0:
        id = id.unsqueeze(0)
    
    # Ensure id is a tensor and handle both scalars and arrays
    id_tensor = torch.tensor([id]) if not isinstance(id, torch.Tensor) else id
    id_tensor = id_tensor.view(-1)  # Flatten to 1D
    
    coord = torch.zeros(len(id_tensor), 3).int().to(device_used)
    coord[:,2] = id_tensor // (dimX*dimY)  # z coordinate
    remaining = id_tensor - coord[:,2] * dimX * dimY
    coord[:,1] = remaining // dimX
    coord[:,0] = remaining % dimX
    return coord
# Add to Env_tsp class
def visualize_mapping(self, mapping, tsv_assignments=None, stage_name=""):
    """Visualizes the 3D NoC mapping with TSV placements."""
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d')
    
    # Plot all routers
    all_routers = torch.arange(self.dimX*self.dimY*self.dimZ, device=self.device)
    router_coords = id2coor(all_routers, self.dimX, self.dimY, self.dimZ).cpu().numpy()
    ax.scatter(router_coords[:,0], router_coords[:,1], router_coords[:,2], c='blue', label='Routers')
    
    # Plot mapped cores
    unique_mapped_routers = torch.unique(mapping)
    mapped_coords = id2coor(unique_mapped_routers, self.dimX, self.dimY, self.dimZ).cpu().numpy()
    ax.scatter(mapped_coords[:,0], mapped_coords[:,1], mapped_coords[:,2], c='red', s=100, label='Mapped Cores')
    
    # Plot TSVs if provided
    if tsv_assignments is not None:
        tsv_coords = id2coor(tsv_assignments, self.dimX, self.dimY, self.dimZ).cpu().numpy()
        ax.scatter(tsv_coords[:,0], tsv_coords[:,1], tsv_coords[:,2], c='green', s=150, 
                   marker='X', label='TSVs')
        
        # Draw vertical connections
        for tsv in tsv_assignments:
            base_coord = id2coor(tsv, self.dimX, self.dimY, self.dimZ).cpu().numpy()
            for z in range(self.dimZ):
                ax.plot([base_coord[0]], [base_coord[1]], [z], 'g--', alpha=0.3)

    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D NoC Mapping - {stage_name}')
    plt.legend()
    
    # Save and show
    plt.savefig(f'plots/{stage_name}_mapping.png', dpi=300)
    plt.close()


def id2coor_batch(id, dimX, dimY, dimZ):
    device_used = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coord = torch.zeros(id.shape[0], 3).int().to(device_used)
    # If you need the z coordinate, compute it; otherwise, adjust accordingly.
    # Here we leave coord[:,2] as zero if not needed.
    coord[:,1] = (id - coord[:,2]*dimX*dimY) / dimX
    coord[:,0] = (id - coord[:,2]*dimX*dimY) % dimX
    return coord


#############################################
# UPDATED Env_tsp class with dynamic TSV variance computation
#############################################

class Env_tsp():
    def __init__(self, args, nw_dict: dict[str,dict], traff_dict: dict, data):
        self.args = args
        self.dimX, self.dimY, self.dimZ = int(args.dimX), int(args.dimY), int(args.dimZ)
        self.R = self.dimX * self.dimY * self.dimZ

        self.device = data.x.device if hasattr(data, "x") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nw_dict = nw_dict if nw_dict else self._make_mesh_dict()

        # Router coordinates as tensors indexed by router-id r in [0, R)
        rx, ry, rz = self._router_coords()
        self.rx, self.ry, self.rz = rx.to(self.device), ry.to(self.device), rz.to(self.device)

        # Demand matrix W (N x N) from (edge_index, edge_attr)
        self.W = self._build_demand_from_data(data).to(self.device)  # float32

        # cached for speed
        self._xy_cache = {}   # key: allowed_xy (Kx2) -> (cx, cy) tensors
        self._mesh_xy = self._all_xy_columns()  # (R,2) list of (x,y) per router

    # ----------------------- helpers -----------------------
    def _make_mesh_dict(self) -> dict[str,dict]:
        d = {}
        rid = 0
        for z in range(self.dimZ):
            for y in range(self.dimY):
                for x in range(self.dimX):
                    d[str(rid)] = {"cor": (x,y,z)}
                    rid += 1
        return d
    def _all_xy_columns(self) -> tensor:
        # (R,2): xy of each router-id (z ignored)
        return torch.stack([self.rx, self.ry], dim=1)
    def _build_demand_from_data(self, data) -> tensor:
        # build W (N x N) from edge_index/edge_attr; treat 1e6 as "no flow" => 0.0
        # If absent, build zeros.
        if getattr(data, "edge_index", None) is None or getattr(data, "edge_attr", None) is None:
            N = int(data.x.size(0))
            return torch.zeros(N, N, dtype=torch.float32)
        idx = data.edge_index  # [2, E]
        attr = data.edge_attr.view(-1)  # [E]
        N = int(max(int(attr.shape[0] > 0 and (idx.max().item() + 1)), data.x.size(0)))
        W = torch.zeros(N, N, dtype=torch.float32, device=attr.device)
        # INF guard
        valid = attr != 1_000_000.0
        if valid.any():
            src = idx[0, valid].long()
            dst = idx[1, valid].long()
            W.index_put_((src, dst), attr[valid].float())
        # zero diagonal (no self-traffic)
        W.fill_diagonal_(0.0)
        return W
    def _dist_fc3d_for_mapping(self, m: tensor) -> tensor:
        """
        Fully-connected 3D Manhattan distance matrix for a mapping 'm' (N,) of task->router-id.
        Returns D (N x N) where D[i,j] = |Δx| + |Δy| + |Δz| between router m[i] and m[j].
        """
        r = m.long()
        xi, yi, zi = self.rx[r], self.ry[r], self.rz[r]  # (N,)
        x_i, y_i, z_i = xi[:, None], yi[:, None], zi[:, None]
        x_j, y_j, z_j = xi[None, :], yi[None, :], zi[None, :]
        D = (x_i - x_j).abs() + (y_i - y_j).abs() + (z_i - z_j).abs()
        return D
    def _cx_cy_from_allowed_tsvs(self, allowed_tsvs: tensor) -> tuple[tensor, tensor, tensor]:
        """
        From allowed TSV router-ids (1D), compute unique XY columns and their z-stacks.
        Returns (cx, cy, unique_xy_index), each on self.device.
        """
        allowed_tsvs = allowed_tsvs.long().to(self.device)
        xy = torch.stack([self.rx[allowed_tsvs], self.ry[allowed_tsvs]], dim=1)  # (M,2)
        # unique xy columns
        xy_u, inv = torch.unique(xy, dim=0, return_inverse=True)  # (K,2), inv in [0..K)
        cx, cy = xy_u[:, 0], xy_u[:, 1]   # (K,)
        return cx, cy, inv

    def _dist_elevator_first(self, m: tensor, cx: tensor, cy: tensor) -> tuple[tensor, tensor]:
        """
        Elevator-first routed distance matrix for mapping m (N,):
        Dmin[i,j] = |Δz| + min_k(|xi-cx[k]|+|yi-cy[k]| + |xj-cx[k]|+|yj-cy[k]|)
        Returns (Dmin, Kstar) where Kstar[i,j] is argmin TSV column index.
        """
        r = m.long()
        xi, yi, zi = self.rx[r], self.ry[r], self.rz[r]  # (N,)
        x_i = xi[:, None, None]
        y_i = yi[:, None, None]
        x_j = xi[None, :, None]
        y_j = yi[None, :, None]
        zdiff = (zi[:, None] - zi[None, :]).abs()[..., None]  # (N,N,1)

        cx = cx[None, None, :]  # (1,1,K)
        cy = cy[None, None, :]
        M = (x_i - cx).abs() + (y_i - cy).abs() + (x_j - cx).abs() + (y_j - cy).abs() + zdiff  # (N,N,K)
        Dmin, Kstar = M.min(dim=2)  # (N,N), (N,N)
        return Dmin, Kstar
    def _router_coords(self) -> tuple[tensor,tensor,tensor]:
        # r -> (x,y,z) with row-major id
        r = torch.arange(self.R, dtype=torch.long)
        x = r % self.dimX
        y = (r // self.dimX) % self.dimY
        z = r // (self.dimX * self.dimY)
        return x, y, z

    # def test_cost2(
    #     self,
    #     mapping: tensor,
    #     allowed_tsvs: Optional[tensor],
    #     stage: int,
    #     tsv_assgn: Optional[tensor] = None
    # ) -> tuple[tensor, dict]:
    #     """
    #     Compute total communication cost for a given mapping and TSV set.
    #     Returns (cost, usage_dict).
    #     usage_dict['total'] is a flat dict keyed by router-id with per-router flow = Fi_col / Z.
    #     """
    #     m = mapping.view(-1).to(self.device)
    #     usage = {'total': {}, 'per_column': {}}

    #     if stage == 1 or allowed_tsvs is None or allowed_tsvs.numel() == 0:
    #         # Fully connected 3D
    #         D = self._dist_fc3d_for_mapping(m)                  # (N,N)
    #         cost = (self.W * D).sum()
    #         return cost, usage

    #     # Stage-2/3: via allowed TSV columns. If a specific assignment list is passed, restrict to those.
    #     allowed = allowed_tsvs.to(self.device).long()
    #     if tsv_assgn is not None and tsv_assgn.numel() > 0:
    #         allowed = tsv_assgn.to(self.device).long()

    #     cx, cy, _ = self._cx_cy_from_allowed_tsvs(allowed)         # (K,)
    #     Dmin, Kstar = self._dist_elevator_first(m, cx, cy)         # (N,N), (N,N)
    #     cost = (self.W * Dmin).sum()

    #     # TSV utilization Fi: sum of FLOWS routed vertically through each column
    #     zi = self.rz[m]
    #     vertical_pairs = (zi[:, None] != zi[None, :])              # (N,N) bool
    #     flow = self.W * vertical_pairs.float()
    #     K = cx.numel()
    #     Fi = torch.zeros(K, dtype=torch.float32, device=self.device)
    #     Fi.scatter_add_(0, Kstar.view(-1), flow.view(-1))

    #     # Fill usage dict: per column and per router-id (Fi/Z per router in that column)
    #     for k in range(K):
    #         col_xy = (int(cx[k].item()), int(cy[k].item()))
    #         usage['per_column'][col_xy] = float(Fi[k].item())
    #         # fill each z-router in this column with Fi/Z
    #         per_router = float(Fi[k].item() / max(1, self.dimZ))
    #         for z in range(self.dimZ):
    #             rid = z * self.dimX * self.dimY + col_xy[1]*self.dimX + col_xy[0]
    #             usage['total'][rid] = usage['total'].get(rid, 0.0) + per_router

    #     return cost, usage
    # --- in env.py, inside class Env_tsp ---
    def test_cost2(
    self,
    mapping: torch.Tensor,
    allowed_tsvs: Optional[torch.Tensor],
    stage: int,
    tsv_assgn: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, dict]:
        """
        Compute total communication cost for a given mapping and TSV set.
        (stage 2: elevator-first; stage 3: each router uses only its assigned TSV)
        Usage dict shows per-router/per-column vertical flows.
        """
        m = mapping.view(-1).to(self.device)
        usage = {'total': {}, 'per_column': {}, 'up': {}, 'down': {}}

        if stage == 1 or allowed_tsvs is None or allowed_tsvs.numel() == 0:
            D = self._dist_fc3d_for_mapping(m)
            cost = (self.W * D).sum()
            return cost, usage

        if (stage == 2):
            # ELEVATOR-FIRST logic (no change)
            allowed = allowed_tsvs.to(self.device).long()
            cx, cy, _ = self._cx_cy_from_allowed_tsvs(allowed)
            Dmin, Kstar = self._dist_elevator_first(m, cx, cy)
            cost = (self.W * Dmin).sum()

            # The rest of the usage-accumulation block is unchanged from your original.

        elif (stage == 3):  # <============ STAGE 3: STRICT PER-ROUTER ASSIGNMENT =============
            assert tsv_assgn is not None and tsv_assgn.numel() == m.numel(), "tsv_assgn must be the per-task assigned TSV router-ids!"
            # print(tsv_assgn, m)
            # Get task and assigned-TSV coordinates
            src_x, src_y, src_z = self.rx[m], self.ry[m], self.rz[m]
            # print(src_x,src_y,src_z)
            tsv_x = self.rx[tsv_assgn]
            tsv_y = self.ry[tsv_assgn]
            tsv_z = self.rz[tsv_assgn]

            # Build target coord arrays for all dests
            dst_x, dst_y, dst_z = src_x[None, :], src_y[None, :], src_z[None, :]
            src_x, src_y, src_z = src_x[:, None], src_y[:, None], src_z[:, None]
            tsv_x, tsv_y = tsv_x[:, None], tsv_y[:, None]  # [N,1] for broadcasting

            # For each source i, flows to all dst j
            # Movement: src -> its assigned TSV column (in XY), vertical hop to Z-dst, TSV -> dst in XY.
            xy_to_tsv = (src_x - tsv_x).abs() + (src_y - tsv_y).abs()  # [N,1]
            tsv_to_dst = (tsv_x - dst_x).abs() + (tsv_y - dst_y).abs()  # [N,N]
            z_diff = (src_z - dst_z).abs()  # [N,N]
            D_fixed = xy_to_tsv + tsv_to_dst + z_diff  # [N,N]
            cost = (self.W * D_fixed).sum()
            
            # print(self.W, D_fixed,sep='\n')

            #basically calculating the distance beteen all the node and then multiplying it with the weight matrix

            # TSV Usage calculation
            Fi = torch.zeros(tsv_assgn.max().item() + 1, dtype=torch.float32, device=self.device)
            Fi_up = torch.zeros_like(Fi)
            Fi_down = torch.zeros_like(Fi)

            # Only count vertical pairs (z differs)
            up_mask = (src_z < dst_z)  # [N,N]
            down_mask = (src_z > dst_z)  # [N,N]
            vert_mask = (src_z != dst_z)

            for i in range(m.numel()):
                assigned_tsv = tsv_assgn[i].item()
                # For all vertical comms out of i
                vert_flows = self.W[i] * vert_mask[i]
                up_flows = self.W[i] * up_mask[i]
                down_flows = self.W[i] * down_mask[i]
                Fi[assigned_tsv] += vert_flows.sum().item()
                Fi_up[assigned_tsv] += up_flows.sum().item()
                Fi_down[assigned_tsv] += down_flows.sum().item()

            # Build usage dicts (per column XY, per-router for this assignment)
            for i in range(tsv_assgn.numel()):
                assigned_tsv = tsv_assgn[i].item()
                col_xy = (int(self.rx[assigned_tsv].item()), int(self.ry[assigned_tsv].item()))
                for z in range(self.dimZ):
                    rid = z * self.dimX * self.dimY + col_xy[1]*self.dimX + col_xy[0]
                    usage['total'][rid] = usage['total'].get(rid, 0.0) + Fi[assigned_tsv].item()/self.dimZ
                    if Fi_up[assigned_tsv] > 0:
                        usage['up'][rid] = usage['up'].get(rid, 0.0) + Fi_up[assigned_tsv].item()/self.dimZ
                    if Fi_down[assigned_tsv] > 0:
                        usage['down'][rid] = usage['down'].get(rid, 0.0) + Fi_down[assigned_tsv].item()/self.dimZ
                usage['per_column'][col_xy] = usage['per_column'].get(col_xy, 0.0) + Fi[assigned_tsv].item()

        else:
            raise ValueError("stage argument must be 1, 2, or 3.")

        return cost, usage


    
    @torch.no_grad()
    def compute_tsv_cost_variance(
    self,
    mapping: torch.Tensor,
    allowed_tsvs: torch.Tensor,
    stage: int,
    tsv_assignments: Optional[torch.Tensor] = None,
    normalize: bool = True
) -> tuple[torch.Tensor, dict]:
        """
        Variance over TSV *columns* (per-direction up/down), aligned with test_cost2().
        - Stage 2: elevator-first routing (same as before)
        - Stage 3: fixed per-router TSV assignment
        """
        m = mapping.view(-1).to(self.device)
        N = m.numel()

        if isinstance(tsv_assignments, list):
            tsv_assignments = torch.tensor(tsv_assignments, device=self.device)

        # Helper: return zero tensors if inputs invalid
        def _zeros_return():
            zero = torch.tensor(0.0, device=self.device)
            empty = torch.zeros(0, device=self.device)
            extras = dict(Fi=empty, Fi_up=empty, Fi_down=empty,
                        avg_up=zero, avg_down=zero,
                        up_sum=zero, down_sum=zero,
                        up_var_per_tsv=zero, down_var_per_tsv=zero,
                        combined_raw=zero, combined_normalized=zero)
            return (zero, extras)

        # ---- Stage 3: strict TSV assignment ----------------------------------
        if stage == 3 and tsv_assignments is not None and tsv_assignments.numel() == N:
            tsv_assignments = tsv_assignments.long().to(self.device)

            zi = self.rz[m]
            vertical_pairs = (zi[:, None] != zi[None, :])
            dir_up = (zi[:, None] < zi[None, :])
            dir_down = (zi[:, None] > zi[None, :])

            flow_all = self.W * vertical_pairs.float()
            flow_up = self.W * dir_up.float()
            flow_down = self.W * dir_down.float()

            max_tsv = int(tsv_assignments.max().item()) + 1
            Fi = torch.zeros(max_tsv, dtype=torch.float32, device=self.device)
            Fi_up = torch.zeros_like(Fi)
            Fi_down = torch.zeros_like(Fi)

            # Accumulate vertical flows per assigned TSV
            for i in range(N):
                assigned_tsv = int(tsv_assignments[i].item())
                Fi[assigned_tsv] += flow_all[i].sum()
                Fi_up[assigned_tsv] += flow_up[i].sum()
                Fi_down[assigned_tsv] += flow_down[i].sum()

            # Compute directional stats
            K = max(1, max_tsv)
            avg_up = Fi_up.sum() / K
            avg_down = Fi_down.sum() / K

            mask_up = Fi_up > 0
            mask_down = Fi_down > 0
            up_sum = ((Fi_up - avg_up).pow(2))[mask_up].sum()
            down_sum = ((Fi_down - avg_down).pow(2))[mask_down].sum()
            valid = int(mask_up.sum().item() + mask_down.sum().item()) or 1
            combined_raw = (up_sum + down_sum) / float(valid)

            z_minus_1 = max(0, self.dimZ - 1)
            up_var_per_tsv = up_sum / max(1, K * z_minus_1)
            down_var_per_tsv = down_sum / max(1, K * z_minus_1)

            beta = self.W.sum()
            combined_normalized = combined_raw / (beta * beta) if (normalize and beta > 0) else combined_raw

            extras = dict(
                Fi=Fi, Fi_up=Fi_up, Fi_down=Fi_down,
                avg_up=avg_up, avg_down=avg_down,
                up_sum=up_sum, down_sum=down_sum,
                up_var_per_tsv=up_var_per_tsv, down_var_per_tsv=down_var_per_tsv,
                combined_raw=combined_raw, combined_normalized=combined_normalized
            )
            return (combined_normalized if normalize else combined_raw), extras

        # ---- Stage 2 or fallback (elevator-first) -----------------------------
        if allowed_tsvs is None or allowed_tsvs.numel() == 0:
            return _zeros_return()

        allowed = allowed_tsvs.view(-1).long().to(self.device)
        cx, cy, _ = self._cx_cy_from_allowed_tsvs(allowed)
        K = int(cx.numel())
        if K == 0:
            return _zeros_return()

        _, Kstar = self._dist_elevator_first(m, cx, cy)
        zi = self.rz[m]
        vertical_pairs = (zi[:, None] != zi[None, :])
        dir_up = (zi[:, None] < zi[None, :])
        dir_down = (zi[:, None] > zi[None, :])

        flow_all = self.W * vertical_pairs.float()
        flow_up = self.W * dir_up.float()
        flow_down = self.W * dir_down.float()

        Fi = torch.zeros(K, dtype=torch.float32, device=self.device)
        Fi_up = torch.zeros_like(Fi)
        Fi_down = torch.zeros_like(Fi)

        Fi.scatter_add_(0, Kstar.view(-1), flow_all.view(-1))
        Fi_up.scatter_add_(0, Kstar.view(-1), flow_up.view(-1))
        Fi_down.scatter_add_(0, Kstar.view(-1), flow_down.view(-1))

        avg_up = Fi_up.sum() / max(1, K)
        avg_down = Fi_down.sum() / max(1, K)

        mask_up = Fi_up > 0
        mask_down = Fi_down > 0
        up_sum = ((Fi_up - avg_up).pow(2))[mask_up].sum()
        down_sum = ((Fi_down - avg_down).pow(2))[mask_down].sum()
        valid = int(mask_up.sum().item() + mask_down.sum().item()) or 1
        combined_raw = (up_sum + down_sum) / float(valid)

        z_minus_1 = max(0, self.dimZ - 1)
        up_var_per_tsv = up_sum / max(1, K * z_minus_1)
        down_var_per_tsv = down_sum / max(1, K * z_minus_1)

        beta = self.W.sum()
        combined_normalized = combined_raw / (beta * beta) if (normalize and beta > 0) else combined_raw

        extras = dict(
            Fi=Fi, Fi_up=Fi_up, Fi_down=Fi_down,
            avg_up=avg_up, avg_down=avg_down,
            up_sum=up_sum, down_sum=down_sum,
            up_var_per_tsv=up_var_per_tsv, down_var_per_tsv=down_var_per_tsv,
            combined_raw=combined_raw, combined_normalized=combined_normalized
        )
        return (combined_normalized if normalize else combined_raw), extras









    
    def test_cost(self, mapp):
        cost = torch.zeros(mapp.shape[0])
        ml = self.dimX * self.dimY * self.dimZ
        for i in list(self.traff_dict.keys()):
            p1 = (mapp[0:ml] == self.traff_dict[i]['src']).nonzero(as_tuple=True)[0]
            p2 = (mapp[0:ml] == self.traff_dict[i]['dst']).nonzero(as_tuple=True)[0]
            hops = hop_cnt(mapp[ml:len(mapp)], p1, p2, self.nw_dict, self.dimX, self.dimY)[0]
            cost += self.traff_dict[i]['val'] * hops
        return cost

    def show(self, nodes, tour, tsv):
        tsv_cords1 = id2coor(tsv, self.dimX, self.dimY, self.dimZ)
        nodes = nodes.cpu().detach()
        tour  = tour[:].cpu().detach()
        tsv_cords1   = tsv_cords1[:].cpu().detach()
        b= torch.stack((tsv_cords1,tsv_cords1),dim=1)
        plt.figure(figsize=(19.20,10.80))
        ax = plt.axes(projection='3d')
        ax.plot3D(nodes[:, 0], nodes[:, 1],nodes[:,2], 'yo', markersize=25)
        for k in range(tsv.shape[0]):
            ax.plot3D([b[k][0][0],b[k][1][0]],[b[k][0][1],b[k][1][1]],[0,1],'r-',linewidth=3)
        i = 0
        for x, y, z in nodes:
            label = '%d' % (tour[i])   
            i = i+1
            ax.text(x, y, z, label)
        i = 0
        for x, y, z in nodes:
            label = '(%d,%d,%d)' % (x,y,z)   
            i = i+1
            ax.text(x-0.1, y-0.1, z-0.1, label)
        for z in range(self.dimZ):
            for y in range(self.dimY):
                for x in range(self.dimX-1):
                    ax.plot3D([x+0.1,x+1-0.1],[y,y],[z,z],'k-')
        for z in range(self.dimZ):
            for x in range(self.dimX):
                for y in range(self.dimY-1):
                    ax.plot3D([x,x],[y+0.2,y+1-0.1],[z,z],'k-')
        ax.view_init(elev=22, azim=-55)
        ax.set_xlabel('$X$', fontsize=20)
        ax.set_ylabel('$Y$',fontsize=20)
        ax.set_zlabel('$Z$', fontsize=20)        
        plt.savefig('plots/mesh3D_'+str(self.dimX)+str(self.dimY)+str(self.dimZ)+'.eps', format='eps',dpi=1200,bbox_inches='tight')
        
    def shuffle(self, inputs):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        inputs = torch.squeeze(inputs)
        shuffle_inputs = torch.zeros(inputs.size())
        for i in range(self.batch):
            perm = torch.randperm(self.city_t)
            shuffle_inputs[i,:,:] = inputs[i,perm,:]
        return shuffle_inputs

    def back_tours(self, pred_shuffle_tours, shuffle_inputs, test_inputs, device):
        test_inputs = torch.squeeze(test_inputs)
        pred_tours = []
        for i in range(self.batch):
            pred_tour = []
            for j in range(int(self.city_t/4)):
                xy_temp = shuffle_inputs[i, pred_shuffle_tours[i, j]].to(device)
                for k in range(self.city_t):
                    if torch.all(torch.eq(xy_temp, test_inputs[i, k])):
                        pred_tour.append(torch.tensor(k))
                        pred_tours.append(torch.stack(pred_tour, dim=0))
                        break
        pred_tours = torch.stack(pred_tours, dim=0)
        return pred_tours

    def get_random_tsv(self):
        tour = []
        while len(tour) < int(self.city_t/4):
            city = np.random.randint(self.city_t)
            if city not in tour:
                tour.append(city)
        tour = torch.from_numpy(np.array(tour))
        return tour

    def stack_random_tsvs(self):
        list_of_tsvs = [self.get_random_tsv() for i in range(self.batch)]
        tours = torch.stack(list_of_tsvs, dim = 0)
        return tours

    def correct_TSVs(self, tsv_ip):
        ln = tsv_ip.shape[1]
        dim1 = int(ln*(ln-1)/2)
        valid_tsvs = torch.zeros(tsv_ip.shape[0],dim1).to(int)
        tsv_pairs = torch.zeros(tsv_ip.shape[0],dim1,2)
        mod_tsp = torch.zeros(tsv_ip.shape).to(int).to('cuda')
        mod_tsp[:] = tsv_ip[:]
        for rep in range(2):
            k = 0
            for i in range(ln):
                j = i+1
                while j<ln:
                    valid_tsvs[:,k] = dist_eval(mod_tsp[:,i],mod_tsp[:,j],self.dimX,self.dimY,self.dimZ)
                    tsv_pairs[:,k][:,0], tsv_pairs[:,k][:,1] = mod_tsp[:,i],mod_tsp[:,j]
                    j = j+1
                    k = k+1
            a = (valid_tsvs == 1).nonzero(as_tuple=True)[0]
            a = torch.unique(a)
            for tnum in a:
                id_nbrs = node_nebrs(tsv_ip[tnum],self.dimX,self.dimY)
                pos = (valid_tsvs[tnum]==1).nonzero(as_tuple=True)[0][0]
                pos1 = (tsv_ip[tnum]== tsv_pairs[tnum][pos][0]).nonzero(as_tuple=True)[0][0]
                not_in_S = random.choice([x for x in range(self.city_t) if x not in id_nbrs])
                mod_tsp[tnum][pos1] = not_in_S
        return mod_tsp.to('cuda')

    def tsv_repair(self, tsvs):
        correct_tsvs = torch.zeros(tsvs.shape).to(int).to('cuda')
        for iter in range(tsvs.shape[1]):
            tsv_cords = id2coor(tsvs[:,iter],self.dimX,self.dimY,self.dimZ)
            tsv_cords[:,2] = 0
            correct_tsvs[:,iter] = coor2id(tsv_cords,self.dimX,self.dimY,self.dimZ)
        return correct_tsvs