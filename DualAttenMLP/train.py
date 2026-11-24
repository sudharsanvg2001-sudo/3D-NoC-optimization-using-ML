"""
train.py
────────
REINFORCE training loop for DualGATMappingNetwork on 3D NoC mapping.

TSV-aware elevator-first routing cost always used — no stage concept.
TSV router IDs passed via --tsvIds and forwarded to cost computation.
"""

from __future__ import annotations

import os
import argparse
import torch
import torch.optim as optim
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple
import random as _random

from env             import Env_tsp
from mapping_network import DualGATMappingNetwork
from reward          import compute_mapping_cost, batch_mapping_cost
from data_utils      import (
    build_noc_topology, mapping_to_tensor,
    random_instance, load_traffic_file, create_mesh,
    load_tsv_placements, load_all_traffic_files,
)


# ─────────────────────────────────────────────────────────────────────────────
# Build Env_tsp for a single problem instance
# ─────────────────────────────────────────────────────────────────────────────

def make_env(
    args       ,                          # argparse.Namespace with dimX, dimY, dimZ
    core_data  : Data,                    # IP core PyG graph
    traff_dict : Optional[Dict] = None,  # from load_traffic_file() or cr_trDic()
    device     : str = 'cpu',
) -> Env_tsp:
    """
    Construct an Env_tsp instance for one problem instance.

    Fix applied: now builds nw_dict via create_mesh() and passes the
    real traff_dict — matching exactly how the previous code called Env_tsp:

        env = Env_tsp(args, nw_dict, traff_dict, traff_data)

    Args:
        args       : argparse.Namespace with dimX, dimY, dimZ
        core_data  : PyG Data for the IP core graph
        traff_dict : traffic link dict from load_traffic_file()
                     If None (e.g. random instances), passes empty dict
    """
    # Fix 2: build nw_dict properly — matches create_mesh() from previous code
    nw_dict    = create_mesh(args.dimX, args.dimY, args.dimZ)
    traff_dict = traff_dict or {}
    return Env_tsp(args=args, nw_dict=nw_dict, traff_dict=traff_dict, data=core_data)


# ─────────────────────────────────────────────────────────────────────────────
# REINFORCE update step  — matches the algorithm exactly:
#
#   ∇J = 1/(B×(k-1)) Σ_i Σ_j  (L(m_j|s_i) − b(s_i)) × ∇log p(m_j|s_i)
#
#   where b(s_i) = mean_j L(m_j|s_i)   (per-instance baseline)
#   B = number of instances in mini-batch
#   k = number of solutions sampled per instance
# ─────────────────────────────────────────────────────────────────────────────

def reinforce_step(
    model        : DualGATMappingNetwork,
    optimizer    : torch.optim.Optimizer,
    batch        : List[Tuple[Data, torch.Tensor, Dict]],
    tsv_pool     : List[torch.Tensor],
    args,
    topo_feat_dim: int,
    topo_edge_dim: int,
    k            : int = 10,
) -> Tuple[float, float]:
    """
    One REINFORCE gradient step over a mini-batch of B instances.

    Each item in the batch gets its own randomly sampled TSV placement
    and its own topology graph — the model trains across a distribution
    of TSV layouts in a single gradient step.

    Args:
        model         : DualGATMappingNetwork (train mode)
        optimizer     : Adam
        batch         : B × (core_data, comm_matrix, traff_dict)
        tsv_pool      : list of TSV placement tensors loaded from file
        args          : argparse.Namespace with dimX, dimY, dimZ
        topo_feat_dim : topology node feature dim (7)
        topo_edge_dim : topology edge feature dim (1)
        k             : solutions sampled per instance
    """
    model.train()
    B          = len(batch)
    total_loss = torch.tensor(0.0)
    total_cost = 0.0

    optimizer.zero_grad()

    for item in batch:
        # ── Unpack instance ───────────────────────────────────────────────
        if len(item) == 3:
            core_data, comm_matrix, traff_dict = item
        else:
            core_data, comm_matrix = item
            traff_dict = None

        num_cores = comm_matrix.size(0)
        env       = make_env(args, core_data, traff_dict=traff_dict)

        # ── Sample one TSV placement for this instance ────────────────────
        tsv_idx  = _random.randint(0, len(tsv_pool) - 1)
        tsv_ids  = tsv_pool[tsv_idx]

        # ── Build topology graph for this TSV placement ───────────────────
        topo_data = build_noc_topology(
            args.dimX, args.dimY, args.dimZ,
            tsv_ids=tsv_ids,
        )

        # ── Sample k solutions ────────────────────────────────────────────
        log_probs_i : List[torch.Tensor]          = []
        mappings_i  : List[List[Tuple[int, int]]] = []

        for _ in range(k):
            mapping, task_lp, router_lp = model(
                core_data, topo_data, greedy=False
            )
            log_prob = (task_lp + router_lp).sum()
            log_probs_i.append(log_prob)
            mappings_i.append(mapping)

        # ── Per-instance costs with this TSV placement ────────────────────
        costs_i    = batch_mapping_cost(
            env, mappings_i, num_cores,
            allowed_tsvs=tsv_ids if tsv_ids.numel() > 0 else None,
        )
        baseline_i = costs_i.mean()

        log_probs_tensor = torch.stack(log_probs_i)
        advantages_i     = (costs_i - baseline_i).detach()
        loss_i           = (advantages_i * log_probs_tensor).sum()

        total_loss = total_loss + loss_i
        total_cost += costs_i.mean().item()

    normalised_loss = total_loss / (B * max(k - 1, 1))
    normalised_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return normalised_loss.item(), total_cost / B


# ─────────────────────────────────────────────────────────────────────────────
# Greedy evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_greedy(
    model        : DualGATMappingNetwork,
    dataset      : List[Tuple[Data, torch.Tensor, Dict]],
    tsv_pool     : List[torch.Tensor],
    args,
    topo_feat_dim: int,
    topo_edge_dim: int,
) -> float:
    """
    Greedy evaluation — for each instance samples a random TSV placement.
    Returns mean WCC over the dataset.
    """
    model.eval()
    total = 0.0
    with torch.no_grad():
        for item in dataset:
            if len(item) == 3:
                core_data, comm_matrix, traff_dict = item
            else:
                core_data, comm_matrix = item
                traff_dict = None

            # Sample TSV placement
            tsv_ids   = tsv_pool[_random.randint(0, len(tsv_pool) - 1)]
            topo_data = build_noc_topology(
                args.dimX, args.dimY, args.dimZ, tsv_ids=tsv_ids
            )

            env     = make_env(args, core_data, traff_dict=traff_dict)
            mapping, _, _ = model(core_data, topo_data, greedy=True)
            cost    = compute_mapping_cost(
                env, mapping, comm_matrix.size(0),
                allowed_tsvs=tsv_ids if tsv_ids.numel() > 0 else None,
            )
            total += cost.item()
    return total / max(len(dataset), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Full training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model        : DualGATMappingNetwork,
    train_dataset: List[Tuple[Data, torch.Tensor, Dict]],
    tsv_pool     : List[torch.Tensor],
    args,
    topo_feat_dim: int,
    topo_edge_dim: int,
    val_dataset  : Optional[List[Tuple[Data, torch.Tensor, Dict]]] = None,
    num_iters    : int   = 400,
    lr           : float = 1e-3,
    B            : int   = 16,
    k            : int   = 10,
    save_path    : str   = 'best_model.pt',
    log_every    : int   = 20,
) -> None:
    """
    Full REINFORCE training pipeline.

    Each iteration samples B random graphs AND B random TSV placements.
    Each batch item gets its own topology graph → model trains across
    a distribution of both communication patterns and TSV layouts.
    """
    import random as _random

    optimizer     = optim.Adam(model.parameters(), lr=lr)
    scheduler     = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters)
    best_val_cost = float('inf')

    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train graphs: {len(train_dataset)}"
          + (f"  Val graphs: {len(val_dataset)}" if val_dataset else ""))
    print(f"TSV pool    : {len(tsv_pool)} placements")
    print(f"Mesh        : {args.dimX}×{args.dimY}×{args.dimZ}")
    print(f"Iterations  : {num_iters}  |  B={B}  k={k}  "
          f"normaliser=B×(k-1)={B*(k-1)}")
    print("-" * 60)

    for t in range(1, num_iters + 1):
        # Sample B random graph instances — with replacement
        batch = _random.choices(train_dataset, k=B)

        loss, mean_cost = reinforce_step(
            model, optimizer, batch,
            tsv_pool, args,
            topo_feat_dim, topo_edge_dim,
            k=k,
        )

        scheduler.step()

        if t % log_every == 0:
            msg = (
                f"Iter {t:5d}/{num_iters}  "
                f"Loss {loss:+8.4f}  "
                f"Train WCC {mean_cost:8.3f}"
            )
            if val_dataset:
                val_cost = evaluate_greedy(
                    model, val_dataset, tsv_pool, args,
                    topo_feat_dim, topo_edge_dim,
                )
                msg += f"  Val WCC {val_cost:8.3f}"
                if val_cost < best_val_cost:
                    best_val_cost = val_cost
                    torch.save(model.state_dict(), save_path)
                    msg += "  ✓"
            print(msg)

    if not val_dataset:
        torch.save(model.state_dict(), save_path)

    print("-" * 60)
    print(f"Done. Model saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description='DualGAT IP-core mapping for 3D NoC via REINFORCE'
    )

    # ── Data paths ────────────────────────────────────────────────────────
    parser.add_argument('--dataDir',  type=str, required=True,
                        help='Directory containing .txt traffic files')
    parser.add_argument('--tsvFile',  type=str, required=True,
                        help='Text file of TSV placements — one per line, '
                             'space-separated router IDs')
    parser.add_argument('--valDir',   type=str, default=None,
                        help='Optional directory of validation .txt files')
    parser.add_argument('--saveDir',  type=str, default='.',
                        help='Directory to save model checkpoints')

    # ── Mesh dimensions ───────────────────────────────────────────────────
    parser.add_argument('--dimX', type=int, default=4, help='Mesh X dimension')
    parser.add_argument('--dimY', type=int, default=4, help='Mesh Y dimension')
    parser.add_argument('--dimZ', type=int, default=2, help='Mesh Z dimension')

    # ── Model architecture ────────────────────────────────────────────────
    parser.add_argument('--embedDim',  type=int,  default=128,
                        help='GAT and attention embedding dimension D')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model checkpoint')

    # ── Training hyperparameters ──────────────────────────────────────────
    parser.add_argument('--numIters', type=int,   default=400,
                        help='T — total REINFORCE gradient steps')
    parser.add_argument('--B',        type=int,   default=16,
                        help='Mini-batch size — graphs per gradient step')
    parser.add_argument('--k',        type=int,   default=10,
                        help='Solutions sampled per instance')
    parser.add_argument('--lr',       type=float, default=1e-3,
                        help='Adam learning rate')
    parser.add_argument('--logEvery', type=int,   default=20,
                        help='Log every N iterations')
    parser.add_argument('--seed',     type=int,   default=42,
                        help='Random seed')

    # ── Device ────────────────────────────────────────────────────────────
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run on')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    args   = get_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    os.makedirs(args.saveDir, exist_ok=True)

    print("=" * 60)
    print(f"Mesh      : {args.dimX}×{args.dimY}×{args.dimZ}")
    print(f"Data dir  : {args.dataDir}")
    print(f"TSV file  : {args.tsvFile}")
    print(f"Device    : {args.device}")
    print("=" * 60)

    # ── Step 1: Load all traffic graphs ──────────────────────────────────
    # load_all_traffic_files() loads every .txt in dataDir
    # Each item: (core_data, comm_matrix, traff_dict)
    train_dataset = load_all_traffic_files(args.dataDir)

    # Optional validation split from separate directory
    val_dataset = None
    if args.valDir:
        val_dataset = load_all_traffic_files(args.valDir)

    # All graphs must have the same number of cores for a shared model
    num_cores_list = [item[0].num_nodes for item in train_dataset]
    if len(set(num_cores_list)) > 1:
        raise ValueError(
            f"All traffic graphs must have the same number of cores. "
            f"Found: {set(num_cores_list)}"
        )
    num_cores = num_cores_list[0]
    print(f"✅ {len(train_dataset)} graphs loaded — {num_cores} cores each")

    # ── Step 2: Load TSV placement pool ──────────────────────────────────
    # Each line in tsvFile is one valid TSV placement
    tsv_pool = load_tsv_placements(args.tsvFile)

    # ── Step 3: Build mesh network dictionary ─────────────────────────────
    # Used by Env_tsp — same for all instances
    nw_dict = create_mesh(args.dimX, args.dimY, args.dimZ)
    print(f"✅ Mesh nw_dict built — {len(nw_dict)} routers")

    # ── Step 4: Get topology feature dims from a sample topology ──────────
    # Build one topology graph just to read topo_feat_dim and topo_edge_dim
    sample_topo = build_noc_topology(
        args.dimX, args.dimY, args.dimZ,
        tsv_ids=tsv_pool[0],
    )
    topo_feat_dim = int(sample_topo.topo_feat_dim)   # 7
    topo_edge_dim = int(sample_topo.topo_edge_dim)   # 1
    print(f"✅ Topology dims — node features: {topo_feat_dim}  "
          f"edge features: {topo_edge_dim}")

    # ── Step 5: Build model ───────────────────────────────────────────────
    model = DualGATMappingNetwork(
        core_feat_dim = num_cores,
        topo_feat_dim = topo_feat_dim,
        embed_dim     = args.embedDim,
        core_edge_dim = 1,
        topo_edge_dim = topo_edge_dim,
    ).to(device)

    if args.pretrained is not None:
        state = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(state)
        print(f"✅ Loaded pretrained weights from {args.pretrained}")

    print(f"✅ Model built — {sum(p.numel() for p in model.parameters()):,} parameters")

    # ── Step 6: Train ────────────────────────────────────────────────────
    save_path = os.path.join(args.saveDir, 'best_model.pt')

    train(
        model         = model,
        train_dataset = train_dataset,
        tsv_pool      = tsv_pool,
        args          = args,
        topo_feat_dim = topo_feat_dim,
        topo_edge_dim = topo_edge_dim,
        val_dataset   = val_dataset,
        num_iters     = args.numIters,
        lr            = args.lr,
        B             = args.B,
        k             = args.k,
        save_path     = save_path,
        log_every     = args.logEvery,
    )

    