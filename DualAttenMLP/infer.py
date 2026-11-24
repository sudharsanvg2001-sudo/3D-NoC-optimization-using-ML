"""
infer.py
────────
Inference pipeline for DualGATMappingNetwork + AMOSA TSV assignment.

Steps:
  1. Load trained model checkpoint
  2. Load traffic file  →  core_data, traff_dict
  3. Load TSV placement from file  →  tsv_ids  (fixed placement)
  4. Build topology graph with TSV features
  5. Run greedy mapping  →  mapping [(core, router), ...]
  6. Run AMOSA offline elevator-SUBSET ASSIGNMENT on the mapping
       Given fixed TSV placement, find per-router elevator subsets
       that minimise utilisation variance + average inter-layer distance
  7. Display Pareto front — user selects a solution
  8. Report WCC comparison

Usage:
    python infer.py \
        --model   checkpoints/best_model.pt \
        --dataTxt data/consumer1.txt \
        --tsvFile tsv_placements.txt \
        --dimX 4 --dimY 4 --dimZ 2 \
        --embedDim 128 \
        --tsvIdx  0 \
        --esetMin 2 --esetMax 3 \
        --select 0
"""

from __future__ import annotations

import os
import sys
import argparse
import torch

from data_utils       import (
    load_traffic_file, create_mesh,
    build_noc_topology, mapping_to_tensor,
    load_tsv_placements,
)
from mapping_network  import DualGATMappingNetwork
from reward           import compute_mapping_cost
from env              import Env_tsp
from amosa_tsv        import run_amosa


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description='Inference + AMOSA elevator-subset assignment for 3D NoC'
    )

    # ── Required ──────────────────────────────────────────────────────────
    parser.add_argument('--model',   type=str, required=True,
                        help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--dataTxt', type=str, required=True,
                        help='Traffic .txt file for this instance')
    parser.add_argument('--tsvFile', type=str, required=True,
                        help='TSV placements file (one placement per line)')

    # ── Mesh ──────────────────────────────────────────────────────────────
    parser.add_argument('--dimX', type=int, default=4)
    parser.add_argument('--dimY', type=int, default=4)
    parser.add_argument('--dimZ', type=int, default=2)

    # ── Model (must match training) ───────────────────────────────────────
    parser.add_argument('--embedDim', type=int, default=128)

    # ── TSV selection ─────────────────────────────────────────────────────
    parser.add_argument('--tsvIdx', type=int, default=0,
                        help='Index into tsvFile to use as fixed TSV placement')

    # ── AMOSA parameters ─────────────────────────────────────────────────
    parser.add_argument('--esetMin',  type=int,   default=2,
                        help='Min elevators per router subset (paper: 2)')
    parser.add_argument('--esetMax',  type=int,   default=3,
                        help='Max elevators per router subset (paper: 3)')
    parser.add_argument('--amoTemp',  type=float, default=100.0)
    parser.add_argument('--amoCool',  type=float, default=0.95)
    parser.add_argument('--amoSteps', type=int,   default=50)
    parser.add_argument('--amoFinal', type=float, default=0.1)
    parser.add_argument('--amoSeed',  type=int,   default=42)

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument('--select', type=int, default=None,
                        help='Pareto front index to select '
                             '(0=lowest variance, -1=lowest distance). '
                             'Omit for interactive prompt.')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'])

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("Inference + AMOSA Elevator-Subset Assignment")
    print(f"  Model    : {args.model}")
    print(f"  Traffic  : {args.dataTxt}")
    print(f"  Mesh     : {args.dimX}×{args.dimY}×{args.dimZ}")
    print(f"  Subset   : {args.esetMin}–{args.esetMax} elevators per router")
    print("=" * 60)

    # ── Step 1: Load traffic file ─────────────────────────────────────────
    core_data, traff_dict, max_val = load_traffic_file(args.dataTxt)
    num_cores   = core_data.num_nodes
    comm_matrix = core_data.x   # [N, N]
    print(f"✅ Traffic loaded — {num_cores} cores, "
          f"{len(traff_dict)} links, max_val={max_val:.2f}")

    # ── Step 2: Load fixed TSV placement ──────────────────────────────────
    # TSV placement is fixed — AMOSA will assign subsets from these TSVs
    tsv_pool    = load_tsv_placements(args.tsvFile)
    tsv_ids_t   = tsv_pool[args.tsvIdx]
    tsv_ids     = tsv_ids_t.tolist()
    print(f"✅ Fixed TSV placement (idx={args.tsvIdx}): {tsv_ids}")
    print(f"   {len(tsv_ids)} TSVs in a "
          f"{args.dimX}×{args.dimY} layer "
          f"({len(tsv_ids)/(args.dimX*args.dimY)*100:.0f}% density)")

    # ── Step 3: Build topology graph with fixed TSV features ──────────────
    topo_data     = build_noc_topology(
        args.dimX, args.dimY, args.dimZ,
        tsv_ids=tsv_ids_t,
    )
    topo_feat_dim = int(topo_data.topo_feat_dim)
    topo_edge_dim = int(topo_data.topo_edge_dim)
    print(f"✅ Topology built — {topo_data.x.size(0)} routers")

    # ── Step 4: Load model ────────────────────────────────────────────────
    model = DualGATMappingNetwork(
        core_feat_dim = num_cores,
        topo_feat_dim = topo_feat_dim,
        embed_dim     = args.embedDim,
        core_edge_dim = 1,
        topo_edge_dim = topo_edge_dim,
    ).to(device)

    if not os.path.exists(args.model):
        print(f"❌ Checkpoint not found: {args.model}")
        sys.exit(1)

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"✅ Model loaded — "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")

    # ── Step 5: Greedy mapping ────────────────────────────────────────────
    print("\n── Greedy Mapping ──────────────────────────────────────────")
    with torch.no_grad():
        mapping, _, _ = model(core_data, topo_data, greedy=True)

    layer_size = args.dimX * args.dimY
    print(f"\nMapping (core → router):")
    for core_idx, router_idx in sorted(mapping):
        x, y, z  = router_idx % args.dimX, \
                   (router_idx % layer_size) // args.dimX, \
                   router_idx // layer_size
        tsv_flag = int(topo_data.x[router_idx, 3].item())
        tsv_str  = "  ← TSV router" if tsv_flag else ""
        print(f"  Core {core_idx:2d}  →  Router {router_idx:2d}  "
              f"(x={x},y={y},z={z}){tsv_str}")

    # WCC with nearest-TSV routing (baseline before AMOSA)
    nw_dict = create_mesh(args.dimX, args.dimY, args.dimZ)
    env     = Env_tsp(
        args=args, nw_dict=nw_dict,
        traff_dict=traff_dict, data=core_data,
    )
    baseline_cost = compute_mapping_cost(
        env, mapping, num_cores,
        allowed_tsvs=tsv_ids_t if tsv_ids_t.numel() > 0 else None,
    )
    print(f"\nBaseline WCC (nearest-TSV routing): {baseline_cost.item():.4f}")

    # ── Step 6: AMOSA elevator-subset assignment ──────────────────────────
    print("\n── AMOSA Elevator-Subset Assignment ────────────────────────")
    print(f"Finding optimal elevator subsets for each router...")
    print(f"Input: fixed placement {tsv_ids}")
    print(f"       {len(tsv_ids)} TSVs  ·  subset size {args.esetMin}–{args.esetMax}")

    pareto_front = run_amosa(
        mapping        = mapping,
        comm_matrix    = comm_matrix,
        tsv_ids        = tsv_ids,
        dimX           = args.dimX,
        dimY           = args.dimY,
        dimZ           = args.dimZ,
        eset_min       = args.esetMin,
        eset_max       = args.esetMax,
        initial_temp   = args.amoTemp,
        final_temp     = args.amoFinal,
        cooling_rate   = args.amoCool,
        steps_per_temp = args.amoSteps,
        seed           = args.amoSeed,
        verbose        = True,
    )

    # ── Step 7: Select from Pareto front ─────────────────────────────────
    print("\n── Solution Selection ──────────────────────────────────────")
    print(f"{'#':>3}  {'Util Variance':>15}  {'Avg Distance':>14}  Note")
    print("-" * 60)
    for idx, sol in enumerate(pareto_front):
        note = ""
        if idx == 0:
            note = "← lowest variance (best load balance)"
        if idx == len(pareto_front) - 1:
            note = "← lowest distance (best energy)"
        print(f"{idx:>3}  {sol['obj1']:>15.6f}  {sol['obj2']:>14.4f}  {note}")

    if args.select is not None:
        sel_idx = args.select % len(pareto_front)
    else:
        sel_idx = int(input(f"\nSelect [0–{len(pareto_front)-1}]: "))
        sel_idx = max(0, min(sel_idx, len(pareto_front) - 1))

    selected   = pareto_front[sel_idx]
    assignment = selected['assignment']
    print(f"\n✅ Selected solution #{sel_idx}")
    print(f"   Utilisation variance : {selected['obj1']:.6f}")
    print(f"   Avg inter-layer dist : {selected['obj2']:.4f}")

    # ── Step 8: Print per-router elevator subset assignment ───────────────
    print(f"\n── Per-Router Elevator Subset Assignment ───────────────────")
    print(f"{'Router':>8}  {'(x,y,z)':>10}  Elevator Subset")
    print("-" * 50)
    for r in range(args.dimX * args.dimY * args.dimZ):
        x = r % args.dimX
        y = (r % layer_size) // args.dimX
        z = r // layer_size
        subset   = sorted(assignment.get(r, []))
        tsv_flag = int(topo_data.x[r, 3].item())
        note     = "  [TSV router]" if tsv_flag else ""
        print(f"{r:>8}  ({x},{y},{z})       {subset}{note}")

    print(f"\n✅ Done. Use this assignment table in your router runtime "
          f"selection (ERR or Distance-Based from AdEle+).")


if __name__ == '__main__':
    main()