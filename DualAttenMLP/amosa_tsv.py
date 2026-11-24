"""
amosa_tsv.py
────────────
Offline TSV elevator-subset assignment using AMOSA
(Archived Multi-Objective Simulated Annealing).

Reference:
  Taheri et al., "AdEle+: An Adaptive Congestion-and-Energy-Aware
  Elevator Selection for Partially Connected 3D Networks-on-Chip"
  IEEE Transactions on Computers, 2023.  Section III-B.

Problem:
  Given:
    - A fixed mapping      (core → router, from model inference)
    - A fixed TSV placement (which routers have TSVs, from generate_tsv_ids)
    - A communication matrix (traffic between cores)

  Find:
    A = {A_1, A_2, ..., A_R}
    where A_i ⊆ E  is the elevator SUBSET assigned to router i.
    |A_i| ∈ [eset_min, eset_max]  (paper uses 2–3)

  Such that two objectives are minimised:

    Objective 1 — Elevator utilisation variance  (Eq. 3)
      U_e = Σ_i (1/|A_i|) Σ_j f_ij · P_ije
      σ²  = (1/E) Σ_e (U_e − μ)²
      Lower variance = more balanced traffic across TSVs.

    Objective 2 — Average inter-layer node distance  (Eq. 5)
      D^e_ij = d_se + |z_i − z_j| + d_ed   (elevator-first distance)
      AD = average of D^e_ij over all cross-layer pairs, weighted by
           uniform selection within each router's subset A_i.
      Lower AD = shorter paths = lower energy.

Solution representation:
  A dict {router_id: [tsv_id, tsv_id, ...]}
  Each router gets a subset of the available TSVs.

Neighbour generation:
  Pick one router, swap one TSV in its subset with another from E.

AMOSA acceptance:
  - Non-dominated neighbour → always accept + add to archive
  - Dominated neighbour     → accept with prob exp(−delta/T)
"""

from __future__ import annotations

import math
import random
import torch
from typing import List, Optional, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def _router_xyz(router_id: int, dimX: int, dimY: int) -> Tuple[int, int, int]:
    layer_size = dimX * dimY
    z  = router_id // layer_size
    xy = router_id %  layer_size
    x  = xy % dimX
    y  = xy // dimX
    return x, y, z


def _manhattan_xy(
    r1: int, r2: int, dimX: int, dimY: int,
) -> int:
    """XY Manhattan distance between two routers (ignores z)."""
    x1, y1, _ = _router_xyz(r1, dimX, dimY)
    x2, y2, _ = _router_xyz(r2, dimX, dimY)
    return abs(x1 - x2) + abs(y1 - y2)


def _elevator_first_dist(
    src: int, dst: int, tsv: int,
    dimX: int, dimY: int,
) -> float:
    """
    Elevator-first routing distance via a specific TSV  (Eq. 4):
        D^e_{ij} = d_se + |z_i − z_j| + d_ed
    Returns 0 if same layer (no cross-layer routing).
    """
    xs, ys, zs = _router_xyz(src, dimX, dimY)
    xd, yd, zd = _router_xyz(dst, dimX, dimY)
    if zs == zd:
        return 0.0
    xe, ye, _  = _router_xyz(tsv, dimX, dimY)
    d_se = abs(xs - xe) + abs(ys - ye)
    d_e  = abs(zs - zd)
    d_ed = abs(xe - xd) + abs(ye - yd)
    return float(d_se + d_e + d_ed)


# ─────────────────────────────────────────────────────────────────────────────
# Build router-level traffic matrix
# ─────────────────────────────────────────────────────────────────────────────

def _build_router_traffic(
    mapping     : List[Tuple[int, int]],
    comm_matrix : torch.Tensor,
    num_routers : int,
) -> torch.Tensor:
    """
    Aggregate core-level communication into router-level traffic.

    F[r1, r2] = Σ_{c1→r1, c2→r2} comm_matrix[c1, c2]

    Returns [num_routers, num_routers] float tensor.
    """
    F = torch.zeros(num_routers, num_routers, dtype=torch.float32)
    core_to_router = {c: r for c, r in mapping}
    N = comm_matrix.size(0)
    for i in range(N):
        ri = core_to_router.get(i)
        if ri is None:
            continue
        for j in range(N):
            rj = core_to_router.get(j)
            if rj is None:
                continue
            if ri != rj:
                F[ri, rj] += comm_matrix[i, j].item()
    return F


# ─────────────────────────────────────────────────────────────────────────────
# Objective functions  (Eq. 1–5 of paper)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_objectives(
    assignment  : Dict[int, List[int]],   # router_id → [tsv_id, ...]
    tsv_ids     : List[int],              # all available TSVs
    F           : torch.Tensor,           # [R, R] router traffic matrix
    dimX        : int,
    dimY        : int,
    dimZ        : int,
) -> Tuple[float, float]:
    """
    Compute both objectives for a given elevator-subset assignment.

    Args:
        assignment : {router_id: [tsv_id, ...]}  subset per router
        tsv_ids    : full set of elevator IDs
        F          : [R, R] router-level traffic
        dimX/Y/Z   : mesh dimensions

    Returns:
        (obj1_utilisation_variance, obj2_avg_inter_layer_distance)
    """
    E          = len(tsv_ids)
    R          = F.size(0)

    # ── Objective 1: Elevator utilisation variance  (Eq. 1–3) ────────────
    # U_e = Σ_i (1/|A_i|) Σ_j f_{ij} · P_ije
    # For uniform selection within subset: P_ije = 1/|A_i| if e ∈ A_i
    # So U_e = Σ_i [ (1/|A_i|²) · Σ_j f_{ij} ]  when e ∈ A_i

    util = {e: 0.0 for e in tsv_ids}

    for src in range(R):
        A_src = assignment.get(src, [])
        if not A_src:
            continue
        sz = len(A_src)
        xs, ys, zs = _router_xyz(src, dimX, dimY)

        for dst in range(R):
            xd, yd, zd = _router_xyz(dst, dimX, dimY)
            if zs == zd:
                continue   # same layer — no elevator
            fij = float(F[src, dst].item())
            if fij == 0:
                continue
            # Each elevator in A_src gets an equal share of traffic
            for e in A_src:
                util[e] += fij / sz

    util_vals = list(util.values())
    mu        = sum(util_vals) / max(E, 1)
    variance  = sum((u - mu) ** 2 for u in util_vals) / max(E, 1)

    # ── Objective 2: Average inter-layer distance  (Eq. 4–5) ─────────────
    # AD = average over all cross-layer pairs of:
    #      (1/|A_i|) Σ_{e ∈ A_i} D^e_{ij}
    total_dist   = 0.0
    cross_pairs  = 0

    for src in range(R):
        A_src = assignment.get(src, [])
        if not A_src:
            continue
        xs, ys, zs = _router_xyz(src, dimX, dimY)

        for dst in range(R):
            xd, yd, zd = _router_xyz(dst, dimX, dimY)
            if zs == zd:
                continue
            # Average elevator-first distance across subset
            avg_d = sum(
                _elevator_first_dist(src, dst, e, dimX, dimY)
                for e in A_src
            ) / len(A_src)
            total_dist  += avg_d
            cross_pairs += 1

    avg_dist = total_dist / max(cross_pairs, 1)

    return variance, avg_dist


# ─────────────────────────────────────────────────────────────────────────────
# Solution initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_assignment(
    num_routers : int,
    tsv_ids     : List[int],
    eset_min    : int,
    eset_max    : int,
    rng         : random.Random,
) -> Dict[int, List[int]]:
    """
    Initialise a random elevator-subset assignment.
    Each router gets a random subset of size in [eset_min, eset_max].
    """
    assignment = {}
    for r in range(num_routers):
        sz             = rng.randint(eset_min, min(eset_max, len(tsv_ids)))
        assignment[r]  = rng.sample(tsv_ids, sz)
    return assignment


# ─────────────────────────────────────────────────────────────────────────────
# Neighbour generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_neighbour(
    assignment  : Dict[int, List[int]],
    tsv_ids     : List[int],
    eset_min    : int,
    eset_max    : int,
    rng         : random.Random,
) -> Dict[int, List[int]]:
    """
    Generate a neighbour by modifying one router's elevator subset.
    Three possible moves (chosen randomly):
      1. Swap one TSV in subset with one not in subset
      2. Add one TSV to subset (if below eset_max)
      3. Remove one TSV from subset (if above eset_min)
    """
    neighbour = {r: s[:] for r, s in assignment.items()}
    r         = rng.randint(0, len(neighbour) - 1)
    subset    = neighbour[r]
    not_in    = [e for e in tsv_ids if e not in subset]

    move      = rng.choice(['swap', 'add', 'remove'])

    if move == 'swap' and not_in:
        # Swap one existing with one not in subset
        idx         = rng.randint(0, len(subset) - 1)
        subset[idx] = rng.choice(not_in)

    elif move == 'add' and len(subset) < eset_max and not_in:
        subset.append(rng.choice(not_in))

    elif move == 'remove' and len(subset) > eset_min:
        subset.pop(rng.randint(0, len(subset) - 1))

    else:
        # Fallback: always do a swap if possible
        if not_in:
            idx         = rng.randint(0, len(subset) - 1)
            subset[idx] = rng.choice(not_in)

    neighbour[r] = subset
    return neighbour


# ─────────────────────────────────────────────────────────────────────────────
# Pareto archive management
# ─────────────────────────────────────────────────────────────────────────────

def _dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """True if a dominates b (lower is better for both objectives)."""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def _update_archive(
    archive  : List[dict],
    new_sol  : dict,
    max_size : int,
) -> List[dict]:
    """Add new_sol to archive if non-dominated. Remove dominated members."""
    new_objs = (new_sol['obj1'], new_sol['obj2'])

    # Reject if dominated by any archive member
    for sol in archive:
        if _dominates((sol['obj1'], sol['obj2']), new_objs):
            return archive

    # Remove archive members dominated by new_sol
    archive = [
        s for s in archive
        if not _dominates(new_objs, (s['obj1'], s['obj2']))
    ]
    archive.append(new_sol)

    # Truncate by crowding distance if needed
    if len(archive) > max_size:
        archive = _truncate_by_crowding(archive, max_size)

    return archive


def _truncate_by_crowding(archive: List[dict], max_size: int) -> List[dict]:
    """Remove solution with smallest crowding distance."""
    sorted_a   = sorted(archive, key=lambda s: s['obj1'])
    n          = len(sorted_a)
    o1_range   = max(s['obj1'] for s in sorted_a) - min(s['obj1'] for s in sorted_a) or 1.0
    o2_range   = max(s['obj2'] for s in sorted_a) - min(s['obj2'] for s in sorted_a) or 1.0

    for s in sorted_a:
        s['_crowd'] = 0.0
    sorted_a[0]['_crowd']  = float('inf')
    sorted_a[-1]['_crowd'] = float('inf')

    for i in range(1, n - 1):
        sorted_a[i]['_crowd'] += (
            (sorted_a[i+1]['obj1'] - sorted_a[i-1]['obj1']) / o1_range +
            (sorted_a[i+1]['obj2'] - sorted_a[i-1]['obj2']) / o2_range
        )

    sorted_a.sort(key=lambda s: s['_crowd'])
    sorted_a.pop(0)
    return sorted_a[:max_size]


# ─────────────────────────────────────────────────────────────────────────────
# AMOSA main
# ─────────────────────────────────────────────────────────────────────────────

def run_amosa(
    mapping          : List[Tuple[int, int]],
    comm_matrix      : torch.Tensor,
    tsv_ids          : List[int],
    dimX             : int,
    dimY             : int,
    dimZ             : int,
    eset_min         : int   = 2,
    eset_max         : int   = 3,
    initial_temp     : float = 100.0,
    final_temp       : float = 0.1,
    cooling_rate     : float = 0.95,
    steps_per_temp   : int   = 50,
    archive_max_size : int   = 100,
    seed             : int   = 42,
    verbose          : bool  = True,
) -> List[dict]:
    """
    Run AMOSA to find Pareto-optimal elevator-subset assignments.

    Args:
        mapping          : [(core_idx, router_idx)] from model inference
        comm_matrix      : [N, N] core-level communication volumes
        tsv_ids          : list of router IDs that have TSVs (fixed placement)
        dimX, dimY, dimZ : mesh dimensions
        eset_min         : min elevators per router subset  (paper uses 2)
        eset_max         : max elevators per router subset  (paper uses 3)
        initial_temp     : AMOSA starting temperature
        final_temp       : AMOSA stopping temperature
        cooling_rate     : geometric cooling factor
        steps_per_temp   : neighbour trials per temperature level
        archive_max_size : max Pareto archive size
        seed             : random seed
        verbose          : print progress

    Returns:
        pareto_front : list of dicts, each with:
            'assignment' : Dict[router_id, List[tsv_id]]
            'obj1'       : elevator utilisation variance
            'obj2'       : average inter-layer distance
    """
    rng         = random.Random(seed)
    num_routers = dimX * dimY * dimZ
    E           = len(tsv_ids)

    if E == 0:
        raise ValueError("No TSV IDs provided — cannot run elevator assignment.")
    if eset_min > E:
        eset_min = E
        eset_max = E

    # ── Build router-level traffic matrix once ────────────────────────────
    F = _build_router_traffic(mapping, comm_matrix, num_routers)

    # ── Initialise ────────────────────────────────────────────────────────
    cur_asgn  = _init_assignment(num_routers, tsv_ids, eset_min, eset_max, rng)
    obj1, obj2 = _compute_objectives(cur_asgn, tsv_ids, F, dimX, dimY, dimZ)
    cur_sol   = {'assignment': cur_asgn, 'obj1': obj1, 'obj2': obj2}
    archive   = [cur_sol.copy()]

    temp      = initial_temp
    num_temps = math.ceil(
        math.log(final_temp / initial_temp) / math.log(cooling_rate)
    )

    if verbose:
        print(f"\n{'─'*60}")
        print(f"AMOSA Elevator-Subset Assignment")
        print(f"  Mesh         : {dimX}×{dimY}×{dimZ}  ({num_routers} routers)")
        print(f"  TSVs (E)     : {E}  →  {tsv_ids}")
        print(f"  Subset size  : {eset_min}–{eset_max} per router")
        print(f"  Temperature  : {initial_temp} → {final_temp} "
              f"(×{cooling_rate}, ~{num_temps} steps)")
        print(f"  Steps/temp   : {steps_per_temp}")
        print(f"{'─'*60}")

    # ── Main AMOSA loop ───────────────────────────────────────────────────
    iteration = 0
    while temp > final_temp:
        for _ in range(steps_per_temp):
            # Generate neighbour assignment
            nbr_asgn  = _generate_neighbour(
                cur_asgn, tsv_ids, eset_min, eset_max, rng
            )
            n1, n2    = _compute_objectives(
                nbr_asgn, tsv_ids, F, dimX, dimY, dimZ
            )
            nbr_sol   = {'assignment': nbr_asgn, 'obj1': n1, 'obj2': n2}

            new_objs  = (n1, n2)

            # Count archive solutions that dominate the neighbour
            dom_sols  = [
                s for s in archive
                if _dominates((s['obj1'], s['obj2']), new_objs)
            ]

            if not dom_sols:
                # Neighbour is non-dominated — accept and update archive
                archive   = _update_archive(archive, nbr_sol, archive_max_size)
                cur_asgn  = nbr_asgn
                cur_sol   = nbr_sol
            else:
                # Dominated — accept with SA probability
                avg_delta = sum(
                    abs(s['obj1'] - n1) + abs(s['obj2'] - n2)
                    for s in dom_sols
                ) / len(dom_sols)
                prob = math.exp(-avg_delta / max(temp, 1e-10))
                if rng.random() < prob:
                    cur_asgn = nbr_asgn
                    cur_sol  = nbr_sol

        temp      *= cooling_rate
        iteration += 1

        if verbose and iteration % max(num_temps // 10, 1) == 0:
            print(f"  Temp {temp:7.3f}  |  Archive: {len(archive):3d}  "
                  f"|  Best variance: {min(s['obj1'] for s in archive):.6f}  "
                  f"Best dist: {min(s['obj2'] for s in archive):.4f}")

    # Sort by obj1 (utilisation variance)
    archive.sort(key=lambda s: s['obj1'])

    if verbose:
        print(f"\n{'─'*60}")
        print(f"AMOSA complete — {len(archive)} Pareto-optimal assignments")
        print(f"{'─'*60}")
        print(f"{'#':>3}  {'Util Variance':>15}  {'Avg Distance':>14}")
        for idx, sol in enumerate(archive):
            print(f"{idx:>3}  {sol['obj1']:>15.6f}  {sol['obj2']:>14.4f}")
        print(f"{'─'*60}")

    return archive