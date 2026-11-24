"""
reward.py
─────────
Mapping cost computation using Env_tsp from env.py.

Always uses elevator-first TSV routing (stage 2 in env.py's hop_cnt).
TSV locations are passed in as allowed_tsvs — required for 3D meshes.

No concept of stages — TSV routing cost is always active.
"""

from __future__ import annotations

import torch
from typing import List, Optional, Tuple

from env        import Env_tsp
from data_utils import mapping_to_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Single mapping cost
# ─────────────────────────────────────────────────────────────────────────────

def compute_mapping_cost(
    env         : Env_tsp,
    mapping     : List[Tuple[int, int]],
    num_cores   : int,
    allowed_tsvs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute Weighted Communication Cost using elevator-first TSV routing.

    Always uses stage=2 (elevator-first) in env.test_cost2().
    TSV locations must be provided for 3D meshes via allowed_tsvs.
    For 2D meshes (dimZ=1) allowed_tsvs can be None — no cross-layer routing.

    Args:
        env          : Env_tsp instance (holds demand matrix and mesh dims)
        mapping      : list of (core_idx, router_idx) from model forward()
        num_cores    : total number of IP cores
        allowed_tsvs : [K] tensor of TSV router IDs for elevator-first routing
                       None → treated as no TSVs (2D mesh or no cross-layer flows)

    Returns:
        cost : scalar float32 tensor (detached — used as RL reward signal)
    """
    mapping_tensor = mapping_to_tensor(mapping, num_cores, device=str(env.device))
    cost, _        = env.test_cost2(
        mapping      = mapping_tensor,
        allowed_tsvs = allowed_tsvs,
        stage        = 2,             # always elevator-first TSV routing
        tsv_assgn    = None,
    )
    return cost.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Batch mapping cost
# ─────────────────────────────────────────────────────────────────────────────

def batch_mapping_cost(
    env         : Env_tsp,
    mappings    : List[List[Tuple[int, int]]],
    num_cores   : int,
    allowed_tsvs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute costs for a batch of mappings of the same problem instance.

    Args:
        env          : Env_tsp instance
        mappings     : B × list of (core_idx, router_idx)
        num_cores    : total number of IP cores
        allowed_tsvs : TSV router IDs for elevator-first routing

    Returns:
        costs : [B] float32 tensor (detached)
    """
    costs = [
        compute_mapping_cost(env, m, num_cores, allowed_tsvs)
        for m in mappings
    ]
    return torch.stack(costs)   # [B]