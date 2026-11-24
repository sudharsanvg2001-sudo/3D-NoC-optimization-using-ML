"""
mapping_network.py
──────────────────
Complete Dual-GAT Mapping Network for NoC IP-core → router assignment.

Adapts the DAN from the paper to the IP-core mapping problem:

  Paper                          │  This implementation
  ─────────────────────────────  │  ──────────────────────────────────
  MPNN graph encoder             │  GAT encoder (your existing code)
  LSTM encoder-decoder           │  GRU context tracker (no LSTM)
  Task-dependency masking        │  No masking (no dependencies)
  Tasks → PEs (many-to-one)      │  IP cores → Routers (one-to-one)

Sequential decision loop per step i:
  1. TaskOrderingAttention.forward()        → score + select next core
  2. RouterScorerMLP.forward()              → score + select router
  3. TaskOrderingAttention.update_context() → GRU updates context
  4. Mark router as occupied

Paper reference: Section 3.2, Algorithm 1
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import List, Optional, Tuple

from gat_model         import FeatureEncoder
from attention_modules import TaskOrderingAttention, RouterScorerMLP


class DualGATMappingNetwork(nn.Module):
    """
    Sequential IP-core mapping network.

    Args:
        core_feat_dim  : node feature dim of the IP core (communication) graph
        topo_feat_dim  : node feature dim of the NoC topology graph
        embed_dim      : internal embedding dimension            (default 128)
        core_edge_dim  : edge feature dim for core graph         (None = none)
        topo_edge_dim  : edge feature dim for topology graph     (None = none)
    """

    def __init__(
        self,
        core_feat_dim : int,
        topo_feat_dim : int,
        embed_dim     : int = 128,
        core_edge_dim : Optional[int] = None,
        topo_edge_dim : Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ── Encoders ─────────────────────────────────────────────────────
        self.core_encoder = FeatureEncoder(
            core_feat_dim, embed_dim, edge_dim=core_edge_dim
        )
        self.topo_encoder = FeatureEncoder(
            topo_feat_dim, embed_dim, edge_dim=topo_edge_dim
        )

        # ── Ordering: which IP core to place next ─────────────────────────
        # GRU lives inside TaskOrderingAttention — owns its own recurrent state
        self.task_ordering = TaskOrderingAttention(embed_dim)

        # ── Assignment: which router to use ──────────────────────────────
        self.router_scorer = RouterScorerMLP(embed_dim)

    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        core_data : Data,
        topo_data : Data,
        greedy    : bool = False,
    ) -> Tuple[List[Tuple[int, int]], torch.Tensor, torch.Tensor]:
        """
        Run the full sequential mapping.

        Args:
            core_data : PyG Data for the IP-core communication graph
                        .x          [num_cores,   core_feat_dim]
                        .edge_index [2, E_core]
                        .edge_attr  [E_core, core_edge_dim]  (optional)

            topo_data : PyG Data for the NoC topology graph
                        .x          [num_routers, topo_feat_dim]
                        .edge_index [2, E_topo]

            greedy    : True  → argmax decisions (inference / evaluation)
                        False → multinomial sampling   (training)

        Returns:
            mapping          : list of (core_idx, router_idx) in order of placement
            task_log_probs   : [num_cores]  log p(core ordering choices)
            router_log_probs : [num_cores]  log p(router assignment choices)

        Note:
            log p(π|s) = sum(task_log_probs + router_log_probs)   [Eq. 26]
        """
        device      = core_data.x.device
        num_cores   = core_data.x.size(0)
        num_routers = topo_data.x.size(0)

        assert num_cores <= num_routers, (
            f"Cannot map {num_cores} cores to {num_routers} routers — "
            "need at least as many routers as cores."
        )

        # ── 1. Encode both graphs ─────────────────────────────────────────
        core_edge_attr = getattr(core_data, 'edge_attr', None)
        topo_edge_attr = getattr(topo_data, 'edge_attr', None)

        h_all = self.core_encoder(
            core_data.x, core_data.edge_index, core_edge_attr
        )  # [num_cores,   embed_dim]
        m_all = self.topo_encoder(
            topo_data.x, topo_data.edge_index, topo_edge_attr
        )  # [num_routers, embed_dim]

        # Global context vectors (mean embeddings, updated at each step)
        h_mean = h_all.mean(0, keepdim=True)   # [1, D]
        m_mean = m_all.mean(0, keepdim=True)   # [1, D]

        # ── 2. Initialise tracking state ──────────────────────────────────
        context         = self.task_ordering.init_context(h_mean)  # [1, D]

        # Use requires_grad=False explicitly — these are control tensors,
        # never part of the computation graph
        scheduled_mask  = torch.zeros(num_cores,   dtype=torch.bool,
                                      device=device)
        router_occupied = torch.zeros(num_routers, dtype=torch.bool,
                                      device=device)

        mapping          : List[Tuple[int, int]] = []
        task_log_probs   : List[torch.Tensor]    = []
        router_log_probs : List[torch.Tensor]    = []

        # ── 3. Sequential placement ───────────────────────────────────────
        for _ in range(num_cores):

            # ── 3a. Select which IP core to place next ────────────────────
            # Clone mask before passing — prevents autograd inplace conflict
            task_logits = self.task_ordering(
                h_all, context, scheduled_mask.clone()
            )
            task_probs  = F.softmax(task_logits, dim=0)    # [num_cores, 1]

            if greedy:
                core_idx = int(task_probs.argmax().item())
            else:
                core_idx = int(
                    torch.multinomial(
                        task_probs.squeeze(1).clamp(min=1e-10),
                        num_samples=1,
                    ).item()
                )

            task_log_probs.append(
                torch.log(task_probs[core_idx].squeeze().clamp(min=1e-10))
            )
            # Update tracking mask AFTER extracting log_prob
            scheduled_mask = scheduled_mask.clone()
            scheduled_mask[core_idx] = True

            # ── 3b. Select which router to assign to this core ────────────
            h_i = h_all[core_idx].unsqueeze(0)   # [1, D]

            # Clone mask before passing — prevents autograd inplace conflict
            router_logits = self.router_scorer(
                h_i, m_all, h_mean, m_mean, router_occupied.clone()
            )
            router_probs = F.softmax(router_logits, dim=0)

            if greedy:
                router_idx = int(router_probs.argmax().item())
            else:
                router_idx = int(
                    torch.multinomial(
                        router_probs.squeeze(1).clamp(min=1e-10),
                        num_samples=1,
                    ).item()
                )

            router_log_probs.append(
                torch.log(router_probs[router_idx].squeeze().clamp(min=1e-10))
            )
            # Update tracking mask AFTER extracting log_prob
            router_occupied = router_occupied.clone()
            router_occupied[router_idx] = True
            mapping.append((core_idx, router_idx))

            # ── 3c. Update GRU context via TaskOrderingAttention ──────────
            m_j     = m_all[router_idx].unsqueeze(0)
            context = self.task_ordering.update_context(
                context, h_i, m_j
            )

        # ── 4. Stack and return ───────────────────────────────────────────
        return (
            mapping,
            torch.stack(task_log_probs),    # [num_cores]
            torch.stack(router_log_probs),  # [num_cores]
        )

    # ─────────────────────────────────────────────────────────────────────
    def solution_log_prob(
        self,
        task_log_probs  : torch.Tensor,
        router_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scalar log p(π|s) = Σ_i [ log p(core_i) + log p(router_i | core_i) ]
        Used as the differentiable term in REINFORCE (Eq. 26 in paper).
        """
        return (task_log_probs + router_log_probs).sum()