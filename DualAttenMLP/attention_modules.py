"""
attention_modules.py
────────────────────
Two attention components that together implement the DAN decision-making
network, adapted for dependency-free NoC IP-core mapping.


"""

import torch
import torch.nn as nn
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Task Ordering Attention  (Eq. 21 / first attention layer)
# ─────────────────────────────────────────────────────────────────────────────

class TaskOrderingAttention(nn.Module):
    """
    GRU-based pointer-network attention that scores each unscheduled
    IP core for selection at the current mapping step.

    The GRU lives INSIDE this module — it owns and updates its own
    recurrent state. The external mapping loop no longer needs to
    manage the GRU cell separately.

    At each sequential step:
        1. forward()        — score all unscheduled cores using current context
        2. update_context() — feed placement event into GRU to update context

    Score for core i  (Bahdanau additive attention):
        score_i = v · tanh( W_ctx · context  +  W_task · h_i )

    Args:
        embed_dim : embedding dimension D
    """

    def __init__(self, embed_dim: int):
        super().__init__()

        # ── GRU — owns recurrent state ────────────────────────────────────
        # Input: [h_i | m_j]  concatenation of placed core + assigned router
        # Hidden: context [1, D]
        self.gru = nn.GRUCell(
            input_size  = embed_dim * 2,   # h_i[D] concat m_j[D]
            hidden_size = embed_dim,
        )

        # ── Additive attention weights ────────────────────────────────────
        self.W_ctx  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_task = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v      = nn.Linear(embed_dim, 1,         bias=False)

    def init_context(self, h_mean: torch.Tensor) -> torch.Tensor:
        """
        Initialise GRU hidden state from the mean core embedding.
        Called once before the sequential loop begins.

        Args:
            h_mean : [1, D]  mean of all core GAT embeddings

        Returns:
            context : [1, D]  initial GRU hidden state
        """
        return h_mean.clone()

    def update_context(
        self,
        context : torch.Tensor,   # [1, D]  current GRU hidden state
        h_i     : torch.Tensor,   # [1, D]  embedding of placed core
        m_j     : torch.Tensor,   # [1, D]  embedding of assigned router
    ) -> torch.Tensor:
        """
        Update GRU context after a (core, router) placement decision.
        Called AFTER each step's core and router have been selected.

        Args:
            context : [1, D]  previous GRU hidden state
            h_i     : [1, D]  embedding of core that was just placed
            m_j     : [1, D]  embedding of router it was assigned to

        Returns:
            new_context : [1, D]  updated GRU hidden state
        """
        step_vec    = torch.cat([h_i, m_j], dim=-1)   # [1, 2D]
        new_context = self.gru(step_vec, context)      # [1, D]
        return new_context

    def forward(
        self,
        h_all          : torch.Tensor,   # [num_cores, D]
        context        : torch.Tensor,   # [1,         D]  current GRU state
        scheduled_mask : torch.Tensor,   # [num_cores] bool
    ) -> torch.Tensor:
        """
        Score all unscheduled cores using current GRU context.

        Args:
            h_all          : all core embeddings from GAT encoder  [N, D]
            context        : current GRU hidden state              [1, D]
            scheduled_mask : True for already-placed cores         [N]

        Returns:
            logits : [num_cores, 1]  (-inf for already-scheduled cores)
        """
        ctx    = self.W_ctx(context)              # [1, D]  broadcasts over N
        keys   = self.W_task(h_all)               # [N, D]
        scores = self.v(torch.tanh(ctx + keys))   # [N, 1]
        scores = scores.masked_fill(
            scheduled_mask.unsqueeze(1), float('-inf')
        )
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Router Scorer MLP  (Eq. 23 / second attention layer)
# ─────────────────────────────────────────────────────────────────────────────

class RouterScorerMLP(nn.Module):
    """
    MLP that scores every router for the currently selected IP core.

    Input feature vector per (core i, router j) pair:
        v_{i,j} = [ h_i | m_j | h_mean | m_mean | status_j ]

    where:
        h_i      = selected core embedding           [embed_dim]
        m_j      = router j embedding                [embed_dim]
        h_mean   = mean over all core embeddings     [embed_dim]
        m_mean   = mean over all router embeddings   [embed_dim]
        status_j = 1 if router j is occupied, else 0 [1]

    Occupied routers are masked to −inf so softmax assigns them 0
    probability.

    Args:
        embed_dim  : embedding dimension
        hidden_dim : MLP hidden width (default = embed_dim)
    """

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        input_dim  = embed_dim * 4 + 1        # 4 embeddings + 1 status bit

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_i            : torch.Tensor,  # [1,           embed_dim]
        m_all          : torch.Tensor,  # [num_routers, embed_dim]
        h_mean         : torch.Tensor,  # [1,           embed_dim]
        m_mean         : torch.Tensor,  # [1,           embed_dim]
        router_occupied: torch.Tensor,  # [num_routers] bool
    ) -> torch.Tensor:
        """
        Returns:
            logits : [num_routers, 1]  (−inf for occupied routers)
        """
        R = m_all.size(0)

        # Expand scalars to match router dimension
        h_i_exp    = h_i.expand(R,    -1)   # [R, D]
        h_mean_exp = h_mean.expand(R, -1)   # [R, D]
        m_mean_exp = m_mean.expand(R, -1)   # [R, D]
        status     = router_occupied.float().unsqueeze(1)   # [R, 1]

        v      = torch.cat(
            [h_i_exp, m_all, h_mean_exp, m_mean_exp, status], dim=-1
        )                                    # [R, 4D+1]
        logits = self.mlp(v)                 # [R, 1]
        logits = logits.masked_fill(
            router_occupied.unsqueeze(1), float('-inf')
        )
        return logits