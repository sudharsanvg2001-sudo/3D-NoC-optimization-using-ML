"""
gat_model.py
────────────
Two-layer GAT encoder that produces node-level embeddings.
Replaces the MPNN used in the original DAN paper.

Enhanced vs original:
  • optional edge_dim for communication-volume edge features
  • add_self_loops=True for isolated nodes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional


class FeatureEncoder(nn.Module):
    """
    Two-layer Graph Attention Network encoder.

    Architecture:
        GAT(in → 64, heads=4, concat=True)  →  LeakyReLU
        GAT(256  → out_channels, heads=1,   concat=False)

    Args:
        in_channels  : input node-feature dimension
        out_channels : output embedding dimension
        edge_dim     : edge-feature dimension (None = no edge features)
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        edge_dim    : Optional[int] = None,
    ):
        super().__init__()
        self.gat1 = GATConv(
            in_channels, 64,
            heads=4, concat=True,
            edge_dim=edge_dim,
            add_self_loops=True,
        )
        self.gat2 = GATConv(
            64 * 4, out_channels,
            heads=1, concat=False,
            edge_dim=edge_dim,
            add_self_loops=True,
        )

    def forward(
        self,
        x          : torch.Tensor,
        edge_index : torch.Tensor,
        edge_attr  : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x          : [N, in_channels]
            edge_index : [2, E]
            edge_attr  : [E, edge_dim] optional
        Returns:
            embeddings : [N, out_channels]
        """
        x = F.leaky_relu(self.gat1(x, edge_index, edge_attr))
        x = self.gat2(x, edge_index, edge_attr)
        return x