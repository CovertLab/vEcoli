"""
Baseline heterogeneous bipartite message-passing GNN for one-step
(t -> t+1) forecasting on the causality network.

This is deliberately the *simple* baseline: messages are combined by a
plain sum over each node's incoming edges (no attention). See the
`AGGREGATION EXTENSION POINT` comment in `MPLayer.forward` for exactly
where to swap in a GAT/GATv2-style softmax-normalized attention
aggregation later -- see the accompanying architecture writeup for the
reasoning on when that upgrade is (and isn't) likely to matter here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EdgeMLP(nn.Module):
    def __init__(self, h: int, d_edge: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * h + d_edge, h),
            nn.ReLU(),
            nn.Linear(h, h),
        )

    def forward(self, h_src, h_dst, edge_attr):
        return self.net(torch.cat([h_src, h_dst, edge_attr], dim=-1))


class MPLayer(nn.Module):
    """One round of bidirectional message passing, sum-aggregated.

    Direction matters here because the graph's edges already encode
    reactant/product direction via the sign of the stoichiometry (and
    catalyst/regulatory edges point species -> process). We pass messages
    in both directions with *separate* learned weights per direction,
    because one-step forecasting needs information to flow both ways
    (a metabolite's next count depends on the processes consuming/
    producing it; a process's next flux depends on its reactants'
    availability).
    """

    def __init__(self, h: int, d_edge: int):
        super().__init__()
        self.fwd = EdgeMLP(h, d_edge)
        self.rev = EdgeMLP(h, d_edge)
        self.update = nn.Sequential(
            nn.Linear(3 * h, h),
            nn.ReLU(),
            nn.Linear(h, h),
        )
        self.norm = nn.LayerNorm(h)

    def forward(self, h, src, dst, edge_attr, n_nodes):
        # h: (B, N, H)
        B = h.shape[0]
        h_src = h.index_select(1, src)  # (B, E, H)
        h_dst = h.index_select(1, dst)  # (B, E, H)

        m_fwd = self.fwd(h_src, h_dst, edge_attr)  # (B, E, H)
        m_rev = self.rev(h_dst, h_src, edge_attr)  # (B, E, H)

        # ---- AGGREGATION EXTENSION POINT -----------------------------
        # Baseline: unweighted sum over each node's incoming messages.
        # This is a real limitation for high-degree "hub" nodes (e.g. ATP,
        # water -- metabolites that participate in thousands of reactions):
        # summing thousands of same-scale messages can wash out the few
        # that actually matter, and gradients through a sum over a huge
        # neighborhood can be poorly scaled. A GAT/GATv2-style replacement
        # would compute a per-edge attention logit (e.g. a small MLP over
        # [h_src, h_dst, edge_attr]), softmax-normalize it over each
        # destination node's incoming edges, and use that as a weight on
        # m_fwd/m_rev before aggregating -- turning the `index_add_` below
        # into a weighted, normalized aggregation. See the writeup for why
        # this is a plausible high-value upgrade but not a baseline
        # requirement (the stoichiometric edges already carry a
        # physically-known "attention weight" in their coefficient).
        agg_fwd = torch.zeros(
            B, n_nodes, m_fwd.shape[-1], device=h.device, dtype=h.dtype
        )
        agg_fwd.index_add_(1, dst, m_fwd)
        agg_rev = torch.zeros(
            B, n_nodes, m_rev.shape[-1], device=h.device, dtype=h.dtype
        )
        agg_rev.index_add_(1, src, m_rev)
        # ---------------------------------------------------------------

        h_new = self.update(torch.cat([h, agg_fwd, agg_rev], dim=-1))
        return self.norm(h + h_new)


class BaselineCausalityGNN(nn.Module):
    def __init__(
        self,
        n_node_types: int,
        n_node_classes: int,
        n_edge_types: int,
        d_type: int = 8,
        d_class: int = 4,
        d_edge: int = 8,
        hidden_dim: int = 16,
        n_layers: int = 2,
    ):
        super().__init__()
        self.type_emb = nn.Embedding(n_node_types, d_type)
        self.class_emb = nn.Embedding(n_node_classes, d_class)
        self.edge_type_emb = nn.Embedding(n_edge_types, d_edge - 3)

        self.in_proj = nn.Linear(1 + d_type + d_class, hidden_dim)
        self.layers = nn.ModuleList(
            [MPLayer(hidden_dim, d_edge) for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self, x_t, node_type_idx, node_class_idx, src, dst, edge_proc_idx, edge_scalar
    ):
        """
        x_t: (B, N) standardized scalar state at time t
        node_type_idx, node_class_idx: (N,) long
        src, dst, edge_proc_idx: (E,) long
        edge_scalar: (E, 3) float -- [sign, log-magnitude, has_stoichiometry]

        Returns: (B, N) predicted standardized delta for every node.
        """
        B, n_nodes = x_t.shape
        type_e = self.type_emb(node_type_idx).unsqueeze(0).expand(B, -1, -1)
        class_e = self.class_emb(node_class_idx).unsqueeze(0).expand(B, -1, -1)
        h = self.in_proj(torch.cat([x_t.unsqueeze(-1), type_e, class_e], dim=-1))

        edge_type_e = self.edge_type_emb(edge_proc_idx).unsqueeze(0).expand(B, -1, -1)
        edge_attr = torch.cat(
            [edge_type_e, edge_scalar.unsqueeze(0).expand(B, -1, -1)], dim=-1
        )

        for layer in self.layers:
            h = layer(h, src, dst, edge_attr, n_nodes)

        return self.out_proj(h).squeeze(-1)
