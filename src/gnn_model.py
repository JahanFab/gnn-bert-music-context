"""GraphSAGE / GAT encoder for music structure graphs (spec Section 4.2).

GraphSAGE update, as given in the spec:

    h_i^(l+1) = sigma( W^(l) . CONCAT( h_i^(l), MEAN_{j in N(i)} h_j^(l) ) )

Graph readout (mean pooling) and classifier:

    g = (1/|V|) sum_i h_i^(L),      y_hat = sigma(W g + b)

`graph_embedding()` returns g without the classification head, which is what the
Task 3 fusion model consumes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_add_pool, global_max_pool, global_mean_pool


CONV_LAYERS = {"sage": SAGEConv, "gat": GATConv, "gcn": GCNConv}
POOLS = {"mean": global_mean_pool, "max": global_max_pool, "sum": global_add_pool}


class MusicGNN(nn.Module):
    """L-layer message-passing encoder with a pooled graph-level classifier.

    Args:
        in_dim: node feature width (83 for segment graphs, 27 for chord graphs)
        num_classes: genres for GTZAN/FMA, or tags in the multi-label setting.
            Pass None to omit the head entirely and use the module as a pure
            encoder -- that is how the Task 3 fusion model consumes it.
        conv: sage | gat | gcn
        readout: mean | max | sum | mean+max (concatenation)
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int | None,
        hidden: int = 128,
        layers: int = 3,
        conv: str = "sage",
        heads: int = 4,
        dropout: float = 0.3,
        readout: str = "mean",
        residual: bool = True,
        batch_norm: bool = True,
    ):
        super().__init__()
        if conv not in CONV_LAYERS:
            raise ValueError(f"unknown conv {conv!r}, expected one of {list(CONV_LAYERS)}")

        self.conv_type = conv
        self.readout = readout
        self.dropout = dropout
        self.residual = residual

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dim = in_dim
        for _ in range(layers):
            self.convs.append(self._make_conv(conv, dim, hidden, heads, dropout))
            self.norms.append(nn.BatchNorm1d(hidden) if batch_norm else nn.Identity())
            dim = hidden

        self.out_dim = hidden * (2 if readout == "mean+max" else 1)
        self.classifier = (
            nn.Identity()
            if num_classes is None
            else nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.out_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
        )

    @staticmethod
    def _make_conv(conv: str, in_dim: int, out_dim: int, heads: int, dropout: float) -> nn.Module:
        if conv == "gat":
            # Multi-head attention with `heads` heads of width out_dim//heads keeps
            # the layer output at exactly out_dim, so layers stay stackable.
            assert out_dim % heads == 0, "hidden must be divisible by heads for GAT"
            return GATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
        if conv == "gcn":
            return GCNConv(in_dim, out_dim)
        return SAGEConv(in_dim, out_dim, aggr="mean")

    def encode_nodes(self, x, edge_index, edge_weight=None):
        """Message passing -> final node embeddings h^(L)."""
        for conv, norm in zip(self.convs, self.norms):
            identity = x
            # Only GCNConv consumes scalar edge weights; SAGE ignores them and GAT
            # learns its own attention, so the weights are passed selectively.
            x = conv(x, edge_index, edge_weight) if self.conv_type == "gcn" else conv(x, edge_index)
            x = norm(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.residual and identity.shape == x.shape:
                x = x + identity
        return x

    def pool(self, h, batch):
        if self.readout == "mean+max":
            return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        return POOLS[self.readout](h, batch)

    def graph_embedding(self, data) -> torch.Tensor:
        """Pooled graph vector g -- the interface Task 3's fusion model calls."""
        edge_weight = data.edge_attr.squeeze(-1) if getattr(data, "edge_attr", None) is not None else None
        h = self.encode_nodes(data.x, data.edge_index, edge_weight)
        return self.pool(h, data.batch)

    def forward(self, data) -> torch.Tensor:
        """Raw class logits; softmax/sigmoid is applied by the loss function."""
        return self.classifier(self.graph_embedding(data))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
