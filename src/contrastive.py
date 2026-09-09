"""Dual-encoder GNN-BERT with InfoNCE, for Task 4 cross-modal alignment.

Spec Section 4.4 / Algorithm 4:

    g_i   = Normalize( GNN(G_i) )
    t_i   = Normalize( BERT_CLS(caption_i) )
    S_ij  = g_i^T t_j / tau
    L_NCE = -(1/N) sum_i log( exp(S_ii) / sum_j exp(S_ij) )

Both encoders project into one shared embedding space, so a caption and the clip
it describes end up near each other and retrieval is a nearest-neighbour lookup
in either direction.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion_model import TextBranch
from gnn_model import MusicGNN


class ProjectionHead(nn.Module):
    """Map a branch's native width into the shared space, then L2-normalise."""

    def __init__(self, in_dim: int, embed_dim: int, hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or max(embed_dim, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class DualEncoderGNNBert(nn.Module):
    """Graph encoder + text encoder sharing one normalised embedding space."""

    def __init__(
        self,
        node_dim: int,
        text_model: str = "distilbert-base-uncased",
        embed_dim: int = 256,
        gnn_hidden: int = 128,
        gnn_layers: int = 3,
        gnn_conv: str = "sage",
        gnn_heads: int = 4,
        gnn_readout: str = "mean",
        dropout: float = 0.1,
        freeze_text_layers: int = 0,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
    ):
        super().__init__()
        self.gnn = MusicGNN(
            in_dim=node_dim,
            num_classes=None,          # encoder only
            hidden=gnn_hidden,
            layers=gnn_layers,
            conv=gnn_conv,
            heads=gnn_heads,
            dropout=dropout,
            readout=gnn_readout,
        )
        self.text = TextBranch(text_model, freeze_text_layers)

        self.graph_proj = ProjectionHead(self.gnn.out_dim, embed_dim, dropout=dropout)
        self.text_proj = ProjectionHead(self.text.hidden_size, embed_dim, dropout=dropout)
        self.embed_dim = embed_dim

        # Fixed tau reproduces the spec equation exactly. The learnable variant is
        # CLIP's logit scale, kept optional because it changes the objective.
        self.learnable_temperature = learnable_temperature
        if learnable_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        else:
            self.register_buffer("fixed_temperature", torch.tensor(float(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        if self.learnable_temperature:
            # Clamp as in CLIP: an unbounded scale collapses training early on.
            return 1.0 / self.logit_scale.clamp(max=math.log(100.0)).exp()
        return self.fixed_temperature

    def encode_graph(self, graph_batch) -> torch.Tensor:
        return self.graph_proj(self.gnn.graph_embedding(graph_batch))

    def encode_text(self, input_ids, attention_mask) -> torch.Tensor:
        return self.text_proj(self.text(input_ids, attention_mask)["cls"])

    def forward(self, graph_batch, input_ids, attention_mask) -> dict:
        return {
            "g": self.encode_graph(graph_batch),
            "t": self.encode_text(input_ids, attention_mask),
            "temperature": self.temperature,
        }

    def param_groups(self, lr_text: float, lr_rest: float, weight_decay: float) -> list[dict]:
        text_params, other_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (text_params if name.startswith("text.") else other_params).append(param)

        groups = []
        if text_params:
            groups.append({"params": text_params, "lr": lr_text, "weight_decay": weight_decay})
        if other_params:
            groups.append({"params": other_params, "lr": lr_rest, "weight_decay": weight_decay})
        return groups


def info_nce(
    g: torch.Tensor, t: torch.Tensor, temperature: torch.Tensor | float, symmetric: bool = True
) -> tuple[torch.Tensor, dict]:
    """InfoNCE over in-batch negatives.

    Inputs are already L2-normalised, so `g @ t.T` is the cosine similarity of
    the spec's sim(u, v) = u^T v / (||u|| ||v||).

    The spec writes the graph->caption direction only. Retrieval is reported in
    both directions, so the symmetric average is the default: training one
    direction and evaluating the other measures something the loss never
    optimised. Set `symmetric: false` to reproduce the one-directional equation.
    """
    logits = g @ t.t() / temperature                 # S_ij, shape (N, N)
    targets = torch.arange(len(g), device=g.device)  # the matched pair is the diagonal

    loss_g2t = F.cross_entropy(logits, targets)
    parts = {"loss_graph_to_text": float(loss_g2t.detach())}

    if symmetric:
        loss_t2g = F.cross_entropy(logits.t(), targets)
        loss = 0.5 * (loss_g2t + loss_t2g)
        parts["loss_text_to_graph"] = float(loss_t2g.detach())
    else:
        loss = loss_g2t

    parts["total"] = float(loss.detach())
    parts["accuracy_in_batch"] = float((logits.argmax(dim=1) == targets).float().mean())
    return loss, parts


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
