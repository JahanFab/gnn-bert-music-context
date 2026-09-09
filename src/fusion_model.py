"""GNN-BERT fusion for multi-context understanding (spec Section 4.3).

Cross-attention fusion, exactly as specified:

    A = softmax( Q K^T / sqrt(d) ),   Q = g W_Q,  K = H_text W_K
    z = CONCAT( g, A H_text ),        y_hat = sigma( W z )

The graph vector g is a single query attending over the caption's token
sequence, so the model learns *which words the audio structure is talking
about*. Those attention rows are also the caption-alignment evidence for the
Task 3 case studies, so `forward` can hand them back.

Ablation modes share one classifier head and one training loop, so the ablation
table isolates the fusion mechanism rather than head capacity:

    bert_only        z = t                    (the Task 1 model)
    gnn_only         z = g                    (the Task 2 encoder)
    concat           z = CONCAT(g, t)         early fusion
    cross_attention  z = CONCAT(g, A H_text)  recommended
    gated            z = CONCAT(g, t) with a learned per-dimension gate
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from gnn_model import MusicGNN

FUSION_MODES = ("bert_only", "gnn_only", "concat", "cross_attention", "gated")


# --------------------------------------------------------------------------- #
# text branch
# --------------------------------------------------------------------------- #
class TextBranch(nn.Module):
    """BERT encoder returning both the CLS vector t and the sequence H_text."""

    def __init__(self, model_name: str = "distilbert-base-uncased", freeze_layers: int = 0):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(
            model_name, config=config, attn_implementation="eager"
        )
        self.hidden_size = config.hidden_size
        if freeze_layers:
            self.freeze_first_layers(freeze_layers)

    def freeze_first_layers(self, n: int) -> None:
        """Freeze embeddings and the first n transformer blocks.

        Spec Algorithm 3 line 6 backprops through "(partial) BERT": the lower
        layers encode generic syntax that a few thousand captions cannot improve,
        and freezing them is what keeps the fusion trainable at small batch size.
        """
        embeddings = getattr(self.encoder, "embeddings", None)
        if embeddings is not None:
            for p in embeddings.parameters():
                p.requires_grad = False

        # BERT nests blocks under .encoder.layer, DistilBERT under .transformer.layer
        blocks = None
        for attr in ("encoder", "transformer"):
            module = getattr(self.encoder, attr, None)
            if module is not None and hasattr(module, "layer"):
                blocks = module.layer
                break
        if blocks is None:
            print("[warn] could not locate transformer blocks; only embeddings frozen")
            return

        for layer in list(blocks)[:n]:
            for p in layer.parameters():
                p.requires_grad = False
        print(f"[model] froze text embeddings + first {min(n, len(blocks))}/{len(blocks)} layers")

    def forward(self, input_ids, attention_mask, output_attentions: bool = False) -> dict:
        out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden = out.last_hidden_state
        result = {"hidden_states": hidden, "cls": hidden[:, 0]}
        if output_attentions:
            result["text_attentions"] = out.attentions
        return result

    def load_task1_encoder(self, ckpt_path: str, device="cpu") -> None:
        """Warm-start from the fine-tuned Task 1 checkpoint.

        Task 1 already adapted this encoder to music vocabulary on the same label
        set; starting fusion from those weights instead of raw pretrained BERT is
        free and makes the BERT-only ablation row a like-for-like comparison.
        """
        state = torch.load(ckpt_path, map_location=device, weights_only=False)["model_state"]
        encoder_state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        print(f"[model] warm-started text branch from {ckpt_path} "
              f"({len(encoder_state)} tensors, {len(missing)} missing, {len(unexpected)} unexpected)")


# --------------------------------------------------------------------------- #
# cross-attention
# --------------------------------------------------------------------------- #
class CrossAttentionFusion(nn.Module):
    """Graph-as-query attention over caption tokens.

    With `heads=1` and `value_proj=False` this is literally the spec equation:
    the values are the raw H_text and A is a single softmax row per example.
    Multi-head is available but reported attention is then averaged over heads.
    """

    def __init__(self, graph_dim: int, text_dim: int, dim: int = 256, heads: int = 1,
                 dropout: float = 0.1, value_proj: bool = False):
        super().__init__()
        if dim % heads:
            raise ValueError(f"fusion dim {dim} must be divisible by heads {heads}")

        self.heads = heads
        self.head_dim = dim // heads
        self.value_proj = value_proj or heads > 1

        self.w_q = nn.Linear(graph_dim, dim)
        self.w_k = nn.Linear(text_dim, dim)
        self.w_v = nn.Linear(text_dim, dim) if self.value_proj else None
        self.out_dim = dim if self.value_proj else text_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, g: torch.Tensor, h_text: torch.Tensor, attention_mask: torch.Tensor):
        b, seq_len, _ = h_text.shape

        q = self.w_q(g).view(b, self.heads, 1, self.head_dim)
        k = self.w_k(h_text).view(b, seq_len, self.heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)).squeeze(2) / math.sqrt(self.head_dim)  # (B, heads, L)

        # Padding tokens must not receive attention mass, or short captions leak
        # their padding into the fused vector.
        scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, torch.finfo(scores.dtype).min)
        attn = self.dropout(F.softmax(scores, dim=-1))

        if self.value_proj:
            v = self.w_v(h_text).view(b, seq_len, self.heads, self.head_dim).transpose(1, 2)
            context = (attn.unsqueeze(2) @ v).squeeze(2).reshape(b, -1)
        else:
            context = (attn.mean(dim=1).unsqueeze(1) @ h_text).squeeze(1)

        return context, attn.mean(dim=1)  # (B, out_dim), (B, L) averaged over heads


# --------------------------------------------------------------------------- #
# fusion model
# --------------------------------------------------------------------------- #
class GNNBertFusion(nn.Module):
    """End-to-end model: GNN structure branch + BERT text branch + fused heads."""

    def __init__(
        self,
        node_dim: int,
        num_tags: int,
        text_model: str = "distilbert-base-uncased",
        mode: str = "cross_attention",
        gnn_hidden: int = 128,
        gnn_layers: int = 3,
        gnn_conv: str = "sage",
        gnn_heads: int = 4,
        gnn_readout: str = "mean",
        fusion_dim: int = 256,
        fusion_heads: int = 1,
        dropout: float = 0.3,
        freeze_text_layers: int = 0,
        predict_emotion: bool = True,
    ):
        super().__init__()
        if mode not in FUSION_MODES:
            raise ValueError(f"unknown fusion mode {mode!r}, expected one of {FUSION_MODES}")
        self.mode = mode
        self.predict_emotion = predict_emotion

        self.gnn = None
        self.text = None
        graph_dim = text_dim = 0

        if mode != "bert_only":
            self.gnn = MusicGNN(
                in_dim=node_dim,
                num_classes=None,          # pure encoder: no internal head
                hidden=gnn_hidden,
                layers=gnn_layers,
                conv=gnn_conv,
                heads=gnn_heads,
                dropout=dropout,
                readout=gnn_readout,
            )
            graph_dim = self.gnn.out_dim

        if mode != "gnn_only":
            self.text = TextBranch(text_model, freeze_text_layers)
            text_dim = self.text.hidden_size

        self.cross_attention = None
        self.gate = None
        if mode == "cross_attention":
            self.cross_attention = CrossAttentionFusion(
                graph_dim, text_dim, fusion_dim, fusion_heads, dropout
            )
            z_dim = graph_dim + self.cross_attention.out_dim
        elif mode == "concat":
            z_dim = graph_dim + text_dim
        elif mode == "gated":
            self.gate = nn.Sequential(nn.Linear(graph_dim + text_dim, text_dim), nn.Sigmoid())
            z_dim = graph_dim + text_dim
        elif mode == "bert_only":
            z_dim = text_dim
        else:  # gnn_only
            z_dim = graph_dim

        self.z_dim = z_dim
        self.tag_head = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Dropout(dropout),
            nn.Linear(z_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_tags),
        )
        # tanh keeps valence/arousal inside the [-1, 1] range the DEAM targets
        # were rescaled to, so the MSE term cannot be driven by unbounded outputs.
        self.emotion_head = (
            nn.Sequential(nn.Linear(z_dim, fusion_dim // 2), nn.ReLU(inplace=True),
                          nn.Linear(fusion_dim // 2, 2), nn.Tanh())
            if predict_emotion
            else None
        )

    def fuse(self, graph_batch=None, input_ids=None, attention_mask=None, return_attention: bool = False):
        """Produce the fused representation z (and the attention row, if asked)."""
        g = self.gnn.graph_embedding(graph_batch) if self.gnn is not None else None
        text_out = (
            self.text(input_ids, attention_mask) if self.text is not None else None
        )
        t = text_out["cls"] if text_out is not None else None

        attn = None
        if self.mode == "bert_only":
            z = t
        elif self.mode == "gnn_only":
            z = g
        elif self.mode == "concat":
            z = torch.cat([g, t], dim=1)
        elif self.mode == "gated":
            gate = self.gate(torch.cat([g, t], dim=1))
            z = torch.cat([g, gate * t], dim=1)
        else:
            context, attn = self.cross_attention(g, text_out["hidden_states"], attention_mask)
            z = torch.cat([g, context], dim=1)

        result = {"z": z, "g": g, "t": t}
        if return_attention:
            result["attention"] = attn
        return result

    def forward(self, graph_batch=None, input_ids=None, attention_mask=None, return_attention: bool = False):
        out = self.fuse(graph_batch, input_ids, attention_mask, return_attention)
        out["tag_logits"] = self.tag_head(out["z"])
        if self.emotion_head is not None:
            out["emotion"] = self.emotion_head(out["z"])
        return out

    def param_groups(self, lr_text: float, lr_rest: float, weight_decay: float) -> list[dict]:
        """Pretrained text weights get a small LR; the freshly initialised rest a large one."""
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


def multitask_loss(
    outputs: dict,
    batch: dict,
    alpha: float = 0.0,
    beta: float = 0.0,
    pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """L = L_tags + alpha ||v - v_hat||^2 + beta ||a - a_hat||^2 (spec Section 4.3).

    Both terms are masked per sample: MusicCaps rows carry tags but no emotion
    targets, DEAM rows the reverse. Averaging over the masked subset (rather than
    the whole batch) keeps each term's scale independent of how the batch happened
    to be composed.
    """
    tag_mask = batch["tag_mask"]
    per_tag = F.binary_cross_entropy_with_logits(
        outputs["tag_logits"], batch["y_tags"], pos_weight=pos_weight, reduction="none"
    )
    tag_loss = (per_tag.mean(dim=1) * tag_mask).sum() / tag_mask.sum().clamp(min=1.0)

    parts = {"tag_loss": float(tag_loss.detach())}
    loss = tag_loss

    if "emotion" in outputs and (alpha > 0 or beta > 0):
        emotion_mask = batch["emotion_mask"]
        squared = (outputs["emotion"] - batch["y_emotion"]) ** 2
        denom = emotion_mask.sum().clamp(min=1.0)
        valence_loss = (squared[:, 0] * emotion_mask).sum() / denom
        arousal_loss = (squared[:, 1] * emotion_mask).sum() / denom
        loss = loss + alpha * valence_loss + beta * arousal_loss
        parts["valence_mse"] = float(valence_loss.detach())
        parts["arousal_mse"] = float(arousal_loss.detach())

    parts["total"] = float(loss.detach())
    return loss, parts


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
