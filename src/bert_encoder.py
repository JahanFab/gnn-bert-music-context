"""BERT text encoder + multi-label tag head (Task 1, spec Section 4.1).

    t       = BERT_CLS(X_text)
    y_hat_k = sigmoid(w_k^T t + b_k)

The encoder is kept as a standalone module because Task 3 re-uses it: the fusion
model needs both the CLS vector `t` and the full token sequence H_text for
cross-attention, so `forward` can return either.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from runtime import autocast_dtype


def build_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


class BertTagClassifier(nn.Module):
    """Pretrained transformer encoder with a linear multi-label classification head.

    Emits raw logits -- the sigmoid lives inside `BCEWithLogitsLoss` for numerical
    stability, and is applied explicitly at inference time.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 50,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        config = AutoConfig.from_pretrained(model_name)
        # `eager` attention is required for output_attentions=True, which the
        # qualitative attention visualisation in evaluate.py depends on.
        self.encoder = AutoModel.from_pretrained(
            model_name, config=config, attn_implementation="eager"
        )
        hidden_size = config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
        return_sequence: bool = False,
    ) -> dict:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden = out.last_hidden_state            # (B, L, d)
        cls = hidden[:, 0]                        # DistilBERT has no pooler -> take [CLS]
        logits = self.classifier(self.dropout(cls))

        result = {"logits": logits, "cls": cls}
        if return_sequence:
            result["hidden_states"] = hidden      # H_text for Task 3 cross-attention
        if output_attentions:
            result["attentions"] = out.attentions  # tuple(L_layers) of (B, heads, L, L)
        return result

    def param_groups(self, lr_encoder: float, lr_head: float, weight_decay: float) -> list[dict]:
        """Discriminative learning rates: small for pretrained weights, large for the head.

        Biases and LayerNorm parameters are excluded from weight decay, as in the
        original BERT fine-tuning recipe.
        """
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")

        def split(module: nn.Module, lr: float) -> list[dict]:
            decay, plain = [], []
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                (plain if any(nd in name for nd in no_decay) else decay).append(param)
            groups = []
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
            if plain:
                groups.append({"params": plain, "lr": lr, "weight_decay": 0.0})
            return groups

        return split(self.encoder, lr_encoder) + split(self.classifier, lr_head)


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: torch.device, amp: bool = True):
    """Run the model over a dataloader, returning (logits, targets) as CPU tensors."""
    model.eval()
    all_logits, all_targets = [], []
    use_amp = amp and device.type == "cuda"
    amp_dtype = autocast_dtype(device)
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(input_ids, attention_mask)["logits"]
        all_logits.append(logits.float().cpu())
        all_targets.append(batch["labels"])
    return torch.cat(all_logits), torch.cat(all_targets)
