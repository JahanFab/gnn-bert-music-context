"""Task 3 training: GNN-BERT fusion with the multi-task objective.

Implements spec Algorithm 3:

    H_text <- BERT(X_text),  t <- H_text[CLS]
    g      <- GNN_Readout(G)
    z      <- CrossAttention(g, H_text)      (or CONCAT(g, t))
    y_hat  <- sigma(W z + b)
    L      <- L_tags + alpha * L_emotion
    backprop through fusion, GNN and (partial) BERT

The ablation runner trains every mode under an identical protocol, which is what
makes the required ablation table meaningful:

    python src/train_fusion.py --config config.yaml --mode cross_attention
    python src/train_fusion.py --config config.yaml --ablation all
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio_datasets import load_config
from data_fusion import FusionDataset, collate_fusion, load_fusion_data
from fusion_model import GNNBertFusion, count_parameters, multitask_loss
from metrics import multilabel_scores, regression_scores, tune_thresholds
from runtime import autocast_dtype, describe_precision, make_scaler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable -- falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def linear_warmup_schedule(optimizer, num_warmup: int, num_total: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup:
            return step / max(1, num_warmup)
        return max(0.0, (num_total - step) / max(1, num_total - num_warmup))

    return LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_loaders(cfg: dict, tokenizer):
    index, graphs, label_vocab, node_dim = load_fusion_data(cfg)
    t3 = cfg["task3"]
    max_length = t3["max_length"]

    loaders, sizes = {}, {}
    for split in ("train", "val", "test"):
        frame = index[index["split"] == split].reset_index(drop=True)
        dataset = FusionDataset(frame, graphs, label_vocab, tokenizer, max_length)
        loaders[split] = DataLoader(
            dataset,
            batch_size=t3["train"]["batch_size"],
            shuffle=(split == "train"),
            collate_fn=collate_fusion,
            num_workers=t3["train"]["num_workers"],
        )
        sizes[split] = len(dataset)

    print(f"[data] paired examples: {sizes} | node_dim={node_dim} | tags={len(label_vocab)}")
    return loaders, label_vocab, node_dim, index


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    moved["graph"] = batch["graph"].to(device, non_blocking=True)
    return moved


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_split(model, loader, device, cfg: dict, threshold=0.5, amp: bool = True) -> dict:
    """Tag metrics plus emotion regression metrics over one split."""
    model.eval()
    t3 = cfg["task3"]
    use_amp = amp and device.type == "cuda"
    amp_dtype = autocast_dtype(device)

    logits_all, targets_all, tag_masks = [], [], []
    emotion_pred, emotion_true, emotion_masks = [], [], []
    total_loss, n = 0.0, 0

    for batch in loader:
        batch = move_batch(batch, device)
        graph = batch["graph"] if model.mode != "bert_only" else None
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(graph, batch["input_ids"], batch["attention_mask"])
            out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
            loss, _ = multitask_loss(out, batch, t3["loss"]["alpha"], t3["loss"]["beta"])

        bs = batch["y_tags"].size(0)
        total_loss += float(loss) * bs
        n += bs
        logits_all.append(out["tag_logits"].cpu())
        targets_all.append(batch["y_tags"].cpu())
        tag_masks.append(batch["tag_mask"].cpu())
        if "emotion" in out:
            emotion_pred.append(out["emotion"].cpu())
            emotion_true.append(batch["y_emotion"].cpu())
            emotion_masks.append(batch["emotion_mask"].cpu())

    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    keep = torch.cat(tag_masks) > 0        # emotion-only rows carry no tag targets
    probs = torch.sigmoid(logits[keep]).numpy()
    y = targets[keep].numpy()

    scores = multilabel_scores(probs, y, threshold)
    scores["loss"] = total_loss / max(1, n)

    if emotion_pred:
        mask = torch.cat(emotion_masks) > 0
        if mask.any():
            scores.update(
                regression_scores(
                    torch.cat(emotion_pred)[mask].numpy(), torch.cat(emotion_true)[mask].numpy()
                )
            )
    return scores


@torch.no_grad()
def collect_probabilities(model, loader, device, amp: bool = True):
    """Sigmoid probabilities and targets for tagged rows only."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    amp_dtype = autocast_dtype(device)
    logits_all, targets_all, masks = [], [], []

    for batch in loader:
        batch = move_batch(batch, device)
        graph = batch["graph"] if model.mode != "bert_only" else None
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(graph, batch["input_ids"], batch["attention_mask"])
        logits_all.append(out["tag_logits"].float().cpu())
        targets_all.append(batch["y_tags"].cpu())
        masks.append(batch["tag_mask"].cpu())

    keep = torch.cat(masks) > 0
    return torch.sigmoid(torch.cat(logits_all)[keep]).numpy(), torch.cat(targets_all)[keep].numpy()


def train_one(cfg: dict, mode: str, loaders, label_vocab, node_dim, device) -> dict:
    t3 = cfg["task3"]
    tcfg, mcfg = t3["train"], t3["model"]
    set_seed(cfg["seed"])  # identical initialisation across ablation arms

    model = GNNBertFusion(
        node_dim=node_dim,
        num_tags=len(label_vocab),
        text_model=mcfg["text_model"],
        mode=mode,
        gnn_hidden=mcfg["gnn_hidden"],
        gnn_layers=mcfg["gnn_layers"],
        gnn_conv=mcfg["gnn_conv"],
        gnn_heads=mcfg["gnn_heads"],
        gnn_readout=mcfg["gnn_readout"],
        fusion_dim=mcfg["fusion_dim"],
        fusion_heads=mcfg["fusion_heads"],
        dropout=mcfg["dropout"],
        freeze_text_layers=mcfg["freeze_text_layers"],
        predict_emotion=t3["loss"]["alpha"] > 0 or t3["loss"]["beta"] > 0,
    ).to(device)

    warm_start = t3.get("warm_start_task1_ckpt")
    if warm_start and model.text is not None and os.path.exists(warm_start):
        model.text.load_task1_encoder(warm_start, device)

    print(f"\n=== mode: {mode} | z_dim={model.z_dim} | {count_parameters(model):,} trainable params ===")

    optimizer = AdamW(model.param_groups(tcfg["lr_text"], tcfg["lr_rest"], tcfg["weight_decay"]))
    total_steps = len(loaders["train"]) * tcfg["epochs"]
    scheduler = linear_warmup_schedule(optimizer, int(tcfg["warmup_ratio"] * total_steps), total_steps)
    use_amp = tcfg["amp"] and device.type == "cuda"
    amp_dtype = autocast_dtype(device)
    scaler = make_scaler(device, use_amp)
    print(f"[train] precision: {describe_precision(device, use_amp)}")

    ckpt_dir = cfg["paths"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"task3_{mode}_best.pt")

    history, best_f1, best_epoch, stale = [], -1.0, -1, 0
    threshold = cfg["eval"]["default_threshold"]

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()

        bar = tqdm(loaders["train"], desc=f"[{mode}] epoch {epoch}/{tcfg['epochs']}", leave=False)
        for batch in bar:
            batch = move_batch(batch, device)
            graph = batch["graph"] if mode != "bert_only" else None

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(graph, batch["input_ids"], batch["attention_mask"])
                out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
                loss, parts = multitask_loss(out, batch, t3["loss"]["alpha"], t3["loss"]["beta"])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = batch["y_tags"].size(0)
            running += parts["total"] * bs
            seen += bs
            bar.set_postfix(loss=f"{running / max(1, seen):.4f}")

        val = evaluate_split(model, loaders["val"], device, cfg, threshold, tcfg["amp"])
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "val_loss": val["loss"],
            "val_macro_f1": val["macro_f1"],
            "val_micro_f1": val["micro_f1"],
            "val_auc_pr": val["auc_pr"],
            "seconds": time.time() - t0,
        }
        for key in ("mae_valence", "mae_arousal"):
            if key in val:
                record[f"val_{key}"] = val[key]
        history.append(record)

        print(
            f"[{mode}] epoch {epoch:>2} | train {record['train_loss']:.4f} | val {record['val_loss']:.4f} "
            f"macro-F1 {record['val_macro_f1']:.4f} micro-F1 {record['val_micro_f1']:.4f} "
            f"AUC-PR {record['val_auc_pr']:.4f} | {record['seconds']:.1f}s"
        )

        if record["val_macro_f1"] > best_f1:
            best_f1, best_epoch, stale = record["val_macro_f1"], epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mode": mode,
                    "label_vocab": label_vocab,
                    "node_dim": node_dim,
                    "config": cfg,
                    "epoch": epoch,
                },
                ckpt_path,
            )
        else:
            stale += 1
            if stale >= tcfg["early_stopping_patience"]:
                print(f"[{mode}] early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state"])

    # Thresholds fitted on validation, then frozen for test -- same protocol as Task 1.
    val_probs, val_y = collect_probabilities(model, loaders["val"], device, tcfg["amp"])
    ecfg = cfg["eval"]
    thresholds = (
        tune_thresholds(
            val_probs, val_y, ecfg["threshold_grid_start"], ecfg["threshold_grid_stop"],
            ecfg["threshold_grid_step"], ecfg["default_threshold"],
        )
        if ecfg["tune_thresholds"]
        else np.full(len(label_vocab), ecfg["default_threshold"], dtype=np.float32)
    )

    test_default = evaluate_split(model, loaders["test"], device, cfg, ecfg["default_threshold"], tcfg["amp"])
    test_tuned = evaluate_split(model, loaders["test"], device, cfg, thresholds, tcfg["amp"])

    return {
        "mode": mode,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "test_default_threshold": test_default,
        "test_tuned_threshold": test_tuned,
        "parameters": count_parameters(model),
        "z_dim": model.z_dim,
        "checkpoint": ckpt_path,
    }


# --------------------------------------------------------------------------- #
# plots + reporting
# --------------------------------------------------------------------------- #
def plot_ablation(results: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for mode, res in results.items():
        epochs = [h["epoch"] for h in res["history"]]
        axes[0].plot(epochs, [h["val_macro_f1"] for h in res["history"]], "o-", label=mode)

    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("val Macro-F1")
    axes[0].set_title("Task 3 ablation: validation Macro-F1"); axes[0].legend(); axes[0].grid(alpha=0.3)

    modes = list(results)
    x = np.arange(len(modes))
    macro = [results[m]["test_tuned_threshold"]["macro_f1"] for m in modes]
    aucpr = [results[m]["test_tuned_threshold"]["auc_pr"] for m in modes]
    axes[1].bar(x - 0.2, macro, 0.4, label="Macro-F1")
    axes[1].bar(x + 0.2, aucpr, 0.4, label="AUC-PR")
    axes[1].set_xticks(x, labels=modes, rotation=20, ha="right")
    axes[1].set_title("Test performance by fusion mode"); axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def print_ablation(results: dict) -> None:
    print(f"\n{'Mode':<18}{'Macro-F1':>10}{'Micro-F1':>10}{'AUC-PR':>10}{'MAE-v':>9}{'Params':>13}")
    print("-" * 70)
    for mode, res in results.items():
        t = res["test_tuned_threshold"]
        mae = f"{t['mae_valence']:.3f}" if "mae_valence" in t else "-"
        print(
            f"{mode:<18}{t['macro_f1']:>10.4f}{t['micro_f1']:>10.4f}"
            f"{t['auc_pr']:>10.4f}{mae:>9}{res['parameters']:>13,}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: GNN-BERT fusion training and ablation.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default=None, help="cross_attention | concat | gated | bert_only | gnn_only")
    parser.add_argument("--ablation", default=None, help="'all' or a comma-separated list of modes")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["task3"]["train"]["epochs"] = args.epochs
    if args.device:
        cfg["device"] = args.device

    if args.ablation:
        modes = cfg["task3"]["ablation_modes"] if args.ablation == "all" else args.ablation.split(",")
    else:
        modes = [args.mode or cfg["task3"]["model"]["mode"]]

    device = resolve_device(cfg["device"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["task3"]["model"]["text_model"])
    loaders, label_vocab, node_dim, _ = build_loaders(cfg, tokenizer)

    results = {}
    for mode in modes:
        results[mode] = train_one(cfg, mode.strip(), loaders, label_vocab, node_dim, device)

    metrics_dir, plots_dir = cfg["paths"]["metrics_dir"], cfg["paths"]["plots_dir"]
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    out_path = os.path.join(metrics_dir, "task3_results.json")
    merged = {}
    if os.path.exists(out_path):  # accumulate across separate runs of single modes
        with open(out_path, encoding="utf-8") as f:
            merged = json.load(f)
    merged.update(results)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"[eval] wrote {out_path}")

    plot_ablation(merged, os.path.join(plots_dir, "task3_ablation.png"))
    print_ablation(merged)


if __name__ == "__main__":
    main()
