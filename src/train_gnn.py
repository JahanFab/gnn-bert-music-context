"""Task 2 training: GNN on music structure graphs vs. CNN mel-spectrogram baseline.

Implements spec Algorithm 2 (graphs are pre-built by graph_builder.py, so the
loop here covers lines 5-14) and the required B2 comparison. Both models share
the splits, the epoch budget, the optimiser family and the metric code, so the
comparison isolates the representation rather than the training setup.

    python src/train_gnn.py --config config.yaml --model gnn
    python src/train_gnn.py --config config.yaml --model cnn
    python src/train_gnn.py --config config.yaml --model both --conv gat
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
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from audio_datasets import load_config, load_index
from cnn_baseline import MelCNN, MelSpectrogramDataset
from gnn_model import MusicGNN, count_parameters
from metrics import confusion, multiclass_scores, per_class_report
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


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_graph_loaders(cfg: dict):
    from torch_geometric.loader import DataLoader as GeoDataLoader

    t2 = cfg["task2"]
    tag = f"{t2['dataset'].lower()}_{t2['graph']['type']}"
    path = os.path.join(cfg["paths"]["processed_dir"], "graphs", f"task2_graphs_{tag}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"graph cache not found: {path}\nrun `python src/graph_builder.py` first")

    blob = torch.load(path, map_location="cpu", weights_only=False)
    graphs, classes = blob["graphs"], blob["classes"]
    bs = t2["train"]["batch_size"]

    loaders = {
        "train": GeoDataLoader(graphs["train"], batch_size=bs, shuffle=True),
        "val": GeoDataLoader(graphs["val"], batch_size=bs, shuffle=False),
        "test": GeoDataLoader(graphs["test"], batch_size=bs, shuffle=False),
    }
    node_dim = int(graphs["train"][0].x.shape[1])
    print(f"[data] graphs: " + ", ".join(f"{k}={len(v)}" for k, v in graphs.items()))
    return loaders, classes, node_dim


def load_cnn_loaders(cfg: dict):
    t2 = cfg["task2"]
    df, classes = load_index(cfg)
    cache_dir = os.path.join(cfg["paths"]["processed_dir"], "audio_cache")
    n_frames = t2["cnn"]["n_frames"]
    bs = t2["train"]["batch_size"]

    loaders = {}
    for split in ("train", "val", "test"):
        frame = df[df["split"] == split].reset_index(drop=True)
        dataset = MelSpectrogramDataset(
            frame, cache_dir, n_frames=n_frames, train=(split == "train"), seed=cfg["seed"]
        )
        loaders[split] = TorchDataLoader(
            dataset,
            batch_size=bs,
            shuffle=(split == "train"),
            num_workers=t2["train"]["num_workers"],
            pin_memory=True,
        )
    print(f"[data] mel crops: " + ", ".join(f"{k}={len(v.dataset)}" for k, v in loaders.items()))
    return loaders, classes


def forward_batch(model, batch, device, kind: str):
    """Unify the PyG `Batch` and the dict-style CNN batch behind one call."""
    if kind == "gnn":
        batch = batch.to(device, non_blocking=True)
        return model(batch), batch.y.view(-1)
    x = batch["x"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)
    return model(x), y


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_split(model, loader, criterion, device, kind: str, amp: bool) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    logits_all, targets_all, total_loss, n = [], [], 0.0, 0
    use_amp = amp and device.type == "cuda"
    amp_dtype = autocast_dtype(device)

    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, targets = forward_batch(model, batch, device, kind)
            loss = criterion(logits.float(), targets)
        total_loss += loss.item() * targets.size(0)
        n += targets.size(0)
        logits_all.append(logits.float().cpu())
        targets_all.append(targets.cpu())

    logits = torch.cat(logits_all).numpy()
    targets = torch.cat(targets_all).numpy()
    scores = multiclass_scores(logits, targets)
    scores["loss"] = total_loss / max(1, n)
    return scores, logits, targets


def train_model(model, loaders, device, cfg: dict, kind: str, classes: list[str]) -> dict:
    tcfg = cfg["task2"]["train"]
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
    optimizer = AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])
    use_amp = tcfg["amp"] and device.type == "cuda"
    amp_dtype = autocast_dtype(device)
    scaler = make_scaler(device, use_amp)
    print(f"[train] precision: {describe_precision(device, use_amp)}")

    ckpt_dir = cfg["paths"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"task2_{kind}_best.pt")

    print(f"[train] {kind}: {count_parameters(model):,} trainable parameters")
    history, best_f1, best_epoch, stale = [], -1.0, -1, 0

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        running_loss, seen, correct = 0.0, 0, 0
        t0 = time.time()

        bar = tqdm(loaders["train"], desc=f"[{kind}] epoch {epoch}/{tcfg['epochs']}", leave=False)
        for batch in bar:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, targets = forward_batch(model, batch, device, kind)
                loss = criterion(logits.float(), targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()

            bs = targets.size(0)
            running_loss += loss.item() * bs
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += bs
            bar.set_postfix(loss=f"{running_loss / max(1, seen):.4f}", acc=f"{correct / max(1, seen):.3f}")

        scheduler.step()
        val_scores, _, _ = evaluate_split(model, loaders["val"], criterion, device, kind, tcfg["amp"])
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "train_accuracy": correct / max(1, seen),
            "val_loss": val_scores["loss"],
            "val_accuracy": val_scores["accuracy"],
            "val_macro_f1": val_scores["macro_f1"],
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        history.append(record)
        print(
            f"[{kind}] epoch {epoch:>3} | train loss {record['train_loss']:.4f} "
            f"acc {record['train_accuracy']:.4f} || val loss {record['val_loss']:.4f} "
            f"acc {record['val_accuracy']:.4f} macro-F1 {record['val_macro_f1']:.4f} "
            f"| {record['seconds']:.1f}s"
        )

        if record["val_macro_f1"] > best_f1:
            best_f1, best_epoch, stale = record["val_macro_f1"], epoch, 0
            torch.save(
                {"model_state": model.state_dict(), "classes": classes, "config": cfg, "epoch": epoch},
                ckpt_path,
            )
        else:
            stale += 1
            if stale >= tcfg["early_stopping_patience"]:
                print(f"[{kind}] early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    # Restore the best checkpoint before scoring the test split.
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state"])
    test_scores, test_logits, test_targets = evaluate_split(
        model, loaders["test"], criterion, device, kind, tcfg["amp"]
    )

    return {
        "kind": kind,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "test": test_scores,
        "per_class": per_class_report(test_logits, test_targets, classes),
        "confusion": confusion(test_logits, test_targets, len(classes)).tolist(),
        "checkpoint": ckpt_path,
        "parameters": count_parameters(model),
    }


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_curves(results: dict[str, dict], out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for kind, res in results.items():
        epochs = [h["epoch"] for h in res["history"]]
        axes[0].plot(epochs, [h["val_accuracy"] for h in res["history"]], "o-", label=f"{kind} val acc")
        axes[0].plot(epochs, [h["train_accuracy"] for h in res["history"]], "--", alpha=0.6, label=f"{kind} train acc")
        axes[1].plot(epochs, [h["val_macro_f1"] for h in res["history"]], "o-", label=f"{kind} val Macro-F1")

    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy")
    axes[0].set_title("Task 2: accuracy vs. epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Validation Macro-F1"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_confusion(matrix: list[list[int]], classes: list[str], title: str, out_path: str) -> None:
    cm = np.asarray(matrix, dtype=np.float32)
    normalized = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(classes), 1 + 0.55 * len(classes)))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(classes)), labels=classes, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)

    for i in range(len(classes)):
        for j in range(len(classes)):
            if normalized[i, j] > 0.01:
                ax.text(
                    j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if normalized[i, j] > 0.5 else "black",
                )

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run(cfg: dict, which: str) -> dict:
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    t2 = cfg["task2"]
    results: dict[str, dict] = {}

    if which in ("gnn", "both"):
        loaders, classes, node_dim = load_graph_loaders(cfg)
        mcfg = t2["model"]
        model = MusicGNN(
            in_dim=node_dim,
            num_classes=len(classes),
            hidden=mcfg["hidden"],
            layers=mcfg["layers"],
            conv=mcfg["conv"],
            heads=mcfg["heads"],
            dropout=mcfg["dropout"],
            readout=mcfg["readout"],
            residual=mcfg["residual"],
        ).to(device)
        print(f"[model] {mcfg['conv'].upper()} | node_dim={node_dim} | classes={len(classes)}")
        results[f"gnn_{mcfg['conv']}"] = train_model(model, loaders, device, cfg, "gnn", classes)
        results[f"gnn_{mcfg['conv']}"]["classes"] = classes

    if which in ("cnn", "both"):
        loaders, classes = load_cnn_loaders(cfg)
        model = MelCNN(
            num_classes=len(classes),
            channels=tuple(t2["cnn"]["channels"]),
            dropout=t2["cnn"]["dropout"],
        ).to(device)
        results["cnn_melspec"] = train_model(model, loaders, device, cfg, "cnn", classes)
        results["cnn_melspec"]["classes"] = classes

    metrics_dir, plots_dir = cfg["paths"]["metrics_dir"], cfg["paths"]["plots_dir"]
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    out_path = os.path.join(metrics_dir, "task2_results.json")
    merged = {}
    if os.path.exists(out_path):  # keep the other model's run when training one at a time
        with open(out_path, encoding="utf-8") as f:
            merged = json.load(f)
    merged.update(results)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"[eval] wrote {out_path}")

    plot_curves(merged, os.path.join(plots_dir, "task2_curves.png"))
    for name, res in results.items():
        plot_confusion(
            res["confusion"], res["classes"], f"Task 2 confusion -- {name}",
            os.path.join(plots_dir, f"task2_confusion_{name}.png"),
        )

    print_comparison(merged)
    return merged


def print_comparison(results: dict[str, dict]) -> None:
    print(f"\n{'Model':<20}{'Accuracy':>10}{'Macro-F1':>10}{'Top-3':>10}{'Params':>12}")
    print("-" * 62)
    for name, res in results.items():
        t = res["test"]
        top3 = f"{t['top3_accuracy']:.4f}" if t.get("top3_accuracy") is not None else "-"
        print(f"{name:<20}{t['accuracy']:>10.4f}{t['macro_f1']:>10.4f}{top3:>10}{res['parameters']:>12,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: GNN on music graphs vs. CNN baseline.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="both", choices=["gnn", "cnn", "both"])
    parser.add_argument("--conv", default=None, help="sage | gat | gcn")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.conv:
        cfg["task2"]["model"]["conv"] = args.conv
    if args.epochs:
        cfg["task2"]["train"]["epochs"] = args.epochs
    if args.lr:
        cfg["task2"]["train"]["lr"] = args.lr
    if args.device:
        cfg["device"] = args.device

    run(cfg, args.model)


if __name__ == "__main__":
    main()
