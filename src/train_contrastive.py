"""Task 4 training: contrastive GNN-BERT alignment on MusicCaps (Algorithm 4).

    for each step:
        g_i   <- Normalize(GNN(G_i))
        t_i   <- Normalize(BERT_CLS(caption_i))
        S_ij  <- g_i^T t_j / tau
        L_NCE <- -(1/N) sum_i log( exp(S_ii) / sum_j exp(S_ij) )
    evaluate R@1, R@5, R@10 on held-out pairs

After training it also produces the three evaluation deliverables: the retrieval
table, 10 qualitative caption -> clip examples (plus a listening-study sheet),
and zero-shot tag prediction measured against the Task 3 supervised model.

    python src/train_contrastive.py --config config.yaml
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
from contrastive import DualEncoderGNNBert, count_parameters, info_nce
from data_fusion import FusionDataset, collate_fusion, load_fusion_data
from runtime import autocast_dtype, describe_precision, make_scaler
from retrieval_eval import (
    compare_with_task3,
    load_clip_times,
    print_retrieval_table,
    print_zero_shot_table,
    qualitative_examples,
    retrieval_metrics,
    write_examples_html,
    write_human_eval_sheet,
    zero_shot_scores,
    zero_shot_tag_metrics,
)


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
    """Paired clips only: contrastive training needs a caption for every graph."""
    index, graphs, label_vocab, node_dim = load_fusion_data(cfg)
    t4 = cfg["task4"]

    paired = index[(index["source"] != "deam") & index["text"].astype(str).str.strip().ne("")]
    dropped = len(index) - len(paired)
    if dropped:
        print(f"[data] dropped {dropped} unpaired rows (no caption) from the contrastive set")

    loaders, frames = {}, {}
    for split in ("train", "val", "test"):
        frame = paired[paired["split"] == split].reset_index(drop=True)
        dataset = FusionDataset(frame, graphs, label_vocab, tokenizer, t4["max_length"])
        loaders[split] = DataLoader(
            dataset,
            batch_size=t4["train"]["batch_size"],
            shuffle=(split == "train"),
            drop_last=(split == "train"),   # a size-1 final batch has no negatives
            collate_fn=collate_fusion,
            num_workers=t4["train"]["num_workers"],
        )
        frames[split] = frame

    print(f"[data] pairs: " + ", ".join(f"{k}={len(v)}" for k, v in frames.items()))
    return loaders, frames, label_vocab, node_dim


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
    moved["graph"] = batch["graph"].to(device, non_blocking=True)
    return moved


@torch.no_grad()
def encode_split(model, loader, device: torch.device, amp: bool = True) -> dict:
    """All graph and text embeddings for a split, in loader order."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    amp_dtype = autocast_dtype(device)
    graph_emb, text_emb, targets, ids = [], [], [], []

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(batch["graph"], batch["input_ids"], batch["attention_mask"])
        graph_emb.append(out["g"].float().cpu())
        text_emb.append(out["t"].float().cpu())
        targets.append(batch["y_tags"].cpu())
        ids.extend(batch["track_id"])

    return {
        "graph": torch.cat(graph_emb).numpy(),
        "text": torch.cat(text_emb).numpy(),
        "targets": torch.cat(targets).numpy(),
        "track_ids": ids,
    }


@torch.no_grad()
def encode_tag_prompts(model, tokenizer, label_vocab, device, prompt: str, max_length: int) -> np.ndarray:
    """Embed each tag as a short sentence, giving zero-shot class vectors."""
    model.eval()
    texts = [prompt.format(tag=tag) for tag in label_vocab]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
    return model.encode_text(enc["input_ids"], enc["attention_mask"]).float().cpu().numpy()


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    t4 = cfg["task4"]
    tcfg, mcfg = t4["train"], t4["model"]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(mcfg["text_model"])
    loaders, frames, label_vocab, node_dim = build_loaders(cfg, tokenizer)

    model = DualEncoderGNNBert(
        node_dim=node_dim,
        text_model=mcfg["text_model"],
        embed_dim=mcfg["embed_dim"],
        gnn_hidden=mcfg["gnn_hidden"],
        gnn_layers=mcfg["gnn_layers"],
        gnn_conv=mcfg["gnn_conv"],
        gnn_heads=mcfg["gnn_heads"],
        gnn_readout=mcfg["gnn_readout"],
        dropout=mcfg["dropout"],
        freeze_text_layers=mcfg["freeze_text_layers"],
        temperature=mcfg["temperature"],
        learnable_temperature=mcfg["learnable_temperature"],
    ).to(device)

    warm_start = t4.get("warm_start_task1_ckpt")
    if warm_start and os.path.exists(warm_start):
        model.text.load_task1_encoder(warm_start, device)

    print(f"[model] dual encoder | embed_dim={mcfg['embed_dim']} | {count_parameters(model):,} params")
    print(f"[model] in-batch negatives per step: {tcfg['batch_size'] - 1}")

    optimizer = AdamW(model.param_groups(tcfg["lr_text"], tcfg["lr_rest"], tcfg["weight_decay"]))
    total_steps = max(1, len(loaders["train"])) * tcfg["epochs"]
    scheduler = linear_warmup_schedule(optimizer, int(tcfg["warmup_ratio"] * total_steps), total_steps)
    use_amp = tcfg["amp"] and device.type == "cuda"
    amp_dtype = autocast_dtype(device)
    scaler = make_scaler(device, use_amp)
    print(f"[train] precision: {describe_precision(device, use_amp)}")

    ckpt_dir = cfg["paths"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "task4_contrastive_best.pt")

    history, best_score, best_epoch, stale = [], -1.0, -1, 0

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        running, seen, batch_acc, t0 = 0.0, 0, 0.0, time.time()

        bar = tqdm(loaders["train"], desc=f"epoch {epoch}/{tcfg['epochs']}", leave=False)
        for batch in bar:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(batch["graph"], batch["input_ids"], batch["attention_mask"])
            # The similarity matrix and its softmax are computed in fp32: fp16
            # logits over N x N with a small tau overflow easily.
            loss, parts = info_nce(
                out["g"].float(), out["t"].float(), model.temperature, tcfg["symmetric_loss"]
            )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            n = out["g"].shape[0]
            running += parts["total"] * n
            batch_acc += parts["accuracy_in_batch"] * n
            seen += n
            bar.set_postfix(loss=f"{running / max(1, seen):.4f}", acc=f"{batch_acc / max(1, seen):.3f}")

        val = encode_split(model, loaders["val"], device, tcfg["amp"])
        val_metrics = retrieval_metrics(val["graph"] @ val["text"].T, tuple(t4["recall_ks"]))
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "train_in_batch_acc": batch_acc / max(1, seen),
            "val_caption_to_audio_R@1": val_metrics["caption_to_audio"]["R@1"],
            "val_caption_to_audio_R@5": val_metrics["caption_to_audio"]["R@5"],
            "val_caption_to_audio_R@10": val_metrics["caption_to_audio"]["R@10"],
            "val_audio_to_caption_R@5": val_metrics["audio_to_caption"]["R@5"],
            "val_mean_R@5": val_metrics["mean_R@5"],
            "temperature": float(model.temperature),
            "seconds": time.time() - t0,
        }
        history.append(record)
        print(
            f"epoch {epoch:>2} | loss {record['train_loss']:.4f} in-batch acc {record['train_in_batch_acc']:.3f} "
            f"|| val C->A R@1 {record['val_caption_to_audio_R@1']:.4f} R@5 "
            f"{record['val_caption_to_audio_R@5']:.4f} R@10 {record['val_caption_to_audio_R@10']:.4f} "
            f"| {record['seconds']:.1f}s"
        )

        if record["val_mean_R@5"] > best_score:
            best_score, best_epoch, stale = record["val_mean_R@5"], epoch, 0
            torch.save(
                {"model_state": model.state_dict(), "label_vocab": label_vocab,
                 "node_dim": node_dim, "config": cfg, "epoch": epoch},
                ckpt_path,
            )
        else:
            stale += 1
            if stale >= tcfg["early_stopping_patience"]:
                print(f"[train] early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state"])
    return evaluate(cfg, model, tokenizer, loaders, frames, label_vocab, device, history, best_epoch)


# --------------------------------------------------------------------------- #
# evaluation + deliverables
# --------------------------------------------------------------------------- #
def evaluate(cfg, model, tokenizer, loaders, frames, label_vocab, device, history, best_epoch) -> dict:
    t4 = cfg["task4"]
    metrics_dir, plots_dir = cfg["paths"]["metrics_dir"], cfg["paths"]["plots_dir"]
    examples_dir = os.path.join(cfg["paths"]["results_dir"], "retrieval_examples")
    for directory in (metrics_dir, plots_dir, examples_dir):
        os.makedirs(directory, exist_ok=True)

    test = encode_split(model, loaders["test"], device, t4["train"]["amp"])
    sim = test["graph"] @ test["text"].T
    metrics = retrieval_metrics(sim, tuple(t4["recall_ks"]))

    caption_by_id = dict(zip(frames["test"]["track_id"].astype(str), frames["test"]["text"].astype(str)))
    captions = [caption_by_id.get(tid, "") for tid in test["track_ids"]]
    clip_times = load_clip_times(cfg)

    examples = qualitative_examples(
        sim, test["track_ids"], captions, clip_times,
        t4["n_examples"], t4["top_k"], cfg["seed"],
    )
    with open(os.path.join(examples_dir, "task4_retrieval_examples.json"), "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
    write_examples_html(examples, os.path.join(examples_dir, "task4_retrieval_examples.html"))
    write_human_eval_sheet(
        examples, os.path.join(examples_dir, "human_eval_sheet.csv"), t4["human_eval_raters"]
    )

    tag_emb = encode_tag_prompts(
        model, tokenizer, label_vocab, device, t4["zero_shot_prompt"], t4["max_length"]
    )
    zero_shot = zero_shot_tag_metrics(zero_shot_scores(test["graph"], tag_emb), test["targets"])
    # Text-side control: the same prompts scored against caption embeddings. It
    # bounds how much of the zero-shot signal comes from the audio graph at all.
    zero_shot_text = zero_shot_tag_metrics(zero_shot_scores(test["text"], tag_emb), test["targets"])
    comparison = compare_with_task3(zero_shot, metrics_dir)
    comparison["task4_zero_shot_from_caption"] = {
        k: zero_shot_text[k] for k in ("macro_f1", "micro_f1", "auc_pr")
    }

    results = {
        "history": history,
        "best_epoch": best_epoch,
        "retrieval": metrics,
        "zero_shot_from_graph": zero_shot,
        "zero_shot_from_caption": zero_shot_text,
        "zero_shot_vs_task3": comparison,
        "n_test_pairs": len(test["track_ids"]),
        "temperature": float(model.temperature),
    }
    with open(os.path.join(metrics_dir, "task4_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote {os.path.join(metrics_dir, 'task4_results.json')}")

    plot_training(history, os.path.join(plots_dir, "task4_retrieval_curves.png"))
    plot_similarity(sim, os.path.join(plots_dir, "task4_similarity_matrix.png"))

    print_retrieval_table(metrics)
    print_zero_shot_table(comparison)
    print(f"\nQualitative examples -> {examples_dir}")
    for i, example in enumerate(examples[: min(3, len(examples))], start=1):
        print(f"\n  Q{i}: {example['query_caption'][:120]}...")
        print(f"      correct clip ranked {example['rank_of_correct_clip']}")
        for item in example["retrieved"]:
            print(f"      #{item['rank']} {item['track_id']} {item['score']:.3f}"
                  f"{'  <-- correct' if item['is_correct'] else ''}")
    return results


def plot_training(history: list[dict], out_path: str) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="InfoNCE")
    ax2 = axes[0].twinx()
    ax2.plot(epochs, [h["train_in_batch_acc"] for h in history], "^:", color="green", label="in-batch acc")
    ax2.set_ylabel("in-batch accuracy", color="green")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Task 4 contrastive training"); axes[0].grid(alpha=0.3); axes[0].legend(loc="upper right")

    for key, style in (("val_caption_to_audio_R@1", "o-"), ("val_caption_to_audio_R@5", "s-"),
                       ("val_caption_to_audio_R@10", "^-")):
        axes[1].plot(epochs, [h[key] for h in history], style, label=key.replace("val_caption_to_audio_", "C->A "))
    axes[1].plot(epochs, [h["val_audio_to_caption_R@5"] for h in history], "d--", label="A->C R@5")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("recall")
    axes[1].set_title("Validation retrieval"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_similarity(sim: np.ndarray, out_path: str, max_n: int = 60) -> None:
    """Top-left corner of the similarity matrix; a bright diagonal means alignment."""
    n = min(max_n, sim.shape[0])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(sim[:n, :n], cmap="magma")
    ax.set_xlabel("caption index"); ax.set_ylabel("clip index")
    ax.set_title(f"graph-caption similarity (first {n} test pairs)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4: contrastive GNN-BERT retrieval.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="larger batch = harder negatives")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["task4"]["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["task4"]["train"]["batch_size"] = args.batch_size
    if args.device:
        cfg["device"] = args.device

    train(cfg)


if __name__ == "__main__":
    main()
