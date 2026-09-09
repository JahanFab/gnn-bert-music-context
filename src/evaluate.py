"""Task 1 evaluation: test metrics, baselines, F1 curves, qualitative predictions.

Produces every Task 1 deliverable listed in spec Section 4.1:

  * Macro-F1 / Micro-F1 curves vs. training epochs   -> results/plots/task1_f1_curves.png
  * Results table vs. baselines B1 / linear text     -> results/metrics/task1_test_metrics.json
  * 5 example predictions + attention visualisation  -> results/plots/task1_attention_*.png

Usage:

    python src/evaluate.py --config config.yaml
    python src/evaluate.py --config config.yaml --ckpt results/checkpoints/task1_bert_best.pt
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from bert_encoder import BertTagClassifier, build_tokenizer, predict_logits
from data_musiccaps import MusicCapsTagDataset, load_config, load_processed
from metrics import (
    majority_baseline,
    multilabel_scores,
    per_tag_report,
    random_baseline,
    tune_thresholds,
)


# --------------------------------------------------------------------------- #
# model + data loading
# --------------------------------------------------------------------------- #
def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg, label_vocab = ckpt["config"], ckpt["label_vocab"]
    model = BertTagClassifier(
        model_name=cfg["model"]["name"],
        num_labels=len(label_vocab),
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[eval] loaded {ckpt_path} (epoch {ckpt['epoch']}, val macro-F1 {ckpt['val_macro_f1']:.4f})")
    return model, cfg, label_vocab


def build_split_loaders(cfg: dict, label_vocab: list[str], tokenizer):
    df, _ = load_processed(cfg)
    max_length = cfg["model"]["max_length"]
    batch_size = cfg["train"]["eval_batch_size"]

    frames, datasets, loaders = {}, {}, {}
    for split in ("train", "val", "test"):
        frame = df[df["split"] == split].reset_index(drop=True)
        dataset = MusicCapsTagDataset(frame, label_vocab, tokenizer, max_length)
        frames[split] = frame
        datasets[split] = dataset
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return frames, datasets, loaders


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_f1_curves(history_path: str, out_path: str) -> None:
    """Macro-F1 / Micro-F1 and loss vs. epoch (required Task 1 deliverable)."""
    with open(history_path, encoding="utf-8") as f:
        blob = json.load(f)
    history = blob["history"]
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [h["train_macro_f1"] for h in history], "o-", label="train Macro-F1")
    axes[0].plot(epochs, [h["val_macro_f1"] for h in history], "o-", label="val Macro-F1")
    axes[0].plot(epochs, [h["train_micro_f1"] for h in history], "s--", label="train Micro-F1")
    axes[0].plot(epochs, [h["val_micro_f1"] for h in history], "s--", label="val Micro-F1")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("F1")
    axes[0].set_title(f"Task 1: {blob['model']} tag classifier")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["train_loss"] for h in history], "o-", label="train BCE")
    axes[1].plot(epochs, [h["val_loss"] for h in history], "o-", label="val BCE")
    ax2 = axes[1].twinx()
    ax2.plot(epochs, [h["val_auc_pr"] for h in history], "^:", color="green", label="val AUC-PR")
    ax2.set_ylabel("AUC-PR", color="green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("BCE loss")
    axes[1].set_title("Loss and AUC-PR")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eval] wrote {out_path}")


def plot_per_tag_f1(report: list[dict], out_path: str, top_n: int = 25) -> None:
    rows = report[:top_n]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y, [r["f1"] for r in rows], color="#4c72b0")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['tag']} (n={r['support']})" for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("test F1")
    ax.set_title(f"Per-tag F1, {len(rows)} most frequent tags")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eval] wrote {out_path}")


def plot_attention(tokens: list[str], weights: np.ndarray, title: str, out_path: str) -> None:
    """Token-level [CLS] attention from the final encoder layer, averaged over heads."""
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(tokens)), 2.2))
    ax.imshow(weights[None, :], aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# qualitative examples
# --------------------------------------------------------------------------- #
@torch.no_grad()
def qualitative_examples(
    model,
    tokenizer,
    frame,
    label_vocab: list[str],
    thresholds: np.ndarray,
    device: torch.device,
    cfg: dict,
    n_examples: int = 5,
) -> list[dict]:
    """Predictions for n test clips, each with an attention heat-map over tokens."""
    plots_dir = cfg["paths"]["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)
    max_length = cfg["model"]["max_length"]

    rng = np.random.default_rng(cfg["seed"])
    idxs = rng.choice(len(frame), size=min(n_examples, len(frame)), replace=False)

    examples = []
    for rank, idx in enumerate(idxs, start=1):
        row = frame.iloc[int(idx)]
        enc = tokenizer(
            str(row["text"]),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        out = model(enc["input_ids"], enc["attention_mask"], output_attentions=True)
        probs = torch.sigmoid(out["logits"])[0].float().cpu().numpy()
        predicted = [label_vocab[k] for k in np.flatnonzero(probs >= thresholds)]
        top5 = [(label_vocab[k], float(probs[k])) for k in np.argsort(-probs)[:5]]

        # Final layer, mean over heads, [CLS] query row -> per-token importance.
        attn = out["attentions"][-1][0].mean(dim=0)[0].float().cpu().numpy()
        n_real = int(enc["attention_mask"][0].sum().item())
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0][:n_real])
        weights = attn[:n_real]

        plot_path = os.path.join(plots_dir, f"task1_attention_{rank}_{row['ytid']}.png")
        plot_attention(tokens, weights, f"[CLS] attention -- {row['ytid']}", plot_path)

        truth = set(row["labels"])
        pred = set(predicted)
        examples.append(
            {
                "ytid": row["ytid"],
                "caption": str(row["caption"])[:400],
                "true_tags": sorted(truth),
                "predicted_tags": sorted(pred),
                "correct": sorted(truth & pred),
                "missed": sorted(truth - pred),
                "spurious": sorted(pred - truth),
                "top5_by_probability": top5,
                "top_attended_tokens": [
                    tokens[i] for i in np.argsort(-weights)[:10] if tokens[i] not in ("[CLS]", "[SEP]")
                ][:8],
                "attention_plot": plot_path,
            }
        )
    return examples


# --------------------------------------------------------------------------- #
# extra reference baseline: TF-IDF + one-vs-rest logistic regression
# --------------------------------------------------------------------------- #
def tfidf_baseline(frames, label_vocab: list[str], seed: int = 42) -> dict:
    """A non-neural text baseline -- shows how much the pretrained encoder buys."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    def multi_hot(frame):
        idx = {t: i for i, t in enumerate(label_vocab)}
        y = np.zeros((len(frame), len(label_vocab)), dtype=np.int8)
        for i, labels in enumerate(frame["labels"]):
            for tag in labels:
                if tag in idx:
                    y[i, idx[tag]] = 1
        return y

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_train = vec.fit_transform(frames["train"]["text"].astype(str))
    x_test = vec.transform(frames["test"]["text"].astype(str))
    y_train, y_test = multi_hot(frames["train"]), multi_hot(frames["test"])

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced", random_state=seed),
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_test)
    return multilabel_scores(probs, y_test, 0.5)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def evaluate(cfg: dict, ckpt_path: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"].startswith("cuda") else "cpu")
    model, ckpt_cfg, label_vocab = load_checkpoint(ckpt_path, device)
    # Evaluation follows the checkpoint's own data settings so a masked-caption
    # model is never scored against unmasked text.
    cfg["data"] = ckpt_cfg["data"]
    cfg["model"] = ckpt_cfg["model"]

    tokenizer = build_tokenizer(cfg["model"]["name"])
    frames, datasets, loaders = build_split_loaders(cfg, label_vocab, tokenizer)

    val_logits, val_targets = predict_logits(model, loaders["val"], device, cfg["train"]["amp"])
    test_logits, test_targets = predict_logits(model, loaders["test"], device, cfg["train"]["amp"])
    val_probs, test_probs = torch.sigmoid(val_logits).numpy(), torch.sigmoid(test_logits).numpy()
    val_y, test_y = val_targets.numpy(), test_targets.numpy()

    ecfg = cfg["eval"]
    default_t = ecfg["default_threshold"]
    results = {
        "model": cfg["model"]["name"],
        "mask_aspect_spans": cfg["data"]["mask_aspect_spans"],
        "num_labels": len(label_vocab),
        "n_test": int(len(test_y)),
        "bert_default_threshold": multilabel_scores(test_probs, test_y, default_t),
    }

    thresholds = np.full(len(label_vocab), default_t, dtype=np.float32)
    if ecfg["tune_thresholds"]:
        thresholds = tune_thresholds(
            val_probs,
            val_y,
            ecfg["threshold_grid_start"],
            ecfg["threshold_grid_stop"],
            ecfg["threshold_grid_step"],
            default_t,
        )
        results["bert_tuned_threshold"] = multilabel_scores(test_probs, test_y, thresholds)
        results["threshold_stats"] = {
            "mean": float(thresholds.mean()),
            "min": float(thresholds.min()),
            "max": float(thresholds.max()),
        }

    train_y = datasets["train"].targets.numpy()
    results["baseline_random_prior"] = random_baseline(train_y, test_y, cfg["seed"])
    results["baseline_majority"] = majority_baseline(train_y, test_y)
    try:
        results["baseline_tfidf_logreg"] = tfidf_baseline(frames, label_vocab, cfg["seed"])
    except Exception as exc:  # noqa: BLE001 - baseline is optional, never block the report
        print(f"[warn] TF-IDF baseline skipped: {exc}")

    report = per_tag_report(test_probs, test_y, label_vocab, thresholds)
    examples = qualitative_examples(
        model, tokenizer, frames["test"], label_vocab, thresholds, device, cfg, ecfg["n_examples"]
    )

    metrics_dir, plots_dir = cfg["paths"]["metrics_dir"], cfg["paths"]["plots_dir"]
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    with open(os.path.join(metrics_dir, "task1_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(metrics_dir, "task1_per_tag.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(metrics_dir, "task1_examples.json"), "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
    np.save(os.path.join(metrics_dir, "task1_thresholds.npy"), thresholds)

    history_path = os.path.join(metrics_dir, "task1_history.json")
    if os.path.exists(history_path):
        plot_f1_curves(history_path, os.path.join(plots_dir, "task1_f1_curves.png"))
    plot_per_tag_f1(report, os.path.join(plots_dir, "task1_per_tag_f1.png"))

    print_summary(results, examples)
    return results


def print_summary(results: dict, examples: list[dict]) -> None:
    key = "bert_tuned_threshold" if "bert_tuned_threshold" in results else "bert_default_threshold"
    rows = [
        ("B1a random (prior)", results["baseline_random_prior"]),
        ("B1b majority tags", results["baseline_majority"]),
    ]
    if "baseline_tfidf_logreg" in results:
        rows.append(("TF-IDF + LogReg", results["baseline_tfidf_logreg"]))
    rows.append(("BERT @0.5", results["bert_default_threshold"]))
    if "bert_tuned_threshold" in results:
        rows.append(("BERT @tuned", results["bert_tuned_threshold"]))

    print(f"\n{'Model':<22}{'Macro-F1':>10}{'Micro-F1':>10}{'AUC-PR':>10}")
    print("-" * 52)
    for name, s in rows:
        print(f"{name:<22}{s['macro_f1']:>10.4f}{s['micro_f1']:>10.4f}{s['auc_pr']:>10.4f}")

    print(f"\nExample predictions (threshold: {key}):")
    for ex in examples:
        print(f"\n  {ex['ytid']}")
        print(f"    caption : {ex['caption'][:140]}...")
        print(f"    true    : {', '.join(ex['true_tags'][:8])}")
        print(f"    pred    : {', '.join(ex['predicted_tags'][:8])}")
        print(f"    attended: {', '.join(ex['top_attended_tokens'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Task 1 BERT tag classifier.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ckpt", default=None, help="path to task1_bert_best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = args.ckpt or os.path.join(cfg["paths"]["ckpt_dir"], "task1_bert_best.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}\nrun `python src/train.py --config {args.config}` first")
    evaluate(cfg, ckpt)


if __name__ == "__main__":
    main()
