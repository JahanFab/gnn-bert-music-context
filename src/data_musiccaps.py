"""MusicCaps caption -> tag proxy dataset for Task 1.

Builds the text/multi-hot-label corpus used by the BERT baseline:

    input  : expert natural-language caption of a 10 s clip
    target : multi-hot vector over the top-K most frequent `aspect_list` tags

Run once before training:

    python src/data_musiccaps.py --config config.yaml
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import urllib.request
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

MUSICCAPS_URL = (
    "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv"
)


# --------------------------------------------------------------------------- #
# download + parsing
# --------------------------------------------------------------------------- #
def download_musiccaps(raw_dir: str, csv_name: str = "musiccaps-public.csv") -> str:
    """Fetch the 2.9 MB MusicCaps metadata CSV if it is not already on disk."""
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, csv_name)
    if os.path.exists(path):
        print(f"[data] found existing {path}")
        return path
    print(f"[data] downloading MusicCaps -> {path}")
    urllib.request.urlretrieve(MUSICCAPS_URL, path)
    return path


def normalize_aspect(aspect: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation noise from a tag string."""
    a = aspect.lower().strip()
    a = re.sub(r"\s+", " ", a)
    a = a.strip(" .,;:!?\"'")
    return a


def parse_aspect_list(cell: str) -> list[str]:
    """`aspect_list` is stored as a stringified Python list."""
    try:
        raw = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return []
    return [normalize_aspect(a) for a in raw if isinstance(a, str) and a.strip()]


def build_label_vocab(aspect_lists: Iterable[Sequence[str]], top_k: int) -> list[str]:
    """The K most frequent aspects across the corpus, ordered by frequency."""
    counts = Counter(a for aspects in aspect_lists for a in aspects)
    return [tag for tag, _ in counts.most_common(top_k)]


def mask_aspect_spans(caption: str, aspects: Sequence[str]) -> str:
    """Delete verbatim occurrences of the label strings from the caption.

    MusicCaps aspects were written by the same annotator as the caption, so many
    of them appear word-for-word in the text. Left untouched, a chunk of the
    proxy task collapses into substring matching. Masking the spans (longest
    first, so 'acoustic drums' goes before 'drums') gives an honest measurement
    of whether the encoder generalises beyond copying.
    """
    masked = caption
    for aspect in sorted(aspects, key=len, reverse=True):
        if not aspect:
            continue
        masked = re.sub(re.escape(aspect), " ", masked, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", masked).strip()


# --------------------------------------------------------------------------- #
# corpus construction
# --------------------------------------------------------------------------- #
def build_corpus(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Return the per-clip dataframe (with a `split` column) and the label vocab."""
    paths, dcfg = cfg["paths"], cfg["data"]
    csv_path = download_musiccaps(paths["raw_dir"], dcfg["csv_name"])
    df = pd.read_csv(csv_path)

    df["aspects"] = df["aspect_list"].map(parse_aspect_list)
    label_vocab = build_label_vocab(df["aspects"], dcfg["top_k_tags"])
    label_set = set(label_vocab)

    # Keep only in-vocabulary aspects; drop clips left with too few labels.
    df["labels"] = df["aspects"].map(lambda a: sorted(label_set.intersection(a)))
    before = len(df)
    df = df[df["labels"].map(len) >= dcfg["min_tags_per_clip"]].reset_index(drop=True)
    print(f"[data] kept {len(df)}/{before} clips with >={dcfg['min_tags_per_clip']} in-vocab tag(s)")

    df["text"] = df["caption"].astype(str)
    if dcfg["mask_aspect_spans"]:
        df["text"] = [
            mask_aspect_spans(cap, asp) for cap, asp in zip(df["text"], df["aspects"])
        ]
        print("[data] aspect spans masked out of captions (harder, leakage-free setting)")

    df["split"] = assign_splits(df, cfg)
    print("[data] split sizes:", df["split"].value_counts().to_dict())
    return df, label_vocab


def assign_splits(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Test = official MusicCaps eval flag, OR a seeded random `test_frac` split.

    The official `is_audioset_eval` partition marks ~53% of usable clips as eval,
    which leaves very few clips for training a 50-tag multi-label fine-tune. Set
    `data.use_official_eval_split: false` to use a seeded `test_frac` / `val_frac`
    random split instead (default 15% / 15% of the remainder -> ~72/13/15).

    Splitting is by `ytid`, which is unique per clip, so no clip can straddle two
    splits. MusicCaps has no artist metadata, so the spec's 'no artist leakage'
    requirement is satisfied as far as the released data allows -- documented in
    the report rather than silently assumed.
    """
    dcfg = cfg["data"]
    rng = np.random.default_rng(cfg["seed"])

    if dcfg["use_official_eval_split"] and "is_audioset_eval" in df.columns:
        is_test = df["is_audioset_eval"].astype(bool).to_numpy()
    else:  # seeded random test split
        is_test = rng.random(len(df)) < dcfg.get("test_frac", 0.15)

    split = np.where(is_test, "test", "train").astype(object)
    trainable = np.flatnonzero(~is_test)
    n_val = int(round(dcfg["val_frac"] * len(trainable)))
    val_idx = rng.choice(trainable, size=n_val, replace=False)
    split[val_idx] = "val"
    return pd.Series(split, index=df.index)


def save_processed(df: pd.DataFrame, label_vocab: list[str], cfg: dict) -> None:
    """Persist the corpus, the label vocabulary, and the split id lists."""
    paths = cfg["paths"]
    os.makedirs(paths["processed_dir"], exist_ok=True)
    os.makedirs(paths["splits_dir"], exist_ok=True)

    out = df[["ytid", "start_s", "end_s", "text", "caption", "labels", "split"]].copy()
    out["labels"] = out["labels"].map(json.dumps)
    corpus_path = os.path.join(paths["processed_dir"], "musiccaps_task1.csv")
    out.to_csv(corpus_path, index=False)

    vocab_path = os.path.join(paths["processed_dir"], "label_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(label_vocab, f, indent=2)

    splits = {s: df.loc[df["split"] == s, "ytid"].tolist() for s in ("train", "val", "test")}
    splits_path = os.path.join(paths["splits_dir"], "task1_splits.json")
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print(f"[data] wrote {corpus_path}\n[data] wrote {vocab_path}\n[data] wrote {splits_path}")


def load_processed(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Read back what `save_processed` wrote, rebuilding it if absent."""
    paths = cfg["paths"]
    corpus_path = os.path.join(paths["processed_dir"], "musiccaps_task1.csv")
    vocab_path = os.path.join(paths["processed_dir"], "label_vocab.json")

    if not (os.path.exists(corpus_path) and os.path.exists(vocab_path)):
        df, label_vocab = build_corpus(cfg)
        save_processed(df, label_vocab, cfg)
        return df, label_vocab

    df = pd.read_csv(corpus_path)
    df["labels"] = df["labels"].map(json.loads)
    with open(vocab_path, encoding="utf-8") as f:
        label_vocab = json.load(f)
    return df, label_vocab


# --------------------------------------------------------------------------- #
# torch Dataset
# --------------------------------------------------------------------------- #
class MusicCapsTagDataset(Dataset):
    """Tokenised captions with multi-hot tag targets."""

    def __init__(self, df: pd.DataFrame, label_vocab: Sequence[str], tokenizer, max_length: int):
        self.texts = df["text"].astype(str).tolist()
        self.ytids = df["ytid"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_vocab = list(label_vocab)
        self.label_to_idx = {t: i for i, t in enumerate(self.label_vocab)}
        self.targets = self._multi_hot(df["labels"].tolist())

    def _multi_hot(self, label_lists: Sequence[Sequence[str]]) -> torch.Tensor:
        y = torch.zeros(len(label_lists), len(self.label_vocab), dtype=torch.float32)
        for i, labels in enumerate(label_lists):
            for tag in labels:
                j = self.label_to_idx.get(tag)
                if j is not None:
                    y[i, j] = 1.0
        return y

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": self.targets[idx],
        }


def compute_pos_weight(dataset: MusicCapsTagDataset, clip: float = 20.0) -> torch.Tensor:
    """(#neg / #pos) per tag for BCEWithLogitsLoss, clipped to keep gradients sane."""
    pos = dataset.targets.sum(dim=0)
    neg = dataset.targets.shape[0] - pos
    weight = neg / pos.clamp(min=1.0)
    return weight.clamp(max=clip)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MusicCaps Task 1 corpus.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df, label_vocab = build_corpus(cfg)
    save_processed(df, label_vocab, cfg)

    counts = np.array([len(l) for l in df["labels"]])
    print(f"\n[stats] clips: {len(df)}  |  tags: {len(label_vocab)}")
    print(f"[stats] tags/clip: mean {counts.mean():.2f}, median {np.median(counts):.0f}, max {counts.max()}")
    print(f"[stats] most frequent tags: {label_vocab[:10]}")


if __name__ == "__main__":
    main()
