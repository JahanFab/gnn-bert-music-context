"""Paired (graph, text, labels) data for Task 3 fusion.

Task 3 needs the full tuple from spec Section 2:

    T = (X_audio, X_text, G, y)

Three sources are supported behind one index format:

  musiccaps    text = expert caption, labels = top-K aspects, graph = 10 s clip
               (the same label vocabulary as Task 1, so the BERT-only ablation
               is literally the Task 1 model and the numbers are comparable)
  fma_medium   text = artist/album/title/tags metadata string,
               labels = multi-label genres
  magnatagatune  text and labels are BOTH drawn from the same 188-tag vocabulary
               (there is no separate caption), so each clip's positive tags are
               randomly split in half per clip: one half becomes a synthesised
               "Tags: ..." text string, the other half stays as the multi-label
               target. This avoids the fusion model just reading its own label
               back as input, at the cost of a nonstandard, weaker text signal
               than MusicCaps captions -- document this in the report.
  deam         valence/arousal only, no text -- appended as auxiliary emotion
               supervision for the L_aux term

Every row carries per-sample masks, so a batch may mix tagged-but-emotionless
MusicCaps clips with emotion-only DEAM clips and each loss term only sees the
samples it actually applies to.

    python src/data_fusion.py --config config.yaml
"""

from __future__ import annotations

import argparse
import ast
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from audio_datasets import load_config
from audio_features import AudioConfig, extract_and_cache
from graph_builder import build_track_graph

EMPTY_TEXT = ""  # DEAM rows have no caption; the text branch sees [CLS][SEP] only


# --------------------------------------------------------------------------- #
# index construction
# --------------------------------------------------------------------------- #
def _index_musiccaps(cfg: dict) -> pd.DataFrame:
    """Reuse the Task 1 corpus and join it against downloaded clip audio."""
    paths, t3 = cfg["paths"], cfg["task3"]
    corpus_path = os.path.join(paths["processed_dir"], "musiccaps_task1.csv")
    if not os.path.exists(corpus_path):
        raise SystemExit(
            f"{corpus_path} not found -- run `python src/data_musiccaps.py` first"
        )

    df = pd.read_csv(corpus_path)
    df["labels"] = df["labels"].map(json.loads)
    audio_dir = t3["musiccaps_audio_dir"]

    df["path"] = [os.path.join(audio_dir, f"{ytid}.wav") for ytid in df["ytid"]]
    present = df["path"].map(os.path.exists)
    missing = int((~present).sum())
    if missing:
        # YouTube attrition is normal here: a slice of MusicCaps ids are private,
        # region-locked or deleted. Report it rather than silently shrinking.
        print(f"[data] {missing}/{len(df)} MusicCaps clips have no local audio and are dropped")
    df = df[present].reset_index(drop=True)

    return pd.DataFrame(
        {
            "track_id": df["ytid"],
            "path": df["path"],
            "text": df["text"].astype(str),
            "labels": df["labels"],
            "split": df["split"],
            "valence": np.nan,
            "arousal": np.nan,
            "source": "musiccaps",
        }
    )


def _index_fma_medium(cfg: dict) -> pd.DataFrame:
    """Metadata-derived text + multi-label genres from FMA's tracks.csv."""
    t3 = cfg["task3"]
    tracks = pd.read_csv(t3["fma_metadata"], index_col=0, header=[0, 1])
    if ("set", "subset") in tracks:
        tracks = tracks[tracks[("set", "subset")] <= t3.get("fma_subset", "medium")]

    genre_names: dict[int, str] = {}
    genres_csv = t3.get("fma_genres")
    if genres_csv and os.path.exists(genres_csv):
        gdf = pd.read_csv(genres_csv, index_col=0)
        genre_names = {int(i): str(r["title"]) for i, r in gdf.iterrows()}

    def field(row, group, name, default=""):
        return row[(group, name)] if (group, name) in row and pd.notna(row[(group, name)]) else default

    rows = []
    for track_id, row in tracks.iterrows():
        tid = f"{int(track_id):06d}"
        path = os.path.join(t3["fma_dir"], tid[:3], f"{tid}.mp3")
        if not os.path.exists(path):
            continue

        raw_genres = field(row, "track", "genres_all", "[]")
        try:
            ids = ast.literal_eval(raw_genres) if isinstance(raw_genres, str) else list(raw_genres)
        except (ValueError, SyntaxError):
            ids = []
        labels = [genre_names.get(int(g), str(g)) for g in ids]
        if not labels:
            continue

        tags = field(row, "track", "tags", "[]")
        try:
            tag_list = ast.literal_eval(tags) if isinstance(tags, str) else list(tags)
        except (ValueError, SyntaxError):
            tag_list = []

        text = (
            f"Artist: {field(row, 'artist', 'name')}. "
            f"Album: {field(row, 'album', 'title')}. "
            f"Title: {field(row, 'track', 'title')}. "
            f"Tags: {', '.join(str(t) for t in tag_list) or 'none'}."
        )
        split = field(row, "set", "split", "training")
        rows.append(
            {
                "track_id": tid,
                "path": path,
                "text": text,
                "labels": labels,
                "split": {"training": "train", "validation": "val", "test": "test"}.get(split, split),
                "valence": np.nan,
                "arousal": np.nan,
                "source": "fma_medium",
            }
        )

    if not rows:
        raise RuntimeError(f"no FMA-medium audio matched metadata under {t3['fma_dir']}")
    return pd.DataFrame(rows)


def _mtat_stem(tag: str) -> str:
    """Loose stem for dedup: lowercase, spaces for _/-, drop a trailing 's'."""
    s = tag.lower().replace("_", " ").replace("-", " ").strip()
    return s[:-1] if s.endswith("s") and len(s) > 3 else s


def _index_magnatagatune(cfg: dict, seed: int) -> pd.DataFrame:
    """Text + multi-label tags from MagnaTagATune.

    annotations_final.csv carries 188 binary tag columns. The top-K by frequency
    are the prediction target; the text is built so it never contains one of
    those K (which would let the model read a label back as input). Two modes,
    via ``task3.mtat_text_mode``:

      descriptive_tags (default)  text = the clip's OWN tags that are NOT in the
          top-K label set (instrument / mood / production descriptors), stem-
          deduped against the labels, most-frequent first; prefixed with the
          clip's title + artist from clip_info_final.csv when available (neither
          is a label). This gives the text branch genuine, non-leaking signal --
          far richer than bare metadata, which was the Task 3/4 bottleneck.
      metadata                    text = "Artist: X. Album: Y. Title: Z." only
          (the previous behaviour; weakest text signal).

    Clips with no top-K label are dropped. Clips with no non-label tag fall back
    to the metadata string (or are dropped if clip_info_final.csv is absent).
    """
    t3 = cfg["task3"]
    text_mode = t3.get("mtat_text_mode", "descriptive_tags")
    ann_path = t3.get("magnatagatune_annotations")
    audio_dir = t3.get("magnatagatune_dir")
    if not ann_path or not os.path.exists(ann_path):
        raise SystemExit(
            f"{ann_path} not found -- set task3.magnatagatune_annotations in config.yaml"
        )

    ann = pd.read_csv(ann_path, sep="\t")
    if ann.shape[1] == 1:  # some mirrors ship it comma-separated instead
        ann = pd.read_csv(ann_path)

    id_col = next((c for c in ann.columns if c.lower() in ("clip_id", "id")), ann.columns[0])
    path_col = next((c for c in ann.columns if "path" in c.lower() or c.lower() == "mp3"), None)
    if path_col is None:
        raise SystemExit(f"{ann_path}: could not find an mp3 path column")

    tag_cols = [c for c in ann.columns if c not in (id_col, path_col)]
    top_k = t3.get("top_k_tags", 50)
    tag_freq = ann[tag_cols].sum(axis=0).sort_values(ascending=False)
    top_tags = tag_freq.index[:top_k].tolist()
    top_set = set(top_tags)
    # stems + individual words of the label tags, for non-leaking text selection
    top_stems = {_mtat_stem(t) for t in top_tags}
    top_words = {w for t in top_tags for w in _mtat_stem(t).split()}
    desc_tag_order = [t for t in tag_freq.index if t not in top_set]  # freq-desc

    clip_info_path = t3.get("magnatagatune_clip_info")
    clip_info = None
    if clip_info_path and os.path.exists(clip_info_path):
        clip_info = pd.read_csv(clip_info_path, sep="\t")
        if clip_info.shape[1] == 1:
            clip_info = pd.read_csv(clip_info_path)
        clip_info.columns = [c.strip().lower() for c in clip_info.columns]
        info_id_col = next((c for c in clip_info.columns if c in ("clip_id", "id")), clip_info.columns[0])
        clip_info = clip_info.set_index(info_id_col)
        print(f"[data] magnatagatune: clip_info from {clip_info_path}; text mode = {text_mode}")
    else:
        print(f"[data] magnatagatune: no clip_info_final.csv; text mode = {text_mode}")

    def metadata_text(clip_id) -> str | None:
        if clip_info is None or clip_id not in clip_info.index:
            return None
        info = clip_info.loc[clip_id]
        return (
            f"Artist: {info.get('artist', 'unknown')}. "
            f"Album: {info.get('album', 'unknown')}. "
            f"Title: {info.get('title', 'unknown')}."
        )

    def descriptive_text(row, clip_id, max_tags: int = 15) -> str | None:
        # the clip's own tags that are NOT labels and don't stem-collide with one
        desc = [
            t for t in desc_tag_order
            if int(row[t]) == 1
            and _mtat_stem(t) not in top_stems
            and not (set(_mtat_stem(t).split()) & top_words)
        ][:max_tags]
        if not desc:
            return None
        phrase = ", ".join(t.replace("_", " ") for t in desc)
        if clip_info is not None and clip_id in clip_info.index:
            info = clip_info.loc[clip_id]
            title, artist = info.get("title", ""), info.get("artist", "")
            prefix = f"'{title}' by {artist}. ".replace("'' by . ", "")
            return f"{prefix}A music track described as: {phrase}."
        return f"A music track described as: {phrase}."

    rng = np.random.default_rng(seed)
    rows, skipped_short = [], 0
    for _, row in ann.iterrows():
        rel_path = str(row[path_col])
        path = os.path.join(audio_dir, rel_path)
        if not os.path.exists(path):
            continue

        labels = [t for t in top_tags if int(row[t]) == 1]
        if not labels:
            continue
        clip_id = row[id_col]

        if text_mode == "metadata":
            text = metadata_text(clip_id)
            if text is None:
                if len(labels) < 2:
                    skipped_short += 1
                    continue
                shuffled = labels.copy()
                rng.shuffle(shuffled)
                half = len(shuffled) // 2
                text = f"Tags: {', '.join(t.replace('_', ' ') for t in shuffled[:half])}."
                labels = shuffled[half:]
        else:  # descriptive_tags
            text = descriptive_text(row, clip_id) or metadata_text(clip_id)
            if text is None:
                skipped_short += 1
                continue

        # Standard MagnaTagATune split (Choi et al. 2016): top-level hex folder
        # 0-b -> train, c -> val, d-f -> test.
        folder = rel_path.split("/")[0].lower()
        split = "train" if folder in "0123456789ab" else ("val" if folder == "c" else "test")

        rows.append(
            {
                "track_id": str(clip_id),
                "path": path,
                "text": text,
                "labels": labels,
                "split": split,
                "valence": np.nan,
                "arousal": np.nan,
                "source": "magnatagatune",
            }
        )

    if not rows:
        raise RuntimeError(f"no MagnaTagATune audio matched annotations under {audio_dir}")
    df_out = pd.DataFrame(rows)
    med_words = int(df_out["text"].str.split().map(len).median())
    print(
        f"[data] magnatagatune: {len(rows)} clips (tag vocab={len(top_tags)}, "
        f"text mode={text_mode}, median {med_words} words/clip)"
        + (f", {skipped_short} clips skipped (no usable text)" if skipped_short else "")
    )
    return df_out


def _index_deam(cfg: dict, seed: int) -> pd.DataFrame:
    """Optional emotion-only rows: static per-song valence/arousal means.

    DEAM ships no text, so these rows train the graph branch and the emotion head
    while the tag loss is masked off for them. That asymmetry is deliberate and
    must be stated in the report -- it is not an oversight of the fusion design.
    """
    t3 = cfg["task3"]
    ann_path, audio_dir = t3.get("deam_annotations"), t3.get("deam_audio_dir")
    if not ann_path or not os.path.exists(ann_path):
        print("[data] DEAM annotations not found -- training without emotion supervision")
        return pd.DataFrame()

    ann = pd.read_csv(ann_path)
    ann.columns = [c.strip() for c in ann.columns]
    col = {c.lower(): c for c in ann.columns}
    v_col = next((col[c] for c in col if "valence" in c and "mean" in c), None)
    a_col = next((col[c] for c in col if "arousal" in c and "mean" in c), None)
    id_col = next((col[c] for c in col if "song" in c or c == "id"), ann.columns[0])
    if v_col is None or a_col is None:
        print("[data] DEAM annotation columns not recognised -- skipping emotion rows")
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows = []
    for _, row in ann.iterrows():
        song_id = int(row[id_col])
        path = os.path.join(audio_dir, f"{song_id}.mp3")
        if not os.path.exists(path):
            continue
        # DEAM rates on [1, 9]; rescale to [-1, 1] so the MSE terms sit on a
        # comparable scale to the BCE tag loss without retuning alpha/beta.
        rows.append(
            {
                "track_id": f"deam_{song_id}",
                "path": path,
                "text": EMPTY_TEXT,
                "labels": [],
                "split": rng.choice(["train", "val", "test"], p=[0.8, 0.1, 0.1]),
                "valence": (float(row[v_col]) - 5.0) / 4.0,
                "arousal": (float(row[a_col]) - 5.0) / 4.0,
                "source": "deam",
            }
        )

    print(f"[data] DEAM: {len(rows)} clips with valence/arousal targets")
    return pd.DataFrame(rows)


def build_paired_index(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the paired index and the label vocabulary."""
    t3 = cfg["task3"]
    name = t3["dataset"].lower()

    if name == "musiccaps":
        df = _index_musiccaps(cfg)
    elif name in ("fma_medium", "fma"):
        df = _index_fma_medium(cfg)
    elif name in ("magnatagatune", "mtat"):
        df = _index_magnatagatune(cfg, cfg["seed"])
    else:
        raise ValueError(f"unknown task3.dataset: {t3['dataset']!r}")

    if t3.get("use_deam", False):
        deam = _index_deam(cfg, cfg["seed"])
        if not deam.empty:
            df = pd.concat([df, deam], ignore_index=True)

    label_vocab = _resolve_label_vocab(cfg, df)
    vocab = set(label_vocab)
    df["labels"] = df["labels"].map(lambda ls: [l for l in ls if l in vocab])

    print(f"[data] paired rows: {len(df)} | labels: {len(label_vocab)}")
    print(f"[data] split sizes: {df['split'].value_counts().to_dict()}")
    return df, label_vocab


def _resolve_label_vocab(cfg: dict, df: pd.DataFrame) -> list[str]:
    """Prefer the Task 1 vocabulary so ablations stay comparable across tasks."""
    task1_vocab = os.path.join(cfg["paths"]["processed_dir"], "label_vocab.json")
    if cfg["task3"]["dataset"].lower() == "musiccaps" and os.path.exists(task1_vocab):
        with open(task1_vocab, encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"[data] reusing Task 1 label vocabulary ({len(vocab)} tags)")
        return vocab

    from collections import Counter

    counts = Counter(l for labels in df["labels"] for l in labels)
    return [t for t, _ in counts.most_common(cfg["task3"]["top_k_tags"])]


# --------------------------------------------------------------------------- #
# graphs
# --------------------------------------------------------------------------- #
def build_fusion_graphs(cfg: dict, index: pd.DataFrame, overwrite: bool = False) -> dict:
    """One structure graph per paired row, cached to disk.

    Task 3 clips are much shorter than Task 2 tracks (10 s for MusicCaps vs. 30 s
    for GTZAN), so `task3.audio` carries its own shorter segment length -- reusing
    the Task 2 window would yield two or three nodes and a nearly empty graph.
    """
    t3 = cfg["task3"]
    acfg = AudioConfig.from_dict(t3["audio"])
    cache_dir = os.path.join(cfg["paths"]["processed_dir"], "audio_cache_task3")

    graphs, failures = {}, 0
    for n, row in enumerate(index.itertuples(index=False), start=1):
        bundle = extract_and_cache(row.path, str(row.track_id), cache_dir, acfg, overwrite)
        if bundle is None:
            failures += 1
            continue
        graph = build_track_graph(bundle, 0, str(row.track_id), t3["graph"])
        if graph is None:
            failures += 1
            continue
        graphs[str(row.track_id)] = graph
        if n % 200 == 0:
            print(f"[graph] {n}/{len(index)} clips processed")

    print(f"[graph] built {len(graphs)} graphs ({failures} skipped)")
    return graphs


def save_fusion_data(index: pd.DataFrame, graphs: dict, label_vocab: list[str], cfg: dict) -> str:
    out_dir = os.path.join(cfg["paths"]["processed_dir"], "fusion")
    os.makedirs(out_dir, exist_ok=True)
    tag = cfg["task3"]["dataset"].lower()

    index = index[index["track_id"].astype(str).isin(graphs)].reset_index(drop=True)
    path = os.path.join(out_dir, f"task3_fusion_{tag}.pt")
    torch.save(
        {
            "index": index.to_dict("records"),
            "graphs": graphs,
            "label_vocab": label_vocab,
            "node_dim": int(next(iter(graphs.values())).x.shape[1]) if graphs else 0,
        },
        path,
    )
    print(f"[data] wrote {path} ({len(index)} paired examples)")
    return path


def load_fusion_data(cfg: dict) -> tuple[pd.DataFrame, dict, list[str], int]:
    tag = cfg["task3"]["dataset"].lower()
    path = os.path.join(cfg["paths"]["processed_dir"], "fusion", f"task3_fusion_{tag}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run `python src/data_fusion.py --config config.yaml` first")

    blob = torch.load(path, map_location="cpu", weights_only=False)
    return pd.DataFrame(blob["index"]), blob["graphs"], blob["label_vocab"], blob["node_dim"]


def fit_node_scaler(index: pd.DataFrame, graphs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    train_ids = index.loc[index["split"] == "train", "track_id"].astype(str)
    stacked = torch.cat([graphs[t].x for t in train_ids if t in graphs], dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0).clamp(min=1e-6)


# --------------------------------------------------------------------------- #
# torch Dataset + collate
# --------------------------------------------------------------------------- #
class FusionDataset(Dataset):
    """Yields (graph, tokenised text, tag targets, emotion targets, masks)."""

    def __init__(self, frame: pd.DataFrame, graphs: dict, label_vocab: list[str], tokenizer, max_length: int):
        self.rows = frame.reset_index(drop=True)
        self.graphs = graphs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_vocab = list(label_vocab)
        self.label_to_idx = {t: i for i, t in enumerate(self.label_vocab)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        enc = self.tokenizer(
            str(row["text"]) if isinstance(row["text"], str) else EMPTY_TEXT,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        y = torch.zeros(len(self.label_vocab), dtype=torch.float32)
        for tag in row["labels"]:
            j = self.label_to_idx.get(tag)
            if j is not None:
                y[j] = 1.0

        has_tags = float(len(row["labels"]) > 0)
        v, a = row.get("valence", np.nan), row.get("arousal", np.nan)
        has_emotion = float(not (pd.isna(v) or pd.isna(a)))

        return {
            "graph": self.graphs[str(row["track_id"])],
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "y_tags": y,
            "tag_mask": torch.tensor(has_tags, dtype=torch.float32),
            "y_emotion": torch.tensor(
                [0.0 if pd.isna(v) else float(v), 0.0 if pd.isna(a) else float(a)],
                dtype=torch.float32,
            ),
            "emotion_mask": torch.tensor(has_emotion, dtype=torch.float32),
            "track_id": str(row["track_id"]),
        }


def collate_fusion(items: list[dict]) -> dict:
    from torch_geometric.data import Batch

    return {
        "graph": Batch.from_data_list([it["graph"] for it in items]),
        "input_ids": torch.stack([it["input_ids"] for it in items]),
        "attention_mask": torch.stack([it["attention_mask"] for it in items]),
        "y_tags": torch.stack([it["y_tags"] for it in items]),
        "tag_mask": torch.stack([it["tag_mask"] for it in items]),
        "y_emotion": torch.stack([it["y_emotion"] for it in items]),
        "emotion_mask": torch.stack([it["emotion_mask"] for it in items]),
        "track_id": [it["track_id"] for it in items],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Task 3 paired fusion dataset.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default=None, help="musiccaps | fma_medium | magnatagatune")
    parser.add_argument("--overwrite-cache", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg["task3"]["dataset"] = args.dataset

    index, label_vocab = build_paired_index(cfg)
    graphs = build_fusion_graphs(cfg, index, args.overwrite_cache)
    if not graphs:
        raise SystemExit("no graphs were built -- check the audio paths in config.yaml task3")

    mean, std = fit_node_scaler(index, graphs)
    for g in graphs.values():
        g.x = (g.x - mean) / std

    save_fusion_data(index, graphs, label_vocab, cfg)


if __name__ == "__main__":
    main()
