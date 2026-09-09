"""Track indexing and splits for Task 2 (GTZAN / FMA-small).

Produces one tidy dataframe per dataset:

    track_id | path | label | artist | split

Both datasets are supported because they trade off differently: GTZAN is 1.2 GB
and trains in minutes but is a flawed benchmark, while FMA-small is 7.2 GB with
proper artist-disjoint official splits. The rest of Task 2 only ever sees the
dataframe, so the choice is a one-line config change.

    python src/audio_datasets.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

AUDIO_EXTENSIONS = (".wav", ".mp3", ".au", ".flac", ".ogg")


# --------------------------------------------------------------------------- #
# GTZAN
# --------------------------------------------------------------------------- #
def index_gtzan(root: str) -> pd.DataFrame:
    """Scan `genres_original/<genre>/<genre>.<nnnnn>.wav`.

    GTZAN caveats to state in the report (Sturm, 2013): exact-duplicate clips,
    several mislabelled tracks, repeated artists across the collection, and a
    corrupt `jazz.00054.wav`. It is used here as the spec's "standard easy
    baseline", not as evidence of state-of-the-art performance.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"GTZAN directory not found: {root}")

    rows = []
    for genre in sorted(os.listdir(root)):
        genre_dir = os.path.join(root, genre)
        if not os.path.isdir(genre_dir):
            continue
        for name in sorted(os.listdir(genre_dir)):
            if name.lower().endswith(AUDIO_EXTENSIONS):
                rows.append(
                    {
                        "track_id": os.path.splitext(name)[0],
                        "path": os.path.join(genre_dir, name),
                        "label": genre,
                        "artist": None,  # GTZAN ships no artist metadata
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"no audio found under {root}")
    return df


# --------------------------------------------------------------------------- #
# FMA-small
# --------------------------------------------------------------------------- #
def index_fma(audio_root: str, metadata_csv: str, subset: str = "small") -> pd.DataFrame:
    """Join `tracks.csv` metadata against the mp3 tree.

    FMA stores audio as `<audio_root>/<first 3 digits>/<6-digit id>.mp3` and
    ships a two-level-header CSV carrying `track.genre_top`, `artist.id` and the
    official `set.split`.
    """
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"FMA metadata not found: {metadata_csv}")

    tracks = pd.read_csv(metadata_csv, index_col=0, header=[0, 1])
    in_subset = tracks[("set", "subset")] <= subset if ("set", "subset") in tracks else slice(None)
    tracks = tracks[in_subset]

    rows = []
    for track_id, row in tracks.iterrows():
        tid = f"{int(track_id):06d}"
        path = os.path.join(audio_root, tid[:3], f"{tid}.mp3")
        genre = row[("track", "genre_top")]
        if not os.path.exists(path) or pd.isna(genre):
            continue
        rows.append(
            {
                "track_id": tid,
                "path": path,
                "label": str(genre),
                "artist": row[("artist", "id")] if ("artist", "id") in row else None,
                "official_split": row[("set", "split")] if ("set", "split") in row else None,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"no FMA audio matched metadata under {audio_root}")
    return df


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def stratified_split(
    df: pd.DataFrame, seed: int, val_frac: float = 0.15, test_frac: float = 0.15
) -> pd.Series:
    """Class-stratified random split, used when the dataset has no official one."""
    rng = np.random.default_rng(seed)
    split = np.empty(len(df), dtype=object)

    for label in df["label"].unique():
        idx = np.flatnonzero((df["label"] == label).to_numpy())
        rng.shuffle(idx)
        n_test = int(round(test_frac * len(idx)))
        n_val = int(round(val_frac * len(idx)))
        split[idx[:n_test]] = "test"
        split[idx[n_test : n_test + n_val]] = "val"
        split[idx[n_test + n_val :]] = "train"

    return pd.Series(split, index=df.index)


def grouped_split(
    df: pd.DataFrame, seed: int, group_col: str = "artist", val_frac: float = 0.15, test_frac: float = 0.15
) -> pd.Series:
    """Split by group so an artist never spans two splits (spec: no artist leakage)."""
    rng = np.random.default_rng(seed)
    groups = df[group_col].fillna("__unknown__").astype(str).to_numpy()
    unique = np.unique(groups)
    rng.shuffle(unique)

    n_test = int(round(test_frac * len(unique)))
    n_val = int(round(val_frac * len(unique)))
    assignment = {g: "test" for g in unique[:n_test]}
    assignment.update({g: "val" for g in unique[n_test : n_test + n_val]})
    assignment.update({g: "train" for g in unique[n_test + n_val :]})
    return pd.Series([assignment[g] for g in groups], index=df.index)


def assign_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    t2 = cfg["task2"]
    seed = cfg["seed"]

    if "official_split" in df.columns and df["official_split"].notna().all():
        # FMA's official splits are already artist-disjoint.
        df["split"] = df["official_split"].replace({"validation": "val"})
        print("[data] using official FMA splits (artist-disjoint by construction)")
    elif df["artist"].notna().any():
        df["split"] = grouped_split(df, seed, "artist", t2["val_frac"], t2["test_frac"])
        print("[data] artist-grouped split")
    else:
        df["split"] = stratified_split(df, seed, t2["val_frac"], t2["test_frac"])
        print(
            "[data] stratified random split -- this dataset ships no artist metadata, "
            "so artist leakage cannot be ruled out (document this in the report)"
        )
    return df


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def build_index(cfg: dict) -> pd.DataFrame:
    t2 = cfg["task2"]
    name = t2["dataset"].lower()

    if name == "gtzan":
        df = index_gtzan(t2["gtzan_dir"])
    elif name in ("fma_small", "fma"):
        df = index_fma(t2["fma_dir"], t2["fma_metadata"], t2.get("fma_subset", "small"))
    else:
        raise ValueError(f"unknown task2.dataset: {t2['dataset']!r} (expected gtzan | fma_small)")

    df = assign_splits(df, cfg)
    labels = sorted(df["label"].unique())
    df["label_idx"] = df["label"].map({l: i for i, l in enumerate(labels)})

    print(f"[data] {name}: {len(df)} tracks, {len(labels)} classes")
    print(f"[data] split sizes: {df['split'].value_counts().to_dict()}")
    return df


def save_index(df: pd.DataFrame, cfg: dict) -> tuple[str, list[str]]:
    paths = cfg["paths"]
    os.makedirs(paths["processed_dir"], exist_ok=True)
    os.makedirs(paths["splits_dir"], exist_ok=True)

    name = cfg["task2"]["dataset"].lower()
    index_path = os.path.join(paths["processed_dir"], f"task2_index_{name}.csv")
    df.to_csv(index_path, index=False)

    labels = sorted(df["label"].unique())
    splits_path = os.path.join(paths["splits_dir"], f"task2_splits_{name}.json")
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": labels,
                "splits": {s: df.loc[df["split"] == s, "track_id"].tolist() for s in ("train", "val", "test")},
            },
            f,
            indent=2,
        )

    print(f"[data] wrote {index_path}\n[data] wrote {splits_path}")
    return index_path, labels


def load_index(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Read the cached index, building it if missing."""
    name = cfg["task2"]["dataset"].lower()
    index_path = os.path.join(cfg["paths"]["processed_dir"], f"task2_index_{name}.csv")

    if not os.path.exists(index_path):
        df = build_index(cfg)
        save_index(df, cfg)
    else:
        df = pd.read_csv(index_path, dtype={"track_id": str})

    return df, sorted(df["label"].unique())


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a Task 2 audio dataset and assign splits.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default=None, help="gtzan | fma_small")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg["task2"]["dataset"] = args.dataset

    df = build_index(cfg)
    save_index(df, cfg)
    print("\n[stats] tracks per class:")
    print(df.groupby(["label", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
