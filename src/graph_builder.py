"""Music structure graphs for Task 2 (spec Section 3, "Graph construction").

Two graph families, selectable via `task2.graph.type`:

  segment graph  nodes = time segments
                 edges = temporal adjacency + cosine similarity of MFCC/chroma > tau
  chord graph    nodes = unique chords (24 major/minor triad templates)
                 edges = observed transitions, weighted by count

Both are emitted as PyTorch Geometric `Data` objects. Node features are
standardised with statistics fitted on the TRAIN split only and reused for
val/test, so no test statistics leak into training.

    python src/graph_builder.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from audio_datasets import load_config, load_index
from audio_features import AudioConfig, extract_and_cache

try:
    from torch_geometric.data import Data
except ImportError as exc:  # pragma: no cover - dependency documented in README
    raise ImportError(
        "torch-geometric is required for Task 2: pip install torch-geometric"
    ) from exc


PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# --------------------------------------------------------------------------- #
# segment graph
# --------------------------------------------------------------------------- #
def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    unit = x / np.clip(norms, 1e-8, None)
    return unit @ unit.T


def build_segment_edges(
    features: np.ndarray,
    temporal: bool = True,
    sim_threshold: float = 0.8,
    max_sim_neighbors: int = 8,
    undirected: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Edges of the segment graph as (edge_index[2, E], edge_weight[E]).

    Two edge types share one adjacency:
      * temporal  -- consecutive segments, weight 1.0, carrying song order
      * similarity -- cosine(h_i, h_j) > tau, weight = similarity, linking
        repeated material (chorus to chorus) that is far apart in time

    Similarity edges are capped at `max_sim_neighbors` per node: on a
    homogeneous track a bare threshold makes the graph nearly complete, which
    washes out message passing and blows up memory.
    """
    n = len(features)
    edges: dict[tuple[int, int], float] = {}

    if temporal:
        for i in range(n - 1):
            edges[(i, i + 1)] = 1.0

    if sim_threshold is not None and n > 1:
        sim = cosine_similarity_matrix(features)
        np.fill_diagonal(sim, -np.inf)
        for i in range(n):
            candidates = np.flatnonzero(sim[i] > sim_threshold)
            if max_sim_neighbors and len(candidates) > max_sim_neighbors:
                candidates = candidates[np.argsort(-sim[i, candidates])[:max_sim_neighbors]]
            for j in candidates:
                key = (i, int(j))
                edges[key] = max(edges.get(key, 0.0), float(sim[i, j]))

    if undirected:
        for (i, j), w in list(edges.items()):
            edges.setdefault((j, i), w)

    if not edges:  # isolated single-segment track
        edges = {(i, i): 1.0 for i in range(n)}

    keys = sorted(edges)
    edge_index = np.asarray(keys, dtype=np.int64).T
    edge_weight = np.asarray([edges[k] for k in keys], dtype=np.float32)
    return edge_index, edge_weight


# --------------------------------------------------------------------------- #
# chord-transition graph
# --------------------------------------------------------------------------- #
def chord_templates() -> tuple[np.ndarray, list[str]]:
    """24 binary triad templates (12 major + 12 minor), L2-normalised."""
    templates, names = [], []
    for quality, intervals in (("maj", (0, 4, 7)), ("min", (0, 3, 7))):
        for root in range(12):
            vec = np.zeros(12, dtype=np.float32)
            for step in intervals:
                vec[(root + step) % 12] = 1.0
            templates.append(vec / np.linalg.norm(vec))
            names.append(f"{PITCH_CLASSES[root]}:{quality}")
    return np.vstack(templates), names


def _smooth_ids(ids: np.ndarray, window: int) -> np.ndarray:
    """Majority filter over a sliding window -- suppresses per-frame chord flicker."""
    if window <= 1 or len(ids) <= window:
        return ids
    half = window // 2
    padded = np.pad(ids, (half, half), mode="edge")
    out = np.empty_like(ids)
    for i in range(len(ids)):
        values, counts = np.unique(padded[i : i + window], return_counts=True)
        out[i] = values[np.argmax(counts)]
    return out


def estimate_chords(chroma: np.ndarray, smooth_window: int = 9) -> np.ndarray:
    """Per-frame chord id in [0, 24) by cosine matching against triad templates."""
    templates, _ = chord_templates()
    norms = np.clip(np.linalg.norm(chroma, axis=0, keepdims=True), 1e-8, None)
    scores = templates @ (chroma / norms)     # (24, T)
    return _smooth_ids(np.argmax(scores, axis=0), smooth_window)


def build_chord_graph(
    chroma: np.ndarray, sec_per_frame: float, smooth_window: int = 9, self_transitions: bool = False
):
    """Nodes = chords observed in the track; edges = transitions weighted by count.

    Node features (27 dims): mean chroma while the chord sounds (12), total
    duration in seconds (1), duration as a fraction of the track (1), occurrence
    count (1), minor flag (1), root one-hot (12).
    """
    frame_ids = estimate_chords(chroma, smooth_window)
    if len(frame_ids) == 0:
        return None

    # Collapse consecutive identical frames into chord events.
    change = np.flatnonzero(np.diff(frame_ids)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(frame_ids)]])
    sequence = frame_ids[starts]

    unique = np.unique(frame_ids)
    if len(unique) < 2:
        return None
    remap = {int(c): i for i, c in enumerate(unique)}

    durations = np.zeros(len(unique), dtype=np.float32)
    counts = np.zeros(len(unique), dtype=np.float32)
    chroma_sums = np.zeros((len(unique), chroma.shape[0]), dtype=np.float32)

    for chord, start, end in zip(sequence, starts, ends):
        node = remap[int(chord)]
        durations[node] += (end - start) * sec_per_frame
        counts[node] += 1.0
        chroma_sums[node] += chroma[:, start:end].sum(axis=1)

    frames_per_node = np.clip(durations / max(sec_per_frame, 1e-8), 1.0, None)
    mean_chroma = chroma_sums / frames_per_node[:, None]
    total_duration = max(float(durations.sum()), 1e-8)

    roots = np.eye(12, dtype=np.float32)[unique % 12]
    is_minor = (unique >= 12).astype(np.float32)[:, None]
    x = np.hstack(
        [
            mean_chroma,
            durations[:, None],
            (durations / total_duration)[:, None],
            counts[:, None],
            is_minor,
            roots,
        ]
    ).astype(np.float32)

    transitions: dict[tuple[int, int], float] = {}
    for a, b in zip(sequence[:-1], sequence[1:]):
        if a == b and not self_transitions:
            continue
        key = (remap[int(a)], remap[int(b)])
        transitions[key] = transitions.get(key, 0.0) + 1.0

    if not transitions:
        transitions = {(i, i): 1.0 for i in range(len(unique))}

    keys = sorted(transitions)
    edge_index = np.asarray(keys, dtype=np.int64).T
    counts_arr = np.asarray([transitions[k] for k in keys], dtype=np.float32)
    edge_weight = counts_arr / counts_arr.max()   # normalised transition frequency

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "chord_names": [chord_templates()[1][int(c)] for c in unique],
    }


# --------------------------------------------------------------------------- #
# per-track graph assembly
# --------------------------------------------------------------------------- #
def build_track_graph(bundle: dict, label_idx: int, track_id: str, gcfg: dict) -> Data | None:
    """Turn one cached feature bundle into a PyG `Data` object."""
    if gcfg["type"] == "chord":
        chord = build_chord_graph(
            bundle["chroma"],
            float(bundle["sec_per_frame"]),
            gcfg.get("chord_smooth_window", 9),
            gcfg.get("chord_self_transitions", False),
        )
        if chord is None:
            return None
        x, edge_index, edge_weight = chord["x"], chord["edge_index"], chord["edge_weight"]
        node_names = chord["chord_names"]
    else:
        x = bundle["node_features"]
        edge_index, edge_weight = build_segment_edges(
            x,
            temporal=gcfg["temporal_edges"],
            sim_threshold=gcfg["sim_threshold"],
            max_sim_neighbors=gcfg["max_sim_neighbors"],
            undirected=gcfg["undirected"],
        )
        node_names = None

    data = Data(
        x=torch.from_numpy(np.asarray(x, dtype=np.float32)),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_weight).unsqueeze(1),
        y=torch.tensor([label_idx], dtype=torch.long),
    )
    data.track_id = track_id
    if node_names is not None:
        data.node_names = node_names
    return data


# --------------------------------------------------------------------------- #
# dataset build
# --------------------------------------------------------------------------- #
def build_graph_dataset(cfg: dict, overwrite: bool = False) -> dict[str, list]:
    """Extract features and build one graph per track, grouped by split."""
    t2 = cfg["task2"]
    acfg = AudioConfig.from_dict(t2["audio"])
    gcfg = t2["graph"]
    df, classes = load_index(cfg)

    cache_dir = os.path.join(cfg["paths"]["processed_dir"], "audio_cache")
    graphs: dict[str, list] = {"train": [], "val": [], "test": []}
    failures = 0

    for n, row in enumerate(df.itertuples(index=False), start=1):
        bundle = extract_and_cache(row.path, str(row.track_id), cache_dir, acfg, overwrite)
        if bundle is None:
            failures += 1
            continue
        graph = build_track_graph(bundle, int(row.label_idx), str(row.track_id), gcfg)
        if graph is None:
            failures += 1
            print(f"[warn] no usable graph for {row.track_id}")
            continue
        graphs[row.split].append(graph)

        if n % 100 == 0:
            print(f"[graph] {n}/{len(df)} tracks processed")

    print(f"[graph] built {sum(len(v) for v in graphs.values())} graphs ({failures} skipped)")
    return graphs


def fit_node_scaler(train_graphs: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std over all TRAIN node features -- never fitted on val or test."""
    stacked = torch.cat([g.x for g in train_graphs], dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0).clamp(min=1e-6)
    return mean, std


def apply_node_scaler(graphs: list, mean: torch.Tensor, std: torch.Tensor) -> None:
    for g in graphs:
        g.x = (g.x - mean) / std


def save_graphs(graphs: dict[str, list], cfg: dict, classes: list[str]) -> str:
    t2 = cfg["task2"]
    out_dir = os.path.join(cfg["paths"]["processed_dir"], "graphs")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{t2['dataset'].lower()}_{t2['graph']['type']}"

    path = os.path.join(out_dir, f"task2_graphs_{tag}.pt")
    torch.save(
        {
            "graphs": graphs,
            "classes": classes,
            "graph_type": t2["graph"]["type"],
            "dataset": t2["dataset"],
            "node_dim": int(graphs["train"][0].x.shape[1]) if graphs["train"] else 0,
        },
        path,
    )
    print(f"[graph] wrote {path}")
    return path


def export_samples(graphs: dict[str, list], cfg: dict, classes: list[str], n: int = 20) -> None:
    """Write n individual graphs as .pt + .json (final submission requirement 2)."""
    sample_dir = os.path.join(cfg["paths"]["processed_dir"], "graph_samples")
    os.makedirs(sample_dir, exist_ok=True)

    pool = graphs["train"] + graphs["val"] + graphs["test"]
    for g in pool[:n]:
        stem = os.path.join(sample_dir, f"graph_{g.track_id}")
        torch.save(g, f"{stem}.pt")
        summary = {
            "track_id": g.track_id,
            "label": classes[int(g.y.item())],
            "num_nodes": int(g.num_nodes),
            "num_edges": int(g.edge_index.shape[1]),
            "node_dim": int(g.x.shape[1]),
            "edge_index": g.edge_index.tolist(),
            "edge_weight": g.edge_attr.squeeze(1).tolist(),
        }
        if hasattr(g, "node_names"):
            summary["node_names"] = g.node_names
        with open(f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(f"[graph] exported {min(n, len(pool))} sample graphs to {sample_dir}")


def graph_statistics(graphs: dict[str, list]) -> dict:
    """Node/edge/degree summary for the report's dataset section."""
    stats = {}
    for split, items in graphs.items():
        if not items:
            continue
        nodes = np.array([g.num_nodes for g in items])
        edges = np.array([g.edge_index.shape[1] for g in items])
        stats[split] = {
            "graphs": len(items),
            "nodes_mean": float(nodes.mean()),
            "nodes_min": int(nodes.min()),
            "nodes_max": int(nodes.max()),
            "edges_mean": float(edges.mean()),
            "mean_degree": float((edges / np.clip(nodes, 1, None)).mean()),
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build music structure graphs for Task 2.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--graph-type", default=None, help="segment | chord")
    parser.add_argument("--dataset", default=None, help="gtzan | fma_small")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--n-samples", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.graph_type:
        cfg["task2"]["graph"]["type"] = args.graph_type
    if args.dataset:
        cfg["task2"]["dataset"] = args.dataset

    _, classes = load_index(cfg)
    graphs = build_graph_dataset(cfg, args.overwrite_cache)

    if not graphs["train"]:
        raise SystemExit("no training graphs were built -- check the audio paths in config.yaml")

    mean, std = fit_node_scaler(graphs["train"])
    for split in graphs:
        apply_node_scaler(graphs[split], mean, std)

    save_graphs(graphs, cfg, classes)
    export_samples(graphs, cfg, classes, args.n_samples)

    stats = graph_statistics(graphs)
    metrics_dir = cfg["paths"]["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "task2_graph_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("\n[stats]", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
