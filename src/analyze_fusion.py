"""Task 3 analysis: t-SNE of z, case studies, and graph coherence.

Covers the three interpretive deliverables of spec Section 4.3:

  * t-SNE of the fused representation z, coloured by genre and by mood
  * 3 case studies showing graph paths alongside caption alignment
  * graph coherence score from Section 6:

        S_graph = (1/|E|) sum_{(i,j) in E} 1[ cos(h_i, h_j) > tau ]

    python src/analyze_fusion.py --config config.yaml --ckpt results/checkpoints/task3_cross_attention_best.pt
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
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from audio_datasets import load_config
from data_fusion import FusionDataset, collate_fusion, load_fusion_data
from fusion_model import GNNBertFusion

# Coarse groupings over the MusicCaps aspect vocabulary. These are *derived*
# views for colouring the projection, not supervision -- the model never sees
# them. A clip matching no keyword is labelled "other" rather than guessed at.
GENRE_KEYWORDS = {
    "rock": ["rock", "punk", "metal", "grunge"],
    "electronic": ["electronic", "edm", "techno", "house", "synth", "trance", "dance"],
    "hip-hop": ["hip hop", "hip-hop", "rap", "trap"],
    "jazz/blues": ["jazz", "blues", "swing", "soul", "funk"],
    "classical": ["classical", "orchestra", "orchestral", "symphony", "chamber", "opera"],
    "folk/country": ["folk", "country", "bluegrass", "acoustic guitar"],
    "pop": ["pop", "ballad"],
    "latin/world": ["latin", "reggae", "salsa", "african", "indian", "world"],
}

MOOD_KEYWORDS = {
    "happy": ["happy", "cheerful", "joyful", "uplifting", "playful"],
    "sad": ["sad", "melancholic", "melancholy", "sorrowful", "dark", "gloomy"],
    "energetic": ["energetic", "uptempo", "fast tempo", "aggressive", "intense", "driving"],
    "calm": ["calm", "mellow", "soft", "relaxing", "gentle", "slow tempo", "soothing"],
    "emotional": ["emotional", "passionate", "soulful", "romantic", "sentimental"],
    "groovy": ["groovy", "funky", "danceable", "rhythmic"],
}


def coarse_label(tags: list[str], keyword_map: dict[str, list[str]]) -> str:
    """First keyword group matched by any of the clip's tags."""
    joined = " | ".join(t.lower() for t in tags)
    for group, keywords in keyword_map.items():
        if any(k in joined for k in keywords):
            return group
    return "other"


# --------------------------------------------------------------------------- #
# model + embeddings
# --------------------------------------------------------------------------- #
def load_fusion_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg, mode, label_vocab, node_dim = ckpt["config"], ckpt["mode"], ckpt["label_vocab"], ckpt["node_dim"]
    mcfg, lcfg = cfg["task3"]["model"], cfg["task3"]["loss"]

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
        predict_emotion=lcfg["alpha"] > 0 or lcfg["beta"] > 0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"[analysis] loaded {ckpt_path} (mode={mode}, epoch={ckpt['epoch']})")
    return model, cfg, label_vocab, mode


@torch.no_grad()
def collect_embeddings(model, loader, device: torch.device) -> dict:
    """Fused vectors z, tag probabilities and attention rows for a whole split."""
    z_all, prob_all, target_all, ids = [], [], [], []

    for batch in loader:
        graph = batch["graph"].to(device) if model.mode != "bert_only" else None
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model(graph, input_ids, attention_mask)
        z_all.append(out["z"].float().cpu())
        prob_all.append(torch.sigmoid(out["tag_logits"]).float().cpu())
        target_all.append(batch["y_tags"])
        ids.extend(batch["track_id"])

    return {
        "z": torch.cat(z_all).numpy(),
        "probs": torch.cat(prob_all).numpy(),
        "targets": torch.cat(target_all).numpy(),
        "track_ids": ids,
    }


# --------------------------------------------------------------------------- #
# t-SNE
# --------------------------------------------------------------------------- #
def plot_tsne(z: np.ndarray, labels: list[str], title: str, out_path: str, seed: int = 42) -> None:
    perplexity = float(min(30, max(5, (len(z) - 1) / 3)))
    coords = TSNE(
        n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed
    ).fit_transform(z)

    groups = sorted(set(labels), key=lambda g: (g == "other", g))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 2)))

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for color, group in zip(colors, groups):
        idx = [i for i, l in enumerate(labels) if l == group]
        ax.scatter(
            coords[idx, 0], coords[idx, 1], s=14, alpha=0.35 if group == "other" else 0.75,
            color="lightgrey" if group == "other" else color, label=f"{group} ({len(idx)})",
        )

    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, markerscale=1.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path} (perplexity {perplexity:.0f})")


# --------------------------------------------------------------------------- #
# graph paths + coherence
# --------------------------------------------------------------------------- #
@torch.no_grad()
def node_embeddings(model, graph, device: torch.device) -> torch.Tensor:
    graph = graph.to(device)
    edge_weight = graph.edge_attr.squeeze(-1) if getattr(graph, "edge_attr", None) is not None else None
    return model.gnn.encode_nodes(graph.x, graph.edge_index, edge_weight).cpu()


def node_importance(h: torch.Tensor) -> np.ndarray:
    """Projection of each node onto the pooled graph vector.

    With mean-pool readout g = (1/|V|) sum_i h_i, every node contributes equally
    in magnitude, so raw norms say little. The projection h_i . g_hat measures how
    much each node pushes g in the direction the classifier actually reads.
    """
    g = h.mean(dim=0)
    g_hat = g / g.norm().clamp(min=1e-8)
    return (h @ g_hat).numpy()


def extract_graph_path(graph, scores: np.ndarray, max_len: int = 6) -> list[int]:
    """Greedy highest-weight walk starting from the most important node.

    For a chord graph this returns an actual chord progression (the C -> G -> Am
    of the project motivation); for a segment graph it is the dominant chain of
    structurally related segments.
    """
    edge_index = graph.edge_index.cpu().numpy()
    weights = (
        graph.edge_attr.squeeze(-1).cpu().numpy()
        if getattr(graph, "edge_attr", None) is not None
        else np.ones(edge_index.shape[1], dtype=np.float32)
    )

    neighbours: dict[int, list[tuple[int, float]]] = {}
    for e in range(edge_index.shape[1]):
        neighbours.setdefault(int(edge_index[0, e]), []).append((int(edge_index[1, e]), float(weights[e])))

    current = int(np.argmax(scores))
    path, visited = [current], {current}
    while len(path) < max_len:
        candidates = [(j, w) for j, w in neighbours.get(current, []) if j not in visited]
        if not candidates:
            break
        # Prefer edges that are both strong and lead somewhere the model weights highly.
        current = max(candidates, key=lambda jw: jw[1] * (1.0 + max(scores[jw[0]], 0.0)))[0]
        path.append(current)
        visited.add(current)
    return path


def graph_coherence(h: torch.Tensor, graph, tau: float = 0.5) -> float:
    """S_graph = fraction of edges whose endpoints have cosine similarity > tau."""
    edge_index = graph.edge_index.to(h.device)
    if edge_index.shape[1] == 0:
        return 0.0
    unit = h / h.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sims = (unit[edge_index[0]] * unit[edge_index[1]]).sum(dim=1)
    return float((sims > tau).float().mean())


# --------------------------------------------------------------------------- #
# case studies
# --------------------------------------------------------------------------- #
@torch.no_grad()
def case_studies(
    model, dataset: FusionDataset, tokenizer, label_vocab, device, cfg, n_cases: int = 3
) -> list[dict]:
    """Graph path + attended caption tokens + predictions, for n test clips."""
    plots_dir = cfg["paths"]["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)

    rng = np.random.default_rng(cfg["seed"])
    # Prefer clips with enough structure for a path to be meaningful.
    candidates = [i for i in range(len(dataset)) if dataset[i]["graph"].num_nodes >= 4]
    if len(candidates) < n_cases:
        candidates = list(range(len(dataset)))
    picks = rng.choice(candidates, size=min(n_cases, len(candidates)), replace=False)

    cases = []
    for rank, idx in enumerate(picks, start=1):
        item = dataset[int(idx)]
        batch = collate_fusion([item])
        graph = batch["graph"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model(graph, input_ids, attention_mask, return_attention=True)
        probs = torch.sigmoid(out["tag_logits"])[0].float().cpu().numpy()

        h = node_embeddings(model, item["graph"], device)
        scores = node_importance(h)
        path = extract_graph_path(item["graph"], scores)
        coherence = graph_coherence(h, item["graph"], cfg["task3"]["analysis"]["coherence_tau"])

        n_real = int(attention_mask[0].sum().item())
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0][:n_real].cpu())
        attn = (
            out["attention"][0][:n_real].float().cpu().numpy()
            if out.get("attention") is not None
            else np.zeros(n_real, dtype=np.float32)
        )

        true_tags = [label_vocab[k] for k in np.flatnonzero(item["y_tags"].numpy() > 0)]
        top_tags = [(label_vocab[k], float(probs[k])) for k in np.argsort(-probs)[:8]]
        node_names = getattr(item["graph"], "node_names", None)
        path_labels = [node_names[i] for i in path] if node_names else [f"seg{i}" for i in path]

        plot_path = os.path.join(plots_dir, f"task3_case_{rank}_{item['track_id']}.png")
        plot_case(item["graph"], scores, path, tokens, attn, path_labels, item["track_id"], plot_path)

        cases.append(
            {
                "track_id": item["track_id"],
                "num_nodes": int(item["graph"].num_nodes),
                "num_edges": int(item["graph"].edge_index.shape[1]),
                "graph_path": path_labels,
                "graph_coherence": coherence,
                "true_tags": true_tags,
                "top_predicted_tags": top_tags,
                "top_attended_tokens": [
                    tokens[i] for i in np.argsort(-attn)[:10] if tokens[i] not in ("[CLS]", "[SEP]", "[PAD]")
                ][:8],
                "figure": plot_path,
            }
        )
    return cases


def plot_case(graph, scores, path, tokens, attn, path_labels, track_id, out_path) -> None:
    """Left: the graph with the extracted path. Right: caption attention."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.1, 1.4]})

    n = int(graph.num_nodes)
    xs = np.arange(n)
    edge_index = graph.edge_index.cpu().numpy()
    weights = (
        graph.edge_attr.squeeze(-1).cpu().numpy()
        if getattr(graph, "edge_attr", None) is not None
        else np.ones(edge_index.shape[1])
    )

    ax = axes[0]
    for e in range(edge_index.shape[1]):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        if i >= j:  # draw each undirected pair once
            continue
        mid = (i + j) / 2
        height = 0.12 * (j - i)
        ax.plot([i, mid, j], [0, height, 0], color="grey", alpha=min(0.85, 0.15 + 0.6 * weights[e]), lw=1.0)

    ax.scatter(xs, np.zeros(n), s=60 + 220 * _minmax(scores), c=scores, cmap="viridis", zorder=3)
    for a, b in zip(path[:-1], path[1:]):
        ax.annotate(
            "", xy=(b, 0), xytext=(a, 0),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=2.0, shrinkA=6, shrinkB=6),
            zorder=4,
        )
    ax.set_title(f"structure graph + dominant path\n{' -> '.join(path_labels)}", fontsize=9)
    ax.set_xlabel("node index (temporal order)")
    ax.set_yticks([])

    ax = axes[1]
    order = np.argsort(-attn)[:20]
    order = order[np.argsort(order)]
    ax.bar(np.arange(len(order)), attn[order], color="#4c72b0")
    ax.set_xticks(np.arange(len(order)), labels=[tokens[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylabel("cross-attention weight")
    ax.set_title(f"graph-to-caption attention -- {track_id}", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.full_like(x, 0.5)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3 analysis: t-SNE, case studies, coherence.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--n-cases", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"].startswith("cuda") else "cpu")
    ckpt = args.ckpt or os.path.join(cfg["paths"]["ckpt_dir"], "task3_cross_attention_best.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}\nrun `python src/train_fusion.py` first")

    model, ckpt_cfg, label_vocab, mode = load_fusion_checkpoint(ckpt, device)
    cfg["task3"] = ckpt_cfg["task3"]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["task3"]["model"]["text_model"])
    index, graphs, _, _ = load_fusion_data(cfg)
    test_frame = index[(index["split"] == "test") & (index["source"] != "deam")].reset_index(drop=True)
    dataset = FusionDataset(test_frame, graphs, label_vocab, tokenizer, cfg["task3"]["max_length"])
    loader = DataLoader(dataset, batch_size=cfg["task3"]["train"]["batch_size"], shuffle=False, collate_fn=collate_fusion)

    embeddings = collect_embeddings(model, loader, device)
    tags_per_clip = [
        [label_vocab[k] for k in np.flatnonzero(row > 0)] for row in embeddings["targets"]
    ]
    genre_labels = [coarse_label(tags, GENRE_KEYWORDS) for tags in tags_per_clip]
    mood_labels = [coarse_label(tags, MOOD_KEYWORDS) for tags in tags_per_clip]

    plots_dir, metrics_dir = cfg["paths"]["plots_dir"], cfg["paths"]["metrics_dir"]
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    plot_tsne(embeddings["z"], genre_labels, f"t-SNE of z by genre ({mode})",
              os.path.join(plots_dir, "task3_tsne_genre.png"), cfg["seed"])
    plot_tsne(embeddings["z"], mood_labels, f"t-SNE of z by mood ({mode})",
              os.path.join(plots_dir, "task3_tsne_mood.png"), cfg["seed"])

    cases = []
    coherence_scores = []
    if model.gnn is not None:
        cases = case_studies(model, dataset, tokenizer, label_vocab, device, cfg, args.n_cases)
        tau = cfg["task3"]["analysis"]["coherence_tau"]
        for i in range(min(len(dataset), cfg["task3"]["analysis"]["coherence_sample"])):
            graph = dataset[i]["graph"]
            coherence_scores.append(graph_coherence(node_embeddings(model, graph, device), graph, tau))

    summary = {
        "mode": mode,
        "n_test": len(dataset),
        "genre_distribution": {g: genre_labels.count(g) for g in sorted(set(genre_labels))},
        "mood_distribution": {m: mood_labels.count(m) for m in sorted(set(mood_labels))},
        "graph_coherence_mean": float(np.mean(coherence_scores)) if coherence_scores else None,
        "graph_coherence_tau": cfg["task3"]["analysis"]["coherence_tau"],
        "case_studies": cases,
    }
    out_path = os.path.join(metrics_dir, "task3_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[analysis] wrote {out_path}")

    if coherence_scores:
        print(f"[analysis] mean graph coherence S_graph = {np.mean(coherence_scores):.4f}")
    for case in cases:
        print(f"\n  {case['track_id']}")
        print(f"    path     : {' -> '.join(case['graph_path'])}")
        print(f"    true     : {', '.join(case['true_tags'][:8])}")
        print(f"    predicted: {', '.join(t for t, _ in case['top_predicted_tags'][:6])}")
        print(f"    attended : {', '.join(case['top_attended_tokens'])}")


if __name__ == "__main__":
    main()
