"""Sanity-check the Task 3 graph-coherence score S_graph.

`task3_analysis.json` reports mean S_graph = 1.0000 at tau = 0.5. A perfect score
across every sampled graph is more likely a saturated metric than a real result:
mean-pool GraphSAGE embeddings can collapse toward a shared direction, which makes
*every* edge clear a low cosine threshold.

This script re-derives the underlying quantity -- the cosine similarity of every
edge's two endpoint embeddings -- and reports:

  * the distribution of those similarities (percentiles), and
  * mean S_graph swept over a range of tau,

so the report can either raise `task3.analysis.coherence_tau` to a value that
actually discriminates, or state plainly that the node embeddings are near-collinear
and the metric is uninformative on this run.

    python src/diagnose_coherence.py --config config.yaml
    python src/diagnose_coherence.py --config config.yaml --ckpt results/checkpoints/task3_gnn_only_best.pt
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from analyze_fusion import load_fusion_checkpoint, node_embeddings
from audio_datasets import load_config
from data_fusion import load_fusion_data

TAU_GRID = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
PERCENTILES = [1, 5, 25, 50, 75, 90, 95, 99]


@torch.no_grad()
def edge_cosines(model, graphs: list, device: torch.device, max_graphs: int) -> np.ndarray:
    """Cosine similarity of both endpoint embeddings, pooled over every edge."""
    sims: list[np.ndarray] = []
    used = 0
    for graph in graphs:
        if graph.edge_index.shape[1] == 0:
            continue
        h = node_embeddings(model, graph, device)  # returns CPU tensor
        unit = h / h.norm(dim=1, keepdim=True).clamp(min=1e-8)
        ei = graph.edge_index.cpu()  # node_embeddings moved the graph to device in place
        s = (unit[ei[0]] * unit[ei[1]]).sum(dim=1).numpy()
        sims.append(s)
        used += 1
        if used >= max_graphs:
            break
    if not sims:
        return np.zeros(0, dtype=np.float32)
    print(f"[diag] pooled {sum(len(s) for s in sims)} edges from {used} graphs")
    return np.concatenate(sims)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the saturated S_graph coherence score.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--max-graphs", type=int, default=400)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"].startswith("cuda") else "cpu")
    ckpt = args.ckpt or os.path.join(cfg["paths"]["ckpt_dir"], "task3_cross_attention_best.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")

    model, ckpt_cfg, _, mode = load_fusion_checkpoint(ckpt, device)
    if model.gnn is None:
        raise SystemExit(f"{mode} has no GNN branch -- pick a checkpoint that does")
    cfg["task3"] = ckpt_cfg["task3"]

    index, graphs_map, _, _ = load_fusion_data(cfg)
    test_ids = index.loc[index["split"] == "test", "track_id"].astype(str)
    test_graphs = [graphs_map[t] for t in test_ids if t in graphs_map]
    print(f"[diag] {len(test_graphs)} test graphs available")

    sims = edge_cosines(model, test_graphs, device, args.max_graphs)
    if sims.size == 0:
        raise SystemExit("no edges found")

    configured_tau = float(cfg["task3"]["analysis"]["coherence_tau"])
    s_graph_by_tau = {f"{t:.3f}": float((sims > t).mean()) for t in TAU_GRID}
    pct = {f"p{p}": float(np.percentile(sims, p)) for p in PERCENTILES}

    summary = {
        "checkpoint": os.path.basename(ckpt),
        "mode": mode,
        "n_edges_pooled": int(sims.size),
        "n_graphs_used": min(len(test_graphs), args.max_graphs),
        "configured_coherence_tau": configured_tau,
        "s_graph_at_configured_tau": float((sims > configured_tau).mean()),
        "edge_cosine_stats": {
            "min": float(sims.min()),
            "mean": float(sims.mean()),
            "max": float(sims.max()),
            "std": float(sims.std()),
            **pct,
        },
        "s_graph_vs_tau": s_graph_by_tau,
        "verdict": (
            "SATURATED: >99% of edges exceed the configured tau; the score cannot "
            "discriminate structure quality on this run. Raise coherence_tau toward "
            "the p50-p90 of the edge-cosine distribution, or report the distribution "
            "itself instead of a single thresholded number."
            if (sims > configured_tau).mean() > 0.99
            else "Not saturated at the configured tau; S_graph is meaningful as reported."
        ),
    }

    out_path = os.path.join(cfg["paths"]["metrics_dir"], "task3_coherence_diagnostic.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[diag] edge cosine: mean {sims.mean():.4f}  p50 {np.percentile(sims, 50):.4f}  "
          f"p90 {np.percentile(sims, 90):.4f}  min {sims.min():.4f}")
    print("[diag] S_graph vs tau:")
    for t, v in s_graph_by_tau.items():
        print(f"         tau={t}: {v:.4f}")
    print(f"\n[diag] {summary['verdict']}")
    print(f"[diag] wrote {out_path}")


if __name__ == "__main__":
    main()
