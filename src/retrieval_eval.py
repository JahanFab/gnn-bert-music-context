"""Retrieval metrics, qualitative examples and zero-shot tagging for Task 4.

Covers spec Section 4.4's evaluation deliverables:

  * Caption -> Audio R@1/5/10 and Audio -> Caption R@K
  * 10 qualitative retrieval examples (query caption -> top-3 clips)
  * zero-shot tag prediction, compared against the Task 3 supervised model
  * a rating sheet for the Section 6 human evaluation (>= 5 listeners, scale 1-5)
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np

from metrics import macro_auc_pr, multilabel_scores


# --------------------------------------------------------------------------- #
# retrieval metrics
# --------------------------------------------------------------------------- #
def _ranks_of_matches(scores: np.ndarray) -> np.ndarray:
    """1-based rank of the correct partner for each query row.

    `scores[i, j]` scores query i against candidate j; the correct partner is the
    diagonal. Ties are broken pessimistically: every candidate scoring >= the
    correct one counts as ranked ahead of it. Counting only strictly-greater
    candidates would hand a perfect R@1 to a collapsed model that emits the same
    score for every pair.

    The `>=` comparison includes the correct candidate itself, which is what
    makes the result 1-based: a strictly-best match ranks 1.
    """
    diagonal = np.diag(scores)[:, None]
    return (scores >= diagonal).sum(axis=1)


def retrieval_metrics(sim: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """Both retrieval directions from one graph-by-text similarity matrix.

    `sim[i, j]` = similarity of graph i to caption j.
      caption_to_audio : query is a caption, candidates are clips -> rank down a column
      audio_to_caption : query is a clip, candidates are captions -> rank across a row
    """
    out = {"n_candidates": int(sim.shape[0])}

    for name, scores in (("audio_to_caption", sim), ("caption_to_audio", sim.T)):
        ranks = _ranks_of_matches(scores)
        block = {f"R@{k}": float((ranks <= k).mean()) for k in ks}
        block["median_rank"] = float(np.median(ranks))
        block["mean_rank"] = float(ranks.mean())
        block["mrr"] = float((1.0 / ranks).mean())
        out[name] = block

    out["mean_R@1"] = 0.5 * (out["audio_to_caption"]["R@1"] + out["caption_to_audio"]["R@1"])
    out["mean_R@5"] = 0.5 * (out["audio_to_caption"]["R@5"] + out["caption_to_audio"]["R@5"])
    return out


def print_retrieval_table(metrics: dict, title: str = "Task 4 retrieval") -> None:
    print(f"\n{title} (N = {metrics['n_candidates']} candidates)")
    print(f"{'Direction':<20}{'R@1':>9}{'R@5':>9}{'R@10':>9}{'MedR':>8}{'MRR':>9}")
    print("-" * 64)
    for direction in ("caption_to_audio", "audio_to_caption"):
        b = metrics[direction]
        print(
            f"{direction:<20}{b['R@1']:>9.4f}{b['R@5']:>9.4f}{b['R@10']:>9.4f}"
            f"{b['median_rank']:>8.0f}{b['mrr']:>9.4f}"
        )


# --------------------------------------------------------------------------- #
# qualitative examples + human evaluation sheet
# --------------------------------------------------------------------------- #
def load_clip_times(cfg: dict) -> dict[str, float]:
    """ytid -> clip start second, so rating links jump to the right moment."""
    csv_path = os.path.join(cfg["paths"]["raw_dir"], cfg["data"]["csv_name"])
    if not os.path.exists(csv_path):
        return {}
    import pandas as pd

    df = pd.read_csv(csv_path, usecols=["ytid", "start_s"])
    return {str(r.ytid): float(r.start_s) for r in df.itertuples(index=False)}


def youtube_link(track_id: str, start_s: float | None) -> str:
    if start_s is None:
        return f"https://www.youtube.com/watch?v={track_id}"
    return f"https://www.youtube.com/watch?v={track_id}&t={int(start_s)}s"


def qualitative_examples(
    sim: np.ndarray,
    track_ids: list[str],
    captions: list[str],
    clip_times: dict[str, float],
    n_examples: int = 10,
    top_k: int = 3,
    seed: int = 42,
) -> list[dict]:
    """n caption queries, each with its top-k retrieved clips."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(track_ids), size=min(n_examples, len(track_ids)), replace=False)

    examples = []
    for query_idx in picks:
        column = sim[:, query_idx]                    # all clips scored against this caption
        order = np.argsort(-column)[:top_k]
        rank_of_truth = int((column > column[query_idx]).sum() + 1)

        examples.append(
            {
                "query_track_id": track_ids[query_idx],
                "query_caption": captions[query_idx],
                "rank_of_correct_clip": rank_of_truth,
                "hit_at_k": bool(rank_of_truth <= top_k),
                "retrieved": [
                    {
                        "rank": rank,
                        "track_id": track_ids[c],
                        "score": float(column[c]),
                        "is_correct": bool(c == query_idx),
                        "caption": captions[c][:300],
                        "url": youtube_link(track_ids[c], clip_times.get(track_ids[c])),
                    }
                    for rank, c in enumerate(order, start=1)
                ],
            }
        )
    return examples


def write_human_eval_sheet(examples: list[dict], out_path: str, n_raters: int = 5) -> None:
    """Blank rating sheet for the Section 6 listening study.

    One row per (rater, query, retrieved clip). `is_correct` is deliberately left
    out of the sheet -- showing raters which clip is the ground-truth pair would
    bias every rating collected.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["rater_id", "query_id", "query_caption", "rank", "clip_id", "clip_url", "rating_1_to_5", "notes"]
        )
        for rater in range(1, n_raters + 1):
            for q, example in enumerate(examples, start=1):
                for item in example["retrieved"]:
                    writer.writerow(
                        [rater, q, example["query_caption"], item["rank"], item["track_id"], item["url"], "", ""]
                    )

    print(f"[eval] wrote human evaluation sheet -> {out_path} "
          f"({n_raters} raters x {len(examples)} queries)")


def write_examples_html(examples: list[dict], out_path: str) -> None:
    """Readable side-by-side view of the retrieval examples for the report."""
    rows = []
    for i, example in enumerate(examples, start=1):
        items = "".join(
            f"<li{' style=\"font-weight:bold;color:#046\"' if item['is_correct'] else ''}>"
            f"#{item['rank']} &middot; score {item['score']:.3f} &middot; "
            f"<a href='{item['url']}'>{item['track_id']}</a><br>"
            f"<small>{item['caption']}</small></li>"
            for item in example["retrieved"]
        )
        rows.append(
            f"<div class='q'><h3>Query {i} &mdash; correct clip ranked "
            f"{example['rank_of_correct_clip']}</h3>"
            f"<p class='cap'>{example['query_caption']}</p><ol>{items}</ol></div>"
        )

    html = (
        "<!doctype html><meta charset='utf-8'><title>Task 4 retrieval examples</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem}"
        ".q{border-top:1px solid #ccc;padding:1rem 0}.cap{background:#f4f4f4;padding:.6rem;border-radius:4px}"
        "li{margin:.5rem 0}</style>"
        "<h1>Task 4 &mdash; caption &rarr; audio retrieval</h1>"
        "<p>Bold entries are the ground-truth pair.</p>" + "".join(rows)
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[eval] wrote {out_path}")


# --------------------------------------------------------------------------- #
# zero-shot tagging
# --------------------------------------------------------------------------- #
def zero_shot_scores(graph_emb: np.ndarray, tag_emb: np.ndarray) -> np.ndarray:
    """Cosine similarity of every clip against every tag prompt (both normalised)."""
    return graph_emb @ tag_emb.T


def zero_shot_tag_metrics(scores: np.ndarray, targets: np.ndarray, top_n: int | None = None) -> dict:
    """Rank-based evaluation of zero-shot tagging.

    There is no validation set to calibrate a threshold on -- that is what makes
    it zero-shot -- so the decision rule is "take the top-n tags", with n set to
    the average number of true tags per clip. AUC-PR is reported alongside
    because it is threshold-free and therefore the fairer headline number.
    """
    if top_n is None:
        top_n = max(1, int(round(targets.sum(axis=1).mean())))

    preds = np.zeros_like(targets, dtype=np.float32)
    ranked = np.argsort(-scores, axis=1)[:, :top_n]
    np.put_along_axis(preds, ranked, 1.0, axis=1)

    result = multilabel_scores(preds, targets, 0.5)
    result["auc_pr"] = macro_auc_pr(scores, targets.astype(np.int8))
    result["top_n"] = top_n
    return result


def compare_with_task3(zero_shot: dict, metrics_dir: str) -> dict:
    """Put the zero-shot row next to the Task 3 supervised numbers, if present."""
    path = os.path.join(metrics_dir, "task3_results.json")
    comparison = {"task4_zero_shot": zero_shot}
    if not os.path.exists(path):
        print("[eval] no task3_results.json found -- skipping the supervised comparison")
        return comparison

    with open(path, encoding="utf-8") as f:
        task3 = json.load(f)
    for mode, res in task3.items():
        block = res.get("test_tuned_threshold") or res.get("test_default_threshold")
        if block:
            comparison[f"task3_{mode}"] = {
                "macro_f1": block["macro_f1"],
                "micro_f1": block["micro_f1"],
                "auc_pr": block["auc_pr"],
            }
    return comparison


def print_zero_shot_table(comparison: dict) -> None:
    print(f"\n{'Model':<28}{'Macro-F1':>10}{'Micro-F1':>10}{'AUC-PR':>10}")
    print("-" * 58)
    for name, block in comparison.items():
        print(f"{name:<28}{block['macro_f1']:>10.4f}{block['micro_f1']:>10.4f}{block['auc_pr']:>10.4f}")
