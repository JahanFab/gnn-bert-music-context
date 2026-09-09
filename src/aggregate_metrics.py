"""Roll the per-task results/metrics/*.json files into one results/metrics.json.

The project brief's repo layout lists a single `results/metrics.json`; this
produces it as a compact digest of the headline numbers. Re-run after any
experiment:

    python src/aggregate_metrics.py
"""
import json
import os

MDIR = "results/metrics"


def L(name):
    with open(os.path.join(MDIR, name), encoding="utf-8") as f:
        return json.load(f)


mc = L("task1_mask_comparison.json")
t2 = L("task2_results.json")
t3 = L("task3_results.json")
t4 = L("task4_results.json")
dl = L("musiccaps_download_log.json")
cd = L("task3_coherence_diagnostic.json")

agg = {
    "_note": "Digest of results/metrics/*.json. Regenerate with python src/aggregate_metrics.py",
    "task1_bert_tags": {
        "dataset": "MusicCaps caption->tag proxy (top-50)",
        "n_test": mc["n_test"],
        "unmasked": {
            "bert_tuned_macro_f1": mc["unmasked"]["bert_tuned"]["macro_f1"],
            "bert_tuned_micro_f1": mc["unmasked"]["bert_tuned"]["micro_f1"],
            "bert_auc_pr": mc["unmasked"]["bert_tuned"]["auc_pr"],
            "tfidf_macro_f1": mc["unmasked"]["baseline_tfidf_logreg"]["macro_f1"],
            "random_macro_f1": mc["unmasked"]["baseline_random_prior"]["macro_f1"],
            "majority_macro_f1": mc["unmasked"]["baseline_majority"]["macro_f1"],
            "best_epoch": mc["unmasked"]["best_epoch"],
        },
        "masked": {
            "bert_tuned_macro_f1": mc["masked"]["bert_tuned"]["macro_f1"],
            "tfidf_macro_f1": mc["masked"]["baseline_tfidf_logreg"]["macro_f1"],
            "best_epoch": mc["masked"]["best_epoch"],
        },
    },
    "task2_gnn_vs_cnn": {
        "dataset": "GTZAN (10 genres, 150 test)",
        "gnn_sage": {
            "params": t2["gnn_sage"]["parameters"],
            "accuracy": t2["gnn_sage"]["test"]["accuracy"],
            "macro_f1": t2["gnn_sage"]["test"]["macro_f1"],
            "top3_accuracy": t2["gnn_sage"]["test"]["top3_accuracy"],
        },
        "cnn_melspec_B2": {
            "params": t2["cnn_melspec"]["parameters"],
            "accuracy": t2["cnn_melspec"]["test"]["accuracy"],
            "macro_f1": t2["cnn_melspec"]["test"]["macro_f1"],
            "top3_accuracy": t2["cnn_melspec"]["test"]["top3_accuracy"],
        },
    },
    "task3_fusion_ablation": {"dataset": "MagnaTagATune (top-50, 4332 test), tuned thresholds"},
    "task3_graph_coherence": {
        "s_graph_at_tau_0.5": cd["s_graph_at_configured_tau"],
        "edge_cosine_mean": cd["edge_cosine_stats"]["mean"],
        "verdict": "saturated - report the cosine distribution, not the scalar",
    },
    "task4_contrastive_retrieval": {
        "dataset": "MagnaTagATune paired (4332 candidates)",
        "best_epoch": t4["best_epoch"],
        "audio_to_caption": {k: t4["retrieval"]["audio_to_caption"][k] for k in ("R@1", "R@5", "R@10", "mrr")},
        "caption_to_audio": {k: t4["retrieval"]["caption_to_audio"][k] for k in ("R@1", "R@5", "R@10", "mrr")},
        "zero_shot_graph_macro_f1": t4["zero_shot_from_graph"]["macro_f1"],
        "zero_shot_caption_macro_f1": t4["zero_shot_from_caption"]["macro_f1"],
    },
    "musiccaps_availability": {
        "nominal_clips": dl["nominal_clips"],
        "clips_present": dl["clips_present"],
        "fraction_present": dl["fraction_present"],
    },
}

for m in ("bert_only", "gnn_only", "concat", "cross_attention"):
    s = t3[m]["test_tuned_threshold"]
    agg["task3_fusion_ablation"][m] = {
        "macro_f1": s["macro_f1"], "micro_f1": s["micro_f1"], "auc_pr": s["auc_pr"],
        "best_epoch": t3[m]["best_epoch"], "params": t3[m]["parameters"],
    }

with open("results/metrics.json", "w", encoding="utf-8") as f:
    json.dump(agg, f, indent=2)
print("wrote results/metrics.json")
print(json.dumps(agg, indent=2))
