"""Identify exactly which MagnaTagATune clips were dropped between indexing and
the saved Task 3 fusion dataset."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_datasets import load_config
from data_fusion import build_paired_index

cfg = load_config("config.yaml")
cfg["task3"]["dataset"] = "magnatagatune"

index, vocab = build_paired_index(cfg)
indexed_ids = [str(t) for t in index["track_id"].tolist()]
print(f"indexed rows: {len(indexed_ids)} (unique {len(set(indexed_ids))})")

blob = torch.load("data/processed/fusion/task3_fusion_magnatagatune.pt", map_location="cpu", weights_only=False)
kept_ids = set(str(k) for k in blob["graphs"].keys())
print(f"kept graphs : {len(kept_ids)}")

skipped = [tid for tid in indexed_ids if tid not in kept_ids]
print(f"\nSKIPPED ({len(skipped)}):")
rows = index[index["track_id"].astype(str).isin(skipped)]
for _, r in rows.iterrows():
    p = r["path"]
    print(f"  clip_id={r['track_id']:>7}  split={r['split']:<5}  exists={os.path.exists(p)}  size={os.path.getsize(p) if os.path.exists(p) else 0}  path={p}")

out = {
    "n_indexed": len(indexed_ids),
    "n_kept": len(kept_ids),
    "n_skipped": len(skipped),
    "skipped": [
        {
            "clip_id": str(r["track_id"]),
            "split": r["split"],
            "path": r["path"],
            "audio_present": os.path.exists(r["path"]),
            "bytes": os.path.getsize(r["path"]) if os.path.exists(r["path"]) else 0,
        }
        for _, r in rows.iterrows()
    ],
}
with open("results/metrics/task3_skipped_clips.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("\nwrote results/metrics/task3_skipped_clips.json")
