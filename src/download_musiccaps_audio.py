"""Fetch the MusicCaps 10 s clips needed for Tasks 3 and 4.

MusicCaps ships captions and YouTube ids, not audio. This pulls each clip's
`[start_s, end_s]` window with yt-dlp and writes a mono wav at the project
sample rate.

Expect attrition: a meaningful share of the 5,521 ids are now private, deleted
or region-locked. Failures are logged to `results/metrics/musiccaps_download_log.json`
so the report can state exactly how many clips the experiments actually used.

    python src/download_musiccaps_audio.py --config config.yaml
    python src/download_musiccaps_audio.py --config config.yaml --limit 500 --workers 4

Requires `yt-dlp` and `ffmpeg` on PATH:  pip install yt-dlp
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from audio_datasets import load_config


def clip_path(out_dir: str, ytid: str) -> str:
    return os.path.join(out_dir, f"{ytid}.wav")


def download_clip(ytid: str, start_s: float, end_s: float, out_dir: str, sr: int, timeout: int = 120) -> tuple[str, bool, str]:
    """Download one clip window. Returns (ytid, ok, message)."""
    dest = clip_path(out_dir, ytid)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return ytid, True, "cached"

    url = f"https://www.youtube.com/watch?v={ytid}"
    # Let yt-dlp hand the stream to ffmpeg and cut only the requested window;
    # downloading full videos for a 10 s excerpt would be wasteful and slow.
    cmd = [
        "yt-dlp", "-q", "--no-warnings", "-f", "bestaudio",
        "--download-sections", f"*{start_s}-{end_s}",
        "--force-keyframes-at-cuts",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", f"ffmpeg:-ac 1 -ar {sr}",
        "-o", os.path.join(out_dir, f"{ytid}.%(ext)s"),
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ytid, False, "timeout"
    except FileNotFoundError:
        raise SystemExit("yt-dlp not found on PATH -- install it with `pip install yt-dlp`")

    if proc.returncode != 0 or not os.path.exists(dest):
        reason = (proc.stderr or proc.stdout or "unknown error").strip().splitlines()
        return ytid, False, reason[-1][:180] if reason else "unknown error"
    return ytid, True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MusicCaps 10 s clips.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="only fetch the first N clips")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    cfg = load_config(args.config)
    csv_path = os.path.join(cfg["paths"]["raw_dir"], cfg["data"]["csv_name"])
    if not os.path.exists(csv_path):
        raise SystemExit(f"{csv_path} not found -- run `python src/data_musiccaps.py` first")

    out_dir = cfg["task3"]["musiccaps_audio_dir"]
    sr = cfg["task3"]["audio"]["sr"]
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if args.limit:
        df = df.head(args.limit)
    print(f"[download] {len(df)} clips -> {out_dir} (workers={args.workers})")

    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_clip, row.ytid, row.start_s, row.end_s, out_dir, sr, args.timeout): row.ytid
            for row in df.itertuples(index=False)
        }
        for n, future in enumerate(as_completed(futures), start=1):
            ytid, ok, message = future.result()
            results.append({"ytid": ytid, "ok": ok, "message": message})
            if not ok:
                failures.append({"ytid": ytid, "reason": message})
            if n % 50 == 0:
                print(f"[download] {n}/{len(df)} attempted, {len(failures)} failed")

    ok_count = sum(r["ok"] for r in results)
    metrics_dir = cfg["paths"]["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)
    log_path = os.path.join(metrics_dir, "musiccaps_download_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {"attempted": len(results), "succeeded": ok_count, "failed": len(failures), "failures": failures},
            f,
            indent=2,
        )

    print(f"\n[download] {ok_count}/{len(results)} clips available ({len(failures)} failed)")
    print(f"[download] log -> {log_path}")


if __name__ == "__main__":
    main()
