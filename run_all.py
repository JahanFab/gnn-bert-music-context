"""Run the whole project, Task 1 -> Task 4, in dependency order.

Each stage is checked before it starts (are the packages installed? is the audio
there?) and skipped if its output already exists, so an interrupted run can be
restarted with the same command and picks up where it stopped.

    python run_all.py                      # everything that is not already done
    python run_all.py --only task1         # one task
    python run_all.py --from task3         # task 3 onwards
    python run_all.py --to task2           # tasks 1 and 2
    python run_all.py --skip-download      # reuse whatever clips are on disk
    python run_all.py --dry-run            # print the plan and exit
    python run_all.py --force              # re-run even completed stages

Run it from the project root.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
TASK_ORDER = ["task1", "task2", "task3", "task4"]


# --------------------------------------------------------------------------- #
# stage definition
# --------------------------------------------------------------------------- #
class Stage:
    def __init__(self, task, name, args, produces, needs_modules=(), needs_paths=(), optional=False):
        self.task = task
        self.name = name
        self.args = args
        self.produces = produces          # file/dir whose presence means "already done"
        self.needs_modules = needs_modules
        self.needs_paths = needs_paths    # (path, human-readable explanation)
        self.optional = optional          # failure warns instead of aborting

    def is_done(self) -> bool:
        path = os.path.join(ROOT, self.produces)
        if os.path.isdir(path):
            return bool(os.listdir(path))
        return os.path.exists(path)

    def missing_requirements(self) -> list[str]:
        problems = [
            f"missing package '{m}'  ->  pip install {PACKAGE_HINTS.get(m, m)}"
            for m in self.needs_modules
            if importlib.util.find_spec(m) is None
        ]
        problems += [
            f"missing data: {path}\n      {why}"
            for path, why in self.needs_paths
            if not os.path.exists(os.path.join(ROOT, path))
        ]
        return problems


PACKAGE_HINTS = {
    "librosa": "librosa soundfile",
    "torch_geometric": "torch-geometric",
    "yt_dlp": "yt-dlp",
}


def build_stages(cfg: dict, skip_download: bool) -> list[Stage]:
    t2, t3 = cfg["task2"], cfg["task3"]
    t2_name = t2["dataset"].lower()
    t2_tag = f"{t2_name}_{t2['graph']['type']}"
    t3_name = t3["dataset"].lower()

    t2_audio_hint = (
        (t2["gtzan_dir"], "unpack GTZAN so genres_original/<genre>/*.wav exists")
        if t2_name == "gtzan"
        else (t2["fma_dir"], "unpack FMA-small audio here")
    )

    stages = [
        Stage("task1", "build MusicCaps corpus", ["src/data_musiccaps.py"],
              "data/processed/musiccaps_task1.csv"),
        Stage("task1", "fine-tune BERT tag classifier", ["src/train.py"],
              "results/checkpoints/task1_bert_best.pt"),
        Stage("task1", "evaluate + plots + examples", ["src/evaluate.py"],
              "results/metrics/task1_test_metrics.json"),

        Stage("task2", "index audio dataset", ["src/audio_datasets.py"],
              f"data/processed/task2_index_{t2_name}.csv",
              needs_paths=[t2_audio_hint]),
        Stage("task2", "extract features + build graphs", ["src/graph_builder.py"],
              f"data/processed/graphs/task2_graphs_{t2_tag}.pt",
              needs_modules=["librosa", "torch_geometric"]),
        Stage("task2", "train GNN + CNN baseline", ["src/train_gnn.py", "--model", "both"],
              "results/metrics/task2_results.json",
              needs_modules=["torch_geometric"]),

        Stage("task3", "download MusicCaps clips", ["src/download_musiccaps_audio.py", "--workers", "4"],
              t3["musiccaps_audio_dir"], needs_modules=["yt_dlp"], optional=True),
        Stage("task3", "build paired fusion dataset", ["src/data_fusion.py"],
              f"data/processed/fusion/task3_fusion_{t3_name}.pt",
              needs_modules=["librosa", "torch_geometric"]),
        Stage("task3", "train fusion ablation", ["src/train_fusion.py", "--ablation", "all"],
              "results/metrics/task3_results.json",
              needs_modules=["torch_geometric"]),
        Stage("task3", "t-SNE + case studies", ["src/analyze_fusion.py"],
              "results/metrics/task3_analysis.json",
              needs_modules=["torch_geometric"]),

        Stage("task4", "train contrastive dual encoder", ["src/train_contrastive.py"],
              "results/metrics/task4_results.json",
              needs_modules=["torch_geometric"]),
    ]

    if skip_download:
        stages = [s for s in stages if s.name != "download MusicCaps clips"]
    return stages


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def preflight() -> None:
    """Report the environment once, up front, instead of per-stage surprises."""
    print("=" * 72)
    print("PREFLIGHT")
    print("=" * 72)

    try:
        import torch

        if torch.cuda.is_available():
            print(f"  torch {torch.__version__} | CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  torch {torch.__version__} | CUDA NOT AVAILABLE -- everything will run on CPU.")
            if "+cpu" in torch.__version__:
                print("    This is a CPU-only build. For GPU:")
                print("    pip install --force-reinstall torch --index-url "
                      "https://download.pytorch.org/whl/cu124")
    except ImportError:
        sys.exit("  torch is not installed -- nothing can run.")

    for module in ("transformers", "librosa", "torch_geometric", "yt_dlp"):
        found = importlib.util.find_spec(module) is not None
        hint = "" if found else f"   (pip install {PACKAGE_HINTS.get(module, module)})"
        print(f"  {'OK     ' if found else 'MISSING'} {module}{hint}")

    ffmpeg = shutil.which("ffmpeg")
    print(f"  {'OK     ' if ffmpeg else 'MISSING'} ffmpeg{'' if ffmpeg else '   (needed to cut MusicCaps clips)'}")
    print()


def run_stage(stage: Stage, index: int, total: int, force: bool, dry_run: bool) -> str:
    header = f"[{index}/{total}] {stage.task.upper()} -- {stage.name}"
    print("=" * 72)
    print(header)
    print("=" * 72)

    if stage.is_done() and not force:
        print(f"  SKIP: {stage.produces} already exists (use --force to redo)\n")
        return "skipped"

    problems = stage.missing_requirements()
    if problems:
        print("  BLOCKED:")
        for problem in problems:
            print(f"    - {problem}")
        print()
        return "blocked"

    command = [sys.executable] + stage.args + ["--config", "config.yaml"]
    print(f"  $ {' '.join(command)}\n")
    if dry_run:
        return "planned"

    start = time.time()
    result = subprocess.run(command, cwd=ROOT)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  FAILED after {elapsed / 60:.1f} min (exit {result.returncode})\n")
        return "failed"

    print(f"\n  done in {elapsed / 60:.1f} min\n")
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tasks 1-4 in order.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--only", choices=TASK_ORDER, help="run just this task")
    parser.add_argument("--from", dest="from_task", choices=TASK_ORDER, help="start here")
    parser.add_argument("--to", dest="to_task", choices=TASK_ORDER, help="stop after this")
    parser.add_argument("--skip-download", action="store_true", help="reuse clips already on disk")
    parser.add_argument("--force", action="store_true", help="re-run completed stages")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()

    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    stages = build_stages(cfg, args.skip_download)
    if args.only:
        stages = [s for s in stages if s.task == args.only]
    else:
        lo = TASK_ORDER.index(args.from_task) if args.from_task else 0
        hi = TASK_ORDER.index(args.to_task) if args.to_task else len(TASK_ORDER) - 1
        stages = [s for s in stages if lo <= TASK_ORDER.index(s.task) <= hi]

    if not args.dry_run:
        preflight()

    outcomes = []
    for i, stage in enumerate(stages, start=1):
        outcome = run_stage(stage, i, len(stages), args.force, args.dry_run)
        outcomes.append((stage, outcome))

        if outcome == "failed" and not stage.optional:
            print("Stopping: later stages depend on this one.")
            break
        if outcome == "blocked" and not stage.optional:
            print("Stopping: install the missing requirement above, then re-run this command.")
            break
        if outcome in ("failed", "blocked") and stage.optional:
            print("  (optional stage -- continuing with whatever data is already present)\n")

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for stage, outcome in outcomes:
        print(f"  {outcome:<8} {stage.task}  {stage.name}")

    remaining = len(stages) - len(outcomes)
    if remaining:
        print(f"  {remaining} stage(s) not reached")
    print("\n  results/metrics/  results/plots/  results/retrieval_examples/")


if __name__ == "__main__":
    main()
