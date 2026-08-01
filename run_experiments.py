"""
Copyright 2025 Universitat Politècnica de Catalunya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Driver for the TensorFlow training-job matrix that produces the baseline the later
# PyTorch port is compared against. Runs 2 datasets x 2 targets x 2 seeds = 8 jobs,
# each as its own `experiment.py` subprocess (clean per-job TF/GPU state + seeding).
#
# Backend is auto-detected: if a GPU is usable, jobs share GPU:0 with memory-growth and
# run with higher concurrency; otherwise they fall back to CPU with concurrency 2 (the
# box has 4 cores). Aggregates every job's metrics.json into:
#   results/<experiment_name>/summary.csv
#   results/<experiment_name>/summary.json
#
# Usage:  python run_experiments.py [--experiment-name tf_baseline] [--epochs 5]
#                                   [--steps 50] [--use-wandb] [--force-cpu]

import argparse
import csv
import glob
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from gpu_setup import configure_gpu_env, cuda_bin_path, cuda_ld_library_path

configure_gpu_env()

DATASETS = ["mawi_pcaps", "trex_multiburst"]
TARGETS = ["delay", "jitter"]
SEEDS = [1, 2]


def detect_gpu(force_cpu):
    if force_cpu:
        return False
    try:
        import tensorflow as tf

        return len(tf.config.list_physical_devices("GPU")) > 0
    except Exception as e:
        print(f"[driver] GPU probe failed ({e}); using CPU.")
        return False


def run_job(job, args, use_gpu):
    dataset, target, seed = job
    log_dir = os.path.join(
        "results", args.experiment_name, dataset, "RouteNetGauss", target, f"seed_{seed}"
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    # Inject the CUDA loader path directly into the child env (deterministic — the child
    # then starts with the right LD_LIBRARY_PATH and never needs to re-exec itself).
    env = dict(os.environ)
    env["_RG_GPU_ENV_READY"] = "1"  # child skips its own re-exec; path is already correct
    if use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = "0"
        # Force incremental VRAM allocation (env var applies regardless of init timing, unlike
        # a late set_memory_growth call) so concurrent jobs share the GPU instead of the first
        # one grabbing all 16 GB and starving the rest.
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        libs = cuda_ld_library_path()
        if libs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                [libs] + [p for p in [env.get("LD_LIBRARY_PATH", "")] if p]
            )
        bins = cuda_bin_path()
        if bins:
            env["PATH"] = os.pathsep.join([bins] + [p for p in [env.get("PATH", "")] if p])
    else:
        env["CUDA_VISIBLE_DEVICES"] = "-1"

    cmd = [
        sys.executable, "experiment.py",
        "--dataset", dataset,
        "--target", target,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--steps", str(args.steps),
        "--shuffle-buffer", str(args.shuffle_buffer),
        "--patience", str(args.patience),
        "--experiment-name", args.experiment_name,
    ]
    if args.use_wandb:
        cmd.append("--use-wandb")
    if args.save_best_only:
        cmd.append("--save-best-only")

    label = f"{dataset}/{target}/seed_{seed}"
    print(f"[driver] START {label}")
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    ok = proc.returncode == 0
    print(f"[driver] {'OK   ' if ok else 'FAIL '} {label} (rc={proc.returncode}, log={log_path})")
    return {"job": label, "returncode": proc.returncode, "ok": ok, "log": log_path}


def aggregate(args):
    """Collect every job's metrics.json into summary.csv + summary.json."""
    root = os.path.join("results", args.experiment_name)
    metric_files = sorted(glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True))
    rows = [json.load(open(mf)) for mf in metric_files]

    summary_json = os.path.join(root, "summary.json")
    with open(summary_json, "w") as f:
        json.dump(rows, f, indent=2)

    summary_csv = os.path.join(root, "summary.csv")
    fields = [
        "dataset", "target", "seed", "backend", "train_seconds",
        "final_train_loss", "final_val_loss", "n_test_predictions",
        "mape_overall", "mae_us_overall", "r2_overall",
    ] + [f"mape_{p}" for p in ["avg", "p50", "p90", "p95", "p99"]]
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {
                "dataset": r["dataset"], "target": r["target"], "seed": r["seed"],
                "backend": r["backend"], "train_seconds": r["train_seconds"],
                "final_train_loss": r["final_train_loss"], "final_val_loss": r["final_val_loss"],
                "n_test_predictions": r["n_test_predictions"],
                "mape_overall": r["test_overall"]["mape"],
                "mae_us_overall": r["test_overall"]["mae_us"],
                "r2_overall": r["test_overall"]["r2"],
            }
            for p in ["avg", "p50", "p90", "p95", "p99"]:
                row[f"mape_{p}"] = r["test_per_percentile"][p]["mape"]
            w.writerow(row)
    return summary_csv, summary_json, len(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Run the RouteNet-Gauss TF job matrix.")
    p.add_argument("--experiment-name", default="tf_baseline")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--shuffle-buffer", type=int, default=200,
                   help="smaller than train.py's 1000 to cut per-job startup stall")
    p.add_argument("--patience", type=int, default=0,
                   help="early-stopping patience on val_loss (0 = fixed epochs). Passed to each job.")
    p.add_argument("--save-best-only", action="store_true",
                   help="only keep the best-val_loss checkpoint per job (good for long runs)")
    p.add_argument("--datasets", default=None,
                   help="comma-separated subset override, e.g. 'trex_multiburst'")
    p.add_argument("--targets", default=None, help="comma-separated subset override, e.g. 'delay,jitter'")
    p.add_argument("--seeds", default=None, help="comma-separated subset override, e.g. '1'")
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--gpu-concurrency", type=int, default=2)
    p.add_argument("--cpu-concurrency", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    use_gpu = detect_gpu(args.force_cpu)
    concurrency = args.gpu_concurrency if use_gpu else args.cpu_concurrency

    # Optional subset overrides (comma-separated) — default to the full 2x2x2 matrix.
    datasets = args.datasets.split(",") if args.datasets else DATASETS
    targets = args.targets.split(",") if args.targets else TARGETS
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    jobs = list(itertools.product(datasets, targets, seeds))

    print(f"[driver] backend={'GPU' if use_gpu else 'CPU'} | {len(jobs)} jobs | "
          f"concurrency={concurrency} | epochs={args.epochs} steps={args.steps} "
          f"patience={args.patience}")

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run_job, j, args, use_gpu): j for j in jobs}
        for fut in as_completed(futures):
            results.append(fut.result())

    failed = [r for r in results if not r["ok"]]
    summary_csv, summary_json, n = aggregate(args)

    print("\n[driver] ===== SUMMARY =====")
    print(f"[driver] {n}/{len(jobs)} jobs produced metrics -> {summary_csv}")
    if failed:
        print(f"[driver] {len(failed)} job(s) FAILED (see their train.log):")
        for r in failed:
            print(f"          - {r['job']}  (rc={r['returncode']}, {r['log']})")
        sys.exit(1)
    print("[driver] all jobs completed successfully.")


if __name__ == "__main__":
    main()
