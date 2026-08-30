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

# Run parity/l0_forward.py for every TF checkpoint in the repo (2 converged GT, 8 quick-baseline
# GT, 6 paper weights) on its test set, each in its own process, and summarise the JSON reports
# into pytorch_version_results/parity/l0_summary.{json,md}.
#
#   python parity/run_l0_all.py --concurrency 2 --threads 1

import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join("pytorch_version_results", "parity")


def latest_ckpt(ckpt_dir):
    with open(os.path.join(ckpt_dir, "checkpoint")) as f:
        for line in f:
            if line.startswith("model_checkpoint_path:"):
                return os.path.join(ckpt_dir, line.split('"')[1])
    raise FileNotFoundError(ckpt_dir)


def jobs():
    out = []
    gt = "tensorflow_version_gt"
    # converged GT (trex_multiburst seed 1, delay + jitter)
    for target in ["delay", "jitter"]:
        cell = f"trex_multiburst/RouteNetGauss/{target}/seed_1"
        out.append(dict(name=f"converged_trex_multiburst_{target}_seed1", dataset="trex_multiburst", target=target,
                        checkpoint=latest_ckpt(f"{gt}/converged/ckpt/{cell}"),
                        z_scores=f"{gt}/converged/normalization/{cell}/z_scores.pkl",
                        gt_predictions=f"{gt}/converged/results/{cell}/predictions.npz"))
    # quick 5x50 baseline GT (8 cells, final epoch-5 checkpoint)
    for dataset in ["mawi_pcaps", "trex_multiburst"]:
        for target in ["delay", "jitter"]:
            for seed in [1, 2]:
                cell = f"{dataset}/RouteNetGauss/{target}/seed_{seed}"
                out.append(dict(name=f"quick_{dataset}_{target}_seed{seed}", dataset=dataset, target=target,
                                checkpoint=latest_ckpt(f"{gt}/ckpt/{cell}"),
                                z_scores=f"{gt}/normalization/{cell}/z_scores.pkl",
                                gt_predictions=f"{gt}/results/{cell}/predictions.npz"))
    # paper weights (evaluation.ipynb: model dir -> test set it is evaluated on)
    paper = [("mawi_pcaps", "delay", "mawi_pcaps"), ("mawi_pcaps", "jitter", "mawi_pcaps"),
             ("trex_multiburst_filtered", "delay", "trex_multiburst"), ("trex_multiburst", "jitter", "trex_multiburst"),
             ("trex_synthetic_filtered", "delay", "trex_synthetic"), ("trex_synthetic", "jitter", "trex_synthetic")]
    for model_ds, target, test_ds in paper:
        d = f"paper_weights/{model_ds}/RouteNetGauss/{target}"
        out.append(dict(name=f"paper_{model_ds}_{target}", dataset=test_ds, target=target,
                        checkpoint=latest_ckpt(f"ckpt/{d}"), z_scores=f"normalization/{d}/z_scores.pkl", gt_predictions=None))
    return out


def run(job, args):
    out_json = os.path.join(OUT_DIR, f"l0_{job['name']}.json")
    log = os.path.join(OUT_DIR, f"l0_{job['name']}.log")
    cmd = [sys.executable, "parity/l0_forward.py", "--checkpoint", job["checkpoint"], "--z-scores", job["z_scores"],
           "--dataset", job["dataset"], "--partition", "test", "--target", job["target"], "--inference-mode",
           "--device", args.device, "--threads", str(args.threads), "--out", out_json]
    if job["gt_predictions"]:
        cmd += ["--gt-predictions", job["gt_predictions"]]
    if args.n:
        cmd += ["--n", str(args.n)]
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"[l0] {'OK  ' if rc == 0 else 'FAIL'} {job['name']} (rc={rc})", flush=True)
    return job["name"], rc, out_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n", type=int, default=None, help="limit scenarios per checkpoint (debug)")
    p.add_argument("--only", default=None, help="comma-separated substring filter on job names")
    args = p.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    js = jobs()
    if args.only:
        js = [j for j in js if any(s in j["name"] for s in args.only.split(","))]
    print(f"[l0] {len(js)} checkpoints | concurrency={args.concurrency} | device={args.device}", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for fut in as_completed([pool.submit(run, j, args) for j in js]):
            results.append(fut.result())

    rows = []
    for name, rc, out_json in sorted(results):
        r = json.load(open(out_json)) if os.path.exists(out_json) else {}
        g = r.get("gt_predictions", {})
        rows.append({
            "checkpoint": name, "device": r.get("device"), "n_scenarios": r.get("n_scenarios"), "n_predictions": r.get("n_predictions"),
            "targets_identical": r.get("all_targets_identical"), "max_abs_diff": r.get("max_abs_diff"),
            "worst_scale_rel_diff": r.get("worst_scale_rel_diff"), "mape_tf": r.get("mape_tf"), "mape_torch": r.get("mape_torch"),
            "tf_vs_gt_predictions_bit_identical": g.get("tf_vs_gt_bit_identical"), "torch_vs_gt_max_abs_diff": g.get("torch_vs_gt_max_abs_diff"),
            "tf_s_per_scenario": r.get("tf_seconds_per_scenario"), "torch_s_per_scenario": r.get("torch_seconds_per_scenario"),
            "passed": r.get("passed"), "rc": rc,
        })
    json.dump(rows, open(os.path.join(OUT_DIR, "l0_summary.json"), "w"), indent=1)
    with open(os.path.join(OUT_DIR, "l0_summary.md"), "w") as f:
        f.write("| checkpoint | scenarios | predictions | targets identical | max abs diff | worst scale-rel diff | MAPE TF | MAPE torch | TF == GT preds | torch vs GT max abs | passed |\n")
        f.write("|---|--:|--:|---|--:|--:|--:|--:|---|--:|---|\n")
        for r in rows:
            fmt = lambda v, s: ("" if v is None else s.format(v))
            f.write(f"| {r['checkpoint']} | {r['n_scenarios']} | {r['n_predictions']} | {r['targets_identical']} | {fmt(r['max_abs_diff'], '{:.2e}')} | "
                    f"{fmt(r['worst_scale_rel_diff'], '{:.2e}')} | {fmt(r['mape_tf'], '{:.4f}')} | {fmt(r['mape_torch'], '{:.4f}')} | "
                    f"{r['tf_vs_gt_predictions_bit_identical']} | {fmt(r['torch_vs_gt_max_abs_diff'], '{:.2e}')} | {r['passed']} |\n")
    n_pass = sum(1 for r in rows if r["passed"])
    print(f"[l0] {n_pass}/{len(rows)} checkpoints passed -> {OUT_DIR}/l0_summary.md", flush=True)
    sys.exit(0 if n_pass == len(rows) else 1)


if __name__ == "__main__":
    main()
