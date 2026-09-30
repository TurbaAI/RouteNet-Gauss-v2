"""Progress of the 2026-09-28 verification runs (Q2b, Q7a) next to their references.

  python results/verification/status.py            # table
  python results/verification/status.py --events   # only what changed since the last --events call

Plateau exit = first epoch (0-based) whose val_loss < 75 (the delay plateau sits at ~86.7).
"""
import argparse
import csv
import json
import os
import subprocess
import time

ROOT = "/home/ubuntu/code/RouteNet-Gauss-v2"
CELL = "trex_multiburst/RouteNetGauss/delay/seed_{s}"
RUNS = [
    # (label, results dir template, experiment name for the process check or None, is_reference)
    ("TF ground truth (4 threads)", "tensorflow_version_gt/converged/results/" + CELL, None, 1),
    ("PyTorch exact replay", "pytorch_version_results/converged/runs/torch_converged/" + CELL, None, 1),
    ("PyTorch torch-init", "pytorch_version_results/converged/runs/torch_converged_torchinit/" + CELL, None, 1),
    ("TF setup check (4 threads, 1 ep)", "results/tf_runner/results/tf_setupcheck/" + CELL, "tf_setupcheck", 1),
    ("TF 1 thread", "results/tf_runner/results/tf_envelope_1thread/" + CELL, "tf_envelope_1thread", 1),
    ("PyTorch native (keras init)", "results/torch_converged_native/" + CELL, "torch_converged_native", 1),
    ("TF probe", "results/tf_runner/results/tf_plateau_probe/" + CELL, "tf_plateau_probe", 2),
    ("TF probe", "results/tf_runner/results/tf_plateau_probe/" + CELL, "tf_plateau_probe", 3),
    ("PyTorch probe (keras init)", "results/torch_plateau_probe/" + CELL, "torch_plateau_probe", 2),
    ("PyTorch probe (keras init)", "results/torch_plateau_probe/" + CELL, "torch_plateau_probe", 3),
    ("DIAG X1 keras init + TF order", "results/torch_diag_X1_tforder/" + CELL, "torch_diag_X1_tforder", 1),
    ("DIAG X2 TF init + torch order", "results/torch_diag_X2_torchorder/" + CELL, "torch_diag_X2_torchorder", 1),
]
EXIT_BELOW = 75.0
STALL_H = 3.0  # a running run whose results dir has not changed for this long is reported once


def running(exp, seed):
    if exp is None:
        return False
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return any(f"--experiment-name {exp}" in l and f"--seed {seed} " in l + " " for l in out.splitlines()
               if "experiment.py" in l)


def summarize(label, tmpl, exp, seed):
    d = os.path.join(ROOT, tmpl.format(s=seed))
    hist = os.path.join(d, "history.csv")
    rows = list(csv.DictReader(open(hist))) if os.path.exists(hist) else []
    vals = [float(r["val_loss"]) for r in rows]
    exit_ep = next((i for i, v in enumerate(vals) if v < EXIT_BELOW), None)
    b45 = min(vals[:45]) if vals else None
    m = json.load(open(os.path.join(d, "metrics.json"))) if os.path.exists(os.path.join(d, "metrics.json")) else None
    is_run = running(exp, seed)
    state = "done" if m else ("running" if is_run else ("reference" if exp is None else ("STOPPED" if rows else "not started")))
    files = [os.path.join(d, x) for x in os.listdir(d)] if os.path.isdir(d) else []
    return {
        "progress_age_h": (time.time() - max(os.path.getmtime(x) for x in files)) / 3600 if files else None,
        "key": f"{label} s{seed}", "label": label, "seed": seed, "state": state, "epochs": len(vals),
        "exit": exit_ep, "exit_val": vals[exit_ep] if exit_ep is not None else None,
        "best45": b45, "best": min(vals) if vals else None, "last": vals[-1] if vals else None,
        "test_mape": (m or {}).get("test_overall", {}).get("mape") if m else None,
        "test_r2": (m or {}).get("test_overall", {}).get("r2") if m else None,
        "epochs_run": (m or {}).get("epochs_run"),
    }


def fmt(x, f=".3f"):
    return "" if x is None else format(x, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", action="store_true")
    p.add_argument("--state", default=os.path.join(ROOT, "results/verification/.status_state.json"))
    a = p.parse_args()
    rows = [summarize(*r) for r in RUNS]
    if not a.events:
        print(f"{'run':34s} {'seed':>4s} {'state':>11s} {'epochs':>6s} {'plateau exit':>12s} "
              f"{'best@45':>8s} {'best':>8s} {'last':>8s} {'test MAPE':>9s} {'test R2':>8s}")
        for r in rows:
            print(f"{r['label']:34s} {r['seed']:>4d} {r['state']:>11s} {r['epochs']:>6d} {fmt(r['exit'], 'd'):>12s} "
                  f"{fmt(r['best45'], '.4f'):>8s} {fmt(r['best'], '.4f'):>8s} {fmt(r['last'], '.4f'):>8s} "
                  f"{fmt(r['test_mape']):>9s} {fmt(r['test_r2'], '.4f'):>8s}")
        return
    old = json.load(open(a.state)) if os.path.exists(a.state) else {}
    now = time.strftime("%m-%d %H:%M")
    for r in rows:
        if r["state"] == "reference":
            continue
        o = old.get(r["key"], {})
        tag = f"[{now}] {r['label']} seed {r['seed']}"
        if r["exit"] is not None and o.get("exit") is None:
            print(f"{tag}: left the plateau at epoch {r['exit']} (val {r['exit_val']:.3f})", flush=True)
        if r["state"] != o.get("state") and r["state"] in ("done", "STOPPED"):
            print(f"{tag}: {r['state']} after {r['epochs']} epochs, best val {fmt(r['best'], '.4f')}, "
                  f"test MAPE {fmt(r['test_mape'])} R2 {fmt(r['test_r2'], '.4f')}", flush=True)
        stalled = r["state"] == "running" and r["epochs"] > 0 and (r["progress_age_h"] or 0) > STALL_H
        if stalled and not o.get("stalled"):
            print(f"{tag}: STALLED — no new output for {r['progress_age_h']:.1f} h at epoch {r['epochs']}", flush=True)
        r["stalled"] = stalled
    with open(a.state, "w") as f:
        json.dump({r["key"]: r for r in rows}, f)


if __name__ == "__main__":
    main()
