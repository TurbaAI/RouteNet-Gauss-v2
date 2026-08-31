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

# Compare PyTorch experiment results with the frozen TensorFlow ground truth, cell by cell
# ("cell" = dataset x target x seed), and emit Markdown tables for PYTORCH_PARITY.md.
#
#   python compare_results.py --tf tensorflow_version_gt --torch results/torch_baseline \
#       --torch results/torch_baseline_torchinit --out pytorch_version_results/quick_comparison.md
#   python compare_results.py --tf tensorflow_version_gt/converged --torch results/torch_converged \
#       --converged --out pytorch_version_results/converged_comparison.md
#
# For every torch cell it reports test MAPE / MAE / R2 next to the TF cell, the deltas, and a
# verdict against the agreed gates:
#   exact-replay delay runs                                    MAPE within +-0.5 pt, R2 within
#                                                              max(+-0.02, 2x TF seed spread)
#   exact-replay jitter runs                                   statistical gate (chaotic objective:
#                                                              TF's own thread-count re-run misses
#                                                              the tight gate, PYTORCH_PARITY.md 3.1)
#   native runs                                               MAPE within +-3 pt, R2 within 2x the TF
#                                                             seed spread (|seed1 - seed2| of the TF cells)
#   converged (--converged)                                   MAPE within +-1 pt, R2 within +-0.03
# For exact-replay runs whose cell has a TF per-step recording (tensorflow_version_gt/replay/...
# step_losses_5x50.csv) it also compares the per-step training losses: max/median relative
# difference, first step where the difference exceeds 1e-3, and the epoch means (history.csv).

import argparse
import csv
import glob
import json
import os

import numpy as np

PERCENTILES = ["avg", "p50", "p90", "p95", "p99"]


def _metrics(y_true, y_pred):
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    mae_us = float(np.mean(np.abs(y_true - y_pred)) * 1e6)
    r2 = float(1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true))))
    return {"mape": mape, "mae_us": mae_us, "r2": r2}


def load_cells(root, clamp_gt=False):
    """{(dataset, target, seed): metrics dict} from every metrics.json under root/results or root.

    clamp_gt: recompute test_overall from predictions.npz with predictions clamped at 0. The TF
    experiment.py set `model.inference_mode = True` AFTER training, which does not affect the
    tf.function traces cached during training, so a fraction of the GT test predictions were
    never clamped (879 negative delay predictions in the converged run, changing its MAPE from
    5.037 to 5.075). The PyTorch evaluation clamps every prediction as intended, so the GT is
    compared on the same footing; the stored (unclamped) values are kept as test_overall_stored."""
    cells = {}
    for mf in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        m = json.load(open(mf))
        m["_dir"] = os.path.dirname(mf)
        if clamp_gt:
            pnpz = os.path.join(m["_dir"], "predictions.npz")
            if os.path.exists(pnpz):
                g = np.load(pnpz)
                m["test_overall_stored"] = m["test_overall"]
                m["n_negative_predictions"] = int((g["y_pred"] < 0).sum())
                m["test_overall"] = _metrics(g["y_true"].flatten(), np.maximum(g["y_pred"], 0).flatten())
        cells[(m["dataset"], m["target"], int(m["seed"]))] = m
    return cells


def read_history(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def step_losses(path):
    with open(path) as f:
        return np.array([float(r["loss"]) for r in csv.DictReader(f)])


def fmt(v, spec="{:.3f}"):
    return "" if v is None else spec.format(v)


def compare(tf_cells, torch_cells, converged, replay_root):
    rows, curves = [], []
    for key, t in sorted(torch_cells.items()):
        dataset, target, seed = key
        g = tf_cells.get(key)
        exact = t.get("sample_order", "native") != "native (ListDataset shuffle)" and t.get("init_weights")
        other_seed = tf_cells.get((dataset, target, 3 - seed))
        spread_r2 = abs(g["test_overall"]["r2"] - other_seed["test_overall"]["r2"]) if g and other_seed else None
        row = {"dataset": dataset, "target": target, "seed": seed, "mode": "exact-replay" if exact else "native",
               "init": t.get("init"), "device": t.get("device"), "epochs_run": t.get("epochs_run"),
               "torch_mape": t["test_overall"]["mape"], "torch_mae_us": t["test_overall"]["mae_us"], "torch_r2": t["test_overall"]["r2"],
               "torch_s_per_step": t.get("seconds_per_train_step_mean"), "torch_train_seconds": t.get("train_seconds")}
        if g:
            row.update({"tf_mape": g["test_overall"]["mape"], "tf_mae_us": g["test_overall"]["mae_us"], "tf_r2": g["test_overall"]["r2"],
                        "tf_mape_stored": g.get("test_overall_stored", g["test_overall"])["mape"],
                        "tf_r2_stored": g.get("test_overall_stored", g["test_overall"])["r2"],
                        "tf_n_negative_predictions": g.get("n_negative_predictions"),
                        "tf_train_seconds": g.get("train_seconds"), "tf_epochs_run": g.get("epochs_run", g.get("epochs")),
                        "d_mape": t["test_overall"]["mape"] - g["test_overall"]["mape"],
                        "d_r2": t["test_overall"]["r2"] - g["test_overall"]["r2"],
                        "tf_seed_spread_r2": spread_r2})
            if converged:
                ok = abs(row["d_mape"]) <= 1.0 and abs(row["d_r2"]) <= 0.03
                gate = "MAPE +-1pt, R2 +-0.03"
            elif exact and target == "delay":
                # R2 of the deliberately unconverged quick models sits far outside [0,1]
                # (trex delay: -32.6), where an absolute +-0.02 is tighter than TF's own
                # seed-to-seed variation (0.21 there); the bound scales with the seed spread
                # exactly like the native gate.
                r2_tol = max(0.02, 2 * spread_r2) if spread_r2 is not None else 0.02
                ok = abs(row["d_mape"]) <= 0.5 and abs(row["d_r2"]) <= r2_tol
                gate = f"MAPE +-0.5pt, R2 +-{r2_tol:.3f} (max(0.02, 2x TF seed spread))"
            elif exact:
                # jitter: the loss landscape is chaotic enough that TF's own re-run under a
                # different thread count ends >1 MAPE point from the GT after 250 steps (see
                # PYTORCH_PARITY.md 3.1) — the tight exact-replay gate is unpassable by TF
                # itself, so jitter exact replays are gated statistically like native runs.
                r2_tol = 2 * spread_r2 if spread_r2 is not None else 0.02
                ok = abs(row["d_mape"]) <= 3.0 and abs(row["d_r2"]) <= r2_tol
                gate = f"exact-replay jitter: MAPE +-3pt, R2 +-{r2_tol:.3f} (chaos envelope, see report)"
            else:
                r2_tol = 2 * spread_r2 if spread_r2 is not None else 0.02
                ok = abs(row["d_mape"]) <= 3.0 and abs(row["d_r2"]) <= r2_tol
                gate = f"MAPE +-3pt, R2 +-{r2_tol:.3f} (2x TF seed spread)"
            row.update({"gate": gate, "passed": bool(ok)})
        rows.append(row)

        # per-step curve comparison for exact replays with a TF recording
        rec = os.path.join(replay_root, dataset, "RouteNetGauss", target, f"seed_{seed}", "step_losses_5x50.csv")
        mine = os.path.join(t["_dir"], "step_losses.csv")
        if exact and os.path.exists(rec) and os.path.exists(mine):
            a, b = step_losses(rec), step_losses(mine)
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            rel = np.abs(a - b) / np.abs(a)
            first_bad = int(np.argmax(rel > 1e-3)) if np.any(rel > 1e-3) else None
            hist_tf = read_history(os.path.join(replay_root, dataset, "RouteNetGauss", target, f"seed_{seed}", "history.csv"))
            hist_t = read_history(os.path.join(t["_dir"], "history.csv"))
            ep = [(float(x["loss"]), float(y["loss"]), float(x["val_loss"]), float(y["val_loss"])) for x, y in zip(hist_tf, hist_t)]
            curves.append({"dataset": dataset, "target": target, "seed": seed, "n_steps": n,
                           "rel_diff_median": float(np.median(rel)), "rel_diff_max": float(rel.max()),
                           "rel_diff_first50_max": float(rel[:50].max()), "first_step_rel_gt_1e-3": first_bad,
                           "corr": float(np.corrcoef(a, b)[0, 1]),
                           "epochs": [{"epoch": i, "tf_loss": e[0], "torch_loss": e[1], "tf_val_loss": e[2], "torch_val_loss": e[3]} for i, e in enumerate(ep)]})
    return rows, curves


def to_markdown(rows, curves, title):
    out = [f"## {title}", ""]
    out.append("TF columns are recomputed from the GT `predictions.npz` with predictions clamped at 0 (as the TF code intended and the PyTorch evaluation does); "
               "`TF MAPE stored` is the value in the GT metrics.json, which includes unclamped negative predictions (`neg`).")
    out.append("")
    out.append("| dataset | target | seed | mode | init | device | TF MAPE (stored / neg) | TF MAPE | torch MAPE | Δ MAPE | TF R² | torch R² | Δ R² | TF MAE µs | torch MAE µs | gate | passed |")
    out.append("|---|---|--:|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|")
    for r in rows:
        out.append(f"| {r['dataset']} | {r['target']} | {r['seed']} | {r['mode']} | {r['init']} | {r['device']} | {fmt(r.get('tf_mape_stored'))} / {r.get('tf_n_negative_predictions', '')} | {fmt(r.get('tf_mape'))} | {fmt(r['torch_mape'])} | "
                   f"{fmt(r.get('d_mape'), '{:+.3f}')} | {fmt(r.get('tf_r2'))} | {fmt(r['torch_r2'])} | {fmt(r.get('d_r2'), '{:+.4f}')} | "
                   f"{fmt(r.get('tf_mae_us'), '{:.2f}')} | {fmt(r['torch_mae_us'], '{:.2f}')} | {r.get('gate', '')} | {r.get('passed', '')} |")
    out.append("")
    out.append("| dataset | target | seed | TF train s | torch train s | torch s/step | TF epochs | torch epochs |")
    out.append("|---|---|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        out.append(f"| {r['dataset']} | {r['target']} | {r['seed']} | {fmt(r.get('tf_train_seconds'), '{:.0f}')} | {fmt(r.get('torch_train_seconds'), '{:.0f}')} | "
                   f"{fmt(r.get('torch_s_per_step'), '{:.2f}')} | {r.get('tf_epochs_run', '')} | {r.get('epochs_run', '')} |")
    if curves:
        out += ["", "### Per-step training loss, exact replay vs TF recording", "",
                "| dataset | target | seed | steps | median rel diff | max rel diff | max rel diff (first 50) | first step > 1e-3 | corr |",
                "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
        for c in curves:
            out.append(f"| {c['dataset']} | {c['target']} | {c['seed']} | {c['n_steps']} | {c['rel_diff_median']:.2e} | {c['rel_diff_max']:.2e} | "
                       f"{c['rel_diff_first50_max']:.2e} | {c['first_step_rel_gt_1e-3']} | {c['corr']:.6f} |")
        out += ["", "| dataset | target | seed | epoch | TF loss | torch loss | TF val_loss | torch val_loss |", "|---|---|--:|--:|--:|--:|--:|--:|"]
        for c in curves:
            for e in c["epochs"]:
                out.append(f"| {c['dataset']} | {c['target']} | {c['seed']} | {e['epoch']} | {e['tf_loss']:.4f} | {e['torch_loss']:.4f} | {e['tf_val_loss']:.4f} | {e['torch_val_loss']:.4f} |")
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser(description="Compare PyTorch results with the TF ground truth")
    p.add_argument("--tf", default="tensorflow_version_gt", help="GT root (contains results/ or metrics.json files)")
    p.add_argument("--torch", action="append", required=True, help="results/<experiment> dir (repeatable)")
    p.add_argument("--converged", action="store_true", help="use the converged gates")
    p.add_argument("--replay-root", default="tensorflow_version_gt/replay")
    p.add_argument("--out", default=None, help="write Markdown here (also printed)")
    p.add_argument("--json", default=None, help="write the raw comparison rows here")
    args = p.parse_args()

    tf_cells = load_cells(os.path.join(args.tf, "results") if os.path.isdir(os.path.join(args.tf, "results")) else args.tf, clamp_gt=True)
    md, all_rows, all_curves = [], [], []
    for troot in args.torch:
        torch_cells = load_cells(troot)
        rows, curves = compare(tf_cells, torch_cells, args.converged, args.replay_root)
        for r in rows:
            r["experiment"] = os.path.basename(troot.rstrip("/"))
        all_rows += rows
        all_curves += curves
        md.append(to_markdown(rows, curves, f"{os.path.basename(troot.rstrip('/'))} vs TF ({'converged' if args.converged else 'quick 5x50'})"))
    text = "\n".join(md)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w").write(text)
    if args.json:
        json.dump({"rows": all_rows, "curves": all_curves}, open(args.json, "w"), indent=1)
    n_pass = sum(1 for r in all_rows if r.get("passed"))
    print(f"{n_pass}/{len(all_rows)} cells within their gates")


if __name__ == "__main__":
    main()
