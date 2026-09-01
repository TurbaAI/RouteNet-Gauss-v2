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

# Figures for PYTORCH_PARITY.md. Each answers one question:
#
#   fig1_converged_curves.png   Do the converged runs follow the same learning trajectory?
#   fig2_converged_metrics.png  Do they reach the same test accuracy, per percentile?
#   fig3_quick_parity.png       Does every quick-set cell land inside its gate?
#   fig4_step_agreement.png     Is the exact replay's step-by-step deviation inside the
#                               envelope TensorFlow shows against *itself* (thread count)?
#
# Run in conda env RG_torch from the repo root:
#   python parity/make_figures.py            # writes pytorch_version_results/figures/*.png
#
# Palette: the validated 3-hue categorical set (blue/orange/aqua) on the light chart surface;
# every series carries a legend AND a direct label (the aqua slot's contrast warning is
# discharged by visible labels plus the tables in PYTORCH_PARITY.md).

import csv
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.environ.get("RG_LOG_DIR", "")          # where converged_*.log live (session scratchpad)
OUT = os.path.join("pytorch_version_results", "figures")
PERC = ["avg", "p50", "p90", "p95", "p99"]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d8d4"
TF, TORCH, TORCH_INIT = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1,2,3

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.9,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.fontsize": 8.5,
    "lines.linewidth": 2, "lines.markersize": 7, "figure.dpi": 200,
})


def style(ax, title=None, xlabel=None, ylabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", color=INK, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def tf_history(target):
    p = f"tensorflow_version_gt/converged/results/trex_multiburst/RouteNetGauss/{target}/seed_1/history.csv"
    rows = list(csv.DictReader(open(p)))
    return [int(r["epoch"]) for r in rows], [float(r["val_loss"]) for r in rows]


def torch_history(log):
    """(epochs, val_losses) from a running/finished converged log."""
    p = os.path.join(LOGS, log)
    if not os.path.exists(p):
        return [], []
    rows = re.findall(r"Epoch (\d+)/300 - \d+s - loss: [\d.]+ - val_loss: ([\d.]+)", open(p, errors="ignore").read())
    return [int(e) for e, _ in rows], [float(v) for _, v in rows]


def annotate_best(ax, xs, ys, color, label):
    if not xs:
        return
    i = int(np.argmin(ys))
    ax.plot([xs[i]], [ys[i]], marker="o", color=color, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
    ax.annotate(f"{label}  {ys[i]:.2f}", (xs[i], ys[i]), textcoords="offset points", xytext=(6, -11),
                color=color, fontsize=8, fontweight="bold")


# ---------------------------------------------------------------- fig 1: learning curves
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    series = {
        "delay": [("TensorFlow (ground truth)", TF, *tf_history("delay")),
                  ("PyTorch, exact replay", TORCH, *torch_history("converged_A_delay.log")),
                  ("PyTorch, torch init", TORCH_INIT, *torch_history("converged_B_delay.log"))],
        "jitter": [("TensorFlow (ground truth)", TF, *tf_history("jitter")),
                   ("PyTorch, exact replay", TORCH, *torch_history("converged_A_jitter.log"))],
    }
    for ax, target in zip(axes, ["delay", "jitter"]):
        for label, color, xs, ys in series[target]:
            if not xs:
                continue
            ax.plot(xs, ys, color=color, label=label, solid_capstyle="round")
            annotate_best(ax, xs, ys, color, "best")
        ax.set_yscale("log")
        ax.set_yticks([6, 10, 20, 40, 100]); ax.set_yticklabels(["6", "10", "20", "40", "100"])
        style(ax, f"{target} — validation loss per epoch", "epoch", "val_loss (MAPE %, log scale)")
        ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Converged runs: PyTorch follows the TensorFlow learning trajectory",
                 x=0.007, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.007, 0.005, "trex_multiburst, seed 1. Exact replay = TF's initial weights, scenario order and z-scores. "
                           "Runs still training are drawn to their current epoch.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig1_converged_curves.png"))
    plt.close(fig)


# ------------------------------------------------------- fig 2: test metrics per percentile
def fig2():
    def load(path):
        return json.load(open(path)) if os.path.exists(path) else None

    H = "pytorch_version_results/converged/harvest"
    data = {}
    for target in ["delay", "jitter"]:
        gt = load(f"tensorflow_version_gt/converged/results/trex_multiburst/RouteNetGauss/{target}/seed_1/metrics.json")
        gtp = np.load(f"tensorflow_version_gt/converged/results/trex_multiburst/RouteNetGauss/{target}/seed_1/predictions.npz")
        # GT metrics recomputed with predictions clamped at 0 (see PYTORCH_PARITY.md §1)
        yt, yp = gtp["y_true"], np.maximum(gtp["y_pred"], 0)
        gt_clamped = {n: {"mape": float(np.mean(np.abs((yt[:, i] - yp[:, i]) / yt[:, i])) * 100),
                          "r2": float(1 - np.sum((yt[:, i] - yp[:, i]) ** 2) / np.sum((yt[:, i] - yt[:, i].mean()) ** 2))}
                      for i, n in enumerate(PERC)}
        data[target] = {"TensorFlow (ground truth)": (TF, gt_clamped),
                        "PyTorch, exact replay": (TORCH, (load(f"{H}/torch_converged/{target}/metrics.json") or {}).get("test_per_percentile"))}
        ti = load(f"{H}/torch_converged_torchinit/{target}/metrics.json")
        if ti:
            data[target]["PyTorch, torch init"] = (TORCH_INIT, ti["test_per_percentile"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, target in enumerate(["delay", "jitter"]):
        for row, metric in enumerate(["mape", "r2"]):
            ax = axes[row][col]
            entries = [(lbl, c, d) for lbl, (c, d) in data[target].items() if d]
            w = 0.8 / max(len(entries), 1)
            for k, (lbl, color, d) in enumerate(entries):
                vals = [d[p][metric] for p in PERC]
                xs = np.arange(len(PERC)) + (k - (len(entries) - 1) / 2) * w
                bars = ax.bar(xs, vals, width=w * 0.92, color=color, label=lbl, zorder=3,
                              edgecolor=SURFACE, linewidth=2)  # 2px surface gap between bars
                for x, v in zip(xs, vals):
                    ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 3 if v >= 0 else -11),
                                ha="center", color=INK2, fontsize=7)
            ax.set_xticks(range(len(PERC))); ax.set_xticklabels(PERC)
            name = "test MAPE (%, lower is better)" if metric == "mape" else "test R² (higher is better)"
            style(ax, f"{target} — {name}", "percentile", None)

    # one figure-level legend under the panels — an in-axes legend collided with the bar labels
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.045), fontsize=8.5)
    fig.suptitle("Converged test accuracy per percentile: PyTorch vs the TensorFlow ground truth",
                 x=0.007, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.007, 0.005, "trex_multiburst test split, seed 1. GT metrics recomputed from its predictions clamped at 0 "
                           "(PYTORCH_PARITY.md §1). PyTorch bars are harvested from each run's best checkpoint.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.085, 1, 0.95))
    fig.savefig(os.path.join(OUT, "fig2_converged_metrics.png"))
    plt.close(fig)


# --------------------------------------------------------------- fig 3: quick-set parity
def fig3():
    sets = [("Exact replay (TF init weights + TF scenario order)", "torch_baseline", TORCH),
            ("Native PyTorch pipeline (torch init + torch shuffle)", "torch_baseline_torchinit", TORCH_INIT)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True)
    for ax, (title, exp, color) in zip(axes, sets):
        p = f"pytorch_version_results/quick/{exp}_vs_gt.json"
        if not os.path.exists(p):
            continue
        rows = json.load(open(p))["rows"]
        rows = sorted(rows, key=lambda r: (r["target"], r["dataset"], r["seed"]))
        labels = [f"{r['dataset'].replace('_pcaps','').replace('trex_','')} {r['target'][:3]} s{r['seed']}" for r in rows]
        d = [r["d_mape"] for r in rows]
        ys = np.arange(len(rows))
        ax.axvspan(-0.5, 0.5, color=TF, alpha=0.10, zorder=0)
        ax.axvspan(-3, 3, color=TF, alpha=0.05, zorder=0)
        ax.axvline(0, color=GRID, linewidth=1.2, zorder=1)
        ax.hlines(ys, 0, d, color=color, linewidth=2, zorder=3)
        ax.plot(d, ys, "o", color=color, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        for y, v in zip(ys, d):
            ax.annotate(f"{v:+.2f}", (v, y), textcoords="offset points",
                        xytext=(8 if v >= 0 else -8, -3), ha="left" if v >= 0 else "right",
                        color=INK2, fontsize=7.5)
        ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(-6.5, 6.5); ax.invert_yaxis()
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, axis="x", zorder=0); ax.set_axisbelow(True)
        ax.set_title(title, loc="left", color=INK, pad=8, fontsize=9.5)
        ax.set_xlabel("Δ test MAPE, PyTorch − TensorFlow (percentage points)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=TF, alpha=0.10, label="±0.5 pt — exact-replay delay gate"),
                        Patch(facecolor=TF, alpha=0.05, label="±3 pt — statistical gate (jitter, native)")],
               frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.055), fontsize=8)
    fig.suptitle("Quick 5×50 comparison sets: every cell against its gate",
                 x=0.007, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.007, 0.005, "8 cells per set (dataset × target × seed). Exact replay: 8/8 inside gate. "
                           "Native: 7/8 (trex jitter seed 2, the most chaotic cell).", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig3_quick_parity.png"))
    plt.close(fig)


# ------------------------------------------------- fig 4: per-step agreement vs chaos envelope
def fig4():
    def steps(p):
        return np.array([float(r["loss"]) for r in csv.DictReader(open(p))]) if os.path.exists(p) else None

    F = "pytorch_version_results/parity/exact_replay_50steps"
    panels = []
    for target in ["delay", "jitter"]:
        gt = steps(f"tensorflow_version_gt/replay/trex_multiburst/RouteNetGauss/{target}/seed_1/step_losses_5x50.csv")
        suffix = "_5ep" if target == "delay" else ""
        tf1 = steps(f"{F}/tf_1thread_step_losses_trex_multiburst_{target}_seed1{suffix}.csv")
        tt = steps(os.path.join(LOGS, "").replace("logs", "") and f"results/torch_converged/trex_multiburst/RouteNetGauss/{target}/seed_1/step_losses.csv")
        if gt is None or tt is None:
            continue
        panels.append((target, gt, tf1, tt))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.1), squeeze=False)
    for ax, (target, gt, tf1, tt) in zip(axes[0], panels):
        n = min(len(gt), len(tt), 250)
        rel_t = np.abs(tt[:n] - gt[:n]) / gt[:n]
        ax.plot(np.arange(n), rel_t, color=TORCH, label="PyTorch vs TF (exact replay)", alpha=0.95)
        if tf1 is not None:
            m = min(len(gt), len(tf1), 250)
            ax.plot(np.arange(m), np.abs(tf1[:m] - gt[:m]) / gt[:m], color=TF,
                    label="TF vs TF, 1 thread (chaos envelope)", alpha=0.8)
        ax.axhline(1e-3, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate("1e-3", (2, 1.15e-3), color=INK2, fontsize=7.5)
        ax.set_yscale("log")
        style(ax, f"{target} — per-step training-loss deviation from the GT", "training step", "|relative difference|")
        ax.legend(frameon=False, loc="lower right")
    fig.suptitle("Exact replay tracks TensorFlow as closely as TensorFlow tracks itself",
                 x=0.007, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.007, 0.005, "trex_multiburst seed 1, first 250 steps (the quick 5×50 config). The blue curve is TF re-run with "
                           "TF_NUM_INTRAOP_THREADS=1 — a pure float32 summation-order change, no framework change.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig4_step_agreement.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn in (fig1, fig2, fig3, fig4):
        try:
            fn()
            print(f"  {fn.__name__}: ok")
        except Exception as e:
            print(f"  {fn.__name__}: FAILED {type(e).__name__}: {e}")
    print("figures ->", OUT)
    for f in sorted(os.listdir(OUT)):
        print(f"   {f}  {os.path.getsize(os.path.join(OUT, f))/1000:.0f} kB")
