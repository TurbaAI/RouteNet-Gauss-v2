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

# Reliability verification report (PYTORCH_PARITY.md §7): applies the rules that were fixed
# before the runs to whatever results exist, and says which checks are still pending.
#
#   §7.2 TF against itself at converged scale — the TF converged delay GT re-run with one intra-op
#        thread (plus the one-epoch setup check that must reproduce the GT's epoch-0 history row);
#   §7.3 the new default pipeline (init=keras drawn by PyTorch, PyTorch shuffle) trained to convergence;
#   §7.4 plateau test across seeds (first 25 epochs, seeds 2 and 3, TF and PyTorch).
#
# TF test metrics are recomputed from predictions.npz with predictions clamped at 0, exactly as
# compare_results.py does for the ground truth (PYTORCH_PARITY.md §1). Epochs are 0-based.
#
#   python parity/reliability_report.py                 # frozen copies in pytorch_version_results/verification
#   python parity/reliability_report.py --live          # the running/finished runs under results/
#   ... --out pytorch_version_results/verification/reliability_report   (writes .md and .json)

import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from compare_results import load_cells  # noqa: E402  (clamped GT metrics, same code as §1-§5)

CELL = os.path.join("trex_multiburst", "RouteNetGauss", "delay", "seed_{seed}")
GT = "tensorflow_version_gt/converged/results"
REFERENCE_RUNS = {  # committed §5 runs, for context rows
    "torch_converged": "pytorch_version_results/converged/runs/torch_converged",
    "torch_converged_torchinit": "pytorch_version_results/converged/runs/torch_converged_torchinit",
}
LIVE = {
    "tf_setupcheck": "results/tf_runner/results/tf_setupcheck",
    "tf_envelope_1thread": "results/tf_runner/results/tf_envelope_1thread",
    "tf_plateau_probe": "results/tf_runner/results/tf_plateau_probe",
    "torch_converged_native": "results/torch_converged_native",
    "torch_plateau_probe": "results/torch_plateau_probe",
}
FROZEN = "pytorch_version_results/verification"
INIT_DRAWS_LIVE = "results/verification/init_draws"
EXIT_BELOW = 75.0       # plateau exit = first epoch with val_loss below this (plateau ~86.7)
BUDGET = 45             # the GT's epoch count, for "best val within TF's budget"
PROBE_EPOCHS = 25
TF_KINDS = ("tf_setupcheck", "tf_envelope_1thread", "tf_plateau_probe")


def run_dir(name, live):
    return LIVE[name] if live else os.path.join(FROZEN, name)


def history(d):
    p = os.path.join(d, "history.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def first_row_text(d):
    p = os.path.join(d, "history.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        lines = f.read().splitlines()
    return (lines[0], lines[1]) if len(lines) > 1 else None


def run(root, seed, clamp):
    """Summary of one run: history-derived quantities + (clamped for TF) test metrics if finished."""
    d = os.path.join(root, CELL.format(seed=seed))
    h = history(d)
    vals = [float(r["val_loss"]) for r in h]
    cells = load_cells(d, clamp_gt=clamp) if os.path.exists(os.path.join(d, "metrics.json")) else {}
    m = next(iter(cells.values()), None)
    return {
        "dir": d,
        "epochs": len(vals),
        "finished": m is not None,
        "exit": next((i for i, v in enumerate(vals) if v < EXIT_BELOW), None),
        "best_within_budget": min(vals[:BUDGET]) if vals else None,
        "best": min(vals) if vals else None,
        "epochs_run": m.get("epochs_run") if m else None,
        "test_mape": m["test_overall"]["mape"] if m else None,
        "test_r2": m["test_overall"]["r2"] if m else None,
        "test_mape_stored": (m.get("test_overall_stored") or {}).get("mape") if m else None,
        "n_negative_predictions": m.get("n_negative_predictions") if m else None,
        "init": m.get("init") if m else None,
    }


def mann_whitney_greater(x, y):
    """Exact one-sided Mann–Whitney U test that x tends to be larger than y: the permutation
    p-value of the rank sum of x over every split of the pooled values (mid-ranks for ties)."""
    pooled = list(x) + list(y)
    order = sorted(range(len(pooled)), key=lambda i: pooled[i])
    ranks = [0.0] * len(pooled)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pooled[order[j + 1]] == pooled[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    observed = sum(ranks[: len(x)])
    splits = list(itertools.combinations(range(len(pooled)), len(x)))
    return sum(1 for c in splits if sum(ranks[k] for k in c) >= observed - 1e-9) / len(splits)


def stopped_on_plateau(r, horizon):
    """Early-stopped before `horizon` epochs while still on the plateau."""
    return r["finished"] and r["epochs"] < horizon and r["exit"] is None


def f(x, spec=".3f"):
    return "—" if x is None else format(x, spec)


def main():
    p = argparse.ArgumentParser(description="PYTORCH_PARITY.md §7 reliability report")
    p.add_argument("--live", action="store_true", help="read the runs under results/ instead of the frozen copies")
    p.add_argument("--out", default=None, help="write <out>.md and <out>.json")
    a = p.parse_args()
    os.chdir(_REPO_ROOT)

    gt = run(GT, 1, clamp=True)
    replay = run(REFERENCE_RUNS["torch_converged"], 1, clamp=False)
    tinit = run(REFERENCE_RUNS["torch_converged_torchinit"], 1, clamp=False)
    tf1 = run(run_dir("tf_envelope_1thread", a.live), 1, clamp=True)
    native = run(run_dir("torch_converged_native", a.live), 1, clamp=False)
    probes = {(k, s): run(run_dir(k, a.live), s, clamp=k in TF_KINDS)
              for k in ("tf_plateau_probe", "torch_plateau_probe") for s in (2, 3)}
    out, res = [], {"live": a.live}

    # §7.2 setup check: one epoch, GT threading, must reproduce the GT's epoch-0 history row.
    sc = first_row_text(os.path.join(run_dir("tf_setupcheck", a.live), CELL.format(seed=1)))
    gt_row = first_row_text(os.path.join(GT, CELL.format(seed=1)))
    setup_ok = None if sc is None else (sc == gt_row)
    res["setup_check_bit_identical"] = setup_ok
    out += ["## 7.2 TF against itself at converged scale", "",
            f"Setup check (one epoch, the ground truth's threading): epoch-0 `history.csv` row "
            f"{'**bit-identical** to the ground truth' if setup_ok else ('pending' if setup_ok is None else '**DIFFERS** from the ground truth')}.", ""]

    d_budget = None if tf1["best_within_budget"] is None else tf1["best_within_budget"] - gt["best_within_budget"]
    budget_complete = tf1["finished"] or tf1["epochs"] >= BUDGET
    d_mape = None if tf1["test_mape"] is None else tf1["test_mape"] - gt["test_mape"]
    d_replay_budget = replay["best_within_budget"] - gt["best_within_budget"]
    d_replay_mape = replay["test_mape"] - gt["test_mape"]
    if not (budget_complete and tf1["finished"]):
        verdict = "pending"
        if budget_complete and abs(d_budget) >= 0.3:
            verdict = "Confirmed (on (ii) already; (iii) pending)"
    elif abs(d_budget) >= 0.3 or abs(d_mape) >= 1.0:
        verdict = "Confirmed"
    elif abs(d_budget) <= 0.1 and abs(d_mape) <= 0.3:
        verdict = "Suspect"
    else:
        verdict = "Inconclusive"
    res["tf_envelope"] = {"gt": gt, "tf_1thread": tf1, "delta_best_within_budget": d_budget,
                          "delta_test_mape": d_mape, "verdict": verdict}
    out += ["| run | epochs | (i) plateau exit | (ii) best val ≤ epoch 44 | Δ(ii) | best val | (iii) test MAPE | Δ(iii) | test R² |",
            "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
            f"| TF ground truth (4 threads) | {gt['epochs']} | {f(gt['exit'], 'd')} | {f(gt['best_within_budget'], '.4f')} | — | "
            f"{f(gt['best'], '.4f')} | {f(gt['test_mape'])} | — | {f(gt['test_r2'], '.4f')} |",
            f"| **TF, one intra-op thread** | {tf1['epochs']}{'' if tf1['finished'] else ' (running)'} | {f(tf1['exit'], 'd')} | "
            f"{f(tf1['best_within_budget'], '.4f')} | {f(d_budget, '+.3f')} | {f(tf1['best'], '.4f')} | {f(tf1['test_mape'])} | "
            f"{f(d_mape, '+.3f')} | {f(tf1['test_r2'], '.4f')} |",
            f"| PyTorch exact replay (§5) | {replay['epochs']} | {f(replay['exit'], 'd')} | {f(replay['best_within_budget'], '.4f')} | "
            f"{f(d_replay_budget, '+.3f')} | {f(replay['best'], '.4f')} | {f(replay['test_mape'])} | {f(d_replay_mape, '+.3f')} | "
            f"{f(replay['test_r2'], '.4f')} |",
            "", f"Rule (fixed before the run): Confirmed if |Δ(ii)| ≥ 0.3 or |Δ(iii)| ≥ 1; Suspect if |Δ(ii)| ≤ 0.1 and "
            f"|Δ(iii)| ≤ 0.3; otherwise Inconclusive. **Verdict: {verdict}.**", ""]

    # §7.3 the new default pipeline, trained to convergence.
    if native["finished"]:
        dm, dr = native["test_mape"] - gt["test_mape"], native["test_r2"] - gt["test_r2"]
        inside = abs(dm) <= 1.0 and abs(dr) <= 0.03
        if inside:
            v73 = "Pass"
        elif tf1["finished"]:
            tf_dm, tf_dr = abs(tf1["test_mape"] - gt["test_mape"]), abs(tf1["test_r2"] - gt["test_r2"])
            v73 = "Pass (outside the band, within TF's own move)" if abs(dm) <= tf_dm and abs(dr) <= tf_dr else "Fail"
        else:
            v73 = "outside the band — pending §7.2"
    else:
        dm = dr = None
        v73 = "pending"
    res["native"] = {"run": native, "delta_test_mape": dm, "delta_test_r2": dr, "verdict": v73}
    out += ["## 7.3 The new default pipeline, trained to convergence", "",
            "| run | init | epochs | plateau exit | best val ≤ epoch 44 | best val | test MAPE | Δ MAPE | test R² | Δ R² |",
            "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|",
            f"| TF ground truth | keras (TF) | {gt['epochs']} | {f(gt['exit'], 'd')} | {f(gt['best_within_budget'], '.4f')} | "
            f"{f(gt['best'], '.4f')} | {f(gt['test_mape'])} | — | {f(gt['test_r2'], '.4f')} | — |",
            f"| **PyTorch, new default** | {native['init'] or 'keras'} | {native['epochs']}{'' if native['finished'] else ' (running)'} | "
            f"{f(native['exit'], 'd')} | {f(native['best_within_budget'], '.4f')} | {f(native['best'], '.4f')} | "
            f"{f(native['test_mape'])} | {f(dm, '+.3f')} | {f(native['test_r2'], '.4f')} | {f(dr, '+.4f')} |",
            f"| PyTorch, torch init (§5) | torch | {tinit['epochs']} | {f(tinit['exit'], 'd')} | {f(tinit['best_within_budget'], '.4f')} | "
            f"{f(tinit['best'], '.4f')} | {f(tinit['test_mape'])} | {f(tinit['test_mape'] - gt['test_mape'], '+.3f')} | "
            f"{f(tinit['test_r2'], '.4f')} | {f(tinit['test_r2'] - gt['test_r2'], '+.4f')} |",
            "", f"Gate (fixed before the run): |Δ MAPE| ≤ 1 and |Δ R²| ≤ 0.03; outside it only passes if TF moves as far "
            f"against itself (§7.2). **Verdict: {v73}.**", ""]

    # §7.4 plateau test across seeds.
    tf_runs = {1: gt, 2: probes[("tf_plateau_probe", 2)], 3: probes[("tf_plateau_probe", 3)]}
    pt_runs = {1: native, 2: probes[("torch_plateau_probe", 2)], 3: probes[("torch_plateau_probe", 3)]}

    def seen(r, s):
        # enough epochs to know the exit (or the 25-epoch probe horizon reached / run finished)
        return r["exit"] is not None or r["finished"] or r["epochs"] >= PROBE_EPOCHS

    all_seen = all(seen(r, s) for r in list(tf_runs.values()) + list(pt_runs.values()) for s in [0])
    tf_exits = [r["exit"] for r in tf_runs.values()]
    pt_exits = [r["exit"] for r in pt_runs.values()]
    tf_stuck = [s for s, r in tf_runs.items() if r["exit"] is None and seen(r, s)]
    pt_stopped = [s for s, r in pt_runs.items() if stopped_on_plateau(r, PROBE_EPOCHS if s > 1 else 10**9)]
    tf_stopped = [s for s, r in tf_runs.items() if stopped_on_plateau(r, PROBE_EPOCHS if s > 1 else 10**9)]
    if not all_seen:
        v74 = "pending"
    else:
        latest_tf = None if tf_stuck else max(tf_exits)
        late = [s for s, e in pt_runs.items() if latest_tf is not None and (e["exit"] is None or e["exit"] > latest_tf + 5)]
        ok_stop = not pt_stopped or bool(tf_stopped)
        v74 = "Pass" if ok_stop and not late else "Fail"
        if tf_stuck:
            v74 += f" (TF seed(s) {tf_stuck} stayed on the plateau themselves: a property of the original model)"
    res["plateau"] = {"tf_exits": tf_exits, "pytorch_exits": pt_exits, "tf_stuck": tf_stuck,
                      "pytorch_stopped_on_plateau": pt_stopped, "tf_stopped_on_plateau": tf_stopped, "verdict": v74}
    out += ["## 7.4 Plateau test across seeds", "",
            "| seed | TF: plateau exit | TF: epochs | PyTorch (new default): plateau exit | PyTorch: epochs |",
            "|--:|--:|--:|--:|--:|"]
    for s in (1, 2, 3):
        t, q = tf_runs[s], pt_runs[s]
        out.append(f"| {s} | {f(t['exit'], 'd')} | {t['epochs']}{'' if t['finished'] else ' (running)'} | "
                   f"{f(q['exit'], 'd')} | {q['epochs']}{'' if q['finished'] else ' (running)'} |")
    out += ["", "Seed 1: TF ground truth and the §7.3 run; seeds 2 and 3: first 25 epochs of the converged config. "
            "Reference: PyTorch's own init left the plateau at epoch 20 (§5, PYTORCH_PORT.md §5.4).", "",
            f"Rule (fixed before the runs): no PyTorch run early-stopped on the plateau, and none leaves it more than 5 epochs "
            f"after the latest TF seed. **Verdict: {v74}.**", ""]

    # §7.6 TF-drawn vs PyTorch-drawn initial weights (one fixed order, no early stopping).
    state_p = os.path.join(INIT_DRAWS_LIVE if a.live else os.path.join(FROZEN, "init_draws"), "state.json")
    draws = json.load(open(state_p)) if os.path.exists(state_p) else {}
    out += ["## 7.6 TF-drawn versus PyTorch-drawn initial weights", "",
            "| group | seed | plateau exit | epochs | status |", "|---|--:|--:|--:|---|"]
    for jid, d in sorted(draws.items(), key=lambda kv: (kv[1]["group"], kv[1]["seed"])):
        out.append(f"| {d['group']} | {d['seed']} | {d['exit'] if d['exit'] is not None else '— (none)'} | "
                   f"{d['epochs']} | {d['status']} |")
    complete = len(draws) == 12 and all(d["status"] in ("done", "failed") for d in draws.values())
    if complete:
        def value(d):  # a run still on the plateau at epoch 24 counts as 25 (rule)
            return d["exit"] if d["exit"] is not None else PROBE_EPOCHS
        tf_v = [value(d) for d in draws.values() if d["group"] == "TF-drawn" and d["status"] == "done"]
        failed = [d for d in draws.values() if d["status"] == "failed"]
        pt_done = [value(d) for d in draws.values() if d["group"] == "PyTorch-drawn" and d["status"] == "done"]
        variants = {"failed runs excluded": (tf_v, pt_done)}
        if failed:  # the rule predates the NaN case: also count a failed run as never leaving the plateau (25)
            variants["failed runs counted as 25"] = (
                tf_v + [PROBE_EPOCHS for d in failed if d["group"] == "TF-drawn"],
                pt_done + [PROBE_EPOCHS for d in failed if d["group"] == "PyTorch-drawn"])
        res["init_draws"] = {"draws": draws, "variants": {}}
        out += ["", "| treatment | TF-drawn exits | PyTorch-drawn exits | median gap | one-sided Mann–Whitney p | verdict |",
                "|---|---|---|--:|--:|---|"]
        for name, (tv, pv) in variants.items():
            p_mw = mann_whitney_greater(pv, tv)
            gap = float(np.median(pv) - np.median(tv))
            verdict = "real difference" if (p_mw < 0.05 and gap >= 5) else "no evidence of a difference"
            res["init_draws"]["variants"][name] = {"tf": tv, "pytorch": pv, "median_gap": gap, "p": p_mw, "verdict": verdict}
            out.append(f"| {name} | {sorted(tv)} | {sorted(pv)} | {gap:+.1f} | {p_mw:.3f} | **{verdict}** |")
        out += ["", "Rule (fixed before the runs): a real difference if the PyTorch-drawn exits are later with one-sided "
                "Mann–Whitney p < 0.05 and a median gap of at least 5 epochs; otherwise no evidence of a difference. "
                "Exits ≥ 25 are counted as 25.", ""]
    else:
        out += ["", "**Verdict: pending** (the 12 runs have not all finished).", ""]

    text = "\n".join(out)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out + ".md", "w") as fh:
            fh.write(text + "\n")
        with open(a.out + ".json", "w") as fh:
            json.dump(res, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
