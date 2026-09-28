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

# Keras-init check: does PyTorch's `RouteNetGauss(init="keras")` (models.init_keras_style_) draw
# its initial weights from the same distributions as the Keras initialisers of the TF original?
#
# Why it needs its own check: every exact-replay run loads TF's *recorded* initial weights, so
# no training run had ever used the PyTorch-drawn Keras init before it became the default of
# experiment.py / run_experiments.py / train.py (PYTORCH_PORT.md §5.4).
#
# References (per parameter tensor, in the PyTorch layout):
#   A. the TF initial weights recorded by tf_reference/replay_tf_run.py
#      (tensorflow_version_gt/replay/<cell>/init_weights.pt, converted by convert_tf_checkpoint.py).
#      The 8 cells hold only 2 distinct draws (one per seed; dataset and target do not enter the
#      Keras initialiser RNG) — duplicates are removed. This reference also validates the
#      TF -> torch layout mapping (fan-in/fan-out, GRU gate order, biases).
#   B. fresh draws of the Keras initialisers themselves (tf.keras.initializers.GlorotUniform /
#      Orthogonal on the Keras shape, transposed to the torch layout) — a larger sample for the
#      distribution test.
# Candidate: --draws PyTorch models built with init="keras" under torch.manual_seed(1..draws).
#
# Checks (all must hold for every tensor):
#   - coverage: every parameter is re-initialised by init_keras_style_ (Linear, GRUCell, GRU);
#   - biases exactly 0;
#   - glorot-uniform kernels: all |w| <= sqrt(6 / (fan_in + fan_out)) and the pooled standard
#     deviation within 5 sampling sigmas of limit / sqrt(3);
#   - orthogonal recurrent kernels ([3H, H] torch = Keras [H, 3H]): max |W^T W - I| <= 1e-5 in
#     the PyTorch draws AND in the recorded TF weights;
#   - distribution: two-sample Kolmogorov-Smirnov test of the pooled PyTorch values against
#     reference A and against reference B, p >= 1e-4.
#
# Run from the repo root in conda env RG_torch (needs tensorflow-cpu for reference B):
#   python parity/check_keras_init.py [--draws 16] [--out pytorch_version_results/parity/keras_init_check]

import argparse
import glob
import hashlib
import json
import os
import pickle
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn

from models import RouteNetGauss

KS_P_MIN = 1e-4
ORTHO_TOL = 1e-5
STD_SIGMAS = 5.0


def ks_2samp(a, b):
    """Two-sample Kolmogorov-Smirnov statistic and asymptotic p-value (no scipy in RG_torch)."""
    a, b = np.sort(np.asarray(a, np.float64)), np.sort(np.asarray(b, np.float64))
    n, m = len(a), len(b)
    both = np.concatenate([a, b])
    d = float(np.max(np.abs(np.searchsorted(a, both, "right") / n - np.searchsorted(b, both, "right") / m)))
    en = np.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 1e-3:
        return d, 1.0
    k = np.arange(1, 101)
    p = 2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * k**2 * lam**2))
    return d, float(min(max(p, 0.0), 1.0))


def param_kinds(model):
    """name -> 'glorot' | 'orthogonal' | 'zero', mirroring models.init_keras_style_."""
    kinds = {}
    for mname, m in model.named_modules():
        pre = f"{mname}." if mname else ""
        if isinstance(m, nn.Linear):
            kinds[pre + "weight"], kinds[pre + "bias"] = "glorot", "zero"
        elif isinstance(m, nn.GRUCell):
            kinds.update({pre + "weight_ih": "glorot", pre + "weight_hh": "orthogonal",
                          pre + "bias_ih": "zero", pre + "bias_hh": "zero"})
        elif isinstance(m, nn.GRU):
            kinds.update({pre + "weight_ih_l0": "glorot", pre + "weight_hh_l0": "orthogonal",
                          pre + "bias_ih_l0": "zero", pre + "bias_hh_l0": "zero"})
    return kinds


def ortho_err(w):
    """max |W^T W - I| for a torch-layout [3H, H] recurrent kernel (orthonormal columns)."""
    w = np.asarray(w, np.float64)
    return float(np.max(np.abs(w.T @ w - np.eye(w.shape[1]))))


def keras_draws(kind, shape, n, seed0):
    """Reference B: n fresh Keras initialiser draws for a torch-layout tensor of `shape`."""
    import tensorflow as tf

    out = []
    for s in range(n):
        if kind == "glorot":
            init = tf.keras.initializers.GlorotUniform(seed=seed0 + s)
        else:
            init = tf.keras.initializers.Orthogonal(seed=seed0 + s)
        out.append(init(shape[::-1]).numpy().T)  # Keras [in, out] / [H, 3H] -> torch [out, in] / [3H, H]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--draws", type=int, default=16, help="PyTorch (and Keras reference B) draws per tensor")
    p.add_argument("--replay-root", default="tensorflow_version_gt/replay")
    p.add_argument("--out", default="pytorch_version_results/parity/keras_init_check",
                   help="writes <out>.json and <out>.md")
    args = p.parse_args()
    torch.set_num_threads(1)

    # Reference A: recorded TF initial weights, deduplicated.
    cells = sorted(glob.glob(os.path.join(args.replay_root, "*", "RouteNetGauss", "*", "seed_*")))
    tf_sets, seen = {}, {}
    for c in cells:
        sd = torch.load(os.path.join(c, "init_weights.pt"), map_location="cpu", weights_only=True)
        h = hashlib.sha1(b"".join(v.numpy().tobytes() for _, v in sorted(sd.items()))).hexdigest()
        seen.setdefault(h, []).append(os.path.relpath(c, args.replay_root))
        tf_sets.setdefault(h, (c, sd))
    print(f"reference A: {len(cells)} recorded cells -> {len(tf_sets)} distinct TF draws")
    for h, cs in seen.items():
        print(f"  {h[:10]}: {', '.join(cs)}")

    # Candidate: PyTorch init="keras" draws (same constructor arguments as experiment.py).
    c0 = next(iter(tf_sets.values()))[0]
    target = "delay" if "/delay/" in c0.replace(os.sep, "/") else "jitter"
    with open(os.path.join(c0, "z_scores.pkl"), "rb") as f:
        z_scores = pickle.load(f)
    torch_sets = []
    for s in range(1, args.draws + 1):
        torch.manual_seed(s)
        model = RouteNetGauss(output_dim=5, mask_field=f"flow_has_{target}",
                              use_trans_delay=target == "delay", z_scores=z_scores, init="keras")
        torch_sets.append({k: v.detach().numpy().copy() for k, v in model.state_dict().items()})
    kinds = param_kinds(model)
    names = [k for k in torch_sets[0] if not k.startswith("z_")]

    rows, ok_all = [], True
    uncovered = [k for k in names if k not in kinds]
    if uncovered:
        ok_all = False
        print("NOT re-initialised by init_keras_style_:", uncovered)
    tf_arrays = [sd for _, sd in tf_sets.values()]
    for i, name in enumerate(names):
        kind = kinds.get(name, "uncovered")
        shape = torch_sets[0][name].shape
        cand = [ts[name] for ts in torch_sets]
        ref_a = [sd[name].numpy() for sd in tf_arrays]
        assert all(a.shape == shape for a in ref_a), (name, shape, [a.shape for a in ref_a])
        row = {"tensor": name, "kind": kind, "shape": list(shape)}
        checks = {}
        if kind == "zero":
            row["torch_max_abs"] = float(max(np.abs(c).max() for c in cand))
            row["tf_max_abs"] = float(max(np.abs(a).max() for a in ref_a))
            checks["zero"] = row["torch_max_abs"] == 0.0 and row["tf_max_abs"] == 0.0
        elif kind in ("glorot", "orthogonal"):
            pooled = np.concatenate([c.ravel() for c in cand])
            ref_b = keras_draws(kind, shape, args.draws, seed0=1000 * (i + 1))
            row["torch_std"] = float(pooled.std())
            row["tf_std"] = float(np.concatenate([a.ravel() for a in ref_a]).std())
            row["keras_std"] = float(np.concatenate([b.ravel() for b in ref_b]).std())
            if kind == "glorot":
                fan_out, fan_in = shape
                limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
                expected = limit / np.sqrt(3.0)
                tol = STD_SIGMAS * np.sqrt(0.2 / pooled.size)  # rel. sd of a uniform sample std
                row.update(limit=limit, expected_std=float(expected),
                           torch_max_abs=float(np.abs(pooled).max()),
                           tf_max_abs=float(max(np.abs(a).max() for a in ref_a)))
                checks["bound"] = row["torch_max_abs"] <= limit * (1 + 1e-6) and row["tf_max_abs"] <= limit * (1 + 1e-6)
                checks["std"] = abs(row["torch_std"] / expected - 1.0) <= tol
            else:
                row["torch_ortho_err"] = max(ortho_err(c) for c in cand)
                row["tf_ortho_err"] = max(ortho_err(a) for a in ref_a)
                checks["orthogonal"] = row["torch_ortho_err"] <= ORTHO_TOL and row["tf_ortho_err"] <= ORTHO_TOL
            row["ks_vs_tf_D"], row["ks_vs_tf_p"] = ks_2samp(pooled, np.concatenate([a.ravel() for a in ref_a]))
            row["ks_vs_keras_D"], row["ks_vs_keras_p"] = ks_2samp(pooled, np.concatenate([b.ravel() for b in ref_b]))
            checks["ks"] = row["ks_vs_tf_p"] >= KS_P_MIN and row["ks_vs_keras_p"] >= KS_P_MIN
        else:
            checks["covered"] = False
        row["checks"] = {k: bool(v) for k, v in checks.items()}
        row["passed"] = all(row["checks"].values())
        ok_all &= row["passed"]
        rows.append(row)

    summary = {
        "passed": bool(ok_all),
        "torch_draws": args.draws,
        "tf_recorded_distinct_draws": len(tf_sets),
        "tf_recorded_cells": {h[:10]: cs for h, cs in seen.items()},
        "keras_reference_draws": args.draws,
        "tensors": len(rows),
        "min_ks_p_vs_tf": min((r["ks_vs_tf_p"] for r in rows if "ks_vs_tf_p" in r), default=None),
        "min_ks_p_vs_keras": min((r["ks_vs_keras_p"] for r in rows if "ks_vs_keras_p" in r), default=None),
        "max_torch_ortho_err": max((r["torch_ortho_err"] for r in rows if "torch_ortho_err" in r), default=None),
        "max_tf_ortho_err": max((r["tf_ortho_err"] for r in rows if "tf_ortho_err" in r), default=None),
        "gates": {"ks_p_min": KS_P_MIN, "ortho_tol": ORTHO_TOL, "std_sigmas": STD_SIGMAS},
        "torch_version": torch.__version__,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump({"summary": summary, "tensors": rows}, f, indent=2)
    lines = [
        "# Keras-init check (`parity/check_keras_init.py`)",
        "",
        f"PyTorch `init=\"keras\"`: {args.draws} draws · reference A: {len(tf_sets)} distinct recorded TF "
        f"initial weight sets ({len(cells)} cells) · reference B: {args.draws} fresh Keras initialiser draws.",
        f"**{'PASSED' if ok_all else 'FAILED'}** — min KS p vs TF {summary['min_ks_p_vs_tf']:.3g}, "
        f"vs Keras {summary['min_ks_p_vs_keras']:.3g} (gate {KS_P_MIN:g}); max |WᵀW−I| torch "
        f"{summary['max_torch_ortho_err']:.1e}, TF {summary['max_tf_ortho_err']:.1e} (gate {ORTHO_TOL:g}).",
        "",
        "| tensor | kind | shape | std torch | std TF | std Keras | expected | max\\|w\\| / limit | ‖WᵀW−I‖ torch / TF | KS p vs TF | KS p vs Keras | passed |",
        "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    for r in rows:
        def g(k, fmt):
            return format(r[k], fmt) if k in r else ""
        ratio = f"{r['torch_max_abs'] / r['limit']:.4f}" if "limit" in r else ""
        ortho = f"{r['torch_ortho_err']:.1e} / {r['tf_ortho_err']:.1e}" if "torch_ortho_err" in r else ""
        if r["kind"] == "zero":
            lines.append(f"| `{r['tensor']}` | zero | {tuple(r['shape'])} | max\\|b\\| {r['torch_max_abs']:g} | "
                         f"max\\|b\\| {r['tf_max_abs']:g} | | | | | | | {r['passed']} |")
            continue
        lines.append(
            f"| `{r['tensor']}` | {r['kind']} | {tuple(r['shape'])} | {g('torch_std', '.5f')} | {g('tf_std', '.5f')} | "
            f"{g('keras_std', '.5f')} | {g('expected_std', '.5f')} | {ratio} | {ortho} | {g('ks_vs_tf_p', '.3f')} | "
            f"{g('ks_vs_keras_p', '.3f')} | {r['passed']} |"
        )
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[3:5]))
    print(f"-> {args.out}.json, {args.out}.md")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
