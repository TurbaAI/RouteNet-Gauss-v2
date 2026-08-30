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

# L1 parity check: loss, gradients and one optimizer step, TensorFlow (tf_reference/) versus
# PyTorch (models.py + training_lib.py), from IDENTICAL weights on IDENTICAL scenarios.
#
# For each of the first --n scenarios of a partition:
#   loss   Keras MeanAbsolutePercentageError  vs  training_lib.keras_mape_loss
#   grads  tf.GradientTape (float32)          vs  loss.backward() (float32), all 34 TF tensors
#                                                 mapped to the torch layout
#   ref    the PyTorch model evaluated in float64 gives a reference gradient that tells how
#          well-CONDITIONED the float32 gradient is on this scenario (see below)
#   step   Keras Adam(lr=1e-3, clipnorm=1.0)  vs  training_lib.KerasAdam(lr=1e-3, clipnorm=1.0)
#          applied once to fresh copies of the weights; the weight UPDATES are compared.
#
# Why the float64 reference: RouteNet-Gauss back-propagates through 50 windows x 8 message-
# passing iterations of GRU updates (400 recurrent steps). At untrained weights the gradients
# explode (1e8..1e24 in float64) and a float32 evaluation of them is not defined to better than
# orders of magnitude — in EITHER framework (TF's own float32 gradient is off by 300x from the
# float64 value on such scenarios). Comparing two float32 implementations there is meaningless,
# so a tensor's gradient is only gated when it is well-conditioned (torch float32 within 1e-3
# of float64); ill-conditioned tensors are reported with both frameworks' distance to the
# float64 reference instead. At trained weights every tensor is well-conditioned and the gate
# applies to all of them.
#
# Gates: loss rel. diff <= 1e-5. Gradients: a tensor is "well-conditioned" on a scenario when
# TF's own float32 gradient is within 1e-3 of the float64 reference; for those tensors the torch
# float32 gradient must be within max(1e-3, 2 x TF's own error) of the float64 reference, i.e.
# torch is never materially farther from the truth than TF on the same tensor (TF-vs-torch is
# reported for information — it is bounded by the sum of the two). On scenarios where every tensor is
# well-conditioned, the Adam update must agree to <= 1e-2 * lr on every element whose gradient is
# not tiny (|g| > 1e-2 * max|g| of its tensor; Adam's first step is ~lr*sign(g), so near-zero
# gradient elements legitimately turn float noise into O(lr) update differences — they are
# counted and reported, not gated). A wrong epsilon placement, bias correction or clipping
# semantics would violate these gates by orders of magnitude.
#
# Run in conda env RG_torch from the repo root, e.g. (trained weights: all tensors gated)
#   python parity/l1_grad_step.py \
#       --checkpoint tensorflow_version_gt/converged/ckpt/trex_multiburst/RouteNetGauss/delay/seed_1/30-7.0663 \
#       --z-scores  tensorflow_version_gt/converged/normalization/trex_multiburst/RouteNetGauss/delay/seed_1/z_scores.pkl \
#       --dataset trex_multiburst --partition training --target delay --n 5

import argparse
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
import tensorflow as tf
import torch

from convert_tf_checkpoint import GRU_ATTRS, MLP_ATTRS, load_tf_arrays, tf_arrays_to_state_dict
from models import RouteNetGauss
from tf_reference.models import RouteNetGauss as TFRouteNetGauss
from tf_reference.utils import load_dataset as tf_load_dataset
from tf_reference.utils import prepare_targets_and_mask as tf_prepare_targets_and_mask
from torch_ragged import Ragged, sample_to_device
from training_lib import KerasAdam, keras_mape_loss
from utils import load_dataset, prepare_targets_and_mask

PERCENTILES = ["avg", "p50", "p90", "p95", "p99"]
LR, CLIPNORM = 0.001, 1.0  # experiment.py: Adam(learning_rate=0.001, clipnorm=1.0)
COND_TOL = 1e-3  # a tensor is well-conditioned if TF's fp32 gradient is within this of the fp64 reference
UPDATE_TOL = 1e-2  # gate on |update_torch - update_tf| / lr for significant elements
# "Significant" elements for the update gate: |g| > SIG_FRAC * max|g| of the tensor. Adam's first
# step is lr * g_c / (|g_c| + eps'), eps' = eps/sqrt(1-beta2) = 3.2e-6 on the per-tensor-clipped
# gradient g_c (norm 1), so d(update)/d(g_c) ~ lr * eps'/|g_c|^2: at |g_c| = 1e-4 a 5e-6 gradient
# difference already moves the update by 1e-3 * lr. Gating at 1e-2 of the max keeps the
# sensitivity at O(1) so the gate tests the optimizer arithmetic, not float noise.
SIG_FRAC = 1e-2


def build_targets(target):
    return [f"flow_{p}_{target}" for p in PERCENTILES], f"flow_has_{target}"


def tf_named_variables(model):
    """{tf checkpoint name: tf.Variable} for the 34 weight tensors of a built TF model."""
    out = {}
    for attr in GRU_ATTRS:
        cell = getattr(model, attr)
        out[f"{attr}/kernel"], out[f"{attr}/recurrent_kernel"], out[f"{attr}/bias"] = cell.kernel, cell.recurrent_kernel, cell.bias
    for attr in MLP_ATTRS:
        layers = [l for l in getattr(model, attr).layers if l.weights]
        for i, layer in enumerate(layers):
            out[f"{attr}/layer_with_weights-{i}/kernel"] = layer.kernel
            out[f"{attr}/layer_with_weights-{i}/bias"] = layer.bias
    return out


def rel(ref, other):
    ref, other = np.asarray(ref, np.float64), np.asarray(other, np.float64)
    return float(np.max(np.abs(ref - other)) / (np.max(np.abs(ref)) + 1e-30))


def cast_sample(x, dt):
    out = {}
    for k, v in x.items():
        if isinstance(v, Ragged):
            out[k] = v.with_values(v.values.to(dt)) if v.values.is_floating_point() else v
        elif isinstance(v, torch.Tensor) and v.is_floating_point():
            out[k] = v.to(dt)
        else:
            out[k] = v
    return out


def torch_loss_grads(sd0, z_scores, mask, target, x, y, device, dtype):
    """Fresh torch model in `dtype` with weights sd0: (loss, grads {name: float64 array}, model)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        m = RouteNetGauss(output_dim=len(PERCENTILES), mask_field=mask, use_trans_delay=target == "delay", z_scores=z_scores)
        m.load_state_dict({k: v.to(dtype) for k, v in sd0.items()}, strict=False)
        m.to(device).train()
        xd, yd = sample_to_device(cast_sample(x, dtype), device), y.to(device).to(dtype)
        y_pred = m(xd)
        loss = keras_mape_loss(yd, y_pred)
        loss.backward()
    finally:
        torch.set_default_dtype(prev)
    grads = {k: p.grad.detach().cpu().numpy().astype(np.float64) for k, p in m.named_parameters()}
    return float(loss.detach()), grads, m, y_pred.detach().cpu().numpy()


def main():
    p = argparse.ArgumentParser(description="L1 parity: loss / gradients / one Adam step, TF vs PyTorch")
    p.add_argument("--checkpoint", required=True, help="TF checkpoint prefix or init_weights.npz")
    p.add_argument("--z-scores", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--partition", default="training")
    p.add_argument("--target", required=True, choices=["delay", "jitter"])
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--tf-data-path", default="data")
    p.add_argument("--torch-data-path", default="data_torch")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    targets, mask = build_targets(args.target)
    z_scores = pickle.load(open(args.z_scores, "rb"))
    arrays = load_tf_arrays(args.checkpoint)
    sd0 = tf_arrays_to_state_dict(arrays)
    old = {k: v.numpy().astype(np.float64) for k, v in sd0.items()}

    tf_ds = tf_load_dataset(f"{args.dataset}/{args.partition}", data_path=args.tf_data_path).map(tf_prepare_targets_and_mask(targets, mask))
    torch_ds = load_dataset(f"{args.dataset}/{args.partition}", data_path=args.torch_data_path).map(prepare_targets_and_mask(targets, mask))
    tf_loss_fn = tf.keras.losses.MeanAbsolutePercentageError()

    rows = []
    for i, ((x_tf, y_tf), (x_t, y_t)) in enumerate(zip(tf_ds, torch_ds)):
        if i >= args.n:
            break
        assert int(x_tf["sample_idx"]) == int(x_t["sample_idx"])

        # ---- TF (float32): fresh model with the reference weights, loss, grads, one Adam step ----
        tf_model = TFRouteNetGauss(output_dim=len(targets), mask_field=mask, use_trans_delay=args.target == "delay", z_scores=z_scores)
        tf_model(x_tf)  # build
        tf_vars = tf_named_variables(tf_model)
        for k, v in tf_vars.items():
            v.assign(arrays[k])
        with tf.GradientTape() as tape:
            y_pred_tf = tf_model(x_tf)
            loss_tf = tf_loss_fn(y_tf, y_pred_tf)
        var_list = list(tf_vars.values())
        grads_tf = {k: g.numpy() for k, g in zip(tf_vars.keys(), tape.gradient(loss_tf, var_list))}
        tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIPNORM).apply_gradients(
            zip([tf.convert_to_tensor(grads_tf[k]) for k in tf_vars], var_list))
        new_tf = {k: v.numpy().astype(np.float64) for k, v in tf_arrays_to_state_dict({k: v.numpy() for k, v in tf_vars.items()}).items()}
        grads_tf = {k: v.numpy().astype(np.float64) for k, v in tf_arrays_to_state_dict(grads_tf).items()}

        # ---- PyTorch float32 (+ one KerasAdam step) and float64 reference ----
        loss_t, grads_t, m32, y_pred_t = torch_loss_grads(sd0, z_scores, mask, args.target, x_t, y_t, device, torch.float32)
        KerasAdam(m32.parameters(), lr=LR, clipnorm=CLIPNORM).step()
        new_t = {k: p.detach().cpu().numpy().astype(np.float64) for k, p in m32.named_parameters()}
        loss_64, grads_64, _, _ = torch_loss_grads(sd0, z_scores, mask, args.target, x_t, y_t, device, torch.float64)

        # ---- gradients: conditioning + gate ----
        per = {}
        n_ill, worst_gated, worst_gated_tf_vs_torch = 0, 0.0, 0.0
        for k in grads_64:
            torch_vs_64 = rel(grads_64[k], grads_t[k])
            tf_vs_64 = rel(grads_64[k], grads_tf[k])
            tf_vs_torch = rel(grads_tf[k], grads_t[k])
            entry = {"max_abs_grad_fp64": float(np.max(np.abs(grads_64[k]))), "torch32_vs_fp64": torch_vs_64,
                     "tf32_vs_fp64": tf_vs_64, "tf32_vs_torch32": tf_vs_torch,
                     "well_conditioned": bool(tf_vs_64 <= COND_TOL)}
            # Gate: torch's float32 error vs the float64 truth is at most max(COND_TOL, 2x TF's own
            # float32 error) — i.e. torch is never materially worse than TF on the same tensor.
            entry["gate_tol"] = max(COND_TOL, 2 * tf_vs_64) if entry["well_conditioned"] else None
            entry["gate_passed"] = bool(torch_vs_64 <= entry["gate_tol"]) if entry["well_conditioned"] else None
            if entry["well_conditioned"]:
                worst_gated = max(worst_gated, torch_vs_64 / entry["gate_tol"])
                worst_gated_tf_vs_torch = max(worst_gated_tf_vs_torch, tf_vs_torch)
            else:
                n_ill += 1
            per[k] = entry
        grads_gate = all(e["gate_passed"] for e in per.values() if e["well_conditioned"])

        # ---- one-step update comparison (only meaningful when everything is well-conditioned) ----
        n_sig, n_sig_bad, worst_sig, n_all_bad = 0, 0, 0.0, 0
        for k in old:
            d = np.abs((new_tf[k] - old[k]) - (new_t[k] - old[k])) / LR
            g = np.abs(grads_tf[k])
            sig = g > SIG_FRAC * (g.max() + 1e-30)
            n_sig += int(sig.sum())
            n_sig_bad += int((d[sig] > UPDATE_TOL).sum())
            n_all_bad += int((d > 1e-2).sum())
            if sig.any():
                worst_sig = max(worst_sig, float(d[sig].max()))
        step_gate = None if n_ill else (n_sig_bad == 0)

        loss_rel = abs(float(loss_tf) - loss_t) / abs(float(loss_tf))
        row = {
            "i": i, "sample_idx": int(x_tf["sample_idx"]), "n_pred": int(y_tf.shape[0]),
            "loss_tf32": float(loss_tf), "loss_torch32": loss_t, "loss_torch64": loss_64, "loss_rel_diff": loss_rel,
            "pred_scale_rel_diff": rel(y_pred_tf.numpy(), y_pred_t),
            "max_abs_grad_fp64": max(e["max_abs_grad_fp64"] for e in per.values()),
            "n_tensors": len(per), "n_ill_conditioned": n_ill,
            "grad_worst_torch32_vs_fp64_well_conditioned": max([e["torch32_vs_fp64"] for e in per.values() if e["well_conditioned"]] or [0.0]),
            "grad_worst_gate_ratio_well_conditioned": worst_gated,
            "grad_worst_tf32_vs_torch32_well_conditioned": worst_gated_tf_vs_torch,
            "grad_worst_tf32_vs_fp64": max(e["tf32_vs_fp64"] for e in per.values()),
            "grad_worst_torch32_vs_fp64": max(e["torch32_vs_fp64"] for e in per.values()),
            "grad_per_tensor": per,
            "update_n_significant_elements": n_sig, "update_n_significant_bad": n_sig_bad,
            "update_worst_significant_diff_over_lr": worst_sig, "update_n_elements_diff_over_1e-2_lr": n_all_bad,
            "loss_gate": bool(loss_rel <= 1e-5), "grads_gate": bool(grads_gate), "step_gate": step_gate,
        }
        row["passed"] = bool(row["loss_gate"] and row["grads_gate"] and (step_gate in (None, True)))
        rows.append(row)
        print(f"[{i}] idx={row['sample_idx']} n={row['n_pred']} | loss tf={row['loss_tf32']:.6f} torch={loss_t:.6f} (fp64 {loss_64:.6f}) rel={loss_rel:.1e} "
              f"| max|grad| fp64={row['max_abs_grad_fp64']:.2e} | ill-conditioned tensors {n_ill}/{len(per)} "
              f"| gated grads: torch32-vs-fp64 worst={row['grad_worst_torch32_vs_fp64_well_conditioned']:.2e} (worst ratio to gate {worst_gated:.2f}; tf32-vs-torch32 {worst_gated_tf_vs_torch:.2e}) | all tensors: tf32-vs-fp64 worst={row['grad_worst_tf32_vs_fp64']:.2e} torch32-vs-fp64 worst={row['grad_worst_torch32_vs_fp64']:.2e} "
              f"| step gate={step_gate} (sig elems {n_sig}, bad {n_sig_bad}, worst Δ/lr={worst_sig:.2e}) | passed={row['passed']}", flush=True)

    report = {
        "checkpoint": args.checkpoint, "z_scores": args.z_scores, "dataset": args.dataset, "partition": args.partition,
        "target": args.target, "device": args.device, "lr": LR, "clipnorm": CLIPNORM, "cond_tol": COND_TOL, "n_scenarios": len(rows),
        "worst_loss_rel_diff": max(r["loss_rel_diff"] for r in rows),
        "worst_gated_grad_torch32_vs_fp64": max(r["grad_worst_torch32_vs_fp64_well_conditioned"] for r in rows),
        "worst_gated_grad_tf32_vs_torch32": max(r["grad_worst_tf32_vs_torch32_well_conditioned"] for r in rows),
        "update_tol_over_lr": UPDATE_TOL,
        "n_scenarios_with_ill_conditioned_tensors": sum(1 for r in rows if r["n_ill_conditioned"]),
        "worst_update_significant_diff_over_lr_gated": max([r["update_worst_significant_diff_over_lr"] for r in rows if r["step_gate"] is not None] or [None]) if any(r["step_gate"] is not None for r in rows) else None,
        "passed": all(r["passed"] for r in rows),
        "scenarios": rows,
    }
    print("\n===== L1 loss / gradient / one-step parity =====")
    for k in ["n_scenarios", "worst_loss_rel_diff", "worst_gated_grad_torch32_vs_fp64", "worst_gated_grad_tf32_vs_torch32", "n_scenarios_with_ill_conditioned_tensors",
              "worst_update_significant_diff_over_lr_gated", "passed"]:
        print(f"  {k:46s} {report[k]}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"  report -> {args.out}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
