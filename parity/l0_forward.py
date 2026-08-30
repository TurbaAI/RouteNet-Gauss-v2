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

# L0 parity check: forward pass of the frozen TensorFlow RouteNetGauss (tf_reference/) versus
# the PyTorch RouteNetGauss (models.py) with IDENTICAL weights on IDENTICAL scenarios.
#
# For every scenario of a dataset partition it runs both models and compares
#   - the target tensors produced by the two data pipelines (must be bit-identical: this
#     validates data_torch/ + utils.prepare_targets_and_mask against tf.data),
#   - the predictions: max |tf - torch| and, per scenario, max |tf - torch| / max |tf|
#     ("scale-relative" error; the gate), plus the elementwise relative error for information,
#   - the resulting MAPE against the targets on both sides.
# Optionally the TF predictions are also compared with a stored GT predictions.npz.
#
# Run in an env with tensorflow(-cpu) AND torch (conda env RG_torch), from the repo root:
#   python parity/l0_forward.py \
#       --checkpoint tensorflow_version_gt/converged/ckpt/trex_multiburst/RouteNetGauss/delay/seed_1/30-7.0663 \
#       --z-scores  tensorflow_version_gt/converged/normalization/trex_multiburst/RouteNetGauss/delay/seed_1/z_scores.pkl \
#       --dataset trex_multiburst --partition test --target delay --inference-mode \
#       --gt-predictions tensorflow_version_gt/converged/results/trex_multiburst/RouteNetGauss/delay/seed_1/predictions.npz

import argparse
import json
import os
import pickle
import sys
import time

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
from torch_ragged import sample_to_device
from utils import load_dataset, prepare_targets_and_mask

PERCENTILES = ["avg", "p50", "p90", "p95", "p99"]


def build_targets(target):
    return [f"flow_{p}_{target}" for p in PERCENTILES], f"flow_has_{target}"


def assign_tf_weights(model, arrays):
    """Assign the 34 tensors of an init_weights.npz / checkpoint dump to a *built* TF model."""
    for attr in GRU_ATTRS:
        cell = getattr(model, attr)
        cell.kernel.assign(arrays[f"{attr}/kernel"])
        cell.recurrent_kernel.assign(arrays[f"{attr}/recurrent_kernel"])
        cell.bias.assign(arrays[f"{attr}/bias"])
    for attr in MLP_ATTRS:
        layers = [l for l in getattr(model, attr).layers if l.weights]
        for i, layer in enumerate(layers):
            layer.kernel.assign(arrays[f"{attr}/layer_with_weights-{i}/kernel"])
            layer.bias.assign(arrays[f"{attr}/layer_with_weights-{i}/bias"])


def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def main():
    p = argparse.ArgumentParser(description="L0 forward parity: TF vs PyTorch RouteNetGauss")
    p.add_argument("--checkpoint", required=True, help="TF checkpoint prefix or init_weights.npz")
    p.add_argument("--z-scores", required=True, help="z_scores.pkl matching the checkpoint")
    p.add_argument("--dataset", required=True)
    p.add_argument("--partition", default="test")
    p.add_argument("--target", required=True, choices=["delay", "jitter"])
    p.add_argument("--n", type=int, default=None, help="limit the number of scenarios")
    p.add_argument("--inference-mode", action="store_true", help="clamp predictions >= 0 (both models)")
    p.add_argument("--device", default="cpu", help="torch device (TF always runs on CPU)")
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads (oversubscription with other jobs is very slow)")
    p.add_argument("--tf-data-path", default="data")
    p.add_argument("--torch-data-path", default="data_torch")
    p.add_argument("--gt-predictions", default=None, help="predictions.npz to compare the TF predictions with")
    # 5e-4: chosen AFTER measuring — across the 16 repo checkpoints on their full test sets the
    # worst per-scenario scale-relative difference spans 1e-5..2.7e-4 (float32 accumulation-order
    # noise; the two >1e-4 cases agree with TF in MAPE to 6 digits). See PYTORCH_PARITY.md §1.
    p.add_argument("--tol-rel", type=float, default=5e-4, help="gate: per-scenario max|tf-torch| / max|tf|")
    p.add_argument("--out", default=None, help="write the report as JSON here")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        torch.use_deterministic_algorithms(True)
        # Ampere+ GPUs run float32 matmuls/RNNs in TF32 (10-bit mantissa) unless told otherwise;
        # full float32 is required for a like-for-like comparison with TF's CPU float32.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    targets, mask = build_targets(args.target)
    z_scores = pickle.load(open(args.z_scores, "rb"))
    arrays = load_tf_arrays(args.checkpoint)

    # --- TF model (frozen original) ---
    tf_model = TFRouteNetGauss(output_dim=len(targets), mask_field=mask, inference_mode=args.inference_mode,
                               use_trans_delay=args.target == "delay", z_scores=z_scores)
    tf_ds = tf_load_dataset(f"{args.dataset}/{args.partition}", data_path=args.tf_data_path).map(
        tf_prepare_targets_and_mask(targets, mask))
    if args.checkpoint.endswith(".npz"):
        x0, _ = next(iter(tf_ds))
        tf_model(x0)  # build
        assign_tf_weights(tf_model, arrays)
    else:
        tf_model.load_weights(args.checkpoint).expect_partial()

    # --- PyTorch model with the same weights ---
    device = torch.device(args.device)
    torch_model = RouteNetGauss(output_dim=len(targets), mask_field=mask, inference_mode=args.inference_mode,
                                use_trans_delay=args.target == "delay", z_scores=z_scores)
    missing, unexpected = torch_model.load_state_dict(tf_arrays_to_state_dict(arrays), strict=False)
    assert not unexpected and all(k.startswith("z_") for k in missing), (missing, unexpected)
    torch_model.to(device).eval()
    torch_ds = load_dataset(f"{args.dataset}/{args.partition}", data_path=args.torch_data_path).map(
        prepare_targets_and_mask(targets, mask))

    rows, tf_time, torch_time = [], 0.0, 0.0
    y_true_all, y_tf_all, y_torch_all = [], [], []
    for i, ((x_tf, y_tf_true), (x_t, y_t_true)) in enumerate(zip(tf_ds, torch_ds)):
        if args.n is not None and i >= args.n:
            break
        assert int(x_tf["sample_idx"]) == int(x_t["sample_idx"]), "datasets out of step"
        t0 = time.time()
        y_tf = tf_model(x_tf).numpy()
        tf_time += time.time() - t0
        t0 = time.time()
        with torch.no_grad():
            y_t = torch_model(sample_to_device(x_t, device)).cpu().numpy()
        torch_time += time.time() - t0

        y_tf_true, y_t_true = y_tf_true.numpy(), y_t_true.numpy()
        targets_identical = y_tf_true.shape == y_t_true.shape and np.array_equal(y_tf_true, y_t_true)
        assert y_tf.shape == y_t.shape, (y_tf.shape, y_t.shape)
        diff = np.abs(y_tf.astype(np.float64) - y_t.astype(np.float64))
        scale = float(np.max(np.abs(y_tf))) + 1e-30
        rows.append({
            "i": i, "sample_idx": int(x_tf["sample_idx"]), "n_pred": int(y_tf.shape[0]),
            "targets_identical": bool(targets_identical),
            "max_abs_diff": float(diff.max()), "max_scale_rel_diff": float(diff.max() / scale),
            "max_elem_rel_diff": float(np.max(diff / (np.abs(y_tf) + 1e-12))),
            "mape_tf": mape(y_tf_true, y_tf), "mape_torch": mape(y_t_true, y_t),
        })
        y_true_all.append(y_tf_true); y_tf_all.append(y_tf); y_torch_all.append(y_t)
        print(f"[{i:4d}] idx={rows[-1]['sample_idx']:5d} n={rows[-1]['n_pred']:6d} targets_identical={targets_identical} "
              f"max|Δ|={rows[-1]['max_abs_diff']:.3e} scale-rel={rows[-1]['max_scale_rel_diff']:.3e} "
              f"elem-rel={rows[-1]['max_elem_rel_diff']:.3e} MAPE tf={rows[-1]['mape_tf']:.4f} torch={rows[-1]['mape_torch']:.4f}", flush=True)

    y_true_all = np.concatenate(y_true_all); y_tf_all = np.concatenate(y_tf_all); y_torch_all = np.concatenate(y_torch_all)
    worst_scale_rel = max(r["max_scale_rel_diff"] for r in rows)
    report = {
        "checkpoint": args.checkpoint, "z_scores": args.z_scores, "dataset": args.dataset, "partition": args.partition,
        "target": args.target, "inference_mode": args.inference_mode, "device": args.device, "threads": args.threads,
        "n_scenarios": len(rows), "n_predictions": int(y_tf_all.shape[0]),
        "all_targets_identical": all(r["targets_identical"] for r in rows),
        "max_abs_diff": max(r["max_abs_diff"] for r in rows),
        "worst_scale_rel_diff": worst_scale_rel,
        "worst_elem_rel_diff": max(r["max_elem_rel_diff"] for r in rows),
        "mape_tf": mape(y_true_all, y_tf_all), "mape_torch": mape(y_true_all, y_torch_all),
        "tf_seconds_per_scenario": tf_time / len(rows), "torch_seconds_per_scenario": torch_time / len(rows),
        "tol_rel": args.tol_rel, "passed": bool(worst_scale_rel <= args.tol_rel and all(r["targets_identical"] for r in rows)),
        "scenarios": rows,
    }
    if args.gt_predictions:
        g = np.load(args.gt_predictions)
        same_shape = g["y_pred"].shape == y_tf_all.shape
        report["gt_predictions"] = {
            "path": args.gt_predictions, "same_shape": bool(same_shape),
            "tf_vs_gt_bit_identical": bool(same_shape and np.array_equal(g["y_pred"], y_tf_all)),
            "tf_vs_gt_max_abs_diff": float(np.max(np.abs(g["y_pred"] - y_tf_all))) if same_shape else None,
            "torch_vs_gt_max_abs_diff": float(np.max(np.abs(g["y_pred"] - y_torch_all))) if same_shape else None,
            "targets_vs_gt_bit_identical": bool(same_shape and np.array_equal(g["y_true"], y_true_all)),
        }
    print("\n===== L0 forward parity =====")
    for k in ["n_scenarios", "n_predictions", "all_targets_identical", "max_abs_diff", "worst_scale_rel_diff",
              "worst_elem_rel_diff", "mape_tf", "mape_torch", "tf_seconds_per_scenario", "torch_seconds_per_scenario", "passed"]:
        print(f"  {k:28s} {report[k]}")
    if args.gt_predictions:
        print(f"  gt_predictions               {report['gt_predictions']}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"  report -> {args.out}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
