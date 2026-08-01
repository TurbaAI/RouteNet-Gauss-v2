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

# Single RouteNet-Gauss training + evaluation job, parameterized for the job-matrix
# runner (run_experiments.py). Designed to be launched as its own subprocess so each
# job gets clean TF/GPU state and independent seeding. Outputs are isolated under an
# `experiment_name` (use something other than "paper_weights" to avoid clobbering the
# shipped paper checkpoints):
#
#   ckpt/<exp>/<dataset>/RouteNetGauss/<target>/seed_<seed>/   -> weights
#   normalization/<exp>/<dataset>/RouteNetGauss/<target>/seed_<seed>/z_scores.pkl
#   results/<exp>/<dataset>/RouteNetGauss/<target>/seed_<seed>/
#       history.csv       -> per-epoch loss/metrics/lr (CSVLogger)
#       metrics.json      -> config + final losses + test-set metrics
#       predictions.npz   -> y_true, y_pred on the test split (for later diffing)
#   tensorboard/<exp>/...  -> TensorBoard scalars

import argparse

# configure_gpu_env() MUST run before `import tensorflow` (it may re-exec the process
# to put the pip CUDA libs on LD_LIBRARY_PATH). No-op under CUDA_VISIBLE_DEVICES=-1.
from gpu_setup import configure_gpu_env

configure_gpu_env()

import json
import math
import os
import time
from random import seed as py_seed

import numpy as np
import tensorflow as tf

from gpu_setup import enable_memory_growth
from models import RouteNetGauss
from training_lib import (
    LearningRateLogger,
    get_positional_mape,
    get_positional_r2,
    get_z_scores_dict,
)
from utils import load_dataset, prepare_targets_and_mask

PERCENTILES = ["avg", "p50", "p90", "p95", "p99"]


def build_targets(target):
    targets = [f"flow_{p}_{target}" for p in PERCENTILES]
    mask = f"flow_has_{target}"
    return targets, mask


def concatenate_ds(ds):
    """Concatenate all target labels of a dataset into a single numpy array.

    Mirrors the evaluation.ipynb helper: yields (x, y) and stacks the y's.
    """
    res = [y.numpy() for _, y in iter(ds)]
    return np.concatenate(res, axis=0)


def _finite_or_none(x):
    """JSON can't hold NaN/Inf; convert those to None so metrics.json stays valid."""
    return x if isinstance(x, (int,)) or (isinstance(x, float) and math.isfinite(x)) else None


def _metrics(y_true, y_pred):
    """MAPE (%), MAE (microseconds) and R2 — the same formulas used in evaluation.ipynb."""
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    mae_us = float(np.mean(np.abs(y_true - y_pred)) * 1e6)
    denom = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = float(1 - np.sum(np.square(y_true - y_pred)) / denom) if denom != 0 else float("nan")
    return {"mape": _finite_or_none(mape), "mae_us": _finite_or_none(mae_us), "r2": _finite_or_none(r2)}


def parse_args():
    p = argparse.ArgumentParser(description="One RouteNet-Gauss train+eval job.")
    p.add_argument("--dataset", required=True, help="e.g. mawi_pcaps, trex_multiburst")
    p.add_argument("--target", required=True, choices=["delay", "jitter"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps", type=int, default=50, help="steps_per_epoch")
    p.add_argument(
        "--shuffle-buffer",
        type=int,
        default=1000,
        help="shuffle buffer size (train.py uses 1000; smaller = faster startup for quick runs)",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=0,
        help="early-stopping patience on val_loss (0 = disabled, fixed epochs). >0 trains to "
        "convergence: stop after this many epochs without val_loss improvement, restore best.",
    )
    p.add_argument(
        "--save-best-only",
        action="store_true",
        help="only keep the best (lowest val_loss) checkpoint instead of one per epoch",
    )
    p.add_argument("--experiment-name", default="tf_baseline")
    p.add_argument("--data-path", default="data")
    p.add_argument("--use-wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility: seed all RNGs used by the pipeline.
    py_seed(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    enable_memory_growth(tf)
    on_gpu = len(tf.config.list_physical_devices("GPU")) > 0

    targets, mask = build_targets(args.target)
    experiment_path = os.path.join(
        args.experiment_name,
        args.dataset,
        RouteNetGauss.__name__,
        args.target,
        f"seed_{args.seed}",
    )
    ckpt_dir = os.path.join("ckpt", experiment_path)
    results_dir = os.path.join("results", experiment_path)
    os.makedirs(results_dir, exist_ok=True)

    print(f"[{experiment_path}] backend={'GPU' if on_gpu else 'CPU'} "
          f"epochs={args.epochs} steps={args.steps}")

    # --- Data ---
    ds_train = (
        load_dataset(f"{args.dataset}/training", data_path=args.data_path)
        .map(prepare_targets_and_mask(targets, mask))
        .shuffle(args.shuffle_buffer, seed=args.seed, reshuffle_each_iteration=True)
        .repeat()
    )
    ds_val = load_dataset(f"{args.dataset}/validation", data_path=args.data_path).map(
        prepare_targets_and_mask(targets, mask)
    )

    z_scores = get_z_scores_dict(
        ds_train,
        RouteNetGauss.z_scores_fields,
        summarize=500,
        flatten=True,
        store_res_path=os.path.join("normalization", experiment_path, "z_scores.pkl"),
        check_existing=True,
    )

    # --- Model ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    loss = tf.keras.losses.MeanAbsolutePercentageError()
    model = RouteNetGauss(
        output_dim=len(targets),
        mask_field=mask,
        use_trans_delay=args.target == "delay",
        z_scores=z_scores,
    )
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[get_positional_mape(i, n) for i, n in enumerate(PERCENTILES)]
        + [get_positional_r2(i, n) for i, n in enumerate(PERCENTILES)],
    )

    # --- Callbacks (log lr / loss / metrics to CSV + TensorBoard) ---
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.4f}"),
            verbose=1,
            save_weights_only=True,
            save_best_only=args.save_best_only,
            monitor="val_loss",
            mode="min",
            save_freq="epoch",
        ),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join("tensorboard", experiment_path)),
        tf.keras.callbacks.CSVLogger(os.path.join(results_dir, "history.csv")),
        LearningRateLogger(),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=10, verbose=1, cooldown=3, mode="min", monitor="loss"
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    # Train-to-convergence: stop when val_loss stops improving and restore the best weights.
    # Mirrors the early_stop config in train.py (which was defined but never wired into fit).
    if args.patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
                start_from_epoch=4,
            )
        )

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            from wandb.integration.keras import WandbMetricsLogger

            wandb_run = wandb.init(
                project="routenet-gauss-tf-baseline",
                name=experiment_path.replace(os.sep, "__"),
                config=vars(args),
                reinit=True,
            )
            callbacks.append(WandbMetricsLogger())
        except Exception as e:  # not installed / not logged in -> keep going without it
            print(f"[wandb] disabled: {e}")

    # --- Train ---
    t0 = time.time()
    history = model.fit(
        ds_train,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        validation_data=ds_val,
        callbacks=callbacks,
    )
    train_seconds = time.time() - t0

    # --- Evaluate on the test split ---
    ds_test = load_dataset(f"{args.dataset}/test", data_path=args.data_path).map(
        prepare_targets_and_mask(targets, mask)
    )
    model.inference_mode = True  # clamp predictions >= 0, matching evaluation.ipynb
    y_pred = model.predict(ds_test)
    y_true = concatenate_ds(ds_test)
    n = min(len(y_true), len(y_pred))  # defensive alignment
    y_true, y_pred = y_true[:n], y_pred[:n]

    test_overall = _metrics(y_true.flatten(), y_pred.flatten())
    test_per_percentile = {
        name: _metrics(y_true[:, i], y_pred[:, i]) for i, name in enumerate(PERCENTILES)
    }

    metrics_out = {
        "dataset": args.dataset,
        "target": args.target,
        "seed": args.seed,
        "epochs": args.epochs,
        "epochs_run": len(history.history["loss"]),  # < epochs if early-stopped
        "patience": args.patience,
        "steps_per_epoch": args.steps,
        "backend": "gpu" if on_gpu else "cpu",
        "train_seconds": round(train_seconds, 2),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        # best (restored) val_loss — the weights actually used for the test eval below
        "best_val_loss": float(min(history.history["val_loss"])),
        "n_test_predictions": int(n),
        "test_overall": test_overall,
        "test_per_percentile": test_per_percentile,
        "experiment_path": experiment_path,
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)
    np.savez_compressed(
        os.path.join(results_dir, "predictions.npz"), y_true=y_true, y_pred=y_pred
    )

    if wandb_run is not None:
        wandb_run.summary.update({"test_overall": test_overall, "train_seconds": train_seconds})
        wandb_run.finish()

    print(f"[{experiment_path}] DONE in {train_seconds:.1f}s | "
          f"val_loss={metrics_out['final_val_loss']:.4f} | "
          f"test MAPE={test_overall['mape']} R2={test_overall['r2']} "
          f"(n={n}) -> {results_dir}/metrics.json")


if __name__ == "__main__":
    main()
