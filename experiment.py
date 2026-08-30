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
#
# PyTorch translation. Every original TensorFlow line is kept as a `#TF:` comment with its
# translation below. Keras' `model.compile/fit` and its callbacks have no PyTorch counterpart:
# `training_lib.fit` writes the loop out and reproduces Keras' arithmetic and bookkeeping
# (verified against the TF replay recordings, see PYTORCH_PORT.md):
#   - loss = MeanAbsolutePercentageError with Keras' 1e-7 floor; Adam(lr, clipnorm) with Keras'
#     per-tensor clipping and epsilon placement (training_lib.KerasAdam);
#   - per-epoch `loss`/`val_loss` are means over steps WEIGHTED by the number of predictions of
#     each scenario (Keras weights its loss Mean by the batch dimension), metrics are plain means
#     over steps; history.csv has the same columns in the same order as the TF one;
#   - callbacks run in Keras' order with Keras' exact logic: ModelCheckpoint, TensorBoard,
#     CSVLogger, LearningRateLogger, ReduceLROnPlateau, TerminateOnNaN, [EarlyStopping], [W&B].
# PyTorch-only additions (all opt-in flags): --device/--threads/--nondeterministic/--allow-tf32,
# --init {torch,keras}, exact-replay inputs (--init-weights, --sample-order, --z-scores or
# --replay-from), per-step loss logging (step_losses.csv), a materialised training order
# (sample_order_used.npy) and --resume (fit checkpoints its full state every epoch).

import argparse

# configure_gpu_env() MUST run before `import tensorflow` (it may re-exec the process
# to put the pip CUDA libs on LD_LIBRARY_PATH). No-op under CUDA_VISIBLE_DEVICES=-1.
# PyTorch: configure_gpu_env() is a no-op (torch bundles its CUDA libs); kept for structure.
from gpu_setup import configure_gpu_env

configure_gpu_env()

import json
import math
import os
import pickle
import time
from random import seed as py_seed

import numpy as np
#TF: import tensorflow as tf
import torch

from gpu_setup import enable_memory_growth
from models import RouteNetGauss
from training_lib import (
    LearningRateLogger,
    get_positional_mape,
    get_positional_r2,
    get_z_scores_dict,
)
# PyTorch-only imports: Keras-exact loss / optimizer / callbacks / training loop (training_lib.py)
from training_lib import (
    KerasAdam,
    KerasEarlyStopping,
    KerasModelCheckpoint,
    KerasReduceLROnPlateau,
    clip_by_norm_,
    fit,
    keras_mape_loss,
    load_resume_state,
)
from torch_ragged import sample_to_device
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


def load_init_weights(model, path):
    """--init-weights: a torch state_dict (.pt) or a TF init_weights.npz from the replay recorder."""
    if path.endswith(".npz"):
        from convert_tf_checkpoint import load_tf_arrays, tf_arrays_to_state_dict

        sd = tf_arrays_to_state_dict(load_tf_arrays(path))
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("z_") for k in missing), (missing, unexpected)


def configure_torch(args):
    """PyTorch-only: device, threads, determinism and TF32 policy (see PYTORCH_PORT.md)."""
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    deterministic = not args.nondeterministic
    if deterministic:
        torch.use_deterministic_algorithms(True)
    # Ampere+ GPUs run float32 matmuls/RNN kernels in TF32 (10-bit mantissa) unless told
    # otherwise, which breaks float32 parity with TensorFlow; off unless --allow-tf32.
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    return device, deterministic


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
    #TF: p.add_argument("--experiment-name", default="tf_baseline")
    p.add_argument("--experiment-name", default="torch_baseline")
    #TF: p.add_argument("--data-path", default="data")
    p.add_argument("--data-path", default="data_torch")
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", default="routenet-gauss-tf-baseline")
    # ---- PyTorch-only options ----
    g = p.add_argument_group("PyTorch options")
    g.add_argument("--device", default="auto", help="cpu | cuda | auto (cuda if available)")
    g.add_argument("--threads", type=int, default=1,
                   help="torch CPU threads. Oversubscribing the cores with several jobs is 10-100x slower than 1 thread each.")
    g.add_argument("--nondeterministic", action="store_true",
                   help="do NOT force torch.use_deterministic_algorithms(True) (faster on GPU, not bit-reproducible)")
    g.add_argument("--allow-tf32", action="store_true",
                   help="allow TensorFloat-32 kernels on Ampere+ GPUs (default off: 10-bit-mantissa arithmetic "
                        "breaks float32 parity with TensorFlow)")
    g.add_argument("--init", choices=["torch", "keras"], default="torch",
                   help="parameter initialisation: torch default, or Keras' glorot/orthogonal/zeros (TF comparisons)")
    g.add_argument("--init-weights", default=None, help="load initial weights (.pt state_dict or TF init_weights.npz)")
    g.add_argument("--sample-order", default=None,
                   help="sample_order.npy: feed the training scenarios in exactly this sample_idx order (exact TF replay)")
    g.add_argument("--z-scores", default=None, help="z_scores.pkl to use instead of computing them (exact TF replay)")
    g.add_argument("--replay-from", default=None,
                   help="a tensorflow_version_gt/replay/<cell> dir: sets --init-weights, --sample-order and --z-scores")
    g.add_argument("--optimizer", choices=["keras_adam", "torch_adam"], default="keras_adam",
                   help="keras_adam reproduces tf.keras Adam arithmetic (eps 1e-7, per-tensor clipnorm); "
                        "torch_adam = torch.optim.Adam(eps=1e-7) + per-tensor clipnorm")
    g.add_argument("--resume", action="store_true", help="continue from results/.../resume.pt if it exists")
    g.add_argument("--no-step-log", action="store_true", help="do not write the per-step step_losses.csv")
    return p.parse_args()


def main():
    args = parse_args()
    if args.replay_from:
        args.init_weights = args.init_weights or os.path.join(args.replay_from, "init_weights.npz")
        args.sample_order = args.sample_order or os.path.join(args.replay_from, "sample_order.npy")
        args.z_scores = args.z_scores or os.path.join(args.replay_from, "z_scores.pkl")
        if args.init_weights.endswith("init_weights.npz"):
            args.init = "keras"  # TF weights are Keras-initialised by construction

    # Reproducibility: seed all RNGs used by the pipeline.
    py_seed(args.seed)
    #TF: tf.random.set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #TF: enable_memory_growth(tf)
    enable_memory_growth(torch)
    #TF: on_gpu = len(tf.config.list_physical_devices("GPU")) > 0
    device, deterministic = configure_torch(args)
    on_gpu = device.type == "cuda"

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
    print(f"[{experiment_path}] torch {torch.__version__} | device={device} | threads={args.threads} | "
          f"deterministic={deterministic} | tf32={args.allow_tf32} | init={args.init} | optimizer={args.optimizer} | "
          f"init_weights={args.init_weights} | sample_order={args.sample_order} | z_scores={args.z_scores}")

    # --- Data ---
    ds_train_raw = load_dataset(f"{args.dataset}/training", data_path=args.data_path)
    ds_train = (
        ds_train_raw
        .map(prepare_targets_and_mask(targets, mask))
        .shuffle(args.shuffle_buffer, seed=args.seed, reshuffle_each_iteration=True)
        .repeat()
    )
    ds_val = load_dataset(f"{args.dataset}/validation", data_path=args.data_path).map(
        prepare_targets_and_mask(targets, mask)
    )

    z_path = os.path.join("normalization", experiment_path, "z_scores.pkl")
    if args.z_scores:
        # Exact replay: the TF z-scores were computed from TF's first shuffled pass, which a
        # torch shuffle cannot reproduce; use the recorded ones (they equal the GT's bit-for-bit,
        # see tensorflow_version_gt/replay/*/replay_check.json).
        z_scores = pickle.load(open(args.z_scores, "rb"))
        os.makedirs(os.path.dirname(z_path), exist_ok=True)
        pickle.dump(z_scores, open(z_path, "wb"))
    else:
        z_scores = get_z_scores_dict(
            ds_train,
            RouteNetGauss.z_scores_fields,
            summarize=500,
            flatten=True,
            store_res_path=z_path,
            check_existing=True,
        )

    # --- Model ---
    #TF: optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    #TF: loss = tf.keras.losses.MeanAbsolutePercentageError()
    model = RouteNetGauss(
        output_dim=len(targets),
        mask_field=mask,
        use_trans_delay=args.target == "delay",
        z_scores=z_scores,
        init=args.init,
    )
    if args.init_weights:
        load_init_weights(model, args.init_weights)
    model.to(device)
    if args.optimizer == "keras_adam":
        optimizer = KerasAdam(model.parameters(), lr=0.001, clipnorm=1.0)
        clip_grads = None  # KerasAdam clips per tensor itself
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-7)
        clip_grads = lambda grads: clip_by_norm_(grads, 1.0)
    loss = keras_mape_loss
    #TF: model.compile(
    #TF:     optimizer=optimizer,
    #TF:     loss=loss,
    #TF:     metrics=[get_positional_mape(i, n) for i, n in enumerate(PERCENTILES)]
    #TF:     + [get_positional_r2(i, n) for i, n in enumerate(PERCENTILES)],
    #TF: )
    metrics = [get_positional_mape(i, n) for i, n in enumerate(PERCENTILES)] + [
        get_positional_r2(i, n) for i, n in enumerate(PERCENTILES)
    ]

    # --- Callbacks (log lr / loss / metrics to CSV + TensorBoard) ---
    #TF: callbacks = [
    #TF:     tf.keras.callbacks.ModelCheckpoint(
    #TF:         filepath=os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.4f}"),
    #TF:         verbose=1,
    #TF:         save_weights_only=True,
    #TF:         save_best_only=args.save_best_only,
    #TF:         monitor="val_loss",
    #TF:         mode="min",
    #TF:         save_freq="epoch",
    #TF:     ),
    #TF:     tf.keras.callbacks.TensorBoard(log_dir=os.path.join("tensorboard", experiment_path)),
    #TF:     tf.keras.callbacks.CSVLogger(os.path.join(results_dir, "history.csv")),
    #TF:     LearningRateLogger(),
    #TF:     tf.keras.callbacks.ReduceLROnPlateau(
    #TF:         factor=0.5, patience=10, verbose=1, cooldown=3, mode="min", monitor="loss"
    #TF:     ),
    #TF:     tf.keras.callbacks.TerminateOnNaN(),
    #TF: ]
    from torch.utils.tensorboard import SummaryWriter

    checkpoint_cb = KerasModelCheckpoint(ckpt_dir, save_best_only=args.save_best_only, verbose=1)
    tb_writer = SummaryWriter(log_dir=os.path.join("tensorboard", experiment_path))
    history_path = os.path.join(results_dir, "history.csv")
    lr_logger = LearningRateLogger()
    reduce_lr_cb = KerasReduceLROnPlateau(factor=0.5, patience=10, verbose=1, cooldown=3, monitor="loss")
    # (TerminateOnNaN is built into training_lib.fit)
    # Train-to-convergence: stop when val_loss stops improving and restore the best weights.
    # Mirrors the early_stop config in train.py (which was defined but never wired into fit).
    early_stop_cb = None
    if args.patience > 0:
        #TF: callbacks.append(
        #TF:     tf.keras.callbacks.EarlyStopping(
        #TF:         monitor="val_loss",
        #TF:         patience=args.patience,
        #TF:         restore_best_weights=True,
        #TF:         start_from_epoch=4,
        #TF:     )
        #TF: )
        early_stop_cb = KerasEarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True, start_from_epoch=4
        )

    # --- Training order (PyTorch-only) ---
    # The scenarios a run will see are materialised up front: either the recorded TF order
    # (exact replay) or the ListDataset's tf.data-style shuffled stream. Writing it to disk
    # makes the run auditable and, together with resume.pt, resumable after a crash/reboot.
    total_steps = args.epochs * args.steps
    if args.sample_order:
        train_order = np.load(args.sample_order).astype(np.int64)
        assert len(train_order) >= total_steps, f"{args.sample_order} has {len(train_order)} steps < {total_steps}"
        train_order = train_order[:total_steps]
    else:
        train_order = ds_train.index_order(total_steps)
    np.save(os.path.join(results_dir, "sample_order_used.npy"), train_order)
    by_idx = {int(x["sample_idx"]): (x, y) for x, y in ds_train_raw}
    map_fn = prepare_targets_and_mask(targets, mask)

    resume_path = os.path.join(results_dir, "resume.pt")
    resume_state = load_resume_state(resume_path) if args.resume else None
    if resume_state:
        print(f"[{experiment_path}] resuming from {resume_path} (epoch {resume_state['epoch'] + 1})")

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            #TF: from wandb.integration.keras import WandbMetricsLogger

            wandb_run = wandb.init(
                #TF: project="routenet-gauss-tf-baseline",
                project=args.wandb_project,
                #TF: name=experiment_path.replace(os.sep, "__"),
                name="torch__" + experiment_path.replace(os.sep, "__"),
                #TF: config=vars(args),
                config=dict(vars(args), framework="torch", torch_version=torch.__version__, deterministic=deterministic),
                reinit=True,
                id=resume_state.get("wandb_id") if resume_state else None,
                resume="allow" if resume_state else None,
            )
            #TF: callbacks.append(WandbMetricsLogger())
        except Exception as e:  # not installed / not logged in -> keep going without it
            print(f"[wandb] disabled: {e}")

    # --- Train ---
    #TF: t0 = time.time()
    #TF: history = model.fit(
    #TF:     ds_train,
    #TF:     epochs=args.epochs,
    #TF:     steps_per_epoch=args.steps,
    #TF:     validation_data=ds_val,
    #TF:     callbacks=callbacks,
    #TF: )
    #TF: train_seconds = time.time() - t0
    history_rows, train_seconds, step_seconds = fit(
        model, optimizer, loss, metrics,
        train_order=train_order, by_idx=by_idx, map_fn=map_fn, ds_val=ds_val,
        epochs=args.epochs, steps_per_epoch=args.steps, device=device,
        checkpoint_cb=checkpoint_cb, reduce_lr_cb=reduce_lr_cb, early_stop_cb=early_stop_cb, lr_logger=lr_logger,
        tb_writer=tb_writer, history_path=history_path,
        step_log_path=None if args.no_step_log else os.path.join(results_dir, "step_losses.csv"),
        resume_path=resume_path, resume_state=resume_state, wandb_run=wandb_run, clip_grads=clip_grads,
    )
    tb_writer.close()
    history = {"loss": [r["loss"] for r in history_rows], "val_loss": [r["val_loss"] for r in history_rows]}

    # --- Evaluate on the test split ---
    ds_test = load_dataset(f"{args.dataset}/test", data_path=args.data_path).map(
        prepare_targets_and_mask(targets, mask)
    )
    model.inference_mode = True  # clamp predictions >= 0, matching evaluation.ipynb
    #TF: y_pred = model.predict(ds_test)
    model.eval()
    t_eval = time.time()
    with torch.no_grad():
        y_pred = np.concatenate([model(sample_to_device(x, device)).cpu().numpy() for x, _ in ds_test], axis=0)
    eval_seconds = time.time() - t_eval
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
        "epochs_run": len(history["loss"]),  # < epochs if early-stopped
        "patience": args.patience,
        "steps_per_epoch": args.steps,
        "backend": "gpu" if on_gpu else "cpu",
        "train_seconds": round(train_seconds, 2),
        "final_train_loss": float(history["loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        # best (restored) val_loss — the weights actually used for the test eval below
        "best_val_loss": float(min(history["val_loss"])),
        "n_test_predictions": int(n),
        "test_overall": test_overall,
        "test_per_percentile": test_per_percentile,
        "experiment_path": experiment_path,
        # ---- PyTorch-only provenance ----
        "framework": "torch",
        "torch_version": torch.__version__,
        "device": str(device),
        "threads": args.threads,
        "deterministic": deterministic,
        "tf32": bool(args.allow_tf32),
        "init": args.init,
        "init_weights": args.init_weights,
        "sample_order": args.sample_order or "native (ListDataset shuffle)",
        "z_scores_source": args.z_scores or z_path,
        "optimizer": args.optimizer,
        "shuffle_buffer": args.shuffle_buffer,
        "resumed": bool(resume_state),
        "seconds_per_train_step_mean": float(np.mean(step_seconds)) if step_seconds else None,
        "test_eval_seconds": round(eval_seconds, 2),
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
