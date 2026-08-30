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

# TF "replay recorder" for the PyTorch parity work.
#
# The frozen TensorFlow ground truth in tensorflow_version_gt/ was produced by
# experiment.py @ 2e30d5d. Those runs depended on three things they never wrote down:
#
#   1. the exact order in which training scenarios were fed (tf.data shuffle RNG),
#   2. the exact initial weights (Keras initializer RNG),
#   3. the z-score normalisation constants (saved, but a consequence of 1.).
#
# TensorFlow regenerates all three bit-exactly when the same code path runs with the
# same seeds (verified: re-running one epoch reproduces the GT epoch-1 checkpoint with
# 34/34 weight tensors bit-identical). This script re-executes the GT setup for one
# "cell" (= dataset x target x seed) and RECORDS what the GT never saved, so that the
# PyTorch port can start from exactly the same state and see exactly the same data:
#
#   --mode order : rebuild the data pipeline (including the "z-score step", which
#                  consumes the first shuffled pass and therefore shifts the order that
#                  model.fit sees) and write the sample_idx fed at every training step
#                  -> sample_order.npy (int32, --order-steps entries).
#   --mode train : rebuild pipeline + model exactly as experiment.py did, dump the
#                  initial weights (init_weights.npz), then re-train the quick 5x50
#                  config with a per-step logger (step_losses_5x50.csv) and PROVE the
#                  replay is exact by comparing every per-epoch checkpoint, history.csv
#                  and the test-set predictions bit-for-bit against the GT
#                  -> replay_check.json.
#
# Both modes assert that the recomputed z-scores equal the GT z_scores.pkl ("fingerprint"
# of the first 500 shuffled scenarios). Each mode MUST run in its own fresh process
# (RNG state), which `--all` takes care of.
#
# Run from the repo root, in the same TF environment that produced the GT (conda env RG):
#   python tf_reference/replay_tf_run.py --all --concurrency 3
#   python tf_reference/replay_tf_run.py --dataset trex_multiburst --target delay --seed 1 --mode both
#
# Outputs: tensorflow_version_gt/replay/<dataset>/RouteNetGauss/<target>/seed_<n>/

import argparse
import os
import sys

# CPU only, exactly like the GT runs (experiment.py was launched with CUDA_VISIBLE_DEVICES=-1).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Make the repo root importable so that `tf_reference.*` (the frozen TF originals) is
# used, never the translated top-level modules.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import csv
import glob
import itertools
import json
import pickle
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import seed as py_seed

import numpy as np
import tensorflow as tf

from tf_reference.models import RouteNetGauss
from tf_reference.training_lib import (
    LearningRateLogger,
    get_positional_mape,
    get_positional_r2,
    get_z_scores_dict,
)
from tf_reference.utils import load_dataset, prepare_targets_and_mask

PERCENTILES = ["avg", "p50", "p90", "p95", "p99"]
GT_DATASETS = ["mawi_pcaps", "trex_multiburst"]
GT_TARGETS = ["delay", "jitter"]
GT_SEEDS = [1, 2]

# Names of the 34 weight tensors as they appear in the TF checkpoints (without the
# "/.ATTRIBUTES/VARIABLE_VALUE" suffix). init_weights.npz uses the same keys so the
# TF->PyTorch weight converter can treat checkpoints and init dumps identically.
GRU_ATTRS = ["flow_update", "link_update", "queue_update", "node_update"]
MLP_ATTRS = ["flow_embedding", "queue_embedding", "link_embedding", "node_embedding", "readout_path"]


def build_targets(target):
    # verbatim: experiment.py
    targets = [f"flow_{p}_{target}" for p in PERCENTILES]
    mask = f"flow_has_{target}"
    return targets, mask


def concatenate_ds(ds):
    # verbatim: experiment.py
    res = [y.numpy() for _, y in iter(ds)]
    return np.concatenate(res, axis=0)


def cell_dir(root, dataset, target, seed):
    return os.path.join(root, dataset, RouteNetGauss.__name__, target, f"seed_{seed}")


def collect_weights(model):
    """All 34 weight tensors of a *built* RouteNetGauss, keyed like the TF checkpoint."""
    out = {}
    for attr in GRU_ATTRS:
        cell = getattr(model, attr)
        out[f"{attr}/kernel"] = cell.kernel.numpy()
        out[f"{attr}/recurrent_kernel"] = cell.recurrent_kernel.numpy()
        out[f"{attr}/bias"] = cell.bias.numpy()
    for attr in MLP_ATTRS:
        layers = [l for l in getattr(model, attr).layers if l.weights]
        for i, layer in enumerate(layers):
            out[f"{attr}/layer_with_weights-{i}/kernel"] = layer.kernel.numpy()
            out[f"{attr}/layer_with_weights-{i}/bias"] = layer.bias.numpy()
    return out


def checkpoint_weight_names(ckpt_prefix):
    names = []
    for name, _ in tf.train.list_variables(ckpt_prefix):
        if "optimizer" in name.lower() or "save_counter" in name or "_CHECKPOINTABLE" in name:
            continue
        if not name.endswith("/.ATTRIBUTES/VARIABLE_VALUE"):
            continue
        short = name[: -len("/.ATTRIBUTES/VARIABLE_VALUE")]
        # skip this script's own recording variables if present in a replay checkpoint
        if short.startswith("_rec_"):
            continue
        names.append(short)
    return sorted(names)


def compare_checkpoints(replay_prefix, gt_prefix):
    """Bit-for-bit comparison of every (non-optimizer) weight tensor in two TF checkpoints."""
    r, g = tf.train.load_checkpoint(replay_prefix), tf.train.load_checkpoint(gt_prefix)
    names = checkpoint_weight_names(gt_prefix)
    n_exact, worst_rel, diffs = 0, 0.0, {}
    for short in names:
        full = short + "/.ATTRIBUTES/VARIABLE_VALUE"
        a, b = r.get_tensor(full), g.get_tensor(full)
        if a.shape == b.shape and np.array_equal(a, b):
            n_exact += 1
            continue
        d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        rel = d / (float(np.max(np.abs(b))) + 1e-30)
        worst_rel = max(worst_rel, rel)
        diffs[short] = {"max_abs_diff": d, "max_rel_diff": rel}
    return {
        "n_tensors": len(names),
        "n_bit_identical": n_exact,
        "all_bit_identical": n_exact == len(names),
        "worst_rel_diff": worst_rel,
        "diffs": diffs,
    }


def z_scores_match(z, gt_path):
    if not os.path.exists(gt_path):
        return None, None
    gt = pickle.load(open(gt_path, "rb"))
    ok = set(z) == set(gt) and all(
        np.float32(z[k][0]) == np.float32(gt[k][0]) and np.float32(z[k][1]) == np.float32(gt[k][1])
        for k in gt
    )
    return bool(ok), {k: [float(v[0]), float(v[1])] for k, v in gt.items()}


class RecordingRouteNetGauss(RouteNetGauss):
    """RouteNetGauss whose train_step additionally records the per-step loss and the
    sample_idx of the scenario it was computed on.

    train_step is Keras 2.15's Model.train_step VERBATIM plus the two `assign` lines
    (recording into non-trainable variables changes no arithmetic). Keras' `logs["loss"]`
    is a running mean over the epoch, so the exact per-step value is only available this
    way.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rec_loss = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="rec_loss")
        self._rec_sample_idx = tf.Variable(-1, trainable=False, dtype=tf.int32, name="rec_sample_idx")

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        # --- recording (added) ---
        self._rec_loss.assign(loss)
        self._rec_sample_idx.assign(x["sample_idx"])
        return self.compute_metrics(x, y, y_pred, sample_weight)


class StepRecorder(tf.keras.callbacks.Callback):
    """Writes one CSV row per training step: exact loss, Keras running-mean loss, lr,
    sample_idx."""

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.rows = []
        self.epoch = 0
        self.global_step = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        lr = self.model.optimizer.learning_rate
        if callable(lr):
            lr = lr(self.model.optimizer.iterations)
        self.rows.append(
            {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "step_in_epoch": batch,
                "sample_idx": int(self.model._rec_sample_idx.numpy()),
                "loss": float(self.model._rec_loss.numpy()),
                "running_mean_loss": float(logs["loss"]) if logs and "loss" in logs else float("nan"),
                "lr": float(tf.keras.backend.get_value(lr)),
            }
        )
        self.global_step += 1

    def on_train_end(self, logs=None):
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


def build_pipeline(args, targets, mask, store_z_path):
    """Steps 2-9 of experiment.py main(), verbatim in order and arguments."""
    # Reproducibility: seed all RNGs used by the pipeline.
    py_seed(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # enable_memory_growth(tf) + on_gpu probe: no GPUs are visible, both are no-ops, but
    # they are kept so the sequence of TF calls before model construction is unchanged.
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    on_gpu = len(tf.config.list_physical_devices("GPU")) > 0
    assert not on_gpu, "replay must run on CPU like the GT"

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

    # The "z-score step": mean/std of flow_traffic and flow_packets over the first 500
    # scenarios of the shuffled stream. This consumes the FIRST shuffled pass; model.fit
    # then gets the second one. check_existing=False forces recomputation so the
    # fingerprint check below is meaningful even on re-runs.
    z_scores = get_z_scores_dict(
        ds_train,
        RouteNetGauss.z_scores_fields,
        summarize=500,
        flatten=True,
        store_res_path=store_z_path,
        check_existing=False,
    )
    return ds_train, ds_val, z_scores


def run_order(args, out_dir, log):
    targets, mask = build_targets(args.target)
    # Full pipeline, exactly as experiment.py: this recomputes the z-scores (fingerprint check)
    # and, as a side effect, consumes iterator #1 of `ds_train` just like the GT did.
    ds_train, _, z_scores = build_pipeline(args, targets, mask, os.path.join(out_dir, "z_scores.pkl"))
    gt_z = os.path.join(args.gt_root, "normalization", args.dataset, "RouteNetGauss", args.target, f"seed_{args.seed}", "z_scores.pkl")
    z_ok, gt_vals = z_scores_match(z_scores, gt_z)
    log(f"z-score fingerprint vs {gt_z}: {z_ok}  (recomputed={ {k: [float(v[0]), float(v[1])] for k, v in z_scores.items()} })")
    if z_ok is False:
        raise SystemExit("z-score fingerprint mismatch: the replayed stream is NOT the GT stream")

    # Recording 150k steps through the full pipeline is slow (every step decompresses a whole
    # scenario, ~1.7 MB for mawi). The shuffle permutation depends only on the shuffle RNG
    # (seed, buffer size, number of upstream elements, iterator/epoch bookkeeping), NOT on the
    # element contents, so a light replica that projects each scenario to its sample_idx and
    # caches the (tiny) result yields the identical order at RAM speed. Same seeds, same
    # position of shuffle/repeat in the chain, same iterator-creation sequence (iterator #1 is
    # created and 502 elements consumed, mirroring get_z_scores_dict; iterator #2 is recorded).
    # The claim is verified below against the full pipeline for the first --check-steps steps,
    # and again by --mode train, which compares the sample_idx fed at every training step.
    ds_light = (
        load_dataset(f"{args.dataset}/training", data_path=args.data_path)
        .map(lambda x, y: x["sample_idx"])
        .cache()
        .shuffle(args.shuffle_buffer, seed=args.seed, reshuffle_each_iteration=True)
        .repeat()
    )
    it1 = iter(ds_light)
    for _ in range(502):  # get_z_scores_dict pulls 1 + 501 elements (summarize=500)
        next(it1)
    t0 = time.time()
    it2 = iter(ds_light)
    order = np.empty(args.order_steps, dtype=np.int32)
    for i in range(args.order_steps):
        order[i] = int(next(it2))
        if (i + 1) % 50000 == 0:
            log(f"  order: {i + 1}/{args.order_steps} steps ({time.time() - t0:.0f}s)")
    order_seconds = time.time() - t0

    # Cross-check against the full pipeline's iterator #2 (the one model.fit would consume).
    it_heavy = iter(ds_train)
    heavy = [int(next(it_heavy)[0]["sample_idx"]) for _ in range(args.check_steps)]
    light_matches_heavy = heavy == order[: args.check_steps].tolist()
    log(f"light pipeline == full pipeline for the first {args.check_steps} steps: {light_matches_heavy}")
    if not light_matches_heavy:
        raise SystemExit("order recorded through the light pipeline differs from the full pipeline")

    np.save(os.path.join(out_dir, "sample_order.npy"), order)
    log(f"sample_order.npy written: {args.order_steps} steps in {order_seconds:.0f}s; first 10 = {order[:10].tolist()}; "
        f"distinct scenarios = {len(np.unique(order))}")
    return {"z_match": z_ok, "order_steps": int(args.order_steps), "order_seconds": round(order_seconds, 1),
            "light_matches_heavy_prefix": light_matches_heavy, "check_steps": int(args.check_steps)}


def run_train(args, out_dir, log):
    targets, mask = build_targets(args.target)
    tmp_dir = tempfile.mkdtemp(prefix=f"rg_replay_{args.dataset}_{args.target}_{args.seed}_", dir=args.tmp_root)
    ckpt_dir = os.path.join(tmp_dir, "ckpt")
    log(f"temporary checkpoint/tensorboard dir: {tmp_dir}")

    ds_train, ds_val, z_scores = build_pipeline(args, targets, mask, os.path.join(out_dir, "z_scores.pkl"))
    gt_cell = cell_dir(os.path.join(args.gt_root, "results"), args.dataset, args.target, args.seed)
    gt_ckpt_dir = cell_dir(os.path.join(args.gt_root, "ckpt"), args.dataset, args.target, args.seed)
    gt_z = os.path.join(cell_dir(os.path.join(args.gt_root, "normalization"), args.dataset, args.target, args.seed), "z_scores.pkl")
    z_ok, _ = z_scores_match(z_scores, gt_z)
    log(f"z-score fingerprint vs GT: {z_ok}")
    if z_ok is False:
        raise SystemExit("z-score fingerprint mismatch: the replayed stream is NOT the GT stream")

    # --- Model (verbatim experiment.py, with the recording subclass) ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    loss = tf.keras.losses.MeanAbsolutePercentageError()
    model = RecordingRouteNetGauss(
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

    # --- Initial weights ---
    # Keras creates the GRU weights lazily on the first call. In the GT run that call was
    # the first training step; the initializer draws only depend on the RNG state and the
    # build order, not on the input, so building here on a VALIDATION scenario yields the
    # same numbers. (Pulling from ds_train would create a new shuffle iterator and shift
    # the order model.fit sees — never do that.) Exactness is proven below by the
    # checkpoint comparison.
    x_val, _ = next(iter(ds_val))
    model(x_val)
    init = collect_weights(model)
    np.savez(os.path.join(out_dir, "init_weights.npz"), **init)
    log(f"init_weights.npz written: {len(init)} tensors, {sum(v.size for v in init.values())} parameters")

    # --- Callbacks (same list and order as experiment.py; TensorBoard -> tmp) ---
    step_csv = os.path.join(out_dir, f"step_losses_{args.epochs}x{args.steps}.csv")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.4f}"),
            verbose=1,
            save_weights_only=True,
            save_best_only=False,
            monitor="val_loss",
            mode="min",
            save_freq="epoch",
        ),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(tmp_dir, "tensorboard")),
        tf.keras.callbacks.CSVLogger(os.path.join(out_dir, "history.csv")),
        LearningRateLogger(),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=10, verbose=1, cooldown=3, mode="min", monitor="loss"
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        StepRecorder(step_csv),
    ]
    wandb_run = None
    if args.use_wandb:
        import wandb
        from wandb.integration.keras import WandbMetricsLogger

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"replay__{args.dataset}__{args.target}__seed_{args.seed}",
            config=vars(args),
            reinit=True,
        )
        callbacks.append(WandbMetricsLogger())

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
    log(f"training done in {train_seconds:.0f}s")

    # --- Test-set predictions (verbatim experiment.py) ---
    ds_test = load_dataset(f"{args.dataset}/test", data_path=args.data_path).map(
        prepare_targets_and_mask(targets, mask)
    )
    model.inference_mode = True
    y_pred = model.predict(ds_test)
    y_true = concatenate_ds(ds_test)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    # --- Checks against the GT ---
    checks = {
        "dataset": args.dataset,
        "target": args.target,
        "seed": args.seed,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps,
        "shuffle_buffer": args.shuffle_buffer,
        "use_wandb": bool(args.use_wandb),
        "train_seconds": round(train_seconds, 1),
        "z_match": z_ok,
        "checkpoints": {},
    }
    all_ok = z_ok is not False
    for epoch in range(1, args.epochs + 1):
        rep = glob.glob(os.path.join(ckpt_dir, f"{epoch:02d}-*.index"))
        gt = glob.glob(os.path.join(gt_ckpt_dir, f"{epoch:02d}-*.index"))
        entry = {"replay_name": None, "gt_name": None}
        if rep:
            entry["replay_name"] = os.path.basename(rep[0])[: -len(".index")]
        if gt:
            entry["gt_name"] = os.path.basename(gt[0])[: -len(".index")]
        if rep and gt:
            entry["same_filename"] = entry["replay_name"] == entry["gt_name"]
            entry.update(compare_checkpoints(rep[0][: -len(".index")], gt[0][: -len(".index")]))
            all_ok = all_ok and entry["all_bit_identical"]
            log(f"epoch {epoch}: replay {entry['replay_name']} vs GT {entry['gt_name']} -> "
                f"{entry['n_bit_identical']}/{entry['n_tensors']} tensors bit-identical")
        else:
            log(f"epoch {epoch}: no GT checkpoint to compare (replay={bool(rep)}, gt={bool(gt)})")
        checks["checkpoints"][f"{epoch:02d}"] = entry

    # history.csv: the GT file may have MORE epochs (never fewer) if this is a shortened run
    gt_hist = os.path.join(gt_cell, "history.csv")
    if os.path.exists(gt_hist):
        with open(gt_hist) as f:
            gt_rows = list(csv.DictReader(f))
        with open(os.path.join(out_dir, "history.csv")) as f:
            rep_rows = list(csv.DictReader(f))
        same = len(rep_rows) <= len(gt_rows) and all(
            all(rr[k] == gr[k] for k in rr if k in gr) for rr, gr in zip(rep_rows, gt_rows)
        )
        checks["history_identical_text"] = bool(same)
        all_ok = all_ok and same
        log(f"history.csv identical (textually, {len(rep_rows)} epochs): {same}")

    gt_pred = os.path.join(gt_cell, "predictions.npz")
    if os.path.exists(gt_pred) and len(history.history["loss"]) == args.epochs:
        g = np.load(gt_pred)
        same_pred = g["y_pred"].shape == y_pred.shape and np.array_equal(g["y_pred"], y_pred)
        same_true = g["y_true"].shape == y_true.shape and np.array_equal(g["y_true"], y_true)
        checks["test_predictions_bit_identical"] = bool(same_pred)
        checks["test_targets_bit_identical"] = bool(same_true)
        if not same_pred and g["y_pred"].shape == y_pred.shape:
            checks["test_predictions_max_abs_diff"] = float(np.max(np.abs(g["y_pred"] - y_pred)))
        # Only demand prediction equality when the full GT config was replayed.
        if args.epochs == 5 and args.steps == 50:
            all_ok = all_ok and same_pred and same_true
        log(f"test predictions bit-identical: {same_pred} (targets: {same_true}, n={n})")

    # Per-step sample order vs the standalone order recording (if it exists).
    order_path = os.path.join(out_dir, "sample_order.npy")
    if os.path.exists(order_path):
        order = np.load(order_path)
        with open(step_csv) as f:
            fed = [int(r["sample_idx"]) for r in csv.DictReader(f)]
        prefix_ok = len(order) >= len(fed) and fed == order[: len(fed)].tolist()
        checks["fed_order_matches_sample_order_prefix"] = bool(prefix_ok)
        all_ok = all_ok and prefix_ok
        log(f"per-step sample_idx == sample_order.npy prefix: {prefix_ok}")

    checks["all_checks_passed"] = bool(all_ok)
    with open(os.path.join(out_dir, "replay_check.json"), "w") as f:
        json.dump(checks, f, indent=2)
    log(f"replay_check.json written; ALL CHECKS PASSED: {all_ok}")

    if wandb_run is not None:
        wandb_run.summary.update({"replay_all_checks_passed": all_ok})
        wandb_run.finish()
    if args.keep_tmp:
        log(f"kept temporary dir {tmp_dir}")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return checks


def run_all(args):
    """Drive order+train for every GT quick cell, each mode in its own subprocess."""
    cells = list(itertools.product(GT_DATASETS, GT_TARGETS, GT_SEEDS))

    def job(cell):
        dataset, target, seed = cell
        out_dir = cell_dir(args.out_root, dataset, target, seed)
        os.makedirs(out_dir, exist_ok=True)
        base = [sys.executable, os.path.abspath(__file__), "--dataset", dataset, "--target", target,
                "--seed", str(seed), "--order-steps", str(args.order_steps), "--epochs", str(args.epochs),
                "--steps", str(args.steps), "--shuffle-buffer", str(args.shuffle_buffer),
                "--out-root", args.out_root, "--gt-root", args.gt_root, "--data-path", args.data_path,
                "--tmp-root", args.tmp_root]
        results = {}
        for mode in ["order", "train"]:
            log_path = os.path.join(out_dir, f"replay_{mode}.log")
            with open(log_path, "w") as logf:
                proc = subprocess.run(base + ["--mode", mode], stdout=logf, stderr=subprocess.STDOUT)
            results[mode] = proc.returncode
            print(f"[replay] {'OK  ' if proc.returncode == 0 else 'FAIL'} {dataset}/{target}/seed_{seed} {mode} (rc={proc.returncode}, log={log_path})", flush=True)
            if proc.returncode != 0:
                break
        return cell, results

    print(f"[replay] {len(cells)} cells | concurrency={args.concurrency} | order_steps={args.order_steps} | {args.epochs}x{args.steps}", flush=True)
    outcomes = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(job, c) for c in cells]
        for fut in as_completed(futs):
            outcomes.append(fut.result())

    summary = {}
    for (dataset, target, seed), rcs in sorted(outcomes):
        check_path = os.path.join(cell_dir(args.out_root, dataset, target, seed), "replay_check.json")
        passed = None
        if os.path.exists(check_path):
            passed = json.load(open(check_path)).get("all_checks_passed")
        summary[f"{dataset}/{target}/seed_{seed}"] = {"returncodes": rcs, "all_checks_passed": passed}
    with open(os.path.join(args.out_root, "replay_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[replay] ===== SUMMARY =====", flush=True)
    for k, v in summary.items():
        print(f"[replay] {k}: rc={v['returncodes']} all_checks_passed={v['all_checks_passed']}", flush=True)
    n_pass = sum(1 for v in summary.values() if v["all_checks_passed"])
    print(f"[replay] {n_pass}/{len(summary)} cells fully verified -> {args.out_root}/replay_summary.json", flush=True)
    return 0 if n_pass == len(summary) else 1


def parse_args():
    p = argparse.ArgumentParser(description="Record the exact TF GT training setup (order, init weights) and prove the replay is exact.")
    p.add_argument("--dataset", choices=GT_DATASETS)
    p.add_argument("--target", choices=GT_TARGETS)
    p.add_argument("--seed", type=int)
    p.add_argument("--mode", choices=["order", "train", "both"], default="both")
    p.add_argument("--all", action="store_true", help="run order+train for all 8 GT quick cells (subprocess per mode)")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--order-steps", type=int, default=150000,
                   help="training steps of scenario order to record (300 epochs x 500 steps covers any converged run)")
    p.add_argument("--check-steps", type=int, default=2000,
                   help="order mode: steps of the full pipeline to compare against the light recording")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--shuffle-buffer", type=int, default=200, help="run_experiments.py default used by every GT run")
    p.add_argument("--out-root", default=os.path.join("tensorflow_version_gt", "replay"))
    p.add_argument("--gt-root", default="tensorflow_version_gt")
    p.add_argument("--data-path", default="data")
    p.add_argument("--tmp-root", default=tempfile.gettempdir(), help="where the throw-away replay checkpoints go")
    p.add_argument("--keep-tmp", action="store_true")
    p.add_argument("--use-wandb", action="store_true", help="mirror experiment.py's W&B logging (checks that it does not alter the RNG sequence)")
    p.add_argument("--wandb-project", default="routenet-gauss-replay-check")
    return p.parse_args()


def main():
    args = parse_args()
    if args.all:
        sys.exit(run_all(args))
    if not (args.dataset and args.target and args.seed is not None):
        raise SystemExit("--dataset, --target and --seed are required (or use --all)")
    if args.mode == "both":
        # Each mode needs a fresh process: re-exec ourselves per mode.
        argv = [a for a in sys.argv[1:] if a not in ("--mode", "both")]
        argv = [a for i, a in enumerate(argv) if not (a == "both" and i > 0 and argv[i - 1] == "--mode")]
        for mode in ["order", "train"]:
            rc = subprocess.call([sys.executable, os.path.abspath(__file__)] + argv + ["--mode", mode])
            if rc != 0:
                sys.exit(rc)
        return

    out_dir = cell_dir(args.out_root, args.dataset, args.target, args.seed)
    os.makedirs(out_dir, exist_ok=True)

    def log(msg):
        print(f"[replay {args.dataset}/{args.target}/seed_{args.seed} {args.mode}] {msg}", flush=True)

    log(f"TF {tf.__version__} | python {sys.version.split()[0]} | out_dir={out_dir}")
    if args.mode == "order":
        run_order(args, out_dir, log)
    else:
        run_train(args, out_dir, log)


if __name__ == "__main__":
    main()
