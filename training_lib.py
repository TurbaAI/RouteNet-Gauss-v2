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

# Shared, importable helpers used by both train.py (single-run script) and
# experiment.py (job-matrix runner). These were originally defined inline in
# train.py; they were extracted here so they can be reused without importing
# train.py (which runs training at import time).
#
# PyTorch translation. Every original TensorFlow line is kept as a `#TF:` comment with its
# translation below (frozen original: tf_reference/training_lib.py). Keras' model.fit/compile
# machinery has no PyTorch counterpart, so this module additionally provides the pieces the
# PyTorch training loop needs to reproduce Keras' arithmetic exactly:
#   - keras_mape_loss:     tf.keras.losses.MeanAbsolutePercentageError (with its 1e-7 floor)
#   - clip_by_norm_:       Keras Optimizer(clipnorm=...) semantics (each tensor clipped separately)
#   - KerasAdam:           Keras 2.15 Adam.update_step (epsilon added outside the bias correction)
#   - MeanTracker:         Keras' per-epoch running mean of loss/metrics (history.csv semantics)

import os
import pickle

import numpy as np
#TF: import tensorflow as tf
import torch


def get_z_scores_dict(
    ds,
    params,
    include_y=None,
    flatten=False,
    summarize=-1,
    store_res_path=None,
    check_existing=False,
):
    """
    Get the mean and the std for different parameters of a dataset. Later used by the
    models for the z-score normalization.

    Parameters
    ----------
    ds
        tensorflow.data.Dataset
    params
        list of strings indicating the parameters to extract the features from
    include_y, optional
        Indicates if to also extract the features of the output variable. Inputs
        indicate the string key used on the return dict. If None, it is not included.
    flatten, optional
        If true, mean and std are computed globally for all dimensions in each feature.
        Otherwise, the values are computed for each dimension separately.
    summarize, optional
        If > 0, only uses the first n samples to compute the mean and std.
    store_res_path, optional
        If not None, the results are stored in the path indicated by the string.
        The dictionary is stored using the pickle library.

    Returns
    -------
    dict
        Dictionary containing the min and the max-min for each parameter.
    """
    # If check_existing is True, check if the file exists and return the dict (if so)
    if store_res_path is not None and check_existing:
        if os.path.exists(store_res_path):
            with open(store_res_path, "rb") as ff:
                return pickle.load(ff)

    # Use first sample to get the shape of the tensors
    iter_ds = iter(ds)
    next_sample = next(iter_ds)
    sample, label = next_sample[0], next_sample[1]
    params_lists = {param: sample[param].numpy() for param in params}
    if include_y:
        params_lists[include_y] = label.numpy()

    if summarize > 0:
        max_samples = summarize - 1

    # Include the rest of the samples
    for ii, (sample, label) in enumerate(map(lambda x: (x[0], x[1]), iter_ds)):
        if summarize > 0 and ii > max_samples:
            break
        for param in params:
            params_lists[param] = np.concatenate(
                (params_lists[param], sample[param].numpy()), axis=0
            )
        if include_y:
            params_lists[include_y] = np.concatenate(
                (params_lists[include_y], label.numpy()), axis=0
            )

    scores = dict()
    axis = None if flatten else 0
    for param, param_list in params_lists.items():
        scores[param] = [np.mean(param_list, axis=axis), np.std(param_list, axis=axis)]
        # Check if std is 0
        if scores[param][1].size == 1 and scores[param][1] == 0:
            print(f"Z-score normalization Warning: {param} has a std of 0.")
            scores[param][1] = 1
        elif scores[param][1].size > 1 and np.any(scores[param][1] == 0):
            print(
                f"Z-score normalization Warning: Several values of {param} has a std of 0."
            )
            scores[param][1][scores[param][1] == 0] = 1

    if store_res_path is not None:
        store_res_dir, _ = os.path.split(store_res_path)
        os.makedirs(store_res_dir, exist_ok=True)
        with open(store_res_path, "wb") as ff:
            pickle.dump(scores, ff)

    return scores


def get_positional_mape(pos: int, name: str) -> callable:
    """Returns the MAPE metric for a specific feature at a given position.

    Parameters
    ----------
    pos : int
        Position of the feature in the output tensor.
    name : str
        Name of the feature.

    Returns
    -------
    callable
        Function to calculate the MAPE metric for a specific feature, to be added to
        model metrics to be included in the tensorboard logs.
    """

    def mape_metric(y_true, y_pred):
        y_true = y_true[:, pos]
        y_pred = y_pred[:, pos]
        #TF: return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100
        return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100

    mape_metric.__name__ = f"mape_{name}_metric"
    return mape_metric


#TF: def r2_score(y_true: tf.TensorSpec, y_pred: tf.TensorSpec) -> float:
def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Returns the R2 score between the true and predicted values.

    Parameters
    ----------
    y_true : tf.TensorSpec
        True values
    y_pred : tf.TensorSpec
        Predicted values

    Returns
    -------
    float
        R2 score
    """
    #TF: residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    residual = torch.sum(torch.square(torch.sub(y_true, y_pred)))
    #TF: total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    total = torch.sum(torch.square(torch.sub(y_true, torch.mean(y_true))))
    #TF: r2 = tf.subtract(1.0, tf.math.divide(residual, total))
    r2 = torch.sub(1.0, torch.div(residual, total))
    return r2


def get_positional_r2(pos: int, name: str) -> callable:
    """Returns the R2 metric for a specific feature at a given position.

    Parameters
    ----------
    pos : int
        Position of the feature in the output tensor.
    name : str
        Name of the feature.

    Returns
    -------
    callable
        Function to calculate the R2 metric for a specific feature, to be added to
        model metrics to be included in the tensorboard logs.
    """

    def r2_metric(y_true, y_pred):
        y_true = y_true[:, pos]
        y_pred = y_pred[:, pos]
        return r2_score(y_true, y_pred)

    r2_metric.__name__ = f"r2_{name}_metric"
    return r2_metric


#TF: class LearningRateLogger(tf.keras.callbacks.Callback):
class LearningRateLogger:
    """Writes the optimizer's current learning rate into the epoch `logs` dict.

    Adding it to the callback list means the LR shows up in the CSVLogger output and
    the TensorBoard scalars alongside loss/metrics, which is useful for inspecting the
    effect of ReduceLROnPlateau on these short runs.

    PyTorch: there is no Keras callback machinery; the training loop in experiment.py calls
    `on_epoch_end(epoch, logs, optimizer)` itself at the same point Keras would.
    """

    #TF: def on_epoch_end(self, epoch, logs=None):
    def on_epoch_end(self, epoch, logs=None, optimizer=None):
        if logs is None:
            return
        #TF: lr = self.model.optimizer.learning_rate
        #TF: # learning_rate may be a schedule or a variable; resolve to a float either way.
        #TF: if callable(lr):
        #TF:     lr = lr(self.model.optimizer.iterations)
        #TF: logs["lr"] = float(tf.keras.backend.get_value(lr))
        logs["lr"] = float(optimizer.param_groups[0]["lr"])


# --------------------------------------------------------------------------------------
# PyTorch-only additions: exact counterparts of the Keras pieces experiment.py relied on.
# --------------------------------------------------------------------------------------

KERAS_EPSILON = 1e-7  # tf.keras.backend.epsilon()


def keras_mape_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """tf.keras.losses.MeanAbsolutePercentageError():
        100 * mean(|y_true - y_pred| / max(|y_true|, epsilon))
    Keras averages the per-element percentage errors over the last axis first and then over
    the batch (reduction SUM_OVER_BATCH_SIZE); with a fixed number of columns the two
    means compose to the plain mean over all elements, which is what is computed here."""
    diff = torch.abs((y_true - y_pred) / torch.clamp(torch.abs(y_true), min=KERAS_EPSILON))
    return 100.0 * torch.mean(diff)


def clip_by_norm_(grads, clipnorm: float) -> None:
    """Keras `Optimizer(clipnorm=c)`: EVERY gradient tensor is clipped separately to L2 norm
    <= c (tf.clip_by_norm per tensor). This is NOT torch.nn.utils.clip_grad_norm_, which
    clips the GLOBAL norm over all parameters (Keras calls that `global_clipnorm`).
    tf.clip_by_norm computes g * c / max(||g||, c); the same expression is used here."""
    for g in grads:
        if g is None:
            continue
        norm = torch.linalg.vector_norm(g)
        g.mul_(clipnorm).div_(torch.maximum(norm, torch.full_like(norm, clipnorm)))


class KerasAdam(torch.optim.Optimizer):
    """Adam with exactly the arithmetic of tf.keras.optimizers.Adam (Keras 2.15):

        alpha = lr * sqrt(1 - beta2**t) / (1 - beta1**t)
        m += (g - m) * (1 - beta1)
        v += (g**2 - v) * (1 - beta2)
        var -= alpha * m / (sqrt(v) + epsilon)

    Differences from torch.optim.Adam: (1) default epsilon 1e-7 (torch: 1e-8);
    (2) Keras adds epsilon to sqrt(v_hat)-free denominator sqrt(v) AFTER folding the
    bias correction into alpha, torch adds epsilon to sqrt(v)/sqrt(1-beta2**t), i.e. torch's
    effective epsilon is epsilon*sqrt(1-beta2**t) — different whenever v is tiny;
    (3) `clipnorm` is applied per tensor before the update (see clip_by_norm_).
    `step()` mirrors Keras' `iterations` counter: t is incremented once per apply."""

    def __init__(self, params, lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=KERAS_EPSILON, clipnorm=None):
        defaults = dict(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm=clipnorm)
        super().__init__(params, defaults)
        self.iterations = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if group["clipnorm"] is not None:
                clip_by_norm_([p.grad for p in params], group["clipnorm"])
            t = self.iterations + 1  # Keras: local_step = iterations + 1
            beta_1, beta_2, lr, eps = group["beta_1"], group["beta_2"], group["lr"], group["epsilon"]
            # Keras computes these in the variable dtype (float32 here) from Python floats.
            beta_1_power = torch.tensor(beta_1, dtype=torch.float32) ** t
            beta_2_power = torch.tensor(beta_2, dtype=torch.float32) ** t
            alpha = torch.tensor(lr, dtype=torch.float32) * torch.sqrt(1 - beta_2_power) / (1 - beta_1_power)
            for p in params:
                state = self.state[p]
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v, g = state["m"], state["v"], p.grad
                m.add_((g - m) * (1 - beta_1))
                v.add_((torch.square(g) - v) * (1 - beta_2))
                p.sub_((m * alpha.to(p.device)) / (torch.sqrt(v) + eps))
        self.iterations += 1
        return loss

    def state_dict(self):
        d = super().state_dict()
        d["iterations"] = self.iterations
        return d

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        self.iterations = int(state_dict.pop("iterations", 0))
        super().load_state_dict(state_dict)


class MeanTracker:
    """Keras `Mean` metric state: the values Keras reports per epoch (`loss`, every
    compiled metric, and their `val_` counterparts) are running means over the steps of
    the epoch, reset at every epoch boundary. history.csv of the PyTorch runs is built with
    this so it is comparable row-by-row with the TF history.csv."""

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, weight=1.0):
        # Keras weights the LOSS by the batch dimension (here: number of predictions of the
        # scenario) and the metrics by 1 per step; pass `weight` accordingly.
        self.total += float(value) * float(weight)
        self.count += float(weight)

    def result(self):
        return self.total / self.count if self.count else float("nan")

    def reset(self):
        self.total, self.count = 0.0, 0


# --------------------------------------------------------------------------------------
# PyTorch-only: Keras callback logic (verbatim from Keras 2.15 keras/callbacks.py) and the
# training loop that replaces model.compile(...) + model.fit(...). Shared by experiment.py
# and train.py.
# --------------------------------------------------------------------------------------

import copy
import csv
import math
import time


class KerasModelCheckpoint:
    """tf.keras.callbacks.ModelCheckpoint(save_weights_only=True, monitor='val_loss', mode='min',
    save_freq='epoch'): saves `{epoch:02d}-{val_loss:.4f}.pt` (epoch is 1-based) every epoch, or
    only on improvement with save_best_only."""

    def __init__(self, ckpt_dir, save_best_only=False, verbose=1):
        self.ckpt_dir, self.save_best_only, self.verbose = ckpt_dir, save_best_only, verbose
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs, model):
        current = logs["val_loss"]
        filepath = os.path.join(self.ckpt_dir, f"{epoch + 1:02d}-{current:.4f}.pt")
        if self.save_best_only:
            if np.less(current, self.best):
                if self.verbose:
                    print(f"\nEpoch {epoch + 1}: val_loss improved from {self.best:.5f} to {current:.5f}, saving model to {filepath}")
                self.best = current
            else:
                if self.verbose:
                    print(f"\nEpoch {epoch + 1}: val_loss did not improve from {self.best:.5f}")
                return
        elif self.verbose:
            print(f"\nEpoch {epoch + 1}: saving model to {filepath}")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), filepath)

    def state(self):
        return {"best": float(self.best)}

    def load_state(self, st):
        self.best = st["best"]


class KerasReduceLROnPlateau:
    """tf.keras.callbacks.ReduceLROnPlateau(factor, patience, verbose, cooldown, mode='min',
    monitor, min_delta=1e-4, min_lr=0) — Keras 2.15 on_epoch_end logic, verbatim."""

    def __init__(self, monitor="loss", factor=0.5, patience=10, verbose=1, cooldown=3, min_delta=1e-4, min_lr=0.0):
        self.monitor, self.factor, self.patience, self.verbose = monitor, factor, patience, verbose
        self.cooldown, self.min_delta, self.min_lr = cooldown, min_delta, min_lr
        self._reset()

    def _reset(self):
        self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
        self.best = np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0

    def on_epoch_end(self, epoch, logs, optimizer):
        logs["lr"] = float(optimizer.param_groups[0]["lr"])
        current = logs.get(self.monitor)
        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(optimizer.param_groups[0]["lr"])
                if old_lr > np.float32(self.min_lr):
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    new_lr = float(np.float32(new_lr))  # Keras keeps the lr in a float32 variable
                    for g in optimizer.param_groups:
                        g["lr"] = new_lr
                    if self.verbose > 0:
                        print(f"\nEpoch {epoch + 1}: ReduceLROnPlateau reducing learning rate to {new_lr}.")
                    self.cooldown_counter = self.cooldown
                    self.wait = 0

    def state(self):
        return {"best": float(self.best), "cooldown_counter": self.cooldown_counter, "wait": self.wait}

    def load_state(self, st):
        self.best, self.cooldown_counter, self.wait = st["best"], st["cooldown_counter"], st["wait"]


class KerasEarlyStopping:
    """tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience, restore_best_weights=True,
    start_from_epoch, min_delta=0, mode='min') — Keras 2.15 logic, verbatim (epoch is 0-based)."""

    def __init__(self, monitor="val_loss", patience=0, restore_best_weights=True, start_from_epoch=0, min_delta=0.0):
        self.monitor, self.patience, self.restore_best_weights = monitor, patience, restore_best_weights
        self.start_from_epoch, self.min_delta = start_from_epoch, abs(min_delta)
        self.wait, self.stopped_epoch, self.best, self.best_weights, self.best_epoch = 0, 0, np.Inf, None, 0

    def _is_improvement(self, monitor_value, reference_value):
        return np.less(monitor_value - self.min_delta, reference_value)

    def on_epoch_end(self, epoch, logs, model):
        """Returns True when training must stop (model weights already restored to the best)."""
        current = logs.get(self.monitor)
        if current is None or epoch < self.start_from_epoch:
            return False
        if self.restore_best_weights and self.best_weights is None:
            self.best_weights = copy.deepcopy(model.state_dict())
        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
            self.wait = 0  # baseline is None
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            if self.restore_best_weights and self.best_weights is not None:
                print(f"Restoring model weights from the end of the best epoch: {self.best_epoch + 1}.")
                model.load_state_dict(self.best_weights)
            print(f"Epoch {epoch + 1}: early stopping")
            return True
        return False

    def state(self):
        return {"wait": self.wait, "stopped_epoch": self.stopped_epoch, "best": float(self.best),
                "best_epoch": self.best_epoch, "best_weights": self.best_weights}

    def load_state(self, st):
        self.wait, self.stopped_epoch, self.best = st["wait"], st["stopped_epoch"], st["best"]
        self.best_epoch, self.best_weights = st["best_epoch"], st["best_weights"]


def load_resume_state(resume_path):
    """The state saved by `fit` at every epoch end, or None."""
    if resume_path and os.path.exists(resume_path):
        return torch.load(resume_path, map_location="cpu", weights_only=False)
    return None


def fit(
    model,
    optimizer,
    loss_fn,
    metrics,
    train_order,
    by_idx,
    map_fn,
    ds_val,
    epochs,
    steps_per_epoch,
    device,
    checkpoint_cb=None,
    reduce_lr_cb=None,
    early_stop_cb=None,
    lr_logger=None,
    tb_writer=None,
    history_path=None,
    step_log_path=None,
    resume_path=None,
    resume_state=None,
    wandb_run=None,
    clip_grads=None,
    verbose=True,
):
    """model.fit(ds_train, epochs, steps_per_epoch, validation_data=ds_val, callbacks=[...])
    with Keras' arithmetic and callback order, written out.

    train_order : array of sample_idx, one per training step (epochs * steps_per_epoch entries)
    by_idx      : {sample_idx: (x, y)} of the raw training scenarios
    map_fn      : prepare_targets_and_mask(...) applied to every scenario
    clip_grads  : optional callable(grads) run before optimizer.step() (KerasAdam clips itself)
    Returns (history_rows, train_seconds, step_seconds). Every epoch end writes `resume_path`
    with the complete training state so an interrupted run can continue (`resume_state`).
    """
    from torch_ragged import sample_to_device

    history_rows, step_rows, start_epoch, train_seconds_prev = [], [], 0, 0.0
    if resume_state:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        if checkpoint_cb is not None:
            checkpoint_cb.load_state(resume_state["checkpoint_cb"])
        if reduce_lr_cb is not None:
            reduce_lr_cb.load_state(resume_state["reduce_lr_cb"])
        if early_stop_cb is not None and resume_state.get("early_stop_cb"):
            early_stop_cb.load_state(resume_state["early_stop_cb"])
        history_rows = resume_state["history_rows"]
        start_epoch = resume_state["epoch"] + 1
        train_seconds_prev = resume_state["train_seconds"]
        torch.set_rng_state(resume_state["torch_rng"])
    else:
        for path in (history_path, step_log_path):
            if path and os.path.exists(path):
                os.remove(path)

    on_gpu = torch.device(device).type == "cuda"
    t0 = time.time()
    stop_training = False
    step_seconds = []
    for epoch in range(start_epoch, epochs):
        model.train()
        loss_tracker = MeanTracker()
        metric_trackers = [MeanTracker() for _ in metrics]
        for step in range(steps_per_epoch):
            global_step = epoch * steps_per_epoch + step
            sample_idx = int(train_order[global_step])
            x, y = map_fn(*by_idx[sample_idx])
            x, y = sample_to_device(x, device), y.to(device)
            ts = time.time()
            y_pred = model(x)
            step_loss = loss_fn(y, y_pred)
            optimizer.zero_grad(set_to_none=True)
            step_loss.backward()
            if clip_grads is not None:
                clip_grads([p.grad for p in model.parameters() if p.grad is not None])
            optimizer.step()
            if on_gpu:
                torch.cuda.synchronize()
            step_seconds.append(time.time() - ts)
            step_loss_value = float(step_loss.detach())
            # Keras bookkeeping: the loss Mean is weighted by the batch dimension (number of
            # predictions of the scenario), the metric Means are unweighted.
            loss_tracker.update(step_loss_value, weight=y.shape[0])
            with torch.no_grad():
                for tr, fn in zip(metric_trackers, metrics):
                    tr.update(float(fn(y, y_pred.detach())))
            if step_log_path:
                step_rows.append({"global_step": global_step, "epoch": epoch, "step_in_epoch": step, "sample_idx": sample_idx,
                                  "loss": step_loss_value, "running_mean_loss": loss_tracker.result(),
                                  "lr": float(optimizer.param_groups[0]["lr"]), "seconds": step_seconds[-1]})
            # tf.keras.callbacks.TerminateOnNaN (on_batch_end)
            if not math.isfinite(step_loss_value):
                print(f"Batch {step}: Invalid loss, terminating training")
                stop_training = True
                break
        if stop_training:
            break

        # validation pass (model.fit's per-epoch evaluate: same weighted/unweighted means)
        model.eval()
        val_loss_tracker = MeanTracker()
        val_metric_trackers = [MeanTracker() for _ in metrics]
        with torch.no_grad():
            for x, y in ds_val:
                x, y = sample_to_device(x, device), y.to(device)
                y_pred = model(x)
                val_loss_tracker.update(float(loss_fn(y, y_pred)), weight=y.shape[0])
                for tr, fn in zip(val_metric_trackers, metrics):
                    tr.update(float(fn(y, y_pred)))

        logs = {"loss": loss_tracker.result()}
        for tr, fn in zip(metric_trackers, metrics):
            logs[fn.__name__] = tr.result()
        logs["val_loss"] = val_loss_tracker.result()
        for tr, fn in zip(val_metric_trackers, metrics):
            logs["val_" + fn.__name__] = tr.result()
        if verbose:
            elapsed = time.time() - t0 + train_seconds_prev
            print(f"Epoch {epoch + 1}/{epochs} - {elapsed:.0f}s - loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f} "
                  f"- mean step {np.mean(step_seconds[-steps_per_epoch:]):.2f}s", flush=True)

        # --- callbacks, in Keras' order: ModelCheckpoint, TensorBoard, CSVLogger,
        #     LearningRateLogger, ReduceLROnPlateau, [EarlyStopping], [W&B] ---
        if checkpoint_cb is not None:
            checkpoint_cb.on_epoch_end(epoch, logs, model)
        if tb_writer is not None:
            for k, v in logs.items():
                tb_writer.add_scalar("epoch_" + k, v, epoch)
        if history_path:
            # CSVLogger: header = "epoch" + sorted(logs.keys()); lr is NOT in the row because
            # LearningRateLogger runs after CSVLogger (same columns as the TF history.csv).
            keys = sorted(logs.keys())
            with open(history_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["epoch"] + keys)
                if f.tell() == 0:
                    w.writeheader()
                w.writerow({"epoch": epoch, **{k: logs[k] for k in keys}})
        if lr_logger is not None:
            lr_logger.on_epoch_end(epoch, logs, optimizer)
        if reduce_lr_cb is not None:
            reduce_lr_cb.on_epoch_end(epoch, logs, optimizer)
        if tb_writer is not None and "lr" in logs:
            tb_writer.add_scalar("epoch_lr", logs["lr"], epoch)
        if early_stop_cb is not None:
            stop_training = early_stop_cb.on_epoch_end(epoch, logs, model)
        if wandb_run is not None:
            wandb_run.log({"epoch/" + k: v for k, v in logs.items()}, step=epoch)
        history_rows.append(logs)

        if step_log_path and step_rows:
            with open(step_log_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
                if f.tell() == 0:
                    w.writeheader()
                w.writerows(step_rows)
            step_rows = []
        if resume_path:
            torch.save({
                "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "checkpoint_cb": checkpoint_cb.state() if checkpoint_cb is not None else None,
                "reduce_lr_cb": reduce_lr_cb.state() if reduce_lr_cb is not None else None,
                "early_stop_cb": early_stop_cb.state() if early_stop_cb is not None else None,
                "history_rows": history_rows, "train_seconds": time.time() - t0 + train_seconds_prev,
                "torch_rng": torch.get_rng_state(), "wandb_id": wandb_run.id if wandb_run is not None else None,
            }, resume_path)
        if stop_training:
            break
    return history_rows, time.time() - t0 + train_seconds_prev, step_seconds
