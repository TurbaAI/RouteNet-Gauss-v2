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

# PyTorch translation of the single-run training script. Every original TensorFlow line is
# kept as a `#TF:` comment with its translation below; Keras' model.compile/fit is replaced by
# training_lib.fit (Keras-exact loop, see experiment.py / PYTORCH_PORT.md). Run from the repo
# root:  python train.py            (conda env RG_torch)
#
# Two deliberate differences from the TF script: (1) `experiment_name` defaults to
# "torch_train" — the TF default "paper_weights" made a plain `python train.py` write epoch
# checkpoints into ckpt/paper_weights/, on top of the shipped paper weights; (2) the device is
# chosen explicitly below (DEVICE), TF forced CPU via CUDA_VISIBLE_DEVICES.

import os

#TF: os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#TF: import tensorflow as tf
import torch

from models import RouteNetGauss
#from data import load_dataset

from utils import prepare_targets_and_mask, load_dataset
from training_lib import (
    get_z_scores_dict,
    get_positional_mape,
    get_positional_r2,
    r2_score,
)
# PyTorch-only: Keras-exact loss / optimizer / callbacks / training loop
from training_lib import (
    KerasAdam,
    KerasEarlyStopping,
    KerasModelCheckpoint,
    KerasReduceLROnPlateau,
    LearningRateLogger,
    fit,
    keras_mape_loss,
)

from random import seed
import numpy as np

# Set all seeds
SEED = 1
seed(SEED)
#TF: tf.random.set_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# RUN EAGERLY -> True for debugging
# PyTorch always runs eagerly; the switch is kept for structural parity only.
RUN_EAGERLY = False
#TF: tf.config.run_functions_eagerly(RUN_EAGERLY)
# RELOAD_WEIGHTS -> True to continue training from a checkpoint
RELOAD_WEIGHTS = False

# PyTorch-only settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THREADS = 1  # torch CPU threads (oversubscription with other jobs is very slow)
INIT = "torch"  # "torch" (PyTorch default init) or "keras" (glorot/orthogonal/zeros, as TF)
torch.set_num_threads(THREADS)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


# NOTE: get_z_scores_dict, get_positional_mape, r2_score and get_positional_r2 were
# moved to training_lib.py so they can be shared with experiment.py (the job-matrix
# runner) without importing this script, which trains at import time. They are imported
# above and used unchanged below.


# Name of the dataset
#ds_name = "data_mawi_pcaps"
ds_name = "mawi_pcaps"
# Experiment identifier
#TF: experiment_name = "paper_weights"
experiment_name = "torch_train"
# Target to be predicted
target = "delay"
# List of subtargets (avg or percentiles)
targets = [
    f"flow_avg_{target}",
    f"flow_p50_{target}",
    f"flow_p90_{target}",
    f"flow_p95_{target}",
    f"flow_p99_{target}",
]
mask = f"flow_has_{target}"
experiment_path = f"{experiment_name}/{ds_name}/{RouteNetGauss.__name__}/{target}"

ds_train_raw = load_dataset(f"{ds_name}/training")
ds_train = (
    ds_train_raw
    .map(prepare_targets_and_mask(targets, mask))
    .shuffle(1000, seed=SEED, reshuffle_each_iteration=True)
    .repeat()
)
ds_val = load_dataset(f"{ds_name}/validation").map(
    prepare_targets_and_mask(targets, mask)
)

# --- Added by Claude: data-insight visualization only, not part of training ---
# The block below (and the visualization/ package it calls into) was added by
# Claude Code purely to give a visual/textual sanity check of the dataset before
# training starts: it reconstructs the network topology of one training sample
# and renders it, together with link capacity and queue (buffer_type) info, to
# visualization/output/<ds_name>/. It reads data only and does not influence the
# model, the loss, or training in any way. See visualization/README.md for
# details, or delete this block (and the import) to remove it entirely.
from visualization import pick_sample, run_sample_deep_dive

# index=None picks a random sample, so each run shows a different topology. Note this
# uses random.SystemRandom internally, NOT the SEED-ed `random` stream above -- a seeded
# draw would hand back the same "random" sample every run. Pass an explicit index (e.g.
# pick_sample(ds_train_raw, index=0)) to reproduce one particular topology.
sample_x, sample_y, sample_index = pick_sample(ds_train_raw)
run_sample_deep_dive(sample_x, sample_y, sample_index=sample_index, dataset_name=ds_name)
# --- End of Claude-added visualization block ---

print("target:", target)
#TF: optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
#TF: loss = tf.keras.losses.MeanAbsolutePercentageError()
model = RouteNetGauss(
    output_dim=len(targets),
    mask_field=mask,
    use_trans_delay=target == "delay",
    z_scores=get_z_scores_dict(
        ds_train,
        RouteNetGauss.z_scores_fields,
        summarize=500,
        flatten=True,
        store_res_path=os.path.join("normalization", experiment_path, "z_scores.pkl"),
        check_existing=True,
    ),
    init=INIT,
)
model.to(DEVICE)
optimizer = KerasAdam(model.parameters(), lr=0.001, clipnorm=1.0)
loss = keras_mape_loss

#TF: model.compile(
#TF:     optimizer=optimizer,
#TF:     loss=loss,
#TF:     run_eagerly=RUN_EAGERLY,
#TF:     metrics=[
#TF:         get_positional_mape(0, "avg"),
#TF:         get_positional_mape(1, "p50"),
#TF:         get_positional_mape(2, "p90"),
#TF:         get_positional_mape(3, "p95"),
#TF:         get_positional_mape(4, "p99"),
#TF:         get_positional_r2(0, "avg"),
#TF:         get_positional_r2(1, "p50"),
#TF:         get_positional_r2(2, "p90"),
#TF:         get_positional_r2(3, "p95"),
#TF:         get_positional_r2(4, "p99"),
#TF:     ],
#TF: )
metrics = [
    get_positional_mape(0, "avg"),
    get_positional_mape(1, "p50"),
    get_positional_mape(2, "p90"),
    get_positional_mape(3, "p95"),
    get_positional_mape(4, "p99"),
    get_positional_r2(0, "avg"),
    get_positional_r2(1, "p50"),
    get_positional_r2(2, "p90"),
    get_positional_r2(3, "p95"),
    get_positional_r2(4, "p99"),
]

ckpt_dir = f"ckpt/{experiment_path}"
#TF: latest = tf.train.latest_checkpoint(ckpt_dir)
_ckpts = sorted(
    (f for f in (os.listdir(ckpt_dir) if os.path.isdir(ckpt_dir) else []) if f.endswith(".pt")),
    key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)),
)
latest = os.path.join(ckpt_dir, _ckpts[-1]) if _ckpts else None
if RELOAD_WEIGHTS and latest is not None:
    print("Found a pretrained model, restoring...")
    #TF: model.load_weights(latest)
    model.load_state_dict(torch.load(latest, map_location=DEVICE, weights_only=True))
else:
    print("Starting training from scratch...")

#TF: filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.4f}")
#TF: cp_callback = tf.keras.callbacks.ModelCheckpoint(
#TF:     filepath=filepath,
#TF:     verbose=1,
#TF:     mode="min",
#TF:     save_best_only=False,
#TF:     save_weights_only=True,
#TF:     save_freq="epoch",
#TF: )
cp_callback = KerasModelCheckpoint(ckpt_dir, save_best_only=False, verbose=1)
#TF: tensorboard_callback = tf.keras.callbacks.TensorBoard(
#TF:     log_dir=f"tensorboard/{experiment_path}", histogram_freq=1
#TF: )
from torch.utils.tensorboard import SummaryWriter

tensorboard_callback = SummaryWriter(log_dir=f"tensorboard/{experiment_path}")
#TF: early_stop = tf.keras.callbacks.EarlyStopping(
#TF:     monitor="val_loss",
#TF:     patience=15,
#TF:     restore_best_weights=True,
#TF:     start_from_epoch=4,
#TF: )
early_stop = KerasEarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    start_from_epoch=4,
)
#TF: reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
#TF:     factor=0.5, patience=10, verbose=1, cooldown=3, mode="min", monitor="loss"
#TF: )
reduce_lr_callback = KerasReduceLROnPlateau(
    factor=0.5, patience=10, verbose=1, cooldown=3, mode="min", monitor="loss"
) if False else KerasReduceLROnPlateau(factor=0.5, patience=10, verbose=1, cooldown=3, monitor="loss")

#TF: model.fit(
#TF:     ds_train,
#TF:     epochs=300,
#TF:     steps_per_epoch=500,
#TF:     validation_data=ds_val,
#TF:     callbacks=[
#TF:         cp_callback,
#TF:         tensorboard_callback,
#TF:         reduce_lr_callback,
#TF:         tf.keras.callbacks.TerminateOnNaN(),
#TF:     ],
#TF:     use_multiprocessing=True,
#TF: )
# PyTorch: the same fit — same epochs/steps, same callbacks (early_stop is defined above but,
# as in the TF script, NOT passed to fit; TerminateOnNaN is built into training_lib.fit). The
# training order is materialised up front (see experiment.py) and the run can be resumed
# with the resume.pt written after every epoch.
EPOCHS, STEPS_PER_EPOCH = 300, 500
results_dir = f"results/{experiment_path}"
os.makedirs(results_dir, exist_ok=True)
train_order = ds_train.index_order(EPOCHS * STEPS_PER_EPOCH)
np.save(os.path.join(results_dir, "sample_order_used.npy"), train_order)
history_rows, train_seconds, step_seconds = fit(
    model, optimizer, loss, metrics,
    train_order=train_order,
    by_idx={int(x["sample_idx"]): (x, y) for x, y in ds_train_raw},
    map_fn=prepare_targets_and_mask(targets, mask),
    ds_val=ds_val,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    device=DEVICE,
    checkpoint_cb=cp_callback,
    tb_writer=tensorboard_callback,
    history_path=os.path.join(results_dir, "history.csv"),
    lr_logger=LearningRateLogger(),
    reduce_lr_cb=reduce_lr_callback,
    step_log_path=os.path.join(results_dir, "step_losses.csv"),
    resume_path=os.path.join(results_dir, "resume.pt"),
)
tensorboard_callback.close()
print(f"training finished in {train_seconds:.0f}s ({len(history_rows)} epochs)")
