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

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from models import RouteNetGauss
#from data import load_dataset

from utils import prepare_targets_and_mask, load_dataset
from training_lib import (
    get_z_scores_dict,
    get_positional_mape,
    get_positional_r2,
    r2_score,
)

from random import seed
import numpy as np

# Set all seeds
SEED = 1
seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# RUN EAGERLY -> True for debugging
RUN_EAGERLY = False
tf.config.run_functions_eagerly(RUN_EAGERLY)
# RELOAD_WEIGHTS -> True to continue training from a checkpoint
RELOAD_WEIGHTS = False


# NOTE: get_z_scores_dict, get_positional_mape, r2_score and get_positional_r2 were
# moved to training_lib.py so they can be shared with experiment.py (the job-matrix
# runner) without importing this script, which trains at import time. They are imported
# above and used unchanged below.


# Name of the dataset
#ds_name = "data_mawi_pcaps"
ds_name = "mawi_pcaps"
# Experiment identifier
experiment_name = "paper_weights"
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
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
loss = tf.keras.losses.MeanAbsolutePercentageError()
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
)

model.compile(
    optimizer=optimizer,
    loss=loss,
    run_eagerly=RUN_EAGERLY,
    metrics=[
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
    ],
)

ckpt_dir = f"ckpt/{experiment_path}"
latest = tf.train.latest_checkpoint(ckpt_dir)
if RELOAD_WEIGHTS and latest is not None:
    print("Found a pretrained model, restoring...")
    model.load_weights(latest)
else:
    print("Starting training from scratch...")

filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.4f}")
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    mode="min",
    save_best_only=False,
    save_weights_only=True,
    save_freq="epoch",
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=f"tensorboard/{experiment_path}", histogram_freq=1
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    start_from_epoch=4,
)
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.5, patience=10, verbose=1, cooldown=3, mode="min", monitor="loss"
)

model.fit(
    ds_train,
    epochs=300,
    steps_per_epoch=500,
    validation_data=ds_val,
    callbacks=[
        cp_callback,
        tensorboard_callback,
        reduce_lr_callback,
        tf.keras.callbacks.TerminateOnNaN(),
    ],
    use_multiprocessing=True,
)
