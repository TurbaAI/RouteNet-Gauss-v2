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

# Draw TensorFlow's own initial weights of RouteNet-Gauss for any seed (PYTORCH_PARITY.md §7.6).
#
# Follows the replay recorder's path to its init dump exactly (tf_reference/replay_tf_run.py
# run_train: build_pipeline -> RecordingRouteNetGauss -> compile -> build on a validation scenario
# -> collect_weights), i.e. the ground-truth pipeline of experiment.py @ 2e30d5d, so a seed yields
# the initial weights the TF original would have trained from. `--check` proves it: for seeds 1
# and 2 the result must equal the recorded tensorflow_version_gt/replay/.../init_weights.npz bit for
# bit. One seed per process (fresh TF RNG state). Output: <out>/init_weights.npz (the recorder's
# format, loadable by experiment.py --init-weights) and <out>/z_scores.pkl.
#
# Run from the repo root in the TF environment that produced the ground truth (conda env RG):
#   python parity/draw_tf_inits.py --seed 3 --out results/verification/init_draws/tf_seed_3
#   python parity/draw_tf_inits.py --seed 1 --out <dir> \
#       --check tensorflow_version_gt/replay/trex_multiburst/RouteNetGauss/delay/seed_1/init_weights.npz

import argparse
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import tensorflow as tf

from tf_reference.replay_tf_run import (
    PERCENTILES,
    RecordingRouteNetGauss,
    build_pipeline,
    build_targets,
    collect_weights,
)
from tf_reference.training_lib import get_positional_mape, get_positional_r2


def main():
    p = argparse.ArgumentParser(description="TF-drawn RouteNet-Gauss initial weights for a seed")
    p.add_argument("--dataset", default="trex_multiburst")
    p.add_argument("--target", default="delay", choices=["delay", "jitter"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--shuffle-buffer", type=int, default=200, help="run_experiments.py default used by every GT run")
    p.add_argument("--data-path", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--check", default=None, help="recorded init_weights.npz that the draw must equal bit for bit")
    a = p.parse_args()
    os.chdir(_REPO_ROOT)
    os.makedirs(a.out, exist_ok=True)

    # Same calls, same order, as replay_tf_run.run_train up to the init dump.
    args = types.SimpleNamespace(seed=a.seed, dataset=a.dataset, data_path=a.data_path, shuffle_buffer=a.shuffle_buffer)
    targets, mask = build_targets(a.target)
    _, ds_val, z_scores = build_pipeline(args, targets, mask, os.path.join(a.out, "z_scores.pkl"))
    model = RecordingRouteNetGauss(
        output_dim=len(targets),
        mask_field=mask,
        use_trans_delay=a.target == "delay",
        z_scores=z_scores,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss=tf.keras.losses.MeanAbsolutePercentageError(),
        metrics=[get_positional_mape(i, n) for i, n in enumerate(PERCENTILES)]
        + [get_positional_r2(i, n) for i, n in enumerate(PERCENTILES)],
    )
    x_val, _ = next(iter(ds_val))
    model(x_val)
    init = collect_weights(model)
    np.savez(os.path.join(a.out, "init_weights.npz"), **init)

    msg = f"[draw_tf_inits] TF {tf.__version__} seed {a.seed}: {len(init)} tensors -> {a.out}/init_weights.npz"
    if a.check:
        ref = np.load(a.check)
        same = sorted(ref.files) == sorted(init) and all(np.array_equal(ref[k], init[k]) for k in init)
        msg += f"; bit-identical to {a.check}: {same}"
        print(msg, flush=True)
        sys.exit(0 if same else 1)
    print(msg, flush=True)


if __name__ == "__main__":
    main()
