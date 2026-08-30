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

# Convert RouteNet-Gauss TensorFlow weights (a Keras `save_weights` checkpoint, or an
# `init_weights.npz` written by tf_reference/replay_tf_run.py) into a PyTorch state_dict
# for the `RouteNetGauss` in models.py.
#
# The mapping is purely a re-layout — no arithmetic — so a converted model computes the same
# function as the TF one (verified by parity/l0_forward.py):
#
#   Keras Dense              kernel [in, out], bias [out]
#   torch nn.Linear          weight [out, in] = kernel.T, bias
#
#   Keras GRUCell            kernel [in, 3H], recurrent_kernel [H, 3H], bias [2, 3H]
#   (reset_after=True)         gate order along the 3H axis: z (update) | r (reset) | h (candidate)
#   torch nn.GRUCell         weight_ih [3H, in], weight_hh [3H, H], bias_ih [3H], bias_hh [3H]
#   (nn.GRU: same + "_l0")     gate order: r | z | n
#   => weight_ih = reorder_gates(kernel).T, weight_hh = reorder_gates(recurrent_kernel).T,
#      bias_ih = reorder_gates(bias[0]), bias_hh = reorder_gates(bias[1]),
#      reorder_gates(x) = concat(x[.., H:2H], x[.., 0:H], x[.., 2H:3H]) along the 3H axis.
#   Both cells compute h' = z*h + (1-z)*tanh(W_h x + b_h + r*(U_h h + b_uh)) with
#   z = sigmoid(W_z x + b_z + U_z h + b_uz), r = sigmoid(W_r x + b_r + U_r h + b_ur).
#
#   Keras Sequential layer_with_weights-k  <->  torch nn.Sequential index 2k
#   (Dense(activation=relu) is Linear + ReLU in torch, so weight-bearing layers sit at even indices).
#
# Usage (needs tensorflow(-cpu) for TF checkpoints; .npz inputs need only numpy/torch):
#   python convert_tf_checkpoint.py ckpt/paper_weights/mawi_pcaps/RouteNetGauss/delay/201-19.7350
#   python convert_tf_checkpoint.py tensorflow_version_gt/replay/.../init_weights.npz
#   python convert_tf_checkpoint.py --all-known      # every GT / paper / replay checkpoint in the repo
# Output: <input>.pt next to the input (a plain state_dict, loadable with weights_only=True).

import argparse
import glob
import os
import sys

import numpy as np
import torch

# TF checkpoint variable name (without "/.ATTRIBUTES/VARIABLE_VALUE") -> torch state_dict key
GRU_ATTRS = ["flow_update", "link_update", "queue_update", "node_update"]
# flow_update is an nn.GRU (sequence form of the cell) in models.py: same tensors, "_l0" suffix.
GRU_PARAM_SUFFIX = {"flow_update": "_l0", "link_update": "", "queue_update": "", "node_update": ""}
MLP_ATTRS = {
    "flow_embedding": 2,
    "queue_embedding": 2,
    "link_embedding": 2,
    "node_embedding": 2,
    "readout_path": 3,
}
SUFFIX = "/.ATTRIBUTES/VARIABLE_VALUE"


def reorder_gates(x: np.ndarray) -> np.ndarray:
    """Keras gate order (z, r, h) -> torch gate order (r, z, n) along the last axis."""
    H = x.shape[-1] // 3
    return np.concatenate([x[..., H : 2 * H], x[..., 0:H], x[..., 2 * H : 3 * H]], axis=-1)


def tf_arrays_to_state_dict(arrays: dict) -> dict:
    """arrays: {tf_name: np.ndarray} with the 34 RouteNet-Gauss weight tensors."""
    sd = {}
    used = set()
    for attr in GRU_ATTRS:
        k, rk, b = arrays[f"{attr}/kernel"], arrays[f"{attr}/recurrent_kernel"], arrays[f"{attr}/bias"]
        assert b.shape[0] == 2, f"{attr}/bias must be [2, 3H] (reset_after=True); got {b.shape}"
        sfx = GRU_PARAM_SUFFIX[attr]
        sd[f"{attr}.weight_ih{sfx}"] = torch.from_numpy(np.ascontiguousarray(reorder_gates(k).T))
        sd[f"{attr}.weight_hh{sfx}"] = torch.from_numpy(np.ascontiguousarray(reorder_gates(rk).T))
        sd[f"{attr}.bias_ih{sfx}"] = torch.from_numpy(np.ascontiguousarray(reorder_gates(b[0])))
        sd[f"{attr}.bias_hh{sfx}"] = torch.from_numpy(np.ascontiguousarray(reorder_gates(b[1])))
        used |= {f"{attr}/kernel", f"{attr}/recurrent_kernel", f"{attr}/bias"}
    for attr, n_layers in MLP_ATTRS.items():
        for i in range(n_layers):
            k, b = arrays[f"{attr}/layer_with_weights-{i}/kernel"], arrays[f"{attr}/layer_with_weights-{i}/bias"]
            sd[f"{attr}.{2 * i}.weight"] = torch.from_numpy(np.ascontiguousarray(k.T))
            sd[f"{attr}.{2 * i}.bias"] = torch.from_numpy(np.ascontiguousarray(b))
            used |= {f"{attr}/layer_with_weights-{i}/kernel", f"{attr}/layer_with_weights-{i}/bias"}
    unused = set(arrays) - used
    assert not unused, f"unmapped TF tensors: {sorted(unused)}"
    assert len(sd) == 38, len(sd)  # 34 TF tensors -> 38 torch tensors (each GRU bias [2,3H] splits into bias_ih, bias_hh)
    return sd


def load_tf_arrays(path: str) -> dict:
    """Load the 34 weight tensors from a TF checkpoint prefix or an init_weights.npz."""
    if path.endswith(".npz"):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    import tensorflow as tf  # only needed for real checkpoints

    reader = tf.train.load_checkpoint(path)
    arrays = {}
    for name, _ in tf.train.list_variables(path):
        if "optimizer" in name.lower() or "save_counter" in name or "_CHECKPOINTABLE" in name:
            continue
        if not name.endswith(SUFFIX):
            continue
        short = name[: -len(SUFFIX)]
        if short.startswith("_rec_"):  # recording variables of the replay recorder
            continue
        arrays[short] = reader.get_tensor(name)
    return arrays


def convert(path: str, out: str = None) -> str:
    arrays = load_tf_arrays(path)
    sd = tf_arrays_to_state_dict(arrays)
    # Extra buffers of the torch model (z-scores) are NOT part of a TF checkpoint; they come
    # from z_scores.pkl at model construction. We add the z-score buffers only if requested.
    out = out or (path[: -len(".npz")] + ".pt" if path.endswith(".npz") else path + ".pt")
    torch.save(sd, out)
    return out


def known_checkpoints():
    paths = []
    for ck in glob.glob("ckpt/paper_weights/*/RouteNetGauss/*/checkpoint") + glob.glob(
        "tensorflow_version_gt/**/checkpoint", recursive=True
    ):
        d = os.path.dirname(ck)
        with open(ck) as f:
            for line in f:
                if line.startswith("model_checkpoint_path:"):
                    paths.append(os.path.join(d, line.split('"')[1]))
                    break
    paths += glob.glob("tensorflow_version_gt/replay/**/init_weights.npz", recursive=True)
    return sorted(set(paths))


def main():
    p = argparse.ArgumentParser(description="TF RouteNet-Gauss weights -> PyTorch state_dict")
    p.add_argument("inputs", nargs="*", help="checkpoint prefix(es) (e.g. ckpt/.../201-19.7350) or init_weights.npz")
    p.add_argument("--all-known", action="store_true", help="convert every GT/paper/replay checkpoint found in the repo")
    p.add_argument("--out", default=None, help="output path (single input only)")
    args = p.parse_args()
    inputs = list(args.inputs) + (known_checkpoints() if args.all_known else [])
    if not inputs:
        p.error("give checkpoint path(s) or --all-known")
    for path in inputs:
        out = convert(path, args.out if len(inputs) == 1 else None)
        n = sum(v.numel() for v in torch.load(out, weights_only=True).values())
        print(f"{path} -> {out} ({n} parameters)")


if __name__ == "__main__":
    main()
