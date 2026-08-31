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

# Verifies the one part of evaluation_torch.ipynb that the L0/L1 harness does NOT cover:
# `concatenate_ds_with_donor_mask`, which builds the OMNeT++ ("simulated") column by selecting
# windows of the *_simulated test set with the flow_has_<metric> mask of the *testbed* set.
# Everything else in the notebook is either the model forward pass (covered by parity/l0_forward.py)
# or the shared metric functions.
#
# Run in conda env RG_torch from the repo root (needs tensorflow-cpu + torch, no model inference):
#   python parity/check_notebook_eval.py
# Expected: "bit-identical: True" for all six (dataset, metric) combinations.

import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/home/ubuntu/code/RouteNet-Gauss-v2")
import numpy as np, tensorflow as tf, torch
from tf_reference.utils import load_dataset as tf_load, prepare_targets_and_mask as tf_prep, seg_to_global_reshape as tf_seg
from utils import load_dataset, prepare_targets_and_mask, seg_to_global_reshape

def tf_donor(ds, mask_ds, metric):                      # verbatim from evaluation.ipynb
    res = []
    targets = [f"flow_{p}_{metric}" for p in ["avg","p50","p90","p95","p99"]]; mask = f"flow_has_{metric}"
    for x, _ in iter(ds):
        mask_field = mask_ds[x["sample_idx"].numpy()][0][mask]
        reshaped_mask = tf.expand_dims(tf_seg(mask_field, num_dims=2), 1)
        val = tf.concat([tf.reshape(tf.boolean_mask(tf_seg(x[t]), reshaped_mask), (-1,1)) for t in targets], axis=1)
        res.append(val.numpy())
    total = np.concatenate(res, axis=0); total[total <= 0] = 0
    return total

def torch_donor(ds, mask_ds, metric):                   # from evaluation_torch.ipynb
    res = []
    targets = [f"flow_{p}_{metric}" for p in ["avg","p50","p90","p95","p99"]]; mask = f"flow_has_{metric}"
    for x, _ in iter(ds):
        mask_field = mask_ds[int(x["sample_idx"])][0][mask]
        reshaped_mask = seg_to_global_reshape(mask_field, num_dims=2).unsqueeze(1)
        val = torch.cat([seg_to_global_reshape(x[t])[reshaped_mask].reshape(-1,1) for t in targets], dim=1)
        res.append(val.numpy())
    total = np.concatenate(res, axis=0); total[total <= 0] = 0
    return total

for true_name, sim_name in [("mawi_pcaps","mawi_pcaps_simulated"), ("trex_multiburst","trex_multiburst_simulated"), ("trex_synthetic","trex_synthetic_simulated")]:
    for metric in ["delay","jitter"]:
        tg = [f"flow_{p}_{metric}" for p in ["avg","p50","p90","p95","p99"]]; mk = f"flow_has_{metric}"
        a = tf_donor(tf_load(f"{sim_name}/test", data_path="data").map(tf_prep(tg, mk)),
                     {x["sample_idx"].numpy(): (x, y) for x, y in tf_load(f"{true_name}/test", data_path="data").map(tf_prep(tg, mk))}, metric)
        b = torch_donor(load_dataset(f"{sim_name}/test").map(prepare_targets_and_mask(tg, mk)),
                        {int(x["sample_idx"]): (x, y) for x, y in load_dataset(f"{true_name}/test").map(prepare_targets_and_mask(tg, mk))}, metric)
        same = a.shape == b.shape and np.array_equal(a, b)
        print(f"{true_name}/{metric}: shape {a.shape} vs {b.shape} | bit-identical: {same}" +
              ("" if same else f" | max|diff| {np.max(np.abs(a-b)):.3e}"), flush=True)
