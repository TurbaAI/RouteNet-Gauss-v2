"""Q10 diagnosis: what distinguishes initial weights that leave the delay plateau early from those
that leave it late? At init the prediction is ~pkt_size*sum(1/capacity) (the queueing term
occupancy/capacity is ~1e-9 s), i.e. the val~86.7 plateau; escaping means growing the readout's
occupancy output by orders of magnitude. For each init: the loss at init and the queueing term's
share of the target on K training scenarios, plus (checks the reconstruction) the loss on the
run's first training scenario against its step_losses.csv row 0.

  python results/verification/diag_init_stats.py [--k 12]
"""
import argparse
import csv
import os
import pickle
import sys

ROOT = "/home/ubuntu/code/RouteNet-Gauss-v2"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import torch

from models import RouteNetGauss
from training_lib import keras_mape_loss
from utils import load_dataset, prepare_targets_and_mask

C = "trex_multiburst/RouteNetGauss/delay/seed_{s}"
R = "tensorflow_version_gt/replay/" + C
TARGETS = [f"flow_{p}_delay" for p in ["avg", "p50", "p90", "p95", "p99"]]
MASK = "flow_has_delay"


def build(init, seed=None, weights=None):
    torch.manual_seed(seed if seed is not None else 0)
    z = pickle.load(open(R.format(s=1) + "/z_scores.pkl", "rb"))
    m = RouteNetGauss(output_dim=5, mask_field=MASK, use_trans_delay=True, z_scores=z, init=init)
    if weights:
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith("z_") for k in missing)
    return m.train()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=12)
    a = p.parse_args()
    torch.set_num_threads(1)
    ds = load_dataset("trex_multiburst/training").map(prepare_targets_and_mask(TARGETS, MASK))
    by_idx = {int(x["sample_idx"]): (x, y) for x, y in ds}
    probe_idx = sorted(by_idx)[:: max(1, len(by_idx) // a.k)][: a.k]

    inits = [  # label, how to build, plateau exit (0-based) observed, run dir whose step 0 checks the build
        ("TF-drawn s1", dict(init="keras", weights=R.format(s=1) + "/init_weights.pt"), "2 / 5 / 6 / 7",
         "pytorch_version_results/converged/runs/torch_converged/" + C.format(s=1)),
        ("TF-drawn s2", dict(init="keras", weights=R.format(s=2) + "/init_weights.pt"), "4 (TF)", None),
        ("torch keras s1", dict(init="keras", seed=1), "20 (and >=8 with TF order)", "results/torch_converged_native/" + C.format(s=1)),
        ("torch keras s2", dict(init="keras", seed=2), "7", "results/torch_plateau_probe/" + C.format(s=2)),
        ("torch keras s3", dict(init="keras", seed=3), ">=13 (running)", "results/torch_plateau_probe/" + C.format(s=3)),
        ("torch default s1", dict(init="torch", seed=1), "20",
         "pytorch_version_results/converged/runs/torch_converged_torchinit/" + C.format(s=1)),
    ]
    print(f"{len(by_idx)} training scenarios, probing {len(probe_idx)}")
    print(f"{'init':18s} {'exit':>26s} {'loss@init':>9s} {'queue share':>12s} {'queue>0':>8s} {'readout out mean':>16s} {'step0 check':>24s}")
    for label, kw, exit_s, rdir in inits:
        m = build(**kw)
        captured = {}
        h = m.readout_path.register_forward_hook(lambda mod, inp, out: captured.__setitem__("occ", out.detach()))
        losses, shares, pos, occ_means = [], [], [], []
        with torch.no_grad():
            for i in probe_idx:
                x, y = by_idx[i]
                pred = m(x)
                occ_means.append(float(captured["occ"].mean()))
                # trans-only prediction: same forward with the readout output zeroed
                h2 = m.readout_path.register_forward_hook(lambda mod, inp, out: out * 0)
                trans = m(x)
                h2.remove()
                q = (pred - trans) / y
                losses.append(float(keras_mape_loss(y, pred)))
                shares.append(float(q.mean()))
                pos.append(float((pred - trans > 0).float().mean()))
            check = ""
            if rdir and os.path.exists(os.path.join(rdir, "step_losses.csv")):
                row0 = next(csv.DictReader(open(os.path.join(rdir, "step_losses.csv"))))
                x, y = by_idx[int(row0["sample_idx"])]
                check = f"{float(keras_mape_loss(y, m(x))):.4f} vs {float(row0['loss']):.4f}"
        h.remove()
        print(f"{label:18s} {exit_s:>26s} {np.mean(losses):9.3f} {np.mean(shares):12.3e} {np.mean(pos):8.3f} "
              f"{np.mean(occ_means):16.4f} {check:>24s}")


if __name__ == "__main__":
    main()
