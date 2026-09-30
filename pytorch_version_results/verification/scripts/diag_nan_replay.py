"""Q10 diagnosis of the §7.6 job pt_s9 ('Batch 282: Invalid loss' in epoch 7): replay epoch 7 from
the run's resume.pt (state at the end of epoch 6) with the same data, device and determinism, and
report the first non-finite value — loss, a gradient tensor before clipping, or a weight after
the update — plus the gradient magnitudes leading up to it.

  python results/verification/diag_nan_replay.py
"""
import math
import os
import pickle
import sys

ROOT = "/home/ubuntu/code/RouteNet-Gauss-v2"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import torch

from models import RouteNetGauss
from training_lib import KerasAdam, keras_mape_loss
from torch_ragged import sample_to_device
from utils import load_dataset, prepare_targets_and_mask

RUN = "results/torch_initdraw_pt_s9/trex_multiburst/RouteNetGauss/delay/seed_9"
ZS = "normalization/torch_converged_native/trex_multiburst/RouteNetGauss/delay/seed_1/z_scores.pkl"
TARGETS = [f"flow_{p}_delay" for p in ["avg", "p50", "p90", "p95", "p99"]]
MASK = "flow_has_delay"
EPOCH, STEPS, UNTIL = 7, 500, 300


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    ds = load_dataset("trex_multiburst/training")
    by_idx = {int(x["sample_idx"]): (x, y) for x, y in ds}
    fn = prepare_targets_and_mask(TARGETS, MASK)
    order = np.load(f"{RUN}/sample_order_used.npy")
    z = pickle.load(open(ZS, "rb"))
    model = RouteNetGauss(output_dim=5, mask_field=MASK, use_trans_delay=True, z_scores=z, init="keras")
    state = torch.load(f"{RUN}/resume.pt", map_location="cpu", weights_only=False)
    print("resume.pt epoch:", state["epoch"], "| optimizer iterations:", state["optimizer"].get("iterations"))
    model.load_state_dict(state["model"])
    model.to(dev)
    opt = KerasAdam(model.parameters(), lr=0.001, clipnorm=1.0)
    opt.load_state_dict(state["optimizer"])
    names = [n for n, _ in model.named_parameters()]
    maxg_hist = []
    for step in range(UNTIL):
        g = EPOCH * STEPS + step
        sidx = int(order[g])
        x, y = fn(*by_idx[sidx])
        x, y = sample_to_device(x, dev), y.to(dev)
        y_pred = model(x)
        loss = keras_mape_loss(y, y_pred)
        if not math.isfinite(float(loss)):
            bad = (~torch.isfinite(y_pred)).sum().item()
            print(f"step {step} (global {g}, sample {sidx}): LOSS non-finite ({float(loss)}); {bad}/{y_pred.numel()} "
                  f"non-finite predictions; weights finite: {all(torch.isfinite(p).all() for p in model.parameters())}")
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        per = {n: p.grad for n, p in zip(names, model.parameters()) if p.grad is not None}
        maxabs = {n: float(t.abs().max()) for n, t in per.items()}
        nonfin = [n for n, t in per.items() if not torch.isfinite(t).all()]
        top = max(maxabs, key=maxabs.get)
        maxg_hist.append((step, float(loss), top, maxabs[top]))
        if nonfin:
            print(f"step {step} (global {g}, sample {sidx}): loss {float(loss):.3f} finite, NON-FINITE GRADIENT in {nonfin}")
            for n in nonfin:
                t = per[n]
                print(f"   {n}: {int((~torch.isfinite(t)).sum())} non-finite of {t.numel()}; finite max |g| "
                      f"{float(t[torch.isfinite(t)].abs().max()) if torch.isfinite(t).any() else float('nan'):.3e}")
            opt.step()
            print(f"   after the (clipped) update, non-finite weights in: "
                  f"{[n for n, p in model.named_parameters() if not torch.isfinite(p).all()]}")
            break
        opt.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            print(f"step {step}: weights non-finite after a finite-gradient update")
            break
    print("largest |gradient| over the last steps before that (step, loss, tensor, max|g|):")
    for s, l, n, m in maxg_hist[-12:]:
        print(f"   {s:3d}  loss {l:8.3f}  {n:28s} {m:.3e}")
    print("overall largest |gradient| seen:", max(maxg_hist, key=lambda r: r[3]))


if __name__ == "__main__":
    main()
