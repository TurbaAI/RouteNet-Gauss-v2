"""Q10 diagnosis, part 2: is the gradient overflow that killed §7.6 job pt_s9 (epoch 7, step 281)
inherent to the model — would ANY float32 implementation, TF included, overflow — or a float32
artefact? Replays epoch 7 from resume.pt exactly as diag_nan_replay.py, and for steps 276-281 also
computes the float64 reference gradient at the same weights (same method as parity/l1_grad_step.py):
if the float64 gradient itself exceeds float32's range (3.4e38), every float32 implementation
overflows there.

  python results/verification/diag_nan_fp64.py
"""
import copy
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
from torch_ragged import Ragged, sample_to_device
from training_lib import KerasAdam, keras_mape_loss
from utils import load_dataset, prepare_targets_and_mask

RUN = "results/torch_initdraw_pt_s9/trex_multiburst/RouteNetGauss/delay/seed_9"
ZS = "normalization/torch_converged_native/trex_multiburst/RouteNetGauss/delay/seed_1/z_scores.pkl"
TARGETS = [f"flow_{p}_delay" for p in ["avg", "p50", "p90", "p95", "p99"]]
MASK = "flow_has_delay"
EPOCH, STEPS, FIRST64, LAST = 7, 500, 276, 281
F32_MAX = float(np.finfo(np.float32).max)


def cast_sample(x, dt):  # parity/l1_grad_step.py
    out = {}
    for k, v in x.items():
        if isinstance(v, Ragged):
            out[k] = v.with_values(v.values.to(dt)) if v.values.is_floating_point() else v
        elif isinstance(v, torch.Tensor) and v.is_floating_point():
            out[k] = v.to(dt)
        else:
            out[k] = v
    return out


def fp64_grads(model32, z, x, y, dev):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        m = RouteNetGauss(output_dim=5, mask_field=MASK, use_trans_delay=True, z_scores=z, init="keras")
        m.load_state_dict({k: v.detach().to(torch.float64) for k, v in model32.state_dict().items()})
        m.to(dev).train()
        loss = keras_mape_loss(y.to(dev).to(torch.float64), m(sample_to_device(cast_sample(x, torch.float64), dev)))
        loss.backward()
    finally:
        torch.set_default_dtype(prev)
    return float(loss.detach()), {n: p.grad.detach() for n, p in m.named_parameters() if p.grad is not None}


def top(grads):
    mx = {n: float(g.abs().max()) if torch.isfinite(g).all() else float("inf") for n, g in grads.items()}
    n = max(mx, key=mx.get)
    return n, mx[n]


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    by_idx = {int(x["sample_idx"]): (x, y) for x, y in load_dataset("trex_multiburst/training")}
    fn = prepare_targets_and_mask(TARGETS, MASK)
    order = np.load(f"{RUN}/sample_order_used.npy")
    z = pickle.load(open(ZS, "rb"))
    model = RouteNetGauss(output_dim=5, mask_field=MASK, use_trans_delay=True, z_scores=z, init="keras")
    state = torch.load(f"{RUN}/resume.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(dev)
    opt = KerasAdam(model.parameters(), lr=0.001, clipnorm=1.0)
    opt.load_state_dict(state["optimizer"])
    print(f"{'step':>4s} {'sample':>6s} {'loss f32':>9s} {'loss f64':>9s}  {'largest |grad| float32':>40s}  {'largest |grad| float64':>40s}")
    for step in range(LAST + 1):
        sidx = int(order[EPOCH * STEPS + step])
        x, y = fn(*by_idx[sidx])
        if step >= FIRST64:
            l64, g64 = fp64_grads(model, z, x, y, dev)
        xd, yd = sample_to_device(x, dev), y.to(dev)
        loss = keras_mape_loss(yd, model(xd))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step >= FIRST64:
            g32 = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
            n32, m32 = top(g32)
            n64, m64 = top(g64)
            print(f"{step:4d} {sidx:6d} {float(loss.detach()):9.3f} {l64:9.3f}  {n32:>28s} {m32:11.3e}  {n64:>28s} {m64:11.3e}"
                  f"{'   <- float64 beyond float32 range' if m64 > F32_MAX else ''}", flush=True)
        opt.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            print(f"weights non-finite after step {step}")
            break


if __name__ == "__main__":
    main()
