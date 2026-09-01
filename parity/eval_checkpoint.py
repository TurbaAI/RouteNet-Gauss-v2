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

# Evaluate a saved PyTorch checkpoint on a dataset's test split, writing the same
# metrics.json / predictions.npz that experiment.py writes at the end of training.
#
# Why this exists: the converged runs keep training for days (early stopping defers every time
# the val_loss improves), but `--save-best-only` has already written their best checkpoint. This
# harvests the accuracy numbers from that checkpoint at any time WITHOUT touching the running
# job — so the comparison against the TensorFlow ground truth can be produced while training
# continues. Re-run it later to refresh the numbers if a better checkpoint appears.
#
# It is also the general "score this checkpoint" tool (paper weights, GT-converted checkpoints,
# any ckpt/**/<epoch>-<val_loss>.pt).
#
# Run in conda env RG_torch from the repo root (no TensorFlow needed):
#   python parity/eval_checkpoint.py \
#       --checkpoint ckpt/torch_converged/trex_multiburst/RouteNetGauss/delay/seed_1/81-5.8431.pt \
#       --z-scores  normalization/torch_converged/trex_multiburst/RouteNetGauss/delay/seed_1/z_scores.pkl \
#       --dataset trex_multiburst --target delay --out pytorch_version_results/converged/harvest/...
# Be kind to concurrent training jobs: `nice -n 10 ... --threads 1`.

import argparse
import json
import os
import pickle
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from experiment import PERCENTILES, _metrics, build_targets, concatenate_ds
from models import RouteNetGauss
from torch_ragged import sample_to_device
from utils import load_dataset, prepare_targets_and_mask


def main():
    p = argparse.ArgumentParser(description="Evaluate a PyTorch RouteNet-Gauss checkpoint on a test split.")
    p.add_argument("--checkpoint", required=True, help="ckpt/**/<epoch>-<val_loss>.pt")
    p.add_argument("--z-scores", required=True, help="z_scores.pkl used by that run")
    p.add_argument("--dataset", required=True)
    p.add_argument("--target", required=True, choices=["delay", "jitter"])
    p.add_argument("--partition", default="test")
    p.add_argument("--data-path", default="data_torch")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", required=True, help="directory for metrics.json + predictions.npz")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)

    targets, mask = build_targets(args.target)
    z_scores = pickle.load(open(args.z_scores, "rb"))
    # inference_mode=True clamps predictions at 0, as experiment.py does before predicting.
    model = RouteNetGauss(output_dim=len(targets), mask_field=mask, inference_mode=True,
                          use_trans_delay=args.target == "delay", z_scores=z_scores)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("z_") for k in missing), (missing, unexpected)
    model.to(device).eval()

    ds = load_dataset(f"{args.dataset}/{args.partition}", data_path=args.data_path).map(
        prepare_targets_and_mask(targets, mask))
    t0 = time.time()
    with torch.no_grad():
        y_pred = np.concatenate([model(sample_to_device(x, device)).cpu().numpy() for x, _ in ds], axis=0)
    y_true = concatenate_ds(ds)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    epoch_val = os.path.basename(args.checkpoint)[: -len(".pt")]
    out = {
        "dataset": args.dataset, "target": args.target, "seed": 1,
        "checkpoint": args.checkpoint, "checkpoint_name": epoch_val,
        "epoch": int(epoch_val.split("-")[0]) if "-" in epoch_val else None,
        "best_val_loss": float(epoch_val.split("-")[1]) if "-" in epoch_val else None,
        "n_test_predictions": int(n),
        "test_overall": _metrics(y_true.flatten(), y_pred.flatten()),
        "test_per_percentile": {name: _metrics(y_true[:, i], y_pred[:, i]) for i, name in enumerate(PERCENTILES)},
        "framework": "torch", "torch_version": torch.__version__, "device": str(device),
        "eval_seconds": round(time.time() - t0, 2),
        "harvested_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "evaluated from the best checkpoint of a still-running training job (see parity/eval_checkpoint.py)",
    }
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(args.out, "predictions.npz"), y_true=y_true, y_pred=y_pred)
    print(f"{args.dataset}/{args.target} {epoch_val}: MAPE {out['test_overall']['mape']:.4f} "
          f"MAE {out['test_overall']['mae_us']:.4f}us R2 {out['test_overall']['r2']:.4f} "
          f"(n={n}, {out['eval_seconds']:.0f}s) -> {args.out}")


if __name__ == "__main__":
    main()
