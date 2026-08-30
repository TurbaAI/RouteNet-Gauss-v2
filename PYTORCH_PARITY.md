# PyTorch port of RouteNet-Gauss — parity report

Measured agreement between the PyTorch port and the frozen TensorFlow ground truth (GT) in
`tensorflow_version_gt/`. How the port was made and every semantic difference:
[PYTORCH_PORT.md](PYTORCH_PORT.md). Terms (cell, z-score step, exact replay, L0–L3, chaos
envelope): [README glossary](README.md#glossary). Raw reports: `pytorch_version_results/`.

All numbers below are from this machine (4-core CPU, RTX A4000; TF 2.15 CPU, torch 2.13.0+cu126).
_Sections marked ⏳ are filled in as the corresponding runs finish._

## 0. Is the TF ground truth itself reproducible? (the yardstick)

Before comparing frameworks, the GT was re-generated with TF to know what "same" can mean:

| check | result |
|---|---|
| re-run one epoch of `trex_multiburst/delay/seed_1` (same seeds, same threading) | checkpoint `01-110.4389` **34/34 tensors bit-identical**, `history.csv` identical |
| full 5×50 replay of all 8 quick cells (`tf_reference/replay_tf_run.py`) | **8/8 cells**: every per-epoch checkpoint bit-identical, `history.csv` identical, test predictions bit-identical, z-score fingerprint equal (`tensorflow_version_gt/replay/*/replay_check.json`) |
| same, but with `TF_NUM_INTRAOP_THREADS=1` | **0/34 tensors identical** after one epoch (`01-110.4414` vs `01-110.4389`); per-step losses within **2.8e-5** relative (median 1.0e-5) |

So TF reproduces itself bit-for-bit only under identical threading; a different summation order
already changes every weight while leaving the loss trajectory within ~3e-5. That is the **chaos
envelope** any float32 re-implementation is measured against. The cause is the model's numerical
conditioning at untrained weights (PYTORCH_PORT.md §6).

## 1. L0 — forward pass, identical weights, identical scenarios

`parity/l0_forward.py`: the frozen TF model and the PyTorch model, loaded with the same weights,
on every scenario of a test set. Gate: per-scenario max|TF − torch| / max|TF| ≤ 1e-4 and the
target tensors of the two data pipelines bit-identical.

| checkpoint | scenarios | predictions | targets identical | max abs diff | worst scale-rel diff | MAPE TF | MAPE torch | passed |
|---|--:|--:|---|--:|--:|--:|--:|---|
| converged trex_multiburst/delay seed 1 (`30-7.0663`), first 3 scenarios | 3 | 12 826 | True | 6.8e-9 | 2.1e-5 | 5.641187 | 5.641189 | True |
| ⏳ all 16 checkpoints (2 converged GT, 8 quick GT, 6 paper weights), full test sets | | | | | | | | `pytorch_version_results/parity/l0_summary.md` |

GPU (deterministic mode, TF32 off) vs CPU on the same scenario: outputs within 3.5e-9 absolute
(8e-6 relative), loss 6.488248 vs 6.488257, gradients within 6.8e-5. With TF32 *on* (PyTorch's
default for cuDNN on Ampere) the same comparison gives 8e-4 on outputs and 2 % on gradients —
hence TF32 is disabled in every run. Two identical GPU runs: outputs, loss and all gradients
bit-identical.

## 2. L1 — loss, gradients, one Adam step

`parity/l1_grad_step.py`, from identical weights on identical scenarios; gradients are judged
against a float64 reference (see PYTORCH_PORT.md §6 for why).

| weights | scenarios | loss rel diff (worst) | ill-conditioned tensors | torch fp32 vs fp64 (gated, worst) | TF fp32 vs torch fp32 (worst) | one-step update gate | passed |
|---|--:|--:|--:|--:|--:|---|---|
| ⏳ converged trex delay seed 1 | 5 | | | | | | |
| ⏳ converged trex jitter seed 1 | 5 | | | | | | |
| ⏳ TF initial weights, trex delay seed 1 | 3 | 2.5e-7 | 37/38 (float64 gradients up to 5.8e24) | n/a | n/a | n/a | loss only |
| ⏳ TF initial weights, mawi delay seed 1 | 3 | | | | | | |

Preliminary (converged delay, 5 scenarios, before the gate reformulation): loss agreed to
≤ 4.4e-6; float32 gradients of both frameworks within 1e-5–3e-3 of float64 on every tensor, and
on the one scenario where a tensor exceeded 1e-3 it was **TF's** float32 gradient (2.8e-3) that
was farther from float64 than torch's (1.8e-4).

## 3. L2 — exact replay of the TF training (TF init weights, TF order, TF z-scores)

### 3.1 First epoch, `trex_multiburst/delay/seed_1` (50 steps)

| pair | per-step loss rel diff: median | max | steps > 1e-4 | epoch mean loss | epoch val_loss |
|---|--:|--:|--:|--:|--:|
| TF (1 thread) vs TF GT | 1.02e-5 | 2.84e-5 | 0 / 50 | 111.90295 vs 111.90171 | 110.4414 vs 110.4389 |
| **torch exact replay vs TF GT** | **1.04e-5** | **3.37e-5** | **0 / 50** | 111.90309 vs 111.90171 | 110.4423 vs 110.4389 |
| torch vs TF (1 thread) | 9.4e-7 | 8.9e-6 | | | |

The torch trajectory sits inside TF's own threading envelope — and is 10× closer to
single-threaded TF than multi-threaded TF is to itself (both use sequential reductions).
Weights after 50 steps differ element-wise on the order of the total weight movement, as TF's own
do under a different thread count (⏳ TF-1-thread weight divergence measured for the table).
Files: `pytorch_version_results/parity/exact_replay_50steps/`.

### 3.2 ⏳ Quick 5×50 set, 8 cells (`torch_baseline`, `--replay-from`)

Gate: test MAPE within ±0.5 pt and R² within ±0.02 of the GT cell; per-step curves vs the TF
recordings reported (`compare_results.py`).

## 4. L3 — native PyTorch pipeline (torch init, torch shuffle)

### 4.1 ⏳ Quick 5×50 set, 8 cells (`torch_baseline_torchinit`)

Gate: MAPE within ±3 pt of the GT cell, R² within 2× the TF seed spread.

## 5. ⏳ Converged runs (`trex_multiburst` seed 1, delay and jitter)

| phase | init / order | delay: TF R² 0.728, MAPE 5.07 % | jitter: TF R² 0.887, MAPE 11.75 % |
|---|---|---|---|
| A | TF init, TF order (exact replay) | | |
| B | torch init, torch shuffle | | |
| C | seed 2 | | |

Gate (Phase A): R² within ±0.03, MAPE within ±1 pt. Phase B reported as "not worse".

## 6. ⏳ Speed

| setting | seconds / training step | notes |
|---|--:|---|
| TF 2.15, CPU (GT quick runs, 4 concurrent jobs) | ≈ 3 (incl. validation share) | from `train_seconds` |
| torch CPU, 1 thread, no contention | | |
| torch CUDA (deterministic, TF32 off) | | |
| torch CPU, 1 thread, 4 concurrent jobs | ≈ 8–10 (measured under heavy contention) | |
