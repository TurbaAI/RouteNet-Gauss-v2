# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Reference implementation of **RouteNet-Gauss** (paper: *Hardware-Enhanced Network Modeling with Machine Learning*, IEEE ToN 2026): a Graph Neural Network predicting per-flow delay/jitter from traffic descriptors, as a surrogate for discrete-event simulation. Research code, no test suite / linter / CI.

This branch (`torch_branch`) is the **PyTorch port** of the original TensorFlow code, verified against the frozen TF results in `tensorflow_version_gt/`. Translation convention in every ported file: the original TF line stays as a `#TF:` comment, the PyTorch line follows it; untouched lines had no TF in them. Keep that convention when editing. The frozen TF originals are importable from `tf_reference/` (never edit them). `PYTORCH_PORT.md` documents every semantic difference; `PYTORCH_PARITY.md` the measured agreement; the README has a glossary (cell, z-score step, exact replay, L0–L3, chaos envelope).

## Environment & commands

Conda env **`RG_torch`** (`requirements-torch.txt`; torch must come from the `cu126` index — the PyPI wheel targets CUDA 13, this machine's driver is 12.8). The old TF env is `RG`; it is only needed to run `tf_reference/replay_tf_run.py`, which must use the exact TF binary that produced the GT. `tensorflow-cpu` in `RG_torch` serves `convert_data_to_torch.py`, `convert_tf_checkpoint.py` and `parity/`.

- One job: `python experiment.py --dataset trex_multiburst --target delay --seed 1 --epochs 5 --steps 50 --experiment-name <name>` (`--help` lists the PyTorch-only flags; `--replay-from tensorflow_version_gt/replay/<cell>` = exact TF replay; `--resume` continues from `results/.../resume.pt`).
- Matrix: `python run_experiments.py --experiment-name <name> ...` (2 datasets × 2 targets × 2 seeds, one subprocess per cell).
- Paper single-run config: `python train.py`. Evaluation: `evaluation_torch.ipynb` (`evaluation.ipynb` is the untouched TF original).
- Parity: `python parity/run_l0_all.py`, `python parity/l1_grad_step.py ...`, `python compare_results.py --torch results/<exp> ...`.
- Run from the repo root (all paths are relative). Long runs: launch with `setsid nohup ... &` — the machine has rebooted mid-run before, and background tasks die with the session.

## Runtime rules that are not obvious from the code

- **One torch thread per concurrent CPU job** (`--threads`; `run_experiments.py` sets it to cores // concurrency). Oversubscribing the 4 cores makes OpenMP spin: 10–100× slower.
- GPU runs are bit-reproducible only with `torch.use_deterministic_algorithms(True)` (default) and **TF32 off** (default; `--allow-tf32` breaks float32 parity with TF).
- Gradients of this model at untrained weights are numerically ill-conditioned (float64 magnitudes 1e8–1e24; float32 is off by orders of magnitude in TF and torch alike). Compare training runs by loss trajectories and test metrics, never by weight equality — see PYTORCH_PORT.md §6.
- Keras semantics that the loop reproduces on purpose: per-tensor `clipnorm`, Adam ε placement (`training_lib.KerasAdam`), loss means weighted by predictions per scenario, callback order. Do not "simplify" these to torch idioms without re-running `parity/`.
- `experiment_name="paper_weights"` writes into the shipped paper checkpoints; use any other name (`train.py` defaults to `torch_train`).

## Data & weights layout

Path template `{experiment_name}/{ds_name}/RouteNetGauss/{target}[/seed_n]` mirrored across `ckpt/`, `normalization/`, `results/`, `tensorboard/`.

- `data_torch/<dataset>/<partition>/<k>.pt.gz` — TF-free datasets, one file per TF shard, loaded only via `utils.load_dataset`. `data/` is the TF original (tf.data snapshots, needs TF). See `data_torch/README.md`.
- `ckpt/**/<epoch>-<val_loss>.pt` — torch state_dicts; the TF checkpoints of the paper/GT have a converted `.pt` next to them (`convert_tf_checkpoint.py --all-known`).
- `tensorflow_version_gt/` — frozen TF ground truth (quick 8-cell baseline + 2 converged cells) and `replay/` (per cell: TF initial weights, 150k-step scenario order, per-step losses, bit-identity proofs). `pytorch_version_results/` mirrors it for the PyTorch side.
- Working outputs `results/`, `ckpt/torch_*`, `normalization/torch_*`, `tensorboard/` are gitignored; committed results live only under `pytorch_version_results/`.
