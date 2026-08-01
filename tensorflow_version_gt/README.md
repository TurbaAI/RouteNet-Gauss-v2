# TensorFlow ground-truth baseline (`tensorflow_version_gt`)

Frozen TensorFlow results used as the reference ("ground truth") for the later PyTorch port
comparison. This folder holds **two** sets of TF results, for two different purposes:

| set | location | config | use it for |
|---|---|---|---|
| **Converged** (primary GT) | `converged/` | 300 epochs × 500 steps, early-stop (patience 15, best weights restored) | comparing PyTorch **accuracy** — these are properly trained models |
| **Quick parity baseline** | top level (`results/`, `ckpt/`, `summary.*`) | 5 epochs × 50 steps | comparing PyTorch **numerics/plumbing** under an identical short config across all 8 cells |

All runs: RouteNet-Gauss, Adam(lr=1e-3, clipnorm=1.0), MAPE loss, CPU backend (this
dynamic-shape GNN retraces per topology and runs *slower* on GPU), W&B project
`salehgh/routenet-gauss-tf-baseline`. Metrics = MAPE (%), MAE (µs), R² vs the test targets,
overall and per percentile (avg/p50/p90/p95/p99).

## `converged/` — the real ground truth (use this for accuracy comparison)

Only **`trex_multiburst`, seed 1** was trained to convergence so far (delay ~21 h, jitter ~70 h
on CPU). These are the models a PyTorch port should be benchmarked against:

| dataset | target | seed | test MAPE | test MAE | test R² | epochs (early-stop) |
|---|---|---|--:|--:|--:|--:|
| trex_multiburst | delay | 1 | 5.07% | 6.70 µs | **0.728** | 45 |
| trex_multiburst | jitter | 1 | 11.75% | 1.88 µs | **0.887** | 193 |

Contents: `converged/results/.../{metrics.json, predictions.npz, history.csv}`,
`converged/ckpt/.../` (best checkpoint only), `converged/normalization/.../z_scores.pkl`,
`converged/summary.{csv,json}`.

**Coverage gap:** `mawi_pcaps` (all seeds) and `trex_multiburst` seed 2 are **not** converged
yet — only the quick baseline exists for them. To fill the matrix, run e.g.:
`python run_experiments.py --experiment-name tf_converged --datasets mawi_pcaps --targets delay,jitter --seeds 1,2 --epochs 300 --steps 500 --patience 15 --save-best-only --force-cpu`

## Top level — quick 5×50 parity baseline (all 8 cells)

Intentionally **un-converged** (MAPE ~85–110%, negative R²): a reproducible parity harness, not
trained models. All 8 cells (datasets {mawi_pcaps, trex_multiburst} × targets {delay, jitter} ×
seeds {1, 2}). Re-generate with:
`python run_experiments.py --experiment-name tf_baseline --epochs 5 --steps 50 --force-cpu --cpu-concurrency 4`

- `results/<ds>/RouteNetGauss/<target>/seed_<n>/` — metrics.json, predictions.npz (y_true,y_pred), history.csv
- `ckpt/...` — weights per job · `normalization/...` — z_scores.pkl per job
- `summary.csv`, `summary.json` — aggregated 8-row table
