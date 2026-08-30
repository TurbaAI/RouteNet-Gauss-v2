# PyTorch results snapshot (`pytorch_version_results`)

Frozen PyTorch-side results, mirroring `tensorflow_version_gt/` (the TensorFlow ground truth they
are compared against). Interpretation and verdicts: [PYTORCH_PARITY.md](../PYTORCH_PARITY.md);
terms: [README glossary](../README.md#glossary).

| location | content |
|---|---|
| `parity/l0_*.json`, `parity/l0_summary.{md,json}` | L0 forward-pass parity per checkpoint (`parity/run_l0_all.py`) |
| `parity/l1_*.json` | L1 loss / gradient / one-step parity reports (`parity/l1_grad_step.py`) |
| `parity/exact_replay_50steps/` | first-epoch exact replay of `trex_multiburst/delay/seed_1`: torch per-step losses + history, and the TF single-thread re-run used as the chaos-envelope reference |
| `quick/` | the two quick 5×50 8-cell sets: `torch_baseline` (exact replay) and `torch_baseline_torchinit` (native), each with `results/` (metrics.json, predictions.npz, history.csv, step_losses.csv, sample_order_used.npy per cell), `ckpt/` (final epoch), `normalization/`, `summary.{csv,json}`, and the `compare_results.py` tables |
| `converged/` | the converged `trex_multiburst` runs (Phase A exact replay, Phase B native, Phase C seed 2): per cell metrics.json, predictions.npz, history.csv, best checkpoint, z-scores; comparison tables |

Everything here is regenerable from the commands recorded in each `metrics.json`
(`experiment.py` flags) and in PYTORCH_PARITY.md; the working copies live in the gitignored
`results/`, `ckpt/torch_*`, `normalization/torch_*`.
