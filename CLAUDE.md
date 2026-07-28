# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Reference implementation of **RouteNet-Gauss** (paper: *Hardware-Enhanced Network Modeling with Machine Learning*, IEEE ToN 2026). It is a TensorFlow/Keras Graph Neural Network that predicts per-flow network performance metrics (delay, jitter) from traffic descriptors, as a fast surrogate for Discrete Event Simulation. This is research code accompanying a paper, not a packaged library — there is no test suite, linter, or build step.

## Environment & commands

`pyproject.toml` lists no dependencies; the real requirement is a specific TensorFlow. Set up per the README:

```bash
virtualenv -p python3 myenv && source myenv/bin/activate
pip install tensorflow==2.11.1 numpy==1.24.2 notebook==7.0.7   # or tensorflow==2.15.0 / numpy==1.26.3
```

- **Train:** `python train.py` (must be run from repo root — dataset/checkpoint paths are relative).
- **Evaluate:** `jupyter notebook evaluation.ipynb` and run the cells top to bottom.
- All entry points force **CPU-only** execution via `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` at the top of the file. Remove/change that line to use a GPU.
- There are no tests, no lint config, and no CI. Note: the branch is named `torch_branch` but the code is entirely TensorFlow.

## Architecture

Three source files, plus data/weights directories.

- **`models.py`** — the `RouteNetGauss` Keras model. It runs iterative **message passing** (default 8 iterations) over four entity types, each with a 32-dim GRU-updated state: **flow** (a.k.a. path), **link**, **queue**, and **node**. Per message-passing iteration the order is link+queue→flow (RNN over the flow's sequence of hops), flow+node→queue, queue→link, queue→node. A path readout MLP produces per-hop occupancy, which is divided by link capacity and summed to a queueing delay; when `use_trans_delay=True` (used for the `delay` target) transmission delay computed from packet size and capacity is added. The model consumes a dict of graph-topology index tensors (`path_to_link`, `link_to_path`, `queue_to_link`, `node_groupings`, ...) that encode the network structure — the model is built dynamically per scenario topology, which is what lets it generalize to unseen/larger networks. Inputs `flow_traffic` and `flow_packets` are z-score normalized inside `call` using the `z_scores` dict passed at construction (`RouteNetGauss.z_scores_fields` lists the required keys). `inference_mode=True` clamps predictions to be non-negative (used only at eval, hurts training).

- **`utils.py`** — `load_dataset(name, data_path="data")` loads a dataset partition; `prepare_targets_and_mask(targets, mask)` returns a `.map()` function that flattens the (segment, flow) target tensors and applies the validity mask; `seg_to_global_reshape` transposes segment-major tensors to flow-major.

- **`train.py`** — training script. The top ~200 lines define reusable metrics (`get_positional_mape`, `get_positional_r2`) and `get_z_scores_dict` (computes/caches normalization stats to `normalization/...`). The bottom half (from ~line 204) is the **experiment config you edit**: `ds_name`, `target` (`delay`/`jitter`), the five `targets` (avg + p50/p90/p95/p99 percentiles), optimizer, loss, and `model.fit` params. `RUN_EAGERLY` (line 36) enables eager mode for debugging; `RELOAD_WEIGHTS` (line 39) resumes from the latest checkpoint.

### Data & weights layout (path convention matters)

Everything is keyed by the path template **`{experiment_name}/{ds_name}/RouteNetGauss/{target}`**, mirrored across three top-level dirs:
- `data/<dataset>/<partition>/<shard-index>/...` — datasets are `tf.data.Dataset` saved with GZIP compression, **split into numbered shards** to stay under Git size limits. Always load via `load_dataset` (it concatenates shards `0..N`); do not open a partition directory directly. Partitions are `training`/`validation`/`test`. See the README "Datasets information" section for what each dataset (`mawi_pcaps`, `trex_multiburst`, `trex_synthetic`, their `_filtered` and `_simulated` variants) represents.
- `ckpt/.../{target}/` — checkpoint weights (`save_weights_only=True`); filenames encode `{epoch}-{val_loss}`.
- `normalization/.../{target}/z_scores.pkl` — pickled z-score stats matching the checkpoint.

`evaluation.ipynb` reconstructs a model with `load_model(model_id, checkpoint, metric)` (builds the model, loads the matching `z_scores.pkl`, then `load_weights`), and compares model predictions against the OMNeT++ `_simulated` datasets using MAPE / MAE / R² (see `evaluate_model_vs_sim`).

## Known caveats in `train.py` (as shipped)

If you run `train.py` unmodified it will not work — treat these as things to fix when adapting it:
- Line 23 `from data import load_dataset` is spurious (`data/` is a directory, not a module) and will raise `ImportError`; the working import is line 24 `from utils import ...`. Remove line 23.
- `get_z_scores_dict` uses `pickle` (caching / `check_existing`) but `pickle` is never imported — add `import pickle`.
- The default `ds_name = "data_mawi_pcaps"` does not match any folder under `data/` (e.g. it should be `mawi_pcaps`). Set `ds_name` to an actual dataset directory name.
