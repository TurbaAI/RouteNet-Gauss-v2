# PyTorch port of RouteNet-Gauss — translation notes

This document describes *how* the TensorFlow/Keras implementation was translated to PyTorch and
every place where the two versions differ in semantics. The measured agreement between the two
is in [PYTORCH_PARITY.md](PYTORCH_PARITY.md); the words used here ("cell", "z-score step",
"exact replay", "L0–L3", …) are defined in the [README glossary](README.md#glossary).

## 1. Layout

| | TensorFlow (frozen reference) | PyTorch (current code) |
|---|---|---|
| model | `tf_reference/models.py` | `models.py` (+ `torch_ragged.py`) |
| data pipeline | `tf_reference/utils.py`, datasets in `data/` (tf.data snapshots) | `utils.py`, datasets in `data_torch/` (see `data_torch/README.md`) |
| training helpers | `tf_reference/training_lib.py` | `training_lib.py` |
| one training job | `experiment.py @ 2e30d5d` (git history) | `experiment.py` |
| job matrix | `run_experiments.py @ 2e30d5d` | `run_experiments.py` |
| single-run script | `train.py @ 2e30d5d` | `train.py` |
| GPU environment | `gpu_setup.py @ 2e30d5d` | `gpu_setup.py` (all no-ops) |
| evaluation notebook | `evaluation.ipynb` (unchanged, TF) | `evaluation_torch.ipynb` |
| weights | `ckpt/**/<epoch>-<val_loss>` (TF checkpoint) | `ckpt/**/<epoch>-<val_loss>.pt` (state_dict); `convert_tf_checkpoint.py` converts TF → torch |

Reading convention inside the translated files: every original TensorFlow line is kept as a
comment prefixed `#TF:` and its PyTorch translation follows directly below; lines without
TensorFlow in them are untouched. Blocks that map to a differently shaped PyTorch block carry a
short explanatory comment. The environment is the conda env `RG_torch`
(`requirements-torch.txt`); TensorFlow (`tensorflow-cpu`) is only needed for the one-time data
and checkpoint conversion and for the side-by-side parity harness in `parity/`.

## 2. Op-by-op mapping (`models.py`)

| TensorFlow / Keras | PyTorch | notes |
|---|---|---|
| `tf.keras.Model` | `torch.nn.Module` | `call(inputs)` → `forward(inputs)`; no `@tf.function`, PyTorch runs eagerly (no per-topology retracing) |
| `tf.keras.layers.Dense(u, activation=relu)` | `nn.Linear(in, u)` + `nn.ReLU()` | Keras kernel `[in, out]` = torch `weight.T`; Keras `layer_with_weights-k` ↔ `nn.Sequential` index `2k` |
| `tf.keras.layers.Input(shape=…)` | — | Keras infers input sizes at build time; torch needs them in the constructor |
| `tf.keras.layers.GRUCell(H)` (`reset_after=True`) | `nn.GRUCell(in, H)` | identical equations (`h' = z·h + (1−z)·tanh(W_h x + b_h + r·(U_h h + b_uh))`); Keras gate order `z,r,h` ↔ torch `r,z,n`; Keras bias `[2,3H]` ↔ torch `bias_ih`, `bias_hh` |
| `tf.keras.layers.RNN(GRUCell, return_sequences, return_state)` on a ragged input | `nn.GRU(in, H, batch_first=True)` via `torch_ragged.run_gru_over_ragged` | `flow_update` is only ever used through the RNN wrapper, so its torch form is `nn.GRU` (same parameter layout, `_l0` suffix). Keras masks rows past their length; a GRU is causal, so the state after the last real hop is the output at position `length−1` — no per-step masking needed |
| `tf.keras.layers.RNN(GRUCell)` on `queue_to_link` `[n_links, 1]` | one `nn.GRUCell` step | the sequence length is always 1 |
| `tf.RaggedTensor` | `torch_ragged.Ragged` (`values` + int64 `row_splits`, or `padded` + `row_lengths`, converted lazily) | same layout as TF; only `ragged_rank=1` is used by the model |
| `tf.gather(params, ragged_idx)` | `ragged_gather` | |
| `tf.gather_nd(ragged_seq, ragged_pairs)` | `ragged_gather_nd(padded, ragged_pairs)` | `(row, position)` pairs index the padded form |
| `tf.reduce_sum(ragged, axis=1)` | `ragged_reduce_sum` (`index_add`) | deterministic on CUDA under `torch.use_deterministic_algorithms(True)` |
| `tf.concat([expand_dims(prev,1), ragged], 1)` | `ragged_prepend` | |
| `ragged[:, 1:].to_tensor()` | `Ragged.inner_slice_from(1).to_padded()` | |
| `tf.RaggedTensor.from_tensor(dense, lengths)` | `Ragged.from_padded` | |
| `tf.one_hot(x, 3)` | `F.one_hot(x.long(), 3).to(dtype)` | |
| `tf.tile(expand_dims(c,1), [1,S,1])` | `c.unsqueeze(1).repeat(1,S,1)` | |
| `tf.transpose(x, [1,0,2])` | `x.permute(1,0,2)` | |
| `tf.boolean_mask(x, m)` | `x[m]` | |
| `tf.keras.activations.relu` | `torch.relu` | |
| `tf.zeros`, `tf.concat`, `tf.squeeze`, `tf.cast` | `torch.zeros`, `torch.cat`, `.squeeze`, `.to(dtype)` | |
| int32 index tensors | `.long()` at the top of `forward` | data stays int32 on disk (as in TF); torch indexing needs int64 |

Initialisation: `RouteNetGauss(init="torch")` (default) keeps PyTorch's defaults;
`init="keras"` re-initialises with the Keras defaults RouteNet-Gauss was trained with
(glorot-uniform kernels, orthogonal recurrent kernels, zero biases) — used by the TF-comparison
runs. Both only fix the *distribution*; the exact-replay experiments load TF's actual initial
weights (`tensorflow_version_gt/replay/**/init_weights.npz`) through the same converter.

## 3. Data pipeline (`utils.py`, `data_torch/`)

`tf.data.Dataset` chains (`load_dataset(...).map(...).shuffle(...).repeat()`) are provided by
`utils.ListDataset`, an in-memory list with the same chainable API:

- `shuffle(buffer, seed, reshuffle_each_iteration=True)` implements tf.data's **buffered
  shuffle** (fill a buffer of `buffer` elements, emit a uniformly random slot, refill, drain at
  the end, reshuffle on every pass) with one seeded `torch.Generator` per `shuffle()` call that is
  shared by every iterator created from the dataset — exactly tf.data's seed-generator
  bookkeeping (the z-score pass and the training pass see different permutations).
- **The random numbers differ from TF's**, so a torch shuffle never reproduces a TF order. Runs
  that must see the TF order load it (`--sample-order`, recorded by
  `tf_reference/replay_tf_run.py`).
- The training order is materialised up front (`results/**/sample_order_used.npy`), which makes
  every run auditable and resumable.
- `prepare_targets_and_mask` and `seg_to_global_reshape` are line-by-line translations; the
  target tensors they produce are bit-identical to TF's for every scenario of every partition
  (checked by `parity/l0_forward.py`).
- Datasets: `data_torch/` is a lossless conversion of `data/` (same scenarios, same shard
  numbering, `flow_packets_per_ms` dropped by default), see `data_torch/README.md`.

## 4. Training loop (`training_lib.fit`, `experiment.py`)

Keras' `compile/fit` has no PyTorch counterpart, so the loop is explicit and reproduces Keras'
arithmetic and bookkeeping, each point verified against the TF replay recordings:

| Keras | PyTorch | verified how |
|---|---|---|
| `MeanAbsolutePercentageError()` | `training_lib.keras_mape_loss`: `100·mean(|y−ŷ| / max(|y|, 1e-7))` | loss agrees to ≤ 5e-6 relative on every scenario tested (L1) |
| `Adam(learning_rate=1e-3, clipnorm=1.0)` | `training_lib.KerasAdam` | see §5.3; one-step update agreement gated in L1 |
| `logs["loss"]` per epoch | `MeanTracker` weighted by the number of predictions of each scenario | Keras weights its loss `Mean` by the batch dimension; reproduces the recorded running mean to 2e-5 (unweighted would be off by 0.2–0.9) |
| compiled metrics per epoch | `MeanTracker`, unweighted | Keras `MeanMetricWrapper` counts one value per step |
| `val_loss` | prediction-count-weighted mean over the validation set (= the global MAPE over all validation predictions) | same as TF's `evaluate` |
| `history.csv` (CSVLogger) | same columns (`epoch` + sorted keys), no `lr` column | `LearningRateLogger` runs after `CSVLogger` in the TF callback list, so TF's file has no `lr` either |
| `ModelCheckpoint(save_weights_only, monitor=val_loss, mode=min[, save_best_only])` | `KerasModelCheckpoint` → `ckpt/**/{epoch:02d}-{val_loss:.4f}.pt` | verbatim logic |
| `ReduceLROnPlateau(factor=.5, patience=10, cooldown=3, monitor=loss, min_delta=1e-4)` | `KerasReduceLROnPlateau` | Keras 2.15 `on_epoch_end` verbatim; note Keras' `min_delta` is **absolute** (torch's `ReduceLROnPlateau` default is relative) |
| `EarlyStopping(val_loss, patience, restore_best_weights, start_from_epoch=4)` | `KerasEarlyStopping` | Keras 2.15 logic verbatim (`wait` increments before the improvement test; `epoch` 0-based) |
| `TerminateOnNaN` | built into `fit` | |
| callback order | ModelCheckpoint → TensorBoard → CSVLogger → LearningRateLogger → ReduceLROnPlateau → [EarlyStopping] → [W&B] | same as the TF `callbacks` list |
| W&B `WandbMetricsLogger` | `wandb_run.log({"epoch/…"})` per epoch, run name `torch__<experiment path>` in the same project | |

PyTorch-only: a per-step CSV (`step_losses.csv`: exact loss, running mean, lr, sample_idx,
seconds), `sample_order_used.npy`, and `resume.pt` (model, optimizer incl. Adam moments and step
counter, callback states, history, RNG state) written every epoch; `--resume` continues a run
after a crash or reboot.

## 5. Semantic differences and how they were handled

### 5.1 Gradient clipping
Keras `Optimizer(clipnorm=c)` clips **each gradient tensor separately** to L2 norm ≤ c
(`tf.clip_by_norm` per tensor). PyTorch's idiom `clip_grad_norm_` clips the *global* norm over all
parameters (what Keras calls `global_clipnorm`). `training_lib.clip_by_norm_` reproduces the
per-tensor semantics with TF's exact expression `g·c / max(‖g‖, c)`; `KerasAdam` applies it
before its update.

### 5.2 MAPE epsilon
Keras' MAPE divides by `max(|y_true|, 1e-7)`; the metric functions in `training_lib`
(`get_positional_mape`) divide by `y_true` directly, as in the TF code. Both are kept as they
were.

### 5.3 Adam arithmetic
`torch.optim.Adam` and `tf.keras.optimizers.Adam` differ in (a) default ε (1e-8 vs 1e-7) and
(b) where ε enters: Keras computes `α = lr·√(1−β₂ᵗ)/(1−β₁ᵗ)` and `θ −= α·m/(√v + ε)`, torch
computes `θ −= lr/(1−β₁ᵗ) · m/(√v/√(1−β₂ᵗ) + ε)`, i.e. torch's effective ε is `ε·√(1−β₂ᵗ)`
(3e-9 at step 1 for ε=1e-7). `KerasAdam` implements Keras' formula exactly; `--optimizer
torch_adam` is available to use `torch.optim.Adam(eps=1e-7)` instead.

### 5.4 Initialisation
Keras: glorot-uniform kernels, orthogonal recurrent kernels, zero biases. PyTorch: kaiming-uniform
Linear weights with uniform biases; uniform(±1/√H) for all GRU tensors. `--init keras` selects
the Keras scheme; the default is `torch` (the user's stated long-term preference). All
TF-comparison runs pass `--init keras` explicitly and record it in `metrics.json`.

### 5.5 Shuffle order and z-scores
See §3. In the TF pipeline the z-score step consumes the first shuffled pass and `model.fit` the
second; `ListDataset` mirrors that bookkeeping but with different random numbers. Exact-replay
runs therefore also load the recorded `z_scores.pkl` (bit-identical to the GT's, proven by the
replay recorder); native runs recompute them from their own first pass.

### 5.6 Device, threads, determinism, TF32
- **TF ground truth**: CPU, bit-reproducible only under identical threading — re-running with
  `TF_NUM_INTRAOP_THREADS=1` changes every weight tensor after one epoch (per-step losses within
  3e-5 relative). This is the natural "envelope" for any comparison.
- **PyTorch CPU**: deterministic. `torch.set_num_threads` must match the number of concurrent
  jobs (`--threads`): oversubscribing the cores makes OpenMP spin and slows steps 10–100×.
- **PyTorch CUDA**: bit-reproducible run-to-run with `torch.use_deterministic_algorithms(True)`
  (default in `experiment.py`; `--nondeterministic` to disable). Ampere+ GPUs execute float32
  matmuls / cuDNN RNN kernels in **TF32** (10-bit mantissa) unless disabled; with TF32 the
  GPU differed from the CPU by 8e-4 relative on outputs and 2 % on gradients, with TF32 off by
  8e-6 and 7e-5. `experiment.py` disables TF32 (`--allow-tf32` re-enables it).
- Speed: see PYTORCH_PARITY.md (speed table).

### 5.7 `inference_mode` — a latent bug in the TF evaluation
As in TF, `model.inference_mode = True` (set after training, before `predict`) clamps
predictions at 0. In TF this attribute is read at **graph-trace time**: `call` is a
`@tf.function`, and `predict` reuses the traces cached during training (per input-shape
signature), which were traced with `inference_mode=False`. Consequently a fraction of the GT test
predictions were never clamped: 879 negative delay predictions (0.10 %) in the converged
`trex_multiburst/delay` run, 50 in the jitter run. Clamping the stored GT predictions reproduces
a fresh TF forward pass and the PyTorch one to 5 digits (MAPE 5.0745 → **5.0368** for delay,
11.7496 → 11.7490 for jitter). PyTorch reads the attribute on every call, so every prediction is
clamped as intended; `compare_results.py` therefore recomputes the GT metrics from clamped
predictions and shows the stored values alongside.

### 5.8 What is *not* translated
`tf.autograph` loop options, `tf.ensure_shape`, `tf.function` retracing, the TF GPU
`LD_LIBRARY_PATH` re-exec (`gpu_setup.py` is a set of no-ops with the original code kept as
comments), Keras layer `name=` arguments.

## 6. Numerical conditioning — why "identical training" is not a meaningful target

RouteNet-Gauss back-propagates through 50 windows × 8 message-passing iterations of GRU updates
(400 recurrent steps). At **untrained** weights the true (float64) gradients explode — 1.6e8 on one
training scenario, 3e24 on another — and a float32 evaluation of them is not defined to better
than orders of magnitude *in either framework*: on the 1.6e8 scenario TF's own float32 gradient is
346× off the float64 value, PyTorch's 42 700×. Because `clipnorm=1.0` normalises every tensor,
training still works — it follows the gradient *direction* — but two implementations that differ
in the last float32 bit take different directions on the tiny-gradient elements. The
consequences, all measured (PYTORCH_PARITY.md):

- **loss trajectories still agree**: over the first 50 training steps the torch exact replay
  tracks the TF recording to ≤ 3.4e-5 relative per step, the same as TF-with-1-thread tracks
  TF-with-4-threads (2.8e-5);
- **weights do not**: after 50 steps the per-element weight differences between torch and TF are
  of the order of the total weight movement — and so are TF's own under a different thread count
  (0/34 tensors identical after one epoch);
- at **trained** weights (the converged checkpoints) the problem is well-conditioned: float32
  gradients agree with float64 to 1e-5–1e-3 in both frameworks, and the L1 gradient/optimizer
  gates apply to every tensor.

Therefore parity is established at three levels — forward pass (L0), loss/gradient/step at
well-conditioned weights (L1), and training *outcomes* (exact-replay and native runs) — and not
as bit-identical weights.

## 7. Verification tools

| tool | what it proves |
|---|---|
| `tf_reference/replay_tf_run.py` | the recorded init weights / order / z-scores reproduce the TF GT bit-for-bit (34/34 tensors per epoch, history.csv, test predictions) |
| `convert_tf_checkpoint.py` | TF ↔ torch weight mapping (pure re-layout) |
| `parity/l0_forward.py`, `parity/run_l0_all.py` | forward-pass parity on every checkpoint's test set; data-pipeline targets bit-identical |
| `parity/l1_grad_step.py` | loss, gradient (vs float64 reference) and one-Adam-step parity |
| `compare_results.py` | training outcomes vs the GT, per cell, with the agreed gates; per-step curves for exact replays |
