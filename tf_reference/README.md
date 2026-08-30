# `tf_reference/` — frozen TensorFlow originals

**Do not edit the module copies in this folder.** They are byte-identical snapshots of the
TensorFlow implementation as of commit `2e30d5d` (the code that produced the ground truth in
`tensorflow_version_gt/`), kept importable so the PyTorch port can be checked against the
real TF model in the same process:

| file | copied from | purpose |
|---|---|---|
| `models.py` | `models.py` @ 2e30d5d | `RouteNetGauss` Keras model |
| `utils.py` | `utils.py` @ 2e30d5d | `load_dataset`, `prepare_targets_and_mask`, `seg_to_global_reshape` |
| `training_lib.py` | `training_lib.py` @ 2e30d5d | z-scores, positional MAPE/R² metrics, `LearningRateLogger` |

Verify at any time with `git diff 2e30d5d:models.py tf_reference/models.py` (etc.) — the diff must be empty.

The top-level `models.py`, `utils.py`, `training_lib.py`, `experiment.py`, `train.py`,
`run_experiments.py` are the **PyTorch** versions (with the original TF lines kept as comments
above each translation). Import from `tf_reference.` when you need TensorFlow behaviour.

## `replay_tf_run.py` — the TF replay recorder

Re-executes the exact setup of a ground-truth training job ("cell" = dataset × target × seed)
and records what the GT never saved — the scenario order fed to `model.fit` and the initial
weights — then proves the replay is exact by comparing per-epoch checkpoints, `history.csv`
and test predictions bit-for-bit against `tensorflow_version_gt/`. Outputs go to
`tensorflow_version_gt/replay/<dataset>/RouteNetGauss/<target>/seed_<n>/`:

| file | content |
|---|---|
| `sample_order.npy` | `sample_idx` of the scenario fed at training step 0, 1, 2, … (150 000 steps by default) |
| `init_weights.npz` | the 34 initial weight tensors, keyed like the TF checkpoint (`flow_update/kernel`, …) |
| `step_losses_5x50.csv` | exact per-step training loss, Keras running-mean loss, lr, sample_idx (quick 5×50 config) |
| `history.csv`, `z_scores.pkl` | regenerated; must equal the GT's |
| `replay_check.json` | the proof: z-score fingerprint, per-epoch checkpoint bit-equality, history equality, prediction equality |

Run from the repo root **in the TF environment that produced the GT** (conda env `RG`, CPU):

```bash
python tf_reference/replay_tf_run.py --all --concurrency 3          # all 8 quick cells
python tf_reference/replay_tf_run.py --dataset trex_multiburst --target delay --seed 1 --mode both
```

Why this works: `tf.data`'s `shuffle(seed=…)` and Keras' initializers are deterministic given
the seeds, so the same code path regenerates the same order and weights. The "z-score step"
(mean/std of the first 500 shuffled scenarios, see the README glossary) consumes the first
shuffled pass, which is why it is replayed too — and why the recomputed z-scores matching the
GT `z_scores.pkl` bit-for-bit is the fingerprint that the replayed stream is the GT's.
