# RouteNet-Gauss: Hardware-Enhanced Network Modeling with Machine Learning

**Carlos Güemes Palau, Miquel Ferrior Galmés, Jordi Paillisse Vilanova, Albert López Brescó, Pere Barlet Ros, Albert Cabellos Aparicio**

This repository is the code of the paper *RouteNet-Gauss: Hardware-Enhanced Network Modeling with Machine Learning*. [[DOI](https://doi.org/10.1109/TON.2026.3668972)]

Contact us: *[carlos.guemes@upc.edu](mailto:carlos.guemes@upc.edu)*, *[contactus@bnn.upc.edu](mailto:contactus@bnn.upc.edu)*

## Abstract

Network simulation is pivotal in network modeling, assisting with tasks ranging from capacity planning to performance estimation. Traditional approaches such as Discrete Event Simulation (DES) face limitations in terms of computational cost and accuracy. This paper introduces RouteNet-Gauss, a novel integration of a testbed network with a Machine Learning (ML) model to address these challenges. By using the testbed as a hardware accelerator, RouteNet-Gauss generates training datasets rapidly and simulates network scenarios with high fidelity to real-world conditions. Experimental results show that RouteNet-Gauss significantly reduces prediction errors by up to 95\% and achieves a 488x speedup in inference time compared to state-of-the-art DES-based methods. RouteNet-Gauss's modular architecture is dynamically constructed based on the specific characteristics of the network scenario, such as topology and routing. 
This enables it to understand and generalize to different network configurations beyond those seen during training, including networks up to 10x larger. Additionally, it supports Temporal Aggregated Performance Estimation (TAPE), providing configurable temporal granularity and maintaining high accuracy in flow performance metrics. This approach shows promise in improving both simulation efficiency and accuracy, offering a valuable tool for network operators.

## Quick start

1. Please ensure that your OS has installed Python 3 (ideally 3.9)
2. Create the virtual environment and activate the environment:
```bash
virtualenv -p python3 myenv
source myenv/bin/activate
```
3. Then we install the required packages (to avoid issues, make sure to install the specific package versions, especially for TensorFlow):
```bash
pip install tensorflow==2.11.1 numpy==1.24.2 notebook==7.0.7
```
- The following configuration with TensorFlow 2.15 was also used successfully:

```bash
pip install tensorflow==2.15.0 numpy==1.26.3 notebook==7.0.7
```

Once done you can either
- Modify and run [`train.py`](train.py) to train the model
- Evaluate the trained models [`evaluation.ipynb`](evaluation.ipynb).

## Repository structure

The repository contains the following structure:
- `ckpt`: Folder containing the checkpoints used in the paper evaluation.
- `data`: Folder containing the datasets used in the paper. For information on these, read [Datasets information](#datasets-information).
- `normalization`: Folder containing the z-score normalizations used by the trained checkpoints (internal path should match the `ckpt` directory).
- [`evaluation.ipynb`](evaluation.ipynb): an interactive Python notebook file used to evaluate the trained models.
- [`models.py`](models.py) contains the implementation of RouteNet-Gauss.
- [`train.py`](train.py): script to train a RouteNet-Fermi model normally, without fine-tuning.
- [`utils.py`](utils.py) contains auxiliary functions common in the previous files.
- [LICENSE](LICENSE): see the file for the full license.

## Datasets information

In the `data` folder, we can find all the variants of the three datasets used in the paper. Inside each directory, data is split according to the training, validation, and test splits. Then each partition is subdivided into shards, to keep the repository's file size under git and GitHub's limits. *NOTE*: please use the `load_dataset` function from `utils.py` to load these shards correctly. The datasets go as follows:

- `mawi_pcaps`: referred to as Real-World Packet Traces in the paper. Includes training, validation, and test partitions.
- `mawi_pcaps_simulated`: version of the `mawi_pcaps` but run with OMNeT++ simulator. Only includes test partition.
- `trex_multiburst`: referred to as TREX-MULTIBURST in the paper. Includes training, validation, and test partitions.
- `trex_multiburst_filtered`: a subset of samples from `trex_multiburst`. Experiments showed that delay models trained from the subset were more accurate later during the evaluation. This includes only training and validation (the test partition is the same as `trex_multiburst`).
- `trex_multiburst_simulated`: version of the `trex_multiburst` but run with OMNeT++ simulator. Only includes test partition.
- `trex_synthetic`: referred to as TREX-SYNTHETIC in the paper. Includes training, validation, and test partitions.
- `trex_synthetic_filtered`: a subset of samples from `trex_synthetic`. Experiments showed that delay models trained from the subset were more accurate later during the evaluation. This includes only training and validation (the test partition is the same as `trex_synthetic`).
- `trex_synthetic_simulated`: version of the `trex_synthetic` but run with OMNeT++ simulator. Only includes test partition.

## Modifying the `train.py` script

The script contains the default hyperparameters and configurations used in the paper. Follow the comments in the code to perform your modifications. In a summary:

- Use the `RUN_EAGERLY` variable (line 36) to run TensorFlow in eager mode.
- Use the `RELOAD_WEIGHTS` variable (line 39) to resume training from the latest recorded checkpoint.
- Modify the experiment configuration to change aspects such as the dataset used (lines 204-219)
- Change the optimizer (and its hyperparameters) and the loss function on lines 231 and 232, respectively.
- Model definition and the remainder of its hyperparameters can be changed on its instantiation (lines 233-245) and the call to fit the model (lines 295-307)

## PyTorch port

The code in this branch is the **PyTorch** version of RouteNet-Gauss. Every original TensorFlow
line was kept as a `#TF:` comment with its PyTorch translation directly below it, so the two can be
read side by side; the frozen TF originals are importable from [`tf_reference/`](tf_reference/README.md).
The port was verified against the frozen TensorFlow results in `tensorflow_version_gt/` — see
[PYTORCH_PORT.md](PYTORCH_PORT.md) (how it was translated, every semantic difference) and
[PYTORCH_PARITY.md](PYTORCH_PARITY.md) (the measured agreement).

### Quick start (PyTorch)

```bash
conda create -y -n RG_torch python=3.10 && conda activate RG_torch
pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu126   # see requirements-torch.txt for why cu126
pip install -r requirements-torch.txt

python experiment.py --dataset trex_multiburst --target delay --seed 1 --epochs 5 --steps 50 --experiment-name my_run   # one job
python run_experiments.py --experiment-name my_matrix --epochs 5 --steps 50                                             # the 2x2x2 matrix
python train.py                                                                                                        # the paper's single-run config
jupyter notebook evaluation_torch.ipynb                                                                                # evaluate checkpoints
```

Datasets are read from `data_torch/` (a lossless, TF-free conversion of `data/`, see
[`data_torch/README.md`](data_torch/README.md)); TensorFlow is **not** needed to train or evaluate.
Important runtime notes: one torch thread per concurrent CPU job (`--threads`, oversubscription is
10–100× slower), GPU runs are deterministic and TF32 is disabled by default, and every run writes
`resume.pt` so it can be continued with `--resume`. To reproduce the TF ground truth exactly, pass
`--replay-from tensorflow_version_gt/replay/<dataset>/RouteNetGauss/<target>/seed_<n>` (TF's own
initial weights, scenario order and z-scores).

Additional files of the port: `torch_ragged.py` (tf.RaggedTensor stand-ins), `training_lib.py`
(Keras-exact loss, Adam, callbacks and training loop), `convert_data_to_torch.py`,
`convert_tf_checkpoint.py`, `compare_results.py`, `parity/` (L0/L1 harness),
`tf_reference/replay_tf_run.py` (TF replay recorder), `pytorch_version_results/` (frozen PyTorch
results mirroring `tensorflow_version_gt/`).

### Glossary

- **Cell** — one combination of the experiment matrix `dataset × target × seed`, e.g.
  `trex_multiburst / delay / seed 1`; one training job with its own results folder
  `results/<experiment>/<dataset>/RouteNetGauss/<target>/seed_<n>/`. The quick TF baseline is the
  2 × 2 × 2 matrix `{mawi_pcaps, trex_multiburst} × {delay, jitter} × {1, 2}` = 8 cells.
- **GT (ground truth)** — the frozen TensorFlow results in `tensorflow_version_gt/`: the 8-cell
  quick 5×50 baseline and the two converged `trex_multiburst` seed-1 runs.
- **Scenario** — one network sample of a dataset (a topology with its flows over `seg_num` time
  windows); one training step processes one scenario (there is no batching).
- **Z-score step** — the normalisation constants for the two standardised inputs
  (`flow_traffic`, `flow_packets`): mean and standard deviation over the first 500 scenarios of
  the *shuffled* training stream (`get_z_scores_dict(..., summarize=500)`), saved as
  `z_scores.pkl`; the model feeds `(x − mean)/std`. It matters for replaying TF exactly because
  it consumes the first shuffled pass (so `model.fit` sees the second), and because recomputing
  it and matching the GT's values bit-for-bit is the *fingerprint* that a replayed stream is the
  GT's.
- **Exact replay** — a PyTorch run that starts from TF's recorded initial weights, sees the
  recorded scenario order and uses the recorded z-scores of a GT cell
  (`tensorflow_version_gt/replay/...`, produced by `tf_reference/replay_tf_run.py`), so that the
  only remaining difference to the GT is framework arithmetic.
- **Native run** — a PyTorch run using PyTorch's own initialisation and its own seeded shuffle,
  i.e. the pipeline as it will be used going forward; compared with the GT statistically.
- **Parity levels** — **L0** forward pass with identical weights on identical scenarios;
  **L1** loss, gradients and one optimizer step from identical weights; **L2** exact-replay
  training; **L3** native training compared statistically (within the TF seed spread).
- **Seed spread** — the difference between the two TF seeds of a cell (`|seed 1 − seed 2|`),
  the natural yardstick for how much a training outcome can move for reasons unrelated to the
  framework.
- **Chaos envelope** — how far the TF ground truth moves from itself when only the thread
  count changes (`TF_NUM_INTRAOP_THREADS=1`): a different rounding order in float32 gives
  different weights after one epoch while the per-step losses stay within ~3e-5. Any two
  float32 implementations differ at least this much; the PyTorch port is compared against it.

## License

See the [file](LICENSE) for the full license:


```
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
```
