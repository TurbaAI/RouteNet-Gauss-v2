# Keras-init check (`parity/check_keras_init.py`)

PyTorch `init="keras"`: 16 draws · reference A: 2 distinct recorded TF initial weight sets (8 cells) · reference B: 16 fresh Keras initialiser draws.
**PASSED** — min KS p vs TF 0.0452, vs Keras 0.0244 (gate 0.0001); max |WᵀW−I| torch 5.7e-07, TF 4.7e-07 (gate 1e-05).

| tensor | kind | shape | std torch | std TF | std Keras | expected | max\|w\| / limit | ‖WᵀW−I‖ torch / TF | KS p vs TF | KS p vs Keras | passed |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| `flow_update.weight_ih_l0` | glorot | (96, 64) | 0.11191 | 0.11164 | 0.11191 | 0.11180 | 1.0000 |  | 0.682 | 0.927 | True |
| `flow_update.weight_hh_l0` | orthogonal | (96, 32) | 0.10206 | 0.10205 | 0.10206 |  |  | 4.7e-07 / 4.2e-07 | 0.367 | 0.627 | True |
| `flow_update.bias_ih_l0` | zero | (96,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `flow_update.bias_hh_l0` | zero | (96,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `link_update.weight_ih` | glorot | (96, 32) | 0.12508 | 0.12503 | 0.12568 | 0.12500 | 1.0000 |  | 0.045 | 0.076 | True |
| `link_update.weight_hh` | orthogonal | (96, 32) | 0.10206 | 0.10206 | 0.10206 |  |  | 5.7e-07 / 4.0e-07 | 0.322 | 0.590 | True |
| `link_update.bias_ih` | zero | (96,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `link_update.bias_hh` | zero | (96,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `queue_update.weight_ih` | glorot | (96, 64) | 0.11175 | 0.11207 | 0.11168 | 0.11180 | 1.0000 |  | 0.830 | 0.970 | True |
| `queue_update.weight_hh` | orthogonal | (96, 32) | 0.10206 | 0.10206 | 0.10206 |  |  | 5.0e-07 / 4.2e-07 | 0.557 | 0.734 | True |
| `queue_update.bias_ih` | zero | (96,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `queue_update.bias_hh` | zero | (96,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `node_update.weight_ih` | glorot | (96, 32) | 0.12480 | 0.12484 | 0.12512 | 0.12500 | 1.0000 |  | 0.137 | 0.506 | True |
| `node_update.weight_hh` | orthogonal | (96, 32) | 0.10206 | 0.10206 | 0.10206 |  |  | 5.4e-07 / 4.7e-07 | 0.701 | 0.957 | True |
| `node_update.bias_ih` | zero | (96,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `node_update.bias_hh` | zero | (96,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `flow_embedding.0.weight` | glorot | (32, 3) | 0.23858 | 0.24159 | 0.24075 | 0.23905 | 0.9998 |  | 0.895 | 0.929 | True |
| `flow_embedding.0.bias` | zero | (32,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `flow_embedding.2.weight` | glorot | (32, 32) | 0.17708 | 0.17568 | 0.17793 | 0.17678 | 0.9998 |  | 0.479 | 0.623 | True |
| `flow_embedding.2.bias` | zero | (32,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `queue_embedding.0.weight` | glorot | (32, 3) | 0.23441 | 0.25030 | 0.23438 | 0.23905 | 0.9998 |  | 0.505 | 0.294 | True |
| `queue_embedding.0.bias` | zero | (32,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `queue_embedding.2.weight` | glorot | (32, 32) | 0.17716 | 0.17851 | 0.17743 | 0.17678 | 1.0000 |  | 0.487 | 0.299 | True |
| `queue_embedding.2.bias` | zero | (32,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `link_embedding.0.weight` | glorot | (32, 1) | 0.23677 | 0.24792 | 0.24030 | 0.24618 | 0.9974 |  | 0.057 | 0.071 | True |
| `link_embedding.0.bias` | zero | (32,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `link_embedding.2.weight` | glorot | (32, 32) | 0.17609 | 0.17474 | 0.17581 | 0.17678 | 0.9999 |  | 0.890 | 0.390 | True |
| `link_embedding.2.bias` | zero | (32,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `node_embedding.0.weight` | glorot | (32, 32) | 0.17651 | 0.17560 | 0.17650 | 0.17678 | 1.0000 |  | 0.259 | 0.118 | True |
| `node_embedding.0.bias` | zero | (32,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `node_embedding.2.weight` | glorot | (32, 32) | 0.17644 | 0.17930 | 0.17610 | 0.17678 | 1.0000 |  | 0.724 | 0.142 | True |
| `node_embedding.2.bias` | zero | (32,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `readout_path.0.weight` | glorot | (16, 32) | 0.20183 | 0.20008 | 0.20397 | 0.20412 | 0.9999 |  | 0.845 | 0.049 | True |
| `readout_path.0.bias` | zero | (16,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `readout_path.2.weight` | glorot | (16, 16) | 0.25128 | 0.25479 | 0.25102 | 0.25000 | 0.9999 |  | 0.459 | 0.024 | True |
| `readout_path.2.bias` | zero | (16,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
| `readout_path.4.weight` | glorot | (5, 16) | 0.30518 | 0.32727 | 0.31127 | 0.30861 | 0.9996 |  | 0.574 | 0.586 | True |
| `readout_path.4.bias` | zero | (5,) | max\|b\| 0 | max\|b\| 0 | | | | | | | True |
