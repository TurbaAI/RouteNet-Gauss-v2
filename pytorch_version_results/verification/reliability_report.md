## 7.2 TF against itself at converged scale

Setup check (one epoch, the ground truth's threading): epoch-0 `history.csv` row **bit-identical** to the ground truth.

| run | epochs | (i) plateau exit | (ii) best val ≤ epoch 44 | Δ(ii) | best val | (iii) test MAPE | Δ(iii) | test R² |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| TF ground truth (4 threads) | 45 | 6 | 7.0663 | — | 7.0663 | 5.037 | — | 0.7402 |
| **TF, one intra-op thread** | 94 | 5 | 7.0049 | -0.061 | 5.5902 | 3.668 | -1.369 | 0.8148 |
| PyTorch exact replay (§5) | 152 | 2 | 7.6533 | +0.587 | 4.9780 | 3.045 | -1.992 | 0.8465 |

Rule (fixed before the run): Confirmed if |Δ(ii)| ≥ 0.3 or |Δ(iii)| ≥ 1; Suspect if |Δ(ii)| ≤ 0.1 and |Δ(iii)| ≤ 0.3; otherwise Inconclusive. **Verdict: Confirmed.**

## 7.3 The new default pipeline, trained to convergence

| run | init | epochs | plateau exit | best val ≤ epoch 44 | best val | test MAPE | Δ MAPE | test R² | Δ R² |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| TF ground truth | keras (TF) | 45 | 6 | 7.0663 | 7.0663 | 5.037 | — | 0.7402 | — |
| **PyTorch, new default** | keras | 57 | 20 | 7.0970 | 7.0970 | 5.116 | +0.080 | 0.7412 | +0.0010 |
| PyTorch, torch init (§5) | torch | 62 | 20 | 7.8501 | 6.6055 | 4.837 | -0.200 | 0.7397 | -0.0006 |

Gate (fixed before the run): |Δ MAPE| ≤ 1 and |Δ R²| ≤ 0.03; outside it only passes if TF moves as far against itself (§7.2). **Verdict: Pass.**

## 7.4 Plateau test across seeds

| seed | TF: plateau exit | TF: epochs | PyTorch (new default): plateau exit | PyTorch: epochs |
|--:|--:|--:|--:|--:|
| 1 | 6 | 45 | 20 | 57 |
| 2 | 4 | 25 | 7 | 25 |
| 3 | 11 | 25 | 13 | 25 |

Seed 1: TF ground truth and the §7.3 run; seeds 2 and 3: first 25 epochs of the converged config. Reference: PyTorch's own init left the plateau at epoch 20 (§5, PYTORCH_PORT.md §5.4).

Rule (fixed before the runs): no PyTorch run early-stopped on the plateau, and none leaves it more than 5 epochs after the latest TF seed. **Verdict: Fail.**

