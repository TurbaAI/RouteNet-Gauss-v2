L0 forward parity, all repo checkpoints on their full test sets (tol 5e-4 scale-relative; see PYTORCH_PARITY.md §1).

| checkpoint | scenarios | predictions | targets identical | max abs diff | worst scale-rel | MAPE TF | MAPE torch | TF s/scen | torch s/scen | passed |
|---|--:|--:|---|--:|--:|--:|--:|--:|--:|---|
| converged_trex_multiburst_delay_seed1 | 51 | 175028 | True | 8.53e-09 | 3.22e-05 | 5.03678 | 5.03677 | 4.0 | 1.2 | True |
| converged_trex_multiburst_jitter_seed1 | 51 | 174242 | True | 5.38e-10 | 1.04e-05 | 11.74896 | 11.74896 | 3.9 | 1.2 | True |
| paper_mawi_pcaps_delay | 172 | 247483 | True | 2.10e-09 | 2.36e-06 | 18.39743 | 18.39743 | 7.1 | 1.4 | True |
| paper_mawi_pcaps_jitter | 172 | 247452 | True | 6.55e-10 | 3.33e-06 | 14.41977 | 14.41977 | 7.1 | 1.3 | True |
| paper_trex_multiburst_filtered_delay | 51 | 175028 | True | 6.31e-08 | 2.69e-04 | 2.55248 | 2.55249 | 6.5 | 2.2 | True |
| paper_trex_multiburst_jitter | 51 | 174242 | True | 1.48e-09 | 2.50e-05 | 11.61226 | 11.61226 | 6.4 | 2.3 | True |
| paper_trex_synthetic_filtered_delay | 178 | 333784 | True | 2.18e-08 | 1.10e-04 | 2.70482 | 2.70481 | 8.7 | 1.9 | True |
| paper_trex_synthetic_jitter | 178 | 333619 | True | 8.55e-11 | 2.16e-06 | 9.87167 | 9.87168 | 8.4 | 1.8 | True |
| quick_mawi_pcaps_delay_seed1 | 172 | 247483 | True | 1.09e-11 | 3.18e-07 | 87.33584 | 87.33584 | 6.8 | 1.3 | True |
| quick_mawi_pcaps_delay_seed2 | 172 | 247483 | True | 1.09e-11 | 3.04e-07 | 87.20522 | 87.20521 | 6.8 | 1.3 | True |
| quick_mawi_pcaps_jitter_seed1 | 172 | 247452 | True | 2.27e-13 | 3.18e-07 | 96.81611 | 96.81611 | 6.2 | 1.2 | True |
| quick_mawi_pcaps_jitter_seed2 | 172 | 247452 | True | 4.55e-13 | 2.97e-07 | 94.01715 | 94.01715 | 6.1 | 1.2 | True |
| quick_trex_multiburst_delay_seed1 | 51 | 175028 | True | 1.75e-10 | 2.76e-07 | 111.11534 | 111.11535 | 5.8 | 1.9 | True |
| quick_trex_multiburst_delay_seed2 | 51 | 175028 | True | 1.75e-10 | 3.22e-07 | 111.46502 | 111.46505 | 5.8 | 1.9 | True |
| quick_trex_multiburst_jitter_seed1 | 51 | 174242 | True | 6.82e-13 | 3.35e-07 | 92.87832 | 92.87832 | 6.1 | 2.0 | True |
| quick_trex_multiburst_jitter_seed2 | 51 | 174242 | True | 6.82e-13 | 3.04e-07 | 91.04993 | 91.04993 | 6.0 | 2.1 | True |
