## torch_baseline vs TF (quick 5x50)

TF columns are recomputed from the GT `predictions.npz` with predictions clamped at 0 (as the TF code intended and the PyTorch evaluation does); `TF MAPE stored` is the value in the GT metrics.json, which includes unclamped negative predictions (`neg`).

| dataset | target | seed | mode | init | device | TF MAPE (stored / neg) | TF MAPE | torch MAPE | Δ MAPE | TF R² | torch R² | Δ R² | TF MAE µs | torch MAE µs | gate | passed |
|---|---|--:|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| mawi_pcaps | delay | 1 | exact-replay | keras | cpu | 87.336 / 0 | 87.336 | 87.643 | +0.307 | -0.159 | -0.159 | -0.0005 | 243.42 | 243.79 | MAPE +-0.5pt, R2 +-0.020 (max(0.02, 2x TF seed spread)) | True |
| mawi_pcaps | delay | 2 | exact-replay | keras | cpu | 87.205 / 0 | 87.205 | 87.205 | -0.000 | -0.158 | -0.158 | +0.0000 | 243.15 | 243.15 | MAPE +-0.5pt, R2 +-0.020 (max(0.02, 2x TF seed spread)) | True |
| mawi_pcaps | jitter | 1 | exact-replay | keras | cpu | 96.816 / 0 | 96.816 | 94.143 | -2.673 | -0.979 | -0.955 | +0.0239 | 39.27 | 38.78 | exact-replay jitter: MAPE +-3pt, R2 +-0.096 (chaos envelope, see report) | True |
| mawi_pcaps | jitter | 2 | exact-replay | keras | cpu | 94.017 / 0 | 94.017 | 94.016 | -0.001 | -0.931 | -0.931 | +0.0000 | 38.55 | 38.55 | exact-replay jitter: MAPE +-3pt, R2 +-0.096 (chaos envelope, see report) | True |
| trex_multiburst | delay | 1 | exact-replay | keras | cpu | 111.115 / 0 | 111.115 | 111.158 | +0.043 | -32.630 | -32.655 | -0.0253 | 150.06 | 150.12 | MAPE +-0.5pt, R2 +-0.416 (max(0.02, 2x TF seed spread)) | True |
| trex_multiburst | delay | 2 | exact-replay | keras | cpu | 111.465 / 0 | 111.465 | 111.109 | -0.356 | -32.838 | -32.633 | +0.2050 | 150.55 | 150.07 | MAPE +-0.5pt, R2 +-0.416 (max(0.02, 2x TF seed spread)) | True |
| trex_multiburst | jitter | 1 | exact-replay | keras | cpu | 92.878 / 0 | 92.878 | 90.799 | -2.079 | -2.928 | -2.869 | +0.0594 | 16.46 | 16.26 | exact-replay jitter: MAPE +-3pt, R2 +-0.396 (chaos envelope, see report) | True |
| trex_multiburst | jitter | 2 | exact-replay | keras | cpu | 91.050 / 0 | 91.050 | 90.400 | -0.650 | -2.730 | -2.700 | +0.0307 | 16.08 | 16.00 | exact-replay jitter: MAPE +-3pt, R2 +-0.396 (chaos envelope, see report) | True |

| dataset | target | seed | TF train s | torch train s | torch s/step | TF epochs | torch epochs |
|---|---|--:|--:|--:|--:|--:|--:|
| mawi_pcaps | delay | 1 | 2040 | 4082 | 5.68 | 5 | 5 |
| mawi_pcaps | delay | 2 | 1947 | 4095 | 5.74 | 5 | 5 |
| mawi_pcaps | jitter | 1 | 1936 | 4070 | 5.68 | 5 | 5 |
| mawi_pcaps | jitter | 2 | 2010 | 2765 | 4.10 | 5 | 5 |
| trex_multiburst | delay | 1 | 1572 | 2336 | 5.96 | 5 | 5 |
| trex_multiburst | delay | 2 | 1610 | 2369 | 6.06 | 5 | 5 |
| trex_multiburst | jitter | 1 | 1640 | 1887 | 4.69 | 5 | 5 |
| trex_multiburst | jitter | 2 | 1598 | 1913 | 4.77 | 5 | 5 |

### Per-step training loss, exact replay vs TF recording

| dataset | target | seed | steps | median rel diff | max rel diff | max rel diff (first 50) | first step > 1e-3 | corr |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| mawi_pcaps | delay | 1 | 250 | 4.97e-05 | 3.91e-03 | 1.81e-05 | 184 | 0.999386 |
| mawi_pcaps | delay | 2 | 250 | 1.35e-06 | 4.76e-06 | 4.49e-07 | None | 1.000000 |
| mawi_pcaps | jitter | 1 | 250 | 1.89e-03 | 3.09e-02 | 1.87e-03 | 25 | 0.978219 |
| mawi_pcaps | jitter | 2 | 250 | 4.74e-06 | 1.78e-05 | 8.25e-06 | None | 1.000000 |
| trex_multiburst | delay | 1 | 250 | 1.77e-04 | 8.75e-04 | 3.37e-05 | None | 0.999996 |
| trex_multiburst | delay | 2 | 250 | 2.06e-04 | 3.63e-03 | 9.47e-06 | 171 | 0.999937 |
| trex_multiburst | jitter | 1 | 250 | 1.69e-03 | 2.50e-02 | 2.39e-04 | 116 | 0.999732 |
| trex_multiburst | jitter | 2 | 250 | 8.10e-04 | 7.53e-03 | 3.95e-04 | 83 | 0.999752 |

| dataset | target | seed | epoch | TF loss | torch loss | TF val_loss | torch val_loss |
|---|---|--:|--:|--:|--:|--:|--:|
| mawi_pcaps | delay | 1 | 0 | 88.3938 | 88.3933 | 88.6532 | 88.6520 |
| mawi_pcaps | delay | 1 | 1 | 88.2031 | 88.2004 | 88.6315 | 88.6275 |
| mawi_pcaps | delay | 1 | 2 | 88.4294 | 88.4323 | 88.5225 | 88.5413 |
| mawi_pcaps | delay | 1 | 3 | 88.5690 | 88.6251 | 88.2522 | 88.3642 |
| mawi_pcaps | delay | 1 | 4 | 88.0458 | 88.2303 | 87.7825 | 88.0737 |
| mawi_pcaps | delay | 2 | 0 | 88.4215 | 88.4216 | 88.6228 | 88.6228 |
| mawi_pcaps | delay | 2 | 1 | 88.1022 | 88.1022 | 88.5500 | 88.5499 |
| mawi_pcaps | delay | 2 | 2 | 88.7000 | 88.6999 | 88.3978 | 88.3976 |
| mawi_pcaps | delay | 2 | 3 | 88.3163 | 88.3161 | 88.1224 | 88.1222 |
| mawi_pcaps | delay | 2 | 4 | 87.4524 | 87.4521 | 87.6607 | 87.6605 |
| mawi_pcaps | jitter | 1 | 0 | 99.9492 | 99.9949 | 99.8816 | 99.9889 |
| mawi_pcaps | jitter | 1 | 1 | 99.7675 | 99.9472 | 99.6197 | 99.8513 |
| mawi_pcaps | jitter | 1 | 2 | 99.4075 | 99.5849 | 99.1257 | 99.1683 |
| mawi_pcaps | jitter | 1 | 3 | 98.7550 | 98.3978 | 98.2562 | 97.3271 |
| mawi_pcaps | jitter | 1 | 4 | 97.6981 | 96.0130 | 96.9575 | 94.3320 |
| mawi_pcaps | jitter | 2 | 0 | 99.9139 | 99.9137 | 99.7743 | 99.7740 |
| mawi_pcaps | jitter | 2 | 1 | 99.5728 | 99.5730 | 99.3001 | 99.3006 |
| mawi_pcaps | jitter | 2 | 2 | 98.9119 | 98.9121 | 98.3388 | 98.3386 |
| mawi_pcaps | jitter | 2 | 3 | 97.5736 | 97.5729 | 96.6333 | 96.6323 |
| mawi_pcaps | jitter | 2 | 4 | 95.4768 | 95.4757 | 94.2083 | 94.2073 |
| trex_multiburst | delay | 1 | 0 | 111.2072 | 111.2086 | 110.4389 | 110.4423 |
| trex_multiburst | delay | 1 | 1 | 110.0053 | 110.0118 | 110.4228 | 110.4344 |
| trex_multiburst | delay | 1 | 2 | 109.1230 | 109.1438 | 110.3587 | 110.3922 |
| trex_multiburst | delay | 1 | 3 | 112.4377 | 112.4903 | 110.1749 | 110.2410 |
| trex_multiburst | delay | 1 | 4 | 110.4690 | 110.5267 | 109.8466 | 109.8862 |
| trex_multiburst | delay | 2 | 0 | 113.5182 | 113.5179 | 110.4420 | 110.4415 |
| trex_multiburst | delay | 2 | 1 | 110.9109 | 110.9076 | 110.4361 | 110.4272 |
| trex_multiburst | delay | 2 | 2 | 109.6209 | 109.5965 | 110.4124 | 110.3613 |
| trex_multiburst | delay | 2 | 3 | 111.7288 | 111.6267 | 110.3315 | 110.1684 |
| trex_multiburst | delay | 2 | 4 | 110.0556 | 109.8159 | 110.1683 | 109.8399 |
| trex_multiburst | jitter | 1 | 0 | 99.9844 | 99.9903 | 99.9536 | 99.9735 |
| trex_multiburst | jitter | 1 | 1 | 99.8481 | 99.8718 | 99.6479 | 99.6357 |
| trex_multiburst | jitter | 1 | 2 | 99.1550 | 98.9353 | 98.5370 | 98.0400 |
| trex_multiburst | jitter | 1 | 3 | 97.4917 | 96.6270 | 96.2308 | 94.9836 |
| trex_multiburst | jitter | 1 | 4 | 94.5635 | 92.8836 | 92.9144 | 90.8444 |
| trex_multiburst | jitter | 2 | 0 | 99.9147 | 99.9270 | 99.7533 | 99.7977 |
| trex_multiburst | jitter | 2 | 1 | 99.3801 | 99.4590 | 98.9075 | 98.9991 |
| trex_multiburst | jitter | 2 | 2 | 98.1643 | 98.2111 | 97.2099 | 97.1765 |
| trex_multiburst | jitter | 2 | 3 | 95.9009 | 95.7400 | 94.5543 | 94.2462 |
| trex_multiburst | jitter | 2 | 4 | 92.7389 | 92.2577 | 91.0787 | 90.4305 |
