## torch_baseline_torchinit vs TF (quick 5x50)

TF columns are recomputed from the GT `predictions.npz` with predictions clamped at 0 (as the TF code intended and the PyTorch evaluation does); `TF MAPE stored` is the value in the GT metrics.json, which includes unclamped negative predictions (`neg`).

| dataset | target | seed | mode | init | device | TF MAPE (stored / neg) | TF MAPE | torch MAPE | Δ MAPE | TF R² | torch R² | Δ R² | TF MAE µs | torch MAE µs | gate | passed |
|---|---|--:|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| mawi_pcaps | delay | 1 | native | torch | cpu | 87.336 / 0 | 87.336 | 87.615 | +0.279 | -0.159 | -0.159 | -0.0002 | 243.42 | 243.70 | MAPE +-3pt, R2 +-0.001 (2x TF seed spread) | True |
| mawi_pcaps | delay | 2 | native | torch | cpu | 87.205 / 0 | 87.205 | 87.406 | +0.201 | -0.158 | -0.158 | -0.0005 | 243.15 | 243.43 | MAPE +-3pt, R2 +-0.001 (2x TF seed spread) | True |
| mawi_pcaps | jitter | 1 | native | torch | cpu | 96.816 / 0 | 96.816 | 94.898 | -1.918 | -0.979 | -0.950 | +0.0298 | 39.27 | 38.81 | MAPE +-3pt, R2 +-0.096 (2x TF seed spread) | True |
| mawi_pcaps | jitter | 2 | native | torch | cpu | 94.017 / 0 | 94.017 | 95.969 | +1.952 | -0.931 | -0.955 | -0.0243 | 38.55 | 38.95 | MAPE +-3pt, R2 +-0.096 (2x TF seed spread) | True |
| trex_multiburst | delay | 1 | native | torch | cpu | 111.115 / 0 | 111.115 | 110.891 | -0.224 | -32.630 | -32.504 | +0.1257 | 150.06 | 149.76 | MAPE +-3pt, R2 +-0.416 (2x TF seed spread) | True |
| trex_multiburst | delay | 2 | native | torch | cpu | 111.465 / 0 | 111.465 | 111.207 | -0.258 | -32.838 | -32.686 | +0.1519 | 150.55 | 150.19 | MAPE +-3pt, R2 +-0.416 (2x TF seed spread) | True |
| trex_multiburst | jitter | 1 | native | torch | cpu | 92.878 / 0 | 92.878 | 94.449 | +1.570 | -2.928 | -2.936 | -0.0079 | 16.46 | 16.59 | MAPE +-3pt, R2 +-0.396 (2x TF seed spread) | True |
| trex_multiburst | jitter | 2 | native | torch | cpu | 91.050 / 0 | 91.050 | 96.538 | +5.488 | -2.730 | -3.017 | -0.2869 | 16.08 | 16.83 | MAPE +-3pt, R2 +-0.396 (2x TF seed spread) | False |

| dataset | target | seed | TF train s | torch train s | torch s/step | TF epochs | torch epochs |
|---|---|--:|--:|--:|--:|--:|--:|
| mawi_pcaps | delay | 1 | 2040 | 2688 | 3.74 | 5 | 5 |
| mawi_pcaps | delay | 2 | 1947 | 2683 | 3.75 | 5 | 5 |
| mawi_pcaps | jitter | 1 | 1936 | 2677 | 3.72 | 5 | 5 |
| mawi_pcaps | jitter | 2 | 2010 | 2665 | 3.73 | 5 | 5 |
| trex_multiburst | delay | 1 | 1572 | 2261 | 5.59 | 5 | 5 |
| trex_multiburst | delay | 2 | 1610 | 2251 | 5.56 | 5 | 5 |
| trex_multiburst | jitter | 1 | 1640 | 1889 | 4.71 | 5 | 5 |
| trex_multiburst | jitter | 2 | 1598 | 1900 | 4.70 | 5 | 5 |
