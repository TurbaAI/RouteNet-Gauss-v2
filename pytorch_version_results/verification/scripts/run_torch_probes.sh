#!/usr/bin/env bash
# Q7(a) plateau test, PyTorch side: seeds 2 and 3, first 25 epochs of the converged delay config
# with the new default init. Seeds whose metrics.json exists are skipped; after a reboot re-launch
# with --resume.
#   setsid nohup results/verification/run_torch_probes.sh > results/verification/logs/torch_plateau_probe.log 2>&1 < /dev/null &
cd /home/ubuntu/code/RouteNet-Gauss-v2 || exit 1
for s in 2 3; do
    [ -f "results/torch_plateau_probe/trex_multiburst/RouteNetGauss/delay/seed_$s/metrics.json" ] && continue
    /home/ubuntu/anaconda3/envs/RG_torch/bin/python experiment.py --dataset trex_multiburst --target delay --seed "$s" \
        --epochs 25 --steps 500 --shuffle-buffer 200 --patience 15 --save-best-only \
        --experiment-name torch_plateau_probe --device cuda --threads 1 --use-wandb "$@" || exit $?
done
