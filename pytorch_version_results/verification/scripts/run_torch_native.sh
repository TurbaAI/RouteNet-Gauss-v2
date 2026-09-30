#!/usr/bin/env bash
# Q7(a) check 2: PyTorch full training run with the new default init (Keras-style, drawn by
# PyTorch's RNG) and PyTorch's own shuffle — no TF input at all. Same converged config as the TF
# ground truth. --init is deliberately NOT passed: this exercises the default.
# After a reboot: re-launch with --resume.
#   setsid nohup results/verification/run_torch_native.sh > results/verification/logs/torch_converged_native.log 2>&1 < /dev/null &
cd /home/ubuntu/code/RouteNet-Gauss-v2 || exit 1
exec /home/ubuntu/anaconda3/envs/RG_torch/bin/python experiment.py --dataset trex_multiburst --target delay --seed 1 \
    --epochs 300 --steps 500 --shuffle-buffer 200 --patience 15 --save-best-only \
    --experiment-name torch_converged_native --device cuda --threads 1 --use-wandb "$@"
