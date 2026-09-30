#!/usr/bin/env bash
# Q2(b): the TF converged delay ground truth re-run with ONE intra-op thread. Everything else is the
# GT command (run_experiments.py @ 2e30d5d: CPU, shuffle buffer 200, patience 15, best-only
# checkpoint, W&B), run by the frozen TF code copied from 2e30d5d into results/tf_runner/.
# TF cannot resume: after a reboot, re-launch this script (the run is deterministic, so it
# retraces the same trajectory from the start).
#   setsid nohup results/verification/run_tf_envelope.sh > results/verification/logs/tf_envelope_1thread.log 2>&1 < /dev/null &
cd /home/ubuntu/code/RouteNet-Gauss-v2/results/tf_runner || exit 1
export CUDA_VISIBLE_DEVICES=-1 _RG_GPU_ENV_READY=1 TF_NUM_INTRAOP_THREADS=1
exec /home/ubuntu/anaconda3/envs/RG/bin/python experiment.py --dataset trex_multiburst --target delay --seed 1 \
    --epochs 300 --steps 500 --shuffle-buffer 200 --patience 15 \
    --experiment-name tf_envelope_1thread --use-wandb --save-best-only
