#!/usr/bin/env bash
# Q7(a) plateau test, TF side: seeds 2 and 3, first 25 epochs of the converged delay config
# (same command as the GT, --epochs 25). One intra- and one inter-op thread so the four
# concurrent jobs fit the four cores. Seeds whose metrics.json exists are skipped.
#   setsid nohup results/verification/run_tf_probes.sh > results/verification/logs/tf_plateau_probe.log 2>&1 < /dev/null &
cd /home/ubuntu/code/RouteNet-Gauss-v2/results/tf_runner || exit 1
export CUDA_VISIBLE_DEVICES=-1 _RG_GPU_ENV_READY=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
for s in 2 3; do
    [ -f "results/tf_plateau_probe/trex_multiburst/RouteNetGauss/delay/seed_$s/metrics.json" ] && continue
    /home/ubuntu/anaconda3/envs/RG/bin/python experiment.py --dataset trex_multiburst --target delay --seed "$s" \
        --epochs 25 --steps 500 --shuffle-buffer 200 --patience 15 \
        --experiment-name tf_plateau_probe --use-wandb --save-best-only || exit $?
done
