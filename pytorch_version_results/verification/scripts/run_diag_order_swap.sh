#!/usr/bin/env bash
# Q10 diagnosis of the §7.4 plateau result: does the SCENARIO ORDER (with its z-scores) or the INIT
# decide when trex_multiburst/delay leaves the val~86.7 plateau? 2x2 cross-over on seed 1 (first 25
# epochs of the converged config; nothing else changed):
#   known:  TF init + TF order    (PyTorch exact replay)            -> exit epoch 2
#   known:  torch init/keras init + torch order (native runs)       -> exit epoch 20 (both inits)
#   X1:     PyTorch keras init (seed 1, as the native run) + TF's recorded order and z-scores
#   X2:     TF's recorded init weights + PyTorch's seed-1 order and z-scores
#   setsid nohup results/verification/run_diag_order_swap.sh X1|X2 > results/verification/logs/diag_X1|X2.log 2>&1 < /dev/null &
cd /home/ubuntu/code/RouteNet-Gauss-v2 || exit 1
C=trex_multiburst/RouteNetGauss/delay/seed_1
R=tensorflow_version_gt/replay/$C
common=(--dataset trex_multiburst --target delay --seed 1 --epochs 25 --steps 500 --shuffle-buffer 200 --patience 15
        --save-best-only --device cuda --threads 1 --use-wandb)
case "$1" in
    X1) exec /home/ubuntu/anaconda3/envs/RG_torch/bin/python experiment.py "${common[@]}" --experiment-name torch_diag_X1_tforder \
            --sample-order "$R/sample_order.npy" --z-scores "$R/z_scores.pkl" "${@:2}" ;;
    X2) exec /home/ubuntu/anaconda3/envs/RG_torch/bin/python experiment.py "${common[@]}" --experiment-name torch_diag_X2_torchorder \
            --init-weights "$R/init_weights.npz" \
            --sample-order "results/torch_converged_native/$C/sample_order_used.npy" \
            --z-scores "normalization/torch_converged_native/$C/z_scores.pkl" "${@:2}" ;;
    *) echo "usage: $0 X1|X2 [--resume]"; exit 2 ;;
esac
