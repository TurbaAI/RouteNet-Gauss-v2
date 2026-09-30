#!/usr/bin/env bash
# Copy the finished 2026-09-28/30 verification runs into pytorch_version_results/verification/
# (mirrors converged/runs/: <experiment>/trex_multiburst/RouteNetGauss/delay/seed_<n>/...). Never
# overwrites finished-run files (cp --update=none); the §7.6 state and histories are refreshed. Afterwards:
#   python parity/reliability_report.py --out pytorch_version_results/verification/reliability_report
cd /home/ubuntu/code/RouteNet-Gauss-v2 || exit 1
C=trex_multiburst/RouteNetGauss/delay
F=pytorch_version_results/verification
refresh() {  # like put, but overwrites: for files that grow while the §7.6 driver runs
    local src=$1 dst=$2; shift 2
    mkdir -p "$dst"
    for f in "$@"; do [ -e "$src/$f" ] && cp -f "$src/$f" "$dst/"; done
}
put() {  # src_dir dst_dir file...
    local src=$1 dst=$2; shift 2
    mkdir -p "$dst"
    for f in "$@"; do [ -e "$src/$f" ] && cp --update=none "$src/$f" "$dst/"; done
}
# TF runs (frozen TF code @ 2e30d5d): predictions only where the gated test metric needs them
put results/tf_runner/results/tf_setupcheck/$C/seed_1       $F/tf_setupcheck/$C/seed_1       history.csv metrics.json
put results/tf_runner/results/tf_envelope_1thread/$C/seed_1 $F/tf_envelope_1thread/$C/seed_1 history.csv metrics.json predictions.npz
put results/tf_runner/ckpt/tf_envelope_1thread/$C/seed_1    $F/tf_envelope_1thread/$C/seed_1 79-5.5902.index 79-5.5902.data-00000-of-00001
for s in 2 3; do
    put results/tf_runner/results/tf_plateau_probe/$C/seed_$s $F/tf_plateau_probe/$C/seed_$s history.csv metrics.json
done
# PyTorch runs
put results/torch_converged_native/$C/seed_1       $F/torch_converged_native/$C/seed_1 history.csv metrics.json predictions.npz step_losses.csv sample_order_used.npy
put ckpt/torch_converged_native/$C/seed_1          $F/torch_converged_native/$C/seed_1 42-7.0970.pt
put normalization/torch_converged_native/$C/seed_1 $F/torch_converged_native/$C/seed_1 z_scores.pkl
for s in 2 3; do
    put results/torch_plateau_probe/$C/seed_$s       $F/torch_plateau_probe/$C/seed_$s history.csv metrics.json step_losses.csv
    put normalization/torch_plateau_probe/$C/seed_$s $F/torch_plateau_probe/$C/seed_$s z_scores.pkl
done
# Q10 diagnosis: seed-1 order/init cross-over (stopped once past the plateau)
for x in torch_diag_X1_tforder torch_diag_X2_torchorder; do
    put results/$x/$C/seed_1 $F/diagnosis/$x/$C/seed_1 history.csv step_losses.csv
done
# §7.6 init draws (whatever has finished) + the TF-drawn initial weights
refresh results/verification/init_draws $F/init_draws state.json
for d in results/verification/init_draws/tf_seed_*; do
    put "$d" "$F/init_draws/$(basename "$d")" init_weights.npz
done
for d in results/torch_initdraw_*; do
    [ -d "$d" ] || continue
    n=$(basename "$d"); s=${n##*_s}
    refresh "$d/$C/seed_$s" "$F/init_draws/runs/$n/$C/seed_$s" history.csv
done
# L0 re-check reports and the scripts that produced everything above
put results/verification/l0_recheck $F/l0_recheck l0_converged_trex_multiburst_delay_seed1.json \
    l0_paper_mawi_pcaps_delay.json l0_paper_trex_multiburst_filtered_delay.json
put results/verification $F/scripts run_tf_envelope.sh run_tf_probes.sh run_torch_native.sh run_torch_probes.sh \
    run_l0_recheck.sh run_diag_order_swap.sh run_init_draws.py diag_init_stats.py status.py freeze.sh
du -sh $F
