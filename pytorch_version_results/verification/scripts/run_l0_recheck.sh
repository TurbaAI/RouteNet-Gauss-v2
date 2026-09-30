#!/usr/bin/env bash
# Re-run the L0 forward-parity check today on 3 representative checkpoints (converged trex delay,
# paper mawi delay, paper trex_multiburst_filtered delay — the tightest one) and compare every
# summary field with the committed reports in pytorch_version_results/parity/.
#   results/verification/run_l0_recheck.sh 2>&1 | tee results/verification/logs/l0_recheck.log
cd /home/ubuntu/code/RouteNet-Gauss-v2 || exit 1
PY=/home/ubuntu/anaconda3/envs/RG_torch/bin/python
OUT=results/verification/l0_recheck
mkdir -p "$OUT"
run() {  # name checkpoint z_scores dataset [gt_predictions]
    local extra=()
    [ -n "$5" ] && extra=(--gt-predictions "$5")
    $PY parity/l0_forward.py --checkpoint "$2" --z-scores "$3" --dataset "$4" --partition test --target delay \
        --inference-mode --device cpu --threads 1 --out "$OUT/l0_$1.json" "${extra[@]}" > "$OUT/l0_$1.log" 2>&1
    echo "[l0 recheck] $1 rc=$?"
}
G=tensorflow_version_gt/converged
C=trex_multiburst/RouteNetGauss/delay/seed_1
run converged_trex_multiburst_delay_seed1 "$G/ckpt/$C/30-7.0663" "$G/normalization/$C/z_scores.pkl" trex_multiburst "$G/results/$C/predictions.npz"
P=paper_weights/mawi_pcaps/RouteNetGauss/delay
run paper_mawi_pcaps_delay "ckpt/$P/201-19.7350" "normalization/$P/z_scores.pkl" mawi_pcaps
P=paper_weights/trex_multiburst_filtered/RouteNetGauss/delay
run paper_trex_multiburst_filtered_delay "ckpt/$P/214-4.7122" "normalization/$P/z_scores.pkl" trex_multiburst
$PY - <<'EOF'
import json
KEYS = ["n_scenarios", "n_predictions", "all_targets_identical", "max_abs_diff", "worst_scale_rel_diff",
        "mape_tf", "mape_torch", "passed"]
for name in ["converged_trex_multiburst_delay_seed1", "paper_mawi_pcaps_delay", "paper_trex_multiburst_filtered_delay"]:
    new = json.load(open(f"results/verification/l0_recheck/l0_{name}.json"))
    old = json.load(open(f"pytorch_version_results/parity/l0_{name}.json"))
    same = all(new[k] == old[k] for k in KEYS)
    per_scen = new["scenarios"] == old["scenarios"]
    print(f"## {name}: summary fields {'IDENTICAL' if same else 'DIFFER'} to the committed report; "
          f"per-scenario rows {'identical' if per_scen else 'differ'}")
    for k in KEYS:
        print(f"   {k:24s} committed={old[k]!s:24s} today={new[k]!s:24s} {'' if new[k] == old[k] else '<-'}")
EOF
