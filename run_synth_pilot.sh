#!/usr/bin/env bash
# Fold-0 pilot: fine-tune the pseudo-label WRN with and without re-degraded
# clean silhouettes (same schedule), then compare accuracy per noise decile.
set -euo pipefail

cd "$(dirname "$0")"
py=${PYTHON_BIN:-python3}
if [[ -d cuda_extra ]]; then
    export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
L=outputs/large
status=$L/synth_pilot_status.txt
step() { printf '%s %s\n' "$1" "$(date -u +%FT%TZ)" >> "$status"; }
: > "$status"
trap 'step failed' ERR

base=$L/wrn28_10_mild_pseudo_f0_stage2_best.pt
common=(--model wrn28_10 --fold 0 --training-seed 314159 --batch-size 128
        --translate-pixels 1.0 --rotate-degrees 8 --scale-min 0.95 --scale-max 1.05
        --data-dir . --output-dir $L --external-bank $L/pseudo_bank_f0.npz --external-fraction 0.35
        --initial-weights $base --epochs 25 --learning-rate 1e-4 --min-learning-rate 1e-7
        --warmup-epochs 1 --min-epochs 25 --patience 25 --min-improvement 0.0005)

[[ -f $L/degradation_f0.npz ]] || $py -u build_degradation_data.py --fold 0 > logs/degradation_f0.log 2>&1

step "synthetic fine-tune"
$py -u train_large_image_external.py "${common[@]}" --run-name wrn28_10_mild_pseudo_synth_f0 \
    --synthetic-data $L/degradation_f0.npz --synthetic-fraction 0.5 \
    > logs/wrn28_10_mild_pseudo_synth_f0.log 2>&1
grep '^BEST ' logs/wrn28_10_mild_pseudo_synth_f0.log | tail -1 >> "$status"

step "control fine-tune"
$py -u train_large_image_external.py "${common[@]}" --run-name wrn28_10_mild_pseudo_ctrl_f0 \
    > logs/wrn28_10_mild_pseudo_ctrl_f0.log 2>&1
grep '^BEST ' logs/wrn28_10_mild_pseudo_ctrl_f0.log | tail -1 >> "$status"

step "noise-decile evaluation"
$py -u evaluate_noise_bins.py --fold 0 --output $L/synth_pilot_noise_bins.json \
    --checkpoints $base $L/wrn28_10_mild_pseudo_synth_f0_last.pt $L/wrn28_10_mild_pseudo_ctrl_f0_last.pt \
    > logs/synth_pilot_noise_bins.log 2>&1
cat logs/synth_pilot_noise_bins.log >> "$status"
step complete
