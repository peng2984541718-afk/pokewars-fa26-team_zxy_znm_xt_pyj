#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
py=/home/pyjama/pokewars-venv/bin/python
out=outputs/stage2_channels_v1
mkdir -p "$out" logs
exec 9>"$out/noise_weighted_queue.lock"
flock -n 9 || exit 1
status="$out/noise_weighted_status.txt"
trap 'printf "FAILED %s\n" "$(date -u +%FT%TZ)" >> "$status"' ERR
"$py" tests/check_stage2_channels.py
for mode in noise_weighted; do
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)" >> "$status"
    extra=(--noise-sampling-weight 1.5)
    "$py" -u train_stage2_channels.py --model wrn28_10 --fold 0 \
        --initial-weights outputs/large/wrn28_10_imp_f0_best.pt \
        --data-dir . --output-dir "$out" --run-name "$mode" \
        --external-bank outputs/large/transductive_bank_f0.npz --external-fraction 0.45 \
        --epochs 20 --min-epochs 5 --patience 4 --min-improvement 0.0001 \
        --batch-size 128 --eval-batch-size 256 --learning-rate 5e-5 \
        --min-learning-rate 1e-6 --warmup-epochs 1 --training-seed 271828 \
        --translate-pixels 1 --rotate-degrees 8 --scale-min 0.95 --scale-max 1.05 \
        "${extra[@]}" > "logs/stage2_${mode}.log" 2>&1
    "$py" -u export_stage2_channels.py --mode "$mode" > "logs/stage2_${mode}_export.log" 2>&1
    printf 'DONE %s %s\n' "$mode" "$(date -u +%FT%TZ)" >> "$status"
done
"$py" -u evaluate_noise_integration.py > logs/noise_integration.log 2>&1
printf 'ALL_COMPLETE %s\n' "$(date -u +%FT%TZ)" >> "$status"
