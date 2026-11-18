#!/usr/bin/env bash
# Mild-augmentation WRN-28-10 with fold-safe pseudo-labeled images, all three folds.
set -euo pipefail

cd "$(dirname "$0")"
python_bin=${PYTHON_BIN:-python3}
if [[ -d cuda_extra ]]; then
    export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
mkdir -p logs outputs/large
status_file=outputs/large/pseudo_wrn_queue_status.txt
printf 'running %s\n' "$(date -u +%FT%TZ)" > "$status_file"
trap 'printf "failed %s\n" "$(date -u +%FT%TZ)" > "$status_file"' ERR

wait_pid=${1:-}
if [[ -n "$wait_pid" ]]; then
    while kill -0 "$wait_pid" 2>/dev/null; do sleep 15; done
fi

common=(--model wrn28_10 --training-seed 314159 --batch-size 128
        --translate-pixels 1.0 --rotate-degrees 8 --scale-min 0.95 --scale-max 1.05
        --data-dir . --output-dir outputs/large --external-fraction 0.35)

for fold in 0 1 2; do
    prefix="wrn28_10_mild_pseudo_f${fold}"
    bank="outputs/large/pseudo_bank_f${fold}.npz"
    if [[ ! -f "$bank" ]]; then
        "$python_bin" -u predict_wrn_topk.py --data-dir unl --variant ensemble \
            --folds "$fold" --tta rotate_5deg --limit 68675 --top-k 20 \
            --output "outputs/large/unlabeled_wrn_rot5_top20_f${fold}only.csv" \
            > "logs/unlabeled_wrn_predict_f${fold}.log" 2>&1
        "$python_bin" -u build_pseudo_label_bank.py \
            --priors "outputs/large/unlabeled_wrn_rot5_top20_f${fold}only.csv" \
            --output "$bank" > "logs/pseudo_bank_f${fold}.log" 2>&1
    fi
    if [[ ! -f "outputs/large/${prefix}_best.pt" ]] || ! grep -q '^BEST ' "logs/${prefix}.log"; then
        "$python_bin" -u train_large_image_external.py "${common[@]}" --fold "$fold" \
            --epochs 40 --run-name "$prefix" --external-bank "$bank" \
            > "logs/${prefix}.log" 2>&1
    fi
    grep '^BEST ' "logs/${prefix}.log" | tail -1
    "$python_bin" -u train_large_image_external.py "${common[@]}" --fold "$fold" \
        --epochs 30 --initial-weights "outputs/large/${prefix}_best.pt" \
        --learning-rate 1e-4 --min-learning-rate 1e-7 --warmup-epochs 1 \
        --min-epochs 15 --patience 8 --min-improvement 0.001 \
        --run-name "${prefix}_stage2" --external-bank "$bank" \
        > "logs/${prefix}_stage2.log" 2>&1
    grep '^BEST ' "logs/${prefix}_stage2.log" | tail -1
done
printf 'complete %s\n' "$(date -u +%FT%TZ)" > "$status_file"
