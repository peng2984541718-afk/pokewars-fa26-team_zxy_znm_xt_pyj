#!/usr/bin/env bash
# Train a second, lighter-augmentation WRN after the first WRN CV queue.
set -euo pipefail

python_bin=${PYTHON_BIN:-python3}
cd "$(dirname "$0")"
mkdir -p logs outputs/large

if [[ "${1:-}" != "--execute" ]]; then
    previous_queue_pid=${1:?Expected PID of the existing WRN queue}
    while ! grep -q '^Completed all queued wrn folds$' logs/wrn_queue.log 2>/dev/null; do
        if ! kill -0 "$previous_queue_pid" 2>/dev/null; then
            echo "Previous WRN queue exited before completion" >&2
            exit 1
        fi
        sleep 20
    done
    exec timeout 4h bash "$0" --execute
fi

for fold in 0 1 2; do
    prefix="wrn28_10_mild_f${fold}"
    echo "Starting ${prefix}"
    "$python_bin" -u train_large_image.py --model wrn28_10 --fold "$fold" \
        --training-seed 314159 --epochs 40 --batch-size 128 \
        --translate-pixels 1.0 --rotate-degrees 8 \
        --scale-min 0.95 --scale-max 1.05 \
        --data-dir . --output-dir outputs/large --run-name "$prefix" \
        > "logs/${prefix}.log" 2>&1
    grep '^BEST ' "logs/${prefix}.log" | tail -1

    echo "Starting ${prefix}_stage2"
    "$python_bin" -u train_large_image.py --model wrn28_10 --fold "$fold" \
        --training-seed 314159 --epochs 30 --batch-size 128 \
        --translate-pixels 1.0 --rotate-degrees 8 \
        --scale-min 0.95 --scale-max 1.05 \
        --initial-weights "outputs/large/${prefix}_best.pt" \
        --learning-rate 1e-4 --min-learning-rate 1e-7 \
        --warmup-epochs 1 --min-epochs 15 --patience 8 \
        --min-improvement 0.001 --data-dir . --output-dir outputs/large \
        --run-name "${prefix}_stage2" > "logs/${prefix}_stage2.log" 2>&1
    grep '^BEST ' "logs/${prefix}_stage2.log" | tail -1
done

echo 'Completed mild-augmentation WRN three-fold experiment'
