#!/usr/bin/env bash
# Keep two architecture queues occupied on one GPU after fold-0 screening.
set -euo pipefail

architecture=${1:?Expected wrn or conv}
wait_pid=${2:?Expected PID of the currently running fold-0 job}
wait_log=${3:?Expected log path for that job}
python_bin=${PYTHON_BIN:-python3}
cd "$(dirname "$0")"
mkdir -p logs outputs/large

while ! grep -q '^BEST ' "$wait_log" 2>/dev/null; do
    if ! kill -0 "$wait_pid" 2>/dev/null; then
        echo "Waiting job exited without BEST: $wait_log" >&2
        exit 1
    fi
    sleep 15
done

case "$architecture" in
    wrn)
        model=wrn28_10
        prefix=wrn28_10
        stage2_lr=1e-4
        extra=()
        ;;
    conv)
        model=convnextv2_nano
        prefix=convnextv2_scratch64
        stage2_lr=5e-5
        extra=(--convnext-size 64)
        ;;
    *) echo "Unknown architecture: $architecture" >&2; exit 2 ;;
esac

run_stage2() {
    local fold=$1
    local initial="outputs/large/${prefix}_f${fold}_best.pt"
    local run_name="${prefix}_f${fold}_stage2"
    echo "Starting $run_name from $initial"
    "$python_bin" -u train_large_image.py --model "$model" --fold "$fold" \
        --epochs 60 --batch-size 128 --data-dir . --output-dir outputs/large \
        --run-name "$run_name" --initial-weights "$initial" \
        --learning-rate "$stage2_lr" --min-learning-rate 1e-7 \
        --warmup-epochs 1 --min-epochs 20 --patience 10 \
        --min-improvement 0.001 "${extra[@]}" > "logs/${run_name}.log" 2>&1
    grep '^BEST ' "logs/${run_name}.log" | tail -1
}

if [[ "$architecture" == wrn ]]; then
    run_stage2 0
fi

for fold in 1 2; do
    run_name="${prefix}_f${fold}"
    echo "Starting $run_name"
    "$python_bin" -u train_large_image.py --model "$model" --fold "$fold" \
        --epochs 40 --batch-size 128 --data-dir . --output-dir outputs/large \
        --run-name "$run_name" "${extra[@]}" > "logs/${run_name}.log" 2>&1
    grep '^BEST ' "logs/${run_name}.log" | tail -1
    run_stage2 "$fold"
done

echo "Completed all queued $architecture folds"
