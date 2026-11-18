#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python_bin=${PYTHON_BIN:-python3}
if [[ -d cuda_extra ]]; then
    export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
mkdir -p logs outputs/large
status_file=outputs/large/wrn40_background_status.txt
printf 'running %s\n' "$(date -u +%FT%TZ)" > "$status_file"
trap 'printf "failed %s\n" "$(date -u +%FT%TZ)" > "$status_file"' ERR

existing_pid=${1:-}
if [[ -n "$existing_pid" ]]; then
    while ps -p "$existing_pid" -o args= 2>/dev/null | grep -q 'train_large_image.py --model wrn40_10 --fold 0'; do
        sleep 15
    done
fi

train_fold() {
    local fold=$1
    local log_file="logs/wrn40_10_f${fold}_background.log"
    "$python_bin" -u train_large_image.py --model wrn40_10 --fold "$fold" \
        --epochs 40 --warmup-epochs 2 --patience 8 --min-epochs 25 \
        --run-name "wrn40_10_f${fold}" --data-dir . > "$log_file" 2>&1
    if ! grep -q '^BEST ' "$log_file"; then
        echo "Fold $fold did not finish with a best checkpoint" >&2
        return 1
    fi
}

if [[ ! -f outputs/large/wrn40_10_f0_best.pt ]] || \
   ! grep -q '^BEST ' logs/wrn40_10_f0.log; then
    echo "Original fold-0 process stopped early; restarting it in detached mode" >&2
    train_fold 0
fi

for fold in 0 1 2; do
    if [[ "$fold" -gt 0 ]]; then
        train_fold "$fold"
    fi
    "$python_bin" -u export_new_model_oof.py --model wrn40_10 --fold "$fold" \
        --checkpoint "outputs/large/wrn40_10_f${fold}_best.pt" \
        --output "outputs/large/wrn40_10_f${fold}_oof.npz"
done

"$python_bin" -u assemble_new_model_oof.py --prefix wrn40_10 \
    --output outputs/large/multibackbone_wrn40_10_oof.npy
"$python_bin" -u evaluate_four_backbone_oof.py
printf 'complete %s\n' "$(date -u +%FT%TZ)" > "$status_file"
