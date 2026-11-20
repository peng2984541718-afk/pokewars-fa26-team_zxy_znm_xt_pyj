#!/usr/bin/env bash
# Leak-free ResNeXt-29 on folds 1 and 2. Stage 1 uses that fold's transductive
# bank. Stage 2 distills only the labeled OOF teacher. The full unlabeled pack
# was fit on every fold, so it is not a legal soft target here.
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
py="${PYTHON_BIN:-$HOME/pokewars-venv/bin/python}"
L=outputs/large
mkdir -p logs "$L"
status=$L/resnext_status.txt
step() { printf '%s %s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$status"; }

for fold in 1 2; do
    step "STAGE1_F${fold}"
    "$py" -u train_large_image_external.py \
        --model resnext29_8x64d --fold "$fold" \
        --training-seed 271828 --batch-size 64 --eval-batch-size 128 --clean-impulses \
        --epochs 30 --warmup-epochs 2 --min-epochs 18 --patience 8 --min-improvement 0.0002 \
        --learning-rate 8e-4 --min-learning-rate 1e-6 \
        --translate-pixels 1.5 --rotate-degrees 12 --scale-min 0.9 --scale-max 1.1 \
        --external-bank "$L/transductive_bank_f${fold}.npz" --external-fraction 0.45 \
        --data-dir . --output-dir "$L" \
        --run-name "resnext29_f${fold}" > "logs/resnext29_f${fold}.log" 2>&1

    stage1=$("$py" -c "import json; d=json.load(open('$L/resnext29_f${fold}_best.json')); print(d['epoch'], d['top1'])")
    step "STAGE1_F${fold}_BEST=${stage1}"
    "$py" - << PY
epoch, top1 = "${stage1}".split()
if float(top1) < 0.940:
    raise SystemExit(f"fold ${fold} stage1 {float(top1):.4f} is below 0.940")
PY

    step "STAGE2_F${fold}"
    "$py" -u train_large_image_external.py \
        --model resnext29_8x64d --fold "$fold" \
        --training-seed 271828 --batch-size 64 --eval-batch-size 128 --clean-impulses \
        --epochs 12 --warmup-epochs 1 --min-epochs 8 --patience 6 --min-improvement 0.0002 \
        --learning-rate 1e-4 --min-learning-rate 1e-6 \
        --translate-pixels 1.0 --rotate-degrees 8 --scale-min 0.95 --scale-max 1.05 \
        --initial-weights "$L/resnext29_f${fold}_best.pt" \
        --distill-labeled "$L/distill_labeled_oof.npz" --distill-alpha 0.5 \
        --save-valid-probs "$L/resnext29_dist_f${fold}_probs.npy" \
        --data-dir . --output-dir "$L" \
        --run-name "resnext29_dist_f${fold}" > "logs/resnext29_dist_f${fold}.log" 2>&1

    pilot=$("$py" -c "import json; d=json.load(open('$L/resnext29_dist_f${fold}_best.json')); print(d['epoch'], d['top1'])")
    step "DISTILL_F${fold}_BEST=${pilot}"
done
step "FOLDS_12_DONE"
