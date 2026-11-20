#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:${LD_LIBRARY_PATH:-}"
py="${PYTHON_BIN:-$HOME/pokewars-venv/bin/python}"
L=outputs/large
mkdir -p logs "$L"
step() { printf '%s %s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$L/fast_ab_status.txt"; }

step "DEBIAS0"
"$py" -u joint_constrained_inference.py --mode predict-submission \
    --unlabeled-priors "$L/phase2_imp_ranker_unlabeled.npz" \
    --test-npz "$L/phase2_imp_ranker_test.npz" \
    --table "$L/pair_semi_v23_blend_table.npy" \
    --scale 1.05 --debias 0 \
    --output "$L/submission_debias0.csv" | tee logs/fast_ab_debias0.log

step "PACK_GATED"
"$py" -u pack_gated_remain.py

step "GATED_REMAIN"
"$py" -u joint_constrained_inference.py --mode predict-submission \
    --unlabeled-priors "$L/phase2_imp_ranker_unlabeled.npz" \
    --test-npz "$L/gated_remain_test.npz" \
    --table "$L/pair_semi_v23_blend_table.npy" \
    --scale 1.05 --debias 0 \
    --output "$L/submission_gated_remain.csv" | tee logs/fast_ab_gated.log

step "FAST_AB_COMPLETE"
