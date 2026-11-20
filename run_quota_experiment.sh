#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:${LD_LIBRARY_PATH:-}"
py="${PYTHON_BIN:-$HOME/pokewars-venv/bin/python}"
L=outputs/large
mkdir -p logs "$L"
step() { printf '%s %s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$L/quota_uniform_status.txt"; }

step "PACK"
"$py" -u pack_image_quota.py

step "FOLD0"
"$py" -u joint_constrained_inference.py --mode evaluate-fold0 \
    --fold0-scores "$L/quota_uniform_fold0_scores.npz" \
    --unlabeled-priors "$L/phase2_imp_ranker_unlabeled_f0only.npz" \
    --table "$L/pair_semi_nofold0_new_table_round1.npy" \
    --output "$L/score_fold0_quota_uniform.json" | tee logs/quota_uniform_fold0.log

scale=$("$py" -c "import json;print(json.load(open('$L/score_fold0_quota_uniform.json'))['selected']['scale'])")
f0=$("$py" -c "import json;print(round(json.load(open('$L/score_fold0_quota_uniform.json'))['selected']['test_score'],3))")
step "QUOTA_FOLD0=${f0}_scale=${scale}"

step "SUBMISSION"
"$py" -u joint_constrained_inference.py --mode predict-submission \
    --unlabeled-priors "$L/phase2_imp_ranker_unlabeled.npz" \
    --test-npz "$L/quota_uniform_test.npz" \
    --table "$L/pair_semi_v23_blend_table.npy" \
    --scale "$scale" \
    --output "$L/submission_quota_uniform.csv" | tee logs/quota_uniform_submission.log

step "QUOTA_COMPLETE"
