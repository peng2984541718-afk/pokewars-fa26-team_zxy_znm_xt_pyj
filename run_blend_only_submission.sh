#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:${LD_LIBRARY_PATH:-}"
# Keep the running 6-view gallery on the GPU. Cavity is a few GB of 40x40 messages; CPU is enough.
export CUDA_VISIBLE_DEVICES=""
py="${PYTHON_BIN:-$HOME/pokewars-venv/bin/python}"
L=outputs/large

step() { printf '%s %s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$L/blend_only_status.txt"; }

step "BLEND_PACKS"
"$py" -u build_blend_only_packs.py

step "FOLD0_EVAL"
"$py" -u joint_constrained_inference.py --mode evaluate-fold0 \
    --fold0-scores "$L/blend_only_fold0_scores.npz" \
    --unlabeled-priors "$L/blend_only_unlabeled_f0only.npz" \
    --table "$L/pair_semi_nofold0_new_table_round1.npy" \
    --output "$L/score_fold0_blend_only.json" | tee logs/blend_only_fold0_eval.log

scale=$("$py" -c "import json;print(json.load(open('$L/score_fold0_blend_only.json'))['selected']['scale'])")
f0=$("$py" -c "import json;print(round(json.load(open('$L/score_fold0_blend_only.json'))['selected']['test_score'],3))")
step "BLEND_FOLD0=${f0}_scale=${scale}"

step "SUBMISSION"
"$py" -u joint_constrained_inference.py --mode predict-submission \
    --unlabeled-priors "$L/blend_only_unlabeled.npz" \
    --test-npz "$L/blend_only_test.npz" \
    --table "$L/pair_semi_v23_blend_table.npy" \
    --scale "$scale" \
    --output "$L/submission_phase2_blend_only.csv" | tee logs/blend_only_submission.log

step "BLEND_ONLY_COMPLETE"
