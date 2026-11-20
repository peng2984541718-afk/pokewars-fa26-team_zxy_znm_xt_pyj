#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:${LD_LIBRARY_PATH:-}"
py="${PYTHON_BIN:-$HOME/pokewars-venv/bin/python}"
L=outputs/large
status=$L/distill_status.txt
step() { printf "%s %s\n" "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$status"; }
step "EXPORT_RESUME"
"$py" -u export_probs_rot5.py --model densenet169_cifar --prefixes densenet169_distfull \
    --folds 0 --clean-impulses --split query --query-dir unl \
    --output $L/unl_dense_distill.npy > logs/distill_export_unl_dense.log 2>&1
"$py" -u export_probs_rot5.py --model wrn28_10 --prefixes wrn28_10_distfull \
    --folds 0 --clean-impulses --split query --query-dir unl \
    --output $L/unl_wrn28_distill.npy > logs/distill_export_unl_wrn.log 2>&1
"$py" -u export_probs_rot5.py --model densenet169_cifar --prefixes densenet169_distfull \
    --folds 0 --clean-impulses --split query --query-dir . \
    --output $L/test_dense_distill.npy > logs/distill_export_test_dense.log 2>&1
"$py" -u export_probs_rot5.py --model wrn28_10 --prefixes wrn28_10_distfull \
    --folds 0 --clean-impulses --split query --query-dir . \
    --output $L/test_wrn28_distill.npy > logs/distill_export_test_wrn.log 2>&1
step "SUBMISSION"
"$py" -u mix_distill_posteriors.py \
    --ranker $L/phase2_imp_ranker_unlabeled.npz \
    --probs $L/unl_dense_distill.npy $L/unl_wrn28_distill.npy --prob-weights 0.6 0.4 \
    --weight 0.25 --output $L/distill_mix_unlabeled.npz
"$py" -u mix_distill_posteriors.py \
    --ranker $L/phase2_imp_ranker_test.npz \
    --probs $L/test_dense_distill.npy $L/test_wrn28_distill.npy --prob-weights 0.6 0.4 \
    --weight 0.25 --output $L/distill_mix_test.npz
scale=$("$py" -c "import json;print(json.load(open(\"$L/score_fold0_phase2_impulse_hammer.json\"))[\"selected\"][\"scale\"])")
"$py" -u joint_constrained_inference.py --mode predict-submission \
    --unlabeled-priors $L/distill_mix_unlabeled.npz \
    --test-npz $L/distill_mix_test.npz \
    --table $L/pair_semi_v23_blend_table.npy \
    --scale "$scale" \
    --output $L/submission_distill.csv | tee logs/distill_submission.log
step "DISTILL_COMPLETE"
