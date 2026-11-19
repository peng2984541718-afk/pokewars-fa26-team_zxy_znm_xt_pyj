#!/usr/bin/env bash
# Global constrained graphical inference + v2 unlabeled ranker priors + v3 PairMLP.
set -euo pipefail

cd "$(dirname "$0")"
py=${PYTHON_BIN:-python3}
if [[ -d cuda_extra ]]; then
    export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
L=outputs/large
mkdir -p logs "$L"
status=$L/constrained_v3_status.txt
step() { printf '%s %s\n' "$1" "$(date -u +%FT%TZ)" >> "$status"; }
: > "$status"
trap 'step failed' ERR

prefixes=(wrn28_10 wrn28_10_mild wrn28_10_mild_pseudo wrn28_10_mild_pseudo_synth wrn28_10_mild_pseudo_ctrl)
weights=(1 1 2 2 2)
run=feature_rerank_v2_top20

step "1_oof_and_test_top20"
[[ -f $L/meta_rerank_v2_top20_oof.npz ]] || \
    "$py" -u evaluate_meta_rerank.py --run-name $run --class-feature \
        --dense-full $L/multibackbone_densenet169_oof.npy \
        --save-predictions $L/meta_rerank_v2_top20_oof.npz \
        --output $L/meta_rerank_v2_top20_cv.json > logs/meta_rerank_v2_oof.log 2>&1

[[ -f $L/test_meta_v2_top20.npz ]] || \
    "$py" -u predict_meta_top20_test.py --run-name $run \
        --test-features test_wrn_v2_rot5_top20_features.npz \
        --save-top20-npz $L/test_meta_v2_top20.npz \
        --cv-report meta_rerank_v2_top20_cv.json \
        --output $L/test_meta_v2_top20_top10.csv > logs/test_meta_v2_top20.log 2>&1

step "2_unlabeled_v2_ranker_priors"
[[ -f $L/unlabeled_wrn_v2_f0only_top20.csv ]] || \
    "$py" -u predict_wrn_topk.py --data-dir unl \
        --prefixes "${prefixes[@]}" --weights "${weights[@]}" --folds 0 \
        --tta rotate_5deg --limit 68675 --top-k 20 \
        --output $L/unlabeled_wrn_v2_f0only_top20.csv > logs/unlabeled_wrn_v2_f0only.log 2>&1

[[ -f $L/unlabeled_wrn_v2_rot5_top20.csv ]] || \
    "$py" -u predict_wrn_topk.py --data-dir unl \
        --prefixes "${prefixes[@]}" --weights "${weights[@]}" \
        --tta rotate_5deg --limit 68675 --top-k 20 \
        --output $L/unlabeled_wrn_v2_rot5_top20.csv > logs/unlabeled_wrn_v2_top20.log 2>&1

[[ -f $L/unlabeled_convnextv2_probs.npy ]] || \
    "$py" -u predict_backbone_test_probs.py --data-dir unl \
        --model convnextv2_nano --checkpoint-prefix convnextv2_scratch64 \
        --output $L/unlabeled_convnextv2_probs.npy > logs/unlabeled_convnext.log 2>&1

[[ -f $L/unlabeled_densenet169_probs.npy ]] || \
    "$py" -u predict_backbone_test_probs.py --data-dir unl \
        --model densenet169_cifar --checkpoint-prefix densenet169_cifar \
        --output $L/unlabeled_densenet169_probs.npy > logs/unlabeled_densenet.log 2>&1

[[ -f $L/unlabeled_wrn_v2_rot5_top20_features.npz ]] || \
    "$py" -u predict_test_top20_features.py --data-dir . --query-dir unl \
        --prefixes "${prefixes[@]}" \
        --candidates $L/unlabeled_wrn_v2_rot5_top20.csv \
        --output $L/unlabeled_wrn_v2_rot5_top20_features.npz \
        > logs/unlabeled_v2_top20_features.log 2>&1

[[ -f $L/unlabeled_meta_v2_top20.npz ]] || \
    "$py" -u predict_meta_top20_test.py --data-dir . --query-dir unl --run-name $run \
        --test-features unlabeled_wrn_v2_rot5_top20_features.npz \
        --conv-probs unlabeled_convnextv2_probs.npy \
        --dense-probs unlabeled_densenet169_probs.npy \
        --save-top20-npz $L/unlabeled_meta_v2_top20.npz \
        --cv-report meta_rerank_v2_top20_cv.json \
        --output $L/unlabeled_meta_v2_top20_top10.csv > logs/unlabeled_meta_v2.log 2>&1

step "3_fold0_constrained_eval"
"$py" -u constrained_inference.py --mode evaluate-fold0 \
    --oof-npz $L/meta_rerank_v2_top20_oof.npz \
    --ranker-fold0-scores $L/meta_v2_top20_fold0_scores.npz \
    --unlabeled-priors $L/unlabeled_wrn_v2_f0only_top20.csv \
    --table $L/pair_semi_nofold0_new_table_round1.npy \
    --output $L/score_fold0_constrained_v2.json > logs/score_fold0_constrained_v2.log 2>&1
best_scale=$("$py" -c "import json;print(json.load(open('$L/score_fold0_constrained_v2.json'))['selected']['scale'])")
best_f0=$("$py" -c "import json;print(round(json.load(open('$L/score_fold0_constrained_v2.json'))['selected']['test_score'],3))")
step "fold0_v2_constrained_score=${best_f0}_scale=${best_scale}"

step "4_submission_v2_constrained"
"$py" -u constrained_inference.py --mode predict-submission \
    --oof-npz $L/meta_rerank_v2_top20_oof.npz \
    --unlabeled-priors $L/unlabeled_meta_v2_top20.npz \
    --test-npz $L/test_meta_v2_top20.npz \
    --table $L/pair_semi_v2_full_table_round1.npy \
    --scale "$best_scale" \
    --save-pseudo-battles $L/unlabeled_constrained_battles_v2.npz \
    --output $L/submission_v2_constrained.csv > logs/submission_v2_constrained.log 2>&1

step "5_pair_semi_v3_training"
[[ -f $L/pair_semi_v3_full_table_round1.npy ]] || \
    "$py" -u train_pair_semi.py --full --rounds 1 --seeds 0 1 2 3 4 \
        --initial-table $L/pair_semi_v2_full_table_round1.npy \
        --pseudo-battles $L/unlabeled_constrained_battles_v2.npz \
        --output $L/pair_semi_v3_full.json > logs/pair_semi_v3_full.log 2>&1

step "6_submission_v3_constrained"
"$py" -u constrained_inference.py --mode predict-submission \
    --oof-npz $L/meta_rerank_v2_top20_oof.npz \
    --unlabeled-priors $L/unlabeled_meta_v2_top20.npz \
    --test-npz $L/test_meta_v2_top20.npz \
    --table $L/pair_semi_v3_full_table_round1.npy \
    --scale "$best_scale" \
    --output $L/submission_v3_constrained.csv > logs/submission_v3_constrained.log 2>&1

step complete
