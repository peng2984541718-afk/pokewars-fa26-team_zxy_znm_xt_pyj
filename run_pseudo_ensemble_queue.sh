#!/usr/bin/env bash
# Rebuild the pipeline on the o + m + 2p WRN ensemble (p = pseudo-label WRN):
# OOF/test priors -> unlabeled priors and pair tables -> kNN features and ranker
# -> fold-0 end-to-end score -> submission.
set -euo pipefail

cd /home/pyjama/pokewars
py=/home/pyjama/pokewars-venv/bin/python
export LD_LIBRARY_PATH=/home/pyjama/pokewars/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:/home/pyjama/pokewars/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:/usr/local/cuda/lib64
mkdir -p logs outputs/large
L=outputs/large
status=$L/pseudo_ensemble_queue_status.txt
step() { printf '%s %s\n' "$1" "$(date -u +%FT%TZ)" >> "$status"; }
: > "$status"
trap 'step failed' ERR

prefixes=(wrn28_10 wrn28_10_mild wrn28_10_mild_pseudo)
weights=(1 1 2)
run=feature_rerank_pseudo_top20

step "1a combine OOF"
[[ -f $L/wrn_pseudo_top20_oof_compact.npz ]] || \
    $py -u combine_oof_probs.py \
        --inputs $L/oof_wrn28_10_rot5.npy $L/oof_wrn28_10_mild_rot5.npy $L/oof_wrn28_10_mild_pseudo_rot5.npy \
        --weights "${weights[@]}" --output $L/multibackbone_wrn_pseudo_rot5_oof.npy \
        --compact $L/wrn_pseudo_top20_oof_compact.npz > logs/combine_oof_pseudo.log 2>&1

step "1b test top20"
[[ -f $L/test_wrn_pseudo_rot5_top20.csv ]] || \
    $py -u predict_wrn_topk.py --prefixes "${prefixes[@]}" --weights "${weights[@]}" \
        --tta rotate_5deg --limit 160000 --top-k 20 \
        --output $L/test_wrn_pseudo_rot5_top20.csv > logs/test_wrn_pseudo_top20.log 2>&1

step "2a unlabeled priors"
[[ -f $L/unlabeled_wrn_pseudo_rot5_top20.csv ]] || \
    $py -u predict_wrn_topk.py --data-dir unl --prefixes "${prefixes[@]}" --weights "${weights[@]}" \
        --tta rotate_5deg --limit 68675 --top-k 20 \
        --output $L/unlabeled_wrn_pseudo_rot5_top20.csv > logs/unlabeled_wrn_pseudo_top20.log 2>&1

step "2b semi table full"
[[ -f $L/pair_semi_v2_full_table_round1.npy ]] || \
    $py -u train_pair_semi.py --full --rounds 1 --seeds 0 1 2 3 4 \
        --initial-table $L/pair_semi_full_table_round1.npy \
        --priors $L/unlabeled_wrn_pseudo_rot5_top20.csv \
        --output $L/pair_semi_v2_full.json > logs/pair_semi_v2_full.log 2>&1

step "2c semi tables without fold-0 pairs"
[[ -f $L/pair_semi_nofold0_new_table_round2.npy ]] || \
    $py -u train_pair_semi.py --exclude-fold0-pairs --rounds 2 --seeds 0 1 2 \
        --initial-table $L/pair_table_no_fold0.npy \
        --priors $L/unlabeled_wrn_pseudo_rot5_top20.csv \
        --output $L/pair_semi_nofold0_new.json > logs/pair_semi_nofold0_new.log 2>&1
[[ -f $L/pair_semi_nofold0_old_table_round2.npy ]] || \
    $py -u train_pair_semi.py --exclude-fold0-pairs --rounds 2 --seeds 0 1 2 \
        --initial-table $L/pair_table_no_fold0.npy \
        --priors $L/unlabeled_wrn_rot5_top20.csv \
        --output $L/pair_semi_nofold0_old.json > logs/pair_semi_nofold0_old.log 2>&1

step "3a kNN features"
for fold in 0 1 2; do
    [[ -f $L/${run}_f${fold}.npz ]] || \
        $py -u feature_rerank.py --fold $fold --prefixes "${prefixes[@]}" \
            --full-probs $L/multibackbone_wrn_pseudo_rot5_oof.npy --top-k 20 \
            --run-name $run > logs/${run}_f${fold}.log 2>&1
done

step "3b ranker CV"
$py -u evaluate_meta_rerank.py --run-name $run --class-feature \
    --dense-full $L/multibackbone_densenet169_oof.npy \
    --output $L/meta_rerank_pseudo_top20_cv.json > logs/meta_rerank_pseudo_top20.log 2>&1
$py -u export_ranker_fold0_scores.py --run-name $run --expected-top1 -1 \
    --output $L/meta_pseudo_top20_fold0_scores.npz > logs/export_ranker_pseudo_fold0.log 2>&1

step "3c fold-0 end-to-end"
for ranker in old new; do
    scores=$L/meta_top20_fold0_scores.npz
    [[ $ranker == new ]] && scores=$L/meta_pseudo_top20_fold0_scores.npz
    for table in mlp semi_old semi_new; do
        case $table in
            mlp) path=$L/pair_table_no_fold0.npy ;;
            semi_old) path=$L/pair_semi_nofold0_old_table_round2.npy ;;
            semi_new) path=$L/pair_semi_nofold0_new_table_round2.npy ;;
        esac
        $py -u evaluate_score_fold0_mlp.py --ranker-scores $scores --table $path \
            --output $L/score_fold0_e2e_${ranker}ranker_${table}.json \
            > logs/score_fold0_e2e_${ranker}ranker_${table}.log 2>&1
    done
done

step "3d test ranker"
[[ -f $L/test_wrn_pseudo_rot5_top20_features.npz ]] || \
    $py -u predict_test_top20_features.py --prefixes "${prefixes[@]}" \
        --candidates $L/test_wrn_pseudo_rot5_top20.csv \
        --output $L/test_wrn_pseudo_rot5_top20_features.npz > logs/test_pseudo_top20_features.log 2>&1
$py -u predict_meta_top20_test.py --run-name $run \
    --test-features test_wrn_pseudo_rot5_top20_features.npz \
    --cv-report meta_rerank_pseudo_top20_cv.json \
    --output $L/test_meta_pseudo_top20_top10.csv > logs/test_meta_pseudo_top20.log 2>&1

step "3e submission"
$py -u predict_score_submission.py --table $L/pair_semi_v2_full_table_round1.npy \
    --candidates $L/test_meta_pseudo_top20_top10.csv \
    --output $L/submission_pseudo_ensemble.csv > logs/submission_pseudo_ensemble.log 2>&1
step complete
