#!/usr/bin/env bash
# Fine-tune all 3 folds (synth + ctrl), rebuild OOF ensemble, kNN features,
# ranker, fold-0 end-to-end evaluation, and final submission.
set -euo pipefail

cd /home/pyjama/pokewars
py=/home/pyjama/pokewars-venv/bin/python
export LD_LIBRARY_PATH=/home/pyjama/pokewars/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:/home/pyjama/pokewars/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:/usr/local/cuda/lib64
L=outputs/large
status=$L/synth_all_folds_status.txt
step() { printf '%s %s\n' "$1" "$(date -u +%FT%TZ)" >> "$status"; }
: > "$status"
trap 'step failed' ERR
mkdir -p logs

# Fold-0 checkpoints already exist from run_synth_pilot.sh.
# Rename _last.pt → _stage2_best.pt style so export_wrn_oof_probs.py finds them.
cp_if_missing() { [[ -f "$2" ]] || cp "$1" "$2"; }
cp_if_missing $L/wrn28_10_mild_pseudo_synth_f0_last.pt $L/wrn28_10_mild_pseudo_synth_f0_stage2_best.pt
cp_if_missing $L/wrn28_10_mild_pseudo_ctrl_f0_last.pt  $L/wrn28_10_mild_pseudo_ctrl_f0_stage2_best.pt

common_ft=(--model wrn28_10 --training-seed 314159 --batch-size 128
           --translate-pixels 1.0 --rotate-degrees 8 --scale-min 0.95 --scale-max 1.05
           --data-dir . --output-dir $L
           --external-fraction 0.35
           --epochs 25 --learning-rate 1e-4 --min-learning-rate 1e-7
           --warmup-epochs 1 --min-epochs 25 --patience 25 --min-improvement 0.0005)

# ── Step 1: fine-tune folds 1 and 2 ──────────────────────────────────────────
for fold in 1 2; do
    step "synth_f${fold}"
    bank=$L/pseudo_bank_f${fold}.npz
    deg=$L/degradation_f${fold}.npz
    base=$L/wrn28_10_mild_pseudo_f${fold}_stage2_best.pt

    [[ -f $deg ]] || $py -u build_degradation_data.py --fold $fold \
        > logs/degradation_f${fold}.log 2>&1

    [[ -f $L/wrn28_10_mild_pseudo_synth_f${fold}_stage2_best.pt ]] || {
        $py -u train_large_image_external.py "${common_ft[@]}" --fold $fold \
            --external-bank $bank \
            --initial-weights $base \
            --synthetic-data $deg --synthetic-fraction 0.5 \
            --run-name wrn28_10_mild_pseudo_synth_f${fold} \
            > logs/wrn28_10_mild_pseudo_synth_f${fold}.log 2>&1
        cp $L/wrn28_10_mild_pseudo_synth_f${fold}_last.pt \
           $L/wrn28_10_mild_pseudo_synth_f${fold}_stage2_best.pt
    }
    grep '^BEST ' logs/wrn28_10_mild_pseudo_synth_f${fold}.log | tail -1 >> "$status"

    step "ctrl_f${fold}"
    [[ -f $L/wrn28_10_mild_pseudo_ctrl_f${fold}_stage2_best.pt ]] || {
        $py -u train_large_image_external.py "${common_ft[@]}" --fold $fold \
            --external-bank $bank \
            --initial-weights $base \
            --run-name wrn28_10_mild_pseudo_ctrl_f${fold} \
            > logs/wrn28_10_mild_pseudo_ctrl_f${fold}.log 2>&1
        cp $L/wrn28_10_mild_pseudo_ctrl_f${fold}_last.pt \
           $L/wrn28_10_mild_pseudo_ctrl_f${fold}_stage2_best.pt
    }
    grep '^BEST ' logs/wrn28_10_mild_pseudo_ctrl_f${fold}.log | tail -1 >> "$status"
done

# ── Step 2: export per-prefix OOF probabilities ───────────────────────────────
step "oof_synth"
[[ -f $L/oof_wrn28_10_mild_pseudo_synth_rot5.npy ]] || \
    $py -u export_wrn_oof_probs.py --prefix wrn28_10_mild_pseudo_synth \
        --output $L/oof_wrn28_10_mild_pseudo_synth_rot5.npy \
        > logs/oof_synth.log 2>&1

step "oof_ctrl"
[[ -f $L/oof_wrn28_10_mild_pseudo_ctrl_rot5.npy ]] || \
    $py -u export_wrn_oof_probs.py --prefix wrn28_10_mild_pseudo_ctrl \
        --output $L/oof_wrn28_10_mild_pseudo_ctrl_rot5.npy \
        > logs/oof_ctrl.log 2>&1

# ── Step 3: combine OOF  o(1)+m(1)+p(2)+synth(2)+ctrl(2) ────────────────────
step "combine_oof_v2"
[[ -f $L/wrn_v2_top20_oof_compact.npz ]] || \
    $py -u combine_oof_probs.py \
        --inputs \
            $L/oof_wrn28_10_rot5.npy \
            $L/oof_wrn28_10_mild_rot5.npy \
            $L/oof_wrn28_10_mild_pseudo_rot5.npy \
            $L/oof_wrn28_10_mild_pseudo_synth_rot5.npy \
            $L/oof_wrn28_10_mild_pseudo_ctrl_rot5.npy \
        --weights 1 1 2 2 2 \
        --output $L/multibackbone_wrn_v2_rot5_oof.npy \
        --compact $L/wrn_v2_top20_oof_compact.npz \
        > logs/combine_oof_v2.log 2>&1
cat logs/combine_oof_v2.log

# ── Step 4: test top-20 with 5-prefix ensemble ───────────────────────────────
step "test_top20_v2"
prefixes=(wrn28_10 wrn28_10_mild wrn28_10_mild_pseudo wrn28_10_mild_pseudo_synth wrn28_10_mild_pseudo_ctrl)
weights_arr=(1 1 2 2 2)
[[ -f $L/test_wrn_v2_rot5_top20.csv ]] || \
    $py -u predict_wrn_topk.py \
        --prefixes "${prefixes[@]}" --weights "${weights_arr[@]}" \
        --tta rotate_5deg --limit 160000 --top-k 20 \
        --output $L/test_wrn_v2_rot5_top20.csv \
        > logs/test_wrn_v2_top20.log 2>&1

# ── Step 5: kNN features ──────────────────────────────────────────────────────
run=feature_rerank_v2_top20
step "knn_features"
for fold in 0 1 2; do
    [[ -f $L/${run}_f${fold}.npz ]] || \
        $py -u feature_rerank.py --fold $fold \
            --prefixes "${prefixes[@]}" \
            --full-probs $L/multibackbone_wrn_v2_rot5_oof.npy \
            --top-k 20 --run-name $run \
            > logs/${run}_f${fold}.log 2>&1
done

# ── Step 6: ranker CV + fold-0 export ────────────────────────────────────────
step "ranker_cv"
$py -u evaluate_meta_rerank.py --run-name $run --class-feature \
    --dense-full $L/multibackbone_densenet169_oof.npy \
    --output $L/meta_rerank_v2_top20_cv.json \
    > logs/meta_rerank_v2.log 2>&1
cat logs/meta_rerank_v2.log

step "ranker_fold0"
$py -u export_ranker_fold0_scores.py --run-name $run --expected-top1 -1 \
    --output $L/meta_v2_top20_fold0_scores.npz \
    > logs/export_ranker_v2_fold0.log 2>&1
cat logs/export_ranker_v2_fold0.log

# ── Step 7: test features + test ranker ──────────────────────────────────────
step "test_features"
[[ -f $L/test_wrn_v2_rot5_top20_features.npz ]] || \
    $py -u predict_test_top20_features.py \
        --prefixes "${prefixes[@]}" \
        --candidates $L/test_wrn_v2_rot5_top20.csv \
        --output $L/test_wrn_v2_rot5_top20_features.npz \
        > logs/test_v2_top20_features.log 2>&1

step "test_ranker"
$py -u predict_meta_top20_test.py \
    --run-name $run \
    --test-features test_wrn_v2_rot5_top20_features.npz \
    --cv-report meta_rerank_v2_top20_cv.json \
    --output $L/test_meta_v2_top20_top10.csv \
    > logs/test_meta_v2.log 2>&1

# ── Step 8: fold-0 end-to-end ─────────────────────────────────────────────────
step "fold0_e2e"
# Use the two best tables from the previous run
for table in semi_old semi_new; do
    case $table in
        semi_old) path=$L/pair_semi_nofold0_old_table_round1.npy ;;
        semi_new) path=$L/pair_semi_nofold0_new_table_round1.npy ;;
    esac
    $py -u evaluate_score_fold0_mlp.py \
        --ranker-scores $L/meta_v2_top20_fold0_scores.npz \
        --table $path \
        --output $L/score_fold0_e2e_v2ranker_${table}.json \
        > logs/score_fold0_e2e_v2ranker_${table}.log 2>&1
    score=$( ~/pokewars-venv/bin/python -c \
        "import json;d=json.load(open('$L/score_fold0_e2e_v2ranker_${table}.json'));print(round(d['soft_test_score'],3))" )
    echo "v2ranker + ${table}: ${score}" | tee -a "$status"
done

# ── Step 9: submission ────────────────────────────────────────────────────────
step "submission"
$py -u predict_score_submission.py \
    --table $L/pair_semi_v2_full_table_round1.npy \
    --candidates $L/test_meta_v2_top20_top10.csv \
    --output $L/submission_v2_ensemble.csv \
    > logs/submission_v2.log 2>&1
step complete
