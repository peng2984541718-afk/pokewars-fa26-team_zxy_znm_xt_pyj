#!/usr/bin/env bash
# Phase 2 Impulse-Repaired Hammer:
#   1. Build fold-safe ~195k transductive banks using impulse-repaired 4-arch priors.
#   2. Fine-tune all 4 backbones (DenseNet-169, ConvNeXt V2, WRN-40-10, WRN-28-10)
#      across all 3 folds on exact 25/245 impulse-repaired images + 45% transductive bank.
#   3. Export impulse-repaired rot5 probabilities for all upgraded + legacy models.
#   4. Extract top-40 impulse-repaired rot5 kNN features with expanded galleries,
#      train Listwise DeepSets + LightGBM ranker, run bidirectional Unl<->Test cavity,
#      verify Fold-0 e2e score, and emit submission_phase2_impulse_hammer.csv.
set -euo pipefail

cd "$(dirname "$0")"
py=${PYTHON_BIN:-python3}
if [[ -d cuda_extra ]]; then
    export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
L=outputs/large
mkdir -p logs "$L"
status=$L/phase2_impulse_hammer_status.txt
step() { printf '%s %s\n' "$1" "$(date -u +%FT%TZ)" >> "$status"; }
: > "$status"
trap 'step failed' ERR

step "0_waiting_for_phase1_fastattack"
while ! grep -q '^P1_SUBMISSION_READY ' "$L/fast_and_hammer_status.txt" 2>/dev/null; do
    sleep 10
done
# Stop the legacy queue before it starts non-impulse Phase 2
pkill -f 'run_fast_and_hammer_queue.sh' 2>/dev/null || true
pkill -f 'build_fold_transductive_banks.py' 2>/dev/null || true
sleep 3

step "P2_1_build_fold_safe_transductive_banks_impulse"
"$py" -u build_fold_transductive_banks.py > logs/p2_imp_build_banks.log 2>&1

common_p2=(--training-seed 271828 --batch-size 128 --clean-impulses
           --translate-pixels 1.0 --rotate-degrees 8 --scale-min 0.95 --scale-max 1.05
           --data-dir . --output-dir $L --external-fraction 0.45
           --epochs 20 --warmup-epochs 1 --min-epochs 20 --patience 20 --min-improvement 0.0001)

for fold in 0 1 2; do
    bank=$L/transductive_bank_f${fold}.npz

    step "P2_2_densenet_imp_f${fold}"
    [[ -f $L/densenet169_imp_f${fold}_best.pt ]] || \
        "$py" -u train_large_image_external.py "${common_p2[@]}" \
            --model densenet169_cifar --fold $fold --external-bank $bank \
            --initial-weights $L/densenet169_cifar_f${fold}_best.pt \
            --learning-rate 2e-4 --min-learning-rate 1e-6 \
            --run-name densenet169_imp_f${fold} > logs/p2_imp_dense_f${fold}.log 2>&1

    step "P2_2_convnext_imp_f${fold}"
    [[ -f $L/convnextv2_imp_f${fold}_best.pt ]] || \
        "$py" -u train_large_image_external.py "${common_p2[@]}" \
            --model convnextv2_nano --fold $fold --external-bank $bank \
            --initial-weights $L/convnextv2_scratch64_f${fold}_stage2_best.pt \
            --learning-rate 1.5e-4 --min-learning-rate 1e-6 \
            --run-name convnextv2_imp_f${fold} > logs/p2_imp_conv_f${fold}.log 2>&1

    step "P2_2_wrn40_imp_f${fold}"
    [[ -f $L/wrn40_10_imp_f${fold}_best.pt ]] || \
        "$py" -u train_large_image_external.py "${common_p2[@]}" \
            --model wrn40_10 --fold $fold --external-bank $bank \
            --initial-weights $L/wrn40_10_f${fold}_best.pt \
            --learning-rate 1.5e-4 --min-learning-rate 1e-6 \
            --run-name wrn40_10_imp_f${fold} > logs/p2_imp_wrn40_f${fold}.log 2>&1

    step "P2_2_wrn28_imp_f${fold}"
    [[ -f $L/wrn28_10_imp_f${fold}_best.pt ]] || \
        "$py" -u train_large_image_external.py "${common_p2[@]}" \
            --model wrn28_10 --fold $fold --external-bank $bank \
            --initial-weights $L/wrn28_10_mild_pseudo_ctrl_f${fold}_stage2_best.pt \
            --learning-rate 1e-4 --min-learning-rate 1e-7 \
            --run-name wrn28_10_imp_f${fold} > logs/p2_imp_wrn28_f${fold}.log 2>&1
done

step "P2_3_export_impulse_repaired_probs"
w28_p2=(wrn28_10_mild wrn28_10_mild_pseudo wrn28_10_mild_pseudo_synth wrn28_10_mild_pseudo_ctrl wrn28_10_imp)
w28_p2_wts=(1 1 2 2 6)
w28_p2_gal=(wrn28_10_mild_pseudo_synth wrn28_10_mild_pseudo_ctrl wrn28_10_imp)

"$py" -u export_probs_rot5.py --model densenet169_cifar --prefixes densenet169_cifar densenet169_imp \
    --weights 1 4 --clean-impulses --split oof --output $L/oof_dense_p2imp.npy > logs/p2_imp_oof_dense.log 2>&1
"$py" -u export_probs_rot5.py --model convnextv2_nano --prefixes convnextv2_scratch64 convnextv2_imp \
    --weights 1 4 --clean-impulses --split oof --output $L/oof_conv_p2imp.npy > logs/p2_imp_oof_conv.log 2>&1
"$py" -u export_probs_rot5.py --model wrn40_10 --prefixes wrn40_10 wrn40_10_imp \
    --weights 1 4 --clean-impulses --split oof --output $L/oof_wrn40_p2imp.npy > logs/p2_imp_oof_wrn40.log 2>&1
"$py" -u export_probs_rot5.py --model wrn28_10 --prefixes "${w28_p2[@]}" --weights "${w28_p2_wts[@]}" \
    --clean-impulses --split oof --output $L/oof_wrn28_p2imp.npy > logs/p2_imp_oof_wrn28.log 2>&1

for split_tag in unl test; do
    qdir=.
    [[ $split_tag == unl ]] && qdir=unl
    "$py" -u export_probs_rot5.py --model wrn28_10 --prefixes "${w28_p2[@]}" --weights "${w28_p2_wts[@]}" \
        --clean-impulses --split query --query-dir $qdir --output $L/${split_tag}_wrn28_p2imp.npy > logs/p2_imp_${split_tag}_wrn28.log 2>&1
    "$py" -u export_probs_rot5.py --model wrn40_10 --prefixes wrn40_10 wrn40_10_imp --weights 1 4 \
        --clean-impulses --split query --query-dir $qdir --output $L/${split_tag}_wrn40_p2imp.npy > logs/p2_imp_${split_tag}_wrn40.log 2>&1
    "$py" -u export_probs_rot5.py --model densenet169_cifar --prefixes densenet169_cifar densenet169_imp --weights 1 4 \
        --clean-impulses --split query --query-dir $qdir --output $L/${split_tag}_dense_p2imp.npy > logs/p2_imp_${split_tag}_dense.log 2>&1
    "$py" -u export_probs_rot5.py --model convnextv2_nano --prefixes convnextv2_scratch64 convnextv2_imp --weights 1 4 \
        --clean-impulses --split query --query-dir $qdir --output $L/${split_tag}_conv_p2imp.npy > logs/p2_imp_${split_tag}_conv.log 2>&1
done

"$py" -u export_probs_rot5.py --model wrn28_10 --prefixes "${w28_p2[@]}" --weights "${w28_p2_wts[@]}" \
    --clean-impulses --split query --folds 0 --query-dir unl --output $L/unl_f0_wrn28_p2imp.npy > logs/p2_imp_unl_f0_wrn28.log 2>&1
"$py" -u export_probs_rot5.py --model wrn40_10 --prefixes wrn40_10 wrn40_10_imp --weights 1 4 \
    --clean-impulses --split query --folds 0 --query-dir unl --output $L/unl_f0_wrn40_p2imp.npy > logs/p2_imp_unl_f0_wrn40.log 2>&1
"$py" -u export_probs_rot5.py --model densenet169_cifar --prefixes densenet169_cifar densenet169_imp --weights 1 4 \
    --clean-impulses --split query --folds 0 --query-dir unl --output $L/unl_f0_dense_p2imp.npy > logs/p2_imp_unl_f0_dense.log 2>&1
"$py" -u export_probs_rot5.py --model convnextv2_nano --prefixes convnextv2_scratch64 convnextv2_imp --weights 1 4 \
    --clean-impulses --split query --folds 0 --query-dir unl --output $L/unl_f0_conv_p2imp.npy > logs/p2_imp_unl_f0_conv.log 2>&1

step "P2_4_gallery_and_listwise_ranker"
for fold in 0 1 2; do
    "$py" -u extract_gallery_features_v2.py --mode oof-fold --fold $fold --top-k 40 --clean-impulses \
        --wrn28-prefixes "${w28_p2_gal[@]}" --wrn40-prefixes wrn40_10_imp \
        --wrn28-probs $L/oof_wrn28_p2imp.npy \
        --wrn40-probs $L/oof_wrn40_p2imp.npy \
        --dense-probs $L/oof_dense_p2imp.npy \
        --conv-probs $L/oof_conv_p2imp.npy \
        --extra-bank $L/transductive_bank_f${fold}.npz \
        --output $L/gallery_p2imp_top40_f${fold}.npz > logs/p2_imp_gallery_f${fold}.log 2>&1
done

"$py" -u extract_gallery_features_v2.py --mode query --folds 0 --query-dir unl --top-k 40 --clean-impulses \
    --wrn28-prefixes "${w28_p2_gal[@]}" --wrn40-prefixes wrn40_10_imp \
    --wrn28-probs $L/unl_f0_wrn28_p2imp.npy \
    --wrn40-probs $L/unl_f0_wrn40_p2imp.npy \
    --dense-probs $L/unl_f0_dense_p2imp.npy \
    --conv-probs $L/unl_f0_conv_p2imp.npy \
    --extra-bank $L/transductive_bank_f0.npz \
    --output $L/gallery_p2imp_top40_unl_f0only.npz > logs/p2_imp_gallery_unl_f0.log 2>&1

"$py" -u extract_gallery_features_v2.py --mode query --query-dir unl --top-k 40 --clean-impulses \
    --wrn28-prefixes "${w28_p2_gal[@]}" --wrn40-prefixes wrn40_10_imp \
    --wrn28-probs $L/unl_wrn28_p2imp.npy \
    --wrn40-probs $L/unl_wrn40_p2imp.npy \
    --dense-probs $L/unl_dense_p2imp.npy \
    --conv-probs $L/unl_conv_p2imp.npy \
    --output $L/gallery_p2imp_top40_unlabeled.npz > logs/p2_imp_gallery_unl.log 2>&1

"$py" -u extract_gallery_features_v2.py --mode query --query-dir . --top-k 40 --clean-impulses \
    --wrn28-prefixes "${w28_p2_gal[@]}" --wrn40-prefixes wrn40_10_imp \
    --wrn28-probs $L/test_wrn28_p2imp.npy \
    --wrn40-probs $L/test_wrn40_p2imp.npy \
    --dense-probs $L/test_dense_p2imp.npy \
    --conv-probs $L/test_conv_p2imp.npy \
    --extra-bank $L/unl_gallery_bank_95.npz \
    --output $L/gallery_p2imp_top40_test.npz > logs/p2_imp_gallery_test.log 2>&1

"$py" -u train_listwise_ranker.py \
    --oof-prefix gallery_p2imp_top40 \
    --unlabeled-f0-features $L/gallery_p2imp_top40_unl_f0only.npz \
    --unlabeled-features $L/gallery_p2imp_top40_unlabeled.npz \
    --test-features $L/gallery_p2imp_top40_test.npz \
    --output-prefix $L/phase2_imp_ranker > logs/p2_imp_listwise_ranker.log 2>&1

step "P2_5_final_inference_and_submission"
"$py" -u joint_constrained_inference.py --mode evaluate-fold0 \
    --fold0-scores $L/phase2_imp_ranker_fold0_scores.npz \
    --unlabeled-priors $L/phase2_imp_ranker_unlabeled_f0only.npz \
    --table $L/pair_semi_nofold0_new_table_round1.npy \
    --output $L/score_fold0_phase2_impulse_hammer.json > logs/p2_imp_fold0_eval.log 2>&1
p2_scale=$("$py" -c "import json;print(json.load(open('$L/score_fold0_phase2_impulse_hammer.json'))['selected']['scale'])")
p2_f0=$("$py" -c "import json;print(round(json.load(open('$L/score_fold0_phase2_impulse_hammer.json'))['selected']['test_score'],3))")
step "P2_IMP_DONE_fold0=${p2_f0}_scale=${p2_scale}"

"$py" -u joint_constrained_inference.py --mode predict-submission \
    --unlabeled-priors $L/phase2_imp_ranker_unlabeled.npz \
    --test-npz $L/phase2_imp_ranker_test.npz \
    --table $L/pair_semi_v23_blend_table.npy \
    --scale "$p2_scale" \
    --save-transductive-bank $L/transductive_bank_p2imp_full.npz \
    --output $L/submission_phase2_impulse_hammer.csv > logs/p2_imp_submission.log 2>&1

step "ALL_COMPLETE"
