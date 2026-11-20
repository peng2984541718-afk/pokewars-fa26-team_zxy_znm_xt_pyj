#!/usr/bin/env bash
# Two-stage offensive:
#   Phase 1 (Fast Attack): 4-backbone rot5 TTA + top-40 candidate pool + expanded
#           rot5 kNN gallery + noise-aware DeepSets/LightGBM listwise ranker +
#           bidirectional Unlabeled<->Test cavity with out-of-pool escape.
#   Phase 2 (Hammer): Strictly fold-safe ~195k-image transductive banks (Unl+Test)
#           fine-tuning DenseNet-169, ConvNeXt V2, WRN-40-10, and WRN-28-10 across
#           all 3 folds, followed by full Phase-2 rebuild and submission.
set -euo pipefail

cd "$(dirname "$0")"
py=${PYTHON_BIN:-python3}
if [[ -d cuda_extra ]]; then
    export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
L=outputs/large
mkdir -p logs "$L"
status=$L/fast_and_hammer_status.txt
step() { printf '%s %s\n' "$1" "$(date -u +%FT%TZ)" >> "$status"; }
: > "$status"
trap 'step failed' ERR

step "0_waiting_for_v3_pair_mlp"
if [[ -f $L/constrained_v3_status.txt ]]; then
    while ! grep -q '^complete ' "$L/constrained_v3_status.txt" 2>/dev/null; do
        if grep -q '^failed ' "$L/constrained_v3_status.txt" 2>/dev/null; then
            break
        fi
        sleep 20
    done
fi

# Blend 5-seed v2 and 5-seed v3 PairMLP tables into a 10-seed ensemble table
"$py" -c "
import numpy as np
from pathlib import Path
t2 = np.load('$L/pair_semi_v2_full_table_round1.npy')
p3 = Path('$L/pair_semi_v3_full_table_round1.npy')
t = 0.5 * (t2 + np.load(p3)) if p3.exists() else t2
np.save('$L/pair_semi_v23_blend_table.npy', t)
"

# ==============================================================================
# PHASE 1: FAST ATTACK (No CNN retraining)
# ==============================================================================
w28_all=(wrn28_10 wrn28_10_mild wrn28_10_mild_pseudo wrn28_10_mild_pseudo_synth wrn28_10_mild_pseudo_ctrl)
w28_wts=(1 1 2 2 2)
w28_gal=(wrn28_10_mild wrn28_10_mild_pseudo_synth wrn28_10_mild_pseudo_ctrl)

step "P1_1_export_4backbone_rot5_probs"
[[ -f $L/oof_wrn40_10_rot5.npy ]] || \
    "$py" -u export_probs_rot5.py --model wrn40_10 --prefixes wrn40_10 \
        --split oof --output $L/oof_wrn40_10_rot5.npy > logs/p1_oof_wrn40.log 2>&1
[[ -f $L/oof_densenet169_rot5.npy ]] || \
    "$py" -u export_probs_rot5.py --model densenet169_cifar --prefixes densenet169_cifar \
        --split oof --output $L/oof_densenet169_rot5.npy > logs/p1_oof_dense.log 2>&1
[[ -f $L/oof_convnextv2_rot5.npy ]] || \
    "$py" -u export_probs_rot5.py --model convnextv2_nano --prefixes convnextv2_scratch64 \
        --split oof --output $L/oof_convnextv2_rot5.npy > logs/p1_oof_conv.log 2>&1

# Unlabeled full-ensemble rot5 probabilities (all 4 backbones)
[[ -f $L/unl_wrn28_v2_rot5.npy ]] || \
    "$py" -u export_probs_rot5.py --model wrn28_10 --prefixes "${w28_all[@]}" --weights "${w28_wts[@]}" \
        --split query --query-dir unl --output $L/unl_wrn28_v2_rot5.npy > logs/p1_unl_wrn28.log 2>&1
[[ -f $L/unl_wrn40_rot5.npy ]] || \
    "$py" -u export_probs_rot5.py --model wrn40_10 --prefixes wrn40_10 \
        --split query --query-dir unl --output $L/unl_wrn40_rot5.npy > logs/p1_unl_wrn40.log 2>&1
[[ -f $L/unl_densenet169_rot5.npy ]] || \
    "$py" -u export_probs_rot5.py --model densenet169_cifar --prefixes densenet169_cifar \
        --split query --query-dir unl --output $L/unl_densenet169_rot5.npy > logs/p1_unl_dense.log 2>&1
[[ -f $L/unl_convnextv2_rot5.npy ]] || \
    "$py" -u export_probs_rot5.py --model convnextv2_nano --prefixes convnextv2_scratch64 \
        --split query --query-dir unl --output $L/unl_convnextv2_rot5.npy > logs/p1_unl_conv.log 2>&1

# Unlabeled fold-0-only rot5 probabilities (for leak-free Fold-0 benchmark)
[[ -f $L/unl_f0_wrn28_v2_rot5.npy ]] || \
    "$py" -u export_probs_rot5.py --model wrn28_10 --prefixes "${w28_all[@]}" --weights "${w28_wts[@]}" \
        --split query --folds 0 --query-dir unl --output $L/unl_f0_wrn28_v2_rot5.npy > logs/p1_unl_f0_wrn28.log 2>&1
[[ -f $L/unl_f0_wrn40_rot5.npy ]] || \
    "$py" -u export_probs_rot5.py --model wrn40_10 --prefixes wrn40_10 \
        --split query --folds 0 --query-dir unl --output $L/unl_f0_wrn40_rot5.npy > logs/p1_unl_f0_wrn40.log 2>&1
[[ -f $L/unl_f0_densenet169_rot5.npy ]] || \
    "$py" -u export_probs_rot5.py --model densenet169_cifar --prefixes densenet169_cifar \
        --split query --folds 0 --query-dir unl --output $L/unl_f0_densenet169_rot5.npy > logs/p1_unl_f0_dense.log 2>&1
[[ -f $L/unl_f0_convnextv2_rot5.npy ]] || \
    "$py" -u export_probs_rot5.py --model convnextv2_nano --prefixes convnextv2_scratch64 \
        --split query --folds 0 --query-dir unl --output $L/unl_f0_convnextv2_rot5.npy > logs/p1_unl_f0_conv.log 2>&1

# Test full-ensemble rot5 probabilities (all 4 backbones)
[[ -f $L/test_wrn28_v2_rot5.npy ]] || \
    "$py" -u export_probs_rot5.py --model wrn28_10 --prefixes "${w28_all[@]}" --weights "${w28_wts[@]}" \
        --split query --query-dir . --output $L/test_wrn28_v2_rot5.npy > logs/p1_test_wrn28.log 2>&1
[[ -f $L/test_wrn40_rot5.npy ]] || \
    "$py" -u export_probs_rot5.py --model wrn40_10 --prefixes wrn40_10 \
        --split query --query-dir . --output $L/test_wrn40_rot5.npy > logs/p1_test_wrn40.log 2>&1
[[ -f $L/test_densenet169_rot5.npy ]] || \
    "$py" -u export_probs_rot5.py --model densenet169_cifar --prefixes densenet169_cifar \
        --split query --query-dir . --output $L/test_densenet169_rot5.npy > logs/p1_test_dense.log 2>&1
[[ -f $L/test_convnextv2_rot5.npy ]] || \
    "$py" -u export_probs_rot5.py --model convnextv2_nano --prefixes convnextv2_scratch64 \
        --split query --query-dir . --output $L/test_convnextv2_rot5.npy > logs/p1_test_conv.log 2>&1

step "P1_2_extract_top40_expanded_gallery"
for fold in 0 1 2; do
    [[ -f $L/gallery_p1_top40_f${fold}.npz ]] || \
        "$py" -u extract_gallery_features_v2.py --mode oof-fold --fold $fold --top-k 40 \
            --wrn28-prefixes "${w28_gal[@]}" --wrn40-prefixes wrn40_10 \
            --wrn28-probs $L/multibackbone_wrn_v2_rot5_oof.npy \
            --wrn40-probs $L/oof_wrn40_10_rot5.npy \
            --dense-probs $L/oof_densenet169_rot5.npy \
            --conv-probs $L/oof_convnextv2_rot5.npy \
            --extra-bank $L/pseudo_bank_f${fold}.npz \
            --output $L/gallery_p1_top40_f${fold}.npz > logs/p1_gallery_f${fold}.log 2>&1
done

[[ -f $L/gallery_p1_top40_unl_f0only.npz ]] || \
    "$py" -u extract_gallery_features_v2.py --mode query --folds 0 --query-dir unl --top-k 40 \
        --wrn28-prefixes "${w28_gal[@]}" --wrn40-prefixes wrn40_10 \
        --wrn28-probs $L/unl_f0_wrn28_v2_rot5.npy \
        --wrn40-probs $L/unl_f0_wrn40_rot5.npy \
        --dense-probs $L/unl_f0_densenet169_rot5.npy \
        --conv-probs $L/unl_f0_convnextv2_rot5.npy \
        --extra-bank $L/pseudo_bank_f0.npz \
        --output $L/gallery_p1_top40_unl_f0only.npz > logs/p1_gallery_unl_f0.log 2>&1

[[ -f $L/gallery_p1_top40_unlabeled.npz ]] || \
    "$py" -u extract_gallery_features_v2.py --mode query --query-dir unl --top-k 40 \
        --wrn28-prefixes "${w28_gal[@]}" --wrn40-prefixes wrn40_10 \
        --wrn28-probs $L/unl_wrn28_v2_rot5.npy \
        --wrn40-probs $L/unl_wrn40_rot5.npy \
        --dense-probs $L/unl_densenet169_rot5.npy \
        --conv-probs $L/unl_convnextv2_rot5.npy \
        --output $L/gallery_p1_top40_unlabeled.npz > logs/p1_gallery_unl.log 2>&1

# Build full high-conf unlabeled bank for test kNN gallery!
"$py" -c "
import numpy as np
imgs = np.load('unl/test_32x32.npy')
z = np.load('$L/pair_semi_full_pseudo_labels.npz')
m = z['confidence'] >= 0.95
np.savez_compressed('$L/unl_gallery_bank_95.npz', images=imgs[m], labels=(z['pseudo_label'][m]-1).astype(np.int16))
"
[[ -f $L/gallery_p1_top40_test.npz ]] || \
    "$py" -u extract_gallery_features_v2.py --mode query --query-dir . --top-k 40 \
        --wrn28-prefixes "${w28_gal[@]}" --wrn40-prefixes wrn40_10 \
        --wrn28-probs $L/test_wrn28_v2_rot5.npy \
        --wrn40-probs $L/test_wrn40_rot5.npy \
        --dense-probs $L/test_densenet169_rot5.npy \
        --conv-probs $L/test_convnextv2_rot5.npy \
        --extra-bank $L/unl_gallery_bank_95.npz \
        --output $L/gallery_p1_top40_test.npz > logs/p1_gallery_test.log 2>&1

step "P1_3_listwise_ranker"
"$py" -u train_listwise_ranker.py \
    --oof-prefix gallery_p1_top40 \
    --unlabeled-f0-features $L/gallery_p1_top40_unl_f0only.npz \
    --unlabeled-features $L/gallery_p1_top40_unlabeled.npz \
    --test-features $L/gallery_p1_top40_test.npz \
    --output-prefix $L/phase1_ranker > logs/p1_listwise_ranker.log 2>&1

step "P1_4_joint_inference_and_submission"
"$py" -u joint_constrained_inference.py --mode evaluate-fold0 \
    --fold0-scores $L/phase1_ranker_fold0_scores.npz \
    --unlabeled-priors $L/phase1_ranker_unlabeled_f0only.npz \
    --table $L/pair_semi_nofold0_new_table_round1.npy \
    --output $L/score_fold0_phase1_fastattack.json > logs/p1_fold0_eval.log 2>&1
p1_scale=$("$py" -c "import json;print(json.load(open('$L/score_fold0_phase1_fastattack.json'))['selected']['scale'])")
p1_f0=$("$py" -c "import json;print(round(json.load(open('$L/score_fold0_phase1_fastattack.json'))['selected']['test_score'],3))")
step "P1_DONE_fold0=${p1_f0}_scale=${p1_scale}"

"$py" -u joint_constrained_inference.py --mode predict-submission \
    --unlabeled-priors $L/phase1_ranker_unlabeled.npz \
    --test-npz $L/phase1_ranker_test.npz \
    --table $L/pair_semi_v23_blend_table.npy \
    --scale "$p1_scale" \
    --save-pseudo-battles $L/unlabeled_battles_phase1.npz \
    --save-transductive-bank $L/transductive_bank_full.npz \
    --output $L/submission_phase1_fastattack.csv > logs/p1_submission.log 2>&1
step "P1_SUBMISSION_READY"

# ==============================================================================
# PHASE 2: HAMMER (200k Fold-Safe Transductive Self-Training Across All 4 Archs)
# ==============================================================================
step "P2_1_build_fold_safe_transductive_banks"
"$py" -u build_fold_transductive_banks.py > logs/p2_build_banks.log 2>&1

common_p2=(--training-seed 271828 --batch-size 128
           --translate-pixels 1.0 --rotate-degrees 8 --scale-min 0.95 --scale-max 1.05
           --data-dir . --output-dir $L --external-fraction 0.45
           --epochs 20 --warmup-epochs 1 --min-epochs 20 --patience 20 --min-improvement 0.0001)

for fold in 0 1 2; do
    bank=$L/transductive_bank_f${fold}.npz

    step "P2_2_densenet_f${fold}"
    [[ -f $L/densenet169_trans_f${fold}_best.pt ]] || \
        "$py" -u train_large_image_external.py "${common_p2[@]}" \
            --model densenet169_cifar --fold $fold --external-bank $bank \
            --initial-weights $L/densenet169_cifar_f${fold}_best.pt \
            --learning-rate 2e-4 --min-learning-rate 1e-6 \
            --run-name densenet169_trans_f${fold} > logs/p2_dense_f${fold}.log 2>&1

    step "P2_2_convnext_f${fold}"
    [[ -f $L/convnextv2_trans_f${fold}_best.pt ]] || \
        "$py" -u train_large_image_external.py "${common_p2[@]}" \
            --model convnextv2_nano --fold $fold --external-bank $bank \
            --initial-weights $L/convnextv2_scratch64_f${fold}_stage2_best.pt \
            --learning-rate 1.5e-4 --min-learning-rate 1e-6 \
            --run-name convnextv2_trans_f${fold} > logs/p2_conv_f${fold}.log 2>&1

    step "P2_2_wrn40_f${fold}"
    [[ -f $L/wrn40_10_trans_f${fold}_best.pt ]] || \
        "$py" -u train_large_image_external.py "${common_p2[@]}" \
            --model wrn40_10 --fold $fold --external-bank $bank \
            --initial-weights $L/wrn40_10_f${fold}_best.pt \
            --learning-rate 1.5e-4 --min-learning-rate 1e-6 \
            --run-name wrn40_10_trans_f${fold} > logs/p2_wrn40_f${fold}.log 2>&1

    step "P2_2_wrn28_f${fold}"
    [[ -f $L/wrn28_10_trans_f${fold}_best.pt ]] || \
        "$py" -u train_large_image_external.py "${common_p2[@]}" \
            --model wrn28_10 --fold $fold --external-bank $bank \
            --initial-weights $L/wrn28_10_mild_pseudo_ctrl_f${fold}_stage2_best.pt \
            --learning-rate 1e-4 --min-learning-rate 1e-7 \
            --run-name wrn28_10_trans_f${fold} > logs/p2_wrn28_f${fold}.log 2>&1
done

step "P2_3_export_upgraded_probs"
"$py" -u export_probs_rot5.py --model densenet169_cifar --prefixes densenet169_cifar densenet169_trans \
    --weights 1 3 --split oof --output $L/oof_densenet169_p2_rot5.npy > logs/p2_oof_dense.log 2>&1
"$py" -u export_probs_rot5.py --model convnextv2_nano --prefixes convnextv2_scratch64 convnextv2_trans \
    --weights 1 3 --split oof --output $L/oof_convnextv2_p2_rot5.npy > logs/p2_oof_conv.log 2>&1
"$py" -u export_probs_rot5.py --model wrn40_10 --prefixes wrn40_10 wrn40_10_trans \
    --weights 1 3 --split oof --output $L/oof_wrn40_p2_rot5.npy > logs/p2_oof_wrn40.log 2>&1
"$py" -u export_probs_rot5.py --model wrn28_10 --prefixes wrn28_10_trans \
    --split oof --output $L/oof_wrn28_trans_rot5.npy > logs/p2_oof_wrn28.log 2>&1

w28_p2=(wrn28_10 wrn28_10_mild wrn28_10_mild_pseudo wrn28_10_mild_pseudo_synth wrn28_10_mild_pseudo_ctrl wrn28_10_trans)
w28_p2_wts=(1 1 2 2 2 4)
w28_p2_gal=(wrn28_10_mild_pseudo_synth wrn28_10_mild_pseudo_ctrl wrn28_10_trans)

"$py" -u combine_oof_probs.py \
    --inputs $L/oof_wrn28_10_rot5.npy $L/oof_wrn28_10_mild_rot5.npy $L/oof_wrn28_10_mild_pseudo_rot5.npy \
             $L/oof_wrn28_10_mild_pseudo_synth_rot5.npy $L/oof_wrn28_10_mild_pseudo_ctrl_rot5.npy \
             $L/oof_wrn28_trans_rot5.npy \
    --weights "${w28_p2_wts[@]}" \
    --output $L/oof_wrn28_p2_rot5.npy --compact $L/wrn28_p2_top20_compact.npz > logs/p2_combine_wrn28.log 2>&1

for split_tag in unl test; do
    qdir=.
    [[ $split_tag == unl ]] && qdir=unl
    "$py" -u export_probs_rot5.py --model wrn28_10 --prefixes "${w28_p2[@]}" --weights "${w28_p2_wts[@]}" \
        --split query --query-dir $qdir --output $L/${split_tag}_wrn28_p2_rot5.npy > logs/p2_${split_tag}_wrn28.log 2>&1
    "$py" -u export_probs_rot5.py --model wrn40_10 --prefixes wrn40_10 wrn40_10_trans --weights 1 3 \
        --split query --query-dir $qdir --output $L/${split_tag}_wrn40_p2_rot5.npy > logs/p2_${split_tag}_wrn40.log 2>&1
    "$py" -u export_probs_rot5.py --model densenet169_cifar --prefixes densenet169_cifar densenet169_trans --weights 1 3 \
        --split query --query-dir $qdir --output $L/${split_tag}_dense_p2_rot5.npy > logs/p2_${split_tag}_dense.log 2>&1
    "$py" -u export_probs_rot5.py --model convnextv2_nano --prefixes convnextv2_scratch64 convnextv2_trans --weights 1 3 \
        --split query --query-dir $qdir --output $L/${split_tag}_conv_p2_rot5.npy > logs/p2_${split_tag}_conv.log 2>&1
done

"$py" -u export_probs_rot5.py --model wrn28_10 --prefixes "${w28_p2[@]}" --weights "${w28_p2_wts[@]}" \
    --split query --folds 0 --query-dir unl --output $L/unl_f0_wrn28_p2_rot5.npy > logs/p2_unl_f0_wrn28.log 2>&1
"$py" -u export_probs_rot5.py --model wrn40_10 --prefixes wrn40_10 wrn40_10_trans --weights 1 3 \
    --split query --folds 0 --query-dir unl --output $L/unl_f0_wrn40_p2_rot5.npy > logs/p2_unl_f0_wrn40.log 2>&1
"$py" -u export_probs_rot5.py --model densenet169_cifar --prefixes densenet169_cifar densenet169_trans --weights 1 3 \
    --split query --folds 0 --query-dir unl --output $L/unl_f0_dense_p2_rot5.npy > logs/p2_unl_f0_dense.log 2>&1
"$py" -u export_probs_rot5.py --model convnextv2_nano --prefixes convnextv2_scratch64 convnextv2_trans --weights 1 3 \
    --split query --folds 0 --query-dir unl --output $L/unl_f0_conv_p2_rot5.npy > logs/p2_unl_f0_conv.log 2>&1

step "P2_4_gallery_and_ranker"
for fold in 0 1 2; do
    "$py" -u extract_gallery_features_v2.py --mode oof-fold --fold $fold --top-k 40 \
        --wrn28-prefixes "${w28_p2_gal[@]}" --wrn40-prefixes wrn40_10_trans \
        --wrn28-probs $L/oof_wrn28_p2_rot5.npy \
        --wrn40-probs $L/oof_wrn40_p2_rot5.npy \
        --dense-probs $L/oof_densenet169_p2_rot5.npy \
        --conv-probs $L/oof_convnextv2_p2_rot5.npy \
        --extra-bank $L/transductive_bank_f${fold}.npz \
        --output $L/gallery_p2_top40_f${fold}.npz > logs/p2_gallery_f${fold}.log 2>&1
done

"$py" -u extract_gallery_features_v2.py --mode query --folds 0 --query-dir unl --top-k 40 \
    --wrn28-prefixes "${w28_p2_gal[@]}" --wrn40-prefixes wrn40_10_trans \
    --wrn28-probs $L/unl_f0_wrn28_p2_rot5.npy \
    --wrn40-probs $L/unl_f0_wrn40_p2_rot5.npy \
    --dense-probs $L/unl_f0_dense_p2_rot5.npy \
    --conv-probs $L/unl_f0_conv_p2_rot5.npy \
    --extra-bank $L/transductive_bank_f0.npz \
    --output $L/gallery_p2_top40_unl_f0only.npz > logs/p2_gallery_unl_f0.log 2>&1

"$py" -u extract_gallery_features_v2.py --mode query --query-dir unl --top-k 40 \
    --wrn28-prefixes "${w28_p2_gal[@]}" --wrn40-prefixes wrn40_10_trans \
    --wrn28-probs $L/unl_wrn28_p2_rot5.npy \
    --wrn40-probs $L/unl_wrn40_p2_rot5.npy \
    --dense-probs $L/unl_dense_p2_rot5.npy \
    --conv-probs $L/unl_conv_p2_rot5.npy \
    --output $L/gallery_p2_top40_unlabeled.npz > logs/p2_gallery_unl.log 2>&1

"$py" -u extract_gallery_features_v2.py --mode query --query-dir . --top-k 40 \
    --wrn28-prefixes "${w28_p2_gal[@]}" --wrn40-prefixes wrn40_10_trans \
    --wrn28-probs $L/test_wrn28_p2_rot5.npy \
    --wrn40-probs $L/test_wrn40_p2_rot5.npy \
    --dense-probs $L/test_dense_p2_rot5.npy \
    --conv-probs $L/test_conv_p2_rot5.npy \
    --extra-bank $L/unl_gallery_bank_95.npz \
    --output $L/gallery_p2_top40_test.npz > logs/p2_gallery_test.log 2>&1

"$py" -u train_listwise_ranker.py \
    --oof-prefix gallery_p2_top40 \
    --unlabeled-f0-features $L/gallery_p2_top40_unl_f0only.npz \
    --unlabeled-features $L/gallery_p2_top40_unlabeled.npz \
    --test-features $L/gallery_p2_top40_test.npz \
    --output-prefix $L/phase2_ranker > logs/p2_listwise_ranker.log 2>&1

step "P2_5_final_inference_and_submission"
"$py" -u joint_constrained_inference.py --mode evaluate-fold0 \
    --fold0-scores $L/phase2_ranker_fold0_scores.npz \
    --unlabeled-priors $L/phase2_ranker_unlabeled_f0only.npz \
    --table $L/pair_semi_nofold0_new_table_round1.npy \
    --output $L/score_fold0_phase2_hammer.json > logs/p2_fold0_eval.log 2>&1
p2_scale=$("$py" -c "import json;print(json.load(open('$L/score_fold0_phase2_hammer.json'))['selected']['scale'])")
p2_f0=$("$py" -c "import json;print(round(json.load(open('$L/score_fold0_phase2_hammer.json'))['selected']['test_score'],3))")
step "P2_DONE_fold0=${p2_f0}_scale=${p2_scale}"

"$py" -u joint_constrained_inference.py --mode predict-submission \
    --unlabeled-priors $L/phase2_ranker_unlabeled.npz \
    --test-npz $L/phase2_ranker_test.npz \
    --table $L/pair_semi_v23_blend_table.npy \
    --scale "$p2_scale" \
    --output $L/submission_phase2_hammer.csv > logs/p2_submission.log 2>&1

step "ALL_COMPLETE"
