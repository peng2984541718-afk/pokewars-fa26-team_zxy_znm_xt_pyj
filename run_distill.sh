#!/usr/bin/env bash
# Soft distillation of the 79.9 ranker. Waits until the facing job releases the GPU.
# Fold 0 is trained without its own labels. Full-data students and a submission are
# produced only if that fold actually improves.
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:${LD_LIBRARY_PATH:-}"
py="${PYTHON_BIN:-$HOME/pokewars-venv/bin/python}"
L=outputs/large
mkdir -p logs "$L"
status=$L/distill_status.txt
step() { printf '%s %s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$status"; }

step "WAIT_FOR_FACING"
while pgrep -f 'bash ./run_facing.sh' >/dev/null; do
    sleep 30
done

step "PACK"
"$py" -u build_distill_pack.py | tee logs/distill_pack.log

common=(--training-seed 271828 --batch-size 128 --clean-impulses
        --translate-pixels 1.0 --rotate-degrees 8 --scale-min 0.95 --scale-max 1.05
        --data-dir . --output-dir $L
        --epochs 12 --warmup-epochs 1 --min-epochs 8 --patience 6 --min-improvement 0.0002
        --distill-labeled $L/distill_labeled_oof.npz --distill-alpha 0.5
        --distill-fraction 0.40)

step "PILOT_FOLD0"
"$py" -u train_large_image_external.py "${common[@]}" \
    --model densenet169_cifar --fold 0 \
    --initial-weights $L/densenet169_imp_f0_best.pt \
    --learning-rate 1e-4 --min-learning-rate 1e-6 \
    --distill-external-images unl/test_32x32.npy \
    --distill-external-pack $L/phase2_imp_ranker_unlabeled_f0only.npz \
    --save-valid-probs $L/distill_pilot_f0_probs.npy \
    --run-name densenet169_dist_f0 > logs/distill_pilot_f0.log 2>&1

pilot=$("$py" -c "import json;d=json.load(open('$L/densenet169_dist_f0_best.json'));print(d['epoch'],d['top1'])")
step "PILOT_BEST=${pilot}"
"$py" - << PY
epoch, top1 = "${pilot}".split()
if int(epoch) < 1 or float(top1) < 0.9561:
    raise SystemExit(f"pilot did not beat 0.9551 no-TTA baseline ({epoch} {top1})")
PY

step "PILOT_FOLD0_SCORE"
"$py" - << 'PY'
import numpy as np
from image_experiment import SEED, stratified_folds
labels = np.load("labeled_ids.npy").astype(np.int64)
valid = np.flatnonzero(stratified_folds(labels, 3, SEED) == 0)
with np.load("outputs/large/phase2_imp_ranker_fold0_scores.npz") as z:
    if not np.array_equal(z["valid_indices"].astype(np.int64), valid):
        raise SystemExit("fold-0 ranker rows are not the validation split")
probs = np.load("outputs/large/distill_pilot_f0_probs.npy", mmap_mode="r")
if len(probs) != len(valid):
    raise SystemExit(f"pilot probs {len(probs)} != validation {len(valid)}")
print(f"fold0 alignment ok {len(valid)}")
PY
"$py" -u mix_distill_posteriors.py \
    --ranker $L/phase2_imp_ranker_fold0_scores.npz \
    --probs $L/distill_pilot_f0_probs.npy --prob-weights 1 \
    --weight 0.25 \
    --output $L/distill_pilot_fold0_scores.npz
"$py" -u joint_constrained_inference.py --mode evaluate-fold0 \
    --fold0-scores $L/distill_pilot_fold0_scores.npz \
    --unlabeled-priors $L/phase2_imp_ranker_unlabeled_f0only.npz \
    --table $L/pair_semi_nofold0_new_table_round1.npy \
    --output $L/score_fold0_distill_pilot.json | tee logs/distill_pilot_fold0.log
f0=$("$py" -c "import json;print(json.load(open('$L/score_fold0_distill_pilot.json'))['selected']['test_score'])")
step "PILOT_FOLD0_SCORE=${f0}"
"$py" - << PY
if float("${f0}") < 78.22:
    raise SystemExit(f"pilot fold0 {float('${f0}'):.3f} did not beat 78.22")
PY

step "FULL_DENSE"
"$py" -u train_large_image_external.py "${common[@]}" --train-all \
    --model densenet169_cifar --fold 0 \
    --initial-weights $L/densenet169_imp_f0_best.pt \
    --learning-rate 1e-4 --min-learning-rate 1e-6 \
    --distill-external-images $L/distill_external_images.npy \
    --distill-external-pack $L/distill_external_full.npz \
    --run-name densenet169_distfull > logs/distill_full_dense.log 2>&1

step "FULL_WRN28"
"$py" -u train_large_image_external.py "${common[@]}" --train-all \
    --model wrn28_10 --fold 0 \
    --initial-weights $L/wrn28_10_imp_f0_best.pt \
    --learning-rate 8e-5 --min-learning-rate 1e-7 \
    --distill-external-images $L/distill_external_images.npy \
    --distill-external-pack $L/distill_external_full.npz \
    --run-name wrn28_10_distfull > logs/distill_full_wrn.log 2>&1

step "EXPORT"
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
scale=$("$py" -c "import json;print(json.load(open('$L/score_fold0_phase2_impulse_hammer.json'))['selected']['scale'])")
"$py" -u joint_constrained_inference.py --mode predict-submission \
    --unlabeled-priors $L/distill_mix_unlabeled.npz \
    --test-npz $L/distill_mix_test.npz \
    --table $L/pair_semi_v23_blend_table.npy \
    --scale "$scale" \
    --output $L/submission_distill.csv | tee logs/distill_submission.log
step "DISTILL_COMPLETE"
