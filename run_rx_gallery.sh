#!/usr/bin/env bash
# ResNeXt replaces ConvNeXt in the gallery. Test queries also see the
# high-confidence test images, with each query removed from its own kNN.
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
py="${PYTHON_BIN:-$HOME/pokewars-venv/bin/python}"
L=outputs/large
mkdir -p logs
status=$L/rx_gallery_status.txt
step() { printf '%s %s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$status"; }

step "EXPORT_RESNEXT_PROBS"
if [[ ! -s $L/oof_resnext29_dist.npy ]]; then
    "$py" -u export_probs_rot5.py --model resnext29_8x64d --prefixes resnext29_dist \
        --folds 0 1 2 --clean-impulses --split oof --batch-size 128 \
        --output $L/oof_resnext29_dist.npy > logs/rx_oof_probs.log 2>&1
fi
if [[ ! -s $L/unl_f0_resnext29_dist.npy ]]; then
    "$py" -u export_probs_rot5.py --model resnext29_8x64d --prefixes resnext29_dist \
        --folds 0 --clean-impulses --split query --query-dir unl --batch-size 128 \
        --output $L/unl_f0_resnext29_dist.npy > logs/rx_unl_f0_probs.log 2>&1
fi
if [[ ! -s $L/unl_resnext29_dist.npy ]]; then
    "$py" -u export_probs_rot5.py --model resnext29_8x64d --prefixes resnext29_dist \
        --folds 0 1 2 --clean-impulses --split query --query-dir unl --batch-size 128 \
        --output $L/unl_resnext29_dist.npy > logs/rx_unl_probs.log 2>&1
fi
if [[ ! -s $L/test_resnext29_dist.npy ]]; then
    "$py" -u export_probs_rot5.py --model resnext29_8x64d --prefixes resnext29_dist \
        --folds 0 1 2 --clean-impulses --split query --query-dir . --batch-size 128 \
        --output $L/test_resnext29_dist.npy > logs/rx_test_probs.log 2>&1
fi
step "EXPORT_DONE"

step "BUILD_TEST_BANK"
if [[ ! -s $L/test_backfill_bank.npz ]]; then
"$py" - << 'PY'
from pathlib import Path
import numpy as np
L = Path("outputs/large")
full = np.load(L / "transductive_bank_p2imp_full.npz")
unl = np.load(L / "unl_gallery_bank_95.npz")
n = len(full["images"])
# The full bank was written as unlabeled images followed by test images.
test_n = 153544
if n <= test_n:
    raise SystemExit(f"full bank has {n} rows, expected more than {test_n}")
test_imgs = full["images"][-test_n:]
test_labs = full["labels"][-test_n:].astype(np.int64)
images = np.concatenate([unl["images"], test_imgs], axis=0)
labels = np.concatenate([unl["labels"].astype(np.int64), test_labs], axis=0)
out = L / "test_backfill_bank.npz"
np.savez_compressed(out, images=images, labels=labels.astype(np.int16))
print(f"wrote {out} unlabeled={len(unl['images'])} test={test_n} total={len(images)}", flush=True)
PY
fi
step "BANK_DONE"

common=(--clean-impulses --batch-size 256
        --wrn28-prefixes wrn28_10_mild_pseudo_synth wrn28_10_mild_pseudo_ctrl wrn28_10_imp
        --wrn40-prefixes wrn40_10_imp
        --resnext-prefixes resnext29_dist
        --blend-weights 0.28 0.16 0.28 0.28)

step "GALLERY_OOF"
for fold in 0 1 2; do
    if [[ ! -s $L/gallery_rx_top40_f${fold}.npz ]]; then
        "$py" -u extract_gallery_features_v2.py --mode oof-fold --fold $fold --top-k 40 \
            "${common[@]}" \
            --wrn28-probs $L/oof_wrn28_p2imp.npy \
            --wrn40-probs $L/oof_wrn40_p2imp.npy \
            --dense-probs $L/oof_dense_p2imp.npy \
            --conv-probs $L/oof_resnext29_dist.npy \
            --extra-bank $L/transductive_bank_f${fold}.npz \
            --output $L/gallery_rx_top40_f${fold}.npz > logs/rx_gallery_f${fold}.log 2>&1
    fi
    step "GALLERY_F${fold}_DONE"
done

step "GALLERY_UNL_F0"
if [[ ! -s $L/gallery_rx_top40_unl_f0only.npz ]]; then
    "$py" -u extract_gallery_features_v2.py --mode query --folds 0 --query-dir unl --top-k 40 \
        "${common[@]}" \
        --wrn28-probs $L/unl_f0_wrn28_p2imp.npy \
        --wrn40-probs $L/unl_f0_wrn40_p2imp.npy \
        --dense-probs $L/unl_f0_dense_p2imp.npy \
        --conv-probs $L/unl_f0_resnext29_dist.npy \
        --extra-bank $L/transductive_bank_f0.npz \
        --output $L/gallery_rx_top40_unl_f0only.npz > logs/rx_gallery_unl_f0.log 2>&1
fi
step "GALLERY_UNL_F0_DONE"

step "GALLERY_UNL"
if [[ ! -s $L/gallery_rx_top40_unlabeled.npz ]]; then
    "$py" -u extract_gallery_features_v2.py --mode query --folds 0 1 2 --query-dir unl --top-k 40 \
        "${common[@]}" \
        --wrn28-probs $L/unl_wrn28_p2imp.npy \
        --wrn40-probs $L/unl_wrn40_p2imp.npy \
        --dense-probs $L/unl_dense_p2imp.npy \
        --conv-probs $L/unl_resnext29_dist.npy \
        --output $L/gallery_rx_top40_unlabeled.npz > logs/rx_gallery_unl.log 2>&1
fi
step "GALLERY_UNL_DONE"

step "GALLERY_TEST"
if [[ ! -s $L/gallery_rx_top40_test.npz ]]; then
    "$py" -u extract_gallery_features_v2.py --mode query --folds 0 1 2 --query-dir . --top-k 40 --loo \
        "${common[@]}" \
        --wrn28-probs $L/test_wrn28_p2imp.npy \
        --wrn40-probs $L/test_wrn40_p2imp.npy \
        --dense-probs $L/test_dense_p2imp.npy \
        --conv-probs $L/test_resnext29_dist.npy \
        --extra-bank $L/test_backfill_bank.npz \
        --output $L/gallery_rx_top40_test.npz > logs/rx_gallery_test.log 2>&1
fi
step "GALLERY_TEST_DONE"

step "RANKER"
if [[ ! -s $L/rx_ranker_test.npz ]]; then
    "$py" -u train_listwise_ranker.py \
        --oof-prefix gallery_rx_top40 \
        --unlabeled-f0-features $L/gallery_rx_top40_unl_f0only.npz \
        --unlabeled-features $L/gallery_rx_top40_unlabeled.npz \
        --test-features $L/gallery_rx_top40_test.npz \
        --output-prefix $L/rx_ranker > logs/rx_ranker.log 2>&1
fi
step "RANKER_DONE"

step "FOLD0"
if [[ ! -s $L/score_fold0_rx_ranker.json ]]; then
    "$py" -u joint_constrained_inference.py --mode evaluate-fold0 \
        --fold0-scores $L/rx_ranker_fold0_scores.npz \
        --unlabeled-priors $L/rx_ranker_unlabeled_f0only.npz \
        --table $L/pair_semi_nofold0_new_table_round1.npy \
        --output $L/score_fold0_rx_ranker.json > logs/rx_fold0_eval.log 2>&1
fi
step "FOLD0_DONE"
