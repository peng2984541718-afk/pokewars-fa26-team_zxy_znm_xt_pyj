#!/usr/bin/env bash
# One facing per image: gallery prototype picks the flip, ±5° stays, the other side is dropped.
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:${LD_LIBRARY_PATH:-}"
py="${PYTHON_BIN:-$HOME/pokewars-venv/bin/python}"
L=outputs/large
mkdir -p logs "$L"
status=$L/facing_status.txt
step() { printf '%s %s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$status"; }

step "SELECT"
"$py" -u select_facing.py | tee logs/facing_select.log

export_one() {
    local model="$1" out="$2" split="$3" facing="$4"
    shift 4
    "$py" -u export_probs_rot5.py --model "$model" "$@" \
        --clean-impulses --facing "$facing" --split "$split" --output "$L/$out" \
        > "logs/facing_${out%.npy}.log" 2>&1
}

w28=(--prefixes wrn28_10_mild wrn28_10_mild_pseudo wrn28_10_mild_pseudo_synth wrn28_10_mild_pseudo_ctrl wrn28_10_imp --weights 1 1 2 2 6)
w40=(--prefixes wrn40_10 wrn40_10_imp --weights 1 4)
den=(--prefixes densenet169_cifar densenet169_imp --weights 1 4)
conv=(--prefixes convnextv2_scratch64 convnextv2_imp --weights 1 4)

step "OOF"
export_one densenet169_cifar oof_dense_face.npy oof "$L/facing_labeled.npy" "${den[@]}"
export_one convnextv2_nano oof_conv_face.npy oof "$L/facing_labeled.npy" "${conv[@]}"
export_one wrn40_10 oof_wrn40_face.npy oof "$L/facing_labeled.npy" "${w40[@]}"
export_one wrn28_10 oof_wrn28_face.npy oof "$L/facing_labeled.npy" "${w28[@]}"

blend=$("$py" - << 'PY'
import numpy as np
labels = np.load("labeled_ids.npy")
w = {"wrn28": 0.35, "wrn40": 0.22, "dense": 0.33, "conv": 0.10}
files = {k: np.load(f"outputs/large/oof_{k}_face.npy", mmap_mode="r") for k in w}
n = len(labels)
hit = 0
for s in range(0, n, 8192):
    e = min(n, s + 8192)
    b = None
    for k, wt in w.items():
        sub = np.asarray(files[k][s:e], np.float32)
        b = wt * sub if b is None else b + wt * sub
    hit += int((b.argmax(1) == labels[s:e]).sum())
print(f"{hit / n:.5f}")
PY
)
step "OOF_BLEND_TOP1=${blend}"
python3 - << PY
blend = float("${blend}")
if blend < 0.960:
    raise SystemExit(f"single-facing OOF blend {blend:.5f} is below 0.960; not exporting test")
PY

step "QUERY"
for tag in unl test; do
    qdir=.
    face=$L/facing_test.npy
    [[ $tag == unl ]] && qdir=unl && face=$L/facing_unl.npy
    export_one wrn28_10 ${tag}_wrn28_face.npy query "$face" "${w28[@]}" --query-dir "$qdir"
    export_one wrn40_10 ${tag}_wrn40_face.npy query "$face" "${w40[@]}" --query-dir "$qdir"
    export_one densenet169_cifar ${tag}_dense_face.npy query "$face" "${den[@]}" --query-dir "$qdir"
    export_one convnextv2_nano ${tag}_conv_face.npy query "$face" "${conv[@]}" --query-dir "$qdir"
done
export_one wrn28_10 unl_f0_wrn28_face.npy query "$L/facing_unl_f0.npy" "${w28[@]}" --query-dir unl --folds 0
export_one wrn40_10 unl_f0_wrn40_face.npy query "$L/facing_unl_f0.npy" "${w40[@]}" --query-dir unl --folds 0
export_one densenet169_cifar unl_f0_dense_face.npy query "$L/facing_unl_f0.npy" "${den[@]}" --query-dir unl --folds 0
export_one convnextv2_nano unl_f0_conv_face.npy query "$L/facing_unl_f0.npy" "${conv[@]}" --query-dir unl --folds 0

step "SPLICE_AND_RANKER"
"$py" -u splice_facing_galleries.py | tee logs/facing_splice.log
"$py" -u train_listwise_ranker.py \
    --oof-prefix gallery_face_top40 \
    --unlabeled-f0-features $L/gallery_face_top40_unl_f0only.npz \
    --unlabeled-features $L/gallery_face_top40_unlabeled.npz \
    --test-features $L/gallery_face_top40_test.npz \
    --output-prefix $L/phase2_face_ranker | tee logs/facing_ranker.log

step "FOLD0"
"$py" -u joint_constrained_inference.py --mode evaluate-fold0 \
    --fold0-scores $L/phase2_face_ranker_fold0_scores.npz \
    --unlabeled-priors $L/phase2_face_ranker_unlabeled_f0only.npz \
    --table $L/pair_semi_nofold0_new_table_round1.npy \
    --output $L/score_fold0_facing.json | tee logs/facing_fold0.log
scale=$("$py" -c "import json;print(json.load(open('$L/score_fold0_facing.json'))['selected']['scale'])")
f0=$("$py" -c "import json;print(json.load(open('$L/score_fold0_facing.json'))['selected']['test_score'])")
step "FACING_FOLD0=${f0}_scale=${scale}"
python3 - << PY
f0 = float("${f0}")
if f0 < 78.22:
    raise SystemExit(f"fold0 {f0:.3f} did not beat 78.22; no submission written")
PY

step "SUBMISSION"
"$py" -u joint_constrained_inference.py --mode predict-submission \
    --unlabeled-priors $L/phase2_face_ranker_unlabeled.npz \
    --test-npz $L/phase2_face_ranker_test.npz \
    --table $L/pair_semi_v23_blend_table.npy \
    --scale "$scale" \
    --output $L/submission_facing.csv | tee logs/facing_submission.log
step "FACING_COMPLETE"
