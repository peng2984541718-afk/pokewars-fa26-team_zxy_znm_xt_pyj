#!/usr/bin/env bash
set -euo pipefail

cd /home/pyjama/pokewars
python_bin=/home/pyjama/pokewars-venv/bin/python
export LD_LIBRARY_PATH=/home/pyjama/pokewars/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:/home/pyjama/pokewars/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:/usr/local/cuda/lib64
mkdir -p logs outputs/large
status_file=outputs/large/meta_top20_test_status.txt
printf 'running %s\n' "$(date -u +%FT%TZ)" > "$status_file"
trap 'printf "failed %s\n" "$(date -u +%FT%TZ)" > "$status_file"' ERR

if [[ ! -s outputs/large/test_wrn_rot5_top20.csv ]]; then
    "$python_bin" -u predict_wrn_topk.py --data-dir . --variant ensemble \
        --tta rotate_5deg --limit 160000 --top-k 20 --batch-size 256 \
        --output outputs/large/test_wrn_rot5_top20.csv \
        > logs/meta_top20_wrn.log 2>&1
fi
if [[ ! -s outputs/large/test_convnextv2_probs.npy ]]; then
    "$python_bin" -u predict_backbone_test_probs.py --data-dir . \
        --model convnextv2_nano --checkpoint-prefix convnextv2_scratch64 \
        --output outputs/large/test_convnextv2_probs.npy \
        > logs/meta_top20_convnext.log 2>&1
fi
if [[ ! -s outputs/large/test_densenet169_probs.npy ]]; then
    "$python_bin" -u predict_backbone_test_probs.py --data-dir . \
        --model densenet169_cifar --checkpoint-prefix densenet169_cifar \
        --output outputs/large/test_densenet169_probs.npy \
        > logs/meta_top20_densenet.log 2>&1
fi
if [[ ! -s outputs/large/test_wrn_rot5_top20_features.npz ]]; then
    "$python_bin" -u predict_test_top20_features.py --data-dir . \
        --candidates outputs/large/test_wrn_rot5_top20.csv \
        --output outputs/large/test_wrn_rot5_top20_features.npz \
        > logs/meta_top20_knn.log 2>&1
fi
"$python_bin" -u predict_meta_top20_test.py --data-dir . \
    --output outputs/large/test_meta_three_arch_top20_160000_top10.csv \
    > logs/meta_top20_ranker.log 2>&1
printf 'complete %s\n' "$(date -u +%FT%TZ)" > "$status_file"
