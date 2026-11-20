#!/usr/bin/env bash
set -euo pipefail

cd /home/pyjama/pokewars
python_bin=/home/pyjama/pokewars-venv/bin/python
export LD_LIBRARY_PATH=/home/pyjama/pokewars/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:/home/pyjama/pokewars/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:/usr/local/cuda/lib64
status_file=outputs/large/external_pilot_status.txt
mkdir -p logs outputs/large
printf 'waiting %s\n' "$(date -u +%FT%TZ)" > "$status_file"
trap 'printf "failed %s\n" "$(date -u +%FT%TZ)" > "$status_file"' ERR

# The existing WRN40 job owns the 5090 until its three folds and OOF finish.
while true; do
    status=$(head -n 1 outputs/large/wrn40_background_status.txt)
    case "$status" in
        complete*) break ;;
        failed*) echo "WRN40 queue failed: $status" >&2; exit 1 ;;
    esac
    sleep 60
done

printf 'running %s\n' "$(date -u +%FT%TZ)" > "$status_file"
common=(--model wrn28_10 --fold 0 --epochs 8 --warmup-epochs 1
        --patience 3 --min-epochs 4 --learning-rate 4e-5
        --min-learning-rate 1e-6 --initial-weights outputs/large/wrn28_10_f0_stage2_best.pt
        --data-dir .)
"$python_bin" -u train_large_image_external.py "${common[@]}" \
    --run-name external_control_f0 > logs/external_control_f0.log 2>&1
"$python_bin" -u train_large_image_external.py "${common[@]}" \
    --external-bank data/external/pokeapi/train_bank_fold0.npz \
    --external-fraction 0.1 --run-name external_sprites_f0 \
    > logs/external_sprites_f0.log 2>&1

"$python_bin" - <<'PY'
import json
from pathlib import Path
root=Path('outputs/large')
result={name:json.loads((root/f'{name}_best.json').read_text())
        for name in ('external_control_f0','external_sprites_f0')}
(root/'external_pilot_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
PY
printf 'complete %s\n' "$(date -u +%FT%TZ)" > "$status_file"
