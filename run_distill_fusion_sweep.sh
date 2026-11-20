#!/usr/bin/env bash
# Fold-0 fusion grid: log vs linear, alpha, and student temperature.
# Writes outputs/large/submission_distill_fusion.csv only when the untouched
# fold-0 test half scores above 78.42.
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
py="${PYTHON_BIN:-$HOME/pokewars-venv/bin/python}"
mkdir -p logs outputs/large
"$py" -u sweep_distill_fusion.py "$@" | tee logs/distill_fusion_sweep.log
