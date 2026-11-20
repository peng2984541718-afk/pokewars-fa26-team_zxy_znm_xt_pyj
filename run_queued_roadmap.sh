#!/usr/bin/env bash
# Queued post-80.2 -> 81.46+ roadmap runner.
# Automatically waits for `run_rx_gallery.sh` to complete, then runs Milestones 1..5
# sequentially on the GPU with strict Fold-0 tune/test anti-leakage gates.
set -uo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/libcusparseLt/13:$PWD/cuda_extra/usr/lib/x86_64-linux-gnu/nvshmem/12:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
py="${PYTHON_BIN:-$HOME/pokewars-venv/bin/python}"
L=outputs/large
mkdir -p logs "$L"

status="$L/roadmap_queue_status.txt"
step() { printf '%s %s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$status"; }

update_scoreboard() {
    "$py" - << 'PY'
import json
from pathlib import Path
L = Path("outputs/large")
board = {"baseline_80_2_fold0_test": 78.9263, "milestones": {}, "best_submission_csv": None, "best_fold0_test_score": 78.9263}

f0_raw = L / "score_fold0_rx_ranker.json"
if f0_raw.exists():
    d = json.loads(f0_raw.read_text())
    board["milestones"]["0_rx_ranker_only"] = d["selected"]

m1 = L / "milestone1_rx_fusion.json"
if m1.exists():
    d = json.loads(m1.read_text())
    sc = d["winner"]["selected"]["test_score"]
    board["milestones"]["1_rx_6view_fusion"] = {
        "fold0_tune_score": d["winner"]["selected"]["tune_score"],
        "fold0_test_score": sc,
        "submission_csv": d["submission_csv"],
    }
    board["best_fold0_test_score"] = sc
    board["best_submission_csv"] = d["submission_csv"]

for key, fname in [
    ("2_pair_table_v4", "milestone2_pair_table_v4.json"),
    ("3_hetero_bp_and_noise_cal", "milestone3_calibration.json"),
    ("4_gen2_3fold_distill", "milestone4_gen2_distill.json"),
    ("5_sepyramid_5th_arch", "milestone5_sepyramid.json"),
]:
    p = L / fname
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    passed = bool(d.get("passed", False))
    sub = d.get("submission_csv")
    node = d.get("chosen") or d.get("winner") or {}
    sel = node.get("selected", node)
    ts = sel.get("test_score")
    board["milestones"][key] = {
        "passed": passed,
        "fold0_tune_score": sel.get("tune_score"),
        "fold0_test_score": ts,
        "submission_csv": sub,
    }
    if passed and sub and ts is not None:
        board["best_fold0_test_score"] = ts
        board["best_submission_csv"] = sub

if board["best_submission_csv"] and Path(board["best_submission_csv"]).exists():
    latest = L / "submission_latest_best.csv"
    latest.write_bytes(Path(board["best_submission_csv"]).read_bytes())

(L / "roadmap_scoreboard.json").write_text(json.dumps(board, indent=2) + "\n")
print("SCOREBOARD UPDATED:", json.dumps(board, indent=2), flush=True)
PY
}

step "QUEUE_WAITING_FOR_RX_GALLERY"
while pgrep -f 'bash run_rx_gallery.sh' >/dev/null; do
    sleep 20
done
step "RX_GALLERY_RELEASED"

if [[ ! -s "$L/score_fold0_rx_ranker.json" || ! -s "$L/rx_ranker_test.npz" ]]; then
    step "ABORT_MISSING_RX_RANKER_OUTPUTS"
    exit 1
fi
update_scoreboard

step "MILESTONE_1_START"
if [[ -s "$L/milestone1_rx_fusion.json" && -s "$L/submission_rx_v1.csv" ]]; then
    step "MILESTONE_1_ALREADY_DONE"
    update_scoreboard
elif "$py" -u milestone1_rx_fusion_and_pack.py > logs/milestone1_rx_fusion.log 2>&1; then
    step "MILESTONE_1_DONE"
    update_scoreboard
else
    step "MILESTONE_1_FAILED"
    exit 1
fi

step "MILESTONE_2_START"
if [[ -s "$L/milestone2_pair_table_v4.json" ]]; then
    step "MILESTONE_2_ALREADY_DONE"
    update_scoreboard
elif "$py" -u milestone2_pair_table_v4.py > logs/milestone2_pair_table_v4.log 2>&1; then
    step "MILESTONE_2_DONE"
    update_scoreboard
else
    step "MILESTONE_2_FAILED"
fi

step "MILESTONE_3_START"
if [[ -s "$L/milestone3_calibration.json" ]]; then
    step "MILESTONE_3_ALREADY_DONE"
    update_scoreboard
elif "$py" -u milestone3_inference_calibration.py > logs/milestone3_calibration.log 2>&1; then
    step "MILESTONE_3_DONE"
    update_scoreboard
else
    step "MILESTONE_3_FAILED"
fi

step "MILESTONE_4_START"
if [[ -s "$L/milestone4_gen2_distill.json" ]]; then
    step "MILESTONE_4_ALREADY_DONE"
    update_scoreboard
elif "$py" -u milestone4_gen2_distill_and_fuse.py > logs/milestone4_gen2_distill.log 2>&1; then
    step "MILESTONE_4_DONE"
    update_scoreboard
else
    step "MILESTONE_4_FAILED"
fi

step "MILESTONE_5_START"
if [[ -s "$L/milestone5_sepyramid.json" ]]; then
    step "MILESTONE_5_ALREADY_DONE"
    update_scoreboard
elif "$py" -u milestone5_fifth_backbone_sepyramid.py > logs/milestone5_sepyramid.log 2>&1; then
    step "MILESTONE_5_DONE"
    update_scoreboard
else
    step "MILESTONE_5_FAILED"
fi

step "ALL_QUEUED_MILESTONES_COMPLETE"
