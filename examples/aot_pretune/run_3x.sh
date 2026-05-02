#!/bin/bash
# Run all kernels 3 times, then merge to pick best per shape.
# Each run uses the orchestrator which spawns subprocesses per kernel on
# different GPUs.  Each AOT run creates a new timestamped subdir under
# .helion_aot/job_<kernel>/, so runs accumulate naturally.
#
# Usage:
#   ./run_3x.sh <gpus_csv> <max_workers> [extra args]
# e.g.:
#   ./run_3x.sh 1,2,3,4,5,7 6
set -e
cd "$(dirname "$0")"

GPUS="${1:-7,5,4,3}"
MAX_WORKERS="${2:-4}"
shift 2 || true

PYTHON=/home/jongsokchoi/.conda/envs/helion/bin/python
RUNNER=/home/jongsokchoi/helion_2/examples/aot_pretune/pretune_runner.py

export PYTHONPATH=/home/jongsokchoi/helion_2
# Use the default fork precompile path. If a kernel hits unrecoverable
# CUDA errors, escalate that one with HELION_AUTOTUNE_PRECOMPILE_WORKERS=N
# (worker pool from choijon5/stack/31) rather than switching to spawn.

for run in 1 2 3; do
  echo "============================================================"
  echo "Pretune run $run/3 starting at $(date '+%H:%M:%S')"
  echo "============================================================"
  "$PYTHON" "$RUNNER" --gpus "$GPUS" --max-workers "$MAX_WORKERS" "$@"
done

echo
echo "All 3 runs complete; merging best configs..."
"$PYTHON" /home/jongsokchoi/helion_2/examples/aot_pretune/pick_best_configs.py
