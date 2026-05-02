#!/bin/bash
# Re-run fp8_gemm × 3 on GPU 4.
# Use the worker pool (HELION_AUTOTUNE_PRECOMPILE_WORKERS=4) to isolate
# the recurring Triton compile failures that hung the previous run.

set -e
cd /home/jongsokchoi/helion_2

PYTHON=/home/jongsokchoi/.conda/envs/helion/bin/python
RUNNER=/home/jongsokchoi/helion_2/examples/aot_pretune/pretune_runner.py
OUTPUT_ROOT=/home/jongsokchoi/helion_2/.helion_aot
LOG_DIR="$OUTPUT_ROOT/logs/scheduler"
mkdir -p "$LOG_DIR"

export PYTHONPATH=/home/jongsokchoi/helion_2
# Use the long-lived worker pool from choijon5/stack/31 explicitly to keep
# Triton compile failures from blocking the main process.
export HELION_AUTOTUNE_PRECOMPILE_WORKERS=4

LOGFILE="$LOG_DIR/gpu4_fp8.log"
{
    echo "[GPU 4] fp8_gemm × 3 starting at $(date '+%H:%M:%S')"
    for i in 1 2 3; do
        echo "[GPU 4] fp8_gemm iter $i/3 at $(date '+%H:%M:%S')"
        "$PYTHON" "$RUNNER" \
            --gpus 4 --max-workers 1 \
            --kernels fp8_gemm \
            --output-root "$OUTPUT_ROOT" \
            || echo "[GPU 4] iter $i FAILED"
    done
    echo "[GPU 4] fp8_gemm × 3 done at $(date '+%H:%M:%S')"
} > "$LOGFILE" 2>&1 &
echo "Launched fp8_gemm × 3 on GPU 4 with worker pool. PID=$!"
echo "Log: $LOGFILE"
