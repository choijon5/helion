#!/bin/bash
# Wait for the current fp8_gemm tuning to finish, then run one more
# iteration with subprocess benchmark + worker pool so we end up with
# at least 3 successful fp8_gemm runs.
set -e
cd /home/jongsokchoi/helion_2

PYTHON=/home/jongsokchoi/.conda/envs/helion/bin/python
RUNNER=/home/jongsokchoi/helion_2/examples/aot_pretune/pretune_runner.py
OUTPUT_ROOT=/home/jongsokchoi/helion_2/.helion_aot
LOG_DIR="$OUTPUT_ROOT/logs/scheduler"
LOGFILE="$LOG_DIR/gpu4_fp8_one_more.log"

export PYTHONPATH=/home/jongsokchoi/helion_2
export HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1
export HELION_AUTOTUNE_PRECOMPILE_WORKERS=4
export HELION_AUTOTUNE_BENCHMARK_TIMEOUT=60

{
    echo "[GPU 4] waiting for current fp8_gemm to finish..."
    while pgrep -f "tutorial_kernels.py --kernel fp8_gemm" > /dev/null; do
        sleep 60
    done
    echo "[GPU 4] starting one more fp8_gemm iter at $(date '+%H:%M:%S')"
    "$PYTHON" "$RUNNER" \
        --gpus 4 --max-workers 1 \
        --kernels fp8_gemm \
        --output-root "$OUTPUT_ROOT" \
        || echo "[GPU 4] one_more FAILED"
    echo "[GPU 4] done at $(date '+%H:%M:%S')"
} > "$LOGFILE" 2>&1 &
echo "Launched 1-more fp8_gemm. PID=$!"
echo "Log: $LOGFILE"
