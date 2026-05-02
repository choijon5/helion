#!/bin/bash
# Run fp8_gemm × 3 with subprocess benchmark + worker pool.
# Use after the initial run_fp8_retry.sh has shown the subprocess-benchmark
# flag fixes the hangs.  Each iteration is self-contained: hang-protected
# benchmark + pool-isolated precompile = no orchestrator-level retries needed.

set -e
cd /home/jongsokchoi/helion_2

PYTHON=/home/jongsokchoi/.conda/envs/helion/bin/python
RUNNER=/home/jongsokchoi/helion_2/examples/aot_pretune/pretune_runner.py
OUTPUT_ROOT=/home/jongsokchoi/helion_2/.helion_aot
LOG_DIR="$OUTPUT_ROOT/logs/scheduler"
LOGFILE="$LOG_DIR/gpu4_fp8_3x_subprocess.log"

export PYTHONPATH=/home/jongsokchoi/helion_2
export HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1
export HELION_AUTOTUNE_PRECOMPILE_WORKERS=4
export HELION_AUTOTUNE_BENCHMARK_TIMEOUT=60

# Wait for any current fp8_gemm tuning to finish before starting our queue.
wait_fp8_done() {
    while pgrep -f "tutorial_kernels.py --kernel fp8_gemm" > /dev/null; do
        sleep 60
    done
}

{
    echo "[GPU 4] fp8_gemm × 3 (subprocess+pool) starting at $(date '+%H:%M:%S')"
    echo "[GPU 4] waiting for any in-progress fp8_gemm to finish first..."
    wait_fp8_done
    for i in 1 2 3; do
        echo "[GPU 4] iter $i/3 starting at $(date '+%H:%M:%S')"
        "$PYTHON" "$RUNNER" \
            --gpus 4 --max-workers 1 \
            --kernels fp8_gemm \
            --output-root "$OUTPUT_ROOT" \
            || echo "[GPU 4] iter $i FAILED"
    done
    echo "[GPU 4] fp8_gemm × 3 done at $(date '+%H:%M:%S')"
} > "$LOGFILE" 2>&1 &
echo "Launched fp8_gemm × 3 (subprocess+pool, queued behind current retry). PID=$!"
echo "Log: $LOGFILE"
