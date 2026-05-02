#!/bin/bash
# Final fp8_gemm retry: subprocess benchmark + worker pool to isolate the
# CUDA-runtime errors that crashed iters 1 and 3 of the previous attempt.
set -e
cd /home/jongsokchoi/helion_2

PYTHON=/home/jongsokchoi/.conda/envs/helion/bin/python
RUNNER=/home/jongsokchoi/helion_2/examples/aot_pretune/pretune_runner.py
OUTPUT_ROOT=/home/jongsokchoi/helion_2/.helion_aot
LOG_DIR="$OUTPUT_ROOT/logs/scheduler"
LOGFILE="$LOG_DIR/gpu4_fp8_retry.log"

export PYTHONPATH=/home/jongsokchoi/helion_2
# Run the benchmark phase in a long-lived spawn subprocess; lets the autotuner
# kill a hung kernel and continue with the next config.  Activates the
# long-lived precompile worker pool too.
export HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1
export HELION_AUTOTUNE_PRECOMPILE_WORKERS=4
# Bump per-config timeout from the default 30s — fp8_gemm can take longer.
export HELION_AUTOTUNE_BENCHMARK_TIMEOUT=60

{
    echo "[GPU 4] fp8_gemm retry (subprocess benchmark + pool) at $(date '+%H:%M:%S')"
    "$PYTHON" "$RUNNER" \
        --gpus 4 --max-workers 1 \
        --kernels fp8_gemm \
        --output-root "$OUTPUT_ROOT" \
        || echo "[GPU 4] retry FAILED"
    echo "[GPU 4] fp8_gemm retry done at $(date '+%H:%M:%S')"
} > "$LOGFILE" 2>&1 &
echo "Launched fp8_gemm retry on GPU 4. PID=$!"
echo "Log: $LOGFILE"
