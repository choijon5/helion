#!/bin/bash
# Sequential per-GPU scheduler.  Each GPU runs a queue of kernels, one at a time.
# Same kernel always uses the same GPU (sticky), so Triton compile cache stays warm.
#
# CURRENT STATE (count of run 1 already completed/in-progress):
#   - softmax    : 1 in progress on GPU 3
#   - layer_norm : 1 in progress on GPU 4
#   - vector_add : 1 in progress on GPU 7
#   - attention  : 1 in progress on GPU 6 (handled separately, not in this script)
#   - matmul       : 0 (need 3 fresh runs)
#   - grouped_gemm : 0 (need 3 fresh runs)
#   - fp8_gemm     : 0 (need 3 fresh runs)
#
# This script queues:
#   GPU 3: softmax × 2 more       (2 to bring total to 3)
#   GPU 4: layer_norm × 2 more, fp8_gemm × 3
#   GPU 7: matmul × 3, grouped_gemm × 3, vector_add × 2 more
#         (vector_add's run 1 is finishing; if it's done by the time we start,
#          this still works — we just queue 2 more runs of vector_add at the end)
# attention (GPU 6) handled by a separate, simpler script.

set -e
cd /home/jongsokchoi/helion_2  # IMPORTANT: keep cwd at repo root so .helion_aot/ resolves correctly

PYTHON=/home/jongsokchoi/.conda/envs/helion/bin/python
RUNNER=/home/jongsokchoi/helion_2/examples/aot_pretune/pretune_runner.py
OUTPUT_ROOT=/home/jongsokchoi/helion_2/.helion_aot
LOG_DIR="$OUTPUT_ROOT/logs/scheduler"
mkdir -p "$LOG_DIR"
export PYTHONPATH=/home/jongsokchoi/helion_2

run_one() {
    local gpu=$1
    local kernel=$2
    "$PYTHON" "$RUNNER" --gpus "$gpu" --max-workers 1 --kernels "$kernel" \
        --output-root "$OUTPUT_ROOT"
}

# Wait until the specified GPU has no compute apps in nvidia-smi.
wait_gpu_idle() {
    local gpu=$1
    while true; do
        local uuid
        uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | \
            awk -F',' -v g="$gpu" '{gsub(/ /,""); if($1==g) print $2}')
        local apps
        apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | \
            grep -F "$uuid" | wc -l)
        if [ "$apps" -eq 0 ]; then
            return 0
        fi
        sleep 30
    done
}

run_queue() {
    local gpu=$1
    shift
    local jobs=("$@")  # entries like "kernel:N" meaning run kernel N times
    local logfile="$LOG_DIR/gpu${gpu}.log"
    {
        echo "[GPU $gpu] queue: ${jobs[*]}"
        # Wait once at the start for the GPU to be idle (in case run-1 of any
        # kernel is still finishing up on this GPU).
        echo "[GPU $gpu] waiting for GPU $gpu to be idle..."
        wait_gpu_idle "$gpu"
        echo "[GPU $gpu] GPU $gpu is idle, starting queue at $(date '+%H:%M:%S')"
        for job in "${jobs[@]}"; do
            local kernel="${job%:*}"
            local n="${job#*:}"
            for i in $(seq 1 "$n"); do
                echo "[GPU $gpu] start $kernel iter $i/$n at $(date '+%H:%M:%S')"
                run_one "$gpu" "$kernel" || echo "[GPU $gpu] $kernel iter $i FAILED"
            done
        done
        echo "[GPU $gpu] queue done at $(date '+%H:%M:%S')"
    } > "$logfile" 2>&1
}

# Start each GPU's queue in parallel
run_queue 3 softmax:2 > /dev/null 2>&1 &
PID_3=$!
run_queue 4 layer_norm:2 fp8_gemm:3 > /dev/null 2>&1 &
PID_4=$!
run_queue 6 attention:2 > /dev/null 2>&1 &
PID_6=$!
run_queue 7 matmul:2 grouped_gemm:3 > /dev/null 2>&1 &
PID_7=$!

echo "Launched 4 GPU pipelines:"
echo "  GPU 3: softmax × 2 more  (PID $PID_3)"
echo "  GPU 4: layer_norm × 2 more, fp8_gemm × 3  (PID $PID_4)"
echo "  GPU 6: attention × 2 more  (PID $PID_6)"
echo "  GPU 7: matmul × 2 more, grouped_gemm × 3  (PID $PID_7)"
echo "  (vector_add already has 5 runs — no more needed)"
echo "  Per-GPU logs in $LOG_DIR"
echo ""

wait $PID_3 $PID_4 $PID_6 $PID_7
echo "All pipelines done at $(date '+%H:%M:%S')."
echo
echo "=== Naive merge (lowest CSV timing per shape) ==="
"$PYTHON" /home/jongsokchoi/helion_2/examples/aot_pretune/pick_best_configs.py
echo
echo "=== Robust pick (re-benchmark top-3, take median of 5) ==="
# Use one free GPU for rebenchmarks (defaults to GPU 7 — safe since the
# scheduler is finished by now and 7 is part of our pool).
"$PYTHON" /home/jongsokchoi/helion_2/examples/aot_pretune/robust_pick.py --gpu 7
