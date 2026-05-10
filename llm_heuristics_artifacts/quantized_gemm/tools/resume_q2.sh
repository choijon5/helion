#!/usr/bin/env bash
# Resume Q2 archive tuning for matmul_bf16_int4 and nvfp4_matmul.
# _bf16xint16_gemm already finished all 40 shapes on 2026-05-09.
#
# Per-shape subprocess isolation: run_full_tuning.py is invoked once
# per (kernel, bucket, tag). If a Triton config triggers an
# unrecoverable CUDA error (misaligned address, etc.), the crash is
# contained to that shape's subprocess. Subsequent shapes get a fresh
# CUDA context.

set -uo pipefail

REPO=/home/dev/helion_choijon5
LOG_DIR=$REPO/llm_heuristics_artifacts/quantized_gemm/iterations/Q2_expansion
mkdir -p "$LOG_DIR"

source /home/dev/miniconda3/etc/profile.d/conda.sh
conda activate helion_choijon5
cd "$REPO"

export HELION_AUTOTUNE_EFFORT=full
export HELION_AUTOTUNER=LLMSeededLFBOTreeSearch
export HELION_AUTOTUNE_IGNORE_ERRORS=1
export HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1
export HELION_LLM_PROVIDER=bedrock
export HELION_LLM_MODEL=us.anthropic.claude-opus-4-7
export HELION_LLM_ANTHROPIC_THINKING_BUDGET=8000
export CUDA_VISIBLE_DEVICES=0

SHAPES=$REPO/llm_heuristics_artifacts/quantized_gemm/iterations/N1_expand/expansion_shapes.json
ARCHIVE=$REPO/aot_pretune_data/b200
RUN_ID=20260509_q2_llmseeded
RUNNER=$REPO/llm_heuristics_artifacts/quantized_gemm/tools/run_full_tuning.py

ts() { date -Iseconds; }

run_shape() {
    local kernel=$1 bucket=$2 tag=$3 log_base=$4
    local shape_log="$LOG_DIR/${kernel}_${tag}.log"
    echo "[$(ts)] >>> $kernel / $bucket / $tag" | tee -a "$log_base"
    python "$RUNNER" \
        --kernel "$kernel" \
        --shapes-json "$SHAPES" \
        --archive-root "$ARCHIVE" \
        --run-id "$RUN_ID" \
        --only-bucket "$bucket" \
        --only-tag "$tag" \
        > "$shape_log" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        grep -E "configs recorded|Elapsed:" "$shape_log" | tail -2 | tee -a "$log_base"
    else
        echo "[$(ts)] !!! $kernel/$tag exited $rc — see $shape_log" | tee -a "$log_base"
    fi
}

MAIN_LOG=$LOG_DIR/resume_q2_driver.log
: > "$MAIN_LOG"

echo "=== [$(ts)] resume Q2 driver (per-shape subprocess) ===" | tee -a "$MAIN_LOG"
echo "HELION_AUTOTUNER=$HELION_AUTOTUNER  effort=$HELION_AUTOTUNE_EFFORT" | tee -a "$MAIN_LOG"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "$MAIN_LOG"

# ---- matmul_bf16_int4 : 3 remaining shapes ----
echo "=== [$(ts)] matmul_bf16_int4 resume ===" | tee -a "$MAIN_LOG"
# bucket,tag pairs
for pair in \
    "rectangular rect_2k_1k_4k" \
    "rectangular rect_8k_2k_512" \
    "rectangular rect_1536_3072_1536"
do
    read -r bucket tag <<<"$pair"
    run_shape matmul_bf16_int4 "$bucket" "$tag" "$MAIN_LOG"
done

# ---- nvfp4_matmul : 33 remaining shapes ----
echo "=== [$(ts)] nvfp4_matmul resume ===" | tee -a "$MAIN_LOG"
for pair in \
    "balanced_mid sq3072" \
    "skinny_m m16_k2048_n2048" \
    "skinny_m m32" \
    "skinny_m m64" \
    "skinny_m m128_k2048_n2048" \
    "skinny_m m256_k2048_n2048" \
    "skinny_m m16_k4k_n4k" \
    "skinny_m m64_k4k_n4k" \
    "skinny_m m256_k4k_n4k" \
    "skinny_n n16_m2048_k2048" \
    "skinny_n n32" \
    "skinny_n n64" \
    "skinny_n n128_m2048_k2048" \
    "skinny_n n256_m2048_k2048" \
    "skinny_n n32_m4k_k4k" \
    "skinny_n n128_m4k_k4k" \
    "skinny_n n256_m4k_k4k" \
    "skinny_k k32_m2k_n2k" \
    "skinny_k k64" \
    "skinny_k k128_m2k_n2k" \
    "skinny_k k256" \
    "skinny_k k512" \
    "skinny_k k128_m4k_n4k" \
    "skinny_k k256_m4k_n4k" \
    "skinny_k k512_m4k_n4k" \
    "rectangular rect_1k_2k_4k" \
    "rectangular rect_4k_2k_1k" \
    "rectangular rect_2k_1k_4k" \
    "rectangular rect_2k_4k_1k" \
    "rectangular rect_512_2k_8k" \
    "rectangular rect_8k_2k_512" \
    "rectangular rect_1536_3072_1536" \
    "rectangular rect_3072_1536_3072"
do
    read -r bucket tag <<<"$pair"
    run_shape nvfp4_matmul "$bucket" "$tag" "$MAIN_LOG"
done

echo "=== [$(ts)] driver done ===" | tee -a "$MAIN_LOG"
