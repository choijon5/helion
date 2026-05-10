#!/usr/bin/env bash
# Q6: Exp-2 live score — LLMGuidedSearch max_rounds=1 baseline vs heuristic seeded.
# Opus 4.7 via Bedrock, adaptive thinking.
# 3 kernels × 2 arms × 12 shapes × 3 repeats = 216 LLM-guided searches.

set -uo pipefail

REPO=/home/dev/helion_choijon5
cd "$REPO"
source /home/dev/miniconda3/etc/profile.d/conda.sh
conda activate helion_choijon5

export CUDA_VISIBLE_DEVICES=0
export HELION_LLM_PROVIDER=bedrock
export HELION_LLM_MODEL=us.anthropic.claude-opus-4-7
export HELION_LLM_ANTHROPIC_THINKING_BUDGET=8000
export HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1
export HELION_AUTOTUNE_IGNORE_ERRORS=1

OUT=$REPO/llm_heuristics_artifacts/quantized_gemm/iterations/Q6_exp2
mkdir -p "$OUT/baseline" "$OUT/heuristics" "$OUT/logs"

SHAPES=$REPO/llm_heuristics_artifacts/quantized_gemm/iterations/N0_live/shape_grid.json
HEUR=$REPO/llm_heuristics_artifacts/quantized_gemm/iterations/N6_full_tune/heuristic

ts() { date -Iseconds; }
echo "[$(ts)] === Q6 Exp-2 start ==="
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv

# Baseline arm
for k in matmul_bf16_int4 _bf16xint16_gemm nvfp4_matmul; do
    echo "[$(ts)] baseline $k"
    unset HELION_LLM_ROUND0_HEURISTIC_PATH || true
    python llm_heuristics_artifacts/quantized_gemm/tools/run_live.py \
        --kernel "$k" \
        --arm baseline \
        --shape-grid "$SHAPES" \
        --output-dir "$OUT/baseline" \
        --repeats 3 \
        --configs-per-round 5 \
        --initial-random-configs 3 > "$OUT/logs/${k}_baseline.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[$(ts)] !!! baseline $k exit=$rc"
    fi
done

# Heuristics arm
for pair in \
    "matmul_bf16_int4 heuristic_matmul_bf16_int4.py" \
    "_bf16xint16_gemm heuristic_bf16xint16_gemm.py" \
    "nvfp4_matmul heuristic_nvfp4_matmul.py"
do
    read -r k fname <<<"$pair"
    echo "[$(ts)] heuristics $k"
    export HELION_LLM_ROUND0_HEURISTIC_PATH="$HEUR/$fname"
    python llm_heuristics_artifacts/quantized_gemm/tools/run_live.py \
        --kernel "$k" \
        --arm heuristics \
        --shape-grid "$SHAPES" \
        --output-dir "$OUT/heuristics" \
        --repeats 3 \
        --configs-per-round 5 \
        --initial-random-configs 3 > "$OUT/logs/${k}_heuristics.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[$(ts)] !!! heuristics $k exit=$rc"
    fi
done

echo "[$(ts)] scoring"
python llm_heuristics_artifacts/quantized_gemm/tools/compute_round0_geo.py \
    --baseline "$OUT/baseline" \
    --heuristics "$OUT/heuristics" \
    --output "$OUT/scores.json" 2>&1 | tee "$OUT/logs/scoring.log"

echo "[$(ts)] === Q6 Exp-2 done ==="
