#!/usr/bin/env bash
# Q6 re-benchmark against the schema-v2 upstream-style JSON.
#
# Same runner, same shape grid, same Opus model as the original Q6 — the
# only change is that the dispatcher reads
# helion/autotuner/data/observed_heuristics_b200_quantized.json (21 rules,
# schema v2, per-rule validation blocks) instead of the 57-rule derived
# file under iterations/N6_full_tune/. If the family heldout number
# stays near 0.66, we have a shippable JSON that lives in the same
# location as PR 2378's observed_heuristics_b200.json.

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
# Redirect the dispatcher to read the upstream schema-v2 JSON:
export HELION_QUANTIZED_GEMM_OBSERVED_HEURISTICS_PATH=$REPO/helion/autotuner/data/observed_heuristics_b200_quantized.json

OUT=$REPO/llm_heuristics_artifacts/quantized_gemm/iterations/Q6b_schema_v2
mkdir -p "$OUT/baseline" "$OUT/heuristics" "$OUT/logs"

SHAPES=$REPO/llm_heuristics_artifacts/quantized_gemm/iterations/N0_live/shape_grid.json
HEUR=$REPO/llm_heuristics_artifacts/quantized_gemm/iterations/N6_full_tune/heuristic

ts() { date -Iseconds; }
echo "[$(ts)] === Q6b schema-v2 start ==="
echo "HELION_QUANTIZED_GEMM_OBSERVED_HEURISTICS_PATH=$HELION_QUANTIZED_GEMM_OBSERVED_HEURISTICS_PATH"

# Baseline arm — identical to original Q6 (no heuristic seed). We could
# reuse Q6_exp2/baseline/*.csv, but re-run for timing parity (Opus
# roll of the dice differs run-to-run).
for k in matmul_bf16_int4 _bf16xint16_gemm nvfp4_matmul; do
    echo "[$(ts)] baseline $k"
    unset HELION_LLM_ROUND0_HEURISTIC_PATH || true
    python llm_heuristics_artifacts/quantized_gemm/tools/run_live.py \
        --kernel "$k" --arm baseline \
        --shape-grid "$SHAPES" --output-dir "$OUT/baseline" \
        --repeats 3 --configs-per-round 5 --initial-random-configs 3 \
        > "$OUT/logs/${k}_baseline.log" 2>&1
done

# Heuristic arm — same dispatcher files, pointed at schema-v2 JSON
for pair in \
    "matmul_bf16_int4 heuristic_matmul_bf16_int4.py" \
    "_bf16xint16_gemm heuristic_bf16xint16_gemm.py" \
    "nvfp4_matmul heuristic_nvfp4_matmul.py"
do
    read -r k fname <<<"$pair"
    echo "[$(ts)] heuristics $k"
    export HELION_LLM_ROUND0_HEURISTIC_PATH="$HEUR/$fname"
    python llm_heuristics_artifacts/quantized_gemm/tools/run_live.py \
        --kernel "$k" --arm heuristics \
        --shape-grid "$SHAPES" --output-dir "$OUT/heuristics" \
        --repeats 3 --configs-per-round 5 --initial-random-configs 3 \
        > "$OUT/logs/${k}_heuristics.log" 2>&1
done

echo "[$(ts)] scoring"
python llm_heuristics_artifacts/quantized_gemm/tools/compute_round0_geo.py \
    --baseline "$OUT/baseline" --heuristics "$OUT/heuristics" \
    --output "$OUT/scores.json" 2>&1 | tee "$OUT/logs/scoring.log"

echo "[$(ts)] === Q6b done ==="
