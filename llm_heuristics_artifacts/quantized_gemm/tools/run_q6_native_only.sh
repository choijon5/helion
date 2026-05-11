#!/usr/bin/env bash
set -uo pipefail
REPO=/tmp/obs-heur-worktree
cd "$REPO"
source /home/dev/miniconda3/etc/profile.d/conda.sh
conda activate helion_choijon5

export CUDA_VISIBLE_DEVICES=0
export HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1
export HELION_AUTOTUNE_IGNORE_ERRORS=1
export HELION_LLM_PROVIDER=bedrock
export HELION_LLM_MODEL=us.anthropic.claude-opus-4-7
export HELION_LLM_ANTHROPIC_THINKING_BUDGET=8000

SHAPES=$REPO/llm_heuristics_artifacts/quantized_gemm/iterations/N0_live/shape_grid.json
OUT=$REPO/llm_heuristics_artifacts/quantized_gemm/iterations/Q6_native
rm -rf "$OUT"
mkdir -p "$OUT/baseline" "$OUT/heuristics" "$OUT/logs"

ts() { date -Iseconds; }
echo "[$(ts)] === Q6 native (LLM-on) start ==="
for arm in baseline heuristics; do
    for k in matmul_bf16_int4 _bf16xint16_gemm nvfp4_matmul; do
        echo "[$(ts)] $arm $k"
        python llm_heuristics_artifacts/quantized_gemm/tools/run_live_native.py \
            --kernel "$k" --arm "$arm" --shape-grid "$SHAPES" \
            --output-dir "$OUT/$arm" --repeats 3 \
            --configs-per-round 5 --initial-random-configs 3 \
            > "$OUT/logs/${k}_${arm}.log" 2>&1
    done
done
echo "[$(ts)] scoring"
python llm_heuristics_artifacts/quantized_gemm/tools/compute_round0_geo.py \
    --baseline "$OUT/baseline" --heuristics "$OUT/heuristics" \
    --output "$OUT/scores.json" 2>&1 | tee "$OUT/logs/scoring.log"
echo "[$(ts)] done"
