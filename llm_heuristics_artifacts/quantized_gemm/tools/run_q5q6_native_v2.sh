#!/usr/bin/env bash
# Q5 + Q6 native re-benchmark, v2: correct env-var hygiene.
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
# Sentinels in Python require explicit values; always export both:
export HELION_AUTOTUNE_EFFORT=none  # overridden below for Q6, reset for Q5
export HELION_AUTOTUNE_OBSERVED_HEURISTICS=0

SHAPES=$REPO/llm_heuristics_artifacts/quantized_gemm/iterations/N0_live/shape_grid.json
ts() { date -Iseconds; }

# ---- Q5 (no-autotune) ----
OUT=$REPO/llm_heuristics_artifacts/quantized_gemm/iterations/Q5_native_v2
rm -rf "$OUT"; mkdir -p "$OUT/baseline" "$OUT/heuristics" "$OUT/logs"
export HELION_AUTOTUNE_EFFORT=none
echo "[$(ts)] === Q5 native v2 start ==="
for arm in baseline heuristics; do
    if [[ "$arm" == "heuristics" ]]; then export HELION_AUTOTUNE_OBSERVED_HEURISTICS=1
    else export HELION_AUTOTUNE_OBSERVED_HEURISTICS=0; fi
    for k in matmul_bf16_int4 _bf16xint16_gemm nvfp4_matmul; do
        echo "[$(ts)] Q5 $arm $k"
        python llm_heuristics_artifacts/quantized_gemm/tools/run_no_autotune_native.py \
            --kernel "$k" --arm "$arm" --shape-grid "$SHAPES" \
            --output-dir "$OUT/$arm" --repeats 3 > "$OUT/logs/${k}_${arm}.log" 2>&1
    done
done
echo "[$(ts)] Q5 scoring"
python llm_heuristics_artifacts/quantized_gemm/tools/compute_round0_geo.py \
    --baseline "$OUT/baseline" --heuristics "$OUT/heuristics" \
    --output "$OUT/scores.json" 2>&1 | tee "$OUT/logs/scoring.log"
echo "[$(ts)] Q5 done"

# ---- Q6 (LLM-on) ----
OUT=$REPO/llm_heuristics_artifacts/quantized_gemm/iterations/Q6_native_v2
rm -rf "$OUT"; mkdir -p "$OUT/baseline" "$OUT/heuristics" "$OUT/logs"
# effort is not gated here; LLMGuidedSearch ignores effort
echo "[$(ts)] === Q6 native v2 start ==="
for arm in baseline heuristics; do
    if [[ "$arm" == "heuristics" ]]; then export HELION_AUTOTUNE_OBSERVED_HEURISTICS=1
    else export HELION_AUTOTUNE_OBSERVED_HEURISTICS=0; fi
    for k in matmul_bf16_int4 _bf16xint16_gemm nvfp4_matmul; do
        echo "[$(ts)] Q6 $arm $k"
        python llm_heuristics_artifacts/quantized_gemm/tools/run_live_native.py \
            --kernel "$k" --arm "$arm" --shape-grid "$SHAPES" \
            --output-dir "$OUT/$arm" --repeats 3 \
            --configs-per-round 5 --initial-random-configs 3 \
            > "$OUT/logs/${k}_${arm}.log" 2>&1
    done
done
echo "[$(ts)] Q6 scoring"
python llm_heuristics_artifacts/quantized_gemm/tools/compute_round0_geo.py \
    --baseline "$OUT/baseline" --heuristics "$OUT/heuristics" \
    --output "$OUT/scores.json" 2>&1 | tee "$OUT/logs/scoring.log"
echo "[$(ts)] Q6 done"

echo "[$(ts)] === all done ==="
