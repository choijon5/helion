#!/usr/bin/env bash
# Q5 + Q6 re-benchmark running on their branch with merged 30-rule JSON.
# No dispatcher files; uses the native observed-heuristics runtime.

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
OUT_BASE=$REPO/llm_heuristics_artifacts/quantized_gemm/iterations

ts() { date -Iseconds; }

# --- Q5 native ---
echo "[$(ts)] === Q5 native start ==="
OUT=$OUT_BASE/Q5_native
mkdir -p "$OUT/baseline" "$OUT/heuristics" "$OUT/logs"
for arm in baseline heuristics; do
    for k in matmul_bf16_int4 _bf16xint16_gemm nvfp4_matmul; do
        echo "[$(ts)] Q5 $arm $k"
        python llm_heuristics_artifacts/quantized_gemm/tools/run_no_autotune_native.py \
            --kernel "$k" --arm "$arm" --shape-grid "$SHAPES" \
            --output-dir "$OUT/$arm" --repeats 3 \
            > "$OUT/logs/${k}_${arm}.log" 2>&1
    done
done
echo "[$(ts)] Q5 scoring"
python llm_heuristics_artifacts/quantized_gemm/tools/compute_round0_geo.py \
    --baseline "$OUT/baseline" --heuristics "$OUT/heuristics" \
    --output "$OUT/scores.json" 2>&1 | tee "$OUT/logs/scoring.log"
echo "[$(ts)] Q5 done"

# --- Q6 native (LLM-on) ---
# Their run_live.py needs to have observed heuristics on for the heuristics
# arm and off for the baseline arm. The LLMGuidedSearch integration already
# reads them via observed_heuristic_seed_configs_for_kernel; just toggle env.
echo "[$(ts)] === Q6 native start ==="
OUT=$OUT_BASE/Q6_native
mkdir -p "$OUT/baseline" "$OUT/heuristics" "$OUT/logs"
for arm in baseline heuristics; do
    for k in matmul_bf16_int4 _bf16xint16_gemm nvfp4_matmul; do
        echo "[$(ts)] Q6 $arm $k"
        if [[ "$arm" == "heuristics" ]]; then
            export HELION_AUTOTUNE_OBSERVED_HEURISTICS=1
        else
            export HELION_AUTOTUNE_OBSERVED_HEURISTICS=0
        fi
        unset HELION_LLM_ROUND0_HEURISTIC_PATH || true
        python llm_heuristics_artifacts/quantized_gemm/tools/run_live.py \
            --kernel "$k" --arm baseline --shape-grid "$SHAPES" \
            --output-dir "$OUT/$arm" --repeats 3 \
            --configs-per-round 5 --initial-random-configs 3 \
            > "$OUT/logs/${k}_${arm}.log" 2>&1 || true
    done
done
echo "[$(ts)] Q6 scoring"
python llm_heuristics_artifacts/quantized_gemm/tools/compute_round0_geo.py \
    --baseline "$OUT/baseline" --heuristics "$OUT/heuristics" \
    --output "$OUT/scores.json" 2>&1 | tee "$OUT/logs/scoring.log"
echo "[$(ts)] Q6 done"

echo "[$(ts)] === all done ==="
