#!/usr/bin/env bash
# Fixed Q5 native runner — sets env vars in shell before Python starts.
set -uo pipefail
REPO=/tmp/obs-heur-worktree
cd "$REPO"
source /home/dev/miniconda3/etc/profile.d/conda.sh
conda activate helion_choijon5
export CUDA_VISIBLE_DEVICES=0
export HELION_AUTOTUNE_EFFORT=none   # <-- critical: before Python

SHAPES=$REPO/llm_heuristics_artifacts/quantized_gemm/iterations/N0_live/shape_grid.json
OUT=$REPO/llm_heuristics_artifacts/quantized_gemm/iterations/Q5_native_fixed
rm -rf "$OUT"; mkdir -p "$OUT/baseline" "$OUT/heuristics" "$OUT/logs"

ts() { date -Iseconds; }
echo "[$(ts)] === Q5 native (fixed env) ==="
for arm in baseline heuristics; do
    if [[ "$arm" == "heuristics" ]]; then
        export HELION_AUTOTUNE_OBSERVED_HEURISTICS=1
    else
        export HELION_AUTOTUNE_OBSERVED_HEURISTICS=0
    fi
    for k in matmul_bf16_int4 _bf16xint16_gemm nvfp4_matmul; do
        echo "[$(ts)] $arm $k"
        python llm_heuristics_artifacts/quantized_gemm/tools/run_no_autotune_native.py \
            --kernel "$k" --arm "$arm" --shape-grid "$SHAPES" \
            --output-dir "$OUT/$arm" --repeats 3 \
            > "$OUT/logs/${k}_${arm}.log" 2>&1
    done
done
echo "[$(ts)] scoring"
python llm_heuristics_artifacts/quantized_gemm/tools/compute_round0_geo.py \
    --baseline "$OUT/baseline" --heuristics "$OUT/heuristics" \
    --output "$OUT/scores.json" 2>&1 | tee "$OUT/logs/scoring.log"
