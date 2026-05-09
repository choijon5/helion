# Phase 1: Per-Round Analysis Report

*Auto-generated from baseline/{kernel}/{kernel}_baseline_round_progress.csv*


## Per-Kernel Summary

| Kernel | Shapes | Rounds | Total Δ | Plateau@ | Regressions |
|--------|--------|--------|---------|----------|-------------|
| attention | 12 | 4 | +19.7% | 3 | — |
| cross_entropy | 12 | 4 | -10.3% | 1 | 3 |
| layernorm | 12 | 4 | +38.1% | 1 | 1 |
| matmul | 12 | 4 | -4.2% | 3 | 3 |
| softmax | 12 | 4 | -24.1% | 1 | 3 |

## Per-Round Improvement (% vs previous round)

| Kernel | R0 | R1 | R2 | R3 |
|--------|--------|--------|--------|--------|
| attention | —    | +11.7% | +7.2% | +2.0% |
| cross_entropy | —    | +0.9% | +2.8% | -14.5% |
| layernorm | —    | -7.0% | +15.0% | +32.0% |
| matmul | —    | +10.7% | +3.8% | -21.2% |
| softmax | —    | +1.0% | +2.3% | -28.3% |

## Per-Round Geometric Mean (ms)

| Kernel | R0 (init) | R1 | R2 | R3 |
|--------|--------|--------|--------|--------|
| attention | 27.519 | 24.309 | 22.558 | 22.107 |
| cross_entropy | 39.301 | 38.955 | 37.882 | 43.366 |
| layernorm | 13.654 | 14.615 | 12.428 | 8.456 |
| matmul | 45.600 | 40.727 | 39.187 | 47.501 |
| softmax | 23.036 | 22.817 | 22.290 | 28.599 |

## Recommendations for Phase 2

- Biggest average gain is at round 2 (+6.2%). → Focus prompt optimization on round 2.
- Early plateau (round ≤1) on: cross_entropy, layernorm, softmax. → Try **Approach 6 (Success Pattern Learning)** to help LLM exploit winners faster.
- Regression detected: cross_entropy (round [3]), layernorm (round [1]), matmul (round [3]), softmax (round [3]). → Try **Approach 1 (Multi-Round Tuning, cap rounds)** or simplify late-round prompts.
- Kernel-type gap detected: memory-bound avg 1.2%, compute-bound avg 7.7% (Δ 6.6%, compute-bound improves more). → Try **Approach 4 (Theoretical Guidance)** or **Approach 7 (Compiler-Detected Patterns)**.
