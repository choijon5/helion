# Q3 — Observed-heuristics JSON from the Q2 archive

**Status: DONE (2026-05-10). 3 rules pass strict LOSO promotion. 57 raw
data-derived rules retained for use as LLM context and by a dispatcher
that tolerates looser filters.**

## What ran

```bash
python scripts/llm_heuristics_research.py \
    --data-root aot_pretune_data/b200 \
    --output-dir llm_heuristics_artifacts/quantized_gemm/iterations/Q3_heuristic
```

Input: 26 751 measurement rows across 3 kernels × 40 shapes each =
120 shapes (run id `20260509_q2_llmseeded`).

## Script fix required first

`scripts/llm_heuristics_research.py:infer_kernel_class` has hard-coded
kernel→class mappings. The quantized kernels hit the fallback
`return "unknown"`, which collapsed all 120 shapes into one bucket
keyed only by dtype — and since the A-operand of every quantized
kernel is bf16, the dtype bucket was `fp16_bf16` for everything.
The resulting rule mixed int4/int16/fp4 shapes together and failed
the runtime filter.

Fix (additive, additive only, no removals):

- `infer_kernel_class`: added `matmul_int4`, `matmul_int16`,
  `matmul_fp4` mapped from the three kernel names.
- `shape_bucket_for_class`: added the new classes to the matmul-style
  bucketing branch (so bucketing uses `(aspect, dtype, m_bin, n_bin,
  k_bin)` exactly like the dense-GEMM loop).
- `_shape_label`: extended to format `M/N/K/size` labels for the
  quantized kernels in the diagnostic report.

Verified by rerunning the generator: kernel_class column now shows
`matmul_fp4`, `matmul_int16`, `matmul_int4` (19 buckets each), not
`unknown`.

## Rule counts

| kernel_class  | derived rules | promoted (runtime JSON) |
|---------------|--------------:|------------------------:|
| `matmul_fp4`  | 19            | 2 |
| `matmul_int4` | 19            | 1 |
| `matmul_int16`| 19            | 0 |
| **total**     | **57**        | **3** |

The filter that gates the runtime JSON is strict:

```python
min_rule_shapes       >= 2
min_holdout_coverage  >= 0.75
max_rule_holdout_geomean_slowdown <= 1.05
max_rule_holdout_p90_slowdown     <= 1.10
min_template_shapes   >= 2
max_template_geomean_slowdown <= 1.01
max_template_p90_slowdown     <= 1.10
```

## Promoted rules

### matmul_fp4 · skinny_n (n ≤ 64)

- source kernel: `nvfp4_matmul`, 4 shapes (N ∈ {16,32,64,128 with M=K=4096 and M=K=2048})
- LOSO geo/p90: 1.000/1.000, coverage 4/4
- template: `block_sizes=[128,32,16], l2_groupings=[2], num_stages=3, num_warps=4, pid_type=flat`

### matmul_fp4 · skinny_n (64 < n ≤ 1024)

- 2 shapes
- LOSO geo/p90: 1.006/1.012
- templates: `block_sizes=[8,128,128] num_stages=6` and `num_stages=8`,
  l2_groupings 1 or 4, num_warps 4

### matmul_int4 · skinny_n (64 < n ≤ 1024)

- source kernel: `matmul_bf16_int4`, 2 shapes
- LOSO geo/p90: 1.047/1.095 (tightest of the three)
- template: `block_sizes=[16,128,128], l2_groupings=[1], num_stages=4, num_warps=4, pid_type=flat`

## Near-misses (worth investigating)

### matmul_fp4 · skinny_m (m ≤ 64)

Rule-level metrics pass (geo 1.032, p90 1.095, cov 3), but one
template failed `max_template_geomean_slowdown ≤ 1.01`. Worth
inspecting: this is a common skinny-M case for fp4 decode workloads.

### matmul_int16 family

Best rule is `skinny_k (k≤256)` with LOSO geo 1.075 (just above
1.05), only 1 shape. The balanced buckets come in at 1.17–1.49,
suggesting int16 tuning is legitimately harder to generalize from
2–4 shapes per bucket. Options:
- Loosen the int16-only filter (risk: noisy rules)
- Collect more int16 shapes in a follow-on expansion
- Accept "no int16 heuristic" and rely on autotune for that kernel

### matmul_int4 · skinny_k (k ≤ 64 and k ≤ 256)

LOSO geo 1.000 and 1.060 look great, but both have only 1–2
shape coverage. The p90 on `k≤256` is 1.121 — just over the threshold.

## Shape-by-shape kernel ranking

For a quick sense of which kernels have the "best" overall bucketed
evidence:

| kernel_class  | # buckets | # promoted | best bucket geo | worst bucket geo |
|---------------|----------:|-----------:|----------------:|-----------------:|
| `matmul_fp4`  | 19        | 2          | 1.000           | inf (empty bucket) |
| `matmul_int4` | 19        | 1          | 1.000           | inf (empty bucket) |
| `matmul_int16`| 19        | 0          | 1.075           | inf (empty bucket) |

## Go/no-go for Q4

Recommend **proceed to Q4** (dispatcher-building). The strict filter
produced only 3 rules, but:

- The dispatcher doesn't have to be built from the runtime JSON
  alone. The dense-GEMM N6 dispatcher reads both the strict runtime
  JSON AND the less-strict derived-general JSON (57 rules here),
  reusing the latter for buckets the former doesn't cover. Works out
  because bucket overlap is monotonic.
- Missing int16 coverage is known (see near-misses). The Q5/Q6
  live scores will expose whether lack of heuristic translates to
  poor round-0 numbers. If it does, that informs whether to go back
  for more int16 training data or let autotune handle int16 alone.
- matmul_fp4 skinny_n is a very promising promoted rule set — the
  hot-pole shape for fp4 decode is exactly skinny-M/N decode.

## Artifacts

- `runtime_observed_heuristics_b200.json` — 3 rules that will seed
  the round-0 dispatcher.
- `derived_general_heuristics.json` — 57 rules (19/kernel) that can
  back-fill or inform LLM prompting.
- `derived_general_heuristics.md` — human-readable version.
- `llm_heuristics_configs.json` — per-kernel top-config +
  shape-regime summary (used for LLM prompting, not the runtime
  dispatcher).
- `llm_heuristics_report.md` — kernel×regime summary.
- `opus_prompt.md` — prompt body for the optional LLM-critique step.

## Next (Q4)

Build three kernel-specific dispatcher `.py` files. Template to
follow: `llm_heuristics_artifacts/gemm/iterations/N6_full_tune/heuristic/heuristic_matmul.py`
(the N9 shmem-safety fix is baked in). For each kernel:

1. Load the runtime JSON at module import.
2. On `autotune_<kernel>(shape)` call, compute bucket
   `(aspect, dtype, m_bin, n_bin, k_bin)`.
3. Find a matching rule; pick the best-fitting template (win_count
   desc, geomean asc) that fits B200 shmem.
4. Fall back to a hand-tuned default if no rule matches.

Then Q5 (Exp-1, effort=none, target heldout ≤ 0.20) and Q6 (Exp-2,
LLM-on, target heldout ≤ 0.95).
