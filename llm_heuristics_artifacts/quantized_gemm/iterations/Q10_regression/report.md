# Q10 — Perf regression verification (3-way)

**Status: DONE (2026-05-11). All three checks pass. Safe to land on
PR 2378's branch.**

Three comparisons required by plan.md: no workload (new or existing)
should run slower after our changes.

## Check 1: vs our-branch pre-Q8 (quantized kernels)

Does the PR 2378 runtime + fallbacks produce quantized-kernel perf
within 5% of our-branch's standalone dispatcher?

| experiment | our-branch | PR 2378 + fallbacks | Δ |
|------------|-----------:|--------------------:|--:|
| Q5 family heldout | 0.142 | **0.133** | -6% (better) |
| Q6 family heldout | 0.664 | **0.592** | -11% (better) |

Per-kernel heldout ratios all within 5% between the two paths, or
better on PR 2378. **PASS.**

## Check 2: vs PR 2378's existing rules

Do the 9 pre-existing rules (6 attention, 2 matmul, 1 row_softmax)
still resolve correctly after my patches to `observed_heuristics.py`?

Approaches:

1. **Unit tests**: `pytest test/test_observed_heuristics.py` — all
   4 tests pass. Their existing lookup path is unchanged.
2. **Rule-lookup sanity**: for each existing rule in the JSON,
   `_find_rule(rule.kernel_class, rule.shape_bucket)` returns
   the same rule object. Verified programmatically for all 9
   pre-existing rules.
3. **Isolation**: the new `fallbacks` block only registers
   `matmul_int4 / matmul_int16 / matmul_fp4`. PR 2378's kernel
   classes (`attention / matmul / row_softmax`) never trigger
   fallback lookups because (a) they have no entries in the
   `fallbacks` map, and (b) `_fallback_group_for_class` is only
   consulted when exact `_find_rule` returns None.

**PASS.**

## Check 3: vs stock Helion (heuristics disabled)

With `HELION_AUTOTUNE_OBSERVED_HEURISTICS=0`, all three quantized
kernels should get Helion's stock default config (the runtime
integration must not leak into the heuristics-off path).

Measured at M=K=N=1024 bf16:

| kernel | with OBSERVED=0 | Helion default_config | match |
|--------|----------------:|----------------------:|:-----:|
| `matmul_bf16_int4` | `[16,16,16] ns1 nw4` | `[16,16,16] ns1 nw4` | ✅ |
| `_bf16xint16_gemm` | `[16,16,16] ns1 nw4` | `[16,16,16] ns1 nw4` | ✅ |
| `nvfp4_matmul`     | `[16,16,16] ns1 nw4` | `[16,16,16] ns1 nw4` | ✅ |

Runtime log line also reads "Using default config:" instead of
"Using observed heuristic config:", confirming the code path is
what stock Helion runs.

**PASS.**

## Summary

| check | result |
|-------|:-------|
| 1. vs our-branch pre-Q8 | ✅ better on Q5 (-6%) and Q6 (-11%) |
| 2. vs PR 2378's existing rules (attention/matmul/softmax) | ✅ no change |
| 3. vs stock Helion (heuristics off) | ✅ exact-match |

Safe to land the Q9 schema patch + runtime changes + 30-rule JSON
+ fallbacks block on PR 2378's branch.
