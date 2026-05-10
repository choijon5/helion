# N9 — Fix MM_003 shmem-overflow crash

**Status: DONE. Fix lands the MM_003 shape as a 6× speedup instead of an error.**

## Bug

N6 heuristic dispatched `block_sizes=[128, 64, 128] num_stages=6` for
the `m_bin<=1024, n_bin<=1024, k_bin<=1024, balanced, fp16_bf16`
bucket. For MM_003 (1024×1024×1024 fp16) that config requests:

```
A tile: 128 * 128 * 2 B = 32 KB
B tile: 128 *  64 * 2 B = 16 KB
per-stage: 48 KB × 6 stages = 288 KB
```

B200's shared-memory capacity is 232 KB. The config was a winner on
archive shapes where other tile sizes fit (the exact config compiles
on e.g. 384³ or 256³), but on 1024³ it triggers
`OutOfResources: shared memory, Required: 294928, Hardware limit:
232448`.

In the N6 LLM-on arm and in N7-B (LLM-off with random seeds), the bad
config is masked by other seeds that win the round-0 min. In Exp-1
(single-config dispatch, `run_no_autotune.py`), the failure becomes
visible: the heuristic arm emits an uncompilable config and the row is
dropped from the geomean.

## Fix

Added a static shmem-budget check to both
`heuristic_matmul.py` and `heuristic_fp8_gemm.py`. Three components:

1. **`_estimate_matmul_shmem_bytes`**: computes
   `(block_m * block_k + block_k * block_n) * dtype_size * num_stages`
   as the shared-memory footprint of a Helion matmul tile. Accumulator
   lives in registers, not shmem.
2. **`_fits_shmem_budget`**: returns True iff the estimate is ≤ 90%
   of B200's 232 KB capacity (safety margin for Triton runtime
   reservations).
3. **`_lookup_template`** now returns *all* templates for the bucket,
   ordered best-first by `(win_count desc, geomean_slowdown asc)`.
4. **`autotune_matmul`** / **`autotune_fp8_gemm`** iterate templates
   and pick the first one that fits the budget for the live query
   shape. If every template exceeds budget (should not happen on
   current data), `_shrink_to_fit` halves `num_stages` then `block_k`
   until the config fits.
5. **The same shmem filter** is applied to the N3b fallback config,
   in case the hand-coded adaptive branch ever picks a too-big tile.

No changes to the JSON schema, no runtime compile probes.

## Before vs after (Exp-1, single-config, no autotune)

| kernel   | scope   | D6 pre-fix | D9 post-fix | D9 errors |
|----------|---------|-----------:|------------:|----------:|
| matmul   | train   | 0.0861 (MM_003 dropped) | 0.0942 (MM_003 = 0.164 ratio) | 0 |
| matmul   | heldout | 0.0877     | 0.0875      | 0 |
| fp8_gemm | train   | 0.0871     | 0.0867      | 0 |
| fp8_gemm | heldout | 0.0741     | 0.0739      | 0 |

MM_003 now lands as:

| rep | baseline (default) | heuristic (post-fix) | ratio |
|----:|-------------------:|---------------------:|------:|
| 0 | 94.11 μs | 15.36 μs | 0.163 |
| 1 | 93.34 μs | 15.36 μs | 0.165 |
| 2 | 93.36 μs | 15.36 μs | 0.165 |

Still a legitimate ~6× speedup on MM_003, just not the 23× that the
un-shrunk template would have achieved if it compiled. Non-MM_003
shapes are within rounding of the prior D6 scores — the fix is
surgical.

## What the dispatcher picks now for MM_003

Without shmem check: `[128, 64, 128] stages=6` (288 KB → ERROR).
With shmem check: next template in the same bucket —
`[64, 64, 128] stages=4` (128 KB → ok). That template was rank-2 by
`win_count` in the N6 JSON for this bucket.

## Why this is the right layer

Two other places this fix could live:

- **Generator (`scripts/llm_heuristics_research.py`)**: probe-compile
  each candidate template on a representative shape from its bucket
  before emitting. Stronger guarantee (never ships a bad template),
  but expensive: requires a GPU + one compile per candidate per
  bucket, and "representative shape" is slippery for buckets with
  wide dim ranges.
- **Runtime compile probe in the dispatcher**: try to compile the
  picked config; on failure, fall through to the next. Robust against
  unknown-unknown failure modes (register pressure, etc) but adds a
  compile to the round-0 critical path.

The static estimator is the cheapest option that catches the observed
failure. If we see a different hardware-limit failure later (e.g.
register spill on Hopper), add a similar check. If we see a failure
we can't estimate statically, fall back to the runtime probe.

## Known constant that needs updating for non-B200 hardware

`_B200_SHMEM_LIMIT_BYTES = 232448` is hardcoded. A general fix would
read `torch.cuda.get_device_properties(...).shared_memory_per_block`
but this loop is B200-only per the plan. Flagged for N10 packaging.

## Artifacts

- Updated `iterations/N6_full_tune/heuristic/heuristic_matmul.py`
- Updated `iterations/N6_full_tune/heuristic/heuristic_fp8_gemm.py`
- `iterations/N9_shmem_fix/heuristics_no_autotune/*.csv` — post-fix
  Exp-1 data with MM_003 succeeding.
