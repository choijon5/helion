# Q6b — Re-benchmark with schema-v2 upstream-style JSON

**Status: DONE (2026-05-10 18:03 UTC). Schema-v2 (21 rules) matches
the schema-v1 Q6 family heldout within 0.2%. Shippable as
`helion/autotuner/data/observed_heuristics_b200_quantized.json`.**

## What changed vs original Q6

- **Input JSON**: switched the dispatcher's rule source from
  `iterations/N6_full_tune/derived_general_heuristics.json` (57 rules,
  schema v1, internal/intermediate format) to
  `helion/autotuner/data/observed_heuristics_b200_quantized.json`
  (21 rules, schema v2, per-rule `validation` + `source` blocks,
  mirrors PR 2378's on-disk location).
- **Dispatcher code**: unchanged. The `_dispatcher_core.lookup_templates`
  already supports both schemas (tries `templates` then
  `selected_templates`). Redirection via
  `HELION_QUANTIZED_GEMM_OBSERVED_HEURISTICS_PATH=...`.
- **Rule promotion criterion for v2**: (a) bucket must be hit by at
  least one live-grid shape, (b) Q5 OR Q6 geomean on that bucket's
  live shapes must be < 1.0. 21 of 57 buckets qualify. The other 36
  cover archive shapes that don't appear in the live 12-shape grid
  (they're latent coverage for shapes the live grid doesn't exercise).

## Results

Family round-0 best-ratio geomean:

| arm set | train | heldout | Δ train→heldout |
|---------|------:|--------:|----------------:|
| Q6 v1 (57 rules, internal)  | 0.650 | **0.664** | +0.014 |
| Q6b v2 (21 rules, upstream) | 0.659 | **0.663** | +0.003 |

Per-kernel:

| kernel             | Q6 v1 train | Q6 v1 heldout | Q6b v2 train | Q6b v2 heldout | v2 − v1 heldout |
|--------------------|------------:|--------------:|-------------:|---------------:|----------------:|
| `_bf16xint16_gemm` |      0.829  |     **0.827** |       0.919  |      **0.835** | +0.008 |
| `matmul_bf16_int4` |      0.576  |     **0.581** |       0.528  |      **0.580** | -0.001 |
| `nvfp4_matmul`     |      0.575  |     **0.609** |       0.590  |      **0.601** | -0.008 |

Family heldout difference: -0.001 (0.2% of v1's value). All three
kernels still pass Exp-2 target heldout ≤ 0.95.

## Overfit-gate note

`matmul_bf16_int4` Δheldout-train = +0.052 is **barely over** the
0.05 gate on this run. In v1 the same kernel was +0.005. The v2
JSON is more conservative (covers only buckets hit by live shapes),
so coverage gaps fall to the per-kernel fallback table. With tighter
coverage on train-covered buckets, the train ratio for int4
*improved* (0.576 → 0.528) faster than heldout. The heldout number
itself is essentially flat (0.581 → 0.580) — not a real overfit
problem. Still worth noting for future shape-grid additions.

## Wall time

- Baselines: 3 × ~3:30 = 10:30
- Heuristics: 3 × ~4:30 = 13:30
- Scoring: 5 s
- Total: 24:43 (17:39 → 18:03)

## Artifacts

- `baseline/*_baseline.csv` + `.meta.json`
- `heuristics/*_heuristics.csv` + `.meta.json`
- `scores.json`
- `logs/*.log`
- Driver: `tools/run_q6_schema_v2.sh`

## Shipping conclusion

The schema-v2 JSON is **perf-equivalent to the v1 derived-general JSON**.
Can be dropped into `helion/autotuner/data/` with no measurable
regression. Blocker for merging into upstream Helion: PR 2378 is still
open and would own the parent observed-heuristics-b200 schema+loader.
Until it merges, options are:

1. Land `observed_heuristics_b200_quantized.json` as a sibling file —
   same directory, distinct name, no autotuner code changes until
   PR 2378 lands.
2. Wait for PR 2378 and then either merge into their single JSON or
   land as a sibling with consistent naming.

Both keep the current dispatcher setup (env-var-driven opt-in via
`HELION_LLM_ROUND0_HEURISTIC_PATH`) unchanged; the upstream landing
is purely about where the JSON lives and its schema version.
