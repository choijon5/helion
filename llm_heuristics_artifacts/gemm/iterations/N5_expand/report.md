# N5 — Observed-heuristics JSON stacked on N3b hybrid

**Status: FAIL (heldout ≤ 0.80 not met). N3b-v2 remains champion.**

## Result

| kernel     | scope   | round0_best_geo | delta (hld − trn) |
|------------|--------:|----------------:|------------------:|
| matmul     | train   | 0.8567          |                    |
| matmul     | heldout | 0.9129          | +0.0562            |
| fp8_gemm   | train   | 0.9173          |                    |
| fp8_gemm   | heldout | 0.8857          | −0.0316            |
| family     | train   | 0.8865          |                    |
| family     | heldout | **0.8992**      | +0.0127            |

## Comparison to prior gates

| gate    | fam train | fam heldout | delta  |
|---------|----------:|------------:|-------:|
| N0 baseline       | 1.000 | 1.000 | 0.000  |
| N2 archive tree   | 0.793 | 0.862 | +0.069 |
| N3 adaptive       | 0.879 | 0.880 | +0.001 |
| **N3b-v2 hybrid** | **0.795** | **0.862** | +0.068 |
| N5 JSON + N3b     | 0.887 | 0.899 | +0.013 |

N5 improved the overfit delta (+0.013 vs N3b's +0.068) but at the cost
of both train and heldout absolute. That's a worse trade than keeping
N3b-v2.

## What went wrong

1. **Random-sample expansion tuning is shallower than the archive's
   dense sweep.** For in-distribution shapes (e.g. 1024³ matmul), the
   original archive had ~60 converged configs; our 40 random configs
   produced a "winner" that was 5–10% slower than the archive's
   actually-tuned winner. So for buckets where N5 could dispatch via
   its JSON templates, those templates were *worse* than what N2's
   archive tree had for the same shape.
2. **Template fields omit tuning knobs that matter.** The
   observed-heuristics schema drops `indexing`,
   `load_eviction_policies`, `range_multi_buffers` (deliberately — the
   generator marks them "noisy"). But those knobs account for most of
   the N2 archive dispatcher's per-shape wins. A template merged with
   generic defaults is a stripped-down config.
3. **Buckets with 0 templates after filtering mean fallback fires for
   many shapes.** Only 9/12 matmul shapes matched a JSON rule with a
   template; 3 fell through to the N3b fallback. Those fallback
   shapes were fine, but the 9 "matched" shapes got the degraded
   templates.

## What we learned

- **Dispatcher format is not the bottleneck.** N5's 5-dim bucketing is
  structurally better than N2's M-only tree, but better bucketing with
  worse configs loses to worse bucketing with better configs.
- **Tuning depth is the bottleneck.** To get heldout < 0.80 on skewed
  shapes, we need configs tuned *for those skewed shapes*, not
  random-sample winners. That's N6.
- **Expansion shape list is correct.** The 40-shape grid (8 per bucket
  × 5 buckets) produced the desired bucket coverage — matmul now has
  23 rules across balanced and all three skinny aspects. The data is
  fine; the tuning is shallow.

## Artifacts

- `expansion_shapes.json` — 40 shapes per kernel.
- Archive CSVs:
  `aot_pretune_data/b200/matmul/runs/20260508_expand_skewed/`,
  `aot_pretune_data/b200/fp8_gemm/runs/20260508_expand_skewed/`.
  ~1,280 rows per kernel. Keep.
- `derived_general_heuristics.json` — 59 rules, 107 templates. Keep
  for diagnostic reference; inactive in champion path.
- `runtime_observed_heuristics_b200.json` — 3 rules (strict filter).
  Keep.
- `heuristic/heuristic_matmul.py`,
  `heuristic/heuristic_fp8_gemm.py` — N5 dispatcher. Archived; not
  wired into champion path.
- `heuristics/matmul_heuristics.csv`,
  `heuristics/fp8_gemm_heuristics.csv` — live run output.
- `scores.json` — numeric scores.

## Next

- **N6**: DifferentialEvolution per-shape tuning on the 40 expansion
  shapes per kernel. Target pop=24, gens=8 (≈192 configs/shape, vs 40
  random in N5). Budget: ~16 min/shape × 80 shapes ≈ 21 GPU-hours
  (overnight). Then regenerate the observed-heuristics JSON on the
  DE-tuned archive and retry N5's dispatch logic.
- **Champion unchanged**: N3b-v2 at family heldout 0.862.
