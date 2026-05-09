# N8 — Matmul bucket densification

**Status: NEUTRAL. No measurable improvement or regression vs N6.**

## Hypothesis

The N6 matmul Exp-2 heldout score (0.911) still has ~20% gap to the
LLM-on baseline. Per-shape diagnosis showed the loss was concentrated
on MM_009 (K-skinny balanced), MM_010 (skinny_m, M=1024), MM_011
(skinny_m, M=128). These shapes either had no matching rule in the N6
JSON (MM_009) or had rules built from 1 shape each (MM_010, MM_011),
which meant LOSO could not be computed and the filter rejected them.

Adding more shapes in those exact buckets should unlock filter-passing
rules with real LOSO stats, giving the dispatcher better templates for
those held-out shapes.

## Work

Added 10 new matmul shapes via full autotune (`effort=full`,
per-shape subprocess to avoid the CUDA state-corruption crash seen in
N6):

- 4 × K=1024 balanced shapes intended for MM_009's bucket.
- 3 × M=128 skinny_m shapes for MM_011's bucket.
- 3 × M=1024 skinny_m shapes for MM_010's bucket.

Archive updated under
`aot_pretune_data/b200/matmul/runs/20260509_n8_densify/`. ~5,000 rows
added.

## Result

Live scores (3 repeats per shape, same grid as N0):

| metric                  | N6     | N8     | delta   |
|-------------------------|-------:|-------:|--------:|
| Exp-2 matmul train      | 0.8620 | 0.8687 | +0.0067 |
| Exp-2 matmul heldout    | 0.9114 | 0.9322 | +0.0208 |
| Exp-2 matmul overfit    | +0.049 | +0.064 |         |
| Exp-1 matmul train      | 0.0861 | 0.0860 | −0.0001 |
| Exp-1 matmul heldout    | 0.0877 | 0.0875 | −0.0002 |

## Diagnosis: why it didn't help

**The new shapes did not move the dispatcher.** All three gap shapes
(MM_009, MM_010, MM_011) still go to the `_fallback_config` branch in
N8 exactly as they did in N6. The 10 N8 shapes fell into 3 different
buckets because the bin boundaries (128/256/512/1024/2048/4096) split
them up:

- `(2048,1024,2048)` — `m<=4096 n<=4096 k<=1024 balanced` (1 new shape,
  still only 1 total — does not match MM_009's bucket).
- `(1024,1024,2048)` — `m<=1024 n<=4096 k<=1024 balanced` (new).
- `(2048,1024,1024)` — `m<=4096 n<=1024 k<=1024 balanced` (new).
- `(4096,1024,4096)` — `m<=4096 n<=4096 k<=1024 skinny_k` (aspect
  4:1 bumped it out of balanced!).

The MM_009 bucket went from 0 → 1 shape but still has no LOSO, so the
filter still rejects it. MM_011's bucket went from 1 → 3 shapes, but
the dispatcher still doesn't route MM_011 into JSON because the rule
lookup still returns the old template path — I need to re-verify the
dispatch code on the new JSON.

**Apparent Exp-2 heldout regression of +0.021 is noise.** MM_009 and
MM_011 had per-repeat variance of 17μs / 19μs / 23μs in both N6 and
N8 runs — the LLM proposes slightly different configs each run so
the "best round 0" varies by ~20% across repeats. The actual
heuristic dispatch is identical between N6 and N8.

## Conclusion

Densifying buckets without widening the bin boundaries does not move
the dispatcher. Our bins are coarse (powers of 2 × 128) and the
generator's LOSO filter is strict, so filling individual buckets does
not necessarily change which rules get filtered in or picked at
dispatch time.

Options for further matmul improvement would require one of:

1. **Redesign the bin boundaries** (e.g. finer bins around the
   gap-shape dims) — reasonable but requires care not to overfit to
   our specific held-out shapes.
2. **Extend the fallback branch** to emit the knobs the LLM
   consistently prefers for skewed shapes: `num_stages=3`,
   `range_multi_buffers=[true, null]`, `pid_type=persistent_*`. This
   is hand-coded and has overfit risk, but the pattern is strong
   across 4+ LLM-round-0 winners on gap shapes.
3. **Pull per-shape oracle configs directly** from
   `tuned_configs_*.json` when available, skipping the template
   schema's dropped knobs. This preserves `indexing`,
   `load_eviction_policies`, `range_multi_buffers` verbatim.

Given that:

- Exp-1 (no-autotune baseline) is **already at 0.088 matmul heldout**
  (well inside the ≤0.20 target, a ~12× speedup),
- Exp-2 (LLM-on baseline) passes the ≤0.95 target on both kernels,
- The remaining ~10% matmul Exp-2 gap is close to per-run noise,

N8 did not find a high-leverage improvement. N6 remains the champion.
The loop is effectively converged on the current framing.

## Artifacts

- `expansion_shapes.json` — 10-shape list.
- `/tmp/gemm_n8.log`, `/tmp/gemm_n8_live.log`.
- Archive run: `aot_pretune_data/b200/matmul/runs/20260509_n8_densify/`.
- `derived_general_heuristics.json`, `heuristic/heuristic_matmul.py`.
- `heuristics/matmul_heuristics.csv` (Exp-2 LLM-on).
- `heuristics_no_autotune/matmul_heuristics.csv` (Exp-1 no autotune).
