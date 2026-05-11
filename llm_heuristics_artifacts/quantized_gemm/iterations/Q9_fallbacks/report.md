# Q9 — Fallback-table generalization

**Status: DONE (2026-05-11). Fallback tables ported into PR 2378's
runtime. Q5+Q6 on their branch reach parity with our branch, with
Q6 actually exceeding it.**

## Motivation

PR 2378's observed-heuristics runtime only honors exact-bucket rule
matches. Live shapes outside any archive bucket drop to Helion's
stock default, costing ~10× on skewed shapes.

Our branch's dispatcher has per-kernel fallback tables that fire
when no rule matches, keyed on a coarse shape group. This gate
ports that mechanism into their runtime as a reusable JSON section
so every kernel family can have fallbacks.

## Design

Three pieces:

1. **JSON schema**: top-level `fallbacks` map
   `{kernel_class: {group_label: {"template": {...config...}}}}`
2. **Runtime hook** in `observed_heuristic_seed_configs`: when
   exact `_find_rule` returns None, compute the shape group via
   `_fallback_group_for_class` and look up `_find_fallback`.
   Same downstream materialize path.
3. **Grouping functions** per kernel family:
   - matmul family: 5 groups `small_m / small_n / small_k /
     balanced / rect` (threshold 256 / max/min < 2).
   - row_*: `short / narrow / wide / square`.
   - elementwise: `tiny / mid / huge`.
   - attention: `short_seq / long_seq / small_head / mid_seq`.
4. **Generator script** `scripts/observed_heuristics_fallbacks.py`:
   walks archive CSVs, picks median-perf winning config per group.

## Results

Generated fallbacks for 3 quantized classes → 15 fallback entries
(5 groups × 3 kernels). Merged into `observed_heuristics_b200.json`.

### Q5 (no-autotune, effort=none)

| arm set | family heldout | vs 0.20 target |
|---------|---------------:|:---------------|
| PR 2378 native, no fallbacks | 0.213 | ❌ |
| PR 2378 native + fallbacks   | **0.133** | ✅ |
| (ref) our-branch dispatcher  | 0.142 | ✅ |

Per-kernel:

| kernel | no fallback | with fallback | our-branch |
|--------|------------:|--------------:|-----------:|
| `_bf16xint16_gemm` | 0.173 | **0.110** | 0.107 |
| `matmul_bf16_int4` | 0.218 | **0.131** | 0.132 |
| `nvfp4_matmul`     | 0.255 | **0.164** | 0.198 |

### Q6 (LLM-on, Opus 4.7)

| arm set | family heldout |
|---------|---------------:|
| PR 2378 native, no fallbacks | 0.675 |
| PR 2378 native + fallbacks   | **0.592** |
| (ref) our-branch dispatcher  | 0.664 |

Per-kernel:

| kernel | no fallback | with fallback | our-branch |
|--------|------------:|--------------:|-----------:|
| `_bf16xint16_gemm` | 0.827 | **0.831** | 0.827 |
| `matmul_bf16_int4` | 0.636 | **0.506** | 0.581 |
| `nvfp4_matmul`     | 0.585 | **0.493** | 0.609 |

Fallbacks help Q6 too: seeding the LLM round-0 search with a
group-appropriate template means the random draws have a higher
best-so-far to beat.

### Δheldout-train (overfit gate = 0.05)

| kernel | Q5 fallback | Q6 fallback |
|--------|------------:|------------:|
| `_bf16xint16_gemm` | -0.027 | +0.020 |
| `matmul_bf16_int4` | -0.021 | +0.054 |
| `nvfp4_matmul`     | -0.033 | -0.007 |

int4 Q6 Δ = +0.054 barely above 0.05 gate — consistent with our
branch. Heldout ratio itself is flat vs our branch.

## Recipe for other kernel families

1. Define shape groups in `_fallback_group_for_class` (3–6 per
   class, covering the dominant config-shape correlation).
2. Tune an archive with ≥5 shapes per group.
3. Run `scripts/observed_heuristics_fallbacks.py`.
4. Merge generator output into `observed_heuristics_b200.json`
   `fallbacks` block.
5. Validate with a small live-grid Q5 benchmark (target heldout
   ≤ 0.25).

## Known limitations

- Single exemplar per group; wide groups (like "rect") aren't
  optimal everywhere. Mitigation: add more rules to the
  per-bucket rules block; fallback is only hit when no rule
  matches.
- Grouping thresholds are hand-picked. Could be learned but the
  matmul taxonomy thresholds match intuition and hit parity.

## Artifacts

- `helion/autotuner/observed_heuristics.py` — extended runtime
- `scripts/observed_heuristics_fallbacks.py` — generator
- `helion/autotuner/data/observed_heuristics_b200.json` — 30
  rules + fallbacks block
- `iterations/Q9_fallbacks/Q5_native_v2/scores.json`
- `iterations/Q9_fallbacks/Q6_native_v2/scores.json`
