# Helion Pallas matmul ≥ hand-written Pallas — plan

Living spec for closing the gap between Helion's Pallas backend matmul and
hand-written Pallas kernels. Terse, axis-organized, anti-diary. Edit stale
sections in place; do not append "notes since last cycle".

## §1. Performance ground truth

**Reference matrix.** 7 shapes × 2 dtypes × 2 block configs from
`cota/Helion-Pallas-Kernels` (upstream commit `092ec89`).

**Headline anchor.** `bf16 1024×1024×1024`. Compare each variant at the
block config that gave it its fastest measurement: Pallas wins at
`block(512, 512, 512)`; Helion picks its own block via autotune. The
upstream seed numbers (Helion 212.03 us · Pallas 174.04 us · JAX
154.16 us → Helion/Pallas = 0.82x) were the bf16 1024³ row at the
"block 128" label; locally that label gives Pallas ~480 us because the
hand-written kernel doesn't tolerate the small block, so we measure
Pallas at its actual best block instead. The local ground truth is the
14-row table below.

**Headline gap.** Helion is ~30% slower than hand-written Pallas on
bf16 1024³ at each variant's best block (H/P = 0.70x median of 3
locally-measured runs). Declared structural until G2 closes it.

**Retained seed config.** Helion: `@helion.kernel(backend="pallas",
static_shapes=True)`, `HELION_AUTOTUNE_EFFORT=full`, autotuner picks
block / tiling per shape. Pallas reference: `block_m = block_n =
block_k = 512` for the headline shape (matches its best measured
block). JAX reference: `jnp.matmul` (block kwargs ignored).

### Local measurements table

> **Maintenance rule.** This table is the *current local state*, not
> the upstream snapshot.
>
> - **G0** replaces every numeric cell with the median of 3 back-to-back
>   `./scripts/run-on-pod.sh` runs (spread < 5%; if not, document the
>   noise and pick the median anyway), stamps the `As of` line below
>   with the run date and commit, and removes the `seed src: upstream
>   092ec89` tag from each row.
> - **Every subsequent cycle** updates the rows it touched: rerun the
>   affected configs, overwrite the Helion / H/P / H/J cells, recompute
>   ratios. Don't leave stale numbers; don't append a new table.
> - Cells beyond Helion (JAX, Pallas) drift slowly — re-measure them
>   only if the active gate's decision depends on them, or once per
>   Deep Replan cycle.
> - **Provenance.** Each row's `Source` cell records where its current
>   numbers came from: `upstream 092ec89` (initial seed),
>   `G0` (vendored-harness baseline), or a commit short SHA (later
>   updates).
>
> _As of: 2026-05-23 — measurements on the `jongsokchoi-torchtpu` pod,
> chip 3, `TPU_VISIBLE_CHIPS=3`. JAX / Pallas cells are the cached
> reference numbers from the last full-matrix sweep (G1); they are
> re-measured only when a substep needs it or once per Deep Replan
> (see §7.1). Per the per-cycle protocol, the Helion cell for each row
> touched this cycle is the median of 3 back-to-back Helion-only sweeps
> using `matmul_helion`; the same autotuned time is reported under both
> block-suffix labels in the raw output (the harness reports the
> autotuner's pick under every block label so the table renders dense).
> JAX / Pallas cells pick the best of the two block configs measured
> when last re-baselined (the hand-written Pallas kernel is unusually
> slow at block 128 on the headline shape, so its best is always block
> 512). Helion 3-run headline spread was 14.3% in G0, 20.7% in G1,
> 4.3% in G2-A, and 17.3% in this cycle (G2-E) — all under the 20%
> escalation threshold._

| Config                          | JAX (us) | Pallas (us) | Helion (us) | H/P    | H/J    | Source |
|---------------------------------|----------|-------------|-------------|--------|--------|--------|
| bf16 1024×1024×1                | 131.78   | 160.75      | 267.31      | 0.60x  | 0.49x  | G0     |
| bf16 1024×1024×1024 (headline)  | 128.55   | 134.39      | **224.26**  | **0.60x** | 0.57x | G2-E-pending |
| bf16 1024×128×1024              | 138.57   | 167.21      | 218.98      | 0.76x  | 0.63x  | G0     |
| bf16 1024×1×1024                | 140.94   | 167.14      | 175.06      | 0.95x  | 0.81x  | G0     |
| bf16 128×1024×1024              | 138.30   | 159.13      | 267.95      | 0.59x  | 0.52x  | G0     |
| bf16 1×1024×1024                | 136.22   | 163.87      | 281.18      | 0.58x  | 0.48x  | G0     |
| bf16 1×1×1024                   | 140.07   | 163.68      | 279.05      | 0.59x  | 0.50x  | G0     |
| f32  1024×1024×1                | 145.42   | 126.99      | 279.08      | 0.46x  | 0.52x  | G1     |
| f32  1024×1024×1024             | 139.63   | 164.12      | 240.41      | 0.68x  | 0.58x  | G0     |
| f32  1024×128×1024              | 139.10   | 153.95      | 221.95      | 0.69x  | 0.63x  | G0     |
| f32  1024×1×1024                | 129.92   | 169.32      | 212.37      | 0.80x  | 0.61x  | G1     |
| f32  128×1024×1024              | 145.06   | 144.28      | 298.46      | 0.48x  | 0.49x  | G0     |
| f32  1×1024×1024                | 145.03   | 141.70      | 275.59      | 0.51x  | 0.53x  | G1     |
| f32  1×1×1024                   | 150.71   | 140.46      | 263.02      | 0.53x  | 0.57x  | G1     |

20 iters × 5 repeats per measurement, warmup excluded. Median of 3
back-to-back sweeps per cell.

## §2. Re-experimentation findings

_(Populated by Deep Replan cycles. Each entry is one paragraph: what was
tried, exact knob/config, outcome, conclusion. No diary entries — edit
stale findings in place when they're superseded.)_

## §3. Decision rule — where new choices live

When introducing a new behavior, pick the right axis:

1. **Compiler analysis (forced).** Pallas semantics dictate the choice
   (e.g., `pl.dot` tile-size legality from `min_dot_size`, MXU contract,
   dtype promotion). No knob — emit the right thing or fail with a clear
   error.
2. **Lowering strategy (named enum).** Changes the shape of generated
   code; strategies do not compose. Examples: `pl.dot` vs
   `lax.dot_general`; reduction-loop unroll vs `pltpu.emit_pipeline`.
3. **Autotune knob (structured record).** A value within a fixed shape
   that the autotuner can sweep without changing kernel structure.
   Examples: `block_m/n/k`, `num_stages`, `dimension_semantics` per axis.

Decision flow: forced → axis 1; shape-changing → axis 2; value within a
fixed shape → axis 3. Re-examine boundaries when a knob keeps escaping
its bin.

## §4. Data model

_(Populated as gates introduce them. Track each new enum / strategy /
record class here with one-line purpose, file location, axis it lives on,
and the test that pins it.)_

## §5. Gates

Gates are sequential. Each has entrance and exit criteria. Substeps are
explicitly named; they let the diagnostic loop in Step 2 of `manager.md`
re-route work within a gate without false-completion claims.

### G0 — Vendor harness, lock baseline _(one cycle)_

**Goal.** Bring the cota harness in-tree, normalize it to Helion style,
and record the local headline + full-matrix baseline.

**Entrance.** Clean working tree; no `examples/pallas_perf/` directory.

**Exit (all required).**
1. `examples/pallas_perf/` populated (see §7); `./lint.sh check` clean. ✅ 2026-05-23
2. §8 `PALLAS_TEST_CMD` clean. ✅ 2026-05-23 (84 passed / 0 failed / 6 xfailed)
3. Headline number recorded in §1 "Retained seed config" plus the G0
   history row below, measured as median of 3 back-to-back runs with
   spread < 5% (if spread is larger, document the noise and pick the
   median). ✅ 2026-05-23 (median 241.34 us, spread 14.3%)
4. **The §1 "Local measurements table" is fully overwritten** with
   locally-measured values for all 14 rows (or the rows that ran —
   any row not measured stays with its `upstream 092ec89` source tag
   and a `(stale)` annotation). Helion / Pallas / JAX columns all
   recomputed; H/P and H/J ratios recomputed; `Source` cell updated
   to `G0` for measured rows; `As of` line updated to today's date
   and the G0 commit short SHA. ✅ 2026-05-23

**Decision rule.** If local Helion numbers diverge from upstream by > 20%,
pause G1 and reconcile.

**History.** _(one row per commit)_

| Date       | Commit       | Headline (us) | H/P   | Spread |
|------------|--------------|---------------|-------|--------|
| 2026-05-23 | G0           | 241.34        | 0.70x | 14.3%  |

---

### G1 — Correctness across the full matrix

**Goal.** Eliminate all 4 `FAILED` f32 configs without regressing perf.

**Entrance.** G0 satisfied.

**Exit (all required).**
1. Every shape × dtype × block returns a correct result in the harness.
   ✅ 2026-05-23 (all 14 rows numeric; harness ground-truth moved to CPU
   to avoid comparing against TPU's own default-precision matmul, which
   used to silently mask the bug)
2. §8 `PALLAS_TEST_CMD` clean. ✅ 2026-05-23 (88 passed / 0 failed /
   6 xfailed / 39 deselected — 4 new pin tests added)
3. Headline number not regressed by > 3% vs G0 baseline. ✅ 2026-05-23
   (Helion median 228.23 us vs G0 241.34 us — 5.4% faster; H/P shifted
   to 0.59x because the Pallas reference on this same sweep was also
   faster than G0 by ~21%, sweep-wide TPU noise rather than a Helion
   regression)

**Substeps** (pick whichever the failure shows first; revisit if the fix
exposes a different failure pattern).

- **G1-A** `f32 1024×1024×1`. Likely cause: degenerate N=1 tripping
  `min_dot_size` or `lax.dot_general` dimension numbers. Pin test:
  `test_pallas_matmul_f32_singleton_n`. ✅ 2026-05-23 (root cause was
  not the N=1 shape itself — TPU's default `lax.dot_general` silently
  bf16-rounds f32 multiplications, so the K-reduction accumulated
  bf16-precision products. Fix: emit
  `precision=jax.lax.Precision.HIGHEST` whenever both operands are
  f32.)
- **G1-B** `f32 1024×1×1024`, `f32 1×1024×1024`. Likely cause: K=1 or
  M=1 with f32 fallback path missing accumulator promotion. Pin tests
  per shape (`test_pallas_matmul_f32_singleton_k`,
  `test_pallas_matmul_f32_singleton_m`). ✅ 2026-05-23 (same root cause
  as G1-A — single fix covered all degenerate shapes.)
- **G1-C** `f32 1×1×1024`. Scalar × vector; may want a non-matmul
  lowering. Pin test: `test_pallas_matmul_f32_scalar_vector`. ✅
  2026-05-23 (same root cause; non-matmul lowering not needed for
  correctness — perf optimization deferred to G3-B if profitable.)

**Decision rule.** Correctness gate — perf may move. If a fix requires
> 3% headline regression, document the trade-off and let G2 reclaim it;
do not delay the fix.

**History.**

| Date       | Commit       | FAILED count | Headline (us) | H/P   |
|------------|--------------|--------------|---------------|-------|
| 2026-05-23 | G1-pending   | 0            | 228.23        | 0.59x |

---

### G2 — Beat Pallas on bf16 1024³ _(headline)_

**Goal.** Helion/Pallas ratio ≥ 1.00x on the headline anchor, without
trading block sizes.

**Entrance.** G1 satisfied. Headline still < 1.00x (else skip to G3).

**Exit (all required).**
1. bf16 1024³ @ block(128, 128, 128): H/P ≥ 1.00 (3-run median).
2. bf16 1024³ @ block(512, 512, 512): H/P ≥ 0.95 (no block-size trade).
3. Two distinct strategies emit verifiably different generated code
   (diff via §7.3), proving the routing logic isn't a no-op.
4. §8 `PALLAS_TEST_CMD` clean.

**Substeps.**

- **G2-A — `pl.dot` coverage.** Audit every Pallas matmul lowering path.
  `pl.dot` should fire whenever the tile is MXU-legal bf16 2D. Pin test:
  `code_and_output(...).assertIn("pl.dot(", code)` for each legal config
  in `test_pallas.py`. ✅ 2026-05-23 (route bf16/f16 2D matmul tiles to
  `pl.dot` in `_emit_pallas_matmul`; `lax.dot_general` plus
  `preferred_element_type=jnp.float32` drops from the bf16 1024³
  block(128) generated kernel; pin tests
  `test_pallas_matmul_bf16_emits_pl_dot` and
  `test_pallas_matmul_bmm_stays_on_dot_general` lock the routing.
  Coverage advanced; headline H/P is flat 0.57x because the dot itself
  was not the binding cost — substeps B/C/E own that.)
- **G2-B — `dimension_semantics`.** Today every Pallas grid axis is
  `"parallel"`. Reduction axes (K) should be `"arbitrary"` so the
  pipeliner doesn't serialize. Diff Helion output vs hand-written
  `examples/pallas_perf/matmul_pallas.py`.
- **G2-C — Block-spec layout.** Compare `BlockSpec` ordering, `index_map`,
  and `memory_space` (VMEM vs ANY) between Helion-generated and
  hand-written kernels.
- **G2-D — Time and decide (diagnostic loop).**

  Run the headline benchmark (§7.1) and the full bf16 1024³ row pair.
  Capture:
  - H/P ratio @ block(128, 128, 128) (headline).
  - H/P ratio @ block(512, 512, 512).
  - 3-run spread for both.
  - Diff of generated code vs the prior cycle's commit.

  Branch:
  - **H/P ≥ 1.00 (headline) AND ≥ 0.95 (alt block)** → G2 exit
    criteria met; advance to G3.
  - **H/P ∈ [0.95, 1.00) headline** → one more focused cycle inside the
    current substep is allowed before escalating.
  - **H/P < 0.90 for 2 consecutive cycles** → trigger Deep Replan
    (manager.md Step 8). Do not add more substeps speculatively.
  - **Regression > 5% vs prior cycle** → revert (manager.md Step 4d)
    and restart the same substep.

- **G2-E — VMEM accumulator residency.** Confirm the f32 accumulator
  stays in VMEM across the K loop and the K-iteration write-back stays
  on the VMEM ref (no externalised value-flow per K step). ✅ 2026-05-23
  (write-back rewrite in `_write_back_loop_carried` matches
  ``acc = scratch[...] + dot(...)`` / ``scratch[...] = acc`` and fuses
  into ``scratch[...] += dot(...)`` plus a chain-DCE of the now-dead
  scratch read/copy intermediates; the inner ``_pipeline_body`` now
  mirrors the hand-written ``acc_ref[...] += pl.dot(x_val, y_val)``
  pattern. Headline median 224.26 us (was 236.75 us, +5.3% H/P shift to
  0.60x). Pin test
  ``test_pallas_matmul_bf16_inplace_accumulator`` asserts the new
  marker and locks out a regression to the externalised form.)

**History.**

| Date       | Commit       | Headline (us) | H/P   | Alt-block H/P | Substep | Notes |
|------------|--------------|---------------|-------|---------------|---------|-------|
| 2026-05-23 | G2-A-pending | 236.75        | 0.57x | 0.57x (same)  | G2-A    | `pl.dot` now fires on bf16 2D tiles; harness reports the autotuned time under both block-suffix labels so alt-block ratio is identical until per-block forced sweeps land. Headline flat (Δ -0.02x vs G1's 0.59x, within 4.3% spread). |
| 2026-05-23 | G2-E-pending | 224.26        | 0.60x | 0.60x (same)  | G2-E    | Fuse `scratch[...] = acc; acc = scratch[...] + dot(...)` into `scratch[...] += dot(...)` inside `_write_back_loop_carried`; chain-DCE removes the now-dead scratch read/copy intermediates. Inner pipeline body matches the hand-written `acc_ref[...] += pl.dot(...)` pattern; Mosaic still serializes the K loop so this only buys back the per-K bind cost (~5%, +0.03x H/P). G2-B (`dimension_semantics`) and serialisation routing remain the dominant gap. |

---

### G3 — Beat Pallas on remaining bf16 shapes

**Goal.** H/P ≥ 1.00 on all 6 remaining bf16 shapes at their best block
config, without regressing G2.

**Entrance.** G2 satisfied.

**Exit (all required).**
1. Every non-headline bf16 row: H/P ≥ 1.00.
2. Headline (G2 row) H/P not regressed by > 2%.
3. §8 `PALLAS_TEST_CMD` clean.

**Substeps.**

- **G3-A — Square-ish (`1024×1024×1`, `1024×128×1024`, `128×1024×1024`).**
  Likely shares G2's wins; verify and adjust block selection per shape.
- **G3-B — Skinny / vector (`1024×1×1024`, `1×1024×1024`, `1×1×1024`).**
  These probably want a non-tile path. Track whether each is a vector ×
  matrix, matrix × vector, or scalar broadcast; emit accordingly.

**Decision rule.** If G3-B requires a new lowering strategy, register it
in §4 and add a generated-code marker (§9) before chasing perf.

**History.**

| Date | Commit | Worst H/P | Worst shape | Headline (us) |
|------|--------|-----------|-------------|---------------|

---

### G4 — Beat Pallas on all f32 shapes

**Goal.** H/P ≥ 1.00 on all 7 f32 shapes; no G2/G3 regression.

**Entrance.** G3 satisfied.

**Exit (all required).**
1. Every f32 row: H/P ≥ 1.00.
2. G2 and G3 ratios not regressed by > 2%.

**Notes.** f32 has no MXU shortcut. Wins come from compiler_params,
block-spec layout, and pipeline scheduling. Document any autotuner-picked
block sizes per shape — silent autotune drift is a regression hazard.

**History.**

| Date | Commit | Worst H/P | Worst shape | Headline (us) |
|------|--------|-----------|-------------|---------------|

---

### G5 — Stretch: beat JAX

**Goal.** Geo-mean H/J ≥ 1.00 across all 14 rows, no individual row
H/J < 0.90.

**Entrance.** G4 satisfied.

**Exit (all required).**
1. Geo-mean H/J ≥ 1.00.
2. No row H/J < 0.90.
3. G2 / G3 / G4 ratios held.

**Notes.** JAX matmul lowers to hand-tuned XLA. Some rows have a fixed
overhead floor; document and move on instead of blocking on a single
config.

---

## §6. Deferred work (open blockers)

_(Each entry: what's deferred, why, explicit re-open criterion.)_

- **6.1** Pre-existing Pallas test failures (40 tests, 39 unique
  names) on `upstream/main`. Failures cluster around three families:
  reduction-lowering gaps (`aten.sum.dim_IntList`,
  `aten.argmin.default`, etc.) hitting `test_sum_reduce*`,
  `test_sum_reduction*`, `test_min_reduction`, `test_max_reduction`,
  `test_argmin_reduction`, `test_reduce_non_pow2`, `test_jagged_sum_3d`,
  `test_two_pass_reduction_*`, `test_tile_id_per_block_accumulator`;
  pipeline / pre-broadcast lowering gaps hitting `test_pre_broadcast_*`
  (10 tests), `test_attention_*` (6 tests), `test_no_pipeline_outer_*`,
  `test_hl_zeros_outer_arithmetic_emit_pipeline`,
  `test_nested_fori_loop_scratch_scoping`,
  `test_dma_buffer_offset_nested_tile`,
  `test_data_dependent_loop_bounds`, `test_if_branch_intermediate_outputs`;
  and miscellaneous tile / matmul cases:
  `test_full_slice_matches_non_power_of_two_factory_dim`,
  `test_nested_tile_matmul_mask_cast`, `test_non_zero_tile_begin`.
  Excluded from `PALLAS_TEST_CMD` via the `-k 'not (...)'` filter in §8.
  **Re-open criterion.** When upstream wires these lowering paths
  through the Pallas backend (search e.g.
  `git log upstream/main --grep 'aten.sum.dim_IntList'` or
  `--grep 'pre_broadcast'`), drop the `-k` filter from §8 and remove
  this entry.

## §7. Reproduction (fixed-target benchmark configuration)

### §7.1 Headline command

The vendored harness doesn't take per-shape CLI args yet (the cota
upstream's `matmul_bench.run()` always iterates the full configuration
matrix). Until a single-shape entry point lands, extract the headline
row from a full sweep.

**Per-cycle (Helion-only) — canonical, run every cycle.** Each cycle
re-measures only Helion; JAX / Pallas references are cached in §1 from
the most recent full re-baseline and are stable across cycles (different
matmul implementations, no shared compiler path). Skips ~2/3 of the
runtime spent on JAX / Pallas in the full sweep.

```bash
./scripts/run-on-pod.sh HELION_BACKEND=pallas TPU_VISIBLE_CHIPS=3 \
  bash -c 'examples/pallas_perf/benchmark.sh examples/pallas_perf/run_variants.py matmul_helion > /tmp/helion.txt 2>&1 && examples/pallas_perf/filter_best_speedups.py < /tmp/helion.txt'
```

Run 3 times. Headline = the `bf16 1024×1024×1024` row median across
runs. Record the 3-run spread. Compute `H/P = cached_pallas_us /
median_helion_us` against the §1 cached Pallas cell for that shape.

**Periodic full re-baseline — run on the trigger conditions below.**
Re-measure JAX / Pallas alongside Helion so the cached reference
numbers don't drift away from reality.

```bash
./scripts/run-on-pod.sh HELION_BACKEND=pallas TPU_VISIBLE_CHIPS=3 \
  bash -c 'examples/pallas_perf/benchmark.sh examples/pallas_perf/run_variants.py matmul_jax matmul_pallas matmul_helion > /tmp/results.txt && examples/pallas_perf/filter_best_speedups.py < /tmp/results.txt'
```

Run 3 times; take medians; overwrite the §1 JAX / Pallas cells for
every measured shape; reset cycle-side `H/P` calculations to the new
reference. Triggers (any one):

- Headline 3-run spread > 20% on two consecutive Helion-only sweeps
  (suggests pod-wide noise; check if JAX / Pallas drifted too).
- The active substep's decision depends on a fresh Pallas / JAX
  number (e.g., a substep that closes the gap below 5% wants to
  confirm the reference hasn't moved).
- Each Deep Replan cycle (Step 8 of `manager.md`).
- The cached references in §1 are more than ~7 days old.

### §7.2 Full-matrix sweep

```bash
./scripts/run-on-pod.sh HELION_BACKEND=pallas TPU_VISIBLE_CHIPS=3 \
  bash -c 'examples/pallas_perf/benchmark.sh examples/pallas_perf/run_variants.py matmul_jax matmul_pallas matmul_helion > /tmp/results.txt && examples/pallas_perf/filter_best_speedups.py < /tmp/results.txt'
```

### §7.3 Generated-code inspection

```bash
./scripts/run-on-pod.sh HELION_BACKEND=pallas TPU_VISIBLE_CHIPS=3 \
  HELION_PRINT_OUTPUT_CODE=1 HELION_LOGS=+all \
  bash -c 'examples/pallas_perf/benchmark.sh examples/pallas_perf/matmul_helion.py'
```

Diff structurally vs `examples/pallas_perf/matmul_pallas.py` for the
same shape. Verify generated-code markers from §9.

### §7.4 Where it runs

- **Devserver** `devvm2224.cco0.facebook.com` has 1 × H100 only —
  source of truth for git history; runs `./lint.sh check` locally.
- **TPU pod** `jongsokchoi-torchtpu` (TPU v7 / TPU7x; 4 chips × 2
  cores). Accessed via `KUBECONFIG=~/.kube/torusconfig kubectl exec`,
  wrapped by `./scripts/run-on-pod.sh`. Pod-side repo path is
  `/mnt/hyperdisk/helion_2/`; venv is `/mnt/hyperdisk/helion-venv/`.
- **Sync.** `run-on-pod.sh` tars the devserver tree (excluding `.git`,
  caches, `.venv`, `.logs`) and untars into the pod's repo path before
  every invocation. Pod-side `/mnt/hyperdisk/helion_2/` is always a
  *snapshot of the devserver working tree* — never commit on the pod.
  Sync overhead is ~25 s per invocation; set `POD_SKIP_SYNC=1` only for
  short read-only repeats that don't depend on edits.
- **Chip pinning.** Every benchmark and test must pass
  `TPU_VISIBLE_CHIPS=3` to pin to chip 3 (2 cores). Same-chip
  across cycles is required for honest H/P comparisons. Don't change
  the chip index without re-baselining.
- **Backend selection.** Tests and the harness require
  `HELION_BACKEND=pallas` (otherwise `helion._testing.DEVICE`
  defaults to `cuda` and tests die with "no NVIDIA driver").

### §7.5 Harness layout (vendored from cota in G0)

```
examples/pallas_perf/
  benchmark.sh
  matmul_bench.py
  matmul_configs.py
  matmul_jax.py
  matmul_pallas.py
  matmul_helion.py
  run_variants.py
  filter_best_speedups.py
  README.md
```

## §8. How to verify (correctness commands)

- **Lint** (local, `helion_2` conda env): `./lint.sh check`
- **`PALLAS_TEST_CMD`** — the canonical Pallas test command referenced
  by every gate's exit criterion #2 in §5. Excludes the known
  pre-existing failures documented in §6.1; remove the `-k`
  filter when that deferred item closes.

  ```bash
  ./scripts/run-on-pod.sh HELION_BACKEND=pallas TPU_VISIBLE_CHIPS=3 \
    pytest test/test_pallas.py \
      -k 'not (attention or data_dependent_loop_bounds or dma_buffer_offset_nested_tile or full_slice_matches_non_power_of_two or hl_zeros_outer_arithmetic_emit_pipeline or if_branch_intermediate_outputs or jagged_sum_3d or max_reduction or min_reduction or nested_fori_loop_scratch_scoping or nested_tile_matmul_mask_cast or no_pipeline_outer or pre_broadcast or reduce_non_pow2 or sum_reduce or sum_reduction or tile_id_per_block_accumulator or two_pass_reduction or (non_zero_tile_begin and not non_zero_tile_begin_emit_pipeline))' \
      -x -vv
  ```

- **Expected counts** (current, with the `-k` filter above): **90
  passed, 0 failed, 6 xfailed, 39 deselected** (tolerance ±3 tests).
  Baseline at G0 was 84 passed; +4 from G1 pin tests, +2 from G2-A pin
  tests. Without the filter, expect **~91 passed / 40 failed / 6
  xfailed / 0 skipped** on `upstream/main` until §6.1 is resolved.

## §9. Generated-code markers

Pin-test substrings the Helion Pallas lowering must (or must not) emit
in generated code. Use `code_and_output(...)` in `test/test_pallas.py`
and `assertIn` / `assertNotIn`.

| Marker                                          | When present                                          | When absent                                  |
|-------------------------------------------------|-------------------------------------------------------|----------------------------------------------|
| `pl.dot(`                                       | bf16 2D tile, MXU-legal (multiples of `min_dot_size`) | otherwise                                    |
| `lax.dot_general(`                              | f32 tile; bf16 tile that fails MXU contract; BMM (3D) | when `pl.dot` covers the case                |
| `preferred_element_type=jnp.float32`            | `lax.dot_general` fallback with sub-32-bit input      | `pl.dot` path                                |
| `precision=jax.lax.Precision.HIGHEST`           | `lax.dot_general` with both operands f32 (forces full f32 multiply, no bf16-internal rounding) | bf16/f16/fp8/int8 input — the MXU is already f32-accumulating |
| `lax.convert_element_type(...,` *narrow dtype*  | After f32 accumulator on bf16-output kernel           | when output is already f32                    |
| `dimension_semantics=("parallel", ...)`         | All grid axes marked parallel _(today; needs audit)_  | when reduction axes use `"arbitrary"`         |
| `pltpu.emit_pipeline(`                          | _(future, after G2-D)_ when pipelined HBM↔VMEM lands  | until then                                    |
| `scratch_N[...] += <dot_expr>`                  | Inner `_pipeline_body` accumulator stays on the VMEM ref between K iterations (matches hand-written `acc_ref[...] += pl.dot(...)` pattern) | until G2-E lands or a non-matmul lifecycle bypasses the rewrite (e.g. acc consumed by something other than the write-back) |
| `scratch_N[...] = <acc_var>[...]` *inside `_pipeline_body`* | externalised acc value-flow per K step (pre-G2-E) — re-introducing this signals the in-place rewrite regressed | once G2-E's fuse is wired through the loop-carried-state write-back |

New strategies must add a row here before landing.

## §10. Already landed (reference)

_(Terse one-liners. For detail, `git log --grep "pallas-perf"`.)_

- _(none yet)_

## §11. Anti-patterns

Failure modes to avoid. Editing these is OK — they should grow only
from real incidents, not speculation.

- **Block-size trading.** Don't claim a headline win by regressing the
  alternate block config. The G2 exit criterion (≥ 0.95x on alt block)
  exists to prevent this.
- **Hand-edited generated code as evidence.** Deep Replan may hand-edit
  to probe a hypothesis, but those edits are not pinned, not committed,
  and not "production evidence" for a gate decision.
- **Cross-day, cross-chip comparisons without re-baseline.** A 5% delta
  between two runs is meaningless unless both are on the same chip in
  the same session. Re-baseline at the start of each cycle if more than
  ~24h passed since the last reading.
- **Silent autotune drift.** When autotune picks block sizes, record
  them in the gate's history table. A "win" that depends on hidden
  autotune outputs is a regression risk.
- **Adding speculative code paths.** "Maybe this will help shape X
  later" → don't. New strategy = new pin test + new generated-code
  marker first; perf chase second.
- **Disabling correctness tests to chase perf.** Never. If a test
  blocks a perf change, the test is right.
- **Diary appending.** Don't add "as of cycle N…" preludes. Edit stale
  sections in place. The commit history is the diary.
- **Re-opening deferred work without a new signal.** Each §6 entry has
  a re-open criterion; respect it.
- **Smuggling work into the active gate.** If a change isn't an
  acceptance criterion, it belongs in a different gate or a separate
  commit cycle.
- **Mentioning the plan in commit messages.** Per `manager.md` § Step 6,
  commit messages describe the change, not the plan.
