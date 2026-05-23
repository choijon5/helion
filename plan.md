# Helion Pallas matmul ≥ hand-written Pallas — plan

Living spec for closing the gap between Helion's Pallas backend matmul and
hand-written Pallas kernels. Terse, axis-organized, anti-diary. Edit stale
sections in place; do not append "notes since last cycle".

## §1. Performance ground truth

**Reference matrix.** 7 shapes × 2 dtypes × 2 block configs from
`cota/Helion-Pallas-Kernels` (upstream commit `092ec89`).

**Headline anchor.** `bf16 1024×1024×1024 @ block(128, 128, 128)`.
The seed numbers below (Helion 212.03 us · Pallas 174.04 us · JAX
154.16 us → Helion/Pallas = 0.82x, Helion/JAX = 0.73x) come from the
upstream comparison commit. G0 replaces them with locally-measured
numbers on the `jongsokchoi-torchtpu` chip and we treat the local
numbers as ground truth thereafter.

**Headline gap.** ~22% slower than hand-written Pallas at the same shape
and block (upstream seed). Declared structural until G2 closes it.
Refresh this percentage when G0 lands.

**Retained seed config.** _(populated by G0; until then, "the upstream
default" — `block_m = block_n = block_k = 128`, `static_shapes=True`,
backend Pallas, no extra knobs.)_

### Local measurements table

> **Maintenance rule.** This table is the *current local state*, not
> the upstream snapshot.
>
> - **G0** replaces every numeric cell with the median of 3 back-to-back
>   `./scripts/run-on-tpu.sh` runs (spread < 5%; if not, document the
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
> _As of: 2026-05-22 — values are the upstream seed; G0 will overwrite._

| Config                          | JAX (us) | Pallas (us) | Helion (us) | H/P    | H/J    | Source              |
|---------------------------------|----------|-------------|-------------|--------|--------|---------------------|
| bf16 1024×1024×1                | 165.26   | 176.26      | 247.19      | 0.71x  | 0.67x  | upstream 092ec89    |
| bf16 1024×1024×1024 (headline)  | 154.16   | 174.04      | **212.03**  | **0.82x** | 0.73x | upstream 092ec89 |
| bf16 1024×128×1024              | 151.73   | 159.42      | 254.65      | 0.63x  | 0.60x  | upstream 092ec89    |
| bf16 1024×1×1024                | 153.54   | 183.95      | 215.89      | 0.85x  | 0.71x  | upstream 092ec89    |
| bf16 128×1024×1024              | 152.88   | 176.95      | 223.84      | 0.79x  | 0.68x  | upstream 092ec89    |
| bf16 1×1024×1024                | 153.85   | 164.00      | 238.75      | 0.69x  | 0.64x  | upstream 092ec89    |
| bf16 1×1×1024                   | 171.87   | 174.22      | 205.18      | 0.85x  | 0.84x  | upstream 092ec89    |
| f32  1024×1024×1                | 153.02   | 164.67      | **FAILED**  | n/a    | n/a    | upstream 092ec89    |
| f32  1024×1024×1024             | 154.75   | 155.97      | 210.92      | 0.74x  | 0.73x  | upstream 092ec89    |
| f32  1024×128×1024              | 152.84   | 158.25      | 204.81      | 0.77x  | 0.75x  | upstream 092ec89    |
| f32  1024×1×1024                | 154.27   | 171.98      | **FAILED**  | n/a    | n/a    | upstream 092ec89    |
| f32  128×1024×1024              | 152.46   | 158.01      | 237.20      | 0.67x  | 0.64x  | upstream 092ec89    |
| f32  1×1024×1024                | 153.38   | 155.22      | **FAILED**  | n/a    | n/a    | upstream 092ec89    |
| f32  1×1×1024                   | 156.04   | 157.50      | **FAILED**  | n/a    | n/a    | upstream 092ec89    |

20 iters × 5 repeats per measurement, warmup excluded.

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
1. `examples/pallas_perf/` populated (see §7); `./lint.sh check` clean.
2. `./scripts/run-on-tpu.sh pytest test/test_pallas.py -x -vv` clean.
3. Headline number recorded in §1 "Retained seed config" plus the G0
   history row below, measured as median of 3 back-to-back runs with
   spread < 5% (if spread is larger, document the noise and pick the
   median).
4. **The §1 "Local measurements table" is fully overwritten** with
   locally-measured values for all 14 rows (or the rows that ran —
   any row not measured stays with its `upstream 092ec89` source tag
   and a `(stale)` annotation). Helion / Pallas / JAX columns all
   recomputed; H/P and H/J ratios recomputed; `Source` cell updated
   to `G0` for measured rows; `As of` line updated to today's date
   and the G0 commit short SHA.

**Decision rule.** If local Helion numbers diverge from upstream by > 20%,
pause G1 and reconcile.

**History.** _(one row per commit)_

| Date | Commit | Headline (us) | H/P  | Spread |
|------|--------|---------------|------|--------|

---

### G1 — Correctness across the full matrix

**Goal.** Eliminate all 4 `FAILED` f32 configs without regressing perf.

**Entrance.** G0 satisfied.

**Exit (all required).**
1. Every shape × dtype × block returns a correct result in the harness.
2. `pytest test/test_pallas.py -x -vv` clean.
3. Headline number not regressed by > 3% vs G0 baseline.

**Substeps** (pick whichever the failure shows first; revisit if the fix
exposes a different failure pattern).

- **G1-A** `f32 1024×1024×1`. Likely cause: degenerate N=1 tripping
  `min_dot_size` or `lax.dot_general` dimension numbers. Pin test:
  `test_pallas_matmul_f32_singleton_n`.
- **G1-B** `f32 1024×1×1024`, `f32 1×1024×1024`. Likely cause: K=1 or
  M=1 with f32 fallback path missing accumulator promotion. Pin tests
  per shape.
- **G1-C** `f32 1×1×1024`. Scalar × vector; may want a non-matmul
  lowering. Pin test: `test_pallas_matmul_f32_scalar_vector`.

**Decision rule.** Correctness gate — perf may move. If a fix requires
> 3% headline regression, document the trade-off and let G2 reclaim it;
do not delay the fix.

**History.**

| Date | Commit | FAILED count | Headline (us) | H/P  |
|------|--------|--------------|---------------|------|

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
4. `pytest test/test_pallas.py -x -vv` clean.

**Substeps.**

- **G2-A — `pl.dot` coverage.** Audit every Pallas matmul lowering path.
  `pl.dot` should fire whenever the tile is MXU-legal bf16 2D. Pin test:
  `code_and_output(...).assertIn("pl.dot(", code)` for each legal config
  in `test_pallas.py`.
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
  stays in VMEM across the K loop. If Helion materializes to HBM per
  step, fix the lowering.

**History.**

| Date | Commit | Headline (us) | H/P  | Alt-block H/P | Substep | Notes |
|------|--------|---------------|------|---------------|---------|-------|

---

### G3 — Beat Pallas on remaining bf16 shapes

**Goal.** H/P ≥ 1.00 on all 6 remaining bf16 shapes at their best block
config, without regressing G2.

**Entrance.** G2 satisfied.

**Exit (all required).**
1. Every non-headline bf16 row: H/P ≥ 1.00.
2. Headline (G2 row) H/P not regressed by > 2%.
3. `pytest test/test_pallas.py -x -vv` clean.

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

- **6.1** _(none yet)_

## §7. Reproduction (fixed-target benchmark configuration)

### §7.1 Headline command

```bash
./scripts/run-on-tpu.sh \
  python examples/pallas_perf/matmul_bench.py \
    --variant helion --dtype bfloat16 \
    --m 1024 --k 1024 --n 1024 --block 128 \
    --iters 20 --repeats 5
```

Run 3 times. Use the median. Record the spread.

### §7.2 Full-matrix sweep

```bash
./scripts/run-on-tpu.sh \
  'examples/pallas_perf/benchmark.sh run_variants.py matmul_jax matmul_pallas matmul_helion > /tmp/results.txt && examples/pallas_perf/filter_best_speedups.py < /tmp/results.txt'
```

### §7.3 Generated-code inspection

```bash
./scripts/run-on-tpu.sh \
  HELION_PRINT_OUTPUT_CODE=1 HELION_LOGS=+all \
  python examples/pallas_perf/matmul_bench.py \
    --variant helion --dtype bfloat16 \
    --m 1024 --k 1024 --n 1024 --block 128 \
    --iters 1 --repeats 1
```

Diff structurally vs `examples/pallas_perf/matmul_pallas.py` for the
same shape. Verify generated-code markers from §9.

### §7.4 Where it runs

- Devserver `devvm2224.cco0.facebook.com` has 1 × H100 only — local
  for editing and `./lint.sh check`.
- Pallas tests + benchmarks run on the remote TPU pod
  (`jongsokchoi-torchtpu`) via `./scripts/run-on-tpu.sh`. The script
  pins `TPU_VISIBLE_CHIPS=1`, `TPU_HOST_BOUNDS=1,1,1`,
  `TPU_DEVICE_BOUNDS=1,1,1`, sets `ALLOW_MULTIPLE_LIBTPU_LOAD=1`.
  **Do not override these.**

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

- Lint (local, `helion_2` conda env): `./lint.sh check`
- Pallas tests (remote TPU):
  `./scripts/run-on-tpu.sh pytest test/test_pallas.py -x -vv`
- Expected counts: _(populated by G0; tolerance ±3 tests. If counts
  drift by more than the tolerance, update this section in the same
  commit.)_

## §9. Generated-code markers

Pin-test substrings the Helion Pallas lowering must (or must not) emit
in generated code. Use `code_and_output(...)` in `test/test_pallas.py`
and `assertIn` / `assertNotIn`.

| Marker                                          | When present                                          | When absent                                  |
|-------------------------------------------------|-------------------------------------------------------|----------------------------------------------|
| `pl.dot(`                                       | bf16 2D tile, MXU-legal (multiples of `min_dot_size`) | otherwise                                    |
| `lax.dot_general(`                              | f32 tile; bf16 tile that fails MXU contract; BMM (3D) | when `pl.dot` covers the case                |
| `preferred_element_type=jnp.float32`            | `lax.dot_general` fallback with sub-32-bit input      | `pl.dot` path                                |
| `lax.convert_element_type(...,` *narrow dtype*  | After f32 accumulator on bf16-output kernel           | when output is already f32                    |
| `dimension_semantics=("parallel", ...)`         | All grid axes marked parallel _(today; needs audit)_  | when reduction axes use `"arbitrary"`         |
| `pltpu.emit_pipeline(`                          | _(future, after G2-D)_ when pipelined HBM↔VMEM lands  | until then                                    |

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
