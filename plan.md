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

**Headline gap.** Helion is ~40% slower than hand-written Pallas on
bf16 1024³ at each variant's best block (H/P = 0.61x median of 3
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
> (see §7.1). Helion cells for rows touched in pre-G2-G cycles are the
> median of 3 back-to-back Helion-only sweeps using `matmul_helion`;
> the same autotuned time is reported under both block-suffix labels in
> the raw output. Starting at G2-G the per-cycle protocol (§7.1) drops
> to a **single** ``measure_headline.py`` measurement for the headline
> row only — gate-exit verification re-runs the 3-sweep `matmul_helion`
> form. The G2-J headline cell (170.20 us) is therefore a 1-call median
> of `n_iter=20 × n_repeats=5` timeit samples on the bf16 1024³ shape,
> not a 3-sweep median. JAX / Pallas cells pick the best of the two
> block configs measured when last re-baselined (the hand-written Pallas
> kernel is unusually slow at block 128 on the headline shape, so its
> best is always block 512). 3-run headline spread was 14.3% in G0,
> 20.7% in G1, 4.3% in G2-A, 17.3% in G2-E, 14.7% in G2-B, and 6.3% in
> G2-F — all under the 20% escalation threshold (G2-B sweep needed 5
> raw runs for the last-3 spread to stay within the threshold; runs 1–3
> alone hit 24.7%. G2-F similarly needed 6 raw runs — runs 1–3 had
> spread 32.4% with one 234.6 us outlier; the last 3 sweeps stabilised
> at 189.1 / 192.1 / 201.2 us, spread 6.3%, median 192.1 us). Per-cycle
> single ``measure_headline.py`` runs since G2-G show measurable
> autotuner-pick variance: at G2-J the dead `_outer_pid_0` / `_outer_pid_1`
> reads are DCE'd from generated `outer_grid` and `emit_pipeline`
> bodies (verified via a one-shot `HELION_PRINT_OUTPUT_CODE=1` dump),
> but the autotuner alternates between
> ``pallas_loop_type='outer_grid'``, ``'unroll'``, and
> ``'emit_pipeline'`` depending on per-run noise, with two back-to-back
> G2-J single-call medians of 194.54 / 170.20 us (run 1 picked
> ``unroll [1024, 1024, 256] pb=F`` at 194.54 us; run 2 settled at
> 170.20 us — within the documented G2-H 163–184 us band)._

| Config                          | JAX (us) | Pallas (us) | Helion (us) | H/P    | H/J    | Source |
|---------------------------------|----------|-------------|-------------|--------|--------|--------|
| bf16 1024×1024×1                | 131.78   | 160.75      | 267.31      | 0.60x  | 0.49x  | G0     |
| bf16 1024×1024×1024 (headline)  | 128.55   | 134.39      | **170.20**  | **0.79x** | 0.76x | G2-J-pending |
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

### 2.1 The headline gap is split across at least three layers (Deep Replan 2026-05-23)

Probing the bf16 1024³ headline with hand-written Pallas kernels (raw
``pl.pallas_call`` issued from a standalone script on the pod, no Helion
launcher in the path) showed the structural gap **is not a single
binding decision**. Three separable causes account for the H/P =
0.61x:

(a) **Autotuner is finding a sub-optimal config (~10-15%).** The
``HELION_AUTOTUNE_EFFORT=full`` search picks
``block_sizes=[512, 1024, 512]`` with ``pallas_loop_type='unroll'`` and
``pallas_pre_broadcast=False``. Hand-fixing the same kernel to
``block_sizes=[512, 512, 512]`` ``pallas_loop_type='unroll'`` is
consistently faster (median Helion 190 us vs autotuner 224 us across 3
runs on chip 3, both back-to-back).  The same probe also reveals
``pallas_pre_broadcast=True`` is the better pick at small blocks
(matters most at bm=bn=bk=256 where Helion goes from 216 → 199 us)
but the autotuner kept ``False``. Numerical noise spread is high
(±10-20%) so the autotuner can mis-rank close configs.

(b) **At Helion's true best config, there is still a ~30% gap to the
hand-written best.** Helion best (``[512,512,512]`` unroll) ≈ 190 us
median, hand-written best (``[512,512,512]`` 3-axis grid) ≈ 131 us
median, H/P = 0.69x. Helion's ``unroll`` lowering emits a
2-axis outer pallas_call grid ``(grid_m, grid_n)`` with x as ``(bm, k)``
strip and y as ``(k, bn)`` strip both loaded to outer VMEM per (m,n)
iteration, then a Python-unrolled K loop inside the kernel does
``pl.dot(x[:, k_slice], y[k_slice, :])``.  When the autotuner picks
bn=1024 (the full N dim) the y strip becomes the full 1024×1024 matrix
(2MB) — reloaded twice (once per outer m-tile).  Hand-written's 3-axis
grid ``(grid_m, grid_n, grid_k)`` keeps per-(m,n,k) iteration VMEM at
~2.5MB and lets Mosaic pipeline across the 8 grid iterations.

(c) **For the ``emit_pipeline`` path specifically, Helion's
generated code is 1.7-2.5x slower than a hand-equivalent emit_pipeline
kernel issued via raw ``pl.pallas_call``.** Concretely at
bm=bn=bk=128: Helion-emit_pipeline 526 us vs hand-rewritten
emit_pipeline mirror 286 us (raw pallas_call probe, both kernels
structurally identical: 2-axis outer grid, inner ``pltpu.emit_pipeline``
over K, outer ``pl.BlockSpec(memory_space=pltpu.HBM)`` for x/y). The
gap shrinks at larger blocks (block 512 is within 5%). However this
path is only chosen by the autotuner at small blocks; it doesn't
account for the headline gap directly, but it confirms a Helion
launcher / codegen overhead exists.  Specifically swapping the outer
in_specs from HBM-only to ``(bm, k) / (k, bn)`` VMEM strips
(``pl.BlockSpec((bm, k), lambda i, j: (i, 0))``) inside the same
emit_pipeline structure cuts the block-128 time from 478 → 249 us
(1.9x). That is the same recipe Helion's ``unroll`` lowering uses;
it's only the ``emit_pipeline`` codegen that misses it.

### 2.2 ``dimension_semantics`` is empirically a no-op for matmul (confirmed Deep Replan 2026-05-23)

Two ablations confirm earlier G2-B finding: (1) on a hand-written 3-axis
grid kernel, flipping K-axis between ``"parallel"`` and ``"arbitrary"``
changes time by < 1% (130.7 vs 130.0 us at bm=bn=bk=512); (2) on
Helion's 2-axis outer grid, flipping all combinations
``(parallel, parallel)``, ``(parallel, arbitrary)``,
``(arbitrary, parallel)``, ``(arbitrary, arbitrary)`` are all
within ~3%. The K loop lives in either ``pltpu.emit_pipeline`` (which
ignores outer ``dimension_semantics`` and runs a serial
``lax.fori_loop`` with ``num_stages=max_buffer_count`` from
``pl.Buffered(buffer_count=2)``) or a Python-unrolled loop (no
scheduling at all), so the marker has nothing to act on.  G2-B's
infrastructure is correct but the lever is empty for matmul.

### 2.3 ``emit_pipeline`` is a serial ``lax.fori_loop`` with 2-stage DMA double-buffering (Deep Replan 2026-05-23)

Read JAX/Mosaic ``pipeline.py`` source. ``pltpu.emit_pipeline`` lowers
to a ``lax.fori_loop`` with prologue prefetch + per-iter
copy_in/wait/body/copy_out/wait scheduler; ``num_stages`` defaults to
``max_buffer_count``, set by ``pl.Buffered(buffer_count=N)`` on
the BlockSpec.  Mosaic itself does no cross-iteration compute
pipelining for emit_pipeline; only DMA is overlapped. The outer
pallas_call ``CompilerParams(dimension_semantics=...)`` controls only
the outer grid; it has no effect on ``emit_pipeline``'s inner loop.
Bumping ``buffer_count`` from 2 → 4 was not measured (TPU was busy
with autotune; deferred), but the structural ceiling is small.

### 2.4 Autotuner picks for representative shapes (Deep Replan 2026-05-23, refreshed at G2-H)

Captured with ``HELION_AUTOTUNE_EFFORT=full`` on chip 3 after G2-F + G2-G
+ G2-H landed:

| Shape | block_sizes (m, k, n) | loop_orders | pallas_loop_type | pre_broadcast | Aggregate autotuner us |
|---|---|---|---|---|---|
| bf16 1024×1×1024            | [1024, 128, 1]   | [[1,0]] | unroll        | True  | 181 |
| bf16 1024×1024×1024 (headline) | [1024, 512, 1024]| [[1,0]] | unroll        | True  | 179 |
| bf16 1024×128×1024          | [1024, 1024, 128]| [[1,0]] | **outer_grid**| True  | 166 |
| bf16 128×1024×1024          | [128, 256, 256]  | [[1,0]] | unroll        | False | 169 |
| bf16 1×1024×1024            | [1, 1024, 1024]  | [[0,1]] | unroll        | False | 163 |
| bf16 1×1×1024               | [1, 1024, 1]     | [[0,1]] | **fori_loop** | False | 164 |
| bf16 1024×1024×1            | [512, 1, 1024]   | [[0,1]] | emit_pipeline | True  | 167 |

Notes: every pick except 1024×128×1024 and 1×1×1024 still resolves to
``unroll`` or ``emit_pipeline``. ``outer_grid`` only wins natural-pick
on 1024×128×1024 (where it edged out unroll [512, 512, 128] by 0.7 us
in final-pick verification). ``fori_loop`` shows up as a natural pick
for the scalar-vector shape 1×1×1024 (first time observed in any G2
log) — its eligibility / lowering hasn't been studied and may benefit
from its own probe later. The 14-shape sweep ran in 5 min total on chip
3 with G2-F's final-pick verification enabled; ~92-136 configs per
shape get verified.

### 2.5 Cross-shape outer_grid forced timings + correctness bug on M=1 (Deep Replan 2026-05-23, G2 closure)

Forced ``pallas_loop_type='outer_grid'`` measured per shape (all bf16,
median of 5 timeit repeats × 20 iters, single chip 3 measurement):

| Shape | autotuner us | og[512] pb=F us | og[256] pb=F us | unroll[512] pb=T us | emit_pipeline[512] pb=F us | Pallas cached us | H/P best |
|---|---|---|---|---|---|---|---|
| bf16 1024×1×1024 (M=K, K=1)    | 181 | FAIL\* | FAIL\* | 175.0 | 188.3 | 167.14 | 0.96 (unroll) |
| bf16 1024×1024×1024 (headline) | 179 | 166.7 | 234.2 | 164.6 | **161.1** | 134.39 | **0.83** (ep) |
| bf16 1024×128×1024             | 166 | 437.4 | 423.7 | 439.2 | 409.4 | 167.21 | **1.01** (autotuner) |
| bf16 128×1024×1024             | 169 | 184.2 | 199.7 | 173.4 | 172.3 | 159.13 | 0.94 |
| bf16 1×1024×1024 (M=1)         | 163 | 183.9† | 171.1† | 179.1 | 168.5 | 163.87 | **1.00** |
| bf16 1×1×1024 (M=1, K=1)       | 164 | FAIL‡ | FAIL‡ | 157.6 | 165.8 | 163.68 | **1.04** (unroll) |
| bf16 1024×1024×1 (N=1)         | 167 | 179.9 | 184.7 | 167.1 | 168.7 | 160.75 | 0.96 |

\* Mosaic alignment error (E2003) at N=1.
‡ Mosaic alignment error (E2003) at K=1 with bk=512/256.
† **silently incorrect** — see correctness bug below.

**Correctness bug (NEW finding)**: ``pallas_loop_type='outer_grid'``
produces **mathematically wrong outputs** on M=1 (and other shapes where
``bm < M`` AND the inner K loop has > 1 iteration). The autotuner's
accuracy check (``HELION_AUTOTUNE_ACCURACY_CHECK=1`` default) catches
this — it skipped 10 outer_grid configs for M=1 K=1024 N=1024 with
relative diffs up to 4.5e6 (NaN at some configs). But the **forced
compile path is not validated**, so a user pinning
``pallas_loop_type='outer_grid'`` on M=1 silently gets garbage. Direct
numerical comparison (``.deep_replan_validate_og_m1.py``, ground truth =
CPU f32 matmul):

| Config                       | max_abs_diff | mismatch / total |
|------------------------------|-------------|------------------|
| outer_grid[1,512,512] pb=F  | NaN | 1024/1024 |
| outer_grid[1,256,256] pb=F  | 43773.1 | 1018/1024 |
| outer_grid[1,1024,1024] pb=F (single K iter) | 0.24 | 0/1024 (CORRECT) |
| outer_grid[1,512,512] pb=T  | 2142.5 | 1010/1024 |
| emit_pipeline[1,512,512] pb=F (control) | 0.24 | 0/1024 (CORRECT) |

Pattern: the bug appears only when the K loop has > 1 iteration AND
``bm == 1``. Single-K-iteration configs (``[1, 1024, 1024]``) are
correct because the rewrite's init/store guards reduce to "always
init, always store". The root cause is likely the body rewrite (or
pre-broadcast handling) on M=1: with ``bm=1`` and a 1-row output tile,
the loop-carried accumulator pattern that ``_codegen_outer_grid_or_fallback``
detects may be matching a non-accumulator scratch (e.g. a per-column
``jnp.sum`` over the broadcast outer product) and rewriting it
incorrectly. Either way the eligibility check is **too permissive**
and must refuse ``bm == 1`` outright.

**Cross-shape impact summary**: outer_grid is **not generically useful**
across the bf16 set today. It wins natural-pick on exactly 1 of 7 bf16
shapes (1024×128×1024, by 0.4%). For 6 of 7 shapes, the autotuner
picks ``unroll`` / ``emit_pipeline`` / ``fori_loop``. Most non-headline
H/P ratios are already ≥ 0.94 (the geo-mean of non-headline-bf16 H/P is
~0.98). The headline is the outlier.

### 2.6 Headline ablation: what's the missing 22%? (Deep Replan 2026-05-23, G2 closure)

Hand-edited hand-written-Pallas (matmul_pallas.py mirror at
``[512,512,512]``, ``dim_sem=("parallel","parallel","arbitrary")``,
``PrefetchScalarGridSpec``) variants timed via raw ``pl.pallas_call``
(cached JIT outside the timed loop;
``.deep_replan_handed_ablation_v2.py``):

| Variant | us | x of HW baseline | Delta |
|---|---|---|---|
| **HW baseline** (matmul_pallas.py mirror) | **125.5** | 1.000x | — |
| HW + redundant ``_outer_pid_0/_1`` reads (mimic Helion's outer_grid) | 132.9 | 1.059x | **+7.4 us / +5.9%** |
| HW ``dim_sem=("parallel","parallel","parallel")`` | 131.7 | 1.050x | +6.3 us |
| HW ``dim_sem=("arbitrary","parallel","arbitrary")`` | 129.3 | 1.030x | +3.8 us |
| HW ``dim_sem=None`` | 140.2 | 1.118x | +14.8 us / +11.8% |
| HW ``prefetch=N`` (``grid+in_specs`` instead of ``grid_spec=PrefetchScalarGridSpec``) | 127.8 | 1.019x | +2.3 us (noise) |
| HW block ``[128,128,128]`` | 440.8 | 3.51x | — (8x more iterations) |
| HW block ``[256,256,256]`` | 175.4 | 1.40x | — |
| HW block ``[1024,1024,1024]`` (no K loop) | 132.1 | 1.053x | +6.6 us |
| HW + ``pl.dot(precision=None)`` (default) | 133.4 | 1.063x | +7.9 us (within noise) |
| HW + ``pl.dot(precision=jax.lax.Precision.DEFAULT)`` | 158.0 | 1.260x | **+32.5 us / +25.9% regression** |
| HW + ``pl.dot(precision=jax.lax.Precision.HIGH)`` | FAIL | — | "Unsupported dot precision: HIGH" |
| HW + ``pl.dot(precision=jax.lax.Precision.HIGHEST)`` | FAIL | — | Mosaic "Bad lhs type" on bf16 |

Back-to-back rerun noise on identical kernel: HW prefetch=Y first run
125.5 us, second run (under "dim_sem=p,p,a control") 134.6 us — that's
~7% same-kernel variance even with cached JIT. Treat any single delta
< 5% as within-noise; only the ``+7.4 us`` redundant-pids and
``+25.9% precision=DEFAULT regression`` are above the noise floor.

**Key conclusions**:

(a) **The redundant ``_outer_pid_0`` and ``_outer_pid_1`` reads cost
~5%** on the hand-written 3-axis kernel (125.5 → 132.9 us). Helion's
``outer_grid`` codegen emits them too (verified via
``HELION_PRINT_OUTPUT_CODE=1`` dump in ``.deep_replan_dump_v2.py``):

```python
def _helion_helion_matmul_kernel(x, y, out, scratch_0):
    _outer_pid_0 = pl.program_id(0)   # UNUSED
    _outer_pid_1 = pl.program_id(1)   # UNUSED
    _outer_pid_2 = pl.program_id(2)   # used by _init / _store guards
```

The ``emit_pipeline`` lowering emits the same dead reads. Removing
them is the highest-confidence remaining lever for ~5% headline gain
without restructuring anything else. Sized: ~7 us × converts to ~4%
relative headline gain if applied via Helion (current Helion headline
163-167 us, hand-written 134-138 us — closing 7 us narrows the gap from
22% to ~17%).

(b) **``pl.dot precision`` is not a usable lever for bf16**. ``HIGH`` is
rejected by Mosaic; ``HIGHEST`` fails to compile on bf16 inputs
("Bad lhs type"). ``DEFAULT`` regresses by 26% (worth avoiding —
Helion's current code path correctly omits the kwarg, which matches the
fast path; see ``helion/_compiler/matmul_utils.py:215`` where
``pl.dot`` is emitted with no precision kwarg for bf16/f16).

(c) **``PrefetchScalarGridSpec`` vs ``grid+in_specs`` is noise** on
the headline (2.3 us delta, within 5% same-kernel variance). Not
worth chasing.

(d) **``dimension_semantics`` is still a no-op** on matmul (consistent
with §2.2): ``p,p,p`` / ``a,p,a`` / ``p,p,a`` all within ~3% of each
other.

(e) **``buffer_count`` (deferred §6.2) was not measured this cycle** —
the TPU was occupied for ~30 minutes by the cross-shape autotune sweep
and the hand-edit ablation, exceeding the cycle budget. The probe
remains queued in §6.2 — but given (a) accounts for ~25% of the
remaining gap on its own, buffer_count is likely a smaller lever.

(f) **No "binding cost we missed"** in the launcher itself. The
``PrefetchScalarGridSpec`` / cache hit / ``JaxCallable`` plumbing is
not the bottleneck (the prior Deep Replan's launcher overhead probes
were unable to attribute > 5 us to the launcher path itself, and the
cached-jit form measured 125-135 us on the hand-written kernel — close
to the ``matmul_pallas.py`` 134 us cached number in §1).

(g) **Same-kernel variance is the noise floor (~5%)**. The headline
gap (22%) is now larger than what any single remaining lever can
plausibly recover with high confidence. Cleaning up redundant pids
(item (a) above) is the best ~5% lever; the remaining ~17% is
distributed across DMA latency, Mosaic's scheduler choices, and the
fundamental Helion launcher dispatch / Python-host overhead — none of
which is a single binding decision.

## §3. Decision rule — where new choices live

When introducing a new behavior, pick the right axis:

1. **Compiler analysis (forced).** Pallas semantics dictate the choice
   (e.g., `pl.dot` tile-size legality from `min_dot_size`, MXU contract,
   dtype promotion). No knob — emit the right thing or fail with a clear
   error.
2. **Lowering strategy (named enum).** Changes the shape of generated
   code; strategies do not compose. Examples: `pl.dot` vs
   `lax.dot_general`; reduction-loop unroll vs `pltpu.emit_pipeline` vs
   `jax.lax.fori_loop`; **outer-grid shape** (2-axis `(grid_m, grid_n)`
   with inner K loop vs 3-axis `(grid_m, grid_n, grid_k)` with `pl.when`
   init/store guards on `program_id(2)`).
3. **Autotune knob (structured record).** A value within a fixed shape
   that the autotuner can sweep without changing kernel structure.
   Examples: `block_m/n/k`, `num_stages`, `dimension_semantics` per axis,
   `pre_broadcast` toggle.

Decision flow: forced → axis 1; shape-changing → axis 2; value within a
fixed shape → axis 3. Re-examine boundaries when a knob keeps escaping
its bin.

**Refinement from Deep Replan 2026-05-23.** ``dimension_semantics`` was
classified as axis 3 in earlier G2-B work, but measurements show it has
no effect on matmul perf because the K reduction lives in either
``pltpu.emit_pipeline`` (which ignores it) or a Python-unrolled loop
(no scheduling). It remains axis 3 for kernels whose outer grid
**does** carry a reduction (e.g. an explicit non-rolled K outer loop)
but the matmul code path will not exercise it until §5 G2-F or a later
substep restructures matmul into a 3-axis outer grid.

## §4. Data model

_(Populated as gates introduce them. Track each new enum / strategy /
record class here with one-line purpose, file location, axis it lives on,
and the test that pins it.)_

- **``pallas_loop_type`` value ``"outer_grid"``** — axis 2 strategy that
  lifts a single user-written inner ``hl.tile`` (the K reduction) into
  the outer ``pl.pallas_call`` grid with ``pl.when(pid_K == 0)`` init
  guard / ``pl.when(pid_K == nsteps - 1)`` store guard, matching the
  hand-written ``examples/pallas_perf/matmul_pallas.py`` 3-axis form.
  Lives in ``helion/autotuner/config_spec.py`` (``VALID_PALLAS_LOOP_TYPES``);
  routing dispatch in ``helion/language/_tracing_ops.py``
  (``_codegen_outer_grid_or_fallback``); body rewrite in
  ``_apply_outer_grid_rewrites`` (called from
  ``DeviceFunction.codegen_function_def``); records lift sites on
  ``DeviceFunction.pallas_outer_grid_lifted`` (a
  ``PallasOuterGridLiftedAxis`` dataclass per lift); launcher reuses the
  existing pipeline launcher with the lifted K axis added to
  ``pid_info``. Eligibility check (single inner block_id + loop-carried
  state + non-reduction outer pids) gates the rewrite; on miss the
  codegen falls back transparently to ``emit_pipeline``. Pin test:
  ``test_pallas_matmul_bf16_outer_grid_lifts_k_axis``.

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

**Goal.** Helion/Pallas ratio ≥ 1.00x on the headline anchor.

**Entrance.** G1 satisfied. Headline still < 1.00x (else skip to G3).

**Exit (all required).**
1. bf16 1024³ headline: H/P ≥ 1.00 (single-call median per §7.1).
2. **Full 14-row sweep verification (3×)**: re-run the §7.1
   gate-exit verification 3 times after the H/P-1.0 single-shape gate
   fires; no other bf16 1024-anything row regressed > 5% vs the G1
   baseline in §1.
3. Two distinct strategies emit verifiably different generated code
   (diff via §7.3), proving the routing logic isn't a no-op.
4. §8 `PALLAS_TEST_CMD` clean.

> **Historical note.** A prior exit #2 read "bf16 1024³
> block(512, 512, 512) H/P ≥ 0.95 (no block-size trade)" — intended to
> prevent winning at block 128 by tanking block 512. Removed
> 2026-05-23: with autotune on, the harness reports the same picked
> kernel under both block-suffix labels (the per-shape autotuner picks
> its own block; the BLOCK_CONFIGS row label is just for table
> rendering). The cross-shape regression check moved to G2-exit
> verification per the §7.1 per-gate protocol table.

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
- **G2-B — `dimension_semantics`.** ✅ 2026-05-23 (audited and threaded;
  lever empirically a no-op for matmul perf). Generalised the
  outer-grid ``CompilerParams(dimension_semantics=...)`` to mark axes
  per ``_compute_reduction_grid_dims`` — any outer-grid axis whose
  block_id has ``reduction=True`` lands as ``"arbitrary"`` and the
  rest stay ``"parallel"``. For Helion matmul the outer grid is
  ``(grid_m, grid_n)`` and neither M nor N is a reduction, so the
  marker resolves to ``("parallel", "parallel")`` (unchanged from
  before). The K loop lives inside ``pltpu.emit_pipeline`` /
  ``jax.lax.fori_loop`` whose own ``dimension_semantics`` defaults to
  ``ARBITRARY`` upstream, so it was already correct. Hand-edit ablation
  on a Helion-shape kernel (probe at G2-B time):
  ``("parallel","parallel")`` ↔ ``("arbitrary","arbitrary")`` ↔ mixed
  combinations were all within ~3 % at ~475 us — i.e. no measurable
  outer-grid ``dimension_semantics`` lever exists in the current
  emit_pipeline architecture. Pin test:
  ``test_pallas_matmul_bf16_outer_grid_no_reduction_dim`` asserts the
  launcher receives no ``_reduction_grid_dims=`` kwarg for matmul (and
  will need updating if a later substep restructures matmul into the
  3-axis grid shape the hand-written reference uses). Headline median
  219.12 us (was 224.26 us); H/P 0.61x (was 0.60x) — within 14.7 %
  spread, treat as flat. The remaining ~40 % gap was decomposed by
  Deep Replan 2026-05-23 (see §2.1): autotuner sub-optimal config
  ~10-15%, structural Helion-vs-hand-written gap ~30% at each side's
  best config. Next-cycle substep ordering moved to **G2-F** (autotuner
  tuning), then **G2-G** (emit_pipeline outer in_specs to VMEM strips),
  then **G2-H** (3-axis outer grid restructure, deferred).
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

#### Remaining substeps (re-ordered by Deep Replan 2026-05-23 G2 closure)

The substeps below are ordered by expected H/P leverage, not by index.
G2-F (autotuner final-pick verification), G2-G (emit_pipeline outer
VMEM strips), G2-H (outer_grid 3-axis restructure), G2-I (M=1
correctness guard), and G2-J (dead outer-pid DCE) have landed (0.59 →
0.79 cumulative; see History). The remaining substep candidates are:

- **G2-K** — Tighten autotuner toward the fastest known headline
  config family. Likely candidates: bias the initial population to
  include ``[512,512,512] emit_pipeline pb=F`` as a seed; add a
  ``block_n == N`` penalty to the surrogate's objective; bump
  ``HELION_AUTOTUNE_FINAL_PICK_TOP_K`` from 5 → 10. §2.5 row 2
  measured ``emit_pipeline [512,512,512] pb=F`` at 161.1 us (H/P
  0.83) and §2.5 / §2.6 show the autotuner currently mis-ranks it
  against ``unroll`` family picks.
- **buffer_count probe (§6.2)** — JAX/Mosaic supports
  ``pl.Buffered(buffer_count=N)`` for ``N ∈ {3, 4}`` on the inner
  ``pltpu.emit_pipeline`` BlockSpec. Estimated ceiling small but
  not yet measured.
- **Deep Replan** if the above two don't close the gap to H/P ≥
  1.00. The remaining ~22% headline gap is distributed across DMA
  latency, Mosaic scheduler choices, and Helion launcher dispatch
  overhead per §2.5 + §2.6 — fresh hypotheses required.

G2 closes only at headline H/P ≥ 1.00 (3-sweep verified per the
"G2 — Closure" entry below); we don't quit short of that.

- **G2-F — Autotuner: rank stability via final-pick verification.**
  ✅ 2026-05-23 (added a final-pick verification phase to
  `PopulationBasedSearch`. After the main search loop finishes (and
  after `run_finishing_phase`), the top-K population members are
  rebenchmarked `HELION_AUTOTUNE_FINAL_PICK_PASSES` extra times in
  interleaved `pl.pallas_call`-pinned passes and re-ranked by the
  median of per-pass medians; the env knobs default to 3 passes /
  top-5 candidates with the implementation in
  `helion/autotuner/base_search.py`
  (`PopulationBasedSearch.run_final_pick_verification`,
  `final_pick_settings`). The call site lives in both
  `helion/autotuner/surrogate_pattern_search.py:LFBOPatternSearch._autotune`
  and `helion/autotuner/pattern_search.py:PatternSearch._autotune` so
  every population-based path picks up the phase. Live autotuner
  output now shows lines like ``Final-pick verification re-picked
  Config(block_sizes=[512, 1, 1024], ...) over previous best
  Config(block_sizes=[1024, 1, 1024], ...)`` — the verification fires
  and re-ranks past noisy initial measurements. For the headline
  shape, the autotuner now picks ``block_sizes=[512, 512, 512]`` with
  ``pallas_pre_broadcast=True`` (the config Deep Replan identified as
  the true best) when the verification fires, instead of the prior
  noisy ``[512, 1024, 512]`` pick. Headline median 192.08 us (was
  219.12 us, +12.3% faster, H/P shifted to 0.70x from 0.61x). Pin
  test ``test_pallas_autotuner_final_pick_picks_true_best_on_noisy_initial_rank``
  asserts that when initial perf misranks the
  ``[512, 1024, 512]`` candidate as best, the verification phase
  re-ranks into the ``[512, 512, 512]`` / ``[256, 256, 256]`` family.
  Did **not** add a `bn == N` heuristic (option (c) in the original
  plan); the verification phase alone moved headline past the exit
  criterion so the surgical heuristic is deferred unless future
  cycles need more headroom.)

- **G2-G — `emit_pipeline` outer in_specs: VMEM strips instead of
  HBM refs.** ✅ 2026-05-23 (added a strip-eligibility pass in
  ``_codegen_emit_pipeline`` (`helion/language/_tracing_ops.py`) that,
  per pipelined tensor, computes the outer working-set footprint as
  the product of inner-pipeline-tiled dim full extents × outer-grid
  block sizes × dtype itemsize, opts in when the doubled (Buffered
  buffer_count=2) sum across candidates fits
  ``_OUTER_VMEM_STRIP_BUDGET_BYTES = 10 MB`` (defined in
  ``helion/runtime/__init__.py``; TPU v7's scoped VMEM is 65 MB so 10
  MB is conservative). Strip-tagged ids flow through
  ``DeviceFunction.pallas_pipeline_vmem_strip``, the launcher arg
  ``_pipeline_vmem_strip_indices``, and a new branch in
  ``_pallas_build_pipeline_specs`` that emits a real BlockSpec
  (existing ``_pallas_make_block_spec`` already produces the
  ``(bm, K) lambda i, j: (i, 0)`` strip shape from
  ``_block_spec_info``) instead of the HBM ref.  Inside the inner
  emit_pipeline BlockSpec, outer-grid dims for strip-tagged operands
  switch from ``_outer_pid_N`` to ``0`` because the outer ref is now
  pre-sliced for the outer grid coord. Headline autotuner pick
  shifted to ``pallas_loop_type='emit_pipeline'``
  ``block_sizes=[1024, 512, 1024]`` ``pre_broadcast=False`` and both
  ``x`` and ``y`` are now strip-tagged. Headline median 169.98 us (was
  192.08 us, -11.5% faster; H/P 0.79x from 0.70x — single
  ``measure_headline.py`` measurement per G2 hill-climb protocol).
  Pin test ``test_pallas_matmul_bf16_emit_pipeline_outer_vmem_strip``
  locks the marker pair: ``_pipeline_vmem_strip_indices=`` in the
  launcher call and ``lambda _j: (0, _j)`` / ``lambda _j: (_j, 0)``
  in the inner BlockSpec lambdas, with negative assertions for the
  pre-G2-G ``_outer_pid_N`` form. Decision rule: outer VMEM working
  set ≤ 10 MB; larger configs fall back to the HBM ref form via the
  same launcher path. fori_loop launcher consumes the same
  ``_pipeline_vmem_strip_indices`` kwarg via
  ``_pallas_build_pipeline_specs`` so any future fori_loop matmul
  picking up an outer-grid-only pipeline strip would inherit the
  same path without code duplication; the matmul autotuner today
  picks emit_pipeline so only that branch is exercised.)

- **G2-H — Restructure matmul into a 3-axis outer grid with `pl.when`
  init/store guards.** ✅ 2026-05-23 (added
  ``pallas_loop_type="outer_grid"`` enum value in
  ``helion/autotuner/config_spec.py:VALID_PALLAS_LOOP_TYPES``; routed
  in ``helion/language/_tracing_ops.py`` via
  ``_codegen_outer_grid_or_fallback``, which runs the existing
  ``_codegen_emit_pipeline`` for body emission then registers a pending
  body rewrite + appends the K block_id to ``DeviceFunction.pid.pid_info``
  so the host wrapper emits a 3-axis ``(grid_m, grid_n, grid_k)`` grid
  with the K dim's ``_block_spec_info`` mapping set to ``(0, 2)`` /
  ``(2, 1)`` instead of the prior ``(0, None)`` / ``(None, 1)``
  emit_pipeline form. The rewrite (``_apply_outer_grid_rewrites``,
  invoked from ``DeviceFunction.codegen_function_def``) walks the
  finalised body, finds the ``def _pipeline_body`` + ``pltpu.emit_pipeline``
  pair by content, wraps the surrounding init / post statements in
  ``@pl.when(_outer_pid_K == 0)`` / ``@pl.when(_outer_pid_K == _k_nsteps - 1)``
  guards, and inlines the K body with parameter substitution
  (``x_vmem`` → ``x`` outer ref, ``_pipeline_indices[0]`` → ``_outer_pid_K``).
  ``_compute_reduction_grid_dims`` (in ``helion/_compiler/backend.py``)
  reads ``DeviceFunction.pallas_outer_grid_lifted`` so the launcher
  marks the K axis ``"arbitrary"`` in ``dimension_semantics`` — no
  global ``BlockSizeInfo.reduction`` flag mutation. Eligibility check
  in ``_codegen_outer_grid_or_fallback`` gates the rewrite (single
  inner block_id + loop-carried state + non-reduction outer pids); on
  miss, codegen falls back transparently to ``emit_pipeline``. Pin
  test ``test_pallas_matmul_bf16_outer_grid_lifts_k_axis`` asserts
  ``_reduction_grid_dims=[2]`` in the launcher kwargs, the 3-axis
  ``_block_spec_info`` shape, ``@pl.when(_outer_pid_2 == 0)`` /
  ``@pl.when(_outer_pid_2 == _k_nsteps_2 - 1)`` body guards, no
  ``pltpu.emit_pipeline(`` survives, and the pre-G2-G strip kwargs
  are absent.

  Headline single ``measure_headline.py`` runs: 163.20 us (H/P 0.82x,
  was 169.98 us / 0.79x at G2-G). Autotuner now picks
  ``pallas_loop_type='outer_grid'`` on ~half of runs (e.g. run picked
  ``outer_grid [1024, 1024, 256]`` → 167.01 us; ``unroll [1024, 1024, 1024]``
  → 163.20 us / 167.53 us in alternate runs).  Hand-fixed probe at
  bm=bn=bk=512 ``pre_broadcast=False``: outer_grid 236.49 us vs
  emit_pipeline 248.19 us vs unroll 250.93 us — outer_grid wins at
  this block by ~5%.  At small block 128 outer_grid degrades (546 us
  vs emit_pipeline 360 us) — expected; the small-block 8x8x8 = 512
  outer grid iterations amortise scratch-init/store per-iter overhead
  poorly. The autotuner with G2-F's final-pick verification picks the
  block-512 outer_grid family when it lands the right combination.
  H/P movement (+3 pp on the headline) is materially smaller than the
  Deep Replan structural-probe estimate (~7% at one block) because the
  G2-F + G2-G stack already harvested most of the structural slack;
  the remaining gap is shared across several small lever costs that no
  single restructure closes. See §11 anti-patterns for why this is
  recorded as a partial G2-H landing rather than a full G2 exit.)

- **G2-C — Block-spec layout (DEMOTED).** Originally drafted as
  "compare BlockSpec ordering, index_map, and memory_space (VMEM vs
  ANY) between Helion-generated and hand-written kernels". §2.1 (c)
  + G2-G subsume the `memory_space` question (HBM ref vs VMEM strip);
  the rest (ordering, index_map shape) showed no measurable delta in
  probe #2. Demote to a documentation-only follow-up after G2-F/G/H.

- **G2-I — Correctness gate: refuse ``outer_grid`` on singleton
  outer-tile block sizes.** ✅ 2026-05-23 (extended the eligibility
  check in ``_codegen_outer_grid_or_fallback``
  (``helion/language/_tracing_ops.py``) to refuse the lift whenever
  any outer-grid axis's configured block size (resolved via
  ``BlockSizeInfo.from_config(state.config)``) equals 1; new helper
  ``_outer_pid_block_is_singleton``. The check fires on M=1
  (``bm == 1``) shapes like bf16 1×1024×1024, on N=1 (``bn == 1``)
  shapes, and on any future kernel whose outer pids include a
  singleton tile, falling back transparently to
  ``_codegen_emit_pipeline`` — the same path Helion picked before
  ``outer_grid`` existed. Without the guard, the body rewrite on
  ``bm == 1`` matched on the loop-carried scratch but reinterpreted
  it as a 2D matmul accumulator, producing silently-wrong outputs
  whenever the K loop had > 1 iteration (relative diffs up to 4.5e6
  vs CPU f32 — see §2.5 correctness bug and the validation script
  ``.deep_replan_validate_og_m1.py``). The autotuner's accuracy
  check (``HELION_AUTOTUNE_ACCURACY_CHECK=1``) had been protecting
  autotuned runs by skipping the broken configs, but a forced
  ``pallas_loop_type='outer_grid'`` config on M=1 was unprotected.
  Pin tests:
  ``test_pallas_matmul_outer_grid_falls_back_on_singleton_m`` forces
  ``pallas_loop_type='outer_grid'`` on bf16 1×1024×1024 with
  ``block_sizes=[1, 128, 128]`` and asserts ``pltpu.emit_pipeline(``
  survives while ``@pl.when(_outer_pid_2 == 0)`` and
  ``_reduction_grid_dims=[2]`` do not appear; companion pin
  ``test_pallas_matmul_outer_grid_fires_on_multi_m`` keeps the
  existing multi-row M path honest by forcing the same loop type on
  bf16 1024×1024×1024 with ``block_sizes=[128, 128, 128]`` and
  asserting the outer-grid markers still appear. Headline single
  ``measure_headline.py`` median 174.83 us — within the documented
  autotuner-pick variance band (G2-H samples ranged 163.20-183.68 us
  across 5 back-to-back single-call measurements); the guard is a
  pure-correctness no-op on the headline because every headline
  config has ``bm > 1``.)

- **G2-J — Drop dead ``_outer_pid_N`` reads from outer_grid +
  emit_pipeline generated kernels.** ✅ 2026-05-23 (added a
  ``_drop_dead_outer_pid_reads`` AST DCE pass in
  ``helion/language/_tracing_ops.py`` and wired it into
  ``DeviceFunction.codegen_function_def``
  (``helion/_compiler/device_function.py``) so it runs after
  ``_apply_outer_grid_rewrites`` on every Pallas device function.
  The pass walks the flat body, collects all ``Name`` references
  inside top-level statements (using ``ast.walk`` so nested
  ``@pl.when`` / ``_pipeline_body`` / BlockSpec lambdas are
  inspected), and drops every ``_outer_pid_N = pl.program_id(N)``
  setup whose LHS isn't read elsewhere. The K pid (e.g.
  ``_outer_pid_2``) survives because the ``@pl.when(_outer_pid_K ==
  0)`` / ``@pl.when(_outer_pid_K == _k_nsteps_K - 1)`` guards read
  it. For the matmul ``outer_grid`` body the M / N pids (axes 0 /
  1) are dead and now DCE'd; for the matmul ``emit_pipeline`` strip
  path both M and N are dead (the BlockSpec lambdas emit ``0`` for
  strip-tagged outer dims). HBM-ref ``emit_pipeline`` keeps both
  pids alive (the lambdas read them to slice the outer ref). Pin
  tests: ``test_pallas_matmul_bf16_outer_grid_omits_dead_pids``,
  ``test_pallas_matmul_bf16_emit_pipeline_omits_dead_pids``, and
  ``test_pallas_matmul_bf16_emit_pipeline_keeps_used_pids_on_hbm_ref``
  cover both DCE paths and the negative HBM-ref path. Verified via
  one-shot ``HELION_PRINT_OUTPUT_CODE=1`` dump on the bf16 1024³
  shape: the generated ``outer_grid`` body emits only
  ``_outer_pid_2`` + ``_k_nsteps_2`` plus the
  ``@pl.when(_outer_pid_2 == 0)`` / ``@pl.when(_outer_pid_2 ==
  _k_nsteps_2 - 1)`` decorators. Headline single
  ``measure_headline.py`` median 170.20 us (H/P 0.79x). Two
  back-to-back single-call runs were 194.54 / 170.20 us — within
  the documented G2-H 163–184 us autotuner-pick variance band
  (run 1 picked ``unroll [1024, 1024, 256] pb=F``; run 2 settled
  ~163-170 us). PALLAS_TEST_CMD: 100 passed / 0 failed / 6 xfailed
  / 39 deselected (+3 pin tests vs prior 97). G2-J landed
  structurally; the autotuner-pick variance dominates the
  per-cycle headline signal so the DCE win is masked at the single
  measurement.)

- **G2-K — (Optional) Tighten autotuner toward emit_pipeline
  ``[512,512,512] pb=F``.** §2.5 row 2 shows
  ``emit_pipeline[512,512,512] pb=F`` measures the fastest headline
  config at 161.1 us (H/P 0.83), but the autotuner picks
  ``unroll[1024,512,1024] pb=T`` (179 us aggregate / H/P 0.75).
  G2-F's final-pick verification helps but doesn't fully steer the
  search toward emit_pipeline at the small-block family. Possible
  fixes: (a) bias the initial population to include
  ``[512,512,512] emit_pipeline pb=F`` as a seed; (b) add a
  ``block_n == N`` penalty to the surrogate's objective; (c) bump
  ``HELION_AUTOTUNE_FINAL_PICK_TOP_K`` default from 5 → 10 so close
  configs near the median get more verification passes. Only land
  G2-K if G2-J doesn't close the gap to H/P ≥ 1.00 on its own.
  Estimated effort: medium (search-space heuristic + autotuner test
  update). Expected gain: 5-10% if the right config is reachable
  but mis-ranked.

- **G2-D — Time and decide (diagnostic loop).**

  Run the per-cycle headline single-shape probe (§7.1). Capture:
  - Headline H/P (single-call median).
  - Diff of generated code vs the prior cycle's commit.

  Branch:
  - **H/P ≥ 1.00** → trigger the G2-exit verification per the §7.1
    table (3-sweep full Helion-only run). On clean verification,
    advance to G3.
  - **H/P ∈ [0.95, 1.00)** → one more focused cycle inside the
    current substep is allowed before escalating.
  - **H/P < 0.90 for 2 consecutive cycles** → trigger Deep Replan
    (manager.md Step 8). Do not add more substeps speculatively.
  - **Regression > 5% vs prior cycle** → revert (manager.md Step 4d)
    and restart the same substep.

- **G2 — Closure.**

  **G2 closes only when bf16 1024³ headline H/P ≥ 1.00**, verified
  by the §7.1 3-sweep gate-exit protocol (full 14-row Helion-only
  sweep × 3; the bf16 1024³ row's median across the 3 sweeps must
  be ≥ 1.00x and no other bf16 row may regress > 5% vs the §1
  baseline). No "documented shortfall" / "close at 0.85x" escape
  hatch — if the current substep set doesn't hit the bar, the next
  substep does; if the substep menu is empty, Deep Replan finds
  more (manager.md Step 8).

  Hard decision rules:
  - **H/P ≥ 1.00 single-call** → trigger the 3-sweep verification
    immediately. Clean verification → G2 ✅ and advance to G3.
  - **H/P ∈ [0.85, 1.00)** → G2 stays open; the current substep is
    done; pick the next substep from the menu below (or trigger
    Deep Replan if the menu is empty).
  - **H/P < 0.85 for 2 consecutive cycles** → trigger Deep Replan.
  - **Regression > 5% vs prior cycle** → revert (manager.md Step
    4d) and restart the same substep.

  Remaining substep candidates (after G2-J): **G2-K** (autotuner
  heuristics — bias toward emit_pipeline ``[512,512,512] pb=F`` or
  add a ``block_n == N`` penalty); the deferred ``buffer_count``
  probe (§6.2) if not yet measured; otherwise queue a Deep Replan
  for fresh hypotheses.

**History.**

| Date       | Commit       | Headline (us) | H/P   | Substep | Notes |
|------------|--------------|---------------|-------|---------|-------|
| 2026-05-23 | G2-A-pending | 236.75        | 0.57x | G2-A    | `pl.dot` now fires on bf16 2D tiles; harness reports the autotuned time under both block-suffix labels. Headline flat (Δ -0.02x vs G1's 0.59x, within 4.3% spread). |
| 2026-05-23 | G2-E-pending | 224.26        | 0.60x | G2-E    | Fuse `scratch[...] = acc; acc = scratch[...] + dot(...)` into `scratch[...] += dot(...)` inside `_write_back_loop_carried`; chain-DCE removes the now-dead scratch read/copy intermediates. Inner pipeline body matches the hand-written `acc_ref[...] += pl.dot(...)` pattern; Mosaic still serializes the K loop so this only buys back the per-K bind cost (~5%, +0.03x H/P). G2-B (`dimension_semantics`) and serialisation routing remain the dominant gap. |
| 2026-05-23 | G2-B-pending | 219.12        | 0.61x | G2-B    | Threaded reduction-axis info from `_compute_reduction_grid_dims` (backend.py) into both pallas launchers and built `dimension_semantics` per-axis. For matmul the outer grid has no reduction axis, so the marker resolves to the same `("parallel","parallel")` as before; no perf delta vs G2-E (within 14.7% spread). Hand-edit ablations (probe time): outer-grid `dimension_semantics` value is empirically a no-op (~475 us regardless of label), but structurally switching to the hand-written 3-axis grid recovers ~7% (458 → 426 us). Recommend Deep Replan to weigh G2-C block-spec vs the 3-axis restructure. |
| 2026-05-23 | G2-F-pending | 192.08        | 0.70x | G2-F    | Added `PopulationBasedSearch.run_final_pick_verification`: after the main search + finishing phase, the top-5 population members are rebenchmarked 3 extra times (configurable via `HELION_AUTOTUNE_FINAL_PICK_PASSES` / `HELION_AUTOTUNE_FINAL_PICK_TOP_K`) and re-ranked by the median of per-pass medians, drowning out the ±10-20% per-call noise that mis-ranked close configs. Headline autotuner picks shifted from `[512, 1024, 512] pre_broadcast=False` (Deep Replan baseline) to `[512, 512, 512] pre_broadcast=True` (Deep Replan-identified true best). Headline median 192.08 us (was 219.12 us, -12.3%, H/P 0.61x → 0.70x). 6-run sweep needed (first 3 had spread 32.4% with a 234.6us outlier); last 3 stable at 189.1 / 192.1 / 201.2 us spread 6.3%. |
| 2026-05-23 | G2-G-pending | 169.98        | 0.79x | G2-G    | Strip-eligible pipelined tensors keep a real ``(outer_block, full_inner)`` BlockSpec at the outer pallas_call level instead of an HBM ref, gated by ``_OUTER_VMEM_STRIP_BUDGET_BYTES = 10 MB``; inner emit_pipeline BlockSpec lambda uses ``0`` for outer-grid dims of strip-tagged operands. Autotuner reordered to ``pallas_loop_type='emit_pipeline'`` ``block_sizes=[1024, 512, 1024]`` (was ``[512, 512, 512]`` unroll under G2-F) once the path stopped paying per-K HBM→VMEM DMA. Both ``x`` and ``y`` strip-tag (combined doubled strip ≈ 4 MB, well within budget). Generated code emits ``_pipeline_vmem_strip_indices=[0, 1]`` and ``lambda _j: (0, _j)`` / ``lambda _j: (_j, 0)``. Headline median 169.98 us (was 192.08 us, -11.5%, H/P 0.70x → 0.79x; single measurement per the new G2 per-cycle protocol). |
| 2026-05-23 | G2-H-pending | 163.20        | 0.82x | G2-H    | Added ``pallas_loop_type='outer_grid'`` enum value (3-axis outer grid ``(grid_m, grid_n, grid_k)`` with ``dimension_semantics=('parallel', 'parallel', 'arbitrary')``, ``@pl.when(_outer_pid_K == 0)`` init guard, ``@pl.when(_outer_pid_K == _k_nsteps - 1)`` store guard — matches ``examples/pallas_perf/matmul_pallas.py``). Routed via ``_codegen_outer_grid_or_fallback`` in ``helion/language/_tracing_ops.py`` with a body-rewrite pass in ``_apply_outer_grid_rewrites`` (invoked from ``DeviceFunction.codegen_function_def``). The eligibility check (single inner block_id + loop-carried state + non-reduction outer pids) falls back to ``emit_pipeline`` on miss. Pin test ``test_pallas_matmul_bf16_outer_grid_lifts_k_axis``. Headline single-call median 163.20 us (was 169.98 us, -4.0%, H/P 0.79x → 0.82x). Autotuner now alternates between ``outer_grid`` and the pre-existing paths depending on noise (3 of 5 single runs landed in 163–168 us; one run picked an unfortunate ``outer_grid [1024, 1024, 512]`` config at 183.68 us). Hand-fixed bm=bn=bk=512 ``pre_broadcast=False`` probe: outer_grid 236.49 us vs emit_pipeline 248.19 us vs unroll 250.93 us — the new path is ~5% faster at this block. At small block 128 outer_grid degrades (546 us vs emit_pipeline 360 us); the autotuner's final-pick verification (G2-F) correctly steers away from those. **Known issue (Deep Replan 2026-05-23 G2 closure)**: outer_grid produces silently-wrong outputs on M=1 shapes with multi-K-iteration configs (see §2.5 correctness bug). The autotuner skips them via accuracy check; forced configs are unprotected. **G2-I** must land before G2 closure to fix the eligibility check. |
| 2026-05-23 | G2-I-pending | 174.83        | 0.77x | G2-I    | Extended ``_codegen_outer_grid_or_fallback`` eligibility check to refuse the lift when any outer-grid axis's configured block size resolves to 1; new helper ``_outer_pid_block_is_singleton`` queries ``BlockSizeInfo.from_config(state.config)`` per outer pid. Without the guard, the body rewrite on ``bm == 1`` (e.g. bf16 1×1024×1024) matched the loop-carried scratch but reinterpreted it as a 2D matmul accumulator, producing silently-wrong outputs whenever the K loop had > 1 iteration (relative diffs up to 4.5e6 vs CPU f32 — §2.5). The autotuner's accuracy check had been masking the bug for autotuned runs; forced ``pallas_loop_type='outer_grid'`` configs were unprotected. New pin tests: ``test_pallas_matmul_outer_grid_falls_back_on_singleton_m`` (M=1, ``block_sizes=[1, 128, 128]``: asserts ``pltpu.emit_pipeline(`` present + outer-K markers absent) and ``test_pallas_matmul_outer_grid_fires_on_multi_m`` (M=1024, ``block_sizes=[128, 128, 128]``: asserts the existing outer-grid markers still appear). Headline single ``measure_headline.py`` median 174.83 us (was 163.20 us at G2-H; 3 consecutive runs landed at 174.83 / 183.72 / 192.54 us with the autotuner alternating ``outer_grid``, ``unroll``, and ``unroll`` final picks). The guard cannot affect any path on the bf16 1024³ headline (every config has ``bm > 1``); the headline movement reflects the documented G2-H autotuner-pick variance band rather than a regression caused by this change. PALLAS_TEST_CMD: 97 passed / 0 failed / 6 xfailed (+2 vs prior 95). |
| 2026-05-23 | G2-J-pending | 170.20        | 0.79x | G2-J    | Added ``_drop_dead_outer_pid_reads`` AST DCE pass in ``helion/language/_tracing_ops.py`` and wired into ``DeviceFunction.codegen_function_def`` (``helion/_compiler/device_function.py``) so every Pallas device function runs the DCE after ``_apply_outer_grid_rewrites``. Drops top-level ``_outer_pid_N = pl.program_id(N)`` setups whose LHS isn't referenced elsewhere in the body; uses ``ast.walk`` so nested ``@pl.when`` / ``_pipeline_body`` / BlockSpec lambdas are inspected. The K pid (e.g. ``_outer_pid_2``) survives because the init / store guards read it; matmul's ``outer_grid`` body drops M / N (dead), matmul's strip-path ``emit_pipeline`` drops M / N too (lambdas emit ``0`` for strip-tagged outer dims), HBM-ref ``emit_pipeline`` keeps them (lambdas slice via the pids). Generated-code dump (``HELION_PRINT_OUTPUT_CODE=1`` on a bf16 1024³ ``outer_grid`` build) confirms the body now emits only ``_outer_pid_2`` + ``_k_nsteps_2`` + the two ``@pl.when`` decorators. Pin tests: ``test_pallas_matmul_bf16_outer_grid_omits_dead_pids``, ``test_pallas_matmul_bf16_emit_pipeline_omits_dead_pids``, ``test_pallas_matmul_bf16_emit_pipeline_keeps_used_pids_on_hbm_ref``. Headline single ``measure_headline.py`` runs: 194.54 / 170.20 us (run 1 picked ``unroll [1024, 1024, 256] pb=F``; run 2 settled within the documented G2-H 163–184 us autotuner-pick variance band). Cycle-end headline = 170.20 us (H/P 0.79x); the DCE win is masked by autotuner-pick noise at the per-cycle single-call signal but the dead-pid markers ARE gone from generated code. G2 stays open (manager directive: G2 closes only at H/P ≥ 1.00, 3-sweep verified). PALLAS_TEST_CMD: 100 passed / 0 failed / 6 xfailed / 39 deselected (+3 pin tests vs prior 97). |

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

- **6.2** `pltpu.emit_pipeline` `num_stages` / `pl.Buffered(buffer_count=N)`
  tuning. JAX/Mosaic `pipeline.py` accepts `pl.Buffered(buffer_count=N)`
  with `num_stages=max_buffer_count` in the scheduler (default 2). A
  buffer_count probe to N ∈ {3, 4} was queued but blocked by the TPU
  being held by the autotuner during Deep Replan 2026-05-23.
  **Re-open criterion.** Run the buffer_count probe after G2-G lands
  (i.e. once the emit_pipeline path is on the critical path again);
  if the bump moves headline by ≥ 3%, promote to an autotune knob.
  Probe script: `/home/jongsokchoi/helion_2/.deep_replan_buffers_probe.py`
  (not committed; local working tree only). _G2 closure replan
  2026-05-23: still not run — the cross-shape autotune sweep + hand-edit
  ablation occupied chip 3 for the cycle's TPU budget. §2.6 (e) notes
  that buffer_count is likely a smaller lever than the redundant-pids
  cleanup (~5%) recommended in G2-J. Re-open after G2-J lands if
  headline H/P is still < 1.00 and G2-K hasn't fired yet._

- **6.3** Full 14-shape autotuner-pick capture (deferred from Deep
  Replan 2026-05-23). The full-matrix autotune sweep was bounded out
  by the 30-min-budget cap; only 3 representative shapes were captured.
  **Re-open criterion.** When G2-F lands and the autotuner search
  changes (different objective / repeat count), re-run the 14-shape
  capture to confirm the new picks aren't worse than the old. _G2
  closure replan 2026-05-23: 7 bf16 shapes captured (see §2.4 refresh);
  full f32 sweep still deferred. Re-open this entry as a §6.4 split
  when G4 (f32 frontier) opens._

## §7. Reproduction (fixed-target benchmark configuration)

### §7.1 Headline command

**Per-gate benchmark scope protocol.** Hill-climb on one signal at a
time per cycle, then broaden at gate-exit verification.

| Gate | Per-cycle (hill-climb iter)              | Gate-exit verification          |
|------|-------------------------------------------|----------------------------------|
| G2   | bf16 1024³ × **1** measurement            | bf16 1024³ × **3** sweeps        |
| G3   | + remaining bf16 shapes × 1 each          | full bf16 set × 3 sweeps         |
| G4   | + all f32 shapes × 1 each                 | full 14-row matrix × 3 sweeps    |
| G5   | full matrix × 1 each                       | full matrix × 3 sweeps           |

Rationale: hill-climb on one signal at a time; verify with 3 sweeps at
gate exit; broaden scope only at the next gate. A change that moves the
per-cycle headline by ≥ 3% (G2) is "on the right track"; the
generated-code marker / structural diff is the secondary signal when the
delta is smaller.

**Per-cycle headline (single-shape, single-measurement).** Use the
single-shape probe; it imports the kernel from ``matmul_helion.py`` so
any kernel-side change is picked up by both the full harness and the
probe.

```bash
./scripts/run-on-pod.sh HELION_BACKEND=pallas TPU_VISIBLE_CHIPS=3 \
  examples/pallas_perf/benchmark.sh examples/pallas_perf/measure_headline.py
```

Prints `helion_bf16_1024x1024x1024: median=<us> us` to stdout. One
measurement per cycle for G2; broaden per the table above as later
gates open. Compute `H/P = cached_pallas_us / median_helion_us` against
the §1 cached Pallas cell.

**Gate-exit verification (3-sweep Helion-only).** Use the full
single-variant sweep so the per-shape autotuner picks land:

```bash
./scripts/run-on-pod.sh HELION_BACKEND=pallas TPU_VISIBLE_CHIPS=3 \
  bash -c 'examples/pallas_perf/benchmark.sh examples/pallas_perf/run_variants.py matmul_helion > /tmp/helion.txt 2>&1 && examples/pallas_perf/filter_best_speedups.py < /tmp/helion.txt'
```

Run 3 times at gate exit. Headline = the `bf16 1024×1024×1024` row
median across runs. Record the 3-run spread. JAX / Pallas references
are cached in §1 from the most recent full re-baseline.

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

- Headline 3-run spread > 20% on two consecutive gate-exit sweeps
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
  measure_headline.py        # single-shape bf16 1024³ probe (§7.1)
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

- **Expected counts** (current, with the `-k` filter above): **100
  passed, 0 failed, 6 xfailed, 39 deselected** (tolerance ±3 tests).
  Baseline at G0 was 84 passed; +4 from G1 pin tests, +2 from G2-A pin
  tests, +1 from G2-E, +1 from G2-B, +1 from G2-F, +1 from G2-G, +1 from
  G2-H, +2 from G2-I, +3 from G2-J. Without the filter, expect **~101
  passed / 40 failed / 6 xfailed / 0 skipped** on `upstream/main` until
  §6.1 is resolved.

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
| `dimension_semantics=("parallel", ...)`         | Outer grid axes that are not reduction blocks (all of matmul M/N today; the marker comes out of the launcher call site so it does not appear in `code_and_output` text — verify it via the launcher kwarg below or by instrumenting the runtime) | when an outer-grid axis is a reduction (`_compute_reduction_grid_dims` flags it) |
| `_reduction_grid_dims=`                          | Host wrapper passes the kwarg only when the outer grid has at least one reduction axis (matmul does not — K lives inside `pltpu.emit_pipeline`); presence flips the launcher's matching grid dim from `"parallel"` to `"arbitrary"` | matmul and other kernels whose outer grid has no reduction axis (kwarg omitted; launcher defaults every outer axis to `"parallel"`) |
| `pltpu.emit_pipeline(`                          | Inner-pipelined K loop (autotuner pick `pallas_loop_type='emit_pipeline'`); already lands today. The marker is in the device-fn body, not the launcher. | when autotuner picks `unroll` (which Python-unrolls the K loop) or `fori_loop` |
| `scratch_N[...] += <dot_expr>`                  | Inner `_pipeline_body` accumulator stays on the VMEM ref between K iterations (matches hand-written `acc_ref[...] += pl.dot(...)` pattern) | until G2-E lands or a non-matmul lifecycle bypasses the rewrite (e.g. acc consumed by something other than the write-back) |
| `scratch_N[...] = <acc_var>[...]` *inside `_pipeline_body`* | externalised acc value-flow per K step (pre-G2-E) — re-introducing this signals the in-place rewrite regressed | once G2-E's fuse is wired through the loop-carried-state write-back |
| `_pipeline_vmem_strip_indices=` *launcher kwarg for an `emit_pipeline` kernel* | pipelined arg(s) whose outer working-set fits the strip budget keep a real BlockSpec (`(outer_block, full_inner)` VMEM strip) at the outer pallas_call level; the inner emit_pipeline BlockSpec slices the strip from VMEM instead of DMAing per inner iter from HBM. Budget: `_OUTER_VMEM_STRIP_BUDGET_BYTES = 10 MB` in `helion/runtime/__init__.py`; lambda for VMEM-strip operands also flips outer-grid coords from `_outer_pid_N` to `0` (see `_make_block_spec` in `helion/language/_tracing_ops.py`). | kwarg omitted when no pipelined tensor fits the budget (large blocks) or none has an outer-grid dim to slice; launcher falls back to the HBM ref form for every pipelined arg |
| `pl.BlockSpec(memory_space=pltpu.HBM)` *outer in_specs for `emit_pipeline`* | pipelined arg whose outer strip footprint exceeds `_OUTER_VMEM_STRIP_BUDGET_BYTES` (or whose strip can't be sized at codegen time) falls back to an HBM ref so emit_pipeline does per-iter DMA | when the outer VMEM working set fits the chip budget — see the `_pipeline_vmem_strip_indices=` marker above |
| `@pl.when(_outer_pid_K == 0)` *body decorator* | ``pallas_loop_type='outer_grid'`` lifted the inner K loop into the outer pallas_call grid; init guard on first K iter. The matching ``@pl.when(_outer_pid_K == _k_nsteps - 1)`` decorator guards the store on the last K iter. K's index ``_outer_pid_K`` = ``pl.program_id(<grid_dim>)`` where ``<grid_dim>`` is the index assigned by ``DeviceFunction.pid.pid_info.append`` in ``_codegen_outer_grid_or_fallback`` (typically 2 for matmul, after M=0, N=1). | matmul's K is in the inner ``pltpu.emit_pipeline`` (default ``emit_pipeline`` path) or Python-unrolled (``unroll`` path), and init is unconditional before the pipeline |
| `_reduction_grid_dims=[<k_grid_dim>]` *launcher kwarg for an `outer_grid` kernel* | matmul (or any kernel with a lifted reduction axis) under ``pallas_loop_type='outer_grid'`` flags the K grid dim as reduction so the launcher marks it ``"arbitrary"`` in ``dimension_semantics``. Computed in ``_compute_reduction_grid_dims`` (``backend.py``) via ``DeviceFunction.pallas_outer_grid_lifted`` — no global ``BlockSizeInfo.reduction`` flag mutation. | other ``pallas_loop_type`` values (``emit_pipeline`` / ``fori_loop`` / ``unroll``) where the K loop stays inside the kernel and the outer grid carries only M / N |
| `_block_spec_info=[((bm, bk), (0, k_grid_dim)), ((bk, bn), (k_grid_dim, 1)), ((bm, bn), (0, 1))]` *3-axis matmul launcher entry* | ``pallas_loop_type='outer_grid'`` matmul: X's K dim is bound to grid dim ``k_grid_dim``, Y's K dim too, out stays on M/N only. Pre-G2-H matmul had ``((bm, None), (0, None))`` / ``((None, bn), (None, 1))`` because K was inside ``pltpu.emit_pipeline`` and not in the outer grid. | ``emit_pipeline`` / ``fori_loop`` / ``unroll`` paths keep the pre-G2-H 2-axis ``_block_spec_info`` shape because K is not in the outer grid |
| `pltpu.emit_pipeline(` *device-fn body* | ``pallas_loop_type='emit_pipeline'``, OR the ``outer_grid`` eligibility check failed and we fell back to ``emit_pipeline`` | ``pallas_loop_type='outer_grid'`` and the eligibility check passed — the K loop is the outer grid now, no inner ``pltpu.emit_pipeline`` call survives |
| `_outer_pid_0` / `_outer_pid_1` *unused reads in `outer_grid` or strip-`emit_pipeline` body* | absent — the `_drop_dead_outer_pid_reads` AST DCE pass (in `helion/language/_tracing_ops.py`, called from `DeviceFunction.codegen_function_def`) strips every top-level `_outer_pid_N = pl.program_id(N)` whose LHS isn't referenced elsewhere in the body. For matmul `outer_grid` the K pid (`_outer_pid_2`) is the only one used (by the `@pl.when` guards); for matmul strip-path `emit_pipeline` no outer pids are used (lambdas emit `0`). Hand-edit ablation measured the dead reads at +7.4 us / +5.9% on the bf16 1024³ headline (§2.6 (a)). Pin tests: `test_pallas_matmul_bf16_outer_grid_omits_dead_pids`, `test_pallas_matmul_bf16_emit_pipeline_omits_dead_pids`. | HBM-ref `emit_pipeline` (when the outer VMEM strip footprint exceeds `_OUTER_VMEM_STRIP_BUDGET_BYTES`) keeps `_outer_pid_0` / `_outer_pid_1` alive because the inner BlockSpec lambdas (`lambda _j: (_outer_pid_0, _j)` / `lambda _j: (_j, _outer_pid_1)`) read them to slice the HBM ref. Pin test: `test_pallas_matmul_bf16_emit_pipeline_keeps_used_pids_on_hbm_ref`. |
| `pltpu.emit_pipeline(` *for forced `outer_grid` on any outer-axis with block size 1* | when `pallas_loop_type='outer_grid'` is forced on a shape whose configured block size for any outer-grid axis (M or N today) is 1, the eligibility check falls back to `emit_pipeline` — the marker that survived is `pltpu.emit_pipeline(`, NOT `@pl.when(_outer_pid_K == 0)`. Companion pin `test_pallas_matmul_outer_grid_fires_on_multi_m` confirms the guard does not over-refuse on the working multi-row M path (`bm > 1`). Pin test `test_pallas_matmul_outer_grid_falls_back_on_singleton_m` asserts `pltpu.emit_pipeline(` present and `@pl.when(_outer_pid_2 == 0)` / `_reduction_grid_dims=[2]` absent on the bf16 1×1024×1024 / `block_sizes=[1, 128, 128]` shape with `pallas_loop_type='outer_grid'`. | the outer-grid body rewrite (pre-guard) used to fire on M=1 too, producing silently-wrong outputs whenever the K loop had > 1 iteration (relative diffs up to 4.5e6 vs CPU f32; see §2.5 correctness bug). The autotuner's accuracy check skipped these configs but forced configs were unprotected. |

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
- **Trusting a single autotuner run as ground truth for what Helion can
  achieve.** Deep Replan 2026-05-23 showed autotuner picks for the
  headline shape were 10-15% slower than a hand-fixed
  `block_sizes=[512, 512, 512]` config. Benchmark noise (±10-20%
  spread) is large enough to scramble ranking on close configs. Before
  declaring a structural gap, manually pin a few alternate configs and
  measure them back-to-back. (See §2.1 (a).)
- **Comparing kernels through different host launchers.** Probing
  "Helion-emit_pipeline vs hand-written" via two different launch
  paths (one via Helion's `_default_pallas_pipeline_launcher`, the
  other via raw `pl.pallas_call`) mixes launcher overhead with kernel
  perf. Either rewrite both kernels for raw `pl.pallas_call` and
  measure, or wrap both through identical Helion plumbing. §2.1 (c)
  separated structure from launcher by issuing both via raw
  `pl.pallas_call`.

- **Re-creating the pallas_call inside the timed lambda.** A common
  bench bug: `_bench(lambda: pl.pallas_call(...)(x, y))` re-traces and
  re-compiles the kernel on every timed call (~140 ms per call!).
  Cache the `jax.jit(pl.pallas_call(...))` outside the lambda; the
  lambda should only invoke the already-jitted function. Caught in the
  Deep Replan 2026-05-23 G2 closure (v1 vs v2 ablation scripts).

- **Trusting forced configs without numerical validation.** The
  cross-shape sweep in §2.5 timed forced ``outer_grid`` configs on
  M=1 shapes and got "fast" numbers — but the outputs were silently
  wrong. The autotuner's accuracy check protects autotuned configs;
  forced ones are unprotected. Always validate forced-config outputs
  against a CPU f32 reference at deep-replan time (see
  ``.deep_replan_validate_og_m1.py``) before drawing conclusions
  from "this forced config is X us faster".

- **Eligibility checks that match on structural shape alone.** The
  G2-H ``_codegen_outer_grid_or_fallback`` eligibility check looked
  at "single inner block_id + loop-carried state + non-reduction
  outer pids" — all structural properties of the device IR. It
  missed that M=1 (i.e. ``bm == 1``) breaks the body rewrite's
  assumption that the loop-carried scratch holds a partial matmul
  accumulator. Eligibility checks for body rewrites must include
  **block-size sanity** (``bm > 1``, ``bn > 1`` etc.) when the
  rewrite assumes 2D vector operations.

- **Knob ablations dominated by JIT compile.** When ablating
  ``pl.dot precision`` / ``dimension_semantics`` etc. on bf16
  1024^3, the per-call latency is ~125-180 us but the per-call
  compile cost is ~140 ms. If the harness pays compile per call,
  the ablation table is just compile-time noise. Always cache the
  jitted kernel and warm up at least 2x before the timed loop.

- **Lessons from the redundant-pids find (§2.6 (a)).** Generated
  code that "looks clean" can still emit dead variable reads when
  the codegen emits all outer pids regardless of body references.
  ~5% headline gain available by DCE'ing them. Whenever a codegen
  pass writes "boilerplate" outer-grid setup statements
  unconditionally, audit whether the body actually reads them
  before declaring the path optimised.
