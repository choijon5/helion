"""Single-shape headline measurements for bf16/f32 matmul (default bf16 1024x1024x1024).

Cycle-18 autotuner-seed refactor (manager directive 2026-05-23): the
prior pinned-config path (``_PINNED_KERNEL_ONLY_CONFIGS``) was a
measurement crutch that bypassed the autotuner entirely. Real Helion
users never get the pinned configs — only our measurement did, so
closing G2 / G3-A on the pinned medians measured a *kernel-quality
ceiling*, not the *real-user* experience. The fix: seed the autotuner
deterministically (``HELION_AUTOTUNE_RANDOM_SEED``) so every measurement
run picks the same config. The autotuner-picked config is what users
get; the seed makes it reproducible; the remaining per-sweep variance
is only chip thermal noise.

This script emits **three** kernel-only measurements per shape (Helion
plus two references) in addition to the Helion full-path number, all
at the autotuner-picked config (pinning is gone):

1. **Real-user full-path** (production behavior): Helion via
   ``torch_tpu`` end-to-end at the autotuner-picked config. Includes
   launcher overhead AND torch_tpu C++ dispatch.
2. **Real-user kernel-only** (gating since cycle 18 for G2/G3/G4;
   diagnostic-only for G5): Helion's generated Pallas kernel pulled
   out of the launcher cache at the *autotuner-picked* config and
   invoked through ``jax.jit(pl.pallas_call(...))`` with JAX arrays,
   identical to the hand-written Pallas reference. Isolates the
   kernel body from launcher / torch_tpu dispatch overhead.
3. **Pallas reference kernel-only**: ``pallas_matmul`` from
   ``matmul_pallas.py`` called through the same ``jax.jit`` path.
4. **JAX reference kernel-only**: a jitted ``jnp.matmul(x_jax,
   y_jax)`` called through the same ``jax.jit`` path. Same dispatch,
   same JAX inputs — the only difference is the kernel body (XLA's
   matmul lowering vs Helion's emitted Pallas kernel vs hand-written
   Pallas). G5 gates on ``full_path_H_over_J`` (this JAX us / Helion
   full-path us — matches the H/P convention where the ratio is
   ``reference_us / helion_us`` so ≥ 1.00 means Helion is ≥ as fast
   as the reference); ``kernel_only_H_over_J`` is a diagnostic split
   that tells the substep menu whether the gap is kernel-side or
   launcher-side.

CLI:

```
python measure_headline.py                    # default bf16 1024x1024x1024
python measure_headline.py --shape 1024 128 1024
python measure_headline.py --dtype float32 --shape 1024 1024 1024   # G4 f32 path
python measure_headline.py --seed 7           # override default seed (0)
python measure_headline.py --timing-mode interleaved                 # DR#6 canonical
```

Output (parseable lines, one per metric):

```
helion_bf16_<M>x<K>x<N>: median=<us> us                                                  # back-compat full-path
helion_full_path_<M>x<K>x<N> [autotuner pick: <config>, seed=<n>]: median=<us> us
helion_kernel_only_<M>x<K>x<N> [autotuner pick: <config>, seed=<n>]: median=<us> us       # HP-leg median in interleaved mode
helion_kernel_only_hj_<M>x<K>x<N> [autotuner pick: <config>, seed=<n>]: median=<us> us    # HJ-leg Helion median (divisor of kernel_only_H_over_J)
pallas_kernel_only_<M>x<K>x<N>: median=<us> us
jax_kernel_only_<M>x<K>x<N>: median=<us> us
full_path_H_over_P: <ratio>
kernel_only_H_over_P: <ratio>             # GATING for G2/G3/G4
full_path_H_over_J: <ratio>               # GATING for G5 (provisional; G5-methodology follow-up — see plan.md §5)
kernel_only_H_over_J: <ratio>             # diagnostic for G5 substep selection (kernel vs launcher lever)
kernel_only_P_over_J: <ratio>             # tracking — hand-written Pallas vs JAX baseline
launcher_overhead_us: <full - kernel> us  # Helion-internal launcher overhead vs HP-leg kernel-only
launcher_overhead_vs_jax_us: <full - jax> us  # Helion full-path overhead vs JAX (G5 substep target)
```

The kernel-only path is built by patching
``helion.runtime._pallas_build_callable`` to stash the JAX ``jit_fn``
argument (``pl.pallas_call(...)``) right before Helion wraps it in
torch_tpu's ``JaxCallable`` (which throws away the original ``jit_fn``
reference by ``jax.export``-serializing the body into a binary blob
bound to ``call_custom_kernel``). The captured ``jit_fn`` is re-wrapped
in ``jax.jit`` to match the JaxCallable construction site, and we time
it directly with JAX inputs — identical to the way the hand-written
``pallas_matmul`` reference is called.

Interleaved timing: the canonical DR#6 methodology pairs every Helion
call with a reference call inside the same ``time.perf_counter_ns()``
window so chip-thermal drift cancels in the ratio. To stay
apples-to-apples with the G2/G3/G4 closure data (DR#6 2-way), the
paired variant (``_time_interleaved_paired``) runs **two** 2-way
interleaved legs back-to-back: Helion-vs-Pallas (for H/P) and
Helion-vs-JAX (for H/J). A single 3-way Helion → Pallas → JAX cycle
would systematically depress H/P by ~1-2% on some shapes (longer
per-iteration window picks up scheduler state from JAX wind-down) and
break the apples-to-apples comparison with the cycle 21-24 H/P
closures. Cost is ~2× per sweep — still within the per-shape budget.

Timing convention matches the production harness everywhere: 20 iters
x 5 repeats, warmup excluded, ``synchronize_device`` (or
``jax.block_until_ready``) between calls.

Known harness limitation (M=1 shapes, see plan.md §6.5): for some
autotuner-picked configs on ``M=1, N=1024`` shapes (e.g. block sizes
``[1, *, 1024]``), the kernel-only replay through
``jax.jit(pl.pallas_call(...))`` raises a Mosaic divisibility error
("the last two dimensions of your block shape are divisible by 8 and
128 respectively") because the harness re-issues the cached
``jit_fn`` directly with JAX inputs without applying the launcher's
runtime padding (``_pallas_apply_ds_padding`` lives in Helion's
production launcher path, not in the cached ``jit_fn``). The script
itself does NOT catch this error — it raises and exits with a
non-zero status. The production full-path launcher handles these
configs correctly so real users see no error; outer sweep harnesses
(e.g. shell loops that retry per shape) are expected to tolerate the
crash and aggregate the surviving sweeps' medians. Affected shapes
typically drop 2-3 of 5 sweeps in a multi-sweep baseline.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
import timeit
from typing import Callable

# Force full autotuning effort + seed the autotuner deterministically;
# set before importing helion so the values are picked up at autotuner
# initialization. Cycle-18 reproducibility refactor (manager directive
# 2026-05-23): ``HELION_AUTOTUNE_RANDOM_SEED`` pins the random sampling
# trajectory through config space; the CLI ``--seed`` flag overrides
# the default 0 for multi-seed sweeps (real-user H/P distribution).
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "full")
os.environ.setdefault("HELION_AUTOTUNE_RANDOM_SEED", "0")

import jax
import jax.numpy as jnp
import numpy as np
import torch

from helion._testing import DEVICE
from helion.autotuner.benchmarking import synchronize_device

# Import the kernel from matmul_helion so any kernel-side change is picked
# up by both the full harness and this single-shape probe.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matmul_helion import helion_matmul_kernel  # pyrefly: ignore [missing-import]
from matmul_pallas import pallas_matmul  # pyrefly: ignore [missing-import]

# Default (headline) shape for back-compat with the legacy single-shape probe.
_DEFAULT_SHAPE: tuple[int, int, int] = (1024, 1024, 1024)
# Default autotuner seed mirrored from the env var setdefault above.
_DEFAULT_AUTOTUNE_SEED = int(os.environ["HELION_AUTOTUNE_RANDOM_SEED"])
# CLI ``--dtype`` choices map to ``(torch dtype, jax dtype)``. bf16 is the
# default for back-compat with cycles 15-22 invocations; float32 is the G4
# f32 path (TPU MXU has no f32 shortcut, so Helion routes through
# ``lax.dot_general(precision=HIGHEST)``).
_DTYPE_CHOICES = {
    "bfloat16": (torch.bfloat16, jnp.bfloat16),
    "float32": (torch.float32, jnp.float32),
}


# Module-level slot populated by ``_install_jit_fn_capture`` whenever
# Helion's runtime calls ``_pallas_build_callable``. After Helion finishes
# its autotune + first-call cycle, we lift the most recently captured
# ``jit_fn`` out of this slot for the kernel-only path. The capture has to
# happen *before* ``_pallas_build_callable`` wraps the ``jit_fn`` in
# torch_tpu's ``JaxCallable`` because ``JaxCallable`` does not retain a
# reference to the underlying jitted function -- it ``jax.export.export``s
# it into a serialized blob and routes everything through
# ``tpu_torch_pallas.call_custom_kernel``.
_CAPTURED_HELION_JIT_FN: Callable[..., object] | None = None


def _install_jit_fn_capture() -> None:
    """Patch ``helion.runtime._pallas_build_callable`` to stash its
    ``jit_fn`` argument so we can re-time it as a pure-JAX call.

    Idempotent: re-importing this module re-patches but only the latest
    wrapping is active. The patch is local to the probe script -- the
    Helion runtime is unmodified.
    """
    import helion.runtime as helion_runtime

    if getattr(helion_runtime, "_pallas_build_callable_orig", None) is not None:
        return  # already patched

    orig_build_callable = helion_runtime._pallas_build_callable  # type: ignore[attr-defined]
    helion_runtime._pallas_build_callable_orig = orig_build_callable  # type: ignore[attr-defined]

    def _wrapped_build_callable(*args: object, **kwargs: object) -> object:
        global _CAPTURED_HELION_JIT_FN
        # ``_pallas_build_callable``'s positional signature is
        # ``(pallas_kernel, grid, jit_fn, _output_indices, ...)``; the
        # third positional arg is the JAX callable we want.
        if len(args) >= 3:
            jit_fn = args[2]
            # Wrap in ``jax.jit`` here (mirrors what
            # ``_pallas_build_callable`` passes to torch_tpu's
            # ``JaxCallable``: ``jit_fn=jax.jit(jit_fn)``).
            _CAPTURED_HELION_JIT_FN = jax.jit(jit_fn)  # pyrefly: ignore[bad-argument-type]
        return orig_build_callable(*args, **kwargs)  # pyrefly: ignore[bad-argument-type, missing-argument]

    helion_runtime._pallas_build_callable = _wrapped_build_callable  # type: ignore[attr-defined]


def _find_helion_jit_fn() -> Callable[..., object]:
    """Return the most recently captured Helion-side ``jit_fn``.

    Raises if the capture patch was never installed or Helion never built
    a callable (e.g. autotune skipped or the kernel went through the
    interpret-mode path that bypasses ``_pallas_build_callable``).
    """
    if _CAPTURED_HELION_JIT_FN is None:
        raise RuntimeError(
            "No Helion ``jit_fn`` was captured. Make sure "
            "``_install_jit_fn_capture`` is called before binding the "
            "kernel, and that the kernel runs through the production "
            "(non-interpret) Pallas launcher path."
        )
    return _CAPTURED_HELION_JIT_FN


def _reset_capture() -> None:
    """Clear the captured ``jit_fn`` so the next ``_pallas_build_callable``
    call repopulates it.

    Used by ``_refresh_capture_for_compiled_fn`` to guarantee the post-
    autotune capture is the one that corresponds to the autotuner-picked
    config, not whatever the last autotuner trial happened to build.
    """
    global _CAPTURED_HELION_JIT_FN
    _CAPTURED_HELION_JIT_FN = None


def _refresh_capture_for_compiled_fn(
    compiled_fn: Callable[..., object],
    *invocation_args: object,
) -> None:
    """Refresh the captured ``jit_fn`` so it matches the launcher built by
    ``compiled_fn`` (the autotuner-picked config).

    The autotuner evaluates many configs, each triggering
    ``_pallas_build_callable`` and overwriting ``_CAPTURED_HELION_JIT_FN``.
    By the time ``bound.compile_config(best_config)`` returns, the
    callable's underlying ``pallas_kernel`` already has a populated
    ``_pallas_cache`` (because the autotuner exercised that exact
    Python module while ranking), so the subsequent first call of
    ``compiled_fn`` hits the cache and does NOT re-invoke
    ``_pallas_build_callable``. The result is a kernel-only timing that
    references the LAST autotuner trial's ``jit_fn`` -- a completely
    unrelated config in the typical case -- inflating noise and
    occasionally pinning H/P far below 1.00 even when the picked kernel
    is competitive.

    Fix: walk the module that holds ``compiled_fn`` (returned from
    ``PyCodeCache.load`` inside ``BoundKernel.compile_config``), find each
    inner ``pallas_kernel`` Python function, and null its
    ``_pallas_cache`` attribute. Then invoke ``compiled_fn`` once with
    the supplied torch args so the launcher rebuilds via
    ``_pallas_build_callable`` -- our capture wrapper observes that call
    and stores the correct ``jit_fn``.

    Safe to call multiple times; idempotent after a fresh autotune.
    """
    mod = inspect.getmodule(compiled_fn)
    if mod is None:  # pragma: no cover - defensive
        raise RuntimeError(
            "Could not locate the module holding the compiled callable to "
            "invalidate the stale launcher cache."
        )

    # The three Pallas launcher families
    # (``default_pallas_launcher`` / ``default_pallas_pipeline_launcher`` /
    # ``default_pallas_fori_launcher``) each maintain their own per-
    # ``pallas_kernel`` cache attribute. Clear whichever ones are populated
    # so the next call rebuilds via ``_pallas_build_callable``.
    cache_attrs = ("_pallas_cache", "_pallas_pipeline_cache", "_pallas_fori_cache")
    cleared = 0
    for attr_name in dir(mod):
        if attr_name.startswith(("_default_", "__")):
            continue
        candidate = getattr(mod, attr_name)
        for cache_attr in cache_attrs:
            if (
                hasattr(candidate, cache_attr)
                and getattr(candidate, cache_attr) is not None
            ):
                setattr(candidate, cache_attr, None)
                cleared += 1

    _reset_capture()
    compiled_fn(*invocation_args)
    synchronize_device(invocation_args[0])

    if _CAPTURED_HELION_JIT_FN is None:  # pragma: no cover - defensive
        raise RuntimeError(
            "Launcher cache invalidation did not trigger the capture "
            f"wrapper (cleared {cleared} cache entry/entries). The chosen "
            "kernel may be on an interpret path or "
            "``_install_jit_fn_capture`` was not installed early enough."
        )


def _time(fn: Callable[[], object], n_iter: int = 20, n_repeats: int = 5) -> float:
    """Return the median per-call latency in microseconds.

    Same shape as the production harness (``matmul_helion.py`` /
    ``matmul_pallas.py``): ``n_iter`` calls inside each ``timeit.repeat``
    sample, ``n_repeats`` samples, take the median. Five warmup calls
    instead of one because JAX's lazy compilation on the first call (and
    XLA caching warmup on the second) can otherwise leak into the first
    timed iteration -- particularly visible on the kernel-only Pallas
    reference path where the first ``pallas_matmul`` call also pays
    ``functools.partial(jax.jit, static_argnames=...)`` cache build.
    """
    for _ in range(5):
        fn()
    samples = np.array(timeit.repeat(fn, repeat=n_repeats, number=n_iter)) / n_iter
    return float(np.median(samples)) * 1e6


def _time_interleaved(
    fn_helion: Callable[[], object],
    fn_pallas: Callable[[], object],
    n_iter: int = 20,
    n_repeats: int = 5,
) -> tuple[float, float]:
    """Paired-sample timing of Helion vs Pallas. Returns (helion_us, pallas_us).

    For each of ``n_iter * n_repeats`` iterations, we time one Helion
    call immediately followed by one Pallas call via
    ``time.perf_counter_ns()``. Each call has a per-call ``block_until_ready``
    so the measured window is the full host+device latency, matching the
    sequential ``_time`` semantics. We then take the median across all
    ``n_iter * n_repeats`` per-call samples per side.

    Rationale: per-call thermal / scheduler noise correlates across the
    pair (both calls run in the same ~microsecond window), so the H/P
    ratio is dominated by kernel quality rather than drift. The sequential
    form measures Helion and Pallas in fully separate windows; if pod
    temperature drifts between the two windows, the ratio absorbs the
    drift.

    Warmup: 5 paired warmup iterations (same as sequential ``_time``).
    """
    for _ in range(5):
        fn_helion()
        fn_pallas()

    helion_samples_ns: list[int] = []
    pallas_samples_ns: list[int] = []
    total = n_iter * n_repeats
    for _ in range(total):
        t0 = time.perf_counter_ns()
        fn_helion()
        t1 = time.perf_counter_ns()
        fn_pallas()
        t2 = time.perf_counter_ns()
        helion_samples_ns.append(t1 - t0)
        pallas_samples_ns.append(t2 - t1)

    helion_us = float(np.median(np.array(helion_samples_ns))) / 1000.0
    pallas_us = float(np.median(np.array(pallas_samples_ns))) / 1000.0
    return helion_us, pallas_us


def _time_interleaved_paired(
    fn_helion: Callable[[], object],
    fn_pallas: Callable[[], object],
    fn_jax: Callable[[], object],
    n_iter: int = 20,
    n_repeats: int = 5,
) -> tuple[float, float, float, float]:
    """Paired-sample timing of Helion vs Pallas (H/P) and Helion vs JAX (H/J).

    Returns ``(helion_p_us, pallas_us, helion_j_us, jax_us)`` — two
    independent Helion medians, one paired against Pallas (for H/P
    tracking) and one paired against JAX (for the H/J diagnostic that
    feeds G5 substep selection). Each pair runs in its own
    ``time.perf_counter_ns()`` window pattern (Helion → reference →
    Helion → reference → ...), so the H/P ratio matches the canonical
    DR#6 2-way methodology that G2/G3/G4 closed under, and the H/J
    ratio uses the same 2-way pattern apples-to-apples.

    Rationale for two-pass-pair rather than one 3-way window: a 3-way
    interleaved cycle (Helion → Pallas → JAX) systematically depresses
    the H/P ratio by ~1-2% on some shapes because the per-iteration
    cycle is longer (Helion's window picks up scheduler state from
    JAX's wind-down). The G2/G3/G4 closure relies on the DR#6 2-way
    methodology; preserving that for H/P keeps the gate comparison
    consistent across cycles. Cost is ~2× longer per sweep
    (two interleaved windows back-to-back vs one), still well within
    the per-shape sweep budget.

    Methodology gap (see plan.md §5 G5 "G5-methodology" follow-up):
    the ``full_path_H_over_J`` ratio that G5 gates on uses
    ``helion_full_us`` measured by ``_time(_run_full_path)`` in an
    isolated sequential window OUTSIDE this paired helper, then
    divides by ``jax_us`` from the HJ leg below. The two numbers come
    from different windows so between-window thermal drift is NOT
    canceled in the gate ratio (it IS canceled in the kernel H/J
    ratio because both numerator and denominator come from the HJ
    leg). The drift is empirically small (sequential us spread ~5-15
    us at the headline; the HJ leg's JAX us is within ~3us of a
    standalone sequential JAX measurement, well below the ~40us
    launcher-overhead signal G5 substeps target) — but it's a real
    methodology asymmetry that should be closed by adding a third
    paired leg (Helion-full-path vs JAX). A probe of the third leg
    showed ``helion_full_us`` rises from ~161 → ~192 us in the paired
    window because the JAX call inherits scheduler state from
    helion-full's wind-down, so the paired-leg full H/J ratio is
    more pessimistic than the sequential / paired-kernel mix. Closing
    this requires re-measuring the whole 14-shape G5-setup baseline.

    Warmup: 5 paired warmup iterations per leg (10 total).
    """
    helion_p_us, pallas_us = _time_interleaved(
        fn_helion, fn_pallas, n_iter=n_iter, n_repeats=n_repeats
    )
    helion_j_us, jax_us = _time_interleaved(
        fn_helion, fn_jax, n_iter=n_iter, n_repeats=n_repeats
    )
    return helion_p_us, pallas_us, helion_j_us, jax_us


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Helion (real-user full-path + real-user kernel-only) "
            "vs hand-written Pallas for a single matmul shape (bf16 default, "
            "f32 via --dtype). Defaults to the bf16 1024x1024x1024 headline. "
            "The Helion autotuner is seeded via "
            "``HELION_AUTOTUNE_RANDOM_SEED`` so every run picks the same "
            "config — real-user metric, reproducible."
        )
    )
    parser.add_argument(
        "--shape",
        nargs=3,
        type=int,
        metavar=("M", "K", "N"),
        default=list(_DEFAULT_SHAPE),
        help="Matmul shape as three ints: M K N (default: 1024 1024 1024).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Autotuner random seed. Defaults to 0 (or whatever the "
            "``HELION_AUTOTUNE_RANDOM_SEED`` env var is set to BEFORE "
            "the script is imported). Overriding here re-sets the env "
            "var, but only takes effect if the autotuner reads the seed "
            "at autotune time rather than import time — kept as a CLI "
            "for multi-seed sweep harnesses."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(_DTYPE_CHOICES.keys()),
        default="bfloat16",
        help=(
            "Element dtype for both Helion and the Pallas reference. "
            "Defaults to ``bfloat16`` (back-compat with cycles 15-22 "
            "invocations). ``float32`` opts into the G4 f32 path which "
            "has no MXU shortcut — Helion routes through "
            "``lax.dot_general(precision=HIGHEST)`` and the Pallas "
            "reference does the same."
        ),
    )
    parser.add_argument(
        "--timing-mode",
        choices=("sequential", "interleaved", "both"),
        default="sequential",
        help=(
            "Kernel-only timing methodology. ``sequential`` (default, "
            "back-compat with cycles 15-19) measures Helion and Pallas "
            "in separate timing windows. ``interleaved`` pairs each "
            "Helion call with a Pallas call inside the same per-call "
            "``time.perf_counter_ns()`` window, so chip-thermal drift "
            "cancels in the H/P ratio. ``both`` runs each mode in "
            "sequence and prints both result blocks (DR#6 methodology "
            "comparison probe). Full-path Helion timing is unaffected "
            "by this flag (it always uses the sequential form)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    m, k, n = args.shape
    torch_dtype, jax_dtype = _DTYPE_CHOICES[args.dtype]

    # If CLI seed was given, re-set the env var. ``Settings`` reads the
    # env var via ``_get_autotune_random_seed`` at the point the
    # ``Settings`` instance is constructed (typically the first time
    # ``@helion.kernel`` decorates a function), so this override is
    # effective whenever the kernel hasn't already been bound. Setting
    # it before ``bound = helion_matmul_kernel.bind(...)`` is the
    # safest place — ``Settings`` for ``helion_matmul_kernel`` is
    # already constructed at import time, but ``base_search.prepare()``
    # uses ``self.settings.autotune_random_seed`` which is read from
    # the live env if not pre-cached. For belt-and-suspenders we set
    # both the env var AND seed the Python global ``random`` here.
    seed = args.seed if args.seed is not None else _DEFAULT_AUTOTUNE_SEED
    os.environ["HELION_AUTOTUNE_RANDOM_SEED"] = str(seed)

    torch.manual_seed(0)
    x_torch = torch.randn((m, k), dtype=torch_dtype, device=DEVICE)
    torch.manual_seed(1)
    y_torch = torch.randn((k, n), dtype=torch_dtype, device=DEVICE)

    # Install the ``_pallas_build_callable`` capture so we can lift the
    # autotuner-picked ``jit_fn`` (real-user kernel-only) out of Helion's
    # cache.
    _install_jit_fn_capture()

    # ----- Helion full-path: bind + autotune + compile + run (production path).
    bound = helion_matmul_kernel.bind((x_torch, y_torch))
    # Force a fresh autotune at the seeded random state so seed determines
    # the pick (``force=True`` bypasses the autotune cache). The kernel's
    # ``Settings`` already read ``HELION_AUTOTUNE_RANDOM_SEED`` from the
    # env at decoration time; setting the env var above is a no-op for
    # the already-constructed Settings instance. The autotuner reads
    # ``settings.autotune_random_seed`` from its own Settings reference,
    # which IS the same Settings instance — so to actually override it
    # mid-process we patch the live settings here.
    bound.kernel.settings.autotune_random_seed = seed  # type: ignore[attr-defined]
    best_config = bound.autotune((x_torch, y_torch), force=True)
    print(
        f"Optimal autotuned config (seed={seed}): {best_config}",
        file=sys.stderr,
    )
    compiled_fn = bound.compile_config(best_config, allow_print=False)

    # The autotuner-driven path leaves the launcher cache populated by
    # whichever trial last compiled this exact ``pallas_kernel``; the
    # capture wrapper therefore points at that trial's ``jit_fn`` rather
    # than the one we actually picked. Invalidate the stale cache and
    # re-run once so the launcher rebuilds via ``_pallas_build_callable``
    # and the capture refreshes to the chosen config's ``jit_fn``. This
    # has to happen BEFORE ``_time(_run_full_path)`` so the full-path
    # measurement is also against the freshly-built launcher (matches the
    # production first-call cost) and so the kernel-only path that
    # follows lifts the correct ``jit_fn`` out of the capture slot.
    _refresh_capture_for_compiled_fn(compiled_fn, x_torch, y_torch)

    def _run_full_path() -> None:
        out = compiled_fn(x_torch, y_torch)
        synchronize_device(out)

    helion_full_us = _time(_run_full_path)
    # Back-compat line for older log scrapers. The ``helion_bf16_…`` name
    # is preserved unchanged when ``--dtype bfloat16`` (default) so cycles
    # 15-22 log scrapers keep parsing; ``--dtype float32`` emits
    # ``helion_float32_…`` so the dtype is parseable downstream.
    back_compat_dtype_tag = "bf16" if args.dtype == "bfloat16" else args.dtype
    print(f"helion_{back_compat_dtype_tag}_{m}x{k}x{n}: median={helion_full_us:.2f} us")
    print(
        f"helion_full_path_{m}x{k}x{n} "
        f"[autotuner pick: {best_config}, seed={seed}]: "
        f"median={helion_full_us:.2f} us"
    )

    # Snapshot the autotuner-picked ``jit_fn`` (captured during the
    # ``compile_config(best_config)`` above) for the kernel-only
    # measurement.
    autotuned_jit_fn = _find_helion_jit_fn()

    # ----- Materialise JAX inputs once for the kernel-only timings.
    # The cached pallas_call's input refs are ordered as
    # ``[tensor_inputs..., output_only_refs...]``; Helion's matmul has no
    # output-only input that needs a host-side buffer here -- the kernel
    # body's ``out = torch.empty(...)`` lowers to a pallas_call output
    # buffer, so ``jit_fn`` has signature ``jit_fn(x_jax, y_jax) -> z_jax``.
    # Materialise inputs on the TPU device the same way the Pallas
    # reference path does (``jax.random.normal`` -> ``jax.device_put`` -> jit).
    # Values are arbitrary; only the shape / dtype affect timing.
    key0, key1 = jax.random.split(jax.random.PRNGKey(0))
    x_jax = jax.device_put(jax.random.normal(key0, (m, k), dtype=jax_dtype))
    y_jax = jax.device_put(jax.random.normal(key1, (k, n), dtype=jax_dtype))
    jax.block_until_ready((x_jax, y_jax))

    # ----- Helion kernel-only: time the autotuner-picked ``jit_fn``
    # (what real users get) through the pure-JAX path.
    def _run_helion_kernel_only() -> None:
        out = autotuned_jit_fn(x_jax, y_jax)
        jax.block_until_ready(out)

    # ----- Pallas reference kernel-only: hand-written ``pallas_matmul`` via
    # the same pure-JAX path. ``pallas_matmul`` is already ``@jax.jit``
    # (``static_argnames=["bm", "bk", "bn"]``), so calling it directly is
    # apples-to-apples with Helion's cached ``jax.jit(pl.pallas_call(...))``.
    # ``pallas_matmul`` clamps each block to ``min(dim, requested_block)``,
    # so passing 512 across the board gives the hand-written best block for
    # the headline shape and the natural cap for skinny / shallow shapes.
    def _run_pallas_kernel_only() -> None:
        out = pallas_matmul(x_jax, y_jax, bm=512, bk=512, bn=512)
        jax.block_until_ready(out)

    # ----- JAX reference kernel-only: ``jnp.matmul`` jitted once outside the
    # timed loop and called with the same JAX inputs. JAX lowers to XLA's
    # hand-tuned matmul (no Pallas in the path); this is the G5 baseline.
    # Using ``jax.jit`` here matches the apples-to-apples dispatch path of
    # the Pallas + Helion kernel-only references.
    #
    # Precision: for f32 inputs, route through ``jax.lax.dot_general``
    # with ``precision=Precision.HIGHEST`` so JAX does the full f32
    # multiply (matching Helion's ``lax.dot_general(precision=HIGHEST)``
    # f32 path and the ``pl.dot(precision=HIGHEST)`` Pallas reference in
    # ``matmul_pallas.py``). Without this override, ``jnp.matmul``
    # defaults to ``Precision.DEFAULT``, which silently bf16-rounds f32
    # multiplications on TPU — apples-to-oranges with Helion / Pallas.
    # bf16/fp16 inputs go through the default ``jnp.matmul`` path
    # because the MXU is already f32-accumulating for those dtypes (no
    # precision knob needed).
    if jax_dtype == jnp.float32:

        def _jax_matmul_highest(x: jax.Array, y: jax.Array) -> jax.Array:
            return jax.lax.dot_general(
                x,
                y,
                dimension_numbers=(((1,), (0,)), ((), ())),
                precision=jax.lax.Precision.HIGHEST,
            )

        jax_matmul_jit = jax.jit(_jax_matmul_highest)
    else:
        jax_matmul_jit = jax.jit(jnp.matmul)

    def _run_jax_kernel_only() -> None:
        out = jax_matmul_jit(x_jax, y_jax)
        jax.block_until_ready(out)

    timing_modes = (
        ("sequential", "interleaved")
        if args.timing_mode == "both"
        else (args.timing_mode,)
    )
    # When only one mode is requested, omit the per-line tag so the
    # printed format stays bit-identical with cycles 15-19 log scrapers
    # (back-compat). When both modes run, prepend a ``[sequential]`` /
    # ``[interleaved]`` tag so each block is parseable separately.
    tag_lines = len(timing_modes) > 1

    for mode in timing_modes:
        if mode == "sequential":
            helion_kernel_us = _time(_run_helion_kernel_only)
            pallas_kernel_us = _time(_run_pallas_kernel_only)
            jax_kernel_us = _time(_run_jax_kernel_only)
            # Sequential mode reports a single Helion us (one timing window),
            # which both ratios divide. Set both Helion legs to the same value
            # so the downstream printing logic doesn't need to special-case
            # the mode. ``helion_full_us`` keeps the sequentially-measured
            # value from outside the loop; ``jax_full_us`` is the same as
            # ``jax_kernel_us`` (JAX has no separate launcher — full path
            # IS kernel-only for JAX).
            helion_kernel_us_for_p = helion_kernel_us
            helion_kernel_us_for_j = helion_kernel_us
            jax_full_us = jax_kernel_us
            mode_tag = "[sequential] " if tag_lines else ""
        else:  # interleaved (canonical 2-way × 2 paired legs: HP and HJ)
            (
                helion_kernel_us_for_p,
                pallas_kernel_us,
                helion_kernel_us_for_j,
                jax_kernel_us,
            ) = _time_interleaved_paired(
                _run_helion_kernel_only,
                _run_pallas_kernel_only,
                _run_jax_kernel_only,
            )
            # Report Helion as the HP-leg median (the leg that pairs with
            # Pallas and matches the canonical DR#6 2-way invariant that
            # G2/G3/G4 closed under). The HJ leg's Helion us is typically
            # within ~0.5us of the HP leg's value on the same shape; the
            # reconstructible-ratio property requires reporting ONE leg's
            # us as the printed median (so ``pallas_us / helion_us``
            # equals the printed H/P), not a mean of two medians which
            # would break the median claim and the reconstruction. The
            # per-ratio Helion us in the divisions below still uses the
            # per-leg values to keep H/P apples-to-apples with the HP
            # leg and H/J apples-to-apples with the HJ leg.
            #
            # ``helion_full_us`` stays at the sequential value measured
            # before the loop (see ``_time_interleaved_paired`` docstring's
            # methodology-gap note); for JAX, "full path" and "kernel only"
            # are the same path so ``jax_full_us`` equals ``jax_kernel_us``.
            helion_kernel_us = helion_kernel_us_for_p
            jax_full_us = jax_kernel_us
            mode_tag = "[interleaved] " if tag_lines else ""

        # Pre-shape tag (after the shape suffix, before the rest of the
        # line) preserves the bit-identical legacy single-mode output;
        # tagged output for the both-mode comparison adds the tag suffix
        # to the metric name itself so each metric is uniquely parseable
        # (``kernel_only_H_over_P_sequential`` / ``..._interleaved``).
        ratio_tag = f"_{mode}" if tag_lines else ""
        line_tag = f" {mode_tag.strip()}" if tag_lines else ""
        # In interleaved mode the printed ``helion_kernel_only_*`` median is
        # the HP-leg's Helion median (paired with Pallas). Downstream log
        # scrapers can reconstruct ``kernel_only_H_over_P`` exactly as
        # ``pallas_kernel_only / helion_kernel_only``. For H/J
        # reconstruction we also emit ``helion_kernel_only_hj_*`` carrying
        # the HJ-leg's Helion median (the divisor used in the printed
        # ``kernel_only_H_over_J`` ratio); ``jax_kernel_only /
        # helion_kernel_only_hj`` recovers the printed H/J. In sequential
        # mode both lines carry the same single-window value.
        print(
            f"helion_kernel_only_{m}x{k}x{n}{line_tag} "
            f"[autotuner pick: {best_config}, seed={seed}]: "
            f"median={helion_kernel_us:.2f} us"
        )
        print(
            f"helion_kernel_only_hj_{m}x{k}x{n}{line_tag} "
            f"[autotuner pick: {best_config}, seed={seed}]: "
            f"median={helion_kernel_us_for_j:.2f} us"
        )
        print(
            f"pallas_kernel_only_{m}x{k}x{n}{line_tag}: "
            f"median={pallas_kernel_us:.2f} us"
        )
        print(f"jax_kernel_only_{m}x{k}x{n}{line_tag}: median={jax_kernel_us:.2f} us")

        # ----- Derived ratios.
        # H/P uses the Helion us paired with Pallas (DR#6 canonical); H/J
        # uses the Helion us paired with JAX. In sequential mode both
        # Helion legs share the same value, so the ratio math is
        # identical to the legacy 2-way path.
        #
        # Full-path ratios use the single Helion full-path us measured once
        # outside the kernel-only loop (the same value drives both
        # ``full_path_H_over_P`` and ``full_path_H_over_J``). For JAX,
        # "full path" and "kernel only" are identical (no torch_tpu /
        # Helion launcher in the path), so ``jax_kernel_us`` doubles as the
        # JAX full-path baseline. ``launcher_overhead_vs_jax_us`` is the
        # raw delta between Helion full-path and JAX — the absolute gap a
        # G5 launcher-side substep has to close on shapes where the
        # kernel is already fast enough. The methodology asymmetry
        # between sequential ``helion_full_us`` and paired
        # ``jax_kernel_us`` is documented in ``_time_interleaved_paired``'s
        # docstring (see plan.md §5 G5 "G5-methodology" follow-up).
        full_h_over_p = pallas_kernel_us / helion_full_us
        kernel_h_over_p = pallas_kernel_us / helion_kernel_us_for_p
        full_h_over_j = jax_full_us / helion_full_us
        kernel_h_over_j = jax_kernel_us / helion_kernel_us_for_j
        kernel_p_over_j = jax_kernel_us / pallas_kernel_us
        # ``launcher_overhead_us`` is the Helion-internal launcher overhead:
        # full-path Helion minus kernel-only Helion. In interleaved mode
        # ``helion_kernel_us`` is the HP-leg's median (see the
        # interleaved branch above), so this is "sequential full-path
        # minus HP-leg kernel-only median". The HJ leg's kernel-only
        # median is typically within ~0.5us; using the HP leg keeps
        # this metric directly comparable to the H/P ratio's
        # denominator.
        launcher_overhead_us = helion_full_us - helion_kernel_us
        launcher_overhead_vs_jax_us = helion_full_us - jax_full_us

        print(f"full_path_H_over_P{ratio_tag}: {full_h_over_p:.3f}")
        print(f"kernel_only_H_over_P{ratio_tag}: {kernel_h_over_p:.3f}")
        print(f"full_path_H_over_J{ratio_tag}: {full_h_over_j:.3f}")
        print(f"kernel_only_H_over_J{ratio_tag}: {kernel_h_over_j:.3f}")
        print(f"kernel_only_P_over_J{ratio_tag}: {kernel_p_over_j:.3f}")
        print(f"launcher_overhead_us{ratio_tag}: {launcher_overhead_us:.2f} us")
        print(
            f"launcher_overhead_vs_jax_us{ratio_tag}: "
            f"{launcher_overhead_vs_jax_us:.2f} us"
        )


if __name__ == "__main__":
    main()
