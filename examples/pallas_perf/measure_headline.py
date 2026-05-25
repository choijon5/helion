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
python measure_headline.py --timing-mode interleaved                 # DR#6 HP + G5-methodology HJ-full
python measure_headline.py --timing-mode interleaved-4way            # G6-methodology-v2 unified 4-way (cycle 31)
python measure_headline.py --device-us-calls 0                       # skip device_us collection (default 200)
```

Output (parseable lines, one per metric):

```
helion_bf16_<M>x<K>x<N>: median=<us> us                                                  # back-compat full-path (always sequential window)
helion_full_path_<M>x<K>x<N> [autotuner pick: <config>, seed=<n>]: median=<us> us         # sequential full-path (always; tracking; divisor of full_path_H_over_P)
helion_kernel_only_<M>x<K>x<N> [autotuner pick: <config>, seed=<n>]: median=<us> us       # HP-leg Helion-kernel median (paired with Pallas) in interleaved mode; 4-way Helion-kernel median in interleaved-4way; sequential window median in sequential mode
helion_kernel_only_hj_<M>x<K>x<N> [autotuner pick: <config>, seed=<n>]: median=<us> us    # HJ-full 3-way leg Helion-kernel median (divisor of kernel_only_H_over_J) in interleaved mode; same as the 4-way median in interleaved-4way (one unified Helion-kernel sample feeds both ratios); same as helion_kernel_only in sequential mode
helion_full_path_hj_<M>x<K>x<N> [autotuner pick: <config>, seed=<n>]: median=<us> us      # HJ-full 3-way leg Helion-full median (divisor of full_path_H_over_J — GATING for G5) in interleaved mode; 4-way Helion-full median in interleaved-4way; same value as helion_full_path in sequential mode
pallas_kernel_only_<M>x<K>x<N>: median=<us> us
jax_kernel_only_<M>x<K>x<N>: median=<us> us
full_path_H_over_P: <ratio>               # tracking; always uses the standalone sequential full-path us
kernel_only_H_over_P: <ratio>             # GATING for G2/G3/G4 (DR#6 canonical when --timing-mode interleaved or interleaved-4way — adjacent-pair invariant preserved under both)
full_path_H_over_J: <ratio>               # GATING for G5 (paired-sample HJ-full 3-way leg when --timing-mode interleaved; G5-methodology closed cycle 26; paired-sample adjacent under --timing-mode interleaved-4way)
kernel_only_H_over_J: <ratio>             # diagnostic for G5 substep selection (kernel vs launcher lever); G6-kernel-A target reference under interleaved-4way
kernel_only_P_over_J: <ratio>             # tracking — hand-written Pallas vs JAX baseline; emitted alongside the back-compat alias ``pallas_over_jax`` for cycle-31 schema parity
pallas_over_jax: <ratio>                  # cycle-31 G6-methodology-v2 spec name for kernel_only_P_over_J (same value); drives G6-kernel-A headroom map
launcher_overhead_us: <us>                # Helion-internal launcher overhead = helion_full − helion_kernel; paired-sample (both terms from HJ-full leg) in interleaved; paired-sample two-slots-off in interleaved-4way; sequential-window in sequential
launcher_overhead_vs_jax_us: <us>         # Helion full-path overhead vs JAX = helion_full − jax; paired-sample (both terms from HJ-full leg) in interleaved; paired-sample adjacent in interleaved-4way; sequential-window in sequential
launcher_overhead_vs_pallas_us: <us>      # Helion full-path overhead vs Pallas = helion_full − pallas; paired-sample adjacent in interleaved-4way (separates wrapper-vs-Pallas from wrapper-vs-JAX so G6-launcher-C substep can isolate which side moved); cross-leg / sequential under the other modes

helion_full_path_device_us_<M>x<K>x<N>: <us>     # G7-dispatch-amortize (cycle 36) — per-call on-device us for Helion full-path callable, jax.profiler 200-call avg of dominant /device:TPU:0 event
helion_kernel_only_device_us_<M>x<K>x<N>: <us>   # per-call on-device us for Helion kernel-only (autotuner-picked jit_fn)
pallas_kernel_only_device_us_<M>x<K>x<N>: <us>   # per-call on-device us for hand-Pallas reference
jax_kernel_only_device_us_<M>x<K>x<N>: <us>      # per-call on-device us for JAX baseline
device_H_over_P: <ratio>                  # device-level kernel ratio = pallas_device_us / helion_kernel_device_us (Helion beats Pallas on-device when > 1.00)
device_H_over_J: <ratio>                  # device-level kernel ratio = jax_device_us / helion_kernel_device_us (Helion beats JAX on-device when > 1.00)
device_P_over_J: <ratio>                  # device-level reference ratio = jax_device_us / pallas_device_us (XLA vs hand-Pallas; <1.00 means JAX wins on-device)
device_full_H_over_J: <ratio>             # device-level full-path ratio = jax_device_us / helion_full_device_us (Helion full-path vs JAX at the device level — typically tracks device_H_over_J closely because static-shapes kernels do the same device work on full-path vs kernel-only)

theoretical_min_us_<M>x<K>x<N>: <us>                  # G7-dispatch-amortize (cycle 36 manager refinement) — per-shape FLOPs / peak_FLOPS ceiling = 2*M*K*N / (peak_tflops * 1e6). Peak depends on dtype: bf16 = --peak-tflops-bf16 (default 1155 = TPU v7), f32 HIGHEST = --peak-tflops-f32 (default 192.5).
helion_full_path_device_pct_of_min_<M>x<K>x<N>: <ratio>     # theoretical_min_us / helion_full_device_us — 1.00 = at theoretical peak. Per-shape headroom signal; "data-bounded" shapes (small/skinny) sit far below 1.00 not because the kernel is bad but because dispatch / chip-latency dominates a microseconds-fraction theoretical min. Compute-bound shapes (1024³, 2048³, 4096³) sit at meaningful fractions of peak and have real kernel-level headroom.
helion_kernel_only_device_pct_of_min_<M>x<K>x<N>: <ratio>   # theoretical_min_us / helion_kernel_device_us
pallas_kernel_only_device_pct_of_min_<M>x<K>x<N>: <ratio>   # theoretical_min_us / pallas_device_us
jax_kernel_only_device_pct_of_min_<M>x<K>x<N>: <ratio>      # theoretical_min_us / jax_device_us — XLA reference fraction of peak; cycle-36 baseline shows JAX hits ~33% peak on 1024³ headline, ~66% on 2048³ large, near 1.00 on 4096³ large (sustained MXU)
```

**Reconstruction note for log scrapers (interleaved mode, cycle 26+).**
Each printed ratio uses these specific *_us lines as numerator /
denominator:
  - ``kernel_only_H_over_P``  = ``pallas_kernel_only`` / ``helion_kernel_only``     (HP 2-way leg)
  - ``kernel_only_H_over_J``  = ``jax_kernel_only`` / ``helion_kernel_only_hj``     (HJ-full 3-way leg)
  - ``full_path_H_over_J``    = ``jax_kernel_only`` / ``helion_full_path_hj``       (HJ-full 3-way leg — GATING for G5)
  - ``full_path_H_over_P``    = ``pallas_kernel_only`` / ``helion_full_path``       (mixes HP-leg pallas with sequential helion-full; tracking only)
  - ``launcher_overhead_us``  = ``helion_full_path_hj`` − ``helion_kernel_only_hj`` (both from HJ-full leg)
  - ``launcher_overhead_vs_jax_us`` = ``helion_full_path_hj`` − ``jax_kernel_only`` (both from HJ-full leg)
Naive ``helion_full_path`` − ``helion_kernel_only`` recovers the
*sequential* launcher overhead (cycle 25 semantics), NOT the printed
``launcher_overhead_us``. Use the ``*_hj_*`` lines for the cycle-26
paired-sample reconstruction.

**Reconstruction note for log scrapers (interleaved-4way mode, cycle 31+).**
Under the unified 4-way methodology every per-sweep ``*_us`` line is the
median of the **same** per-iteration ``perf_counter_ns()`` window (one
window per iteration timing JAX → Helion-full → Pallas → Helion-kernel
back-to-back), so cross-leg predecessor asymmetry is eliminated:
  - ``kernel_only_H_over_P``  = ``pallas_kernel_only`` / ``helion_kernel_only``     (4-way window, adjacent slots P → Hkernel — preserves DR#6 canonical adjacency)
  - ``kernel_only_H_over_J``  = ``jax_kernel_only`` / ``helion_kernel_only_hj``     (same value as ``helion_kernel_only``; 4-way Helion-kernel slot, 2-slots-off from JAX inside the window — diagnostic, almost-paired)
  - ``full_path_H_over_J``    = ``jax_kernel_only`` / ``helion_full_path_hj``       (4-way window, adjacent slots J → Hfull — GATING for G5)
  - ``pallas_over_jax``       = ``jax_kernel_only`` / ``pallas_kernel_only``        (same value as ``kernel_only_P_over_J``; 4-way window, 2-slots-off — drives G6-kernel-A headroom map)
  - ``launcher_overhead_us``  = ``helion_full_path_hj`` − ``helion_kernel_only_hj`` (4-way window, 2-slots-off Hfull → Hkernel — tracking)
  - ``launcher_overhead_vs_jax_us`` = ``helion_full_path_hj`` − ``jax_kernel_only`` (4-way window, adjacent J → Hfull)
  - ``launcher_overhead_vs_pallas_us`` = ``helion_full_path_hj`` − ``pallas_kernel_only`` (4-way window, adjacent Hfull → P — NEW G6 schema)

The kernel-only path is built by patching
``helion.runtime._pallas_build_callable`` to stash the JAX ``jit_fn``
argument (``pl.pallas_call(...)``) right before Helion wraps it in
torch_tpu's ``JaxCallable`` (which throws away the original ``jit_fn``
reference by ``jax.export``-serializing the body into a binary blob
bound to ``call_custom_kernel``). The captured ``jit_fn`` is re-wrapped
in ``jax.jit`` to match the JaxCallable construction site, and we time
it directly with JAX inputs — identical to the way the hand-written
``pallas_matmul`` reference is called.

Interleaved timing: see ``_time_interleaved_paired``'s docstring for
the canonical methodology rationale. Brief summary: a 3-way HJ-full
leg (``Helion-kernel → Helion-full → JAX``) makes the G5 gate ratio
``full_path_H_over_J`` paired-sample with the Helion-full ↔ JAX
adjacency; a separate 2-way HP leg (``Helion-kernel → Pallas``)
preserves the DR#6 canonical methodology that the G2/G3/G4 closure
verdicts landed under. Cost is ~2× per sweep (two paired legs
back-to-back), still within the per-shape sweep budget.

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
import contextlib
import inspect
import os
import sys
import tempfile
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

# Per-dtype theoretical MXU peak (TFLOPS/s) for TPU v7 / TPU7x. Sourced from
# ``jax._src.pallas.mosaic.tpu_info`` ``_get_tpu_info_impl`` for
# ``ChipVersion.TPU_7X`` with ``tensor_cores_per_chip=2`` and the
# ``TPU_VISIBLE_CHIPS=3`` pin (one logical core, ``num_cores=1``):
#   - bf16 / fp16 MXU peak = **1155 TFLOPS/s** (2.31e15 ops/s / 2).
#   - f32 MXU ``precision=HIGHEST`` effective peak = **~192.5 TFLOPS/s**
#     (standard multi-pass emulation ratio ~bf16/6; DR#7 §5 G7 validated
#     empirically — f32 1024³ device-only run hit 128.2 TFLOPS = 66.6% of
#     this estimate, matching expected near-peak behavior for a 1024³
#     HIGHEST matmul). Overridable from the CLI via ``--peak-tflops-bf16``
#     / ``--peak-tflops-f32`` for future TPU generations or
#     calibration runs.
_DEFAULT_PEAK_TFLOPS = {
    "bfloat16": 1155.0,
    "float32": 192.5,
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


def _time_interleaved_3way(
    fn_a: Callable[[], object],
    fn_b: Callable[[], object],
    fn_c: Callable[[], object],
    n_iter: int = 20,
    n_repeats: int = 5,
) -> tuple[float, float, float]:
    """Paired-sample 3-way timing of three callables. Returns ``(a_us, b_us, c_us)``.

    Times ``fn_a() → fn_b() → fn_c()`` consecutively inside one
    ``time.perf_counter_ns()`` window per iteration (one counter call
    between each fn invocation), so every per-call sample for a, b, c
    is collected within a single chip-thermal-noise window. Per-side
    median is the median across ``n_iter * n_repeats`` per-call
    samples on that side.

    Only **adjacent** pairs (a↔b, b↔c) have their common-mode drift
    fully cancelled in their ratio; non-adjacent pairs (a↔c) have
    fn_b's per-call jitter in between. Pick the ordering so the gate
    ratio's numerator and denominator are adjacent — see
    ``_time_interleaved_paired`` for the full G5-methodology rationale
    on why we use a 3-way HJ-full leg (gate-pair-adjacent) plus a
    separate 2-way HP leg (DR#6 G2/G3/G4 invariant) rather than one
    4-way window.

    Warmup: 5 3-way warmup iterations (each runs all three callables
    once).
    """
    for _ in range(5):
        fn_a()
        fn_b()
        fn_c()

    a_samples_ns: list[int] = []
    b_samples_ns: list[int] = []
    c_samples_ns: list[int] = []
    total = n_iter * n_repeats
    for _ in range(total):
        t0 = time.perf_counter_ns()
        fn_a()
        t1 = time.perf_counter_ns()
        fn_b()
        t2 = time.perf_counter_ns()
        fn_c()
        t3 = time.perf_counter_ns()
        a_samples_ns.append(t1 - t0)
        b_samples_ns.append(t2 - t1)
        c_samples_ns.append(t3 - t2)

    a_us = float(np.median(np.array(a_samples_ns))) / 1000.0
    b_us = float(np.median(np.array(b_samples_ns))) / 1000.0
    c_us = float(np.median(np.array(c_samples_ns))) / 1000.0
    return a_us, b_us, c_us


def _time_interleaved_paired(
    fn_helion_full: Callable[[], object],
    fn_helion_kernel: Callable[[], object],
    fn_pallas: Callable[[], object],
    fn_jax: Callable[[], object],
    n_iter: int = 20,
    n_repeats: int = 5,
) -> tuple[float, float, float, float, float]:
    """Paired-sample timing for the G5 tri-metric output schema.

    Returns ``(helion_full_us, helion_kernel_p_us, pallas_us,
    helion_kernel_j_us, jax_us)`` measured under two paired legs.

    Leg order (HP first, then HJ-full) preserves the DR#6 cycle-15-25
    pre-leg chip-thermal state for the HP leg's pair — running HP
    first means the HP leg starts after the same warmup pattern that
    cycles 21-25 closed G2/G3/G4 under, so the
    ``kernel_only_H_over_P`` invariant is preserved structurally (not
    just empirically). The HJ-full leg follows; its own gate pair
    (Helion-full ↔ JAX) is paired-sample within the same per-iteration
    window so chip-thermal drift between the two legs doesn't leak
    into the G5 gate ratio.

    1. **HP 2-way leg** (DR#6 canonical, unchanged):
       ``fn_helion_kernel → fn_pallas`` inside the same per-iteration
       ``perf_counter_ns()`` window. Produces ``helion_kernel_p_us``,
       ``pallas_us``.
    2. **HJ-full 3-way leg** (G5-methodology closure): the helper
       times ``fn_helion_kernel → fn_helion_full → fn_jax`` inside
       the same per-iteration window, ordered so the *gate pair*
       (Helion-full, JAX) is **adjacent** — paired-sample timing
       only cancels common-mode drift between calls that are adjacent
       in the timing window, and the G5 gate signal is
       ``full_path_H_over_J`` (= ``jax_us / helion_full_us``) so the
       helion-full → jax slot order matters most. The HJ-leg
       Helion-kernel us (taken one slot earlier in the same window)
       is used as the ``kernel_only_H_over_J`` divisor; its
       drift-cancellation with the JAX sample is one call worse than
       the gate's (helion-full sits between them), but the kernel
       H/J ratio is a diagnostic split — not the gate — so the
       trade-off prioritises the gate pair correctly. Produces
       ``helion_full_us``, ``helion_kernel_j_us``, ``jax_us``.

    The two legs run back-to-back; cost is one extra timing pass per
    shape vs the legacy 2-pass-pair, well within the per-shape sweep
    budget.

    Rationale for two legs rather than one 4-way window: the G2/G3/G4
    closure verdict landed under the canonical DR#6 2-way
    Helion-vs-Pallas paired methodology. Mixing a JAX call into the
    H/P window changes the per-iteration cycle length and could drift
    the H/P ratio away from the closed values by an amount the gate
    signal can't separate from a real regression. The 3-way HJ-full
    leg is new (closes the G5-methodology asymmetry); the 2-way HP leg
    is unchanged in shape AND in pre-leg state (preserves
    apples-to-apples comparability with G2/G3/G4 closures).

    Diagnostic-ratio caveat: ``kernel_only_H_over_J`` divides
    ``jax_kernel_us`` (HJ-full leg, post-helion-full slot) by
    ``helion_kernel_us`` (HJ-full leg, pre-helion-full slot) — the
    two terms are paired-sample with one ``helion_full`` call between
    them, so the ratio is "almost paired" (drift cancellation is one
    call worse than the G5 gate's strict adjacency).
    ``kernel_only_P_over_J`` divides ``jax_kernel_us`` (HJ-full leg)
    by ``pallas_kernel_us`` (HP leg) — the two terms come from
    different legs and different predecessor calls (Pallas-after-
    Helion-kernel vs JAX-after-Helion-full), so the ratio is NOT
    paired-sample. Both diagnostics drive G5 bucket selection (see
    plan.md §5 G5 bucket rule) so the selection inherits that
    asymmetry; cycle-26 documents the effect explicitly in the
    methodology closure write-up.

    Warmup: 5 warmup iterations per leg (10 total).
    """
    helion_kernel_p_us, pallas_us = _time_interleaved(
        fn_helion_kernel, fn_pallas, n_iter=n_iter, n_repeats=n_repeats
    )
    # Order: helion_kernel → helion_full → jax. Putting helion_full and
    # jax adjacent makes the gate ratio ``full_path_H_over_J = jax_us /
    # helion_full_us`` a true paired-sample number (no other callable
    # between them inside the per-iteration window). helion_kernel sits
    # one slot earlier so the kernel-H/J diagnostic is "almost paired"
    # — accepted trade-off, since the kernel H/J ratio is a substep
    # selector, not the gate.
    helion_kernel_j_us, helion_full_us, jax_us = _time_interleaved_3way(
        fn_helion_kernel, fn_helion_full, fn_jax, n_iter=n_iter, n_repeats=n_repeats
    )
    return (
        helion_full_us,
        helion_kernel_p_us,
        pallas_us,
        helion_kernel_j_us,
        jax_us,
    )


def _time_interleaved_4way(
    fn_helion_full: Callable[[], object],
    fn_helion_kernel: Callable[[], object],
    fn_pallas: Callable[[], object],
    fn_jax: Callable[[], object],
    n_iter: int = 20,
    n_repeats: int = 5,
) -> tuple[float, float, float, float]:
    """Unified 4-way paired-sample timing of all four callables in one window.

    Returns ``(helion_full_us, helion_kernel_us, pallas_us, jax_us)``
    measured from a single per-iteration timing window that issues
    all four callables back-to-back. Median per side across
    ``n_iter * n_repeats`` per-call samples.

    Ordering (fixed): ``fn_jax → fn_helion_full → fn_pallas →
    fn_helion_kernel`` inside one ``perf_counter_ns()`` window per
    iteration (one counter call between each fn invocation), so every
    per-call sample for jax, helion_full, pallas, helion_kernel is
    collected within a single chip-thermal-noise window.

    Adjacency map under this ordering (only adjacent pairs cancel
    common-mode chip-thermal drift fully in their ratio):

    - **JAX ↔ Helion-full**: paired-sample → ``full_path_H_over_J``
      (G5 / G6-launcher-C gate) and ``launcher_overhead_vs_jax_us``.
      JAX runs first (predecessor for Helion-full), so the JAX
      sample is essentially predecessor-free at the start of the
      window — symmetric inheritance with the cycle-26 HJ-full leg
      where JAX inherits Helion-full's wind-down.
    - **Helion-full ↔ Pallas**: paired-sample →
      ``launcher_overhead_vs_pallas_us`` (NEW G6 schema separates
      Helion's full-path overhead vs the hand-written Pallas
      reference from the JAX overhead so substep diagnosis can
      tell which side moved).
    - **Pallas ↔ Helion-kernel**: paired-sample →
      ``kernel_only_H_over_P`` (G2/G3/G4 gate; the canonical
      DR#6 adjacency that the prior closures landed under is
      preserved here — Helion-kernel still pairs immediately with
      Pallas, so a re-measurement of any closed shape under 4-way
      methodology is apples-to-apples with the cycle-21-26 H/P
      verdict).

    Non-adjacent (almost-paired) ratios:

    - **Helion-kernel ↔ JAX** (kernel H/J, G6-kernel-A input):
      two callables apart (Helion-full + Pallas between them).
      Diagnostic split — not gating — so the trade-off is accepted.
      Same precision class as the cycle-26 HJ-full leg's kernel
      H/J ratio (also one or two callables apart depending on
      framing).
    - **Pallas ↔ JAX** (P/J, reference Pallas headroom over JAX,
      drives G6-kernel-A target selection): two callables apart
      (Helion-full between them). The headroom map G6-kernel-A
      uses this ratio as a sizer; "almost-paired" precision is
      sufficient.
    - **Helion-full ↔ Helion-kernel** (launcher_overhead_us):
      two callables apart (Pallas between them). Tracking-only;
      cycle-26 had this adjacency, but the new 4-way methodology
      sacrifices it to keep the gating ratios fully paired. The
      kernel-side overhead diagnostic is recoverable via
      ``launcher_overhead_vs_jax_us - (helion_full_us -
      helion_kernel_us)``-style decomposition if needed.

    Cost: 1 paired leg per sweep (vs the cycle-26 HP-leg + HJ-full
    leg's 2 paired legs), so 4-way is faster per sweep than the
    cycle-26 2-leg form. The G2/G3/G4 H/P invariant is preserved
    because Helion-kernel is still paired-sample-adjacent to
    Pallas (the canonical DR#6 ordering).

    Warmup: 5 4-way warmup iterations (each runs all four
    callables once).
    """
    for _ in range(5):
        fn_jax()
        fn_helion_full()
        fn_pallas()
        fn_helion_kernel()

    jax_samples_ns: list[int] = []
    helion_full_samples_ns: list[int] = []
    pallas_samples_ns: list[int] = []
    helion_kernel_samples_ns: list[int] = []
    total = n_iter * n_repeats
    for _ in range(total):
        t0 = time.perf_counter_ns()
        fn_jax()
        t1 = time.perf_counter_ns()
        fn_helion_full()
        t2 = time.perf_counter_ns()
        fn_pallas()
        t3 = time.perf_counter_ns()
        fn_helion_kernel()
        t4 = time.perf_counter_ns()
        jax_samples_ns.append(t1 - t0)
        helion_full_samples_ns.append(t2 - t1)
        pallas_samples_ns.append(t3 - t2)
        helion_kernel_samples_ns.append(t4 - t3)

    jax_us = float(np.median(np.array(jax_samples_ns))) / 1000.0
    helion_full_us = float(np.median(np.array(helion_full_samples_ns))) / 1000.0
    pallas_us = float(np.median(np.array(pallas_samples_ns))) / 1000.0
    helion_kernel_us = float(np.median(np.array(helion_kernel_samples_ns))) / 1000.0
    return helion_full_us, helion_kernel_us, pallas_us, jax_us


def _time_device_us(
    fn: Callable[[], object],
    n_calls: int = 200,
    n_warmup: int = 5,
) -> float:
    """Return per-call on-device us under a ``jax.profiler`` trace.

    Wraps ``n_calls`` invocations of ``fn`` in a single
    ``jax.profiler.start_trace`` / ``stop_trace`` window, parses the
    resulting ``.xplane.pb`` via ``jax.profiler.ProfileData.from_file``,
    finds the dominant compute event on the ``/device:TPU:0`` plane
    (largest total ``duration_ns`` across all events on that plane), and
    returns ``total_duration_ns / n_calls / 1000``.

    Rationale (DR#7 §5 G7 — see plan.md §1 ``device_us`` block). The
    existing single-call us metric is ~96-98% PJRT + ``pallas_call``
    dispatch overhead on small / medium matmuls. ``device_us`` exposes
    the kernel-actual time so substeps targeting kernel quality
    (G7-prefetch, G7-launch-fusion) have a signal that isn't drowned
    by dispatch noise. Both metrics are tracked side-by-side: the
    single-call us reflects what users actually pay per call, the
    ``device_us`` reflects on-device throughput.

    Aggregation rule: per dominant compute event name (e.g.
    ``jit_wrapped(<hash>)`` for Helion-kernel, ``jit_pallas_matmul(<hash>)``
    for hand-Pallas, ``jit_matmul(<hash>)`` for JAX). DR#7 Track 3
    confirmed the per-event device us is uniform across calls inside
    a single trace window (no scheduler-induced drift across the
    200-call loop), so the avg per call equals the median per call
    within sub-us. Returning the average is simpler and matches DR#7's
    reported numbers (JAX 5.50us / Pallas 9.91us / Helion 6.12us on
    the bf16 1024³ headline).

    Args:
      fn: zero-arg callable that issues one device-side op and (via
        a tail ``jax.block_until_ready`` or equivalent inside the
        callable) makes the result visible to ``stop_trace``. The
        existing per-callable wrappers (``_run_full_path``,
        ``_run_helion_kernel_only``, ``_run_pallas_kernel_only``,
        ``_run_jax_kernel_only``) already do this.
      n_calls: number of calls to amortize over. Default 200 matches
        the DR#7 Track 3 probe; tighter than the 5-sweep single-call
        median (each sweep is 20×5=100 calls) and the ``ProfileData``
        per-event averages are sub-us stable at 200 calls.
      n_warmup: warmup calls before ``start_trace``. Default 5 matches
        the rest of the harness.

    Raises:
      FileNotFoundError: no ``.xplane.pb`` was written under the
        trace dir (means ``start_trace`` failed silently — typically a
        XLA flag mis-set; the harness should surface this).
      RuntimeError: trace contains no ``/device:TPU:0`` plane (means
        the run ran on a non-TPU device, e.g. the chip pin
        ``TPU_VISIBLE_CHIPS=3`` was missed).
    """
    for _ in range(n_warmup):
        fn()

    with tempfile.TemporaryDirectory(prefix="device_us_trace_") as trace_dir:
        jax.profiler.start_trace(trace_dir)
        last_out: object = None
        for _ in range(n_calls):
            last_out = fn()
        # Belt-and-suspenders: the per-call ``fn`` already calls
        # ``block_until_ready`` (or its torch-side equivalent), but a
        # final sync makes sure the last call's device work is included
        # in the trace before ``stop_trace`` finalises the .pb. The
        # ``contextlib.suppress`` covers callables whose return value is
        # ``None`` (full-path callables synchronise inside the fn) or a
        # non-JAX-array type — ``block_until_ready`` raises ``TypeError`` /
        # ``AttributeError`` on those, which is benign here.
        if last_out is not None:
            with contextlib.suppress(TypeError, AttributeError):
                jax.block_until_ready(last_out)
        jax.profiler.stop_trace()

        # Find the .xplane.pb file
        pb_path: str | None = None
        for root, _dirs, files in os.walk(trace_dir):
            for f in files:
                if f.endswith(".xplane.pb"):
                    pb_path = os.path.join(root, f)
                    break
            if pb_path is not None:
                break
        if pb_path is None:
            raise FileNotFoundError(
                f"No .xplane.pb under {trace_dir}; jax.profiler.start_trace "
                "produced no output — check XLA flags / pod TPU pin."
            )

        # Parse via the public ProfileData API. The trace's /device:TPU:0
        # plane has multiple lines (host-side jit summary + device-side
        # HLO events + SparseCore counters + DVFS / hardware P-state
        # counter samples); the dominant compute event is the largest
        # per-event total across the lines whose event count equals
        # ``n_calls`` (one event per call — these are the jit / HLO ops
        # that actually issued once per call). The count filter is
        # required to exclude the device's hardware ``P state`` counter
        # line: when ``LIBTPU_INIT_ARGS=--xla_tpu_dvfs_p_state=7`` is set
        # (canonical in ``examples/pallas_perf/benchmark.sh``), the trace
        # gains a second ``_counters_`` line carrying ~17 DVFS-sampled
        # ``P state`` events whose total spans the full 200-call window
        # (~52 ms total). Without the count filter the aggregator picks
        # that line, computes ``52000us / 200 = 260us/call``, and reports
        # ~45× the actual on-device time. DR#7 validation: post-filter,
        # this is uniformly the top-level ``jit_<name>(<hash>)`` line
        # (line "XLA Modules") for all 3 paths (Helion / Pallas / JAX) —
        # the same line that aggregates every device-side op for that
        # jit call, matching DR#7 Track 3's reported per-call us (5.50
        # JAX / 9.91 Pallas / 6.12 Helion on bf16 1024³).
        pd = jax.profiler.ProfileData.from_file(pb_path)
        best_total_ns = 0
        saw_device_plane = False
        for plane in pd.planes:
            if plane.name != "/device:TPU:0":
                continue
            saw_device_plane = True
            for line in plane.lines:
                per_event_totals: dict[str, int] = {}
                per_event_counts: dict[str, int] = {}
                for ev in line.events:
                    per_event_totals[ev.name] = (
                        per_event_totals.get(ev.name, 0) + ev.duration_ns
                    )
                    per_event_counts[ev.name] = per_event_counts.get(ev.name, 0) + 1
                for name, total in per_event_totals.items():
                    if per_event_counts[name] != n_calls:
                        continue
                    if total > best_total_ns:
                        best_total_ns = total
        if not saw_device_plane:
            raise RuntimeError(
                f"jax.profiler trace at {pb_path} has no /device:TPU:0 plane "
                "— is TPU_VISIBLE_CHIPS=3 set and the device JAX-visible?"
            )
        if best_total_ns == 0:
            raise RuntimeError(
                f"jax.profiler trace at {pb_path} has no /device:TPU:0 event "
                f"with count == {n_calls}; the per-call kernel may have a "
                "different cardinality than 1 (e.g. inner loop emits multiple "
                "events per call) — adjust the count filter or use a different "
                "n_calls."
            )

    # us per call = ns_total / n_calls / 1000.
    return best_total_ns / n_calls / 1000.0


def _theoretical_min_us(m: int, k: int, n: int, peak_tflops: float) -> float:
    """Return the theoretical minimum per-call us for a matmul (manager
    refinement 2026-05-25 — see plan.md §1 device-us block).

    The per-shape achievable ceiling is shape-dependent — small / skinny
    shapes can never hit 90% of MXU peak because there aren't enough
    arithmetic ops to amortize MXU pipeline fill / drain, while compute-bound
    shapes (e.g. 1024³, 2048³, 4096³) can saturate the chip. The correct
    per-shape lower bound is::

        theoretical_min_us = FLOPs / peak_FLOPS
                           = (2 * M * K * N) / (peak_tflops * 1e12) / 1e-6
                           = (2 * M * K * N) / (peak_tflops * 1e6)

    Examples (bf16 1155 TFLOPS/s on TPU v7):
      - 1024×1024×1024: 2.15 GFLOP → ~1.86 us min; headline Helion 6.12 us
        device → ~30% of theoretical min (PLENTY of room for kernel work
        on this and larger shapes).
      - 1024×1×1024: 2.10 MFLOP → ~0.0018 us min; structurally
        dispatch-bounded — device_pct_of_min only reaches the chip
        latency floor on this kind of shape, not the MXU peak.
      - 1×1×1024: 2 KFLOP → ~0.0000017 us min; latency-bounded, cannot
        approach peak under any kernel strategy.
      - 4096×4096×4096: 137 GFLOP → ~119 us min; large enough for true
        MXU sustained-peak work — the canonical signal shape for any
        future G7-prefetch / G7-launch-fusion substep.

    Args:
      m, k, n: matmul dims (A is m × k, B is k × n, C is m × n).
      peak_tflops: per-dtype TPU MXU peak in TFLOPS/s
        (``_DEFAULT_PEAK_TFLOPS[dtype]`` or the ``--peak-tflops-{bf16,f32}``
        CLI override).

    Returns:
      Theoretical minimum per-call wall-clock us — the device cannot
      finish the matmul in less time even if every cycle hit MXU peak.
    """
    flops = 2.0 * m * k * n
    return flops / (peak_tflops * 1e6)


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
        "--n-sweeps",
        type=int,
        default=1,
        help=(
            "Number of measurement sweeps to print, sharing one "
            "autotune + compile + capture refresh setup. Each sweep is "
            "an independent ``timeit.repeat`` (sequential) or "
            "interleaved-window block, prefixed with ``[sweep <i>]`` "
            "when ``n_sweeps > 1`` so per-sweep medians are parseable "
            "by outer harnesses. Default 1 (back-compat with cycles "
            "15-30 invocations). Amortizes the ~50s autotune across "
            "multiple sweeps when the outer harness wants per-shape "
            "medians of N sweeps; for cross-shape sweeps the outer "
            "shell loop still re-invokes per shape to honor the "
            "per-shape ``compile_config(best_config)`` + capture "
            "refresh that the autotuner needs."
        ),
    )
    parser.add_argument(
        "--timing-mode",
        choices=("sequential", "interleaved", "interleaved-4way", "both"),
        default="sequential",
        help=(
            "Timing methodology for the kernel-only and G5 full-path-vs-JAX "
            "ratios. ``sequential`` (default, back-compat with cycles "
            "15-19) measures Helion / Pallas / JAX / Helion-full in "
            "separate timing windows. ``interleaved`` pairs every Helion "
            "call with its reference call(s) inside the same per-call "
            "``time.perf_counter_ns()`` window so chip-thermal drift "
            "cancels in the ratios: the HP leg is a 2-way "
            "Helion-vs-Pallas pair (canonical DR#6 since 2026-05-23), "
            "the HJ-full leg is a 3-way ``Helion-kernel → Helion-full → "
            "JAX`` cycle so the G5 gate signal ``full_path_H_over_J`` "
            "and ``launcher_overhead_vs_jax_us`` are paired-sample "
            "(G5-methodology closure cycle 26). "
            "``interleaved-4way`` (cycle 31 unified-methodology) times "
            "all four callables (Helion-full, Helion-kernel, Pallas, "
            "JAX) in one 4-way ``perf_counter_ns()`` window per "
            "iteration with ordering ``JAX → Helion-full → Pallas → "
            "Helion-kernel`` — chosen so the three gating / new-G6-schema "
            "ratios (``full_path_H_over_J``, ``launcher_overhead_vs_pallas_us``, "
            "``kernel_only_H_over_P``) all pair adjacent slots and "
            "every ratio is recovered from one unified per-iteration "
            "window (no cross-leg predecessor asymmetry). ``both`` "
            "runs sequential + interleaved (the cycle-26 form) in "
            "sequence and prints both result blocks (methodology "
            "comparison probe; does NOT include the new 4-way mode). "
            "The back-compat ``helion_full_path_*`` "
            "line + ``full_path_H_over_P`` ratio always use the "
            "standalone sequential ``_time(_run_full_path)`` window "
            "regardless of this flag (preserves the cycle 15-25 log "
            "scraper contract for full-path-vs-Pallas tracking)."
        ),
    )
    parser.add_argument(
        "--device-us-calls",
        type=int,
        default=200,
        help=(
            "Number of calls to amortize per ``jax.profiler.start_trace`` "
            "window for the ``device_us`` metric (DR#7 §5 G7 — see "
            "plan.md §1 ``device_us`` block). Default 200 matches the "
            "DR#7 Track 3 probe. Set to 0 to skip device-us collection "
            "entirely (e.g. on hosts where ``jax.profiler`` traces don't "
            "produce a parseable .xplane.pb; the single-call us metric "
            "is unaffected). Emitted as one ``<path>_device_us`` line "
            "per callable (Helion-full, Helion-kernel, Pallas, JAX) "
            "plus derived ``device_H_over_P`` / ``device_H_over_J`` / "
            "``device_P_over_J`` ratios."
        ),
    )
    parser.add_argument(
        "--peak-tflops-bf16",
        type=float,
        default=_DEFAULT_PEAK_TFLOPS["bfloat16"],
        help=(
            "TPU MXU bf16/fp16 peak TFLOPS/s used to compute the "
            "per-shape ``theoretical_min_us`` and ``device_pct_of_min`` "
            "lines (manager refinement 2026-05-25 — see plan.md §1 "
            "device-us block). Default 1155.0 = TPU v7 / TPU7x bf16 "
            "per-tensor-core peak per "
            "``jax._src.pallas.mosaic.tpu_info`` ``_get_tpu_info_impl`` "
            "(``ChipVersion.TPU_7X`` with ``tensor_cores_per_chip=2``; "
            "the ``TPU_VISIBLE_CHIPS=3`` pin sees one logical core)."
        ),
    )
    parser.add_argument(
        "--peak-tflops-f32",
        type=float,
        default=_DEFAULT_PEAK_TFLOPS["float32"],
        help=(
            "TPU MXU f32 ``precision=HIGHEST`` effective peak TFLOPS/s "
            "used to compute the per-shape ``theoretical_min_us`` and "
            "``device_pct_of_min`` lines. Default 192.5 = standard "
            "multi-pass emulation ratio ~bf16/6 (DR#7 §5 G7 — f32 1024³ "
            "device-only run validated empirically at 128.2 TFLOPS = "
            "66.6% of this estimate, matching expected near-peak behavior "
            "for a 1024³ HIGHEST matmul)."
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
    n_sweeps = max(1, int(args.n_sweeps))

    for sweep_idx in range(n_sweeps):
        if n_sweeps > 1:
            print(f"=== sweep {sweep_idx + 1}/{n_sweeps} ===")
        for mode in timing_modes:
            if mode == "sequential":
                helion_kernel_us = _time(_run_helion_kernel_only)
                pallas_kernel_us = _time(_run_pallas_kernel_only)
                jax_kernel_us = _time(_run_jax_kernel_only)
                # Sequential mode reports a single Helion-kernel us (one timing
                # window), which both kernel-only ratios (kernel H/P and kernel
                # H/J) divide. Set both Helion-kernel legs to the same value
                # so the downstream printing logic doesn't need to special-case
                # the mode. ``helion_full_us_for_j`` defaults to the sequential
                # ``helion_full_us`` measured outside the loop; ``jax_full_us``
                # is the same as ``jax_kernel_us`` (JAX has no separate launcher
                # — full path IS kernel-only for JAX).
                helion_kernel_us_for_p = helion_kernel_us
                helion_kernel_us_for_j = helion_kernel_us
                helion_full_us_for_j = helion_full_us
                jax_full_us = jax_kernel_us
                mode_tag = "[sequential] " if tag_lines else ""
            elif mode == "interleaved-4way":
                # Cycle-31 unified 4-way methodology: a single per-iteration
                # window times all four callables back-to-back so every
                # ratio (H/P, H/J full, H/J kernel, P/J, launcher overhead)
                # is recovered from one paired-sample window — no cross-leg
                # predecessor asymmetry. See ``_time_interleaved_4way``'s
                # docstring for the adjacency map under the chosen
                # ``JAX → Helion-full → Pallas → Helion-kernel`` ordering.
                (
                    helion_full_us_for_j,
                    helion_kernel_us,
                    pallas_kernel_us,
                    jax_kernel_us,
                ) = _time_interleaved_4way(
                    _run_full_path,
                    _run_helion_kernel_only,
                    _run_pallas_kernel_only,
                    _run_jax_kernel_only,
                )
                # Unified methodology: one Helion-kernel sample feeds both
                # the H/P (paired with Pallas, adjacent) and H/J (paired
                # 2-slots-off with JAX) ratios so the divisor is shared.
                # The cycle-26 split between HP-leg and HJ-full-leg Helion
                # samples collapses to a single sample under 4-way.
                helion_kernel_us_for_p = helion_kernel_us
                helion_kernel_us_for_j = helion_kernel_us
                jax_full_us = jax_kernel_us
                mode_tag = "[interleaved-4way] " if tag_lines else ""
            else:  # interleaved (G5-methodology: 3-way HJ-full leg + 2-way HP leg)
                (
                    helion_full_us_for_j,
                    helion_kernel_us_for_p,
                    pallas_kernel_us,
                    helion_kernel_us_for_j,
                    jax_kernel_us,
                ) = _time_interleaved_paired(
                    _run_full_path,
                    _run_helion_kernel_only,
                    _run_pallas_kernel_only,
                    _run_jax_kernel_only,
                )
                # Report Helion-kernel as the HP-leg median (the leg that pairs
                # with Pallas and matches the canonical DR#6 2-way invariant
                # that G2/G3/G4 closed under). The HJ-full leg's Helion-kernel
                # us is typically within ~0.5us of the HP leg's value on the same
                # shape; the reconstructible-ratio property requires reporting
                # ONE leg's us as the printed median (so ``pallas_us /
                # helion_us`` equals the printed H/P), not a mean of two
                # medians which would break the median claim and the
                # reconstruction. The per-ratio Helion us in the divisions
                # below uses the per-leg values to keep H/P apples-to-apples
                # with the HP leg and H/J apples-to-apples with the HJ-full
                # leg.
                #
                # ``helion_full_us_for_j`` is the HJ-full 3-way leg's Helion
                # full-path median (G5-methodology gate signal divisor) and
                # also the ``launcher_overhead_us`` / ``launcher_overhead_vs_jax_us``
                # numerator (cycle 26 methodology change — both terms in the
                # overhead deltas now come from the paired HJ-full leg). The
                # standalone sequential ``helion_full_us`` (measured outside
                # the loop) stays as the back-compat ``helion_full_path_*``
                # output and the ``full_path_H_over_P`` divisor only; the
                # G5-gating ``full_path_H_over_J`` uses ``helion_full_us_for_j``
                # so it's paired-sample with ``jax_kernel_us``. For JAX, "full
                # path" and "kernel only" are the same path so ``jax_full_us``
                # equals ``jax_kernel_us``.
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
            # the HJ-full 3-way leg's Helion-kernel median (the divisor used
            # in the printed ``kernel_only_H_over_J`` ratio); ``jax_kernel_only
            # / helion_kernel_only_hj`` recovers the printed H/J. In sequential
            # mode both lines carry the same single-window value.
            # ``helion_full_path_hj_*`` carries the HJ-full 3-way leg's Helion
            # full-path median (the divisor for the gating ``full_path_H_over_J``
            # ratio); the standalone ``helion_full_path_*`` line above is the
            # sequential measurement and stays for back-compat / tracking. In
            # sequential mode both full-path lines carry the same value.
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
                f"helion_full_path_hj_{m}x{k}x{n}{line_tag} "
                f"[autotuner pick: {best_config}, seed={seed}]: "
                f"median={helion_full_us_for_j:.2f} us"
            )
            print(
                f"pallas_kernel_only_{m}x{k}x{n}{line_tag}: "
                f"median={pallas_kernel_us:.2f} us"
            )
            print(
                f"jax_kernel_only_{m}x{k}x{n}{line_tag}: median={jax_kernel_us:.2f} us"
            )

            # ----- Derived ratios.
            # H/P uses the Helion-kernel us paired with Pallas (DR#6 canonical);
            # kernel H/J uses the Helion-kernel us paired with JAX (HJ-full 3-way
            # leg); full H/J uses the Helion-full us paired with JAX (HJ-full
            # 3-way leg — G5-methodology closure). In sequential mode all four
            # Helion legs share the same per-window value, so the ratio math is
            # identical to the legacy 2-way path. The back-compat
            # ``full_path_H_over_P`` ratio keeps using the sequential
            # ``helion_full_us`` so existing log scrapers stay valid; it's a
            # tracking number only.
            #
            # For JAX, "full path" and "kernel only" are identical (no torch_tpu
            # / Helion launcher in the path), so ``jax_kernel_us`` doubles as
            # the JAX full-path baseline. ``launcher_overhead_vs_jax_us`` is
            # the raw delta between Helion full-path (the paired-leg version,
            # so it's directly comparable to ``jax_kernel_us``) and JAX — the
            # absolute gap a G5 launcher-side substep has to close on shapes
            # where the kernel is already fast enough.
            full_h_over_p = pallas_kernel_us / helion_full_us
            kernel_h_over_p = pallas_kernel_us / helion_kernel_us_for_p
            full_h_over_j = jax_full_us / helion_full_us_for_j
            kernel_h_over_j = jax_kernel_us / helion_kernel_us_for_j
            kernel_p_over_j = jax_kernel_us / pallas_kernel_us
            # ``launcher_overhead_us`` is the Helion-internal launcher overhead:
            # full-path Helion minus kernel-only Helion. In interleaved mode
            # both terms come from the HJ-full 3-way leg so they're
            # paired-sample (full-path and kernel-only timed back-to-back
            # inside the same per-iteration window). In sequential mode both
            # are single-window medians (full-path from outside the loop,
            # kernel-only from the per-mode block) so this is the sequential-
            # window delta. ``launcher_overhead_vs_jax_us`` is the analogous
            # subtraction against JAX: the paired-leg full-path us minus JAX
            # (interleaved) or the sequential full-path us minus JAX
            # (sequential), matching the ``full_path_H_over_J`` ratio's
            # pairing (gate signal denominator is paired-sample with JAX in
            # interleaved mode).
            launcher_overhead_us = helion_full_us_for_j - helion_kernel_us_for_j
            launcher_overhead_vs_jax_us = helion_full_us_for_j - jax_full_us
            # ``launcher_overhead_vs_pallas_us`` (cycle 31 G6-methodology-v2
            # output schema): Helion full-path us − Pallas kernel-only us.
            # Separates how much of the launcher overhead the user is paying
            # vs the hand-written Pallas reference from how much they're
            # paying vs JAX (``launcher_overhead_vs_jax_us``). Under the
            # 4-way unified methodology the two terms are paired-sample
            # adjacent (``Helion-full → Pallas`` slot pair in the ordering),
            # so the delta is a true single-window measurement. Under
            # ``sequential`` and the cycle-26 ``interleaved`` modes the two
            # terms come from different windows / legs respectively, so the
            # delta is sequential-window / cross-leg under those modes
            # (still printed for symmetry with the JAX overhead, but the
            # G6-launcher-C substep should consume it under the 4-way mode).
            launcher_overhead_vs_pallas_us = helion_full_us_for_j - pallas_kernel_us

            print(f"full_path_H_over_P{ratio_tag}: {full_h_over_p:.3f}")
            print(f"kernel_only_H_over_P{ratio_tag}: {kernel_h_over_p:.3f}")
            print(f"full_path_H_over_J{ratio_tag}: {full_h_over_j:.3f}")
            print(f"kernel_only_H_over_J{ratio_tag}: {kernel_h_over_j:.3f}")
            print(f"kernel_only_P_over_J{ratio_tag}: {kernel_p_over_j:.3f}")
            print(f"pallas_over_jax{ratio_tag}: {kernel_p_over_j:.3f}")
            print(f"launcher_overhead_us{ratio_tag}: {launcher_overhead_us:.2f} us")
            print(
                f"launcher_overhead_vs_jax_us{ratio_tag}: "
                f"{launcher_overhead_vs_jax_us:.2f} us"
            )
            print(
                f"launcher_overhead_vs_pallas_us{ratio_tag}: "
                f"{launcher_overhead_vs_pallas_us:.2f} us"
            )

            # ----- Device-us collection (G7-dispatch-amortize, cycle 36).
            # Per DR#7 (plan.md §5 G7 ceiling-verification block), the
            # single-call us above is ~96-98% PJRT + ``pallas_call`` dispatch
            # overhead on small / medium matmuls — kernel-side substeps
            # (G7-prefetch, G7-launch-fusion) producing 5-10% on-device wins
            # only move the single-call us ~0.5% and disappear into the
            # autotuner-pick noise band. ``device_us`` is the per-call
            # on-device matmul time measured via a ``jax.profiler`` trace
            # over ``--device-us-calls`` calls (default 200), divided by the
            # call count. Both metrics are tracked side-by-side: single-call
            # us reflects user-perceived per-call latency (dispatch dominates),
            # ``device_us`` reflects kernel-actual on-device time (substeps
            # targeting kernel quality have a meaningful signal). The
            # ``device_us`` is independent of ``--timing-mode`` (it's a
            # profile-trace average, not a paired-sample wall-clock window),
            # but is emitted under the same ``[mode]`` / ``_<mode>`` tag
            # for log-scraper consistency.
            #
            # Device us is also useful for the §1 14-row table's
            # ``device_H_over_P`` / ``device_H_over_J`` / ``device_P_over_J``
            # columns; the cycle-36 DR#7 baseline on the headline shape
            # was JAX 5.50 / Pallas 9.91 / Helion 6.12 us (Helion beats
            # hand-Pallas 1.6× on-device but is 11% slower than JAX —
            # the inverse of the single-call story where all three converge
            # at ~120us).
            if args.device_us_calls > 0:
                helion_full_device_us = _time_device_us(
                    _run_full_path, n_calls=args.device_us_calls
                )
                helion_kernel_device_us = _time_device_us(
                    _run_helion_kernel_only, n_calls=args.device_us_calls
                )
                pallas_device_us = _time_device_us(
                    _run_pallas_kernel_only, n_calls=args.device_us_calls
                )
                jax_device_us = _time_device_us(
                    _run_jax_kernel_only, n_calls=args.device_us_calls
                )

                print(
                    f"helion_full_path_device_us_{m}x{k}x{n}{line_tag}: "
                    f"{helion_full_device_us:.4f} us  "
                    f"[{args.device_us_calls}-call jax.profiler avg]"
                )
                print(
                    f"helion_kernel_only_device_us_{m}x{k}x{n}{line_tag}: "
                    f"{helion_kernel_device_us:.4f} us  "
                    f"[{args.device_us_calls}-call jax.profiler avg]"
                )
                print(
                    f"pallas_kernel_only_device_us_{m}x{k}x{n}{line_tag}: "
                    f"{pallas_device_us:.4f} us  "
                    f"[{args.device_us_calls}-call jax.profiler avg]"
                )
                print(
                    f"jax_kernel_only_device_us_{m}x{k}x{n}{line_tag}: "
                    f"{jax_device_us:.4f} us  "
                    f"[{args.device_us_calls}-call jax.profiler avg]"
                )

                # Device-level ratios mirror the kernel-only ratio shape:
                # ``device_H_over_P`` = pallas_device_us / helion_kernel_device_us
                # (Helion beats Pallas when > 1.00); same for H/J and P/J.
                # The Helion-full vs Helion-kernel comparison at the device
                # level isolates how much of Helion's full-path on-device
                # time is the launched-via-torch_tpu dispatch path bumping
                # things vs the kernel itself — typically near-zero on
                # ``static_shapes=True`` kernels because the device-side
                # work is identical, but tracked for symmetry with the
                # single-call ``launcher_overhead_us``.
                device_h_over_p = pallas_device_us / helion_kernel_device_us
                device_h_over_j = jax_device_us / helion_kernel_device_us
                device_p_over_j = jax_device_us / pallas_device_us
                device_full_h_over_j = jax_device_us / helion_full_device_us
                print(f"device_H_over_P{ratio_tag}: {device_h_over_p:.3f}")
                print(f"device_H_over_J{ratio_tag}: {device_h_over_j:.3f}")
                print(f"device_P_over_J{ratio_tag}: {device_p_over_j:.3f}")
                print(f"device_full_H_over_J{ratio_tag}: {device_full_h_over_j:.3f}")

                # ----- Per-shape data-size-adjusted ceiling (manager
                # refinement 2026-05-25 — see plan.md §1 device-us block).
                # The right per-shape lower bound for device_us is
                # ``theoretical_min_us = FLOPs / peak_FLOPS``, NOT a
                # universal "% peak" — small / skinny shapes can never
                # hit MXU peak because there are not enough arithmetic
                # ops to amortize MXU pipeline fill / drain (the per-call
                # latency floor sits at the chip's irreducible dispatch
                # latency, not at the MXU FLOPS ceiling). Examples on
                # TPU v7 bf16 (1155 TFLOPS/s):
                #   - 1024³: 2.15 GFLOP → 1.86 us min; headline Helion
                #     ~6.12 us device → device_pct_of_min ~30% (plenty
                #     of room for kernel work — this is the canonical
                #     compute-bound headroom signal).
                #   - 1024×1×1024 / 1×1×1024 / 1×1024×1024: KFLOPs–MFLOP
                #     range → us-fraction theoretical min; structurally
                #     dispatch-bounded, device_pct_of_min sits at the
                #     chip latency floor (% of peak is irrelevant — the
                #     headroom signal these shapes carry is "no kernel
                #     headroom; this row is §6.4 territory").
                #   - 2048³ / 4096³ (manager-added large-shape extension):
                #     17 / 137 GFLOP → 14.9 / 119 us min — large enough
                #     to exercise true MXU sustained-peak work; the
                #     canonical compute-bound headroom shapes for any
                #     future G7-prefetch / G7-launch-fusion substep.
                #
                # ``device_pct_of_min`` is reported as
                # ``theoretical_min_us / device_us`` (in [0, 1]+; 1.0
                # = at theoretical MXU peak — meaning device_us equals
                # the theoretical minimum, the chip cannot finish the
                # matmul any faster; values < 1 indicate room above the
                # min). One ``theoretical_min_us`` line is emitted per
                # sweep (per timing-mode tag, identical across sweeps —
                # pure shape × dtype × peak_tflops function); one
                # ``device_pct_of_min`` line is emitted per callable
                # (Helion-full / Helion-kernel / Pallas / JAX) so the
                # per-shape headroom map is parseable per path.
                peak_tflops = (
                    args.peak_tflops_bf16
                    if args.dtype == "bfloat16"
                    else args.peak_tflops_f32
                )
                theoretical_min_us = _theoretical_min_us(m, k, n, peak_tflops)
                print(
                    f"theoretical_min_us_{m}x{k}x{n}{line_tag}: "
                    f"{theoretical_min_us:.6f} us  "
                    f"[2*M*K*N FLOPs / {peak_tflops:.1f} TFLOPS]"
                )
                helion_full_pct = theoretical_min_us / helion_full_device_us
                helion_kernel_pct = theoretical_min_us / helion_kernel_device_us
                pallas_pct = theoretical_min_us / pallas_device_us
                jax_pct = theoretical_min_us / jax_device_us
                print(
                    f"helion_full_path_device_pct_of_min_{m}x{k}x{n}"
                    f"{line_tag}: {helion_full_pct:.4f}"
                )
                print(
                    f"helion_kernel_only_device_pct_of_min_{m}x{k}x{n}"
                    f"{line_tag}: {helion_kernel_pct:.4f}"
                )
                print(
                    f"pallas_kernel_only_device_pct_of_min_{m}x{k}x{n}"
                    f"{line_tag}: {pallas_pct:.4f}"
                )
                print(
                    f"jax_kernel_only_device_pct_of_min_{m}x{k}x{n}"
                    f"{line_tag}: {jax_pct:.4f}"
                )


if __name__ == "__main__":
    main()
