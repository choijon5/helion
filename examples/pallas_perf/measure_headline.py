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

This script now emits **two** Helion measurements per shape (both at
the autotuner-picked config; pinning is gone):

1. **Real-user full-path** (production behavior): Helion via
   ``torch_tpu`` end-to-end at the autotuner-picked config. Includes
   launcher overhead AND torch_tpu C++ dispatch.
2. **Real-user kernel-only** (gating since cycle 18): Helion's
   generated Pallas kernel pulled out of the launcher cache at the
   *autotuner-picked* config and invoked through
   ``jax.jit(pl.pallas_call(...))`` with JAX arrays, identical to the
   hand-written Pallas reference. Isolates the kernel body from
   launcher / torch_tpu dispatch overhead.

CLI:

```
python measure_headline.py                    # default bf16 1024x1024x1024
python measure_headline.py --shape 1024 128 1024
python measure_headline.py --dtype float32 --shape 1024 1024 1024   # G4 f32 path
python measure_headline.py --seed 7           # override default seed (0)
```

Output (parseable lines, one per metric):

```
helion_bf16_<M>x<K>x<N>: median=<us> us                                                  # back-compat full-path
helion_full_path_<M>x<K>x<N> [autotuner pick: <config>, seed=<n>]: median=<us> us
helion_kernel_only_<M>x<K>x<N> [autotuner pick: <config>, seed=<n>]: median=<us> us
pallas_kernel_only_<M>x<K>x<N>: median=<us> us
full_path_H_over_P: <ratio>
kernel_only_H_over_P: <ratio>     # GATING since cycle 18 (real-user, seeded autotuner)
launcher_overhead_us: <full - kernel> us
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

Timing convention matches the production harness everywhere: 20 iters
x 5 repeats, warmup excluded, ``synchronize_device`` (or
``jax.block_until_ready``) between calls.
"""

from __future__ import annotations

import argparse
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
            mode_tag = "[sequential] " if tag_lines else ""
        else:  # interleaved
            helion_kernel_us, pallas_kernel_us = _time_interleaved(
                _run_helion_kernel_only, _run_pallas_kernel_only
            )
            mode_tag = "[interleaved] " if tag_lines else ""

        # Pre-shape tag (after the shape suffix, before the rest of the
        # line) preserves the bit-identical legacy single-mode output;
        # tagged output for the both-mode comparison adds the tag suffix
        # to the metric name itself so each metric is uniquely parseable
        # (``kernel_only_H_over_P_sequential`` / ``..._interleaved``).
        ratio_tag = f"_{mode}" if tag_lines else ""
        line_tag = f" {mode_tag.strip()}" if tag_lines else ""
        print(
            f"helion_kernel_only_{m}x{k}x{n}{line_tag} "
            f"[autotuner pick: {best_config}, seed={seed}]: "
            f"median={helion_kernel_us:.2f} us"
        )
        print(
            f"pallas_kernel_only_{m}x{k}x{n}{line_tag}: "
            f"median={pallas_kernel_us:.2f} us"
        )

        # ----- Derived ratios.
        full_h_over_p = pallas_kernel_us / helion_full_us
        kernel_h_over_p = pallas_kernel_us / helion_kernel_us
        launcher_overhead_us = helion_full_us - helion_kernel_us

        print(f"full_path_H_over_P{ratio_tag}: {full_h_over_p:.3f}")
        print(f"kernel_only_H_over_P{ratio_tag}: {kernel_h_over_p:.3f}")
        print(f"launcher_overhead_us{ratio_tag}: {launcher_overhead_us:.2f} us")


if __name__ == "__main__":
    main()
