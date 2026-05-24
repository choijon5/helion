"""Single-shape headline measurements for bf16 matmul (default 1024x1024x1024).

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


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Helion (real-user full-path + real-user kernel-only) "
            "vs hand-written Pallas for a single bf16 matmul shape. "
            "Defaults to the bf16 1024x1024x1024 headline. The Helion "
            "autotuner is seeded via ``HELION_AUTOTUNE_RANDOM_SEED`` so "
            "every run picks the same config — real-user metric, "
            "reproducible."
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    m, k, n = args.shape

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
    x_torch = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE)
    torch.manual_seed(1)
    y_torch = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE)

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
    # Back-compat line for older log scrapers.
    print(f"helion_bf16_{m}x{k}x{n}: median={helion_full_us:.2f} us")
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
    x_jax = jax.device_put(jax.random.normal(key0, (m, k), dtype=jnp.bfloat16))
    y_jax = jax.device_put(jax.random.normal(key1, (k, n), dtype=jnp.bfloat16))
    jax.block_until_ready((x_jax, y_jax))

    # ----- Helion kernel-only: time the autotuner-picked ``jit_fn``
    # (what real users get) through the pure-JAX path.
    def _run_helion_kernel_only() -> None:
        out = autotuned_jit_fn(x_jax, y_jax)
        jax.block_until_ready(out)

    helion_kernel_us = _time(_run_helion_kernel_only)
    print(
        f"helion_kernel_only_{m}x{k}x{n} "
        f"[autotuner pick: {best_config}, seed={seed}]: "
        f"median={helion_kernel_us:.2f} us"
    )

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

    pallas_kernel_us = _time(_run_pallas_kernel_only)
    print(f"pallas_kernel_only_{m}x{k}x{n}: median={pallas_kernel_us:.2f} us")

    # ----- Derived ratios.
    full_h_over_p = pallas_kernel_us / helion_full_us
    kernel_h_over_p = pallas_kernel_us / helion_kernel_us
    launcher_overhead_us = helion_full_us - helion_kernel_us

    print(f"full_path_H_over_P: {full_h_over_p:.3f}")
    print(f"kernel_only_H_over_P: {kernel_h_over_p:.3f}")
    print(f"launcher_overhead_us: {launcher_overhead_us:.2f} us")


if __name__ == "__main__":
    main()
