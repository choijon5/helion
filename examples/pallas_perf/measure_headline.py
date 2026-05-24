"""Single-shape headline measurements for bf16 matmul (default 1024x1024x1024).

Emits two complementary metrics in one run (G2-closure dual-metric setup,
per the manager's cycle-15 decision):

1. **Full-path metric** (tracked, not gating): Helion via ``torch_tpu``
   end-to-end (the production user-facing path) vs hand-written Pallas
   via pure JAX. Captures launcher overhead AND torch_tpu C++ dispatch.
   The full-path measurement uses the **autotuner-picked** config
   (matches production behavior).

2. **Kernel-only metric** (gating for G2/G3/G4/G5): Helion's generated
   Pallas kernel pulled out of the launcher cache and invoked through
   ``jax.jit(pl.pallas_call(...))`` with JAX arrays, identical to how
   the hand-written Pallas reference is invoked. Isolates the kernel
   body from launcher / torch_tpu dispatch overhead. For shapes with a
   known-best ablated config (registered in
   ``_PINNED_KERNEL_ONLY_CONFIGS``) the kernel-only measurement uses a
   pinned config so the per-sweep signal is deterministic
   (autotuner-pick variance is the dominant noise source for
   kernel-only timing -- see plan.md §2.9 (h)).  Shapes not in the
   table fall back to the autotuner's full-path pick for the
   kernel-only timing, which gives an apples-to-apples kernel-vs-
   kernel comparison without speculative per-shape pinning.  A
   ``--kernel-only-config`` CLI flag overrides both the pin table and
   the autotuner fallback for one-off ablation runs.

The kernel-only path is built by patching ``helion.runtime._pallas_build_callable``
to stash the JAX ``jit_fn`` argument (``pl.pallas_call(...)``) right
before Helion wraps it in torch_tpu's ``JaxCallable`` (which throws away
the original ``jit_fn`` reference by ``jax.export``-serializing the body
into a binary blob bound to ``call_custom_kernel``). The captured
``jit_fn`` is re-wrapped in ``jax.jit`` to match the JaxCallable
construction site, and we time it directly with JAX inputs -- identical
to the way the hand-written ``pallas_matmul`` reference is called.

CLI:

```
python measure_headline.py                    # default bf16 1024x1024x1024
python measure_headline.py --shape 1024 128 1024
```

Output (parseable lines, one per metric):

```
helion_full_path_bf16_<M>x<K>x<N> [autotuner pick: <config>]: median=<us> us
helion_kernel_only_bf16_<M>x<K>x<N> [<pin label>]: median=<us> us
pallas_kernel_only_bf16_<M>x<K>x<N>: median=<us> us
full_path_H_over_P: <ratio>
kernel_only_H_over_P: <ratio>
launcher_overhead_us: <helion_full - helion_kernel_only> us
```

The ``helion_bf16_<shape>`` legacy line (full-path median) is also
emitted for back-compat with older log scrapers.

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

# Force full autotuning effort to mirror the production harness; set before
# importing helion so the value is picked up at autotuner initialization.
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "full")

import jax
import jax.numpy as jnp
import numpy as np
import torch

import helion
from helion._testing import DEVICE
from helion.autotuner.benchmarking import synchronize_device

# Import the kernel from matmul_helion so any kernel-side change is picked
# up by both the full harness and this single-shape probe.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matmul_helion import helion_matmul_kernel  # pyrefly: ignore [missing-import]
from matmul_pallas import pallas_matmul  # pyrefly: ignore [missing-import]

# Default (headline) shape for back-compat with the legacy single-shape probe.
_DEFAULT_SHAPE: tuple[int, int, int] = (1024, 1024, 1024)

# Per-shape pinned configs for the kernel-only measurement. The headline
# (1024, 1024, 1024) entry is the Deep Replan §2.5 row 2 known-best
# ``emit_pipeline [512, 512, 512] pb=False`` family (161us full-path /
# ~126us kernel-only); the G3-A entries are the per-shape ablation winners
# (single-sweep us picked from a 3-4 candidate set per shape, then verified
# 5-sweep). Kernel-only timing is deterministic at these pinned configs
# (no autotuner picks leaking into the per-sweep signal), unlike the
# full-path measurement which still goes through the autotuner per
# plan.md §7.1. Shapes not in the table fall back to the autotuner's
# full-path pick for kernel-only too (see ``_resolve_kernel_only_config``
# below).
_PINNED_KERNEL_ONLY_CONFIGS: dict[tuple[int, int, int], helion.Config] = {
    (1024, 1024, 1024): helion.Config(
        block_sizes=[512, 512, 512],
        pallas_loop_type="emit_pipeline",
        pallas_pre_broadcast=False,
    ),
    (1024, 1024, 1): helion.Config(
        block_sizes=[1024, 1024, 1],
        pallas_loop_type="unroll",
        pallas_pre_broadcast=True,
    ),
    (1024, 128, 1024): helion.Config(
        block_sizes=[1024, 128, 128],
        pallas_loop_type="emit_pipeline",
        pallas_pre_broadcast=False,
    ),
    (128, 1024, 1024): helion.Config(
        block_sizes=[128, 1024, 1024],
        pallas_loop_type="unroll",
        pallas_pre_broadcast=True,
    ),
}


def _pin_label(config: helion.Config) -> str:
    loop_type = config.config.get("pallas_loop_type", "unroll")
    pre_broadcast = bool(config.config.get("pallas_pre_broadcast", False))
    return (
        f"pinned: {loop_type} {list(config.block_sizes)} "
        f"pb={'T' if pre_broadcast else 'F'}"
    )


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


def _resolve_kernel_only_config(
    shape: tuple[int, int, int],
    autotuned_config: helion.Config,
    override_config: helion.Config | None = None,
) -> tuple[helion.Config | None, str]:
    """Pick the kernel-only Helion config for ``shape``.

    Resolution order:
      1. ``override_config`` (CLI ``--kernel-only-config``) wins
         outright -- the caller pinned a specific config for one-off
         ablation; pin it without consulting the table.
      2. Shapes registered in ``_PINNED_KERNEL_ONLY_CONFIGS`` use their
         per-shape pin -- the G2 closure / G3-A-pin known-bests.
      3. Otherwise return ``(None, ...)`` so the caller reuses the
         autotuner's full-path pick for the kernel-only timing too --
         still apples-to-apples vs the hand-written Pallas reference at
         its own best block.

    Returns ``(config_to_recompile, label)``. When ``config_to_recompile``
    is ``None``, the caller should reuse the already-captured full-path
    autotuner ``jit_fn``; otherwise the caller should ``compile_config``
    the returned config to trigger a fresh ``_pallas_build_callable``
    capture.
    """
    if override_config is not None:
        return override_config, _pin_label(override_config) + " (CLI override)"
    if shape in _PINNED_KERNEL_ONLY_CONFIGS:
        config = _PINNED_KERNEL_ONLY_CONFIGS[shape]
        return config, _pin_label(config)
    return None, f"autotuner pick: {autotuned_config}"


def _parse_kernel_only_config(spec: str) -> helion.Config:
    """Parse a ``--kernel-only-config`` argument of the form
    ``<loop_type>:<bm>,<bk>,<bn>:<pb>`` (e.g. ``emit_pipeline:512,512,512:F``).

    Used by per-shape ablation harnesses that drive ``measure_headline.py``
    once per candidate config to time the kernel-only path. ``<pb>`` is
    ``T`` or ``F`` for ``pallas_pre_broadcast``.
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected --kernel-only-config of form "
            f"'<loop_type>:<bm>,<bk>,<bn>:<T|F>' (got {spec!r})"
        )
    loop_type, block_str, pb_str = parts
    block_sizes = [int(x) for x in block_str.split(",")]
    if len(block_sizes) != 3:
        raise argparse.ArgumentTypeError(
            f"expected 3 block sizes in --kernel-only-config (got {block_str!r})"
        )
    if pb_str not in ("T", "F"):
        raise argparse.ArgumentTypeError(
            "expected T or F for pre_broadcast in --kernel-only-config "
            f"(got {pb_str!r})"
        )
    return helion.Config(
        block_sizes=block_sizes,
        pallas_loop_type=loop_type,
        pallas_pre_broadcast=(pb_str == "T"),
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Helion (full-path + kernel-only) vs hand-written Pallas "
            "for a single bf16 matmul shape. Defaults to the bf16 1024x1024x1024 "
            "headline."
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
        "--kernel-only-config",
        type=_parse_kernel_only_config,
        default=None,
        metavar="LOOP_TYPE:BM,BK,BN:PB",
        help=(
            "Override the per-shape pin table for the kernel-only timing. "
            "Format: '<loop_type>:<bm>,<bk>,<bn>:<T|F>' "
            "(e.g. 'emit_pipeline:512,512,512:F'). Used by per-shape "
            "ablation harnesses; not needed for routine cycle runs."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    m, k, n = args.shape
    shape: tuple[int, int, int] = (m, k, n)

    torch.manual_seed(0)
    x_torch = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE)
    torch.manual_seed(1)
    y_torch = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE)

    # Install the ``_pallas_build_callable`` capture so we can lift the
    # final compiled config's ``jit_fn`` for the kernel-only path.
    _install_jit_fn_capture()

    # ----- Helion full-path: bind + autotune + compile + run (production path).
    # Same ``bound`` instance is reused for any pinned compile_config below
    # (the Helion compile cache keys on Config equality so the two compiles
    # produce different cache entries -- and therefore different
    # ``_pallas_build_callable`` invocations + captures).
    bound = helion_matmul_kernel.bind((x_torch, y_torch))
    best_config = bound.autotune((x_torch, y_torch), force=True)
    print(f"Optimal autotuned config: {best_config}", file=sys.stderr)
    compiled_fn = bound.compile_config(best_config, allow_print=False)

    def _run_full_path() -> None:
        out = compiled_fn(x_torch, y_torch)
        synchronize_device(out)

    helion_full_us = _time(_run_full_path)
    # Back-compat line for older log scrapers.
    print(f"helion_bf16_{m}x{k}x{n}: median={helion_full_us:.2f} us")
    print(
        f"helion_full_path_bf16_{m}x{k}x{n} "
        f"[autotuner pick: {best_config}]: median={helion_full_us:.2f} us"
    )

    # Snapshot the autotuner-picked ``jit_fn`` (captured during the
    # ``compile_config(best_config)`` above) before any further
    # ``compile_config`` calls overwrite the module-level slot. Non-headline
    # shapes time the kernel-only metric against this snapshot.
    autotuned_jit_fn = _find_helion_jit_fn()

    # ----- Helion kernel-only: compile the pinned config explicitly so the
    # capture patch records a deterministic, known-best kernel jit_fn rather
    # than whatever the autotuner happened to pick this run (the autotuner
    # optimises *full-path* time, which is not necessarily optimal for the
    # kernel-only signal; per-sweep autotuner-pick variance is the dominant
    # source of kernel-only timing noise -- see plan.md §2.9 (h)).
    kernel_only_config, kernel_only_label = _resolve_kernel_only_config(
        shape, best_config, override_config=args.kernel_only_config
    )
    if kernel_only_config is not None:
        pinned_compiled_fn = bound.compile_config(kernel_only_config, allow_print=False)
        # The pinned compile_config caches a fresh pallas_kernel; running it
        # once triggers ``_pallas_build_callable`` and the capture patch
        # records the pinned config's ``jit_fn`` as the most-recent capture.
        pinned_compiled_fn(x_torch, y_torch)
        synchronize_device(x_torch)
        jit_fn = _find_helion_jit_fn()
    else:
        # No pin candidate for this shape; reuse the autotuner-picked
        # ``jit_fn`` captured above.
        jit_fn = autotuned_jit_fn

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

    def _run_helion_kernel_only() -> None:
        out = jit_fn(x_jax, y_jax)
        jax.block_until_ready(out)

    helion_kernel_us = _time(_run_helion_kernel_only)
    print(
        f"helion_kernel_only_bf16_{m}x{k}x{n} "
        f"[{kernel_only_label}]: median={helion_kernel_us:.2f} us"
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
    print(f"pallas_kernel_only_bf16_{m}x{k}x{n}: median={pallas_kernel_us:.2f} us")

    # ----- Derived ratios.
    full_path_h_over_p = pallas_kernel_us / helion_full_us
    kernel_only_h_over_p = pallas_kernel_us / helion_kernel_us
    launcher_overhead_us = helion_full_us - helion_kernel_us

    print(f"full_path_H_over_P: {full_path_h_over_p:.3f}")
    print(f"kernel_only_H_over_P: {kernel_only_h_over_p:.3f}")
    print(f"launcher_overhead_us: {launcher_overhead_us:.2f} us")


if __name__ == "__main__":
    main()
