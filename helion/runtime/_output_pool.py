"""Output buffer pool for kernel-output ``torch.empty`` allocations.

The generated host wrapper's ``torch.empty(...)`` calls that produce
kernel-output tensors are routed through ``_output_pool_alloc(...)``
(see ``helion/_compiler/generate_ast.py``), which can reuse buffers
across calls to amortize the per-call ``cudaMalloc`` cost (~8 µs per
launch on small kernels).

Activation contract:

* **Default (env unset).** Pooling is active ONLY inside the
  ``_pool_active()`` context manager. The autotuner enters that scope
  around per-trial benchmark loops so trial allocations hit a buffer
  cache instead of ``torch.empty``. For all other callers (i.e.
  user-facing kernel invocations) ``_output_pool_alloc`` is a plain
  passthrough to ``torch.empty`` — there is no behavior change from
  pre-pool semantics and no aliasing risk for user code that retains
  prior outputs.

* **``HELION_REUSE_OUTPUT_BUFFERS=1`` (opt-in).** Pooling is active for
  every caller, including user kernel calls. Useful for tight inference
  loops where the user knows they drop the previous output before the
  next call and wants the per-launch cudaMalloc removed.

When pooling IS active, safety is preserved by three signals that
together cover ≥99% of usage:

  1. Tensor reference count: a cached tensor has exactly 3 refs (dict
     slot + the local variable holding the lookup result + the
     ``sys.getrefcount`` arg) when no external code holds it. If the
     refcount exceeds 3 — either because the user kept the previous
     output, took a view (PyTorch view ops bump the parent's refcount
     via ``_base`` for autograd version tracking), or autograd saved
     it for backward — we allocate fresh instead.

  2. CUDA stream tagging: the cache key includes the current CUDA
     stream id. A buffer first produced on stream A is never returned
     to a caller running on stream B, so cross-stream consumers don't
     observe stale data.

  3. CUDA graph capture: when ``torch.cuda.is_current_stream_capturing()``
     is True, the pool is bypassed entirely (always ``torch.empty``).
     CUDA graph buffer reuse is decided by the graph builder, not by
     runtime liveness, so the pool would interfere otherwise.

Concurrency note: the safety guarantees here are per Python process
(single-Python-thread), not thread-safe across multiple Python threads
that share a Kernel. The refcount-based liveness check between the
``dict.get`` and the ``return`` is not atomic across a thread switch.
Real Helion workloads are GPU-driven on a single Python thread, so this
matches existing usage.
"""

from __future__ import annotations

import os
import sys
import threading
from typing_extensions import Self

import torch

_REUSE_OUTPUT_BUFFERS_ENV = "HELION_REUSE_OUTPUT_BUFFERS"
_reuse_outputs_user_opt_in: bool | None = None

# Thread-local flag toggled by the ``_pool_active`` context manager.
# Using ``threading.local`` so that ad-hoc multi-threaded callers (e.g.
# tests, parallel benchmark drivers) don't cross-contaminate each other's
# pool activation state.
_pool_scope = threading.local()


class _pool_active:
    """Enable the safe output pool for kernel calls inside this scope.

    Used internally by the autotuner so per-trial allocations hit a buffer
    cache instead of ``torch.empty``, removing the per-launch cudaMalloc
    overhead from trial timing. The pool is NOT active for user-facing
    kernel calls outside this scope unless ``HELION_REUSE_OUTPUT_BUFFERS=1``
    is set.

    On exit, drops cached buffers so they don't leak across scopes (e.g.
    between consecutive autotune runs, or between an autotune scope and a
    later user-facing call sequence).
    """

    def __enter__(self) -> Self:
        _pool_scope.active = True
        return self

    def __exit__(self, *exc: object) -> None:
        _pool_scope.active = False
        _output_pool_safe_cache.clear()


# Bind the low-level C entry points once at import time so the hot path
# avoids the Python wrapper layer. ``torch.cuda.current_stream(device)`` is
# ~10× slower than ``torch._C._cuda_getCurrentStream(device_index)`` and
# the public ``is_current_stream_capturing`` wrapper has a similar
# overhead profile.
_cuda_is_capturing = getattr(torch._C, "_cuda_isCurrentStreamCapturing", lambda: False)
_cuda_get_current_stream = getattr(
    torch._C, "_cuda_getCurrentStream", lambda _idx: (0, 0, 1)
)

# Safe-mode cache: keyed on (dtype, shape, device, stream_id). Values are
# strong references to the cached tensor; ``_output_pool_alloc`` uses
# ``sys.getrefcount`` to decide whether external code also holds the
# tensor (in which case we allocate fresh instead of reusing).
_output_pool_safe_cache: dict[
    tuple[torch.dtype, tuple[int, ...], torch.device, int], torch.Tensor
] = {}

# ``sys.getrefcount(x)`` always counts the temporary binding it makes for
# its argument, so a value held by exactly ``(dict slot, local var)``
# reports a refcount of 3. Anything higher means external code holds it.
_OUTPUT_POOL_BASELINE_REFCOUNT = 3


def _resolve_user_opt_in() -> bool:
    """Read ``HELION_REUSE_OUTPUT_BUFFERS`` once and cache the result.

    The env var is read at most once per process so the hot path stays a
    single dict lookup + boolean check.
    """
    global _reuse_outputs_user_opt_in
    if _reuse_outputs_user_opt_in is None:
        raw = os.environ.get(_REUSE_OUTPUT_BUFFERS_ENV, "").strip().lower()
        _reuse_outputs_user_opt_in = raw in ("1", "true", "on", "yes")
    return _reuse_outputs_user_opt_in


def _output_pool_alloc(
    *shape_args: object,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    **extra_kwargs: object,
) -> torch.Tensor:
    """Pooled replacement for ``torch.empty(*shape, dtype=..., device=...)``.

    Matches ``torch.empty``'s overloaded shape signature: callers can pass
    a single tuple/list ``([1024, 1024])`` or multiple positional dims
    ``(M, N)``. Generated host wrappers emit whichever the user wrote, so
    both shapes need to work transparently. ``dtype`` and ``device`` are
    optional to mirror ``torch.empty``'s defaults; ``extra_kwargs`` is a
    catch-all (e.g. ``pin_memory``) forwarded to ``torch.empty`` on the
    miss path.

    Pooling is active only when running inside ``_pool_active()`` (used by
    the autotuner) or when ``HELION_REUSE_OUTPUT_BUFFERS=1`` is set —
    otherwise this is a passthrough to ``torch.empty``. See the module
    docstring above for full details. Returns a tensor whose contents are
    undefined (same contract as ``torch.empty``); the kernel is expected
    to fully write it before the caller reads.
    """
    if not (getattr(_pool_scope, "active", False) or _resolve_user_opt_in()):
        # Fast path: no pooling. Pure passthrough to ``torch.empty`` so
        # the user sees exact pre-pool semantics. Kept lean to minimize
        # overhead for the (default) case.
        kwargs: dict[str, object] = dict(extra_kwargs)
        if dtype is not None:
            kwargs["dtype"] = dtype
        if device is not None:
            kwargs["device"] = device
        return torch.empty(*shape_args, **kwargs)  # type: ignore[arg-type]
    # Normalize ``shape_args`` → a flat tuple of ints (or a tuple containing
    # one tuple/list/torch.Size that we flatten). Mirrors torch.empty's
    # accepted overloads.
    if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list, torch.Size)):
        shape_t = tuple(shape_args[0])
    else:
        shape_t = tuple(shape_args)  # type: ignore[arg-type]
    # Resolve defaults the same way torch.empty does so the cache key
    # captures the actual realized dtype/device. Use ``get_default_device``
    # rather than ``torch.tensor(0.0).device`` so the device-defaulting
    # branch (theoretical — generated wrappers always pass ``device=``)
    # doesn't allocate a throwaway tensor per call.
    resolved_dtype = dtype if dtype is not None else torch.get_default_dtype()
    resolved_device = device if device is not None else torch.get_default_device()
    if isinstance(resolved_device, str):
        resolved_device = torch.device(resolved_device)
    if extra_kwargs:
        # Uncommon: extra kwargs (e.g. ``layout``, ``pin_memory``) change
        # storage characteristics, so don't pool — just delegate.
        # pyrefly: ignore [no-matching-overload]
        return torch.empty(
            shape_t, dtype=resolved_dtype, device=resolved_device, **extra_kwargs
        )
    # Pool active. Three guards: CG capture bypass, stream-keyed cache,
    # and a refcount-based liveness check on cache hit.
    if resolved_device.type == "cuda":
        # Use the lowest-level C APIs available — the public Python
        # wrappers (``torch.cuda.is_current_stream_capturing``,
        # ``torch.cuda.current_stream``) are ~10× slower per call.
        if _cuda_is_capturing():
            # CUDA graph capture handles buffer reuse at graph-build time;
            # the runtime pool must not interfere.
            # pyrefly: ignore [no-matching-overload]
            return torch.empty(shape_t, dtype=resolved_dtype, device=resolved_device)
        device_index = resolved_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        # ``_cuda_getCurrentStream`` returns ``(stream_id, device, raw_id)``;
        # we use the first element as the stream tag.
        stream_id = _cuda_get_current_stream(device_index)[0]
    else:
        # Non-CUDA devices: the stream concept doesn't apply, but we still
        # want pooling per device. Use a constant stream tag.
        stream_id = 0
    key_safe = (resolved_dtype, shape_t, resolved_device, stream_id)
    # pyrefly: ignore [bad-argument-type]
    cached = _output_pool_safe_cache.get(key_safe)
    if cached is not None:
        # sys.getrefcount returns one more than the "real" count because
        # of the temporary binding it creates for its argument. Our pool
        # contributes (dict slot, ``cached`` local) → 2 + 1 (getrefcount
        # arg) = 3 baseline. Anything higher means at least one external
        # holder (direct ref, view via ``_base``, or autograd save).
        if sys.getrefcount(cached) <= _OUTPUT_POOL_BASELINE_REFCOUNT:
            return cached
    # pyrefly: ignore [no-matching-overload]
    fresh = torch.empty(shape_t, dtype=resolved_dtype, device=resolved_device)
    # pyrefly: ignore [unsupported-operation]
    _output_pool_safe_cache[key_safe] = fresh
    return fresh


def output_pool_clear() -> None:
    """Drop all pooled output buffers (for tests that need fresh storage)."""
    _output_pool_safe_cache.clear()


def _reset_output_pool_user_opt_in_cache() -> None:
    """Reset the cached env-var lookup so tests can flip the opt-in mid-process."""
    global _reuse_outputs_user_opt_in
    _reuse_outputs_user_opt_in = None
    output_pool_clear()
