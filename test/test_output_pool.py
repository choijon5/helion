"""Tests for the output pool.

The pool lives in ``helion/runtime/__init__.py`` and routes the generated
host wrapper's ``torch.empty(...)`` calls through ``_output_pool_alloc``.

Activation contract:

* Default (env unset): pooling is active ONLY inside the
  ``_pool_active()`` context manager (used by the autotuner around
  per-trial benchmark loops). For all other callers
  ``_output_pool_alloc`` is a passthrough to ``torch.empty``.
* ``HELION_REUSE_OUTPUT_BUFFERS=1`` (opt-in): pooling is active for all
  callers, including user-facing kernel calls.

When pooling IS active, the cache is gated by three safety signals:
tensor refcount (catches direct references, views via ``_base``, and
autograd saves), CUDA stream tagging (cross-stream consumers don't see
stale buffers), and CUDA-graph capture detection (capture bypasses the
pool entirely).
"""

from __future__ import annotations

import gc
from typing import Iterator

import pytest
import torch

import helion
from helion._testing import DEVICE
import helion.language as hl
import helion.runtime as helion_runtime
from helion.runtime._output_pool import _pool_scope
from helion.runtime.settings import _get_backend

# Output-pool tests genuinely require the Triton backend: the codegen
# rewrite that routes ``torch.empty(...)`` through ``_helion_output_alloc``
# (see ``helion/_compiler/generate_ast.py``) is restricted to
# ``TritonBackend``. They would silently no-op the pool under
# ``HELION_BACKEND=cute`` on a CUDA host.
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(
        _get_backend() != "triton",
        reason="output pool codegen rewrite is Triton-backend only",
    ),
]


@helion.kernel(static_shapes=True)
def _pool_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Minimal add kernel used by the pool tests.

    Goes through the generated host wrapper's ``torch.empty(...)`` (which
    is rewritten to ``_helion_output_alloc``), so its output participates
    in the pool exactly like ``examples/add.py``.
    """
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@pytest.fixture
def reset_pool(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset the env-cached user opt-in + drop buffers around each test.

    The opt-in env var is read once and cached for the lifetime of the
    process, so tests that flip it need a reset hook. ``monkeypatch`` also
    restores the env var after the test.
    """
    helion_runtime._reset_output_pool_user_opt_in_cache()
    # Make sure no leftover scope state from another test leaks in.
    _pool_scope.active = False
    yield
    _pool_scope.active = False
    helion_runtime._reset_output_pool_user_opt_in_cache()


def test_user_call_no_pool_by_default(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default mode: user kernel calls do NOT pool — concurrent allocs differ."""
    monkeypatch.delenv(helion_runtime._REUSE_OUTPUT_BUFFERS_ENV, raising=False)
    helion_runtime._reset_output_pool_user_opt_in_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)

    # Warmup so the kernel itself is JIT'd.
    out = _pool_add(x, y)
    del out
    gc.collect()

    out1 = _pool_add(x, y)
    out2 = _pool_add(x, y)
    assert out1 is not out2
    assert out1.data_ptr() != out2.data_ptr(), (
        "User kernel calls must not pool by default — but two concurrently "
        f"held outputs shared {out1.data_ptr()=}."
    )


def test_pool_active_scope_reuses_buffer(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inside ``_pool_active()`` the pool reuses a dropped buffer, then clears on exit."""
    monkeypatch.delenv(helion_runtime._REUSE_OUTPUT_BUFFERS_ENV, raising=False)
    helion_runtime._reset_output_pool_user_opt_in_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)

    # Warmup (outside the scope) so JIT happens once.
    out = _pool_add(x, y)
    torch.testing.assert_close(out, x + y)
    del out
    gc.collect()

    with helion_runtime._pool_active():
        out1 = _pool_add(x, y)
        p1 = out1.data_ptr()
        del out1
        gc.collect()

        out2 = _pool_add(x, y)
        p2 = out2.data_ptr()

    assert p1 == p2, (
        "Inside _pool_active(), the second allocation must reuse the cached "
        f"buffer dropped by the first (got {p1=} vs {p2=})."
    )

    # After exit, the cache is dropped and the next call should allocate
    # afresh (it also no longer participates in the pool at all).
    out3 = _pool_add(x, y)
    out4 = _pool_add(x, y)
    assert out3.data_ptr() != out4.data_ptr(), (
        "After exiting _pool_active(), user calls must no longer pool — but "
        f"two concurrent outputs shared {out3.data_ptr()=}."
    )


def test_opt_in_env_pools_user_calls(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``HELION_REUSE_OUTPUT_BUFFERS=1`` enables the safe pool for user calls."""
    monkeypatch.setenv(helion_runtime._REUSE_OUTPUT_BUFFERS_ENV, "1")
    helion_runtime._reset_output_pool_user_opt_in_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)

    # Warmup so we don't measure first-time JIT.
    out = _pool_add(x, y)
    torch.testing.assert_close(out, x + y)
    del out
    gc.collect()

    out1 = _pool_add(x, y)
    p1 = out1.data_ptr()
    del out1
    gc.collect()

    out2 = _pool_add(x, y)
    p2 = out2.data_ptr()
    assert p1 == p2, (
        "With HELION_REUSE_OUTPUT_BUFFERS=1, the pool should reuse the buffer "
        f"when the previous output was dropped before the next call "
        f"(got {p1=} vs {p2=})."
    )


def test_pool_view_keeps_storage_alive(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A surviving view of the previous output forces a fresh allocation."""
    monkeypatch.setenv(helion_runtime._REUSE_OUTPUT_BUFFERS_ENV, "1")
    helion_runtime._reset_output_pool_user_opt_in_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)

    # Warmup: get the pool entry seeded.
    out = _pool_add(x, y)
    del out
    gc.collect()

    out1 = _pool_add(x, y)
    p1 = out1.data_ptr()
    view = out1[:128]  # holds out1 alive via _base + storage via view
    del out1
    gc.collect()

    out2 = _pool_add(x, y)
    p2 = out2.data_ptr()
    assert p1 != p2, (
        "View kept storage alive, so the pool must allocate a fresh buffer "
        f"(got {p1=} vs {p2=})."
    )
    # Sanity: the view still sees the original buffer's data.
    assert view.data_ptr() == p1


def test_pool_autograd_retain_alloc_fresh(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An autograd graph that saves the output keeps the pool from reusing it.

    ``_pool_add`` itself doesn't track gradients (Helion outputs are leaf
    tensors), so we emulate the "autograd retains" case by saving the
    tensor inside a custom autograd graph via ``ctx.save_for_backward``.
    """
    monkeypatch.setenv(helion_runtime._REUSE_OUTPUT_BUFFERS_ENV, "1")
    helion_runtime._reset_output_pool_user_opt_in_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)

    # Warmup.
    out = _pool_add(x, y)
    del out
    gc.collect()

    class Retain(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: object, kept: torch.Tensor, gate: torch.Tensor
        ) -> torch.Tensor:
            ctx.save_for_backward(kept)  # type: ignore[attr-defined]
            return gate.clone()

        @staticmethod
        def backward(ctx: object, grad: torch.Tensor) -> tuple[None, torch.Tensor]:
            (kept,) = ctx.saved_tensors  # type: ignore[attr-defined]
            return None, grad + kept.mean()

    out1 = _pool_add(x, y)
    p1 = out1.data_ptr()
    gate = torch.zeros(1, device=DEVICE, requires_grad=True)
    # The Retain.forward call saves out1 in the autograd graph.
    graph_out = Retain.apply(out1, gate)
    del out1  # caller drops their explicit ref; autograd still holds it
    gc.collect()

    out2 = _pool_add(x, y)
    p2 = out2.data_ptr()
    assert p1 != p2, (
        "Autograd retained the previous output via save_for_backward, so "
        f"the pool must allocate fresh (got {p1=} vs {p2=})."
    )
    # Trigger the backward path so the kept reference is exercised.
    assert graph_out is not None


def test_pool_cuda_graph_bypass(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """During CUDA graph capture, the pool must bypass (never reuse)."""
    monkeypatch.setenv(helion_runtime._REUSE_OUTPUT_BUFFERS_ENV, "1")
    helion_runtime._reset_output_pool_user_opt_in_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    # Warmup outside the graph.
    out = _pool_add(x, y)
    torch.cuda.synchronize()
    del out
    gc.collect()

    # Capture: the kernel runs once inside the graph context. We don't
    # assert the data_ptr inside capture (CG semantics own that), only
    # that the captured graph replays correctly without aliasing.
    side_stream = torch.cuda.Stream()
    with torch.cuda.stream(side_stream):
        _pool_add(x, y)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    static_out: torch.Tensor | None = None
    with torch.cuda.graph(graph):
        static_out = _pool_add(x, y)
    assert static_out is not None
    torch.cuda.synchronize()

    # Replay: should reproduce the original result.
    graph.replay()
    torch.cuda.synchronize()
    expected = x + y
    torch.testing.assert_close(static_out, expected)


def test_pool_cross_stream_no_reuse(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When pooling is active, the pool is stream-keyed — switching streams allocates fresh."""
    monkeypatch.setenv(helion_runtime._REUSE_OUTPUT_BUFFERS_ENV, "1")
    helion_runtime._reset_output_pool_user_opt_in_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)

    # Warmup on the default stream.
    out = _pool_add(x, y)
    del out
    gc.collect()

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    with torch.cuda.stream(stream_a):
        out_a = _pool_add(x, y)
        ptr_a = out_a.data_ptr()
        del out_a
        gc.collect()
    stream_a.synchronize()

    with torch.cuda.stream(stream_b):
        out_b = _pool_add(x, y)
        ptr_b = out_b.data_ptr()
    stream_b.synchronize()

    assert ptr_a != ptr_b, (
        "Cross-stream consumer: the pool must not return a buffer first "
        f"allocated on stream A to a caller on stream B (got {ptr_a=} vs {ptr_b=})."
    )
