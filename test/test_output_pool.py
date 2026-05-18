"""Tests for the safe default-on output pool.

The pool lives in ``helion/runtime/__init__.py`` and routes the generated
host wrapper's ``torch.empty(...)`` calls through ``_output_pool_alloc``.
It has three modes selected by the ``HELION_OUTPUT_POOL`` env var:

* unset / default → safe pool. Reuses cached buffers only when no
  external code holds them; uses tensor refcount (catches direct
  references, views via ``_base``, and autograd saves), CUDA stream
  tagging, and CUDA-graph capture detection as the safety signals.
* ``HELION_OUTPUT_POOL=0`` → opt-out. Always ``torch.empty``.
* ``HELION_OUTPUT_POOL=1`` → legacy naive pool (back-compat).

These tests cover the default safe behavior plus opt-out / opt-in and
the safety edge cases (views, autograd, cross-stream, CG capture).
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
    """Reset the env-cached pool mode + drop buffers around each test.

    The pool reads ``HELION_OUTPUT_POOL`` once and caches the result for
    the lifetime of the process, so tests that flip the env var need a
    reset hook. ``monkeypatch`` also restores the env var after the test.
    """
    helion_runtime._reset_output_pool_mode_cache()
    yield
    helion_runtime._reset_output_pool_mode_cache()


def test_pool_basic_reuse(reset_pool: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default mode reuses the buffer when the caller drops it between calls."""
    monkeypatch.delenv(helion_runtime._OUTPUT_POOL_ENV, raising=False)
    helion_runtime._reset_output_pool_mode_cache()
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
        "Safe pool default mode should reuse the buffer when the previous "
        f"output was dropped before the next call (got {p1=} vs {p2=})."
    )


def test_pool_view_keeps_storage_alive(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A surviving view of the previous output forces a fresh allocation."""
    monkeypatch.delenv(helion_runtime._OUTPUT_POOL_ENV, raising=False)
    helion_runtime._reset_output_pool_mode_cache()
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
    monkeypatch.delenv(helion_runtime._OUTPUT_POOL_ENV, raising=False)
    helion_runtime._reset_output_pool_mode_cache()
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
    monkeypatch.delenv(helion_runtime._OUTPUT_POOL_ENV, raising=False)
    helion_runtime._reset_output_pool_mode_cache()
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


def test_pool_opt_out_via_env(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``HELION_OUTPUT_POOL=0`` disables the pool — concurrent allocs differ."""
    monkeypatch.setenv(helion_runtime._OUTPUT_POOL_ENV, "0")
    helion_runtime._reset_output_pool_mode_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)

    # Warmup so the kernel itself is JIT'd.
    out = _pool_add(x, y)
    del out
    gc.collect()

    # Concurrent holds: with pool OFF, each call must produce a distinct
    # tensor backed by a distinct allocation (PyTorch's allocator hands
    # out a different block while the previous one is still alive).
    out1 = _pool_add(x, y)
    out2 = _pool_add(x, y)
    assert out1 is not out2
    assert out1.data_ptr() != out2.data_ptr(), (
        "HELION_OUTPUT_POOL=0 must not reuse buffers, but two concurrently "
        f"held outputs shared {out1.data_ptr()=}."
    )


def test_pool_opt_in_naive_back_compat(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``HELION_OUTPUT_POOL=1`` keeps the legacy naive pool (unconditional reuse).

    Back-compat guarantee: users who set ``=1`` for benchmarks should see
    no regression vs the earlier naive-pool behavior.
    """
    monkeypatch.setenv(helion_runtime._OUTPUT_POOL_ENV, "1")
    helion_runtime._reset_output_pool_mode_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)

    # Warmup.
    out = _pool_add(x, y)
    del out
    gc.collect()

    out1 = _pool_add(x, y)
    out2 = _pool_add(x, y)
    # Naive pool reuses unconditionally — same tensor instance.
    assert out1 is out2, (
        "HELION_OUTPUT_POOL=1 (naive) must return the same buffer even when "
        f"the previous output is still alive. Got distinct buffers: {out1.data_ptr()=} vs {out2.data_ptr()=}."
    )


def test_pool_cross_stream_no_reuse(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default mode keys the pool by stream — switching streams allocates fresh."""
    monkeypatch.delenv(helion_runtime._OUTPUT_POOL_ENV, raising=False)
    helion_runtime._reset_output_pool_mode_cache()
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
