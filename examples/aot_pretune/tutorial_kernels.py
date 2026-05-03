"""
Tutorial-style pretuned kernels using Helion AOT autotuning.

Mirrors the Triton tutorial set (matmul, softmax, layer-norm, fused-attention,
grouped-gemm, fp8-gemm) so users get a database of (kernel, shape) -> config
that they can ship and re-use via HELION_AOT_MODE=evaluate.

Usage:
    # Tune one kernel (used by pretune_runner.py per-GPU subprocess):
    HELION_AOT_MODE=collect HELION_AOT_DATA_DIR=.helion_aot/matmul \\
        python tutorial_kernels.py --kernel matmul

    # All kernels (single-GPU sequential):
    HELION_AOT_MODE=collect python tutorial_kernels.py
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Callable
from typing import TypeVar

import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
import helion.experimental
import helion.language as hl

_T = TypeVar("_T")


def _shape_set(kind: str) -> str:
    default = os.environ.get("HELION_PRETUNE_SHAPE_SET", "base")
    return os.environ.get(
        f"HELION_PRETUNE_{kind.upper()}_SHAPE_SET", default
    ).lower()


def _choose_shapes(base: list[_T], extra: list[_T], kind: str) -> list[_T]:
    shape_set = _shape_set(kind)
    if shape_set == "base":
        return list(base)
    if shape_set == "additional":
        return list(extra)
    if shape_set == "expanded":
        return [*base, *extra]
    raise ValueError(
        "HELION_PRETUNE_*_SHAPE_SET must be one of: base, additional, expanded"
    )


def _workflow_only() -> bool:
    return os.environ.get("HELION_PRETUNE_WORKFLOW_ONLY") == "1"


def _run_workflow_once(
    kernel: Callable[..., object],
    inputs: list[tuple[object, ...]],
) -> bool:
    if not _workflow_only():
        return False
    if inputs:
        kernel(*inputs[0])
    return True


# ---------------------------------------------------------------------------
# vector_add (Triton tutorial 01)  shapes: size = 2**i for i in range(12, 27)
# ---------------------------------------------------------------------------

# Triton tutorial 01: x_vals=[2**i for i in range(12, 28, 1)]  → 16 shapes
# (sizes from 4096 to 134_217_728 elements).
_VECTOR_ADD_BASE_SIZES = [2**i for i in range(12, 28)]  # 16 shapes
_VECTOR_ADD_EXTRA_SIZES = [2**i for i in range(28, 32)]  # 4 bigger shapes


def _vector_add_sizes(kind: str = "benchmark") -> list[int]:
    return _choose_shapes(_VECTOR_ADD_BASE_SIZES, _VECTOR_ADD_EXTRA_SIZES, kind)


def _vector_add_inputs(
    kind: str = "benchmark",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.randn([n], device=DEVICE, dtype=torch.float32),
            torch.randn([n], device=DEVICE, dtype=torch.float32),
        )
        for n in _vector_add_sizes(kind)
    ]


def _vector_add_collect_inputs() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _vector_add_inputs("collect")


def _vector_add_measure_inputs() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _vector_add_inputs("measure")


@helion.experimental.aot_kernel(
    collect_fn=_vector_add_collect_inputs,
    measure_fn=_vector_add_measure_inputs,
)
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


def benchmark_vector_add() -> None:
    print("=== vector_add ===")
    inputs = _vector_add_inputs()
    if _run_workflow_once(vector_add, inputs):
        return
    for x, y in inputs:
        vector_add(x, y)
        print(f"  N={x.numel()} done")


# ---------------------------------------------------------------------------
# matmul (Triton tutorial 03)  shapes: square M=N=K
# ---------------------------------------------------------------------------

# Triton tutorial 03: x_vals=[128 * i for i in range(2, 33)]  → 31 shapes
# (square M=N=K from 256 to 4096).
_MATMUL_BASE_SHAPES = [(128 * i, 128 * i, 128 * i) for i in range(2, 33)]
_MATMUL_EXTRA_SHAPES = [
    (4608, 4608, 4608),
    (5120, 5120, 5120),
    (6144, 6144, 6144),
    (7168, 7168, 7168),
    (8192, 8192, 8192),
    (4096, 4096, 12288),
    (4096, 12288, 4096),
    (8192, 4096, 4096),
]


def _matmul_shapes(kind: str = "benchmark") -> list[tuple[int, int, int]]:
    return _choose_shapes(_MATMUL_BASE_SHAPES, _MATMUL_EXTRA_SHAPES, kind)


def _matmul_inputs(kind: str = "benchmark") -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE),
        )
        for m, n, k in _matmul_shapes(kind)
    ]


def _matmul_collect_inputs() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _matmul_inputs("collect")


def _matmul_measure_inputs() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _matmul_inputs("measure")


@helion.experimental.aot_kernel(
    collect_fn=_matmul_collect_inputs,
    measure_fn=_matmul_measure_inputs,
    static_shapes=True,
)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def benchmark_matmul() -> None:
    print("=== matmul ===")
    inputs = _matmul_inputs()
    if _run_workflow_once(matmul, inputs):
        return
    for x, y in inputs:
        m, k = x.shape
        _, n = y.shape
        matmul(x, y)
        print(f"  {m}x{k} @ {k}x{n} done")


# ---------------------------------------------------------------------------
# softmax (Triton tutorial 02)  shapes: M=4096, N varies
# ---------------------------------------------------------------------------

# Triton tutorial 02: x_vals=[128 * i for i in range(2, 100)]  → 98 shapes
# (M=4096 fixed, N = 256, 384, ..., 12672).
_SOFTMAX_SHAPES = [(4096, 128 * i) for i in range(2, 100)]  # 98 shapes


def _softmax_inputs() -> list[tuple[torch.Tensor]]:
    return [
        (torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE),)
        for (m, n) in _SOFTMAX_SHAPES
    ]


@helion.experimental.aot_kernel(
    collect_fn=_softmax_inputs,
    measure_fn=_softmax_inputs,
)
def softmax(x: torch.Tensor) -> torch.Tensor:
    m, _n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        out[tile_m, :] = torch.nn.functional.softmax(x[tile_m, :], dim=1)
    return out


def benchmark_softmax() -> None:
    print("=== softmax ===")
    for (x,) in _softmax_inputs():
        m, n = x.shape
        softmax(x)
        print(f"  {m}x{n} done")


# ---------------------------------------------------------------------------
# layer_norm (Triton tutorial 05)  shapes: M=4096, N varies
# ---------------------------------------------------------------------------

# Triton tutorial 05: x_vals=[512 * i for i in range(2, 32)]  → 30 shapes
# (M=4096 fixed, N = 1024, 1536, ..., 15872, step 512).
_LAYER_NORM_SHAPES = [(4096, 512 * i) for i in range(2, 32)]  # 30 shapes


def _layer_norm_inputs() -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([n], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([n], device=DEVICE, dtype=HALF_DTYPE),
        )
        for (m, n) in _LAYER_NORM_SHAPES
    ]


@helion.experimental.aot_kernel(
    collect_fn=_layer_norm_inputs,
    measure_fn=_layer_norm_inputs,
)
def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        mean_val = torch.sum(acc, dim=-1) / n
        centered = acc - mean_val[:, None]
        var_val = torch.sum(centered * centered, dim=-1) / n
        rstd_val = torch.rsqrt(var_val + 1e-5)
        out[tile_m, :] = (
            centered * rstd_val[:, None] * weight[:].to(torch.float32)
            + bias[:].to(torch.float32)
        ).to(x.dtype)
    return out


def benchmark_layer_norm() -> None:
    print("=== layer_norm ===")
    for x, w, b in _layer_norm_inputs():
        m, n = x.shape
        layer_norm(x, w, b)
        print(f"  {m}x{n} done")


# ---------------------------------------------------------------------------
# attention (Triton tutorial 06)  shapes: B=4, H=32, N_CTX varies, D=64
# ---------------------------------------------------------------------------

# Triton tutorial 06 sweeps:  BATCH=4, H=32, HEAD_DIM ∈ {64, 128},
#   N_CTX = 2^i for i in range(10, 15)  →  10 shapes (HEAD_DIM × N_CTX, fwd non-causal)
_ATTN_BATCH = 4
_ATTN_HEAD = 32
_ATTN_HEAD_DIMS = [64, 128]
_ATTN_N_CTX = [2**i for i in range(10, 15)]  # 1024, 2048, 4096, 8192, 16384
_ATTN_BASE_SHAPES = [
    (_ATTN_BATCH, _ATTN_HEAD, n_ctx, head_dim)
    for head_dim in _ATTN_HEAD_DIMS
    for n_ctx in _ATTN_N_CTX
]
_ATTN_EXTRA_SHAPES = [
    (4, 32, 1536, 64),
    (4, 32, 3072, 64),
    (4, 32, 6144, 64),
    (4, 32, 12288, 64),
    (4, 32, 14336, 64),
    (4, 32, 1536, 128),
    (4, 32, 3072, 128),
    (4, 32, 6144, 128),
    (4, 32, 12288, 128),
    (4, 32, 14336, 128),
]


def _attention_shapes(kind: str = "benchmark") -> list[tuple[int, int, int, int]]:
    return _choose_shapes(_ATTN_BASE_SHAPES, _ATTN_EXTRA_SHAPES, kind)


def _attention_inputs(
    kind: str = "benchmark",
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    out = []
    for batch, heads, n_ctx, head_dim in _attention_shapes(kind):
        shape = [batch, heads, n_ctx, head_dim]
        q = torch.randn(shape, device=DEVICE, dtype=HALF_DTYPE)
        k = torch.randn(shape, device=DEVICE, dtype=HALF_DTYPE)
        v = torch.randn(shape, device=DEVICE, dtype=HALF_DTYPE)
        out.append((q, k, v))
    return out


def _attention_collect_inputs() -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return _attention_inputs("collect")


def _attention_measure_inputs() -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return _attention_inputs("measure")


@helion.experimental.aot_kernel(
    collect_fn=_attention_collect_inputs,
    measure_fn=_attention_measure_inputs,
    static_shapes=True,
)
def attention(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    """Scaled dot-product attention — same kernel as examples/attention.py."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


def benchmark_attention() -> None:
    print("=== attention ===")
    inputs = _attention_inputs()
    if _run_workflow_once(attention, inputs):
        return
    for q, k, v in inputs:
        attention(q, k, v)
        print(
            f"  B={q.shape[0]} H={q.shape[1]} N_CTX={q.shape[2]} "
            f"D={q.shape[3]} done"
        )


# ---------------------------------------------------------------------------
# grouped_gemm (Triton tutorial 08)  shapes: G groups of M=N=K matmul
# ---------------------------------------------------------------------------

# Triton tutorial 08 has two benchmarks (G=4 always):
#   benchmark_square_matrices: M=N=K = 2^i for i in range(7, 11)  → 4 shapes
#   benchmark_batches:         M = 2^i for i in range(7, 11), N=K=8192  → 4 shapes
# Total: 8 (M, N, K) triples.
_GG_SQUARE_SIZES = [2**i for i in range(7, 11)]  # 128, 256, 512, 1024
_GG_BATCHES_M = [2**i for i in range(7, 11)]  # 128, 256, 512, 1024
_GG_BATCHES_NK = 8192
_GG_BASE_SHAPES = [(4, n, n, n) for n in _GG_SQUARE_SIZES] + [
    (4, m, _GG_BATCHES_NK, _GG_BATCHES_NK) for m in _GG_BATCHES_M
]
_GG_EXTRA_SHAPES = [
    (4, 1536, 1536, 1536),
    (4, 2048, 2048, 2048),
    (4, 256, 4096, 4096),
    (4, 512, 4096, 4096),
    (4, 1024, 4096, 4096),
    (4, 2048, 4096, 4096),
    (4, 128, 16384, 8192),
    (4, 256, 16384, 8192),
    (8, 256, 4096, 4096),
    (8, 512, 4096, 4096),
    (8, 1024, 2048, 2048),
    (16, 256, 2048, 2048),
]
# kept for benchmark printer
_GG_SIZES = _GG_SQUARE_SIZES


def _grouped_gemm_shapes(kind: str = "benchmark") -> list[tuple[int, int, int, int]]:
    return _choose_shapes(_GG_BASE_SHAPES, _GG_EXTRA_SHAPES, kind)


def _grouped_gemm_inputs() -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    for groups, m, n, k in _grouped_gemm_shapes():
        a = torch.randn([groups, m, k], device=DEVICE, dtype=HALF_DTYPE)
        b = torch.randn([groups, k, n], device=DEVICE, dtype=HALF_DTYPE)
        out.append((a, b))
    return out


def _grouped_gemm_collect_inputs() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.randn([groups, m, k], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([groups, k, n], device=DEVICE, dtype=HALF_DTYPE),
        )
        for groups, m, n, k in _grouped_gemm_shapes("collect")
    ]


def _grouped_gemm_measure_inputs() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.randn([groups, m, k], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([groups, k, n], device=DEVICE, dtype=HALF_DTYPE),
        )
        for groups, m, n, k in _grouped_gemm_shapes("measure")
    ]


@helion.experimental.aot_kernel(
    collect_fn=_grouped_gemm_collect_inputs,
    measure_fn=_grouped_gemm_measure_inputs,
    static_shapes=True,
)
def grouped_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched/grouped matmul, matches examples/bmm.py."""
    g, m, k = a.size()
    g, k, n = b.size()
    out = torch.empty(
        [g, m, n], device=a.device, dtype=torch.promote_types(a.dtype, b.dtype)
    )
    for tile_g, tile_m, tile_n in hl.tile([g, m, n]):
        acc = hl.zeros([tile_g, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc, a[tile_g, tile_m, tile_k], b[tile_g, tile_k, tile_n]
            )
        out[tile_g, tile_m, tile_n] = acc
    return out


def benchmark_grouped_gemm() -> None:
    print("=== grouped_gemm ===")
    inputs = _grouped_gemm_inputs()
    if _run_workflow_once(grouped_gemm, inputs):
        return
    for a, b in inputs:
        g, m, k = a.shape
        _, _, n = b.shape
        grouped_gemm(a, b)
        print(f"  G={g} {m}x{k} @ {k}x{n} done")


# ---------------------------------------------------------------------------
# fp8_gemm (Triton tutorial 10)  shapes: square M=N=K in fp8
# ---------------------------------------------------------------------------

# fp8_gemm: helion's fp8_gemm is plain fp8 matmul (no block scaling), so it
# uses the same shape sweep as matmul (30 shapes).  Triton tutorial 10 is
# block-scaled (mxfp4/nvfp4/mxfp8) which is structurally different.
_FP8_BASE_SHAPES = list(_MATMUL_BASE_SHAPES)
_FP8_EXTRA_SHAPES = list(_MATMUL_EXTRA_SHAPES)


def _fp8_gemm_shapes(kind: str = "benchmark") -> list[tuple[int, int, int]]:
    return _choose_shapes(_FP8_BASE_SHAPES, _FP8_EXTRA_SHAPES, kind)


def _fp8_gemm_inputs(
    kind: str = "benchmark",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.randn([m, k], device=DEVICE).to(torch.float8_e4m3fn),
            torch.randn([k, n], device=DEVICE).to(torch.float8_e4m3fn),
        )
        for m, n, k in _fp8_gemm_shapes(kind)
    ]


def _fp8_gemm_collect_inputs() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _fp8_gemm_inputs("collect")


def _fp8_gemm_measure_inputs() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _fp8_gemm_inputs("measure")


@helion.experimental.aot_kernel(
    collect_fn=_fp8_gemm_collect_inputs,
    measure_fn=_fp8_gemm_measure_inputs,
    static_shapes=True,
)
def fp8_gemm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out = torch.empty([m, n], dtype=HALF_DTYPE, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(HALF_DTYPE)
    return out


def benchmark_fp8_gemm() -> None:
    print("=== fp8_gemm ===")
    inputs = _fp8_gemm_inputs()
    if _run_workflow_once(fp8_gemm, inputs):
        return
    for x, y in inputs:
        m, k = x.shape
        _, n = y.shape
        fp8_gemm(x, y)
        print(f"  {m}x{k} @ {k}x{n} done")


KERNEL_BENCHMARKS: dict[str, Callable[[], None]] = {
    "vector_add": benchmark_vector_add,
    "matmul": benchmark_matmul,
    "softmax": benchmark_softmax,
    "layer_norm": benchmark_layer_norm,
    "attention": benchmark_attention,
    "grouped_gemm": benchmark_grouped_gemm,
    "fp8_gemm": benchmark_fp8_gemm,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tutorial pretune kernels")
    parser.add_argument(
        "--kernel",
        "-k",
        type=str,
        action="append",
        dest="kernels",
        help="Kernel(s) to run (default: all)",
    )
    args = parser.parse_args()

    kernels = args.kernels
    if kernels is None:
        env_kernels = os.environ.get("HELION_AOT_KERNELS", "")
        if env_kernels:
            kernels = env_kernels.split(",")

    aot_mode = os.environ.get("HELION_AOT_MODE", "disabled")
    print(f"AOT mode: {aot_mode}")

    targets = kernels or list(KERNEL_BENCHMARKS.keys())
    for name in targets:
        if name not in KERNEL_BENCHMARKS:
            print(f"Unknown kernel: {name}")
            continue
        KERNEL_BENCHMARKS[name]()


if __name__ == "__main__":
    main()
