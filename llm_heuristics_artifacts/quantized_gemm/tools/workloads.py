"""Materialize kernel args for the quantized-GEMM shape grid.

Three kernels share a matmul-shaped tuning surface but differ in how the
weight operand is stored:

- ``matmul_bf16_int4`` (examples.int4_gemm): A=[M,K] bf16, B=[K//2, N] int8
  (two int4 nibbles per byte).
- ``_bf16xint16_gemm`` (examples.bf16xint16_gemm): x=[M,K] bf16, w=[K, N] int16.
- ``nvfp4_matmul`` (examples.nvfp4_gemm): A=[M,K] bf16, B_packed=[K//2, N] int8
  (two fp4_e2m1 nibbles per byte).

Shape grid entries use **logical** M, K, N — the builder handles packing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _rand_bf16(rows: int, cols: int, device: str) -> torch.Tensor:
    return torch.randn(rows, cols, device=device, dtype=torch.bfloat16)


def _rand_int4_packed(k: int, n: int, device: str) -> torch.Tensor:
    """Random int4 values in [-8, 7], packed two-per-byte along dim 0."""
    from examples.int4_gemm import _pack_int4_matrix

    assert k % 2 == 0, f"int4 requires K even, got K={k}"
    unpacked = torch.randint(-8, 8, (k, n), device=device, dtype=torch.int8)
    return _pack_int4_matrix(unpacked)


def _rand_int16(k: int, n: int, device: str) -> torch.Tensor:
    # int16 values; kernel casts to bf16 before dot, so any reasonable range is fine.
    return torch.randint(-32768, 32767, (k, n), device=device, dtype=torch.int16)


def _rand_fp4_packed(k: int, n: int, device: str) -> torch.Tensor:
    """Random fp4_e2m1 nibble indices, packed two-per-byte along dim 0."""
    from examples.nvfp4_gemm import pack_fp4

    assert k % 2 == 0, f"fp4 requires K even, got K={k}"
    # Valid nibble indices are 0..15 (sign bit + 3-bit magnitude).
    indices = torch.randint(0, 16, (k, n), device=device, dtype=torch.uint8)
    return pack_fp4(indices)


def _args_matmul_bf16_int4(shape: dict[str, Any], device: str):
    from examples.int4_gemm import matmul_bf16_int4

    M, K, N = shape["M"], shape["K"], shape["N"]
    A = _rand_bf16(M, K, device)
    B_packed = _rand_int4_packed(K, N, device)
    return matmul_bf16_int4, (A, B_packed)


def _args_bf16xint16_gemm(shape: dict[str, Any], device: str):
    from examples.bf16xint16_gemm import _bf16xint16_gemm

    M, K, N = shape["M"], shape["K"], shape["N"]
    x = _rand_bf16(M, K, device)
    w = _rand_int16(K, N, device)
    return _bf16xint16_gemm, (x, w)


def _args_nvfp4_matmul(shape: dict[str, Any], device: str):
    from examples.nvfp4_gemm import nvfp4_matmul

    M, K, N = shape["M"], shape["K"], shape["N"]
    A = _rand_bf16(M, K, device)
    B_packed = _rand_fp4_packed(K, N, device)
    return nvfp4_matmul, (A, B_packed)


_BUILDERS: dict[str, Callable[..., Any]] = {
    "matmul_bf16_int4": _args_matmul_bf16_int4,
    "_bf16xint16_gemm": _args_bf16xint16_gemm,
    "nvfp4_matmul": _args_nvfp4_matmul,
}


def build_kernel_and_args(
    kernel: str, shape_entry: dict[str, Any], dtype_name: str | None = None,
    device: str = "cuda",
):
    """Return (kernel_callable, args_tuple) for one shape entry.

    ``dtype_name`` is ignored (each kernel has a fixed dtype signature).
    Shape entry must contain logical ``M``, ``K``, ``N``.
    """
    if kernel not in _BUILDERS:
        raise ValueError(f"Unknown kernel {kernel!r}")
    return _BUILDERS[kernel](shape_entry["args"], device)
