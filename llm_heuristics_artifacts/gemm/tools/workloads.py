"""Materialize kernel args for the GEMM shape grid.

GEMM shapes are specified as (M, K, N) triples:
  x: [M, K], y: [K, N], out: [M, N].

Supported kernels: matmul, fp8_gemm.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _dtype_from_str(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }[name]


def _rand_fp_like(rows: int, cols: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    if dtype == torch.float8_e4m3fn:
        x = torch.randn(rows, cols, device=device, dtype=torch.float32) * 0.1
        return x.clamp(-448.0, 448.0).to(dtype)
    return torch.randn(rows, cols, device=device, dtype=dtype)


def _args_matmul(shape_args: dict[str, Any], dtype: torch.dtype, device: str):
    from examples.matmul import matmul

    M, K, N = shape_args["M"], shape_args["K"], shape_args["N"]
    x = _rand_fp_like(M, K, dtype, device)
    y = _rand_fp_like(K, N, dtype, device)
    return matmul, (x, y)


def _args_fp8_gemm(shape_args: dict[str, Any], dtype: torch.dtype, device: str):
    from examples.fp8_gemm import fp8_gemm

    M, K, N = shape_args["M"], shape_args["K"], shape_args["N"]
    fp8 = torch.float8_e4m3fn
    x = _rand_fp_like(M, K, fp8, device)
    y = _rand_fp_like(K, N, fp8, device)
    return fp8_gemm, (x, y)


_BUILDERS: dict[str, Callable[..., Any]] = {
    "matmul": _args_matmul,
    "fp8_gemm": _args_fp8_gemm,
}


# Per-kernel default dtype when the shape entry does not override it.
_DEFAULT_DTYPE: dict[str, str] = {
    "matmul": "float16",
    "fp8_gemm": "float8_e4m3fn",
}


def build_kernel_and_args(
    kernel: str, shape_entry: dict[str, Any], dtype_name: str | None, device: str = "cuda"
):
    """Return (kernel_callable, args_tuple) for one shape entry.

    ``dtype_name`` is the grid-level default. A shape entry may override
    via ``args["dtype"]``; for fp8_gemm the dtype is always forced to
    float8_e4m3fn regardless.
    """
    if kernel not in _BUILDERS:
        raise ValueError(f"Unknown kernel {kernel!r}")
    args = shape_entry["args"]
    resolved_dtype_name = args.get("dtype") or dtype_name or _DEFAULT_DTYPE[kernel]
    if kernel == "fp8_gemm":
        resolved_dtype_name = "float8_e4m3fn"
    dtype = _dtype_from_str(resolved_dtype_name)
    return _BUILDERS[kernel](args, dtype, device)
