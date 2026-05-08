"""Materialize kernel args for the norm shape grid."""

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
    }[name]


def _args_layer_norm(shape_args: dict[str, Any], dtype: torch.dtype, device: str):
    from examples.layer_norm import layer_norm_fwd

    rows, cols = shape_args["rows"], shape_args["cols"]
    x = torch.randn(rows, cols, device=device, dtype=dtype)
    weight = torch.randn(cols, device=device, dtype=dtype)
    bias = torch.randn(cols, device=device, dtype=dtype)
    return layer_norm_fwd, (x, [cols], weight, bias)


def _args_rms_norm(shape_args: dict[str, Any], dtype: torch.dtype, device: str):
    from examples.rms_norm import rms_norm_fwd

    rows, cols = shape_args["rows"], shape_args["cols"]
    x = torch.randn(rows, cols, device=device, dtype=dtype)
    weight = torch.randn(cols, device=device, dtype=dtype)
    return rms_norm_fwd, (x, weight)


def _args_softmax(shape_args: dict[str, Any], dtype: torch.dtype, device: str):
    from examples.softmax import softmax_two_pass

    rows, cols = shape_args["rows"], shape_args["cols"]
    x = torch.randn(rows, cols, device=device, dtype=dtype)
    return softmax_two_pass, (x,)


_BUILDERS: dict[str, Callable[..., Any]] = {
    "layer_norm": _args_layer_norm,
    "rms_norm": _args_rms_norm,
    "softmax": _args_softmax,
}


def build_kernel_and_args(
    kernel: str, shape_entry: dict[str, Any], dtype_name: str, device: str = "cuda"
):
    """Return (kernel_callable, args_tuple) for one shape entry."""
    if kernel not in _BUILDERS:
        raise ValueError(f"Unknown kernel {kernel!r}")
    dtype = _dtype_from_str(dtype_name)
    return _BUILDERS[kernel](shape_entry["args"], dtype, device)
