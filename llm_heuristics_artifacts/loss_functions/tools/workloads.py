"""Materialize kernel args for the loss function shape grid."""

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


def _args_cross_entropy(shape_args: dict[str, Any], dtype: torch.dtype, device: str):
    from examples.cross_entropy import cross_entropy
    from helion._testing import LONG_INT_TYPE

    batch_size, vocab_size = shape_args["batch_size"], shape_args["vocab_size"]
    logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
    labels = torch.randint(0, vocab_size, (batch_size,), device=device, dtype=LONG_INT_TYPE)
    return cross_entropy, (logits, labels)


def _args_jsd(shape_args: dict[str, Any], dtype: torch.dtype, device: str):
    from examples.jsd import jsd_forward

    batch_size, vocab_size = shape_args["batch_size"], shape_args["vocab_size"]
    # JSD expects log-space inputs
    student = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
    teacher = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
    beta = shape_args.get("beta", 0.5)

    # shift_labels is optional; use None for now
    return jsd_forward, (student, teacher, None, beta)


def _args_kl_div(shape_args: dict[str, Any], dtype: torch.dtype, device: str):
    from examples.kl_div import kl_div_forward

    batch_size, vocab_size = shape_args["batch_size"], shape_args["vocab_size"]
    # KL divergence: KL(P || Q)
    p_logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
    q_logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
    return kl_div_forward, (p_logits, q_logits)


def _args_grpo_loss(shape_args: dict[str, Any], dtype: torch.dtype, device: str):
    from examples.grpo_loss import grpo_loss
    from helion._testing import LONG_INT_TYPE

    batch_size, vocab_size = shape_args["batch_size"], shape_args["vocab_size"]
    seq_len = shape_args.get("seq_len", 128)  # Default sequence length

    # GRPO loss typically operates on sequences
    policy_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=dtype)
    ref_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=dtype)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=LONG_INT_TYPE)

    return grpo_loss, (policy_logits, ref_logits, labels)


def _args_fused_linear_jsd(shape_args: dict[str, Any], dtype: torch.dtype, device: str):
    from examples.fused_linear_jsd import fused_linear_jsd

    batch_size = shape_args["batch_size"]
    hidden_size = shape_args.get("hidden_size", 4096)
    vocab_size = shape_args["vocab_size"]

    # Input activations
    x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    # Linear layer weights
    weight = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype)
    # Teacher logits
    teacher_logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)

    return fused_linear_jsd, (x, weight, teacher_logits)


def _args_softmax(shape_args: dict[str, Any], dtype: torch.dtype, device: str):
    from examples.softmax import softmax

    batch_size = shape_args["batch_size"]
    dim = shape_args["dim"]

    # Softmax operates on 2D tensor [batch, dim]
    x = torch.randn(batch_size, dim, device=device, dtype=dtype)
    return softmax, (x,)


_BUILDERS: dict[str, Callable[..., Any]] = {
    "cross_entropy": _args_cross_entropy,
    "jsd": _args_jsd,
    "kl_div": _args_kl_div,
    "grpo_loss": _args_grpo_loss,
    "fused_linear_jsd": _args_fused_linear_jsd,
    "softmax": _args_softmax,
}


def build_kernel_and_args(
    kernel: str, shape_entry: dict[str, Any], dtype_name: str, device: str = "cuda"
):
    """Return (kernel_callable, args_tuple) for one shape entry.

    Args:
        kernel: Kernel name (e.g., 'cross_entropy', 'jsd', 'kl_div')
        shape_entry: Dict with 'args' key containing shape parameters
        dtype_name: Data type name ('bfloat16', 'float16', 'float32')
        device: Device string (default 'cuda')

    Returns:
        (kernel_callable, args_tuple): Kernel function and its arguments
    """
    if kernel not in _BUILDERS:
        raise ValueError(f"Unknown kernel {kernel!r}. Available: {list(_BUILDERS.keys())}")
    dtype = _dtype_from_str(dtype_name)
    return _BUILDERS[kernel](shape_entry["args"], dtype, device)
