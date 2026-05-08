# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Compare LLM-guided autotuning with and without observed heuristics.

The script launches one fresh Python subprocess per kernel/shape/arm so Python
and Triton compile caches do not leak across arms.  Each worker runs full
``LLMGuidedSearch``, verifies the winning config, and writes a JSON result.
"""

from __future__ import annotations

import argparse
import collections
import csv
from dataclasses import dataclass
from html import escape
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import TYPE_CHECKING
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("/tmp/helion_llm_heuristics_experiment")
DEFAULT_MODEL = "gpt-5-2"
DEFAULT_PROVIDER = ""
DEFAULT_AUTOTUNER = "LLMGuidedSearch"
AUTOTUNERS = ("LLMGuidedSearch", "LLMSeededLFBOTreeSearch")
LLM_ROUND0_RECORD_PATH_ENV = "HELION_LLM_ROUND0_RECORD_PATH"
LLM_ROUND0_REPLAY_PATH_ENV = "HELION_LLM_ROUND0_REPLAY_PATH"
LLM_ROUND0_MODE_ENV = "HELION_LLM_ROUND0_MODE"
LLM_ROUND0_PAIRED_BASELINE_PATH_ENV = "HELION_LLM_ROUND0_PAIRED_BASELINE_PATH"
LLM_ROUND0_PAIRED_ACTIVE_MATCH_ENV = (
    "HELION_LLM_ROUND0_PAIRED_ACTIVE_HEURISTIC_MATCH"
)
LLM_ROUND0_PAIRED_ACTION_ENV = "HELION_LLM_ROUND0_PAIRED_ACTION"
ARM_SETTINGS = {
    # observed, prompt guidance, seed configs, max templates, disabled classes, prompt mode
    "baseline": (False, False, False, None, "", "template"),
    "prompt": (True, True, False, None, "", "template"),
    "range_prompt": (True, True, False, None, "", "range"),
    "seeds": (True, False, True, None, "", "template"),
    "heuristics": (True, True, True, None, "", "template"),
    "heuristics_top1": (True, True, True, 1, "", "template"),
    "heuristics_no_matmul": (True, True, True, None, "matmul", "template"),
    "heuristics_top1_no_matmul": (True, True, True, 1, "matmul", "template"),
    "heuristics_product": (
        True,
        True,
        True,
        None,
        "matmul,matmul_fp8,batched_matmul,split_k_matmul,grouped_matmul",
        "template",
    ),
    "heuristics_product_top1": (
        True,
        True,
        True,
        1,
        "matmul,matmul_fp8,batched_matmul,split_k_matmul,grouped_matmul",
        "template",
    ),
}
ARM_COLORS = {
    "baseline": "#7f1d1d",
    "prompt": "#047857",
    "range_prompt": "#2563eb",
    "seeds": "#9333ea",
    "heuristics": "#1d4ed8",
    "heuristics_top1": "#0f766e",
    "heuristics_no_matmul": "#c2410c",
    "heuristics_top1_no_matmul": "#4338ca",
    "heuristics_product": "#be123c",
    "heuristics_product_top1": "#0891b2",
}


@dataclass(frozen=True)
class Workload:
    name: str
    module: str
    kernel_attr: str
    shape_name: str
    shape: tuple[int, ...]


WORKLOADS: dict[str, Workload] = {
    "add_1m": Workload("add_1m", "examples.add", "add", "N=1048576", (1048576,)),
    "add_16m": Workload("add_16m", "examples.add", "add", "N=16777216", (16777216,)),
    "add_64m": Workload("add_64m", "examples.add", "add", "N=67108864", (67108864,)),
    "softmax_4k": Workload(
        "softmax_4k", "examples.softmax", "softmax", "M=4096,N=4096", (4096, 4096)
    ),
    "softmax_1k_1k": Workload(
        "softmax_1k_1k",
        "examples.softmax",
        "softmax",
        "M=1024,N=1024",
        (1024, 1024),
    ),
    "softmax_1k_2k": Workload(
        "softmax_1k_2k",
        "examples.softmax",
        "softmax",
        "M=1024,N=2048",
        (1024, 2048),
    ),
    "softmax_2k_1k": Workload(
        "softmax_2k_1k",
        "examples.softmax",
        "softmax",
        "M=2048,N=1024",
        (2048, 1024),
    ),
    "softmax_2k_2k": Workload(
        "softmax_2k_2k",
        "examples.softmax",
        "softmax",
        "M=2048,N=2048",
        (2048, 2048),
    ),
    "softmax_1k_4k": Workload(
        "softmax_1k_4k",
        "examples.softmax",
        "softmax",
        "M=1024,N=4096",
        (1024, 4096),
    ),
    "softmax_2k_4k": Workload(
        "softmax_2k_4k",
        "examples.softmax",
        "softmax",
        "M=2048,N=4096",
        (2048, 4096),
    ),
    "softmax_4k_1k": Workload(
        "softmax_4k_1k",
        "examples.softmax",
        "softmax",
        "M=4096,N=1024",
        (4096, 1024),
    ),
    "softmax_4k_2k": Workload(
        "softmax_4k_2k",
        "examples.softmax",
        "softmax",
        "M=4096,N=2048",
        (4096, 2048),
    ),
    "softmax_2k_8k": Workload(
        "softmax_2k_8k",
        "examples.softmax",
        "softmax",
        "M=2048,N=8192",
        (2048, 8192),
    ),
    "softmax_8k_8k": Workload(
        "softmax_8k_8k",
        "examples.softmax",
        "softmax",
        "M=8192,N=8192",
        (8192, 8192),
    ),
    "softmax_1k_16k": Workload(
        "softmax_1k_16k",
        "examples.softmax",
        "softmax",
        "M=1024,N=16384",
        (1024, 16384),
    ),
    "layer_norm_4k": Workload(
        "layer_norm_4k",
        "examples.layer_norm",
        "layer_norm_fwd",
        "M=4096,N=4096",
        (4096, 4096),
    ),
    "layer_norm_4k_1k": Workload(
        "layer_norm_4k_1k",
        "examples.layer_norm",
        "layer_norm_fwd",
        "M=4096,N=1024",
        (4096, 1024),
    ),
    "layer_norm_4k_2k": Workload(
        "layer_norm_4k_2k",
        "examples.layer_norm",
        "layer_norm_fwd",
        "M=4096,N=2048",
        (4096, 2048),
    ),
    "layer_norm_2k_8k": Workload(
        "layer_norm_2k_8k",
        "examples.layer_norm",
        "layer_norm_fwd",
        "M=2048,N=8192",
        (2048, 8192),
    ),
    "layer_norm_8k_8k": Workload(
        "layer_norm_8k_8k",
        "examples.layer_norm",
        "layer_norm_fwd",
        "M=8192,N=8192",
        (8192, 8192),
    ),
    "layer_norm_1k_16k": Workload(
        "layer_norm_1k_16k",
        "examples.layer_norm",
        "layer_norm_fwd",
        "M=1024,N=16384",
        (1024, 16384),
    ),
    "rms_norm_4k": Workload(
        "rms_norm_4k",
        "examples.rms_norm",
        "rms_norm_fwd",
        "M=4096,N=4096",
        (4096, 4096),
    ),
    "rms_norm_4k_1k": Workload(
        "rms_norm_4k_1k",
        "examples.rms_norm",
        "rms_norm_fwd",
        "M=4096,N=1024",
        (4096, 1024),
    ),
    "rms_norm_4k_2k": Workload(
        "rms_norm_4k_2k",
        "examples.rms_norm",
        "rms_norm_fwd",
        "M=4096,N=2048",
        (4096, 2048),
    ),
    "rms_norm_2048x4096": Workload(
        "rms_norm_2048x4096",
        "examples.rms_norm",
        "rms_norm_fwd",
        "M=2048,N=4096",
        (2048, 4096),
    ),
    "rms_norm_8192x2048": Workload(
        "rms_norm_8192x2048",
        "examples.rms_norm",
        "rms_norm_fwd",
        "M=8192,N=2048",
        (8192, 2048),
    ),
    "rms_norm_1024x8192": Workload(
        "rms_norm_1024x8192",
        "examples.rms_norm",
        "rms_norm_fwd",
        "M=1024,N=8192",
        (1024, 8192),
    ),
    "rms_norm_1024x16384": Workload(
        "rms_norm_1024x16384",
        "examples.rms_norm",
        "rms_norm_fwd",
        "M=1024,N=16384",
        (1024, 16384),
    ),
    "cross_entropy_32k": Workload(
        "cross_entropy_32k",
        "examples.cross_entropy",
        "cross_entropy",
        "N=2048,V=32768",
        (2048, 32768),
    ),
    "cross_entropy_4k_16k": Workload(
        "cross_entropy_4k_16k",
        "examples.cross_entropy",
        "cross_entropy",
        "N=4096,V=16384",
        (4096, 16384),
    ),
    "cross_entropy_1k_64k": Workload(
        "cross_entropy_1k_64k",
        "examples.cross_entropy",
        "cross_entropy",
        "N=1024,V=65536",
        (1024, 65536),
    ),
    "matmul_1k": Workload(
        "matmul_1k",
        "examples.matmul",
        "matmul",
        "M=1024,K=1024,N=1024",
        (1024, 1024, 1024),
    ),
    "matmul_256": Workload(
        "matmul_256",
        "examples.matmul",
        "matmul",
        "M=256,K=256,N=256",
        (256, 256, 256),
    ),
    "matmul_512": Workload(
        "matmul_512",
        "examples.matmul",
        "matmul",
        "M=512,K=512,N=512",
        (512, 512, 512),
    ),
    "matmul_2k": Workload(
        "matmul_2k",
        "examples.matmul",
        "matmul",
        "M=2048,K=2048,N=2048",
        (2048, 2048, 2048),
    ),
    "matmul_skinny_m": Workload(
        "matmul_skinny_m",
        "examples.matmul",
        "matmul",
        "M=128,K=4096,N=4096",
        (128, 4096, 4096),
    ),
    "matmul_skinny_n": Workload(
        "matmul_skinny_n",
        "examples.matmul",
        "matmul",
        "M=4096,K=4096,N=128",
        (4096, 4096, 128),
    ),
    "matmul_k_heavy": Workload(
        "matmul_k_heavy",
        "examples.matmul",
        "matmul",
        "M=256,K=16384,N=256",
        (256, 16384, 256),
    ),
    "attention_1k_d64": Workload(
        "attention_1k_d64",
        "examples.attention",
        "attention",
        "B=2,H=32,S=1024,D=64",
        (2, 32, 1024, 64),
    ),
    "attention_512_d64": Workload(
        "attention_512_d64",
        "examples.attention",
        "attention",
        "B=2,H=32,S=512,D=64",
        (2, 32, 512, 64),
    ),
    "attention_2k_d64": Workload(
        "attention_2k_d64",
        "examples.attention",
        "attention",
        "B=2,H=16,S=2048,D=64",
        (2, 16, 2048, 64),
    ),
    "attention_2k_d128": Workload(
        "attention_2k_d128",
        "examples.attention",
        "attention",
        "B=2,H=16,S=2048,D=128",
        (2, 16, 2048, 128),
    ),
    "attention_4k_d64": Workload(
        "attention_4k_d64",
        "examples.attention",
        "attention",
        "B=1,H=16,S=4096,D=64",
        (1, 16, 4096, 64),
    ),
    "attention_4k_d128": Workload(
        "attention_4k_d128",
        "examples.attention",
        "attention",
        "B=1,H=8,S=4096,D=128",
        (1, 8, 4096, 128),
    ),
    "batch_softmax_16x512x1024": Workload(
        "batch_softmax_16x512x1024",
        "examples.batch_softmax",
        "batch_softmax",
        "B=16,M=512,N=1024",
        (16, 512, 1024),
    ),
    "batch_softmax_8x1024x2048": Workload(
        "batch_softmax_8x1024x2048",
        "examples.batch_softmax",
        "batch_softmax",
        "B=8,M=1024,N=2048",
        (8, 1024, 2048),
    ),
    "bmm_8x256x384x512": Workload(
        "bmm_8x256x384x512",
        "examples.bmm",
        "bmm",
        "B=8,M=256,K=384,N=512",
        (8, 256, 384, 512),
    ),
    "bmm_16x128x512x256": Workload(
        "bmm_16x128x512x256",
        "examples.bmm",
        "bmm",
        "B=16,M=128,K=512,N=256",
        (16, 128, 512, 256),
    ),
    "concat2d_1500x400_600": Workload(
        "concat2d_1500x400_600",
        "examples.concatenate",
        "concat2d_dim1",
        "M=1500,N1=400,N2=600",
        (1500, 400, 600),
    ),
    "embedding_8192x64": Workload(
        "embedding_8192x64",
        "examples.embedding",
        "embedding",
        "tokens=8192,V=16384,D=64",
        (8192, 16384, 64),
    ),
    "exp_1m": Workload("exp_1m", "examples.exp", "exp_fwd", "N=1048576", (1048576,)),
    "exp_16m": Workload(
        "exp_16m", "examples.exp", "exp_fwd", "N=16777216", (16777216,)
    ),
    "geglu_4096x4096": Workload(
        "geglu_4096x4096",
        "examples.geglu",
        "geglu",
        "M=4096,N=4096",
        (4096, 4096),
    ),
    "geglu_2048x8192": Workload(
        "geglu_2048x8192",
        "examples.geglu",
        "geglu",
        "M=2048,N=8192",
        (2048, 8192),
    ),
    "matmul_split_k_64x32768x64": Workload(
        "matmul_split_k_64x32768x64",
        "examples.matmul_split_k",
        "matmul_split_k",
        "M=64,K=32768,N=64",
        (64, 32768, 64),
    ),
    "sum_5120x2560": Workload(
        "sum_5120x2560",
        "examples.sum",
        "sum_kernel",
        "M=5120,N=2560",
        (5120, 2560),
    ),
    "sum_4096x1024": Workload(
        "sum_4096x1024",
        "examples.sum",
        "sum_kernel",
        "M=4096,N=1024",
        (4096, 1024),
    ),
    "sum_2048x8192": Workload(
        "sum_2048x8192",
        "examples.sum",
        "sum_kernel",
        "M=2048,N=8192",
        (2048, 8192),
    ),
    "swiglu_4096x4096": Workload(
        "swiglu_4096x4096",
        "examples.swiglu",
        "swiglu_fwd",
        "M=4096,N=4096",
        (4096, 4096),
    ),
    "swiglu_2048x8192": Workload(
        "swiglu_2048x8192",
        "examples.swiglu",
        "swiglu_fwd",
        "M=2048,N=8192",
        (2048, 8192),
    ),
    "welford_4096x1024": Workload(
        "welford_4096x1024",
        "examples.welford",
        "welford",
        "M=4096,N=1024",
        (4096, 1024),
    ),
}


def _json_default(value: object) -> object:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    return str(value)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _value_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    filtered = [value for value in values if math.isfinite(value)]
    if not filtered:
        return {
            "count": 0,
            "total": 0.0,
            "mean": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": len(filtered),
        "total": sum(filtered),
        "mean": statistics.mean(filtered),
        "median": statistics.median(filtered),
        "p75": _percentile(filtered, 0.75),
        "p90": _percentile(filtered, 0.90),
        "max": max(filtered),
    }


def _prepend_path_env(env: dict[str, str], name: str, path: Path) -> None:
    value = str(path)
    existing = env.get(name)
    if existing:
        parts = existing.split(os.pathsep)
        if value in parts:
            return
        env[name] = os.pathsep.join([value, existing])
    else:
        env[name] = value


def _ensure_repo_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _make_args(workload: Workload) -> tuple[object, ...]:
    import torch

    from helion._testing import DEVICE
    from helion._testing import HALF_DTYPE
    from helion._testing import LONG_INT_TYPE

    if workload.module == "examples.add" and workload.kernel_attr == "add":
        (n,) = workload.shape
        return (
            torch.randn([n], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([n], device=DEVICE, dtype=HALF_DTYPE),
        )
    if workload.module == "examples.softmax" and workload.kernel_attr == "softmax":
        m, n = workload.shape
        return (torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE),)
    if (
        workload.module == "examples.layer_norm"
        and workload.kernel_attr == "layer_norm_fwd"
    ):
        m, n = workload.shape
        return (
            torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE),
            [n],
            torch.randn([n], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([n], device=DEVICE, dtype=HALF_DTYPE),
        )
    if (
        workload.module == "examples.rms_norm"
        and workload.kernel_attr == "rms_norm_fwd"
    ):
        m, n = workload.shape
        return (
            torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([n], device=DEVICE, dtype=HALF_DTYPE),
        )
    if (
        workload.module == "examples.cross_entropy"
        and workload.kernel_attr == "cross_entropy"
    ):
        n, vocab = workload.shape
        return (
            torch.randn([n, vocab], device=DEVICE, dtype=torch.float32),
            torch.randint(0, vocab, [n], device=DEVICE, dtype=LONG_INT_TYPE),
        )
    if workload.module == "examples.matmul" and workload.kernel_attr == "matmul":
        m, k, n = workload.shape
        return (
            torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE),
        )
    if workload.module == "examples.attention" and workload.kernel_attr == "attention":
        b, h, s, d = workload.shape
        return tuple(
            torch.randn([b, h, s, d], device=DEVICE, dtype=HALF_DTYPE) for _ in range(3)
        )
    if (
        workload.module == "examples.batch_softmax"
        and workload.kernel_attr == "batch_softmax"
    ):
        b, m, n = workload.shape
        return (torch.randn([b, m, n], device=DEVICE, dtype=HALF_DTYPE),)
    if workload.module == "examples.bmm" and workload.kernel_attr == "bmm":
        b, m, k, n = workload.shape
        return (
            torch.randn([b, m, k], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([b, k, n], device=DEVICE, dtype=HALF_DTYPE),
        )
    if workload.name == "concat2d_1500x400_600":
        m, n1, n2 = workload.shape
        return (
            torch.randn([m, n1], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([m, n2], device=DEVICE, dtype=HALF_DTYPE),
        )
    if workload.name == "embedding_8192x64":
        tokens, vocab, dim = workload.shape
        return (
            torch.randint(0, vocab, [tokens], device=DEVICE, dtype=torch.int32),
            torch.randn([vocab, dim], device=DEVICE, dtype=HALF_DTYPE),
        )
    if workload.module == "examples.exp" and workload.kernel_attr == "exp_fwd":
        (n,) = workload.shape
        return (torch.randn([n], device=DEVICE, dtype=torch.float32),)
    if workload.module in {"examples.geglu", "examples.swiglu"}:
        m, n = workload.shape
        return (
            torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE),
        )
    if workload.name == "matmul_split_k_64x32768x64":
        m, k, n = workload.shape
        return (
            torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE),
        )
    if workload.module == "examples.sum" and workload.kernel_attr == "sum_kernel":
        m, n = workload.shape
        return (torch.randn([m, n], device=DEVICE, dtype=torch.float32),)
    if workload.name == "welford_4096x1024":
        m, n = workload.shape
        return (
            torch.rand([n], device=DEVICE, dtype=torch.float32),
            torch.rand([n], device=DEVICE, dtype=torch.float32),
            torch.rand([m, n], device=DEVICE, dtype=torch.float32),
        )
    raise KeyError(workload.name)


def _load_kernel(workload: Workload) -> object:
    module = importlib.import_module(workload.module)
    return getattr(module, workload.kernel_attr)


def _build_search(
    kernel: object,
    args: tuple[object, ...],
    *,
    autotuner: str,
    effort: str,
    llm_max_rounds: int | None,
    llm_configs_per_round: int | None,
    llm_initial_random_configs: int | None,
) -> object:
    from helion.autotuner import search_algorithms
    from helion.autotuner.effort_profile import AutotuneEffort
    from helion.autotuner.effort_profile import get_effort_profile

    strategy_cls = search_algorithms[autotuner]
    bound = kernel.bind(args)
    profile = get_effort_profile(cast("AutotuneEffort", effort))
    kwargs = strategy_cls.get_kwargs_from_profile(profile, bound.settings)
    parameters = inspect.signature(strategy_cls.__init__).parameters
    if llm_max_rounds is not None:
        if "max_rounds" in parameters:
            kwargs["max_rounds"] = llm_max_rounds
        elif "llm_max_rounds" in parameters:
            kwargs["llm_max_rounds"] = llm_max_rounds
    if llm_configs_per_round is not None:
        if "configs_per_round" in parameters:
            kwargs["configs_per_round"] = llm_configs_per_round
        elif "llm_configs_per_round" in parameters:
            kwargs["llm_configs_per_round"] = llm_configs_per_round
    if llm_initial_random_configs is not None:
        if "initial_random_configs" in parameters:
            kwargs["initial_random_configs"] = llm_initial_random_configs
        elif "llm_initial_random_configs" in parameters:
            kwargs["llm_initial_random_configs"] = llm_initial_random_configs
    return strategy_cls(bound, args, **kwargs)


def _selected_perf_ms(search: object) -> float:
    try:
        return float(search.best.perf)
    except Exception:
        return float(search.best_perf_so_far)


def _verify_config(
    kernel: object,
    args: tuple[object, ...],
    config: object,
    *,
    runs: int,
) -> dict[str, object]:
    import torch

    from helion.autotuner.benchmarking import do_bench

    bound = kernel.bind(args)
    compiled_fn = bound.compile_config(config)
    for _ in range(3):
        compiled_fn(*args)
    torch.cuda.synchronize()
    times_ms = [float(do_bench(lambda: compiled_fn(*args))) for _ in range(runs)]
    return {
        "times_ms": times_ms,
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
    }


def _read_convergence_csv(
    path: Path, *, stage: str, generation_offset: int, config_offset: int
) -> tuple[list[dict[str, object]], int, int]:
    events: list[dict[str, object]] = []
    best_ms = math.inf
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("status") != "ok":
                continue
            perf_text = row.get("perf_ms") or ""
            if perf_text == "":
                continue
            perf_ms = float(perf_text)
            if not math.isfinite(perf_ms):
                continue
            raw_generation = int(row["generation"])
            generation = raw_generation + generation_offset
            config_index = int(row["config_index"]) + config_offset
            best_ms = min(best_ms, perf_ms)
            events.append(
                {
                    "config_index": config_index,
                    "timestamp_s": (
                        float(row["timestamp_s"]) if row.get("timestamp_s") else None
                    ),
                    "generation": generation,
                    "raw_generation": raw_generation,
                    "stage": stage,
                    "perf_ms": perf_ms,
                    "best_ms": best_ms,
                }
            )
    next_generation_offset = (
        generation_offset
        + max([int(event["raw_generation"]) for event in events], default=-1)
        + 1
    )
    return events, next_generation_offset, config_offset + len(events)


def _parse_convergence_logs(base_path: Path) -> dict[str, list[dict[str, object]]]:
    log_paths = [
        ("main", base_path.with_suffix(".csv")),
        (
            "stage1_llm",
            base_path.with_name(f"{base_path.name}_stage1_llm").with_suffix(".csv"),
        ),
        (
            "stage2_lfbo",
            base_path.with_name(f"{base_path.name}_stage2_lfbo").with_suffix(".csv"),
        ),
    ]
    events: list[dict[str, object]] = []
    generation_offset = 0
    config_offset = 0
    for stage, path in log_paths:
        if not path.exists():
            continue
        stage_events, generation_offset, config_offset = _read_convergence_csv(
            path,
            stage=stage,
            generation_offset=generation_offset,
            config_offset=config_offset,
        )
        events.extend(stage_events)

    if not events:
        return {"events": [], "by_generation": []}

    running_best = math.inf
    best_by_generation: dict[int, float] = {}
    configs_by_generation: dict[int, int] = {}
    for event in events:
        generation = int(event["generation"])
        running_best = min(running_best, float(event["perf_ms"]))
        event["best_ms"] = running_best
        best_by_generation[generation] = min(
            best_by_generation.get(generation, math.inf), running_best
        )
        configs_by_generation[generation] = configs_by_generation.get(generation, 0) + 1

    running_best = math.inf
    cumulative_configs = 0
    by_generation: list[dict[str, object]] = []
    for generation in sorted(best_by_generation):
        running_best = min(running_best, best_by_generation[generation])
        cumulative_configs += configs_by_generation[generation]
        by_generation.append(
            {
                "generation": generation,
                "best_ms": running_best,
                "configs_seen": cumulative_configs,
            }
        )
    return {"events": events, "by_generation": by_generation}


def _active_heuristic_match_for_arm(
    arm: str,
    *,
    matched_observed_rule: bool,
    matched_range_policy: bool,
) -> bool:
    """Return whether this arm has an active matched heuristic for the workload."""
    (
        observed_enabled,
        prompt_enabled,
        seeds_enabled,
        _max_templates,
        _disabled_classes,
        prompt_mode,
    ) = ARM_SETTINGS[arm]
    if not observed_enabled:
        return False

    prompt_match = prompt_enabled and (
        matched_range_policy if prompt_mode == "range" else matched_observed_rule
    )
    seed_match = seeds_enabled and matched_observed_rule
    return prompt_match or seed_match


def _configure_worker_round0_env(
    args: argparse.Namespace,
    *,
    matched_observed_rule: bool,
    matched_range_policy: bool,
) -> dict[str, object]:
    """Set worker-local round-0 record/replay env after match detection."""
    mode = os.environ.get(LLM_ROUND0_MODE_ENV, "off")
    active_match = _active_heuristic_match_for_arm(
        args.arm,
        matched_observed_rule=matched_observed_rule,
        matched_range_policy=matched_range_policy,
    )
    metadata: dict[str, object] = {
        "mode": mode,
        "action": mode,
        "active_heuristic_match": active_match,
    }
    if mode != "paired-no-match":
        return metadata

    baseline_path = os.environ.get(LLM_ROUND0_PAIRED_BASELINE_PATH_ENV)
    if baseline_path is None:
        raise RuntimeError(
            f"{LLM_ROUND0_MODE_ENV}=paired-no-match requires "
            f"{LLM_ROUND0_PAIRED_BASELINE_PATH_ENV}"
        )

    os.environ[LLM_ROUND0_PAIRED_ACTIVE_MATCH_ENV] = "1" if active_match else "0"
    os.environ.pop(LLM_ROUND0_RECORD_PATH_ENV, None)
    os.environ.pop(LLM_ROUND0_REPLAY_PATH_ENV, None)

    if args.arm == "baseline":
        os.environ[LLM_ROUND0_RECORD_PATH_ENV] = baseline_path
        action = "record"
    elif active_match:
        action = "off_matched_heuristic"
    else:
        os.environ[LLM_ROUND0_REPLAY_PATH_ENV] = baseline_path
        action = "replay"

    os.environ[LLM_ROUND0_PAIRED_ACTION_ENV] = action
    metadata["action"] = action
    return metadata


def _run_worker(args: argparse.Namespace) -> int:
    _ensure_repo_on_path()
    workload = WORKLOADS[args.workload]
    kernel = _load_kernel(workload)
    kernel_args = _make_args(workload)

    import torch

    from helion._compat import get_device_name
    from helion.autotuner.llm.heuristics import _find_rule
    from helion.autotuner.llm.heuristics import _matched_range_policy
    from helion.autotuner.llm.heuristics import _shape_bucket_for_class
    from helion.autotuner.llm.heuristics import classify_runtime_kernel
    from helion.autotuner.llm.workload import detect_workload_traits

    start = time.perf_counter()
    search = _build_search(
        kernel,
        kernel_args,
        autotuner=args.autotuner,
        effort=args.effort,
        llm_max_rounds=args.llm_max_rounds,
        llm_configs_per_round=args.llm_configs_per_round,
        llm_initial_random_configs=args.llm_initial_random_configs,
    )
    workload_traits = detect_workload_traits(
        search.kernel,
        config_spec=search.config_spec,
    )
    kernel_class = classify_runtime_kernel(
        kernel_args,
        workload_traits=workload_traits,
        config_spec=search.config_spec,
    )
    shape_bucket = (
        _shape_bucket_for_class(kernel_class, kernel_args) if kernel_class else {}
    )
    matched_rule = (
        _find_rule(kernel_class, shape_bucket) is not None if kernel_class else False
    )
    _range_kernel_class, _range_shape_bucket, range_policy = _matched_range_policy(
        kernel_args,
        workload_traits=workload_traits,
        config_spec=search.config_spec,
    )
    matched_range_policy = range_policy is not None
    round0_metadata = _configure_worker_round0_env(
        args,
        matched_observed_rule=matched_rule,
        matched_range_policy=matched_range_policy,
    )
    best_config = search.autotune()
    wall_time_s = time.perf_counter() - start
    verified = _verify_config(kernel, kernel_args, best_config, runs=args.verify_runs)

    metrics = search._autotune_metrics
    convergence = _parse_convergence_logs(Path(args.autotune_log))
    benchmark_results = getattr(search, "_all_benchmark_results", [])
    compile_times_s = [
        float(result.compile_time)
        for result in benchmark_results
        if getattr(result, "compile_time", None) is not None
    ]
    benchmark_times_s = [
        float(value) for value in getattr(search, "_benchmark_times", [])
    ]
    result = {
        "workload": workload.name,
        "shape": workload.shape_name,
        "arm": args.arm,
        "autotuner": args.autotuner,
        "model": os.environ.get("HELION_LLM_MODEL", ""),
        "workload_traits": sorted(workload_traits),
        "kernel_class": kernel_class,
        "shape_bucket": shape_bucket,
        "matched_observed_rule": matched_rule,
        "matched_range_policy": matched_range_policy,
        "active_heuristic_match": round0_metadata["active_heuristic_match"],
        "observed_heuristics": os.environ.get("HELION_LLM_OBSERVED_HEURISTICS", ""),
        "observed_heuristic_prompt": os.environ.get(
            "HELION_LLM_OBSERVED_HEURISTIC_PROMPT", ""
        ),
        "observed_heuristic_prompt_mode": os.environ.get(
            "HELION_LLM_OBSERVED_HEURISTIC_PROMPT_MODE", ""
        ),
        "observed_heuristic_seeds": os.environ.get(
            "HELION_LLM_OBSERVED_HEURISTIC_SEEDS", ""
        ),
        "range_heuristics_path": os.environ.get("HELION_LLM_RANGE_HEURISTICS_PATH", ""),
        "llm_round0_mode": round0_metadata["mode"],
        "llm_round0_action": round0_metadata["action"],
        "llm_round0_record_path": os.environ.get(LLM_ROUND0_RECORD_PATH_ENV, ""),
        "llm_round0_replay_path": os.environ.get(LLM_ROUND0_REPLAY_PATH_ENV, ""),
        "llm_round0_paired_baseline_path": os.environ.get(
            LLM_ROUND0_PAIRED_BASELINE_PATH_ENV,
            "",
        ),
        "device": get_device_name(torch.device("cuda")) or "cuda",
        "best_config": dict(best_config),
        "autotuner_perf_ms": _selected_perf_ms(search),
        "verified": verified,
        "configs_tested": metrics.num_configs_tested,
        "compile_failures": metrics.num_compile_failures,
        "accuracy_failures": metrics.num_accuracy_failures,
        "wall_time_s": wall_time_s,
        "llm_call_times_s": getattr(search, "_llm_call_times", []),
        "benchmark_times_s": benchmark_times_s,
        "compile_time_per_config_stats": _value_stats(compile_times_s),
        "benchmark_time_per_batch_stats": _value_stats(benchmark_times_s),
        "hybrid_stage_breakdown": getattr(search, "hybrid_stage_breakdown", None),
        "autotune_log_csv": str(Path(args.autotune_log).with_suffix(".csv")),
        "convergence_events": convergence["events"],
        "convergence_by_generation": convergence["by_generation"],
    }
    Path(args.output).write_text(json.dumps(result, indent=2, default=_json_default))
    print(json.dumps(result, default=_json_default))
    return 0


def _arm_env(
    base_env: dict[str, str],
    *,
    output_dir: Path,
    workload: str,
    arm: str,
    autotuner: str,
    range_heuristics_path: Path | None,
) -> dict[str, str]:
    env = dict(base_env)
    conda_prefix = Path(env.get("CONDA_PREFIX", Path(sys.executable).parents[1]))
    _prepend_path_env(env, "LD_LIBRARY_PATH", conda_prefix / "lib")
    env["HELION_AUTOTUNER"] = autotuner
    env["HELION_AUTOTUNE_EFFORT"] = "full"
    env["HELION_FORCE_AUTOTUNE"] = "1"
    env["HELION_SKIP_CACHE"] = "1"
    env["HELION_AUTOTUNE_PROGRESS_BAR"] = "0"
    env["HELION_AUTOTUNE_LOG_LEVEL"] = "20"
    env["HELION_LLM_MODEL"] = env.get("HELION_LLM_MODEL", DEFAULT_MODEL)
    env["HELION_AUTOTUNE_BENCHMARK_SUBPROCESS"] = "1"
    env["HELION_AUTOTUNE_BENCH_SUBPROCESS"] = "1"
    env["HELION_CACHE_DIR"] = str(output_dir / "cache" / workload / arm)
    env.pop("TRITON_CACHE_DIR", None)
    if "HELION_LLM_API_BASE" not in env and env.get("CODEX_BASE_URL"):
        env["HELION_LLM_API_BASE"] = env["CODEX_BASE_URL"]
    if "HELION_LLM_API_KEY" not in env and env.get("OPENAI_API_KEY"):
        env["HELION_LLM_API_KEY"] = env["OPENAI_API_KEY"]
    if "HELION_LLM_CLIENT_CERT" not in env and env.get("CODEX_CLIENT_CERT"):
        env["HELION_LLM_CLIENT_CERT"] = env["CODEX_CLIENT_CERT"]
    if "HELION_LLM_CLIENT_KEY" not in env and env.get("CODEX_CLIENT_KEY"):
        env["HELION_LLM_CLIENT_KEY"] = env["CODEX_CLIENT_KEY"]
    if "HELION_LLM_CA_BUNDLE" not in env and env.get("CURL_CA_BUNDLE"):
        env["HELION_LLM_CA_BUNDLE"] = env["CURL_CA_BUNDLE"]

    (
        observed_enabled,
        prompt_enabled,
        seeds_enabled,
        max_templates,
        disabled_classes,
        prompt_mode,
    ) = ARM_SETTINGS[arm]
    if observed_enabled:
        env["HELION_LLM_OBSERVED_HEURISTICS"] = "1"
        env["HELION_LLM_OBSERVED_HEURISTIC_PROMPT"] = "1" if prompt_enabled else "0"
        env["HELION_LLM_OBSERVED_HEURISTIC_PROMPT_MODE"] = prompt_mode
        env["HELION_LLM_OBSERVED_HEURISTIC_SEEDS"] = "1" if seeds_enabled else "0"
        if prompt_mode == "range" and range_heuristics_path is not None:
            env["HELION_LLM_RANGE_HEURISTICS_PATH"] = str(range_heuristics_path)
        else:
            env.pop("HELION_LLM_RANGE_HEURISTICS_PATH", None)
        if max_templates is None:
            env.pop("HELION_LLM_OBSERVED_HEURISTIC_MAX_TEMPLATES", None)
        else:
            env["HELION_LLM_OBSERVED_HEURISTIC_MAX_TEMPLATES"] = str(max_templates)
        if disabled_classes:
            env["HELION_LLM_OBSERVED_HEURISTIC_DISABLED_CLASSES"] = disabled_classes
        else:
            env.pop("HELION_LLM_OBSERVED_HEURISTIC_DISABLED_CLASSES", None)
    else:
        env.pop("HELION_LLM_OBSERVED_HEURISTICS", None)
        env.pop("HELION_LLM_OBSERVED_HEURISTIC_PROMPT", None)
        env.pop("HELION_LLM_OBSERVED_HEURISTIC_PROMPT_MODE", None)
        env.pop("HELION_LLM_OBSERVED_HEURISTIC_SEEDS", None)
        env.pop("HELION_LLM_RANGE_HEURISTICS_PATH", None)
        env.pop("HELION_LLM_OBSERVED_HEURISTIC_MAX_TEMPLATES", None)
        env.pop("HELION_LLM_OBSERVED_HEURISTIC_DISABLED_CLASSES", None)
    return env


def _llm_round0_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    configured = getattr(args, "llm_round0_dir", None)
    if configured is not None:
        return Path(configured)
    return output_dir / "llm_round0"


def _llm_round0_artifact_path(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    workload: str,
    arm: str,
) -> Path:
    return _llm_round0_dir(args, output_dir) / f"{workload}_{arm}.json"


def _apply_suite_round0_env(
    env: dict[str, str],
    *,
    args: argparse.Namespace,
    output_dir: Path,
    workload: str,
    arm: str,
) -> None:
    """Wire suite CLI round-0 diagnostics into the worker subprocess env."""
    mode = getattr(args, "llm_round0_mode", "off")
    if mode == "off":
        return

    env[LLM_ROUND0_MODE_ENV] = mode
    env.pop(LLM_ROUND0_RECORD_PATH_ENV, None)
    env.pop(LLM_ROUND0_REPLAY_PATH_ENV, None)
    env.pop(LLM_ROUND0_PAIRED_BASELINE_PATH_ENV, None)
    env.pop(LLM_ROUND0_PAIRED_ACTIVE_MATCH_ENV, None)
    env.pop(LLM_ROUND0_PAIRED_ACTION_ENV, None)

    if mode == "record":
        env[LLM_ROUND0_RECORD_PATH_ENV] = str(
            _llm_round0_artifact_path(
                args,
                output_dir,
                workload=workload,
                arm=arm,
            )
        )
        return

    if mode == "replay":
        env[LLM_ROUND0_REPLAY_PATH_ENV] = str(
            _llm_round0_artifact_path(
                args,
                output_dir,
                workload=workload,
                arm=arm,
            )
        )
        return

    if mode == "paired-no-match":
        env[LLM_ROUND0_PAIRED_BASELINE_PATH_ENV] = str(
            _llm_round0_artifact_path(
                args,
                output_dir,
                workload=workload,
                arm="baseline",
            )
        )
        return

    raise ValueError(f"Unknown --llm-round0-mode {mode!r}")


def _run_one_subprocess(
    *,
    workload: str,
    arm: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, object]:
    result_path = output_dir / "raw" / f"{workload}_{arm}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs" / f"{workload}_{arm}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    autotune_log = output_dir / "autotune_logs" / f"{workload}_{arm}"
    env = _arm_env(
        os.environ,
        output_dir=output_dir,
        workload=workload,
        arm=arm,
        autotuner=args.autotuner,
        range_heuristics_path=args.range_heuristics_path,
    )
    env["HELION_AUTOTUNE_LOG"] = str(autotune_log)
    if args.model:
        env["HELION_LLM_MODEL"] = args.model
    if args.provider:
        env["HELION_LLM_PROVIDER"] = args.provider
    _apply_suite_round0_env(
        env,
        args=args,
        output_dir=output_dir,
        workload=workload,
        arm=arm,
    )
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--workload",
        workload,
        "--arm",
        arm,
        "--effort",
        args.effort,
        "--autotuner",
        args.autotuner,
        "--autotune-log",
        str(autotune_log),
        "--verify-runs",
        str(args.verify_runs),
        "--output",
        str(result_path),
    ]
    if args.llm_max_rounds is not None:
        cmd.extend(["--llm-max-rounds", str(args.llm_max_rounds)])
    if args.llm_configs_per_round is not None:
        cmd.extend(["--llm-configs-per-round", str(args.llm_configs_per_round)])
    if args.llm_initial_random_configs is not None:
        cmd.extend(
            [
                "--llm-initial-random-configs",
                str(args.llm_initial_random_configs),
            ]
        )
    with log_path.open("w") as log:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=args.timeout_s,
        )
    if completed.returncode != 0:
        return {
            "workload": workload,
            "arm": arm,
            "error": f"exit {completed.returncode}",
            "log": str(log_path),
        }
    return json.loads(result_path.read_text())


def _print_summary(results: Sequence[dict[str, object]]) -> None:
    by_workload: dict[str, dict[str, dict[str, object]]] = {}
    for result in results:
        by_workload.setdefault(str(result["workload"]), {})[str(result["arm"])] = result
    header = (
        f"{'workload':<28} {'arm':<10} {'base ms':>10} {'arm ms':>10} "
        f"{'perf cost':>10} {'base s':>9} {'arm s':>9} {'time cost':>10} "
        f"{'cfg cost':>9}"
    )
    print(header)
    print("-" * len(header))
    costs_by_arm: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: {"perf": [], "time": [], "cfg": []}
    )
    for workload, arms in sorted(by_workload.items()):
        base = arms.get("baseline")
        if not base or "error" in base:
            print(f"{workload:<22} incomplete")
            continue
        base_ms = float(cast("dict[str, object]", base["verified"])["median_ms"])
        base_s = float(base["wall_time_s"])
        base_cfgs = int(base["configs_tested"])
        for arm in sorted(arms):
            if arm == "baseline":
                continue
            candidate = arms[arm]
            if "error" in candidate:
                print(f"{workload:<28} {arm:<10} incomplete")
                continue
            arm_ms = float(
                cast("dict[str, object]", candidate["verified"])["median_ms"]
            )
            arm_s = float(candidate["wall_time_s"])
            arm_cfgs = int(candidate["configs_tested"])
            perf_cost = arm_ms / base_ms
            time_cost = arm_s / base_s
            cfg_cost = arm_cfgs / base_cfgs
            costs_by_arm[arm]["perf"].append(perf_cost)
            costs_by_arm[arm]["time"].append(time_cost)
            costs_by_arm[arm]["cfg"].append(cfg_cost)
            print(
                f"{workload:<28} {arm:<10} {base_ms:10.4f} {arm_ms:10.4f} "
                f"{perf_cost:10.2f} {base_s:9.1f} {arm_s:9.1f} "
                f"{time_cost:10.2f} {cfg_cost:9.2f}"
            )

    def geomean(costs: Sequence[float]) -> float:
        return math.exp(sum(math.log(cost) for cost in costs) / len(costs))

    if costs_by_arm:
        print("-" * len(header))
    for arm, costs in sorted(costs_by_arm.items()):
        perf_costs = costs["perf"]
        time_costs = costs["time"]
        cfg_costs = costs["cfg"]
        if not perf_costs:
            continue
        print(
            f"{'GEOMEAN':<28} {arm:<10} {'':>10} {'':>10} "
            f"{geomean(perf_costs):10.2f} {'':>9} {'':>9} "
            f"{geomean(time_costs):10.2f} {geomean(cfg_costs):9.2f}"
        )


def _convergence_rows(
    result: dict[str, object],
) -> Sequence[dict[str, object]]:
    return cast(
        "Sequence[dict[str, object]]", result.get("convergence_by_generation", [])
    )


def _write_convergence_artifacts(
    results: Sequence[dict[str, object]], output_dir: Path
) -> list[Path]:
    conv_dir = output_dir / "convergence"
    conv_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = conv_dir / "convergence.tsv"
    with tsv_path.open("w", encoding="utf-8") as out:
        out.write("workload\tarm\tgeneration\tconfigs_seen\tbest_ms\n")
        for result in results:
            if "error" in result:
                continue
            for row in _convergence_rows(result):
                out.write(
                    f"{result['workload']}\t{result['arm']}\t"
                    f"{row['generation']}\t{row['configs_seen']}\t"
                    f"{float(row['best_ms']):.6f}\n"
                )

    by_workload: dict[str, dict[str, dict[str, object]]] = {}
    for result in results:
        if "error" in result:
            continue
        by_workload.setdefault(str(result["workload"]), {})[str(result["arm"])] = result

    svg_paths = [tsv_path]
    for workload, arms in sorted(by_workload.items()):
        series: dict[str, list[dict[str, object]]] = {
            arm: list(_convergence_rows(result)) for arm, result in arms.items()
        }
        series = {arm: rows for arm, rows in series.items() if rows}
        if not series:
            continue

        width = 720
        height = 420
        left = 72
        right = 24
        top = 48
        bottom = 62
        plot_w = width - left - right
        plot_h = height - top - bottom
        generations = [
            int(row["generation"]) for rows in series.values() for row in rows
        ]
        values = [float(row["best_ms"]) for rows in series.values() for row in rows]
        max_x = max([*generations, 1])
        min_y = min(values)
        max_y = max(values)
        if min_y == max_y:
            pad = min_y * 0.05 if min_y else 0.001
            min_y -= pad
            max_y += pad
        else:
            pad = (max_y - min_y) * 0.08
            min_y -= pad
            max_y += pad

        def point(
            row: dict[str, object],
            *,
            left: int = left,
            max_x: int = max_x,
            max_y: float = max_y,
            min_y: float = min_y,
            plot_h: int = plot_h,
            plot_w: int = plot_w,
            top: int = top,
        ) -> tuple[float, float]:
            generation = int(row["generation"])
            best_ms = float(row["best_ms"])
            x = left + generation / max_x * plot_w
            y = top + (max_y - best_ms) / (max_y - min_y) * plot_h
            return x, y

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{left}" y="28" font-family="sans-serif" font-size="18" fill="#111827">{escape(workload)}</text>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#6b7280"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#6b7280"/>',
            f'<text x="{left + plot_w / 2 - 64}" y="{height - 18}" font-family="sans-serif" font-size="13" fill="#374151">autotune generation</text>',
            f'<text x="16" y="{top + plot_h / 2 + 48}" transform="rotate(-90 16 {top + plot_h / 2 + 48})" font-family="sans-serif" font-size="13" fill="#374151">best perf so far (ms)</text>',
            f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-family="monospace" font-size="11" fill="#374151">{min_y:.4f}</text>',
            f'<text x="{left - 8}" y="{top + plot_h}" text-anchor="end" font-family="monospace" font-size="11" fill="#374151">{max_y:.4f}</text>',
            f'<text x="{left}" y="{top + plot_h + 20}" text-anchor="middle" font-family="monospace" font-size="11" fill="#374151">0</text>',
            f'<text x="{left + plot_w}" y="{top + plot_h + 20}" text-anchor="middle" font-family="monospace" font-size="11" fill="#374151">{max_x}</text>',
        ]
        legend_x = width - right - 190
        legend_y = 28
        for index, arm in enumerate(sorted(series)):
            rows = series.get(arm)
            if not rows:
                continue
            color = ARM_COLORS.get(arm, "#374151")
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(point, rows))
            lines.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
            )
            for x, y in map(point, rows):
                lines.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>'
                )
            y = legend_y + index * 18
            lines.extend(
                [
                    f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="2.5"/>',
                    f'<text x="{legend_x + 32}" y="{y + 4}" font-family="sans-serif" font-size="13" fill="#111827">{arm}</text>',
                ]
            )
        lines.append("</svg>")
        svg_path = conv_dir / f"{workload}.svg"
        svg_path.write_text("\n".join(lines), encoding="utf-8")
        svg_paths.append(svg_path)
    return svg_paths


def _run_suite(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workloads = args.workloads.split(",")
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    if "baseline" not in arms:
        arms.insert(0, "baseline")
    elif args.llm_round0_mode == "paired-no-match" and arms[0] != "baseline":
        arms = ["baseline", *[arm for arm in arms if arm != "baseline"]]
    invalid_arms = sorted(set(arms) - set(ARM_SETTINGS))
    if invalid_arms:
        raise KeyError(f"Unknown arm(s) {invalid_arms}; valid: {sorted(ARM_SETTINGS)}")
    results: list[dict[str, object]] = []
    for workload in workloads:
        if workload not in WORKLOADS:
            raise KeyError(f"Unknown workload {workload!r}; valid: {sorted(WORKLOADS)}")
        for arm in arms:
            print(f"Running {workload} / {arm}", flush=True)
            result = _run_one_subprocess(
                workload=workload,
                arm=arm,
                args=args,
                output_dir=output_dir,
            )
            results.append(result)
            if "error" in result:
                print(f"  failed: {result['error']} log={result['log']}", flush=True)
            else:
                verified = cast("dict[str, object]", result["verified"])
                print(
                    f"  median={float(verified['median_ms']):.4f}ms "
                    f"configs={result['configs_tested']} "
                    f"time={float(result['wall_time_s']):.1f}s",
                    flush=True,
                )
    (output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, default=_json_default)
    )
    _print_summary(results)
    artifact_paths = _write_convergence_artifacts(results, output_dir)
    if len(artifact_paths) > 1:
        print(
            f"Wrote convergence graphs to {output_dir / 'convergence'} "
            f"({len(artifact_paths) - 1} SVGs)"
        )
    else:
        print(f"Wrote convergence table to {artifact_paths[0]}")
    print(f"\nWrote results to {output_dir}")
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd")

    suite = subparsers.add_parser("suite")
    suite.add_argument(
        "--workloads",
        default="add_1m,softmax_4k,rms_norm_4k",
        help="Comma-separated workload names.",
    )
    suite.add_argument("--model", default=DEFAULT_MODEL)
    suite.add_argument(
        "--autotuner",
        default=DEFAULT_AUTOTUNER,
        choices=AUTOTUNERS,
        help="Autotuner class to compare.",
    )
    suite.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help="Optional HELION_LLM_PROVIDER override, e.g. claude_cli or codex_cli.",
    )
    suite.add_argument("--effort", default="full", choices=["quick", "full"])
    suite.add_argument(
        "--arms",
        default="baseline,heuristics",
        help=(
            "Comma-separated experiment arms. Valid: "
            f"{', '.join(sorted(ARM_SETTINGS))}."
        ),
    )
    suite.add_argument("--verify-runs", type=int, default=10)
    suite.add_argument("--timeout-s", type=float, default=1800.0)
    suite.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    suite.add_argument(
        "--range-heuristics-path",
        type=Path,
        default=None,
        help="Optional JSON path used by the range_prompt arm.",
    )
    suite.add_argument(
        "--llm-max-rounds",
        type=int,
        default=None,
        help="Override LLMGuidedSearch max_rounds or hybrid llm_max_rounds.",
    )
    suite.add_argument(
        "--llm-configs-per-round",
        type=int,
        default=None,
        help=(
            "Override LLMGuidedSearch configs_per_round or hybrid "
            "llm_configs_per_round."
        ),
    )
    suite.add_argument(
        "--llm-initial-random-configs",
        type=int,
        default=None,
        help=(
            "Override LLMGuidedSearch initial_random_configs or hybrid "
            "llm_initial_random_configs."
        ),
    )
    suite.add_argument(
        "--llm-round0-mode",
        choices=("off", "record", "replay", "paired-no-match"),
        default="off",
        help=(
            "Diagnostic round-0 LLM response record/replay mode. "
            "paired-no-match records baseline and replays it only for arms "
            "without an active heuristic match."
        ),
    )
    suite.add_argument(
        "--llm-round0-dir",
        type=Path,
        default=None,
        help=(
            "Directory for round-0 record/replay artifacts. Defaults to "
            "<output-dir>/llm_round0 when --llm-round0-mode is not off."
        ),
    )

    worker = subparsers.add_parser("worker")
    worker.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    worker.add_argument("--arm", required=True, choices=sorted(ARM_SETTINGS))
    worker.add_argument("--effort", default="full", choices=["quick", "full"])
    worker.add_argument("--autotuner", default=DEFAULT_AUTOTUNER, choices=AUTOTUNERS)
    worker.add_argument("--autotune-log", required=True)
    worker.add_argument("--verify-runs", type=int, default=10)
    worker.add_argument("--output", required=True)
    worker.add_argument("--llm-max-rounds", type=int, default=None)
    worker.add_argument("--llm-configs-per-round", type=int, default=None)
    worker.add_argument("--llm-initial-random-configs", type=int, default=None)

    parsed = parser.parse_args(argv)
    if parsed.cmd is None:
        parser.print_help()
        raise SystemExit(2)
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.cmd == "worker":
        return _run_worker(args)
    if args.cmd == "suite":
        return _run_suite(args)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
