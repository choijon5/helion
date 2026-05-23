from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import torch

from ...runtime.config import Config
from .common import clamp_block_size_targets
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


# Headline-family fast config measured by the Deep Replan 2026-05-23 cross-shape
# probe (plan.md §2.5 row 2): on bf16 1024^3 the forced
# ``[512, 512, 512] emit_pipeline pb=False`` config landed at 161 us, the
# fastest known Helion configuration -- but the autotuner has been mis-ranking
# it against ``unroll [1024, _, _]``-family picks because of pod-wide
# ~10-20% per-call noise on short kernels.
#
# Seeding the initial population with this config guarantees the autotuner
# always considers it: combined with the existing final-pick verification
# phase the seed survives the noisy initial rank and reaches the top-K
# rebench so consistently picks (or stays close to) this family.
_PALLAS_SQUARE_BLOCK = 512


class PallasMatmulSquareSeedHeuristic(AutotunerHeuristic):
    """Seed config for square-ish Pallas bf16/fp16 matmul on TPU.

    Fires when every static dimension is large enough to admit the
    ``[512, 512, 512]`` block tile (M, N, K all >= 512), the matmul is 2D
    (lhs.ndim == rhs.ndim == 2), and the inputs are bf16/fp16. Emits the
    ``emit_pipeline pre_broadcast=False`` family that the cross-shape probe
    identified as fastest for the headline shape -- so the search starts
    from a config already inside the best-known family rather than
    discovering it by mutation.
    """

    name = "pallas_matmul_square_seed"
    backend = "pallas"
    BLOCK_TARGETS = (_PALLAS_SQUARE_BLOCK, _PALLAS_SQUARE_BLOCK, _PALLAS_SQUARE_BLOCK)

    @classmethod
    def _eligible_fact(cls, env: CompileEnvironment) -> bool:
        facts = env.config_spec.matmul_facts
        if len(facts) != 1:
            return False
        fact = facts[0]
        if fact.lhs_ndim != 2 or fact.rhs_ndim != 2:
            return False
        if fact.lhs_dtype not in (torch.bfloat16, torch.float16):
            return False
        if fact.rhs_dtype != fact.lhs_dtype:
            return False
        if (
            fact.static_m is None
            or fact.static_n is None
            or fact.static_k is None
            or fact.m_block_id is None
            or fact.n_block_id is None
            or fact.k_block_id is None
        ):
            return False
        if min(fact.static_m, fact.static_n, fact.static_k) < _PALLAS_SQUARE_BLOCK:
            return False
        return (
            clamp_block_size_targets(
                env,
                [
                    (fact.m_block_id, fact.static_m, cls.BLOCK_TARGETS[0]),
                    (fact.k_block_id, fact.static_k, cls.BLOCK_TARGETS[1]),
                    (fact.n_block_id, fact.static_n, cls.BLOCK_TARGETS[2]),
                ],
            )
            is not None
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not cls._eligible_fact(env):
            return False
        # Only seed the loop-type/pre-broadcast pair when the spec actually
        # exposes those knobs; gracefully no-op otherwise.
        spec = env.config_spec
        return spec.supports_config_key(
            "pallas_loop_type"
        ) and spec.supports_config_key("pallas_pre_broadcast")

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        fact = env.config_spec.matmul_facts[0]
        assert fact.m_block_id is not None
        assert fact.n_block_id is not None
        assert fact.k_block_id is not None
        assert fact.static_m is not None
        assert fact.static_n is not None
        assert fact.static_k is not None
        block_sizes = clamp_block_size_targets(
            env,
            [
                (fact.m_block_id, fact.static_m, cls.BLOCK_TARGETS[0]),
                (fact.k_block_id, fact.static_k, cls.BLOCK_TARGETS[1]),
                (fact.n_block_id, fact.static_n, cls.BLOCK_TARGETS[2]),
            ],
        )
        if block_sizes is None:
            return None
        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
            "pallas_loop_type": "emit_pipeline",
            "pallas_pre_broadcast": False,
        }
        return Config(**seed)
