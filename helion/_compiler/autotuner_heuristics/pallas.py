from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple

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

# Skinny-N family (plan.md §5 G3-A). Cycle-17 G3-A-pin ablation showed
# the ``1024×1024×1`` shape's per-shape best is
# ``unroll [1024, 1024, 1] pb=True`` (pinned-ceiling H/P 1.042 single
# sweep). Under the real-user seeded-autotuner methodology the autotuner
# instead lands on smaller-M / smaller-K ``unroll`` variants and ends up
# just below 1.00 (median 0.990). Seeding the per-shape winner guarantees
# the autotuner always considers it.
_PALLAS_SKINNY_N_MAX_M = 1024
_PALLAS_SKINNY_N_MAX_K = 1024
_PALLAS_SKINNY_N_MIN_MK = 256

# Tall-M family (plan.md §5 G3-A). Cycle-17 G3-A-pin ablation showed
# the ``128×1024×1024`` shape's per-shape best is
# ``unroll [128, 1024, 1024] pb=True`` (pinned-ceiling H/P 1.021 single
# sweep). The autotuner instead picks ``emit_pipeline`` with smaller
# block sizes on K/N and lands median 0.992. Seeding the per-shape winner
# fixes the search bias.
_PALLAS_TALL_M_MAX_M = 256
_PALLAS_TALL_M_MAX_K = 1024
_PALLAS_TALL_M_MAX_N = 1024
_PALLAS_TALL_M_MIN_KN = 512

# f32 family (see plan.md §5 G4 for per-shape closure status and the
# residual gaps). f32 has no MXU shortcut (TPU MXU is bf16/fp8 only),
# so Helion routes through ``lax.dot_general(precision=HIGHEST)`` and
# the Pallas reference mirrors (see ``examples/pallas_perf/matmul_pallas.py``).
# Under the seeded autotuner the f32 picks drift across families;
# seeding the per-shape ablation winners improves pick consistency.
# (It does NOT necessarily close H/P ≥ 1.00 on every shape — see
# plan.md §5 G4 for which shapes remain 🟡 deferred.)
_PALLAS_F32_SQUARE_BLOCK = 512
# Skinny-N family clamp targets. Note: ``clamp_block_size_targets``
# returns the block sizes in INPUT ORDER, and ``Config`` positionally
# maps ``[m_id=0, n_id=1, k_id=2]``. The f32 seed heuristic passes
# ``[(m_id, ...), (n_id, ...), (k_id, ...)]`` in matching positional
# order so the seed encodes the true intent. The bf16
# ``PallasMatmulSkinnyNSeedHeuristic`` is documented to pass
# ``[m, k, n]`` order which works only by accidental encoding when one
# operand is a singleton (the trailing dim is 1 → the wrong-slot lookup
# returns the same value); the f32 variant keeps the canonical
# ``[m, n, k]`` order so the seed is robust under any future shape.
_PALLAS_F32_SKINNY_N_MIN_MK = 256
_PALLAS_F32_SKINNY_N_M_TARGET = 512
_PALLAS_F32_SKINNY_N_K_TARGET = 512


class _PallasMatmulFactDims(NamedTuple):
    """Static-shape view of a single 2D bf16/fp16 matmul fact.

    Narrows ``MatmulFact``'s ``int | None`` fields to ``int`` after the
    shared eligibility gate confirms every field is known, so the
    per-shape heuristics can index without re-asserting.
    """

    m: int
    k: int
    n: int
    m_block_id: int
    k_block_id: int
    n_block_id: int


_BF16_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float16)
_F32_DTYPES: tuple[torch.dtype, ...] = (torch.float32,)


def _pallas_matmul_seed_dims_or_none(
    env: CompileEnvironment,
    *,
    allowed_dtypes: tuple[torch.dtype, ...] = _BF16_DTYPES,
) -> _PallasMatmulFactDims | None:
    """Return the single 2D matmul fact's static dims, or ``None``.

    Shared eligibility gate for the per-shape Pallas seed heuristics:
    one matmul fact, 2D x 2D, same dtype across operands, all of
    ``static_{m,n,k}`` and ``{m,n,k}_block_id`` known. Returns a
    narrowed view so each heuristic can layer its own shape predicate
    on top. Default ``allowed_dtypes=(bfloat16, float16)`` matches the
    historical bf16/fp16 family heuristics; pass ``(float32,)`` (or any
    other dtype tuple) when adding f32-specific seeds.
    """
    facts = env.config_spec.matmul_facts
    if len(facts) != 1:
        return None
    fact = facts[0]
    if fact.lhs_ndim != 2 or fact.rhs_ndim != 2:
        return None
    if fact.lhs_dtype not in allowed_dtypes:
        return None
    if fact.rhs_dtype != fact.lhs_dtype:
        return None
    if (
        fact.static_m is None
        or fact.static_n is None
        or fact.static_k is None
        or fact.m_block_id is None
        or fact.n_block_id is None
        or fact.k_block_id is None
    ):
        return None
    return _PallasMatmulFactDims(
        m=fact.static_m,
        k=fact.static_k,
        n=fact.static_n,
        m_block_id=fact.m_block_id,
        k_block_id=fact.k_block_id,
        n_block_id=fact.n_block_id,
    )


def _pallas_matmul_supports_loop_type_and_pre_broadcast(
    env: CompileEnvironment,
) -> bool:
    spec = env.config_spec
    return spec.supports_config_key("pallas_loop_type") and spec.supports_config_key(
        "pallas_pre_broadcast"
    )


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
    def _eligible_dims(cls, env: CompileEnvironment) -> _PallasMatmulFactDims | None:
        dims = _pallas_matmul_seed_dims_or_none(env)
        if dims is None:
            return None
        if min(dims.m, dims.n, dims.k) < _PALLAS_SQUARE_BLOCK:
            return None
        if (
            clamp_block_size_targets(
                env,
                [
                    (dims.m_block_id, dims.m, cls.BLOCK_TARGETS[0]),
                    (dims.k_block_id, dims.k, cls.BLOCK_TARGETS[1]),
                    (dims.n_block_id, dims.n, cls.BLOCK_TARGETS[2]),
                ],
            )
            is None
        ):
            return None
        return dims

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if cls._eligible_dims(env) is None:
            return False
        # Only seed the loop-type/pre-broadcast pair when the spec actually
        # exposes those knobs; gracefully no-op otherwise.
        return _pallas_matmul_supports_loop_type_and_pre_broadcast(env)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        dims = cls._eligible_dims(env)
        if dims is None:
            return None
        block_sizes = clamp_block_size_targets(
            env,
            [
                (dims.m_block_id, dims.m, cls.BLOCK_TARGETS[0]),
                (dims.k_block_id, dims.k, cls.BLOCK_TARGETS[1]),
                (dims.n_block_id, dims.n, cls.BLOCK_TARGETS[2]),
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


class PallasMatmulSkinnyNSeedHeuristic(AutotunerHeuristic):
    """Seed config for skinny-N (``N == 1``) Pallas bf16/fp16 matmul on TPU.

    Fires on 2D bf16/fp16 matmul with ``N == 1`` and ``M, K >= 256``.
    Seeds ``unroll [M, K, 1] pre_broadcast=True`` -- the per-shape winner
    identified by the cycle-17 G3-A-pin ablation (single-sweep H/P 1.042
    on bf16 1024×1024×1). Under the real-user seeded-autotuner metric
    (cycle-18 methodology), the autotuner consistently lands on
    ``unroll`` configs but with smaller M / K blocks and ends up just
    below the 1.00 bar (median 0.990); seeding the full-1024 winner
    fixes the search bias.
    """

    name = "pallas_matmul_skinny_n_seed"
    backend = "pallas"

    @classmethod
    def _eligible_dims(cls, env: CompileEnvironment) -> _PallasMatmulFactDims | None:
        dims = _pallas_matmul_seed_dims_or_none(env)
        if dims is None:
            return None
        if dims.n != 1:
            return None
        if dims.m < _PALLAS_SKINNY_N_MIN_MK or dims.k < _PALLAS_SKINNY_N_MIN_MK:
            return None
        if (
            clamp_block_size_targets(
                env,
                [
                    (dims.m_block_id, dims.m, min(dims.m, _PALLAS_SKINNY_N_MAX_M)),
                    (dims.k_block_id, dims.k, min(dims.k, _PALLAS_SKINNY_N_MAX_K)),
                    (dims.n_block_id, dims.n, 1),
                ],
            )
            is None
        ):
            return None
        return dims

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if cls._eligible_dims(env) is None:
            return False
        return _pallas_matmul_supports_loop_type_and_pre_broadcast(env)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        dims = cls._eligible_dims(env)
        if dims is None:
            return None
        block_sizes = clamp_block_size_targets(
            env,
            [
                (dims.m_block_id, dims.m, min(dims.m, _PALLAS_SKINNY_N_MAX_M)),
                (dims.k_block_id, dims.k, min(dims.k, _PALLAS_SKINNY_N_MAX_K)),
                (dims.n_block_id, dims.n, 1),
            ],
        )
        if block_sizes is None:
            return None
        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
            "pallas_loop_type": "unroll",
            "pallas_pre_broadcast": True,
        }
        return Config(**seed)


class PallasMatmulTallMSeedHeuristic(AutotunerHeuristic):
    """Seed config for tall-M (small ``M``, large ``K``/``N``) Pallas matmul.

    Fires on 2D bf16/fp16 matmul with ``M <= 256`` and
    ``K, N >= 512``. Seeds ``unroll [M, K, N] pre_broadcast=True`` -- the
    per-shape winner identified by the cycle-17 G3-A-pin ablation
    (single-sweep H/P 1.021 on bf16 128×1024×1024). Under the real-user
    seeded-autotuner metric the autotuner consistently picks
    ``emit_pipeline`` with smaller K/N blocks and lands median 0.992;
    seeding the full-1024 unroll winner fixes the search bias.
    """

    name = "pallas_matmul_tall_m_seed"
    backend = "pallas"

    @classmethod
    def _eligible_dims(cls, env: CompileEnvironment) -> _PallasMatmulFactDims | None:
        dims = _pallas_matmul_seed_dims_or_none(env)
        if dims is None:
            return None
        if dims.m > _PALLAS_TALL_M_MAX_M:
            return None
        if dims.k < _PALLAS_TALL_M_MIN_KN or dims.n < _PALLAS_TALL_M_MIN_KN:
            return None
        if (
            clamp_block_size_targets(
                env,
                [
                    (dims.m_block_id, dims.m, dims.m),
                    (dims.k_block_id, dims.k, min(dims.k, _PALLAS_TALL_M_MAX_K)),
                    (dims.n_block_id, dims.n, min(dims.n, _PALLAS_TALL_M_MAX_N)),
                ],
            )
            is None
        ):
            return None
        return dims

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if cls._eligible_dims(env) is None:
            return False
        return _pallas_matmul_supports_loop_type_and_pre_broadcast(env)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        dims = cls._eligible_dims(env)
        if dims is None:
            return None
        block_sizes = clamp_block_size_targets(
            env,
            [
                (dims.m_block_id, dims.m, dims.m),
                (dims.k_block_id, dims.k, min(dims.k, _PALLAS_TALL_M_MAX_K)),
                (dims.n_block_id, dims.n, min(dims.n, _PALLAS_TALL_M_MAX_N)),
            ],
        )
        if block_sizes is None:
            return None
        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
            "pallas_loop_type": "unroll",
            "pallas_pre_broadcast": True,
        }
        return Config(**seed)


class PallasMatmulF32SquareSeedHeuristic(AutotunerHeuristic):
    """Seed config for square-ish Pallas f32 matmul on TPU.

    Fires when every static dimension is ``>= 512``, the matmul is 2D
    (lhs.ndim == rhs.ndim == 2), and both inputs are ``float32``. Emits
    ``unroll [512, 512, 512] pre_broadcast=True`` -- the per-shape best
    identified by G4 f32 ablation. f32 has no MXU shortcut so Helion
    routes through ``lax.dot_general(precision=HIGHEST)`` and the
    Pallas reference mirrors with ``pl.dot(..., precision=HIGHEST)``
    for apples-to-apples; the seeded config improves pick consistency
    (the autotuner lands the seed family on most sweeps rather than
    drifting). See plan.md §5 G4 for the 10-sweep closure status —
    this heuristic does NOT, by itself, close H/P ≥ 1.00 on the f32
    headline shape (the kernel at the seed config is ~3% behind
    Pallas under matched-precision timing; G4-headline-tuner-v2
    follow-up queued).
    """

    name = "pallas_matmul_f32_square_seed"
    backend = "pallas"
    BLOCK_TARGETS = (
        _PALLAS_F32_SQUARE_BLOCK,
        _PALLAS_F32_SQUARE_BLOCK,
        _PALLAS_F32_SQUARE_BLOCK,
    )

    @classmethod
    def _eligible_dims(cls, env: CompileEnvironment) -> _PallasMatmulFactDims | None:
        dims = _pallas_matmul_seed_dims_or_none(env, allowed_dtypes=_F32_DTYPES)
        if dims is None:
            return None
        if min(dims.m, dims.n, dims.k) < _PALLAS_F32_SQUARE_BLOCK:
            return None
        if (
            clamp_block_size_targets(
                env,
                [
                    (dims.m_block_id, dims.m, cls.BLOCK_TARGETS[0]),
                    (dims.n_block_id, dims.n, cls.BLOCK_TARGETS[1]),
                    (dims.k_block_id, dims.k, cls.BLOCK_TARGETS[2]),
                ],
            )
            is None
        ):
            return None
        return dims

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if cls._eligible_dims(env) is None:
            return False
        return _pallas_matmul_supports_loop_type_and_pre_broadcast(env)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        dims = cls._eligible_dims(env)
        if dims is None:
            return None
        block_sizes = clamp_block_size_targets(
            env,
            [
                (dims.m_block_id, dims.m, cls.BLOCK_TARGETS[0]),
                (dims.n_block_id, dims.n, cls.BLOCK_TARGETS[1]),
                (dims.k_block_id, dims.k, cls.BLOCK_TARGETS[2]),
            ],
        )
        if block_sizes is None:
            return None
        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
            "pallas_loop_type": "unroll",
            "pallas_pre_broadcast": True,
        }
        return Config(**seed)


class PallasMatmulF32SkinnyNSeedHeuristic(AutotunerHeuristic):
    """Seed config for skinny-N (``N == 1``) Pallas f32 matmul on TPU.

    Fires on 2D float32 matmul with ``N == 1`` and ``M, K >= 256``.
    Seeds ``unroll [512, 1, 512] pre_broadcast=True`` -- the per-shape
    winner identified by G4 f32 ablation. The autotuner under seed=0
    picks ``unroll [m, 1, k]`` family but often lands on small k_tile
    (128) on many sweeps; seeding the better block split improves pick
    consistency. See plan.md §5 G4 for the 10-sweep closure status —
    the seed brings the median within paired-sample precision of 1.00
    but does NOT, by itself, reliably clear the bar
    (G4-skinny-N-tuner-v2 follow-up queued).

    Unlike ``PallasMatmulSkinnyNSeedHeuristic`` (bf16/fp16) which passes
    its clamp arguments in ``[m, k, n]`` order, this f32 variant passes
    ``[m, n, k]`` order so the returned block sizes line up with
    ``Config.block_sizes`` positional interpretation (block_id 0 = m,
    1 = n, 2 = k). The bf16 heuristic's seed is documented to work
    accidentally because ``n=1`` short-circuits the ``from_config``
    lookup at the wrong slot -- the f32 heuristic keeps the canonical
    order so the seed is robust under any future shape.
    """

    name = "pallas_matmul_f32_skinny_n_seed"
    backend = "pallas"

    @classmethod
    def _eligible_dims(cls, env: CompileEnvironment) -> _PallasMatmulFactDims | None:
        dims = _pallas_matmul_seed_dims_or_none(env, allowed_dtypes=_F32_DTYPES)
        if dims is None:
            return None
        if dims.n != 1:
            return None
        if dims.m < _PALLAS_F32_SKINNY_N_MIN_MK or dims.k < _PALLAS_F32_SKINNY_N_MIN_MK:
            return None
        if (
            clamp_block_size_targets(
                env,
                [
                    (
                        dims.m_block_id,
                        dims.m,
                        min(dims.m, _PALLAS_F32_SKINNY_N_M_TARGET),
                    ),
                    (dims.n_block_id, dims.n, 1),
                    (
                        dims.k_block_id,
                        dims.k,
                        min(dims.k, _PALLAS_F32_SKINNY_N_K_TARGET),
                    ),
                ],
            )
            is None
        ):
            return None
        return dims

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if cls._eligible_dims(env) is None:
            return False
        return _pallas_matmul_supports_loop_type_and_pre_broadcast(env)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        dims = cls._eligible_dims(env)
        if dims is None:
            return None
        block_sizes = clamp_block_size_targets(
            env,
            [
                (
                    dims.m_block_id,
                    dims.m,
                    min(dims.m, _PALLAS_F32_SKINNY_N_M_TARGET),
                ),
                (dims.n_block_id, dims.n, 1),
                (
                    dims.k_block_id,
                    dims.k,
                    min(dims.k, _PALLAS_F32_SKINNY_N_K_TARGET),
                ),
            ],
        )
        if block_sizes is None:
            return None
        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
            "pallas_loop_type": "unroll",
            "pallas_pre_broadcast": True,
        }
        return Config(**seed)
