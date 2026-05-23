"""Shared benchmark runner used by ``matmul_jax`` and ``matmul_pallas``.

Iterates over the configuration matrix defined in ``matmul_configs`` and prints
one ``RESULT:`` line per shape / dtype / block combination so the variant
runner can parse it.
"""

from __future__ import annotations

import sys
import timeit
from typing import Callable

import jax
import jax.numpy as jnp

# Sibling modules in this directory are imported by name because the benchmark
# scripts are invoked as standalone files (``python matmul_jax.py``). Pyrefly
# resolves the project from the repo root and can't see them, so suppress.
from matmul_configs import BLOCK_CONFIGS  # pyrefly: ignore [missing-import]
from matmul_configs import DTYPES  # pyrefly: ignore [missing-import]
from matmul_configs import SHAPES  # pyrefly: ignore [missing-import]
import numpy as np

_JAX_DTYPES = {
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
}
dtypes = [_JAX_DTYPES[d] for d in DTYPES]


def run(matmul_fn: Callable[..., jax.Array]) -> None:
    for dtype in dtypes:
        for shape in SHAPES:
            m, k, n = shape

            x = jax.random.normal(jax.random.key(0), (m, k), dtype=dtype)
            y = jax.random.normal(jax.random.key(1), (k, n), dtype=dtype)
            expected = jnp.dot(x, y, preferred_element_type=jnp.float32)

            for bm, bk, bn in BLOCK_CONFIGS:
                # Correctness check
                try:
                    actual = matmul_fn(x, y, bm=bm, bk=bk, bn=bn).block_until_ready()
                    rtol = 5e-2 if dtype == jnp.bfloat16 else (5e-2 if n == 1 else 1e-3)
                    atol = 5e-2 if dtype == jnp.bfloat16 else (5e-2 if n == 1 else 1e-3)
                    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
                except Exception as e:
                    print(
                        f"FAILED correctness or compilation for {shape} {bm}x{bk}x{bn}"
                        f" {dtype}: {e}",
                        file=sys.stderr,
                    )
                    continue

                def _run(
                    matmul_fn: Callable[..., jax.Array] = matmul_fn,
                    x: jax.Array = x,
                    y: jax.Array = y,
                    bm: int = bm,
                    bk: int = bk,
                    bn: int = bn,
                ) -> None:
                    matmul_fn(x, y, bm=bm, bk=bk, bn=bn).block_until_ready()

                # Warm up.
                _run()

                n_iter = 20
                n_repeats = 5
                samples = (
                    np.array(timeit.repeat(_run, repeat=n_repeats, number=n_iter))
                    / n_iter
                )
                mean = np.mean(samples) * 1e6
                std = np.std(samples) * 1e6

                print(
                    f"RESULT:{np.dtype(dtype).name}_{m}x{k}x{n}_{bm}x{bk}x{bn}:{mean}:{std}"
                )


def main() -> None:
    """Entry point only used to satisfy the ``main()``-in-examples lint.

    The real driver code lives in the per-variant scripts (``matmul_jax``,
    ``matmul_pallas``, ``matmul_helion``); each of them imports ``run`` and
    passes its own matmul function in. Running this file directly does nothing
    useful — point at one of those scripts instead.
    """
    print(
        "matmul_bench is a shared benchmark runner; "
        "invoke matmul_jax / matmul_pallas / matmul_helion instead.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
