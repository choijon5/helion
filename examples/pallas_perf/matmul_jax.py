"""Plain JAX (``jnp.matmul``) reference variant for the matmul benchmark."""

from __future__ import annotations

import functools

from absl import app
import jax
import jax.numpy as jnp
import matmul_bench  # pyrefly: ignore [missing-import]


@functools.partial(jax.jit, static_argnames=["bm", "bk", "bn"])
def jax_matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
) -> jax.Array:
    # Block kwargs are part of the shared API; pure JAX ignores them.
    return jnp.matmul(x, y)


def main(argv: list[str]) -> None:
    matmul_bench.run(jax_matmul)


if __name__ == "__main__":
    app.run(main)
