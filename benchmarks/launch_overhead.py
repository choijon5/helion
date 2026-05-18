"""Per-call launch-overhead microbenchmark for the Helion ``add`` kernel.

This script measures the *median per-call wall-clock* time of four launch
paths so we can quantify the host-side overhead Helion adds on top of
Triton's launcher, and how far that sits from torch eager. All four paths
do the same work: element-wise add of two 1024x1024 bfloat16 tensors.

Columns
-------
* **Helion**: full Helion entry point (``Kernel.__call__`` -> ``BoundKernel``
  -> generated host wrapper -> ``default_launcher`` -> ``triton_kernel.run`` ->
  Triton's C launcher).
* **raw triton_kernel.run**: bypass Helion's wrapper and call the cached
  ``JITFunction.run(...)`` directly with the same kwargs Helion would
  pass. Still goes through Triton's Python launcher (binder +
  ``compute_cache_key`` + dict lookup) before reaching the C launcher.
* **raw compiled_kernel.run**: the C launcher Triton produces. We recover
  the cached ``CompiledKernel`` via Triton's documented ``warmup=True``
  API and call it directly. In G0 Helion does not cache a
  ``CompiledKernel`` itself; we extract it via Triton's public API. If
  that extraction fails on a given Triton version, this column prints
  ``null`` (allowed by the plan for G0).
* **torch eager**: ``torch.add(x, y)`` in a loop with the same shapes.

How to reproduce
----------------
On the gb200 devserver, pin GPU 0 (canonical GPU for this project) and
run:

```bash
cd <repo root>
source /home/jongsokchoi/helion_2/.venv/bin/activate
# If running from a worktree, point PYTHONPATH at it so import resolves
# to the worktree, not the shared editable install.
export PYTHONPATH=$(pwd):$PYTHONPATH

# Without CUDA graphs (the launcher-overhead view we actually optimize):
CUDA_VISIBLE_DEVICES=0 HELION_BENCHMARK_CUDAGRAPH=0 \\
    python benchmarks/launch_overhead.py

# With CUDA graphs (sanity check: should knock the per-call overhead
# down to roughly the same number across all rows):
CUDA_VISIBLE_DEVICES=0 HELION_BENCHMARK_CUDAGRAPH=1 \\
    python benchmarks/launch_overhead.py
```

Methodology
-----------
For each path we run ``N = 10_000`` calls in a tight Python loop and
report the per-call wall-clock in microseconds. The loop is repeated
``TRIALS`` times with a ``torch.cuda.synchronize()`` between trials; the
reported number is the median across trials. Within a trial we
synchronize only once at the end, so ``(wall_seconds / N) * 1e6``
captures per-call host overhead without per-iter sync noise.

Median is preferred over mean because the Python loop occasionally
suffers GC pauses; the median is robust to those one-off blips while
still rewarding genuine per-call wins.

When ``HELION_BENCHMARK_CUDAGRAPH=1``, each path is wrapped with
``torch.cuda.CUDAGraph`` capture+replay (a thin re-implementation of
``helion/autotuner/benchmarking.py::_make_cudagraph_replay``). The
captured graph replays the same kernel launch repeatedly; this is the
mode ``examples/add.py`` uses today via ``default_cudagraph=True``.

This benchmark intentionally avoids ``do_bench``-style CUDA events: we
want wall-clock from Python's perspective so that we measure the host
overhead Helion adds (allocator, Python frames, dict allocations) and
not just GPU activity. Triton/CUDA event-based timing would hide the
overhead we are trying to reduce.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
import statistics
import time
from typing import Protocol
from typing import cast
import warnings

from examples.add import add
import torch
from triton import knobs
from triton.runtime.driver import driver

N = 10_000
TRIALS = 5
SHAPE = (1024, 1024)
_TRUE_LITERALS = {"1", "true", "yes", "on"}
_FALSE_LITERALS = {"0", "false", "no", "off"}

Binder = Callable[..., tuple[dict[str, object], object, object]]


class ActiveDriver(Protocol):
    def get_current_device(self) -> object: ...

    def get_current_stream(self, device: object) -> object: ...


class TritonKernel(Protocol):
    device_caches: dict[object, tuple[object, object, object, object, Binder]]

    def run(
        self,
        *args: object,
        grid: tuple[int, ...],
        warmup: bool,
        **kwargs: object,
    ) -> object: ...


class CompiledKernel(Protocol):
    function: object
    packed_metadata: object

    @property
    def run(self) -> Callable[..., object]: ...

    def launch_metadata(
        self, grid: tuple[int, ...], stream: object, *args: object
    ) -> object: ...


def _active_driver() -> ActiveDriver:
    return cast("ActiveDriver", driver.active)


@dataclass(frozen=True)
class LaunchArgs:
    triton_kernel: TritonKernel
    grid: tuple[int, ...]
    args: tuple[object, ...]
    kwargs: dict[str, object]


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or (value := value.strip()) == "":
        return default
    lowered = value.lower()
    if lowered in _TRUE_LITERALS:
        return True
    if lowered in _FALSE_LITERALS:
        return False
    raise ValueError(
        f"{name} must be one of {_TRUE_LITERALS | _FALSE_LITERALS}, got {value!r}"
    )


def _make_cudagraph_replay(fn: Callable[[], object]) -> Callable[[], object]:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    static_output: list[object] = []
    with torch.cuda.graph(graph):
        static_output.append(fn())
    torch.cuda.synchronize()

    def replay() -> object:
        graph.replay()
        return static_output[0]

    return replay


def _capture_launch_args(x: torch.Tensor, y: torch.Tensor) -> LaunchArgs:
    captured: LaunchArgs | None = None
    bound = add.bind((x, y))
    bound_run = bound._run
    if bound_run is None:
        raise RuntimeError("add kernel was not compiled before launch capture")

    def capture_launcher(
        triton_kernel: TritonKernel,
        grid: tuple[int, ...],
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal captured
        captured = LaunchArgs(triton_kernel, grid, args, dict(kwargs))
        return triton_kernel.run(*args, grid=grid, warmup=False, **kwargs)

    bound_run(x, y, _launcher=capture_launcher)
    torch.cuda.synchronize()
    if captured is None:
        raise RuntimeError("failed to capture generated Triton launch arguments")
    return captured


def _raw_triton_runner(launch_args: LaunchArgs) -> Callable[[], object]:
    def run() -> object:
        return launch_args.triton_kernel.run(
            *launch_args.args,
            grid=launch_args.grid,
            warmup=False,
            **launch_args.kwargs,
        )

    return run


def _compiled_kernel_runner(launch_args: LaunchArgs) -> Callable[[], object] | None:
    try:
        compiled_kernel = cast(
            "CompiledKernel | None",
            launch_args.triton_kernel.run(
                *launch_args.args,
                grid=launch_args.grid,
                warmup=True,
                **launch_args.kwargs,
            ),
        )
        if compiled_kernel is None:
            return None

        active_driver = _active_driver()
        device = active_driver.get_current_device()
        _kernel_cache, _kernel_key_cache, _target, _backend, binder = (
            launch_args.triton_kernel.device_caches[device]
        )
        bound_args, _specialization, _options = binder(
            *launch_args.args, **launch_args.kwargs
        )
        kernel_args = tuple(bound_args.values())
        compiled_run = compiled_kernel.run
        grid_size = len(launch_args.grid)
        grid_0 = launch_args.grid[0]
        grid_1 = launch_args.grid[1] if grid_size > 1 else 1
        grid_2 = launch_args.grid[2] if grid_size > 2 else 1

        def run() -> object:
            stream = active_driver.get_current_stream(device)
            launch_metadata = compiled_kernel.launch_metadata(
                launch_args.grid, stream, *kernel_args
            )
            compiled_run(
                grid_0,
                grid_1,
                grid_2,
                stream,
                compiled_kernel.function,
                compiled_kernel.packed_metadata,
                launch_metadata,
                knobs.runtime.launch_enter_hook,
                knobs.runtime.launch_exit_hook,
                *kernel_args,
            )
            return None

        return run
    except Exception as error:
        warnings.warn(
            f"raw compiled_kernel.run is not reachable in this checkout: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _benchmark(fn: Callable[[], object], *, use_cudagraph: bool) -> float:
    fn()
    torch.cuda.synchronize()
    benchmark_fn = _make_cudagraph_replay(fn) if use_cudagraph else fn

    times_us: list[float] = []
    for _ in range(TRIALS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(N):
            benchmark_fn()
        torch.cuda.synchronize()
        times_us.append((time.perf_counter() - start) * 1_000_000 / N)
    return statistics.median(times_us)


def _format_us(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:.3f}"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for launch_overhead.py")

    use_cudagraph = _env_flag("HELION_BENCHMARK_CUDAGRAPH")
    x = torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16)

    helion_out = add(x, y)
    torch.testing.assert_close(helion_out, torch.add(x, y))
    torch.cuda.synchronize()

    launch_args = _capture_launch_args(x, y)
    raw_triton = _raw_triton_runner(launch_args)
    raw_compiled = _compiled_kernel_runner(launch_args)

    helion_us = _benchmark(lambda: add(x, y), use_cudagraph=use_cudagraph)
    raw_triton_us = _benchmark(raw_triton, use_cudagraph=use_cudagraph)
    raw_compiled_us = (
        None
        if raw_compiled is None
        else _benchmark(raw_compiled, use_cudagraph=use_cudagraph)
    )
    torch_us = _benchmark(lambda: torch.add(x, y), use_cudagraph=use_cudagraph)

    headers = (
        "Helion (us)",
        "raw triton_kernel.run (us)",
        "raw compiled_kernel.run (us)",
        "torch eager (us)",
    )
    values = (
        _format_us(helion_us),
        _format_us(raw_triton_us),
        _format_us(raw_compiled_us),
        _format_us(torch_us),
    )
    print(f"N={N} trials={TRIALS} HELION_BENCHMARK_CUDAGRAPH={int(use_cudagraph)}")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    print("| " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
