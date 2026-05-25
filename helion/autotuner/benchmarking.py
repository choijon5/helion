from __future__ import annotations

import contextlib
import functools
import logging
import math
import os
import statistics
import tempfile
import time
from typing import Any
from typing import Callable
from typing import TypeVar

import torch

from ..runtime.settings import _env_get_bool
from ..runtime.settings import _get_backend
from ..runtime.settings import is_pallas_interpret
from .progress_bar import iter_with_progress
from helion._dist_utils import sync_object

T = TypeVar("T")

_log = logging.getLogger(__name__)
_BENCHMARK_CUDAGRAPH_ENV = "HELION_BENCHMARK_CUDAGRAPH"


def _cudagraph_unavailable_reason() -> str | None:
    if getattr(torch.version, "hip", None) is not None:
        return "CUDA graph benchmarking is only enabled for NVIDIA CUDA"
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if torch.cuda.is_current_stream_capturing():
        return "the current CUDA stream is already capturing"
    return None


def _make_cudagraph_replay(fn: Callable[[], T]) -> Callable[[], T]:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    static_output: list[T] = []
    with torch.cuda.graph(graph):
        static_output.append(fn())
    torch.cuda.synchronize()

    def replay() -> T:
        graph.replay()
        return static_output[0]

    return replay


def _maybe_cudagraph_replay(
    fn: Callable[[], T], *, default_enabled: bool = False
) -> Callable[[], T]:
    if not _env_get_bool(_BENCHMARK_CUDAGRAPH_ENV, default=default_enabled):
        return fn

    reason = _cudagraph_unavailable_reason()
    if reason is not None:
        _log.debug("Skipping CUDA graph benchmarking: %s", reason)
        return fn

    try:
        return _make_cudagraph_replay(fn)
    except Exception:
        _log.debug("CUDA graph benchmark capture failed; falling back", exc_info=True)
        return fn


def synchronize_device(result: object = None) -> None:
    """Wait for device computation to complete.

    For TPU tensors, uses ``torch_tpu``'s tensor-level sync which truly
    blocks until the device finishes (``torch.accelerator.synchronize()``
    does not reliably wait on ``torch_tpu``).  For all other cases, falls
    back to ``torch.accelerator.synchronize()``.
    """
    if isinstance(result, torch.Tensor) and result.device.type == "tpu":
        try:
            from torch_tpu._internal.sync import (  # pyrefly: ignore[missing-import]
                synchronize as tpu_sync,
            )

            tpu_sync(result, wait=True)
            return
        except ImportError:
            raise ImportError(
                "torch_tpu is required for reliable device synchronization on TPU. "
                "Install torch_tpu or torch.accelerator.synchronize() will return "
                "before device computation finishes, producing incorrect benchmarks."
            ) from None
    if (
        not is_pallas_interpret()
        and _get_backend() != "pallas"
        and torch.accelerator.is_available()
    ):
        torch.accelerator.synchronize()


def compute_repeat(
    fn: Callable[[], object],
    *,
    target_ms: float = 100.0,
    min_repeat: int = 10,
    max_repeat: int = 1000,
    estimate_runs: int = 5,
    default_cudagraph: bool = False,
) -> int:
    """
    Estimate how many repetitions are needed to collect a stable benchmark for a
    single function call, mirroring Triton's ``do_bench`` heuristic while
    clamping the result between ``min_repeat`` and ``max_repeat``.
    """
    from triton import runtime

    di = runtime.driver.active.get_device_interface()  # type: ignore[attr-defined]
    cache = runtime.driver.active.get_empty_cache_for_benchmark()  # type: ignore[attr-defined]

    # Warm the pipeline once before collecting timing samples.
    fn()
    di.synchronize()
    benchmark_function = _maybe_cudagraph_replay(fn, default_enabled=default_cudagraph)

    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(estimate_runs):
        runtime.driver.active.clear_cache(cache)  # type: ignore[attr-defined]
        benchmark_function()
    end_event.record()
    di.synchronize()

    estimate_ms = start_event.elapsed_time(end_event) / max(estimate_runs, 1)
    if not math.isfinite(estimate_ms) or estimate_ms <= 0:
        return max_repeat

    repeat = int(target_ms / estimate_ms)
    return max(min_repeat, min(max_repeat, max(1, repeat)))


def compute_repeat_generic(
    fn: Callable[[], object],
    *,
    target_ms: float = 100.0,
    min_repeat: int = 10,
    max_repeat: int = 1000,
    estimate_runs: int = 5,
    default_cudagraph: bool = False,  # accepted for API symmetry; wall-clock timing doesn't use CG
) -> int:
    """
    Estimate how many repetitions are needed using wall-clock timing.
    Used for backends that don't have Triton's event-based timing (e.g., Pallas/TPU).
    """
    # Warm the pipeline once before collecting timing samples.
    out = fn()
    synchronize_device(out)

    start = time.perf_counter()
    for _ in range(estimate_runs):
        out = fn()
    synchronize_device(out)
    end = time.perf_counter()

    estimate_ms = (end - start) * 1000 / max(estimate_runs, 1)
    if not math.isfinite(estimate_ms) or estimate_ms <= 0:
        return max_repeat

    repeat = int(target_ms / estimate_ms)
    return max(min_repeat, min(max_repeat, max(1, repeat)))


def interleaved_bench(
    fns: list[Callable[[], object]],
    *,
    repeat: int,
    desc: str | None = None,
    default_cudagraph: bool = False,
) -> list[float]:
    """
    Benchmark multiple functions at once, interleaving their executions to reduce
    the impact of external factors (e.g., load, temperature) on the
    measurements.

    Args:
        fns: List of functions to benchmark
        repeat: Number of times to repeat each benchmark
        desc: Optional description for progress bar
    """
    from triton import runtime

    # warmup
    for fn in fns:
        fn()
    clear_cache = functools.partial(
        runtime.driver.active.clear_cache,  # type: ignore[attr-defined]
        runtime.driver.active.get_empty_cache_for_benchmark(),  # type: ignore[attr-defined]
    )
    clear_cache()
    di = runtime.driver.active.get_device_interface()  # type: ignore[attr-defined]
    start_events = [
        [di.Event(enable_timing=True) for _ in range(repeat)] for _ in range(len(fns))
    ]
    end_events = [
        [di.Event(enable_timing=True) for _ in range(repeat)] for _ in range(len(fns))
    ]

    di.synchronize()
    benchmark_functions = [
        _maybe_cudagraph_replay(fn, default_enabled=default_cudagraph) for fn in fns
    ]

    # When a description is supplied we show a progress bar so the user can
    # track the repeated benchmarking loop.
    iterator = iter_with_progress(
        range(repeat),
        total=repeat,
        description=desc,
        enabled=desc is not None,
    )
    for i in iterator:
        for j in range(len(benchmark_functions)):
            clear_cache()
            start_events[j][i].record()
            benchmark_functions[j]()
            end_events[j][i].record()
    di.synchronize()

    return [
        statistics.median(
            [
                s.elapsed_time(e)
                for s, e in zip(start_events[j], end_events[j], strict=True)
            ]
        )
        for j in range(len(fns))
    ]


def interleaved_bench_generic(
    fns: list[Callable[[], object]],
    *,
    repeat: int,
    desc: str | None = None,
    default_cudagraph: bool = False,  # accepted for API symmetry; wall-clock timing doesn't use CG
) -> list[float]:
    """
    Benchmark multiple functions using wall-clock timing.
    Used for backends that don't have Triton's event-based timing (e.g., Pallas/TPU).
    """
    # warmup
    out: object = None
    for fn in fns:
        out = fn()
    synchronize_device(out)

    all_times: list[list[float]] = [[] for _ in range(len(fns))]

    iterator = iter_with_progress(
        range(repeat),
        total=repeat,
        description=desc,
        enabled=desc is not None,
    )
    for _i in iterator:
        for j in range(len(fns)):
            synchronize_device(out)
            start = time.perf_counter()
            out = fns[j]()
            synchronize_device(out)
            end = time.perf_counter()
            all_times[j].append((end - start) * 1000)  # convert to ms

    return [statistics.median(times) for times in all_times]


def paired_interleaved_bench(
    fns: list[Callable[..., object]],
    reference_fn: Callable[..., object],
    *,
    repeat: int,
    desc: str | None = None,
) -> list[tuple[float, float]]:
    """Paired-sample timing: each candidate is paired with ``reference_fn``.

    For every iteration and every candidate ``fns[j]``, the helper times
    one ``fns[j]()`` call immediately followed by one ``reference_fn()``
    call (both with per-call ``synchronize_device``).  The candidate and
    its paired reference time are accumulated together so that any
    chip-thermal or scheduling drift that affects one call also affects
    the other inside the same ~microsecond window.

    The returned ``(median_ms, paired_delta_median_ms)`` per fn lets the
    caller rank by paired delta instead of absolute median.  Ranking by
    paired delta cancels common-mode drift that accumulates across the
    ``repeat`` axis (the dominant noise source on noisy TPU pods, per
    plan.md §2.10 / Deep Replan 6); ranking by absolute median does not.

    Wall-clock based via ``time.perf_counter()`` + ``synchronize_device``
    so the same helper works for CUDA, Pallas / TPU, CuTe, and any other
    backend that exposes a device synchronization primitive.

    Args:
        fns: List of candidate functions to benchmark.
        reference_fn: Stable reference function (typically the cohort's
            current best) timed once per candidate per iteration so the
            paired deltas are one-to-one.
        repeat: Number of paired iterations per candidate.
        desc: Optional description for progress bar.

    Returns:
        A list of ``(median_ms, paired_delta_median_ms)`` tuples, one per
        candidate in ``fns``.  ``paired_delta_median_ms`` is positive when
        the candidate is slower than the reference and negative when
        faster.
    """
    # warmup
    out: object = None
    for fn in fns:
        out = fn()
    out = reference_fn()
    synchronize_device(out)

    fn_times: list[list[float]] = [[] for _ in range(len(fns))]
    ref_paired_times: list[list[float]] = [[] for _ in range(len(fns))]

    iterator = iter_with_progress(
        range(repeat),
        total=repeat,
        description=desc,
        enabled=desc is not None,
    )
    for _i in iterator:
        for j in range(len(fns)):
            synchronize_device(out)
            t0 = time.perf_counter()
            out = fns[j]()
            synchronize_device(out)
            t1 = time.perf_counter()
            out = reference_fn()
            synchronize_device(out)
            t2 = time.perf_counter()
            fn_times[j].append((t1 - t0) * 1000)
            ref_paired_times[j].append((t2 - t1) * 1000)

    results: list[tuple[float, float]] = []
    for j in range(len(fns)):
        candidate_median = statistics.median(fn_times[j])
        paired_deltas = [
            c - r for c, r in zip(fn_times[j], ref_paired_times[j], strict=True)
        ]
        paired_delta_median = statistics.median(paired_deltas)
        results.append((candidate_median, paired_delta_median))
    return results


def paired_device_us_bench(
    fns: list[Callable[..., object]],
    reference_fn: Callable[..., object],
    *,
    device_us_fn: Callable[[Callable[[], object]], float],
    passes: int = 1,
    desc: str | None = None,
) -> list[tuple[float, float]]:
    """Device-us paired-sample timing for autotuner final-pick re-rank.

    For every candidate ``fns[j]`` the helper invokes the
    ``device_us_fn`` ``passes`` times on the candidate and ``passes``
    times on ``reference_fn``, then returns ``(median_candidate_us,
    median_paired_delta_us)`` where each pass's paired delta is
    ``candidate_us - reference_us``.  Negative median delta means the
    candidate is faster on-device than the reference; positive means
    slower.  Common-mode noise inside ``device_us_fn`` (e.g.
    chip-thermal / DVFS variation that drifts across the autotune
    session) is suppressed in the per-pass paired delta because both
    calls run inside the same ``device_us_fn`` invocation window; the
    cross-pass median further suppresses any per-pass scheduler /
    cache-state variance that single-pass timing leaves on the table.

    Why this exists (plan.md §5 G7-autotune-device).  The autotuner's
    final-pick verification path historically ranked the top-K
    candidate cohort by single-call wall-clock us, which on small /
    medium Pallas matmuls is ~96-98% PJRT + ``pallas_call`` dispatch
    overhead.  Two configs whose chip work differs by 3-9 us
    register as the same ~125 us at the user-call level, and the
    autotuner ships the dispatch-cheap-but-device-expensive pick.
    Ranking by ``device_us`` collected via ``jax.profiler`` (the
    ``device_us_fn`` parameter on Pallas / TPU) gives a kernel-quality
    signal that isn't drowned by dispatch noise; the multi-pass
    median then suppresses single-window cross-call drift so the
    decision survives on close pairs (e.g. G7-prefetch's no-tiling
    seed vs neighboring tiled configs).

    Args:
      fns: List of candidate callables to time (``passes`` traces per
        call).
      reference_fn: Stable reference callable (typically the cohort's
        incoming best) timed alongside each candidate so the paired
        deltas are one-to-one per pass.
      device_us_fn: Backend-supplied helper that takes a zero-arg
        callable and returns its per-call on-device us under a
        many-call profiler trace (e.g. the Pallas backend wraps
        ``jax.profiler.start_trace`` over ``n_calls`` calls and
        averages the dominant device event).  Per-event sub-us stable
        at ~200 calls per trace; cross-trace drift on close pairs
        ~0.05-0.1 us, which the per-pass median below suppresses.
      passes: Number of paired-sample passes per candidate (default 1
        for the unit-test scaffold; production callers should
        explicitly request ``passes >= 2`` to tighten the
        cross-window noise floor — see
        ``_PALLAS_AUTOTUNE_DEVICE_US_DEFAULT_PASSES`` for the
        production default applied by the Pallas backend's
        ``make_pallas_paired_device_us_bench`` factory).  Taking the
        median across passes suppresses single-pass scheduler /
        cache-state variance; signal tightens by roughly
        ``sqrt(passes)`` for ``passes`` * per-trace overhead.
      desc: Optional description for the progress bar.

    Returns:
      A list of ``(median_device_us, paired_delta_device_us)`` tuples,
      one per candidate in ``fns``.  ``paired_delta_device_us`` is
      positive when the candidate is slower on-device than the
      reference, negative when faster.
    """
    iterator = iter_with_progress(
        range(len(fns)),
        total=len(fns),
        description=desc,
        enabled=desc is not None,
    )
    n_passes = max(1, passes)
    results: list[tuple[float, float]] = []
    for j in iterator:
        candidate_samples: list[float] = []
        delta_samples: list[float] = []
        for _pass in range(n_passes):
            candidate_us = device_us_fn(fns[j])
            reference_us = device_us_fn(reference_fn)
            candidate_samples.append(candidate_us)
            # Avoid ``inf - inf == nan`` poisoning the median when a
            # pass's trace data is unusable.  Keep a per-pass delta
            # only when both endpoints are finite; otherwise carry
            # ``inf`` so the per-axis filter below still yields the
            # "candidate failed this pass" signal.
            if math.isfinite(candidate_us) and math.isfinite(reference_us):
                delta_samples.append(candidate_us - reference_us)
            else:
                delta_samples.append(math.inf)
        finite_candidate_samples = [s for s in candidate_samples if math.isfinite(s)]
        finite_delta_samples = [s for s in delta_samples if math.isfinite(s)]
        if not finite_candidate_samples or not finite_delta_samples:
            # Every pass produced an unusable trace; mark the
            # candidate as failed so the caller deranks it.
            results.append((math.inf, math.inf))
            continue
        results.append(
            (
                statistics.median(finite_candidate_samples),
                statistics.median(finite_delta_samples),
            )
        )
    return results


# Default per-trace call count for the device-us autotune re-rank.  200
# calls matches the cycle-36 device_us harness in
# ``examples/pallas_perf/measure_headline.py``: empirically the per-event
# avg over 200 calls is sub-us stable so the avg == the median and the
# helper returns a single float per fn.  Lower n_calls (e.g. 50) drops
# the per-candidate wall cost from ~1-2 s to ~0.3-0.5 s but widens the
# noise band to ~0.5-1 us, which can flip ranks on close pairs;
# 200 is the sweet spot.
_PALLAS_AUTOTUNE_DEVICE_US_DEFAULT_N_CALLS = 200
_PALLAS_AUTOTUNE_DEVICE_US_DEFAULT_N_WARMUP = 5
# Multi-pass median per candidate suppresses cross-window scheduler /
# cache-state variance that single-pass traces leave on the table.  At
# 3 passes, the per-candidate cost is ~3-6 sec (3 trace windows ~1-2
# sec each, plus 3 paired reference traces), so per top-K cohort of
# 10 the device-us re-rank adds ~60-120 sec to autotune wall-time.
# Single-pass (passes=1) is faster but the rank can flip on close
# pairs whose paired delta is within the ~0.05-0.1us cross-window
# noise — for the headline shape the no-tiling-vs-tiled pair is
# routinely inside that band, so multi-pass is required to hold the
# G7-prefetch closure.
_PALLAS_AUTOTUNE_DEVICE_US_DEFAULT_PASSES = 3


def _autotune_rank_by_device_us() -> bool:
    """Return True iff ``HELION_AUTOTUNE_RANK_BY`` selects device-us.

    Default is ``device_us`` (on); set ``HELION_AUTOTUNE_RANK_BY=wall_us``
    to fall back to the legacy single-call wall-clock paired-sample
    ranking.  Any unknown value also falls back to wall-clock (defensive
    against typos / future renames).
    """
    value = os.environ.get("HELION_AUTOTUNE_RANK_BY", "device_us").strip().lower()
    return value == "device_us"


def _pallas_device_us_for_fn(
    fn: Callable[[], object],
    *,
    n_calls: int,
    n_warmup: int,
) -> float:
    """Per-call on-device us under a ``jax.profiler`` trace.

    Wraps ``n_calls`` invocations of ``fn`` in a single
    ``jax.profiler.start_trace`` / ``stop_trace`` window, parses the
    resulting ``.xplane.pb`` via ``jax.profiler.ProfileData.from_file``,
    finds the dominant compute event on the ``/device:TPU:0`` plane
    (largest total ``duration_ns`` across events whose count == ``n_calls``
    — the count filter excludes the DVFS ``P state`` counter line whose
    ~17 sampled events over the trace window would otherwise dominate
    the aggregation by ~45x), and returns ``total_ns / n_calls / 1000``.

    Mirrors ``examples/pallas_perf/measure_headline.py``'s
    ``_time_device_us`` (see plan.md §1 device-us block + §5 G7-prefetch
    cycle 36 history row) but lives here so the autotuner can call it
    without depending on the examples package.

    Returns ``+inf`` when ``jax`` is unavailable, when the trace
    produces no ``.xplane.pb`` (``jax.profiler`` silently dropped the
    trace), when the trace has no ``/device:TPU:0`` plane, or when
    no event on the plane has cardinality == ``n_calls`` (the
    per-call kernel emits a different number of events than
    expected).  All four cases are "the trace data is unusable, not
    the kernel is broken", so the caller's paired-delta median uses
    ``inf`` as the "candidate failed" sentinel.  Exceptions raised by
    the kernel itself (``fn``) propagate so a real kernel bug
    surfaces instead of being silently re-ranked.
    """
    try:
        import jax  # pyrefly: ignore[missing-module-attribute]
    except ImportError:
        return math.inf

    # Warmup OUTSIDE the trace window so first-call compile / cache
    # population doesn't pollute the dominant-event aggregation.
    # Exceptions from ``fn`` propagate (a kernel that doesn't even
    # warm up successfully isn't a benchmark target — caller should
    # see the real error).
    for _ in range(n_warmup):
        fn()

    with tempfile.TemporaryDirectory(prefix="helion_autotune_device_us_") as td:
        jax.profiler.start_trace(td)
        last_out: object = None
        try:
            for _ in range(n_calls):
                last_out = fn()
        finally:
            # Belt-and-suspenders: the per-call ``fn`` already runs a
            # device-sync (Pallas TPU launcher path), but a final
            # ``block_until_ready`` on the last output guarantees
            # ``stop_trace`` sees every device-side event before
            # finalising the .xplane.pb.
            if last_out is not None:
                with contextlib.suppress(TypeError, AttributeError):
                    jax.block_until_ready(last_out)
            jax.profiler.stop_trace()

        pb_path: str | None = None
        for root, _dirs, files in os.walk(td):
            for f in files:
                if f.endswith(".xplane.pb"):
                    pb_path = os.path.join(root, f)
                    break
            if pb_path is not None:
                break
        if pb_path is None:
            return math.inf

        pd = jax.profiler.ProfileData.from_file(pb_path)
        best_total_ns = 0
        for plane in pd.planes:
            if plane.name != "/device:TPU:0":
                continue
            for line in plane.lines:
                per_event_totals: dict[str, int] = {}
                per_event_counts: dict[str, int] = {}
                for ev in line.events:
                    per_event_totals[ev.name] = (
                        per_event_totals.get(ev.name, 0) + ev.duration_ns
                    )
                    per_event_counts[ev.name] = per_event_counts.get(ev.name, 0) + 1
                for name, total in per_event_totals.items():
                    if per_event_counts[name] != n_calls:
                        continue
                    if total > best_total_ns:
                        best_total_ns = total
        if best_total_ns == 0:
            return math.inf
        return best_total_ns / n_calls / 1000.0


def make_pallas_paired_device_us_bench(
    *,
    n_calls: int = _PALLAS_AUTOTUNE_DEVICE_US_DEFAULT_N_CALLS,
    n_warmup: int = _PALLAS_AUTOTUNE_DEVICE_US_DEFAULT_N_WARMUP,
    passes: int = _PALLAS_AUTOTUNE_DEVICE_US_DEFAULT_PASSES,
) -> Callable[..., list[tuple[float, float]]] | None:
    """Build a paired device-us bench callable for the Pallas backend.

    Returns ``None`` when the user opted out via
    ``HELION_AUTOTUNE_RANK_BY=wall_us``, in which case the autotuner
    keeps its legacy wall-clock paired-sample ranking.  Returns a
    closure with the
    :func:`paired_device_us_bench` signature otherwise; the closure
    captures ``n_calls`` / ``n_warmup`` / ``passes`` so the autotuner
    doesn't have to pass them at call time.

    Why this lives next to ``paired_device_us_bench``: the autotuner's
    ``_run_final_pick_verification_paired`` calls
    ``backend.get_paired_device_us_bench()`` once per final-pick phase
    and reuses the returned callable across the top-K candidates;
    constructing it here keeps the per-call import of ``jax`` and the
    env-var read off the autotune hot path.
    """
    if not _autotune_rank_by_device_us():
        return None

    def _bench(
        fns: list[Callable[..., object]],
        reference_fn: Callable[..., object],
        *,
        desc: str | None = None,
    ) -> list[tuple[float, float]]:
        def _device_us_fn(fn: Callable[[], object]) -> float:
            return _pallas_device_us_for_fn(fn, n_calls=n_calls, n_warmup=n_warmup)

        return paired_device_us_bench(
            fns,
            reference_fn,
            device_us_fn=_device_us_fn,
            passes=passes,
            desc=desc,
        )

    return _bench


def _summarize_statistics_fallback(
    times: list[float],
    quantiles: list[float] | None,
    return_mode: str,
) -> float | tuple[float, ...]:
    """Fallback statistics summarizer when triton.testing._summarize_statistics is unavailable."""
    if return_mode == "min":
        return min(times)
    if return_mode == "max":
        return max(times)
    if return_mode == "mean":
        return statistics.mean(times)
    if return_mode == "median":
        return statistics.median(times)
    # "all" mode
    if quantiles is not None:
        sorted_times = sorted(times)
        n = len(sorted_times)
        result = []
        for q in quantiles:
            idx = min(int(q * n), n - 1)
            result.append(sorted_times[idx])
        return tuple(result)
    return statistics.median(times)


# This function is copied from triton._testing.do_bench with modification
# to make sure different ranks run the benchmark for the same number
# of times.
def do_bench(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    grad_to_none: torch.Tensor | None = None,
    quantiles: list[float] | None = None,
    return_mode: str = "mean",
    process_group_name: str | None = None,
    *,
    default_cudagraph: bool = False,
) -> float | tuple[float, ...]:
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all". Default is "mean".
    :type return_mode: str
    """
    from triton import runtime
    from triton.testing import _summarize_statistics

    assert return_mode in ["min", "max", "mean", "median", "all"]

    di = runtime.driver.active.get_device_interface()  # pyrefly: ignore

    fn()
    di.synchronize()
    # Backward benchmarks mutate grad fields between iterations, so keep their
    # existing launch path.
    benchmark_function = (
        fn
        if grad_to_none is not None
        else _maybe_cudagraph_replay(fn, default_enabled=default_cudagraph)
    )

    cache = runtime.driver.active.get_empty_cache_for_benchmark()  # pyrefly: ignore

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        runtime.driver.active.clear_cache(cache)  # pyrefly: ignore
        benchmark_function()
    end_event.record()
    di.synchronize()
    estimate_ms = sync_object(
        start_event.elapsed_time(end_event) / 5, process_group_name=process_group_name
    )

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        benchmark_function()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        runtime.driver.active.clear_cache(cache)  # pyrefly: ignore
        # record time of `fn`
        start_event[i].record()
        benchmark_function()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event, strict=True)]
    return _summarize_statistics(times, quantiles, return_mode)  # pyrefly: ignore


def do_bench_generic(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    grad_to_none: torch.Tensor | None = None,
    quantiles: list[float] | None = None,
    return_mode: str = "mean",
    process_group_name: str | None = None,
    *,
    default_cudagraph: bool = False,  # accepted for API symmetry; wall-clock timing doesn't use CG
) -> float | tuple[float, ...]:
    """
    Benchmark using wall-clock timing for backends without Triton event timing.
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    out = fn()
    synchronize_device(out)

    # Estimate the runtime of the function
    synchronize_device(out)
    start = time.perf_counter()
    for _ in range(5):
        out = fn()
    synchronize_device(out)
    end = time.perf_counter()
    estimate_ms = sync_object(
        (end - start) * 1000 / 5, process_group_name=process_group_name
    )

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    times: list[float] = []
    for _i in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        synchronize_device(out)
        t0 = time.perf_counter()
        out = fn()
        synchronize_device(out)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # convert to ms
    return _summarize_statistics_fallback(times, quantiles, return_mode)
