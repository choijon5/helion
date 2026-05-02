"""
Multi-GPU AOT pretune orchestrator.

Runs `helion.experimental.aot_runner --phase all` for each kernel in
`tutorial_kernels.py` in parallel across unused GPUs (max 4 workers).

Each subprocess gets its own output dir to avoid races on the JSON cache.
After all jobs finish, JSONs and heuristic files are merged into a single
`.helion_aot/` directory that ships as the pretuned config database.

Usage:
    python pretune_runner.py                       # all kernels, max 4 GPUs
    python pretune_runner.py --kernels matmul softmax
    python pretune_runner.py --max-workers 2
    python pretune_runner.py --gpus 0,1,2,3        # force specific GPUs
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ALL_KERNELS = [
    "vector_add",
    "matmul",
    "softmax",
    "layer_norm",
    "attention",
    "grouped_gemm",
    "fp8_gemm",
]

# Sticky kernel → GPU mapping.  Same kernel always runs on the same GPU
# across runs so the Triton compile cache and L2 layout stay warm.
# GPUs 0, 1 are usually shared with other users; GPU 5 is broken right now.
# GPU 6 is dedicated to the long-running attention job.
# 4 GPUs total in use: 3, 4, 6, 7.
KERNEL_GPU_MAP: dict[str, int] = {
    "softmax": 3,        # longest run (~4h × 3)
    "layer_norm": 4,     # ~2h × 3, then fp8_gemm queues here
    "fp8_gemm": 4,
    "vector_add": 7,     # ~15min × 3, then matmul + grouped_gemm queue here
    "matmul": 7,
    "grouped_gemm": 7,
    "attention": 6,      # already running on 6, leave alone
}

SCRIPT_DIR = Path(__file__).resolve().parent
TUTORIAL_KERNELS = SCRIPT_DIR / "tutorial_kernels.py"
DEFAULT_OUTPUT_ROOT = Path.cwd() / ".helion_aot"


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def detect_unused_gpus(
    max_util: int = 5, max_mem_mib: int = 200
) -> list[int]:
    """Return GPU indices that have NO other compute apps and very low usage.

    Strict: skips any GPU with another user's process holding more than
    max_mem_mib of memory.  Returns indices in DESCENDING order so the
    orchestrator prefers later GPUs.
    """
    # First filter by util/memory
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpu_info: dict[int, tuple[str, int, int]] = {}  # idx -> (uuid, util, mem)
    for line in out.strip().splitlines():
        idx_s, uuid_s, util_s, mem_s = (p.strip() for p in line.split(","))
        gpu_info[int(idx_s)] = (uuid_s, int(util_s), int(mem_s))

    # Find which GPUs have any compute apps from OTHER users / processes
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    busy_uuids: set[str] = set()
    own_pids = {os.getpid(), os.getppid()}
    for line in apps.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        uuid_s = parts[0]
        try:
            mem = int(parts[1])
        except ValueError:
            continue
        if mem > max_mem_mib:
            busy_uuids.add(uuid_s)

    indices: list[int] = []
    for idx, (uuid, util, mem) in gpu_info.items():
        if util > max_util or mem > max_mem_mib:
            continue
        if uuid in busy_uuids:
            continue
        indices.append(idx)
    indices.sort(reverse=True)
    return indices


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Job:
    kernel: str
    output_dir: Path
    log_path: Path

    def cmd(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "helion.experimental.aot_runner",
            "--phase",
            "all",
            "--kernel",
            self.kernel,
            "--output-dir",
            str(self.output_dir),
            "--",
            sys.executable,
            str(TUTORIAL_KERNELS),
            "--kernel",
            self.kernel,
        ]


def run_job(job: Job, gpu: int) -> tuple[Job, int, float]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HELION_AOT_KERNELS"] = job.kernel

    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] [GPU {gpu}] start "
        f"{job.kernel} -> {job.log_path}"
    )
    with open(job.log_path, "w") as f:
        proc = subprocess.run(  # noqa: S603 — controlled args
            job.cmd(),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] [GPU {gpu}] {status} "
        f"{job.kernel} in {elapsed:.0f}s"
    )
    return job, proc.returncode, elapsed


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


class GpuWorkerPool:
    """Schedules jobs across a fixed set of GPUs.  Each GPU runs one job at a time."""

    def __init__(self, gpus: list[int]):
        self.gpus = gpus
        # gpu_index -> (subprocess.Popen, Job, start_time, log_file)
        self.running: dict[int, tuple[subprocess.Popen, Job, float, object]] = {}
        self.results: list[tuple[Job, int, float, int]] = []

    def has_capacity(self) -> bool:
        return len(self.running) < len(self.gpus)

    def free_gpu(self, preferred: int | None = None) -> int | None:
        if preferred is not None and preferred in self.gpus and preferred not in self.running:
            return preferred
        for g in self.gpus:
            if g not in self.running:
                return g
        return None

    def launch(self, job: Job) -> None:
        gpu = self.free_gpu(KERNEL_GPU_MAP.get(job.kernel))
        assert gpu is not None
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["HELION_AOT_KERNELS"] = job.kernel
        # Force subprocesses to import helion from this checkout, not whatever
        # editable install (e.g. /home/$USER/helion) is on the system path.
        repo_root = str(Path(__file__).resolve().parents[2])
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{repo_root}:{existing}" if existing else repo_root
        )

        job.output_dir.mkdir(parents=True, exist_ok=True)
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(job.log_path, "w")
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] [GPU {gpu}] start "
            f"{job.kernel} -> {job.log_path}",
            flush=True,
        )
        proc = subprocess.Popen(  # noqa: S603
            job.cmd(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        self.running[gpu] = (proc, job, time.time(), log_file)

    def reap(self, block: bool = False) -> int:
        """Reap any finished job(s); returns number reaped.

        With block=True: wait until at least one job finishes, then return.
        With block=False: do one non-blocking pass and return.
        """
        reaped = 0
        while True:
            done = [
                (gpu, proc, job, t0, log_file)
                for gpu, (proc, job, t0, log_file) in self.running.items()
                if proc.poll() is not None
            ]
            for gpu, proc, job, t0, log_file in done:
                log_file.close()
                elapsed = time.time() - t0
                rc = proc.returncode
                status = "OK" if rc == 0 else f"FAIL({rc})"
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"[GPU {gpu}] {status} {job.kernel} in {elapsed:.0f}s",
                    flush=True,
                )
                self.results.append((job, rc, elapsed, gpu))
                del self.running[gpu]
                reaped += 1
            # Return as soon as we have anything to reap (block=True), or
            # immediately in non-blocking mode, or if nothing is running.
            if reaped > 0 or not block or not self.running:
                return reaped
            time.sleep(2.0)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_outputs(per_kernel_dirs: list[Path], merged_dir: Path) -> None:
    """Combine per-kernel JSONs and copy heuristic files to a single root dir."""
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_configs: dict[str, dict[str, list]] = {}  # hw_id -> kernel -> list

    for d in per_kernel_dirs:
        if not d.exists():
            continue
        for json_path in d.glob("tuned_configs_*.json"):
            hw_id = json_path.stem.removeprefix("tuned_configs_")
            data = json.loads(json_path.read_text())
            merged_configs.setdefault(hw_id, {})
            for kernel, configs in data.items():
                merged_configs[hw_id].setdefault(kernel, []).extend(configs)

        # Copy CSV measurements (append per-row)
        for csv_path in d.glob("measurements_*.csv"):
            target = merged_dir / csv_path.name
            with csv_path.open() as src, target.open("a") as dst:
                if target.stat().st_size == 0:
                    dst.write(src.read())
                else:
                    # skip header on subsequent files
                    next(src, None)
                    dst.write(src.read())

        # Copy generated heuristic files
        for heur_path in d.glob("heuristic_*.py"):
            shutil.copy2(heur_path, merged_dir / heur_path.name)

        # Copy standalone files
        for sa_path in d.glob("*_standalone.py"):
            shutil.copy2(sa_path, merged_dir / sa_path.name)

    for hw_id, kernels in merged_configs.items():
        out_path = merged_dir / f"tuned_configs_{hw_id}.json"
        out_path.write_text(json.dumps(kernels, indent=2))
        n_total = sum(len(v) for v in kernels.values())
        print(
            f"[merge] {out_path}: {len(kernels)} kernels, {n_total} tuned configs"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="AOT pretune orchestrator")
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=ALL_KERNELS,
        help="Kernels to tune (default: all 6 tutorial kernels)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="Comma-separated GPU indices to use (default: auto-detect unused)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max number of GPUs to use in parallel (default: 4)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory (default: .helion_aot)",
    )
    args = parser.parse_args()

    if args.gpus:
        gpus = [int(g) for g in args.gpus.split(",")]
    else:
        gpus = detect_unused_gpus()
        if not gpus:
            print("No unused GPUs found.", file=sys.stderr)
            return 2

    gpus = gpus[: args.max_workers]
    print(f"Using GPUs: {gpus} ({len(gpus)} workers)")
    print(f"Kernels: {args.kernels}")
    print(f"Output root: {args.output_root}")

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"

    # Build job queue: each kernel writes to its own subdir
    queue: list[Job] = [
        Job(
            kernel=k,
            output_dir=output_root / f"job_{k}",
            log_path=log_dir / f"{k}.log",
        )
        for k in args.kernels
    ]

    pool = GpuWorkerPool(gpus)

    t0 = time.time()
    pending = list(queue)
    while pending or pool.running:
        while pending and pool.has_capacity():
            pool.launch(pending.pop(0))
        # Block for at least one job to finish if at capacity, else loop
        block = (not pool.has_capacity()) or (not pending and pool.running)
        pool.reap(block=block)

    total_elapsed = time.time() - t0
    print(f"\nAll jobs finished in {total_elapsed:.0f}s")

    # Summary
    n_ok = sum(1 for _, rc, _, _ in pool.results if rc == 0)
    n_fail = len(pool.results) - n_ok
    print(f"Results: {n_ok} OK, {n_fail} FAIL")
    for job, rc, elapsed, gpu in pool.results:
        print(
            f"  {job.kernel:>14s} (GPU {gpu}): "
            f"{'OK' if rc == 0 else f'FAIL({rc})'} in {elapsed:.0f}s"
        )

    # Merge
    print("\nMerging outputs...")
    merge_outputs(
        [j.output_dir for j in queue],
        merged_dir=output_root,
    )

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
