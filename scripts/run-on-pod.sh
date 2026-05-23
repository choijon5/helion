#!/bin/bash
# Sync the local helion tree to the torchtpu pod and run a command inside
# the pre-built helion-venv. Mirrors scripts/run-on-tpu.sh, but uses
# kubectl exec against the per-user pod rather than ssh + uv wheels.
#
# Usage: scripts/run-on-pod.sh [ENV_VAR=val ...] command [args ...]
#
# Examples:
#   scripts/run-on-pod.sh python -c 'import jax; print(jax.devices())'
#   scripts/run-on-pod.sh TPU_VISIBLE_CHIPS=3 pytest test/test_pallas.py -x -vv
#   scripts/run-on-pod.sh 'cd examples/pallas_perf && bash benchmark.sh matmul_helion.py'
#
# Environment variables:
#   KUBECONFIG           kubeconfig file (default: ~/.kube/torusconfig)
#   TORCHTPU_POD         pod name (default: jongsokchoi-torchtpu)
#   TORCHTPU_NAMESPACE   pod namespace (default: default)
#   TORCHTPU_REPO        pod-side repo path (default: /mnt/hyperdisk/helion_2)
#   TORCHTPU_VENV        pod-side venv activate path
#                        (default: /mnt/hyperdisk/helion-venv/bin/activate)
#   POD_SKIP_SYNC=1      skip the tar sync (use last-known pod state)
#
# The script always syncs the devserver tree to the pod before running
# the command unless POD_SKIP_SYNC=1. Sync excludes .git, caches, and
# build artifacts.

set -euo pipefail

KUBECONFIG_PATH="${KUBECONFIG:-${HOME}/.kube/torusconfig}"
POD="${TORCHTPU_POD:-jongsokchoi-torchtpu}"
NAMESPACE="${TORCHTPU_NAMESPACE:-default}"
POD_PATH="${TORCHTPU_REPO:-/mnt/hyperdisk/helion_2}"
VENV_ACTIVATE="${TORCHTPU_VENV:-/mnt/hyperdisk/helion-venv/bin/activate}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
    echo "error: cannot find pyproject.toml in ${REPO_ROOT}" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [ENV_VAR=val ...] command [args ...]" >&2
    exit 1
fi

# Split leading ENV=val arguments from the command
env_exports=""
cmd_args=()
for arg in "$@"; do
    if [[ "$arg" =~ ^[A-Z_][A-Z_0-9]*=.* ]] && [[ ${#cmd_args[@]} -eq 0 ]]; then
        env_exports="${env_exports}export $(printf '%q' "$arg"); "
    else
        cmd_args+=("$arg")
    fi
done

if [[ ${#cmd_args[@]} -eq 0 ]]; then
    echo "error: no command specified" >&2
    exit 1
fi

# Quote each command argument for safe transport
quoted_cmd=""
for arg in "${cmd_args[@]}"; do
    quoted_cmd="${quoted_cmd} $(printf '%q' "$arg")"
done

# Step 1: sync devserver tree to pod (unless skipped)
if [[ "${POD_SKIP_SYNC:-0}" != "1" ]]; then
    # Exclusions: VCS metadata, caches, build artifacts, and trees that
    # never get exercised by Pallas TPU runs (docs/, benchmarks/ which is
    # CUDA-only). examples/ is included so examples/pallas_perf/ syncs.
    tar -C "${REPO_ROOT}" \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.pyo' \
        --exclude='.pytest_cache' \
        --exclude='.ruff_cache' \
        --exclude='dist' \
        --exclude='site' \
        --exclude='*.egg-info' \
        --exclude='.venv' \
        --exclude='.logs' \
        --exclude='docs' \
        --exclude='benchmarks' \
        -cf - . | \
    KUBECONFIG="${KUBECONFIG_PATH}" kubectl exec -i -n "${NAMESPACE}" "${POD}" -- \
        /bin/bash -c "mkdir -p '${POD_PATH}' && cd '${POD_PATH}' && tar -xf -" >&2
fi

# Step 2: run command inside pod, in repo dir, with venv activated
KUBECONFIG="${KUBECONFIG_PATH}" kubectl exec -n "${NAMESPACE}" "${POD}" -- \
    /bin/bash -lc "
        cd '${POD_PATH}' &&
        source '${VENV_ACTIVATE}' &&
        ${env_exports}
        ${quoted_cmd}
    "
