#!/bin/bash
# Sync the local helion tree to the torchtpu pod and run a command inside
# the pre-built helion-venv. Uses kubectl exec against the per-user pod.
#
# Sync modes (chosen automatically per call):
#   - First call ever (or POD_FULL_SYNC=1, or marker missing): full tar
#     sync of the whole repo (~8.5 MB; ~22 s on top of kubectl overhead).
#   - Subsequent calls: incremental — only files modified since the
#     previous successful sync (mtime > marker), tracked via a marker
#     file under ~/.cache/run-on-pod/. Typically <100 KB, few seconds.
#   - POD_SKIP_SYNC=1: skip sync entirely (use last-known pod state).
#
# The sync stream and the in-pod command share a single kubectl exec
# call: `tar -cf - ... | kubectl exec -i -- bash 'cd repo && tar -xf - &&
# source venv && <cmd>'`. tar consumes stdin until EOF, then bash runs
# the command — saves ~4 s vs two separate kubectl exec calls.
#
# Note: file deletions on the devserver are NOT propagated (pod will
# accumulate stale files). Force a clean re-sync with POD_FULL_SYNC=1
# when this matters (or `kubectl exec ... rm -rf /mnt/hyperdisk/helion_2`
# first; the next call will full-sync).
#
# Usage: scripts/run-on-pod.sh [ENV_VAR=val ...] command [args ...]
#
# Examples:
#   scripts/run-on-pod.sh python -c 'import jax; print(jax.devices())'
#   scripts/run-on-pod.sh HELION_BACKEND=pallas TPU_VISIBLE_CHIPS=3 \
#       pytest test/test_pallas.py -x -vv
#   POD_FULL_SYNC=1 scripts/run-on-pod.sh ls /mnt/hyperdisk/helion_2
#
# Environment variables:
#   KUBECONFIG          kubeconfig file (default: ~/.kube/torusconfig)
#   TORCHTPU_POD        pod name (default: jongsokchoi-torchtpu)
#   TORCHTPU_NAMESPACE  pod namespace (default: default)
#   TORCHTPU_REPO       pod-side repo path (default: /mnt/hyperdisk/helion_2)
#   TORCHTPU_VENV       pod-side venv activate path
#                       (default: /mnt/hyperdisk/helion-venv/bin/activate)
#   POD_FULL_SYNC=1     force full tar sync this call
#   POD_SKIP_SYNC=1     skip sync entirely this call

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

MARKER_DIR="${HOME}/.cache/run-on-pod"
MARKER="${MARKER_DIR}/${POD}.${NAMESPACE}.synced"
mkdir -p "${MARKER_DIR}"

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

quoted_cmd=""
for arg in "${cmd_args[@]}"; do
    quoted_cmd="${quoted_cmd} $(printf '%q' "$arg")"
done

# Decide sync mode
if [[ "${POD_SKIP_SYNC:-0}" == "1" ]]; then
    sync_mode="none"
elif [[ "${POD_FULL_SYNC:-0}" == "1" ]] || [[ ! -f "${MARKER}" ]]; then
    sync_mode="full"
else
    sync_mode="incremental"
fi

TAR_EXCLUDES=(
    --exclude='.git'
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='*.pyo'
    --exclude='.pytest_cache'
    --exclude='.ruff_cache'
    --exclude='dist'
    --exclude='site'
    --exclude='*.egg-info'
    --exclude='.venv'
    --exclude='.logs'
    --exclude='docs'
    --exclude='benchmarks'
)

# Stage a new marker file (mtime captured at script start) so that any
# file modified during this call will still be picked up by the next
# call's incremental scan. Promote on success.
new_marker="$(mktemp "${MARKER_DIR}/.in-flight.XXXXXX")"
promoted=0
cleanup() {
    if [[ "${promoted}" != "1" ]]; then
        rm -f "${new_marker}"
    fi
}
trap cleanup EXIT

# Commands run inside the pod after sync. tar -xf - consumes stdin
# until EOF, then bash continues to source the venv and run the cmd.
RUN_CMD="cd '${POD_PATH}' && source '${VENV_ACTIVATE}' && ${env_exports} ${quoted_cmd}"
SYNC_RUN_CMD="cd '${POD_PATH}' && tar -xf - && source '${VENV_ACTIVATE}' && ${env_exports} ${quoted_cmd}"

case "${sync_mode}" in
    none)
        echo "run-on-pod: skipping sync (POD_SKIP_SYNC=1)" >&2
        KUBECONFIG="${KUBECONFIG_PATH}" kubectl exec -n "${NAMESPACE}" "${POD}" -- \
            /bin/bash -lc "${RUN_CMD}"
        ;;
    full)
        echo "run-on-pod: full sync (first call, marker missing, or POD_FULL_SYNC=1)" >&2
        tar -C "${REPO_ROOT}" "${TAR_EXCLUDES[@]}" -cf - . | \
            KUBECONFIG="${KUBECONFIG_PATH}" kubectl exec -i -n "${NAMESPACE}" "${POD}" -- \
            /bin/bash -lc "${SYNC_RUN_CMD}"
        ;;
    incremental)
        mapfile -t changed_files < <(
            cd "${REPO_ROOT}" && find . -type f -newer "${MARKER}" \
                -not -path './.git/*' \
                -not -path '*/__pycache__/*' \
                -not -name '*.pyc' -not -name '*.pyo' \
                -not -path './.pytest_cache/*' \
                -not -path './.ruff_cache/*' \
                -not -path './dist/*' \
                -not -path './site/*' \
                -not -path '*.egg-info/*' \
                -not -name '*.egg-info' \
                -not -path './.venv/*' \
                -not -path './.logs/*' \
                -not -path './docs/*' \
                -not -path './benchmarks/*' \
                -printf '%P\n' || true
        )
        if [[ ${#changed_files[@]} -eq 0 ]]; then
            echo "run-on-pod: incremental sync (0 files changed, skipping tar)" >&2
            KUBECONFIG="${KUBECONFIG_PATH}" kubectl exec -n "${NAMESPACE}" "${POD}" -- \
                /bin/bash -lc "${RUN_CMD}"
        else
            echo "run-on-pod: incremental sync (${#changed_files[@]} files)" >&2
            tar -C "${REPO_ROOT}" -cf - "${changed_files[@]}" | \
                KUBECONFIG="${KUBECONFIG_PATH}" kubectl exec -i -n "${NAMESPACE}" "${POD}" -- \
                /bin/bash -lc "${SYNC_RUN_CMD}"
        fi
        ;;
esac

# Promote marker on success
mv "${new_marker}" "${MARKER}"
promoted=1
