#!/bin/bash
# Build the Helion Pallas locked-path C extension
# (helion/_helion_c_launcher.c) against the current Python.  Output is
# placed alongside the source under ``helion/`` as
# ``_helion_c_launcher.cpython-<ver>-<plat>.so`` so the standard
# import path picks it up.
#
# G6-launcher-C (plan.md §5 G6): the extension wraps the
# ``_DirectCallKernel`` locked hot path in C to eliminate the
# ~11 us per-frame CPython overhead.  Falls back to the Python
# closure when the extension is unavailable, so this build step is
# optional (skip on environments without a compiler).
#
# Pod execution model: build on the devserver (or any host with a C
# compiler matching the pod's Python version), and let
# ``scripts/run-on-pod.sh`` tar-sync the resulting ``.so`` to the pod.
# The pod itself has no compiler installed.
#
# Re-run after editing ``helion/_helion_c_launcher.c``.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${REPO_ROOT}/helion/_helion_c_launcher.c"

if [[ ! -f "${SRC}" ]]; then
    echo "error: source not found: ${SRC}" >&2
    exit 1
fi

PYTHON="${PYTHON:-python}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "error: python interpreter '${PYTHON}' not found on PATH" >&2
    exit 1
fi

PY_INCLUDE=$("${PYTHON}" -c "import sysconfig; print(sysconfig.get_paths()['include'])")
EXT_SUFFIX=$("${PYTHON}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT="${REPO_ROOT}/helion/_helion_c_launcher${EXT_SUFFIX}"

CC="${CC:-gcc}"
if ! command -v "${CC}" >/dev/null 2>&1; then
    echo "error: C compiler '${CC}' not found on PATH (skip C extension build)" >&2
    exit 1
fi

echo "build_c_launcher: ${CC} -> ${OUT}" >&2
"${CC}" \
    -O3 \
    -fPIC \
    -shared \
    -Wall \
    -Wno-unused-parameter \
    -I"${PY_INCLUDE}" \
    "${SRC}" \
    -o "${OUT}"

echo "build_c_launcher: built ${OUT}" >&2
