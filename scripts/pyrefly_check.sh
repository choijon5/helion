#!/bin/bash
# Wrapper for pyrefly check.
# On macOS, triton/cutlass are not installable, so we ignore their imports.
EXTRA=""
if [ "$(uname -s)" = "Darwin" ]; then
  for mod in triton "triton.*" cutlass "cutlass.*"; do
    EXTRA="$EXTRA --ignore-missing-imports $mod"
  done
fi

# Pyrefly's default interpreter discovery prefers a project-local .venv and
# falls back to system python; it does not honor an activated conda env via
# CONDA_PREFIX. Pin to whatever `python` is first on PATH so conda activate
# and uv .venv both work.
PYTHON_BIN="$(command -v python || command -v python3)"
if [ -n "$PYTHON_BIN" ]; then
  EXTRA="$EXTRA --python-interpreter-path $PYTHON_BIN"
fi

exec pyrefly check $EXTRA
