#!/usr/bin/env bash
# Play defend_the_line and health_gathering_supreme with a local Intern-Decision checkpoint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON_BIN:-$PWD/.venv-doom/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON. Create it from the ROCm (or CUDA) interpreter, then install ViZDoom and Transformers 5.14.1." >&2
  exit 1
fi
exec "$PYTHON" -m src.doom.play "$@"
