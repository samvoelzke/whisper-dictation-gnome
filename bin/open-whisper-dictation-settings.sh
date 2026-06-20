#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

if [[ "$(uname)" == "Darwin" ]]; then
  exec "${ROOT}/.venv/bin/python" "${ROOT}/gui/settings_macos.py"
else
  # GTK4 GUI needs PyGObject (gi), a system package not present in the venv.
  exec python3 "${ROOT}/gui/settings.py"
fi
