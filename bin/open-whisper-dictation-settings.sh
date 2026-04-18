#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

if [[ "$(uname)" == "Darwin" ]]; then
  exec "${ROOT}/.venv/bin/python" "${ROOT}/gui/settings_macos.py"
else
  exec "${ROOT}/.venv/bin/python" "${ROOT}/gui/settings.py"
fi
