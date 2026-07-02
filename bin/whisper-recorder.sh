#!/usr/bin/env bash
# Thin wrapper so the GUI can drive the long-form recorder with the project venv
# python (faster-whisper / openvino-genai live there, not in the system python).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PY="${ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || PY="python3"

exec "${PY}" "${ROOT}/dictation/recorder.py" "$@"
