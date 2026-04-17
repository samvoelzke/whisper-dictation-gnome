#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
LOG_DIR="${HOME}/.cache/whisper-dictation"
PID_FILE="${LOG_DIR}/daemon.pid"
DAEMON_PATTERN='dictation/daemon.py'

mkdir -p "${LOG_DIR}"

stop_daemon() {
  if [[ -f "${PID_FILE}" ]]; then
    kill "$(cat "${PID_FILE}")" 2>/dev/null || true
    rm -f "${PID_FILE}"
  fi
  pkill -f "${DAEMON_PATTERN}" 2>/dev/null || true
  for _ in {1..20}; do
    if ! pgrep -f "${DAEMON_PATTERN}" >/dev/null 2>&1; then
      return
    fi
    sleep 0.2
  done
  pkill -9 -f "${DAEMON_PATTERN}" 2>/dev/null || true
}

start_daemon() {
  setsid "${ROOT}/.venv/bin/python" -u "${ROOT}/dictation/daemon.py" >>"${LOG_DIR}/daemon.log" 2>&1 </dev/null &
  local launcher_pid=$!
  echo "${launcher_pid}" > "${PID_FILE}"
  sleep 1
  if ! pgrep -f "${DAEMON_PATTERN}" >/dev/null 2>&1; then
    echo "Daemon failed to start. Check ${LOG_DIR}/daemon.log" >&2
    rm -f "${PID_FILE}"
    return 1
  fi
}

case "${1:-}" in
  --restart)
    stop_daemon
    start_daemon
    ;;
  --status)
    if pgrep -f "${DAEMON_PATTERN}" >/dev/null 2>&1; then
      echo running
    else
      echo stopped
    fi
    ;;
  --stop)
    stop_daemon
    ;;
  *)
    exec "${ROOT}/.venv/bin/python" -u "${ROOT}/dictation/daemon.py"
    ;;
esac
