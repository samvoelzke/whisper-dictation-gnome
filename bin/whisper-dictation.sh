#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
LOG_DIR="${HOME}/.cache/whisper-dictation"
PID_FILE="${LOG_DIR}/daemon.pid"
# Match only the real daemon (venv python + full script path) — never an
# editor/pager that merely has "daemon.py" somewhere on its command line.
DAEMON_PATTERN="${ROOT}/.venv/bin/python -u ${ROOT}/dictation/daemon.py"
UNIT="whisper-dictation.service"

mkdir -p "${LOG_DIR}"

# ydotool (Wayland paste) talks to ydotoold over this socket.
export YDOTOOL_SOCKET="${YDOTOOL_SOCKET:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/.ydotool_socket}"

# Prefer the systemd --user service when it is installed; otherwise fall back
# to a plain setsid background process managed via a PID file.
have_unit() {
  command -v systemctl >/dev/null 2>&1 && systemctl --user cat "${UNIT}" >/dev/null 2>&1
}

manual_stop() {
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

manual_start() {
  setsid "${ROOT}/.venv/bin/python" -u "${ROOT}/dictation/daemon.py" >>"${LOG_DIR}/daemon.log" 2>&1 </dev/null &
  echo "$!" > "${PID_FILE}"
  sleep 1
  if ! pgrep -f "${DAEMON_PATTERN}" >/dev/null 2>&1; then
    echo "Daemon failed to start. Check ${LOG_DIR}/daemon.log" >&2
    rm -f "${PID_FILE}"
    return 1
  fi
}

case "${1:-}" in
  --restart)
    if have_unit; then systemctl --user restart "${UNIT}"; else manual_stop; manual_start; fi
    ;;
  --stop)
    if have_unit; then systemctl --user stop "${UNIT}"; else manual_stop; fi
    ;;
  --reload)
    # Apply config changes without reloading the model (unless model/device changed).
    if have_unit; then systemctl --user reload "${UNIT}"; else pkill -HUP -f "${DAEMON_PATTERN}" 2>/dev/null || true; fi
    ;;
  --toggle)
    # Start/stop recording (bindable to a GNOME custom shortcut).
    if have_unit; then systemctl --user kill -s SIGUSR1 "${UNIT}"; else pkill -USR1 -f "${DAEMON_PATTERN}" 2>/dev/null || true; fi
    ;;
  --status)
    if have_unit; then
      [[ "$(systemctl --user is-active "${UNIT}")" == "active" ]] && echo running || echo stopped
    elif pgrep -f "${DAEMON_PATTERN}" >/dev/null 2>&1; then
      echo running
    else
      echo stopped
    fi
    ;;
  *)
    exec "${ROOT}/.venv/bin/python" -u "${ROOT}/dictation/daemon.py"
    ;;
esac
