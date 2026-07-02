#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
LOG_DIR="${HOME}/.cache/whisper-dictation"
PLIST="${HOME}/Library/LaunchAgents/io.whisper-dictation.daemon.plist"
# Match only a real daemon (python executing the script), never an editor
# that has "daemon.py" on its command line (pkill -9 would kill it).
DAEMON_PATTERN='python[^ ]* .*dictation/daemon\.py'

mkdir -p "${LOG_DIR}"

start_daemon() {
  launchctl load "${PLIST}" 2>/dev/null || true
  sleep 2
  if pgrep -f "${DAEMON_PATTERN}" >/dev/null 2>&1; then
    echo "Daemon gestartet."
  else
    echo "Daemon konnte nicht starten. Prüfe: tail -f ${LOG_DIR}/daemon.log" >&2
    exit 1
  fi
}

stop_daemon() {
  launchctl unload "${PLIST}" 2>/dev/null || true
  pkill -f "${DAEMON_PATTERN}" 2>/dev/null || true
  for _ in {1..20}; do
    pgrep -f "${DAEMON_PATTERN}" >/dev/null 2>&1 || { echo "Daemon gestoppt."; return; }
    sleep 0.2
  done
  pkill -9 -f "${DAEMON_PATTERN}" 2>/dev/null || true
}

case "${1:-}" in
  --start)   start_daemon ;;
  --stop)    stop_daemon ;;
  --restart) stop_daemon; start_daemon ;;
  --status)
    if pgrep -f "${DAEMON_PATTERN}" >/dev/null 2>&1; then
      echo "running"
    else
      echo "stopped"
    fi
    ;;
  --log)
    tail -f "${LOG_DIR}/daemon.log"
    ;;
  *)
    echo "Usage: $0 --start | --stop | --restart | --status | --log"
    ;;
esac
