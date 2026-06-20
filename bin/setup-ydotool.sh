#!/usr/bin/env bash
# Grant the user access to /dev/uinput and run ydotoold as a user service so
# ydotool can inject the paste shortcut on Wayland.
set -euo pipefail

UDEV_RULE="/etc/udev/rules.d/60-ydotool-uinput.rules"
SERVICE_DIR="${HOME}/.config/systemd/user"
SERVICE_FILE="${SERVICE_DIR}/ydotoold.service"
SOCKET="%t/.ydotool_socket"

if ! command -v ydotool >/dev/null 2>&1; then
  echo "ydotool nicht installiert - ueberspringe Setup."
  exit 0
fi

# ── 1. udev rule: make /dev/uinput group-writable for 'input' ───────────────
need_rule=1
if [[ -f "${UDEV_RULE}" ]]; then
  need_rule=0
fi
if [[ "${need_rule}" -eq 1 ]]; then
  echo "Richte udev-Regel fuer /dev/uinput ein (einmalig sudo noetig)..."
  printf 'KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"\n' \
    | sudo tee "${UDEV_RULE}" >/dev/null
  sudo udevadm control --reload-rules
  sudo udevadm trigger /dev/uinput || true
fi

if ! id -nG | tr ' ' '\n' | grep -qx input; then
  echo "WARNUNG: Dein User ist nicht in der Gruppe 'input'."
  echo "  Bitte ausfuehren:  sudo usermod -aG input \"$USER\"  (danach neu anmelden)."
fi

# ── 2. ydotoold as a systemd --user service ─────────────────────────────────
YDOTOOLD_BIN="$(command -v ydotoold)"
mkdir -p "${SERVICE_DIR}"
cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=ydotoold virtual input daemon

[Service]
ExecStart=${YDOTOOLD_BIN} --socket-path=${SOCKET} --socket-perm=0600
Restart=always
RestartSec=2

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now ydotoold.service || \
  echo "WARNUNG: konnte ydotoold-Service nicht starten (evtl. erst nach Neuanmeldung)."

echo "ydotool-Setup abgeschlossen."
echo "Socket: \${XDG_RUNTIME_DIR}/.ydotool_socket"
