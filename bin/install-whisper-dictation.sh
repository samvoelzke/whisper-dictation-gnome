#!/usr/bin/env bash
# Installer for Whisper Dictation on GNOME / Wayland (Fedora & friends).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
AUTOSTART_DIR="${HOME}/.config/autostart"
APPLICATIONS_DIR="${HOME}/.local/share/applications"
CONFIG_DIR="${HOME}/.config/whisper-dictation"
CONFIG_FILE="${CONFIG_DIR}/config.json"
DESKTOP_FILE="${AUTOSTART_DIR}/whisper-dictation.desktop"
SETTINGS_DESKTOP_FILE="${APPLICATIONS_DIR}/whisper-dictation-settings.desktop"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
SERVICE_FILE="${SYSTEMD_USER_DIR}/whisper-dictation.service"
VENV="${ROOT}/.venv"

mkdir -p "${AUTOSTART_DIR}" "${APPLICATIONS_DIR}" "${CONFIG_DIR}" "${SYSTEMD_USER_DIR}" "${HOME}/.cache/whisper-dictation"

# ── System dependencies (best effort hint) ─────────────────────────────────
missing=()
for tool in arecord wl-copy ydotool notify-send; do
  command -v "${tool}" >/dev/null 2>&1 || missing+=("${tool}")
done
if ((${#missing[@]})); then
  echo "WARNUNG: Folgende Tools fehlen: ${missing[*]}"
  echo "  Auf Fedora:  sudo dnf install alsa-utils wl-clipboard ydotool libnotify"
  echo "  (Installation laeuft trotzdem weiter.)"
fi

# ── Python venv (rebuild if it points at a foreign interpreter, e.g. macOS) ──
venv_ok=0
if [[ -x "${VENV}/bin/python" ]] && "${VENV}/bin/python" -c "" >/dev/null 2>&1; then
  venv_ok=1
fi
if [[ "${venv_ok}" -ne 1 ]]; then
  echo "Erstelle frisches Python-venv unter ${VENV} ..."
  rm -rf "${VENV}"
  python3 -m venv "${VENV}"
  "${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
fi

# ── Python dependencies ─────────────────────────────────────────────────────
# evdev (hotkey) + numpy; faster-whisper (CPU fallback); openvino-genai for
# Intel GPU/NPU acceleration. None of these pull torch -> the venv stays lean.
if ! "${VENV}/bin/python" -c "import faster_whisper, evdev, numpy" >/dev/null 2>&1; then
  echo "Installiere faster-whisper, evdev, numpy ..."
  "${VENV}/bin/python" -m pip install faster-whisper evdev numpy
fi
if ! "${VENV}/bin/python" -c "import openvino_genai, huggingface_hub" >/dev/null 2>&1; then
  echo "Installiere openvino-genai (Intel GPU/NPU) ..."
  "${VENV}/bin/python" -m pip install openvino openvino-genai huggingface_hub
fi

# Report which OpenVINO devices are available (GPU = Intel Arc; NPU needs the
# Intel NPU userspace runtime, see README-GNOME.md).
"${VENV}/bin/python" - <<'PY' 2>/dev/null || true
try:
    import openvino as ov
    print("OpenVINO-Devices:", ", ".join(ov.Core().available_devices))
except Exception as e:
    print("OpenVINO-Check fehlgeschlagen:", e)
PY

if [[ ! -f "${CONFIG_FILE}" ]]; then
  cp "${ROOT}/dictation/config.example.json" "${CONFIG_FILE}"
fi

# ── ydotool: uinput udev rule + user service (needs sudo once) ──────────────
if command -v ydotool >/dev/null 2>&1; then
  bash "${ROOT}/bin/setup-ydotool.sh" || \
    echo "WARNUNG: ydotool-Setup uebersprungen/fehlgeschlagen - Auto-Paste evtl. inaktiv."
fi

# ── systemd --user service (replaces the old autostart .desktop) ────────────
# Remove a previous autostart entry so the daemon is not launched twice.
rm -f "${DESKTOP_FILE}"

cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Whisper Dictation daemon (local speech-to-text)
After=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
ExecStart=${VENV}/bin/python -u ${ROOT}/dictation/daemon.py
ExecReload=/bin/kill -HUP \$MAINPID
Restart=on-failure
RestartSec=3
Environment=YDOTOOL_SOCKET=%t/.ydotool_socket
StandardOutput=append:%h/.cache/whisper-dictation/daemon.log
StandardError=append:%h/.cache/whisper-dictation/daemon.log

[Install]
WantedBy=default.target
EOF

cat > "${SETTINGS_DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=Whisper Dictation Settings
Comment=Configure local Whisper dictation
Exec=${ROOT}/bin/open-whisper-dictation-settings.sh
Icon=audio-input-microphone
Terminal=false
Categories=Utility;AudioVideo;
StartupNotify=true
EOF

if command -v systemctl >/dev/null 2>&1; then
  systemctl --user daemon-reload
  systemctl --user enable --now whisper-dictation.service || \
    echo "WARNUNG: konnte den Dienst nicht starten (evtl. erst nach Neuanmeldung)."
else
  "${ROOT}/bin/whisper-dictation.sh" --restart
fi

printf '\nFertig.\n'
printf 'systemd-Dienst:   %s\n' "${SERVICE_FILE}"
printf 'Settings-Starter: %s\n' "${SETTINGS_DESKTOP_FILE}"
printf 'Config:           %s\n' "${CONFIG_FILE}"
printf 'Daemon-Log:       %s\n' "${HOME}/.cache/whisper-dictation/daemon.log"
printf '\nHotkey: doppelt %s tippen zum Starten/Stoppen der Aufnahme.\n' "Right Ctrl"
