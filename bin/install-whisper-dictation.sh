#!/usr/bin/env bash
# Installer for Whisper Dictation on GNOME / Wayland (Fedora & friends).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
AUTOSTART_DIR="${HOME}/.config/autostart"
APPLICATIONS_DIR="${HOME}/.local/share/applications"
CONFIG_DIR="${HOME}/.config/whisper-dictation"
CONFIG_FILE="${CONFIG_DIR}/config.json"
DESKTOP_FILE="${AUTOSTART_DIR}/whisper-dictation.desktop"
APP_ID="io.voelzke.WhisperDictation"
SETTINGS_DESKTOP_FILE="${APPLICATIONS_DIR}/${APP_ID}.desktop"
ICON_DIR="${HOME}/.local/share/icons/hicolor/scalable/apps"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
SERVICE_FILE="${SYSTEMD_USER_DIR}/whisper-dictation.service"
VENV="${ROOT}/.venv"

mkdir -p "${AUTOSTART_DIR}" "${APPLICATIONS_DIR}" "${CONFIG_DIR}" "${SYSTEMD_USER_DIR}" "${HOME}/.cache/whisper-dictation"

# ── System dependencies (best effort hint) ─────────────────────────────────
missing=()
for tool in arecord wl-copy ydotool notify-send ffmpeg ffprobe pactl; do
  command -v "${tool}" >/dev/null 2>&1 || missing+=("${tool}")
done
if ((${#missing[@]})); then
  echo "WARNUNG: Folgende Tools fehlen: ${missing[*]}"
  echo "  (ffmpeg/ffprobe/pactl werden vom Rekorder fuer Langaufnahmen gebraucht.)"
  echo "  Auf Fedora:  sudo dnf install alsa-utils wl-clipboard ydotool libnotify ffmpeg-free pipewire-utils"
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
# sherpa-onnx: optional speaker recognition (no torch, ~4 MB). Models are
# downloaded on first activation in the app.
if ! "${VENV}/bin/python" -c "import sherpa_onnx" >/dev/null 2>&1; then
  echo "Installiere sherpa-onnx (Sprechererkennung, optional) ..."
  "${VENV}/bin/python" -m pip install sherpa-onnx || \
    echo "  (sherpa-onnx optional — Sprechererkennung bleibt sonst deaktiviert)"
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
# Cap restarts so a persistent failure (e.g. no model) can't loop forever.
StartLimitIntervalSec=120
StartLimitBurst=5

[Service]
Type=simple
ExecStart=${VENV}/bin/python -u ${ROOT}/dictation/daemon.py
ExecReload=/bin/kill -HUP \$MAINPID
Restart=on-failure
RestartSec=5
Environment=YDOTOOL_SOCKET=%t/.ydotool_socket
StandardOutput=append:%h/.cache/whisper-dictation/daemon.log
StandardError=append:%h/.cache/whisper-dictation/daemon.log

[Install]
# graphical-session, not default: the daemon needs XDG_RUNTIME_DIR + Wayland
# (IPC socket, evdev), which only exist once the graphical session is up.
WantedBy=graphical-session.target
EOF

# App icon (shown in the GNOME dash / app grid and on the window).
mkdir -p "${ICON_DIR}"
cp "${ROOT}/assets/${APP_ID}.svg" "${ICON_DIR}/${APP_ID}.svg"
# Remove the pre-rename launcher if it lingers from an older install.
rm -f "${APPLICATIONS_DIR}/whisper-dictation-settings.desktop"

# Desktop entry — basename matches the GTK application_id so GNOME links the
# running window to this icon.
cat > "${SETTINGS_DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=Whisper Dictation
GenericName=Diktat
Comment=Lokale Sprache-zu-Text per Doppel-Tastendruck
Exec=${ROOT}/bin/open-whisper-dictation-settings.sh
Icon=${APP_ID}
Terminal=false
Categories=AudioVideo;Accessibility;
Keywords=dictation;speech;whisper;voice;diktat;sprache;
StartupNotify=true
StartupWMClass=${APP_ID}
EOF

gtk-update-icon-cache -f -t "${HOME}/.local/share/icons/hicolor" 2>/dev/null || true
update-desktop-database "${APPLICATIONS_DIR}" 2>/dev/null || true

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
