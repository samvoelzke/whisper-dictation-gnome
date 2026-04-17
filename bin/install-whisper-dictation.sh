#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
AUTOSTART_DIR="${HOME}/.config/autostart"
APPLICATIONS_DIR="${HOME}/.local/share/applications"
CONFIG_DIR="${HOME}/.config/whisper-dictation"
CONFIG_FILE="${CONFIG_DIR}/config.json"
DESKTOP_FILE="${AUTOSTART_DIR}/whisper-dictation.desktop"
SETTINGS_DESKTOP_FILE="${APPLICATIONS_DIR}/whisper-dictation-settings.desktop"

mkdir -p "${AUTOSTART_DIR}" "${APPLICATIONS_DIR}" "${CONFIG_DIR}" "${HOME}/.cache/whisper-dictation"

if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  python3 -m venv "${ROOT}/.venv"
  "${ROOT}/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
fi

if ! "${ROOT}/.venv/bin/python" -c "import whisper, torch, pynput" >/dev/null 2>&1; then
  "${ROOT}/.venv/bin/python" -m pip install torch pynput
  "${ROOT}/.venv/bin/python" -m pip install -e "${ROOT}/whisper"
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  cp "${ROOT}/dictation/config.example.json" "${CONFIG_FILE}"
fi

cat > "${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=Whisper Dictation
Comment=Double Right Ctrl to dictate with local Whisper
Exec=${ROOT}/bin/whisper-dictation.sh
Terminal=false
X-GNOME-Autostart-enabled=true
StartupNotify=false
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

"${ROOT}/bin/whisper-dictation.sh" --restart

printf 'Installed autostart entry at %s\n' "${DESKTOP_FILE}"
printf 'Installed settings launcher at %s\n' "${SETTINGS_DESKTOP_FILE}"
printf 'Config file: %s\n' "${CONFIG_FILE}"
printf 'Daemon log: %s\n' "${HOME}/.cache/whisper-dictation/daemon.log"
