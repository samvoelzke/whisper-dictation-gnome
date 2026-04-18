#!/usr/bin/env bash
# Whisper Dictation – macOS Setup (Apple Silicon)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT"

echo "=== Whisper Dictation – macOS Setup ==="
echo ""

# 1. Python prüfen
if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 nicht gefunden. Installiere Python 3.11+ von https://www.python.org"
  exit 1
fi
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python $PY_VERSION gefunden."

# 2. Homebrew-Abhängigkeiten
echo ""
echo "Prüfe Homebrew-Pakete..."
if ! command -v brew &>/dev/null; then
  echo "ERROR: Homebrew nicht gefunden. Installiere es von https://brew.sh"
  exit 1
fi
brew list portaudio &>/dev/null || brew install portaudio
echo "portaudio OK"

# tkinter für die Settings-GUI (Homebrew Python hat es nicht eingebaut)
PY_MAJOR_MINOR=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
brew list "python-tk@${PY_MAJOR_MINOR}" &>/dev/null || brew install "python-tk@${PY_MAJOR_MINOR}" || true
echo "python-tk OK"

# 3. venv erstellen
echo ""
echo "Erstelle Python-Umgebung..."
python3 -m venv .venv
source .venv/bin/activate

# 4. PyTorch mit MPS-Support
echo ""
echo "Installiere PyTorch (Apple Silicon MPS)..."
pip install --upgrade pip --quiet
pip install torch torchvision torchaudio --quiet

# 5. Python-Abhängigkeiten
echo ""
echo "Installiere Python-Pakete..."
pip install \
  openai-whisper \
  pynput \
  sounddevice \
  numpy \
  --quiet

echo "Alle Pakete installiert."

# 6. Whisper-Repo klonen falls noch nicht vorhanden
if [[ ! -d "$ROOT/whisper/.git" ]]; then
  echo ""
  echo "Klone openai/whisper Repo..."
  git clone --depth=1 https://github.com/openai/whisper.git "$ROOT/whisper"
fi

# 7. macOS-Standardconfig schreiben
CONFIG_DIR="$HOME/.config/whisper-dictation"
CONFIG_FILE="$CONFIG_DIR/config.json"
mkdir -p "$CONFIG_DIR"
if [[ ! -f "$CONFIG_FILE" ]]; then
  cat > "$CONFIG_FILE" <<'JSON'
{
  "double_tap_key": "ctrl_r",
  "double_tap_window_ms": 400,
  "language": "de",
  "model": "turbo",
  "paste_mode": "cmd_v",
  "record_device": "default",
  "max_record_seconds": 180,
  "initial_prompt": ""
}
JSON
  echo "Config erstellt: $CONFIG_FILE"
fi

# 8. App ins /Applications kopieren
echo ""
echo "Installiere Whisper Dictation.app nach /Applications..."
APP_SRC="${ROOT}/app/Whisper Dictation.app"
APP_DST="/Applications/Whisper Dictation.app"
if [[ -d "${APP_DST}" ]]; then
  rm -rf "${APP_DST}"
fi
cp -R "${APP_SRC}" "${APP_DST}"
# Launcher ausführbar machen
chmod +x "${APP_DST}/Contents/MacOS/whisper-dictation-launcher"
# Quarantine-Flag entfernen damit macOS nicht blockiert
xattr -rd com.apple.quarantine "${APP_DST}" 2>/dev/null || true
echo "App installiert: ${APP_DST}"

# 9. LaunchAgent – startet die .app beim Login
PLIST="$HOME/Library/LaunchAgents/io.whisper-dictation.daemon.plist"
PYTHON_BIN="${ROOT}/.venv/bin/python"

echo ""
echo "Erstelle/aktualisiere LaunchAgent..."
cat > "$PLIST" <<XML
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>io.whisper-dictation.daemon</string>
  <key>ProgramArguments</key>
  <array>
    <string>PYTHON_BIN_PLACEHOLDER</string>
    <string>-u</string>
    <string>ROOT_PLACEHOLDER/app/menubar.py</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>HOME_PLACEHOLDER/.cache/whisper-dictation/daemon.log</string>
  <key>StandardErrorPath</key>
  <string>HOME_PLACEHOLDER/.cache/whisper-dictation/daemon.log</string>
  <key>WorkingDirectory</key>
  <string>ROOT_PLACEHOLDER</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>ROOT_PLACEHOLDER/.venv/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
</dict>
</plist>
XML
sed -i '' \
  "s|PYTHON_BIN_PLACEHOLDER|${PYTHON_BIN}|g; \
   s|ROOT_PLACEHOLDER|${ROOT}|g; \
   s|HOME_PLACEHOLDER|${HOME}|g" "$PLIST"

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST" 2>/dev/null || true
echo "LaunchAgent installiert (startet bei jedem Login automatisch)."

echo ""
echo "======================================================"
echo "  Installation abgeschlossen!"
echo "======================================================"
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  WICHTIG: Accessibility-Berechtigung einmalig einrichten!        ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                  ║"
echo "║  Ohne diese Berechtigung erkennt pynput keine Tastendrücke.     ║"
echo "║                                                                  ║"
echo "║  Du musst EINEN der folgenden Einträge hinzufügen:              ║"
echo "║                                                                  ║"
echo "║  WICHTIG: macOS prüft den ECHTEN Binary-Pfad, nicht den Symlink! ║"
echo "║                                                                  ║"
REAL_PYTHON=$(readlink -f "${PYTHON_BIN}" 2>/dev/null || echo "${PYTHON_BIN}")
echo "║  Diesen Pfad in Bedienungshilfen eintragen:                     ║"
echo "║  → ${REAL_PYTHON}"
echo "║                                                                  ║"
echo "║  (Nicht .venv/bin/python — macOS ignoriert Symlinks!)           ║"
echo "║                                                                  ║"
echo "║  Option B (wenn du manuell aus dem Terminal startest):           ║"
echo "║    → Ghostty / Terminal.app / iTerm2 (je nach Terminal)         ║"
echo "║                                                                  ║"
echo "║  So geht's:                                                      ║"
echo "║  1. Systemeinstellungen → Datenschutz & Sicherheit              ║"
echo "║     → Bedienungshilfen                                           ║"
echo "║  2. Das Schloss unten links entsperren                           ║"
echo "║  3. + klicken → oben genannten Pfad/App hinzufügen              ║"
echo "║  4. Schalter aktivieren                                          ║"
echo "║  5. Daemon neu starten                                           ║"
echo "║                                                                  ║"
echo "║  Oder direkt öffnen:                                             ║"
echo "║  open 'x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility'"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Daemon starten:      bin/whisper-dictation-mac.sh --start"
echo "Daemon stoppen:      bin/whisper-dictation-mac.sh --stop"
echo "Status prüfen:       bin/whisper-dictation-mac.sh --status"
echo "Log anzeigen:        bin/whisper-dictation-mac.sh --log"
echo "Einstellungen (GUI): bin/open-whisper-dictation-settings.sh"
echo ""
echo "Doppelt rechtes Ctrl drücken = Aufnahme starten/stoppen"
echo ""

# Systemeinstellungen direkt öffnen
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility" 2>/dev/null || true
