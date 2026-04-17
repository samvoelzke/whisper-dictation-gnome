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

# 8. LaunchAgent für Autostart
PLIST="$HOME/Library/LaunchAgents/io.whisper-dictation.daemon.plist"
if [[ ! -f "$PLIST" ]]; then
  echo ""
  echo "Erstelle LaunchAgent für Autostart..."
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
    <string>${ROOT}/.venv/bin/python</string>
    <string>-u</string>
    <string>${ROOT}/dictation/daemon.py</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <false/>
  <key>StandardOutPath</key>
  <string>${HOME}/.cache/whisper-dictation/daemon.log</string>
  <key>StandardErrorPath</key>
  <string>${HOME}/.cache/whisper-dictation/daemon.log</string>
  <key>WorkingDirectory</key>
  <string>${ROOT}</string>
</dict>
</plist>
XML
  # Variablen in plist expandieren
  sed -i '' "s|\${ROOT}|${ROOT}|g; s|\${HOME}|${HOME}|g" "$PLIST"
  launchctl load "$PLIST" 2>/dev/null || true
  echo "LaunchAgent installiert und geladen."
fi

echo ""
echo "======================================================"
echo "  Installation abgeschlossen!"
echo "======================================================"
echo ""
echo "WICHTIG – Accessibility-Berechtigung erforderlich:"
echo "  System Settings → Privacy & Security → Accessibility"
echo "  → Terminal (oder iTerm2) hinzufügen und aktivieren"
echo ""
echo "Daemon starten:   bin/whisper-dictation-mac.sh --start"
echo "Daemon stoppen:   bin/whisper-dictation-mac.sh --stop"
echo "Status prüfen:    bin/whisper-dictation-mac.sh --status"
echo "Log anzeigen:     tail -f ~/.cache/whisper-dictation/daemon.log"
echo ""
echo "Doppelt rechtes Ctrl drücken = Aufnahme starten/stoppen"
