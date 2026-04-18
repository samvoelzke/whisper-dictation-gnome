#!/usr/bin/env bash
# Whisper Dictation – macOS Setup (Apple Silicon)
# Installiert alle Abhängigkeiten automatisch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT"

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

step()  { echo -e "\n${BOLD}▶ $*${RESET}"; }
ok()    { echo -e "${GREEN}✓ $*${RESET}"; }
warn()  { echo -e "${YELLOW}⚠ $*${RESET}"; }
die()   { echo -e "${RED}✗ $*${RESET}"; exit 1; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║       Whisper Dictation – macOS Installation         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── 1. Xcode Command Line Tools ───────────────────────────────────────────────
step "Xcode Command Line Tools prüfen..."
if ! xcode-select -p &>/dev/null; then
  warn "Xcode CLT nicht gefunden – wird installiert (Popup erscheint)..."
  xcode-select --install 2>/dev/null || true
  echo "Warte auf Installation... (drücke Enter wenn fertig)"
  read -r
fi
ok "Xcode CLT OK"

# ── 2. Homebrew ───────────────────────────────────────────────────────────────
step "Homebrew prüfen..."
if ! command -v brew &>/dev/null; then
  warn "Homebrew nicht gefunden – wird automatisch installiert..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Homebrew PATH für Apple Silicon
  eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || true
fi
ok "Homebrew $(brew --version | head -1)"

# ── 3. Python ─────────────────────────────────────────────────────────────────
step "Python prüfen..."
if ! command -v python3 &>/dev/null || ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
  warn "Python 3.11+ nicht gefunden – wird installiert..."
  brew install python@3.11
fi
PYTHON=$(command -v python3.14 || command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3)
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ok "Python $PY_VERSION ($PYTHON)"

# ── 4. Homebrew-Pakete ────────────────────────────────────────────────────────
step "System-Pakete installieren..."
brew list portaudio &>/dev/null || brew install portaudio
ok "portaudio"

# tkinter für Settings-GUI
brew list "python-tk@${PY_VERSION}" &>/dev/null || brew install "python-tk@${PY_VERSION}" 2>/dev/null || \
  warn "python-tk@${PY_VERSION} nicht verfügbar – GUI könnte eingeschränkt sein"
ok "python-tk"

# Ollama (optional – für Text-Cleanup nach Transkription)
if ! command -v ollama &>/dev/null; then
  echo ""
  read -r -p "  Ollama installieren? (für KI-Textkorrektur nach Diktat) [j/N] " INSTALL_OLLAMA
  if [[ "${INSTALL_OLLAMA,,}" == "j" ]]; then
    brew install ollama
    ok "Ollama installiert"
  else
    warn "Ollama übersprungen – kann später über das Menü installiert werden"
  fi
else
  ok "Ollama $(ollama --version 2>/dev/null || echo '')"
fi

# ── 5. Python venv ────────────────────────────────────────────────────────────
step "Python-Umgebung erstellen..."
"$PYTHON" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet
ok "venv erstellt"

# ── 6. PyTorch ────────────────────────────────────────────────────────────────
step "PyTorch installieren (Apple Silicon MPS)..."
pip install torch torchvision torchaudio --quiet
ok "PyTorch installiert"

# ── 7. Python-Pakete ──────────────────────────────────────────────────────────
step "Python-Pakete installieren..."
pip install openai-whisper pynput sounddevice numpy --quiet
ok "Alle Pakete installiert"

# ── 8. Whisper-Repo ───────────────────────────────────────────────────────────
step "Whisper-Repo prüfen..."
if [[ ! -d "$ROOT/whisper/.git" ]]; then
  git clone --depth=1 https://github.com/openai/whisper.git "$ROOT/whisper"
fi
ok "Whisper-Repo vorhanden"

# ── 9. Config ────────────────────────────────────────────────────────────────
step "Konfiguration schreiben..."
CONFIG_DIR="$HOME/.config/whisper-dictation"
CONFIG_FILE="$CONFIG_DIR/config.json"
mkdir -p "$CONFIG_DIR" "$HOME/.cache/whisper-dictation"
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
  "initial_prompt": "",
  "ollama_postprocess": false,
  "ollama_model": "llama3.2:3b",
  "ollama_host": "http://localhost:11434"
}
JSON
  ok "Config erstellt: $CONFIG_FILE"
else
  ok "Config bereits vorhanden: $CONFIG_FILE"
fi

# ── 10. App ins /Applications ─────────────────────────────────────────────────
step "Whisper Dictation.app installieren..."
APP_SRC="${ROOT}/app/Whisper Dictation.app"
APP_DST="/Applications/Whisper Dictation.app"
rm -rf "${APP_DST}"
cp -R "${APP_SRC}" "${APP_DST}"
chmod +x "${APP_DST}/Contents/MacOS/whisper-dictation-launcher"
xattr -rd com.apple.quarantine "${APP_DST}" 2>/dev/null || true
ok "App installiert: ${APP_DST}"

# ── 11. LaunchAgents (Daemon + Menubar) ───────────────────────────────────────
step "LaunchAgents einrichten (Autostart bei Login)..."
PYTHON_BIN="${ROOT}/.venv/bin/python"
LAUNCHAGENTS="$HOME/Library/LaunchAgents"
mkdir -p "$LAUNCHAGENTS"

# Daemon – 24/7, KeepAlive
cat > "${LAUNCHAGENTS}/io.whisper-dictation.daemon.plist" <<XML
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>io.whisper-dictation.daemon</string>
  <key>ProgramArguments</key><array>
    <string>PYTHON_BIN_PH</string><string>-u</string>
    <string>ROOT_PH/dictation/daemon.py</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>HOME_PH/.cache/whisper-dictation/daemon.log</string>
  <key>StandardErrorPath</key><string>HOME_PH/.cache/whisper-dictation/daemon.log</string>
  <key>WorkingDirectory</key><string>ROOT_PH</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>ROOT_PH/.venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
</dict></plist>
XML

# Menubar – Menüleisten-App
cat > "${LAUNCHAGENTS}/io.whisper-dictation.menubar.plist" <<XML
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>io.whisper-dictation.menubar</string>
  <key>ProgramArguments</key><array>
    <string>PYTHON_BIN_PH</string><string>-u</string>
    <string>ROOT_PH/app/menubar.py</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>5</integer>
  <key>StandardOutPath</key><string>HOME_PH/.cache/whisper-dictation/menubar.log</string>
  <key>StandardErrorPath</key><string>HOME_PH/.cache/whisper-dictation/menubar.log</string>
  <key>WorkingDirectory</key><string>ROOT_PH</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>ROOT_PH/.venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
</dict></plist>
XML

# Platzhalter ersetzen
for PLIST in "${LAUNCHAGENTS}/io.whisper-dictation.daemon.plist" \
             "${LAUNCHAGENTS}/io.whisper-dictation.menubar.plist"; do
  sed -i '' \
    "s|PYTHON_BIN_PH|${PYTHON_BIN}|g; s|ROOT_PH|${ROOT}|g; s|HOME_PH|${HOME}|g" \
    "$PLIST"
  launchctl unload "$PLIST" 2>/dev/null || true
  launchctl load  "$PLIST" 2>/dev/null || true
done
ok "LaunchAgents geladen (starten bei jedem Login automatisch)"

# ── 12. Accessibility ────────────────────────────────────────────────────────
REAL_PYTHON=$(readlink -f "${PYTHON_BIN}" 2>/dev/null || echo "${PYTHON_BIN}")

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════╗"
echo    "║  LETZTER SCHRITT: Accessibility-Berechtigung einmalig vergeben!  ║"
echo    "╠══════════════════════════════════════════════════════════════════╣"
echo    "║                                                                  ║"
echo    "║  1. Systemeinstellungen → Datenschutz → Bedienungshilfen        ║"
echo    "║  2. Schloss entsperren                                           ║"
echo    "║  3. + klicken → Cmd+Shift+G → diesen Pfad einfügen:            ║"
echo -e "║     ${YELLOW}${REAL_PYTHON}${RESET}"
echo -e "${BOLD}║  4. Schalter aktivieren → fertig!                                ║"
echo    "╚══════════════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo "  Systemeinstellungen werden jetzt geöffnet..."
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility" 2>/dev/null || true

echo ""
echo -e "${GREEN}${BOLD}Installation abgeschlossen!${RESET}"
echo ""
echo "  🎙 Menüleiste: Whisper Dictation läuft im Hintergrund"
echo "  ⌨️  Doppelt rechtes Ctrl = Aufnahme starten/stoppen"
echo "  ⚙️  Einstellungen: Klick auf 🎙 in der Menüleiste → Einstellungen"
echo ""
