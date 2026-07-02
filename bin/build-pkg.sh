#!/usr/bin/env bash
# Baut ein macOS .pkg Installationspaket für Whisper Dictation
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
BUILD_DIR="$ROOT/build-pkg"
PKG_ROOT="$BUILD_DIR/root"
SCRIPTS_DIR="$BUILD_DIR/scripts"
VERSION="1.0"
OUTPUT="$ROOT/WhisperDictation-${VERSION}.pkg"

BOLD="\033[1m"; GREEN="\033[0;32m"; RESET="\033[0m"
step() { echo -e "\n${BOLD}▶ $*${RESET}"; }
ok()   { echo -e "${GREEN}✓ $*${RESET}"; }

step "Build-Verzeichnis vorbereiten..."
rm -rf "$BUILD_DIR"
mkdir -p "$PKG_ROOT/usr/local/lib/whisper-dictation"
mkdir -p "$PKG_ROOT/Applications"
mkdir -p "$SCRIPTS_DIR"

step "Projektdateien kopieren..."
rsync -a --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
  --exclude='build-pkg' --exclude='*.pkg' --exclude='whisper' \
  "$ROOT/" "$PKG_ROOT/usr/local/lib/whisper-dictation/"
ok "Projektdateien kopiert"

step "App Bundle kopieren..."
cp -R "$ROOT/app/Whisper Dictation.app" "$PKG_ROOT/Applications/"
ok "App Bundle kopiert"

step "Postinstall-Script erstellen..."
cat > "$SCRIPTS_DIR/postinstall" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="/usr/local/lib/whisper-dictation"
LOG="/tmp/whisper-dictation-install.log"

echo "Whisper Dictation Installation gestartet..." > "$LOG"
echo "Nutzer: $USER" >> "$LOG"
echo "Home: $HOME" >> "$LOG"

# PATH für Homebrew setzen
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

# Homebrew PATH evaluieren falls vorhanden
if [[ -f /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# install-macos.sh ausführen als aktueller Nutzer
cd "$INSTALL_DIR"
bash "$INSTALL_DIR/bin/install-macos.sh" >> "$LOG" 2>&1 || true

echo "Installation abgeschlossen." >> "$LOG"
exit 0
SCRIPT
chmod +x "$SCRIPTS_DIR/postinstall"
ok "Postinstall-Script erstellt"

step "Welcome/Conclusion Texte erstellen..."
cat > "$BUILD_DIR/welcome.html" <<'HTML'
<html><body>
<h2>Willkommen bei Whisper Dictation</h2>
<p>Dieser Installer richtet alles automatisch ein:</p>
<ul>
  <li>🍺 Homebrew (falls nicht installiert)</li>
  <li>🐍 Python + alle Abhängigkeiten</li>
  <li>🎙 Whisper Dictation App</li>
  <li>🔄 Autostart bei Login</li>
</ul>
<p><b>Die Installation dauert ca. 5–10 Minuten.</b><br>
Danach erscheint ein 🎙 Icon in deiner Menüleiste.</p>
</body></html>
HTML

cat > "$BUILD_DIR/conclusion.html" <<'HTML'
<html><body>
<h2>✅ Installation abgeschlossen!</h2>
<p>Whisper Dictation läuft jetzt im Hintergrund.</p>
<h3>Letzter Schritt: Accessibility-Berechtigung</h3>
<p>macOS benötigt einmalig eine Berechtigung damit die App globale Tastenkürzel erkennen kann:</p>
<ol>
  <li>Öffne <b>Systemeinstellungen → Datenschutz → Bedienungshilfen</b></li>
  <li>Klicke das Schloss → klicke <b>+</b></li>
  <li>Drücke <b>Cmd+Shift+G</b> und füge den Python-Pfad ein<br>
      <i>(Der genaue Pfad steht im Installations-Log unter /tmp/whisper-dictation-install.log)</i></li>
  <li>Schalter aktivieren → fertig!</li>
</ol>
<p>🎙 Doppelt <b>rechtes Ctrl</b> drücken = Aufnahme starten/stoppen</p>
</body></html>
HTML
ok "HTML-Texte erstellt"

step "Component Package erstellen..."
pkgbuild \
  --root "$PKG_ROOT" \
  --scripts "$SCRIPTS_DIR" \
  --identifier "io.whisper-dictation.app" \
  --version "$VERSION" \
  --install-location "/" \
  "$BUILD_DIR/component.pkg"
ok "Component Package erstellt"

step "Distribution Package erstellen..."
cat > "$BUILD_DIR/distribution.xml" <<DIST
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>Whisper Dictation</title>
    <welcome file="welcome.html" mime-type="text/html"/>
    <conclusion file="conclusion.html" mime-type="text/html"/>
    <options customize="never" require-scripts="false" rootVolumeOnly="true"/>
    <pkg-ref id="io.whisper-dictation.app"/>
    <choices-outline>
        <line choice="default">
            <line choice="io.whisper-dictation.app"/>
        </line>
    </choices-outline>
    <choice id="default"/>
    <choice id="io.whisper-dictation.app" visible="false">
        <pkg-ref id="io.whisper-dictation.app"/>
    </choice>
    <pkg-ref id="io.whisper-dictation.app" version="${VERSION}" onConclusion="none">component.pkg</pkg-ref>
</installer-gui-script>
DIST

productbuild \
  --distribution "$BUILD_DIR/distribution.xml" \
  --package-path "$BUILD_DIR" \
  --resources "$BUILD_DIR" \
  "$OUTPUT"
ok "Distribution Package erstellt"

step "Aufräumen..."
rm -rf "$BUILD_DIR"

echo ""
echo -e "${GREEN}${BOLD}✅ Fertig!${RESET}"
echo ""
echo "  📦 Paket: $OUTPUT"
echo "  $(du -sh "$OUTPUT" | cut -f1) groß"
echo ""
echo "  Dein Bruder muss nur:"
echo "  1. Doppelklick auf WhisperDictation-${VERSION}.pkg"
echo "  2. Weiter → Weiter → Installieren"
echo "  3. Accessibility-Berechtigung einmal vergeben"
echo ""
