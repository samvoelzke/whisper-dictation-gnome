# Whisper Dictation For GNOME

Dieses Setup startet einen kleinen Hintergrundprozess, der auf eine doppelte `Right Ctrl`-Taste hoert.

## Verhalten

- Doppelt `Right Ctrl`: Aufnahme starten
- Doppelt `Right Ctrl`: Aufnahme stoppen
- Danach wird der Text lokal mit Whisper transkribiert und in das aktuell fokussierte Fenster eingefuegt

## Dateien

- `dictation/daemon.py`: Hintergrundprozess fuer Hotkey, Aufnahme, Transkription und Paste
- `bin/install-whisper-dictation.sh`: installiert Autostart und startet den Daemon neu
- `bin/whisper-dictation.sh --restart`: startet den Daemon manuell neu
- `bin/whisper-dictation.sh --stop`: beendet den Daemon
- `bin/open-whisper-dictation-settings.sh`: oeffnet die grafische Einstellungsoberflaeche
- `gui/settings.py`: GTK4-GUI fuer Modellwahl und die wichtigsten Laufzeitoptionen

## Konfiguration

Die Laufzeitkonfiguration liegt in `~/.config/whisper-dictation/config.json`.

Zusatzlich wird ein Desktop-Starter `Whisper Dictation Settings` unter `~/.local/share/applications/` angelegt. Darueber kannst du Modell, Sprache, Hotkey und Paste-Modus aendern und den Daemon direkt neu starten.

Wichtige Optionen:

- `double_tap_key`: `ctrl_r`, `ctrl_l`, `alt_r`, `alt_l`, `f8`, `f9`, `f10`, `pause`
- `model`: standardmaessig `turbo`
- `language`: standardmaessig `de`
- `paste_mode`: `auto`, `ctrl_v`, `ctrl_shift_v`, `shift_insert`
- `record_device`: ALSA-Aufnahmegeraet, standardmaessig `default`

## Logs

Das Laufzeitlog liegt unter `~/.cache/whisper-dictation/daemon.log`.
