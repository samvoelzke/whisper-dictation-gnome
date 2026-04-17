# Whisper Dictation

Lokale Spracheingabe für den Desktop — doppelt eine Taste drücken, sprechen, nochmal drücken. Der Text erscheint direkt an der Cursorposition in jeder App. Kein Cloud-Dienst, alles läuft lokal via [OpenAI Whisper](https://github.com/openai/whisper).

**Unterstützte Plattformen:** Linux (GNOME/X11) · macOS (Apple Silicon)

---

## Wie es funktioniert

1. Doppelt die konfigurierte Taste drücken → Aufnahme startet (Benachrichtigung erscheint)
2. Sprechen
3. Nochmal doppelt drücken → Whisper transkribiert lokal → Text wird eingefügt

---

## Linux (GNOME / X11)

### Voraussetzungen

```bash
sudo apt install python3 python3-venv git alsa-utils xclip
# NVIDIA GPU empfohlen (CUDA), funktioniert aber auch auf CPU
```

### Installation

```bash
git clone https://github.com/samvoelzke/whisper-dictation-gnome
cd whisper-dictation-gnome
bash bin/install-whisper-dictation.sh
```

Das Script:
- Klont das openai/whisper Repo
- Erstellt ein Python venv mit PyTorch (CUDA wenn verfügbar)
- Schreibt einen Autostart-Eintrag (`~/.config/autostart/`)
- Legt die Standardconfig an (`~/.config/whisper-dictation/config.json`)

### Daemon steuern

```bash
bin/whisper-dictation.sh --status
bin/whisper-dictation.sh --restart
bin/whisper-dictation.sh --stop
```

### Einstellungen (GUI)

```bash
bin/open-whisper-dictation-settings.sh
# oder im App-Menü: "Whisper Dictation Settings"
```

---

## macOS (Apple Silicon – M1/M2/M3/M4)

### Voraussetzungen

- macOS 12.3+
- [Homebrew](https://brew.sh)
- Python 3.11+
- **Accessibility-Berechtigung** (einmalig, Schritt unten)

### Installation

```bash
git clone https://github.com/samvoelzke/whisper-dictation-gnome
cd whisper-dictation-gnome
bash bin/install-macos.sh
```

Das Script installiert PyTorch mit MPS-Support, sounddevice, pynput und richtet einen LaunchAgent für den Autostart ein.

### Accessibility-Berechtigung (einmalig!)

Ohne diese Berechtigung kann pynput keine globalen Tastendrücke erkennen:

1. **System Settings → Privacy & Security → Accessibility**
2. Das `+` klicken
3. Das Terminal hinzufügen, aus dem du den Daemon startest (Terminal.app, iTerm2, etc.)
4. Den Schalter aktivieren

> Falls der Daemon trotzdem keine Tastendrücke erkennt: Terminal neu starten und `bin/whisper-dictation-mac.sh --restart` ausführen.

### Daemon steuern

```bash
bin/whisper-dictation-mac.sh --start
bin/whisper-dictation-mac.sh --stop
bin/whisper-dictation-mac.sh --restart
bin/whisper-dictation-mac.sh --status
bin/whisper-dictation-mac.sh --log
```

---

## Konfiguration

Config-Datei: `~/.config/whisper-dictation/config.json`

| Schlüssel | Standard | Beschreibung |
|-----------|----------|--------------|
| `double_tap_key` | `ctrl_r` | Taste: `ctrl_r`, `ctrl_l`, `alt_r`, `alt_l`, `f8`, `f9`, `f10`, `pause` |
| `double_tap_window_ms` | `400` | Maximale Zeit zwischen zwei Tastendrücken (ms) |
| `language` | `de` | Sprache für Whisper (`de`, `en`, `fr`, … oder leer für Auto) |
| `model` | `turbo` | Whisper-Modell (Empfehlung: `turbo`) |
| `paste_mode` | `auto` | `auto`, `ctrl_v`, `cmd_v` (macOS), `ctrl_shift_v`, `shift_insert` |
| `record_device` | `default` | ALSA-Device (Linux: z.B. `plughw:4,0`) oder `default` |
| `max_record_seconds` | `180` | Maximale Aufnahmedauer in Sekunden |
| `initial_prompt` | `""` | Optionaler Whisper-Prompt (Fachbegriffe, Kontext) |

### Modellwahl

| Modell | Qualität | Geschwindigkeit | Empfehlung |
|--------|----------|-----------------|------------|
| **turbo** | ★★★★★ | ★★★★★ | **Beste Wahl für alle** |
| large-v3 | ★★★★★ | ★★☆☆☆ | Nur bei sehr kritischen Transkriptionen |
| medium | ★★★☆☆ | ★★★☆☆ | Schwächere Hardware ohne GPU |
| small | ★★☆☆☆ | ★★★★☆ | Sehr schwache Hardware |

---

## Log & Debugging

```bash
# Linux
tail -f ~/.cache/whisper-dictation/daemon.log

# macOS
bin/whisper-dictation-mac.sh --log
```

---

## Lizenz

MIT
