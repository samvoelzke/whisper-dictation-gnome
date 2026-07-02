# Whisper Dictation für GNOME (Wayland)

Lokale Sprache-zu-Text-Diktierfunktion: doppelt `Right Ctrl` tippen, sprechen,
wieder doppelt tippen — der Text wird mit Whisper transkribiert und ins
fokussierte Fenster eingefügt. Alles lokal, keine Cloud.

Getestet auf **Fedora / GNOME Shell 50 / Wayland**.

## Verhalten

- Doppelt `Right Ctrl`: Aufnahme starten
- Doppelt `Right Ctrl`: Aufnahme stoppen → transkribieren → einfügen
- Alternativ **Push-to-Talk** (`hotkey_mode`): Taste halten = aufnehmen,
  loslassen = einfügen. Kurze Tipps (< 250 ms, z. B. `Strg+C`) starten
  **keine** Aufnahme.
- `Esc` während der Aufnahme: abbrechen ohne Transkription
- Status erscheint als Desktop-Benachrichtigung (kein Tray-Icon nötig)
- Läuft ein Clipboard-Manager (z. B. **Vicinae**), bleibt das Diktat an
  **erster Stelle** seiner History (das Zurücksetzen der Zwischenablage wird
  dann automatisch übersprungen — der alte Inhalt steckt ja im Manager).

## Warum kein Tray-Icon?

GNOME (Shell 50, Wayland) hat keinen klassischen System-Tray. Statt eines
Indikators nutzt dieses Setup **Desktop-Benachrichtigungen** für den Status
und ein **GTK4-Einstellungsfenster** zur Steuerung (Start/Stop/Modellwahl/Log).

## Wie es auf Wayland funktioniert

Der X11-Stack (pynput-Listener, xclip, Xlib-Fenstererkennung) funktioniert auf
GNOME-Wayland nicht. Dieses Setup nutzt stattdessen:

| Aufgabe | Werkzeug |
|---|---|
| Globaler Hotkey | `evdev` (liest `/dev/input`, braucht User in Gruppe `input`) |
| Transkription | **`openvino-genai`** (Intel GPU/NPU) mit `faster-whisper` (CPU) als Fallback |
| Aufnahme | `arecord` (ALSA/PipeWire) |
| Zwischenablage | `wl-copy` |
| Text einfügen | `ydotool` (virtuelles Tippen via `/dev/uinput`) |

### Beschleunigung (Intel GPU/NPU)

Auf Intel-Hardware läuft Whisper über **OpenVINO** auf der GPU — deutlich
schneller und akkuschonender als CPU. Gemessen (large-v3-turbo, 11 s Audio):

| Pfad | Zeit | Faktor |
|---|---|---|
| OpenVINO GPU (Intel Arc) | 0,38 s | ~17× schneller |
| faster-whisper CPU (int8) | 6,58 s | Fallback |

Es werden die **offiziellen vorkonvertierten `OpenVINO/*`-Modelle** von Hugging
Face genutzt (kein PyTorch/optimum nötig). Geräteauswahl via `ov_device`
(`AUTO` bevorzugt **GPU** > NPU > CPU — die iGPU ist am schnellsten und
kompiliert zuverlässig).

**NPU (experimentell):** Die Panther-Lake-NPU lässt sich via
`sudo dnf install intel-npu-driver` aktivieren (OpenVINO listet danach `NPU`).
Stand 2026-06 kann der NPU-Compiler den Whisper-Graph aber **nicht übersetzen**
(`ZE_RESULT_ERROR_UNSUPPORTED_FEATURE`) — weder fp16 noch int8. Bei
`ov_device: "NPU"` fällt der Daemon daher automatisch auf die GPU zurück. Die
NPU wäre ohnehin nur stromsparender, nicht schneller als die GPU.

## Installation

```bash
# Systempakete (Fedora)
sudo dnf install alsa-utils wl-clipboard ydotool libnotify python3 \
  python3-gobject gtk4 libadwaita libcanberra-gtk3 sound-theme-freedesktop

# Projekt einrichten (venv, openvino-genai + faster-whisper, evdev,
# ydotool-Setup, systemd-Dienst)
bash bin/install-whisper-dictation.sh
```

Der Installer ruft `bin/setup-ydotool.sh`, das **einmalig `sudo`** braucht, um:

1. eine udev-Regel für `/dev/uinput` (Gruppe `input`, Modus 0660) anzulegen,
2. `ydotoold` als `systemctl --user`-Dienst zu starten.

Stelle sicher, dass dein User in der `input`-Gruppe ist (für den Hotkey):

```bash
sudo usermod -aG input "$USER"   # danach einmal neu anmelden
```

## Steuerung

Der Daemon läuft als **`systemctl --user`-Dienst** (Autostart + Auto-Restart):

```bash
systemctl --user status whisper-dictation     # Status
systemctl --user restart whisper-dictation    # Neustart (lädt Modell neu)
systemctl --user reload  whisper-dictation     # Config live übernehmen (kein Modell-Reload)
journalctl --user -u whisper-dictation -f      # Live-Log
```

- **Doppel-`Right Ctrl`** startet/stoppt die Aufnahme. Dezenter Ton bei Start/Fertig.
- Optional **GNOME-Tastenkombi** statt Doppel-Tap: Einstellungen → Tastatur →
  eigene Kürzel → Befehl `…/bin/whisper-dictation.sh --toggle`.

## Die App (GTK4/libadwaita)

`bin/open-whisper-dictation-settings.sh` öffnet die App mit drei Ansichten:

- **Werkbank** — in ein Textfeld diktieren und den Text per KI-Anweisung
  umformen (Presets: strukturieren, formeller, kürzer, übersetzen, …)
- **Rekorder** — Langaufnahmen (siehe unten)
- **Verlauf** — alle Diktate: durchsuchen (`Strg+F`), Volltext aufklappen,
  kopieren (auch den Rohtext vor der KI), einzeln löschen

Die **Einstellungen** öffnen sich GNOME-typisch als Dialog (`Strg+,` oder
Menü) und **gelten sofort** — kein Speichern-Knopf. Nur ein Tastenwechsel
zeigt ein Banner „Daemon-Neustart nötig". Weitere Kürzel: `Strg+R` Aufnahme
in der Werkbank, `Strg+1–3` Ansichten, `Strg+Q` beenden.

## Wörterbuch, Diktier-Modi & Schnipsel

- **Eigene Begriffe** (Einstellungen → Wörterbuch): Namen/Fachbegriffe, ein
  Begriff pro Zeile — fließen in die Erkennung ein (Hotwords + Prompt).
- **Ersetzungen**: hartnäckige Fehlerkennungen automatisch korrigieren,
  Format `falsch = richtig` (ganze Wörter, Groß-/Kleinschreibung egal).
- **Sprach-Schnipsel**: Sprich exakt den Auslöser („Grußformel"), und der
  hinterlegte Text wird eingefügt. Format `auslöser = Text` (`\n` = Umbruch).
- **Diktier-Modus** (Einstellungen → KI): `Standard`, `E-Mail` (formell, KI
  immer an), `Chat` (locker, KI immer an) oder `Roh` (KI nie).

## Dateien

- `dictation/daemon.py` — Hintergrunddienst (Hotkey, Aufnahme, Transkription, Paste)
- `bin/install-whisper-dictation.sh` — Installer (venv, Deps, systemd-Dienst)
- `bin/setup-ydotool.sh` — udev-Regel + ydotoold-Dienst (sudo)
- `bin/whisper-dictation.sh --restart|--stop|--status|--reload|--toggle` — Daemon steuern
- `bin/open-whisper-dictation-settings.sh` — Einstellungen öffnen
- `gui/settings.py` — libadwaita-GUI (Modell/Backend/Gerät/Sprache/Hotkey/…)

## Konfiguration

`~/.config/whisper-dictation/config.json`:

| Schlüssel | Werte / Standard |
|---|---|
| `double_tap_key` | `ctrl_r` (Standard), `ctrl_l`, `alt_r`, `alt_l`, `f8`–`f10`, `pause` |
| `model` | `turbo` (= large-v3-turbo, Standard), `large-v3`, `distil-large-v3` (GPU); `tiny`…`medium`, `*.en` (CPU); HF-CTranslate2-Repo (z. B. Deutsch-Finetune) |
| `backend` | `auto` (Linux→OpenVINO wenn möglich, sonst faster-whisper), `openvino`, `faster`, `openai` |
| `ov_device` | `AUTO` (GPU > NPU > CPU), `GPU`, `NPU` (experimentell), `CPU` |
| `vad_filter` | `true` (Stille filtern: weniger Halluzinationen + weniger Rechenzeit) |
| `hotwords` | Fachbegriffe/Namen, die bevorzugt erkannt werden (faster-whisper) |
| `beam_size` | `5` (kleiner = schneller) |
| `language` | `de` (Standard), `en`, oder leer/`auto` |
| `paste_mode` | `auto`/`ctrl_v` (Standard), `ctrl_shift_v` (Terminal), `shift_insert` |
| `record_device` | ALSA-Gerät, Standard `default` |
| `ollama_postprocess` | `false` (Standard). LLM-Textverbesserung an/aus |
| `ollama_model` | `qwen2.5:7b` (Standard) |
| `dictation_mode` | `standard` (Standard), `email`, `chat`, `raw` |
| `dictionary` | Liste eigener Begriffe (GUI: Wörterbuch) |
| `replacements` | `{"falsch": "richtig"}` — Wort-Korrekturen nach der Erkennung |
| `snippets` | `{"auslöser": "Text"}` — Sprach-Schnipsel |
| `hotkey_mode` | `double_tap` (Standard) oder `push_to_talk` |
| `restore_clipboard` | `true`; pausiert automatisch bei laufendem Clipboard-Manager |

## Textverbesserung (Ollama, optional)

Ein optionaler LLM-Schritt entfernt Füllwörter (äh, halt, …), korrigiert
Grammatik/Zeichensetzung und behält englische Fachbegriffe im DE+EN-Mix.
**Standard: aus** (kostet ~2–4 s extra pro Diktat auf CPU).

Aktivieren:

```bash
sudo dnf install ollama          # falls noch nicht vorhanden
sudo systemctl enable --now ollama
ollama pull qwen2.5:7b           # empfohlenes Modell (DE+EN, gute Qualität)
```

Dann in der GUI **„Ollama-Nachbearbeitung"** einschalten (oder
`ollama_postprocess: true`). Läuft Ollama nicht, fällt der Daemon automatisch
auf den Rohtext zurück. Kleinere Modelle (< 7B) neigen dazu, den Text zu
*beantworten* statt zu korrigieren — `qwen2.5:7b` ist der getestete Sweet Spot.

## Rekorder (Langaufnahmen: Vorlesungen & Calls)

Zusätzlich zum Live-Diktat gibt es im GUI den Tab **„Rekorder"** für stunden­lange
Aufnahmen (Vorlesungen, Meetings, Zoom/Teams-Calls). Bewusst dreistufig, damit
ein Absturz nicht alles verliert:

1. **Aufnahme** – `ffmpeg` schreibt **laufend** eine Opus-Datei (3 h ≈ ~40 MB).
   Quelle wählbar: **Mikrofon + System-Ton** (Standard), nur System-Ton
   (PipeWire-Monitor – für Online-Calls/Videos) oder nur Mikrofon.
2. **Transkription** – erst nach der Aufnahme, in **~5-Minuten-Chunks** über das
   Whisper-Backend (Standard `large-v3`). Chunk-Grenzen werden an **Sprechpausen
   ausgerichtet** (Silence-Detection), damit keine Wörter zerschnitten werden.
   Teil-Transkript + Fortschritt werden nach **jedem** Chunk gespeichert →
   abbruchsicher und fortsetzbar (auch nach Absturz/Kill, inkl. abgeschnittener
   Dateien ohne Dauer-Metadaten). Chunk-Länge: `recorder_chunk_seconds`.
   Das Transkript bekommt **`[mm:ss]`-Zeitmarken** pro Absatz — ein Klick auf
   eine Marke springt im Player genau dorthin. Danach schlägt die KI
   automatisch einen **Titel** vor (abschaltbar: `recorder_auto_title`).
3. **Zusammenfassung** – optional via Ollama (Map-Reduce). Ein Klick auf ein
   **Preset** (Vorlesungsnotizen / Meeting-Protokoll / Action-Items) oder ein
   eigener **Fokus-Prompt** (z. B. „prüfungsrelevante Definitionen") liefert
   strukturierte Notizen. Transkript + Notizen lassen sich als Markdown
   **nach Obsidian exportieren** (Vault-Pfad: `obsidian_vault`).

Im GUI gibt's außerdem live **Pegel-/Wellen-Anzeigen** (sehen, ob Ton ankommt —
auch vor der Aufnahme), Pause/Fortsetzen und während der Aufnahme die
**wachsende Dateigröße** (Beweis, dass wirklich geschrieben wird). Die
Aufnahmen-Liste ist **nach Datum gruppiert** (Heute/Gestern/Diese Woche/Älter),
und jede Zeile kann direkt **angehört** (▶) und **gelöscht** werden. Die
Quelle (Mic + System / System / Mic) wechselst du mit einem Klick per
Toggle-Gruppe. Pro Aufnahme gibt es eine **Detail-Seite** mit richtigem
**Audio-Player (Spulen, ±10 s, Pause, Lautstärke)**, klickbaren
Zeitmarken, **Suche im Transkript** (mit Treffer-Hervorhebung), editierbarem
Transkript und Notizen. Über das ⋮-Menü: Obsidian-Export, erneut
transkribieren, **„Audio entfernen, Transkript behalten"** (spart bei
Vorlesungen ~40 MB pro Stunde) und Löschen. Die Visualisierung ist in den
Einstellungen umschaltbar: **Wellen / Balken / Aus**.

Dateien liegen unter `~/.local/share/whisper-dictation/recordings/`
(`.opus`, `.txt`, `.summary.md`). CLI direkt nutzbar:

```bash
bin/whisper-recorder.sh record-start --source both --title "Vorlesung"
bin/whisper-recorder.sh record-stop
bin/whisper-recorder.sh transcribe <id> --model large-v3
bin/whisper-recorder.sh summarize  <id> --focus "Action-Items"
```

> **Hinweis:** Das Mitschneiden des nicht-öffentlich gesprochenen Worts anderer
> ohne deren Einwilligung ist in DE rechtlich heikel (§201 StGB). Eigene
> Vorlesungen / Aufnahmen mit Einwilligung sind unproblematisch.

Braucht `ffmpeg`/`ffprobe` und `pactl` (PipeWire):
`sudo dnf install ffmpeg-free pipewire-utils`.

## Logs

`~/.cache/whisper-dictation/daemon.log`

## Fehlersuche

- **Hotkey reagiert nicht** → User in `input`-Gruppe? (`id -nG | grep input`), danach neu anmelden.
- **Text landet nur in der Zwischenablage** → `ydotoold` läuft? (`systemctl --user status ydotoold`). Falls nicht: `bash bin/setup-ydotool.sh`.
- **Mikrofon falsch** → Gerät im GTK-Einstellungsfenster wählen.
- **Läuft auf CPU statt GPU?** → Geräte prüfen: `.venv/bin/python -c "import openvino as ov; print(ov.Core().available_devices)"`. Steht `GPU` nicht dabei, fehlt der Intel-Compute-Runtime (`intel-opencl`/Level-Zero). Der Daemon fällt dann automatisch auf faster-whisper (CPU) zurück (siehe `daemon.log`).
