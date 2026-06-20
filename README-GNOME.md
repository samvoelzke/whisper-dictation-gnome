# Whisper Dictation für GNOME (Wayland)

Lokale Sprache-zu-Text-Diktierfunktion: doppelt `Right Ctrl` tippen, sprechen,
wieder doppelt tippen — der Text wird mit Whisper transkribiert und ins
fokussierte Fenster eingefügt. Alles lokal, keine Cloud.

Getestet auf **Fedora / GNOME Shell 50 / Wayland**.

## Verhalten

- Doppelt `Right Ctrl`: Aufnahme starten
- Doppelt `Right Ctrl`: Aufnahme stoppen → transkribieren → einfügen
- Status erscheint als Desktop-Benachrichtigung (kein Tray-Icon nötig)

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

## Logs

`~/.cache/whisper-dictation/daemon.log`

## Fehlersuche

- **Hotkey reagiert nicht** → User in `input`-Gruppe? (`id -nG | grep input`), danach neu anmelden.
- **Text landet nur in der Zwischenablage** → `ydotoold` läuft? (`systemctl --user status ydotoold`). Falls nicht: `bash bin/setup-ydotool.sh`.
- **Mikrofon falsch** → Gerät im GTK-Einstellungsfenster wählen.
- **Läuft auf CPU statt GPU?** → Geräte prüfen: `.venv/bin/python -c "import openvino as ov; print(ov.Core().available_devices)"`. Steht `GPU` nicht dabei, fehlt der Intel-Compute-Runtime (`intel-opencl`/Level-Zero). Der Daemon fällt dann automatisch auf faster-whisper (CPU) zurück (siehe `daemon.log`).
