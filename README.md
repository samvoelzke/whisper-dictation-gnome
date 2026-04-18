# Whisper Dictation

Local speech-to-text for your desktop — double-tap a key, speak, double-tap again. The transcribed text is instantly pasted wherever your cursor is. No cloud, no subscription, everything runs locally via [OpenAI Whisper](https://github.com/openai/whisper).

**Supported platforms:** macOS (Apple Silicon M1/M2/M3/M4) · Linux (GNOME/X11)

---

## macOS – Quick Start

> **One command installs everything automatically** (Homebrew, Python, PyTorch, all dependencies).

```bash
git clone https://github.com/samvoelzke/whisper-dictation-gnome
cd whisper-dictation-gnome
bash bin/install-macos.sh
```

The installer will:
- Install **Homebrew** if missing (with your confirmation)
- Install **Python**, **portaudio**, **tkinter**
- Optionally install **Ollama** for AI text cleanup
- Create a Python virtual environment with PyTorch (MPS/Apple Silicon)
- Install the app to `/Applications/Whisper Dictation.app`
- Set up **LaunchAgents** so everything starts automatically at login
- Open **System Settings → Accessibility** for the one required manual step

### One manual step: Accessibility permission

macOS requires you to grant Accessibility access once so the app can detect global key presses:

1. **System Settings → Privacy & Security → Accessibility**
2. Click the lock to unlock
3. Click **+** → press `Cmd+Shift+G` → paste this path:
   ```
   /opt/homebrew/Cellar/python@3.XX/3.XX.X/Frameworks/Python.framework/Versions/3.XX/bin/python3.XX
   ```
   *(The installer prints the exact path for your system)*
4. Enable the toggle

> **Why this path?** macOS checks the real binary, not symlinks. `.venv/bin/python` won't work — you need the Homebrew Python path.

---

## How it works

1. Double-tap the configured key (default: **Right Ctrl**) → recording starts
2. Speak
3. Double-tap again → Whisper transcribes locally → text is pasted at cursor

### Menubar app

After installation a 🎙 icon appears in your menubar. Click it to:

| Menu item | Description |
|-----------|-------------|
| ● Daemon läuft / ○ gestoppt | Current recording daemon status |
| ▶ Daemon starten | Start the daemon |
| ↺ Daemon neu starten | Restart (after config changes) |
| 🤖 Ollama | AI text cleanup submenu |
| ⚙ Einstellungen | Open settings GUI |
| 🔐 Accessibility prüfen | Check/fix accessibility permission |

### Ollama AI text cleanup (optional)

After Whisper transcribes your speech, a local Ollama model can clean up punctuation, capitalization and grammar — similar to how ChatGPT Voice works.

**Setup via menubar:**
1. Click 🎙 → **🤖 Ollama** → **▶ Ollama starten**
2. Select a model (e.g. `llama3.2:3b`) — you'll be asked to download it automatically
3. Enable **◻ Text-Cleanup aktivieren**

**Recommended models for Apple Silicon:**

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `llama3.2:3b` | 2 GB | ⭐⭐⭐⭐ | ~1s |
| `phi3:mini` | 2.3 GB | ⭐⭐⭐ | ~0.8s |
| `gemma3:1b` | 0.8 GB | ⭐⭐ | ~0.4s |

---

## Configuration

Config file: `~/.config/whisper-dictation/config.json`

| Key | Default | Description |
|-----|---------|-------------|
| `double_tap_key` | `ctrl_r` | Hotkey: `ctrl_r`, `ctrl_l`, `alt_r`, `alt_l`, `f8`–`f10` |
| `double_tap_window_ms` | `400` | Max time between two key presses (ms) |
| `language` | `de` | Whisper language code (`de`, `en`, `fr`, … or empty for auto) |
| `model` | `turbo` | Whisper model — see table below |
| `paste_mode` | `cmd_v` | `cmd_v` (macOS), `ctrl_v`, `ctrl_shift_v`, `shift_insert`, `auto` |
| `record_device` | `default` | Microphone device |
| `max_record_seconds` | `180` | Max recording length |
| `initial_prompt` | `""` | Optional Whisper context prompt |
| `ollama_postprocess` | `false` | Enable Ollama AI text cleanup |
| `ollama_model` | `llama3.2:3b` | Ollama model to use |
| `ollama_host` | `http://localhost:11434` | Ollama API URL |

### Whisper model comparison

| Model | Quality | Speed | Recommendation |
|-------|---------|-------|----------------|
| **turbo** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Best for everyone** |
| large-v3 | ⭐⭐⭐⭐⭐ | ⭐⭐ | Only for critical accuracy needs |
| medium | ⭐⭐⭐ | ⭐⭐⭐ | Weak hardware without GPU |
| small | ⭐⭐ | ⭐⭐⭐⭐ | Very weak hardware |

---

## Linux (GNOME / X11)

```bash
sudo apt install python3 python3-venv git alsa-utils xclip
git clone https://github.com/samvoelzke/whisper-dictation-gnome
cd whisper-dictation-gnome
bash bin/install-whisper-dictation.sh
```

### Linux daemon control

```bash
bin/whisper-dictation.sh --status
bin/whisper-dictation.sh --restart
bin/whisper-dictation.sh --stop
```

### Linux settings GUI

```bash
bin/open-whisper-dictation-settings.sh
```

---

## Troubleshooting

**No keyboard events detected (macOS)**
→ Check Accessibility permission. The menubar icon shows 🎙⚠ if missing. Click → 🔐 Accessibility prüfen.

**"Daemon gestoppt" in menubar**
→ Click ↺ Daemon neu starten, or check the log: `~/.cache/whisper-dictation/daemon.log`

**Text not pasted after transcription**
→ Try changing `paste_mode` in settings. For most apps: `cmd_v`. For terminals: `ctrl_shift_v`.

**Whisper model download on first run**
→ The `turbo` model is ~1.5 GB and downloads automatically on first use. This takes a few minutes depending on your connection.

---

## License

MIT
