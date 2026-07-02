#!/usr/bin/env python3
"""Whisper Dictation settings — a libadwaita (GNOME) app.

Writes ~/.config/whisper-dictation/config.json and applies changes live via
`whisper-dictation.sh --reload` (no model reload unless model/device changed).
"""

from __future__ import annotations

import collections
import json
import math
import os
import re
import signal
import socket
import subprocess
import tempfile
import threading
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk


def detect_alsa_capture_devices() -> list[tuple[str, str]]:
    """Return list of (alsa_device_string, human_label) for all capture cards."""
    devices: list[tuple[str, str]] = [("default", "default (Systemstandard)")]
    try:
        out = subprocess.run(
            ["arecord", "--list-devices"],
            capture_output=True, text=True, check=False,
        ).stdout
        for line in out.splitlines():
            m = re.match(r"card\s+(\d+):\s+\S+\s+\[(.+?)\].*device\s+(\d+):\s+\S+\s+\[(.+?)\]", line)
            if m:
                card, card_name, dev, dev_name = m.group(1), m.group(2), m.group(3), m.group(4)
                hw = f"plughw:{card},{dev}"
                label = f"{hw}  —  {card_name} / {dev_name}"
                devices.append((hw, label))
    except Exception:
        pass
    return devices


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path.home() / ".config" / "whisper-dictation"
CONFIG_FILE = CONFIG_DIR / "config.json"
IPC_SOCKET = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "whisper-dictation.sock"


def ipc_call(req: dict, timeout: float = 200) -> dict:
    """Send one JSON request to the running daemon and return its reply."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(IPC_SOCKET))
        s.sendall((json.dumps(req) + "\n").encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode("utf-8") or "{}")
    finally:
        s.close()


LOG_FILE = Path.home() / ".cache" / "whisper-dictation" / "daemon.log"
HISTORY_FILE = Path.home() / ".cache" / "whisper-dictation" / "history.jsonl"
DAEMON_SCRIPT = PROJECT_ROOT / "bin" / "whisper-dictation.sh"


def read_history(limit: int = 100) -> list:
    if not HISTORY_FILE.exists():
        return []
    out = []
    try:
        for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines()[-limit:]:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return out


def format_ts(ts) -> str:
    try:
        import datetime
        return datetime.datetime.fromtimestamp(float(ts)).strftime("%d.%m. %H:%M")
    except Exception:
        return ""

DEFAULT_CONFIG = {
    "double_tap_key": "ctrl_r",
    "hotkey_mode": "double_tap",
    "double_tap_window_ms": 400,
    "language": "de",
    "model": "turbo",
    "backend": "auto",
    "ov_device": "AUTO",
    "beam_size": 5,
    "vad_filter": True,
    "hotwords": "",
    "voice_commands": False,
    "sound_cue": True,
    "restore_clipboard": True,
    "save_history": True,
    "paste_mode": "auto",
    "record_device": "default",
    "max_record_seconds": 180,
    "initial_prompt": "",  # filled with DEFAULT_INITIAL_PROMPT in the UI when empty
    "ollama_postprocess": False,
    "ollama_model": "qwen2.5:7b",
    "llm_toggle_key": "",
    "command_key": "",
    "recorder_source": "both",
    "recorder_model": "large-v3",
    "recorder_bitrate": "32k",
    "recorder_chunk_seconds": 300,
    "recorder_language": "",
    "recorder_mic_device": "",
    "recorder_monitor_device": "",
    "recorder_auto_process": False,
    # Live audio visualization: "waves" | "bar" | "none" (Werkbank + Rekorder).
    "audio_visualizer": "waves",
}

# turbo/large-v3/distil-large-v3 run on the OpenVINO backend (Intel GPU/NPU)
# when available; the others fall back to faster-whisper on CPU. The German
# finetune is a CTranslate2 HF model and always runs via faster-whisper.
DE_FINETUNE = "TheChola/whisper-large-v3-turbo-german-faster-whisper"
MODEL_OPTIONS = [
    ("turbo", "turbo (large-v3-turbo) — GPU"),
    (DE_FINETUNE, "Deutsch-Finetune (turbo, CPU)"),
    ("distil-large-v3", "distil-large-v3 (EN) — GPU"),
    ("large-v3", "large-v3 — GPU"),
    ("tiny", "tiny (CPU)"),
    ("base", "base (CPU)"),
    ("small", "small (CPU)"),
    ("medium", "medium (CPU)"),
    ("tiny.en", "tiny.en (CPU)"),
    ("base.en", "base.en (CPU)"),
    ("small.en", "small.en (CPU)"),
    ("medium.en", "medium.en (CPU)"),
    ("large-v2", "large-v2 (CPU)"),
]

MODEL_HINTS = {
    "turbo": "Beste Standardwahl: stark + multilingual (DE/EN gemischt), laeuft auf der Intel-GPU.",
    DE_FINETUNE: "Deutsch-Finetune (WER ~2.6%). Bestes Deutsch, aber CPU-only und schwaecher bei Englisch.",
    "distil-large-v3": "Nur Englisch, distilliert: am schnellsten. Laeuft auf der GPU.",
    "large-v3": "Hoechste Qualitaet, multilingual, GPU. Etwas langsamer als turbo.",
    "tiny": "Extrem schnell, aber die geringste Genauigkeit (CPU).",
    "base": "Etwas genauer als tiny, immer noch sehr leicht (CPU).",
    "small": "Guter Mittelweg (CPU).",
    "medium": "Deutlich genauer, aber merklich schwerer (CPU).",
    "tiny.en": "Nur Englisch, maximal leichtgewichtig (CPU).",
    "base.en": "Nur Englisch, kompakt (CPU).",
    "small.en": "Nur Englisch, guter Kompromiss (CPU).",
    "medium.en": "Nur Englisch, stark aber schwerer (CPU).",
    "large-v2": "Aelteres grosses Modell (CPU). Meist nur fuer Vergleiche.",
}

# backend + ov_device stay in config (defaults: auto -> OpenVINO GPU, with
# automatic fallback to faster-whisper CPU). They are not exposed in the GUI
# because "auto" is right for virtually everyone; power users can edit the JSON.

LANGUAGE_OPTIONS = [
    ("de", "Deutsch"),
    ("en", "English"),
    ("", "Auto-Erkennung"),
    ("fr", "Français"),
    ("es", "Español"),
    ("it", "Italiano"),
    ("nl", "Nederlands"),
    ("pt", "Português"),
]

# Pre-filled context that biases recognition toward DE + common English tech
# terms (Whisper does not echo this into the output).
DEFAULT_INITIAL_PROMPT = (
    "Diktat auf Deutsch, teils mit englischen Fachbegriffen wie Pull Request, "
    "Deployment, Bug, Backend, Repository, Meeting."
)

HOTKEY_MODE_OPTIONS = [
    ("double_tap", "Doppel-Tap (start/stopp)"),
    ("push_to_talk", "Push-to-Talk (halten)"),
]

HOTKEY_OPTIONS = [
    ("ctrl_r", "Right Ctrl"),
    ("ctrl_l", "Left Ctrl"),
    ("alt_r", "Right Alt"),
    ("alt_l", "Left Alt"),
    ("f8", "F8"),
    ("f9", "F9"),
    ("f10", "F10"),
    ("pause", "Pause"),
]

# Key whose double-tap toggles Ollama cleanup on/off ("" = disabled).
LLM_TOGGLE_OPTIONS = [("", "Aus")] + HOTKEY_OPTIONS

PASTE_OPTIONS = [
    ("auto", "Auto (Ctrl+V)"),
    ("ctrl_v", "Ctrl+V"),
    ("ctrl_shift_v", "Ctrl+Shift+V (Terminal)"),
    ("shift_insert", "Shift+Insert"),
]

# Ollama cleanup models. Stars = quality for the cleanup task; bigger models are
# stronger but slower and need more RAM. Must be installed (ollama pull <name>).
LLM_MODEL_OPTIONS = [
    ("qwen2.5:7b", "qwen2.5:7b  ★★★★★  empfohlen (DE+EN)"),
    ("qwen2.5:14b", "qwen2.5:14b  ★★★★★  stärker, langsamer"),
    ("gemma3:4b", "gemma3:4b  ★★★★☆  schnell, multilingual"),
    ("qwen3:4b", "qwen3:4b  ★★★★☆"),
    ("qwen2.5:3b", "qwen2.5:3b  ★★★☆☆  schnell"),
    ("llama3.2:3b", "llama3.2:3b  ★★☆☆☆  sehr schnell, schwächer"),
]


# ── Long-form recorder (lectures / calls) ────────────────────────────────────
RECORDER_SCRIPT = PROJECT_ROOT / "bin" / "whisper-recorder.sh"
RECORDINGS_DIR = Path.home() / ".local" / "share" / "whisper-dictation" / "recordings"

REC_SOURCE_OPTIONS = [
    ("both", "Mikrofon + System-Ton (Meeting)"),
    ("system", "Nur System-Ton (Call/Video)"),
    ("mic", "Nur Mikrofon (Präsenz)"),
]
REC_SOURCE_SHORT = {"both": "Mic+System", "system": "System", "mic": "Mic"}

# Recorder-specific model / quality choices, surfaced directly in the tab.
REC_MODEL_OPTIONS = [
    ("large-v3", "large-v3 — genau (empfohlen)"),
    ("turbo", "turbo — schnell"),
    ("distil-large-v3", "distil-large-v3 (nur EN)"),
]
REC_QUALITY_OPTIONS = [
    ("24k", "Sprache — klein (24 kbit)"),
    ("32k", "Standard (32 kbit)"),
    ("64k", "Hoch (64 kbit)"),
    ("96k", "Musik (96 kbit)"),
]

# Live audio visualization style (Werkbank + Rekorder).
VISUALIZER_OPTIONS = [
    ("waves", "Wellen (Lautstärke-Verlauf)"),
    ("bar", "Balken (Pegel)"),
    ("none", "Aus"),
]
ACCENT_RGB = (0.21, 0.52, 0.89)  # GNOME blue #3584e4, readable on light + dark
_APP_CSS_PROVIDER = None


def install_app_css() -> None:
    """Small app-local polish for free text editors.

    libadwaita already handles the main layout; this only makes large text
    fields look intentional instead of like plain embedded widgets.
    """
    global _APP_CSS_PROVIDER
    if _APP_CSS_PROVIDER is not None:
        return
    display = Gdk.Display.get_default()
    if display is None:
        return
    provider = Gtk.CssProvider()
    provider.load_from_data(b"""
        .editor-card {
          border: 1px solid @borders;
          border-radius: 14px;
          background: @view_bg_color;
        }
        .editor-card:focus-within {
          border-color: @accent_color;
        }
        .editor-card textview,
        .editor-card textview text {
          background: transparent;
        }
        .editor-card textview {
          padding: 2px;
        }
    """)
    Gtk.StyleContext.add_provider_for_display(
        display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )
    _APP_CSS_PROVIDER = provider


def recorder_call(*args: str, timeout: float = 25) -> dict:
    """Run the recorder wrapper and parse its last (JSON) stdout line."""
    try:
        out = subprocess.run(
            [str(RECORDER_SCRIPT), *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
        return json.loads(lines[-1]) if lines else {"error": (out.stderr.strip() or "no output")}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _pcm_level(data: bytes) -> float:
    """Perceptual 0..1 level from raw s16 PCM: peak → dBFS → mapped so normal
    speech fills most of the bar and loud vs quiet differ clearly. A noise gate
    keeps ambient hiss from making the waves shimmer. (A flat linear peak scale
    made everything look low + uniform on a quiet laptop mic.)"""
    data = data[: len(data) // 2 * 2]
    if not data:
        return 0.0
    try:
        peak = max((abs(s) for s in memoryview(data).cast("h")), default=0) / 32768.0
    except (ValueError, TypeError):
        return 0.0
    if peak < 0.025:                      # ~ -32 dBFS noise gate (ignores room hiss)
        return 0.0
    db = 20.0 * math.log10(peak)
    return max(0.0, min(1.0, (db + 32.0) / 28.0))   # -32 dB → 0, -4 dB → 1


def _die_with_parent():
    """preexec_fn: ask the kernel to SIGTERM this child if the GUI dies, so
    pw-record/ffplay never linger as orphans after a crash/kill."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def default_source_name() -> str:
    """The default PipeWire mic source name (for live meters)."""
    try:
        return subprocess.run(["pactl", "get-default-source"],
                              capture_output=True, text=True, check=False).stdout.strip()
    except Exception:
        return ""


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds or 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d} h" if h else f"{m}:{s:02d} min"


def key_label(value: str) -> str:
    """Human label for a stored key (named / KEY_xxx / captured 'code:N[:label]')."""
    v = str(value or "")
    if not v:
        return ""
    if v.startswith("code:"):
        parts = v.split(":", 2)
        return parts[2] if len(parts) > 2 and parts[2] else f"Taste {parts[1] if len(parts) > 1 else '?'}"
    for code, lbl in HOTKEY_OPTIONS:
        if code == v.lower():
            return lbl
    if v.startswith("KEY_"):
        return v[4:]
    return v


def key_options(base: list, current: str) -> list:
    """Return base options, appending the current key if it's a captured one."""
    opts = list(base)
    if current and current not in [v for v, _ in opts]:
        opts.append((current, key_label(current)))
    return opts


def load_config() -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(
            json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    loaded = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    config = DEFAULT_CONFIG.copy()
    config.update(loaded)
    return config


def save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps(config, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def daemon_running() -> bool:
    result = subprocess.run(
        [str(DAEMON_SCRIPT), "--status"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() == "running"


# One-click instructions for the workbench (the free-text field stays for custom).
WB_PRESETS = [
    ("Strukturieren", "Strukturiere den Text übersichtlich, mit Absätzen oder Stichpunkten wo sinnvoll."),
    ("Formeller", "Schreibe den Text formeller."),
    ("Freundlicher", "Schreibe den Text freundlicher."),
    ("Menschlicher", "Schreibe den Text natürlicher und menschlicher, weniger steif."),
    ("Kürzer", "Fasse den Text kürzer, ohne Wichtiges zu verlieren."),
    ("Zusammenfassen", "Fasse den Text in wenigen Sätzen zusammen."),
    ("Korrigieren", "Korrigiere nur Rechtschreibung, Grammatik und Zeichensetzung."),
    ("Stichpunkte", "Formuliere den Text als Stichpunkt-Liste."),
    ("Englisch", "Übersetze den Text ins Englische."),
]


class AudioVisualizer(Gtk.DrawingArea):
    """Live mic/audio visualization: a scrolling volume waveform ("waves") or a
    single fill bar ("bar"). Feed it 0..1 levels via push(); choose the look via
    set_mode(); reset() clears it."""

    def __init__(self, mode: str = "waves", height: int = 46):
        super().__init__()
        self._mode = mode
        self._level = 0.0
        self._silent_runs = 0
        self._visible_bars = 250
        self._history = collections.deque(maxlen=400)
        self.set_content_height(height)
        self.set_hexpand(True)
        self.add_css_class("card")
        self.set_draw_func(self._draw)

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.queue_draw()

    def push(self, level: float) -> None:
        self._level = max(0.0, min(1.0, level))
        self._history.append(self._level)
        # Skip redrawing once the view is fully silent: a flat line scrolling
        # flat looks identical, so this cuts idle CPU/stutter to ~nothing.
        if self._level <= 0.0:
            self._silent_runs += 1
            if self._silent_runs > self._visible_bars + 2:
                return   # visible area has fully scrolled to flat
        else:
            self._silent_runs = 0
        self.queue_draw()

    def reset(self) -> None:
        self._level = 0.0
        self._history.clear()
        self.queue_draw()

    def _draw(self, _area, cr, width, height) -> None:
        r, g, b = ACCENT_RGB
        if self._mode == "bar":
            cr.set_source_rgba(r, g, b, 0.16)
            cr.rectangle(0, height * 0.30, width, height * 0.40)
            cr.fill()
            cr.set_source_rgba(r, g, b, 0.95)
            cr.rectangle(0, height * 0.30, max(2.0, width * self._level), height * 0.40)
            cr.fill()
            return
        # waves: centered bars, newest on the right, scrolling left
        mid = height / 2.0
        step = 4.0           # px per sample (bar + gap)
        bar_w = 2.0
        n = max(1, int(width / step))
        self._visible_bars = n
        hist = list(self._history)[-n:]
        cr.set_source_rgba(r, g, b, 0.95)
        for i, lvl in enumerate(hist):
            x = width - (len(hist) - i) * step
            bh = max(2.0, lvl * (height - 4))
            cr.rectangle(x, mid - bh / 2.0, bar_w, bh)
        cr.fill()


class LevelMeter:
    """Reads peak levels from a PipeWire device via pw-record and calls
    `callback(level)` on the GTK main thread (~20 Hz). No numpy/audioop needed."""

    def __init__(self, callback):
        self._cb = callback
        self._stop = None
        self._proc = None

    def start(self, device: str) -> None:
        self.stop()
        if not device:
            return
        stop = threading.Event()
        self._stop = stop

        def work():
            try:
                proc = subprocess.Popen(
                    ["pw-record", "--target", device, "--rate", "16000",
                     "--channels", "1", "--format", "s16", "-"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    preexec_fn=_die_with_parent)
            except FileNotFoundError:
                return
            self._proc = proc
            while not stop.is_set():
                data = proc.stdout.read(1600)   # ~50 ms blocks -> ~20 Hz
                if not data:
                    break
                level = _pcm_level(data)
                if not stop.is_set():
                    GLib.idle_add(self._cb, level)
            try:
                proc.terminate()
            except Exception:
                pass

        threading.Thread(target=work, daemon=True).start()

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
            self._stop = None
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None


class WorkbenchView(Gtk.Box):
    """Dictate into a scratchpad, then give the AI free-form instructions."""

    def __init__(self, toast_cb=None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12,
                         margin_top=18, margin_bottom=18, margin_start=24, margin_end=24)
        self.rec_proc: subprocess.Popen | None = None
        self.rec_wav: str | None = None
        self._toast_cb = toast_cb

        clamp = Adw.Clamp(maximum_size=1100, tightening_threshold=900)
        clamp.set_vexpand(True)
        self.append(clamp)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        clamp.set_child(box)

        rec_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.rec_btn = Gtk.Button(label="🔴 Aufnehmen")
        self.rec_btn.add_css_class("pill")
        self.rec_btn.add_css_class("suggested-action")
        self.rec_btn.connect("clicked", self._toggle_record)
        rec_row.append(self.rec_btn)
        self.status = Gtk.Label(label="Bereit", xalign=0, hexpand=True)
        self.status.add_css_class("dim-label")
        rec_row.append(self.status)
        box.append(rec_row)

        # Live volume visualization while dictating (waves / bar / off).
        self._viz = AudioVisualizer(mode=str(load_config().get("audio_visualizer", "waves")), height=40)
        self._viz.set_visible(False)
        box.append(self._viz)
        self._meter = LevelMeter(self._viz.push)

        scroller = Gtk.ScrolledWindow(vexpand=True)
        scroller.add_css_class("card")
        scroller.add_css_class("editor-card")
        self.text_view = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD_CHAR,
            top_margin=12, bottom_margin=12, left_margin=12, right_margin=12,
        )
        self.text_view.add_css_class("editor-view")
        scroller.set_child(self.text_view)
        box.append(scroller)

        presets = Gtk.FlowBox(
            selection_mode=Gtk.SelectionMode.NONE,
            column_spacing=4, row_spacing=4, max_children_per_line=12,
            halign=Gtk.Align.START,
        )
        for label, instruction in WB_PRESETS:
            chip = Gtk.Button(label=label)
            chip.add_css_class("flat")
            chip.connect("clicked", lambda _b, i=instruction: self._do_instruct(i))
            presets.append(chip)
        box.append(presets)

        instr_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.instr = Gtk.Entry(hexpand=True)
        self.instr.set_placeholder_text("Anweisung an die KI … (formaler · zusammenfassen · auf Englisch)")
        self.instr.connect("activate", self._run_instruction)
        instr_row.append(self.instr)
        self.send_btn = Gtk.Button(label="Ausführen")
        self.send_btn.add_css_class("suggested-action")
        self.send_btn.connect("clicked", self._run_instruction)
        instr_row.append(self.send_btn)
        box.append(instr_row)

        bottom = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8, halign=Gtk.Align.END)
        clear_btn = Gtk.Button(label="Leeren")
        clear_btn.connect("clicked", lambda *_: self._set_text(""))
        bottom.append(clear_btn)
        copy_btn = Gtk.Button(label="Kopieren")
        copy_btn.add_css_class("suggested-action")
        copy_btn.connect("clicked", self._copy)
        bottom.append(copy_btn)
        box.append(bottom)

    def _start_viz(self) -> None:
        mode = str(load_config().get("audio_visualizer", "waves"))
        self._viz.set_mode(mode)
        if mode == "none":
            self._viz.set_visible(False)
            return
        self._viz.reset()
        self._viz.set_visible(True)
        self._meter.start(default_source_name())

    def _stop_viz(self) -> None:
        self._meter.stop()
        self._viz.reset()
        self._viz.set_visible(False)

    def _text(self) -> str:
        buf = self.text_view.get_buffer()
        return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).strip()

    def _set_text(self, text: str) -> None:
        self.text_view.get_buffer().set_text(text)

    def _toggle_record(self, *_a) -> None:
        if self.rec_proc is None:
            self.rec_wav = tempfile.mktemp(suffix=".wav")
            device = str(load_config().get("record_device", "default"))
            try:
                self.rec_proc = subprocess.Popen([
                    "arecord", "-q", "-D", device, "-f", "S16_LE",
                    "-r", "16000", "-c", "1", "-t", "wav", self.rec_wav,
                ])
            except Exception as exc:
                self.status.set_text(f"arecord-Fehler: {exc}")
                return
            self.rec_btn.set_label("⏹ Stopp")
            self.status.set_text("● Aufnahme läuft …")
            self._start_viz()
            return

        try:
            self.rec_proc.send_signal(signal.SIGINT)
            self.rec_proc.wait(timeout=3)
        except Exception:
            pass
        self.rec_proc = None
        self._stop_viz()
        self.rec_btn.set_label("🔴 Aufnehmen")
        self.status.set_text("Transkribiere …")
        wav = self.rec_wav

        def work():
            try:
                r = ipc_call({"cmd": "transcribe", "wav": wav})
            except Exception as exc:
                r = {"error": str(exc)}
            GLib.idle_add(self._after_transcribe, r)
        threading.Thread(target=work, daemon=True).start()

    def _after_transcribe(self, r: dict) -> bool:
        if "error" in r:
            self.status.set_text(f"Fehler: {r['error']}")
            return False
        text = str(r.get("text", "")).strip()
        cur = self._text()
        joined = (cur + " " + text).strip() if cur else text
        self._set_text(joined)
        self.status.set_text("Bereit" if text else "Nichts erkannt")
        return False

    def _run_instruction(self, *_a) -> None:
        self._do_instruct(self.instr.get_text().strip())

    def _do_instruct(self, instruction: str) -> None:
        text = self._text()
        if not text:
            self.status.set_text("Erst etwas aufnehmen oder eingeben.")
            return
        if not instruction:
            return
        self.send_btn.set_sensitive(False)
        self.status.set_text("🤖 KI arbeitet …")

        def work():
            try:
                r = ipc_call({"cmd": "instruct", "text": text, "instruction": instruction})
            except Exception as exc:
                r = {"error": str(exc)}
            GLib.idle_add(self._after_instruct, r)
        threading.Thread(target=work, daemon=True).start()

    def _after_instruct(self, r: dict) -> bool:
        self.send_btn.set_sensitive(True)
        if "error" in r:
            self.status.set_text(f"Fehler: {r['error']}")
            return False
        self._set_text(str(r.get("text", "")).strip())
        self.instr.set_text("")
        self.status.set_text("Bereit")
        return False

    def _copy(self, *_a) -> None:
        subprocess.run(["wl-copy"], input=self._text().encode("utf-8"), check=False)
        self.status.set_text("In Zwischenablage kopiert ✓")


class RecorderView(Gtk.Box):
    """Long-form recorder: live meters, visible options, per-recording detail page.

    Built for hour-long lectures/calls where a crash must not lose everything.
    Uses an Adw.NavigationView so the recordings list and a full detail page
    (audio player, editable transcript, notes, actions) share one tab.
    """

    def __init__(self, toast_cb=None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._toast_cb = toast_cb
        self._busy: set[str] = set()
        self._busy_action: dict = {}          # base -> "transcribe" | "summarize"
        self._busy_started: dict = {}         # base -> monotonic start (for ETA)
        self._prog_cp: dict = {}              # base -> (seconds_done, wall) last checkpoint
        self._rows_by_base: dict = {}
        self._recording_base = None
        self._rec_start = 0.0
        self._paused = False
        self._frozen = 0.0
        self._timer_id = None
        self._poll_id = None
        # live meters
        self._meters_on = False
        self._meter_procs: list = []
        self._meter_stops: list = []
        self._mic_viz = None
        self._sys_viz = None
        self._default_mic = ""
        self._default_monitor = ""
        self._devices_loaded = False
        self._loading_devices = False
        # detail page
        self._detail_base = None
        self._detail: dict = {}
        self._play_proc = None
        self._play_timer = None
        self._play_start = 0.0

        self.nav = Adw.NavigationView()
        self.append(self.nav)
        self.nav.add(Adw.NavigationPage(title="Rekorder", child=self._build_list_page()))
        self.nav.connect("popped", self._on_popped)

    # ── tab lifecycle (called by the window) ─────────────────────────────────
    def on_shown(self):
        self.refresh()
        if self._detail_base is not None:
            return
        if self._devices_loaded:
            if self._recording_base is not None and not self._paused:
                self._start_meters()
            else:
                self._stop_meters()
        else:
            self._load_devices_async(then_start_meters=self._recording_base is not None and not self._paused)

    def on_hidden(self):
        self._stop_meters()
        self._stop_play()

    # ── small helpers ────────────────────────────────────────────────────────
    def _combo(self, title, options, current):
        row = Adw.ComboRow(title=title)
        row.set_model(Gtk.StringList.new([l for _, l in options]))
        row.set_selected(next((i for i, (v, _) in enumerate(options) if v == current), 0))
        return row

    @staticmethod
    def _cv(row, options):
        i = row.get_selected()
        return options[i][0] if 0 <= i < len(options) else ""

    def _toast(self, t):
        if self._toast_cb:
            self._toast_cb(t)

    def _persist(self, k, v):
        cfg = load_config()
        cfg[k] = v
        save_config(cfg)

    def _copy(self, text):
        subprocess.run(["wl-copy"], input=(text or "").encode("utf-8"), check=False)
        self._toast("In Zwischenablage kopiert ✓")

    def _load_devices_async(self, then_start_meters: bool = False):
        """Enumerate capture devices off the main thread (the call is ~180 ms),
        so opening the tab never blocks. Populates the combos when ready."""
        def work():
            d = recorder_call("devices", timeout=8)
            GLib.idle_add(self._apply_devices, d, then_start_meters)
        threading.Thread(target=work, daemon=True).start()

    def _apply_devices(self, d, then_start_meters):
        d = d if isinstance(d, dict) else {}
        self._default_mic = d.get("default_mic", "")
        self._default_monitor = d.get("default_monitor", "")
        mics = [("", "Standard-Mikrofon")]
        mons = [("", "Standard-Ausgang")]
        for m in d.get("mics", []):
            mics.append((m["name"], (m.get("desc") or m["name"])[:46]))
        for m in d.get("monitors", []):
            mons.append((m["name"], (m.get("desc") or m["name"])[:46]))
        cfg = load_config()
        self._loading_devices = True
        for row, opts, key in ((self.mic_row, mics, "recorder_mic_device"),
                               (self.mon_row, mons, "recorder_monitor_device")):
            row.set_model(Gtk.StringList.new([l for _, l in opts]))
            cur = str(cfg.get(key, ""))
            row.set_selected(next((i for i, (v, _) in enumerate(opts) if v == cur), 0))
        self._mic_opts, self._mon_opts = mics, mons
        self._loading_devices = False
        self._devices_loaded = True
        if then_start_meters and self._recording_base is not None and not self._paused:
            self._start_meters()
        return False

    # ── list page ────────────────────────────────────────────────────────────
    def _build_list_page(self):
        scroller = Gtk.ScrolledWindow(vexpand=True)
        clamp = Adw.Clamp(maximum_size=940, tightening_threshold=720,
                          margin_top=18, margin_bottom=18, margin_start=12, margin_end=12)
        scroller.set_child(clamp)
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        clamp.set_child(outer)
        cfg = load_config()

        ctl = Adw.PreferencesGroup(
            title="Neue Aufnahme",
            description="Für Vorlesungen und Calls. Wird laufend gespeichert — "
                        "ein Absturz kostet höchstens die letzten Sekunden.")
        outer.append(ctl)
        self.source_row = self._combo("Quelle", REC_SOURCE_OPTIONS,
                                      str(cfg.get("recorder_source", "both")))
        self.source_row.connect("notify::selected", self._on_source_changed)
        ctl.add(self.source_row)
        self.title_row = Adw.EntryRow(title="Titel (optional)")
        ctl.add(self.title_row)

        opt = Adw.ExpanderRow(title="Optionen", subtitle="Geräte, Modell, Qualität, Sprache, Chunk-Länge")
        ctl.add(opt)
        # Device combos start with just the defaults; the real list is filled in
        # asynchronously by _load_devices_async() so construction never blocks.
        self._mic_opts = [("", "Standard-Mikrofon")]
        self._mon_opts = [("", "Standard-Ausgang")]
        self.mic_row = self._combo("Mikrofon", self._mic_opts, "")
        self.mic_row.connect("notify::selected", self._on_device_changed)
        opt.add_row(self.mic_row)
        self.mon_row = self._combo("System-Ausgang (Monitor)", self._mon_opts, "")
        self.mon_row.connect("notify::selected", self._on_device_changed)
        opt.add_row(self.mon_row)
        self.model_row = self._combo("Modell", REC_MODEL_OPTIONS, str(cfg.get("recorder_model", "large-v3")))
        self.model_row.connect("notify::selected",
                               lambda *_: self._persist("recorder_model", self._cv(self.model_row, REC_MODEL_OPTIONS)))
        opt.add_row(self.model_row)
        self.quality_row = self._combo("Qualität", REC_QUALITY_OPTIONS, str(cfg.get("recorder_bitrate", "32k")))
        self.quality_row.connect("notify::selected",
                                 lambda *_: self._persist("recorder_bitrate", self._cv(self.quality_row, REC_QUALITY_OPTIONS)))
        opt.add_row(self.quality_row)
        self.lang_row = self._combo("Sprache", LANGUAGE_OPTIONS, str(cfg.get("recorder_language", "")).lower())
        self.lang_row.connect("notify::selected",
                              lambda *_: self._persist("recorder_language", self._cv(self.lang_row, LANGUAGE_OPTIONS)))
        opt.add_row(self.lang_row)
        self.chunk_row = Adw.SpinRow.new_with_range(60, 900, 30)
        self.chunk_row.set_title("Chunk-Länge (s)")
        self.chunk_row.set_subtitle("Teil-Speicherung; Grenzen werden an Sprechpausen ausgerichtet")
        self.chunk_row.set_value(float(cfg.get("recorder_chunk_seconds", 300)))
        self.chunk_row.connect("notify::value",
                               lambda *_: self._persist("recorder_chunk_seconds", int(self.chunk_row.get_value())))
        opt.add_row(self.chunk_row)
        self.auto_row = Adw.SwitchRow(title="Nach Stopp automatisch transkribieren",
                                      subtitle="Zusammenfassung danach manuell mit Fokus.")
        self.auto_row.set_active(bool(cfg.get("recorder_auto_process", False)))
        self.auto_row.connect("notify::active",
                              lambda *_: self._persist("recorder_auto_process", bool(self.auto_row.get_active())))
        opt.add_row(self.auto_row)

        viz_mode = str(cfg.get("audio_visualizer", "waves"))
        self._meters_group = Adw.PreferencesGroup(
            title="Live-Pegel", description="Wird während der Aufnahme angezeigt.")
        outer.append(self._meters_group)
        mic_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        mic_box.append(Gtk.Label(label="🎤 Mikrofon", xalign=0, css_classes=["dim-label"]))
        self._mic_viz = AudioVisualizer(mode=viz_mode)
        mic_box.append(self._mic_viz)
        self._meters_group.add(mic_box)
        sys_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, margin_top=8)
        sys_box.append(Gtk.Label(label="🔊 System-Ton", xalign=0, css_classes=["dim-label"]))
        self._sys_viz = AudioVisualizer(mode=viz_mode)
        sys_box.append(self._sys_viz)
        self._meters_group.add(sys_box)
        self._meters_group.set_visible(False)

        ctlbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12,
                         halign=Gtk.Align.CENTER, margin_top=6)
        outer.append(ctlbox)
        self.rec_btn = Gtk.Button(label="🔴 Aufnahme starten")
        self.rec_btn.add_css_class("pill")
        self.rec_btn.add_css_class("suggested-action")
        self.rec_btn.connect("clicked", self._toggle_record)
        ctlbox.append(self.rec_btn)
        self.pause_btn = Gtk.Button(label="⏸ Pause")
        self.pause_btn.add_css_class("pill")
        self.pause_btn.set_visible(False)
        self.pause_btn.connect("clicked", self._toggle_pause)
        ctlbox.append(self.pause_btn)
        self.timer_label = Gtk.Label(label="")
        self.timer_label.add_css_class("title-2")
        self.timer_label.add_css_class("numeric")
        ctlbox.append(self.timer_label)

        self.list_group = Adw.PreferencesGroup(title="Aufnahmen")
        refresh = Gtk.Button(icon_name="view-refresh-symbolic", valign=Gtk.Align.CENTER)
        refresh.add_css_class("flat")
        refresh.set_tooltip_text("Aktualisieren")
        refresh.connect("clicked", lambda *_: self.refresh())
        self.list_group.set_header_suffix(refresh)
        outer.append(self.list_group)

        self.refresh()
        return scroller

    def _on_source_changed(self, *_):
        self._persist("recorder_source", self._cv(self.source_row, REC_SOURCE_OPTIONS))
        if self._meters_on:
            self._start_meters()

    def _on_device_changed(self, *_):
        if self._loading_devices:        # ignore programmatic repopulation
            return
        self._persist("recorder_mic_device", self._cv(self.mic_row, self._mic_opts))
        self._persist("recorder_monitor_device", self._cv(self.mon_row, self._mon_opts))
        if self._meters_on:
            self._start_meters()

    # ── live level meters (pw-record + memoryview, no numpy/audioop needed) ──
    def _should_run_meters(self) -> bool:
        return (
            str(load_config().get("audio_visualizer", "waves")) != "none"
            and self._recording_base is not None
            and not self._paused
        )

    def _sync_meters_visibility(self) -> None:
        if hasattr(self, "_meters_group"):
            self._meters_group.set_visible(self._should_run_meters())

    def _start_meters(self):
        self._stop_meters()
        if not self._should_run_meters():
            self._sync_meters_visibility()
            return
        self._meters_on = True
        self._sync_meters_visibility()
        src = self._cv(self.source_row, REC_SOURCE_OPTIONS)
        mic = self._cv(self.mic_row, self._mic_opts) or self._default_mic
        mon = self._cv(self.mon_row, self._mon_opts) or self._default_monitor
        if src in ("both", "mic") and mic:
            self._spawn_meter(mic, self._mic_viz)
        if src in ("both", "system") and mon:
            self._spawn_meter(mon, self._sys_viz)

    def _spawn_meter(self, device, viz):
        stop = threading.Event()
        self._meter_stops.append(stop)

        def work():
            try:
                # pw-record is the reliable PipeWire capture (parec can yield no
                # data on some setups). "-" streams raw s16 to stdout.
                proc = subprocess.Popen(
                    ["pw-record", "--target", device, "--rate", "16000",
                     "--channels", "1", "--format", "s16", "-"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    preexec_fn=_die_with_parent)
            except FileNotFoundError:
                return
            self._meter_procs.append(proc)
            while not stop.is_set():
                data = proc.stdout.read(1600)   # ~50 ms blocks -> ~20 Hz
                if not data:
                    break
                level = _pcm_level(data)
                if not stop.is_set():
                    GLib.idle_add(viz.push, level)
            try:
                proc.terminate()
            except Exception:
                pass

        threading.Thread(target=work, daemon=True).start()

    def _stop_meters(self):
        self._meters_on = False
        for s in self._meter_stops:
            s.set()
        for p in self._meter_procs:
            try:
                p.terminate()
            except Exception:
                pass
        self._meter_stops = []
        self._meter_procs = []
        if self._mic_viz:
            self._mic_viz.reset()
        if self._sys_viz:
            self._sys_viz.reset()
        self._sync_meters_visibility()

    def set_visualizer_mode(self, mode: str) -> None:
        """Apply a Wellen/Balken/Aus choice live."""
        if self._mic_viz:
            self._mic_viz.set_mode(mode)
        if self._sys_viz:
            self._sys_viz.set_mode(mode)
        self._sync_meters_visibility()
        # Only re-spawn capture if meters are already live (recorder tab shown);
        # otherwise they'll start with the new mode on the next on_shown().
        if self._meters_on:
            self._start_meters()

    # ── recording control ────────────────────────────────────────────────────
    def _toggle_record(self, *_):
        if self._recording_base is None:
            self._start_record()
        else:
            self._stop_record()

    def _start_record(self):
        src = self._cv(self.source_row, REC_SOURCE_OPTIONS)
        title = self.title_row.get_text().strip()
        mic = self._cv(self.mic_row, self._mic_opts)
        mon = self._cv(self.mon_row, self._mon_opts)
        bitrate = self._cv(self.quality_row, REC_QUALITY_OPTIONS)
        args = ["record-start", "--source", src, "--title", title, "--bitrate", bitrate]
        if mic:
            args += ["--mic-device", mic]
        if mon:
            args += ["--monitor-device", mon]
        r = recorder_call(*args)
        if "error" in r:
            self._toast(f"Aufnahme-Fehler: {r['error']}")
            return
        self._recording_base = r.get("base")
        self._rec_start = GLib.get_monotonic_time() / 1e6
        self._paused = False
        self.rec_btn.set_label("⏹ Aufnahme stoppen")
        self.rec_btn.remove_css_class("suggested-action")
        self.rec_btn.add_css_class("destructive-action")
        self.pause_btn.set_label("⏸ Pause")
        self.pause_btn.set_visible(True)
        self._toast("● Aufnahme läuft")
        if self._timer_id is None:
            self._timer_id = GLib.timeout_add(500, self._tick)
        self._start_meters()

    def _stop_record(self):
        base = self._recording_base
        r = recorder_call("record-stop", timeout=20)
        self._recording_base = None
        self._paused = False
        if self._timer_id is not None:
            GLib.source_remove(self._timer_id)
            self._timer_id = None
        self.timer_label.set_label("")
        self.rec_btn.set_label("🔴 Aufnahme starten")
        self.rec_btn.remove_css_class("destructive-action")
        self.rec_btn.add_css_class("suggested-action")
        self.pause_btn.set_visible(False)
        self.title_row.set_text("")
        self._stop_meters()
        self._toast(f"Aufnahme gespeichert ({fmt_duration(r.get('duration_seconds', 0))})")
        self.refresh()
        if base and self.auto_row.get_active():
            self._transcribe(base)

    def _toggle_pause(self, *_):
        if self._recording_base is None:
            return
        if not self._paused:
            recorder_call("record-pause")
            self._paused = True
            self._frozen = GLib.get_monotonic_time() / 1e6 - self._rec_start
            self.pause_btn.set_label("▶ Fortsetzen")
            self._stop_meters()          # no signal is captured while paused → flat
            self._toast("⏸ Pausiert")
        else:
            recorder_call("record-resume")
            self._paused = False
            self._rec_start = GLib.get_monotonic_time() / 1e6 - self._frozen
            self.pause_btn.set_label("⏸ Pause")
            self._start_meters()
            self._toast("● Weiter")

    def _tick(self):
        if self._recording_base is None:
            return False
        if self._paused:
            dot, el = "<span foreground='#f5c211'>⏸</span>", self._frozen
        else:
            dot, el = "<span foreground='#e01b24'>●</span>", GLib.get_monotonic_time() / 1e6 - self._rec_start
        self.timer_label.set_markup(f"{dot} {int(el) // 60}:{int(el) % 60:02d}")
        return True

    def _apply_record_status(self, r: dict):
        if r.get("recording") and self._recording_base is None:
            self._recording_base = r.get("base")
            self._paused = bool(r.get("paused"))
            el = float(r.get("elapsed", 0))
            self._rec_start = GLib.get_monotonic_time() / 1e6 - el
            self._frozen = el
            self.rec_btn.set_label("⏹ Aufnahme stoppen")
            self.rec_btn.remove_css_class("suggested-action")
            self.rec_btn.add_css_class("destructive-action")
            self.pause_btn.set_label("▶ Fortsetzen" if self._paused else "⏸ Pause")
            self.pause_btn.set_visible(True)
            if self._timer_id is None:
                self._timer_id = GLib.timeout_add(500, self._tick)
            if not self._paused and self._detail_base is None:
                if self._devices_loaded:
                    self._start_meters()
                else:
                    self._load_devices_async(then_start_meters=True)
            else:
                self._sync_meters_visibility()

    # ── recordings list ──────────────────────────────────────────────────────
    def refresh(self):
        """Fetch status + list off the main thread so switching to the tab is
        instant (the recorder CLI calls take ~150 ms each)."""
        def work():
            status = recorder_call("record-status", timeout=8)
            data = recorder_call("list", timeout=10)
            GLib.idle_add(self._apply_refresh, status, data)
        threading.Thread(target=work, daemon=True).start()

    def _apply_refresh(self, status: dict, data: dict):
        self._apply_record_status(status)
        for row in self._rows_by_base.values():
            self.list_group.remove(row)
        self._rows_by_base = {}
        items = data.get("recordings", []) if isinstance(data, dict) else []
        if not items:
            row = Adw.ActionRow(title="Noch keine Aufnahmen",
                                subtitle="Starte oben eine Aufnahme – sie erscheint dann hier.")
            row.add_prefix(Gtk.Image(icon_name="audio-input-microphone-symbolic",
                                     valign=Gtk.Align.CENTER))
            self.list_group.add(row)
            self._rows_by_base["__empty__"] = row
            return False
        for item in items:
            self._add_row(item)
        if self._busy and self._poll_id is None:
            self._poll_id = GLib.timeout_add(400, self._poll_progress)
        return False

    def _status_line(self, item):
        parts = [fmt_duration(item.get("duration_seconds", 0)),
                 REC_SOURCE_SHORT.get(item.get("source", ""), item.get("source", ""))]
        if item.get("transcribed"):
            parts.append("✓ Transkript")
        if item.get("summarized"):
            parts.append("✓ Notizen")
        return " · ".join(p for p in parts if p)

    def _add_row(self, item):
        base = item["base"]
        row = Adw.ActionRow(title=item.get("title", base))
        self.list_group.add(row)
        self._rows_by_base[base] = row
        if item.get("recording"):
            row.set_subtitle("● nimmt auf …")
            icon = Gtk.Image(icon_name="media-record-symbolic", valign=Gtk.Align.CENTER)
            icon.add_css_class("error")
            row.add_prefix(icon)
            return
        # status icon: at-a-glance state of each recording
        if base in self._busy:
            ico = "content-loading-symbolic"
        elif item.get("summarized"):
            ico = "emblem-ok-symbolic"
        elif item.get("transcribed"):
            ico = "audio-x-generic-symbolic"
        else:
            ico = "audio-input-microphone-symbolic"
        row.add_prefix(Gtk.Image(icon_name=ico, valign=Gtk.Align.CENTER))
        row.set_activatable(True)
        row.connect("activated", lambda _r, x=base: self._open_detail(x))
        if base in self._busy:
            row.set_subtitle(self._busy_subtitle(base))
            row.add_suffix(Gtk.Spinner(spinning=True, valign=Gtk.Align.CENTER))
            return
        row.set_subtitle(self._status_line(item))
        if not item.get("transcribed"):
            b = Gtk.Button(label="Transkribieren", valign=Gtk.Align.CENTER)
            b.add_css_class("flat")
            b.connect("clicked", lambda _b, x=base: self._transcribe(x))
            row.add_suffix(b)
        row.add_suffix(Gtk.Image(icon_name="go-next-symbolic", valign=Gtk.Align.CENTER))

    # ── transcribe / summarize (background) ──────────────────────────────────
    def _transcribe(self, base):
        if base in self._busy:
            return
        self._busy.add(base)
        self._busy_action[base] = "transcribe"
        self._busy_started[base] = GLib.get_monotonic_time() / 1e6
        model = self._cv(self.model_row, REC_MODEL_OPTIONS) if hasattr(self, "model_row") \
            else str(load_config().get("recorder_model", "large-v3"))
        self.refresh()
        if self._detail_base == base:
            self._load_detail_content(base)
        self._toast(f"Transkribiere mit {model} …")

        def work():
            try:
                out = subprocess.run([str(RECORDER_SCRIPT), "transcribe", base, "--model", model],
                                     capture_output=True, text=True, check=False)
                ok = out.returncode == 0
            except Exception:
                ok = False
            GLib.idle_add(self._done, base, "Transkription fertig" if ok else "Transkription fehlgeschlagen")

        threading.Thread(target=work, daemon=True).start()
        if self._poll_id is None:
            self._poll_id = GLib.timeout_add(400, self._poll_progress)

    def _ask_focus(self, base):
        dlg = Adw.AlertDialog(
            heading="Worauf soll sich die Zusammenfassung fokussieren?",
            body="Beschreibe den Fokus — z. B. »Prüfungsrelevante Definitionen«, "
                 "»Action-Items und Entscheidungen« oder »Kernargumente des Vortrags«.")
        entry = Gtk.Entry(hexpand=True)
        entry.set_placeholder_text("Fokus (leer = wichtigste Inhalte und Action-Items)")
        dlg.set_extra_child(entry)
        dlg.add_response("cancel", "Abbrechen")
        dlg.add_response("go", "Zusammenfassen")
        dlg.set_response_appearance("go", Adw.ResponseAppearance.SUGGESTED)
        dlg.set_default_response("go")
        dlg.connect("response", lambda _d, resp, e=entry, x=base:
                    self._summarize(x, e.get_text().strip()) if resp == "go" else None)
        dlg.present(self.get_root())

    def _summarize(self, base, focus):
        if base in self._busy:
            return
        self._busy.add(base)
        self._busy_action[base] = "summarize"
        self._busy_started[base] = GLib.get_monotonic_time() / 1e6
        self.refresh()
        if self._detail_base == base:
            self._load_detail_content(base)
        self._toast("🤖 Erstelle Zusammenfassung …")

        def work():
            args = [str(RECORDER_SCRIPT), "summarize", base]
            if focus:
                args += ["--focus", focus]
            try:
                out = subprocess.run(args, capture_output=True, text=True, check=False)
                ok = out.returncode == 0
            except Exception:
                ok = False
            GLib.idle_add(self._done, base, "Zusammenfassung fertig" if ok else "Zusammenfassung fehlgeschlagen")

        threading.Thread(target=work, daemon=True).start()
        if self._poll_id is None:
            self._poll_id = GLib.timeout_add(400, self._poll_progress)

    def _done(self, base, msg):
        self._busy.discard(base)
        self._busy_action.pop(base, None)
        self._busy_started.pop(base, None)
        self._prog_cp.pop(base, None)
        self._toast(msg)
        self.refresh()
        if self._detail_base == base:
            self._load_detail_content(base)
        return False

    def _read_progress(self, base):
        try:
            return json.loads((RECORDINGS_DIR / f"{base}.progress.json").read_text())
        except Exception:
            return {}

    def _eta(self, base, pct):
        started = self._busy_started.get(base)
        if not started or not pct or pct <= 0:
            return ""
        elapsed = GLib.get_monotonic_time() / 1e6 - started
        remain = max(0, elapsed / pct * 100 - elapsed)
        return f" · noch ~{int(remain) // 60}:{int(remain) % 60:02d}"

    def _smoothed_pct(self, base):
        """Real per-chunk progress, interpolated between checkpoints by the
        measured processing speed → a continuously moving percentage (Whisper/
        OpenVINO gives no sub-call progress for long audio, so we estimate)."""
        d = self._read_progress(base)
        if d.get("status") == "loading":
            return None
        if d.get("status") == "done":
            return 100
        dur = float(d.get("duration") or 0)
        if dur <= 0:
            return d.get("percent")
        secs = float(d.get("seconds_done") or 0)
        chunk = float(d.get("chunk_seconds") or 300)
        started = self._busy_started.get(base)
        now = GLib.get_monotonic_time() / 1e6
        cp = self._prog_cp.get(base)
        if cp is None or secs != cp[0]:
            cp = (secs, now)
            self._prog_cp[base] = cp
        cp_secs, cp_wall = cp
        if started and cp_secs > 0 and cp_wall > started:
            rate = cp_secs / (cp_wall - started)      # measured audio-sec per wall-sec
        else:
            rate = chunk / 12.0                        # rough guess during first chunk
        est = min(cp_secs + rate * (now - cp_wall), cp_secs + chunk, dur)
        return max(1, min(99, int(est / dur * 100)))

    def _busy_subtitle(self, base):
        if self._busy_action.get(base) == "summarize":
            return "🤖 Fasse zusammen …"
        d = self._read_progress(base)
        if d.get("status") == "loading":
            return "⏳ Lädt Modell …"
        pct = self._smoothed_pct(base)
        if pct is not None:
            return f"⏳ Transkribiere … {pct} %{self._eta(base, pct)}"
        return "⏳ Transkribiere …"

    def _poll_progress(self):
        for base in list(self._busy):
            row = self._rows_by_base.get(base)
            if row is not None:
                row.set_subtitle(self._busy_subtitle(base))
            if base == self._detail_base:
                self._update_detail_progress(base)
        if not self._busy:
            self._poll_id = None
            return False
        return True

    # ── detail page ──────────────────────────────────────────────────────────
    def _open_detail(self, base):
        self._stop_meters()
        self._detail_base = base
        try:
            meta = json.loads((RECORDINGS_DIR / f"{base}.meta.json").read_text())
        except Exception:
            meta = {}
        self._detail = {}
        scroller = Gtk.ScrolledWindow(vexpand=True)
        clamp = Adw.Clamp(maximum_size=920, tightening_threshold=720,
                          margin_top=12, margin_bottom=18, margin_start=12, margin_end=12)
        scroller.set_child(clamp)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        clamp.set_child(box)

        # header: back + title + rename
        head = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        back = Gtk.Button(icon_name="go-previous-symbolic", valign=Gtk.Align.CENTER)
        back.add_css_class("flat")
        back.set_tooltip_text("Zurück")
        back.connect("clicked", lambda *_: self.nav.pop())
        head.append(back)
        title_lbl = Gtk.Label(label=meta.get("title", base), xalign=0, hexpand=True, wrap=True)
        title_lbl.add_css_class("title-2")
        head.append(title_lbl)
        self._detail["title_lbl"] = title_lbl
        rename = Gtk.Button(label="Titel ändern", valign=Gtk.Align.CENTER)
        rename.add_css_class("flat")
        rename.set_tooltip_text("Titel bearbeiten")
        rename.connect("clicked", lambda *_: self._rename(base))
        head.append(rename)
        box.append(head)

        # metadata + player
        info = Adw.PreferencesGroup()
        box.append(info)
        from datetime import datetime as _dt
        try:
            created = _dt.fromisoformat(meta.get("created", "")).strftime("%d.%m.%Y %H:%M")
        except Exception:
            created = ""
        current_title = meta.get("title", base)
        title_row = Adw.ActionRow(title="Titel", subtitle=current_title)
        title_row.set_activatable(True)
        title_row.add_prefix(Gtk.Image(icon_name="document-edit-symbolic", valign=Gtk.Align.CENTER))
        edit_title = Gtk.Button(label="Bearbeiten", valign=Gtk.Align.CENTER)
        edit_title.add_css_class("flat")
        edit_title.connect("clicked", lambda *_: self._rename(base))
        title_row.add_suffix(edit_title)
        title_row.connect("activated", lambda *_: self._rename(base))
        info.add(title_row)
        self._detail["title_row"] = title_row
        meta_line = " · ".join(p for p in [
            created, fmt_duration(meta.get("duration_seconds", 0)),
            REC_SOURCE_SHORT.get(meta.get("source", ""), meta.get("source", "")),
        ] if p)
        player = Adw.ActionRow(title="Audio", subtitle=meta_line)
        play_btn = Gtk.Button(label="▶ Abspielen", valign=Gtk.Align.CENTER)
        play_btn.add_css_class("flat")
        play_lbl = Gtk.Label(label="")
        play_lbl.add_css_class("numeric")
        play_lbl.add_css_class("dim-label")
        play_btn.connect("clicked", lambda *_: self._toggle_play(base, play_btn, play_lbl))
        player.add_suffix(play_lbl)
        player.add_suffix(play_btn)
        self._detail["play_btn"] = play_btn
        self._detail["play_lbl"] = play_lbl
        info.add(player)

        # progress (transcription / summary)
        prog_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._detail["progress_box"] = prog_box
        prog_lbl = Gtk.Label(label="", xalign=0)
        prog_lbl.add_css_class("dim-label")
        bar = Gtk.ProgressBar(show_text=False)
        self._detail["progress_lbl"] = prog_lbl
        self._detail["progress_bar"] = bar
        prog_box.append(prog_lbl)
        prog_box.append(bar)
        box.append(prog_box)

        # transcript section
        tr_group = Adw.PreferencesGroup(title="Transkript")
        tr_copy = Gtk.Button(icon_name="edit-copy-symbolic", valign=Gtk.Align.CENTER)
        tr_copy.add_css_class("flat")
        tr_copy.set_tooltip_text("Transkript kopieren")
        tr_copy.connect("clicked", lambda *_: self._copy(self._transcript_text()))
        tr_group.set_header_suffix(tr_copy)
        box.append(tr_group)
        tr_scroller = Gtk.ScrolledWindow(min_content_height=220)
        tr_scroller.add_css_class("card")
        tr_scroller.add_css_class("editor-card")
        tr_view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR, top_margin=10, bottom_margin=10,
                               left_margin=10, right_margin=10)
        tr_view.add_css_class("editor-view")
        tr_scroller.set_child(tr_view)
        self._detail["tr_view"] = tr_view
        self._detail["tr_scroller"] = tr_scroller
        tr_group.add(self._row_wrap(tr_scroller))
        tr_actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8, halign=Gtk.Align.END, margin_top=6)
        save_btn = Gtk.Button(label="Speichern")
        save_btn.connect("clicked", lambda *_: self._save_transcript(base))
        tr_actions.append(save_btn)
        self._detail["tr_actions"] = tr_actions
        tr_group.add(self._row_wrap(tr_actions))
        # empty-state
        tr_empty = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        tr_empty_lbl = Gtk.Label(label="Noch nicht transkribiert.", xalign=0, hexpand=True)
        tr_empty_lbl.add_css_class("dim-label")
        tr_empty.append(tr_empty_lbl)
        tr_btn = Gtk.Button(label="Transkribieren")
        tr_btn.add_css_class("suggested-action")
        tr_btn.connect("clicked", lambda *_: self._transcribe(base))
        tr_empty.append(tr_btn)
        self._detail["tr_empty"] = tr_empty
        tr_group.add(self._row_wrap(tr_empty))

        # notes section
        nt_group = Adw.PreferencesGroup(title="Notizen (LLM-Zusammenfassung)")
        nt_copy = Gtk.Button(icon_name="edit-copy-symbolic", valign=Gtk.Align.CENTER)
        nt_copy.add_css_class("flat")
        nt_copy.set_tooltip_text("Notizen kopieren")
        nt_copy.connect("clicked", lambda *_: self._copy(self._notes_text()))
        nt_group.set_header_suffix(nt_copy)
        box.append(nt_group)
        nt_scroller = Gtk.ScrolledWindow(min_content_height=180)
        nt_scroller.add_css_class("card")
        nt_scroller.add_css_class("editor-card")
        nt_view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR, editable=False, top_margin=10,
                               bottom_margin=10, left_margin=10, right_margin=10)
        nt_view.add_css_class("editor-view")
        nt_scroller.set_child(nt_view)
        self._detail["nt_view"] = nt_view
        self._detail["nt_scroller"] = nt_scroller
        nt_group.add(self._row_wrap(nt_scroller))
        nt_actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8, halign=Gtk.Align.END, margin_top=6)
        refocus = Gtk.Button(label="LLM-Fokus neu …")
        refocus.connect("clicked", lambda *_: self._ask_focus(base))
        nt_actions.append(refocus)
        self._detail["nt_actions"] = nt_actions
        nt_group.add(self._row_wrap(nt_actions))
        nt_empty = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        nt_empty_lbl = Gtk.Label(label="Noch keine Notizen.", xalign=0, hexpand=True)
        nt_empty_lbl.add_css_class("dim-label")
        nt_empty.append(nt_empty_lbl)
        nt_btn = Gtk.Button(label="Zusammenfassen …")
        nt_btn.connect("clicked", lambda *_: self._ask_focus(base))
        nt_empty.append(nt_btn)
        self._detail["nt_empty"] = nt_empty
        self._detail["nt_btn"] = nt_btn
        nt_group.add(self._row_wrap(nt_empty))

        # bottom actions
        bottom = Adw.PreferencesGroup()
        box.append(bottom)
        retr = Adw.ActionRow(title="Erneut transkribieren", subtitle="Verwirft das aktuelle Transkript")
        retr.set_activatable(True)
        retr.add_prefix(Gtk.Image(icon_name="view-refresh-symbolic"))
        retr.connect("activated", lambda *_: self._retranscribe(base))
        bottom.add(retr)
        folder = Adw.ActionRow(title="Ordner öffnen")
        folder.set_activatable(True)
        folder.add_prefix(Gtk.Image(icon_name="folder-symbolic"))
        folder.connect("activated", lambda *_: self._open_folder())
        bottom.add(folder)
        delete = Adw.ActionRow(title="Löschen")
        delete.add_css_class("error")
        delete.set_activatable(True)
        delete.add_prefix(Gtk.Image(icon_name="user-trash-symbolic"))
        delete.connect("activated", lambda *_: self._delete(base, from_detail=True))
        bottom.add(delete)

        page = Adw.NavigationPage(title=current_title, child=scroller)
        self._detail["page"] = page
        self.nav.push(page)
        self._load_detail_content(base)

    @staticmethod
    def _row_wrap(widget):
        # PreferencesGroup expects rows; wrap arbitrary widgets so they sit in the card.
        return widget

    def _on_popped(self, _nav, _page):
        self._detail_base = None
        self._detail = {}
        self._stop_play()
        # back on the list: resume live meters
        self._start_meters()

    def _transcript_text(self):
        try:
            return (RECORDINGS_DIR / f"{self._detail_base}.txt").read_text(encoding="utf-8")
        except Exception:
            return ""

    def _notes_text(self):
        try:
            return (RECORDINGS_DIR / f"{self._detail_base}.summary.md").read_text(encoding="utf-8")
        except Exception:
            return ""

    def _load_detail_content(self, base):
        if self._detail_base != base or not self._detail:
            return
        txt_path = RECORDINGS_DIR / f"{base}.txt"
        prog = self._read_progress(base)
        has_txt = txt_path.exists() and prog.get("status") == "done"
        busy = base in self._busy
        try:
            txt = txt_path.read_text(encoding="utf-8")
        except Exception:
            txt = ""
        self._detail["tr_view"].get_buffer().set_text(txt)
        self._detail["tr_scroller"].set_visible(has_txt)
        self._detail["tr_actions"].set_visible(has_txt)
        self._detail["tr_empty"].set_visible(not has_txt and not busy)

        notes = self._notes_text()
        has_notes = bool(notes.strip())
        self._detail["nt_view"].get_buffer().set_text(notes)
        self._detail["nt_scroller"].set_visible(has_notes)
        self._detail["nt_actions"].set_visible(has_notes)
        self._detail["nt_empty"].set_visible(not has_notes and not busy)
        # can only summarize once a transcript exists
        if not has_notes:
            self._detail["nt_btn"].set_sensitive(has_txt)
        self._update_detail_progress(base)

    def _update_detail_progress(self, base):
        if self._detail_base != base or not self._detail:
            return
        box = self._detail.get("progress_box")
        if box is None:
            return
        if base not in self._busy:
            box.set_visible(False)
            return
        box.set_visible(True)
        bar = self._detail["progress_bar"]
        lbl = self._detail["progress_lbl"]
        if self._busy_action.get(base) == "summarize":
            lbl.set_label("🤖 Erstelle Zusammenfassung …")
            bar.pulse()
            return
        d = self._read_progress(base)
        if d.get("status") == "loading":
            lbl.set_label("⏳ Lädt Modell …")
            bar.pulse()
            return
        pct = self._smoothed_pct(base)
        if pct is not None:
            bar.set_fraction(min(1.0, pct / 100.0))
            lbl.set_label(f"⏳ Transkribiere … {pct} %{self._eta(base, pct)}")
        else:
            lbl.set_label("⏳ Transkribiere …")
            bar.pulse()

    # ── detail actions ───────────────────────────────────────────────────────
    def _save_transcript(self, base):
        buf = self._detail["tr_view"].get_buffer()
        txt = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        (RECORDINGS_DIR / f"{base}.txt").write_text(txt, encoding="utf-8")
        self._toast("Transkript gespeichert ✓")

    def _retranscribe(self, base):
        if base in self._busy:
            return
        self._busy.add(base)
        self._busy_action[base] = "transcribe"
        self._busy_started[base] = GLib.get_monotonic_time() / 1e6
        model = self._cv(self.model_row, REC_MODEL_OPTIONS)
        self.refresh()
        if self._detail_base == base:
            self._load_detail_content(base)
        self._toast(f"Transkribiere neu mit {model} …")

        def work():
            try:
                out = subprocess.run([str(RECORDER_SCRIPT), "transcribe", base, "--restart", "--model", model],
                                     capture_output=True, text=True, check=False)
                ok = out.returncode == 0
            except Exception:
                ok = False
            GLib.idle_add(self._done, base, "Transkription fertig" if ok else "Transkription fehlgeschlagen")

        threading.Thread(target=work, daemon=True).start()
        if self._poll_id is None:
            self._poll_id = GLib.timeout_add(400, self._poll_progress)

    def _rename(self, base):
        try:
            meta = json.loads((RECORDINGS_DIR / f"{base}.meta.json").read_text())
        except Exception:
            meta = {}
        dlg = Adw.AlertDialog(heading="Umbenennen", body="Neuer Titel der Aufnahme:")
        entry = Gtk.Entry(hexpand=True)
        entry.set_text(meta.get("title", base))
        dlg.set_extra_child(entry)
        dlg.add_response("cancel", "Abbrechen")
        dlg.add_response("ok", "Speichern")
        dlg.set_response_appearance("ok", Adw.ResponseAppearance.SUGGESTED)
        dlg.set_default_response("ok")

        def on_resp(_d, resp):
            if resp != "ok":
                return
            new = entry.get_text().strip()
            recorder_call("rename", base, "--title", new)
            shown = new or base
            if self._detail.get("title_lbl"):
                self._detail["title_lbl"].set_label(shown)
            if self._detail.get("title_row"):
                self._detail["title_row"].set_subtitle(shown)
            if self._detail.get("page"):
                self._detail["page"].set_title(shown)
            self.refresh()
            self._toast("Umbenannt ✓")

        dlg.connect("response", on_resp)
        dlg.present(self.get_root())

    def _open_folder(self):
        Gio.AppInfo.launch_default_for_uri(GLib.filename_to_uri(str(RECORDINGS_DIR), None), None)

    def _delete(self, base, from_detail=False):
        dlg = Adw.AlertDialog(heading="Aufnahme löschen?",
                              body="Audio, Transkript und Notizen werden entfernt.")
        dlg.add_response("cancel", "Abbrechen")
        dlg.add_response("del", "Löschen")
        dlg.set_response_appearance("del", Adw.ResponseAppearance.DESTRUCTIVE)

        def on_resp(_d, resp):
            if resp != "del":
                return
            recorder_call("delete", base)
            if from_detail and self._detail_base == base:
                self.nav.pop()
            self.refresh()
            self._toast("Gelöscht")

        dlg.connect("response", on_resp)
        dlg.present(self.get_root())

    # ── audio playback (ffplay, no window) ───────────────────────────────────
    def _toggle_play(self, base, btn, lbl):
        if self._play_proc and self._play_proc.poll() is None:
            self._stop_play()
            return
        path = RECORDINGS_DIR / f"{base}.opus"
        if not path.exists():
            self._toast("Keine Audiodatei.")
            return
        try:
            self._play_proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
                preexec_fn=_die_with_parent)
        except FileNotFoundError:
            self._toast("ffplay nicht gefunden (ffmpeg installieren).")
            return
        self._play_start = GLib.get_monotonic_time() / 1e6
        btn.set_label("⏹ Stopp")
        self._play_timer = GLib.timeout_add(500, self._play_tick, btn, lbl)

    def _play_tick(self, btn, lbl):
        if not self._play_proc or self._play_proc.poll() is not None:
            self._reset_play_button(btn, lbl)
            self._play_proc = None
            self._play_timer = None
            return False
        el = GLib.get_monotonic_time() / 1e6 - self._play_start
        lbl.set_label(f"{int(el) // 60}:{int(el) % 60:02d}")
        return True

    def _reset_play_button(self, btn, lbl):
        try:
            btn.set_label("▶ Abspielen")
            lbl.set_label("")
        except Exception:
            pass

    def _stop_play(self):
        if self._play_timer is not None:
            GLib.source_remove(self._play_timer)
            self._play_timer = None
        if self._play_proc is not None:
            try:
                self._play_proc.terminate()
            except Exception:
                pass
            self._play_proc = None
        if self._detail.get("play_btn"):
            self._reset_play_button(self._detail["play_btn"], self._detail.get("play_lbl"))


class SettingsWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application):
        super().__init__(application=app, title="Whisper Dictation")
        install_app_css()
        self.set_default_size(940, 720)
        self.set_size_request(420, 480)
        self.config = load_config()
        self.device_options = detect_alsa_capture_devices()
        self._capturing = None          # (combo, opts_attr, button) while learning a key
        self._capture_ctrl = None

        self.toasts = Adw.ToastOverlay()
        self.set_content(self.toasts)

        toolbar = Adw.ToolbarView()
        self.toasts.set_child(toolbar)

        header = Adw.HeaderBar()
        toolbar.add_top_bar(header)

        self.stack = Adw.ViewStack()
        switcher = Adw.ViewSwitcher(policy=Adw.ViewSwitcherPolicy.WIDE)
        switcher.set_stack(self.stack)
        header.set_title_widget(switcher)
        toolbar.set_content(self.stack)

        self.save_button = Gtk.Button(label="Speichern")
        self.save_button.add_css_class("suggested-action")
        self.save_button.connect("clicked", self._on_save)
        header.pack_start(self.save_button)

        menu = Gio.Menu()
        menu.append("Diagnose", "win.diagnose")
        menu.append("Daemon starten", "win.start")
        menu.append("Daemon neu starten", "win.restart")
        menu.append("Daemon stoppen", "win.stop")
        menu.append("Log oeffnen", "win.log")
        menu.append("Ueber Whisper Dictation", "win.about")
        menu_button = Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu)
        header.pack_end(menu_button)
        for name, handler in (
            ("start", self._on_start), ("restart", self._on_restart),
            ("stop", self._on_stop), ("log", self._on_log),
            ("diagnose", self._on_diagnose), ("about", self._on_about),
        ):
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", handler)
            self.add_action(action)

        # ── Werkbank (Hauptansicht) ──────────────────────────────────────────
        self.workbench = WorkbenchView(toast_cb=self._toast)
        self.stack.add_titled_with_icon(
            self.workbench, "werkbank", "Werkbank", "audio-input-microphone-symbolic")

        # ── Rekorder (Langaufnahme: Vorlesungen / Calls) ─────────────────────
        self.recorder = RecorderView(toast_cb=self._toast)
        self.stack.add_titled_with_icon(
            self.recorder, "rekorder", "Rekorder", "media-record-symbolic")

        # ── Verlauf ──────────────────────────────────────────────────────────
        self._history_rows = []
        self.stack.add_titled_with_icon(
            self._build_history_page(), "verlauf", "Verlauf", "document-open-recent-symbolic")

        # ── Einstellungen (zweite Ansicht) ───────────────────────────────────
        page = Adw.PreferencesPage()
        self.stack.add_titled_with_icon(
            page, "settings", "Einstellungen", "applications-system-symbolic")

        # ── Status ───────────────────────────────────────────────────────────
        status_group = Adw.PreferencesGroup()
        page.add(status_group)
        self.status_row = Adw.ActionRow(title="Daemon")
        status_icon = Gtk.Image(icon_name="audio-input-microphone-symbolic")
        self.status_row.add_prefix(status_icon)
        status_group.add(self.status_row)

        # ── Erkennung ────────────────────────────────────────────────────────
        rec = Adw.PreferencesGroup(title="Erkennung")
        page.add(rec)
        self.model_row = self._combo("Modell", MODEL_OPTIONS, str(self.config["model"]))
        self.model_row.connect("notify::selected", lambda *_: self._update_model_hint())
        rec.add(self.model_row)
        self.language_row = self._combo(
            "Sprache", LANGUAGE_OPTIONS, str(self.config.get("language", "de")).lower()
        )
        rec.add(self.language_row)
        self.hotwords_row = Adw.EntryRow(title="Hotwords (Komma-getrennt)")
        self.hotwords_row.set_text(str(self.config.get("hotwords", "")))
        rec.add(self.hotwords_row)
        self.vad_row = Adw.SwitchRow(title="VAD", subtitle="Stille filtern (weniger Halluzinationen)")
        self.vad_row.set_active(bool(self.config.get("vad_filter", True)))
        rec.add(self.vad_row)
        self.voice_row = Adw.SwitchRow(
            title="Sprachbefehle",
            subtitle="neue Zeile, neuer Absatz, Doppelpunkt, Fragezeichen …",
        )
        self.voice_row.set_active(bool(self.config.get("voice_commands", False)))
        rec.add(self.voice_row)

        # ── Eingabe ──────────────────────────────────────────────────────────
        inp = Adw.PreferencesGroup(title="Eingabe")
        page.add(inp)
        self.mode_row = self._combo(
            "Modus", HOTKEY_MODE_OPTIONS, str(self.config.get("hotkey_mode", "double_tap"))
        )
        inp.add(self.mode_row)
        hotkey_cur = str(self.config["double_tap_key"])
        self._hotkey_opts = key_options(HOTKEY_OPTIONS, hotkey_cur)
        self.hotkey_row = self._combo("Aufnahme-Taste", self._hotkey_opts, hotkey_cur)
        inp.add(self.hotkey_row)
        inp.add(self._make_capture_row("hotkey_row", "_hotkey_opts"))
        self.double_tap_row = Adw.SpinRow.new_with_range(150, 1200, 10)
        self.double_tap_row.set_title("Double-Tap-Fenster (ms)")
        self.double_tap_row.set_value(float(self.config["double_tap_window_ms"]))
        inp.add(self.double_tap_row)
        self.paste_row = self._combo("Paste-Modus", PASTE_OPTIONS, str(self.config["paste_mode"]))
        inp.add(self.paste_row)
        self.max_record_row = Adw.SpinRow.new_with_range(15, 900, 5)
        self.max_record_row.set_title("Max. Aufnahme (s)")
        self.max_record_row.set_value(float(self.config["max_record_seconds"]))
        inp.add(self.max_record_row)
        self.sound_row = Adw.SwitchRow(title="Sound-Feedback", subtitle="Ton bei Start/Fertig")
        self.sound_row.set_active(bool(self.config.get("sound_cue", True)))
        inp.add(self.sound_row)
        self.clipboard_row = Adw.SwitchRow(
            title="Zwischenablage schonen",
            subtitle="Inhalt nach dem Einfügen wiederherstellen",
        )
        self.clipboard_row.set_active(bool(self.config.get("restore_clipboard", True)))
        inp.add(self.clipboard_row)
        self.history_row = Adw.SwitchRow(
            title="Verlauf speichern",
            subtitle="Diktate im Verlauf-Tab merken",
        )
        self.history_row.set_active(bool(self.config.get("save_history", True)))
        inp.add(self.history_row)

        # ── Audio + Erweitert ────────────────────────────────────────────────
        audio = Adw.PreferencesGroup(title="Audio")
        page.add(audio)
        self.device_row = self._combo("Mikrofon", self.device_options, str(self.config["record_device"]))
        audio.add(self.device_row)
        self.viz_row = self._combo("Live-Visualisierung", VISUALIZER_OPTIONS,
                                   str(self.config.get("audio_visualizer", "waves")))
        self.viz_row.set_subtitle("Während der Aufnahme in Werkbank und Rekorder: "
                                  "Wellen (Lautstärke-Verlauf), Balken oder aus.")
        self.viz_row.connect("notify::selected", self._on_visualizer_changed)
        audio.add(self.viz_row)

        # ── Textverbesserung (Ollama, optional) ──────────────────────────────
        llm = Adw.PreferencesGroup(
            title="Textverbesserung (Ollama)",
            description="Optionaler LLM-Schritt: entfernt Fuellwoerter, fixt Grammatik. "
                        "Kostet ~2-4 s extra. Standard: aus.",
        )
        page.add(llm)
        self.ollama_row = Adw.SwitchRow(
            title="Ollama-Nachbearbeitung",
            subtitle="Braucht laufenden Ollama-Server",
        )
        self.ollama_row.set_active(bool(self.config.get("ollama_postprocess", False)))
        llm.add(self.ollama_row)
        current_model = str(self.config.get("ollama_model", "qwen2.5:7b"))
        installed = self._installed_ollama_models()
        self._llm_model_opts = [
            (v, (("✓ " if v in installed else "") + lbl)) for v, lbl in LLM_MODEL_OPTIONS
        ]
        if current_model not in [v for v, _ in self._llm_model_opts]:
            self._llm_model_opts.append((current_model, f"{current_model}  (eigenes)"))
        self.ollama_model_row = self._combo("Modell", self._llm_model_opts, current_model)
        self.ollama_model_row.set_subtitle("Mehr Sterne = stärker, aber langsamer. Muss via 'ollama pull' installiert sein.")
        llm.add(self.ollama_model_row)
        toggle_cur = str(self.config.get("llm_toggle_key", ""))
        self._llm_toggle_opts = key_options(LLM_TOGGLE_OPTIONS, toggle_cur)
        self.llm_toggle_row = self._combo(
            "Umschalt-Taste (Doppel-Tap)", self._llm_toggle_opts, toggle_cur,
        )
        self.llm_toggle_row.set_subtitle("Schaltet Cleanup an/aus. Muss sich von der Aufnahme-Taste unterscheiden.")
        llm.add(self.llm_toggle_row)
        llm.add(self._make_capture_row("llm_toggle_row", "_llm_toggle_opts"))

        command_cur = str(self.config.get("command_key", ""))
        self._command_opts = key_options(LLM_TOGGLE_OPTIONS, command_cur)
        self.command_row = self._combo("Befehl-Taste (markierten Text bearbeiten)", self._command_opts, command_cur)
        self.command_row.set_subtitle("Doppel-Tap, dann Anweisung sprechen → ersetzt die Markierung. Funktioniert in Textfeldern, nicht im Terminal.")
        llm.add(self.command_row)
        llm.add(self._make_capture_row("command_row", "_command_opts"))

        adv = Adw.PreferencesGroup(
            title="Kontext (Initial Prompt)",
            description="Lenkt die Erkennung Richtung deiner Begriffe — wird nicht "
                        "mitgeschrieben. Leer lassen = aus.",
        )
        page.add(adv)
        prompt_scroller = Gtk.ScrolledWindow(min_content_height=84)
        prompt_scroller.add_css_class("card")
        self.prompt_view = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD_CHAR,
            top_margin=8, bottom_margin=8, left_margin=8, right_margin=8,
        )
        self.prompt_view.get_buffer().set_text(str(self.config.get("initial_prompt", "")))
        prompt_scroller.set_child(self.prompt_view)
        adv.add(prompt_scroller)

        self._update_model_hint()
        self._refresh_status()

        # Open on the Werkbank; the Speichern button only matters in settings.
        self.stack.set_visible_child_name("werkbank")
        self.stack.connect("notify::visible-child-name", self._on_view_changed)
        self._on_view_changed()
        # Stop background meter/playback processes when the window is closed.
        self.connect("close-request", self._on_close)

    def _on_close(self, *_a) -> bool:
        self.recorder.on_hidden()
        return False

    def _on_visualizer_changed(self, *_a) -> None:
        mode = self._combo_value(self.viz_row, VISUALIZER_OPTIONS)
        cfg = load_config()
        cfg["audio_visualizer"] = mode
        save_config(cfg)
        self.config["audio_visualizer"] = mode
        self.recorder.set_visualizer_mode(mode)        # live in the recorder tab
        self.workbench._viz.set_mode(mode)             # next dictation uses it
        self._toast({"waves": "Wellen", "bar": "Balken", "none": "Aus"}.get(mode, mode))

    def _on_view_changed(self, *_a) -> None:
        name = self.stack.get_visible_child_name()
        self.save_button.set_visible(name == "settings")
        if name == "verlauf":
            self._refresh_history()
        if name == "rekorder":
            self.recorder.on_shown()
        else:
            self.recorder.on_hidden()  # stop live meters / playback when leaving

    # ── Verlauf (history) ───────────────────────────────────────────────────────

    def _build_history_page(self) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage()
        self._history_group = Adw.PreferencesGroup(
            title="Verlauf der Diktate",
            description="Zuletzt eingesprochene Texte — kopieren oder in der Werkbank weiterbearbeiten.",
        )
        clear = Gtk.Button(label="Leeren", valign=Gtk.Align.CENTER)
        clear.add_css_class("flat")
        clear.connect("clicked", self._clear_history)
        self._history_group.set_header_suffix(clear)
        page.add(self._history_group)
        return page

    def _refresh_history(self) -> None:
        for row in self._history_rows:
            self._history_group.remove(row)
        self._history_rows = []
        entries = read_history(100)
        if not entries:
            row = Adw.ActionRow(title="Noch keine Diktate")
            self._history_group.add(row)
            self._history_rows.append(row)
            return
        for entry in reversed(entries):
            text = str(entry.get("text", "")).strip()
            row = Adw.ActionRow(
                title=(text[:90] + "…") if len(text) > 90 else (text or "(leer)"),
                subtitle=format_ts(entry.get("ts")),
            )
            copy = Gtk.Button(icon_name="edit-copy-symbolic", valign=Gtk.Align.CENTER)
            copy.add_css_class("flat")
            copy.set_tooltip_text("Kopieren")
            copy.connect("clicked", lambda _b, t=text: self._copy_text(t))
            load = Gtk.Button(label="In Werkbank", valign=Gtk.Align.CENTER)
            load.add_css_class("flat")
            load.connect("clicked", lambda _b, t=text: self._load_to_workbench(t))
            row.add_suffix(copy)
            row.add_suffix(load)
            self._history_group.add(row)
            self._history_rows.append(row)

    def _copy_text(self, text: str) -> None:
        subprocess.run(["wl-copy"], input=text.encode("utf-8"), check=False)
        self._toast("In Zwischenablage kopiert ✓")

    def _load_to_workbench(self, text: str) -> None:
        self.workbench._set_text(text)
        self.stack.set_visible_child_name("werkbank")

    def _clear_history(self, *_a) -> None:
        try:
            HISTORY_FILE.write_text("", encoding="utf-8")
        except Exception:
            pass
        self._refresh_history()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _combo(self, title: str, options: list[tuple[str, str]], current: str) -> Adw.ComboRow:
        row = Adw.ComboRow(title=title)
        row.set_model(Gtk.StringList.new([label for _, label in options]))
        idx = next((i for i, (value, _) in enumerate(options) if value == current), 0)
        row.set_selected(idx)
        return row

    @staticmethod
    def _combo_value(row: Adw.ComboRow, options: list[tuple[str, str]]) -> str:
        return options[row.get_selected()][0]

    # ── Key capture (press a key to set it) ─────────────────────────────────────

    def _make_capture_row(self, combo_attr: str, opts_attr: str) -> Adw.ActionRow:
        row = Adw.ActionRow(title="… oder Taste drücken zum Festlegen")
        btn = Gtk.Button(label="🎯 Taste erfassen", valign=Gtk.Align.CENTER)
        btn.add_css_class("flat")
        btn.connect("clicked", lambda *_: self._start_capture(combo_attr, opts_attr, btn))
        row.add_suffix(btn)
        row.set_activatable_widget(btn)
        return row

    def _start_capture(self, combo_attr: str, opts_attr: str, btn: Gtk.Button) -> None:
        if self._capturing is not None:
            return
        self._capturing = (combo_attr, opts_attr, btn)
        btn.set_label("… drück eine Taste (Esc = Abbruch)")
        ctrl = Gtk.EventControllerKey()
        ctrl.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        ctrl.connect("key-pressed", self._on_capture_key)
        self.add_controller(ctrl)
        self._capture_ctrl = ctrl

    def _on_capture_key(self, _ctrl, keyval, keycode, _state) -> bool:
        if self._capturing is None:
            return False
        combo_attr, opts_attr, btn = self._capturing
        if keyval == Gdk.KEY_Escape:
            self._end_capture(btn)
            return True
        evdev = keycode - 8  # GTK/X hardware keycode -> evdev code
        name = Gdk.keyval_name(keyval) or f"Taste{evdev}"
        value = f"code:{evdev}:{name}"
        combo = getattr(self, combo_attr)
        opts = getattr(self, opts_attr)
        vals = [v for v, _ in opts]
        if value not in vals:
            opts.append((value, key_label(value)))
            combo.get_model().append(key_label(value))
            vals.append(value)
        combo.set_selected(vals.index(value))
        self._end_capture(btn)
        self._toast(f"Taste erfasst: {key_label(value)} — jetzt Speichern")
        return True

    def _end_capture(self, btn: Gtk.Button) -> None:
        btn.set_label("🎯 Taste erfassen")
        if self._capture_ctrl is not None:
            self.remove_controller(self._capture_ctrl)
            self._capture_ctrl = None
        self._capturing = None

    def _update_model_hint(self) -> None:
        model = self._combo_value(self.model_row, MODEL_OPTIONS)
        self.model_row.set_subtitle(MODEL_HINTS.get(model, ""))

    def _refresh_status(self) -> None:
        # Off the main thread: the status check spawns a subprocess (~30 ms) and
        # is called at startup + after every daemon action.
        def work():
            running = daemon_running()
            GLib.idle_add(self.status_row.set_subtitle, "● läuft" if running else "○ gestoppt")
        threading.Thread(target=work, daemon=True).start()

    def _prompt_text(self) -> str:
        buf = self.prompt_view.get_buffer()
        return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).strip()

    def _toast(self, text: str) -> None:
        self.toasts.add_toast(Adw.Toast.new(text))

    def _config_from_form(self) -> dict:
        # Start from the on-disk config so keys edited elsewhere (e.g. the
        # Rekorder tab's source/auto choice) are preserved on save. The form
        # fields below overwrite every key this page owns.
        config = load_config()
        config.update({
            "model": self._combo_value(self.model_row, MODEL_OPTIONS),
            "language": self._combo_value(self.language_row, LANGUAGE_OPTIONS),
            "hotwords": self.hotwords_row.get_text().strip(),
            "vad_filter": bool(self.vad_row.get_active()),
            "voice_commands": bool(self.voice_row.get_active()),
            "sound_cue": bool(self.sound_row.get_active()),
            "restore_clipboard": bool(self.clipboard_row.get_active()),
            "save_history": bool(self.history_row.get_active()),
            "hotkey_mode": self._combo_value(self.mode_row, HOTKEY_MODE_OPTIONS),
            "double_tap_key": self._combo_value(self.hotkey_row, self._hotkey_opts),
            "double_tap_window_ms": int(self.double_tap_row.get_value()),
            "paste_mode": self._combo_value(self.paste_row, PASTE_OPTIONS),
            "max_record_seconds": int(self.max_record_row.get_value()),
            "record_device": self._combo_value(self.device_row, self.device_options),
            "audio_visualizer": self._combo_value(self.viz_row, VISUALIZER_OPTIONS),
            "initial_prompt": self._prompt_text(),
            "ollama_postprocess": bool(self.ollama_row.get_active()),
            "ollama_model": self._combo_value(self.ollama_model_row, self._llm_model_opts),
            "llm_toggle_key": self._combo_value(self.llm_toggle_row, self._llm_toggle_opts),
            "command_key": self._combo_value(self.command_row, self._command_opts),
        })
        return config

    def _run_daemon(self, arg: str) -> tuple[int, str]:
        result = subprocess.run(
            [str(DAEMON_SCRIPT), arg], capture_output=True, text=True, check=False,
        )
        return result.returncode, (result.stdout + result.stderr).strip()

    @staticmethod
    def _ollama_list() -> tuple[bool, set]:
        """Return (server_up, set_of_installed_model_names)."""
        try:
            out = subprocess.run(["ollama", "list"], capture_output=True,
                                 text=True, timeout=3, check=False)
            if out.returncode != 0:
                return False, set()
            return True, {ln.split()[0] for ln in out.stdout.splitlines()[1:] if ln.split()}
        except Exception:
            return False, set()

    def _installed_ollama_models(self) -> set:
        return self._ollama_list()[1]

    def _ollama_model_installed(self, model: str) -> bool | None:
        up, installed = self._ollama_list()
        return (model in installed) if up else None

    # ── Actions ────────────────────────────────────────────────────────────────

    # Changing these keys re-grabs the evdev listener, which only happens at
    # startup -> a full restart is needed. Everything else applies live.
    RESTART_KEYS = ("double_tap_key", "llm_toggle_key", "command_key")

    def _on_save(self, _button: Gtk.Button) -> None:
        old, new = self.config, self._config_from_form()
        self.config = new
        save_config(new)
        needs_restart = any(old.get(k) != new.get(k) for k in self.RESTART_KEYS)
        if needs_restart:
            code, output = self._run_daemon("--restart")
            msg = "Gespeichert und Daemon neu gestartet." if code == 0 else f"Neustart-Fehler: {output}"
        else:
            code, output = self._run_daemon("--reload")
            msg = "Gespeichert — Aenderungen sind aktiv." if code == 0 else f"Reload-Fehler: {output}"
        if new.get("ollama_postprocess") and self._ollama_model_installed(new.get("ollama_model", "")) is False:
            msg += f"  ⚠ Modell nicht installiert: ollama pull {new['ollama_model']}"
        self._toast(msg)
        self._refresh_status()

    def _on_start(self, *_a) -> None:
        code, output = self._run_daemon("--restart")
        self._toast("Daemon gestartet." if code == 0 else f"Start fehlgeschlagen: {output}")
        self._refresh_status()

    def _on_restart(self, *_a) -> None:
        code, output = self._run_daemon("--restart")
        self._toast("Daemon neu gestartet." if code == 0 else f"Neustart fehlgeschlagen: {output}")
        self._refresh_status()

    def _on_stop(self, *_a) -> None:
        code, output = self._run_daemon("--stop")
        self._toast("Daemon gestoppt." if code == 0 else f"Stop-Fehler: {output}")
        self._refresh_status()

    def _on_log(self, *_a) -> None:
        if LOG_FILE.exists():
            Gio.AppInfo.launch_default_for_uri(GLib.filename_to_uri(str(LOG_FILE), None), None)
        else:
            self._toast("Noch keine Logdatei vorhanden.")

    def _on_diagnose(self, *_a) -> None:
        import re
        device = backend = "?"
        try:
            log = LOG_FILE.read_text(errors="ignore")
            dev = re.findall(r"using device=(\w+)", log)
            be = re.findall(r"backend=(\w+)", log)
            device = dev[-1] if dev else "?"
            backend = be[-1] if be else "?"
        except Exception:
            pass
        up, installed = self._ollama_list()
        model = self._combo_value(self.ollama_model_row, self._llm_model_opts)
        body = (
            f"Daemon: {'läuft' if daemon_running() else 'gestoppt'}\n"
            f"Backend: {backend}\n"
            f"Gerät: {device}\n"
            f"Ollama-Server: {'läuft' if up else 'aus'}\n"
            f"Cleanup-Modell ({model}): "
            f"{'installiert' if model in installed else ('nicht installiert' if up else '?')}"
        )
        dlg = Adw.AlertDialog(heading="Diagnose", body=body)
        dlg.add_response("ok", "OK")
        dlg.present(self)

    def _on_about(self, *_a) -> None:
        about = Adw.AboutDialog(
            application_name="Whisper Dictation",
            application_icon="io.voelzke.WhisperDictation",
            version="1.0",
            developer_name="Sam Völzke",
            comments="Lokales Sprache-zu-Text-Diktat für GNOME/Wayland — "
                     "Whisper auf der Intel-GPU via OpenVINO.",
            license_type=Gtk.License.MIT_X11,
        )
        about.present(self)


class WhisperDictationApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id="io.voelzke.WhisperDictation")

    def do_activate(self) -> None:
        window = self.props.active_window
        if window is None:
            window = SettingsWindow(self)
        window.present()


def main() -> int:
    app = WhisperDictationApp()
    return app.run([])


if __name__ == "__main__":
    raise SystemExit(main())
