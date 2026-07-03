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
import sys
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
# Config schema, load/save (atomic) and hotkey labels live in the shared core.
sys.path.insert(0, str(PROJECT_ROOT / "dictation"))
from common import (  # noqa: E402
    DEFAULT_CONFIG,
    DICTATION_MODES,
    HISTORY_FILE,
    HOTKEY_SPECS,
    IPC_SOCKET,
    LOG_FILE,
    RECORDINGS_DIR,
    dictionary_terms,
    key_label,
    load_config,
    save_config,
)

DICT_MODE_OPTIONS = [(key, label) for key, (label, _prompt) in DICTATION_MODES.items()]


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

HOTKEY_MODE_OPTIONS = [
    ("double_tap", "Doppel-Tap (start/stopp)"),
    ("push_to_talk", "Push-to-Talk (halten)"),
]

# Derived from the daemon's key table so GUI and daemon can never drift.
HOTKEY_OPTIONS = [(name, spec[0]) for name, spec in HOTKEY_SPECS.items()]

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
_FALLBACK_ACCENT = (0.21, 0.52, 0.89)  # GNOME blue #3584e4
_accent_cache: tuple[float, float, float] | None = None
_accent_watching = False


def _invalidate_accent(*_a) -> None:
    global _accent_cache
    _accent_cache = None


def accent_rgb() -> tuple[float, float, float]:
    """The user's system accent color (GNOME setting), with a blue fallback.

    Cached: the meters redraw ~40×/s for hours, and the value only changes
    when the user picks a new accent (invalidated via notify::accent-color).
    """
    global _accent_cache, _accent_watching
    if _accent_cache is not None:
        return _accent_cache
    try:
        manager = Adw.StyleManager.get_default()
        rgba = manager.get_accent_color_rgba()
        _accent_cache = (rgba.red, rgba.green, rgba.blue)
        if not _accent_watching:
            manager.connect("notify::accent-color", _invalidate_accent)
            _accent_watching = True
    except Exception:
        _accent_cache = _FALLBACK_ACCENT
    return _accent_cache


def copy_to_clipboard(text: str) -> bool:
    """Copy via wl-copy; False if unavailable, so callers never show a false
    'kopiert ✓' toast."""
    try:
        subprocess.run(["wl-copy"], input=(text or "").encode("utf-8"), check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def combo_row(title: str, options: list, current: str) -> Adw.ComboRow:
    row = Adw.ComboRow(title=title)
    row.set_model(Gtk.StringList.new([label for _, label in options]))
    row.set_selected(next((i for i, (v, _) in enumerate(options) if v == current), 0))
    return row


def combo_value(row: Adw.ComboRow, options: list) -> str:
    i = row.get_selected()
    return options[i][0] if 0 <= i < len(options) else ""


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
          padding: 4px;
        }
        /* Hero record buttons (GNOME-Sound-Recorder-style) */
        .record-circle {
          min-width: 64px;
          min-height: 64px;
          -gtk-icon-size: 26px;
        }
        .record-circle-small {
          min-width: 42px;
          min-height: 42px;
          -gtk-icon-size: 17px;
        }
        /* Idle record button: soft red tint (red dot = record, universally
           understood); while recording the button switches to solid
           destructive red with a stop icon. */
        .record-idle {
          color: @error_color;
          background: alpha(@error_color, 0.1);
        }
        .record-idle:hover {
          background: alpha(@error_color, 0.18);
        }
        .record-idle:active {
          background: alpha(@error_color, 0.26);
        }
        /* Ghost title entry in the hero: no gray box until it's used */
        entry.inline-title {
          background: transparent;
          border: none;
          box-shadow: none;
        }
        entry.inline-title:focus-within {
          background: alpha(@window_fg_color, 0.05);
          border-radius: 8px;
        }
        /* Preset chips: quiet pills instead of naked text */
        .chip {
          border-radius: 9999px;
          padding: 3px 12px;
          background: alpha(@window_fg_color, 0.07);
        }
        .chip:hover {
          background: alpha(@window_fg_color, 0.13);
        }
        .hero-timer {
          font-size: 1.5em;
          font-weight: 300;
        }
        .hero-hint {
          font-size: 0.85em;
        }
        /* List row actions appear on hover/keyboard focus only (calm rows) */
        row .row-actions {
          opacity: 0;
          transition: opacity 150ms ease;
        }
        row:hover .row-actions,
        row:focus-within .row-actions {
          opacity: 1;
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


def pw_record_cmd(device: str) -> list[str]:
    """Build the pw-record command for a capture target.

    pw-record cannot resolve Pulse-style '<sink>.monitor' names — it silently
    falls back to the DEFAULT SOURCE (the mic!), so a 'System' meter would
    show microphone waves. Monitors are captured from the sink node itself
    via stream.capture.sink=true (verified against a test tone).
    """
    cmd = ["pw-record"]
    if device.endswith(".monitor"):
        cmd += ["--target", device[: -len(".monitor")], "-P", "stream.capture.sink=true"]
    else:
        cmd += ["--target", device]
    return cmd + ["--rate", "16000", "--channels", "1", "--format", "s16", "-"]


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


def fmt_size(num_bytes: int) -> str:
    if num_bytes < 1024 * 1024:
        return f"{max(1, round(num_bytes / 1024))} KB"
    mb = num_bytes / (1024 * 1024)
    return f"{mb:.0f} MB" if mb >= 10 else f"{mb:.1f} MB"


def date_bucket(created_iso: str) -> str:
    """Group label for the recordings list (Heute/Gestern/Diese Woche/Älter)."""
    import datetime as _dt
    try:
        d = _dt.datetime.fromisoformat(created_iso).date()
    except (ValueError, TypeError):
        return "Älter"
    today = _dt.date.today()
    if d == today:
        return "Heute"
    if d == today - _dt.timedelta(days=1):
        return "Gestern"
    if d >= today - _dt.timedelta(days=6):
        return "Diese Woche"
    return "Älter"


def key_options(base: list, current: str) -> list:
    """Return base options, appending the current key if it's a captured one."""
    opts = list(base)
    if current and current not in [v for v, _ in opts]:
        opts.append((current, key_label(current)))
    return opts


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
        r, g, b = accent_rgb()
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
                    pw_record_cmd(device),
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

        # Hero: one big circular record button, status underneath.
        hero = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8,
                       halign=Gtk.Align.CENTER, margin_top=4)
        self.rec_btn = Gtk.Button(icon_name="media-record-symbolic",
                                  halign=Gtk.Align.CENTER)
        self.rec_btn.add_css_class("circular")
        self.rec_btn.add_css_class("record-circle")
        self.rec_btn.add_css_class("record-idle")
        self.rec_btn.set_tooltip_text("Aufnehmen (Strg+R)")
        self.rec_btn.connect("clicked", self._toggle_record)
        hero.append(self.rec_btn)
        status_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6,
                             halign=Gtk.Align.CENTER)
        self.spinner = Gtk.Spinner(visible=False)
        status_row.append(self.spinner)
        self.status = Gtk.Label(label="Bereit")
        self.status.add_css_class("dim-label")
        status_row.append(self.status)
        hero.append(status_row)
        box.append(hero)

        # Live volume visualization while dictating (waves / bar / off).
        self._viz = AudioVisualizer(mode=str(load_config().get("audio_visualizer", "waves")), height=40)
        self._viz.set_visible(False)
        box.append(self._viz)
        self._meter = LevelMeter(self._viz.push)

        # Editor with its own compact toolbar (title + clear/copy).
        editor_head = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        editor_title = Gtk.Label(label="Text", xalign=0, hexpand=True)
        editor_title.add_css_class("heading")
        editor_head.append(editor_title)
        clear_btn = Gtk.Button(icon_name="edit-clear-all-symbolic", valign=Gtk.Align.CENTER)
        clear_btn.add_css_class("flat")
        clear_btn.set_tooltip_text("Leeren")
        clear_btn.connect("clicked", lambda *_: self._set_text(""))
        editor_head.append(clear_btn)
        copy_btn = Gtk.Button(icon_name="edit-copy-symbolic", valign=Gtk.Align.CENTER)
        copy_btn.add_css_class("flat")
        copy_btn.set_tooltip_text("Kopieren")
        copy_btn.connect("clicked", self._copy)
        editor_head.append(copy_btn)
        box.append(editor_head)

        scroller = Gtk.ScrolledWindow(vexpand=True)
        scroller.add_css_class("card")
        scroller.add_css_class("editor-card")
        self.text_view = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD_CHAR,
            top_margin=12, bottom_margin=12, left_margin=12, right_margin=12,
        )
        self.text_view.add_css_class("editor-view")
        scroller.set_child(self.text_view)
        # Placeholder in the empty editor (vanishes with the first character).
        overlay = Gtk.Overlay()
        overlay.set_child(scroller)
        self._placeholder = Gtk.Label(
            label="Diktiere mit dem Aufnahme-Knopf –\noder tippe einfach los …",
            halign=Gtk.Align.CENTER, valign=Gtk.Align.CENTER,
            justify=Gtk.Justification.CENTER,
        )
        self._placeholder.add_css_class("dim-label")
        self._placeholder.set_can_target(False)  # clicks fall through to the editor
        overlay.add_overlay(self._placeholder)
        self.text_view.get_buffer().connect(
            "changed",
            lambda buf: self._placeholder.set_visible(buf.get_char_count() == 0),
        )
        box.append(overlay)

        # Adw.WrapBox (libadwaita >= 1.7) wraps unevenly sized chips naturally;
        # FlowBox stays as fallback for older libadwaita.
        if hasattr(Adw, "WrapBox"):
            presets = Adw.WrapBox(child_spacing=4, line_spacing=4)
        else:
            presets = Gtk.FlowBox(
                selection_mode=Gtk.SelectionMode.NONE,
                column_spacing=4, row_spacing=4, max_children_per_line=12,
                halign=Gtk.Align.START,
            )
        for label, instruction in WB_PRESETS:
            chip = Gtk.Button(label=label)
            chip.add_css_class("flat")
            chip.add_css_class("chip")
            chip.connect("clicked", lambda _b, i=instruction: self._do_instruct(i))
            presets.append(chip)
        box.append(presets)

        instr_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.instr = Gtk.Entry(hexpand=True)
        self.instr.set_placeholder_text("Anweisung an die KI … (formaler · zusammenfassen · auf Englisch)")
        self.instr.connect("activate", self._run_instruction)
        instr_row.append(self.instr)
        self.send_btn = Gtk.Button()
        self.send_btn.set_child(Adw.ButtonContent(icon_name="document-send-symbolic",
                                                  label="Ausführen"))
        self.send_btn.connect("clicked", self._run_instruction)
        instr_row.append(self.send_btn)
        box.append(instr_row)

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

    def _set_status(self, text: str, error: bool = False, busy: bool = False) -> None:
        """One status line: gray for state, red for errors, spinner while busy."""
        self.status.set_text(text)
        if error:
            self.status.remove_css_class("dim-label")
            self.status.add_css_class("error")
        else:
            self.status.remove_css_class("error")
            self.status.add_css_class("dim-label")
        self.spinner.set_visible(busy)
        self.spinner.set_spinning(busy)

    def _toggle_record(self, *_a) -> None:
        if self.rec_proc is None:
            handle = tempfile.NamedTemporaryFile(
                prefix="whisper-werkbank-", suffix=".wav", delete=False,
            )
            handle.close()
            self.rec_wav = handle.name
            device = str(load_config().get("record_device", "default"))
            try:
                self.rec_proc = subprocess.Popen([
                    "arecord", "-q", "-D", device, "-f", "S16_LE",
                    "-r", "16000", "-c", "1", "-t", "wav", self.rec_wav,
                ])
            except Exception as exc:
                self._set_status(f"arecord-Fehler: {exc}", error=True)
                return
            self.rec_btn.set_icon_name("media-playback-stop-symbolic")
            self.rec_btn.set_tooltip_text("Stopp (Strg+R)")
            self.rec_btn.remove_css_class("record-idle")
            self.rec_btn.add_css_class("destructive-action")
            self._set_status("Aufnahme läuft …")
            self._start_viz()
            return

        try:
            self.rec_proc.send_signal(signal.SIGINT)
            self.rec_proc.wait(timeout=3)
        except Exception:
            pass
        self.rec_proc = None
        self._stop_viz()
        self.rec_btn.set_icon_name("media-record-symbolic")
        self.rec_btn.set_tooltip_text("Aufnehmen (Strg+R)")
        self.rec_btn.remove_css_class("destructive-action")
        self.rec_btn.add_css_class("record-idle")
        self._set_status("Transkribiere …", busy=True)
        wav = self.rec_wav

        def work():
            try:
                r = ipc_call({"cmd": "transcribe", "wav": wav})
            except Exception as exc:
                r = {"error": str(exc)}
            finally:
                try:
                    os.unlink(wav)
                except OSError:
                    pass
            GLib.idle_add(self._after_transcribe, r)
        threading.Thread(target=work, daemon=True).start()

    def _after_transcribe(self, r: dict) -> bool:
        if "error" in r:
            self._set_status(f"Fehler: {r['error']}", error=True)
            return False
        text = str(r.get("text", "")).strip()
        cur = self._text()
        joined = (cur + " " + text).strip() if cur else text
        self._set_text(joined)
        self._set_status("Bereit" if text else "Nichts erkannt")
        return False

    def _run_instruction(self, *_a) -> None:
        self._do_instruct(self.instr.get_text().strip())

    def _do_instruct(self, instruction: str) -> None:
        text = self._text()
        if not text:
            self._set_status("Erst etwas aufnehmen oder eingeben.")
            return
        if not instruction:
            return
        self.send_btn.set_sensitive(False)
        self._set_status("KI arbeitet …", busy=True)

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
            self._set_status(f"Fehler: {r['error']}", error=True)
            return False
        self._set_text(str(r.get("text", "")).strip())
        self.instr.set_text("")
        self._set_status("Bereit")
        return False

    def _copy(self, *_a) -> None:
        ok = copy_to_clipboard(self._text())
        if self._toast_cb:
            self._toast_cb("In Zwischenablage kopiert ✓" if ok else "Kopieren fehlgeschlagen (wl-copy fehlt?)")


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
        self._media = None              # native detail player (Gtk.MediaFile)
        self._qa_busy = False           # one transcript question at a time
        # inline row preview (one at a time)
        self._preview_media = None
        self._preview_base = None
        self._preview_btn = None

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
    _combo = staticmethod(combo_row)
    _cv = staticmethod(combo_value)

    def _rec_button_state(self, recording: bool, saving: bool = False) -> None:
        if saving:
            self.rec_btn.set_icon_name("media-playback-stop-symbolic")
            self.rec_btn.set_tooltip_text("Speichere …")
            return
        if recording:
            self.rec_btn.set_icon_name("media-playback-stop-symbolic")
            self.rec_btn.set_tooltip_text("Aufnahme stoppen")
            self.rec_btn.remove_css_class("record-idle")
            self.rec_btn.add_css_class("destructive-action")
        else:
            self.rec_btn.set_icon_name("media-record-symbolic")
            self.rec_btn.set_tooltip_text("Aufnahme starten")
            self.rec_btn.remove_css_class("destructive-action")
            self.rec_btn.add_css_class("record-idle")

    def _pause_button_state(self, paused: bool) -> None:
        self.pause_btn.set_icon_name(
            "media-playback-start-symbolic" if paused else "media-playback-pause-symbolic")
        self.pause_btn.set_tooltip_text("Fortsetzen" if paused else "Pause")

    def _toast(self, t):
        if self._toast_cb:
            self._toast_cb(t)

    def _persist(self, k, v):
        cfg = load_config()
        cfg[k] = v
        save_config(cfg)

    def _copy(self, text):
        ok = copy_to_clipboard(text or "")
        self._toast("In Zwischenablage kopiert ✓" if ok else "Kopieren fehlgeschlagen (wl-copy fehlt?)")

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

        # Hero: big circular record button + pause + timer, source toggle and
        # title underneath (GNOME-Sound-Recorder-style, no form rows).
        hero = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12,
                       halign=Gtk.Align.CENTER, margin_top=4)
        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16,
                           halign=Gtk.Align.CENTER)
        self.pause_btn = Gtk.Button(icon_name="media-playback-pause-symbolic",
                                    valign=Gtk.Align.CENTER)
        self.pause_btn.add_css_class("circular")
        self.pause_btn.add_css_class("record-circle-small")
        self.pause_btn.set_tooltip_text("Pause")
        self.pause_btn.set_visible(False)
        self.pause_btn.connect("clicked", self._toggle_pause)
        controls.append(self.pause_btn)
        self.rec_btn = Gtk.Button(icon_name="media-record-symbolic",
                                  valign=Gtk.Align.CENTER)
        self.rec_btn.add_css_class("circular")
        self.rec_btn.add_css_class("record-circle")
        self.rec_btn.add_css_class("record-idle")
        self.rec_btn.set_tooltip_text("Aufnahme starten")
        self.rec_btn.connect("clicked", self._toggle_record)
        controls.append(self.rec_btn)
        self.timer_label = Gtk.Label(label="", valign=Gtk.Align.CENTER)
        self.timer_label.add_css_class("hero-timer")
        self.timer_label.add_css_class("numeric")
        controls.append(self.timer_label)
        hero.append(controls)

        current_source = str(cfg.get("recorder_source", "both"))
        if hasattr(Adw, "ToggleGroup"):
            # One-click switching between call (Mic+System) and lecture (Mic).
            self._source_toggle = Adw.ToggleGroup(halign=Gtk.Align.CENTER)
            self._source_toggle.add_css_class("flat")
            self._source_toggle.add_css_class("round")
            for value, label in (("both", "Mic + System"), ("system", "System"), ("mic", "Mic")):
                toggle = Adw.Toggle(label=label)
                toggle.set_name(value)
                self._source_toggle.add(toggle)
            self._source_toggle.set_active_name(current_source)
            self._source_toggle.connect("notify::active", self._on_source_changed)
            hero.append(self._source_toggle)
            self.source_row = None
        else:
            self._source_toggle = None
        # Ghost entry: reads as a dim caption, becomes a field on focus.
        self.title_row = Gtk.Entry(halign=Gtk.Align.CENTER, width_chars=30, xalign=0.5)
        self.title_row.add_css_class("inline-title")
        self.title_row.set_placeholder_text("Titel (optional)")
        self.title_row.set_tooltip_text(
            "Wird laufend gespeichert — ein Absturz kostet höchstens Sekunden.")
        hero.append(self.title_row)
        outer.append(hero)

        opt_group = Adw.PreferencesGroup()
        outer.append(opt_group)
        opt = Adw.ExpanderRow(title="Optionen", subtitle="Geräte, Modell, Qualität, Sprache, Chunk-Länge")
        opt_group.add(opt)
        if self._source_toggle is None:
            self.source_row = self._combo("Quelle", REC_SOURCE_OPTIONS, current_source)
            self.source_row.connect("notify::selected", self._on_source_changed)
            opt.add_row(self.source_row)
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
        self.auto_title_row = Adw.SwitchRow(
            title="Titel automatisch vergeben",
            subtitle="Nach der Transkription schlägt die KI einen kurzen Titel vor.")
        self.auto_title_row.set_active(bool(cfg.get("recorder_auto_title", True)))
        self.auto_title_row.connect(
            "notify::active",
            lambda *_: self._persist("recorder_auto_title", bool(self.auto_title_row.get_active())))
        opt.add_row(self.auto_title_row)
        vault = str(cfg.get("obsidian_vault", "")).strip()
        self.vault_row = Adw.ActionRow(
            title="Obsidian-Vault",
            subtitle=vault or "Nicht gesetzt — wird beim ersten Export gewählt")
        vault_btn = Gtk.Button(label="Wählen …", valign=Gtk.Align.CENTER)
        vault_btn.add_css_class("flat")
        vault_btn.connect("clicked", self._pick_vault)
        self.vault_row.add_suffix(vault_btn)
        opt.add_row(self.vault_row)

        viz_mode = str(cfg.get("audio_visualizer", "waves"))
        self._meters_group = Adw.PreferencesGroup(
            title="Live-Pegel", description="Wird während der Aufnahme angezeigt.")
        outer.append(self._meters_group)
        self._mic_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._mic_box.append(Gtk.Label(label="Mikrofon", xalign=0, css_classes=["dim-label"]))
        self._mic_viz = AudioVisualizer(mode=viz_mode)
        self._mic_box.append(self._mic_viz)
        self._meters_group.add(self._mic_box)
        self._sys_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, margin_top=8)
        self._sys_box.append(Gtk.Label(label="System-Ton", xalign=0, css_classes=["dim-label"]))
        self._sys_viz = AudioVisualizer(mode=viz_mode)
        self._sys_box.append(self._sys_viz)
        self._meters_group.add(self._sys_box)
        self._meters_group.set_visible(False)

        # Recordings are grouped by date (Heute/Gestern/…) at refresh time.
        self._list_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        outer.append(self._list_container)

        self._empty_status = Adw.StatusPage(
            icon_name="audio-input-microphone-symbolic",
            title="Noch keine Aufnahmen",
            description="Starte oben eine Aufnahme — sie wird laufend gespeichert "
                        "und erscheint dann hier.",
        )
        self._empty_status.set_visible(False)
        outer.append(self._empty_status)

        self.refresh()
        return scroller

    def _current_source(self) -> str:
        if self._source_toggle is not None:
            return self._source_toggle.get_active_name() or "both"
        return self._cv(self.source_row, REC_SOURCE_OPTIONS)

    def _on_source_changed(self, *_):
        self._persist("recorder_source", self._current_source())
        if self._meters_on:
            self._start_meters()

    def _pick_vault(self, *_a):
        dialog = Gtk.FileDialog(title="Obsidian-Vault wählen")

        def done(dlg, result):
            try:
                folder = dlg.select_folder_finish(result)
            except Exception:
                return
            path = folder.get_path() if folder else None
            if not path:
                return
            self._persist("obsidian_vault", path)
            self.vault_row.set_subtitle(path)
            self._toast("Obsidian-Vault gesetzt ✓")

        dialog.select_folder(self.get_root(), None, done)

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
        src = self._current_source()
        mic = self._cv(self.mic_row, self._mic_opts) or self._default_mic
        mon = self._cv(self.mon_row, self._mon_opts) or self._default_monitor
        # Only the meters of active sources are shown — an always-empty
        # 'Mikrofon' row during a system-only recording just confuses.
        show_mic = src in ("both", "mic")
        show_sys = src in ("both", "system")
        if hasattr(self, "_mic_box"):
            self._mic_box.set_visible(show_mic)
            self._sys_box.set_visible(show_sys)
        if show_mic and mic:
            self._spawn_meter(mic, self._mic_viz)
        if show_sys and mon:
            self._spawn_meter(mon, self._sys_viz)

    def _spawn_meter(self, device, viz):
        stop = threading.Event()
        self._meter_stops.append(stop)

        def work():
            try:
                # pw-record is the reliable PipeWire capture (parec can yield no
                # data on some setups). "-" streams raw s16 to stdout.
                proc = subprocess.Popen(
                    pw_record_cmd(device),
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
        src = self._current_source()
        title = self.title_row.get_text().strip()
        mic = self._cv(self.mic_row, self._mic_opts)
        mon = self._cv(self.mon_row, self._mon_opts)
        bitrate = self._cv(self.quality_row, REC_QUALITY_OPTIONS)
        args = ["record-start", "--source", src, "--title", title, "--bitrate", bitrate]
        if mic:
            args += ["--mic-device", mic]
        if mon:
            args += ["--monitor-device", mon]
        self.rec_btn.set_sensitive(False)

        def work():
            r = recorder_call(*args)
            GLib.idle_add(self._after_start_record, r)
        threading.Thread(target=work, daemon=True).start()

    def _after_start_record(self, r: dict) -> bool:
        self.rec_btn.set_sensitive(True)
        if "error" in r:
            self._toast(f"Aufnahme-Fehler: {r['error']}")
            return False
        self._recording_base = r.get("base")
        self._rec_start = GLib.get_monotonic_time() / 1e6
        self._paused = False
        self._rec_button_state(recording=True)
        self._pause_button_state(paused=False)
        self.pause_btn.set_visible(True)
        if self._timer_id is None:
            self._timer_id = GLib.timeout_add(500, self._tick)
        self._start_meters()
        return False

    def _stop_record(self):
        # ffmpeg needs up to a few seconds to finalise the Opus container;
        # run that off the main loop so the window never freezes.
        base = self._recording_base
        self.rec_btn.set_sensitive(False)
        self._rec_button_state(recording=True, saving=True)
        self._stop_meters()

        def work():
            r = recorder_call("record-stop", timeout=20)
            GLib.idle_add(self._after_stop_record, base, r)
        threading.Thread(target=work, daemon=True).start()

    def _after_stop_record(self, base, r: dict) -> bool:
        self.rec_btn.set_sensitive(True)
        self._recording_base = None
        self._paused = False
        if self._timer_id is not None:
            GLib.source_remove(self._timer_id)
            self._timer_id = None
        self.timer_label.set_label("")
        self._rec_button_state(recording=False)
        self.pause_btn.set_visible(False)
        self.title_row.set_text("")
        self._toast(f"Aufnahme gespeichert ({fmt_duration(r.get('duration_seconds', 0))})")
        self.refresh()
        if base and self.auto_row.get_active():
            self._transcribe(base)
        return False

    def _toggle_pause(self, *_):
        if self._recording_base is None:
            return
        going_paused = not self._paused
        self.pause_btn.set_sensitive(False)

        def work():
            r = recorder_call("record-pause" if going_paused else "record-resume")
            GLib.idle_add(self._after_toggle_pause, going_paused, r)
        threading.Thread(target=work, daemon=True).start()

    def _after_toggle_pause(self, paused: bool, r: dict) -> bool:
        self.pause_btn.set_sensitive(True)
        if "error" in r:
            # Don't fake a paused UI over a recording that kept running.
            self._toast(f"Pause fehlgeschlagen: {r['error']}")
            self.refresh()
            return False
        self._paused = paused
        self._pause_button_state(paused)
        if paused:
            self._frozen = GLib.get_monotonic_time() / 1e6 - self._rec_start
            self._stop_meters()          # no signal is captured while paused → flat
        else:
            self._rec_start = GLib.get_monotonic_time() / 1e6 - self._frozen
            self._start_meters()
        return False

    def _tick(self):
        if self._recording_base is None:
            return False
        if self._paused:
            el = self._frozen
            self.timer_label.remove_css_class("error")
            self.timer_label.add_css_class("warning")
        else:
            el = GLib.get_monotonic_time() / 1e6 - self._rec_start
            self.timer_label.remove_css_class("warning")
            self.timer_label.add_css_class("error")
        text = f"● {int(el) // 60}:{int(el) % 60:02d}"
        # Growing file size = visible proof the recording is really being
        # written (reassuring during hour-long calls).
        if not self._paused and self._recording_base:
            try:
                size = (RECORDINGS_DIR / f"{self._recording_base}.opus").stat().st_size
                if size > 0:
                    text += f" · {fmt_size(size)}"
            except OSError:
                pass
        self.timer_label.set_text(text)
        return True

    def _apply_record_status(self, r: dict):
        if r.get("recording") and self._recording_base is None:
            self._recording_base = r.get("base")
            self._paused = bool(r.get("paused"))
            el = float(r.get("elapsed", 0))
            self._rec_start = GLib.get_monotonic_time() / 1e6 - el
            self._frozen = el
            self._rec_button_state(recording=True)
            self._pause_button_state(self._paused)
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
        child = self._list_container.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self._list_container.remove(child)
            child = nxt
        self._rows_by_base = {}
        items = data.get("recordings", []) if isinstance(data, dict) else []
        has_items = bool(items)
        self._list_container.set_visible(has_items)
        if hasattr(self, "_empty_status"):
            self._empty_status.set_visible(not has_items)
        if not has_items:
            return False
        groups: dict[str, Adw.PreferencesGroup] = {}
        for item in items:  # already sorted newest-first
            bucket = date_bucket(str(item.get("created", "")))
            group = groups.get(bucket)
            if group is None:
                group = Adw.PreferencesGroup(title=bucket)
                if not groups:  # first (newest) group carries the refresh button
                    refresh_btn = Gtk.Button(icon_name="view-refresh-symbolic",
                                             valign=Gtk.Align.CENTER)
                    refresh_btn.add_css_class("flat")
                    refresh_btn.set_tooltip_text("Aktualisieren")
                    refresh_btn.connect("clicked", lambda *_: self.refresh())
                    group.set_header_suffix(refresh_btn)
                groups[bucket] = group
                self._list_container.append(group)
            self._add_row(item, group)
        if self._busy and self._poll_id is None:
            self._poll_id = GLib.timeout_add(400, self._poll_progress)
        return False

    def _status_line(self, item):
        # Slim on purpose: transcript/notes state lives in the leading icon,
        # source + more detail on the detail page.
        parts = [fmt_duration(item.get("duration_seconds", 0))]
        try:
            parts.append(fmt_size((RECORDINGS_DIR / f"{item['base']}.opus").stat().st_size))
        except OSError:
            parts.append("Audio entfernt")
        return " · ".join(p for p in parts if p)

    def _add_row(self, item, group):
        base = item["base"]
        row = Adw.ActionRow()
        row.set_title(GLib.markup_escape_text(item.get("title", base)))
        group.add(row)
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
        # Play/delete only appear on hover or keyboard focus (calm rows);
        # 'Transkribieren' stays visible — it is the row's pending action.
        play = Gtk.Button(icon_name="media-playback-start-symbolic", valign=Gtk.Align.CENTER)
        play.add_css_class("flat")
        play.add_css_class("row-actions")
        play.set_tooltip_text("Anhören")
        play.connect("clicked", lambda b, x=base: self._toggle_row_preview(x, b))
        row.add_suffix(play)
        if not item.get("transcribed"):
            b = Gtk.Button(label="Transkribieren", valign=Gtk.Align.CENTER)
            b.add_css_class("flat")
            b.connect("clicked", lambda _b, x=base: self._transcribe(x))
            row.add_suffix(b)
        trash = Gtk.Button(icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER)
        trash.add_css_class("flat")
        trash.add_css_class("row-actions")
        trash.set_tooltip_text("Löschen")
        trash.connect("clicked", lambda _b, x=base: self._delete(x))
        row.add_suffix(trash)
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

    # One-click summary styles for the common cases (free text stays possible).
    FOCUS_PRESETS = (
        ("Vorlesungsnotizen",
         "prüfungsrelevante Inhalte, Definitionen und Beispiele — als strukturierte Lernnotizen"),
        ("Meeting-Protokoll",
         "besprochene Themen, Argumente und getroffene Entscheidungen — als Protokoll"),
        ("Action-Items",
         "Aufgaben, Verantwortliche und Fristen — als kompakte Action-Item-Liste"),
    )

    def _ask_focus(self, base):
        dlg = Adw.AlertDialog(
            heading="Wie soll zusammengefasst werden?",
            body="Preset wählen — oder eigenen Fokus beschreiben.")
        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        chips = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6,
                        halign=Gtk.Align.CENTER)
        for label, focus in self.FOCUS_PRESETS:
            chip = Gtk.Button(label=label)
            chip.add_css_class("pill")
            chip.connect("clicked", lambda _b, f=focus: (dlg.force_close(),
                                                         self._summarize(base, f)))
            chips.append(chip)
        content.append(chips)
        entry = Gtk.Entry(hexpand=True)
        entry.set_placeholder_text("Eigener Fokus (leer = wichtigste Inhalte und Action-Items)")
        content.append(entry)
        dlg.set_extra_child(content)
        dlg.add_response("cancel", "Abbrechen")
        dlg.add_response("go", "Zusammenfassen")
        dlg.set_response_appearance("go", Adw.ResponseAppearance.SUGGESTED)
        dlg.set_default_response("go")
        dlg.connect("response", lambda _d, resp, e=entry, x=base:
                    self._summarize(x, e.get_text().strip()) if resp == "go" else None)
        # Enter in the entry = confirm (instead of doing nothing)
        entry.connect("activate", lambda *_: (dlg.force_close(),
                                              self._summarize(base, entry.get_text().strip())))
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
        self._toast("Erstelle Zusammenfassung …")

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
            return "Fasse zusammen …"
        d = self._read_progress(base)
        if d.get("status") == "loading":
            return "Lädt Modell …"
        pct = self._smoothed_pct(base)
        if pct is not None:
            return f"Transkribiere … {pct} %{self._eta(base, pct)}"
        return "Transkribiere …"

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

        current_title = meta.get("title", base)

        # header: back | title | edit pencil | actions menu
        head = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        back = Gtk.Button(icon_name="go-previous-symbolic", valign=Gtk.Align.CENTER)
        back.add_css_class("flat")
        back.set_tooltip_text("Zurück")
        back.connect("clicked", lambda *_: self.nav.pop())
        head.append(back)
        title_lbl = Gtk.Label(label=current_title, xalign=0, hexpand=True, wrap=True)
        title_lbl.add_css_class("title-2")
        head.append(title_lbl)
        self._detail["title_lbl"] = title_lbl
        qa_btn = Gtk.Button(icon_name="dialog-question-symbolic", valign=Gtk.Align.CENTER)
        qa_btn.add_css_class("flat")
        qa_btn.set_tooltip_text("Frag die Aufnahme — inhaltliche Fragen ans Transkript")
        qa_btn.connect("clicked", lambda *_: self._open_qa(base))
        self._detail["qa_btn"] = qa_btn
        head.append(qa_btn)
        rename = Gtk.Button(icon_name="document-edit-symbolic", valign=Gtk.Align.CENTER)
        rename.add_css_class("flat")
        rename.set_tooltip_text("Titel bearbeiten")
        rename.connect("clicked", lambda *_: self._rename(base))
        head.append(rename)
        head.append(self._detail_menu_button(base))
        box.append(head)

        # audio metadata + player
        info = Adw.PreferencesGroup()
        box.append(info)
        from datetime import datetime as _dt
        try:
            created = _dt.fromisoformat(meta.get("created", "")).strftime("%d.%m.%Y %H:%M")
        except Exception:
            created = ""
        meta_line = " · ".join(p for p in [
            created, fmt_duration(meta.get("duration_seconds", 0)),
            REC_SOURCE_SHORT.get(meta.get("source", ""), meta.get("source", "")),
        ] if p)
        audio_path = RECORDINGS_DIR / f"{base}.opus"
        try:
            meta_line += f" · {fmt_size(audio_path.stat().st_size)}"
        except OSError:
            meta_line += " · Audio entfernt"
        player = Adw.ActionRow(title="Audio", subtitle=meta_line)
        info.add(player)
        if hasattr(Gtk, "MediaControls") and audio_path.exists():
            # Native seekable player (play/pause, scrubbing, volume) with
            # ±10 s jump buttons — for navigating hour-long lectures.
            self._media = Gtk.MediaFile.new_for_filename(str(audio_path))
            controls = Gtk.MediaControls(media_stream=self._media)
            controls.set_hexpand(True)
            back10 = Gtk.Button(icon_name="media-seek-backward-symbolic",
                                valign=Gtk.Align.CENTER)
            back10.add_css_class("flat")
            back10.set_tooltip_text("10 Sekunden zurück")
            back10.connect("clicked", lambda *_: self._seek_relative(-10))
            fwd10 = Gtk.Button(icon_name="media-seek-forward-symbolic",
                               valign=Gtk.Align.CENTER)
            fwd10.add_css_class("flat")
            fwd10.set_tooltip_text("10 Sekunden vor")
            fwd10.connect("clicked", lambda *_: self._seek_relative(10))
            pbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4,
                           margin_top=4, margin_bottom=4,
                           margin_start=8, margin_end=8)
            pbox.append(back10)
            pbox.append(controls)
            pbox.append(fwd10)
            controls_row = Gtk.ListBoxRow(activatable=False, selectable=False,
                                          child=pbox)
            info.add(controls_row)
        elif audio_path.exists():
            # Fallback without a GStreamer GTK backend: simple ffplay toggle.
            play_btn = Gtk.Button(icon_name="media-playback-start-symbolic",
                                  valign=Gtk.Align.CENTER)
            play_btn.set_tooltip_text("Abspielen")
            play_btn.add_css_class("flat")
            play_lbl = Gtk.Label(label="")
            play_lbl.add_css_class("numeric")
            play_lbl.add_css_class("dim-label")
            play_btn.connect("clicked", lambda *_: self._toggle_play(base, play_btn, play_lbl))
            player.add_suffix(play_lbl)
            player.add_suffix(play_btn)
            self._detail["play_btn"] = play_btn
            self._detail["play_lbl"] = play_lbl

        # chapters (AI-detected topic jump marks, filled in _load_chapters)
        chapters_row = Gtk.ListBoxRow(activatable=False, selectable=False)
        if hasattr(Adw, "WrapBox"):
            chapters_box = Adw.WrapBox(child_spacing=6, line_spacing=6)
        else:
            chapters_box = Gtk.FlowBox(selection_mode=Gtk.SelectionMode.NONE,
                                       column_spacing=6, row_spacing=6)
        chapters_box.set_margin_top(6)
        chapters_box.set_margin_bottom(8)
        chapters_box.set_margin_start(10)
        chapters_box.set_margin_end(10)
        chapters_row.set_child(chapters_box)
        chapters_row.set_visible(False)
        info.add(chapters_row)
        self._detail["chapters_row"] = chapters_row
        self._detail["chapters_box"] = chapters_box

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

        # transcript section (search + copy in the header; [mm:ss] markers in
        # the text are clickable and seek the player)
        tr_group = Adw.PreferencesGroup(title="Transkript")
        tr_suffix = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self._tr_search = Gtk.SearchEntry(placeholder_text="Im Transkript suchen …")
        self._tr_search.set_max_width_chars(22)
        self._tr_search.connect("search-changed", lambda *_: self._tr_do_search(reset=True))
        self._tr_search.connect("activate", lambda *_: self._tr_do_search(reset=False))
        tr_suffix.append(self._tr_search)
        tr_copy = Gtk.Button(icon_name="edit-copy-symbolic", valign=Gtk.Align.CENTER)
        tr_copy.add_css_class("flat")
        tr_copy.set_tooltip_text("Transkript kopieren")
        tr_copy.connect("clicked", lambda *_: self._copy(self._transcript_text()))
        tr_suffix.append(tr_copy)
        tr_group.set_header_suffix(tr_suffix)
        box.append(tr_group)
        tr_scroller = Gtk.ScrolledWindow(min_content_height=220)
        tr_scroller.add_css_class("card")
        tr_scroller.add_css_class("editor-card")
        tr_view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR, top_margin=10, bottom_margin=10,
                               left_margin=10, right_margin=10)
        tr_view.add_css_class("editor-view")
        click = Gtk.GestureClick()
        click.connect("released", self._on_transcript_click)
        tr_view.add_controller(click)
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

        page = Adw.NavigationPage(title=current_title, child=scroller)
        self._detail["page"] = page
        self.nav.push(page)
        self._load_detail_content(base)

    @staticmethod
    def _row_wrap(widget):
        # PreferencesGroup expects rows; wrap arbitrary widgets so they sit in the card.
        return widget

    # ── detail: actions menu ─────────────────────────────────────────────────

    def _detail_menu_button(self, base) -> Gtk.MenuButton:
        actions = Gio.SimpleActionGroup()
        for name, callback in (
            ("ask", lambda: self._open_qa(base)),
            ("chapters", lambda: self._make_chapters(base)),
            ("export-obsidian", lambda: self._export_obsidian(base)),
            ("retranscribe", lambda: self._confirm_retranscribe(base)),
            ("open-folder", self._open_folder),
            ("drop-audio", lambda: self._drop_audio(base)),
            ("delete", lambda: self._delete(base, from_detail=True)),
        ):
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", lambda _a, _p, cb=callback: cb())
            actions.add_action(action)
        menu = Gio.Menu()
        section = Gio.Menu()
        section.append("Frag die Aufnahme …", "detail.ask")
        section.append("Kapitel erkennen", "detail.chapters")
        section.append("Nach Obsidian exportieren", "detail.export-obsidian")
        section.append("Erneut transkribieren", "detail.retranscribe")
        section.append("Ordner öffnen", "detail.open-folder")
        menu.append_section(None, section)
        section = Gio.Menu()
        section.append("Audio entfernen (Transkript bleibt)", "detail.drop-audio")
        section.append("Aufnahme löschen", "detail.delete")
        menu.append_section(None, section)
        btn = Gtk.MenuButton(icon_name="view-more-symbolic", valign=Gtk.Align.CENTER)
        btn.add_css_class("flat")
        btn.set_tooltip_text("Aktionen")
        btn.set_menu_model(menu)
        btn.insert_action_group("detail", actions)
        return btn

    # ── detail: player seek + clickable transcript markers ──────────────────

    _TS_RE = re.compile(r"\[(\d+):(\d{2})(?::(\d{2}))?\]")

    @classmethod
    def _ts_to_seconds(cls, stamp: str) -> int | None:
        if not stamp.startswith("["):
            stamp = f"[{stamp}]"
        m = cls._TS_RE.fullmatch(stamp)
        if not m:
            return None
        first, mm, ss = m.group(1), m.group(2), m.group(3)
        if ss is not None:
            return int(first) * 3600 + int(mm) * 60 + int(ss)
        return int(first) * 60 + int(mm)

    def _seek_to(self, seconds: float) -> None:
        if self._media is None:
            return
        self._media.seek(max(0, int(seconds * 1_000_000)))
        if not self._media.get_playing():
            self._media.play()

    def _seek_relative(self, delta: float) -> None:
        if self._media is None:
            return
        self._seek_to(max(0.0, self._media.get_timestamp() / 1_000_000 + delta))

    def _on_transcript_click(self, _gesture, _n_press, x, y) -> None:
        """Click on a [mm:ss] marker → jump the player to that moment."""
        view = self._detail.get("tr_view")
        if view is None or self._media is None:
            return
        bx, by = view.window_to_buffer_coords(Gtk.TextWindowType.WIDGET, int(x), int(y))
        ok, it = view.get_iter_at_location(bx, by)
        if not ok:
            return
        buf = view.get_buffer()
        line_start = it.copy()
        line_start.set_line_offset(0)
        line_end = it.copy()
        if not line_end.ends_line():
            line_end.forward_to_line_end()
        line = buf.get_text(line_start, line_end, False)
        offset = it.get_line_offset()
        for m in self._TS_RE.finditer(line):
            if m.start() <= offset <= m.end():
                seconds = self._ts_to_seconds(m.group(0))
                if seconds is not None:
                    self._seek_to(seconds)
                return

    def _decorate_transcript(self) -> None:
        """Timestamps get accent color + bold — the transcript reads as
        structured paragraphs instead of a gray wall of text."""
        view = self._detail.get("tr_view")
        if view is None:
            return
        buf = view.get_buffer()
        tag = buf.get_tag_table().lookup("ts-marker")
        if tag is None:
            r, g, b = accent_rgb()
            rgba = Gdk.RGBA()
            rgba.red, rgba.green, rgba.blue, rgba.alpha = r, g, b, 1.0
            tag = buf.create_tag("ts-marker", weight=700, scale=0.85)
            tag.set_property("foreground-rgba", rgba)
        start, end = buf.get_bounds()
        buf.remove_tag(tag, start, end)
        text = buf.get_text(start, end, False)
        for m in self._TS_RE.finditer(text):
            buf.apply_tag(tag,
                          buf.get_iter_at_offset(m.start()),
                          buf.get_iter_at_offset(m.end()))

    def _load_chapters(self, base) -> None:
        row = self._detail.get("chapters_row")
        box = self._detail.get("chapters_box")
        if row is None or box is None:
            return
        child = box.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            box.remove(child)
            child = nxt
        try:
            chapters = json.loads(
                (RECORDINGS_DIR / f"{base}.chapters.json").read_text(encoding="utf-8"))
        except Exception:
            chapters = []
        usable = (isinstance(chapters, list) and chapters
                  and self._media is not None)
        row.set_visible(bool(usable))
        if not usable:
            return
        for chapter in chapters:
            stamp = str(chapter.get("time", ""))
            title = str(chapter.get("title", "")).strip()
            seconds = self._ts_to_seconds(stamp)
            if seconds is None or not title:
                continue
            btn = Gtk.Button(label=f"{stamp} · {title}")
            btn.add_css_class("flat")
            btn.add_css_class("chip")
            btn.set_tooltip_text("Im Audio dorthin springen")
            btn.connect("clicked", lambda _b, s=seconds: self._seek_to(s))
            box.append(btn)

    def _make_chapters(self, base) -> None:
        if not (RECORDINGS_DIR / f"{base}.txt").exists():
            self._toast("Erst transkribieren — Kapitel entstehen aus dem Transkript.")
            return
        self._toast("Erkenne Kapitel …")

        def done(r: dict) -> bool:
            if "error" in r:
                self._toast(f"Kapitel fehlgeschlagen: {r['error']}")
            else:
                self._toast(f"{len(r.get('chapters', []))} Kapitel erkannt ✓")
                if self._detail_base == base:
                    self._load_chapters(base)
            return False

        def work():
            r = recorder_call("chapters", base, timeout=300)
            GLib.idle_add(done, r)
        threading.Thread(target=work, daemon=True).start()

    # ── detail: transcript search with highlight ────────────────────────────

    def _tr_do_search(self, reset: bool) -> None:
        view = self._detail.get("tr_view")
        if view is None or not hasattr(self, "_tr_search"):
            return
        buf = view.get_buffer()
        tag = buf.get_tag_table().lookup("search-hit")
        if tag is None:
            r, g, b = accent_rgb()
            rgba = Gdk.RGBA()
            rgba.red, rgba.green, rgba.blue, rgba.alpha = r, g, b, 0.35
            tag = buf.create_tag("search-hit")
            tag.set_property("background-rgba", rgba)
        start, end = buf.get_bounds()
        buf.remove_tag(tag, start, end)
        needle = self._tr_search.get_text().strip()
        if not needle:
            return
        flags = Gtk.TextSearchFlags.CASE_INSENSITIVE | Gtk.TextSearchFlags.TEXT_ONLY
        it = buf.get_start_iter()
        first_match = None
        while True:
            result = it.forward_search(needle, flags, None)
            if not result:
                break
            m_start, m_end = result
            buf.apply_tag(tag, m_start, m_end)
            if first_match is None:
                first_match = m_start.copy()
            it = m_end
        if reset:
            target = first_match
        else:  # Enter: jump to the next match after the cursor
            cursor = buf.get_iter_at_mark(buf.get_insert())
            cursor.forward_char()
            result = cursor.forward_search(needle, flags, None)
            target = result[0] if result else first_match
        if target is not None:
            buf.place_cursor(target)
            view.scroll_to_iter(target, 0.1, False, 0.0, 0.0)

    # ── detail: destructive/maintenance actions ─────────────────────────────

    def _confirm_retranscribe(self, base) -> None:
        dlg = Adw.AlertDialog(heading="Erneut transkribieren?",
                              body="Das aktuelle Transkript wird verworfen und neu erstellt.")
        dlg.add_response("cancel", "Abbrechen")
        dlg.add_response("go", "Neu transkribieren")
        dlg.set_response_appearance("go", Adw.ResponseAppearance.DESTRUCTIVE)
        dlg.connect("response",
                    lambda _d, resp: self._retranscribe(base) if resp == "go" else None)
        dlg.present(self.get_root())

    def _drop_audio(self, base) -> None:
        dlg = Adw.AlertDialog(
            heading="Audio entfernen?",
            body="Die Audiodatei wird gelöscht, Transkript und Notizen bleiben "
                 "erhalten. Spart Speicherplatz — Anhören ist danach nicht mehr möglich.")
        dlg.add_response("cancel", "Abbrechen")
        dlg.add_response("drop", "Audio entfernen")
        dlg.set_response_appearance("drop", Adw.ResponseAppearance.DESTRUCTIVE)

        def on_resp(_d, resp):
            if resp != "drop":
                return
            self._stop_play()
            (RECORDINGS_DIR / f"{base}.opus").unlink(missing_ok=True)
            self._toast("Audio entfernt — Transkript bleibt")
            if self._detail_base == base:
                self.nav.pop()
                self._open_detail(base)
            self.refresh()

        dlg.connect("response", on_resp)
        dlg.present(self.get_root())

    # ── detail: Q&A (Frag die Aufnahme) — dezenter On-Demand-Dialog ──────────

    def _open_qa(self, base) -> None:
        if not (RECORDINGS_DIR / f"{base}.txt").exists():
            self._toast("Erst transkribieren — dann kannst du Fragen stellen.")
            return
        dlg = self._detail.get("qa_dialog")
        if dlg is None:
            dlg = Adw.Dialog(title="Frag die Aufnahme",
                             content_width=560, content_height=520)
            toolbar = Adw.ToolbarView()
            toolbar.add_top_bar(Adw.HeaderBar())
            outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10,
                            margin_top=6, margin_bottom=12,
                            margin_start=14, margin_end=14)

            scroller = Gtk.ScrolledWindow(vexpand=True)
            qa_answers = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
            self._detail["qa_answers"] = qa_answers
            hint = Gtk.Label(
                label="Stell eine inhaltliche Frage zur Aufnahme —\n"
                      "die KI antwortet nur aus dem Transkript.",
                justify=Gtk.Justification.CENTER, vexpand=True,
                valign=Gtk.Align.CENTER)
            hint.add_css_class("dim-label")
            self._detail["qa_hint"] = hint
            qa_answers.append(hint)
            scroller.set_child(qa_answers)
            outer.append(scroller)

            qa_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            qa_entry = Gtk.Entry(hexpand=True)
            qa_entry.set_placeholder_text("Was wurde zu … gesagt?")
            qa_entry.connect("activate", lambda *_: self._ask_question(base))
            self._detail["qa_entry"] = qa_entry
            qa_row.append(qa_entry)
            qa_send = Gtk.Button(label="Fragen")
            qa_send.add_css_class("suggested-action")
            qa_send.connect("clicked", lambda *_: self._ask_question(base))
            self._detail["qa_send"] = qa_send
            qa_row.append(qa_send)
            outer.append(qa_row)

            toolbar.set_content(outer)
            dlg.set_child(toolbar)
            self._detail["qa_dialog"] = dlg
        dlg.present(self.get_root())
        self._detail["qa_entry"].grab_focus()

    def _ask_question(self, base) -> None:
        entry = self._detail.get("qa_entry")
        if entry is None or self._qa_busy:
            return
        question = entry.get_text().strip()
        if not question:
            return
        self._qa_busy = True
        entry.set_text("")
        hint = self._detail.get("qa_hint")
        if hint is not None and hint.get_parent() is not None:
            self._detail["qa_answers"].remove(hint)
        send = self._detail.get("qa_send")
        if send is not None:
            send.set_sensitive(False)

        q_lbl = Gtk.Label(label=question, xalign=0, wrap=True, selectable=True)
        q_lbl.add_css_class("heading")
        a_lbl = Gtk.Label(label="Die KI liest das Transkript …", xalign=0,
                          wrap=True, selectable=True)
        a_lbl.add_css_class("dim-label")
        item = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        item.append(q_lbl)
        item.append(a_lbl)
        self._detail["qa_answers"].append(item)

        def done(r: dict) -> bool:
            self._qa_busy = False
            if send is not None:
                try:
                    send.set_sensitive(True)
                except Exception:
                    pass
            try:
                if "error" in r:
                    a_lbl.set_text(f"Fehler: {r['error']}")
                    a_lbl.remove_css_class("dim-label")
                    a_lbl.add_css_class("error")
                else:
                    answer = str(r.get("answer", "")).strip() or "(keine Antwort)"
                    a_lbl.set_text(answer)
                    a_lbl.remove_css_class("dim-label")
                    # Cited [mm:ss] marks become jump buttons into the audio.
                    stamps: list[str] = []
                    for match in self._TS_RE.finditer(answer):
                        if match.group(0) not in stamps:
                            stamps.append(match.group(0))
                    if stamps and self._media is not None:
                        chips = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL,
                                        spacing=6)
                        for stamp in stamps[:6]:
                            seconds = self._ts_to_seconds(stamp)
                            if seconds is None:
                                continue
                            jump = Gtk.Button(label=f"▶ {stamp.strip('[]')}")
                            jump.add_css_class("flat")
                            jump.add_css_class("chip")
                            jump.set_tooltip_text("Im Audio dorthin springen")
                            jump.connect("clicked",
                                         lambda _b, s=seconds: self._seek_to(s))
                            chips.append(jump)
                        item.append(chips)
            except Exception:
                pass  # Detail-Seite wurde inzwischen geschlossen
            return False

        def work():
            r = recorder_call("ask", base, "--question", question, timeout=600)
            GLib.idle_add(done, r)
        threading.Thread(target=work, daemon=True).start()

    # ── detail: Obsidian export ──────────────────────────────────────────────

    def _export_obsidian(self, base) -> None:
        vault = str(load_config().get("obsidian_vault", "")).strip()
        if vault and Path(vault).is_dir():
            self._write_obsidian_note(base, Path(vault))
            return

        dialog = Gtk.FileDialog(title="Obsidian-Vault wählen")

        def done(dlg, result):
            try:
                folder = dlg.select_folder_finish(result)
            except Exception:
                return  # abgebrochen
            path = folder.get_path() if folder else None
            if not path:
                return
            cfg = load_config()
            cfg["obsidian_vault"] = path
            save_config(cfg)
            self._write_obsidian_note(base, Path(path))

        dialog.select_folder(self.get_root(), None, done)

    def _write_obsidian_note(self, base, vault: Path) -> None:
        try:
            meta = json.loads((RECORDINGS_DIR / f"{base}.meta.json").read_text())
        except Exception:
            meta = {}
        title = str(meta.get("title", base)).strip() or base
        try:
            transcript = (RECORDINGS_DIR / f"{base}.txt").read_text(encoding="utf-8").strip()
        except OSError:
            transcript = ""
        try:
            notes = (RECORDINGS_DIR / f"{base}.summary.md").read_text(encoding="utf-8").strip()
        except OSError:
            notes = ""
        created = str(meta.get("created", ""))[:10]
        safe_title = "".join(c for c in title if c not in '\\/:*?"<>|').strip() or base
        target_dir = vault / "Aufnahmen"
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{created + ' ' if created else ''}{safe_title}.md"
            n = 2
            while target.exists():
                target = target_dir / f"{created + ' ' if created else ''}{safe_title} ({n}).md"
                n += 1
            parts = [f"# {title}", ""]
            facts = [p for p in (
                f"Datum: {created}" if created else "",
                f"Dauer: {fmt_duration(meta.get('duration_seconds', 0))}",
            ) if p]
            parts += [" · ".join(facts), ""]
            if notes:
                parts += ["## Notizen", "", notes, ""]
            if transcript:
                parts += ["## Transkript", "", transcript, ""]
            target.write_text("\n".join(parts), encoding="utf-8")
            self._toast(f"Exportiert: {target.name}")
        except OSError as exc:
            self._toast(f"Export fehlgeschlagen: {exc}")

    def _on_popped(self, _nav, _page):
        qa_dialog = self._detail.get("qa_dialog")
        if qa_dialog is not None:
            try:
                qa_dialog.force_close()
            except Exception:
                pass
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

        self._decorate_transcript()
        self._load_chapters(base)

        notes = self._notes_text()
        has_notes = bool(notes.strip())
        self._detail["nt_view"].get_buffer().set_text(notes)
        self._detail["nt_scroller"].set_visible(has_notes)
        self._detail["nt_actions"].set_visible(has_notes)
        self._detail["nt_empty"].set_visible(not has_notes and not busy)
        # can only summarize/ask once a transcript exists
        if not has_notes:
            self._detail["nt_btn"].set_sensitive(has_txt)
        if self._detail.get("qa_btn") is not None:
            self._detail["qa_btn"].set_sensitive(has_txt)
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
            lbl.set_label("Erstelle Zusammenfassung …")
            bar.pulse()
            return
        d = self._read_progress(base)
        if d.get("status") == "loading":
            lbl.set_label("Lädt Modell …")
            bar.pulse()
            return
        pct = self._smoothed_pct(base)
        if pct is not None:
            bar.set_fraction(min(1.0, pct / 100.0))
            lbl.set_label(f"Transkribiere … {pct} %{self._eta(base, pct)}")
        else:
            lbl.set_label("Transkribiere …")
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
        # Enter in the entry = confirm
        entry.connect("activate", lambda *_: (dlg.force_close(), on_resp(dlg, "ok")))

        def apply_ui(new: str) -> bool:
            shown = new or base
            if self._detail.get("title_lbl"):
                self._detail["title_lbl"].set_label(shown)
            if self._detail.get("title_row"):
                self._detail["title_row"].set_subtitle(shown)
            if self._detail.get("page"):
                self._detail["page"].set_title(shown)
            self.refresh()
            self._toast("Umbenannt ✓")
            return False

        def on_resp(_d, resp):
            if resp != "ok":
                return
            new = entry.get_text().strip()

            def work():
                recorder_call("rename", base, "--title", new)
                GLib.idle_add(apply_ui, new)
            threading.Thread(target=work, daemon=True).start()

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

        def apply_ui() -> bool:
            if from_detail and self._detail_base == base:
                self.nav.pop()
            self.refresh()
            self._toast("Gelöscht")
            return False

        def on_resp(_d, resp):
            if resp != "del":
                return

            def work():
                recorder_call("delete", base)
                GLib.idle_add(apply_ui)
            threading.Thread(target=work, daemon=True).start()

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
        btn.set_icon_name("media-playback-stop-symbolic")
        btn.set_tooltip_text("Stopp")
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
            btn.set_icon_name("media-playback-start-symbolic")
            btn.set_tooltip_text("Abspielen")
            lbl.set_label("")
        except Exception:
            pass

    def _stop_play(self):
        # native detail player
        if self._media is not None:
            try:
                self._media.pause()
            except Exception:
                pass
            self._media = None
        self._stop_row_preview()
        # ffplay fallback
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

    # ── inline row preview (listen without opening the detail page) ──────────
    def _toggle_row_preview(self, base, btn):
        if self._preview_base == base:
            self._stop_row_preview()
            return
        self._stop_row_preview()
        path = RECORDINGS_DIR / f"{base}.opus"
        if not path.exists() or not hasattr(Gtk, "MediaControls"):
            self._open_detail(base)   # fallback: detail page has a player
            return
        try:
            media = Gtk.MediaFile.new_for_filename(str(path))
        except Exception:
            self._toast("Wiedergabe nicht möglich.")
            return
        self._preview_media = media
        self._preview_base = base
        self._preview_btn = btn
        btn.set_icon_name("media-playback-stop-symbolic")
        btn.set_tooltip_text("Stopp")
        # Keep the button visible while playing (it is hover-only otherwise).
        btn.remove_css_class("row-actions")
        media.connect("notify::ended",
                      lambda m, _p: self._stop_row_preview() if m is self._preview_media else None)
        media.play()

    def _stop_row_preview(self):
        if self._preview_media is not None:
            try:
                self._preview_media.pause()
            except Exception:
                pass
        if self._preview_btn is not None:
            try:
                self._preview_btn.set_icon_name("media-playback-start-symbolic")
                self._preview_btn.set_tooltip_text("Anhören")
                self._preview_btn.add_css_class("row-actions")
            except Exception:
                pass
        self._preview_media = None
        self._preview_base = None
        self._preview_btn = None


# Changing these keys re-grabs the evdev listener, which only happens at
# daemon startup — the dialog shows a restart banner instead of restarting
# silently mid-session.
RESTART_KEYS = ("double_tap_key", "llm_toggle_key", "command_key")


class PrefsDialog(Adw.PreferencesDialog):
    """GNOME-style instant-apply settings: every change is saved immediately
    and live-reloaded into the daemon (debounced). No save button."""

    def __init__(self, win: SettingsWindow):
        super().__init__(title="Einstellungen")
        self.win = win
        self.config = load_config()
        self._reload_id = None
        self._pending: dict = {}
        self._debounce_id = None
        self._updating = False          # guard: programmatic combo rebuilds
        self._capturing = None
        self._capture_ctrl = None
        self._banners: list[Adw.Banner] = []
        # Filled asynchronously — arecord probing can stall on some hardware
        # and must never block opening the dialog.
        current_dev = str(self.config.get("record_device", "default"))
        self.device_options = [(
            current_dev,
            "default (Systemstandard)" if current_dev == "default" else current_dev,
        )]
        # Debounced free-text edits must not be lost when the dialog closes.
        self.connect("closed", lambda *_: self.flush_now())

        # ── Seite 1: Diktat ──────────────────────────────────────────────
        page = Adw.PreferencesPage(title="Diktat", icon_name="audio-input-microphone-symbolic")
        self.add(page)
        self._attach_banner(page)

        status_group = Adw.PreferencesGroup()
        page.add(status_group)
        self.status_row = Adw.ActionRow(title="Daemon")
        self.status_row.add_prefix(Gtk.Image(icon_name="audio-input-microphone-symbolic"))
        restart_btn = Gtk.Button(label="Neu starten", valign=Gtk.Align.CENTER)
        restart_btn.add_css_class("flat")
        restart_btn.connect("clicked", lambda *_: self.win._on_restart())
        self.status_row.add_suffix(restart_btn)
        status_group.add(self.status_row)

        rec = Adw.PreferencesGroup(title="Erkennung")
        page.add(rec)
        self.model_row = self._combo("Modell", MODEL_OPTIONS, str(self.config["model"]))
        self._update_model_hint()
        self.model_row.connect("notify::selected", self._on_model_changed)
        rec.add(self.model_row)
        self.language_row = self._combo(
            "Sprache", LANGUAGE_OPTIONS, str(self.config.get("language", "de")).lower())
        self._bind_combo(self.language_row, LANGUAGE_OPTIONS, "language")
        rec.add(self.language_row)
        self.hotwords_row = Adw.EntryRow(title="Hotwords (Komma-getrennt)")
        self.hotwords_row.set_text(str(self.config.get("hotwords", "")))
        self._bind_entry(self.hotwords_row, "hotwords")
        rec.add(self.hotwords_row)
        self.vad_row = Adw.SwitchRow(title="VAD", subtitle="Stille filtern (weniger Halluzinationen)")
        self.vad_row.set_active(bool(self.config.get("vad_filter", True)))
        self._bind_switch(self.vad_row, "vad_filter")
        rec.add(self.vad_row)
        self.voice_row = Adw.SwitchRow(
            title="Sprachbefehle",
            subtitle="neue Zeile, neuer Absatz, Doppelpunkt, Fragezeichen …",
        )
        self.voice_row.set_active(bool(self.config.get("voice_commands", False)))
        self._bind_switch(self.voice_row, "voice_commands")
        rec.add(self.voice_row)

        inp = Adw.PreferencesGroup(title="Eingabe")
        page.add(inp)
        self.mode_row = self._combo(
            "Modus", HOTKEY_MODE_OPTIONS, str(self.config.get("hotkey_mode", "double_tap")))
        self._bind_combo(self.mode_row, HOTKEY_MODE_OPTIONS, "hotkey_mode")
        inp.add(self.mode_row)
        hotkey_cur = str(self.config["double_tap_key"])
        self._hotkey_opts = key_options(HOTKEY_OPTIONS, hotkey_cur)
        self.hotkey_row = self._combo("Aufnahme-Taste", self._hotkey_opts, hotkey_cur)
        self._bind_combo(self.hotkey_row, self._hotkey_opts, "double_tap_key")
        inp.add(self.hotkey_row)
        inp.add(self._make_capture_row("hotkey_row", "_hotkey_opts"))
        self.double_tap_row = Adw.SpinRow.new_with_range(150, 1200, 10)
        self.double_tap_row.set_title("Double-Tap-Fenster (ms)")
        self.double_tap_row.set_value(float(self.config["double_tap_window_ms"]))
        self._bind_spin(self.double_tap_row, "double_tap_window_ms")
        inp.add(self.double_tap_row)
        self.paste_row = self._combo("Paste-Modus", PASTE_OPTIONS, str(self.config["paste_mode"]))
        self._bind_combo(self.paste_row, PASTE_OPTIONS, "paste_mode")
        inp.add(self.paste_row)
        self.max_record_row = Adw.SpinRow.new_with_range(15, 900, 5)
        self.max_record_row.set_title("Max. Aufnahme (s)")
        self.max_record_row.set_value(float(self.config["max_record_seconds"]))
        self._bind_spin(self.max_record_row, "max_record_seconds")
        inp.add(self.max_record_row)
        self.sound_row = Adw.SwitchRow(title="Sound-Feedback", subtitle="Ton bei Start/Fertig")
        self.sound_row.set_active(bool(self.config.get("sound_cue", True)))
        self._bind_switch(self.sound_row, "sound_cue")
        inp.add(self.sound_row)
        self.clipboard_row = Adw.SwitchRow(
            title="Zwischenablage schonen",
            subtitle="Inhalt nach dem Einfügen wiederherstellen. Pausiert automatisch, "
                     "wenn ein Clipboard-Manager (z. B. Vicinae) läuft — so bleibt das "
                     "Diktat dort an erster Stelle.",
        )
        self.clipboard_row.set_active(bool(self.config.get("restore_clipboard", True)))
        self._bind_switch(self.clipboard_row, "restore_clipboard")
        inp.add(self.clipboard_row)
        self.history_row = Adw.SwitchRow(
            title="Verlauf speichern",
            subtitle="Diktate im Verlauf-Tab merken",
        )
        self.history_row.set_active(bool(self.config.get("save_history", True)))
        self._bind_switch(self.history_row, "save_history")
        inp.add(self.history_row)

        audio = Adw.PreferencesGroup(title="Audio")
        page.add(audio)
        self.device_row = self._combo("Mikrofon", self.device_options, str(self.config["record_device"]))
        self._bind_combo(self.device_row, self.device_options, "record_device")
        audio.add(self.device_row)
        self.viz_row = self._combo("Live-Visualisierung", VISUALIZER_OPTIONS,
                                   str(self.config.get("audio_visualizer", "waves")))
        self.viz_row.set_subtitle("Während der Aufnahme in Werkbank und Rekorder: "
                                  "Wellen (Lautstärke-Verlauf), Balken oder aus.")
        self._bind_combo(self.viz_row, VISUALIZER_OPTIONS, "audio_visualizer")
        audio.add(self.viz_row)
        self._load_devices_async()

        # ── Seite 2: KI & Kontext ────────────────────────────────────────
        page2 = Adw.PreferencesPage(title="KI & Kontext", icon_name="text-editor-symbolic")
        self.add(page2)
        self._attach_banner(page2)
        mode_group = Adw.PreferencesGroup(
            title="Diktier-Modus",
            description="Bestimmt, wie diktierter Text formuliert wird. "
                        "E-Mail und Chat nutzen immer die KI, Roh nie.",
        )
        page2.add(mode_group)
        self.dict_mode_row = self._combo(
            "Modus", DICT_MODE_OPTIONS, str(self.config.get("dictation_mode", "standard")))
        self._bind_combo(self.dict_mode_row, DICT_MODE_OPTIONS, "dictation_mode")
        mode_group.add(self.dict_mode_row)
        llm = Adw.PreferencesGroup(
            title="Textverbesserung (Ollama)",
            description="Optionaler LLM-Schritt: entfernt Füllwörter, korrigiert "
                        "Grammatik. Kostet ~2–4 s extra. Standard: aus.",
        )
        page2.add(llm)
        self.ollama_row = Adw.SwitchRow(
            title="Ollama-Nachbearbeitung",
            subtitle="Braucht laufenden Ollama-Server",
        )
        self.ollama_row.set_active(bool(self.config.get("ollama_postprocess", False)))
        self.ollama_row.connect("notify::active", self._on_ollama_toggled)
        llm.add(self.ollama_row)
        current_model = str(self.config.get("ollama_model", "qwen2.5:7b"))
        self._llm_model_opts = list(LLM_MODEL_OPTIONS)
        if current_model not in [v for v, _ in self._llm_model_opts]:
            self._llm_model_opts.append((current_model, f"{current_model}  (eigenes)"))
        self.ollama_model_row = self._combo("Modell", self._llm_model_opts, current_model)
        self.ollama_model_row.set_subtitle(
            "Mehr Sterne = stärker, aber langsamer. Muss via 'ollama pull' installiert sein.")
        self.ollama_model_row.connect("notify::selected", self._on_ollama_model_changed)
        llm.add(self.ollama_model_row)
        self._mark_installed_ollama_models()
        toggle_cur = str(self.config.get("llm_toggle_key", ""))
        self._llm_toggle_opts = key_options(LLM_TOGGLE_OPTIONS, toggle_cur)
        self.llm_toggle_row = self._combo(
            "Umschalt-Taste (Doppel-Tap)", self._llm_toggle_opts, toggle_cur)
        self.llm_toggle_row.set_subtitle(
            "Schaltet Cleanup an/aus. Muss sich von der Aufnahme-Taste unterscheiden.")
        self._bind_combo(self.llm_toggle_row, self._llm_toggle_opts, "llm_toggle_key")
        llm.add(self.llm_toggle_row)
        llm.add(self._make_capture_row("llm_toggle_row", "_llm_toggle_opts"))
        command_cur = str(self.config.get("command_key", ""))
        self._command_opts = key_options(LLM_TOGGLE_OPTIONS, command_cur)
        self.command_row = self._combo(
            "Befehl-Taste (markierten Text bearbeiten)", self._command_opts, command_cur)
        self.command_row.set_subtitle(
            "Doppel-Tap, dann Anweisung sprechen → ersetzt die Markierung. "
            "Funktioniert in Textfeldern, nicht im Terminal.")
        self._bind_combo(self.command_row, self._command_opts, "command_key")
        llm.add(self.command_row)
        llm.add(self._make_capture_row("command_row", "_command_opts"))

        adv = Adw.PreferencesGroup(
            title="Kontext (Initial Prompt)",
            description="Lenkt die Erkennung Richtung deiner Begriffe — wird nicht "
                        "mitgeschrieben. Leer lassen = aus.",
        )
        page2.add(adv)
        prompt_scroller = Gtk.ScrolledWindow(min_content_height=110)
        prompt_scroller.add_css_class("card")
        prompt_scroller.add_css_class("editor-card")
        self.prompt_view = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD_CHAR,
            top_margin=8, bottom_margin=8, left_margin=8, right_margin=8,
        )
        self.prompt_view.get_buffer().set_text(str(self.config.get("initial_prompt", "")))
        self.prompt_view.get_buffer().connect("changed", self._on_prompt_changed)
        prompt_scroller.set_child(self.prompt_view)
        adv.add(prompt_scroller)

        # ── Seite 3: Wörterbuch ──────────────────────────────────────────
        page3 = Adw.PreferencesPage(title="Wörterbuch", icon_name="accessories-dictionary-symbolic")
        self.add(page3)

        dict_group = Adw.PreferencesGroup(
            title="Eigene Begriffe",
            description="Namen, Fachbegriffe, Abkürzungen — ein Begriff pro Zeile. "
                        "Sie fließen in die Erkennung ein und werden richtig geschrieben.",
        )
        page3.add(dict_group)
        dict_group.add(self._editor_card(
            "\n".join(dictionary_terms(self.config)), self._on_dictionary_changed))

        repl_group = Adw.PreferencesGroup(
            title="Ersetzungen",
            description="Hartnäckige Fehlerkennungen automatisch korrigieren. "
                        "Eine pro Zeile im Format: falsch = richtig",
        )
        page3.add(repl_group)
        repl = self.config.get("replacements") or {}
        repl_group.add(self._editor_card(
            "\n".join(f"{k} = {v}" for k, v in repl.items()),
            self._on_replacements_changed))

        snip_group = Adw.PreferencesGroup(
            title="Sprach-Schnipsel",
            description="Sprich exakt den Auslöser, und der hinterlegte Text wird "
                        "eingefügt — z. B. für Grußformeln oder Adressen. "
                        "Eine pro Zeile: auslöser = Text (\\n = Zeilenumbruch)",
        )
        page3.add(snip_group)
        snippets = self.config.get("snippets") or {}
        snip_group.add(self._editor_card(
            "\n".join(f"{k} = {self._escape_value(str(v))}" for k, v in snippets.items()),
            self._on_snippets_changed))

        self.refresh_status()

    # ── Instant apply ─────────────────────────────────────────────────────

    def _apply(self, key: str, value) -> None:
        self._apply_many({key: value})

    def _apply_many(self, changes: dict) -> None:
        # Re-read from disk so keys the Rekorder tab persists are never lost;
        # one read+write per batch, not per key.
        cfg = load_config()
        changed = {k: v for k, v in changes.items() if cfg.get(k) != v}
        if not changed:
            return
        cfg.update(changed)
        save_config(cfg)
        self.config = cfg
        self.win.config = cfg
        if "audio_visualizer" in changed:
            mode = str(changed["audio_visualizer"])
            self.win.recorder.set_visualizer_mode(mode)
            self.win.workbench._viz.set_mode(mode)
        if any(k in RESTART_KEYS for k in changed):
            self._show_restart_banner()
        if any(k not in RESTART_KEYS for k in changed):
            # Model/backend switches make the daemon unload+reload a multi-GB
            # model — debounce those longer so browsing the combo doesn't
            # trigger a reload per entry.
            heavy = any(k in ("model", "backend", "ov_device") for k in changed)
            self._schedule_reload(2000 if heavy else 600)

    def _bind_combo(self, row: Adw.ComboRow, options: list, key: str) -> None:
        row.connect(
            "notify::selected",
            lambda *_: None if self._updating else self._apply(key, self._cv(row, options)),
        )

    def _bind_switch(self, row: Adw.SwitchRow, key: str) -> None:
        row.connect("notify::active", lambda *_: self._apply(key, bool(row.get_active())))

    def _bind_spin(self, row: Adw.SpinRow, key: str) -> None:
        row.connect("notify::value", lambda *_: self._apply(key, int(row.get_value())))

    def _bind_entry(self, row: Adw.EntryRow, key: str) -> None:
        row.connect("notify::text",
                    lambda *_: self._apply_debounced(key, row.get_text().strip()))

    def _on_prompt_changed(self, buf) -> None:
        self._apply_debounced("initial_prompt", self._buffer_text(buf).strip())

    # ── Wörterbuch / Ersetzungen / Schnipsel ──────────────────────────────

    def _editor_card(self, initial: str, on_changed) -> Gtk.ScrolledWindow:
        scroller = Gtk.ScrolledWindow(min_content_height=96)
        scroller.add_css_class("card")
        scroller.add_css_class("editor-card")
        view = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD_CHAR,
            top_margin=8, bottom_margin=8, left_margin=8, right_margin=8,
        )
        view.get_buffer().set_text(initial)
        view.get_buffer().connect("changed", on_changed)
        scroller.set_child(view)
        return scroller

    @staticmethod
    def _buffer_text(buf) -> str:
        return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)

    def _on_dictionary_changed(self, buf) -> None:
        words = [ln.strip() for ln in self._buffer_text(buf).splitlines() if ln.strip()]
        self._apply_debounced("dictionary", words)

    def _on_replacements_changed(self, buf) -> None:
        self._apply_debounced(
            "replacements", self._parse_pairs(self._buffer_text(buf), unescape=False))

    def _on_snippets_changed(self, buf) -> None:
        self._apply_debounced(
            "snippets", self._parse_pairs(self._buffer_text(buf), unescape=True))

    @staticmethod
    def _escape_value(value: str) -> str:
        # Escape backslashes FIRST, then newlines — otherwise a literal "\n"
        # in a snippet round-trips into a real newline and corrupts the text.
        return value.replace("\\", "\\\\").replace("\n", "\\n")

    @staticmethod
    def _unescape_value(value: str) -> str:
        return value.replace("\\\\", "\x00").replace("\\n", "\n").replace("\x00", "\\")

    @classmethod
    def _parse_pairs(cls, text: str, unescape: bool) -> dict:
        pairs: dict[str, str] = {}
        for line in text.splitlines():
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and value:
                pairs[key] = cls._unescape_value(value) if unescape else value
        return pairs

    def _apply_debounced(self, key: str, value) -> None:
        """Free-text fields save 800 ms after the last keystroke, not on each."""
        self._pending[key] = value
        if self._debounce_id is not None:
            GLib.source_remove(self._debounce_id)
        self._debounce_id = GLib.timeout_add(800, self._flush_pending)

    def _flush_pending(self) -> bool:
        self._debounce_id = None
        pending, self._pending = self._pending, {}
        if pending:
            self._apply_many(pending)
        return False

    def flush_now(self) -> None:
        """Write out pending debounced edits + fire a scheduled reload
        immediately (dialog/window is closing; the timeouts would be lost)."""
        if self._debounce_id is not None:
            GLib.source_remove(self._debounce_id)
            self._debounce_id = None
        pending, self._pending = self._pending, {}
        if pending:
            self._apply_many(pending)
        if self._reload_id is not None:
            GLib.source_remove(self._reload_id)
            self._do_reload()

    def _schedule_reload(self, delay_ms: int = 600) -> None:
        if self._reload_id is not None:
            GLib.source_remove(self._reload_id)
        self._reload_id = GLib.timeout_add(delay_ms, self._do_reload)

    def _do_reload(self) -> bool:
        self._reload_id = None

        def work():
            subprocess.run([str(DAEMON_SCRIPT), "--reload"], capture_output=True, check=False)
        threading.Thread(target=work, daemon=True).start()
        return False

    # ── Restart banner (one per page: two of the three hotkey rows live on
    # page 2, and a banner widget can only have one parent) ────────────────

    def _attach_banner(self, page: Adw.PreferencesPage) -> None:
        if not hasattr(page, "set_banner"):
            return  # libadwaita < 1.7
        banner = Adw.Banner(title="Tasten-Änderung — Daemon-Neustart nötig")
        banner.set_button_label("Neu starten")
        banner.connect("button-clicked", self._restart_for_keys)
        page.set_banner(banner)
        self._banners.append(banner)

    def _show_restart_banner(self) -> None:
        if not self._banners:
            # No banner support: restart right away (old behavior) instead
            # of leaving the new key silently inactive.
            self._restart_for_keys()
            return
        for banner in self._banners:
            banner.set_revealed(True)

    def _restart_for_keys(self, *_a) -> None:
        for banner in self._banners:
            banner.set_revealed(False)
        self.win._daemon_action(
            "--restart", "Daemon neu gestartet — neue Taste aktiv.", "Neustart fehlgeschlagen")

    def _on_ollama_toggled(self, *_a) -> None:
        active = bool(self.ollama_row.get_active())
        self._apply("ollama_postprocess", active)
        if active:
            self._warn_if_model_missing()

    def _on_ollama_model_changed(self, *_a) -> None:
        if self._updating:
            return
        self._apply("ollama_model", self._cv(self.ollama_model_row, self._llm_model_opts))
        if bool(self.config.get("ollama_postprocess")):
            self._warn_if_model_missing()

    def _warn_if_model_missing(self) -> None:
        model = self._cv(self.ollama_model_row, self._llm_model_opts)

        def work():
            if self.win._ollama_model_installed(model) is False:
                GLib.idle_add(self._toast, f"Modell fehlt: ollama pull {model}")
        threading.Thread(target=work, daemon=True).start()

    def _toast(self, text: str) -> bool:
        self.add_toast(Adw.Toast.new(text))
        return False

    # ── UI helpers ────────────────────────────────────────────────────────

    _combo = staticmethod(combo_row)
    _cv = staticmethod(combo_value)

    def _load_devices_async(self) -> None:
        def apply(devices: list) -> bool:
            cur = str(load_config().get("record_device", "default"))
            if cur not in [v for v, _ in devices]:
                devices = devices + [(cur, cur)]
            self._updating = True
            try:
                self.device_options[:] = devices  # in place: the bind closure holds this list
                self.device_row.set_model(Gtk.StringList.new([lbl for _, lbl in devices]))
                self.device_row.set_selected(
                    next((i for i, (v, _) in enumerate(devices) if v == cur), 0))
            finally:
                self._updating = False
            return False

        def work():
            GLib.idle_add(apply, detect_alsa_capture_devices())
        threading.Thread(target=work, daemon=True).start()

    def _on_model_changed(self, *_a) -> None:
        self._update_model_hint()
        if not self._updating:
            self._apply("model", self._cv(self.model_row, MODEL_OPTIONS))

    def _update_model_hint(self) -> None:
        self.model_row.set_subtitle(
            MODEL_HINTS.get(self._cv(self.model_row, MODEL_OPTIONS), ""))

    def refresh_status(self) -> None:
        def work():
            running = daemon_running()
            GLib.idle_add(self.status_row.set_subtitle, "läuft" if running else "gestoppt")
        threading.Thread(target=work, daemon=True).start()

    def _mark_installed_ollama_models(self) -> None:
        """Prefix installed models with '✓' once `ollama list` answered (async)."""
        def apply_marks(installed: set) -> bool:
            if not installed:
                return False
            self._updating = True
            try:
                selected = self.ollama_model_row.get_selected()
                self._llm_model_opts[:] = [
                    (v, ("✓ " + lbl if v in installed and not lbl.startswith("✓ ") else lbl))
                    for v, lbl in self._llm_model_opts
                ]
                self.ollama_model_row.set_model(
                    Gtk.StringList.new([lbl for _, lbl in self._llm_model_opts]))
                self.ollama_model_row.set_selected(selected)
            finally:
                self._updating = False
            return False

        def work():
            GLib.idle_add(apply_marks, self.win._installed_ollama_models())
        threading.Thread(target=work, daemon=True).start()

    # ── Key capture (press a key to set it) ───────────────────────────────

    def _make_capture_row(self, combo_attr: str, opts_attr: str) -> Adw.ActionRow:
        row = Adw.ActionRow(title="… oder Taste drücken zum Festlegen")
        btn = Gtk.Button(valign=Gtk.Align.CENTER)
        btn.set_child(Adw.ButtonContent(icon_name="input-keyboard-symbolic",
                                        label="Taste erfassen"))
        btn.add_css_class("flat")
        btn.connect("clicked", lambda *_: self._start_capture(combo_attr, opts_attr, btn))
        row.add_suffix(btn)
        row.set_activatable_widget(btn)
        return row

    def _start_capture(self, combo_attr: str, opts_attr: str, btn: Gtk.Button) -> None:
        if self._capturing is not None:
            return
        self._capturing = (combo_attr, opts_attr, btn)
        btn.set_child(Adw.ButtonContent(icon_name="input-keyboard-symbolic",
                                        label="Taste drücken … (Esc bricht ab)"))
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
        # set_selected fires the bound handler -> the key is applied instantly
        # and the restart banner appears.
        combo.set_selected(vals.index(value))
        self._end_capture(btn)
        self._toast(f"Taste erfasst: {key_label(value)}")
        return True

    def _end_capture(self, btn: Gtk.Button) -> None:
        btn.set_child(Adw.ButtonContent(icon_name="input-keyboard-symbolic",
                                        label="Taste erfassen"))
        if self._capture_ctrl is not None:
            self.remove_controller(self._capture_ctrl)
            self._capture_ctrl = None
        self._capturing = None


class SettingsWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application):
        super().__init__(application=app, title="Whisper Dictation")
        install_app_css()
        self.config = load_config()
        # Restore the last window geometry (stored in the config on close).
        self.set_default_size(int(self.config.get("window_width", 940)),
                              int(self.config.get("window_height", 720)))
        if self.config.get("window_maximized"):
            self.maximize()
        self.set_size_request(360, 480)
        self._prefs_dialog: PrefsDialog | None = None

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

        # Narrow windows: the switcher moves into a bottom bar (HIG pattern).
        self.switcher_bar = Adw.ViewSwitcherBar()
        self.switcher_bar.set_stack(self.stack)
        toolbar.add_bottom_bar(self.switcher_bar)
        try:
            bp = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 620sp"))
            self._narrow_title = Adw.WindowTitle(title="Whisper Dictation")
            bp.add_setter(header, "title-widget", self._narrow_title)
            bp.add_setter(self.switcher_bar, "reveal", True)
            self.add_breakpoint(bp)
        except Exception:
            pass  # very old libadwaita: keep the header switcher only

        menu = Gio.Menu()
        section = Gio.Menu()
        section.append("Einstellungen", "win.prefs")
        section.append("Tastenkürzel", "win.shortcuts")
        menu.append_section(None, section)
        section = Gio.Menu()
        section.append("Daemon neu starten", "win.restart")
        section.append("Daemon stoppen", "win.stop")
        section.append("Diagnose", "win.diagnose")
        section.append("Log öffnen", "win.log")
        menu.append_section(None, section)
        section = Gio.Menu()
        section.append("Über Whisper Dictation", "win.about")
        menu.append_section(None, section)
        menu_button = Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu)
        menu_button.set_tooltip_text("Hauptmenü")
        header.pack_end(menu_button)
        for name, handler in (
            ("prefs", self._on_prefs), ("shortcuts", self._on_shortcuts),
            ("restart", self._on_restart), ("stop", self._on_stop),
            ("log", self._on_log), ("diagnose", self._on_diagnose),
            ("about", self._on_about), ("search", self._on_search),
            ("record", self._on_record_accel),
            ("view1", lambda *_: self.stack.set_visible_child_name("werkbank")),
            ("view2", lambda *_: self.stack.set_visible_child_name("rekorder")),
            ("view3", lambda *_: self.stack.set_visible_child_name("verlauf")),
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


        # Open on the Werkbank.
        self.stack.set_visible_child_name("werkbank")
        self.stack.connect("notify::visible-child-name", self._on_view_changed)
        self._on_view_changed()
        # Stop meter/playback processes and persist the geometry on close.
        self.connect("close-request", self._on_close)

    def _on_close(self, *_a) -> bool:
        self.recorder.on_hidden()
        if self._prefs_dialog is not None:
            # Debounced edits + scheduled reloads would die with the main loop.
            self._prefs_dialog.flush_now()
        try:
            cfg = load_config()
            cfg["window_width"] = self.get_width()
            cfg["window_height"] = self.get_height()
            cfg["window_maximized"] = bool(self.is_maximized())
            save_config(cfg)
        except Exception:
            pass
        return False

    def _on_prefs(self, *_a) -> None:
        if self._prefs_dialog is None:
            self._prefs_dialog = PrefsDialog(self)
            self._prefs_dialog.connect("closed", self._on_prefs_closed)
        self._prefs_dialog.present(self)

    def _on_prefs_closed(self, *_a) -> None:
        self._prefs_dialog = None

    def _on_shortcuts(self, *_a) -> None:
        entries = (
            ("<primary>comma", "Einstellungen öffnen"),
            ("<primary>r", "Aufnahme starten/stoppen (Werkbank)"),
            ("<primary>f", "Verlauf durchsuchen"),
            ("<primary>1", "Werkbank"),
            ("<primary>2", "Rekorder"),
            ("<primary>3", "Verlauf"),
            ("<primary>w", "Fenster schließen"),
            ("<primary>q", "Beenden"),
        )
        try:
            dlg = Adw.ShortcutsDialog()
            sec = Adw.ShortcutsSection(title="Allgemein")
            for accel, title in entries:
                sec.add(Adw.ShortcutsItem.new(title, accel))
            dlg.add(sec)
            dlg.present(self)
        except (AttributeError, TypeError):
            # libadwaita < 1.8: plain list fallback
            body = "\n".join(
                f"{Gtk.accelerator_get_label(*Gtk.accelerator_parse(a)[1:])} — {t}"
                for a, t in entries
            )
            dlg = Adw.AlertDialog(heading="Tastenkürzel", body=body)
            dlg.add_response("ok", "OK")
            dlg.present(self)

    def _on_search(self, *_a) -> None:
        self.stack.set_visible_child_name("verlauf")
        self._history_search.grab_focus()

    def _on_record_accel(self, *_a) -> None:
        self.stack.set_visible_child_name("werkbank")
        self.workbench._toggle_record()

    def _on_view_changed(self, *_a) -> None:
        name = self.stack.get_visible_child_name()
        if name == "verlauf":
            self._refresh_history()
        if name == "rekorder":
            self.recorder.on_shown()
        else:
            self.recorder.on_hidden()  # stop live meters / playback when leaving

    # ── Verlauf (history) ───────────────────────────────────────────────────────

    def _build_history_page(self) -> Gtk.Box:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        search_clamp = Adw.Clamp(maximum_size=940, tightening_threshold=720,
                                 margin_top=12, margin_start=12, margin_end=12)
        self._history_search = Gtk.SearchEntry(placeholder_text="Verlauf durchsuchen …")
        self._history_search.connect("search-changed", lambda *_: self._filter_history())
        search_clamp.set_child(self._history_search)
        box.append(search_clamp)

        self._history_page = Adw.PreferencesPage(vexpand=True)
        self._history_group = Adw.PreferencesGroup(
            title="Verlauf der Diktate",
            description="Zuletzt eingesprochene Texte — kopieren oder in der Werkbank weiterbearbeiten.",
        )
        clear = Gtk.Button(label="Alle löschen", valign=Gtk.Align.CENTER)
        clear.add_css_class("flat")
        clear.connect("clicked", self._clear_history)
        self._history_group.set_header_suffix(clear)
        self._history_page.add(self._history_group)
        box.append(self._history_page)

        self._history_empty = Adw.StatusPage(
            icon_name="document-open-recent-symbolic",
            title="Noch keine Diktate",
            description="Diktiere irgendwo Text per Hotkey — er erscheint dann hier "
                        "zum Nachschlagen und Weiterbearbeiten.",
            vexpand=True,
        )
        self._history_empty.set_visible(False)
        box.append(self._history_empty)
        return box

    def _refresh_history(self) -> None:
        for row in self._history_rows:
            self._history_group.remove(row)
        self._history_rows = []
        entries = read_history(100)
        has_entries = bool(entries)
        self._history_page.set_visible(has_entries)
        self._history_search.set_visible(has_entries)
        self._history_empty.set_visible(not has_entries)
        if not has_entries:
            return
        for entry in reversed(entries):
            text = str(entry.get("text", "")).strip()
            ts = entry.get("ts")
            preview = (text[:90] + "…") if len(text) > 90 else (text or "(leer)")
            row = Adw.ExpanderRow(subtitle=format_ts(ts))
            row.set_title(GLib.markup_escape_text(preview))
            row._search_text = text.lower()

            # Actions appear on hover/focus only — rows stay calm.
            copy = Gtk.Button(icon_name="edit-copy-symbolic", valign=Gtk.Align.CENTER)
            copy.add_css_class("flat")
            copy.add_css_class("row-actions")
            copy.set_tooltip_text("Kopieren")
            copy.connect("clicked", lambda _b, t=text: self._copy_text(t))
            row.add_suffix(copy)
            raw = str(entry.get("raw", "")).strip()
            if raw and raw != text:
                raw_btn = Gtk.Button(icon_name="edit-undo-symbolic", valign=Gtk.Align.CENTER)
                raw_btn.add_css_class("flat")
                raw_btn.add_css_class("row-actions")
                raw_btn.set_tooltip_text("Rohtext kopieren (vor KI-Bearbeitung)")
                raw_btn.connect("clicked", lambda _b, t=raw: self._copy_text(t))
                row.add_suffix(raw_btn)
            load = Gtk.Button(label="In Werkbank", valign=Gtk.Align.CENTER)
            load.add_css_class("flat")
            load.add_css_class("row-actions")
            load.connect("clicked", lambda _b, t=text: self._load_to_workbench(t))
            row.add_suffix(load)
            trash = Gtk.Button(icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER)
            trash.add_css_class("flat")
            trash.add_css_class("row-actions")
            trash.set_tooltip_text("Eintrag löschen")
            trash.connect("clicked", lambda _b, t=ts: self._delete_history_entry(t))
            row.add_suffix(trash)

            full = Gtk.Label(label=text, wrap=True, xalign=0, selectable=True,
                             margin_top=8, margin_bottom=10,
                             margin_start=14, margin_end=14)
            row.add_row(full)

            self._history_group.add(row)
            self._history_rows.append(row)
        self._filter_history()

    def _filter_history(self) -> None:
        needle = self._history_search.get_text().strip().lower()
        for row in self._history_rows:
            row.set_visible(not needle or needle in getattr(row, "_search_text", ""))

    def _delete_history_entry(self, ts) -> None:
        try:
            lines = HISTORY_FILE.read_text(encoding="utf-8").splitlines()
            kept = []
            for line in lines:
                try:
                    if json.loads(line).get("ts") == ts:
                        continue
                except Exception:
                    pass
                kept.append(line)
            HISTORY_FILE.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        except Exception:
            pass
        self._refresh_history()
        self._toast("Eintrag gelöscht")

    def _copy_text(self, text: str) -> None:
        ok = copy_to_clipboard(text)
        self._toast("In Zwischenablage kopiert ✓" if ok else "Kopieren fehlgeschlagen (wl-copy fehlt?)")

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

    def _refresh_status(self) -> None:
        if self._prefs_dialog is not None:
            self._prefs_dialog.refresh_status()

    def _toast(self, text: str) -> None:
        self.toasts.add_toast(Adw.Toast.new(text))

    def _run_daemon(self, arg: str) -> tuple[int, str]:
        result = subprocess.run(
            [str(DAEMON_SCRIPT), arg], capture_output=True, text=True, check=False,
        )
        return result.returncode, (result.stdout + result.stderr).strip()

    def _daemon_action(self, arg: str, ok_msg: str, err_prefix: str) -> None:
        """Run a daemon control command off the main loop (restart = model
        reload = seconds); toast the outcome when done."""
        def work():
            code, output = self._run_daemon(arg)
            GLib.idle_add(done, code, output)

        def done(code: int, output: str) -> bool:
            self._toast(ok_msg if code == 0 else f"{err_prefix}: {output}")
            self._refresh_status()
            return False

        threading.Thread(target=work, daemon=True).start()

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

    def _on_restart(self, *_a) -> None:
        self._daemon_action("--restart", "Daemon neu gestartet.", "Neustart fehlgeschlagen")

    def _on_stop(self, *_a) -> None:
        self._daemon_action("--stop", "Daemon gestoppt.", "Stop-Fehler")

    def _on_log(self, *_a) -> None:
        if LOG_FILE.exists():
            Gio.AppInfo.launch_default_for_uri(GLib.filename_to_uri(str(LOG_FILE), None), None)
        else:
            self._toast("Noch keine Logdatei vorhanden.")

    def _on_diagnose(self, *_a) -> None:
        model = str(load_config().get("ollama_model", ""))

        def show(running: bool, backend: str, device: str, up: bool, installed: set) -> bool:
            body = (
                f"Daemon: {'läuft' if running else 'gestoppt'}\n"
                f"Backend: {backend}\n"
                f"Gerät: {device}\n"
                f"Ollama-Server: {'läuft' if up else 'aus'}\n"
                f"Cleanup-Modell ({model}): "
                f"{'installiert' if model in installed else ('nicht installiert' if up else '?')}"
            )
            dlg = Adw.AlertDialog(heading="Diagnose", body=body)
            dlg.add_response("ok", "OK")
            dlg.present(self)
            return False

        def work():
            device = backend = "?"
            try:
                log = LOG_FILE.read_text(errors="ignore")
                dev = re.findall(r"using device=(\w+)", log)
                be = re.findall(r"backend=(\w+)", log)
                device = dev[-1] if dev else "?"
                backend = be[-1] if be else "?"
            except Exception:
                pass
            GLib.idle_add(show, daemon_running(), backend, device, *self._ollama_list())

        threading.Thread(target=work, daemon=True).start()

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
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", lambda *_: self.quit())
        self.add_action(quit_action)
        for action, accels in {
            "app.quit": ["<primary>q"],
            "window.close": ["<primary>w"],
            "win.prefs": ["<primary>comma"],
            "win.search": ["<primary>f"],
            "win.record": ["<primary>r"],
            "win.view1": ["<primary>1"],
            "win.view2": ["<primary>2"],
            "win.view3": ["<primary>3"],
        }.items():
            self.set_accels_for_action(action, accels)

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
