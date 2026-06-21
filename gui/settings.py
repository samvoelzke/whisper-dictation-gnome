#!/usr/bin/env python3
"""Whisper Dictation settings — a libadwaita (GNOME) app.

Writes ~/.config/whisper-dictation/config.json and applies changes live via
`whisper-dictation.sh --reload` (no model reload unless model/device changed).
"""

from __future__ import annotations

import json
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
DAEMON_SCRIPT = PROJECT_ROOT / "bin" / "whisper-dictation.sh"

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
    "paste_mode": "auto",
    "record_device": "default",
    "max_record_seconds": 180,
    "initial_prompt": "",  # filled with DEFAULT_INITIAL_PROMPT in the UI when empty
    "ollama_postprocess": False,
    "ollama_model": "qwen2.5:7b",
    "llm_toggle_key": "",
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


class WorkbenchWindow(Adw.Window):
    """Dictate into a scratchpad, then give the AI free-form instructions."""

    def __init__(self, parent: Adw.ApplicationWindow):
        super().__init__(title="Werkbank", transient_for=parent)
        self.set_default_size(600, 620)
        self.rec_proc: subprocess.Popen | None = None
        self.rec_wav: str | None = None

        toolbar = Adw.ToolbarView()
        self.set_content(toolbar)
        toolbar.add_top_bar(Adw.HeaderBar())

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10,
                      margin_top=12, margin_bottom=12, margin_start=12, margin_end=12)
        toolbar.set_content(box)

        rec_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.rec_btn = Gtk.Button(label="🔴 Aufnehmen")
        self.rec_btn.add_css_class("pill")
        self.rec_btn.connect("clicked", self._toggle_record)
        rec_row.append(self.rec_btn)
        self.status = Gtk.Label(label="Bereit", xalign=0)
        self.status.add_css_class("dim-label")
        rec_row.append(self.status)
        box.append(rec_row)

        scroller = Gtk.ScrolledWindow(vexpand=True)
        scroller.add_css_class("card")
        self.text_view = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD_CHAR,
            top_margin=8, bottom_margin=8, left_margin=8, right_margin=8,
        )
        scroller.set_child(self.text_view)
        box.append(scroller)

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
        copy_btn = Gtk.Button(label="Kopieren")
        copy_btn.connect("clicked", self._copy)
        bottom.append(copy_btn)
        box.append(bottom)

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
            return

        try:
            self.rec_proc.send_signal(signal.SIGINT)
            self.rec_proc.wait(timeout=3)
        except Exception:
            pass
        self.rec_proc = None
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
        instruction = self.instr.get_text().strip()
        text = self._text()
        if not instruction or not text:
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


class SettingsWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application):
        super().__init__(application=app, title="Whisper Dictation")
        self.set_default_size(560, 720)
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

        save_button = Gtk.Button(label="Speichern")
        save_button.add_css_class("suggested-action")
        save_button.connect("clicked", self._on_save)
        header.pack_start(save_button)

        menu = Gio.Menu()
        menu.append("Werkbank (Diktat + KI)", "win.werkbank")
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
            ("werkbank", self._on_werkbank),
        ):
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", handler)
            self.add_action(action)

        page = Adw.PreferencesPage()
        toolbar.set_content(page)

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

        # ── Audio + Erweitert ────────────────────────────────────────────────
        audio = Adw.PreferencesGroup(title="Audio")
        page.add(audio)
        self.device_row = self._combo("Mikrofon", self.device_options, str(self.config["record_device"]))
        audio.add(self.device_row)

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
        running = daemon_running()
        self.status_row.set_subtitle("● läuft" if running else "○ gestoppt")

    def _prompt_text(self) -> str:
        buf = self.prompt_view.get_buffer()
        return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).strip()

    def _toast(self, text: str) -> None:
        self.toasts.add_toast(Adw.Toast.new(text))

    def _config_from_form(self) -> dict:
        config = dict(self.config)
        config.update({
            "model": self._combo_value(self.model_row, MODEL_OPTIONS),
            "language": self._combo_value(self.language_row, LANGUAGE_OPTIONS),
            "hotwords": self.hotwords_row.get_text().strip(),
            "vad_filter": bool(self.vad_row.get_active()),
            "voice_commands": bool(self.voice_row.get_active()),
            "sound_cue": bool(self.sound_row.get_active()),
            "restore_clipboard": bool(self.clipboard_row.get_active()),
            "hotkey_mode": self._combo_value(self.mode_row, HOTKEY_MODE_OPTIONS),
            "double_tap_key": self._combo_value(self.hotkey_row, self._hotkey_opts),
            "double_tap_window_ms": int(self.double_tap_row.get_value()),
            "paste_mode": self._combo_value(self.paste_row, PASTE_OPTIONS),
            "max_record_seconds": int(self.max_record_row.get_value()),
            "record_device": self._combo_value(self.device_row, self.device_options),
            "initial_prompt": self._prompt_text(),
            "ollama_postprocess": bool(self.ollama_row.get_active()),
            "ollama_model": self._combo_value(self.ollama_model_row, self._llm_model_opts),
            "llm_toggle_key": self._combo_value(self.llm_toggle_row, self._llm_toggle_opts),
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
    RESTART_KEYS = ("double_tap_key", "llm_toggle_key")

    def _on_save(self, _button: Gtk.Button) -> None:
        old, new = self.config, self._config_from_form()
        self.config = new
        save_config(new)
        needs_restart = any(old.get(k) != new.get(k) for k in self.RESTART_KEYS)
        if needs_restart:
            code, output = self._run_daemon("--restart")
            msg = "Gespeichert & Daemon neu gestartet." if code == 0 else f"Neustart-Fehler: {output}"
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

    def _on_werkbank(self, *_a) -> None:
        if not IPC_SOCKET.exists():
            self._toast("Daemon läuft nicht — Werkbank braucht den laufenden Dienst.")
            return
        WorkbenchWindow(self).present()

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
