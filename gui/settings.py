#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import threading
from pathlib import Path

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gio, Gtk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR   = Path.home() / ".config" / "whisper-dictation"
CONFIG_FILE  = CONFIG_DIR / "config.json"
LOG_FILE     = Path.home() / ".cache" / "whisper-dictation" / "daemon.log"
DAEMON_SCRIPT = PROJECT_ROOT / "bin" / "whisper-dictation.sh"

DEFAULT_CONFIG = {
    "double_tap_key": "ctrl_r",
    "double_tap_window_ms": 400,
    "language": "de",
    "model": "turbo",
    "paste_mode": "auto",
    "record_device": "default",
    "max_record_seconds": 180,
    "initial_prompt": "",
    "push_to_talk": False,
    "postprocess": False,
    "postprocess_model": "qwen3:14b-q4_K_M",
    "postprocess_prompt": "",
}

# value, display label, star-rating (1-5), short review
MODEL_OPTIONS: list[tuple[str, str, int, str]] = [
    ("turbo",     "turbo  ★★★★★  Empfohlen",   5, "Beste Wahl. Genauso gut wie large-v3, aber 6x schneller. Ideal fuer Deutsch."),
    ("large-v3",  "large-v3  ★★★★★  Max. Qualitaet", 5, "Hoechste Genauigkeit, besonders bei Fachbegriffen. ~3x langsamer als turbo."),
    ("medium",    "medium  ★★★☆☆  Mittelklasse", 3, "Gut fuer schwaeachere Hardware ohne dedizierte GPU."),
    ("small",     "small  ★★☆☆☆  Leichtgewicht", 2, "Schnell, aber spuerbar ungenauer. Nur fuer sehr alte Hardware."),
    ("base",      "base  ★☆☆☆☆  Minimal",       1, "Kaum brauchbar fuer Deutsch. Nur fuer Tests."),
    ("tiny",      "tiny  ★☆☆☆☆  Winzig",        1, "Extrem ungenau. Nicht fuer produktiven Einsatz geeignet."),
    ("large-v2",  "large-v2  ★★★★☆  Veraltet",  4, "Vorgaenger von large-v3. Nur fuer Vergleiche interessant."),
    ("medium.en", "medium.en  ★★★☆☆  Nur Englisch", 3, "Englisch-optimiert, etwas schneller als medium."),
    ("small.en",  "small.en  ★★☆☆☆  Nur Englisch", 2, "Englisch-only, kompakt."),
    ("base.en",   "base.en  ★☆☆☆☆  Nur Englisch", 1, "Englisch-only, minimal."),
    ("tiny.en",   "tiny.en  ★☆☆☆☆  Nur Englisch", 1, "Englisch-only, winzig."),
]
MODEL_OPTS_SIMPLE = [(v, l) for v, l, _, _ in MODEL_OPTIONS]
MODEL_REVIEWS     = {v: (s, r) for v, _, s, r in MODEL_OPTIONS}

LANGUAGE_OPTIONS = [
    ("de",  "Deutsch (de)"),
    ("en",  "Englisch (en)"),
    ("",    "Automatisch erkennen"),
    ("fr",  "Französisch (fr)"),
    ("es",  "Spanisch (es)"),
    ("it",  "Italienisch (it)"),
    ("pt",  "Portugiesisch (pt)"),
    ("nl",  "Niederländisch (nl)"),
    ("pl",  "Polnisch (pl)"),
    ("ru",  "Russisch (ru)"),
    ("zh",  "Chinesisch (zh)"),
    ("ja",  "Japanisch (ja)"),
]

HOTKEY_OPTIONS = [
    ("ctrl_r", "Rechtes Strg"),
    ("ctrl_l", "Linkes Strg"),
    ("alt_r",  "Rechtes Alt"),
    ("alt_l",  "Linkes Alt"),
    ("f8",     "F8"),
    ("f9",     "F9"),
    ("f10",    "F10"),
    ("pause",  "Pause"),
]

PASTE_OPTIONS = [
    ("auto",        "Auto (empfohlen)"),
    ("ctrl_v",      "Ctrl+V"),
    ("cmd_v",       "Cmd+V (macOS)"),
    ("ctrl_shift_v","Ctrl+Shift+V (Terminal)"),
    ("shift_insert","Shift+Insert (xterm)"),
]


def detect_alsa_capture_devices() -> list[tuple[str, str]]:
    devices: list[tuple[str, str]] = [("default", "Standard-Mikrofon")]
    try:
        out = subprocess.run(
            ["arecord", "--list-devices"], capture_output=True, text=True, check=False,
        ).stdout
        for line in out.splitlines():
            m = re.match(r"card\s+(\d+):\s+\S+\s+\[(.+?)\].*device\s+(\d+):\s+\S+\s+\[(.+?)\]", line)
            if m:
                card, card_name, dev, dev_name = m.group(1), m.group(2), m.group(3), m.group(4)
                hw = f"plughw:{card},{dev}"
                devices.append((hw, f"{card_name} / {dev_name}  ({hw})"))
    except Exception:
        pass
    return devices


OLLAMA_RATINGS: list[tuple[str, int, str]] = [
    ("gemma4:27b", 5, "Neu & stark. Google Gemma 4 27B — Top 3 weltweit, exzellent fuer Deutsch/Englisch. Empfohlen fuer maximale Qualitaet."),
    ("qwen3:32b",  5, "Maximale Qualitaet. Erkennbar besser bei langen Saetzen & Fachbegriffen. ~3-5 Sek. auf GPU."),
    ("qwen3:14b",  4, "Beste Balance. Schnell (~1-2 Sek.), sehr gute Bereinigung. Ideal fuer den taeglichen Einsatz."),
    ("qwen3:8b",   3, "Gut & schnell. Merklich kompakter als 14b, fuer die meisten Texte ausreichend."),
    ("qwen3:4b",   2, "Kompakt, aber bei komplexen Saetzen spuerbar ungenauer als 14b/8b."),
    ("qwen3:1b",   1, "Sehr schnell, aber Bereinigungsqualitaet gering — nur fuer schwache Hardware."),
    ("llama3",     3, "Gutes Allround-Modell, Deutsch etwas schlechter als Qwen3."),
    ("mistral",    2, "Solide fuer Englisch, fuer Deutsche Transkripte nicht ideal."),
    ("gemma3",     3, "Vorgaenger von Gemma 4. Gute Qualitaet bei kurzen Texten."),
]

def _ollama_rating_for(name: str) -> tuple[int, str]:
    nl = name.lower()
    for prefix, stars, review in OLLAMA_RATINGS:
        if nl.startswith(prefix.lower()):
            return stars, review
    return 3, "Kein Bewertungsprofil fuer dieses Modell vorhanden."


def detect_ollama_models() -> list[tuple[str, str]]:
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            data = json.loads(r.read())
        result = []
        for m in data.get("models", []):
            name = m["name"]
            stars, _ = _ollama_rating_for(name)
            bar = "★" * stars + "☆" * (5 - stars)
            result.append((name, f"{name}  {bar}"))
        return result if result else [("qwen3:14b-q4_K_M", "qwen3:14b-q4_K_M  ★★★★★")]
    except Exception:
        return [("qwen3:14b-q4_K_M", "qwen3:14b-q4_K_M  ★★★★★  (Ollama nicht erreichbar)")]


def load_config() -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8")
    loaded = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(loaded)
    return cfg


def save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def daemon_running() -> bool:
    r = subprocess.run(["pgrep", "-f", "dictation/daemon.py"], capture_output=True, text=True, check=False)
    return r.returncode == 0 and bool(r.stdout.strip())


# ── Main Window ───────────────────────────────────────────────────────────────

class SettingsWindow(Adw.ApplicationWindow):
    def __init__(self, app: Gtk.Application):
        super().__init__(application=app, title="Whisper Dictation")
        self.set_default_size(660, -1)
        self.config = load_config()
        self._apply_button: Gtk.Button | None = None

        # CSS
        css = Gtk.CssProvider()
        css.load_from_data(b"""
            .section-title { font-weight: bold; font-size: 11pt; margin-top: 6px; }
            .review-box { background: alpha(currentColor, 0.06); border-radius: 8px; padding: 8px 12px; }
            .status-running { color: #26a269; }
            .status-stopped { color: #e5a50a; }
            dropdown > button { border-radius: 6px; }
            spinbutton { border-radius: 6px; }
            .textarea-field { border-radius: 6px; border: 1px solid alpha(currentColor, 0.2); }
            .placeholder-active { color: alpha(currentColor, 0.45); }
        """)
        Gtk.StyleContext.add_provider_for_display(
            self.get_display(), css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        toolbar = Adw.ToolbarView()
        self.set_content(toolbar)

        header = Adw.HeaderBar()
        header.set_title_widget(Adw.WindowTitle(title="Whisper Dictation", subtitle="Lokale Spracherkennung"))
        toolbar.add_top_bar(header)

        scroll = Gtk.ScrolledWindow(vexpand=True, hscrollbar_policy=Gtk.PolicyType.NEVER)
        toolbar.set_content(scroll)

        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10,
                        margin_top=12, margin_bottom=12, margin_start=20, margin_end=20)
        scroll.set_child(outer)

        # ── Whisper-Modell ────────────────────────────────────────────────────
        outer.append(self._section("Whisper-Modell"))

        self.model_opts = MODEL_OPTS_SIMPLE
        self.model_dropdown = self._dropdown(self.model_opts, str(self.config["model"]))
        self.model_dropdown.connect("notify::selected", self._on_model_changed)
        outer.append(self._row("Modell", self.model_dropdown))

        self.model_review = Gtk.Label(wrap=True, xalign=0)
        self.model_review.add_css_class("review-box")
        self._update_model_review()
        outer.append(self.model_review)

        self.language_dropdown = self._dropdown(LANGUAGE_OPTIONS, str(self.config["language"]))
        outer.append(self._row("Sprache", self.language_dropdown))

        self.initial_prompt_entry = self._textarea(
            str(self.config["initial_prompt"]),
            "z.B. Python, CUDA, Fachbegriffe — hilft Whisper beim Erkennen"
        )
        outer.append(self._row("Whisper-Prompt", self.initial_prompt_entry))

        # ── Aufnahme ──────────────────────────────────────────────────────────
        outer.append(self._section("Aufnahme"))

        self.device_options = detect_alsa_capture_devices()
        self.device_dropdown = self._dropdown(self.device_options, str(self.config["record_device"]))
        outer.append(self._row("Mikrofon", self.device_dropdown))

        self.max_record_spin = Gtk.SpinButton.new_with_range(15, 900, 5)
        self.max_record_spin.set_value(float(self.config["max_record_seconds"]))
        outer.append(self._row("Max. Aufnahmedauer (s)", self.max_record_spin))

        # ── Steuerung ─────────────────────────────────────────────────────────
        outer.append(self._section("Steuerung"))

        self.hotkey_dropdown = self._dropdown(HOTKEY_OPTIONS, str(self.config["double_tap_key"]))
        outer.append(self._row("Aktivierungstaste", self.hotkey_dropdown))

        self.ptt_switch = self._switch(bool(self.config.get("push_to_talk", False)))
        outer.append(self._row("Push-to-Talk  (halten statt doppelt tippen)", self.ptt_switch, expand=False))

        self.double_tap_spin = Gtk.SpinButton.new_with_range(150, 1200, 10)
        self.double_tap_spin.set_value(float(self.config["double_tap_window_ms"]))
        outer.append(self._row("Doppeltipp-Fenster (ms)", self.double_tap_spin))

        self.paste_dropdown = self._dropdown(PASTE_OPTIONS, str(self.config["paste_mode"]))
        outer.append(self._row("Einfuge-Modus", self.paste_dropdown))

        # ── LLM-Bereinigung ───────────────────────────────────────────────────
        outer.append(self._section("LLM-Bereinigung (Ollama)"))

        hint = Gtk.Label(
            label="Verbessert den transkribierten Text: entfernt Fuellwoerter, korrigiert Grammatik und Zeichensetzung.",
            wrap=True, xalign=0,
        )
        hint.add_css_class("dim-label")
        outer.append(hint)

        self.postprocess_switch = self._switch(bool(self.config.get("postprocess", False)))
        outer.append(self._row("Bereinigung aktivieren", self.postprocess_switch, expand=False))

        self.ollama_options = detect_ollama_models()
        self.ollama_dropdown = self._dropdown(
            self.ollama_options, str(self.config.get("postprocess_model", "qwen3:14b-q4_K_M"))
        )
        self.ollama_dropdown.connect("notify::selected", self._on_ollama_model_changed)
        outer.append(self._row("Ollama-Modell", self.ollama_dropdown))

        self.ollama_review = Gtk.Label(wrap=True, xalign=0)
        self.ollama_review.add_css_class("review-box")
        self._update_ollama_review()
        outer.append(self.ollama_review)

        self.postprocess_thinking_switch = self._switch(bool(self.config.get("postprocess_thinking", False)))
        outer.append(self._row("Thinking-Modus  (langsamer, gründlicher)", self.postprocess_thinking_switch, expand=False))

        self.postprocess_prompt_entry = self._textarea(
            str(self.config.get("postprocess_prompt", "")),
            "Leer = Standard-Prompt wird verwendet"
        )
        outer.append(self._row("Eigene Anweisung", self.postprocess_prompt_entry))

        # ── Status ────────────────────────────────────────────────────────────
        outer.append(self._section("Status"))

        self.status_label = Gtk.Label(wrap=True, xalign=0)
        outer.append(self.status_label)
        self._refresh_status()

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8, margin_top=4)
        btn_box.set_homogeneous(True)
        outer.append(btn_box)

        self._apply_button = Gtk.Button(label="Speichern & Neustart")
        self._apply_button.add_css_class("suggested-action")
        self._apply_button.connect("clicked", self._on_apply_clicked)
        btn_box.append(self._apply_button)

        start_btn = Gtk.Button(label="Starten")
        start_btn.connect("clicked", self._on_start_clicked)
        btn_box.append(start_btn)

        stop_btn = Gtk.Button(label="Stoppen")
        stop_btn.connect("clicked", self._on_stop_clicked)
        btn_box.append(stop_btn)

        log_btn = Gtk.Button(label="Log")
        log_btn.connect("clicked", self._on_log_clicked)
        btn_box.append(log_btn)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _section(self, title: str) -> Gtk.Label:
        lbl = Gtk.Label(label=title, xalign=0)
        lbl.add_css_class("section-title")
        lbl.add_css_class("heading")
        return lbl

    def _row(self, label: str, widget: Gtk.Widget, expand: bool = True) -> Gtk.Box:
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        lbl = Gtk.Label(label=label, xalign=0, width_chars=28, wrap=True)
        lbl.add_css_class("dim-label")
        lbl.set_halign(Gtk.Align.START)
        if expand:
            widget.set_hexpand(True)
        else:
            widget.set_halign(Gtk.Align.START)
        box.append(lbl)
        box.append(widget)
        return box

    def _textarea(self, text: str, placeholder: str = "") -> Gtk.TextView:
        tv = Gtk.TextView()
        tv.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        tv.set_top_margin(6)
        tv.set_bottom_margin(6)
        tv.set_left_margin(8)
        tv.set_right_margin(8)
        tv.set_size_request(-1, 52)
        tv.add_css_class("card")

        is_placeholder = not text.strip()
        tv.get_buffer().set_text(placeholder if is_placeholder else text)
        if is_placeholder:
            tv.add_css_class("placeholder-active")

        def _on_enter(ctrl, *_):
            widget = ctrl.get_widget()
            if "placeholder-active" in widget.get_css_classes():
                widget.get_buffer().set_text("")
                widget.remove_css_class("placeholder-active")

        def _on_leave(ctrl, *_):
            widget = ctrl.get_widget()
            buf = widget.get_buffer()
            if not buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).strip():
                buf.set_text(placeholder)
                widget.add_css_class("placeholder-active")

        fc_in = Gtk.EventControllerFocus()
        fc_in.connect("enter", _on_enter)
        fc_out = Gtk.EventControllerFocus()
        fc_out.connect("leave", _on_leave)
        tv.add_controller(fc_in)
        tv.add_controller(fc_out)
        return tv

    def _textarea_text(self, tv: Gtk.TextView) -> str:
        if "placeholder-active" in tv.get_css_classes():
            return ""
        buf = tv.get_buffer()
        return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).strip()

    def _dropdown(self, options: list[tuple[str, str]], current: str) -> Gtk.DropDown:
        labels = [lbl for _, lbl in options]
        dd = Gtk.DropDown.new_from_strings(labels)
        idx = next((i for i, (v, _) in enumerate(options) if v == current), 0)
        dd.set_selected(idx)
        return dd

    def _selected(self, dd: Gtk.DropDown, options: list[tuple[str, str]]) -> str:
        return options[dd.get_selected()][0]

    def _switch(self, active: bool) -> Gtk.Switch:
        sw = Gtk.Switch()
        sw.set_active(active)
        sw.set_valign(Gtk.Align.CENTER)
        return sw

    def _update_model_review(self) -> None:
        model = self._selected(self.model_dropdown, self.model_opts)
        stars, review = MODEL_REVIEWS.get(model, (3, ""))
        bar = "★" * stars + "☆" * (5 - stars)
        self.model_review.set_label(f"{bar}  {review}")

    def _refresh_status(self, extra: str = "") -> None:
        running = daemon_running()
        if running:
            self.status_label.set_label(f"Daemon laeuft  ·  Config: {CONFIG_FILE}" + (f"\n{extra}" if extra else ""))
            self.status_label.remove_css_class("status-stopped")
            self.status_label.add_css_class("status-running")
        else:
            self.status_label.set_label(f"Daemon gestoppt" + (f"\n{extra}" if extra else ""))
            self.status_label.remove_css_class("status-running")
            self.status_label.add_css_class("status-stopped")

    def _config_from_form(self) -> dict:
        return {
            "model":               self._selected(self.model_dropdown, self.model_opts),
            "language":            self._selected(self.language_dropdown, LANGUAGE_OPTIONS),
            "double_tap_key":      self._selected(self.hotkey_dropdown, HOTKEY_OPTIONS),
            "double_tap_window_ms":int(self.double_tap_spin.get_value()),
            "paste_mode":          self._selected(self.paste_dropdown, PASTE_OPTIONS),
            "max_record_seconds":  int(self.max_record_spin.get_value()),
            "record_device":       self._selected(self.device_dropdown, self.device_options),
            "initial_prompt":      self._textarea_text(self.initial_prompt_entry),
            "push_to_talk":        self.ptt_switch.get_active(),
            "postprocess":         self.postprocess_switch.get_active(),
            "postprocess_model":   self._selected(self.ollama_dropdown, self.ollama_options),
            "postprocess_thinking": self.postprocess_thinking_switch.get_active(),
            "postprocess_prompt":  self._textarea_text(self.postprocess_prompt_entry),
        }

    # ── Events ────────────────────────────────────────────────────────────────

    def _update_ollama_review(self) -> None:
        model = self._selected(self.ollama_dropdown, self.ollama_options)
        stars, review = _ollama_rating_for(model)
        bar = "★" * stars + "☆" * (5 - stars)
        self.ollama_review.set_label(f"{bar}  {review}")

    def _on_model_changed(self, *_: object) -> None:
        self._update_model_review()

    def _on_ollama_model_changed(self, *_: object) -> None:
        self._update_ollama_review()

    def _run_daemon_cmd(self, arg: str) -> tuple[int, str]:
        r = subprocess.run([str(DAEMON_SCRIPT), arg], capture_output=True, text=True, check=False)
        return r.returncode, (r.stdout + r.stderr).strip()

    def _run_async(self, btn: Gtk.Button, busy_label: str, arg: str, done_msg: str) -> None:
        original = btn.get_label()
        btn.set_label(busy_label)
        btn.set_sensitive(False)
        self._refresh_status(busy_label)

        def _work() -> None:
            code, out = self._run_daemon_cmd(arg)
            msg = done_msg if code == 0 else f"Fehler: {out}"
            GLib.idle_add(self._done_async, btn, original, msg)

        threading.Thread(target=_work, daemon=True).start()

    def _done_async(self, btn: Gtk.Button, original_label: str, msg: str) -> bool:
        btn.set_label(original_label)
        btn.set_sensitive(True)
        self._refresh_status(msg)
        return False

    def _on_apply_clicked(self, btn: Gtk.Button) -> None:
        save_config(self._config_from_form())
        self._run_async(btn, "Wird neu gestartet...", "--restart", "Gespeichert und neu gestartet.")

    def _on_start_clicked(self, btn: Gtk.Button) -> None:
        self._run_async(btn, "Startet...", "--restart", "Daemon gestartet.")

    def _on_stop_clicked(self, btn: Gtk.Button) -> None:
        self._run_async(btn, "Stoppt...", "--stop", "Daemon gestoppt.")

    def _on_log_clicked(self, _btn: Gtk.Button) -> None:
        if LOG_FILE.exists():
            Gio.AppInfo.launch_default_for_uri(f"file://{LOG_FILE}", None)
        self._refresh_status("Logdatei geöffnet.")


# ── App ───────────────────────────────────────────────────────────────────────

def main() -> int:
    import uuid
    app = Adw.Application(
        application_id=f"local.whisper.dictation.s{uuid.uuid4().hex[:12]}",
        flags=Gio.ApplicationFlags.NON_UNIQUE,
    )

    def on_activate(a):
        win = SettingsWindow(a)
        win.set_default_size(740, 860)
        win.present()

    app.connect("activate", on_activate)
    return app.run([])


if __name__ == "__main__":
    raise SystemExit(main())
