#!/usr/bin/env python3
"""Whisper Dictation settings — a libadwaita (GNOME) app.

Writes ~/.config/whisper-dictation/config.json and applies changes live via
`whisper-dictation.sh --reload` (no model reload unless model/device changed).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk


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
LOG_FILE = Path.home() / ".cache" / "whisper-dictation" / "daemon.log"
DAEMON_SCRIPT = PROJECT_ROOT / "bin" / "whisper-dictation.sh"

DEFAULT_CONFIG = {
    "double_tap_key": "ctrl_r",
    "double_tap_window_ms": 400,
    "language": "de",
    "model": "turbo",
    "backend": "auto",
    "ov_device": "AUTO",
    "beam_size": 5,
    "vad_filter": True,
    "hotwords": "",
    "sound_cue": True,
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


class SettingsWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application):
        super().__init__(application=app, title="Whisper Dictation")
        self.set_default_size(560, 720)
        self.config = load_config()
        self.device_options = detect_alsa_capture_devices()

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
        menu.append("Daemon starten", "win.start")
        menu.append("Daemon neu starten", "win.restart")
        menu.append("Daemon stoppen", "win.stop")
        menu.append("Log oeffnen", "win.log")
        menu_button = Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu)
        header.pack_end(menu_button)
        for name, handler in (
            ("start", self._on_start), ("restart", self._on_restart),
            ("stop", self._on_stop), ("log", self._on_log),
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

        # ── Eingabe ──────────────────────────────────────────────────────────
        inp = Adw.PreferencesGroup(title="Eingabe")
        page.add(inp)
        self.hotkey_row = self._combo("Doppeltaste", HOTKEY_OPTIONS, str(self.config["double_tap_key"]))
        inp.add(self.hotkey_row)
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
        self.ollama_model_row = Adw.EntryRow(title="Ollama-Modell")
        self.ollama_model_row.set_text(str(self.config.get("ollama_model", "qwen2.5:7b")))
        llm.add(self.ollama_model_row)
        self.llm_toggle_row = self._combo(
            "Umschalt-Taste (Doppel-Tap)", LLM_TOGGLE_OPTIONS,
            str(self.config.get("llm_toggle_key", "")).lower(),
        )
        self.llm_toggle_row.set_subtitle("Schaltet Cleanup an/aus. Muss sich von der Aufnahme-Taste unterscheiden.")
        llm.add(self.llm_toggle_row)

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
            "sound_cue": bool(self.sound_row.get_active()),
            "double_tap_key": self._combo_value(self.hotkey_row, HOTKEY_OPTIONS),
            "double_tap_window_ms": int(self.double_tap_row.get_value()),
            "paste_mode": self._combo_value(self.paste_row, PASTE_OPTIONS),
            "max_record_seconds": int(self.max_record_row.get_value()),
            "record_device": self._combo_value(self.device_row, self.device_options),
            "initial_prompt": self._prompt_text(),
            "ollama_postprocess": bool(self.ollama_row.get_active()),
            "ollama_model": self.ollama_model_row.get_text().strip() or "qwen2.5:7b",
            "llm_toggle_key": self._combo_value(self.llm_toggle_row, LLM_TOGGLE_OPTIONS),
        })
        return config

    def _run_daemon(self, arg: str) -> tuple[int, str]:
        result = subprocess.run(
            [str(DAEMON_SCRIPT), arg], capture_output=True, text=True, check=False,
        )
        return result.returncode, (result.stdout + result.stderr).strip()

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
