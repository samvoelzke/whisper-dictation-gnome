#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gio, Gtk


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
    "paste_mode": "auto",
    "record_device": "default",
    "max_record_seconds": 180,
    "initial_prompt": "",
}

MODEL_OPTIONS = [
    ("turbo", "turbo"),
    ("tiny", "tiny"),
    ("base", "base"),
    ("small", "small"),
    ("medium", "medium"),
    ("large-v3", "large-v3"),
    ("tiny.en", "tiny.en"),
    ("base.en", "base.en"),
    ("small.en", "small.en"),
    ("medium.en", "medium.en"),
    ("large-v2", "large-v2"),
]

MODEL_HINTS = {
    "turbo": "Schnell und sehr stark. Gute Standardwahl fuer lokale Diktate.",
    "tiny": "Extrem schnell, aber die geringste Genauigkeit.",
    "base": "Etwas genauer als tiny, immer noch sehr leicht.",
    "small": "Guter Mittelweg fuer viele Systeme.",
    "medium": "Deutlich genauer, aber merklich schwerer.",
    "large-v3": "Beste Qualitaet, braucht am meisten VRAM und Ladezeit.",
    "tiny.en": "Nur Englisch, maximal leichtgewichtig.",
    "base.en": "Nur Englisch, kompakt und etwas praeziser.",
    "small.en": "Nur Englisch, guter Kompromiss.",
    "medium.en": "Nur Englisch, stark aber schwerer.",
    "large-v2": "Aelteres grosses Modell. Meist nur noetig fuer direkte Vergleiche.",
}

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

PASTE_OPTIONS = [
    ("auto", "Auto"),
    ("ctrl_v", "Ctrl+V"),
    ("cmd_v", "Cmd+V (macOS)"),
    ("ctrl_shift_v", "Ctrl+Shift+V"),
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
        ["pgrep", "-f", "dictation/daemon.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


class SettingsWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application):
        super().__init__(application=app, title="Whisper Dictation")
        self.set_default_size(620, 520)
        self.config = load_config()

        outer = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=14,
            margin_top=18,
            margin_bottom=18,
            margin_start=18,
            margin_end=18,
        )
        self.set_child(outer)

        title = Gtk.Label(
            label="Lokale Whisper-Diktierfunktion fuer GNOME",
            xalign=0,
        )
        title.add_css_class("title-2")
        outer.append(title)

        subtitle = Gtk.Label(
            label=(
                "Die GUI schreibt dieselbe Config wie der Hintergrunddienst. "
                "Mit Speichern und Neustarten wird das gewaehlte Modell sofort aktiv."
            ),
            wrap=True,
            xalign=0,
        )
        outer.append(subtitle)

        grid = Gtk.Grid(column_spacing=12, row_spacing=12)
        outer.append(grid)

        self.model_dropdown = self._dropdown(MODEL_OPTIONS, str(self.config["model"]))
        self.hotkey_dropdown = self._dropdown(
            HOTKEY_OPTIONS, str(self.config["double_tap_key"])
        )
        self.paste_dropdown = self._dropdown(
            PASTE_OPTIONS, str(self.config["paste_mode"])
        )

        self.language_entry = Gtk.Entry(text=str(self.config["language"]))
        self.language_entry.set_placeholder_text("de, en oder leer fuer auto")

        self.double_tap_spin = Gtk.SpinButton.new_with_range(150, 1200, 10)
        self.double_tap_spin.set_value(float(self.config["double_tap_window_ms"]))

        self.max_record_spin = Gtk.SpinButton.new_with_range(15, 900, 5)
        self.max_record_spin.set_value(float(self.config["max_record_seconds"]))

        self.device_options = detect_alsa_capture_devices()
        self.device_dropdown = self._dropdown(
            self.device_options, str(self.config["record_device"])
        )
        self.initial_prompt_entry = Gtk.Entry(text=str(self.config["initial_prompt"]))
        self.initial_prompt_entry.set_placeholder_text("Optionaler Whisper-Prompt")

        self._attach_row(grid, 0, "Modell", self.model_dropdown)
        self._attach_row(grid, 1, "Sprache", self.language_entry)
        self._attach_row(grid, 2, "Doppeltaste", self.hotkey_dropdown)
        self._attach_row(grid, 3, "Double-Tap Fenster (ms)", self.double_tap_spin)
        self._attach_row(grid, 4, "Paste-Modus", self.paste_dropdown)
        self._attach_row(grid, 5, "Max. Aufnahme (s)", self.max_record_spin)
        self._attach_row(grid, 6, "Mikrofon", self.device_dropdown)
        self._attach_row(grid, 7, "Initial Prompt", self.initial_prompt_entry)

        self.model_hint = Gtk.Label(wrap=True, xalign=0)
        outer.append(self.model_hint)
        self._update_model_hint()
        self.model_dropdown.connect("notify::selected", self._on_model_changed)

        self.status_label = Gtk.Label(wrap=True, xalign=0)
        outer.append(self.status_label)
        self._set_status()

        button_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        outer.append(button_row)

        apply_button = Gtk.Button(label="Speichern und Daemon neu starten")
        apply_button.connect("clicked", self._on_apply_clicked)
        apply_button.add_css_class("suggested-action")
        button_row.append(apply_button)

        start_button = Gtk.Button(label="Daemon starten")
        start_button.connect("clicked", self._on_start_clicked)
        button_row.append(start_button)

        stop_button = Gtk.Button(label="Daemon stoppen")
        stop_button.connect("clicked", self._on_stop_clicked)
        button_row.append(stop_button)

        log_button = Gtk.Button(label="Log oeffnen")
        log_button.connect("clicked", self._on_log_clicked)
        button_row.append(log_button)

    def _attach_row(self, grid: Gtk.Grid, row: int, label_text: str, widget: Gtk.Widget) -> None:
        label = Gtk.Label(label=label_text, xalign=0)
        grid.attach(label, 0, row, 1, 1)
        widget.set_hexpand(True)
        grid.attach(widget, 1, row, 1, 1)

    def _dropdown(self, options: list[tuple[str, str]], current_value: str) -> Gtk.DropDown:
        labels = [label for _, label in options]
        dropdown = Gtk.DropDown.new_from_strings(labels)
        selected = next(
            (index for index, (value, _) in enumerate(options) if value == current_value),
            0,
        )
        dropdown.set_selected(selected)
        return dropdown

    def _selected_value(self, dropdown: Gtk.DropDown, options: list[tuple[str, str]]) -> str:
        return options[dropdown.get_selected()][0]

    def _config_from_form(self) -> dict:
        return {
            "model": self._selected_value(self.model_dropdown, MODEL_OPTIONS),
            "language": self.language_entry.get_text().strip(),
            "double_tap_key": self._selected_value(self.hotkey_dropdown, HOTKEY_OPTIONS),
            "double_tap_window_ms": int(self.double_tap_spin.get_value()),
            "paste_mode": self._selected_value(self.paste_dropdown, PASTE_OPTIONS),
            "max_record_seconds": int(self.max_record_spin.get_value()),
            "record_device": self._selected_value(self.device_dropdown, self.device_options),
            "initial_prompt": self.initial_prompt_entry.get_text().strip(),
        }

    def _update_model_hint(self) -> None:
        model = self._selected_value(self.model_dropdown, MODEL_OPTIONS)
        hint = MODEL_HINTS.get(model, "")
        self.model_hint.set_label(
            f"Modellhinweis: {hint}\nHinweis: Ein Modellwechsel fuehrt beim naechsten Start einmalig zu einem Download, falls es noch nicht lokal im Cache liegt."
        )

    def _set_status(self, extra: str = "") -> None:
        running = daemon_running()
        state = "Daemon laeuft im Hintergrund." if running else "Daemon ist gerade gestoppt."
        active_model = self._selected_value(self.model_dropdown, MODEL_OPTIONS)
        suffix = f"\n{extra}" if extra else ""
        self.status_label.set_label(
            f"{state}\nAktuell ausgewaehltes Modell: {active_model}\nConfig: {CONFIG_FILE}{suffix}"
        )

    def _run_daemon_command(self, arg: str) -> tuple[int, str]:
        result = subprocess.run(
            [str(DAEMON_SCRIPT), arg],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode, output

    def _on_model_changed(self, *_args) -> None:
        self._update_model_hint()
        self._set_status()

    def _on_apply_clicked(self, _button: Gtk.Button) -> None:
        config = self._config_from_form()
        save_config(config)
        code, output = self._run_daemon_command("--restart")
        message = "Gespeichert und Daemon neu gestartet."
        if code != 0:
            message = f"Config gespeichert, aber Neustart schlug fehl: {output}"
        self._set_status(message)

    def _on_start_clicked(self, _button: Gtk.Button) -> None:
        code, output = self._run_daemon_command("--restart")
        message = "Daemon gestartet."
        if code != 0:
            message = f"Start fehlgeschlagen: {output}"
        self._set_status(message)

    def _on_stop_clicked(self, _button: Gtk.Button) -> None:
        code, output = self._run_daemon_command("--stop")
        message = "Daemon gestoppt."
        if code != 0:
            message = f"Stop meldete einen Fehler: {output}"
        self._set_status(message)

    def _on_log_clicked(self, _button: Gtk.Button) -> None:
        if LOG_FILE.exists():
            Gio.AppInfo.launch_default_for_uri(f"file://{LOG_FILE}", None)
            self._set_status("Logdatei geoeffnet.")
            return
        self._set_status("Noch keine Logdatei vorhanden.")


class SettingsApplication(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="local.whisper.dictation.settings")

    def do_activate(self) -> None:
        window = self.props.active_window
        if window is None:
            window = SettingsWindow(self)
        window.present()


def main() -> int:
    app = SettingsApplication()
    return app.run([])


if __name__ == "__main__":
    raise SystemExit(main())
