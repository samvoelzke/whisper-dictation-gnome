#!/usr/bin/env python3
"""macOS Settings GUI for Whisper Dictation — uses tkinter (stdlib, no GTK needed)."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path.home() / ".config" / "whisper-dictation"
CONFIG_FILE = CONFIG_DIR / "config.json"
LOG_FILE = Path.home() / ".cache" / "whisper-dictation" / "daemon.log"
DAEMON_SCRIPT = PROJECT_ROOT / "bin" / "whisper-dictation-mac.sh"

DEFAULT_CONFIG = {
    "double_tap_key": "ctrl_r",
    "double_tap_window_ms": 400,
    "language": "de",
    "model": "turbo",
    "paste_mode": "cmd_v",
    "record_device": "default",
    "max_record_seconds": 180,
    "initial_prompt": "",
}

MODEL_OPTIONS = [
    "turbo", "tiny", "base", "small", "medium", "large-v3",
    "tiny.en", "base.en", "small.en", "medium.en", "large-v2",
]

MODEL_HINTS = {
    "turbo": "Schnell und sehr stark. Beste Standardwahl.",
    "tiny": "Extrem schnell, geringste Genauigkeit.",
    "base": "Etwas genauer als tiny, sehr leicht.",
    "small": "Guter Mittelweg.",
    "medium": "Deutlich genauer, aber schwerer.",
    "large-v3": "Beste Qualität, braucht am meisten RAM/Zeit.",
    "tiny.en": "Nur Englisch, maximal leichtgewichtig.",
    "base.en": "Nur Englisch, kompakt.",
    "small.en": "Nur Englisch, guter Kompromiss.",
    "medium.en": "Nur Englisch, stark aber schwerer.",
    "large-v2": "Älteres großes Modell.",
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
    ("cmd_v", "Cmd+V (macOS Standard)"),
    ("auto", "Auto"),
    ("ctrl_v", "Ctrl+V"),
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
        capture_output=True, text=True, check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def ax_is_process_trusted() -> bool:
    try:
        import ctypes, ctypes.util
        lib = ctypes.cdll.LoadLibrary(
            ctypes.util.find_library("ApplicationServices") or
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
        lib.AXIsProcessTrusted.restype = ctypes.c_bool
        return bool(lib.AXIsProcessTrusted())
    except Exception:
        return True


def open_accessibility_settings() -> None:
    subprocess.run(
        ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"],
        check=False, capture_output=True,
    )


def run_daemon_command(arg: str) -> tuple[int, str]:
    result = subprocess.run(
        [str(DAEMON_SCRIPT), arg],
        capture_output=True, text=True, check=False,
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def detect_sounddevice_devices() -> list[tuple[str, str]]:
    devices: list[tuple[str, str]] = [("default", "default (Systemstandard)")]
    try:
        import sounddevice as sd  # type: ignore[import]
        for dev in sd.query_devices():
            if dev["max_input_channels"] > 0:  # type: ignore[index]
                idx = str(dev["index"])  # type: ignore[index]
                name = str(dev["name"])  # type: ignore[index]
                devices.append((idx, f"{idx}: {name}"))
    except Exception:
        pass
    return devices


class SettingsApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Whisper Dictation – Einstellungen")
        self.resizable(False, False)
        self.config_data = load_config()
        self.device_options = detect_sounddevice_devices()
        self._build_ui()
        self._refresh_status()

    def _build_ui(self) -> None:
        pad = {"padx": 12, "pady": 4}

        # Title
        title = tk.Label(self, text="Whisper Dictation", font=("SF Pro Display", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(14, 2), padx=14, sticky="w")

        subtitle = tk.Label(
            self,
            text="Lokale Spracherkennung  ·  macOS",
            font=("SF Pro Text", 11),
            fg="#666",
        )
        subtitle.grid(row=1, column=0, columnspan=2, padx=14, sticky="w")

        ttk.Separator(self, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=8, padx=14
        )

        # Accessibility warning banner (shown if permission missing)
        self.ax_frame = tk.Frame(self, bg="#fff3cd", padx=10, pady=8)
        self.ax_label = tk.Label(
            self.ax_frame,
            text="",
            bg="#fff3cd",
            fg="#856404",
            font=("SF Pro Text", 11),
            wraplength=520,
            justify="left",
        )
        self.ax_label.pack(side="left", fill="x", expand=True)
        self.ax_button = tk.Button(
            self.ax_frame,
            text="Systemeinstellungen öffnen",
            command=self._open_ax_settings,
            font=("SF Pro Text", 11),
        )
        self.ax_button.pack(side="right", padx=(8, 0))

        row = 3
        fields: list[tuple[str, tk.Widget]] = []

        # Model
        self.model_var = tk.StringVar(value=str(self.config_data["model"]))
        model_cb = ttk.Combobox(self, textvariable=self.model_var, values=MODEL_OPTIONS, state="readonly", width=22)
        model_cb.bind("<<ComboboxSelected>>", self._on_model_changed)
        fields.append(("Modell", model_cb))

        # Model hint
        self.model_hint_var = tk.StringVar()
        self.model_hint_label = tk.Label(
            self, textvariable=self.model_hint_var, fg="#555", font=("SF Pro Text", 10), wraplength=380, justify="left"
        )

        # Language
        self.lang_var = tk.StringVar(value=str(self.config_data["language"]))
        lang_entry = tk.Entry(self, textvariable=self.lang_var, width=24)
        fields.append(("Sprache (de, en, …)", lang_entry))

        # Hotkey
        hotkey_labels = [label for _, label in HOTKEY_OPTIONS]
        hotkey_values = [v for v, _ in HOTKEY_OPTIONS]
        current_hk = str(self.config_data["double_tap_key"])
        hk_idx = hotkey_values.index(current_hk) if current_hk in hotkey_values else 0
        self.hotkey_var = tk.StringVar(value=hotkey_labels[hk_idx])
        self._hotkey_map = dict(zip(hotkey_labels, hotkey_values))
        hk_cb = ttk.Combobox(self, textvariable=self.hotkey_var, values=hotkey_labels, state="readonly", width=22)
        fields.append(("Doppeltaste", hk_cb))

        # Double-tap window
        self.double_tap_var = tk.IntVar(value=int(self.config_data["double_tap_window_ms"]))
        dt_spin = tk.Spinbox(self, from_=150, to=1200, increment=10, textvariable=self.double_tap_var, width=23)
        fields.append(("Double-Tap Fenster (ms)", dt_spin))

        # Paste mode
        paste_labels = [label for _, label in PASTE_OPTIONS]
        paste_values = [v for v, _ in PASTE_OPTIONS]
        current_paste = str(self.config_data["paste_mode"])
        paste_idx = paste_values.index(current_paste) if current_paste in paste_values else 0
        self.paste_var = tk.StringVar(value=paste_labels[paste_idx])
        self._paste_map = dict(zip(paste_labels, paste_values))
        paste_cb = ttk.Combobox(self, textvariable=self.paste_var, values=paste_labels, state="readonly", width=22)
        fields.append(("Paste-Modus", paste_cb))

        # Max record seconds
        self.max_rec_var = tk.IntVar(value=int(self.config_data["max_record_seconds"]))
        max_spin = tk.Spinbox(self, from_=15, to=900, increment=5, textvariable=self.max_rec_var, width=23)
        fields.append(("Max. Aufnahme (s)", max_spin))

        # Microphone
        dev_labels = [label for _, label in self.device_options]
        dev_values = [v for v, _ in self.device_options]
        current_dev = str(self.config_data["record_device"])
        dev_idx = dev_values.index(current_dev) if current_dev in dev_values else 0
        self.device_var = tk.StringVar(value=dev_labels[dev_idx])
        self._device_map = dict(zip(dev_labels, dev_values))
        dev_cb = ttk.Combobox(self, textvariable=self.device_var, values=dev_labels, state="readonly", width=22)
        fields.append(("Mikrofon", dev_cb))

        # Initial prompt
        self.prompt_var = tk.StringVar(value=str(self.config_data["initial_prompt"]))
        prompt_entry = tk.Entry(self, textvariable=self.prompt_var, width=24)
        fields.append(("Initial Prompt", prompt_entry))

        for i, (label_text, widget) in enumerate(fields):
            lbl = tk.Label(self, text=label_text, anchor="w")
            lbl.grid(row=row + i, column=0, sticky="w", **pad)
            widget.grid(row=row + i, column=1, sticky="ew", **pad)

        # Model hint below model row
        self.model_hint_label.grid(row=row, column=1, sticky="w", padx=12, pady=(0, 4))
        self._update_model_hint()

        last_row = row + len(fields)

        ttk.Separator(self, orient="horizontal").grid(
            row=last_row, column=0, columnspan=2, sticky="ew", pady=8, padx=14
        )

        # Status label
        self.status_var = tk.StringVar()
        status_lbl = tk.Label(
            self, textvariable=self.status_var, wraplength=520, justify="left",
            font=("SF Pro Text", 11), fg="#333",
        )
        status_lbl.grid(row=last_row + 1, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 6))

        # Buttons
        btn_frame = tk.Frame(self)
        btn_frame.grid(row=last_row + 2, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 14))

        tk.Button(
            btn_frame, text="Speichern & Neustart", command=self._on_apply,
            font=("SF Pro Text", 12, "bold"), bg="#007AFF", fg="white", padx=8, pady=4,
        ).pack(side="left", padx=(0, 6))

        tk.Button(btn_frame, text="Starten", command=self._on_start, padx=8, pady=4).pack(side="left", padx=(0, 4))
        tk.Button(btn_frame, text="Stoppen", command=self._on_stop, padx=8, pady=4).pack(side="left", padx=(0, 4))
        tk.Button(btn_frame, text="Log öffnen", command=self._on_log, padx=8, pady=4).pack(side="left", padx=(0, 4))
        tk.Button(btn_frame, text="Accessibility prüfen", command=self._check_ax, padx=8, pady=4).pack(side="right")

    def _update_model_hint(self) -> None:
        model = self.model_var.get()
        self.model_hint_var.set(MODEL_HINTS.get(model, ""))

    def _on_model_changed(self, _event=None) -> None:
        self._update_model_hint()

    def _config_from_form(self) -> dict:
        return {
            "model": self.model_var.get(),
            "language": self.lang_var.get().strip(),
            "double_tap_key": self._hotkey_map.get(self.hotkey_var.get(), "ctrl_r"),
            "double_tap_window_ms": self.double_tap_var.get(),
            "paste_mode": self._paste_map.get(self.paste_var.get(), "cmd_v"),
            "max_record_seconds": self.max_rec_var.get(),
            "record_device": self._device_map.get(self.device_var.get(), "default"),
            "initial_prompt": self.prompt_var.get().strip(),
        }

    def _refresh_status(self) -> None:
        running = daemon_running()
        state = "Daemon läuft." if running else "Daemon gestoppt."
        self.status_var.set(f"{state}  |  Config: {CONFIG_FILE}")

        # Accessibility banner
        if not ax_is_process_trusted():
            real = __import__("os").path.realpath(sys.executable)
            self.ax_label.config(
                text=(
                    "Accessibility-Berechtigung fehlt — pynput erkennt keine Tasten.\n"
                    f"Systemeinstellungen öffnen → + → diesen Pfad einfügen:\n{real}"
                )
            )
            self.ax_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 4))
        else:
            self.ax_frame.grid_remove()

    def _open_ax_settings(self) -> None:
        open_accessibility_settings()

    def _check_ax(self) -> None:
        trusted = ax_is_process_trusted()
        self.status_var.set(
            "Accessibility: OK — Tasten werden erkannt." if trusted
            else "Accessibility fehlt — Systemeinstellungen öffnen und Python eintragen."
        )
        self._refresh_status()

    def _on_apply(self) -> None:
        save_config(self._config_from_form())
        code, out = run_daemon_command("--restart")
        msg = "Gespeichert und Daemon neu gestartet." if code == 0 else f"Gespeichert, Neustart fehlgeschlagen: {out}"
        self.status_var.set(msg)
        self.after(2000, self._refresh_status)

    def _on_start(self) -> None:
        code, out = run_daemon_command("--start")
        self.status_var.set("Daemon gestartet." if code == 0 else f"Start fehlgeschlagen: {out}")
        self.after(2000, self._refresh_status)

    def _on_stop(self) -> None:
        run_daemon_command("--stop")
        self.status_var.set("Daemon gestoppt.")
        self.after(1000, self._refresh_status)

    def _on_log(self) -> None:
        if LOG_FILE.exists():
            subprocess.run(["open", str(LOG_FILE)], check=False)
        else:
            self.status_var.set("Noch keine Logdatei vorhanden.")


def main() -> int:
    if platform.system() != "Darwin":
        print("Diese GUI ist nur für macOS. Auf Linux: gui/settings.py", file=sys.stderr)
        return 1
    app = SettingsApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
