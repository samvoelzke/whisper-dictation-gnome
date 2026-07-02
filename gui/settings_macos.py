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
    "ollama_postprocess": False,
    "ollama_model": "llama3.2:3b",
    "ollama_thinking": False,
    "ollama_host": "http://localhost:11434",
    "ollama_system_prompt": "",
}

WHISPER_CATALOG = [
    ("turbo",    "★★★★☆", "800 MB", "Empfohlen · Schnell und sehr stark"),
    ("small",    "★★★☆☆", "465 MB", "Guter Mittelweg · Sparsam mit RAM"),
    ("medium",   "★★★★☆", "1.5 GB", "Deutlich genauer · Braucht mehr RAM"),
    ("large-v3", "★★★★★", "3.0 GB", "Beste Qualität · Braucht am meisten RAM"),
    ("base",     "★★☆☆☆", "145 MB", "Kompakt · Einfache Texte"),
    ("tiny",     "★☆☆☆☆", "75 MB",  "Minimal · Geringste Genauigkeit"),
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

        self.geometry("640x780")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # ── Scrollable canvas setup ────────────────────────────────────────
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.grid(row=0, column=0, sticky="nsew")

        p = tk.Frame(canvas)
        p.columnconfigure(1, weight=1)
        win_id = canvas.create_window((0, 0), window=p, anchor="nw")

        def _on_frame_configure(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfig(win_id, width=event.width)

        p.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            # macOS: delta ist bereits in scroll-units, nicht ×120 wie Windows
            delta = event.delta
            if delta == 0:
                return
            canvas.yview_scroll(-1 if delta > 0 else 1, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── Header ────────────────────────────────────────────────────────
        tk.Label(p, text="Whisper Dictation", font=("SF Pro Display", 16, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(14, 2), padx=14, sticky="w"
        )
        tk.Label(p, text="Lokale Spracherkennung  ·  macOS", font=("SF Pro Text", 11), fg="#666").grid(
            row=1, column=0, columnspan=2, padx=14, sticky="w"
        )
        ttk.Separator(p, orient="horizontal").grid(row=2, column=0, columnspan=2, sticky="ew", pady=8, padx=14)

        # Accessibility warning banner
        self.ax_frame = tk.Frame(p, bg="#fff3cd", padx=10, pady=8)
        self.ax_label = tk.Label(
            self.ax_frame, text="", bg="#fff3cd", fg="#856404",
            font=("SF Pro Text", 11), wraplength=520, justify="left",
        )
        self.ax_label.pack(side="left", fill="x", expand=True)
        tk.Button(
            self.ax_frame, text="Systemeinstellungen öffnen",
            command=self._open_ax_settings, font=("SF Pro Text", 11),
        ).pack(side="right", padx=(8, 0))

        row = 3

        # ── Whisper Modell-Galerie ─────────────────────────────────────────
        tk.Label(p, text="🎙  Whisper Modell wählen",
                 font=("SF Pro Display", 13, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 4)
        )
        row += 1

        self.model_var = tk.StringVar(value=str(self.config_data["model"]))
        for model_id, stars, size, desc in WHISPER_CATALOG:
            is_selected = self.model_var.get() == model_id
            bg = "#f0f7ff" if is_selected else "white"
            f = tk.Frame(p, bg=bg, highlightbackground="#ddd", highlightthickness=1)
            f.grid(row=row, column=0, columnspan=2, sticky="ew", padx=14, pady=2, ipadx=6, ipady=4)
            rb = tk.Radiobutton(f, variable=self.model_var, value=model_id,
                                text=model_id, font=("SF Pro Text", 12, "bold"),
                                bg=bg, activebackground=bg,
                                command=lambda: self._on_model_changed())
            rb.pack(side="left")
            tk.Label(f, text=stars, font=("SF Pro Text", 11), fg="#FF9500", bg=bg).pack(side="left", padx=(6, 2))
            tk.Label(f, text=f"{size}  ·  {desc}", font=("SF Pro Text", 10), fg="#555", bg=bg).pack(side="left", padx=(2, 0))
            row += 1

        ttk.Separator(p, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8, padx=14)
        row += 1

        fields: list[tuple[str, tk.Widget]] = []

        # Language
        LANG_OPTIONS = ["auto", "de", "en", "fr", "es", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko", "ar", "tr"]
        current_lang = str(self.config_data["language"])
        if current_lang not in LANG_OPTIONS:
            LANG_OPTIONS.append(current_lang)
        self.lang_var = tk.StringVar(value=current_lang)
        lang_cb = ttk.Combobox(p, textvariable=self.lang_var, values=LANG_OPTIONS, width=22)
        fields.append(("Sprache (de, en, …)", lang_cb))

        # Hotkey
        hotkey_labels = [label for _, label in HOTKEY_OPTIONS]
        hotkey_values = [v for v, _ in HOTKEY_OPTIONS]
        current_hk = str(self.config_data["double_tap_key"])
        hk_idx = hotkey_values.index(current_hk) if current_hk in hotkey_values else 0
        self.hotkey_var = tk.StringVar(value=hotkey_labels[hk_idx])
        self._hotkey_map = dict(zip(hotkey_labels, hotkey_values))
        hk_cb = ttk.Combobox(p, textvariable=self.hotkey_var, values=hotkey_labels, state="readonly", width=22)
        fields.append(("Doppeltaste", hk_cb))

        # Double-tap window
        self.double_tap_var = tk.IntVar(value=int(self.config_data["double_tap_window_ms"]))
        dt_spin = tk.Spinbox(p, from_=150, to=1200, increment=10, textvariable=self.double_tap_var, width=23)
        fields.append(("Double-Tap Fenster (ms)", dt_spin))

        # Paste mode
        paste_labels = [label for _, label in PASTE_OPTIONS]
        paste_values = [v for v, _ in PASTE_OPTIONS]
        current_paste = str(self.config_data["paste_mode"])
        paste_idx = paste_values.index(current_paste) if current_paste in paste_values else 0
        self.paste_var = tk.StringVar(value=paste_labels[paste_idx])
        self._paste_map = dict(zip(paste_labels, paste_values))
        paste_cb = ttk.Combobox(p, textvariable=self.paste_var, values=paste_labels, state="readonly", width=22)
        fields.append(("Paste-Modus", paste_cb))

        # Max record seconds
        self.max_rec_var = tk.IntVar(value=int(self.config_data["max_record_seconds"]))
        max_spin = tk.Spinbox(p, from_=15, to=900, increment=5, textvariable=self.max_rec_var, width=23)
        fields.append(("Max. Aufnahme (s)", max_spin))

        # Microphone
        dev_labels = [label for _, label in self.device_options]
        dev_values = [v for v, _ in self.device_options]
        current_dev = str(self.config_data["record_device"])
        dev_idx = dev_values.index(current_dev) if current_dev in dev_values else 0
        self.device_var = tk.StringVar(value=dev_labels[dev_idx])
        self._device_map = dict(zip(dev_labels, dev_values))
        dev_cb = ttk.Combobox(p, textvariable=self.device_var, values=dev_labels, state="readonly", width=22)
        fields.append(("Mikrofon", dev_cb))

        # Initial prompt
        self.prompt_var = tk.StringVar(value=str(self.config_data["initial_prompt"]))
        prompt_entry = tk.Entry(p, textvariable=self.prompt_var, width=24)
        fields.append(("Initial Prompt", prompt_entry))

        # ── Ollama ─────────────────────────────────────────────────────────
        self.ollama_var = tk.BooleanVar(value=bool(self.config_data.get("ollama_postprocess", False)))
        ollama_check = tk.Checkbutton(p, variable=self.ollama_var, text="Aktiv")
        fields.append(("Ollama Text-Cleanup", ollama_check))

        self.ollama_thinking_var = tk.BooleanVar(value=bool(self.config_data.get("ollama_thinking", False)))
        ollama_thinking_check = tk.Checkbutton(p, variable=self.ollama_thinking_var, text="Aktiv (langsamer, bessere Qualität)")
        fields.append(("Ollama Thinking", ollama_thinking_check))

        self.ollama_host_var = tk.StringVar(value=str(self.config_data.get("ollama_host", "http://localhost:11434")))
        ollama_host_entry = tk.Entry(p, textvariable=self.ollama_host_var, width=24)
        fields.append(("Ollama Host", ollama_host_entry))

        _DEFAULT_PROMPT = (
            "Du bist ein Diktat-Korrektor. Korrigiere nur Grammatik,\n"
            "Satzzeichen und Groß-/Kleinschreibung.\n"
            "Antworte nie auf den Inhalt. Gib nur den\n"
            "korrigierten Text aus."
        )
        saved_prompt = str(self.config_data.get("ollama_system_prompt", "")).strip()
        self._prompt_text = tk.Text(p, width=28, height=4, font=("SF Pro Text", 11),
                                    wrap="word", relief="solid", borderwidth=1)
        if saved_prompt:
            self._prompt_text.insert("1.0", saved_prompt)
            self._prompt_text.config(fg="black")
        else:
            self._prompt_text.insert("1.0", _DEFAULT_PROMPT)
            self._prompt_text.config(fg="#aaa")

        def _prompt_focus_in(_e):
            if self._prompt_text.cget("fg") == "#aaa":
                self._prompt_text.delete("1.0", "end")
                self._prompt_text.config(fg="black")

        def _prompt_focus_out(_e):
            if not self._prompt_text.get("1.0", "end").strip():
                self._prompt_text.insert("1.0", _DEFAULT_PROMPT)
                self._prompt_text.config(fg="#aaa")

        self._prompt_text.bind("<FocusIn>", _prompt_focus_in)
        self._prompt_text.bind("<FocusOut>", _prompt_focus_out)
        fields.append(("Ollama System Prompt\n(leer = Standard)", self._prompt_text))

        for i, (label_text, widget) in enumerate(fields):
            tk.Label(p, text=label_text, anchor="w").grid(row=row + i, column=0, sticky="w", **pad)
            widget.grid(row=row + i, column=1, sticky="ew", **pad)

        last_row = row + len(fields)

        # ── Ollama Modell-Auswahl (Galerie) ────────────────────────────────
        ttk.Separator(p, orient="horizontal").grid(row=last_row, column=0, columnspan=2, sticky="ew", pady=8, padx=14)
        last_row += 1

        tk.Label(p, text="🤖  Ollama Modell wählen & installieren",
                 font=("SF Pro Display", 13, "bold")).grid(
            row=last_row, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 6)
        )
        last_row += 1

        self._ollama_model_var = tk.StringVar(value=str(self.config_data.get("ollama_model", "llama3.2:3b")))
        self._model_row_widgets: dict = {}
        installed = set(self._get_ollama_models())

        OLLAMA_CATALOG = [
            ("llama3.2:3b",  "★★★★☆", "2 GB",   "Empfohlen · Sehr gut auf Deutsch & Englisch"),
            ("llama3.2:1b",  "★★★☆☆", "0.8 GB", "Sehr schnell · Einfache Texte"),
            ("gemma3:1b",    "★★☆☆☆", "0.8 GB", "Mini · Maximal schnell"),
            ("phi3:mini",    "★★★☆☆", "2.3 GB", "Kompakt · Etwas schwächer"),
            ("mistral:7b",   "★★★★★", "4.1 GB", "Profi · Beste Qualität, braucht mehr RAM"),
        ]

        for model_id, stars, size, desc in OLLAMA_CATALOG:
            is_installed = model_id in installed
            bg = "#f9f9f9" if is_installed else "white"
            f = tk.Frame(p, bg=bg, highlightbackground="#ddd", highlightthickness=1)
            f.grid(row=last_row, column=0, columnspan=2, sticky="ew", padx=14, pady=2, ipadx=6, ipady=4)

            rb = tk.Radiobutton(f, variable=self._ollama_model_var, value=model_id,
                                text=model_id, font=("SF Pro Text", 12, "bold"),
                                bg=bg, activebackground=bg,
                                state="normal" if is_installed else "disabled")
            rb.pack(side="left")
            tk.Label(f, text=stars, font=("SF Pro Text", 11), fg="#FF9500", bg=bg).pack(side="left", padx=(6, 2))
            tk.Label(f, text=f"{size}  ·  {desc}", font=("SF Pro Text", 10), fg="#555", bg=bg).pack(side="left", padx=(2, 0))

            if is_installed:
                tk.Label(f, text="✓ installiert", font=("SF Pro Text", 10, "bold"),
                         fg="#34C759", bg=bg).pack(side="right", padx=8)
            else:
                btn = tk.Button(f, text="⬇ Installieren",
                                font=("SF Pro Text", 10), bg="#007AFF", fg="white",
                                activebackground="#005ecb", relief="flat", padx=6, pady=2,
                                command=lambda m=model_id, fr=f, rb=rb: self._install_ollama_model(m, fr, rb))
                btn.pack(side="right", padx=8)
                self._model_row_widgets[model_id] = (f, rb, btn)

            last_row += 1

        self._ollama_dl_label = tk.Label(p, text="", font=("SF Pro Text", 11), fg="#FF9500")
        self._ollama_dl_label.grid(row=last_row, column=0, columnspan=2, sticky="w", padx=14)
        self._ollama_dl_bar = ttk.Progressbar(p, mode="determinate", maximum=100, length=500)
        self._ollama_dl_bar.grid(row=last_row + 1, column=0, columnspan=2, sticky="ew", padx=14, pady=(2, 6))
        last_row += 2

        ttk.Separator(p, orient="horizontal").grid(row=last_row, column=0, columnspan=2, sticky="ew", pady=8, padx=14)

        # Status label
        self.status_var = tk.StringVar()
        tk.Label(p, textvariable=self.status_var, wraplength=520, justify="left",
                 font=("SF Pro Text", 11), fg="#333").grid(
            row=last_row + 1, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 6)
        )

        # Buttons
        btn_frame = tk.Frame(p)
        btn_frame.grid(row=last_row + 2, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 14))

        tk.Button(
            btn_frame, text="Speichern & Neustart", command=self._on_apply,
            font=("SF Pro Text", 12, "bold"), bg="#007AFF", fg="white", padx=8, pady=4,
        ).pack(side="left", padx=(0, 6))
        tk.Button(btn_frame, text="Starten", command=self._on_start, padx=8, pady=4).pack(side="left", padx=(0, 4))
        tk.Button(btn_frame, text="Stoppen", command=self._on_stop, padx=8, pady=4).pack(side="left", padx=(0, 4))
        tk.Button(btn_frame, text="Log öffnen", command=self._on_log, padx=8, pady=4).pack(side="left", padx=(0, 4))
        tk.Button(btn_frame, text="Accessibility prüfen", command=self._check_ax, padx=8, pady=4).pack(side="right")

    def _install_ollama_model(self, model: str, frame: tk.Frame, rb: tk.Radiobutton) -> None:
        import threading, json as _json
        self._ollama_dl_label.config(text=f"Lade {model} herunter… 0%", fg="#FF9500")
        self._ollama_dl_bar.config(value=0)

        def run():
            proc = subprocess.Popen(
                ["ollama", "pull", model],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = _json.loads(line)
                    t = data.get("total", 0) or 0
                    c = data.get("completed", 0) or 0
                    if t > 0:
                        pct = min(int(c / t * 100), 100)
                        mb_done = c / (1024 ** 2)
                        mb_total = t / (1024 ** 2)
                        lbl = f"Lade {model}… {pct}%  ({mb_done:.0f} / {mb_total:.0f} MB)"
                        self.after(0, lambda l=lbl, p=pct: (
                            self._ollama_dl_label.config(text=l),
                            self._ollama_dl_bar.config(value=p),
                        ))
                except Exception:
                    pass
            proc.wait()
            self.after(0, lambda: self._model_install_done(model, frame, rb))

        threading.Thread(target=run, daemon=True).start()

    def _model_install_done(self, model: str, frame: tk.Frame, rb: tk.Radiobutton) -> None:
        installed = set(self._get_ollama_models())
        if model in installed:
            self._ollama_dl_label.config(text=f"✓ {model} erfolgreich installiert!", fg="#34C759")
            self._ollama_dl_bar.config(value=100)
            # Zeile aktualisieren: Button entfernen, "✓ installiert" zeigen, Radiobutton aktivieren
            frame.config(bg="#f9f9f9")
            for w in frame.winfo_children():
                w.config(bg="#f9f9f9") if hasattr(w, 'config') else None
            rb.config(state="normal", bg="#f9f9f9", activebackground="#f9f9f9")
            if model in self._model_row_widgets:
                _, _, btn = self._model_row_widgets[model]
                btn.destroy()
                tk.Label(frame, text="✓ installiert", font=("SF Pro Text", 10, "bold"),
                         fg="#34C759", bg="#f9f9f9").pack(side="right", padx=8)
            self._ollama_model_var.set(model)
            # Dropdown aktualisieren
            new_models = self._get_ollama_models()
            self._ollama_model_var.set(model)
        else:
            self._ollama_dl_label.config(text=f"✗ Download fehlgeschlagen. Erneut versuchen.", fg="#FF3B30")

    def _get_ollama_models(self) -> list[str]:
        try:
            r = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
            models = []
            for line in r.stdout.splitlines()[1:]:  # erste Zeile = Header
                parts = line.split()
                if parts:
                    models.append(parts[0])
            return models if models else ["llama3.2:3b"]
        except Exception:
            return ["llama3.2:3b"]

    def _on_model_changed(self) -> None:
        pass

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
            "ollama_postprocess": self.ollama_var.get(),
            "ollama_model": self._ollama_model_var.get().strip(),
            "ollama_thinking": self.ollama_thinking_var.get(),
            "ollama_host": self.ollama_host_var.get().strip(),
            "ollama_system_prompt": (
                "" if self._prompt_text.cget("fg") == "#aaa"
                else self._prompt_text.get("1.0", "end").strip()
            ),
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
