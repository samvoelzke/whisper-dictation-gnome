#!/usr/bin/env python3
"""Ollama Setup Wizard – führt den Nutzer durch Installation und Modell-Download."""

from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

MODELS = [
    ("llama3.2:3b",  "llama3.2:3b",  "Empfohlen  •  2 GB  •  Sehr gute Qualität auf Deutsch & Englisch"),
    ("phi3:mini",    "phi3:mini",    "Kompakt    •  2.3 GB  •  Schnell, etwas schwächer"),
    ("gemma3:1b",    "gemma3:1b",    "Mini       •  0.8 GB  •  Sehr schnell, für einfache Texte"),
    ("mistral:7b",   "mistral:7b",   "Profi      •  4.1 GB  •  Beste Qualität, braucht mehr RAM"),
]


def ollama_installed() -> bool:
    return subprocess.run(["which", "ollama"], capture_output=True).returncode == 0


def ollama_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


MODEL_SIZES_GB = {
    "llama3.2:3b": 2.0,
    "llama3.2:1b": 0.8,
    "phi3:mini":   2.3,
    "gemma3:1b":   0.8,
    "mistral:7b":  4.1,
}


def model_downloaded(model: str) -> bool:
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return model.split(":")[0] in r.stdout
    except Exception:
        return False


def free_disk_gb() -> float:
    import shutil
    return shutil.disk_usage(Path.home()).free / (1024 ** 3)


class OllamaWizard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ollama einrichten – Whisper Dictation")
        self.resizable(False, False)
        self.configure(bg="#f5f5f7")
        self._selected_model = tk.StringVar(value="llama3.2:3b")
        self._build()
        self._check_state()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg="#1d1d1f", padx=24, pady=18)
        hdr.grid(row=0, column=0, sticky="ew")
        tk.Label(hdr, text="🤖  Ollama Text-Cleanup einrichten",
                 font=("SF Pro Display", 16, "bold"), fg="white", bg="#1d1d1f").pack(anchor="w")
        tk.Label(hdr,
                 text="Nach dem Diktat verbessert ein lokales KI-Modell Satzzeichen und Grammatik automatisch.",
                 font=("SF Pro Text", 11), fg="#aaa", bg="#1d1d1f", wraplength=500, justify="left").pack(anchor="w", pady=(4,0))

        body = tk.Frame(self, bg="#f5f5f7", padx=24, pady=20)
        body.grid(row=1, column=0, sticky="nsew")

        # ── Schritt 1: Ollama installieren ──────────────────────────────────
        self._step1 = self._card(body, "1", "Ollama installieren", 0)

        self.step1_status = tk.Label(self._step1, text="Wird geprüft...",
                                     font=("SF Pro Text", 11), fg="#666", bg="white")
        self.step1_status.pack(anchor="w", pady=(0, 8))

        self.btn_install = tk.Button(
            self._step1, text="Ollama automatisch installieren",
            command=self._install_ollama,
            font=("SF Pro Text", 12, "bold"), bg="#007AFF", fg="white",
            activebackground="#005ecb", activeforeground="white",
            disabledforeground="#888", padx=14, pady=8, relief="flat", cursor="hand2",
        )
        self.btn_install.pack(anchor="w")

        # ── Schritt 2: Modell wählen ─────────────────────────────────────────
        self._step2 = self._card(body, "2", "KI-Modell wählen", 1)

        free_gb = free_disk_gb()
        disk_color = "#FF3B30" if free_gb < 5 else "#FF9500" if free_gb < 10 else "#34C759"
        self.disk_label = tk.Label(
            self._step2,
            text=f"💾 Freier Speicher: {free_gb:.1f} GB",
            font=("SF Pro Text", 11, "bold"), fg=disk_color, bg="white"
        )
        self.disk_label.pack(anchor="w", pady=(0, 8))

        for model_id, _, desc in MODELS:
            needed = MODEL_SIZES_GB.get(model_id, 2.0)
            already = model_downloaded(model_id)
            enough = already or free_gb > needed + 1
            row = tk.Frame(self._step2, bg="white")
            row.pack(fill="x", pady=3)
            rb = tk.Radiobutton(row, variable=self._selected_model, value=model_id,
                                font=("SF Pro Text", 12, "bold"), bg="white",
                                text=model_id, activebackground="white",
                                state="normal" if enough else "disabled")
            rb.pack(side="left")
            if already:
                tk.Label(row, text="✓ bereits heruntergeladen",
                         font=("SF Pro Text", 10), fg="#34C759", bg="white").pack(side="right", padx=8)
            elif not enough:
                tk.Label(row, text="⚠ Nicht genug Speicher",
                         font=("SF Pro Text", 10), fg="#FF3B30", bg="white").pack(side="right", padx=8)
            tk.Label(row, text=desc, font=("SF Pro Text", 10),
                     fg="#555" if enough else "#aaa", bg="white").pack(side="left", padx=(4,0))

        # ── Schritt 3: Herunterladen ─────────────────────────────────────────
        self._step3 = self._card(body, "3", "Modell herunterladen", 2)

        self.progress_label = tk.Label(self._step3, text="Bereit zum Herunterladen.",
                                       font=("SF Pro Text", 11), fg="#333", bg="white")
        self.progress_label.pack(anchor="w", pady=(0,6))

        self.progress_bar = ttk.Progressbar(self._step3, mode="determinate", maximum=100, length=460)
        self.progress_bar.pack(fill="x", pady=(0,8))

        self.btn_download = tk.Button(
            self._step3, text="⬇  Modell herunterladen",
            command=self._download_model,
            font=("SF Pro Text", 12, "bold"), bg="#34C759", fg="white",
            activebackground="#248a3d", activeforeground="white",
            disabledforeground="#888", padx=14, pady=8, relief="flat", cursor="hand2",
        )
        self.btn_download.pack(anchor="w")

        # ── Schritt 4: Aktivieren ────────────────────────────────────────────
        self._step4 = self._card(body, "4", "Text-Cleanup aktivieren", 3)

        tk.Label(self._step4,
                 text=(
                     "Was macht Text-Cleanup?\n"
                     "Whisper schreibt gesprochenen Text direkt um — ohne Satzzeichen, manchmal mit Fehlern.\n"
                     "Das KI-Modell liest den Text danach nochmal durch und fügt automatisch\n"
                     "Punkte, Kommas, Großschreibung und Korrekturen ein.\n\n"
                     "Beispiel vorher:  \"ich gehe morgen zum arzt er soll sich das mal anschauen\"\n"
                     "Beispiel nachher: \"Ich gehe morgen zum Arzt. Er soll sich das mal anschauen.\""
                 ),
                 font=("SF Pro Text", 11), fg="#333", bg="white", justify="left").pack(anchor="w", pady=(0,10))

        self.btn_activate = tk.Button(
            self._step4, text="✓  Text-Cleanup aktivieren & fertig",
            command=self._activate,
            font=("SF Pro Text", 13, "bold"), bg="#aaa", fg="white",
            activebackground="#248a3d", activeforeground="white",
            disabledforeground="#ccc", padx=16, pady=10, relief="flat",
            state="disabled",
        )
        self.btn_activate.pack(anchor="w")

        self.done_label = tk.Label(self._step4, text="", font=("SF Pro Text", 12, "bold"),
                                   fg="#34C759", bg="white")
        self.done_label.pack(anchor="w", pady=(8,0))

    def _card(self, parent, number, title, row):
        frame = tk.Frame(parent, bg="white", padx=16, pady=14,
                         highlightbackground="#ddd", highlightthickness=1)
        frame.grid(row=row, column=0, sticky="ew", pady=(0,12), ipadx=4)
        parent.columnconfigure(0, weight=1)
        hdr = tk.Frame(frame, bg="white")
        hdr.pack(fill="x", pady=(0,8))
        tk.Label(hdr, text=number, font=("SF Pro Text", 12, "bold"),
                 bg="#007AFF", fg="white", width=2, pady=2).pack(side="left", padx=(0,10))
        tk.Label(hdr, text=title, font=("SF Pro Display", 13, "bold"),
                 bg="white").pack(side="left")
        return frame

    def _check_state(self):
        installed = ollama_installed()
        running = ollama_running()
        # Bereits heruntergeladenes Modell direkt anzeigen (nur wenn Ollama läuft)
        model = self._selected_model.get()
        if running and model_downloaded(model):
            self.progress_label.config(text=f"✓ {model} ist bereits heruntergeladen!", fg="#34C759")
            self.btn_download.config(text="✓ Heruntergeladen", state="disabled", bg="#aaa")
            self.btn_activate.config(state="normal", bg="#34C759")

        if installed:
            self.step1_status.config(
                text="✓ Ollama ist installiert." + (" Läuft." if running else " Nicht gestartet."),
                fg="#34C759"
            )
            self.btn_install.config(
                text="✓ Ollama bereits installiert",
                state="disabled", bg="#aaa"
            )
            if not running:
                self.btn_install.config(
                    text="▶  Ollama starten",
                    state="normal", bg="#007AFF",
                    command=self._start_ollama
                )
        else:
            self.step1_status.config(text="Ollama ist noch nicht installiert.", fg="#FF3B30")

    def _install_ollama(self):
        self.btn_install.config(state="disabled", text="Installiere...", bg="#aaa")
        self.step1_status.config(text="Homebrew installiert Ollama...", fg="#FF9500")

        def run():
            result = subprocess.run(
                ["brew", "install", "ollama"],
                capture_output=True, text=True
            )
            self.after(0, lambda: self._install_done(result.returncode == 0))

        threading.Thread(target=run, daemon=True).start()

    def _install_done(self, success):
        if success:
            self.step1_status.config(text="✓ Ollama erfolgreich installiert!", fg="#34C759")
            self._start_ollama()
        else:
            self.step1_status.config(text="✗ Installation fehlgeschlagen. Bitte Homebrew installieren.", fg="#FF3B30")
            self.btn_install.config(state="normal", text="Erneut versuchen", bg="#007AFF")

    def _start_ollama(self):
        subprocess.Popen(["ollama", "serve"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        self.after(2000, self._check_state)

    def _download_model(self):
        model = self._selected_model.get()
        if model_downloaded(model):
            self.progress_label.config(text=f"✓ {model} ist bereits heruntergeladen!", fg="#34C759")
            return

        if not ollama_running():
            self.progress_label.config(text="Starte Ollama zuerst...", fg="#FF9500")
            self._start_ollama()
            self.after(3000, self._download_model)
            return

        self.btn_download.config(state="disabled", bg="#aaa")
        self.progress_bar.config(mode="determinate", maximum=100, value=0)
        self.progress_label.config(
            text=f"Lade {model} herunter… 0%",
            fg="#FF9500"
        )

        def run():
            import json as _json
            proc = subprocess.Popen(
                ["ollama", "pull", model],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            total = 0
            completed = 0
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = _json.loads(line)
                    t = data.get("total", 0) or 0
                    c = data.get("completed", 0) or 0
                    status = data.get("status", "")
                    if t > 0:
                        total, completed = t, c
                        pct = min(int(c / t * 100), 100)
                        mb_done = c / (1024 ** 2)
                        mb_total = t / (1024 ** 2)
                        label = f"Lade {model}… {pct}%  ({mb_done:.0f} / {mb_total:.0f} MB)"
                        self.after(0, lambda l=label, p=pct: self._update_progress(l, p))
                    elif status:
                        self.after(0, lambda s=status: self.progress_label.config(text=s, fg="#FF9500"))
                except Exception:
                    pass
            proc.wait()
            self.after(0, lambda: self._download_done(model))

        threading.Thread(target=run, daemon=True).start()

    def _update_progress(self, label: str, pct: int):
        self.progress_label.config(text=label, fg="#FF9500")
        self.progress_bar.config(value=pct)

    def _download_done(self, model):
        if model_downloaded(model):
            self.progress_label.config(text=f"✓ {model} erfolgreich heruntergeladen!", fg="#34C759")
            self.btn_download.config(text="✓ Heruntergeladen", state="disabled", bg="#aaa")
            self.btn_activate.config(state="normal", bg="#34C759")
        else:
            self.progress_label.config(text="✗ Download fehlgeschlagen. Bitte erneut versuchen.", fg="#FF3B30")
            self.btn_download.config(state="normal", bg="#34C759")

    def _activate(self):
        model = self._selected_model.get()
        config_file = Path.home() / ".config" / "whisper-dictation" / "config.json"
        import json
        cfg = json.loads(config_file.read_text()) if config_file.exists() else {}
        cfg["ollama_postprocess"] = True
        cfg["ollama_model"] = model
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(json.dumps(cfg, indent=2, ensure_ascii=True) + "\n")

        # Daemon neu starten
        daemon_script = PROJECT_ROOT / "bin" / "whisper-dictation-mac.sh"
        subprocess.run([str(daemon_script), "--restart"], capture_output=True)

        self.btn_activate.config(state="disabled", bg="#aaa")
        self.done_label.config(
            text=f"🎉 Fertig! Text-Cleanup mit {model} ist jetzt aktiv."
        )
        self.after(3000, self.destroy)


def main():
    app = OllamaWizard()
    app.mainloop()


if __name__ == "__main__":
    main()
