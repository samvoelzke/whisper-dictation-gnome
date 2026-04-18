#!/usr/bin/env python3
"""Whisper Dictation – macOS Menüleisten-App via PyObjC."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from AppKit import (
    NSApplication, NSApp, NSStatusBar, NSMenu, NSMenuItem,
    NSVariableStatusItemLength, NSObject,
)
from Foundation import NSTimer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DAEMON_SCRIPT = PROJECT_ROOT / "bin" / "whisper-dictation-mac.sh"
SETTINGS_SCRIPT = PROJECT_ROOT / "gui" / "settings_macos.py"
LOG_FILE = Path.home() / ".cache" / "whisper-dictation" / "daemon.log"
CONFIG_FILE = Path.home() / ".config" / "whisper-dictation" / "config.json"

OLLAMA_MODELS = ["llama3.2:3b", "llama3.2:1b", "phi3:mini", "gemma3:1b", "mistral:7b"]


# ── Hilfsfunktionen (außerhalb der Klasse, damit PyObjC sie nicht als Selektoren registriert)

def mi(menu, title, action, target):
    """Erstellt einen NSMenuItem und hängt ihn ans Menu."""
    item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action or "", "")
    if action and target:
        item.setTarget_(target)
    menu.addItem_(item)
    return item


def daemon_running() -> bool:
    r = subprocess.run(["pgrep", "-f", "dictation/daemon.py"],
                       capture_output=True, text=True, check=False)
    return r.returncode == 0 and bool(r.stdout.strip())


def run_daemon(arg: str) -> None:
    subprocess.run([str(DAEMON_SCRIPT), arg], check=False, capture_output=True)


def ax_trusted() -> bool:
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


def ollama_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=1)
        return True
    except Exception:
        return False


def ollama_installed() -> bool:
    r = subprocess.run(["which", "ollama"], capture_output=True, text=True, check=False)
    return r.returncode == 0


def load_config() -> dict:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return {}


def save_config_key(key: str, value) -> None:
    cfg = load_config()
    cfg[key] = value
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


# ── App Delegate

class AppDelegate(NSObject):

    def applicationDidFinishLaunching_(self, notification):
        self._settings_proc = None
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            NSVariableStatusItemLength
        )
        self.status_item.button().setTitle_("🎙")
        self._build_menu()
        self._update_status()
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            5.0, self, "updateStatus:", None, True
        )

    def _build_menu(self):
        t = self  # target shorthand
        menu = NSMenu.alloc().init()

        self.status_label = mi(menu, "● Daemon läuft", None, t)
        self.status_label.setEnabled_(False)
        menu.addItem_(NSMenuItem.separatorItem())

        mi(menu, "▶  Daemon starten",     "startDaemon:",   t)
        mi(menu, "■  Daemon stoppen",     "stopDaemon:",    t)
        mi(menu, "↺  Daemon neu starten", "restartDaemon:", t)
        menu.addItem_(NSMenuItem.separatorItem())

        # ── Ollama Untermenü
        ollama_menu = NSMenu.alloc().init()

        self.ollama_status_label = mi(ollama_menu, "○ Ollama nicht gestartet", None, t)
        self.ollama_status_label.setEnabled_(False)
        ollama_menu.addItem_(NSMenuItem.separatorItem())

        self.ollama_toggle = mi(ollama_menu, "◻  Text-Cleanup aktivieren", "toggleOllama:", t)
        ollama_menu.addItem_(NSMenuItem.separatorItem())

        # Modell-Untermenü
        model_menu = NSMenu.alloc().init()
        self._model_items: dict = {}
        cfg = load_config()
        current_model = cfg.get("ollama_model", "llama3.2:3b")
        for m in OLLAMA_MODELS:
            prefix = "✓ " if m == current_model else "   "
            item = mi(model_menu, f"{prefix}{m}", "selectModel:", t)
            item.setRepresentedObject_(m)
            self._model_items[m] = item

        model_parent = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "🧠  Modell wählen", None, ""
        )
        model_parent.setSubmenu_(model_menu)
        ollama_menu.addItem_(model_parent)

        mi(ollama_menu, "⬇  Modell herunterladen", "downloadModel:", t)
        ollama_menu.addItem_(NSMenuItem.separatorItem())
        self.ollama_install = mi(ollama_menu, "📦  Ollama installieren (brew)", "installOllama:", t)
        mi(ollama_menu, "▶  Ollama starten", "startOllama:", t)

        ollama_parent = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "🤖  Ollama", None, ""
        )
        ollama_parent.setSubmenu_(ollama_menu)
        menu.addItem_(ollama_parent)

        menu.addItem_(NSMenuItem.separatorItem())
        mi(menu, "⚙  Einstellungen",        "openSettings:", t)
        mi(menu, "📋  Log anzeigen",         "openLog:",      t)
        menu.addItem_(NSMenuItem.separatorItem())
        mi(menu, "🔐  Accessibility prüfen", "checkAX:",      t)
        menu.addItem_(NSMenuItem.separatorItem())
        mi(menu, "Beenden",                  "quitApp:",      t)

        self.status_item.setMenu_(menu)
        self._update_ollama_menu()

    def _update_status(self):
        running = daemon_running()
        trusted = ax_trusted()
        if not trusted:
            self.status_item.button().setTitle_("🎙⚠")
            self.status_label.setTitle_("⚠ Accessibility fehlt!")
        elif running:
            self.status_item.button().setTitle_("🎙")
            self.status_label.setTitle_("● Daemon läuft")
        else:
            self.status_item.button().setTitle_("🎙✕")
            self.status_label.setTitle_("○ Daemon gestoppt")
        self._update_ollama_menu()

    def _update_ollama_menu(self):
        cfg = load_config()
        postprocess = cfg.get("ollama_postprocess", False)
        current_model = cfg.get("ollama_model", "llama3.2:3b")
        running = ollama_running()
        installed = ollama_installed()

        self.ollama_status_label.setTitle_(
            "● Ollama läuft" if running else
            ("○ Ollama gestoppt" if installed else "✗ Ollama nicht installiert")
        )
        self.ollama_toggle.setTitle_(
            "◼  Text-Cleanup deaktivieren" if postprocess else "◻  Text-Cleanup aktivieren"
        )
        self.ollama_install.setEnabled_(not installed)

        for m, item in self._model_items.items():
            item.setTitle_(("✓ " if m == current_model else "   ") + m)

    def updateStatus_(self, timer):
        self._update_status()

    # ── Daemon

    def startDaemon_(self, sender):
        threading.Thread(
            target=lambda: (run_daemon("--start"), time.sleep(2), self._update_status()),
            daemon=True,
        ).start()

    def stopDaemon_(self, sender):
        threading.Thread(
            target=lambda: (run_daemon("--stop"), time.sleep(1), self._update_status()),
            daemon=True,
        ).start()

    def restartDaemon_(self, sender):
        threading.Thread(
            target=lambda: (run_daemon("--restart"), time.sleep(2), self._update_status()),
            daemon=True,
        ).start()

    # ── Ollama

    def toggleOllama_(self, sender):
        cfg = load_config()
        save_config_key("ollama_postprocess", not cfg.get("ollama_postprocess", False))
        run_daemon("--restart")
        time.sleep(1)
        self._update_ollama_menu()

    def selectModel_(self, sender):
        model = sender.representedObject()
        if model:
            save_config_key("ollama_model", model)
            run_daemon("--restart")
            self._update_ollama_menu()

    def downloadModel_(self, sender):
        model = load_config().get("ollama_model", "llama3.2:3b")
        script = f'tell application "Terminal" to do script "ollama pull {model} && echo \\"✓ Fertig!\\""'
        subprocess.run(["osascript", "-e", script], check=False)

    def installOllama_(self, sender):
        script = 'tell application "Terminal" to do script "brew install ollama && ollama serve"'
        subprocess.run(["osascript", "-e", script], check=False)

    def startOllama_(self, sender):
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        time.sleep(2)
        self._update_ollama_menu()

    # ── Sonstiges

    def openSettings_(self, sender):
        if self._settings_proc and self._settings_proc.poll() is None:
            return
        self._settings_proc = subprocess.Popen(
            [PYTHON, str(SETTINGS_SCRIPT)], start_new_session=True
        )

    def openLog_(self, sender):
        if LOG_FILE.exists():
            subprocess.run(["open", str(LOG_FILE)], check=False)

    def checkAX_(self, sender):
        if not ax_trusted():
            subprocess.run(
                ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"],
                check=False,
            )

    def quitApp_(self, sender):
        run_daemon("--stop")
        NSApp.terminate_(None)


def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)  # kein Dock-Icon
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()


if __name__ == "__main__":
    main()
