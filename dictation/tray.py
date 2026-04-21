#!/usr/bin/env python3
"""
Tray icon process — läuft mit System-Python (gi/GTK3 verfügbar).
Empfängt State-Updates vom Daemon via Unix-Socket.
"""
from __future__ import annotations

import socket
import sys
import threading
from pathlib import Path

import gi
gi.require_version("AyatanaAppIndicator3", "0.1")
gi.require_version("Gtk", "3.0")
from gi.repository import AyatanaAppIndicator3, GLib, Gtk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOCKET_PATH = Path.home() / ".cache" / "whisper-dictation" / "tray.sock"

ICONS = {
    "ready":      "audio-input-microphone",
    "recording":  "media-record",
    "processing": "appointment-soon",
    "done":       "emblem-default",
}
LABELS = {
    "ready":      "Whisper Dictation - bereit",
    "recording":  "Whisper Dictation - nimmt auf",
    "processing": "Whisper Dictation - verarbeitet",
    "done":       "Whisper Dictation - eingefügt ✓",
}


class TrayApp:
    def __init__(self) -> None:
        self.indicator = AyatanaAppIndicator3.Indicator.new(
            "whisper-dictation",
            ICONS["ready"],
            AyatanaAppIndicator3.IndicatorCategory.APPLICATION_STATUS,
        )
        self.indicator.set_status(AyatanaAppIndicator3.IndicatorStatus.ACTIVE)
        self.indicator.set_title(LABELS["ready"])

        menu = Gtk.Menu()

        item_rewrite = Gtk.MenuItem(label="✨  Markierung umformulieren")
        item_rewrite.connect("activate", self._open_rewrite)
        menu.append(item_rewrite)

        menu.append(Gtk.SeparatorMenuItem())

        item_settings = Gtk.MenuItem(label="Einstellungen")
        item_settings.connect("activate", self._open_settings)
        menu.append(item_settings)

        item_quit = Gtk.MenuItem(label="Beenden")
        item_quit.connect("activate", self._quit)
        menu.append(item_quit)

        menu.show_all()
        self.indicator.set_menu(menu)

    def set_state(self, state: str) -> None:
        icon = ICONS.get(state, ICONS["ready"])
        label = LABELS.get(state, LABELS["ready"])

        def _update() -> bool:
            self.indicator.set_icon_full(icon, label)
            self.indicator.set_title(label)
            return False

        GLib.idle_add(_update)

    def _open_rewrite(self, _widget: object) -> None:
        import subprocess, json, time
        # Simulate Ctrl+C to copy current selection, then launch rewrite popup
        try:
            from pynput import keyboard as kb
            ctrl = kb.Controller()
            with ctrl.pressed(kb.Key.ctrl):
                ctrl.tap("c")
            time.sleep(0.15)
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True, timeout=2,
            )
            text = result.stdout.decode("utf-8", errors="replace").strip()
        except Exception:
            text = ""
        if not text:
            return
        cfg_file = Path.home() / ".config" / "whisper-dictation" / "config.json"
        try:
            cfg = json.loads(cfg_file.read_text())
            model = str(cfg.get("rewrite_model") or cfg.get("postprocess_model", "qwen3:14b-q4_K_M"))
        except Exception:
            model = "qwen3:14b-q4_K_M"
        rewrite_script = PROJECT_ROOT / "gui" / "rewrite.py"
        subprocess.Popen(["python3", str(rewrite_script), text, model])

    def _open_settings(self, _widget: object) -> None:
        import subprocess
        script = PROJECT_ROOT / "bin" / "open-whisper-dictation-settings.sh"
        subprocess.Popen([str(script)])

    def _quit(self, _widget: object) -> None:
        Gtk.main_quit()


def _socket_listener(app: TrayApp) -> None:
    SOCKET_PATH.unlink(missing_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(SOCKET_PATH))
    srv.listen(5)
    srv.settimeout(1.0)
    while True:
        try:
            conn, _ = srv.accept()
            data = conn.recv(64).decode().strip()
            conn.close()
            if data == "quit":
                GLib.idle_add(Gtk.main_quit)
                break
            app.set_state(data)
        except socket.timeout:
            continue
        except Exception:
            break
    SOCKET_PATH.unlink(missing_ok=True)


def main() -> None:
    app = TrayApp()
    t = threading.Thread(target=_socket_listener, args=(app,), daemon=True)
    t.start()
    Gtk.main()


if __name__ == "__main__":
    main()
