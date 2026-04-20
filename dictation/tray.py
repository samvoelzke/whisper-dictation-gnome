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
