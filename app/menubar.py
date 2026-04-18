#!/usr/bin/env python3
"""Whisper Dictation – macOS Menüleisten-App via PyObjC."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import objc
from AppKit import (
    NSApplication, NSApp, NSStatusBar, NSMenu, NSMenuItem,
    NSImage, NSVariableStatusItemLength, NSObject,
)
from Foundation import NSTimer, NSRunLoop, NSDefaultRunLoopMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DAEMON_SCRIPT = PROJECT_ROOT / "bin" / "whisper-dictation-mac.sh"
SETTINGS_SCRIPT = PROJECT_ROOT / "gui" / "settings_macos.py"
LOG_FILE = Path.home() / ".cache" / "whisper-dictation" / "daemon.log"


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


class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        self._settings_proc = None

        # Status-Item in der Menüleiste
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            NSVariableStatusItemLength
        )
        self.status_item.button().setTitle_("🎙")

        self._build_menu()
        self._update_status()

        # Timer alle 5s für Status-Update
        self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            5.0, self, "updateStatus:", None, True
        )

    def _build_menu(self):
        menu = NSMenu.alloc().init()

        self.status_item_label = self._add_item(menu, "● Daemon läuft", None)
        self.status_item_label.setEnabled_(False)
        menu.addItem_(NSMenuItem.separatorItem())

        self._add_item(menu, "▶  Daemon starten",   "startDaemon:")
        self._add_item(menu, "■  Daemon stoppen",   "stopDaemon:")
        self._add_item(menu, "↺  Daemon neu starten", "restartDaemon:")
        menu.addItem_(NSMenuItem.separatorItem())

        self._add_item(menu, "⚙  Einstellungen",   "openSettings:")
        self._add_item(menu, "📋  Log anzeigen",    "openLog:")
        menu.addItem_(NSMenuItem.separatorItem())

        self._add_item(menu, "🔐  Accessibility prüfen", "checkAX:")
        menu.addItem_(NSMenuItem.separatorItem())

        self._add_item(menu, "Beenden", "quitApp:")

        self.status_item.setMenu_(menu)

    def _add_item(self, menu, title, action):
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            title, action, ""
        )
        item.setTarget_(self)
        menu.addItem_(item)
        return item

    def _update_status(self):
        running = daemon_running()
        trusted = ax_trusted()
        if not trusted:
            self.status_item.button().setTitle_("🎙⚠")
            self.status_item_label.setTitle_("⚠ Accessibility fehlt!")
        elif running:
            self.status_item.button().setTitle_("🎙")
            self.status_item_label.setTitle_("● Daemon läuft")
        else:
            self.status_item.button().setTitle_("🎙✕")
            self.status_item_label.setTitle_("○ Daemon gestoppt")

    def updateStatus_(self, timer):
        self._update_status()

    def startDaemon_(self, sender):
        threading.Thread(target=lambda: (run_daemon("--start"), time.sleep(2), self._update_status()), daemon=True).start()

    def stopDaemon_(self, sender):
        threading.Thread(target=lambda: (run_daemon("--stop"), time.sleep(1), self._update_status()), daemon=True).start()

    def restartDaemon_(self, sender):
        threading.Thread(target=lambda: (run_daemon("--restart"), time.sleep(2), self._update_status()), daemon=True).start()

    def openSettings_(self, sender):
        if self._settings_proc and self._settings_proc.poll() is None:
            return
        self._settings_proc = subprocess.Popen(
            [PYTHON, str(SETTINGS_SCRIPT)],
            start_new_session=True,
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
    # LSUIElement=true in Info.plist verhindert Dock-Icon
    # aber wir setzen es auch hier zur Sicherheit
    app.setActivationPolicy_(1)  # NSApplicationActivationPolicyAccessory

    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()


if __name__ == "__main__":
    main()
