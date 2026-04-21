#!/usr/bin/env python3
"""Text-Rewrite popup — GTK4/Adwaita, reliable input on Wayland."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, GLib, Gio, Gtk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = Path.home() / ".config" / "whisper-dictation" / "config.json"

SYSTEM_PROMPTS = {
    "schoener": (
        "Verbessere den folgenden deutschen Text stilistisch: "
        "bessere Wortwahl, flüssigerer Satz, natürlicher Klang. "
        "Behalte die Bedeutung exakt bei. "
        "Gib NUR den verbesserten Text aus."
    ),
    "formeller": (
        "Mache den folgenden deutschen Text formeller und professioneller. "
        "Geeignet für geschäftliche E-Mails oder offizielle Dokumente. "
        "Gib NUR den verbesserten Text aus."
    ),
    "kuerzer": (
        "Kürze den folgenden deutschen Text prägnant auf das Wesentliche. "
        "Verliere keine wichtigen Informationen. "
        "Gib NUR den gekürzten Text aus."
    ),
}

MODES = [
    ("schoener",  "✨ Schöner"),
    ("formeller", "📋 Formeller"),
    ("kuerzer",   "✂️ Kürzer"),
    ("eigener",   "💬 Eigener Prompt"),
]

CSS_TOOLBAR = """
window.toolbar-win,
window.toolbar-win decoration,
window.toolbar-win > *,
window.toolbar-win > * > * {
    background-color: rgba(0,0,0,0);
    background: unset;
    box-shadow: none;
    border: none;
}
.toolbar-box {
    background-color: #1c1c22;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.12);
    padding: 5px 8px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.6);
}
.toolbar-btn {
    border-radius: 999px;
    padding: 5px 14px;
    background-color: rgba(0,0,0,0);
    border: none;
    color: #e8e8e8;
    font-size: 10.5pt;
    min-height: 0;
}
.toolbar-btn:hover {
    background-color: rgba(255,255,255,0.10);
}
"""

CSS_PANEL = """
.mode-btn {
    border-radius: 99px;
    padding: 4px 12px;
    font-size: 9pt;
}
.mode-btn:checked {
    background: @accent_color;
    color: @accent_fg_color;
}
.orig-view {
    background: alpha(currentColor, 0.04);
    border-radius: 8px;
    padding: 8px;
    font-size: 10pt;
}
.sugg-view {
    background: alpha(@accent_color, 0.07);
    border-radius: 8px;
    padding: 8px;
    font-size: 10pt;
    border: 1px solid alpha(@accent_color, 0.2);
}
.panel-label {
    font-size: 8pt;
    font-weight: bold;
    color: alpha(currentColor, 0.45);
    letter-spacing: 1px;
}
.btn-replace {
    border-radius: 99px;
    padding: 6px 20px;
    font-weight: bold;
}
"""


def _load_model() -> str:
    try:
        cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return str(cfg.get("rewrite_model") or cfg.get("postprocess_model", "qwen3:14b-q4_K_M"))
    except Exception:
        return "qwen3:14b-q4_K_M"


def call_ollama(model: str, mode: str, original: str, custom: str = "") -> str:
    system = custom.strip() if mode == "eigener" else SYSTEM_PROMPTS.get(mode, "")
    if not system:
        system = "Verarbeite den folgenden Text."
    payload = {
        "model": model, "prompt": original, "system": system,
        "stream": False, "options": {"temperature": 0.45}, "think": False,
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return re.sub(r"<think>.*?</think>", "", str(data["response"]), flags=re.DOTALL).strip()


def _write_clipboard(text: str) -> None:
    subprocess.run(["xclip", "-selection", "clipboard"],
                   input=text.encode("utf-8"), check=True, timeout=3)


def _send_ctrl_v() -> None:
    from pynput import keyboard as kb
    c = kb.Controller()
    time.sleep(0.15)
    with c.pressed(kb.Key.ctrl):
        c.tap("v")


def _move_window(title: str, x: int, y: int) -> None:
    """Move window by title using xdotool (XWayland only)."""
    try:
        r = subprocess.run(
            ["xdotool", "search", "--name", title],
            capture_output=True, timeout=3,
        )
        wid = r.stdout.decode().strip().splitlines()[-1] if r.stdout.strip() else ""
        if wid:
            subprocess.run(["xdotool", "windowmove", wid, str(x), str(y)],
                           check=True, timeout=3)
    except Exception:
        pass


# ── Toolbar window (plain GTK, no Adwaita decorations) ───────────────────────

class ToolbarWindow(Gtk.Window):
    def __init__(self, app: Adw.Application, original_text: str, model: str,
                 mouse_x: int, mouse_y: int) -> None:
        super().__init__()
        self.set_application(app)
        self.original_text = original_text
        self.model = model
        self._mouse_x = mouse_x
        self._mouse_y = mouse_y
        self._closing = False

        self.set_decorated(False)
        self.set_resizable(False)
        self.set_title("whisper-rewrite-toolbar")

        prov = Gtk.CssProvider()
        prov.load_from_data(CSS_TOOLBAR.encode())
        Gtk.StyleContext.add_provider_for_display(
            self.get_display(), prov, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self.add_css_class("toolbar-win")

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
        box.add_css_class("toolbar-box")
        self.set_child(box)

        for key_mode, label in MODES:
            btn = Gtk.Button(label=label)
            btn.add_css_class("toolbar-btn")
            btn.connect("clicked", self._on_btn, key_mode)
            box.append(btn)

        key = Gtk.EventControllerKey()
        key.connect("key-pressed", self._on_key)
        self.add_controller(key)

        focus = Gtk.EventControllerFocus()
        focus.connect("enter", self._on_focus_enter)
        focus.connect("leave", self._on_focus_leave)
        self.add_controller(focus)

        self.connect("map", self._on_map)
        self._ever_focused = False

    def _on_map(self, *_: object) -> None:
        if self._mouse_x > 0 or self._mouse_y > 0:
            GLib.timeout_add(60, self._do_move)
        else:
            self.set_opacity(1)

    def _do_move(self) -> bool:
        _move_window("whisper-rewrite-toolbar", self._mouse_x - 200, self._mouse_y - 60)
        self.set_opacity(1)
        return False

    def _on_focus_leave(self, *_: object) -> None:
        if self._ever_focused and not self._closing:
            self._closing = True
            self.close()

    def _on_focus_enter(self, *_: object) -> None:
        self._ever_focused = True

    def _on_btn(self, _btn: object, mode_key: str) -> None:
        self._closing = True
        self.close()
        GLib.idle_add(self._open_panel, mode_key)

    def _open_panel(self, mode_key: str) -> bool:
        win = PanelWindow(self.get_application(), self.original_text, self.model, mode_key)
        win.present()
        return False

    def _on_key(self, _ctrl: object, keyval: int, *_: object) -> bool:
        if keyval == 0xFF1B:
            self._closing = True
            self.close()
            return True
        return False


# ── Panel window (Adwaita, full UI) ──────────────────────────────────────────

class PanelWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application, original_text: str, model: str,
                 mode_key: str) -> None:
        super().__init__(application=app)
        self.original_text = original_text
        self.model = model
        self._suggestion = ""
        self._active_mode = mode_key

        prov = Gtk.CssProvider()
        prov.load_from_data(CSS_PANEL.encode())
        Gtk.StyleContext.add_provider_for_display(
            self.get_display(), prov, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        key = Gtk.EventControllerKey()
        key.connect("key-pressed", self._on_key)
        self.add_controller(key)

        self.set_default_size(780, 400)
        self.set_title("Text umformulieren")
        self._build(mode_key)

    def _build(self, mode_key: str) -> None:
        toolbar_view = Adw.ToolbarView()
        self.set_content(toolbar_view)

        header = Adw.HeaderBar()
        header.set_title_widget(Adw.WindowTitle(title="Text umformulieren", subtitle=self.model))
        toolbar_view.add_top_bar(header)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        root.set_margin_top(10)
        root.set_margin_bottom(14)
        root.set_margin_start(16)
        root.set_margin_end(16)
        toolbar_view.set_content(root)

        # Custom prompt row
        self._custom_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._custom_box.set_visible(mode_key == "eigener")
        lbl = Gtk.Label(label="DEIN PROMPT", xalign=0)
        lbl.add_css_class("panel-label")
        self._custom_box.append(lbl)
        self._custom_entry = Gtk.Entry(placeholder_text="Prompt eingeben und Enter drücken…")
        self._custom_entry.connect("activate", lambda _: self._start_generation())
        self._custom_box.append(self._custom_entry)
        root.append(self._custom_box)

        # Two text panels
        panels = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12, vexpand=True)
        root.append(panels)

        def _panel(title: str, css: str, text: str) -> Gtk.TextView:
            col = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, hexpand=True)
            panels.append(col)
            l2 = Gtk.Label(label=title, xalign=0)
            l2.add_css_class("panel-label")
            col.append(l2)
            sw = Gtk.ScrolledWindow(vexpand=True, hexpand=True)
            sw.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
            tv = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR, editable=False)
            tv.add_css_class(css)
            tv.get_buffer().set_text(text)
            sw.set_child(tv)
            col.append(sw)
            return tv

        _panel("ORIGINAL", "orig-view", self.original_text)
        self._sugg_tv = _panel("VORSCHLAG", "sugg-view", "")

        self._spinner = Gtk.Spinner()
        self._spinner.set_size_request(36, 36)
        self._spinner.set_halign(Gtk.Align.CENTER)
        self._spinner.set_valign(Gtk.Align.CENTER)

        # Action row
        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        root.append(actions)

        pill_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2, hexpand=True)
        grp: Gtk.ToggleButton | None = None
        for k, lbl_text in MODES:
            short = lbl_text.split(" ", 1)[1] if " " in lbl_text else lbl_text
            tb = Gtk.ToggleButton(label=short)
            tb.add_css_class("mode-btn")
            tb.set_active(k == mode_key)
            if grp is None:
                grp = tb
            else:
                tb.set_group(grp)
            tb.connect("toggled", self._on_mode_switch, k)
            pill_box.append(tb)
        actions.append(pill_box)

        self._btn_redraft = Gtk.Button(label="↺ Neu")
        self._btn_redraft.connect("clicked", lambda _: self._start_generation())
        actions.append(self._btn_redraft)

        self._btn_discard = Gtk.Button(label="Verwerfen")
        self._btn_discard.connect("clicked", lambda _: self.close())
        actions.append(self._btn_discard)

        self._btn_replace = Gtk.Button(label="Ersetzen ✓")
        self._btn_replace.add_css_class("suggested-action")
        self._btn_replace.add_css_class("btn-replace")
        self._btn_replace.connect("clicked", self._on_replace)
        actions.append(self._btn_replace)

        if mode_key == "eigener":
            self._set_loading(False)
            self._sugg_tv.get_buffer().set_text("Prompt eingeben und Enter drücken…")
        else:
            self._set_loading(True)
            self._start_generation()

    def _on_mode_switch(self, btn: Gtk.ToggleButton, mode_key: str) -> None:
        if not btn.get_active():
            return
        self._active_mode = mode_key
        self._custom_box.set_visible(mode_key == "eigener")
        if mode_key != "eigener":
            self._start_generation()

    def _start_generation(self) -> None:
        self._set_loading(True)
        self._sugg_tv.get_buffer().set_text("")
        mode = self._active_mode
        custom = self._custom_entry.get_text().strip() if mode == "eigener" else ""
        threading.Thread(target=self._run_ollama, args=(mode, custom), daemon=True).start()

    def _run_ollama(self, mode: str, custom: str) -> None:
        try:
            result = call_ollama(self.model, mode, self.original_text, custom)
            GLib.idle_add(self._show_result, result)
        except Exception as exc:
            GLib.idle_add(self._show_error, str(exc))

    def _show_result(self, text: str) -> bool:
        self._suggestion = text
        self._sugg_tv.get_buffer().set_text(text)
        self._set_loading(False)
        return False

    def _show_error(self, msg: str) -> bool:
        self._suggestion = ""
        self._sugg_tv.get_buffer().set_text(f"[Fehler: {msg}]")
        self._set_loading(False)
        return False

    def _set_loading(self, loading: bool) -> None:
        self._spinner.set_visible(loading)
        if loading:
            self._spinner.start()
        else:
            self._spinner.stop()
        self._btn_replace.set_sensitive(not loading)
        self._btn_redraft.set_sensitive(not loading)

    def _on_replace(self, _btn: object) -> None:
        if not self._suggestion:
            return
        try:
            _write_clipboard(self._suggestion)
            self.close()
            threading.Thread(target=_send_ctrl_v, daemon=True).start()
        except Exception as exc:
            print(f"[rewrite] paste failed: {exc}", file=sys.stderr, flush=True)

    def _on_key(self, _ctrl: object, keyval: int, *_: object) -> bool:
        if keyval == 0xFF1B:
            self.close()
            return True
        return False


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: rewrite.py <text> [model] [mouse_x] [mouse_y]", file=sys.stderr)
        return 1
    original_text = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else _load_model()
    mouse_x = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    mouse_y = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    app = Adw.Application(
        application_id=f"local.whisper.rewrite.s{uuid.uuid4().hex[:12]}",
        flags=Gio.ApplicationFlags.NON_UNIQUE,
    )

    def on_activate(a: Adw.Application) -> None:
        win = ToolbarWindow(a, original_text, model, mouse_x, mouse_y)
        win.present()

    app.connect("activate", on_activate)
    return app.run([])


if __name__ == "__main__":
    raise SystemExit(main())
