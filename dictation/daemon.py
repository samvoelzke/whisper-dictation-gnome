#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

# hf-xet transfers can stall on some networks; force plain HTTPS downloads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    CACHE_DIR,
    HISTORY_FILE,
    HOTKEY_SPECS,
    IPC_SOCKET,
    IS_LINUX,
    IS_MACOS,
    FASTER_MODEL_MAP,
    OV_MODEL_REPOS,
    clipboard_manager_running,
    cuda_available,
    evdev_code_for,
    key_label,
    load_config,
    load_ov_pipeline,
    ollama_chat,
    save_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHISPER_REPO_ROOT = PROJECT_ROOT / "whisper"

# macOS records via sounddevice; Linux records via arecord (PipeWire/ALSA).
if IS_MACOS:
    try:
        import sounddevice as sd
    except ImportError:
        sd = None  # type: ignore[assignment]
else:
    sd = None  # type: ignore[assignment]

# Optional Ollama text cleanup. Chat-style prompt with few-shot examples (the
# pattern competitor dictation tools use) so a mid-size model edits the text
# instead of answering it, and keeps English terms English in DE+EN speech.
OLLAMA_CLEANUP_SYSTEM = (
    "Du bist ein Korrektur-Werkzeug fuer diktierten Text. Gib AUSSCHLIESSLICH "
    "den korrigierten Text zurueck - keine Erklaerungen, keine Antworten auf den Inhalt.\n"
    "Regeln:\n"
    "- Entferne bedeutungslose Fuellwoerter (aeh, aehm, halt, sozusagen, ne, also).\n"
    "- Setze korrekte Zeichensetzung und Gross-/Kleinschreibung.\n"
    "- Wende gesprochene Selbstkorrekturen an (streiche das Verworfene).\n"
    "- Fuege KEINE Woerter hinzu, aendere NICHT die Bedeutung, mache den Ton NICHT formeller.\n"
    "- Uebersetze NICHT: englische Woerter/Fachbegriffe bleiben Englisch (deployen, Pull Request, Meeting).\n"
    "- Antworte NIEMALS auf den Inhalt; korrigiere ihn nur."
)
OLLAMA_CLEANUP_SHOTS = [
    ("wo ist die fernbedienung", "Wo ist die Fernbedienung?"),
    ("das meeting ist aehm morgen um drei uhr", "Das Meeting ist morgen um drei Uhr."),
    ("kannst du mir sagen wie das geht", "Kannst du mir sagen, wie das geht?"),
    ("lass uns das halt mal schnell deployen und dann den pull request mergen",
     "Lass uns das mal schnell deployen und dann den Pull Request mergen."),
    ("can you um send me the file when you get a chance",
     "Can you send me the file when you get a chance?"),
    ("ich brauche nein warte ich brauche zwei tickets",
     "Ich brauche - nein, warte - ich brauche zwei Tickets."),
]

# Spoken formatting commands (opt-in). Kept small + unambiguous on purpose:
# punctuation words like "Punkt"/"Komma" are everyday words and would misfire,
# so only formatting/symbol phrases that are rarely said literally are mapped.
VOICE_COMMANDS = {
    "neuer absatz": "\n\n",
    "neue zeile": "\n",
    "neuezeile": "\n",
    "doppelpunkt": ":",
    "bindestrich": "-",
    "fragezeichen": "?",
    "ausrufezeichen": "!",
}

TERMINAL_HINTS = (
    "gnome-terminal", "kgx", "console", "tilix", "terminator", "kitty",
    "alacritty", "wezterm", "konsole", "xfce4-terminal",
    "mate-terminal", "lxterminal", "iterm2", "terminal",
)


def apply_voice_commands(text: str) -> str:
    """Replace spoken formatting commands and tidy whitespace around them."""
    import re as _re
    out = text
    for phrase, repl in VOICE_COMMANDS.items():
        out = _re.sub(rf"\b{_re.escape(phrase)}\b", repl, out, flags=_re.IGNORECASE)
    out = _re.sub(r"[ \t]+([:?!-])", r"\1", out)   # no space before inserted symbol
    out = _re.sub(r"[ \t]*\n[ \t]*", "\n", out)      # trim spaces around newlines
    return out.strip()

# Linux input event codes for ydotool paste injection.
_YDOTOOL_KEYS = {
    "ctrl_v": ["29:1", "47:1", "47:0", "29:0"],
    "ctrl_shift_v": ["29:1", "42:1", "47:1", "47:0", "42:0", "29:0"],
    "shift_insert": ["42:1", "110:1", "110:0", "42:0"],
}


def notify(summary: str, body: str = "") -> None:
    if IS_MACOS:
        script = f'display notification "{body}" with title "{summary}"'
        subprocess.run(["osascript", "-e", script], check=False, capture_output=True)
        return
    if shutil_which("notify-send") is None:
        return
    command = [
        "notify-send", "-a", "Whisper Dictation",
        "-i", "io.voelzke.WhisperDictation", summary,
    ]
    if body:
        command.append(body)
    subprocess.run(command, check=False)


def _notify_send_linux(
    summary: str,
    body: str = "",
    urgency: str = "normal",
    timeout_ms: int | None = None,
    replace_id: int | None = None,
) -> int | None:
    """Send/update a notification in place; returns its id (for replacing).

    Using --replace-id keeps a single status bubble that changes its text
    instead of stacking one popup per state. --print-id returns the id.
    """
    if shutil_which("notify-send") is None:
        return replace_id
    command = [
        "notify-send", "-a", "Whisper Dictation",
        "-i", "io.voelzke.WhisperDictation", "-p", "-u", urgency,
    ]
    if timeout_ms is not None:
        command += ["-t", str(int(timeout_ms))]
    if replace_id is not None:
        command += ["-r", str(int(replace_id))]
    command.append(summary)
    if body:
        command.append(body)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    out = result.stdout.strip()
    return int(out) if out.isdigit() else replace_id


def shutil_which(binary: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / binary
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _ax_is_process_trusted() -> bool:
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


def check_macos_accessibility() -> None:
    """Exit with clear instructions if Accessibility permission is missing."""
    if _ax_is_process_trusted():
        return

    real_python = os.path.realpath(sys.executable)
    print(
        f"\n[whisper-dictation] FEHLER: Accessibility-Berechtigung fehlt!\n"
        f"  Systemeinstellungen -> Datenschutz & Sicherheit -> Bedienungshilfen\n"
        f"  -> + klicken -> Cmd+Shift+G -> diesen Pfad einfuegen:\n"
        f"  {real_python}\n"
        f"  -> Schalter aktivieren -> Daemon neu starten.\n",
        flush=True,
    )
    notify(
        "Accessibility-Berechtigung fehlt",
        "Systemeinstellungen -> Bedienungshilfen -> Python eintragen, dann Daemon neu starten.",
    )
    subprocess.run(
        ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"],
        check=False, capture_output=True,
    )
    sys.exit(1)


def read_wav_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        audio_bytes = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise RuntimeError(f"Unsupported sample width: {sample_width * 8} bit")

    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sample_rate != 16000:
        raise RuntimeError(
            f"Unexpected sample rate {sample_rate}. Expected 16000 Hz."
        )

    return audio


class WhisperDictationDaemon:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        # The key value is stored as-is: a logical name (ctrl_r), a raw evdev
        # name (KEY_F9), or a captured "code:N[:label]". Resolved per platform.
        self.hotkey_name = str(config["double_tap_key"])
        self.hotkey_label = key_label(self.hotkey_name) or "Right Ctrl"
        self.hotkey_mode = str(config.get("hotkey_mode", "double_tap")).lower()
        self.double_tap_window = max(150, int(config["double_tap_window_ms"])) / 1000.0

        # Optional second hotkey that toggles Ollama cleanup (must differ).
        toggle_value = str(config.get("llm_toggle_key", ""))
        self.llm_toggle_name = toggle_value if (toggle_value and toggle_value != self.hotkey_name) else None
        self.llm_toggle_label = key_label(self.llm_toggle_name)
        self.last_llm_toggle_release: float | None = None

        # Command mode (rewrite the selection by voice).
        command_value = str(config.get("command_key", ""))
        self.command_name = command_value if (command_value and command_value != self.hotkey_name) else None
        self.command_label = key_label(self.command_name)
        self.last_command_release: float | None = None
        self._command_active = False
        self._command_selection = ""

        self.backend = self._resolve_backend()

        # Transcription models (only one is populated, depending on backend).
        self.fw_model: Any = None
        self.model: Any = None
        self.ov_pipe: Any = None

        self.lock = threading.RLock()
        self._infer_lock = threading.Lock()  # serialize model inference (dictation + IPC)

        # Linux: arecord subprocess; macOS: sounddevice thread.
        self.recording_process: subprocess.Popen[bytes] | None = None
        self.recording_sd_thread: threading.Thread | None = None
        self.recording_sd_stop: threading.Event | None = None
        self.recording_sd_frames: list[np.ndarray] = []

        # Hotkey listener handles (one path per platform).
        self.listener: Any = None  # pynput listener (macOS)
        self.listener_thread: threading.Thread | None = None  # evdev (Linux)
        self._evdev_devices: list[Any] = []

        self.recording_file: Path | None = None
        self.recording_timer: threading.Timer | None = None
        self.last_hotkey_release: float | None = None
        self.busy = False
        self.stopping = False
        # Single, in-place-updated status notification for the dictation cycle.
        self._notify_id: int | None = None

    def _resolve_backend(self) -> str:
        backend = str(self.config.get("backend", "auto")).lower()
        if backend in ("faster", "openai", "openvino"):
            return backend
        # auto: openai-whisper on macOS (keeps MPS); on Linux prefer OpenVINO
        # (Intel GPU/NPU) when available and the model has an OV export, else
        # fall back to faster-whisper on CPU.
        if IS_MACOS:
            return "openai"
        import importlib.util
        model_name = str(self.config.get("model", "turbo"))
        if (
            importlib.util.find_spec("openvino_genai") is not None
            and model_name in OV_MODEL_REPOS
        ):
            return "openvino"
        return "faster"

    # -- Linux: ALSA mic volume -------------------------------------------------

    def _init_mic_volume(self) -> None:
        """Raise a near-muted capture volume so recordings aren't silent.

        Resolved by control name ('Capture') — numeric control ids are
        card-specific and would hit a random control on other hardware.
        """
        if not IS_LINUX:
            return
        device = str(self.config.get("record_device", "default"))
        m = re.match(r"(?:plug)?hw:(\d+)", device)
        if not m:
            return
        card = m.group(1)
        result = subprocess.run(
            ["amixer", "-c", card, "sget", "Capture"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            return
        percents = [int(p) for p in re.findall(r"\[(\d+)%\]", result.stdout)]
        if percents and min(percents) < 20:
            subprocess.run(
                ["amixer", "-c", card, "sset", "Capture", "65%"],
                check=False, capture_output=True,
            )
            print(f"[whisper-dictation] mic capture volume raised to 65% on card {card}", flush=True)

    # -- Startup ----------------------------------------------------------------

    def run(self) -> None:
        if IS_MACOS:
            check_macos_accessibility()
        self._init_mic_volume()
        self._load_model()
        self._start_ipc_server()

        if IS_MACOS:
            self._start_listener_macos()
        else:
            self._start_listener_linux()

    def _load_model(self) -> None:
        model_name = str(self.config["model"])
        print(
            f"[whisper-dictation] platform={platform.system()} backend={self.backend} "
            f"loading model={model_name}",
            flush=True,
        )
        notify("Lade Modell", f"{model_name} ({self.backend})")

        if self.backend == "openvino":
            try:
                self._load_openvino(model_name)
            except Exception as exc:
                print(
                    f"[whisper-dictation] OpenVINO load failed ({exc}); "
                    f"falling back to faster-whisper on CPU",
                    file=sys.stderr, flush=True,
                )
                notify("OpenVINO nicht verfuegbar", "Nutze faster-whisper (CPU).")
                self.backend = "faster"
                self.ov_pipe = None

        if self.backend == "faster":
            from faster_whisper import WhisperModel

            fw_name = FASTER_MODEL_MAP.get(model_name, model_name)
            if cuda_available():
                device, compute_type = "cuda", "float16"
            else:
                device, compute_type = "cpu", "int8"
            print(
                f"[whisper-dictation] faster-whisper model={fw_name} "
                f"device={device} compute_type={compute_type}",
                flush=True,
            )
            self.fw_model = WhisperModel(
                fw_name,
                device=device,
                compute_type=compute_type,
                download_root=str(CACHE_DIR / "models-faster"),
            )
        elif self.backend == "openai":
            if str(WHISPER_REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(WHISPER_REPO_ROOT))
            import whisper

            device = "cuda" if cuda_available() else (
                "mps" if (IS_MACOS and self._mps_available()) else "cpu"
            )
            print(f"[whisper-dictation] openai-whisper device={device}", flush=True)
            self.model = whisper.load_model(
                model_name,
                device=device,
                download_root=str(CACHE_DIR / "models"),
            )
            self._openai_device = device

        print("[whisper-dictation] model ready", flush=True)
        notify("Bereit", f"Doppelt {self.hotkey_label} startet oder stoppt die Aufnahme")

    def _load_openvino(self, model_name: str) -> None:
        self.ov_pipe, self._ov_device = load_ov_pipeline(
            self.config, model_name, "whisper-dictation", notify=notify,
        )

    @staticmethod
    def _mps_available() -> bool:
        try:
            import torch
            return bool(torch.backends.mps.is_available())
        except Exception:
            return False

    # -- Hotkey: macOS (pynput) -------------------------------------------------

    def _start_listener_macos(self) -> None:
        from pynput import keyboard

        # macOS only supports the named keys; captured codes fall back.
        name = self.hotkey_name.lower()
        if name not in HOTKEY_SPECS:
            print(f"[whisper-dictation] hotkey {self.hotkey_name!r} unsupported on "
                  f"macOS; using Right Ctrl", file=sys.stderr, flush=True)
            name = "ctrl_r"
        _, pynput_attr, fallback_attrs, _ = HOTKEY_SPECS[name]
        self._mac_hotkey = getattr(keyboard.Key, pynput_attr)
        self._mac_fallbacks = frozenset(
            getattr(keyboard.Key, name) for name in fallback_attrs
        )

        def is_hotkey(key: Any) -> bool:
            return key == self._mac_hotkey or key in self._mac_fallbacks

        def on_press(key: Any) -> None:
            if is_hotkey(key):
                if self.hotkey_mode == "push_to_talk":
                    self._ptt_start()
            else:
                with self.lock:
                    self.last_hotkey_release = None

        def on_release(key: Any) -> None:
            if is_hotkey(key):
                if self.hotkey_mode == "push_to_talk":
                    self._ptt_stop()
                else:
                    self._register_release()

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        print("[whisper-dictation] pynput listener started", flush=True)
        self.listener.start()
        self.listener.join()

    # -- Hotkey: Linux/Wayland (evdev) ------------------------------------------

    def _start_listener_linux(self) -> None:
        try:
            import evdev
            from evdev import ecodes
        except ImportError:
            raise RuntimeError(
                "python-evdev fehlt. Bitte 'pip install evdev' im venv ausfuehren."
            )

        target = evdev_code_for(self.hotkey_name, ecodes)
        if target is None:
            notify("Hotkey ungueltig", f"Taste '{self.hotkey_name}' nicht erkannt.")
            raise RuntimeError(f"Unresolvable hotkey: {self.hotkey_name!r}")
        toggle_target = evdev_code_for(self.llm_toggle_name, ecodes) if self.llm_toggle_name else None
        command_target = evdev_code_for(self.command_name, ecodes) if self.command_name else None
        wanted = {target}
        for extra in (toggle_target, command_target):
            if extra is not None:
                wanted.add(extra)

        devices: list[Any] = []
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
            except Exception:
                continue
            caps = dev.capabilities()
            keys = caps.get(ecodes.EV_KEY, [])
            if any(code in keys for code in wanted):
                devices.append(dev)

        if not devices:
            notify(
                "Hotkey nicht moeglich",
                "Keine Tastatur lesbar. Ist dein User in der 'input'-Gruppe?",
            )
            raise RuntimeError(
                "Keine Eingabegeraete mit der Hotkey-Taste lesbar "
                "(/dev/input). User muss in der 'input'-Gruppe sein."
            )

        self._evdev_devices = devices
        labels = ", ".join(d.name for d in devices)
        extra_info = ""
        if toggle_target:
            extra_info += f" + LLM-Switch={self.llm_toggle_label}"
        if command_target:
            extra_info += f" + Befehl={self.command_label}"
        print(
            f"[whisper-dictation] evdev listener on {len(devices)} device(s): {labels}{extra_info}",
            flush=True,
        )

        self.listener_thread = threading.Thread(
            target=self._evdev_loop, args=(target, toggle_target, command_target, ecodes),
            daemon=True,
        )
        self.listener_thread.start()
        self.listener_thread.join()

    def _evdev_loop(self, target: int, toggle_target: int | None,
                    command_target: int | None, ecodes: Any) -> None:
        import selectors

        # Modifier keys don't reset the double-tap timer. Some keys (e.g. the
        # Copilot/AI key = Meta+Shift+F23) emit modifiers alongside the trigger;
        # without this they would cancel their own double-tap.
        modifier_codes = {
            getattr(ecodes, name) for name in (
                "KEY_LEFTSHIFT", "KEY_RIGHTSHIFT", "KEY_LEFTCTRL", "KEY_RIGHTCTRL",
                "KEY_LEFTALT", "KEY_RIGHTALT", "KEY_LEFTMETA", "KEY_RIGHTMETA",
            ) if hasattr(ecodes, name)
        }
        self._esc_code = getattr(ecodes, "KEY_ESC", None)

        selector = selectors.DefaultSelector()
        for dev in self._evdev_devices:
            selector.register(dev, selectors.EVENT_READ)

        while not self.stopping:
            for key, _mask in selector.select(timeout=0.5):
                dev = key.fileobj
                try:
                    for event in dev.read():
                        if event.type != ecodes.EV_KEY:
                            continue
                        self._process_key_event(
                            event.code, event.value, target, toggle_target,
                            command_target, modifier_codes,
                        )
                except BlockingIOError:
                    # No events ready right now; not an error.
                    continue
                except OSError:
                    # Device went away (unplugged); stop watching it.
                    try:
                        selector.unregister(dev)
                    except Exception:
                        pass

    def _process_key_event(self, code: int, value: int, target: int,
                           toggle_target: int | None, command_target: int | None,
                           modifier_codes: set) -> None:
        # Esc while recording -> cancel without transcribing.
        if value == 1 and code == getattr(self, "_esc_code", None) and self._is_recording():
            self.cancel_recording()
            return
        if code == target:
            if self.hotkey_mode == "push_to_talk":
                if value == 1:      # press -> start recording
                    self._ptt_start()
                elif value == 0:    # release -> stop + transcribe
                    self._ptt_stop()
            elif value == 0:        # double-tap: release counts
                self._register_release()
            elif value == 1:        # press resets the *other* timers
                with self.lock:
                    self.last_llm_toggle_release = None
                    self.last_command_release = None
        elif toggle_target is not None and code == toggle_target:
            if value == 0:
                self._register_llm_toggle()
            elif value == 1:
                with self.lock:
                    self.last_hotkey_release = None
                    self.last_command_release = None
        elif command_target is not None and code == command_target:
            if value == 0:
                self._register_command()
            elif value == 1:
                with self.lock:
                    self.last_hotkey_release = None
                    self.last_llm_toggle_release = None
        elif value == 1 and code not in modifier_codes:
            # a real other key press resets the double-tap timers
            with self.lock:
                self.last_hotkey_release = None
                self.last_llm_toggle_release = None
                self.last_command_release = None

    # -- Double-tap detection (shared) ------------------------------------------

    def _register_release(self) -> None:
        now = time.monotonic()
        with self.lock:
            if self.last_hotkey_release is not None:
                delta = now - self.last_hotkey_release
                self.last_hotkey_release = None
                if delta <= self.double_tap_window:
                    self.toggle_recording()
                    return
            self.last_hotkey_release = now

    def _register_llm_toggle(self) -> None:
        now = time.monotonic()
        with self.lock:
            if self.last_llm_toggle_release is not None:
                delta = now - self.last_llm_toggle_release
                self.last_llm_toggle_release = None
                if delta <= self.double_tap_window:
                    self.toggle_llm_cleanup()
                    return
            self.last_llm_toggle_release = now

    def _register_command(self) -> None:
        now = time.monotonic()
        with self.lock:
            if self.last_command_release is not None:
                delta = now - self.last_command_release
                self.last_command_release = None
                if delta <= self.double_tap_window:
                    self._toggle_command_mode()
                    return
            self.last_command_release = now

    def _toggle_command_mode(self) -> None:
        if self._command_active and self._is_recording():
            self.stop_recording()           # completion applies the instruction
        elif not self.busy and not self._is_recording():
            self._start_command_mode()

    def _start_command_mode(self) -> None:
        """Grab the current selection (Wayland PRIMARY) and record an instruction."""
        if shutil_which("wl-paste") is None:
            self._status("Befehlsmodus nicht moeglich", "wl-paste fehlt.", timeout_ms=4000)
            return
        try:
            sel = subprocess.run(
                ["wl-paste", "--primary", "--no-newline"],
                capture_output=True, timeout=2,
            ).stdout.decode("utf-8", "replace").strip()
        except Exception:
            sel = ""
        if not sel:
            self._status("Nichts markiert", "Markiere zuerst Text, dann doppelt druecken.", timeout_ms=4000)
            return
        self._command_selection = sel
        self._command_active = True
        self.start_recording()

    def toggle_llm_cleanup(self) -> None:
        """Flip Ollama post-processing on/off and persist it."""
        with self.lock:
            enabled = not bool(self.config.get("ollama_postprocess", False))
            self.config["ollama_postprocess"] = enabled
            self._save_config()
        print(f"[whisper-dictation] llm cleanup toggled -> {enabled}", flush=True)
        self._status(
            "🤖 LLM-Cleanup AN" if enabled else "✏️ LLM-Cleanup AUS",
            "Textverbesserung aktiviert" if enabled else "Roher Whisper-Text",
            timeout_ms=2500,
        )
        self._play_sound("done")

    def _save_config(self) -> None:
        try:
            save_config(self.config)
        except Exception as exc:
            print(f"[whisper-dictation] could not save config: {exc}", file=sys.stderr, flush=True)

    def _append_history(self, text: str) -> None:
        if not self.config.get("save_history", True) or not text.strip():
            return
        try:
            with HISTORY_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.time(), "text": text}, ensure_ascii=False) + "\n")
            lines = HISTORY_FILE.read_text(encoding="utf-8").splitlines()
            if len(lines) > 500:  # keep the file bounded
                HISTORY_FILE.write_text("\n".join(lines[-500:]) + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"[whisper-dictation] history write failed: {exc}", file=sys.stderr, flush=True)

    # Shorter presses of the push-to-talk key are treated as normal shortcut
    # use (Ctrl+C etc.) and cancel silently instead of transcribing.
    PTT_MIN_HOLD = 0.25

    def toggle_recording(self) -> None:
        with self.lock:
            if self.busy:
                self._status("Noch beschäftigt", "Letzte Aufnahme wird verarbeitet.", timeout_ms=3000)
                return
            if not self._is_recording():
                self.start_recording()
            else:
                self.stop_recording()

    def _is_recording(self) -> bool:
        return self.recording_process is not None or self.recording_sd_thread is not None

    def _ptt_start(self) -> None:
        """Push-to-talk: key pressed -> start recording."""
        with self.lock:
            if self.busy or self._is_recording():
                return
            self._ptt_pressed_at = time.monotonic()
            # Audio is captured from the very first moment; only the status
            # bubble/sound wait out the grace period so a plain shortcut
            # press doesn't flash a recording notification.
            self.start_recording(ui_delay=self.PTT_MIN_HOLD)

    def _ptt_stop(self) -> None:
        """Push-to-talk: key released -> stop + transcribe (or silent cancel)."""
        with self.lock:
            if not self._is_recording() or self.busy:
                return
            held = time.monotonic() - getattr(self, "_ptt_pressed_at", 0.0)
            if held < self.PTT_MIN_HOLD:
                self.cancel_recording(quiet=True)
                return
            self.stop_recording()

    def cancel_recording(self, quiet: bool = False) -> None:
        """Abort an in-progress recording without transcribing (Esc)."""
        with self.lock:
            if not self._is_recording() or self.busy:
                return
            if self.recording_timer is not None:
                self.recording_timer.cancel()
                self.recording_timer = None
            output_path = self.recording_file
            self.recording_file = None
            if self.recording_process is not None:
                try:
                    self.recording_process.send_signal(signal.SIGINT)
                except Exception:
                    pass
                self.recording_process = None
            if self.recording_sd_stop is not None:
                self.recording_sd_stop.set()
            self.recording_sd_thread = None
            self.recording_sd_stop = None
        if output_path:
            output_path.unlink(missing_ok=True)
        self._command_active = False
        self._command_selection = ""
        print("[whisper-dictation] recording cancelled", flush=True)
        if not quiet:
            self._status("✖ Abgebrochen", "Aufnahme verworfen.", timeout_ms=2500)

    # -- Recording: Linux (arecord) ---------------------------------------------

    def _start_recording_linux(self, output_path: Path) -> None:
        if shutil_which("arecord") is None:
            raise RuntimeError("arecord ist nicht installiert (sudo dnf install alsa-utils).")

        command = [
            "arecord", "-q",
            "-D", str(self.config["record_device"]),
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "-t", "wav",
            str(output_path),
        ]
        self.recording_process = subprocess.Popen(command)

    def _stop_recording_linux(self) -> subprocess.Popen[bytes]:
        process = self.recording_process
        self.recording_process = None
        process.send_signal(signal.SIGINT)  # type: ignore[union-attr]
        return process  # type: ignore[return-value]

    # -- Recording: macOS (sounddevice) -----------------------------------------

    def _start_recording_macos(self, output_path: Path) -> None:
        if sd is None:
            raise RuntimeError(
                "sounddevice ist nicht installiert (pip install sounddevice)."
            )

        self.recording_sd_frames = []
        self.recording_sd_stop = threading.Event()
        stop_event = self.recording_sd_stop
        frames = self.recording_sd_frames

        def _record() -> None:
            device_cfg = str(self.config.get("record_device", "default"))
            device_arg: str | int | None = None if device_cfg == "default" else device_cfg
            try:
                with sd.InputStream(
                    samplerate=16000,
                    channels=1,
                    dtype="int16",
                    device=device_arg,
                    blocksize=1024,
                ) as stream:
                    while not stop_event.is_set():
                        data, _ = stream.read(1024)
                        frames.append(data.copy())
            except Exception as exc:
                print(f"[whisper-dictation] sounddevice error: {exc}", file=sys.stderr, flush=True)

        self.recording_sd_thread = threading.Thread(target=_record, daemon=True)
        self.recording_sd_thread.start()

    def _stop_recording_macos(self, output_path: Path) -> None:
        if self.recording_sd_stop is not None:
            self.recording_sd_stop.set()
        if self.recording_sd_thread is not None:
            self.recording_sd_thread.join(timeout=3)
        self.recording_sd_thread = None
        self.recording_sd_stop = None

        frames = self.recording_sd_frames
        self.recording_sd_frames = []

        if frames:
            audio_data = np.concatenate(frames, axis=0)
            with wave.open(str(output_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data.tobytes())

    # -- Recording: common ------------------------------------------------------

    def start_recording(self, ui_delay: float = 0.0) -> None:
        handle = tempfile.NamedTemporaryFile(
            prefix="whisper-dictation-", suffix=".wav", delete=False,
        )
        handle.close()
        output_path = Path(handle.name)
        self.recording_file = output_path

        if IS_MACOS:
            self._start_recording_macos(output_path)
        else:
            self._start_recording_linux(output_path)

        print(f"[whisper-dictation] recording started file={output_path}", flush=True)
        self.recording_timer = threading.Timer(
            int(self.config["max_record_seconds"]),
            self.auto_stop_recording,
        )
        self.recording_timer.daemon = True
        self.recording_timer.start()
        if ui_delay > 0:
            # Push-to-talk grace: show the recording UI only if the key is
            # still held after the grace period (a quick tap cancels quietly).
            def _deferred() -> None:
                with self.lock:
                    if self._is_recording() and not self.busy:
                        self._recording_ui()
            t = threading.Timer(ui_delay, _deferred)
            t.daemon = True
            t.start()
        else:
            self._recording_ui()

    def _recording_ui(self) -> None:
        if self._command_active:
            self._status("🎙 Befehl sprechen…",
                         f"Doppelt {self.command_label} = auf Auswahl anwenden",
                         urgency="critical", timeout_ms=0)
        else:
            stop_hint = (
                f"{self.hotkey_label} loslassen zum Stoppen"
                if self.hotkey_mode == "push_to_talk"
                else f"Doppelt {self.hotkey_label} zum Stoppen"
            )
            self._status("🔴 Aufnahme läuft", stop_hint, urgency="critical", timeout_ms=0)
        self._play_sound("start")

    def auto_stop_recording(self) -> None:
        with self.lock:
            if not self._is_recording() or self.busy:
                return
            self.stop_recording()

    def stop_recording(self) -> None:
        with self.lock:
            output_path = self.recording_file
            if output_path is None:
                return

            self.recording_file = None
            self.busy = True
            print(f"[whisper-dictation] recording stopped file={output_path}", flush=True)

            if self.recording_timer is not None:
                self.recording_timer.cancel()
                self.recording_timer = None

            if IS_MACOS:
                worker = threading.Thread(
                    target=self._stop_and_transcribe_macos,
                    args=(output_path,),
                    daemon=True,
                )
            else:
                process = self._stop_recording_linux()
                worker = threading.Thread(
                    target=self._stop_and_transcribe_linux,
                    args=(process, output_path),
                    daemon=True,
                )

        worker.start()
        self._status("✍️ Transkribiere…", "Aufnahme wird erkannt", urgency="critical", timeout_ms=0)

    def _stop_and_transcribe_macos(self, output_path: Path) -> None:
        self._stop_recording_macos(output_path)
        self._transcribe_and_paste(output_path)

    def _stop_and_transcribe_linux(
        self, process: subprocess.Popen[bytes], output_path: Path
    ) -> None:
        try:
            process.wait(timeout=5)
        except Exception:
            pass
        self._transcribe_and_paste(output_path)

    def _transcribe_and_paste(self, output_path: Path) -> None:
        try:
            if not output_path.exists() or output_path.stat().st_size < 100:
                self._status("Kein Text erkannt", "Die Aufnahme war leer.", timeout_ms=4000)
                return

            audio = read_wav_mono(output_path)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            print(f"[whisper-dictation] audio rms={rms:.5f}", flush=True)
            if rms < 0.002:
                self._status("Kein Text erkannt", "Die Aufnahme war leer oder zu leise.", timeout_ms=4000)
                return

            with self._infer_lock:
                text = self._transcribe_audio(audio).strip()
            if not text:
                self._status("Kein Text erkannt", "Nichts verstanden.", timeout_ms=4000)
                return

            print(f"[whisper-dictation] transcription ready chars={len(text)}", flush=True)

            # Command mode: the transcription is an INSTRUCTION; apply it to the
            # selection we grabbed and replace the selection in place.
            if self._command_active:
                self._command_active = False
                selection = self._command_selection
                self._command_selection = ""
                if not selection:
                    self._status("Befehl abgebrochen", "Keine Auswahl.", timeout_ms=4000)
                    return
                self._status("🤖 Wende Befehl an…", text[:80], urgency="critical", timeout_ms=0)
                try:
                    result = self._ollama_instruct(selection, text).strip()
                except Exception as exc:
                    self._status("Befehl fehlgeschlagen", f"Ollama: {exc}", urgency="critical", timeout_ms=6000)
                    return
                if not result:
                    self._status("Befehl fehlgeschlagen", "Leeres Ergebnis.", timeout_ms=4000)
                    return
                # Keep the result in the clipboard (restore=False): in an
                # editable field Ctrl+V replaces the selection, but on a
                # read-only page (e.g. Wikipedia) the paste is a no-op and the
                # user can still paste the result wherever they want.
                pasted = self._paste_text(result, restore=False)
                if pasted:
                    self._status("✓ Ersetzt (auch in Zwischenablage)", result[:120], timeout_ms=4000)
                else:
                    self._status("📋 In Zwischenablage", "Manuell mit Strg+V: " + result[:90], timeout_ms=6000)
                self._play_sound("done")
                return

            if self.config.get("ollama_postprocess"):
                self._status("🤖 Verfeinere Text…", "Ollama läuft…", urgency="critical", timeout_ms=0)
                try:
                    text = self._ollama_postprocess(text)
                except Exception as exc:
                    print(f"[whisper-dictation] ollama failed, using raw: {exc}", file=sys.stderr, flush=True)

            if self.config.get("voice_commands"):
                text = apply_voice_commands(text)
                if not text:
                    self._status("Kein Text erkannt", "Nichts uebrig nach Befehlen.", timeout_ms=4000)
                    return

            self._append_history(text)
            pasted = self._paste_text(text)
            if pasted:
                self._status("✓ Eingefügt", text[:120], timeout_ms=4000)
            else:
                self._status("📋 In Zwischenablage", "Manuell mit Strg+V: " + text[:90], timeout_ms=6000)
            self._play_sound("done")
        except Exception as exc:
            self._status("Fehler", str(exc), urgency="critical", timeout_ms=6000)
            print(f"[whisper-dictation] {exc}", file=sys.stderr, flush=True)
        finally:
            self.busy = False
            output_path.unlink(missing_ok=True)

    # -- Transcription ----------------------------------------------------------

    def _transcribe_audio(self, audio: np.ndarray) -> str:
        lang_cfg = str(self.config.get("language") or "").strip().lower()
        language = None if lang_cfg in ("", "auto") else lang_cfg
        initial_prompt = str(self.config.get("initial_prompt") or "").strip() or None

        hotwords = str(self.config.get("hotwords") or "").strip() or None

        if self.backend == "openvino":
            if self.ov_pipe is None:
                raise RuntimeError("Model is not loaded.")
            kwargs: dict[str, Any] = {"task": "transcribe"}
            if language:
                kwargs["language"] = f"<|{language}|>"
            if initial_prompt:
                kwargs["initial_prompt"] = initial_prompt
            result = self.ov_pipe.generate(audio, **kwargs)
            return str(result)

        if self.backend == "faster":
            if self.fw_model is None:
                raise RuntimeError("Model is not loaded.")
            beam_size = int(self.config.get("beam_size", 5))
            segments, _info = self.fw_model.transcribe(
                audio,
                language=language,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
                beam_size=beam_size,
                condition_on_previous_text=False,
                vad_filter=bool(self.config.get("vad_filter", True)),
            )
            return "".join(segment.text for segment in segments)

        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        import torch

        options: dict[str, Any] = {
            "task": "transcribe",
            "language": language,
            "fp16": getattr(self, "_openai_device", "cpu") == "cuda",
            "condition_on_previous_text": False,
            "verbose": False,
        }
        if initial_prompt is not None:
            options["initial_prompt"] = initial_prompt

        with torch.inference_mode():
            result = self.model.transcribe(audio, **options)
        return str(result["text"])

    def _ollama_postprocess(self, text: str) -> str:
        # A custom system prompt (if set) overrides the default cleanup rules.
        system = str(self.config.get("ollama_system_prompt", "")).strip() or OLLAMA_CLEANUP_SYSTEM
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for example_in, example_out in OLLAMA_CLEANUP_SHOTS:
            messages.append({"role": "user", "content": example_in})
            messages.append({"role": "assistant", "content": example_out})
        messages.append({"role": "user", "content": text})

        print(
            f"[whisper-dictation] ollama chat model={self.config.get('ollama_model')} "
            f"input_chars={len(text)}", flush=True,
        )
        cleaned = ollama_chat(
            self.config, messages, temperature=0.1, timeout=120,
            think=bool(self.config.get("ollama_thinking", False)),
        )
        # If the model refuses or returns nothing, fall back to the raw text.
        refusal_hints = ("ich kann", "i cannot", "i'm unable", "tut mir leid", "sorry", "als ki", "as an ai")
        if not cleaned or any(h in cleaned.lower() for h in refusal_hints):
            print("[whisper-dictation] ollama refusal/empty, using raw transcription", flush=True)
            return text

        print(f"[whisper-dictation] ollama cleanup done chars={len(cleaned)}", flush=True)
        return cleaned

    def _ollama_instruct(self, text: str, instruction: str) -> str:
        """Run a free-form user instruction over the text via Ollama."""
        system = (
            "Du bist ein Schreibassistent. Fuehre die Anweisung des Nutzers auf dem "
            "gegebenen Text aus und gib AUSSCHLIESSLICH den ueberarbeiteten Text zurueck "
            "- keine Erklaerungen, keine Vorrede. Behalte englische Fachbegriffe bei."
        )
        user = f"Anweisung: {instruction}\n\nText:\n{text}"
        return ollama_chat(
            self.config,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3, timeout=180,
        )

    # -- IPC server (GUI workbench) ---------------------------------------------

    def _start_ipc_server(self) -> None:
        if IS_MACOS:
            return  # Linux first; macOS could be added later
        import socket as _socket
        try:
            if IPC_SOCKET.exists():
                IPC_SOCKET.unlink()
        except Exception:
            pass
        try:
            srv = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
            srv.bind(str(IPC_SOCKET))
            srv.listen(4)
        except Exception as exc:
            print(f"[whisper-dictation] IPC server failed: {exc}", file=sys.stderr, flush=True)
            return
        self._ipc_srv = srv
        print(f"[whisper-dictation] IPC socket at {IPC_SOCKET}", flush=True)
        threading.Thread(target=self._ipc_accept_loop, args=(srv,), daemon=True).start()

    def _ipc_accept_loop(self, srv: Any) -> None:
        while not self.stopping:
            try:
                conn, _ = srv.accept()
            except OSError:
                break
            threading.Thread(target=self._ipc_handle, args=(conn,), daemon=True).start()

    def _ipc_handle(self, conn: Any) -> None:
        import json as _json
        try:
            conn.settimeout(300)
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            req = _json.loads(data.decode("utf-8") or "{}")
            resp = self._ipc_dispatch(req)
        except Exception as exc:
            resp = {"error": str(exc)}
        try:
            conn.sendall((_json.dumps(resp) + "\n").encode("utf-8"))
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _ipc_dispatch(self, req: dict) -> dict:
        cmd = str(req.get("cmd", ""))
        if cmd == "ping":
            return {"ok": True, "backend": self.backend}
        if cmd == "transcribe":
            wav = req.get("wav")
            if not wav or not Path(wav).exists():
                return {"error": "wav fehlt"}
            audio = read_wav_mono(Path(wav))
            with self._infer_lock:
                text = self._transcribe_audio(audio).strip()
            return {"text": text}
        if cmd == "instruct":
            try:
                out = self._ollama_instruct(str(req.get("text", "")), str(req.get("instruction", "")))
                return {"text": out}
            except Exception as exc:
                return {"error": f"Ollama: {exc}"}
        return {"error": f"unbekannter Befehl {cmd!r}"}

    # -- Paste ------------------------------------------------------------------

    def _paste_text(self, text: str, restore: bool = True) -> bool:
        """Returns True if the text was auto-pasted, False if only copied.

        restore=False keeps the text in the clipboard (used by command mode, so
        a read-only target like a web page still leaves the result available).
        """
        if IS_MACOS:
            return self._paste_macos(text)
        return self._paste_linux(text, restore=restore)

    def _paste_linux(self, text: str, restore: bool = True) -> bool:
        if shutil_which("wl-copy") is None:
            raise RuntimeError("wl-copy ist nicht installiert (sudo dnf install wl-clipboard).")

        # Remember the current clipboard (best effort, text only) to restore it
        # after pasting, so dictation doesn't clobber what the user had copied.
        saved: bytes | None = None
        if restore and self.config.get("restore_clipboard", True) and shutil_which("wl-paste"):
            if clipboard_manager_running():
                print(
                    "[whisper-dictation] clipboard manager running; "
                    "skipping restore (keeps dictation on top of its history)",
                    flush=True,
                )
            else:
                try:
                    saved = subprocess.run(
                        ["wl-paste", "-n"], capture_output=True, timeout=2
                    ).stdout
                except Exception:
                    saved = None

        subprocess.run(["wl-copy"], input=text.encode("utf-8"), check=True)
        time.sleep(0.08)

        paste_mode = self._resolve_paste_mode()
        print(f"[whisper-dictation] paste mode={paste_mode}", flush=True)

        if shutil_which("ydotool") is None:
            return False

        keys = _YDOTOOL_KEYS.get(paste_mode, _YDOTOOL_KEYS["ctrl_v"])
        result = subprocess.run(
            ["ydotool", "key", *keys], check=False, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(
                f"[whisper-dictation] ydotool failed rc={result.returncode}: {result.stderr.strip()}",
                file=sys.stderr, flush=True,
            )
            return False

        if saved is not None:
            # Restore after the paste keystroke has been consumed by the app.
            def _restore(data: bytes) -> None:
                time.sleep(0.4)
                subprocess.run(["wl-copy"], input=data, check=False)
            threading.Thread(target=_restore, args=(saved,), daemon=True).start()
        return True

    def _paste_macos(self, text: str) -> bool:
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        time.sleep(0.15)
        print("[whisper-dictation] paste mode=osascript", flush=True)
        subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to keystroke "v" using command down'],
            check=False, capture_output=True,
        )
        return True

    def _resolve_paste_mode(self) -> str:
        configured = str(self.config["paste_mode"]).lower()
        if configured != "auto":
            return configured
        # On Wayland the focused window class is not exposed, so we cannot
        # auto-switch to the terminal paste shortcut. Default to Ctrl+V.
        return "ctrl_v"

    # -- Status notification (single, in-place) ---------------------------------

    def _status(self, summary: str, body: str = "", urgency: str = "normal",
                timeout_ms: int | None = None) -> None:
        """Update the one dictation status bubble instead of stacking popups.

        urgency="critical" with timeout_ms=0 pins it on screen for the duration
        of the memo; terminal states use "normal" + a short timeout so the
        bubble auto-dismisses.
        """
        if IS_MACOS:
            notify(summary, body)
            return
        self._notify_id = _notify_send_linux(
            summary, body, urgency, timeout_ms, self._notify_id
        )

    # -- Sound feedback ---------------------------------------------------------

    def _play_sound(self, event: str) -> None:
        """Best-effort freedesktop sound cue (tray-less status feedback)."""
        if not self.config.get("sound_cue", True) or IS_MACOS:
            return
        player = shutil_which("canberra-gtk-play")
        if player is None:
            return
        names = {"start": "audio-volume-change", "done": "complete"}
        subprocess.Popen(
            [player, "-i", names.get(event, "bell")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    # -- Live config reload (SIGHUP) --------------------------------------------

    def reload_config(self) -> None:
        """Apply config changes without a full restart.

        Mic, language, VAD, hotwords, paste and beam size are read per use, so
        they take effect on the next dictation. Only a model/backend/device
        change reloads the model; a hotkey change still needs a restart.
        """
        try:
            new = load_config()
        except Exception as exc:
            print(f"[whisper-dictation] reload: cannot read config: {exc}",
                  file=sys.stderr, flush=True)
            return
        old = self.config
        self.config = new
        self.double_tap_window = max(150, int(new.get("double_tap_window_ms", 400))) / 1000.0
        # Mode (double-tap vs push-to-talk) is read live by the evdev loop.
        self.hotkey_mode = str(new.get("hotkey_mode", "double_tap")).lower()

        model_keys = (new.get("model"), new.get("backend"), new.get("ov_device"))
        old_keys = (old.get("model"), old.get("backend"), old.get("ov_device"))
        if model_keys != old_keys:
            print("[whisper-dictation] reloading model after config change", flush=True)
            # _infer_lock: never swap models under a transcription that is
            # still running; the reload waits until inference finishes.
            with self._infer_lock:
                with self.lock:
                    self.backend = self._resolve_backend()
                    self.fw_model = None
                    self.model = None
                    self.ov_pipe = None
                self._load_model()
        elif (new.get("double_tap_key") != old.get("double_tap_key")
              or new.get("llm_toggle_key") != old.get("llm_toggle_key")
              or new.get("command_key") != old.get("command_key")):
            notify("Taste geaendert", "Bitte Daemon neu starten, damit die neue Taste greift.")
        else:
            notify("Einstellungen uebernommen", "Aenderungen sind aktiv.")
        print("[whisper-dictation] config reloaded", flush=True)

    # -- Shutdown ---------------------------------------------------------------

    def shutdown(self) -> None:
        with self.lock:
            self.stopping = True
            if self.recording_timer is not None:
                self.recording_timer.cancel()
                self.recording_timer = None
            if self.recording_process is not None:
                self.recording_process.send_signal(signal.SIGINT)
                self.recording_process = None
            if self.recording_sd_stop is not None:
                self.recording_sd_stop.set()
            if self.listener is not None:
                self.listener.stop()
        srv = getattr(self, "_ipc_srv", None)
        if srv is not None:
            try:
                srv.close()
            except Exception:
                pass
        try:
            IPC_SOCKET.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> int:
    config = load_config()
    daemon = WhisperDictationDaemon(config)

    def handle_signal(signum: int, _frame: Any) -> None:
        print(f"[whisper-dictation] stopping on signal {signum}", flush=True)
        daemon.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # SIGHUP: live-reload config (no model reload unless model/device changed).
    # SIGUSR1: toggle recording (bindable to a GNOME custom shortcut).
    if hasattr(signal, "SIGHUP"):
        signal.signal(
            signal.SIGHUP,
            lambda *_a: threading.Thread(target=daemon.reload_config, daemon=True).start(),
        )
    if hasattr(signal, "SIGUSR1"):
        signal.signal(
            signal.SIGUSR1,
            lambda *_a: threading.Thread(target=daemon.toggle_recording, daemon=True).start(),
        )

    try:
        daemon.run()
        return 0
    except Exception as exc:
        notify("Start fehlgeschlagen", str(exc))
        print(f"[whisper-dictation] startup failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
