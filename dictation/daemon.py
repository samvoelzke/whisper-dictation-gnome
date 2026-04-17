#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHISPER_REPO_ROOT = PROJECT_ROOT / "whisper"
if str(WHISPER_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(WHISPER_REPO_ROOT))

import numpy as np
import torch
import whisper
from pynput import keyboard

try:
    from Xlib import X, display as xdisplay
except Exception:  # pragma: no cover - optional runtime dependency
    X = None
    xdisplay = None

CONFIG_DIR = Path.home() / ".config" / "whisper-dictation"
CONFIG_FILE = CONFIG_DIR / "config.json"
CACHE_DIR = Path.home() / ".cache" / "whisper-dictation"

DEFAULT_CONFIG: dict[str, Any] = {
    "double_tap_key": "ctrl_r",
    "double_tap_window_ms": 400,
    "language": "de",
    "model": "turbo",
    "paste_mode": "auto",
    "record_device": "default",
    "max_record_seconds": 180,
    "initial_prompt": "",
}

# Each entry: (primary_key, fallback_keys_set, label)
# Fallbacks handle systems where pynput reports the generic key instead of left/right variant.
HOTKEYS: dict[str, tuple[keyboard.Key, frozenset[keyboard.Key], str]] = {
    "ctrl_r": (keyboard.Key.ctrl_r, frozenset({keyboard.Key.ctrl}), "Right Ctrl"),
    "ctrl_l": (keyboard.Key.ctrl_l, frozenset({keyboard.Key.ctrl}), "Left Ctrl"),
    "alt_r": (keyboard.Key.alt_r, frozenset({keyboard.Key.alt}), "Right Alt"),
    "alt_l": (keyboard.Key.alt_l, frozenset({keyboard.Key.alt}), "Left Alt"),
    "f8": (keyboard.Key.f8, frozenset(), "F8"),
    "f9": (keyboard.Key.f9, frozenset(), "F9"),
    "f10": (keyboard.Key.f10, frozenset(), "F10"),
    "pause": (keyboard.Key.pause, frozenset(), "Pause"),
}

TERMINAL_HINTS = (
    "gnome-terminal",
    "kgx",
    "tilix",
    "terminator",
    "kitty",
    "alacritty",
    "wezterm",
    "konsole",
    "xfce4-terminal",
    "mate-terminal",
    "lxterminal",
)


def notify(summary: str, body: str = "") -> None:
    if shutil_which("notify-send") is None:
        return
    command = ["notify-send", "-a", "Whisper Dictation", summary]
    if body:
        command.append(body)
    subprocess.run(command, check=False)


def shutil_which(binary: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / binary
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def load_config() -> dict[str, Any]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(
            json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    loaded = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    config = DEFAULT_CONFIG.copy()
    config.update(loaded)
    return config


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
            f"Unexpected sample rate {sample_rate}. Expected 16000 Hz from arecord."
        )

    return audio


class WhisperDictationDaemon:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        hotkey_name = str(config["double_tap_key"]).lower()
        if hotkey_name not in HOTKEYS:
            valid = ", ".join(sorted(HOTKEYS))
            raise RuntimeError(
                f"Unsupported hotkey '{hotkey_name}'. Valid values: {valid}"
            )

        self.hotkey_name = hotkey_name
        self.hotkey, self.hotkey_fallbacks, self.hotkey_label = HOTKEYS[hotkey_name]
        self.double_tap_window = max(150, int(config["double_tap_window_ms"])) / 1000.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: whisper.Whisper | None = None
        self.listener: keyboard.Listener | None = None
        self.controller = keyboard.Controller()
        self.lock = threading.RLock()
        self.recording_process: subprocess.Popen[bytes] | None = None
        self.recording_file: Path | None = None
        self.recording_timer: threading.Timer | None = None
        self.last_hotkey_release: float | None = None
        self.busy = False
        self.stopping = False

    def _init_mic_volume(self) -> None:
        device = str(self.config.get("record_device", "default"))
        import re as _re
        m = _re.match(r"(?:plug)?hw:(\d+)", device)
        if not m:
            return
        card = m.group(1)
        result = subprocess.run(
            ["amixer", "-c", card, "cget", "numid=6"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            return
        current_line = next((l for l in result.stdout.splitlines() if ": values=" in l), "")
        try:
            current_vol = int(current_line.split("values=")[1].split()[0])
        except (IndexError, ValueError):
            current_vol = -1
        if current_vol < 20:
            subprocess.run(["amixer", "-c", card, "cset", "numid=6", "26"], check=False, capture_output=True)
            print(f"[whisper-dictation] mic volume set to 26 on card {card}", flush=True)

    def run(self) -> None:
        self._init_mic_volume()
        print(
            f"[whisper-dictation] loading model={self.config['model']} device={self.device}",
            flush=True,
        )
        notify("Lade Modell", f"{self.config['model']} auf {self.device}")
        self.model = whisper.load_model(
            str(self.config["model"]),
            device=self.device,
            download_root=str(CACHE_DIR / "models"),
        )
        print("[whisper-dictation] model ready", flush=True)
        notify("Bereit", f"Doppelt {self.hotkey_label} startet oder stoppt die Aufnahme")

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )
        print("[whisper-dictation] listener started", flush=True)
        self.listener.start()
        self.listener.join()

    def _is_hotkey(self, key: keyboard.Key | keyboard.KeyCode | None) -> bool:
        return key == self.hotkey or key in self.hotkey_fallbacks

    def on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if not self._is_hotkey(key):
            with self.lock:
                self.last_hotkey_release = None

    def on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if not self._is_hotkey(key):
            return

        now = time.monotonic()
        with self.lock:
            if self.last_hotkey_release is not None:
                delta = now - self.last_hotkey_release
                self.last_hotkey_release = None
                if delta <= self.double_tap_window:
                    self.toggle_recording()
                    return

            self.last_hotkey_release = now

    def toggle_recording(self) -> None:
        if self.busy:
            notify("Noch beschäftigt", "Bitte warten, die letzte Aufnahme wird noch verarbeitet.")
            return
        if self.recording_process is None:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self) -> None:
        if shutil_which("arecord") is None:
            raise RuntimeError("arecord is not installed.")

        handle = tempfile.NamedTemporaryFile(
            prefix="whisper-dictation-",
            suffix=".wav",
            delete=False,
        )
        handle.close()
        output_path = Path(handle.name)

        command = [
            "arecord",
            "-q",
            "-D",
            str(self.config["record_device"]),
            "-f",
            "S16_LE",
            "-r",
            "16000",
            "-c",
            "1",
            "-t",
            "wav",
            str(output_path),
        ]

        self.recording_process = subprocess.Popen(command)
        self.recording_file = output_path
        print(f"[whisper-dictation] recording started file={output_path}", flush=True)
        self.recording_timer = threading.Timer(
            int(self.config["max_record_seconds"]),
            self.auto_stop_recording,
        )
        self.recording_timer.daemon = True
        self.recording_timer.start()
        notify("Aufnahme gestartet", f"Zum Stoppen wieder doppelt {self.hotkey_label} drücken")

    def auto_stop_recording(self) -> None:
        with self.lock:
            if self.recording_process is None or self.busy:
                return
            notify("Aufnahme wird beendet", "Maximale Aufnahmedauer erreicht.")
            self.stop_recording()

    def stop_recording(self) -> None:
        if self.recording_process is None or self.recording_file is None:
            return

        process = self.recording_process
        output_path = self.recording_file
        self.recording_process = None
        self.recording_file = None
        self.busy = True
        print(f"[whisper-dictation] recording stopped file={output_path}", flush=True)

        if self.recording_timer is not None:
            self.recording_timer.cancel()
            self.recording_timer = None

        process.send_signal(signal.SIGINT)
        worker = threading.Thread(
            target=self.transcribe_and_paste,
            args=(process, output_path),
            daemon=True,
        )
        worker.start()
        notify("Transkription läuft", "Die Aufnahme wird gerade erkannt und eingefügt.")

    def transcribe_and_paste(
        self,
        process: subprocess.Popen[bytes],
        output_path: Path,
    ) -> None:
        try:
            process.wait(timeout=5)
            audio = read_wav_mono(output_path)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            print(f"[whisper-dictation] audio rms={rms:.5f}", flush=True)
            if rms < 0.002:
                notify("Kein Text erkannt", "Die Aufnahme war leer oder zu leise.")
                return
            text = self.transcribe_audio(audio).strip()
            if not text:
                notify("Kein Text erkannt", "Die Aufnahme war leer oder zu leise.")
                return

            print(f"[whisper-dictation] transcription ready chars={len(text)}", flush=True)
            self.paste_text(text)
            notify("Eingefügt", text[:100])
        except Exception as exc:
            notify("Fehler", str(exc))
            print(f"[whisper-dictation] {exc}", file=sys.stderr, flush=True)
        finally:
            self.busy = False
            output_path.unlink(missing_ok=True)

    def transcribe_audio(self, audio: np.ndarray) -> str:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        language = str(self.config.get("language") or "").strip() or None
        initial_prompt = str(self.config.get("initial_prompt") or "").strip() or None
        options: dict[str, Any] = {
            "task": "transcribe",
            "language": language,
            "fp16": self.device == "cuda",
            "condition_on_previous_text": False,
            "verbose": False,
        }
        if initial_prompt is not None:
            options["initial_prompt"] = initial_prompt

        with torch.inference_mode():
            result = self.model.transcribe(audio, **options)
        return str(result["text"])

    def paste_text(self, text: str) -> None:
        if shutil_which("xclip") is None:
            raise RuntimeError("xclip is not installed.")

        print(f"[whisper-dictation] paste mode={self.resolve_paste_mode()}", flush=True)
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode("utf-8"),
            check=True,
        )
        time.sleep(0.08)
        self.send_paste_shortcut()

    def send_paste_shortcut(self) -> None:
        paste_mode = self.resolve_paste_mode()
        time.sleep(0.02)
        if paste_mode == "ctrl_shift_v":
            with self.controller.pressed(keyboard.Key.ctrl):
                with self.controller.pressed(keyboard.Key.shift):
                    self.controller.tap("v")
            return

        if paste_mode == "shift_insert":
            with self.controller.pressed(keyboard.Key.shift):
                self.controller.tap(keyboard.Key.insert)
            return

        with self.controller.pressed(keyboard.Key.ctrl):
            self.controller.tap("v")

    def resolve_paste_mode(self) -> str:
        configured = str(self.config["paste_mode"]).lower()
        if configured != "auto":
            return configured

        window_class = self.get_active_window_class()
        if not window_class:
            return "ctrl_v"
        if "xterm" in window_class or "uxterm" in window_class:
            return "shift_insert"
        if any(hint in window_class for hint in TERMINAL_HINTS):
            return "ctrl_shift_v"
        return "ctrl_v"

    def get_active_window_class(self) -> str | None:
        if xdisplay is None or X is None:
            return None

        display = xdisplay.Display()
        try:
            root = display.screen().root
            active_window_atom = display.intern_atom("_NET_ACTIVE_WINDOW")
            prop = root.get_full_property(active_window_atom, X.AnyPropertyType)
            if prop is None or not prop.value:
                return None

            window = display.create_resource_object("window", int(prop.value[0]))
            wm_class = window.get_wm_class()
            if not wm_class:
                return None
            return " ".join(part.lower() for part in wm_class if part)
        finally:
            display.close()

    def shutdown(self) -> None:
        with self.lock:
            self.stopping = True
            if self.recording_timer is not None:
                self.recording_timer.cancel()
                self.recording_timer = None
            if self.recording_process is not None:
                self.recording_process.send_signal(signal.SIGINT)
                self.recording_process = None
            if self.listener is not None:
                self.listener.stop()


def main() -> int:
    config = load_config()
    daemon = WhisperDictationDaemon(config)

    def handle_signal(signum: int, _frame: Any) -> None:
        print(f"[whisper-dictation] stopping on signal {signum}", flush=True)
        daemon.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        daemon.run()
        return 0
    except Exception as exc:
        notify("Start fehlgeschlagen", str(exc))
        print(f"[whisper-dictation] startup failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
