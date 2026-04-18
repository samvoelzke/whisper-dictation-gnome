#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import platform
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

IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

try:
    from Xlib import X, display as xdisplay
except Exception:
    X = None
    xdisplay = None

if IS_MACOS:
    try:
        import sounddevice as sd
    except ImportError:
        sd = None  # type: ignore[assignment]
else:
    sd = None  # type: ignore[assignment]

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
    "ollama_postprocess": False,
    "ollama_model": "llama3.2:3b",
    "ollama_host": "http://localhost:11434",
}

# Each entry: (primary_key, fallback_keys_set, label)
# Fallbacks handle systems where pynput reports the generic key instead of left/right variant.
def _hotkey_entry(key_name: str, fallbacks: frozenset, label: str):
    try:
        return (getattr(keyboard.Key, key_name), fallbacks, label)
    except AttributeError:
        return None

HOTKEYS: dict[str, tuple] = {k: v for k, v in {
    "ctrl_r": _hotkey_entry("ctrl_r", frozenset({keyboard.Key.ctrl}), "Right Ctrl"),
    "ctrl_l": _hotkey_entry("ctrl_l", frozenset({keyboard.Key.ctrl}), "Left Ctrl"),
    "alt_r":  _hotkey_entry("alt_r",  frozenset({keyboard.Key.alt}),  "Right Alt"),
    "alt_l":  _hotkey_entry("alt_l",  frozenset({keyboard.Key.alt}),  "Left Alt"),
    "f8":     _hotkey_entry("f8",     frozenset(), "F8"),
    "f9":     _hotkey_entry("f9",     frozenset(), "F9"),
    "f10":    _hotkey_entry("f10",    frozenset(), "F10"),
    "pause":  _hotkey_entry("pause",  frozenset(), "Pause"),
}.items() if v is not None}

TERMINAL_HINTS = (
    "gnome-terminal", "kgx", "tilix", "terminator", "kitty",
    "alacritty", "wezterm", "konsole", "xfce4-terminal",
    "mate-terminal", "lxterminal", "iterm2", "terminal",
)


def notify(summary: str, body: str = "") -> None:
    if IS_MACOS:
        script = f'display notification "{body}" with title "{summary}"'
        subprocess.run(["osascript", "-e", script], check=False, capture_output=True)
        return
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
        f"  Systemeinstellungen → Datenschutz & Sicherheit → Bedienungshilfen\n"
        f"  → + klicken → Cmd+Shift+G → diesen Pfad einfügen:\n"
        f"  {real_python}\n"
        f"  → Schalter aktivieren → Daemon neu starten.\n",
        flush=True,
    )
    notify(
        "Accessibility-Berechtigung fehlt",
        "Systemeinstellungen → Bedienungshilfen → Python eintragen, dann Daemon neu starten.",
    )
    subprocess.run(
        ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"],
        check=False, capture_output=True,
    )
    sys.exit(1)


def load_config() -> dict[str, Any]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        defaults = DEFAULT_CONFIG.copy()
        if IS_MACOS:
            defaults["record_device"] = "default"
            defaults["paste_mode"] = "cmd_v"
        CONFIG_FILE.write_text(
            json.dumps(defaults, indent=2, ensure_ascii=True) + "\n",
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
            f"Unexpected sample rate {sample_rate}. Expected 16000 Hz."
        )

    return audio


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if IS_MACOS and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
        self.device = _best_device()
        self.model: whisper.Whisper | None = None
        self.listener: keyboard.Listener | None = None
        self.controller = keyboard.Controller()
        self.lock = threading.RLock()

        # Linux: arecord subprocess; macOS: sounddevice thread
        self.recording_process: subprocess.Popen[bytes] | None = None
        self.recording_sd_thread: threading.Thread | None = None
        self.recording_sd_stop: threading.Event | None = None
        self.recording_sd_frames: list[np.ndarray] = []

        self.recording_file: Path | None = None
        self.recording_timer: threading.Timer | None = None
        self.last_hotkey_release: float | None = None
        self.busy = False
        self.stopping = False

    # ── Linux: ALSA mic volume ────────────────────────────────────────────────

    def _init_mic_volume(self) -> None:
        if not IS_LINUX:
            return
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
            subprocess.run(
                ["amixer", "-c", card, "cset", "numid=6", "26"],
                check=False, capture_output=True,
            )
            print(f"[whisper-dictation] mic volume set to 26 on card {card}", flush=True)

    # ── Startup ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        if IS_MACOS:
            check_macos_accessibility()
        self._init_mic_volume()
        print(
            f"[whisper-dictation] platform={platform.system()} "
            f"loading model={self.config['model']} device={self.device}",
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

    # ── Hotkey detection ──────────────────────────────────────────────────────

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
        is_recording = (
            self.recording_process is not None or self.recording_sd_thread is not None
        )
        if not is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    # ── Recording: Linux (arecord) ────────────────────────────────────────────

    def _start_recording_linux(self, output_path: Path) -> None:
        if shutil_which("arecord") is None:
            raise RuntimeError("arecord ist nicht installiert (sudo apt install alsa-utils).")

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

    # ── Recording: macOS (sounddevice) ───────────────────────────────────────

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

    # ── Recording: common ─────────────────────────────────────────────────────

    def start_recording(self) -> None:
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
        notify("Aufnahme gestartet", f"Zum Stoppen wieder doppelt {self.hotkey_label} drücken")

    def auto_stop_recording(self) -> None:
        with self.lock:
            is_recording = (
                self.recording_process is not None or self.recording_sd_thread is not None
            )
            if not is_recording or self.busy:
                return
            notify("Aufnahme wird beendet", "Maximale Aufnahmedauer erreicht.")
            self.stop_recording()

    def stop_recording(self) -> None:
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
        notify("Transkription läuft", "Die Aufnahme wird gerade erkannt und eingefügt.")

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
                notify("Kein Text erkannt", "Die Aufnahme war leer.")
                return

            audio = read_wav_mono(output_path)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            print(f"[whisper-dictation] audio rms={rms:.5f}", flush=True)
            if rms < 0.002:
                notify("Kein Text erkannt", "Die Aufnahme war leer oder zu leise.")
                return

            text = self._transcribe_audio(audio).strip()
            if not text:
                notify("Kein Text erkannt", "Nichts verstanden.")
                return

            print(f"[whisper-dictation] transcription ready chars={len(text)}", flush=True)

            if self.config.get("ollama_postprocess"):
                notify("Verfeinere Text...", "Ollama läuft...")
                text = self._ollama_postprocess(text)
                print(f"[whisper-dictation] ollama postprocess done chars={len(text)}", flush=True)

            self._paste_text(text)
            notify("Eingefügt", text[:100])
        except Exception as exc:
            notify("Fehler", str(exc))
            print(f"[whisper-dictation] {exc}", file=sys.stderr, flush=True)
        finally:
            self.busy = False
            output_path.unlink(missing_ok=True)

    # ── Transcription ─────────────────────────────────────────────────────────

    def _transcribe_audio(self, audio: np.ndarray) -> str:
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

    def _ollama_postprocess(self, text: str) -> str:
        import urllib.request, json as _json
        host = str(self.config.get("ollama_host", "http://localhost:11434")).rstrip("/")
        model = str(self.config.get("ollama_model", "llama3.2:3b"))
        language = str(self.config.get("language") or "de")

        system = (
            "Du bist ein Assistent der diktierten Text korrigiert. "
            "Füge korrekte Satzzeichen, Groß-/Kleinschreibung und Absätze ein. "
            "Korrigiere offensichtliche Erkennungsfehler. "
            "Gib NUR den korrigierten Text zurück — keine Erklärungen, keine Anführungszeichen."
        ) if language == "de" else (
            "You are an assistant that cleans up dictated text. "
            "Add correct punctuation, capitalization, and paragraph breaks. "
            "Fix obvious speech recognition errors. "
            "Return ONLY the corrected text — no explanations, no quotes."
        )

        payload = _json.dumps({
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
        }).encode()

        try:
            req = urllib.request.Request(
                f"{host}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read())
                return str(data["message"]["content"]).strip() or text
        except Exception as exc:
            print(f"[whisper-dictation] ollama error (fallback to raw): {exc}", file=sys.stderr, flush=True)
            return text

    # ── Paste ─────────────────────────────────────────────────────────────────

    def _paste_text(self, text: str) -> None:
        if IS_MACOS:
            self._paste_macos(text)
        else:
            self._paste_linux(text)

    def _paste_linux(self, text: str) -> None:
        if shutil_which("xclip") is None:
            raise RuntimeError("xclip ist nicht installiert (sudo apt install xclip).")

        paste_mode = self._resolve_paste_mode()
        print(f"[whisper-dictation] paste mode={paste_mode}", flush=True)
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode("utf-8"),
            check=True,
        )
        time.sleep(0.08)
        self._send_paste_shortcut(paste_mode)

    def _paste_macos(self, text: str) -> None:
        subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            check=True,
        )
        time.sleep(0.08)
        paste_mode = self._resolve_paste_mode()
        print(f"[whisper-dictation] paste mode={paste_mode}", flush=True)
        self._send_paste_shortcut(paste_mode)

    def _send_paste_shortcut(self, paste_mode: str) -> None:
        time.sleep(0.02)
        if paste_mode == "cmd_v":
            with self.controller.pressed(keyboard.Key.cmd):
                self.controller.tap("v")
        elif paste_mode == "ctrl_shift_v":
            with self.controller.pressed(keyboard.Key.ctrl):
                with self.controller.pressed(keyboard.Key.shift):
                    self.controller.tap("v")
        elif paste_mode == "shift_insert":
            with self.controller.pressed(keyboard.Key.shift):
                self.controller.tap(keyboard.Key.insert)
        else:
            with self.controller.pressed(keyboard.Key.ctrl):
                self.controller.tap("v")

    def _resolve_paste_mode(self) -> str:
        configured = str(self.config["paste_mode"]).lower()
        if configured != "auto":
            return configured

        if IS_MACOS:
            return "cmd_v"

        window_class = self._get_active_window_class()
        if not window_class:
            return "ctrl_v"
        if "xterm" in window_class or "uxterm" in window_class:
            return "shift_insert"
        if any(hint in window_class for hint in TERMINAL_HINTS):
            return "ctrl_shift_v"
        return "ctrl_v"

    def _get_active_window_class(self) -> str | None:
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

    # ── Shutdown ──────────────────────────────────────────────────────────────

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
