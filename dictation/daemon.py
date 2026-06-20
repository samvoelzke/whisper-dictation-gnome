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

import numpy as np

# hf-xet transfers can stall on some networks; force plain HTTPS downloads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHISPER_REPO_ROOT = PROJECT_ROOT / "whisper"

IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# macOS records via sounddevice; Linux records via arecord (PipeWire/ALSA).
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
    # "auto": openvino on Linux when available, else faster-whisper; openai on macOS.
    "backend": "auto",
    # OpenVINO device preference: AUTO picks NPU > GPU > CPU.
    "ov_device": "AUTO",
    "beam_size": 5,
    # VAD trims silence before transcription: fewer hallucinations + less compute.
    "vad_filter": True,
    # Comma-separated domain words to bias recognition (names, jargon).
    "hotwords": "",
    # Audible feedback on record start / text inserted (no tray on GNOME).
    "sound_cue": True,
    "paste_mode": "auto",
    "record_device": "default",
    "max_record_seconds": 180,
    "initial_prompt": "",
    "ollama_postprocess": False,
    "ollama_model": "llama3.2:3b",
    "ollama_host": "http://localhost:11434",
}

# Hotkey spec per logical name:
#   (label, pynput_attr, pynput_fallback_attrs, evdev_ecode_name)
# pynput_* are used on macOS, evdev_ecode_name on Linux/Wayland.
HOTKEY_SPECS: dict[str, tuple[str, str, list[str], str]] = {
    "ctrl_r": ("Right Ctrl", "ctrl_r", ["ctrl"], "KEY_RIGHTCTRL"),
    "ctrl_l": ("Left Ctrl", "ctrl_l", ["ctrl"], "KEY_LEFTCTRL"),
    "alt_r": ("Right Alt", "alt_r", ["alt"], "KEY_RIGHTALT"),
    "alt_l": ("Left Alt", "alt_l", ["alt"], "KEY_LEFTALT"),
    "f8": ("F8", "f8", [], "KEY_F8"),
    "f9": ("F9", "f9", [], "KEY_F9"),
    "f10": ("F10", "f10", [], "KEY_F10"),
    "pause": ("Pause", "pause", [], "KEY_PAUSE"),
}

# faster-whisper model id mapping (openai short names -> CTranslate2 ids).
FASTER_MODEL_MAP = {
    "turbo": "large-v3-turbo",
}

# Official, openvino-genai-2026-compatible pre-converted Whisper models.
# (Community exports often lack the `beam_idx` input and fail to load.)
# Only models listed here can use the OpenVINO backend; others fall back to
# faster-whisper. No torch/optimum needed at runtime.
OV_MODEL_REPOS = {
    "turbo": "OpenVINO/whisper-large-v3-turbo-fp16-ov",
    "large-v3-turbo": "OpenVINO/whisper-large-v3-turbo-fp16-ov",
    "large-v3": "OpenVINO/whisper-large-v3-int8-ov",
    "distil-large-v3": "OpenVINO/distil-whisper-large-v3-int8-ov",
}

TERMINAL_HINTS = (
    "gnome-terminal", "kgx", "console", "tilix", "terminator", "kitty",
    "alacritty", "wezterm", "konsole", "xfce4-terminal",
    "mate-terminal", "lxterminal", "iterm2", "terminal",
)

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


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        try:
            import ctranslate2
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False


class WhisperDictationDaemon:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        hotkey_name = str(config["double_tap_key"]).lower()
        if hotkey_name not in HOTKEY_SPECS:
            valid = ", ".join(sorted(HOTKEY_SPECS))
            raise RuntimeError(
                f"Unsupported hotkey '{hotkey_name}'. Valid values: {valid}"
            )

        self.hotkey_name = hotkey_name
        self.hotkey_label = HOTKEY_SPECS[hotkey_name][0]
        self.double_tap_window = max(150, int(config["double_tap_window_ms"])) / 1000.0
        self.backend = self._resolve_backend()

        # Transcription models (only one is populated, depending on backend).
        self.fw_model: Any = None
        self.model: Any = None
        self.ov_pipe: Any = None

        self.lock = threading.RLock()

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

    # -- Startup ----------------------------------------------------------------

    def run(self) -> None:
        if IS_MACOS:
            check_macos_accessibility()
        self._init_mic_volume()
        self._load_model()

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
            if _cuda_available():
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

            device = "cuda" if _cuda_available() else (
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
        import openvino_genai as ov_genai
        from huggingface_hub import snapshot_download

        repo = OV_MODEL_REPOS[model_name]
        device = self._resolve_ov_device()
        print(f"[whisper-dictation] OpenVINO model={repo} device={device}", flush=True)
        notify("Lade Modell", f"OpenVINO {device}: {model_name}")
        model_dir = snapshot_download(
            repo,
            local_dir=str(CACHE_DIR / "ov-models" / repo.replace("/", "__")),
        )
        kwargs: dict[str, Any] = {}
        if device == "NPU":
            # NPU needs a static-shape pipeline for Whisper.
            kwargs["STATIC_PIPELINE"] = True
        self.ov_pipe = ov_genai.WhisperPipeline(model_dir, device, **kwargs)
        self._ov_device = device

    def _resolve_ov_device(self) -> str:
        import openvino as ov

        available = ov.Core().available_devices
        want = str(self.config.get("ov_device", "AUTO")).upper()
        if want != "AUTO" and want in available:
            return want
        for preferred in ("NPU", "GPU", "CPU"):
            if preferred in available:
                return preferred
        return "CPU"

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

        _, pynput_attr, fallback_attrs, _ = HOTKEY_SPECS[self.hotkey_name]
        self._mac_hotkey = getattr(keyboard.Key, pynput_attr)
        self._mac_fallbacks = frozenset(
            getattr(keyboard.Key, name) for name in fallback_attrs
        )

        def is_hotkey(key: Any) -> bool:
            return key == self._mac_hotkey or key in self._mac_fallbacks

        def on_press(key: Any) -> None:
            if not is_hotkey(key):
                with self.lock:
                    self.last_hotkey_release = None

        def on_release(key: Any) -> None:
            if is_hotkey(key):
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

        target = getattr(ecodes, HOTKEY_SPECS[self.hotkey_name][3])

        devices: list[Any] = []
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
            except Exception:
                continue
            caps = dev.capabilities()
            if ecodes.EV_KEY in caps and target in caps[ecodes.EV_KEY]:
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
        print(
            f"[whisper-dictation] evdev listener on {len(devices)} device(s): {labels}",
            flush=True,
        )

        self.listener_thread = threading.Thread(
            target=self._evdev_loop, args=(target, ecodes), daemon=True
        )
        self.listener_thread.start()
        self.listener_thread.join()

    def _evdev_loop(self, target: int, ecodes: Any) -> None:
        import selectors

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
                        if event.code == target:
                            if event.value == 0:  # release
                                self._register_release()
                        elif event.value == 1:  # other key pressed -> reset
                            with self.lock:
                                self.last_hotkey_release = None
                except OSError:
                    # Device went away (unplugged); stop watching it.
                    try:
                        selector.unregister(dev)
                    except Exception:
                        pass

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

    def toggle_recording(self) -> None:
        if self.busy:
            notify("Noch beschaeftigt", "Bitte warten, die letzte Aufnahme wird noch verarbeitet.")
            return
        is_recording = (
            self.recording_process is not None or self.recording_sd_thread is not None
        )
        if not is_recording:
            self.start_recording()
        else:
            self.stop_recording()

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
        notify("Aufnahme gestartet", f"Zum Stoppen wieder doppelt {self.hotkey_label} druecken")
        self._play_sound("start")

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
        notify("Transkription laeuft", "Die Aufnahme wird gerade erkannt und eingefuegt.")

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
                notify("Verfeinere Text...", "Ollama laeuft...")
                try:
                    text = self._ollama_postprocess(text)
                except Exception as exc:
                    print(f"[whisper-dictation] ollama failed, using raw: {exc}", file=sys.stderr, flush=True)

            self._paste_text(text)
            notify("Eingefuegt", text[:100])
            self._play_sound("done")
        except Exception as exc:
            notify("Fehler", str(exc))
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
        import urllib.request, json as _json, re as _re
        host = str(self.config.get("ollama_host", "http://localhost:11434")).rstrip("/")
        model = str(self.config.get("ollama_model", "llama3.2:3b"))

        custom_prompt = str(self.config.get("ollama_system_prompt", "")).strip()

        if custom_prompt:
            completion_prompt = f"{custom_prompt}\n\nEingabe: {text}\nAusgabe:"
        else:
            # Few-shot format: Modell sieht sich als reines Textwerkzeug, kein Gespraechspartner.
            # Bewusst kein "system"-Feld - llama3.2 triggert Safety-Filter seltener im completion-Modus.
            completion_prompt = (
                "AUFGABE: Zeichensetzung, Gross-/Kleinschreibung und Grammatik korrigieren.\n"
                "REGELN: Nur Korrekturen ausgeben. Niemals auf den Inhalt eingehen oder antworten.\n\n"
                "###\n"
                "Eingabe: wo ist die fernbedienung\n"
                "Ausgabe: Wo ist die Fernbedienung?\n"
                "###\n"
                "Eingabe: das meeting ist morgen um drei uhr\n"
                "Ausgabe: Das Meeting ist morgen um drei Uhr.\n"
                "###\n"
                "Eingabe: kannst du mir sagen wie das geht\n"
                "Ausgabe: Kannst du mir sagen, wie das geht?\n"
                "###\n"
                f"Eingabe: {text}\n"
                "Ausgabe:"
            )

        thinking = bool(self.config.get("ollama_thinking", False))
        payload = {
            "model": model,
            "prompt": completion_prompt,
            "stream": False,
            "options": {"temperature": 0.1, "stop": ["###", "\nEingabe:", "\n\nEingabe:"]},
            "think": thinking,
        }

        print(f"[whisper-dictation] ollama request model={model} input_chars={len(text)}", flush=True)
        req = urllib.request.Request(
            f"{host}/api/generate",
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            response_body = resp.read()

        parsed = _json.loads(response_body)
        raw = str(parsed.get("response", ""))
        print(f"[whisper-dictation] ollama raw response: {raw!r}", flush=True)

        cleaned = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        # Verweigert das Modell, faellt es auf den Original-Text zurueck
        refusal_hints = ("ich kann", "i cannot", "i'm unable", "tut mir leid", "sorry", "als ki", "as an ai")
        if not cleaned or any(h in cleaned.lower() for h in refusal_hints):
            print(f"[whisper-dictation] ollama refusal detected, using raw transcription", flush=True)
            return text

        print(f"[whisper-dictation] ollama postprocess done chars={len(cleaned)}", flush=True)
        return cleaned

    # -- Paste ------------------------------------------------------------------

    def _paste_text(self, text: str) -> None:
        if IS_MACOS:
            self._paste_macos(text)
        else:
            self._paste_linux(text)

    def _paste_linux(self, text: str) -> None:
        if shutil_which("wl-copy") is None:
            raise RuntimeError("wl-copy ist nicht installiert (sudo dnf install wl-clipboard).")

        subprocess.run(["wl-copy"], input=text.encode("utf-8"), check=True)
        time.sleep(0.08)

        paste_mode = self._resolve_paste_mode()
        print(f"[whisper-dictation] paste mode={paste_mode}", flush=True)

        if shutil_which("ydotool") is None:
            notify(
                "Text in der Zwischenablage",
                "ydotool fehlt - bitte manuell mit Strg+V einfuegen.",
            )
            return

        keys = _YDOTOOL_KEYS.get(paste_mode, _YDOTOOL_KEYS["ctrl_v"])
        result = subprocess.run(
            ["ydotool", "key", *keys], check=False, capture_output=True, text=True
        )
        if result.returncode != 0:
            notify(
                "Text in der Zwischenablage",
                "ydotool-Inject fehlgeschlagen - bitte manuell mit Strg+V einfuegen.",
            )
            print(
                f"[whisper-dictation] ydotool failed rc={result.returncode}: {result.stderr.strip()}",
                file=sys.stderr, flush=True,
            )

    def _paste_macos(self, text: str) -> None:
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        time.sleep(0.15)
        print("[whisper-dictation] paste mode=osascript", flush=True)
        subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to keystroke "v" using command down'],
            check=False, capture_output=True,
        )

    def _resolve_paste_mode(self) -> str:
        configured = str(self.config["paste_mode"]).lower()
        if configured != "auto":
            return configured
        # On Wayland the focused window class is not exposed, so we cannot
        # auto-switch to the terminal paste shortcut. Default to Ctrl+V.
        return "ctrl_v"

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

        model_keys = (new.get("model"), new.get("backend"), new.get("ov_device"))
        old_keys = (old.get("model"), old.get("backend"), old.get("ov_device"))
        if model_keys != old_keys:
            print("[whisper-dictation] reloading model after config change", flush=True)
            with self.lock:
                self.backend = self._resolve_backend()
                self.fw_model = None
                self.model = None
                self.ov_pipe = None
            self._load_model()
        elif new.get("double_tap_key") != old.get("double_tap_key"):
            notify("Hotkey geaendert", "Bitte Daemon neu starten, damit die neue Taste greift.")
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
