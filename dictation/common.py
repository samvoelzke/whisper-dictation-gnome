"""Shared core for Whisper Dictation.

Single source of truth for everything the daemon, the long-form recorder and
the GTK settings app all need: config schema + robust load/save, hotkey
specs, model tables, the Ollama chat client and the OpenVINO pipeline loader.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

CONFIG_DIR = Path.home() / ".config" / "whisper-dictation"
CONFIG_FILE = CONFIG_DIR / "config.json"
CACHE_DIR = Path.home() / ".cache" / "whisper-dictation"
# Append-only history of dictations (JSON lines), read by the GUI "Verlauf" tab.
HISTORY_FILE = CACHE_DIR / "history.jsonl"
RECORDINGS_DIR = Path.home() / ".local" / "share" / "whisper-dictation" / "recordings"
LOG_FILE = CACHE_DIR / "daemon.log"


def _ipc_socket_path() -> Path:
    """Unix socket for GUI <-> daemon IPC.

    XDG_RUNTIME_DIR is per-user and 0700. Without it, fall back to a private
    per-uid directory instead of world-writable /tmp directly, so no other
    local user can pre-create/hijack the socket path.
    """
    run_dir = os.environ.get("XDG_RUNTIME_DIR")
    if run_dir:
        return Path(run_dir) / "whisper-dictation.sock"
    fallback = Path(tempfile.gettempdir()) / f"whisper-dictation-{os.getuid()}"
    fallback.mkdir(mode=0o700, exist_ok=True)
    try:
        fallback.chmod(0o700)
    except OSError:
        pass
    return fallback / "whisper-dictation.sock"


IPC_SOCKET = _ipc_socket_path()

DEFAULT_CONFIG: dict[str, Any] = {
    "double_tap_key": "ctrl_r",
    # "double_tap": double-tap to start/stop. "push_to_talk": hold to record.
    "hotkey_mode": "double_tap",
    "double_tap_window_ms": 400,
    "language": "de",
    "model": "turbo",
    # "auto": openvino on Linux when available, else faster-whisper; openai on macOS.
    "backend": "auto",
    # OpenVINO device: AUTO prefers GPU > NPU > CPU (GPU is fastest + reliable
    # for dictation; NPU is opt-in via "NPU" and falls back to GPU if it can't
    # compile the Whisper graph).
    "ov_device": "AUTO",
    "beam_size": 5,
    # VAD trims silence before transcription: fewer hallucinations + less compute.
    "vad_filter": True,
    # Comma-separated domain words to bias recognition (names, jargon).
    "hotwords": "",
    # Audible feedback on record start / text inserted (no tray on GNOME).
    "sound_cue": True,
    # Replace spoken formatting commands ("neue Zeile" -> newline). Off by default.
    "voice_commands": False,
    # Save & restore the clipboard around paste so dictation doesn't clobber it.
    # Skipped automatically while a clipboard manager (Vicinae, ...) is running.
    "restore_clipboard": True,
    # Keep a history of dictations (for the GUI "Verlauf" tab).
    "save_history": True,
    "paste_mode": "auto",
    "record_device": "default",
    "max_record_seconds": 180,
    "initial_prompt": (
        "Diktat auf Deutsch, teils mit englischen Fachbegriffen wie Pull Request, "
        "Deployment, Bug, Backend, Repository, Meeting."
    ),
    "ollama_postprocess": False,
    "ollama_model": "qwen2.5:7b",
    "ollama_host": "http://localhost:11434",
    # Optional override for the built-in cleanup system prompt ("" = default).
    "ollama_system_prompt": "",
    # Let thinking-capable Ollama models think before answering (slower).
    "ollama_thinking": False,
    # How long Ollama keeps the model in RAM after a request ("0" = unload
    # immediately, costs a few seconds reload on the next request; the model
    # is ~5 GB resident while warm).
    "ollama_keep_alive": "10m",
    # Double-tap this key to toggle Ollama cleanup on/off ("" = disabled).
    # Must differ from double_tap_key. Same value set as double_tap_key.
    "llm_toggle_key": "",
    # Command mode: double-tap this key, speak an instruction, and the AI
    # rewrites the currently selected text in place ("" = disabled).
    "command_key": "",
    # Long-form recorder (lectures/calls) — separate from live dictation.
    # Default audio source: "both" (mic + system) | "system" | "mic".
    "recorder_source": "both",
    # Whisper model for recordings (large-v3 = most accurate, slower).
    "recorder_model": "large-v3",
    # Transcription chunk length in seconds (crash-safe partial saves).
    "recorder_chunk_seconds": 300,
    # Recorder language ("" = fall back to the global `language`).
    "recorder_language": "",
    # Preferred capture devices for the recorder ("" = system default).
    "recorder_mic_device": "",
    "recorder_monitor_device": "",
    # GUI live audio visualization: "waves" | "bar" | "none".
    "audio_visualizer": "waves",
    # Opus bitrate; 32k mono is plenty for speech and tiny for hour-long files.
    "recorder_bitrate": "32k",
    # When on, stopping a recording auto-runs transcription (+ summary if a
    # focus prompt is set in the GUI).
    "recorder_auto_process": False,
    # After transcription, let Ollama suggest a short title when the
    # recording still has the default one.
    "recorder_auto_title": True,
    # Obsidian vault folder for exporting transcripts/notes ("" = ask once).
    "obsidian_vault": "",
    # Personal dictionary: names/jargon (list of strings). Biases recognition
    # via hotwords + initial prompt so custom terms are spelled correctly.
    "dictionary": [],
    # Hard post-transcription corrections: {"wrong": "right"} (whole words,
    # case-insensitive). For terms Whisper keeps getting wrong.
    "replacements": {},
    # Voice snippets: speak exactly the trigger phrase -> the stored text is
    # inserted verbatim ({"trigger": "text", ...}).
    "snippets": {},
    # Dictation mode: "standard" (cleanup follows ollama_postprocess),
    # "email" (formal, LLM always on), "chat" (casual, LLM always on),
    # "raw" (never run the LLM).
    "dictation_mode": "standard",
}

# Dictation modes: label + LLM system prompt. None = use the regular cleanup
# pipeline (respects ollama_postprocess); "" = never post-process.
DICTATION_MODES: dict[str, tuple[str, str | None]] = {
    "standard": ("Standard", None),
    "email": (
        "E-Mail — formell",
        "Du bist ein Korrektur-Werkzeug fuer diktierten Text. Forme den Text in eine "
        "formelle, professionelle Formulierung um (E-Mail-Ton): hoeflich, klar, "
        "vollstaendige Saetze, korrekte Zeichensetzung. Entferne Fuellwoerter und "
        "wende gesprochene Selbstkorrekturen an. Erfinde KEINE Inhalte, keine Anrede "
        "und keine Grussformel dazu. Englische Fachbegriffe bleiben Englisch. "
        "Antworte NIEMALS auf den Inhalt. Gib AUSSCHLIESSLICH den Text zurueck.",
    ),
    "chat": (
        "Chat — locker",
        "Du bist ein Korrektur-Werkzeug fuer diktierten Text. Forme den Text in eine "
        "natuerliche, lockere Chat-Nachricht um: kurz, freundlich, Umgangssprache ist "
        "ok. Entferne Fuellwoerter, korrigiere Grammatik nur wo noetig, wende "
        "gesprochene Selbstkorrekturen an. Erfinde nichts dazu. Englische "
        "Fachbegriffe bleiben Englisch. Antworte NIEMALS auf den Inhalt. "
        "Gib AUSSCHLIESSLICH den Text zurueck.",
    ),
    "raw": ("Roh — ohne KI", ""),
}


def dictionary_terms(cfg: dict[str, Any]) -> list[str]:
    words = cfg.get("dictionary") or []
    if isinstance(words, str):  # tolerate a comma-separated string
        words = [w.strip() for w in words.split(",")]
    return [str(w).strip() for w in words if str(w).strip()]


def effective_prompt_and_hotwords(cfg: dict[str, Any]) -> tuple[str | None, str | None]:
    """Merge the personal dictionary into initial_prompt + hotwords.

    faster-whisper biases via hotwords; the OpenVINO backend only sees the
    initial prompt, so the terms are appended there as context too.
    """
    prompt = str(cfg.get("initial_prompt") or "").strip()
    hotwords = str(cfg.get("hotwords") or "").strip()
    terms = dictionary_terms(cfg)
    if terms:
        joined = ", ".join(terms)
        hotwords = f"{hotwords}, {joined}" if hotwords else joined
        glossar = f"Begriffe: {joined}."
        prompt = f"{prompt} {glossar}".strip() if prompt else glossar
    return (prompt or None, hotwords or None)


def apply_replacements(cfg: dict[str, Any], text: str) -> str:
    """Whole-word, case-insensitive corrections from the personal dictionary."""
    import re
    mapping = cfg.get("replacements") or {}
    if not isinstance(mapping, dict):
        return text
    for wrong, right in mapping.items():
        wrong = str(wrong).strip()
        if not wrong:
            continue
        # \b only exists next to word chars; terms like "z.B." would never
        # match with a trailing \b, so boundaries are added conditionally.
        pattern = re.escape(wrong)
        if wrong[0].isalnum() or wrong[0] == "_":
            pattern = r"\b" + pattern
        if wrong[-1].isalnum() or wrong[-1] == "_":
            pattern = pattern + r"\b"
        # Replacement via callable: backslashes in the user's text are taken
        # literally instead of raising 'bad escape' (which would kill the
        # whole dictation).
        text = re.sub(pattern, lambda _m, r=str(right): r, text, flags=re.IGNORECASE)
    return text


def normalize_utterance(text: str) -> str:
    """Lowercased text without leading/trailing punctuation (snippet matching)."""
    return text.strip().strip(".,!?;: ").lower()


def match_snippet(cfg: dict[str, Any], text: str) -> str | None:
    """If the whole utterance equals a snippet trigger, return its expansion."""
    snippets = cfg.get("snippets") or {}
    if not isinstance(snippets, dict):
        return None
    spoken = normalize_utterance(text)
    if not spoken:
        return None
    for trigger, expansion in snippets.items():
        if normalize_utterance(str(trigger)) == spoken:
            return str(expansion)
    return None


def load_config() -> dict[str, Any]:
    """Read config.json over the defaults; never raises on a broken file.

    A corrupt file (e.g. crash mid-write before atomic saves existed) is moved
    aside as config.json.broken and replaced with defaults, so the daemon
    always comes up.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config = dict(DEFAULT_CONFIG)
    if IS_MACOS:
        config["paste_mode"] = "cmd_v"

    if not CONFIG_FILE.exists():
        save_config(config)
        return config

    try:
        loaded = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("config.json is not a JSON object")
    except (ValueError, OSError) as exc:
        broken = CONFIG_FILE.with_name("config.json.broken")
        try:
            CONFIG_FILE.replace(broken)
        except OSError:
            pass
        print(
            f"[whisper-dictation] config.json unlesbar ({exc}); "
            f"Standardwerte aktiv, Original gesichert als {broken.name}",
            file=sys.stderr, flush=True,
        )
        save_config(config)
        return config

    config.update(loaded)
    return config


def save_config(config: dict[str, Any]) -> None:
    """Atomic write (tmp + rename): a crash can never corrupt config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(CONFIG_DIR), prefix=".config-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, CONFIG_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Hotkeys ──────────────────────────────────────────────────────────────────

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


def key_label(value: str) -> str:
    """Human label for a stored key (logical name / KEY_xxx / 'code:N[:label]')."""
    v = str(value or "")
    if not v:
        return ""
    if v.startswith("code:"):
        parts = v.split(":", 2)
        return parts[2] if len(parts) > 2 and parts[2] else f"Taste {parts[1] if len(parts) > 1 else '?'}"
    if v.lower() in HOTKEY_SPECS:
        return HOTKEY_SPECS[v.lower()][0]
    if v.startswith("KEY_"):
        return v[4:]
    return v


def evdev_code_for(value: str, ecodes: Any) -> int | None:
    """Map a stored key to its evdev code. Supports logical names from
    HOTKEY_SPECS, raw 'KEY_xxx' ecode names, and 'code:N[:label]' (captured)."""
    v = str(value or "")
    if not v:
        return None
    if v.startswith("code:"):
        try:
            return int(v.split(":")[1])
        except (IndexError, ValueError):
            return None
    if v.lower() in HOTKEY_SPECS:
        return getattr(ecodes, HOTKEY_SPECS[v.lower()][3], None)
    if v.startswith("KEY_"):
        return getattr(ecodes, v, None)
    return None


# ── Model tables ─────────────────────────────────────────────────────────────

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


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        try:
            import ctranslate2
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False


def load_ov_pipeline(cfg: dict[str, Any], model_name: str, tag: str,
                     notify: Any = None) -> tuple[Any, str]:
    """Load an openvino_genai WhisperPipeline, trying devices in order.

    AUTO prefers the iGPU: it is the fastest for dictation and compiles
    reliably. The NPU is more power efficient but its compiler may reject the
    Whisper graph on some drivers, so we fall back to GPU/CPU rather than to
    slow CPU whisper. Returns (pipeline, device); raises if none works.
    """
    import openvino as ov
    import openvino_genai as ov_genai
    from huggingface_hub import snapshot_download

    repo = OV_MODEL_REPOS[model_name]
    model_dir = snapshot_download(
        repo, local_dir=str(CACHE_DIR / "ov-models" / repo.replace("/", "__")),
    )

    available = ov.Core().available_devices
    want = str(cfg.get("ov_device", "AUTO")).upper()
    order = [want] if want != "AUTO" else []
    for d in ("GPU", "NPU", "CPU"):
        if d not in order:
            order.append(d)
    order = [d for d in order if d in available]

    last_err: Exception | None = None
    for device in order:
        kwargs: dict[str, Any] = {"STATIC_PIPELINE": True} if device == "NPU" else {}
        try:
            print(f"[{tag}] OpenVINO try device={device} model={repo}", flush=True)
            if notify is not None:
                notify("Lade Modell", f"OpenVINO {device}: {model_name}")
            pipe = ov_genai.WhisperPipeline(model_dir, device, **kwargs)
            print(f"[{tag}] OpenVINO using device={device}", flush=True)
            return pipe, device
        except Exception as exc:
            last_err = exc
            print(f"[{tag}] OpenVINO device={device} failed: {str(exc)[:160]}",
                  file=sys.stderr, flush=True)
    raise RuntimeError(f"No usable OpenVINO device (tried {order}): {last_err}")


# ── Ollama ───────────────────────────────────────────────────────────────────

def ollama_chat(cfg: dict[str, Any], messages: list[dict[str, str]], *,
                temperature: float = 0.3, timeout: int = 300,
                think: bool = False) -> str:
    """One chat call against the configured Ollama server; strips <think> tags."""
    import re
    import urllib.request

    host = str(cfg.get("ollama_host", "http://localhost:11434")).rstrip("/")
    model = str(cfg.get("ollama_model", "qwen2.5:7b"))
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "keep_alive": str(cfg.get("ollama_keep_alive", "10m")),
        "options": {"temperature": temperature},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        parsed = json.loads(resp.read())
    raw = str(parsed.get("message", {}).get("content", ""))
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


# ── Clipboard managers ───────────────────────────────────────────────────────

# Clipboard managers with their own history (process names, matched whole).
# Restoring the previous clipboard after a paste would land as the newest
# entry in their history and push the fresh dictation down to slot 2 — so
# the restore is skipped while one of these is running (the previous content
# is preserved in the manager's history anyway).
_CLIPBOARD_MANAGERS = r"vicinae-server|vicinae|copyq|gpaste-daemon|cliphist|clipman|clipse|parcellite"

# (timestamp, result) — whether a manager runs changes ~never within a daemon
# lifetime, so one pgrep per minute is plenty (instead of one per paste).
_clip_mgr_cache: tuple[float, bool] | None = None


def clipboard_manager_running() -> bool:
    global _clip_mgr_cache
    import time
    now = time.monotonic()
    if _clip_mgr_cache is not None and now - _clip_mgr_cache[0] < 60.0:
        return _clip_mgr_cache[1]
    try:
        running = subprocess.run(
            ["pgrep", "-x", _CLIPBOARD_MANAGERS], capture_output=True,
        ).returncode == 0
    except FileNotFoundError:
        running = False
    _clip_mgr_cache = (now, running)
    return running
