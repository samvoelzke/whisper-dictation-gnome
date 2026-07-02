#!/usr/bin/env python3
"""Long-form recorder for Whisper Dictation (lectures, calls, meetings).

This is *additive*: the live dictation daemon (daemon.py) is untouched. The
recorder is built for hour-long sessions where losing everything on a crash is
unacceptable, so it works in three robust, separately triggerable phases:

  1. record   - ffmpeg writes Opus to disk *continuously* (crash = keep so far).
                Source: mic, system audio (PipeWire monitor) or both mixed.
  2. transcribe - 5-minute chunks through the existing Whisper backend; the
                partial transcript + progress are saved after *every* chunk,
                so an abort loses at most one chunk and can be resumed.
  3. summarize - Ollama map-reduce over the transcript with a user focus prompt.

The GUI drives these via bin/whisper-recorder.sh. Run with the project venv
python (faster-whisper / openvino-genai live there).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# numpy and the daemon module (which imports numpy) are imported *lazily* inside
# the transcribe/summarize paths only. That keeps lightweight commands — the
# ones the GUI calls on every tab switch (record-status, list, devices) — at
# ~50 ms startup instead of ~300 ms (numpy alone is ~160 ms).
sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE_DIR = Path.home() / ".cache" / "whisper-dictation"
RECORDINGS_DIR = Path.home() / ".local" / "share" / "whisper-dictation" / "recordings"
STATE_FILE = CACHE_DIR / "recorder-state.json"
CHUNK_SECONDS = 300  # 5-minute transcription chunks
SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono


# ── small helpers ────────────────────────────────────────────────────────────

def _notify(summary: str, body: str = "") -> None:
    try:
        subprocess.run(
            ["notify-send", "-a", "Whisper Dictation", "-i",
             "io.voelzke.WhisperDictation", summary, body],
            check=False,
        )
    except FileNotFoundError:
        pass


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _pactl(*args: str) -> str:
    try:
        out = subprocess.run(["pactl", *args], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def _default_source() -> str:
    return _pactl("get-default-source")


def _monitor_source() -> str:
    sink = _pactl("get-default-sink")
    return f"{sink}.monitor" if sink else ""


def _list_audio_devices() -> dict[str, Any]:
    """Return {'mics':[...], 'monitors':[...]} with name/description/default,
    parsed from `pactl list sources`. Monitors = system-audio capture points."""
    default_src = _default_source()
    default_mon = _monitor_source()
    mics: list[dict[str, Any]] = []
    monitors: list[dict[str, Any]] = []
    name = desc = ""
    is_monitor = False
    text = _pactl("list", "sources")
    for raw in text.splitlines() + ["Source #__end"]:
        line = raw.strip()
        if line.startswith("Source #") or line.startswith("Sink #"):
            if name:
                entry = {"name": name, "desc": desc or name}
                if is_monitor:
                    entry["default"] = (name == default_mon)
                    monitors.append(entry)
                else:
                    entry["default"] = (name == default_src)
                    mics.append(entry)
            name = desc = ""
            is_monitor = False
        elif line.startswith("Name:"):
            name = line.split("Name:", 1)[1].strip()
            is_monitor = name.endswith(".monitor")
        elif line.startswith("Description:"):
            desc = line.split("Description:", 1)[1].strip()
    return {"mics": mics, "monitors": monitors,
            "default_mic": default_src, "default_monitor": default_mon}


def cmd_devices(_args: argparse.Namespace) -> int:
    print(json.dumps(_list_audio_devices()))
    return 0


def _probe_duration(audio: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(audio)],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip() or 0.0)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return 0.0


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except OSError:
        return False
    return True


# ── phase 1: recording ───────────────────────────────────────────────────────

def _ffmpeg_input_args(source: str, mic_dev: str = "", monitor_dev: str = "") -> list[str]:
    """Build ffmpeg -f pulse inputs for mic / system / both, honouring explicit
    device names (empty = system default)."""
    mic = mic_dev or _default_source()
    monitor = monitor_dev or _monitor_source()
    if source == "mic":
        if not mic:
            raise RuntimeError("Kein Standard-Mikrofon gefunden (pactl).")
        return ["-f", "pulse", "-i", mic]
    if source == "system":
        if not monitor:
            raise RuntimeError("Keine System-Audio-Quelle (Monitor) gefunden.")
        return ["-f", "pulse", "-i", monitor]
    # both: mix mic + system into one mono track
    if not mic or not monitor:
        raise RuntimeError("Mikrofon oder System-Audio nicht verfuegbar fuer 'both'.")
    # aresample=async keeps the two independent clocks (mic vs. monitor) in sync
    # over long recordings instead of drifting / producing non-monotonic DTS.
    return [
        "-f", "pulse", "-i", mic,
        "-f", "pulse", "-i", monitor,
        "-filter_complex",
        "[0:a]aresample=async=1000[a0];[1:a]aresample=async=1000[a1];"
        "[a0][a1]amix=inputs=2:duration=longest:normalize=0",
    ]


def cmd_record_start(args: argparse.Namespace) -> int:
    state = _read_json(STATE_FILE, {})
    if state.get("pid") and _pid_alive(int(state["pid"])):
        print(json.dumps({"error": "already_recording", "base": state.get("base")}))
        return 1

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    title = (args.title or "Aufnahme").strip()
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip().replace(" ", "_")
    base = f"{stamp}_{safe}" if safe else stamp
    audio = RECORDINGS_DIR / f"{base}.opus"

    try:
        inputs = _ffmpeg_input_args(args.source, getattr(args, "mic_device", "") or "",
                                    getattr(args, "monitor_device", "") or "")
    except RuntimeError as exc:
        print(json.dumps({"error": str(exc)}))
        return 1

    bitrate = str(args.bitrate or "32k")
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "warning",
        *inputs,
        "-ac", "1", "-c:a", "libopus", "-b:a", bitrate, "-application", "voip",
        str(audio),
    ]
    # Keep the laptop awake for the whole (possibly multi-hour) recording so it
    # cannot suspend mid-meeting and cut the file off. systemd-inhibit holds the
    # lock exactly as long as its child (ffmpeg) runs.
    if shutil.which("systemd-inhibit"):
        cmd = ["systemd-inhibit", "--what=sleep:idle", "--who=Whisper Dictation",
               "--why=Langaufnahme laeuft", "--mode=block", *cmd]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log = open(CACHE_DIR / "recorder-ffmpeg.log", "ab")
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
        start_new_session=True,
    )

    _write_json(RECORDINGS_DIR / f"{base}.meta.json", {
        "title": title, "source": args.source, "created": datetime.now().isoformat(),
        "audio": audio.name, "duration_seconds": 0,
    })
    _write_json(STATE_FILE, {
        "pid": proc.pid, "base": base, "audio": str(audio),
        "source": args.source, "title": title, "started": time.time(),
        "paused": False, "paused_accum": 0.0,
    })
    print(json.dumps({"base": base, "audio": str(audio), "source": args.source}))
    return 0


def cmd_record_stop(_args: argparse.Namespace) -> int:
    state = _read_json(STATE_FILE, {})
    pid = int(state.get("pid", 0) or 0)
    base = state.get("base")
    if not pid or not _pid_alive(pid):
        STATE_FILE.unlink(missing_ok=True)
        print(json.dumps({"error": "not_recording", "base": base}))
        return 1

    # SIGINT lets ffmpeg finalise the Opus/Ogg container cleanly. We signal the
    # whole process group (pid is the group leader via start_new_session) so it
    # also reaches ffmpeg when wrapped in systemd-inhibit.
    def _signal_group(sig: int) -> None:
        try:
            os.killpg(pid, sig)
        except OSError:
            try:
                os.kill(pid, sig)
            except OSError:
                pass

    # If paused (SIGSTOP'd), resume first — a stopped process can't act on SIGINT
    # — and let it settle briefly so a just-started ffmpeg can open + flush the
    # output before we ask it to finalise.
    if state.get("paused"):
        _signal_group(signal.SIGCONT)
        time.sleep(0.3)
    _signal_group(signal.SIGINT)
    for _ in range(80):  # wait up to ~8s for ffmpeg to write the trailer
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    else:
        _signal_group(signal.SIGKILL)

    STATE_FILE.unlink(missing_ok=True)
    duration = 0.0
    if base:
        audio = RECORDINGS_DIR / f"{base}.opus"
        duration = _probe_duration(audio)
        meta_path = RECORDINGS_DIR / f"{base}.meta.json"
        meta = _read_json(meta_path, {})
        meta["duration_seconds"] = round(duration, 1)
        _write_json(meta_path, meta)
    print(json.dumps({"base": base, "duration_seconds": round(duration, 1)}))
    return 0


def _elapsed_audio(state: dict[str, Any]) -> float:
    """Recorded audio seconds so far, excluding paused time."""
    now = time.time()
    el = now - float(state.get("started", now)) - float(state.get("paused_accum", 0.0))
    if state.get("paused"):
        el -= now - float(state.get("paused_at", now))
    return max(0.0, el)


def cmd_record_status(_args: argparse.Namespace) -> int:
    state = _read_json(STATE_FILE, {})
    pid = int(state.get("pid", 0) or 0)
    if pid and _pid_alive(pid):
        print(json.dumps({
            "recording": True, "base": state.get("base"),
            "title": state.get("title"), "source": state.get("source"),
            "paused": bool(state.get("paused", False)),
            "elapsed": round(_elapsed_audio(state), 1),
        }))
    else:
        print(json.dumps({"recording": False}))
    return 0


def _signal_recording_group(state: dict[str, Any], sig: int) -> bool:
    pid = int(state.get("pid", 0) or 0)
    if not pid or not _pid_alive(pid):
        return False
    try:
        os.killpg(pid, sig)
    except OSError:
        try:
            os.kill(pid, sig)
        except OSError:
            return False
    return True


def cmd_record_pause(_args: argparse.Namespace) -> int:
    state = _read_json(STATE_FILE, {})
    if not (int(state.get("pid", 0) or 0) and _pid_alive(int(state.get("pid", 0) or 0))):
        print(json.dumps({"error": "not_recording"}))
        return 1
    if state.get("paused"):
        print(json.dumps({"base": state.get("base"), "paused": True}))
        return 0
    # SIGSTOP freezes ffmpeg; the paused span is simply dropped from the audio.
    _signal_recording_group(state, signal.SIGSTOP)
    state["paused"] = True
    state["paused_at"] = time.time()
    _write_json(STATE_FILE, state)
    print(json.dumps({"base": state.get("base"), "paused": True}))
    return 0


def cmd_record_resume(_args: argparse.Namespace) -> int:
    state = _read_json(STATE_FILE, {})
    if not (int(state.get("pid", 0) or 0) and _pid_alive(int(state.get("pid", 0) or 0))):
        print(json.dumps({"error": "not_recording"}))
        return 1
    if not state.get("paused"):
        print(json.dumps({"base": state.get("base"), "paused": False}))
        return 0
    state["paused_accum"] = float(state.get("paused_accum", 0.0)) + (
        time.time() - float(state.get("paused_at", time.time())))
    state["paused"] = False
    state["paused_at"] = 0
    _write_json(STATE_FILE, state)
    _signal_recording_group(state, signal.SIGCONT)
    print(json.dumps({"base": state.get("base"), "paused": False}))
    return 0


def cmd_rename(args: argparse.Namespace) -> int:
    meta_path = RECORDINGS_DIR / f"{args.base}.meta.json"
    if not meta_path.exists():
        print(json.dumps({"error": "not_found", "base": args.base}))
        return 1
    meta = _read_json(meta_path, {})
    meta["title"] = (args.title or "").strip() or meta.get("title", args.base)
    _write_json(meta_path, meta)
    print(json.dumps({"base": args.base, "title": meta["title"]}))
    return 0


# ── phase 2: transcription (chunked + resumable) ─────────────────────────────

class _Backend:
    """Loads large-v3 via OpenVINO (Intel GPU/NPU) with faster-whisper fallback."""

    def __init__(self, cfg: dict[str, Any], model_name: str):
        self.cfg = cfg
        self.model_name = model_name
        self.kind = ""
        self.ov_pipe = None
        self.fw_model = None
        self._load()

    def _load(self) -> None:
        from daemon import OV_MODEL_REPOS
        want_backend = str(self.cfg.get("backend", "auto"))
        if want_backend != "faster" and self.model_name in OV_MODEL_REPOS:
            try:
                self._load_openvino()
                self.kind = "openvino"
                return
            except Exception as exc:  # noqa: BLE001
                print(f"[recorder] OpenVINO load failed ({str(exc)[:160]}); "
                      f"falling back to faster-whisper", file=sys.stderr, flush=True)
        self._load_faster()
        self.kind = "faster"

    def _load_openvino(self) -> None:
        import openvino as ov
        import openvino_genai as ov_genai
        from huggingface_hub import snapshot_download
        from daemon import OV_MODEL_REPOS

        repo = OV_MODEL_REPOS[self.model_name]
        model_dir = snapshot_download(
            repo, local_dir=str(CACHE_DIR / "ov-models" / repo.replace("/", "__")))
        available = ov.Core().available_devices
        want = str(self.cfg.get("ov_device", "AUTO")).upper()
        order = [want] if want != "AUTO" else []
        for d in ("GPU", "NPU", "CPU"):
            if d not in order:
                order.append(d)
        order = [d for d in order if d in available]
        last: Exception | None = None
        for device in order:
            kwargs = {"STATIC_PIPELINE": True} if device == "NPU" else {}
            try:
                print(f"[recorder] OpenVINO {device}: {repo}", flush=True)
                self.ov_pipe = ov_genai.WhisperPipeline(model_dir, device, **kwargs)
                return
            except Exception as exc:  # noqa: BLE001
                last = exc
                print(f"[recorder] OpenVINO {device} failed: {str(exc)[:140]}",
                      file=sys.stderr, flush=True)
        raise RuntimeError(f"No usable OpenVINO device (tried {order}): {last}")

    def _load_faster(self) -> None:
        from faster_whisper import WhisperModel
        from daemon import FASTER_MODEL_MAP, _cuda_available
        fw_name = FASTER_MODEL_MAP.get(self.model_name, self.model_name)
        device, compute = ("cuda", "float16") if _cuda_available() else ("cpu", "int8")
        print(f"[recorder] faster-whisper {fw_name} on {device}", flush=True)
        self.fw_model = WhisperModel(
            fw_name, device=device, compute_type=compute,
            download_root=str(CACHE_DIR / "models-faster"))

    def transcribe(self, audio: np.ndarray) -> str:
        lang = str(self.cfg.get("recorder_language") or self.cfg.get("language") or "").strip().lower()
        language = None if lang in ("", "auto") else lang
        prompt = str(self.cfg.get("initial_prompt") or "").strip() or None
        if self.kind == "openvino":
            kwargs: dict[str, Any] = {"task": "transcribe"}
            if language:
                kwargs["language"] = f"<|{language}|>"
            if prompt:
                kwargs["initial_prompt"] = prompt
            return str(self.ov_pipe.generate(audio, **kwargs))  # type: ignore[union-attr]
        segments, _ = self.fw_model.transcribe(  # type: ignore[union-attr]
            audio, language=language, initial_prompt=prompt,
            beam_size=int(self.cfg.get("beam_size", 5)),
            condition_on_previous_text=False,
            vad_filter=bool(self.cfg.get("vad_filter", True)),
        )
        return "".join(s.text for s in segments)


def _decode_chunk(audio: Path, start: float, length: float):
    import numpy as np
    cmd = [
        "ffmpeg", "-nostdin", "-v", "quiet",
        "-ss", str(start), "-t", str(length), "-i", str(audio),
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "f32le", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=False).stdout
    # .copy() makes the array writable (faster-whisper/ctranslate2 may need it).
    return np.frombuffer(raw, dtype=np.float32).copy()


def _find_silence_cut(audio: Path, target: float, window: float) -> float | None:
    """Find a silence near `target` (s) to cut on, so chunk boundaries land in
    pauses instead of mid-word. Returns an absolute cut time, or None if the
    window has no clear pause. Only a small window is decoded, so it's cheap."""
    lo = max(0.0, target - window)
    cmd = ["ffmpeg", "-nostdin", "-v", "info", "-ss", str(lo), "-t", str(2 * window),
           "-i", str(audio), "-af", "silencedetect=n=-35dB:d=0.35", "-f", "null", "-"]
    err = subprocess.run(cmd, capture_output=True, text=True, check=False).stderr
    events: list[tuple[str, float]] = []
    for line in err.splitlines():
        for kind, tag in (("s", "silence_start:"), ("e", "silence_end:")):
            if tag in line:
                try:
                    events.append((kind, float(line.split(tag)[1].split()[0])))
                except (ValueError, IndexError):
                    pass
    # Pair starts/ends into silence intervals (relative to lo); a window that
    # begins or ends inside a silence yields an unpaired end/start.
    intervals: list[tuple[float, float]] = []
    cur: float | None = None
    for kind, val in events:
        if kind == "s":
            cur = val
        else:
            intervals.append((cur if cur is not None else 0.0, val))
            cur = None
    if cur is not None:
        intervals.append((cur, cur + 0.4))
    cands = [lo + (s + e) / 2.0 for s, e in intervals]
    cands = [c for c in cands if c > lo + 0.2]
    if not cands:
        return None
    return min(cands, key=lambda c: abs(c - target))


def cmd_transcribe(args: argparse.Namespace) -> int:
    base = args.base
    audio = RECORDINGS_DIR / f"{base}.opus"
    if not audio.exists():
        print(json.dumps({"error": "audio_not_found", "base": base}))
        return 1
    txt_path = RECORDINGS_DIR / f"{base}.txt"
    prog_path = RECORDINGS_DIR / f"{base}.progress.json"

    from daemon import load_config
    cfg = load_config()
    prog = _read_json(prog_path, {})
    # Keep chunk size consistent across resumes: prefer the one stored at the
    # first run, else CLI flag, else config, else default.
    chunk = (int(prog.get("chunk_seconds", 0)) if not args.restart else 0) \
        or int(getattr(args, "chunk_seconds", 0) or 0) \
        or int(cfg.get("recorder_chunk_seconds", CHUNK_SECONDS) or CHUNK_SECONDS)
    chunk = max(5, chunk)

    # A cleanly-stopped file has a duration; a SIGKILL'd (crashed) recording may
    # not. When unknown, we drive the loop by decoding until the audio runs out
    # so a crashed recording is still transcribed *in full*, not just chunk 1.
    # Boundaries are snapped to nearby silences so words are never cut in half.
    duration = _probe_duration(audio)
    have_dur = duration > 0.5
    window = min(15.0, chunk / 3.0)
    start_pos = float(prog.get("seconds_done", 0.0)) if not args.restart else 0.0
    if args.restart:
        txt_path.unlink(missing_ok=True)
    if have_dur and start_pos >= duration - 0.5 and txt_path.exists():
        print(json.dumps({"base": base, "status": "already_done", "chars": len(txt_path.read_text())}))
        return 0

    model_name = args.model or str(cfg.get("recorder_model", "large-v3"))

    def _progress(pos: float, status: str, segs: int) -> None:
        pct = min(100, int(pos / duration * 100)) if have_dur and duration else None
        _write_json(prog_path, {
            "seconds_done": round(pos, 1), "duration": round(duration, 1),
            "percent": pct, "segments": segs, "chunk_seconds": chunk,
            "status": status, "model": model_name, "backend": getattr(backend, "kind", ""),
        })

    backend = None  # type: ignore[assignment]
    _progress(start_pos, "loading", 0)
    backend = _Backend(cfg, model_name)

    pos = start_pos
    segs = 0
    mode = "a" if start_pos > 0 else "w"
    with open(txt_path, mode, encoding="utf-8") as fh:
        while True:
            if have_dur and pos >= duration - 0.5:
                break
            target = pos + chunk
            if have_dur and target >= duration:
                seg_end = duration
            else:
                cut = _find_silence_cut(audio, target, window)
                seg_end = cut if (cut and cut > pos + 1.0) else target
            length = max(1.0, seg_end - pos)
            audio_np = _decode_chunk(audio, pos, length)
            if audio_np.size < SAMPLE_RATE // 2:  # < 0.5s -> end of file
                break
            text = backend.transcribe(audio_np).strip()
            if text:
                fh.write(text + "\n")
                fh.flush()
            pos = seg_end if have_dur else pos + length
            segs += 1
            _progress(pos, "running", segs)
            pct = f"{int(pos / duration * 100)}%" if have_dur and duration else f"{pos:.0f}s"
            print(f"[recorder] segment {segs} -> {pct} ({backend.kind})", flush=True)

    _progress(duration if have_dur else pos, "done", segs)
    chars = len(txt_path.read_text()) if txt_path.exists() else 0
    _notify("Transkription fertig", f"{base} ({chars} Zeichen)")
    print(json.dumps({"base": base, "status": "done", "chars": chars}))
    if args.then_summarize:
        ns = argparse.Namespace(base=base, focus=args.then_summarize)
        cmd_summarize(ns)
    return 0


# ── phase 3: summarize via Ollama (map-reduce) ───────────────────────────────

def _ollama_chat(cfg: dict[str, Any], system: str, user: str, timeout: int = 300) -> str:
    import urllib.request
    host = str(cfg.get("ollama_host", "http://localhost:11434")).rstrip("/")
    model = str(cfg.get("ollama_model", "qwen2.5:7b"))
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False, "think": False, "keep_alive": "10m",
        "options": {"temperature": 0.3},
    }
    req = urllib.request.Request(
        f"{host}/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        parsed = json.loads(resp.read())
    import re
    raw = str(parsed.get("message", {}).get("content", ""))
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def _split_words(text: str, max_words: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)] or [""]


def cmd_summarize(args: argparse.Namespace) -> int:
    base = args.base
    txt_path = RECORDINGS_DIR / f"{base}.txt"
    if not txt_path.exists():
        print(json.dumps({"error": "transcript_not_found", "base": base}))
        return 1
    transcript = txt_path.read_text(encoding="utf-8").strip()
    if not transcript:
        print(json.dumps({"error": "empty_transcript", "base": base}))
        return 1
    focus = (args.focus or "").strip() or "die wichtigsten Inhalte, Kernaussagen und Action-Items"
    from daemon import load_config
    cfg = load_config()

    # ~3000 words (~4k tokens) per map block: well within qwen2.5's 32k context,
    # while keeping the number of LLM calls (and latency) low on long transcripts.
    blocks = _split_words(transcript, 3000)
    if len(blocks) == 1:
        summary = _ollama_chat(cfg,
            "Du bist ein praeziser Notiz-Assistent. Antworte auf Deutsch, behalte "
            "englische Fachbegriffe bei und erfinde nichts.",
            f"Erstelle aus dieser Transkription eine gut strukturierte Zusammenfassung "
            f"mit Markdown-Ueberschriften und Stichpunkten. Fokus: {focus}.\n\n{transcript}")
    else:
        partials = []
        for idx, block in enumerate(blocks, 1):
            print(f"[recorder] summarize map {idx}/{len(blocks)}", flush=True)
            partials.append(_ollama_chat(cfg,
                "Du bist ein praeziser Notiz-Assistent. Behalte englische Fachbegriffe bei, "
                "erfinde nichts.",
                f"Fasse diesen Abschnitt einer laengeren Aufnahme stichpunktartig zusammen. "
                f"Fokus: {focus}.\n\nAbschnitt {idx}/{len(blocks)}:\n{block}"))
        print("[recorder] summarize reduce", flush=True)
        summary = _ollama_chat(cfg,
            "Du bist ein praeziser Notiz-Assistent. Antworte auf Deutsch, behalte englische "
            "Fachbegriffe bei, erfinde nichts.",
            f"Hier sind Teil-Zusammenfassungen einer langen Aufnahme (in Reihenfolge). "
            f"Fuege sie zu EINER zusammenhaengenden, gut strukturierten Zusammenfassung mit "
            f"Markdown-Ueberschriften und Stichpunkten zusammen. Fokus: {focus}.\n\n"
            + "\n\n".join(partials), timeout=420)

    meta = _read_json(RECORDINGS_DIR / f"{base}.meta.json", {})
    header = f"# {meta.get('title', base)}\n\n*Fokus: {focus}*\n\n"
    summary_path = RECORDINGS_DIR / f"{base}.summary.md"
    summary_path.write_text(header + summary + "\n", encoding="utf-8")
    _notify("Zusammenfassung fertig", base)
    print(json.dumps({"base": base, "status": "done", "summary": str(summary_path)}))
    return 0


# ── listing ──────────────────────────────────────────────────────────────────

def cmd_list(_args: argparse.Namespace) -> int:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    state = _read_json(STATE_FILE, {})
    active = state.get("base") if state.get("pid") and _pid_alive(int(state.get("pid", 0) or 0)) else None
    items = []
    for meta_path in sorted(RECORDINGS_DIR.glob("*.meta.json"), reverse=True):
        base = meta_path.name[: -len(".meta.json")]
        meta = _read_json(meta_path, {})
        prog = _read_json(RECORDINGS_DIR / f"{base}.progress.json", {})
        txt = RECORDINGS_DIR / f"{base}.txt"
        items.append({
            "base": base,
            "title": meta.get("title", base),
            "created": meta.get("created", ""),
            "source": meta.get("source", ""),
            "duration_seconds": meta.get("duration_seconds", 0),
            "recording": base == active,
            "transcribed": txt.exists() and prog.get("status") == "done",
            "transcribe_status": prog.get("status", ""),
            "percent": prog.get("percent"),
            "summarized": (RECORDINGS_DIR / f"{base}.summary.md").exists(),
        })
    print(json.dumps({"recordings": items, "active": active}))
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    base = args.base
    for suffix in (".opus", ".meta.json", ".txt", ".progress.json", ".summary.md"):
        (RECORDINGS_DIR / f"{base}{suffix}").unlink(missing_ok=True)
    print(json.dumps({"base": base, "status": "deleted"}))
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Whisper Dictation long-form recorder")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("record-start")
    s.add_argument("--source", choices=["mic", "system", "both"], default="both")
    s.add_argument("--title", default="")
    s.add_argument("--bitrate", default="32k")
    s.add_argument("--mic-device", default="", dest="mic_device")
    s.add_argument("--monitor-device", default="", dest="monitor_device")
    s.set_defaults(func=cmd_record_start)

    sub.add_parser("record-stop").set_defaults(func=cmd_record_stop)
    sub.add_parser("record-status").set_defaults(func=cmd_record_status)
    sub.add_parser("record-pause").set_defaults(func=cmd_record_pause)
    sub.add_parser("record-resume").set_defaults(func=cmd_record_resume)
    sub.add_parser("devices").set_defaults(func=cmd_devices)
    sub.add_parser("list").set_defaults(func=cmd_list)

    s = sub.add_parser("rename")
    s.add_argument("base")
    s.add_argument("--title", default="")
    s.set_defaults(func=cmd_rename)

    s = sub.add_parser("transcribe")
    s.add_argument("base")
    s.add_argument("--model", default="")
    s.add_argument("--restart", action="store_true")
    s.add_argument("--chunk-seconds", type=int, default=0, dest="chunk_seconds",
                   help="override chunk length (default: config recorder_chunk_seconds or 300)")
    s.add_argument("--then-summarize", default="", help="focus prompt; summarize after transcribe")
    s.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("summarize")
    s.add_argument("base")
    s.add_argument("--focus", default="")
    s.set_defaults(func=cmd_summarize)

    s = sub.add_parser("delete")
    s.add_argument("base")
    s.set_defaults(func=cmd_delete)

    args = p.parse_args()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
