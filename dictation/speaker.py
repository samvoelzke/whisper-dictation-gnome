"""Speaker recognition: learn the user's voice from dictations, then label
"Ich" vs other speakers in recordings — fully local via sherpa-onnx (no torch).

Everything degrades gracefully: if sherpa-onnx or the models are missing,
`available()` is False and all features simply don't activate.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import CACHE_DIR, atomic_write

MODELS_DIR = CACHE_DIR / "speaker-models"
EMBED_MODEL = MODELS_DIR / "campplus.onnx"
SEG_MODEL = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
PROFILE_FILE = CACHE_DIR / "voice-profile.json"

# Download sources (sherpa-onnx release assets; offline after the one-time pull).
EMBED_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/"
             "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")
SEG_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
           "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")

SAMPLE_RATE = 16000
_extractor = None
_diarizer = None


def models_present() -> bool:
    return EMBED_MODEL.exists() and SEG_MODEL.exists()


def available() -> bool:
    """True if sherpa-onnx is importable and the models are downloaded."""
    if not models_present():
        return False
    try:
        import sherpa_onnx  # noqa: F401
        return True
    except ImportError:
        return False


def download_models(progress=None) -> bool:
    """One-time model download (~34 MB). Returns True on success. `progress`
    is an optional callback(str)."""
    import tarfile
    import urllib.request
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def say(msg: str) -> None:
        print(f"[speaker] {msg}", flush=True)
        if progress:
            progress(msg)

    try:
        if not EMBED_MODEL.exists():
            say("Lade Stimm-Modell (28 MB) …")
            tmp = EMBED_MODEL.with_suffix(".part")
            urllib.request.urlretrieve(EMBED_URL, tmp)
            tmp.replace(EMBED_MODEL)
        if not SEG_MODEL.exists():
            say("Lade Segmentierungs-Modell (6 MB) …")
            tar_path = MODELS_DIR / "seg.tar.bz2"
            urllib.request.urlretrieve(SEG_URL, tar_path)
            with tarfile.open(tar_path, "r:bz2") as tf:
                tf.extractall(MODELS_DIR)
            tar_path.unlink(missing_ok=True)
        say("Modelle bereit.")
        return models_present()
    except Exception as exc:  # noqa: BLE001
        say(f"Download fehlgeschlagen: {exc}")
        return False


def _get_extractor():
    global _extractor
    if _extractor is None:
        import sherpa_onnx
        _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(EMBED_MODEL), num_threads=4, provider="cpu"))
    return _extractor


def embed(samples, sr: int = SAMPLE_RATE):
    """L2-normalized 192-dim speaker embedding for mono float32 audio, or None.

    Needs at least ~1 s of audio to be meaningful.
    """
    import numpy as np
    if samples is None or len(samples) < sr:  # < 1 s: not enough signal
        return None
    try:
        ex = _get_extractor()
        st = ex.create_stream()
        st.accept_waveform(sample_rate=sr, waveform=samples)
        st.input_finished()
        if not ex.is_ready(st):
            return None
        vec = np.asarray(ex.compute(st), dtype=np.float32)
        n = float(np.linalg.norm(vec))
        return vec / n if n > 1e-9 else None
    except Exception as exc:  # noqa: BLE001
        print(f"[speaker] embed failed: {exc}", file=sys.stderr, flush=True)
        return None


# ── Voice profile (running mean of the user's dictation embeddings) ──────────

def load_profile() -> dict[str, Any]:
    try:
        return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def profile_vector():
    import numpy as np
    data = load_profile()
    vec = data.get("vector")
    if not vec:
        return None
    return np.asarray(vec, dtype=np.float32)


def enroll_samples(samples, sr: int = SAMPLE_RATE) -> bool:
    """Fold one dictation clip into the running voice profile (mean of unit
    embeddings, re-normalized). Best effort — returns True if it updated."""
    import numpy as np
    vec = embed(samples, sr)
    if vec is None:
        return False
    data = load_profile()
    count = int(data.get("count", 0))
    if count and data.get("vector"):
        prev = np.asarray(data["vector"], dtype=np.float32)
        merged = (prev * count + vec) / (count + 1)
    else:
        merged = vec
    n = float(np.linalg.norm(merged))
    merged = merged / n if n > 1e-9 else merged
    atomic_write(PROFILE_FILE, json.dumps({
        "vector": merged.tolist(), "count": count + 1,
    }))
    return True


def reset_profile() -> None:
    PROFILE_FILE.unlink(missing_ok=True)


# ── Named voices: a global registry of people you renamed once ───────────────
# {"Anna": {"vector": [...], "count": 3}, ...} — future diarizations match
# clusters against these, so Anna is recognized automatically next time.

NAMED_FILE = CACHE_DIR / "named-voices.json"


def named_voices() -> dict[str, Any]:
    try:
        data = json.loads(NAMED_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_named_voice(name: str, vector) -> bool:
    """Store or merge (running mean) one voice embedding under a real name."""
    import numpy as np
    vec = np.asarray(vector, dtype=np.float32)
    if vec.size == 0:
        return False
    data = named_voices()
    entry = data.get(name) or {}
    count = int(entry.get("count", 0))
    if count and entry.get("vector"):
        prev = np.asarray(entry["vector"], dtype=np.float32)
        merged = (prev * count + vec) / (count + 1)
    else:
        merged = vec
    n = float(np.linalg.norm(merged))
    merged = merged / n if n > 1e-9 else merged
    data[name] = {"vector": merged.tolist(), "count": count + 1}
    atomic_write(NAMED_FILE, json.dumps(data))
    return True


def delete_named_voice(name: str) -> None:
    data = named_voices()
    if data.pop(name, None) is not None:
        atomic_write(NAMED_FILE, json.dumps(data))


# ── Diarization + labeling ───────────────────────────────────────────────────

def _get_diarizer():
    global _diarizer
    if _diarizer is None:
        import sherpa_onnx
        _diarizer = sherpa_onnx.OfflineSpeakerDiarization(
            sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=str(SEG_MODEL))),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(EMBED_MODEL), num_threads=4, provider="cpu"),
                clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5),
                min_duration_on=0.3, min_duration_off=0.5))
    return _diarizer


def diarize_and_label(samples, sr: int = SAMPLE_RATE, me_threshold: float = 0.45,
                      named_threshold: float = 0.5):
    """Return ([(start, end, label)], {label: embedding-list}).

    The enrolled user becomes 'Ich', voices from the named registry get
    their real name, everyone else 'Sprecher N'. The per-label embeddings
    let the GUI enroll a voice into the registry when the user renames it.
    Returns ([], {}) on failure/unavailable.
    """
    import numpy as np
    if not available():
        return [], {}
    try:
        sd = _get_diarizer()
        if sr != sd.sample_rate:  # sherpa expects 16 kHz
            return [], {}
        segments = sd.process(samples).sort_by_start_time()
    except Exception as exc:  # noqa: BLE001
        print(f"[speaker] diarize failed: {exc}", file=sys.stderr, flush=True)
        return [], {}
    if not segments:
        return [], {}

    profile = profile_vector()
    # Mean embedding per diarized cluster (re-embed each cluster's audio once).
    cluster_vecs: dict[int, list] = {}
    for r in segments:
        chunk = samples[int(r.start * sr):int(r.end * sr)]
        if len(chunk) < int(0.6 * sr):
            continue
        v = embed(chunk, sr)
        if v is not None:
            cluster_vecs.setdefault(r.speaker, []).append(v)
    means = {}
    for spk, vs in cluster_vecs.items():
        m = np.mean(vs, axis=0)
        n = float(np.linalg.norm(m))
        means[spk] = m / n if n > 1e-9 else m

    me_cluster = None
    if profile is not None and means:
        sims = {spk: float(np.dot(profile, m)) for spk, m in means.items()}
        best = max(sims, key=sims.get)
        if sims[best] >= me_threshold:
            me_cluster = best

    # Named registry: greedy best-match, each name and each cluster used once.
    # Slightly stricter threshold than 'Ich' — a wrong real name is worse
    # than a neutral 'Sprecher 2'.
    registry = {name: np.asarray(e["vector"], dtype=np.float32)
                for name, e in named_voices().items()
                if isinstance(e, dict) and e.get("vector")}
    assigned: dict[int, str] = {}
    if registry and means:
        pairs = sorted(
            ((float(np.dot(vec, m)), spk, name)
             for spk, m in means.items() if spk != me_cluster
             for name, vec in registry.items()),
            key=lambda p: p[0], reverse=True)
        for sim, spk, name in pairs:
            if sim < named_threshold:
                break
            if spk in assigned or name in assigned.values():
                continue
            assigned[spk] = name

    # Labels: 'Ich' > real name > 'Sprecher N' (numbered from 2 when the
    # user is present, from 1 otherwise — matching how people count).
    order = sorted({r.speaker for r in segments})
    labels: dict[int, str] = {}
    counter = 1 if me_cluster is not None else 0
    for spk in order:
        if spk == me_cluster:
            labels[spk] = "Ich"
        elif spk in assigned:
            labels[spk] = assigned[spk]
        else:
            counter += 1
            labels[spk] = f"Sprecher {counter}"
    voices = {labels[spk]: m.tolist() for spk, m in means.items()}
    return [(r.start, r.end, labels[r.speaker]) for r in segments], voices
