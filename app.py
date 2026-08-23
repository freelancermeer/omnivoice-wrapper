#!/usr/bin/env python3
"""
OmniVoice Voiceover Studio — local Gradio UI + REST API for k2-fsa/OmniVoice.

Adapted from the official HuggingFace Space app.py with the ZeroGPU wrapper
removed, then hardened against everything a real production batch turned up
(63 clips / 64.7 minutes / 10,269 words, every clip transcribed back and diffed
against the script that was sent).

WHAT THE BATCH FOUND, AND WHERE IT IS FIXED
-------------------------------------------
1. Reference audio leaked into every clip — 218 words that were never sent, one
   voice contributing 185 of them ("forcing" x163). Root cause: a 60s reference
   hard-cut at exactly MAX_REF_SEC landed mid-word, and the model kept finishing
   that word.                        -> audio_fx.smart_trim_reference (cut at a
                                        real pause), reference validation at
                                        registration, and a verifier that trims
                                        a confirmed tail artefact.
2. Words silently dropped — 12 of them, mostly one arm of a repeated structure
   ("He called it perjury / fraud / contempt" lost the middle clause), with
   nothing reporting it.             -> verify.word_diff on every chunk, retry,
                                        and a warning the caller can see.
3. Output loudness tracked the reference (-0.2 dB to -12.6 dB in one batch).
                                     -> BS.1770 loudness normalization on both
                                        the reference and the output.
4. CUDA OOM that never recovered, while /api/health said "ok".
                                     -> gpu_guard: inference_mode, explicit del,
                                        bounded concurrency, OOM recovery, and
                                        /api/ready backed by a real self-test.
5. Rushed short clips, 133-203 wpm across one batch.
                                     -> balanced chunking, and wpm measured and
                                        reported against each voice's own
                                        baseline (a metric, never a gate).
6. Numbers/currency/dashes/acronyms  -> textnorm, with the normalized text
                                        returned so it can be checked without
                                        listening to twenty minutes of audio.

API CONTRACT: POST /api/tts is frozen — multipart in, raw audio bytes out, with
the X-Duration-Sec and X-RTF headers it always had. Everything new is either an
additional response header or lives on /api/v2/tts. See LOCAL_API.md.

Run:  python app.py       (UI on :7860, API on :8001)
"""

import os

# MUST be set before torch initialises CUDA. Made for long-running processes
# whose allocations vary in size, which is exactly a TTS server: it lets the
# allocator grow a segment instead of fragmenting into unusable holes — the
# "GPU at 5% utilisation and out of memory" failure.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import base64
import datetime
import html as html_mod
import io
import json
import logging
import queue as queue_mod
import socket
import re
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from typing import Any, Dict, List, Optional, Tuple

# Windows consoles default to cp1252 and crash when logging non-Latin text
# (language names, normalized output, etc.). Force UTF-8 with replacement.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
log = logging.getLogger("omnivoice.app")
log.setLevel(logging.INFO)
logging.getLogger("omnivoice").setLevel(logging.INFO)

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.cli.demo import _ALL_LANGUAGES

import inspect

import audio_fx
import gpu_guard
import textnorm
import verify
import watermark

API_VERSION = "2.0"

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_MODEL = os.path.join(_HERE, "Model")


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _default_checkpoint() -> str:
    """Prefer the local ./Model folder (already downloaded) over the HF repo."""
    if os.path.isfile(os.path.join(_LOCAL_MODEL, "config.json")):
        return _LOCAL_MODEL
    return "k2-fsa/OmniVoice"


CHECKPOINT = os.environ.get("OMNIVOICE_MODEL") or _default_checkpoint()
ASR_MODEL = os.environ.get("OMNIVOICE_ASR_MODEL", "openai/whisper-large-v3-turbo")

# --- text front-end -------------------------------------------------------
NORMALIZE_LEVEL = os.environ.get("OMNIVOICE_NORMALIZE_LEVEL", "").strip().lower()
if not NORMALIZE_LEVEL:
    # Back-compat with the old on/off switch.
    NORMALIZE_LEVEL = "full" if _env_flag("OMNIVOICE_NORMALIZE") else "off"
LEXICON_PATH = os.environ.get("OMNIVOICE_LEXICON", os.path.join(_HERE, "lexicon.json"))
YEAR_STYLE = _env_flag("OMNIVOICE_YEARS")
# "us" -> "one hundred fifty"; "uk" -> "one hundred and fifty".
textnorm.NUM_STYLE = os.environ.get("OMNIVOICE_NUM_STYLE", "us").strip().lower()
# Community workaround from upstream #116 for swallowed final phonemes. Off by
# default; A/B it with tools/audit_batch.py before trusting it.
textnorm.SPACE_BEFORE_PUNCT = _env_flag("OMNIVOICE_SPACE_BEFORE_PUNCT", "0")
LEXICON = textnorm.load_lexicon(LEXICON_PATH)

CHUNK = _env_flag("OMNIVOICE_CHUNK")
# Target characters per chunk. 100 was too small in both directions: upstream
# #229/#206 report the cloned voice switching speaker on short generations
# ("longer sentences appear more stable"), and every extra chunk is another
# fixed per-call overhead on the RTF. 200 sits inside the 100-250 band the
# long-form literature recommends, while staying well clear of #144 (crackling
# on very long inputs). Nothing shorter than MAX_CHARS/3 is ever emitted.
MAX_CHARS = int(os.environ.get("OMNIVOICE_MAX_CHARS", "200"))
MIN_CHUNK_CHARS = int(os.environ.get("OMNIVOICE_MIN_CHUNK_CHARS",
                                     str(max(40, MAX_CHARS // 3))))
GAP_SEC = float(os.environ.get("OMNIVOICE_CHUNK_GAP", "0.15"))

# --- reference audio ------------------------------------------------------
# The reference length we aim for. NOT a hard cut: see REF_HARD_MAX_SEC.
MAX_REF_SEC = float(os.environ.get("OMNIVOICE_MAX_REF_SEC", "10"))
# How far past the target we will go to let a sentence finish, rather than
# backing off to a much earlier pause and shipping a thin reference.
REF_HARD_MAX_SEC = float(os.environ.get("OMNIVOICE_REF_HARD_MAX_SEC", "15"))
REF_TAIL_SILENCE = float(os.environ.get("OMNIVOICE_REF_TAIL_SILENCE", "0.30"))
REF_TARGET_LUFS = float(os.environ.get("OMNIVOICE_REF_LUFS", "-20"))
# Shortest reference we are willing to cut back to when a clip ends mid-word.
REF_MIN_KEEP_SEC = float(os.environ.get("OMNIVOICE_REF_MIN_KEEP_SEC", "3.0"))
STRICT_REF = _env_flag("OMNIVOICE_STRICT_REF")

# --- output ---------------------------------------------------------------
NORMALIZE_OUTPUT = _env_flag("OMNIVOICE_NORMALIZE_OUTPUT")
OUT_TARGET_LUFS = float(os.environ.get("OMNIVOICE_OUT_LUFS", "-20"))
OUT_PEAK_CEILING = float(os.environ.get("OMNIVOICE_OUT_PEAK_DB", "-1.0"))
# Silence appended to every finished clip. Silence-stripping downstream is a
# documented way to lose a final consonant; 0.3s of headroom costs nothing.
OUT_TAIL_PAD = float(os.environ.get("OMNIVOICE_OUT_TAIL_PAD", "0.30"))
# Pull outlier chunks toward the clip's own median level (not to a fixed
# target, which is what makes stitched long-form output jump in volume).
LEVEL_MATCH = _env_flag("OMNIVOICE_LEVEL_MATCH")

# --- provenance & compliance ---------------------------------------------
# EU AI Act Article 50 (enforceable 2 August 2026) obliges the provider of a
# synthetic-audio system to mark its output machine-readably. Off by default
# because it needs an extra package; turn it on before selling into the EU.
# Only ever applied to generated audio, never to a customer's own recording.
WATERMARK = _env_flag("OMNIVOICE_WATERMARK", "0")
WATERMARK_ALPHA = float(os.environ.get("OMNIVOICE_WATERMARK_ALPHA", "1.0"))
# "consent at enrolment, watermark at generation, detect on complaint" — the
# audit log is the third leg, and the one a dispute actually turns on.
AUDIT = _env_flag("OMNIVOICE_AUDIT")
AUDIT_LOG = os.environ.get("OMNIVOICE_AUDIT_LOG",
                           os.path.join(_HERE, "logs", "generations.jsonl"))
# Scripts are the customer's content: hashes by default, full text only on
# request.
AUDIT_TEXT = _env_flag("OMNIVOICE_AUDIT_TEXT", "0")
REQUIRE_CONSENT = _env_flag("OMNIVOICE_REQUIRE_CONSENT", "0")
PREWARM_VOICES = _env_flag("OMNIVOICE_PREWARM_VOICES", "0")

# --- verification ---------------------------------------------------------
VERIFY = _env_flag("OMNIVOICE_VERIFY")
VERIFY_BUDGET_S = float(os.environ.get("OMNIVOICE_VERIFY_BUDGET", "45"))
VERIFY_RETRIES = int(os.environ.get("OMNIVOICE_VERIFY_RETRIES", "1"))
# "fast"   - one ASR pass over the finished clip; drill into chunks only when
#            that pass finds something wrong. A clean job is the common case,
#            and this is what keeps RTF close to unverified.
# "strict" - one ASR pass per chunk, always.
VERIFY_MODE = os.environ.get("OMNIVOICE_VERIFY_MODE", "fast").strip().lower()
# Speaking rate is compared to each voice's OWN baseline, never to a fixed band.
verify.RATE_LOW = float(os.environ.get("OMNIVOICE_RATE_LOW", verify.RATE_LOW))
verify.RATE_HIGH = float(os.environ.get("OMNIVOICE_RATE_HIGH", verify.RATE_HIGH))

# --- generation -----------------------------------------------------------
FORCE_NUM_STEP = os.environ.get("OMNIVOICE_NUM_STEP")
DO_COMPILE = _env_flag("OMNIVOICE_COMPILE", "0")
BATCH = int(os.environ.get("OMNIVOICE_BATCH", "1"))

# --- limits (one place, read everywhere) ----------------------------------
# Two limits, because there are two situations and they are not alike.
#
# SYNCHRONOUS: an HTTP client is holding the connection open. A script long
# enough to take twenty minutes will time out somewhere in the middle no matter
# what we do, so refusing it up front with a usable message is kinder than
# failing halfway. The message names the async endpoint.
MAX_INPUT_CHARS = int(os.environ.get("OMNIVOICE_MAX_INPUT_CHARS", "8000"))
MAX_INPUT_WORDS = int(os.environ.get("OMNIVOICE_MAX_INPUT_WORDS", "1300"))
# QUEUED (the Studio and /api/tts/async): nobody is waiting on a socket. Paste
# a whole book if you like — it is chunked internally, one chunk on the GPU at
# a time, and written to disk when it finishes. The cap here exists only to
# stop a single job from eating RAM: the finished waveform is held in memory
# before it is written, roughly 100 MB per 10,000 characters.
MAX_INPUT_CHARS_ASYNC = int(os.environ.get("OMNIVOICE_MAX_INPUT_CHARS_ASYNC",
                                           "100000"))
MAX_INPUT_WORDS_ASYNC = int(os.environ.get("OMNIVOICE_MAX_INPUT_WORDS_ASYNC",
                                           "18000"))
MAX_CONCURRENCY = int(os.environ.get("OMNIVOICE_MAX_CONCURRENCY", "1"))
QUEUE_WAIT_S = float(os.environ.get("OMNIVOICE_QUEUE_WAIT", "300"))

# --- recovery -------------------------------------------------------------
AUTO_RELOAD = _env_flag("OMNIVOICE_AUTO_RELOAD")
SELFTEST = _env_flag("OMNIVOICE_SELFTEST")
SELFTEST_EVERY = float(os.environ.get("OMNIVOICE_SELFTEST_EVERY", "120"))
SELFTEST_STALE = float(os.environ.get("OMNIVOICE_SELFTEST_STALE", "600"))


# ---------------------------------------------------------------------------
# Text front-end (see textnorm.py — unit-tested without a GPU)
# ---------------------------------------------------------------------------
def normalize_for_tts(text: str, language: Optional[str]) -> Tuple[str, List[str]]:
    return textnorm.normalize_text(text, language, level=NORMALIZE_LEVEL,
                                   lexicon=LEXICON, years=YEAR_STYLE)


def chunk_text(text: str, max_chars: int = MAX_CHARS) -> List[str]:
    return textnorm.chunk_text(text, max_chars, min_chars=MIN_CHUNK_CHARS)


def concat_audio(parts, sr, gap_sec=GAP_SEC):
    return audio_fx.concat_audio(parts, sr, gap_sec)


def check_input_size(text: str, queued: bool = False) -> Optional[str]:
    """One check, two limits — see MAX_INPUT_CHARS above."""
    max_chars = MAX_INPUT_CHARS_ASYNC if queued else MAX_INPUT_CHARS
    max_words = MAX_INPUT_WORDS_ASYNC if queued else MAX_INPUT_WORDS
    n_chars, n_words = len(text or ""), textnorm.word_count(text)
    where = ("Split it into separate projects." if queued else
             "Send it to POST /api/tts/async instead, which queues the work "
             "and has a far higher limit.")
    if n_chars > max_chars:
        return f"text is {n_chars} characters; the limit is {max_chars}. {where}"
    if n_words > max_words:
        return f"text is {n_words} words; the limit is {max_words}. {where}"
    return None


# ---------------------------------------------------------------------------
# Reference audio handling
# ---------------------------------------------------------------------------
def read_audio(path: str) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    return audio_fx.to_mono_float(data), int(sr)


def write_wav(path: str, x: np.ndarray, sr: int) -> None:
    sf.write(path, audio_fx.to_int16(x), sr, subtype="PCM_16")


def _merge_ref_warnings(report: Dict, trim_info: Dict) -> List[str]:
    """Analysis runs before the repair, so drop what the repair just fixed.

    Telling a customer "your reference ends without a pause" *and* "it was cut
    back to the last finished phrase" in the same breath reads like two
    problems. It is one problem and its fix.
    """
    stale = "ends without a pause"
    before = [w for w in report.get("warnings", [])
              if not (trim_info.get("repaired_mid_word") and stale in w)]
    return before + list(trim_info.get("warnings", []))


def prepare_reference(path: str, normalize: bool = True) -> Tuple[str, Dict]:
    """Trim at a real pause, level it, and write a clean temp wav.

    This is the fix for the worst bug in the batch. The old code cut at exactly
    MAX_REF_SEC, which for one 60s reference landed mid-word — and that half
    word was then spoken at the end of 86-90% of the clips made with it.
    """
    info: Dict[str, Any] = {"source": path, "warnings": []}
    try:
        x, sr = read_audio(path)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not read reference audio %s: %s", path, e)
        info["warnings"].append(f"could not read reference audio: {e}")
        return path, info

    info.update(audio_fx.analyze_reference(x, sr, max_sec=MAX_REF_SEC * 3))
    y, trim_info = audio_fx.smart_trim_reference(
        x, sr, max_sec=MAX_REF_SEC, tail_silence_sec=REF_TAIL_SILENCE,
        min_keep_sec=REF_MIN_KEEP_SEC, hard_max_sec=REF_HARD_MAX_SEC)
    info["warnings"] = _merge_ref_warnings(info, trim_info)
    info.update({k: v for k, v in trim_info.items() if k != "warnings"})

    if normalize:
        y, loud = audio_fx.normalize_loudness(
            y, sr, target_lufs=REF_TARGET_LUFS, peak_ceiling_db=OUT_PEAK_CEILING)
        info["reference_loudness"] = loud

    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="omnivoice_ref_")
    os.close(fd)
    try:
        write_wav(tmp, y, sr)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not write prepared reference: %s", e)
        return path, info
    info["prepared_path"] = tmp
    info["sample_rate"] = sr
    return tmp, info


def trim_reference(path: str):
    """Back-compat shim: (path_to_use, duration_seconds, was_trimmed)."""
    prepared, info = prepare_reference(path)
    return prepared, info.get("duration_sec"), bool(info.get("trimmed"))


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
_AUDIT_LOCK = threading.Lock()


def _sha(data) -> str:
    import hashlib as _h
    if isinstance(data, str):
        data = data.encode("utf-8")
    return _h.sha256(data).hexdigest()[:32]


def audit(event: str, **fields) -> None:
    """Append one line per generation and per voice registration.

    A voice-cloning business is one complaint away from having to show who
    made a clip, from which voice, and on whose authority. Reconstructing that
    afterwards is impossible; writing 300 bytes at the time is free.
    """
    if not AUDIT:
        return
    rec = {"ts": datetime.datetime.now(datetime.timezone.utc)
           .isoformat(timespec="seconds"), "event": event}
    rec.update({k: v for k, v in fields.items() if v is not None})
    try:
        os.makedirs(os.path.dirname(AUDIT_LOG) or ".", exist_ok=True)
        line = json.dumps(rec, ensure_ascii=False)
        with _AUDIT_LOCK:
            with open(AUDIT_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:  # noqa: BLE001 - logging must never break a render
        log.warning("could not write the audit log: %s", e)


def audit_generation(text, rep, voice_id=None, owner=None, project=None,
                     wav=None, sr=None, source="api") -> None:
    audit("generation",
          source=source, tenant=owner, voice_id=voice_id, project=project,
          text_sha256=_sha(text or ""), text_chars=len(text or ""),
          text=(text if AUDIT_TEXT else None),
          audio_sha256=(_sha(wav.tobytes()) if wav is not None else None),
          duration_sec=rep.get("audio_sec"), wpm=rep.get("wpm"),
          verified=rep.get("verified"),
          warnings=(rep.get("warnings") or None),
          watermarked=(rep.get("watermark") or {}).get("watermarked"),
          rtf=rep.get("rtf"), seed=rep.get("seed"))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device_map = "cuda"
    dtype = torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    print(f"Loading model from {CHECKPOINT} to CUDA ({torch.cuda.get_device_name(0)}) ...")
else:
    device_map = "cpu"
    dtype = torch.float32
    print(f"CUDA not available -> loading {CHECKPOINT} on CPU (this will be slow) ...")

print(f"Checkpoint: {CHECKPOINT}")

# ---------------------------------------------------------------------------
# Version-tolerant feature use
# ---------------------------------------------------------------------------
# 0.2.1 is the supported baseline (requirements.txt pins it). Detection is kept
# because FlashInfer landed AFTER 0.2.1 and only exists on git main, and because
# an old install should degrade with a clear message rather than a traceback.
#
#   0.2.0  pad_duration / fade_duration      control over the model's own
#                                            fade-in/out and silence padding
#                                            (upstream #194: the built-in fades
#                                            "sometimes add artifact")
#   0.2.0  fixed punctuation handling        upstream #181
#   0.2.1  asr_device                        stop Whisper always landing on
#                                            GPU 0 (upstream PR #224)
#   0.2.1  VoiceClonePrompt.save()/load()    reuse a built voice across
#                                            restarts instead of re-encoding
#   main   enable_flashinfer                 PR #239, "2-2.9x lossless
#                                            speedup"; 2.1x at batch size 1
def _pops_kwarg(fn, name: str) -> bool:
    """Does `fn` reach into **kwargs and take `name` back out by name?"""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):  # pragma: no cover - C funcs, frozen installs
        return False
    return bool(re.search(r"kwargs\.(?:pop|get)\(\s*['\"]%s['\"]" % re.escape(name),
                          src))


def _accepts(fn, name: str) -> bool:
    """Does this callable take `name` as a real, named parameter?

    A bare **kwargs on its own still tells us nothing — anything that swallows
    everything says nothing about what it does. But a callable that pops the
    name straight back out of **kwargs plainly does support it, and this is
    exactly how OmniVoice.from_pretrained is written: every option it has
    gained since 0.1.5 (asr_device among them) arrives that way. Checking only
    the named parameters reports those options as missing, which silently
    disables our own env-var levers for them.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins/C funcs
        return False
    if name in params:
        return True
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return False
    return _pops_kwarg(fn, name)


def _omnivoice_version() -> str:
    try:
        from importlib.metadata import version
        return version("omnivoice")
    except Exception:  # noqa: BLE001
        return "unknown"


MIN_OMNIVOICE = (0, 2, 1)
OMNIVOICE_VERSION = _omnivoice_version()


def _version_tuple(v: str):
    parts = []
    for chunk in v.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts[:3]) or (0,)


OMNIVOICE_OUTDATED = (OMNIVOICE_VERSION != "unknown"
                      and _version_tuple(OMNIVOICE_VERSION) < MIN_OMNIVOICE)
FEATURES = {
    "asr_device": _accepts(OmniVoice.from_pretrained, "asr_device"),
    "flashinfer": _accepts(OmniVoice.from_pretrained, "enable_flashinfer"),
    "pad_duration": _accepts(OmniVoiceGenerationConfig, "pad_duration"),
    "fade_duration": _accepts(OmniVoiceGenerationConfig, "fade_duration"),
}
ASR_DEVICE = os.environ.get("OMNIVOICE_ASR_DEVICE", "").strip() or None
USE_FLASHINFER = _env_flag("OMNIVOICE_FLASHINFER", "0")
# None keeps whatever the model does by default. Set these only if the built-in
# fade is clipping your final consonants.
PAD_DURATION = os.environ.get("OMNIVOICE_PAD_DURATION", "").strip()
FADE_DURATION = os.environ.get("OMNIVOICE_FADE_DURATION", "").strip()
_ATTN = os.environ.get("OMNIVOICE_ATTN", "sdpa")


def _load_model(attn, extras: Optional[Dict[str, Any]] = None):
    kw: Dict[str, Any] = dict(
        device_map=device_map,
        dtype=dtype,
        load_asr=True,
        asr_model_name=ASR_MODEL,
        attn_implementation=attn,
    )
    kw.update(extras or {})
    return OmniVoice.from_pretrained(CHECKPOINT, **kw)


def _load_extras() -> Dict[str, Any]:
    """Only the newer options this installation actually understands."""
    extras: Dict[str, Any] = {}
    if ASR_DEVICE and FEATURES["asr_device"]:
        extras["asr_device"] = ASR_DEVICE
    elif ASR_DEVICE:
        log.warning("OMNIVOICE_ASR_DEVICE needs omnivoice >= 0.2.1 "
                    "(installed: %s) — ignoring", OMNIVOICE_VERSION)
    if USE_FLASHINFER and FEATURES["flashinfer"]:
        extras["enable_flashinfer"] = True
    elif USE_FLASHINFER:
        log.warning(
            "OMNIVOICE_FLASHINFER is set but this omnivoice (%s) does not "
            "accept enable_flashinfer. It landed after 0.2.1 — install from "
            "git main plus flashinfer-python to use it.", OMNIVOICE_VERSION)
    return extras


_EXTRAS = _load_extras()
try:
    model = _load_model(_ATTN, _EXTRAS)
    print(f"Model loaded successfully! (attention: {_ATTN}"
          + (f", {', '.join(sorted(_EXTRAS))}" if _EXTRAS else "") + ")")
except Exception as e:
    # Two things can go wrong: the attention backend, or one of the newer
    # options. Peel them off in that order rather than failing to start.
    log.warning("model load failed (%s); retrying with plain defaults", e)
    try:
        model = _load_model(_ATTN)
        _EXTRAS = {}
        print(f"Model loaded successfully! (attention: {_ATTN}, extras dropped)")
    except Exception:
        if _ATTN != "sdpa":
            model = _load_model("sdpa")
            _EXTRAS = {}
            print("Model loaded successfully! (attention: sdpa)")
        else:
            raise
print(f"omnivoice {OMNIVOICE_VERSION} · features: "
      + (", ".join(k for k, v in FEATURES.items() if v) or "none detected"))
if OMNIVOICE_OUTDATED:
    print("=" * 60)
    print(f"  WARNING: omnivoice {OMNIVOICE_VERSION} is older than the "
          f"{'.'.join(map(str, MIN_OMNIVOICE))} this wrapper targets.")
    print("  Missing: control over the model's own fade/padding (the fade is")
    print("  reported upstream to add artifacts), the punctuation-handling fix,")
    print("  asr_device, and cached voices that survive a restart.")
    print("      venv\\Scripts\\python -m pip install -r requirements.txt")
    print("=" * 60)
sampling_rate = model.sampling_rate

if DO_COMPILE:
    try:
        if hasattr(model, "llm") and model.llm is not None:
            model.llm = torch.compile(model.llm)
            log.info("torch.compile enabled on model.llm")
        else:
            log.warning("torch.compile requested but model.llm not found; skipped")
    except Exception as e:
        log.warning("torch.compile failed (%s); continuing uncompiled", e)


# Single GPU -> one model op at a time.
MODEL_LOCK = threading.RLock()
# Bounds how many generations are in flight at once. Four concurrent callers
# killed the old server in under a minute; a client should not be able to
# oversubscribe the card by sending more requests.
GEN_SLOTS = threading.BoundedSemaphore(max(1, MAX_CONCURRENCY))
HEALTH = gpu_guard.GpuHealth()
READY = gpu_guard.Readiness(stale_after=SELFTEST_STALE)
STARTED_AT = time.time()
METRICS = {"generations": 0, "verify_failures": 0, "verify_skipped": 0,
           "tail_trims": 0, "regenerations": 0, "oom": 0,
           "unknown_params": {}, "idempotent_replays": 0,
           "audio_sec_total": 0.0, "gen_sec_total": 0.0}
METRICS_LOCK = threading.Lock()


def _bump(key: str, n: int = 1) -> None:
    with METRICS_LOCK:
        METRICS[key] = METRICS.get(key, 0) + n


def reload_model() -> None:
    """Last-resort recovery from a poisoned CUDA context.

    Held under MODEL_LOCK so nobody can run against a half-loaded model. Every
    cached voice prompt is invalidated, because the old ones belong to tensors
    that no longer exist.
    """
    global model, sampling_rate
    with MODEL_LOCK:
        log.warning("reloading the model after a fatal CUDA error")
        try:
            old = model
            model = None  # type: ignore
            del old
        except Exception:  # noqa: BLE001
            pass
        gpu_guard.free_cuda()
        model = _load_model(_ATTN if _ATTN == "sdpa" else "sdpa")
        sampling_rate = model.sampling_rate
        _invalidate_voice_prompts()
        log.warning("model reloaded")


_RELOAD_HOOK = reload_model if AUTO_RELOAD else None


def call_model(fn, label="generate"):
    """Every single model call goes through here: one at a time, no autograd,
    memory freed afterwards, OOM handled."""
    def _run():
        with MODEL_LOCK:
            return fn()

    try:
        return gpu_guard.guarded(_run, retries=1, health=HEALTH,
                                 on_reload=_RELOAD_HOOK, label=label)
    except BaseException as exc:
        if gpu_guard.is_oom(exc):
            _bump("oom")
        raise


# ---------------------------------------------------------------------------
# Generation core
# ---------------------------------------------------------------------------
class _Cancelled(Exception):
    """Raised to abort generation when a job is cancelled."""


# Whisper takes 30 s of mel features at a time. Past that, transformers demands
# return_timestamps=True and raises otherwise — and OmniVoice's own
# transcribe() passes no such argument, so it cannot read anything longer.
ASR_WINDOW_SEC = 30.0


def _too_long_for_whisper(path: str) -> bool:
    try:
        info = sf.info(path)
        return info.frames > int(ASR_WINDOW_SEC * info.samplerate)
    except Exception:  # noqa: BLE001 - unreadable header: let the ASR decide
        return False


def _transcribe_path(path: str) -> str:
    """ASR on a file, whatever its length.

    A generated voiceover clip is routinely minutes long, and both the verifier
    and `POST /api/transcribe` (which is how tools/audit_batch.py re-measures a
    finished batch) land here.
    """
    if _too_long_for_whisper(path):
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return _transcribe_long(x, sr)
    return call_model(lambda: model.transcribe(path), label="transcribe") or ""


def _transcribe_short(x: np.ndarray, sr: int) -> str:
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="omnivoice_vfy_")
    os.close(fd)
    try:
        write_wav(tmp, x, sr)
        return _transcribe_path(tmp)
    finally:
        try:
            os.remove(tmp)
        except Exception:  # noqa: BLE001
            pass


def _transcribe_long(x: np.ndarray, sr: int) -> str:
    """ASR for audio past Whisper's 30 s input window.

    This is the whole long-form case: a two-minute clip is exactly what the
    verifier exists for, and without this it could not be read at all — the
    transcription raised, verification was skipped, and the clip shipped with
    "could not be verified" on it. Ask the pipeline for long-form directly
    where we can; otherwise walk the audio in windows ourselves.
    """
    pipe = getattr(model, "_asr_pipe", None)
    if pipe is not None:
        try:
            out = call_model(
                lambda: pipe({"array": np.asarray(np.squeeze(x), dtype=np.float32),
                              "sampling_rate": sr}, return_timestamps=True),
                label="transcribe-long")
            text = (out or {}).get("text", "")
            if text.strip():
                return text.strip()
            log.warning("long-form ASR returned nothing; falling back to windows")
        except Exception as e:  # noqa: BLE001
            log.warning("long-form ASR failed (%s); falling back to windows", e)

    step = int(ASR_WINDOW_SEC * sr)
    said = []
    for start in range(0, len(x), step):
        piece = x[start:start + step]
        if len(piece) < int(0.2 * sr):  # a sliver of tail carries no words
            continue
        said.append(_transcribe_short(piece, sr))
    return " ".join(p for p in said if p).strip()


def _transcribe_audio(x: np.ndarray, sr: int) -> str:
    """ASR on an in-memory waveform (used by the verifier)."""
    if len(np.squeeze(x)) > int(ASR_WINDOW_SEC * sr):
        return _transcribe_long(x, sr)
    return _transcribe_short(x, sr)


def _gen_core(*args, **kwargs):
    # The GPU lock is taken per model call (per chunk / per prompt / per
    # transcription), NOT around the whole job — so an upload-transcription can
    # run between chunks while a long job is processing.
    return _gen_core_impl(*args, **kwargs)


def _gen_core_impl(
    text,
    language,
    ref_audio,
    instruct,
    num_step,
    guidance_scale,
    denoise,
    speed,
    duration,
    preprocess_prompt,
    postprocess_output,
    mode,
    ref_text=None,
    cancel_check=None,
    prebuilt_prompt=None,
    report=None,
    seed=None,
    baseline_wpm=None,
):
    """Synthesize `text`. Returns ((sr, int16 waveform), status) as it always
    has; everything new is written into `report` so no caller breaks."""
    rep: Dict[str, Any] = report if report is not None else {}
    rep.setdefault("warnings", [])
    rep.setdefault("notes", [])
    rep.setdefault("verified", False)

    if not text or not text.strip():
        return None, "Please enter the text to synthesize."

    too_big = check_input_size(text, queued=True)
    if too_big:
        return None, f"Error: {too_big}"

    notes: List[str] = rep["notes"]
    warnings: List[str] = rep["warnings"]

    if FORCE_NUM_STEP:
        num_step = FORCE_NUM_STEP
        notes.append(f"steps forced to {FORCE_NUM_STEP}")

    syn_text, norm_notes = normalize_for_tts(text.strip(), language)
    notes.extend(norm_notes)
    rep["normalized_text"] = syn_text
    rep["original_text"] = text.strip()

    gpu_guard.seed_everything(seed)
    if seed is not None:
        rep["seed"] = int(seed)

    cfg_kw: Dict[str, Any] = dict(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )
    # 0.2.0+: the model's own fade/pad. Upstream #194 reports the built-in fade
    # "sometimes adds artifact", and #204/#245 are clips losing their last word.
    if PAD_DURATION and FEATURES["pad_duration"]:
        cfg_kw["pad_duration"] = float(PAD_DURATION)
    if FADE_DURATION and FEATURES["fade_duration"]:
        cfg_kw["fade_duration"] = float(FADE_DURATION)
    gen_config = OmniVoiceGenerationConfig(**cfg_kw)

    lang = language if (language and language != "Auto") else None
    kw: Dict[str, Any] = dict(text=syn_text, language=lang, generation_config=gen_config)
    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    instruct_val = instruct.strip() if (instruct and instruct.strip()) else None

    duration_set = duration is not None and float(duration) > 0
    if CHUNK and not duration_set and len(syn_text) > MAX_CHARS:
        chunks = chunk_text(syn_text, MAX_CHARS)
    else:
        chunks = [syn_text]
    rep["chunks"] = len(chunks)

    # ---- the voice ------------------------------------------------------
    clone_prompt = None
    if prebuilt_prompt is not None:
        clone_prompt = prebuilt_prompt
    elif mode == "clone":
        if not ref_audio:
            return None, "Please upload a reference audio."
        ref_path, ref_info = prepare_reference(ref_audio)
        warnings.extend(ref_info.get("warnings", []))
        if ref_info.get("trimmed"):
            notes.append(
                ("reference cut back from a mid-word ending to "
                 if ref_info.get("repaired_mid_word") else "reference trimmed to ")
                + f"{ref_info.get('cut_sec')}s"
                + (" at a pause" if ref_info.get("cut_at_pause") else "")
                + (f" (+{ref_info['overshoot_sec']}s to finish the sentence)"
                   if ref_info.get("overshoot_sec") else ""))
            if ref_text:
                # The supplied transcript describes the untrimmed clip, so it no
                # longer matches the audio — and a mismatched ref_text is the
                # main cause of reference words leaking into every clip.
                warnings.append("the supplied reference transcript was replaced: "
                                "the clip had to be trimmed, so the transcript no "
                                "longer matched the audio")
                ref_text = None
        try:
            clone_prompt = call_model(
                lambda: model.create_voice_clone_prompt(ref_audio=ref_path,
                                                        ref_text=ref_text),
                label="clone_prompt")
        except Exception as e:
            return None, f"Error creating voice prompt: {type(e).__name__}: {e}"
        if not ref_text:
            notes.append("ref text auto-transcribed (Whisper)")

    # ---- generation -----------------------------------------------------
    def _generate(texts, prompt, inst, batch_size):
        out_parts: List[np.ndarray] = []
        i, bs = 0, max(1, batch_size)
        while i < len(texts):
            if cancel_check and cancel_check():
                raise _Cancelled()
            group = texts[i:i + bs]
            bkw: Dict[str, Any] = dict(kw)
            bkw["text"] = group
            if prompt is not None:
                bkw["voice_clone_prompt"] = prompt
            if inst:
                bkw["instruct"] = inst
            raw = None
            try:
                raw = call_model(lambda: model.generate(**bkw), label="generate")
                for a in raw:
                    out_parts.append(_as_cpu_float(a))
                i += bs
            except torch.cuda.OutOfMemoryError:
                gpu_guard.free_cuda()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                log.warning("CUDA OOM -> reducing batch size to %d", bs)
            finally:
                # Named delete: deleting out of locals() looks like cleanup and
                # does nothing at all.
                del raw
        return out_parts

    t0 = time.perf_counter()
    try:
        if clone_prompt is not None or len(chunks) == 1:
            parts = _generate(chunks, clone_prompt, instruct_val, BATCH)
        else:
            # Design mode + multiple chunks: lock the voice from the first
            # chunk so the timbre cannot drift.
            parts = _generate([chunks[0]], None, instruct_val, 1)
            try:
                first = parts[0]
                clone_prompt = call_model(
                    lambda: model.create_voice_clone_prompt(
                        ref_audio=(torch.from_numpy(first).float(), sampling_rate),
                        ref_text=chunks[0]),
                    label="clone_prompt")
            except Exception as e:
                log.warning("Could not lock design voice across chunks: %s", e)
            parts += _generate(chunks[1:], clone_prompt, None, BATCH)

        # ---- did it say what it was sent? -------------------------------
        if VERIFY and not duration_set:
            parts = _verify_stage(
                chunks, parts, rep,
                regen=lambda t: _generate([t], clone_prompt, instruct_val, 1)[0],
                cancel_check=cancel_check)

        waveform_f, join_info = audio_fx.join_chunks(
            parts, sampling_rate, gap_sec=GAP_SEC, level_match=LEVEL_MATCH,
            tail_pad_sec=OUT_TAIL_PAD)
        rep["join"] = join_info
    except _Cancelled:
        return None, "Cancelled"
    except Exception as e:
        log.exception("generation failed")
        return None, f"Error: {type(e).__name__}: {e}"
    finally:
        gpu_guard.free_cuda()

    gen_time = time.perf_counter() - t0
    _bump("generations")

    if len(chunks) > 1:
        notes.append(f"{len(chunks)} chunks" + (f", batch {BATCH}" if BATCH > 1 else ""))

    # ---- loudness -------------------------------------------------------
    if NORMALIZE_OUTPUT and len(waveform_f):
        waveform_f, loud = audio_fx.normalize_loudness(
            waveform_f, sampling_rate, target_lufs=OUT_TARGET_LUFS,
            peak_ceiling_db=OUT_PEAK_CEILING)
        rep["loudness"] = loud
        notes.append(f"{loud['out_lufs']:.1f} LUFS")
        if not loud.get("met_target"):
            why = {
                "peak_ceiling": "the peak ceiling was reached first",
                "gain_cap": "the source was too quiet to lift safely without "
                            "raising its background noise",
                "silence": "there was almost no signal to measure",
            }.get(loud.get("limited_by"), "the target could not be reached")
            warnings.append(
                f"loudness came out at {loud['out_lufs']:.1f} LUFS instead of "
                f"{OUT_TARGET_LUFS:.0f} — {why}")
    else:
        rep["loudness"] = {"in_lufs": round(audio_fx.lufs(waveform_f, sampling_rate), 2)}

    join_info = rep.get("join") or {}
    audio_dur = len(waveform_f) / sampling_rate if sampling_rate else 0.0
    rtf = (gen_time / audio_dur) if audio_dur > 0 else 0.0
    spoken_words = textnorm.word_count(syn_text)
    # Rate is measured over speech, not over the trailing padding we added.
    measured_wpm = audio_fx.wpm(spoken_words,
                                join_info.get("speech_sec") or audio_dur)
    corrections = [c for c in join_info.get("level_corrections", []) if c]
    if corrections:
        notes.append(f"levelled {len(corrections)} chunk(s) "
                     f"(max {max(abs(c) for c in corrections):.1f} dB)")

    rep.update({
        "gen_sec": round(gen_time, 2), "audio_sec": round(audio_dur, 2),
        "rtf": round(rtf, 3), "wpm": measured_wpm, "words": spoken_words,
        "baseline_wpm": baseline_wpm,
    })
    # Speaking rate is a metric, never a gate: a documentary narrator runs ~100
    # wpm and an ads voice ~210, so any fixed band both rejects good clips and
    # passes broken ones. Compared against its own voice, it means something.
    rate_note = verify.rate_warning(measured_wpm, baseline_wpm)
    if rate_note:
        warnings.append(rate_note)
    if join_info.get("ends_abruptly"):
        warnings.append("the clip ends at full volume — it may have been cut short")

    notes.append(f"{gen_time:.1f}s for {audio_dur:.1f}s audio · RTF {rtf:.3f}"
                 f" · {measured_wpm:.0f} wpm")
    # Totals, so /api/metrics can report the RTF across a whole day rather than
    # the average of per-clip ratios (short clips would dominate that).
    with METRICS_LOCK:
        METRICS["audio_sec_total"] = round(
            METRICS.get("audio_sec_total", 0.0) + audio_dur, 2)
        METRICS["gen_sec_total"] = round(
            METRICS.get("gen_sec_total", 0.0) + gen_time, 2)

    if WATERMARK and len(waveform_f):
        # Generated audio only. A customer's own recording is their real voice
        # and must never be stamped as synthetic.
        waveform_f, wm = watermark.embed(waveform_f, sampling_rate,
                                         alpha=WATERMARK_ALPHA)
        rep["watermark"] = wm
        if wm.get("watermarked"):
            notes.append("watermarked")
        elif wm.get("error"):
            warnings.append(f"this clip is NOT watermarked: {wm['error']}")

    waveform = audio_fx.to_int16(waveform_f)
    status = "Done. (" + "; ".join(notes) + ")"
    if warnings:
        status += " ⚠ " + "; ".join(warnings)
    return (sampling_rate, waveform), status


def _as_cpu_float(a) -> np.ndarray:
    """Get a result off the GPU immediately. A CUDA tensor parked in the jobs
    dict keeps its whole allocation alive for as long as the entry exists."""
    if torch is not None and isinstance(a, torch.Tensor):
        a = a.detach().to("cpu", copy=True).float().numpy()
    return np.asarray(a, dtype=np.float32).reshape(-1)


def _verify_stage(chunks, parts, rep, regen, cancel_check=None):
    """Screen the whole clip in one ASR pass; drill into chunks only on failure.

    Verification is by far the biggest cost this wrapper adds, and RTF is the
    number the product gets judged on. One pass over the finished clip answers
    "did it say what it was sent?" for the common case — which is a clean job.
    Per-chunk passes buy *locality*, and locality is only worth paying for once
    something is actually wrong.

    A five-chunk job goes from five extra ASR passes to one.
    """
    if VERIFY_MODE == "strict" or len(chunks) <= 1:
        return _verify_and_repair(chunks, parts, rep, regen, cancel_check)

    started = time.perf_counter()
    screen, _info = audio_fx.join_chunks(parts, sampling_rate, gap_sec=GAP_SEC,
                                         level_match=False, tail_pad_sec=0.0)
    try:
        heard = _transcribe_audio(screen, sampling_rate)
    except Exception as e:  # noqa: BLE001
        log.warning("verification screen failed: %s", e)
        _bump("verify_skipped")
        rep["warnings"].append("this clip could not be verified")
        rep["verified"] = False
        return parts
    finally:
        del screen

    diff = verify.word_diff(" ".join(chunks), heard)
    rep["verify_passes"] = 1
    if verify.clean(diff):
        rep["verified"] = True
        rep["verify_mode"] = "whole-clip"
        rep["verify_sec"] = round(time.perf_counter() - started, 2)
        rep["word_accuracy"] = diff["word_accuracy"]
        note = verify.pronunciation_note(diff)
        if note:
            rep["warnings"].append(note)
        return parts

    # Something is wrong. Now it is worth finding out exactly where.
    log.info("whole-clip check failed (%s) — checking each chunk",
             verify.describe(diff))
    rep["verify_mode"] = "per-chunk"
    out = _verify_and_repair(chunks, parts, rep, regen, cancel_check)
    rep["verify_passes"] = 1 + len(chunks)
    rep["verify_sec"] = round(time.perf_counter() - started, 2)
    return out


def _verify_and_repair(chunks, parts, rep, regen, cancel_check=None):
    """Transcribe each chunk back, diff it against the script, and fix what can
    be fixed.

    Two failure shapes, two answers:
      * extra words *after* the script (reference bleed) -> cut the trailing
        artefact, which is cheap and leaves the real speech untouched;
      * words missing (repetition collapse) -> regenerate the chunk, because
        no amount of editing puts back something that was never said.

    The verifier has a time budget. If it runs out the audio still goes out,
    marked unverified — a checker must never become the reason a customer's
    batch stalls.
    """
    started = time.perf_counter()
    checked = 0
    problems: List[str] = []
    out = list(parts)
    for idx, (chunk, wave) in enumerate(zip(chunks, parts)):
        if cancel_check and cancel_check():
            raise _Cancelled()
        if time.perf_counter() - started > VERIFY_BUDGET_S:
            _bump("verify_skipped")
            rep["warnings"].append(
                "verification ran out of time — this clip was not checked")
            rep["verified"] = False
            return out
        try:
            heard = _transcribe_audio(wave, sampling_rate)
        except Exception as e:  # noqa: BLE001
            log.warning("verification transcription failed: %s", e)
            _bump("verify_skipped")
            rep["warnings"].append("this clip could not be verified")
            rep["verified"] = False
            return out
        diff = verify.word_diff(chunk, heard)
        checked += 1

        # First, the cheap repair: if the ONLY problem is words after the end of
        # the script, cut the artefact. The real speech is untouched, and it
        # costs nothing next to a re-generation.
        repaired = False
        if verify.only_tail_is_wrong(diff):
            fixed, removed = audio_fx.remove_tail_after_gap(wave, sampling_rate)
            if removed:
                _bump("tail_trims")
                log.info("trimmed %.2fs of leaked reference audio from chunk %d "
                         "(%s)", removed, idx + 1, " ".join(diff["tail_inserted"]))
                wave, out[idx], repaired = fixed, fixed, True
                problems.append(
                    f"removed {removed:.1f}s of leftover reference audio "
                    f"(\"{' '.join(diff['tail_inserted'][:4])}\")")

        # Then the expensive one: words that were never spoken cannot be edited
        # back in, so the chunk has to be made again.
        attempts = 0
        # `worth_regenerating`, not `passed`: a proper noun the transcriber
        # fumbled is not worth a second full generation, and the retry would be
        # judged by the same fallible transcriber anyway.
        while (not repaired and verify.worth_regenerating(diff)
               and attempts < VERIFY_RETRIES):
            _bump("regenerations")
            attempts += 1
            try:
                candidate = regen(chunk)
            except Exception as e:  # noqa: BLE001
                log.warning("regeneration failed: %s", e)
                break
            try:
                heard2 = _transcribe_audio(candidate, sampling_rate)
            except Exception:  # noqa: BLE001
                break
            diff2 = verify.word_diff(chunk, heard2)
            better = (len(diff2["missing"]) + len(diff2["extra"])
                      < len(diff["missing"]) + len(diff["extra"]))
            if better:
                wave, diff = candidate, diff2
                out[idx] = candidate
            if not verify.worth_regenerating(diff):
                break
            if verify.only_tail_is_wrong(diff):
                fixed, removed = audio_fx.remove_tail_after_gap(wave, sampling_rate)
                if removed:
                    _bump("tail_trims")
                    wave, out[idx], repaired = fixed, fixed, True
                    problems.append(
                        f"removed {removed:.1f}s of leftover reference audio "
                        f"(\"{' '.join(diff['tail_inserted'][:4])}\")")

        if not repaired and not verify.clean(diff):
            _bump("verify_failures")
            problems.append(f"chunk {idx + 1}: {verify.describe(diff)}")
            rep.setdefault("diffs", []).append(
                {"chunk": idx + 1, "missing": diff["missing"],
                 "extra": diff["extra"], "misheard": diff["misheard"],
                 "dropped": diff["hard_dropped"],
                 "inserted": diff["hard_inserted"],
                 "word_accuracy": diff["word_accuracy"]})
        else:
            note = verify.pronunciation_note(diff)
            if note and note not in problems:
                problems.append(note)

    rep["verified"] = checked > 0
    rep["verify_sec"] = round(time.perf_counter() - started, 2)
    if problems:
        rep["warnings"].extend(problems)
    return out


# ---------------------------------------------------------------------------
# Auto-transcribe reference audio on upload (fills the Reference Text box).
# ---------------------------------------------------------------------------
def transcribe_reference(ref_audio):
    if not ref_audio:
        return "", "Upload a reference clip — it will be auto-transcribed here."
    ref_path, info = prepare_reference(ref_audio)
    try:
        text = _transcribe_path(ref_path)
    except Exception as e:
        return "", f"Auto-transcribe failed: {type(e).__name__}: {e}"
    dur = info.get("cut_sec") or info.get("duration_sec")
    msg = f"Transcribed ~{dur:.1f}s reference." if dur else "Transcribed reference."
    if info.get("warnings"):
        msg += " ⚠ " + " ".join(info["warnings"])
    return text, msg


# ---------------------------------------------------------------------------
# Voiceover Studio — project queue (processed one at a time by a worker thread)
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get("OMNIVOICE_OUTPUT_DIR", os.path.join(_HERE, "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
_OUT_DIR = OUTPUT_DIR
_SHUTDOWN = {"on": False, "fired": False}

JOBS = []  # list of dicts: id, project, status, info, file, params
JOBS_LOCK = threading.Lock()
JOB_Q: "queue_mod.Queue[int]" = queue_mod.Queue()
_JOB_SEQ = 0


def set_output_dir(path):
    global _OUT_DIR
    p = (path or "").strip().strip('"')
    if not p:
        return f"📂 Saving voices to: {_OUT_DIR}"
    try:
        os.makedirs(p, exist_ok=True)
        _OUT_DIR = p
        extra = ""
        if os.path.normpath(p) != os.path.normpath(OUTPUT_DIR):
            extra = ("  ·  files save here on disk; in-browser downloads only "
                     "work for folders listed in OMNIVOICE_ALLOWED_PATHS")
        return f"📂 Voices will auto-save to: {p}{extra}"
    except Exception as e:
        return f"❌ Can't use that folder: {e}"


def set_shutdown(on):
    _SHUTDOWN["on"] = bool(on)
    _SHUTDOWN["fired"] = False
    return ("🛑 PC will shut down ~60s after the queue finishes "
            "(cancel with `shutdown /a` on Windows)." if on
            else "Shutdown-on-complete is OFF.")


def _shutdown_command() -> Optional[str]:
    if sys.platform.startswith("win"):
        return 'shutdown /s /t 60 /c "OmniVoice: queue complete"'
    if sys.platform == "darwin":
        return "sudo shutdown -h +1"
    return "shutdown -h +1"


def _maybe_shutdown():
    if not _SHUTDOWN["on"] or _SHUTDOWN["fired"]:
        return
    with JOBS_LOCK:
        pending = any(j["status"] in ("Queued", "Processing", "Cancelling…")
                      for j in JOBS)
    if not pending:
        _SHUTDOWN["fired"] = True
        cmd = _shutdown_command()
        log.warning("Queue complete -> shutting down in 60s (%s)", cmd)
        try:
            os.system(cmd)
        except Exception as e:  # noqa: BLE001
            log.warning("shutdown command failed: %s", e)


def _safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^\w\- ]+", "", name).strip().replace(" ", "_")
    return name or "voiceover"


def _unique_output_path(project: str) -> str:
    base = _safe_name(project)
    path = os.path.join(_OUT_DIR, base + ".wav")
    i = 2
    while os.path.exists(path):
        path = os.path.join(_OUT_DIR, f"{base}_{i}.wav")
        i += 1
    return path


def _find_job(job_id):
    return next((j for j in JOBS if j["id"] == job_id), None)


def _is_cancelled(job_id) -> bool:
    with JOBS_LOCK:
        j = _find_job(job_id)
        return bool(j and j.get("cancel"))


def _worker_loop():
    while True:
        job_id = JOB_Q.get()
        try:
            with JOBS_LOCK:
                job = _find_job(job_id)
                cancelled = bool(job and job.get("cancel"))
            if job is None:
                continue
            if cancelled:
                with JOBS_LOCK:
                    job["status"] = "Cancelled"
                continue
            with JOBS_LOCK:
                job["status"] = "Processing"
            p = job["params"]
            prebuilt = p.get("clone_prompt")
            mode = "clone" if (p.get("ref_audio") or prebuilt) else "design"
            rep: Dict[str, Any] = {}
            with GEN_SLOTS:
                out, status = _gen_core(
                    p.get("script", ""), p.get("language"), p.get("ref_audio"),
                    p.get("instruct"),
                    p.get("num_step", 16), p.get("guidance_scale", 2.0),
                    p.get("denoise", True),
                    p.get("speed", 1.0), p.get("duration"),
                    p.get("preprocess", True), p.get("postprocess", True),
                    mode=mode, ref_text=p.get("ref_text"),
                    cancel_check=lambda jid=job_id: _is_cancelled(jid),
                    prebuilt_prompt=prebuilt, report=rep,
                    seed=p.get("seed"), baseline_wpm=p.get("baseline_wpm"),
                )
            with JOBS_LOCK:
                job["report"] = rep
                job["warnings"] = rep.get("warnings", [])
                if status == "Cancelled":
                    job["status"], job["info"] = "Cancelled", ""
                elif out is None:
                    job["status"], job["info"] = "Error", status
                else:
                    sr, wav = out
                    path = _unique_output_path(job["project"])
                    sf.write(path, wav, sr)
                    job["status"] = "Done"
                    job["info"] = (status.split(" ⚠ ")[0]
                                   .replace("Done. ", "").strip("() "))
                    job["file"] = path
                    audit_generation(p.get("script", ""), rep,
                                     voice_id=p.get("voice_id"),
                                     owner=p.get("owner"),
                                     project=job["project"], wav=wav, sr=sr,
                                     source="queue")
        except Exception as e:  # noqa: BLE001 - keep the worker alive
            log.exception("worker failed on job %s", job_id)
            with JOBS_LOCK:
                job = _find_job(job_id)
                if job is not None:
                    job["status"], job["info"] = "Error", f"{type(e).__name__}: {e}"
        finally:
            JOB_Q.task_done()
            _maybe_shutdown()


threading.Thread(target=_worker_loop, daemon=True).start()


_STATUS_ICON = {
    "Queued": "⏳", "Processing": "🔊", "Done": "✅", "Error": "❌",
    "Cancelled": "🚫", "Cancelling…": "⏹️",
}
_BADGE_CLASS = {
    "Queued": "badge-queued", "Processing": "badge-processing",
    "Cancelling…": "badge-processing", "Done": "badge-done",
    "Error": "badge-error", "Cancelled": "badge-cancelled",
}


def _download_files():
    with JOBS_LOCK:
        files = [j["file"] for j in JOBS if j["status"] == "Done" and j["file"]]
    return [f for f in files if os.path.exists(f)]


def _job_choices():
    with JOBS_LOCK:
        return [(f"#{j['id']} · {j['project']}  [{j['status']}]", j["id"]) for j in JOBS]


def _picker_update(cur=None):
    choices = _job_choices()
    valid = cur if (cur is not None and any(v == cur for _, v in choices)) else None
    return gr.update(choices=choices, value=valid)


def _gpu_line() -> str:
    snap = gpu_guard.snapshot()
    if snap.get("device") != "cuda":
        return '<div class="vq-gpu">CPU mode — generation will be slow.</div>'
    bits = [f"{snap.get('name', 'GPU')}"]
    if "free_mb" in snap and "total_mb" in snap:
        used = snap["total_mb"] - snap["free_mb"]
        bits.append(f"VRAM {used:.0f} / {snap['total_mb']:.0f} MB")
    if snap.get("fragmentation_mb", 0) > 512:
        bits.append(f"⚠ {snap['fragmentation_mb']:.0f} MB fragmented")
    if not HEALTH.ok:
        bits.append("⚠ generation is failing — restart the app")
    return '<div class="vq-gpu">' + html_mod.escape(" · ".join(bits)) + "</div>"


def _jobs_html():
    with JOBS_LOCK:
        jobs = list(JOBS)
        counts = {}
        for j in jobs:
            counts[j["status"]] = counts.get(j["status"], 0) + 1

    pill_defs = [("Queued", "⏳"), ("Processing", "🔊"), ("Done", "✅"),
                 ("Error", "❌"), ("Cancelled", "🚫")]
    pills = "".join(
        f'<span class="vq-pill">{ico} {counts.get(st, 0)} {st}</span>'
        for st, ico in pill_defs
    )
    # RTF is the number this gets judged on, so it belongs in the header rather
    # than buried at the end of a line that gets ellipsised away.
    rtfs = [(j.get("report") or {}).get("rtf") for j in jobs
            if j["status"] == "Done"]
    rtfs = [r for r in rtfs if r]
    if rtfs:
        avg = sum(rtfs) / len(rtfs)
        cls = "vq-pill-good" if avg < 1.0 else "vq-pill-slow"
        pills += (f'<span class="vq-pill {cls}">⚡ avg RTF {avg:.3f}</span>'
                  f'<span class="vq-pill {cls}">best {min(rtfs):.3f}</span>')
    header = f'<div class="vq-pills">{pills}</div>' + _gpu_line()

    if not jobs:
        return header + ('<div class="vq-empty">🎙️ No projects yet — fill the form '
                         'on the left and click <b>Add to Queue</b>.</div>')

    cards = []
    for j in jobs:
        status = j["status"]
        cls = _BADGE_CLASS.get(status, "badge-queued")
        icon = _STATUS_ICON.get(status, "")
        name = html_mod.escape(j["project"])
        rep = j.get("report") or {}
        script_preview = html_mod.escape((j["params"]["script"] or "")[:60])
        cloned = bool(j["params"].get("clone_prompt"))
        voice = "🎤 cloned" if cloned else "🎛️ designed"
        # The timing note moves to its own uncropped row below; keep the rest.
        notes = "; ".join(n for n in rep.get("notes", []) if "RTF" not in n)
        detail = notes or (f"{script_preview}…" if script_preview else "")
        if not detail and j["info"]:
            detail = j["info"]
        sub = f"{voice} · {html_mod.escape(detail)}" if detail else voice

        chips = []
        rtf = rep.get("rtf")
        if rtf:
            # < 1.0 means faster than real time.
            chips.append(f'<span class="vq-chip '
                         f'{"vq-chip-good" if rtf < 1.0 else "vq-chip-slow"}">'
                         f'⚡ RTF {rtf:.3f}</span>')
        if rep.get("audio_sec"):
            chips.append(f'<span class="vq-chip">🕑 {rep["audio_sec"]:.1f}s audio</span>')
        if rep.get("gen_sec"):
            chips.append(f'<span class="vq-chip">⚙️ {rep["gen_sec"]:.1f}s to make</span>')
        if rep.get("wpm"):
            chips.append(f'<span class="vq-chip">🗣️ {rep["wpm"]:.0f} wpm</span>')
        _lufs = (rep.get("loudness") or {}).get("out_lufs")
        if _lufs is not None:
            chips.append(f'<span class="vq-chip">🔊 {_lufs:.1f} LUFS</span>')
        if rep.get("chunks", 0) > 1:
            chips.append(f'<span class="vq-chip">🧩 {rep["chunks"]} chunks</span>')
        metrics_line = (f'<div class="vq-metrics">{"".join(chips)}</div>'
                        if chips else "")
        ref_used = html_mod.escape((j.get("ref_used") or "")[:60])
        ref_line = (f'<div class="vq-ref">🗣️ voice: "{ref_used}…"</div>'
                    if ref_used else "")
        warn_line = ""
        if j.get("warnings"):
            txt = html_mod.escape("; ".join(j["warnings"])[:220])
            warn_line = f'<div class="vq-warn">⚠ {txt}</div>'
        elif status == "Done" and (j.get("report") or {}).get("verified"):
            warn_line = '<div class="vq-good">✓ checked against your script</div>'
        dl = ""
        if status == "Done" and j["file"]:
            fname = html_mod.escape(os.path.basename(j["file"]))
            dl = f'<div class="vq-file">💾 {fname}</div>'
        cards.append(
            f'<div class="vq-card {cls}">'
            f'<div class="vq-left">'
            f'<span class="vq-id">#{j["id"]}</span>'
            f'<div class="vq-meta"><div class="vq-name">{name}</div>'
            f'<div class="vq-sub">{sub}</div>'
            f'{metrics_line}{ref_line}{warn_line}{dl}</div>'
            f'</div>'
            f'<div class="vq-right">'
            f'<span class="vq-badge {cls}">{icon} {status}</span>'
            f'</div></div>'
        )
    return header + '<div class="vq-board">' + "".join(cards) + "</div>"


def _ui_state(msg="", cur=None):
    return _jobs_html(), _download_files(), _picker_update(cur), msg


# ---------------------------------------------------------------------------
# Voice library — save a clone under a name, reuse it later (UI + API share it).
# ---------------------------------------------------------------------------
VOICES_DIR = os.environ.get("OMNIVOICE_VOICES_DIR", os.path.join(_HERE, "voices"))
os.makedirs(VOICES_DIR, exist_ok=True)
VOICES_INDEX = os.path.join(VOICES_DIR, "index.json")
VOICES: Dict[str, Dict[str, Any]] = {}
VOICES_LOCK = threading.Lock()
NO_SAVED_VOICE = "— none —"

_PERSISTED_VOICE_FIELDS = ("path", "ref_text", "prompt_version",
                           "baseline_wpm", "quality_score",
                           "warnings", "owner", "created", "duration_sec",
                           "lufs", "snr_db",
                           # Who this voice belongs to and on what authority.
                           # Consent is the deciding factor in every framework
                           # that now covers voice cloning, and it has to live
                           # with the voice, not in somebody's inbox.
                           "speaker_name", "consent", "consent_ref")


def _save_voice_index():
    with VOICES_LOCK:
        data = {n: {k: v.get(k) for k in _PERSISTED_VOICE_FIELDS if k in v}
                for n, v in VOICES.items()}
    tmp = VOICES_INDEX + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, VOICES_INDEX)     # atomic: a crash never truncates it
    except Exception as e:  # noqa: BLE001
        log.warning("Could not write voices index: %s", e)


def _load_voice_index():
    """Voices must survive a restart — an OOM restart that loses every
    customer's voice_id breaks every client at once."""
    if not os.path.exists(VOICES_INDEX):
        return
    try:
        with open(VOICES_INDEX, encoding="utf-8") as f:
            data = json.load(f)
        for name, meta in data.items():
            p = meta.get("path")
            if p and os.path.exists(p):
                entry = {k: meta.get(k) for k in _PERSISTED_VOICE_FIELDS
                         if k in meta}
                entry.setdefault("ref_text", "")
                entry["path"] = p
                entry["prompt"] = None          # rebuilt lazily on first use
                VOICES[name] = entry
            else:
                log.warning("voice '%s' points at a missing file (%s) — skipped",
                            name, p)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not load voices index: %s", e)


def _invalidate_voice_prompts():
    """After a model reload the cached prompts belong to tensors that no longer
    exist. Drop them; they rebuild from the stored clips."""
    with VOICES_LOCK:
        for v in VOICES.values():
            v["prompt"] = None
    _VOICE_CACHE.update(key=None, prompt=None, ref_text="")


# --- built-voice persistence (omnivoice 0.2.1+) ---------------------------
# Re-encoding a reference on every restart costs 50-200 ms per voice for a
# result that has not changed since it was registered — and on this server
# "restart" also means "after every OOM recovery". 0.2.1 added save/load for a
# built prompt; where it exists we use it, and where it does not we rebuild.
def _prompt_path(name: str) -> str:
    return os.path.join(VOICES_DIR, _safe_name(name) + ".omniprompt")


def _save_prompt(name: str, prompt) -> bool:
    saver = getattr(prompt, "save", None)
    if not callable(saver):
        return False
    try:
        saver(_prompt_path(name))
        with VOICES_LOCK:
            if name in VOICES:
                VOICES[name]["prompt_version"] = OMNIVOICE_VERSION
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("could not cache the built voice '%s': %s", name, e)
        return False


def _load_prompt(name: str):
    """Return a previously built prompt, or None to rebuild it."""
    path = _prompt_path(name)
    if not os.path.exists(path):
        return None
    with VOICES_LOCK:
        stored = (VOICES.get(name) or {}).get("prompt_version")
    if stored and stored != OMNIVOICE_VERSION:
        # A prompt built by a different library version is not worth trusting.
        log.info("voice '%s' was built on omnivoice %s (now %s) — rebuilding",
                 name, stored, OMNIVOICE_VERSION)
        return None
    try:
        from omnivoice import VoiceClonePrompt
    except Exception:  # noqa: BLE001 - older versions do not export it
        return None
    loader = getattr(VoiceClonePrompt, "load", None)
    if not callable(loader):
        return None
    try:
        prompt = loader(path)
        log.info("voice '%s' restored from cache (no re-encode)", name)
        return prompt
    except Exception as e:  # noqa: BLE001
        log.warning("cached voice '%s' could not be loaded (%s) — rebuilding",
                    name, e)
        return None


def _unique_voice_name(base):
    base = _safe_name(base)
    with VOICES_LOCK:
        existing = set(VOICES.keys())
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def _build_prompt(ref_audio, ref_text):
    """Build a voice clone prompt now and return (prompt, transcript_used)."""
    ref_path, _info = prepare_reference(ref_audio)
    p = call_model(
        lambda: model.create_voice_clone_prompt(ref_audio=ref_path,
                                                ref_text=(ref_text or None)),
        label="clone_prompt")
    return p, (getattr(p, "ref_text", "") or "")


# Cache the last-built voice so adding many scripts with the SAME clip does not
# re-transcribe / re-encode each time.
_VOICE_CACHE = {"key": None, "prompt": None, "ref_text": ""}


def _get_or_build_prompt(ref_audio, ref_text):
    key = (ref_audio, (ref_text or "").strip())
    if _VOICE_CACHE["key"] == key and _VOICE_CACHE["prompt"] is not None:
        return _VOICE_CACHE["prompt"], _VOICE_CACHE["ref_text"]
    prompt, used = _build_prompt(ref_audio, ref_text)
    _VOICE_CACHE.update(key=key, prompt=prompt, ref_text=used)
    return prompt, used


class ReferenceRejected(ValueError):
    """The uploaded reference cannot be used, and the caller is told why."""


def save_voice(name, audio_path, ref_text=None, owner=None, consent=None,
               consent_ref=None, speaker_name=None) -> Tuple[str, Dict]:
    """Register a voice: repair the clip, level it, check the transcript really
    matches the audio, and store a baseline speaking rate.

    Returns (voice_id, report). Never overwrites an existing voice.

    The transcript check is the important one. Reference bleed — reference words
    turning up in generated clips — happens when the model is told one thing and
    hears another, so it treats the difference as something it is supposed to
    say. Catching that here turns a bug that surfaces three hours into a batch
    into a message at upload time.
    """
    if not audio_path:
        raise ReferenceRejected("no audio to save")
    if REQUIRE_CONSENT and not consent:
        raise ReferenceRejected(
            "this server requires proof of consent before a voice can be "
            "registered: send consent=true and consent_ref=<your record id or "
            "URL>. Cloning an identifiable voice without documented, explicit "
            "permission is unlawful in a growing number of places, and the "
            "record has to name the commercial use it covers.")

    x, sr = read_audio(audio_path)
    report = audio_fx.analyze_reference(x, sr, max_sec=MAX_REF_SEC * 3)
    y, trim_info = audio_fx.smart_trim_reference(
        x, sr, max_sec=MAX_REF_SEC, tail_silence_sec=REF_TAIL_SILENCE,
        min_keep_sec=REF_MIN_KEEP_SEC, hard_max_sec=REF_HARD_MAX_SEC)
    warnings = _merge_ref_warnings(report, trim_info)
    y, loud = audio_fx.normalize_loudness(
        y, sr, target_lufs=REF_TARGET_LUFS, peak_ceiling_db=OUT_PEAK_CEILING)

    name = _unique_voice_name(name)
    dst = os.path.join(VOICES_DIR, name + ".wav")
    write_wav(dst, y, sr)
    duration = len(y) / float(sr)

    heard = ""
    try:
        heard = _transcribe_path(dst)
    except Exception as e:  # noqa: BLE001
        log.warning("could not transcribe reference for '%s': %s", name, e)

    supplied = (ref_text or "").strip()
    used_text = heard
    if supplied:
        if trim_info.get("trimmed"):
            warnings.append(
                "your reference transcript was replaced by a transcript of the "
                "trimmed clip — the upload was longer than "
                f"{MAX_REF_SEC:.0f}s, so the text no longer matched the audio")
        elif heard:
            m = verify.reference_matches_audio(supplied, heard)
            if m["matches"]:
                used_text = supplied
            else:
                detail = (f"the transcript you sent matches only "
                          f"{m['word_accuracy']:.0%} of what the recording "
                          f"actually says")
                if STRICT_REF:
                    try:
                        os.remove(dst)
                    except Exception:  # noqa: BLE001
                        pass
                    raise ReferenceRejected(
                        detail + ". A transcript that does not match the audio is "
                        "the main cause of reference words leaking into every "
                        "generated clip. Send the correct text, or omit ref_text "
                        "and it will be transcribed for you. "
                        f"Heard: \"{heard[:160]}\"")
                warnings.append(detail + " — the transcribed text was used instead")
        else:
            used_text = supplied

    baseline = verify.baseline_wpm(used_text, duration)
    entry = {
        "path": dst, "ref_text": used_text, "prompt": None,
        "baseline_wpm": baseline,
        "quality_score": report.get("quality_score"),
        "warnings": warnings, "owner": owner,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "duration_sec": round(duration, 2),
        "lufs": loud.get("out_lufs"), "snr_db": report.get("snr_db"),
        "speaker_name": speaker_name,
        "consent": bool(consent) if consent is not None else None,
        "consent_ref": consent_ref,
    }
    with VOICES_LOCK:
        VOICES[name] = entry
    _save_voice_index()
    audit("voice_registered", voice_id=name, tenant=owner,
          speaker_name=speaker_name, consent=entry["consent"],
          consent_ref=consent_ref, duration_sec=entry["duration_sec"],
          quality_score=report.get("quality_score"),
          warnings=(warnings or None))

    report.update({"voice_id": name, "accepted": True, "ref_text": used_text,
                   "baseline_wpm": baseline, "warnings": warnings,
                   "duration_sec": round(duration, 2),
                   "loudness": loud, "trim": trim_info,
                   "hint": "docs/reference_playbook.md"})
    return name, report


def get_voice_prompt(name):
    """Return (prompt, transcript) for a saved voice, building it on first use."""
    with VOICES_LOCK:
        v = VOICES.get(name)
    if not v:
        return None, ""
    if v.get("prompt") is None:
        cached = _load_prompt(name)
        if cached is not None:
            with VOICES_LOCK:
                v["prompt"] = cached
            return cached, v.get("ref_text", "")
        prompt, ref_used = _build_prompt(v["path"], v.get("ref_text") or None)
        with VOICES_LOCK:
            v["prompt"] = prompt
            if not v.get("ref_text"):
                v["ref_text"] = ref_used
        if _save_prompt(name, prompt):
            _save_voice_index()
    return v["prompt"], v.get("ref_text", "")


def voice_baseline(name) -> Optional[float]:
    with VOICES_LOCK:
        v = VOICES.get(name) or {}
    return v.get("baseline_wpm")


def list_voice_names(owner=None):
    with VOICES_LOCK:
        return sorted(n for n, v in VOICES.items()
                      if owner is None or v.get("owner") in (None, owner))


def voice_exists(name, owner=None) -> bool:
    with VOICES_LOCK:
        v = VOICES.get(name)
    # A voice belonging to another tenant reports as missing, not forbidden:
    # 403 would confirm that the id exists.
    return bool(v and (owner is None or v.get("owner") in (None, owner)))


def delete_voice(name, owner=None):
    with VOICES_LOCK:
        v = VOICES.get(name)
        if not v or (owner is not None and v.get("owner") not in (None, owner)):
            return False
        VOICES.pop(name, None)
    try:
        p = v.get("path")
        if p and os.path.exists(p) and os.path.normpath(
                os.path.dirname(p)) == os.path.normpath(VOICES_DIR):
            os.remove(p)
        cached = _prompt_path(name)
        if os.path.exists(cached):
            os.remove(cached)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not delete voice clip: %s", e)
    _save_voice_index()
    return True


def voice_details(name):
    with VOICES_LOCK:
        v = VOICES.get(name)
    if not v:
        return None, ""
    return v.get("path"), v.get("ref_text", "")


def voice_public(name) -> Dict:
    with VOICES_LOCK:
        v = dict(VOICES.get(name) or {})
    v.pop("prompt", None)
    v.pop("path", None)
    v["voice_id"] = name
    return v


_load_voice_index()


# ---------------------------------------------------------------------------
# UI handlers
# ---------------------------------------------------------------------------
def save_voice_ui(name, ref_audio):
    """UI handler: save the uploaded clip to the permanent library."""
    if not ref_audio:
        return gr.update(), "⚠️ Upload a voice clip above first, then Save."
    if not name or not name.strip():
        return gr.update(), "⚠️ Type a name for the voice, then Save."
    wanted = _safe_name(name)
    try:
        vid, rep = save_voice(name, ref_audio)
    except ReferenceRejected as e:
        return gr.update(), f"❌ {e}"
    except Exception as e:  # noqa: BLE001
        return gr.update(), f"❌ Save failed: {type(e).__name__}: {e}"
    choices = [NO_SAVED_VOICE] + list_voice_names()
    note = (f" (name '{wanted}' was taken)" if vid != wanted else "")
    msg = (f"💾 Your unique voice name is **{vid}**{note} — reusable from the "
           f"dropdown & API. It's saved permanently.")
    if rep.get("baseline_wpm"):
        msg += f"  ·  natural rate ≈ {rep['baseline_wpm']:.0f} wpm"
    if rep.get("warnings"):
        msg += "\n\n⚠️ " + "\n⚠️ ".join(rep["warnings"])
    return gr.update(choices=choices, value=vid), msg


def refresh_voice_dropdown():
    """Repopulate the saved-voice dropdown from the live library (runs on every
    page load/refresh so newly-saved voices always show up)."""
    return gr.update(choices=[NO_SAVED_VOICE] + list_voice_names())


def on_pick_voice(name):
    """When a saved voice is picked, load its sample clip + transcript into the
    form so the user sees exactly what will be used."""
    if not name or name == NO_SAVED_VOICE:
        return None, "", ""
    path, ref = voice_details(name)
    if not path:
        return None, "", f"⚠️ Voice '{name}' not found."
    return (path, ref,
            f"🎙️ Loaded **{name}** — its sample & transcript are shown above.")


def delete_voice_ui(name):
    """Delete the selected saved voice from the library."""
    if not name or name == NO_SAVED_VOICE:
        return gr.update(), None, "", "Pick a saved voice first, then Delete."
    ok = delete_voice(name)
    choices = [NO_SAVED_VOICE] + list_voice_names()
    msg = (f"🗑️ Deleted voice **{name}**." if ok else f"Voice '{name}' not found.")
    return gr.update(choices=choices, value=NO_SAVED_VOICE), None, "", msg


def _resolve_ui_voice(saved_voice, ref_audio, ref_text):
    """Prefer a saved-library voice; else the uploaded clip; else designed.
    Returns (prompt, transcript, baseline_wpm)."""
    if saved_voice and saved_voice != NO_SAVED_VOICE:
        if not voice_exists(saved_voice):
            raise KeyError(f"voice '{saved_voice}' no longer exists")
        prompt, used = get_voice_prompt(saved_voice)
        return prompt, used, voice_baseline(saved_voice)
    if ref_audio:
        prompt, used = _get_or_build_prompt(ref_audio, ref_text)
        return prompt, used, None
    return None, "", None


def _new_job(project, script, prebuilt, ref_used, params) -> int:
    global _JOB_SEQ
    with JOBS_LOCK:
        _JOB_SEQ += 1
        jid = _JOB_SEQ
        proj = (project or "").strip() or f"Project_{jid}"
        JOBS.append({
            "id": jid, "project": proj, "status": "Queued", "info": "",
            "file": None, "cancel": False, "ref_used": ref_used,
            "warnings": [], "report": {},
            "params": dict(params, script=script, clone_prompt=prebuilt),
        })
    JOB_Q.put(jid)
    return jid


def add_job(project, script, ref_audio, ref_text, language, num_step,
            guidance_scale=2.0, denoise=True, speed=1.0, duration=None,
            preprocess=True, postprocess=True, saved_voice=None):
    if not script or not script.strip():
        return _ui_state("⚠️ Script is empty — nothing added.")
    too_big = check_input_size(script, queued=True)
    if too_big:
        return _ui_state(f"⚠️ {too_big}")
    try:
        prebuilt, ref_used, baseline = _resolve_ui_voice(saved_voice, ref_audio, ref_text)
    except Exception as e:
        return _ui_state(f"❌ Voice prompt failed: {type(e).__name__}: {e}")
    params = {
        "ref_audio": None, "ref_text": (ref_text or None), "language": language,
        "num_step": int(num_step or 16), "guidance_scale": float(guidance_scale),
        "denoise": bool(denoise), "speed": float(speed) if speed else 1.0,
        "duration": (float(duration) if duration else None),
        "preprocess": bool(preprocess), "postprocess": bool(postprocess),
        "baseline_wpm": baseline,
        "voice_id": (saved_voice if saved_voice and saved_voice != NO_SAVED_VOICE
                     else None),
    }
    jid = _new_job(project, script, prebuilt, ref_used, params)
    with JOBS_LOCK:
        proj = _find_job(jid)["project"]
    return _ui_state(f"✅ Added '{proj}' to the queue.")


def _cell(v):
    """Normalize a table cell to a clean string ('' for empty/NaN)."""
    if v is None:
        return ""
    try:
        if isinstance(v, float) and v != v:
            return ""
    except Exception:
        pass
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def _rows_to_list(rows):
    """Accept whatever a gradio Dataframe hands us: pandas DataFrame, dict with
    'data', numpy array, or a plain list — and return a list of row-lists."""
    if rows is None:
        return []
    try:
        import pandas as pd
        if isinstance(rows, pd.DataFrame):
            return rows.values.tolist()
    except Exception:
        pass
    if isinstance(rows, dict):
        return rows.get("data") or rows.get("value") or []
    if isinstance(rows, (list, tuple)):
        return list(rows)
    try:
        return rows.tolist()
    except Exception:
        return []


def _normalize_rows(rows):
    """Return table data as a clean list of [name, script] string-pairs."""
    out = []
    for r in _rows_to_list(rows):
        if isinstance(r, (list, tuple)):
            out.append([_cell(r[0]) if len(r) > 0 else "",
                        _cell(r[1]) if len(r) > 1 else ""])
    return out


def _to_df(data):
    """gradio 6 requires a Dataframe *output* to be a DataFrame/dict, not a raw
    list — returning a plain list raises a DataframeData validation error."""
    import pandas as pd
    return pd.DataFrame(data, columns=["Project name", "Script"])


def add_row(rows):
    """Append a blank row so the user can keep adding unlimited rows."""
    data = _normalize_rows(rows)
    data.append(["", ""])
    return _to_df(data)


def clear_rows():
    return _to_df([["", ""]])


def add_table(rows, ref_audio, ref_text, language, num_step,
              guidance_scale=2.0, denoise=True, speed=1.0, duration=None,
              preprocess=True, postprocess=True, saved_voice=None):
    """Queue one project per table row (columns: Project name, Script).

    All rows share the same (cached) voice. Empty-script rows are skipped.
    """
    items = []
    for r in _rows_to_list(rows):
        if r is None:
            continue
        if not isinstance(r, (list, tuple)):
            r = [r]
        name = _cell(r[0]) if len(r) > 0 else ""
        scr = _cell(r[1]) if len(r) > 1 else ""
        if scr:
            items.append((name, scr))
    if not items:
        return _ui_state("⚠️ Fill at least one row with a script.")
    for _n, scr in items:
        too_big = check_input_size(scr, queued=True)
        if too_big:
            return _ui_state(f"⚠️ {too_big}")

    try:
        prebuilt, ref_used, baseline = _resolve_ui_voice(saved_voice, ref_audio, ref_text)
    except Exception as e:
        return _ui_state(f"❌ Voice prompt failed: {type(e).__name__}: {e}")

    params = {
        "ref_audio": None, "ref_text": (ref_text or None), "language": language,
        "num_step": int(num_step or 16), "guidance_scale": float(guidance_scale),
        "denoise": bool(denoise), "speed": float(speed) if speed else 1.0,
        "duration": (float(duration) if duration else None),
        "preprocess": bool(preprocess), "postprocess": bool(postprocess),
        "baseline_wpm": baseline,
        "voice_id": (saved_voice if saved_voice and saved_voice != NO_SAVED_VOICE
                     else None),
    }
    for name, scr in items:
        proj = _safe_name(name) if name and name.lower() != "nan" else ""
        _new_job(proj, scr, prebuilt, ref_used, params)
    return _ui_state(f"✅ Added {len(items)} projects to the queue (same voice).")


def cancel_or_remove(job_id):
    """Cancel a queued/running job, or remove a finished one."""
    if job_id is None:
        return _ui_state("Select a project from the dropdown first.")
    with JOBS_LOCK:
        j = _find_job(int(job_id))
        if j is None:
            msg = "Job not found."
        elif j["status"] == "Queued":
            j["cancel"], j["status"] = True, "Cancelled"
            msg = f"🚫 Cancelled '{j['project']}'."
        elif j["status"] == "Processing":
            j["cancel"], j["status"] = True, "Cancelling…"
            msg = f"⏹️ Cancelling '{j['project']}' (stops at next chunk)…"
        else:
            JOBS.remove(j)
            msg = f"🗑️ Removed '{j['project']}'."
    return _ui_state(msg)


def cancel_all_queued():
    n = 0
    with JOBS_LOCK:
        for j in JOBS:
            if j["status"] == "Queued":
                j["cancel"], j["status"] = True, "Cancelled"
                n += 1
    return _ui_state(f"🚫 Cancelled {n} queued job(s).")


def clear_finished():
    with JOBS_LOCK:
        JOBS[:] = [j for j in JOBS if j["status"] in
                   ("Queued", "Processing", "Cancelling…")]
    return _ui_state("🧹 Cleared finished jobs.")


def zip_all():
    """Bundle all completed outputs into a single zip for one-click download."""
    files = [f for f in _download_files() if f and os.path.exists(f)]
    if not files:
        return None, "No completed files to download yet."
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Always written to OUTPUT_DIR: that folder is what launch() allows the
    # browser to read, whatever the user set as the save folder.
    zip_path = os.path.join(OUTPUT_DIR, f"voiceover_all_{ts}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=os.path.basename(f))
    return zip_path, f"📦 Zipped {len(files)} file(s)."


# ---------------------------------------------------------------------------
# Custom Gradio UI
# ---------------------------------------------------------------------------
def _build_instruct(groups):
    """Turn Voice-Design dropdown selections into an instruct string."""
    selected = [g for g in groups if g and g != "Auto"]
    if not selected:
        return None
    parts = []
    for v in selected:
        if " / " in v:
            en, zh = v.split(" / ", 1)
            if "Dialect" in v.split(" / ")[0]:
                parts.append(zh.strip())
            else:
                parts.append(en.strip())
        else:
            parts.append(v)
    return ", ".join(parts)


def _lang_dropdown(label="Language (optional) / 语种 (可选)"):
    return gr.Dropdown(
        label=label,
        choices=_ALL_LANGUAGES,
        value="Auto",
        allow_custom_value=False,
        info="Keep as Auto to auto-detect the language.",
    )


def build_ui() -> gr.Blocks:
    # In Gradio 6, theme/css are passed to launch(), not the Blocks constructor.
    with gr.Blocks(title="OmniVoice Demo") as demo:
        gr.Markdown(
            "# 🎙️ Voiceover Studio\n"
            "**1)** pick a voice  →  **2)** add your script(s)  →  **3)** render. "
            "Every clip is saved to the `outputs/` folder, checked against your "
            "script, and levelled to the same loudness."
        )

        with gr.Row(equal_height=False):
            # ============ LEFT: create ============
            with gr.Column(scale=5):
                gr.Markdown("### 1️⃣  Pick a voice")
                with gr.Row():
                    st_saved_voice = gr.Dropdown(
                        label="🎙️ Use a saved voice", choices=[NO_SAVED_VOICE] +
                        list_voice_names(), value=NO_SAVED_VOICE, scale=4,
                        info="Pick a saved voice (loads its sample + transcript), "
                        "or upload a new clip below.",
                    )
                    st_del_voice = gr.Button("🗑️ Delete", scale=1)
                st_ref = gr.Audio(
                    label="Upload a 3–10s voice clip   ·   leave empty = AI voice",
                    type="filepath",
                )
                gr.Markdown(
                    "<span style='font-size:0.82em;color:#888'>Best results: 6–10s, "
                    "one speaker, no background noise, and <b>ending on a finished "
                    "sentence</b> — a clip that stops mid-word can leak that word "
                    "into every clip you generate.</span>"
                )
                with gr.Row():
                    st_voice_name = gr.Textbox(
                        label="Save this voice as", placeholder="narrator_a", scale=2,
                    )
                    st_save_voice = gr.Button("💾 Save to library", scale=1)
                with gr.Accordion("Voice transcript (auto · usually leave as is)",
                                  open=False):
                    st_ref_text = gr.Textbox(
                        label="Transcript of the voice clip", lines=2,
                        placeholder="Auto-filled by Whisper when you upload a clip.",
                    )

                gr.Markdown("### 2️⃣  Add your script")
                st_project = gr.Textbox(
                    label="Project name", placeholder="my_intro",
                )
                st_script = gr.Textbox(
                    label="Script", lines=9,
                    placeholder="Type or paste the text to speak…",
                )
                st_add = gr.Button("➕  Add to queue", variant="primary", size="lg")
                gr.Markdown(
                    "<span style='font-size:0.82em;color:#888'>The uploaded voice "
                    "stays loaded — add as many scripts as you like with the same "
                    "voice. Change the clip only when you want a different voice.</span>"
                )

                with gr.Accordion("🧩  Add several at once (table)", open=False):
                    gr.Markdown(
                        "<span style='font-size:0.85em;color:#888'>Each row = one "
                        "project, all using the voice above. Click the bottom row's "
                        "<b>+</b> to add more rows, then <b>Add all rows</b>.</span>"
                    )
                    st_table_in = gr.Dataframe(
                        headers=["Project name", "Script"],
                        datatype=["str", "str"],
                        type="array",
                        value=[["", ""], ["", ""]],
                        row_count=(2, "dynamic"),
                        col_count=(2, "fixed"),
                        interactive=True, wrap=True,
                    )
                    with gr.Row():
                        st_addrow = gr.Button("➕ Add row", size="sm")
                        st_clearrows = gr.Button("🧽 Clear rows", size="sm")
                    st_addtable = gr.Button("✅  Add all rows to queue",
                                            variant="primary")

                with gr.Accordion("3️⃣  Settings (optional)", open=False):
                    st_lang = _lang_dropdown()
                    with gr.Row():
                        st_steps = gr.Slider(
                            4, 64, value=16, step=1, label="Quality steps",
                            info="16 = fast · 32+ = higher quality",
                        )
                        st_speed = gr.Slider(
                            0.5, 1.5, value=1.0, step=0.05, label="Speed",
                            info="1.0 = normal. Ignored if Duration is set.",
                        )
                    with gr.Row():
                        st_cfg = gr.Slider(
                            0.0, 4.0, value=2.0, step=0.1, label="Guidance (CFG)",
                            info="Default 2.0. Upstream #163 reports this has "
                                 "no audible effect through the Python API — "
                                 "change Quality steps instead.",
                        )
                        st_duration = gr.Number(
                            value=None, label="Duration (seconds)",
                            info="Empty = auto. Fixed value disables chunking "
                                 "and script checking.",
                        )
                    with gr.Row():
                        st_denoise = gr.Checkbox(value=True, label="Denoise")
                        st_preprocess = gr.Checkbox(
                            value=True, label="Preprocess prompt",
                            info="Silence-trim reference audio.",
                        )
                        st_postprocess = gr.Checkbox(
                            value=True, label="Postprocess output",
                            info="Remove long silences.",
                        )
                st_msg = gr.Markdown("")

            # ============ RIGHT: queue ============
            with gr.Column(scale=7):
                gr.Markdown("### 🎧  Render queue  ·  one at a time")
                st_board = gr.HTML(_jobs_html())
                with gr.Row():
                    st_picker = gr.Dropdown(
                        label="Select a project to manage", choices=_job_choices(),
                        value=None, scale=3, interactive=True,
                    )
                    st_cancel = gr.Button("✖ Cancel / Remove", scale=1)
                with gr.Row():
                    st_cancel_all = gr.Button("⏹️ Cancel all queued")
                    st_clear = gr.Button("🧹 Clear finished")
                st_shutdown = gr.Checkbox(
                    value=False, label="🛑 Shut down PC when the queue finishes",
                    info="Auto-saves first, then shuts down ~60s later "
                    "(cancel with 'shutdown /a').",
                )
                with gr.Accordion("📂  Save folder & downloads", open=True):
                    st_outdir = gr.Textbox(
                        label="Voices auto-save to this folder",
                        value=_OUT_DIR,
                        info="Files appear here automatically — no clicking needed.",
                    )
                    st_make_zip = gr.Button("📦 Zip all completed")
                    st_zip_file = gr.File(label="Zip — click to download",
                                          interactive=False)
                    st_files = gr.Files(label="Individual files",
                                        value=_download_files())
                st_timer = gr.Timer(2.0)

        # Auto-transcribe the reference on upload (fills the transcript box).
        # If the clip is a saved-library voice, reuse its transcript (no re-ASR).
        def _on_studio_ref(ref_audio):
            if ref_audio:
                with VOICES_LOCK:
                    for nm, v in VOICES.items():
                        if v.get("path") == ref_audio:
                            return (v.get("ref_text", ""),
                                    f"<span style='color:#888;font-size:0.85em'>"
                                    f"Saved voice '{nm}' transcript.</span>")
            text, msg = transcribe_reference(ref_audio)
            return text, f"<span style='color:#888;font-size:0.85em'>{msg}</span>"

        st_ref.change(_on_studio_ref, inputs=[st_ref],
                      outputs=[st_ref_text, st_msg])

        # Pick a saved voice -> load its sample clip + transcript into the form.
        st_saved_voice.change(on_pick_voice, inputs=[st_saved_voice],
                              outputs=[st_ref, st_ref_text, st_msg])
        st_del_voice.click(delete_voice_ui, inputs=[st_saved_voice],
                           outputs=[st_saved_voice, st_ref, st_ref_text, st_msg])

        st_save_voice.click(save_voice_ui, inputs=[st_voice_name, st_ref],
                            outputs=[st_saved_voice, st_msg])

        _OUT = [st_board, st_files, st_picker, st_msg]
        st_add.click(
            add_job,
            inputs=[st_project, st_script, st_ref, st_ref_text, st_lang, st_steps,
                    st_cfg, st_denoise, st_speed, st_duration, st_preprocess,
                    st_postprocess, st_saved_voice],
            outputs=_OUT,
        ).then(lambda: ("", ""), outputs=[st_project, st_script])

        st_addrow.click(add_row, inputs=[st_table_in], outputs=[st_table_in],
                        api_name="add_row")
        st_clearrows.click(clear_rows, outputs=[st_table_in])
        st_addtable.click(
            add_table,
            inputs=[st_table_in, st_ref, st_ref_text, st_lang, st_steps, st_cfg,
                    st_denoise, st_speed, st_duration, st_preprocess, st_postprocess,
                    st_saved_voice],
            outputs=_OUT, api_name="add_table",
        ).then(clear_rows, outputs=[st_table_in])

        st_outdir.change(set_output_dir, inputs=[st_outdir], outputs=[st_msg])
        st_shutdown.change(set_shutdown, inputs=[st_shutdown], outputs=[st_msg])

        st_cancel.click(cancel_or_remove, inputs=[st_picker], outputs=_OUT)
        st_cancel_all.click(cancel_all_queued, outputs=_OUT)
        st_clear.click(clear_finished, outputs=_OUT)
        st_make_zip.click(zip_all, outputs=[st_zip_file, st_msg])
        st_timer.tick(
            lambda cur: (_jobs_html(), _download_files(), _picker_update(cur)),
            inputs=[st_picker],
            outputs=[st_board, st_files, st_picker],
        )

        # On every page load/refresh, repopulate the saved-voice dropdown from
        # the live library (so voices saved after startup always appear).
        demo.load(refresh_voice_dropdown, outputs=[st_saved_voice])

    return demo


demo = build_ui()

_THEME = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
_CSS = """
.gradio-container {max-width: 100% !important;}
.vq-pills {display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px;}
.vq-pill {padding:5px 12px; border-radius:999px; font-size:0.85em; font-weight:600;
  background:var(--block-background-fill); border:1px solid var(--border-color-primary);}
.vq-pill.vq-pill-good {background:#dcfce7; color:#166534; border-color:#86efac;}
.vq-pill.vq-pill-slow {background:#fef3c7; color:#92400e; border-color:#fcd34d;}
/* Metrics get their own row and are never ellipsised: RTF used to sit at the
   end of a truncated line, which is why nobody ever saw it. */
.vq-metrics {display:flex; flex-wrap:wrap; gap:5px; margin-top:5px;}
.vq-chip {font-size:0.72em; font-weight:600; padding:2px 8px; border-radius:7px;
  background:var(--background-fill-secondary, rgba(127,127,127,0.12));
  color:var(--body-text-color-subdued); white-space:nowrap;}
.vq-chip.vq-chip-good {background:#dcfce7; color:#166534;}
.vq-chip.vq-chip-slow {background:#fef3c7; color:#92400e;}
.vq-gpu {font-size:0.76em; color:var(--body-text-color-subdued); margin-bottom:10px;}
.vq-board {display:flex; flex-direction:column; gap:10px; max-height:62vh; overflow-y:auto;
  padding-right:4px;}
.vq-card {display:flex; align-items:center; justify-content:space-between; gap:12px;
  padding:12px 16px; border-radius:14px; background:var(--block-background-fill);
  border:1px solid var(--border-color-primary); border-left:5px solid var(--border-color-primary);
  box-shadow:0 1px 4px rgba(0,0,0,0.06); transition:transform .08s ease;}
.vq-card:hover {transform:translateY(-1px); box-shadow:0 3px 10px rgba(0,0,0,0.10);}
.vq-left {display:flex; align-items:center; gap:14px; min-width:0; flex:1 1 auto;}
.vq-id {font-weight:700; color:var(--body-text-color-subdued); font-size:0.9em;
  min-width:34px; flex-shrink:0;}
.vq-meta {min-width:0; flex:1 1 auto;}
.vq-name {font-weight:600; font-size:1.05em; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; max-width:360px;}
.vq-sub {font-size:0.8em; color:var(--body-text-color-subdued); white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; max-width:420px; margin-top:2px;}
.vq-ref {font-size:0.76em; color:var(--body-text-color-subdued); font-style:italic;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:300px;
  margin-top:2px; opacity:0.85;}
.vq-warn {font-size:0.76em; color:#9a3412; background:#ffedd5; border-radius:8px;
  padding:3px 8px; margin-top:4px; max-width:420px;}
.vq-good {font-size:0.74em; color:#166534; margin-top:4px; opacity:0.9;}
.vq-right {display:flex; align-items:center; gap:12px; flex:0 0 auto;}
.vq-info {font-size:0.8em; color:var(--body-text-color-subdued); white-space:nowrap;}
.vq-file {display:inline-block; margin-top:5px; font-size:0.74em; color:#166534;
  background:#dcfce7; padding:2px 8px; border-radius:8px; max-width:100%;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
.vq-badge {padding:5px 12px; border-radius:999px; font-size:0.8em; font-weight:700;
  white-space:nowrap;}
.vq-badge.badge-queued {background:#fef3c7; color:#92400e;}
.vq-badge.badge-processing {background:#dbeafe; color:#1e40af; animation:vqpulse 1.2s infinite;}
.vq-badge.badge-done {background:#dcfce7; color:#166534;}
.vq-badge.badge-error {background:#fee2e2; color:#991b1b;}
.vq-badge.badge-cancelled {background:#e5e7eb; color:#374151;}
.vq-card.badge-processing {border-left-color:#3b82f6;}
.vq-card.badge-done {border-left-color:#22c55e;}
.vq-card.badge-error {border-left-color:#ef4444;}
.vq-card.badge-queued {border-left-color:#f59e0b;}
.vq-card.badge-cancelled {border-left-color:#9ca3af;}
.vq-empty {padding:36px; text-align:center; color:var(--body-text-color-subdued);
  border:2px dashed var(--border-color-primary); border-radius:14px;}
@keyframes vqpulse {0%,100%{opacity:1;} 50%{opacity:0.45;}}
"""


# ===========================================================================
# Local REST API  (see LOCAL_API.md)
#
# CONTRACT FREEZE: POST /api/tts keeps exactly the shape it has always had —
# multipart in, raw audio bytes out, X-Duration-Sec and X-RTF headers. Every
# new fix (normalization, verification, loudness) applies on that path too;
# only the *shape of the reply* is frozen. Everything new is either an extra
# response header (additive, ignorable) or lives on /api/v2/tts.
# ===========================================================================
import hashlib
from contextlib import contextmanager

API_PORT = int(os.environ.get("OMNIVOICE_API_PORT", "8001"))
STRICT_PARAMS = _env_flag("OMNIVOICE_STRICT_PARAMS", "0")
HEALTH_STRICT = _env_flag("OMNIVOICE_HEALTH_STRICT")
IDEM_TTL_S = float(os.environ.get("OMNIVOICE_IDEMPOTENCY_TTL", str(24 * 3600)))


def _tenant_keys() -> Dict[str, str]:
    """OMNIVOICE_API_KEYS="key1:acme,key2:globex" turns on per-customer voice
    isolation. With the single legacy OMNIVOICE_API_KEY everything stays one
    tenant, exactly as before."""
    raw = os.environ.get("OMNIVOICE_API_KEYS", "").strip()
    out: Dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if ":" in part:
            k, t = part.split(":", 1)
            if k.strip():
                out[k.strip()] = t.strip() or "default"
    return out


TENANT_KEYS = _tenant_keys()
MULTI_TENANT = bool(TENANT_KEYS)


def _encode_audio(wav_i16, sr, fmt="mp3"):
    """int16 mono numpy -> (bytes, media_type, ext). mp3 needs ffmpeg."""
    fmt = (fmt or "mp3").lower()
    if fmt == "wav":
        buf = io.BytesIO()
        sf.write(buf, wav_i16, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue(), "audio/wav", "wav"
    from pydub import AudioSegment
    seg = AudioSegment(wav_i16.tobytes(), frame_rate=int(sr), sample_width=2, channels=1)
    buf = io.BytesIO()
    seg.export(buf, format="mp3", bitrate=os.environ.get("OMNIVOICE_MP3_BITRATE", "192k"))
    return buf.getvalue(), "audio/mpeg", "mp3"


# ---- idempotency ----------------------------------------------------------
_IDEM: Dict[str, Dict[str, Any]] = {}
_IDEM_LOCK = threading.Lock()


IDEM_MAX_ENTRIES = int(os.environ.get("OMNIVOICE_IDEMPOTENCY_MAX", "200"))


def _idem_sweep():
    now = time.time()
    for k in [k for k, v in _IDEM.items() if now - v["at"] > IDEM_TTL_S]:
        _IDEM.pop(k, None)
    # Each entry holds encoded audio, so an unbounded cache is a slow leak.
    if len(_IDEM) > IDEM_MAX_ENTRIES:
        for k, _v in sorted(_IDEM.items(), key=lambda kv: kv[1]["at"])[
                :len(_IDEM) - IDEM_MAX_ENTRIES]:
            _IDEM.pop(k, None)


def _idem_lookup(key: Optional[str]):
    """Returns ('hit', payload) | ('busy', None) | ('new', None).

    A client whose request timed out will retry work the server already did.
    Without this the GPU pays twice, and on a ten-hour day that is real money.
    """
    if not key:
        return "new", None
    with _IDEM_LOCK:
        _idem_sweep()
        entry = _IDEM.get(key)
        if entry is None:
            _IDEM[key] = {"at": time.time(), "state": "running", "payload": None}
            return "new", None
        if entry["state"] == "running":
            return "busy", None
        return "hit", entry["payload"]


def _idem_store(key: Optional[str], payload: Any):
    if not key:
        return
    with _IDEM_LOCK:
        _IDEM[key] = {"at": time.time(), "state": "done", "payload": payload}


def _idem_drop(key: Optional[str]):
    if not key:
        return
    with _IDEM_LOCK:
        if _IDEM.get(key, {}).get("state") == "running":
            _IDEM.pop(key, None)


def _hdr(value: Any, limit: int = 480) -> str:
    """HTTP headers are latin-1; a script is not. Never let a header kill a
    response that is otherwise fine."""
    s = str(value if value is not None else "")
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.encode("ascii", "replace").decode("ascii")
    return s[:limit]


def _queue_depth() -> int:
    with JOBS_LOCK:
        return sum(1 for j in JOBS if j["status"] in ("Queued", "Processing"))


def _api_enqueue(project, script, prebuilt, ref_used, language, num_step, speed,
                 seed=None, baseline=None, voice_id=None, owner=None):
    """Create a queued job (for the async API) and return its id."""
    params = {
        "ref_audio": None, "ref_text": None, "language": language,
        "num_step": int(num_step), "guidance_scale": 2.0, "denoise": True,
        "speed": float(speed), "duration": None,
        "preprocess": True, "postprocess": True,
        "seed": seed, "baseline_wpm": baseline,
        "voice_id": voice_id, "owner": owner,
    }
    return _new_job(_safe_name(project), script, prebuilt, ref_used, params)


def _run_selftest() -> Optional[Tuple[bool, str]]:
    """Generate four words. The only answer to 'is this server working?' that
    is worth anything — a VRAM reading said 'ok' for twenty minutes while every
    request returned 500."""
    if not GEN_SLOTS.acquire(blocking=False):
        return None                     # real work is running; that is 'alive'
    t0 = time.monotonic()
    try:
        rep: Dict[str, Any] = {}
        out, status = _gen_core("Testing one two three.", None, None, None,
                                8, 2.0, True, 1.0, None, True, True,
                                mode="design", report=rep)
        if out is None:
            return False, status
        return True, f"spoke in {(time.monotonic() - t0) * 1000:.0f} ms"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    finally:
        GEN_SLOTS.release()


def _prewarm_voices():
    """Build every saved voice's prompt once at startup.

    Otherwise the first request for each voice pays 50-200 ms of GPU time to
    re-encode a reference that has not changed since it was registered — and it
    pays it again after every restart, which on this server means after every
    OOM recovery.
    """
    time.sleep(5)
    for name in list_voice_names():
        try:
            get_voice_prompt(name)
            log.info("pre-warmed voice %s", name)
        except Exception as e:  # noqa: BLE001
            log.warning("could not pre-warm voice %s: %s", name, e)


def _selftest_loop():
    time.sleep(10)
    while True:
        try:
            res = _run_selftest()
            if res is not None:
                READY.record(res[0], res[1])
                if not res[0]:
                    log.warning("self-test failed: %s", res[1])
        except Exception as e:  # noqa: BLE001
            log.warning("self-test loop error: %s", e)
        time.sleep(max(15.0, SELFTEST_EVERY))


def _build_api():
    from fastapi import (FastAPI, File, Form, Header, HTTPException, Request,
                         UploadFile)
    from fastapi.responses import JSONResponse, Response
    from starlette.concurrency import run_in_threadpool

    api = FastAPI(title="OmniVoice Local API", version=API_VERSION,
                  docs_url="/api/docs", openapi_url="/api/openapi.json")

    # ---- auth / tenants --------------------------------------------------
    def _auth(x_api_key) -> Optional[str]:
        """Returns the owner id for this key (None = single-tenant)."""
        if TENANT_KEYS:
            owner = TENANT_KEYS.get((x_api_key or "").strip())
            if not owner:
                raise HTTPException(status_code=401,
                                    detail="Invalid or missing API key")
            return owner
        key = os.environ.get("OMNIVOICE_API_KEY", "").strip()
        if key and (x_api_key or "") != key:
            raise HTTPException(status_code=401,
                                detail="Invalid or missing API key")
        return None

    # ---- shared plumbing -------------------------------------------------
    @contextmanager
    def _gen_slot():
        """Bound concurrency instead of dying under it. Four simultaneous
        callers used to take the whole server down; now the fourth waits, and
        if the wait is hopeless it gets a 429 with a Retry-After it can obey."""
        if not GEN_SLOTS.acquire(timeout=QUEUE_WAIT_S):
            raise HTTPException(
                status_code=429,
                detail={"error": "busy",
                        "message": f"the GPU is busy; waited {QUEUE_WAIT_S:.0f}s",
                        "queue_depth": _queue_depth()},
                headers={"Retry-After": str(max(5, int(QUEUE_WAIT_S / 10)))})
        try:
            yield
        finally:
            GEN_SLOTS.release()

    async def _reject_unknown(request: Request, allowed) -> List[str]:
        """Unknown parameters are reported, not silently accepted — but they
        do not 400 by default, because every existing client would break on
        day one. Flip OMNIVOICE_STRICT_PARAMS=1 once the metric is quiet."""
        try:
            form = await request.form()
            extra = sorted(set(form.keys()) - set(allowed))
        except Exception:  # noqa: BLE001
            return []
        if extra:
            with METRICS_LOCK:
                for k in extra:
                    METRICS["unknown_params"][k] = \
                        METRICS["unknown_params"].get(k, 0) + 1
            if STRICT_PARAMS:
                raise HTTPException(status_code=400, detail={
                    "error": "unknown_params", "unknown": extra,
                    "valid_keys": sorted(allowed)})
        return extra

    def _resolve_api_voice(voice_bytes, voice_filename, voice_id, ref_text, owner):
        """(prompt, ref_used, mode, baseline_wpm).

        An unknown voice_id is a 404, always. Silently substituting another
        voice is the most expensive failure this product can have: the customer
        pays for ten hours of audio in the wrong voice and only finds out at the
        end. A 500 is visible; a wrong voice is not.
        """
        if voice_id:
            if not voice_exists(voice_id, owner):
                raise KeyError(voice_id)
            prompt, ref_used = get_voice_prompt(voice_id)
            if prompt is None:
                raise KeyError(voice_id)
            return prompt, ref_used, "clone", voice_baseline(voice_id)
        if voice_bytes:
            digest = hashlib.sha1(voice_bytes).hexdigest()
            key = (digest, (ref_text or "").strip())
            if _VOICE_CACHE["key"] == key and _VOICE_CACHE["prompt"] is not None:
                return (_VOICE_CACHE["prompt"], _VOICE_CACHE["ref_text"],
                        "clone", None)
            suffix = os.path.splitext(voice_filename or "ref.wav")[1] or ".wav"
            fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="api_ref_")
            os.close(fd)
            try:
                with open(tmp, "wb") as f:
                    f.write(voice_bytes)
                prompt, ref_used = _build_prompt(tmp, ref_text)
            finally:
                try:
                    os.remove(tmp)      # the old code leaked one file per call
                except Exception:  # noqa: BLE001
                    pass
            _VOICE_CACHE.update(key=key, prompt=prompt, ref_text=ref_used)
            return prompt, ref_used, "clone", None
        return None, "", "design", None

    def _synthesize(text, language, steps, speed, prebuilt, mode, seed, baseline,
                    voice_id=None, owner=None, project=None):
        rep: Dict[str, Any] = {}
        with _gen_slot():
            out, status = _gen_core(
                text, (None if not language or language == "Auto" else language),
                None, None, int(steps), 2.0, True, float(speed), None, True, True,
                mode=mode, ref_text=None, prebuilt_prompt=prebuilt, report=rep,
                seed=seed, baseline_wpm=baseline)
        if out is None:
            if not HEALTH.ok:
                raise HTTPException(
                    status_code=503,
                    detail={"error": "gpu_unavailable", "message": status,
                            **HEALTH.as_dict()},
                    headers={"Retry-After": "60"})
            raise HTTPException(status_code=500, detail=status)
        sr, wav = out
        audit_generation(text, rep, voice_id=voice_id, owner=owner,
                         project=project, wav=wav, sr=sr, source="api")
        return sr, wav, status, rep

    def _headers(rep, project, ext, sr, wav):
        dur = len(wav) / sr if sr else 0.0
        h = {
            "X-Duration-Sec": f"{dur:.2f}",
            "X-RTF": f"{rep.get('rtf', 0):.3f}",
            "X-WPM": f"{rep.get('wpm', 0):.0f}",
            "X-LUFS": f"{(rep.get('loudness') or {}).get('out_lufs', '')}",
            "X-True-Peak-dB": f"{(rep.get('loudness') or {}).get('true_peak_db', '')}",
            "X-Verified": "true" if rep.get("verified") else "false",
            "X-OmniVoice-API-Version": API_VERSION,
            "X-OmniVoice-Normalized-Text": _hdr(rep.get("normalized_text")),
            "Content-Disposition": f'attachment; filename="{_safe_name(project)}.{ext}"',
        }
        if rep.get("warnings"):
            h["X-OmniVoice-Warning"] = _hdr("; ".join(rep["warnings"]))
        return h

    def _meta(rep, project, ext, sr, wav, voice_id, ref_used, mode):
        return {
            "ok": True,
            "project": _safe_name(project),
            "format": ext,
            "duration_sec": round(len(wav) / sr, 2) if sr else 0.0,
            "sampling_rate": sr,
            "voice": "cloned" if mode == "clone" else "designed",
            "voice_id": voice_id,
            "ref_text": ref_used,
            "normalized_text": rep.get("normalized_text"),
            "verified": rep.get("verified", False),
            "verification": rep.get("diffs", []),
            "warnings": rep.get("warnings", []),
            "wpm": rep.get("wpm"),
            "baseline_wpm": rep.get("baseline_wpm"),
            "loudness": rep.get("loudness"),
            "rtf": rep.get("rtf"),
            "gen_sec": rep.get("gen_sec"),
            "chunks": rep.get("chunks"),
            "seed": rep.get("seed"),
            "api_version": API_VERSION,
        }

    # ---- health ----------------------------------------------------------
    @api.get("/api/live")
    def live():
        """Liveness only: the process is up. Never fails while it can answer."""
        return {"status": "alive", "uptime_s": round(time.time() - STARTED_AT, 1)}

    @api.get("/api/ready")
    def ready():
        """Readiness: words actually came out of the model recently.

        Point a load balancer at this one, not /api/live.
        """
        state = READY.as_dict()
        body = {"status": "ok" if state["ready"] else "degraded",
                **state, "queue_depth": _queue_depth(),
                "vram": gpu_guard.snapshot(), **HEALTH.as_dict()}
        if not state["ready"]:
            return JSONResponse(status_code=503, content=body,
                                headers={"Retry-After": "30"})
        return body

    @api.get("/api/selftest")
    def selftest():
        res = _run_selftest()
        if res is None:
            return JSONResponse(status_code=409, content={
                "status": "busy", "detail": "the GPU is generating right now"},
                headers={"Retry-After": "10"})
        READY.record(res[0], res[1])
        body = {"status": "ok" if res[0] else "failed", "detail": res[1],
                "vram": gpu_guard.snapshot()}
        return body if res[0] else JSONResponse(status_code=503, content=body,
                                                headers={"Retry-After": "60"})

    @api.get("/api/health")
    def health():
        with JOBS_LOCK:
            q = sum(1 for j in JOBS if j["status"] == "Queued")
            p = sum(1 for j in JOBS if j["status"] == "Processing")
        body = {
            "status": "ok" if HEALTH.ok else "degraded",
            "model": "OmniVoice", "device": device_map,
            "sampling_rate": sampling_rate,
            "queue": {"queued": q, "processing": p},
            "voices": list_voice_names(),
            "vram": gpu_guard.snapshot(),
            "expandable_segments": gpu_guard.expandable_segments_state(),
            "readiness": READY.as_dict(),
            "watermark": {"enabled": WATERMARK, **watermark.status()},
            "omnivoice_version": OMNIVOICE_VERSION,
            "omnivoice_outdated": OMNIVOICE_OUTDATED,
            "features": FEATURES,
            "audit_log": AUDIT_LOG if AUDIT else None,
            "api_version": API_VERSION,
            **HEALTH.as_dict(),
        }
        if HEALTH_STRICT and not HEALTH.ok:
            # Answering "ok" while every generation returns 500 is how a
            # twenty-minute batch gets started against a dead server.
            return JSONResponse(status_code=503, content=body,
                                headers={"Retry-After": "60"})
        return body

    @api.get("/api/metrics")
    def metrics():
        with METRICS_LOCK:
            snap = dict(METRICS)
            snap["unknown_params"] = dict(snap.get("unknown_params", {}))
        audio_total = snap.get("audio_sec_total") or 0.0
        snap["rtf_overall"] = (round(snap.get("gen_sec_total", 0.0) / audio_total, 4)
                               if audio_total > 0 else None)
        snap.update({"queue_depth": _queue_depth(),
                     "uptime_s": round(time.time() - STARTED_AT, 1),
                     "vram": gpu_guard.snapshot(), **HEALTH.as_dict()})
        return snap

    # ---- voices ----------------------------------------------------------
    @api.post("/api/voices")
    def register_voice(name: str = Form(...), voice: UploadFile = File(...),
                       ref_text: str = Form(None),
                       speaker_name: str = Form(None),
                       consent: bool = Form(None),
                       consent_ref: str = Form(None),
                       x_api_key: str = Header(None)):
        owner = _auth(x_api_key)
        data = voice.file.read()
        suffix = os.path.splitext(voice.filename or "ref.wav")[1] or ".wav"
        fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="api_voice_")
        os.close(fd)
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            vid, rep = save_voice(name, tmp, ref_text, owner=owner,
                                  consent=consent, consent_ref=consent_ref,
                                  speaker_name=speaker_name)
        except ReferenceRejected as e:
            raise HTTPException(status_code=422, detail={
                "error": "reference_rejected", "message": str(e),
                "hint": "docs/reference_playbook.md"})
        finally:
            try:
                os.remove(tmp)
            except Exception:  # noqa: BLE001
                pass
        return {
            "voice_id": vid,               # frozen keys
            "ref_text": rep.get("ref_text", ""),
            "accepted": True,
            "quality_score": rep.get("quality_score"),
            "baseline_wpm": rep.get("baseline_wpm"),
            "duration_sec": rep.get("duration_sec"),
            "lufs": (rep.get("loudness") or {}).get("out_lufs"),
            "snr_db": rep.get("snr_db"),
            "warnings": rep.get("warnings", []),
            "consent": bool(consent) if consent is not None else None,
            "consent_ref": consent_ref,
            "hint": "docs/reference_playbook.md",
        }

    @api.get("/api/voices")
    def list_voices(x_api_key: str = Header(None)):
        owner = _auth(x_api_key) if MULTI_TENANT else None
        return {"voices": [voice_public(n) for n in list_voice_names(owner)]}

    @api.delete("/api/voices/{voice_id}")
    def delete_voice_api(voice_id: str, x_api_key: str = Header(None)):
        owner = _auth(x_api_key)
        if not delete_voice(voice_id, owner):
            raise HTTPException(status_code=404, detail="voice not found")
        return {"deleted": voice_id}

    # ---- synthesis -------------------------------------------------------
    _TTS_KEYS = {"text", "voice", "voice_id", "ref_text", "language", "format",
                 "steps", "speed", "project", "seed"}

    async def _tts_common(request, text, voice, voice_id, ref_text, language,
                          steps, speed, x_api_key, idempotency_key, seed,
                          allowed, queued=False):
        owner = _auth(x_api_key)
        warn = await _reject_unknown(request, allowed)
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="text is required")
        too_big = check_input_size(text, queued=queued)
        if too_big:
            raise HTTPException(status_code=413, detail={
                "error": "input_too_long", "message": too_big,
                "max_chars": (MAX_INPUT_CHARS_ASYNC if queued else MAX_INPUT_CHARS),
                "max_words": (MAX_INPUT_WORDS_ASYNC if queued else MAX_INPUT_WORDS),
                "async_endpoint": None if queued else "/api/tts/async"})
        vbytes = voice.file.read() if voice is not None else None
        try:
            prebuilt, ref_used, mode, baseline = await run_in_threadpool(
                _resolve_api_voice, vbytes, getattr(voice, "filename", None),
                voice_id, ref_text, owner)
        except KeyError:
            raise HTTPException(status_code=404, detail={
                "error": "unknown_voice",
                "message": f"No registered voice '{voice_id}'.",
                "hint": "GET /api/voices to list them, POST /api/voices to add one"})
        return owner, warn, prebuilt, ref_used, mode, baseline

    @api.post("/api/tts")
    async def tts(request: Request,
                  text: Optional[str] = Form(None), voice: UploadFile = File(None),
                  voice_id: str = Form(None), ref_text: str = Form(None),
                  language: str = Form("Auto"), format: str = Form("mp3"),
                  steps: int = Form(16), speed: float = Form(1.0),
                  project: str = Form("tts"), seed: Optional[int] = Form(None),
                  json: int = 0, x_api_key: str = Header(None),
                  idempotency_key: str = Header(None)):
        """FROZEN CONTRACT: multipart in, raw audio bytes out."""
        state, cached = _idem_lookup(idempotency_key)
        if state == "busy":
            raise HTTPException(status_code=409, detail={
                "error": "in_progress",
                "message": "a request with this Idempotency-Key is still running"},
                headers={"Retry-After": "10"})
        if state == "hit":
            _bump("idempotent_replays")
            if json:
                return JSONResponse(cached["json"])
            return Response(content=base64.b64decode(cached["audio"]),
                            media_type=cached["media"],
                            headers={**cached["headers"],
                                     "X-OmniVoice-Idempotent-Replay": "true"})
        try:
            owner, warn, prebuilt, ref_used, mode, baseline = await _tts_common(
                request, text, voice, voice_id, ref_text, language, steps, speed,
                x_api_key, idempotency_key, seed, _TTS_KEYS)
            sr, wav, status, rep = await run_in_threadpool(
                _synthesize, text.strip(), language, steps, speed, prebuilt,
                mode, seed, baseline, voice_id, owner, project)
            audio, media, ext = await run_in_threadpool(_encode_audio, wav, sr, format)
        except BaseException:
            _idem_drop(idempotency_key)
            raise

        headers = _headers(rep, project, ext, sr, wav)
        if warn:
            headers["X-OmniVoice-Warning"] = _hdr(
                (headers.get("X-OmniVoice-Warning", "") + "; " if
                 headers.get("X-OmniVoice-Warning") else "")
                + f"unknown params ignored: {warn}")
        info = status.split(" ⚠ ")[0].replace("Done. ", "").strip("() ")
        payload_json = dict(_meta(rep, project, ext, sr, wav, voice_id, ref_used, mode),
                            info=info)
        if json:
            fname = f"{_safe_name(project)}.{ext}"
            with open(os.path.join(_OUT_DIR, fname), "wb") as f:
                f.write(audio)
            payload_json.update(file=fname, download_url=f"/api/files/{fname}")
            _idem_store(idempotency_key, {"json": payload_json, "audio":
                                          base64.b64encode(audio).decode(),
                                          "media": media, "headers": headers})
            return JSONResponse(payload_json)
        _idem_store(idempotency_key, {"json": payload_json,
                                      "audio": base64.b64encode(audio).decode(),
                                      "media": media, "headers": headers})
        return Response(content=audio, media_type=media, headers=headers)

    @api.post("/api/v2/tts")
    async def tts_v2(request: Request,
                     text: Optional[str] = Form(None), voice: UploadFile = File(None),
                     voice_id: str = Form(None), ref_text: str = Form(None),
                     language: str = Form("Auto"), format: str = Form("mp3"),
                     steps: int = Form(16), speed: float = Form(1.0),
                     project: str = Form("tts"), seed: Optional[int] = Form(None),
                     inline_audio: int = Form(1), x_api_key: str = Header(None),
                     idempotency_key: str = Header(None)):
        """Same inputs, JSON out: every number the verifier produced, plus the
        audio inline (base64) and on disk."""
        allowed = _TTS_KEYS | {"inline_audio"}
        state, cached = _idem_lookup(idempotency_key)
        if state == "busy":
            raise HTTPException(status_code=409, detail={"error": "in_progress"},
                                headers={"Retry-After": "10"})
        if state == "hit":
            _bump("idempotent_replays")
            return JSONResponse(dict(cached["json"], idempotent_replay=True))
        try:
            owner, warn, prebuilt, ref_used, mode, baseline = await _tts_common(
                request, text, voice, voice_id, ref_text, language, steps, speed,
                x_api_key, idempotency_key, seed, allowed)
            sr, wav, status, rep = await run_in_threadpool(
                _synthesize, text.strip(), language, steps, speed, prebuilt,
                mode, seed, baseline, voice_id, owner, project)
            audio, media, ext = await run_in_threadpool(_encode_audio, wav, sr, format)
        except BaseException:
            _idem_drop(idempotency_key)
            raise

        fname = f"{_safe_name(project)}_{uuid.uuid4().hex[:8]}.{ext}"
        try:
            with open(os.path.join(_OUT_DIR, fname), "wb") as f:
                f.write(audio)
            download = f"/api/files/{fname}"
        except Exception as e:  # noqa: BLE001
            log.warning("could not save %s: %s", fname, e)
            download = None
        body = _meta(rep, project, ext, sr, wav, voice_id, ref_used, mode)
        body.update(file=fname, download_url=download, media_type=media,
                    info=status.split(" ⚠ ")[0].replace("Done. ", "").strip("() "))
        if warn:
            body.setdefault("warnings", []).append(
                f"unknown params ignored: {warn}")
        if inline_audio:
            body["audio_base64"] = base64.b64encode(audio).decode()
        _idem_store(idempotency_key, {"json": body})
        return JSONResponse(body)

    @api.post("/api/tts/async")
    async def tts_async(request: Request,
                        text: Optional[str] = Form(None), voice: UploadFile = File(None),
                        voice_id: str = Form(None), ref_text: str = Form(None),
                        language: str = Form("Auto"), steps: int = Form(16),
                        speed: float = Form(1.0), project: str = Form("tts"),
                        seed: Optional[int] = Form(None),
                        x_api_key: str = Header(None)):
        allowed = _TTS_KEYS - {"format"}
        owner, warn, prebuilt, ref_used, mode, baseline = await _tts_common(
            request, text, voice, voice_id, ref_text, language, steps, speed,
            x_api_key, None, seed, allowed, queued=True)
        jid = await run_in_threadpool(
            _api_enqueue, project, text.strip(), prebuilt, ref_used, language,
            steps, speed, seed, baseline, voice_id, owner)
        body = {"job_id": jid, "project": _safe_name(project), "status": "queued",
                "queue_depth": _queue_depth()}
        if warn:
            body["warnings"] = [f"unknown params ignored: {warn}"]
        return body

    def _job_public(j):
        rep = j.get("report") or {}
        return {"job_id": j["id"], "project": j["project"],
                "status": j["status"].lower().rstrip("…"),
                "info": j["info"],
                "warnings": j.get("warnings", []),
                "verified": rep.get("verified", False),
                "verification": rep.get("diffs", []),
                "wpm": rep.get("wpm"), "loudness": rep.get("loudness"),
                "rtf": rep.get("rtf"), "gen_sec": rep.get("gen_sec"),
                "audio_sec": rep.get("audio_sec"), "chunks": rep.get("chunks"),
                "normalized_text": rep.get("normalized_text"),
                "download_url": (f"/api/jobs/{j['id']}/download" if j["file"] else None)}

    @api.get("/api/jobs/{jid}")
    def job_status(jid: int):
        with JOBS_LOCK:
            j = _find_job(jid)
            if not j:
                raise HTTPException(status_code=404, detail="job not found")
            return _job_public(j)

    @api.get("/api/jobs")
    def job_list(limit: int = 50):
        with JOBS_LOCK:
            return {"jobs": [_job_public(j) for j in JOBS[-max(1, limit):]]}

    @api.get("/api/jobs/{jid}/download")
    def job_download(jid: int, format: str = "mp3"):
        with JOBS_LOCK:
            j = _find_job(jid)
            if not j:
                raise HTTPException(status_code=404, detail="job not found")
            path, proj, status = j["file"], j["project"], j["status"]
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=409, detail={
                "error": "not_ready", "status": status.lower().rstrip("…")},
                headers={"Retry-After": "5"})
        wav, sr = sf.read(path, dtype="int16")
        audio, media, ext = _encode_audio(wav, sr, format)
        return Response(content=audio, media_type=media, headers={
            "Content-Disposition": f'attachment; filename="{proj}.{ext}"'})

    @api.post("/api/transcribe")
    async def transcribe_api(request: Request, audio: UploadFile = File(...),
                             text: str = Form(None),
                             x_api_key: str = Header(None)):
        """Transcribe a clip with the Whisper that is already loaded here.

        This exists so the batch audit in the bug report can be reproduced
        without a paid ASR service: send a generated clip plus the script it
        was made from, and get back the same word-level diff the server uses
        internally to check itself.
        """
        _auth(x_api_key)
        data = await audio.read()
        suffix = os.path.splitext(audio.filename or "clip.wav")[1] or ".wav"
        fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="api_asr_")
        os.close(fd)
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            with _gen_slot():
                heard = await run_in_threadpool(_transcribe_path, tmp)
        finally:
            try:
                os.remove(tmp)
            except Exception:  # noqa: BLE001
                pass
        body: Dict[str, Any] = {"text": heard}
        if text:
            body["diff"] = verify.word_diff(text, heard)
            body["ok"] = verify.passed(body["diff"])
            body["summary"] = verify.describe(body["diff"])
        return body

    @api.post("/api/watermark/detect")
    async def watermark_detect(audio: UploadFile = File(...),
                               x_api_key: str = Header(None)):
        """"Detect on complaint": was this clip generated here?

        The other half of EU AI Act Article 50 marking — a mark nobody can read
        back is not provenance.
        """
        _auth(x_api_key)
        data = await audio.read()
        suffix = os.path.splitext(audio.filename or "clip.wav")[1] or ".wav"
        fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="api_wm_")
        os.close(fd)
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            x, sr = await run_in_threadpool(read_audio, tmp)
        finally:
            try:
                os.remove(tmp)
            except Exception:  # noqa: BLE001
                pass
        return await run_in_threadpool(watermark.detect, x, sr)

    @api.get("/api/files/{name}")
    def get_file(name: str):
        from fastapi.responses import FileResponse
        safe = os.path.basename(name)
        path = os.path.join(_OUT_DIR, safe)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(path, filename=safe)

    return api


def _start_api_server():
    try:
        import uvicorn
        uvicorn.run(_build_api(), host="0.0.0.0", port=API_PORT, log_level="warning")
    except Exception as e:  # noqa: BLE001
        log.warning("API server failed to start: %s", e)


def _lan_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0]
            if ":" not in ip and not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    return sorted(ips)


def _allowed_paths():
    paths = [OUTPUT_DIR, VOICES_DIR]
    extra = os.environ.get("OMNIVOICE_ALLOWED_PATHS", "")
    for p in extra.replace(",", os.pathsep).split(os.pathsep):
        p = p.strip().strip('"')
        if p and os.path.isdir(p):
            paths.append(p)
    seen, out = set(), []
    for p in paths:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


if __name__ == "__main__":
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    share = _env_flag("GRADIO_SHARE", "0")
    start_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

    _auth_env = os.environ.get("OMNIVOICE_AUTH", "").strip()
    auth = tuple(_auth_env.split(":", 1)) if ":" in _auth_env else None
    if auth:
        print(f"Login required: user '{auth[0]}'")

    _api_on = _env_flag("OMNIVOICE_API")
    if _api_on:
        threading.Thread(target=_start_api_server, daemon=True).start()
    if SELFTEST:
        threading.Thread(target=_selftest_loop, daemon=True).start()
    if PREWARM_VOICES:
        threading.Thread(target=_prewarm_voices, daemon=True).start()

    app = demo.queue()
    last_err = None
    for port in range(start_port, start_port + 11):
        try:
            _ips = _lan_ips() if server_name == "0.0.0.0" else []
            print("=" * 60)
            print(f"  UI    (local):   http://127.0.0.1:{port}")
            for ip in _ips:
                print(f"  UI    (LAN):     http://{ip}:{port}")
            if _api_on:
                print(f"  API   (local):   http://127.0.0.1:{API_PORT}/api")
                for ip in _ips:
                    print(f"  API   (LAN):     http://{ip}:{API_PORT}/api")
                print(f"  API   docs:      http://127.0.0.1:{API_PORT}/api/docs")
                print(f"  API   ready:     http://127.0.0.1:{API_PORT}/api/ready"
                      "   <- point monitoring here")
            if WATERMARK:
                print(f"  watermark:       {watermark.status()}")
            _feat = ", ".join(k for k, v in FEATURES.items() if v) or "none"
            print(f"  omnivoice {OMNIVOICE_VERSION} · features: {_feat}"
                  + ("   <- OUTDATED, run pip install -r requirements.txt"
                     if OMNIVOICE_OUTDATED else ""))
            print(f"  verify={'on' if VERIFY else 'off'} · "
                  f"normalize={NORMALIZE_LEVEL} · "
                  f"loudness={'on' if NORMALIZE_OUTPUT else 'off'} "
                  f"({OUT_TARGET_LUFS:.0f} LUFS) · concurrency={MAX_CONCURRENCY}"
                  + (f" · tenants={len(TENANT_KEYS)}" if MULTI_TENANT else ""))
            _seg = gpu_guard.expandable_segments_state()
            if _seg["note"]:
                print(f"  note: expandable_segments {_seg['note']}")
            print("=" * 60)
            app.launch(
                server_name=server_name,
                server_port=port,
                share=share,
                auth=auth,
                theme=_THEME,
                css=_CSS,
                allowed_paths=_allowed_paths(),
                inbrowser=_env_flag("OMNIVOICE_OPEN_BROWSER"),
            )
            last_err = None
            break
        except OSError as e:
            last_err = e
            print(f"Port {port} busy, trying {port + 1} ...")
    if last_err is not None:
        raise last_err
