#!/usr/bin/env python3
"""
Local Gradio launcher for OmniVoice (k2-fsa/OmniVoice).

Adapted from the official HuggingFace Space app.py, with the ZeroGPU
(`spaces`) wrapper removed so it runs on a local CUDA GPU, plus fixes for
several well-known OmniVoice quality/speed issues:

  1. Long reference audio degrades quality (GitHub #50)
     -> reference audio is auto-trimmed to <= OMNIVOICE_MAX_REF_SEC (default 10s;
        the model was trained on 3-10s references, ~6s is the sweet spot).

  2. Number normalization ("123" -> garbled)
     -> text is run through a num2words-based front-end before synthesis.

  3. Mathematical / phone-number / long digit sequences sound wrong
     -> long digit runs and phone-style numbers are read digit-by-digit.

  4. Slow inference on consumer GPUs
     -> FP16 (already default), optional torch.compile of the LLM backbone
        (OMNIVOICE_COMPILE=1), and an optional step-count override
        (OMNIVOICE_NUM_STEP, e.g. 16) for a 20-30% speedup.

Run:  python app.py
The model (~3.3 GB) auto-downloads from HuggingFace on first launch and is
cached under ~/.cache/huggingface (or HF_HOME if set).

Environment variables:
  OMNIVOICE_MODEL        model id / path        (default: k2-fsa/OmniVoice)
  OMNIVOICE_MAX_REF_SEC  max reference seconds  (default: 10)
  OMNIVOICE_NORMALIZE    number normalization   (default: 1, set 0 to disable)
  OMNIVOICE_NUM_STEP     force inference steps   (unset = use UI slider; 16 = faster)
  OMNIVOICE_COMPILE      torch.compile the LLM   (default: 0; needs Triton)
  GRADIO_SERVER_NAME / GRADIO_SERVER_PORT / GRADIO_SHARE
"""

import datetime
import html as html_mod
import logging
import os
import queue as queue_mod
import socket
import re
import sys
import tempfile
import threading
import time
import zipfile
from typing import Any, Dict, Optional

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
from omnivoice.cli.demo import _ALL_LANGUAGES, _ATTR_INFO, _CATEGORIES
from omnivoice.utils.lang_map import LANG_NAME_TO_ID

# Number normalization front-end -------------------------------------------------
try:
    from num2words import num2words
    from num2words import CONVERTER_CLASSES as _N2W_CLASSES

    _N2W_LANGS = set(_N2W_CLASSES.keys())
except Exception:  # pragma: no cover - num2words optional
    num2words = None
    _N2W_LANGS = set()

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_MODEL = os.path.join(_HERE, "Model")


def _default_checkpoint() -> str:
    """Prefer the local ./Model folder (already downloaded) over the HF repo."""
    if os.path.isfile(os.path.join(_LOCAL_MODEL, "config.json")):
        return _LOCAL_MODEL
    return "k2-fsa/OmniVoice"


# OMNIVOICE_MODEL overrides; otherwise use the local ./Model dir if present.
CHECKPOINT = os.environ.get("OMNIVOICE_MODEL") or _default_checkpoint()
# Whisper ASR for auto-transcribing reference text. Point this at a local path
# to avoid downloading it from HuggingFace.
ASR_MODEL = os.environ.get("OMNIVOICE_ASR_MODEL", "openai/whisper-large-v3-turbo")
MAX_REF_SEC = float(os.environ.get("OMNIVOICE_MAX_REF_SEC", "10"))
NORMALIZE = os.environ.get("OMNIVOICE_NORMALIZE", "1") == "1"
FORCE_NUM_STEP = os.environ.get("OMNIVOICE_NUM_STEP")  # None or e.g. "16"
DO_COMPILE = os.environ.get("OMNIVOICE_COMPILE", "0") == "1"
CHUNK = os.environ.get("OMNIVOICE_CHUNK", "1") == "1"
MAX_CHARS = int(os.environ.get("OMNIVOICE_MAX_CHARS", "100"))
GAP_SEC = float(os.environ.get("OMNIVOICE_CHUNK_GAP", "0.15"))
# Chunks generated together per GPU batch. Default 1 (sequential): batching
# pads all chunks to the longest one, which wastes compute on short chunks and
# usually *raises* RTF. Opt in with OMNIVOICE_BATCH>1 only if chunks are long
# and uniform. Auto-halves on CUDA OOM.
BATCH = int(os.environ.get("OMNIVOICE_BATCH", "1"))


# ---------------------------------------------------------------------------
# Fix #2 / #3: text / number normalization
# ---------------------------------------------------------------------------
# Capture a digit run that may contain ., , or - as internal separators.
_NUM_RE = re.compile(r"\d[\d.,\-]*\d|\d")


def _resolve_n2w_lang(language: Optional[str], text: str) -> Optional[str]:
    """Map an OmniVoice language label to a num2words language code.

    Returns None when normalization should be skipped (unsupported language or
    auto-detect on non-Latin text), so we never corrupt non-Latin scripts.
    """
    if num2words is None:
        return None

    if not language or language == "Auto":
        # Only safe to assume English digits for predominantly ASCII text.
        if any(ord(c) > 0x2FF for c in text):
            return None
        return "en" if "en" in _N2W_LANGS else None

    code = LANG_NAME_TO_ID.get(language.lower(), language.lower())
    if code in _N2W_LANGS:
        return code
    base = code.split("_")[0]
    return base if base in _N2W_LANGS else None


def normalize_numbers(text: str, language: Optional[str]) -> str:
    """Convert digit sequences to spoken words (language-aware).

    - Long runs (>= 7 digits, e.g. phone numbers) -> read digit by digit.
    - Decimals -> "<int> point <digits>" via num2words float handling.
    - Plain integers (incl. thousands separators) -> cardinal words.
    Falls back to leaving the original token untouched on any failure.
    """
    if not NORMALIZE or not text:
        return text
    code = _resolve_n2w_lang(language, text)
    if code is None:
        return text

    def repl(m: "re.Match") -> str:
        tok = m.group(0)
        digits = re.sub(r"\D", "", tok)
        if not digits:
            return tok
        try:
            # Phone-like / long sequences -> digit by digit.
            if len(digits) >= 7:
                return " ".join(num2words(int(d), lang=code) for d in digits)
            # Ambiguous short hyphenated tokens (e.g. "1-2") -> leave as-is.
            if "-" in tok:
                return tok
            clean = tok.replace(",", "")
            if "." in clean:
                return num2words(float(clean), lang=code)
            return num2words(int(clean), lang=code)
        except Exception:
            return tok

    out = _NUM_RE.sub(repl, text)
    if out != text:
        log.info("Normalized text [%s]: %r -> %r", code, text, out)
    return out


# ---------------------------------------------------------------------------
# Fix: long-text degeneration into non-speech audio (GitHub #144) +
#      voice drift across chunks (GitHub #44).
# Split long text on sentence boundaries into ~MAX_CHARS chunks, synthesize
# each chunk separately, and concatenate. A single shared voice (one
# VoiceClonePrompt) is reused across all chunks to keep the timbre consistent.
# ---------------------------------------------------------------------------
# Sentence terminators for both Latin (. ! ?) and CJK (。！？…) scripts.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？…])\s*")


def chunk_text(text: str, max_chars: int = MAX_CHARS):
    """Split text at sentence boundaries, packing sentences up to max_chars."""
    sentences = [s for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]
    chunks, current = [], ""
    for s in sentences:
        s = s.strip()
        if not current:
            current = s
        elif len(current) + 1 + len(s) <= max_chars:
            current += " " + s
        else:
            chunks.append(current)
            current = s
        # A single sentence longer than max_chars stays as its own chunk.
    if current:
        chunks.append(current)
    return chunks or [text]


def concat_audio(parts, sr, gap_sec=GAP_SEC):
    """Concatenate float waveforms with a short silence between them."""
    if not parts:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(gap_sec * sr), dtype=np.float32)
    out = []
    for i, p in enumerate(parts):
        if i:
            out.append(gap)
        out.append(np.asarray(p, dtype=np.float32).reshape(-1))
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Fix #1: trim long reference audio (GitHub #50)
# ---------------------------------------------------------------------------
def trim_reference(path: str):
    """Trim a reference clip to the first MAX_REF_SEC seconds if it is longer.

    Returns (path_to_use, duration_seconds, was_trimmed). On any error the
    original path is returned unchanged.
    """
    try:
        info = sf.info(path)
        dur = info.frames / float(info.samplerate)
    except Exception as e:
        log.warning("Could not read reference audio %s: %s", path, e)
        return path, None, False

    if dur <= MAX_REF_SEC:
        return path, dur, False

    try:
        data, sr = sf.read(path, frames=int(MAX_REF_SEC * info.samplerate))
        fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="omnivoice_ref_")
        os.close(fd)
        sf.write(tmp, data, sr)
        log.info("Trimmed reference %.1fs -> %.1fs", dur, MAX_REF_SEC)
        return tmp, MAX_REF_SEC, True
    except Exception as e:
        log.warning("Reference trim failed, using original: %s", e)
        return path, dur, False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device_map = "cuda"
    dtype = torch.float16  # Fix #4: FP16 by default
    # Free perf tweak: TF32 matmuls. (cudnn.benchmark is intentionally OFF — TTS
    # inputs vary in length, so autotuning re-runs per shape and HURTS latency.)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    print(f"Loading model from {CHECKPOINT} to CUDA ({torch.cuda.get_device_name(0)}) ...")
else:
    device_map = "cpu"
    dtype = torch.float32
    print(f"CUDA not available -> loading {CHECKPOINT} on CPU (this will be slow) ...")

print(f"Checkpoint: {CHECKPOINT}")
# Attention backend. NOTE: OmniVoice (as of 0.1.5) does NOT support
# flash_attention_2 — it raises at load — so we default to "sdpa" (fast, built
# in, no extra package). You can still try flash via OMNIVOICE_ATTN if a future
# model version adds support; it auto-falls back to sdpa on failure.
_ATTN = os.environ.get("OMNIVOICE_ATTN", "sdpa")


def _load_model(attn):
    return OmniVoice.from_pretrained(
        CHECKPOINT,
        device_map=device_map,
        dtype=dtype,
        load_asr=True,
        asr_model_name=ASR_MODEL,
        attn_implementation=attn,
    )


try:
    model = _load_model(_ATTN)
    print(f"Model loaded successfully! (attention: {_ATTN})")
except Exception as e:
    if _ATTN != "sdpa":
        log.warning("attn '%s' failed (%s); falling back to sdpa", _ATTN, e)
        model = _load_model("sdpa")
        print("Model loaded successfully! (attention: sdpa)")
    else:
        raise
sampling_rate = model.sampling_rate

# Fix #4: optional torch.compile of the LLM backbone (the dominant cost).
# Disabled by default: on Windows the Inductor backend needs Triton, which has
# no official Windows wheels, so compilation typically fails there.
if DO_COMPILE:
    try:
        if hasattr(model, "llm") and model.llm is not None:
            model.llm = torch.compile(model.llm)
            log.info("torch.compile enabled on model.llm")
        else:
            log.warning("torch.compile requested but model.llm not found; skipped")
    except Exception as e:
        log.warning("torch.compile failed (%s); continuing uncompiled", e)


# Single GPU -> one model op at a time. Serializes the batch worker and the
# interactive tabs so they never touch CUDA concurrently.
MODEL_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Generation core (mirrors omnivoice.cli.demo, with the fixes applied)
# ---------------------------------------------------------------------------
class _Cancelled(Exception):
    """Raised to abort generation when a job is cancelled."""


def _gen_core(*args, **kwargs):
    # Note: the GPU lock is taken per model call (per chunk / per prompt / per
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
):
    if not text or not text.strip():
        return None, "Please enter the text to synthesize."

    notes = []

    # Fix #4: optional forced step count (e.g. OMNIVOICE_NUM_STEP=16 for speed).
    if FORCE_NUM_STEP:
        num_step = FORCE_NUM_STEP
        notes.append(f"steps forced to {FORCE_NUM_STEP}")

    # Fix #2/#3: normalize numbers before synthesis.
    syn_text = normalize_numbers(text.strip(), language)
    if syn_text != text.strip():
        notes.append("numbers normalized")

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang = language if (language and language != "Auto") else None

    kw: Dict[str, Any] = dict(
        text=syn_text, language=lang, generation_config=gen_config
    )

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    instruct_val = instruct.strip() if (instruct and instruct.strip()) else None

    # Decide chunking (Fix #144): split long text on sentence boundaries.
    # Skip when a fixed Duration is set — that is a whole-utterance target that
    # cannot be meaningfully divided across chunks.
    duration_set = duration is not None and float(duration) > 0
    if CHUNK and not duration_set and len(syn_text) > MAX_CHARS:
        chunks = chunk_text(syn_text, MAX_CHARS)
    else:
        chunks = [syn_text]

    # Build the (single, shared) voice for the whole utterance.
    clone_prompt = None
    if prebuilt_prompt is not None:
        # Reuse a voice prompt built once (bulk add: same voice, many scripts).
        clone_prompt = prebuilt_prompt
    elif mode == "clone":
        if not ref_audio:
            return None, "Please upload a reference audio."
        # Fix #1: trim overly long references (model trained on 3-10s clips).
        ref_path, ref_dur, trimmed = trim_reference(ref_audio)
        if trimmed:
            notes.append(
                f"reference trimmed {ref_dur:.0f}s -> {MAX_REF_SEC:.0f}s"
                if ref_dur else "reference trimmed"
            )
        elif ref_dur is not None and ref_dur < 3:
            notes.append("warning: reference < 3s may reduce quality")
        try:
            # ref_text=None -> auto-transcribed by multilingual Whisper.
            with MODEL_LOCK:
                clone_prompt = model.create_voice_clone_prompt(
                    ref_audio=ref_path,
                    ref_text=ref_text,
                )
        except Exception as e:
            return None, f"Error creating voice prompt: {type(e).__name__}: {e}"
        if not ref_text:
            notes.append("ref text auto-transcribed (Whisper)")

    # Generate in mini-batches for GPU parallelism. A single shared voice
    # prompt is broadcast across the batch, keeping the timbre consistent
    # (Fix #44). Batch size auto-halves on CUDA OOM.
    def _run_batches(texts, prompt, inst, batch_size):
        out_parts = []
        i = 0
        bs = max(1, batch_size)
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
            try:
                with MODEL_LOCK:
                    outs = model.generate(**bkw)
                out_parts.extend(
                    np.asarray(a, dtype=np.float32).reshape(-1) for a in outs
                )
                i += bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                log.warning("CUDA OOM -> reducing batch size to %d", bs)
        return out_parts

    t0 = time.perf_counter()
    try:
        if clone_prompt is not None or len(chunks) == 1:
            # Clone mode, or a single chunk (instruct applies for design).
            parts = _run_batches(chunks, clone_prompt, instruct_val, BATCH)
        else:
            # Design mode + multiple chunks: synthesize the first chunk, lock the
            # voice from its output, then batch the rest with that voice (Fix #44).
            parts = _run_batches([chunks[0]], None, instruct_val, 1)
            try:
                with MODEL_LOCK:
                    clone_prompt = model.create_voice_clone_prompt(
                        ref_audio=(torch.from_numpy(parts[0]).float(), sampling_rate),
                        ref_text=chunks[0],
                    )
            except Exception as e:
                log.warning("Could not lock design voice across chunks: %s", e)
            parts += _run_batches(chunks[1:], clone_prompt, None, BATCH)

        waveform_f = concat_audio(parts, sampling_rate)
    except _Cancelled:
        return None, "Cancelled"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    gen_time = time.perf_counter() - t0

    if len(chunks) > 1:
        notes.append(f"{len(chunks)} chunks" + (f", batch {BATCH}" if BATCH > 1 else ""))

    # RTF (real-time factor) = compute time / audio duration. Lower is faster;
    # < 1.0 means faster than real time.
    audio_dur = len(waveform_f) / sampling_rate if sampling_rate else 0.0
    rtf = (gen_time / audio_dur) if audio_dur > 0 else 0.0
    notes.append(f"{gen_time:.1f}s for {audio_dur:.1f}s audio · RTF {rtf:.3f}")

    waveform = (np.clip(waveform_f, -1.0, 1.0) * 32767).astype(np.int16)
    status = "Done. (" + "; ".join(notes) + ")"
    return (sampling_rate, waveform), status


# ---------------------------------------------------------------------------
# Auto-transcribe reference audio on upload (fills the Reference Text box).
# Combines Fix #1 (trim to <=MAX_REF_SEC) with multilingual Whisper ASR so the
# user sees and can edit the transcript before generating.
# ---------------------------------------------------------------------------
def transcribe_reference(ref_audio):
    if not ref_audio:
        return "", "Upload a reference clip — it will be auto-transcribed here."
    ref_path, dur, trimmed = trim_reference(ref_audio)
    try:
        with MODEL_LOCK:
            text = model.transcribe(ref_path)
    except Exception as e:
        return "", f"Auto-transcribe failed: {type(e).__name__}: {e}"
    msg = f"Transcribed ~{dur:.1f}s reference." if dur else "Transcribed reference."
    if trimmed:
        msg += f" Using first {MAX_REF_SEC:.0f}s for cloning."
    elif dur is not None and dur < 3:
        msg += " Warning: < 3s may reduce quality."
    return text, msg


# ---------------------------------------------------------------------------
# Voiceover Studio — project queue (processed one at a time by a worker thread)
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get("OMNIVOICE_OUTPUT_DIR", os.path.join(_HERE, "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Live save folder (user-changeable in the UI). Voices auto-save here.
_OUT_DIR = OUTPUT_DIR
# Shut-down-on-complete toggle.
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
        return f"📂 Voices will auto-save to: {p}"
    except Exception as e:
        return f"❌ Can't use that folder: {e}"


def set_shutdown(on):
    _SHUTDOWN["on"] = bool(on)
    _SHUTDOWN["fired"] = False
    return ("🛑 PC will shut down ~60s after the queue finishes "
            "(run `shutdown /a` in CMD to cancel)." if on
            else "Shutdown-on-complete is OFF.")


def _maybe_shutdown():
    if not _SHUTDOWN["on"] or _SHUTDOWN["fired"]:
        return
    with JOBS_LOCK:
        pending = any(j["status"] in ("Queued", "Processing", "Cancelling…")
                      for j in JOBS)
    if not pending:
        _SHUTDOWN["fired"] = True
        log.warning("Queue complete -> shutting down in 60s. Cancel: shutdown /a")
        try:
            os.system('shutdown /s /t 60 /c "OmniVoice: queue complete"')
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
            out, status = _gen_core(
                p.get("script", ""), p.get("language"), p.get("ref_audio"),
                p.get("instruct"),
                p.get("num_step", 16), p.get("guidance_scale", 2.0),
                p.get("denoise", True),
                p.get("speed", 1.0), p.get("duration"),
                p.get("preprocess", True), p.get("postprocess", True),
                mode=mode, ref_text=p.get("ref_text"),
                cancel_check=lambda jid=job_id: _is_cancelled(jid),
                prebuilt_prompt=prebuilt,
            )
            with JOBS_LOCK:
                if status == "Cancelled":
                    job["status"], job["info"] = "Cancelled", ""
                elif out is None:
                    job["status"], job["info"] = "Error", status
                else:
                    sr, wav = out
                    path = _unique_output_path(job["project"])
                    sf.write(path, wav, sr)
                    job["status"] = "Done"
                    job["info"] = status.replace("Done. ", "").strip("() ")
                    job["file"] = path
        except Exception as e:  # noqa: BLE001 - keep the worker alive
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
    # Only surface files that still exist — a moved/deleted output must never
    # break the queue UI (gr.Files errors on missing paths otherwise).
    return [f for f in files if os.path.exists(f)]


def _job_choices():
    with JOBS_LOCK:
        return [(f"#{j['id']} · {j['project']}  [{j['status']}]", j["id"]) for j in JOBS]


def _picker_update(cur=None):
    choices = _job_choices()
    valid = cur if (cur is not None and any(v == cur for _, v in choices)) else None
    return gr.update(choices=choices, value=valid)


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
    header = f'<div class="vq-pills">{pills}</div>'

    if not jobs:
        return header + ('<div class="vq-empty">🎙️ No projects yet — fill the form '
                         'on the left and click <b>Add to Queue</b>.</div>')

    cards = []
    for j in jobs:
        status = j["status"]
        cls = _BADGE_CLASS.get(status, "badge-queued")
        icon = _STATUS_ICON.get(status, "")
        name = html_mod.escape(j["project"])
        info = html_mod.escape(j["info"] or "")
        script_preview = html_mod.escape((j["params"]["script"] or "")[:60])
        cloned = bool(j["params"].get("clone_prompt"))
        voice = "🎤 cloned" if cloned else "🎛️ designed"
        # Sub-line: voice + (timing/RTF once done, else a script preview).
        detail = info if info else (f"{script_preview}…" if script_preview else "")
        sub = f"{voice} · {detail}" if detail else voice
        # Voice transcript actually used for the clone (so you can verify it).
        ref_used = html_mod.escape((j.get("ref_used") or "")[:60])
        ref_line = (f'<div class="vq-ref">🗣️ voice: "{ref_used}…"</div>'
                    if ref_used else "")
        dl = ""
        if status == "Done" and j["file"]:
            fname = html_mod.escape(os.path.basename(j["file"]))
            dl = f'<div class="vq-file">💾 {fname}</div>'
        cards.append(
            f'<div class="vq-card {cls}">'
            f'<div class="vq-left">'
            f'<span class="vq-id">#{j["id"]}</span>'
            f'<div class="vq-meta"><div class="vq-name">{name}</div>'
            f'<div class="vq-sub">{sub}</div>{ref_line}{dl}</div>'
            f'</div>'
            f'<div class="vq-right">'
            f'<span class="vq-badge {cls}">{icon} {status}</span>'
            f'</div></div>'
        )
    return header + '<div class="vq-board">' + "".join(cards) + "</div>"


def _ui_state(msg="", cur=None):
    return _jobs_html(), _download_files(), _picker_update(cur), msg


def _build_prompt(ref_audio, ref_text):
    """Build a voice clone prompt now and return (prompt, transcript_used)."""
    ref_path, _, _ = trim_reference(ref_audio)
    with MODEL_LOCK:
        p = model.create_voice_clone_prompt(
            ref_audio=ref_path, ref_text=(ref_text or None),
        )
    return p, (p.ref_text or "")


# Cache the last-built voice so adding many scripts with the SAME clip does not
# re-transcribe / re-encode each time. Rebuilds only when the clip (or its
# transcript) changes.
_VOICE_CACHE = {"key": None, "prompt": None, "ref_text": ""}


def _get_or_build_prompt(ref_audio, ref_text):
    key = (ref_audio, (ref_text or "").strip())
    if _VOICE_CACHE["key"] == key and _VOICE_CACHE["prompt"] is not None:
        return _VOICE_CACHE["prompt"], _VOICE_CACHE["ref_text"]
    prompt, used = _build_prompt(ref_audio, ref_text)
    _VOICE_CACHE.update(key=key, prompt=prompt, ref_text=used)
    return prompt, used


def add_job(project, script, ref_audio, ref_text, language, num_step,
            guidance_scale=2.0, denoise=True, speed=1.0, duration=None,
            preprocess=True, postprocess=True):
    global _JOB_SEQ
    if not script or not script.strip():
        return _ui_state("⚠️ Script is empty — nothing added.")
    prebuilt, ref_used = None, ""
    if ref_audio:
        try:
            prebuilt, ref_used = _get_or_build_prompt(ref_audio, ref_text)
        except Exception as e:
            return _ui_state(f"❌ Voice prompt failed: {type(e).__name__}: {e}")
    with JOBS_LOCK:
        _JOB_SEQ += 1
        jid = _JOB_SEQ
        proj = (project or "").strip() or f"Project_{jid}"
        JOBS.append({
            "id": jid, "project": proj, "status": "Queued", "info": "",
            "file": None, "cancel": False, "ref_used": ref_used,
            "params": {
                "script": script, "ref_audio": None, "clone_prompt": prebuilt,
                "ref_text": (ref_text or None), "language": language,
                "num_step": int(num_step or 16),
                "guidance_scale": float(guidance_scale),
                "denoise": bool(denoise),
                "speed": float(speed) if speed else 1.0,
                "duration": (float(duration) if duration else None),
                "preprocess": bool(preprocess),
                "postprocess": bool(postprocess),
            },
        })
    JOB_Q.put(jid)
    return _ui_state(f"✅ Added '{proj}' to the queue.")


def _cell(v):
    """Normalize a table cell to a clean string ('' for empty/NaN)."""
    if v is None:
        return ""
    try:
        # NaN check (float('nan') != itself)
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
    # pandas DataFrame
    try:
        import pandas as pd
        if isinstance(rows, pd.DataFrame):
            return rows.values.tolist()
    except Exception:
        pass
    # dict form: {"headers": [...], "data": [[...], ...]}
    if isinstance(rows, dict):
        return rows.get("data") or rows.get("value") or []
    # list / tuple
    if isinstance(rows, (list, tuple)):
        return list(rows)
    # numpy array or anything with tolist()
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
              preprocess=True, postprocess=True):
    """Queue one project per table row (columns: Project name, Script).

    All rows share the same (cached) voice. Empty-script rows are skipped.
    """
    global _JOB_SEQ
    data = _rows_to_list(rows)
    items = []
    for r in data:
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

    prebuilt, ref_used = None, ""
    if ref_audio:
        try:
            prebuilt, ref_used = _get_or_build_prompt(ref_audio, ref_text)
        except Exception as e:
            return _ui_state(f"❌ Voice prompt failed: {type(e).__name__}: {e}")

    new_ids = []
    with JOBS_LOCK:
        for name, scr in items:
            _JOB_SEQ += 1
            jid = _JOB_SEQ
            proj = _safe_name(name) if name and name.lower() != "nan" else f"clip_{jid}"
            JOBS.append({
                "id": jid, "project": proj, "status": "Queued",
                "info": "", "file": None, "cancel": False, "ref_used": ref_used,
                "params": {
                    "script": scr, "ref_audio": None, "clone_prompt": prebuilt,
                    "ref_text": (ref_text or None), "language": language,
                    "num_step": int(num_step or 16),
                    "guidance_scale": float(guidance_scale), "denoise": bool(denoise),
                    "speed": float(speed) if speed else 1.0,
                    "duration": (float(duration) if duration else None),
                    "preprocess": bool(preprocess), "postprocess": bool(postprocess),
                },
            })
            new_ids.append(jid)
    for jid in new_ids:
        JOB_Q.put(jid)
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


def _gen_settings():
    with gr.Accordion("Generation Settings (optional)", open=False):
        sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed",
                       info="1.0 = normal. Ignored if Duration is set.")
        du = gr.Number(value=None, label="Duration (seconds)",
                       info="Leave empty to use speed. Disables chunking.")
        ns = gr.Slider(4, 64, value=16, step=1, label="Inference Steps",
                       info="Default 16 (fast). Raise to 32+ for higher quality.")
        dn = gr.Checkbox(label="Denoise", value=True)
        gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale (CFG)")
        pp = gr.Checkbox(label="Preprocess Prompt", value=True,
                         info="Silence removal/trim on reference audio.")
        po = gr.Checkbox(label="Postprocess Output", value=True,
                         info="Remove long silences from output.")
    return ns, gs, dn, sp, du, pp, po


def build_ui() -> gr.Blocks:
    # In Gradio 6, theme/css are passed to launch(), not the Blocks constructor.
    with gr.Blocks(title="OmniVoice Demo") as demo:
        gr.Markdown(
            "# 🎙️ Voiceover Studio\n"
            "**1)** pick a voice  →  **2)** add your script(s)  →  **3)** render. "
            "Every clip is saved to the `outputs/` folder."
        )

        with gr.Row(equal_height=False):
            # ============ LEFT: create ============
            with gr.Column(scale=5):
                gr.Markdown("### 1️⃣  Pick a voice")
                st_ref = gr.Audio(
                    label="Upload a 3–10s voice clip   ·   leave empty = AI voice",
                    type="filepath",
                )
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
                            info="Default 2.0.",
                        )
                        st_duration = gr.Number(
                            value=None, label="Duration (seconds)",
                            info="Empty = auto. Fixed value disables chunking.",
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
        def _on_studio_ref(ref_audio):
            text, msg = transcribe_reference(ref_audio)
            return text, f"<span style='color:#888;font-size:0.85em'>{msg}</span>"

        st_ref.change(_on_studio_ref, inputs=[st_ref],
                      outputs=[st_ref_text, st_msg])

        _OUT = [st_board, st_files, st_picker, st_msg]
        st_add.click(
            add_job,
            inputs=[st_project, st_script, st_ref, st_ref_text, st_lang, st_steps,
                    st_cfg, st_denoise, st_speed, st_duration, st_preprocess,
                    st_postprocess],
            outputs=_OUT,
        ).then(lambda: ("", ""), outputs=[st_project, st_script])

        st_addrow.click(add_row, inputs=[st_table_in], outputs=[st_table_in],
                        api_name="add_row")
        st_clearrows.click(clear_rows, outputs=[st_table_in])
        st_addtable.click(
            add_table,
            inputs=[st_table_in, st_ref, st_ref_text, st_lang, st_steps, st_cfg,
                    st_denoise, st_speed, st_duration, st_preprocess, st_postprocess],
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

    return demo


demo = build_ui()

_THEME = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
_CSS = """
.gradio-container {max-width: 100% !important;}
.vq-pills {display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px;}
.vq-pill {padding:5px 12px; border-radius:999px; font-size:0.85em; font-weight:600;
  background:var(--block-background-fill); border:1px solid var(--border-color-primary);}
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
  text-overflow:ellipsis; max-width:280px;}
.vq-sub {font-size:0.8em; color:var(--body-text-color-subdued); white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; max-width:300px; margin-top:2px;}
.vq-ref {font-size:0.76em; color:var(--body-text-color-subdued); font-style:italic;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:300px;
  margin-top:2px; opacity:0.85;}
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


if __name__ == "__main__":
    # Default 0.0.0.0 -> always reachable on the local network (LAN) as well as
    # locally. Set GRADIO_SERVER_NAME=127.0.0.1 to restrict to this PC only.
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    start_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

    # Optional login: set OMNIVOICE_AUTH="user:pass" to require a password.
    _auth_env = os.environ.get("OMNIVOICE_AUTH", "").strip()
    auth = tuple(_auth_env.split(":", 1)) if ":" in _auth_env else None
    if auth:
        print(f"Login required: user '{auth[0]}'")

    app = demo.queue()
    # Auto-fallback: if the port is busy, try the next few ports.
    last_err = None
    for port in range(start_port, start_port + 11):
        try:
            print("=" * 56)
            print(f"  Local:    http://127.0.0.1:{port}")
            if server_name == "0.0.0.0":
                for ip in _lan_ips():
                    print(f"  Network:  http://{ip}:{port}   <- share on LAN")
            else:
                print("  (LAN sharing off — run run_lan.bat for network access)")
            print("=" * 56)
            app.launch(
                server_name=server_name,
                server_port=port,
                share=share,
                auth=auth,
                theme=_THEME,
                css=_CSS,
                allowed_paths=[OUTPUT_DIR],
                inbrowser=os.environ.get("OMNIVOICE_OPEN_BROWSER", "1") == "1",
            )
            last_err = None
            break
        except OSError as e:
            last_err = e
            print(f"Port {port} busy, trying {port + 1} ...")
    if last_err is not None:
        raise last_err
