#!/usr/bin/env python3
"""Optional inaudible watermarking of generated speech (Meta's AudioSeal).

WHY THIS IS HERE
----------------
EU AI Act Article 50 became enforceable on **2 August 2026**. It obliges the
provider of a synthetic-audio system to mark its output in a **machine-readable**
way, and the deployer to disclose that the audio is AI-generated. Selling a
voice-cloning tool to the public without any marking is now a compliance gap,
not just a nice-to-have — and if a customer ever misuses a clone, a watermark is
the only thing that lets you prove which clips came from your server.

The accepted practice is narrow and worth following exactly:

    consent at enrolment · watermark at generation · detect on complaint

**Watermark only model output.** Never stamp a customer's own recording — the
reference clip they uploaded is their real voice, not synthetic, and marking it
would be both wrong and misleading.

STATUS
------
This module is written but has **never been executed** — `audioseal` is not
installed here and it needs torch. It is off by default (`OMNIVOICE_WATERMARK=0`)
and every failure path returns the audio untouched with a flag rather than
raising, so enabling it can never cost you a clip. Verify it on the GPU box:

    venv\\Scripts\\python -m pip install audioseal
    set OMNIVOICE_WATERMARK=1
    venv\\Scripts\\python -c "import watermark; print(watermark.self_check())"

AudioSeal is MIT-licensed and explicitly permits commercial use.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger("omnivoice.watermark")

# AudioSeal's published models are trained at 16 kHz. It accepts a sample_rate
# argument and resamples internally; we pass the real rate rather than
# resampling the deliverable ourselves.
NATIVE_SR = 16000

_LOCK = threading.Lock()
_GEN = None
_DET = None
_STATE = {"loaded": False, "error": None, "model": None}


def available() -> bool:
    import importlib.util
    return importlib.util.find_spec("audioseal") is not None


def _load(model_name: str = "audioseal_wm_16bits",
          detector_name: str = "audioseal_detector_16bits"):
    """Load once, lazily. Returns (generator, detector) or (None, None)."""
    global _GEN, _DET
    if _STATE["loaded"] or _STATE["error"]:
        return _GEN, _DET
    with _LOCK:
        if _STATE["loaded"] or _STATE["error"]:
            return _GEN, _DET
        try:
            import torch
            from audioseal import AudioSeal

            _GEN = AudioSeal.load_generator(model_name)
            _DET = AudioSeal.load_detector(detector_name)
            if torch.cuda.is_available():
                # Tiny next to the TTS model, and it keeps the tensors on one
                # device. Falls back silently if the move is refused.
                try:
                    _GEN = _GEN.cuda()
                    _DET = _DET.cuda()
                except Exception:  # pragma: no cover
                    pass
            _STATE.update(loaded=True, model=model_name)
            log.info("audio watermarking enabled (%s)", model_name)
        except Exception as e:  # noqa: BLE001
            _STATE["error"] = f"{type(e).__name__}: {e}"
            log.warning("watermarking unavailable (%s) — clips will be "
                        "unmarked", _STATE["error"])
    return _GEN, _DET


def _to_tensor(x: np.ndarray):
    import torch
    a = np.asarray(x, dtype=np.float32).reshape(1, 1, -1)
    return torch.from_numpy(a)


def embed(x: np.ndarray, sr: int, alpha: float = 1.0,
          message: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    """Return (audio, info). On any failure the audio comes back unchanged.

    A watermark is worth nothing if it can fail a customer's render, so every
    error here is a flag, never an exception.
    """
    info: Dict = {"watermarked": False, "model": None, "error": None}
    if len(x) == 0:
        return x, info
    gen, _det = _load()
    if gen is None:
        info["error"] = _STATE["error"] or "audioseal not installed"
        return x, info
    try:
        import torch

        wav = _to_tensor(x)
        if next(gen.parameters()).is_cuda:
            wav = wav.cuda()
        kw = {"sample_rate": int(sr), "alpha": float(alpha)}
        if message is not None:
            kw["message"] = torch.randint(0, 2, (1, 16)) * 0 + int(message)
        with torch.no_grad():
            marked = gen(wav, **kw)
        out = marked.detach().to("cpu").numpy().reshape(-1).astype(np.float32)
        if out.shape[0] != x.shape[0]:
            # Never hand back audio of a different length than was rendered.
            info["error"] = (f"length changed {x.shape[0]} -> {out.shape[0]}; "
                             f"watermark discarded")
            return x, info
        info.update(watermarked=True, model=_STATE.get("model"))
        return np.clip(out, -1.0, 1.0), info
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
        log.warning("watermark embed failed (%s) — clip returned unmarked", e)
        return x, info


def detect(x: np.ndarray, sr: int) -> Dict:
    """"Detect on complaint": did this clip come out of our server?"""
    out: Dict = {"detected": False, "probability": None, "error": None}
    if len(x) == 0:
        out["error"] = "empty audio"
        return out
    _gen, det = _load()
    if det is None:
        out["error"] = _STATE["error"] or "audioseal not installed"
        return out
    try:
        import torch

        wav = _to_tensor(x)
        if next(det.parameters()).is_cuda:
            wav = wav.cuda()
        with torch.no_grad():
            found = det.detect_watermark(wav, int(sr))
        # Older builds return just the probability; newer ones (prob, message).
        prob_raw = found[0] if isinstance(found, (tuple, list)) else found
        if isinstance(prob_raw, torch.Tensor):
            prob_raw = prob_raw.detach().to("cpu").float().reshape(-1)[0].item()
        prob = float(prob_raw)
        out.update(detected=prob > 0.5, probability=round(prob, 4))
        return out
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        return out


def self_check(sr: int = 24000, seconds: float = 2.0) -> Dict:
    """Watermark a tone and read it back. Run this once on the GPU box."""
    t = np.arange(int(seconds * sr)) / float(sr)
    tone = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    marked, info = embed(tone, sr)
    if not info["watermarked"]:
        return {"ok": False, "stage": "embed", **info}
    found = detect(marked, sr)
    clean = detect(tone, sr)
    return {
        "ok": bool(found.get("detected") and not clean.get("detected")),
        "marked_detected": found.get("detected"),
        "marked_probability": found.get("probability"),
        "clean_detected": clean.get("detected"),
        "model": info.get("model"),
        "error": found.get("error") or clean.get("error"),
    }


def status() -> Dict:
    return {"installed": available(), "loaded": _STATE["loaded"],
            "model": _STATE.get("model"), "error": _STATE["error"]}
