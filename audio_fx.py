#!/usr/bin/env python3
"""Audio measurement and repair for OmniVoice — numpy/scipy only.

No torch, no model, no GPU: this module can be imported and unit-tested on any
machine (`python -m pytest tests/test_audio_fx.py`), which is how the reference
trimming was developed without CUDA.

Three jobs:

1. **Reference repair** (`smart_trim_reference`) — the fix for the worst bug in
   production. A 60s reference cut at exactly MAX_REF_SEC lands mid-word, and
   the half-spoken word leaks into every single generation ("forcing" x163 in
   one batch). We cut at the last real pause before the limit instead, fade,
   and append a controlled silence so the model can tell the reference ended.

2. **Loudness** (`lufs`, `normalize_loudness`) — ITU-R BS.1770-4 K-weighted
   loudness with a true-peak ceiling, so every clip from every voice comes back
   at the same level instead of tracking whatever the reference happened to be
   (the batch measured -0.2 dB to -12.6 dB across one video).

3. **Output repair** (`remove_tail_after_gap`, `ends_abruptly`) — surgical
   removal of a trailing artefact the verifier has *confirmed* is not in the
   script, and a truncation check that is the right way round (a loud tail
   means the clip was cut off; silence means it finished).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("omnivoice.audio_fx")

try:
    from scipy.signal import lfilter, resample_poly

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

EPS = 1e-12


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------
def to_mono_float(x: np.ndarray) -> np.ndarray:
    """int16/float64/stereo -> mono float32 in [-1, 1]."""
    a = np.asarray(x)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    elif a.dtype == np.int32:
        a = a.astype(np.float32) / 2147483648.0
    else:
        a = a.astype(np.float32, copy=False)
    return np.ascontiguousarray(a.reshape(-1))


def to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)


def db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), EPS))


def frame_db(x: np.ndarray, sr: int, frame_ms: float = 20.0) -> Tuple[np.ndarray, int]:
    """Non-overlapping frame energies in dBFS. Cheap and memory-flat."""
    frame = max(1, int(sr * frame_ms / 1000.0))
    n = (len(x) // frame) * frame
    if n == 0:
        rms = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
        return np.array([db(rms)], dtype=np.float32), frame
    f = x[:n].reshape(-1, frame)
    rms = np.sqrt(np.mean(np.square(f, dtype=np.float64), axis=1))
    return (20.0 * np.log10(np.maximum(rms, EPS))).astype(np.float32), frame


def silence_threshold_db(fdb: np.ndarray, drop: float = 35.0,
                         floor: float = -60.0) -> float:
    """A speech-relative silence threshold, so a quiet clip is not all 'speech'."""
    if fdb.size == 0:
        return floor
    top = float(np.percentile(fdb, 95))
    return max(floor, top - drop)


def speech_rms_db(x: np.ndarray, sr: int) -> float:
    """RMS over speech frames only (silence would drag a whole-clip RMS down)."""
    if len(x) == 0:
        return -120.0
    fdb, frame = frame_db(x, sr)
    thr = silence_threshold_db(fdb)
    mask = fdb > thr
    if not mask.any():
        return float(db(np.sqrt(np.mean(np.square(x)))))
    n = (len(x) // frame) * frame
    f = x[:n].reshape(-1, frame)
    sel = f[mask[: f.shape[0]]]
    return float(db(np.sqrt(np.mean(np.square(sel, dtype=np.float64)))))


def sample_peak_db(x: np.ndarray) -> float:
    return db(float(np.max(np.abs(x))) if len(x) else 0.0)


def true_peak_db(x: np.ndarray, oversample: int = 4) -> float:
    """4x-oversampled peak (BS.1770 true peak). Falls back to sample peak."""
    if len(x) == 0:
        return -120.0
    if not _HAVE_SCIPY:
        return sample_peak_db(x)
    try:
        up = resample_poly(x.astype(np.float64), oversample, 1)
        return db(float(np.max(np.abs(up))))
    except Exception:  # pragma: no cover
        return sample_peak_db(x)


# ---------------------------------------------------------------------------
# ITU-R BS.1770-4 loudness (LUFS)
# ---------------------------------------------------------------------------
def _biquad_high_shelf(sr: int, fc=1681.9744509555319, q=0.7071752369554196,
                       gain_db=3.999843853973347):
    a_ = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / sr
    cw, alpha = math.cos(w0), math.sin(w0) / (2.0 * q)
    sa = 2.0 * math.sqrt(a_) * alpha
    b = np.array([a_ * ((a_ + 1) + (a_ - 1) * cw + sa),
                  -2 * a_ * ((a_ - 1) + (a_ + 1) * cw),
                  a_ * ((a_ + 1) + (a_ - 1) * cw - sa)])
    a = np.array([(a_ + 1) - (a_ - 1) * cw + sa,
                  2 * ((a_ - 1) - (a_ + 1) * cw),
                  (a_ + 1) - (a_ - 1) * cw - sa])
    return b / a[0], a / a[0]


def _biquad_high_pass(sr: int, fc=38.13547087602444, q=0.5003270373238773):
    w0 = 2.0 * math.pi * fc / sr
    cw, alpha = math.cos(w0), math.sin(w0) / (2.0 * q)
    b = np.array([(1 + cw) / 2.0, -(1 + cw), (1 + cw) / 2.0])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


def lufs(x: np.ndarray, sr: int) -> float:
    """Integrated loudness in LUFS (mono). Falls back to RMS-ish without scipy."""
    if len(x) == 0:
        return -120.0
    if not _HAVE_SCIPY:
        # Not a real LUFS number, but on the same scale for speech.
        return speech_rms_db(x, sr) - 0.691
    try:
        b1, a1 = _biquad_high_shelf(sr)
        b2, a2 = _biquad_high_pass(sr)
        y = lfilter(b1, a1, x.astype(np.float64))
        y = lfilter(b2, a2, y)
    except Exception:  # pragma: no cover
        return speech_rms_db(x, sr) - 0.691

    block = int(0.400 * sr)
    step = max(1, int(0.100 * sr))          # 75 % overlap
    if len(y) < block:
        z = float(np.mean(np.square(y)))
        return -0.691 + 10.0 * math.log10(max(z, EPS))

    starts = range(0, len(y) - block + 1, step)
    z = np.array([np.mean(np.square(y[s:s + block])) for s in starts])
    loud = -0.691 + 10.0 * np.log10(np.maximum(z, EPS))

    keep = loud > -70.0                      # absolute gate
    if not keep.any():
        return -120.0
    gamma_r = -0.691 + 10.0 * math.log10(max(float(np.mean(z[keep])), EPS)) - 10.0
    keep = keep & (loud > gamma_r)           # relative gate
    if not keep.any():
        return -120.0
    return float(-0.691 + 10.0 * math.log10(max(float(np.mean(z[keep])), EPS)))


def normalize_loudness(x: np.ndarray, sr: int, target_lufs: float = -20.0,
                       peak_ceiling_db: float = -1.0,
                       max_gain_db: float = 24.0) -> Tuple[np.ndarray, Dict]:
    """Bring a clip to `target_lufs` without letting true peak exceed the ceiling.

    Returns (audio, info). `info["met_target"]` is False when the peak ceiling
    forced a quieter result — worth reporting rather than clipping the audio.
    """
    info: Dict[str, Any] = {"limited_by": None}
    if len(x) == 0:
        return x, {"in_lufs": -120.0, "out_lufs": -120.0, "gain_db": 0.0,
                   "true_peak_db": -120.0, "met_target": False,
                   "limited_by": "silence"}
    measured = lufs(x, sr)
    info["in_lufs"] = round(measured, 2)
    if measured <= -119.0:
        info.update(out_lufs=measured, gain_db=0.0,
                    true_peak_db=true_peak_db(x), met_target=False,
                    limited_by="silence")
        return x, info

    wanted = target_lufs - measured
    gain_db = float(np.clip(wanted, -max_gain_db, max_gain_db))
    if abs(gain_db - wanted) > 0.05:
        # Refusing to add 30 dB to a near-silent recording is deliberate: it
        # would bring the room noise up with it.
        info["limited_by"] = "gain_cap"
    y = x * (10.0 ** (gain_db / 20.0))

    tp = true_peak_db(y)
    if tp > peak_ceiling_db:
        trim = peak_ceiling_db - tp
        y = y * (10.0 ** (trim / 20.0))
        gain_db += trim
        tp = peak_ceiling_db
        info["limited_by"] = "peak_ceiling"
    y = np.clip(y, -1.0, 1.0).astype(np.float32)

    out = lufs(y, sr)
    info.update(out_lufs=round(out, 2), gain_db=round(gain_db, 2),
                true_peak_db=round(tp, 2),
                met_target=bool(abs(out - target_lufs) < 1.5))
    return y, info


# ---------------------------------------------------------------------------
# Silence geometry
# ---------------------------------------------------------------------------
def _silent_runs(fdb: np.ndarray, thr: float) -> List[Tuple[int, int]]:
    """[(first_silent_frame, last_silent_frame_exclusive), ...]"""
    quiet = fdb <= thr
    runs, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(quiet)))
    return runs


def trailing_silence_sec(x: np.ndarray, sr: int) -> float:
    if len(x) == 0:
        return 0.0
    fdb, frame = frame_db(x, sr)
    thr = silence_threshold_db(fdb)
    n = 0
    for v in fdb[::-1]:
        if v <= thr:
            n += 1
        else:
            break
    return n * frame / float(sr)


def strip_trailing_silence(x: np.ndarray, sr: int, keep_ms: float = 0.0,
                           drop_db: float = 35.0) -> np.ndarray:
    if len(x) == 0:
        return x
    fdb, frame = frame_db(x, sr)
    thr = silence_threshold_db(fdb, drop=drop_db)
    last = -1
    for i in range(len(fdb) - 1, -1, -1):
        if fdb[i] > thr:
            last = i
            break
    if last < 0:
        return x
    end = min(len(x), (last + 1) * frame + int(sr * keep_ms / 1000.0))
    return x[:end]


def strip_leading_silence(x: np.ndarray, sr: int, keep_ms: float = 40.0,
                          drop_db: float = 35.0) -> np.ndarray:
    if len(x) == 0:
        return x
    fdb, frame = frame_db(x, sr)
    thr = silence_threshold_db(fdb, drop=drop_db)
    first = -1
    for i, v in enumerate(fdb):
        if v > thr:
            first = i
            break
    if first <= 0:
        return x
    start = max(0, first * frame - int(sr * keep_ms / 1000.0))
    return x[start:]


def fade_out(x: np.ndarray, sr: int, ms: float = 25.0) -> np.ndarray:
    n = min(len(x), int(sr * ms / 1000.0))
    if n <= 1:
        return x
    y = x.copy()
    y[-n:] = y[-n:] * np.linspace(1.0, 0.0, n, dtype=np.float32)
    return y


def fade_in(x: np.ndarray, sr: int, ms: float = 10.0) -> np.ndarray:
    n = min(len(x), int(sr * ms / 1000.0))
    if n <= 1:
        return x
    y = x.copy()
    y[:n] = y[:n] * np.linspace(0.0, 1.0, n, dtype=np.float32)
    return y


# ---------------------------------------------------------------------------
# THE reference fix
# ---------------------------------------------------------------------------
def last_pause_before(x: np.ndarray, sr: int, min_pause_ms: float = 140.0,
                      min_keep_samples: int = 0) -> Optional[int]:
    """Sample index where speech last *stopped*, i.e. a safe place to cut.

    Returns None when every pause would leave less than `min_keep_samples` of
    audio — better a warned mid-word clip than a two-second reference.
    """
    if len(x) == 0:
        return None
    fdb, frame = frame_db(x, sr)
    thr = silence_threshold_db(fdb)
    min_frames = max(1, int(min_pause_ms / 20.0))
    best = None
    for s, e in _silent_runs(fdb, thr):
        if e - s < min_frames or s == 0:
            continue
        cut = s * frame                       # where the speech stopped
        if cut >= min_keep_samples:
            best = cut
    return best


def first_pause_within(x: np.ndarray, sr: int, scan_sec: float = 0.6,
                       min_pause_ms: float = 140.0) -> Optional[int]:
    """Sample index where speech first *resumes*, if that happens early enough
    to be worth skipping a partial opening word."""
    if len(x) == 0:
        return None
    fdb, frame = frame_db(x, sr)
    thr = silence_threshold_db(fdb)
    min_frames = max(1, int(min_pause_ms / 20.0))
    scan_frames = int(scan_sec * sr / frame)
    for s, e in _silent_runs(fdb, thr):
        if e - s < min_frames or s == 0 or e >= len(fdb):
            continue
        if e <= scan_frames:
            return e * frame
    return None


def pause_near_target(x: np.ndarray, sr: int, target_sec: float,
                      hard_max_sec: float, min_pause_ms: float = 140.0,
                      min_keep_sec: float = 3.0,
                      overshoot_penalty: float = 1.25) -> Optional[int]:
    """The pause closest to `target_sec`, searching as far as `hard_max_sec`.

    `target_sec` is a target, not a wall. If a sentence finishes at 11 s there
    is no sense in throwing away two and a half seconds to land under 10 —
    especially when the alternative pause is at 4 s, which is the difference
    between a good reference and a thin one. Overshoot is allowed up to
    `hard_max_sec` and costs slightly more than the same undershoot, because
    the model was trained on 3-10 s references.
    """
    if len(x) == 0:
        return None
    scan = x[:int(max(hard_max_sec, target_sec) * sr)]
    fdb, frame = frame_db(scan, sr)
    thr = silence_threshold_db(fdb)
    min_frames = max(1, int(min_pause_ms / 20.0))
    floor, target = min_keep_sec * sr, target_sec * sr
    best, best_cost = None, None
    for s, e in _silent_runs(fdb, thr):
        if e - s < min_frames or s == 0:
            continue
        cut = s * frame                       # where the speech stopped
        if cut < floor:
            continue
        cost = (target - cut) if cut <= target else (cut - target) * overshoot_penalty
        if best_cost is None or cost < best_cost:
            best, best_cost = cut, cost
    return best


def smart_trim_reference(x: np.ndarray, sr: int, max_sec: float = 10.0,
                         tail_silence_sec: float = 0.30,
                         min_keep_ratio: float = 0.55,
                         min_pause_ms: float = 140.0,
                         min_keep_sec: float = 3.0,
                         hard_max_sec: float = 15.0,
                         fade_ms: float = 25.0,
                         repair_mid_word: bool = True,
                         lead_scan_sec: float = 0.6) -> Tuple[np.ndarray, Dict]:
    """Make sure a reference clip ends on a finished word, whatever arrives.

    Two different ways a reference ends mid-word, and both are handled here:

    1. **We cut it there.** A 60s upload chopped at exactly 10.000s landed
       inside the word "forcing" — and the model then said "forcing" at the end
       of 86-90% of that voice's clips, 163 times in one batch. Cut at the last
       real pause instead.
    2. **It arrived that way.** A customer records on a phone and stops the
       recording while still speaking. Nothing trims it, because it is already
       under the limit — so the half word sits at the end of the reference and
       does exactly the same damage. Back it up to the last pause too.

    Cutting back costs a little duration, and that is the right trade: the
    transcript is re-derived from the trimmed audio afterwards, so losing a
    whole final word costs nothing, while keeping half of one is the single
    most expensive defect this product has had. It only refuses when the clip
    would drop below `min_keep_sec`, and then it says so.

    `max_sec` is a **target**, not a wall: the cut lands on the pause nearest
    to it, overshooting as far as `hard_max_sec` when that is what it takes to
    let a sentence finish. Backing off from 11 s to 4 s to respect a round
    number would cost far more quality than the extra second ever could.
    """
    info: Dict = {
        "orig_sec": round(len(x) / float(sr), 2) if sr else 0.0,
        "trimmed": False, "cut_at_pause": False, "cut_sec": None,
        "trailing_silence_sec": 0.0, "ends_mid_word": False,
        "repaired_mid_word": False, "trimmed_lead_sec": 0.0, "warnings": [],
    }
    if len(x) == 0:
        info["warnings"].append("reference audio is empty")
        return x, info

    limit = int(max_sec * sr)
    floor = int(min_keep_sec * sr)
    y = x

    if len(x) > limit:
        # --- case 1: too long, we choose where to cut ---------------------
        # The limit is a target. We look for the pause nearest to it, in either
        # direction, and only fall back to a hard cut when the clip has no
        # pause anywhere in the search window.
        info["trimmed"] = True
        cut = pause_near_target(x, sr, max_sec, hard_max_sec, min_pause_ms,
                                min_keep_sec)
        if cut:
            y = x[:cut]
            info["cut_at_pause"] = True
            if cut > limit:
                info["overshoot_sec"] = round((cut - limit) / float(sr), 2)
        else:
            y = x[:limit]
            info["ends_mid_word"] = True
            info["warnings"].append(
                f"no pause found anywhere in the first {hard_max_sec:.0f}s — the "
                f"reference had to be cut mid-phrase, which can leak words into "
                f"every clip; upload a 6-10s clip that ends on a finished sentence")
    else:
        # --- case 2: short enough, but does it end mid-word? --------------
        tail = trailing_silence_sec(x, sr)
        info["trailing_silence_sec"] = round(tail, 3)
        if tail < 0.05:
            info["ends_mid_word"] = True
            cut = (last_pause_before(x, sr, min_pause_ms,
                                     max(floor, int(len(x) * min_keep_ratio)))
                   if repair_mid_word else None)
            if cut:
                y = x[:cut]
                info["trimmed"] = True
                info["cut_at_pause"] = True
                info["repaired_mid_word"] = True
                info["warnings"].append(
                    f"your clip stopped while you were still speaking, so it was "
                    f"cut back to the last finished phrase "
                    f"({cut / float(sr):.1f}s of {len(x) / float(sr):.1f}s kept) "
                    f"— an unfinished word at the end gets repeated in every clip")
            else:
                info["warnings"].append(
                    f"reference ends without a pause and is too short to cut back "
                    f"(under {min_keep_sec:.0f}s would be left) — please re-record "
                    f"6-10s ending on a finished sentence, then a beat of silence")

    # A clip that *begins* at full volume probably began mid-word too. That is
    # much less harmful than a partial ending — the model continues after the
    # reference, not before it — but it does make the transcript describe a
    # word the audio only half contains. Only ever gives up a fraction of a
    # second, and only if a pause turns up that early.
    if len(y):
        opening = y[:max(1, int(0.02 * sr))]
        if float(np.max(np.abs(opening))) > 0.02:
            lead = first_pause_within(y, sr, lead_scan_sec, min_pause_ms)
            if lead:
                info["trimmed_lead_sec"] = round(lead / float(sr), 2)
                y = y[lead:]

    y = strip_trailing_silence(y, sr)
    y = strip_leading_silence(y, sr)
    y = fade_in(y, sr, 8.0)
    # When the half word had to stay, let it decay instead of stopping dead —
    # it is a weaker "keep going" cue to the model than a hard edge.
    y = fade_out(y, sr, fade_ms * 3 if (info["ends_mid_word"]
                                        and not info["repaired_mid_word"])
                 else fade_ms)
    if tail_silence_sec > 0:
        y = np.concatenate(
            [y, np.zeros(int(tail_silence_sec * sr), dtype=np.float32)])
    info["cut_sec"] = round(len(y) / float(sr), 2)
    return y.astype(np.float32), info


# ---------------------------------------------------------------------------
# Reference quality report (E1)
# ---------------------------------------------------------------------------
def estimate_snr_db(x: np.ndarray, sr: int) -> float:
    """Rough speech-vs-noise-floor estimate. Good enough to warn a customer."""
    if len(x) < sr // 4:
        return 99.0
    fdb, _ = frame_db(x, sr)
    if fdb.size < 5:
        return 99.0
    noise = float(np.percentile(fdb, 10))
    speech = float(np.percentile(fdb, 90))
    return round(speech - noise, 1)


def clipping_ratio(x: np.ndarray, thresh: float = 0.99) -> float:
    return float(np.mean(np.abs(x) > thresh)) if len(x) else 0.0


def analyze_reference(x: np.ndarray, sr: int,
                      min_sec: float = 3.0, max_sec: float = 30.0) -> Dict:
    """Everything a customer should be told about their upload, before they
    generate ten hours of audio with it."""
    dur = len(x) / float(sr) if sr else 0.0
    rep: Dict = {
        "duration_sec": round(dur, 2),
        "lufs": round(lufs(x, sr), 2),
        "true_peak_db": round(true_peak_db(x), 2),
        "snr_db": estimate_snr_db(x, sr),
        "clipping_ratio": round(clipping_ratio(x), 5),
        "trailing_silence_sec": round(trailing_silence_sec(x, sr), 3),
        "warnings": [],
    }
    w = rep["warnings"]
    if dur < min_sec:
        w.append(f"reference is {dur:.1f}s — under {min_sec:.0f}s clones poorly; "
                 f"6-10s is the sweet spot")
    elif dur > max_sec:
        w.append(f"reference is {dur:.0f}s — only the first {max_sec:.0f}s "
                 f"matter, and a long clip raises the risk of leftover words "
                 f"leaking into your clips")
    if rep["snr_db"] < 15:
        w.append(f"reference is noisy (SNR {rep['snr_db']:.0f} dB) — the cloned "
                 f"voice will carry that noise")
    if rep["clipping_ratio"] > 0.001:
        w.append("reference is clipping — re-record a little further from the mic")
    if rep["trailing_silence_sec"] < 0.05:
        w.append("reference ends without a pause — it may be cut mid-word")
    # 0..1, purely advisory
    score = 1.0
    score -= 0.25 if dur < min_sec else 0.0
    score -= 0.10 if dur > max_sec else 0.0
    score -= 0.25 if rep["snr_db"] < 15 else 0.0
    score -= 0.15 if rep["clipping_ratio"] > 0.001 else 0.0
    score -= 0.15 if rep["trailing_silence_sec"] < 0.05 else 0.0
    rep["quality_score"] = round(max(0.0, min(1.0, score)), 2)
    return rep


# ---------------------------------------------------------------------------
# Output repair
# ---------------------------------------------------------------------------
def ends_abruptly(x: np.ndarray, sr: int, tail_sec: float = 0.25,
                  thresh_db: float = -35.0) -> bool:
    """True when the clip stops at full volume — i.e. it was cut off.

    Deliberately the opposite of the naive check: a clip that ends in silence
    finished properly; a clip that ends loud is the truncated one.
    """
    if len(x) < int(tail_sec * sr):
        return False
    tail = x[-int(tail_sec * sr):]
    return db(float(np.sqrt(np.mean(np.square(tail, dtype=np.float64))))) > thresh_db


def find_tail_segment(x: np.ndarray, sr: int, min_gap_ms: float = 110.0,
                      max_tail_sec: float = 2.5,
                      min_keep_sec: float = 0.30) -> Optional[Tuple[int, float]]:
    """Locate a final speech burst separated from the rest by a pause.

    Returns (cut_sample, tail_seconds) or None. Only ever used when the
    verifier has already proven the tail is not in the script — a heuristic
    like this must never be allowed to delete a real word on its own.
    """
    if len(x) == 0:
        return None
    core = strip_trailing_silence(x, sr)
    if len(core) == 0:
        return None
    fdb, frame = frame_db(core, sr)
    thr = silence_threshold_db(fdb)
    min_frames = max(1, int(min_gap_ms / 20.0))
    runs = [r for r in _silent_runs(fdb, thr) if r[1] - r[0] >= min_frames
            and r[1] < len(fdb)]
    if not runs:
        return None
    s, e = runs[-1]
    cut = s * frame
    tail_sec = (len(core) - e * frame) / float(sr)
    if tail_sec <= 0 or tail_sec > max_tail_sec:
        return None
    if cut < min_keep_sec * sr:
        return None
    return cut, round(tail_sec, 3)


def remove_tail_after_gap(x: np.ndarray, sr: int, pad_sec: float = 0.28,
                          **kw) -> Tuple[np.ndarray, float]:
    """Drop a confirmed hallucinated tail; returns (audio, removed_seconds).

    The pad is not cosmetic: it has to be at least as long as the window
    `ends_abruptly` inspects, or repairing a clip would make it look truncated.
    """
    found = find_tail_segment(x, sr, **kw)
    if not found:
        return x, 0.0
    cut, tail_sec = found
    y = fade_out(x[:cut].copy(), sr, 20.0)
    pad = np.zeros(int(max(0.0, pad_sec) * sr), dtype=np.float32)
    return np.concatenate([y, pad]).astype(np.float32), tail_sec


# Chunks are generated independently, so their levels drift apart. Pulling
# each one to a fixed target — what XTTS does — is what makes long-form output
# jump in volume between sentences. Pulling outliers toward the clip's OWN
# median removes the jump and leaves natural dynamics intact.
def match_levels(parts: List[np.ndarray], sr: int,
                 max_correction_db: float = 6.0,
                 deadband_db: float = 1.5) -> Tuple[List[np.ndarray], List[float]]:
    if len(parts) < 2:
        return list(parts), [0.0] * len(parts)
    levels = [speech_rms_db(p, sr) for p in parts]
    usable = [lv for lv in levels if lv > -110.0]
    if len(usable) < 2:
        return list(parts), [0.0] * len(parts)
    median = float(np.median(usable))
    out, applied = [], []
    for p, lv in zip(parts, levels):
        if lv <= -110.0:
            out.append(p)
            applied.append(0.0)
            continue
        delta = median - lv
        if abs(delta) <= deadband_db:
            out.append(p)
            applied.append(0.0)
            continue
        # Correct only the excess beyond the deadband, and cap it.
        corr = math.copysign(
            min(abs(delta) - deadband_db, max_correction_db), delta)
        out.append(np.clip(p * (10.0 ** (corr / 20.0)), -1.0, 1.0).astype(np.float32))
        applied.append(round(corr, 2))
    return out, applied


def join_chunks(parts: List[np.ndarray], sr: int, gap_sec: float = 0.15,
                fade_ms: float = 15.0, edge_keep_ms: float = 60.0,
                level_match: bool = True, tail_pad_sec: float = 0.30,
                edge_drop_db: float = 45.0) -> Tuple[np.ndarray, Dict]:
    """Join generated chunks without an audible seam.

    Four things, each fixing a documented long-form failure:

    * **Even edges.** Each chunk is trimmed to a consistent 60 ms of silence and
      a fixed gap is inserted, so inter-sentence pauses are the same length
      instead of whatever the model happened to leave. The threshold here is
      deliberately 10 dB stricter than elsewhere: a soft final consonant is
      quiet, and trimming it is a documented way to lose the last word.
    * **Level matching** to the clip's own median (see `match_levels`).
    * **Short fades** at every edge — the boundary click is the usual reason a
      stitched clip sounds stitched.
    * **Tail padding** on the finished clip, so whatever the customer runs it
      through next cannot clip the final consonant.
    """
    info: Dict = {"chunks": len(parts), "level_corrections": [],
                  "pad_sec": 0.0, "speech_sec": 0.0, "ends_abruptly": False}
    if not parts:
        return np.zeros(0, dtype=np.float32), info

    # Truncation has to be judged on the LAST CHUNK AS THE MODEL MADE IT.
    # Once we have stripped its trailing silence, every clip ends in speech and
    # the check would fire on all of them — a detector that flags every good
    # clip is worse than no detector.
    info["ends_abruptly"] = ends_abruptly(
        np.asarray(parts[-1], dtype=np.float32).reshape(-1), sr)

    cleaned = []
    for p in parts:
        q = np.asarray(p, dtype=np.float32).reshape(-1)
        q = strip_leading_silence(q, sr, keep_ms=edge_keep_ms, drop_db=edge_drop_db)
        q = strip_trailing_silence(q, sr, keep_ms=edge_keep_ms, drop_db=edge_drop_db)
        cleaned.append(q if len(q) else np.asarray(p, dtype=np.float32).reshape(-1))

    if level_match:
        cleaned, info["level_corrections"] = match_levels(cleaned, sr)

    gap = np.zeros(int(max(0.0, gap_sec) * sr), dtype=np.float32)
    pieces: List[np.ndarray] = []
    for i, p in enumerate(cleaned):
        p = fade_in(p, sr, fade_ms)
        p = fade_out(p, sr, fade_ms)
        if i:
            pieces.append(gap)
        pieces.append(p)
    y = np.concatenate(pieces).astype(np.float32) if pieces else np.zeros(0, np.float32)

    info["speech_sec"] = round(len(y) / float(sr), 3)
    if tail_pad_sec > 0:
        y = np.concatenate([y, np.zeros(int(tail_pad_sec * sr), dtype=np.float32)])
        info["pad_sec"] = tail_pad_sec
    return y, info


def concat_audio(parts: List[np.ndarray], sr: int, gap_sec: float = 0.15) -> np.ndarray:
    if not parts:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(gap_sec * sr), dtype=np.float32)
    out: List[np.ndarray] = []
    for i, p in enumerate(parts):
        if i:
            out.append(gap)
        out.append(np.asarray(p, dtype=np.float32).reshape(-1))
    return np.concatenate(out)


def wpm(word_count: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return round(word_count / (seconds / 60.0), 1)
