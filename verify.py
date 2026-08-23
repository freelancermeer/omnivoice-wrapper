#!/usr/bin/env python3
"""Did the model actually say what it was sent?

The single most valuable check in the whole product: transcribe the generated
audio and align it against the script. It replaces a pile of weaker proxies —
a WPM gate, a duration ratio, a truncation detector — because it produces
*evidence* ("these three words were never spoken") instead of a guess.

Pure Python: `python -m pytest tests/test_verify.py` runs anywhere.

IMPORTANT — this uses sequence alignment, not set membership. The obvious
implementation:

    dropped = [w for w in sent if w not in spoken]

passes the exact bug it exists to catch:

    sent  : "He called it perjury. He called it fraud. He called it contempt."
    spoken: "He called it perjury, fraud, contempt."          <- 6 words gone
    dropped == []            <- because every word still appears *somewhere*

difflib aligns, so a word sent three times and spoken once counts as two drops.
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional

# Filler differences that are transcription artefacts rather than model errors.
# Kept deliberately tiny — every entry here is a bug we agree not to see.
SOFT_TOKENS = {"and", "a", "ah", "uh", "um"}

_CLEAN_RE = re.compile(r"[^a-z0-9' ]")


def words(text: str) -> List[str]:
    """Put the script and the ASR output into one shape."""
    t = (text or "").lower().replace("’", "'").replace("-", " ")
    t = _CLEAN_RE.sub(" ", t)
    return [w for w in (tok.strip("'") for tok in t.split()) if w]


def word_diff(sent_text: str, spoken_text: str) -> Dict:
    """What was sent vs what was said — order and count both respected.

    Returns:
        dropped        words in the script that were not spoken
        inserted       words spoken that were not in the script
        tail_inserted  inserted words that come *after* the whole script
                       (the reference-bleed signature, and the only kind we
                       can repair by trimming audio)
        word_accuracy  matched / len(sent)
        ok             nothing dropped, nothing inserted
    """
    sent, spoken = words(sent_text), words(spoken_text)
    m = difflib.SequenceMatcher(None, sent, spoken, autojunk=False)

    dropped: List[str] = []
    inserted: List[str] = []
    tail_inserted: List[str] = []
    for op, i1, i2, j1, j2 in m.get_opcodes():
        if op in ("delete", "replace"):
            dropped += sent[i1:i2]
        if op in ("insert", "replace"):
            inserted += spoken[j1:j2]
        if op == "insert" and i1 >= len(sent):
            tail_inserted += spoken[j1:j2]

    matched = sum(b.size for b in m.get_matching_blocks())
    hard_dropped = [w for w in dropped if w not in SOFT_TOKENS]
    hard_inserted = [w for w in inserted if w not in SOFT_TOKENS]
    return {
        "sent_words": len(sent),
        "spoken_words": len(spoken),
        "dropped": dropped,
        "inserted": inserted,
        "tail_inserted": tail_inserted,
        "hard_dropped": hard_dropped,
        "hard_inserted": hard_inserted,
        "word_accuracy": round(matched / max(len(sent), 1), 4),
        "ok": not hard_dropped and not hard_inserted,
    }


def passed(diff: Dict, min_accuracy: float = 0.995) -> bool:
    """All three conditions, because accuracy alone cannot see an insertion:
    a clip that says everything it was sent *plus* four extra words scores
    1.000 and is still broken."""
    return (not diff["hard_dropped"]
            and not diff["hard_inserted"]
            and diff["word_accuracy"] >= min_accuracy)


def describe(diff: Dict, limit: int = 6) -> str:
    """One human-readable line for a log, a job card or a response header."""
    bits = []
    if diff["hard_dropped"]:
        d = diff["hard_dropped"]
        bits.append("not spoken: " + " ".join(d[:limit])
                    + ("…" if len(d) > limit else ""))
    if diff["tail_inserted"]:
        t = diff["tail_inserted"]
        bits.append("extra at end: " + " ".join(t[:limit])
                    + ("…" if len(t) > limit else ""))
    extra = [w for w in diff["hard_inserted"] if w not in diff["tail_inserted"]]
    if extra:
        bits.append("extra: " + " ".join(extra[:limit])
                    + ("…" if len(extra) > limit else ""))
    if not bits:
        return f"verified {diff['word_accuracy']:.1%}"
    return "; ".join(bits)


def only_tail_is_wrong(diff: Dict) -> bool:
    """True when the only problem is extra words after the end of the script.

    This is the repairable case: the audio for the script itself is fine and
    the artefact can be cut off, instead of paying for a whole re-generation.
    """
    if diff["hard_dropped"]:
        return False
    if not diff["tail_inserted"]:
        return False
    non_tail = [w for w in diff["hard_inserted"] if w not in diff["tail_inserted"]]
    return not non_tail


def reference_matches_audio(ref_text: str, heard_text: str,
                            min_accuracy: float = 0.90) -> Dict:
    """Does the supplied reference transcript match the reference recording?

    A mismatch here is the root cause of reference bleed: the model is told one
    thing and hears another, and it treats the difference as something it is
    supposed to say. Checking it at registration turns a bug that surfaces
    three hours into a batch into a message at upload time.
    """
    d = word_diff(ref_text, heard_text)
    d["matches"] = d["word_accuracy"] >= min_accuracy
    return d


def baseline_wpm(ref_text: str, duration_sec: float) -> Optional[float]:
    """A voice's own natural speaking rate, measured once at registration.

    A fixed 140-180 WPM gate is wrong in both directions: a documentary
    narrator runs ~100 and would fail every clip, while a 210 WPM voice that
    has drifted down to 175 is genuinely broken and would pass. Comparing a
    voice against itself is the only rate check that means anything.
    """
    n = len(words(ref_text))
    if not n or duration_sec <= 0:
        return None
    return round(n / (duration_sec / 60.0), 1)


# Only used when a voice has no baseline (older voices, no ref_text). Wide
# enough that it can only ever catch a catastrophe, and it warns, never fails.
CATASTROPHIC_LOW, CATASTROPHIC_HIGH = 50.0, 280.0

# Band around a voice's own baseline, as fractions. Overridable from the
# environment (OMNIVOICE_RATE_LOW / OMNIVOICE_RATE_HIGH) because the right
# width depends on how varied the customer's scripts are.
#
# NOTE on the numbers: the production note argues for 0.75-1.30 and then offers
# "a 210 wpm voice that drifted to 175" as the case it catches - but 175/210 is
# 0.83, comfortably inside 0.75. Tightening to 0.85 to catch that example would
# fire constantly: in the measured batch a single voice ranged 160-203 wpm
# (about +/-12%) on ordinary clips. So 0.75-1.30 stays the default, and a 17%
# drift is treated as within a voice's normal range rather than as a warning
# nobody would trust after the first fifty false alarms.
RATE_LOW, RATE_HIGH = 0.75, 1.30


def rate_warning(measured: float, baseline: Optional[float],
                 low: Optional[float] = None,
                 high: Optional[float] = None) -> Optional[str]:
    if not measured:
        return None
    low = RATE_LOW if low is None else low
    high = RATE_HIGH if high is None else high
    if baseline:
        lo, hi = low * baseline, high * baseline
        if not lo <= measured <= hi:
            return (f"speaking rate {measured:.0f} wpm is outside this voice's "
                    f"usual {lo:.0f}-{hi:.0f} wpm")
        return None
    if measured < CATASTROPHIC_LOW:
        return f"speaking rate {measured:.0f} wpm is implausibly slow"
    if measured > CATASTROPHIC_HIGH:
        return f"speaking rate {measured:.0f} wpm is implausibly fast — the clip may be cut short"
    return None
