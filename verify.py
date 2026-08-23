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
from typing import Dict, List, Optional, Tuple

# Filler differences that are transcription artefacts rather than model errors.
# Kept deliberately tiny — every entry here is a bug we agree not to see.
SOFT_TOKENS = {"and", "a", "ah", "uh", "um"}

# Words where "close enough" is not close enough. Confusing "million" with
# "billion", or "not" with "now", is not a transcription quirk — it changes what
# the clip says, and a paid voiceover cannot ship it.
CRITICAL_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "trillion",
    "point", "percent", "dollar", "dollars", "pound", "pounds", "euro", "euros",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth",
    "not", "no", "never", "none", "nothing", "cannot", "without", "nor",
}

# Below this, two words are different words rather than one word heard twice.
ASR_SIMILARITY = 0.72


def likely_asr_artifact(sent_word: str, heard_word: str,
                        threshold: float = None) -> bool:
    """Is this substitution the transcriber, or the model?

    The production batch saw "Bessent" transcribed as "bessant" five times and
    "Hegseth" as "hexeth" twice — proper nouns the transcriber misheard. Every
    one of those would otherwise trigger a full re-generation that fixes
    nothing, which is real GPU time spent on a non-problem.
    """
    if sent_word == heard_word:
        return True
    if sent_word in CRITICAL_WORDS or heard_word in CRITICAL_WORDS:
        return False
    thr = ASR_SIMILARITY if threshold is None else threshold
    return difflib.SequenceMatcher(None, sent_word, heard_word).ratio() >= thr

_CLEAN_RE = re.compile(r"[^a-z0-9' ]")

try:  # the same converter the script side already went through
    import textnorm as _textnorm
except Exception:  # pragma: no cover - verify.py still works without it
    _textnorm = None


def _as_spoken(text: str) -> str:
    """Numbers, money, percentages and ordinals in the shape they were said.

    The script that reaches the model has already been through textnorm, so it
    reads "ninety pages" and "one hundred fifty million dollars". Whisper
    writes the same speech back in its own notation — "90 pages", "$150
    million", "11,000", "4.7", "6th". Those are the transcriber's spelling
    choices, not words the model dropped.

    Left alone they are scored as drops *and* insertions, because the number
    words sit in CRITICAL_WORDS — rightly, since "million" heard as "billion"
    must never be waved through. Measured on the GPU box, that cost a warning
    and a wasted regeneration on nearly every clip containing a figure.

    So both sides go through the one normalizer the script already used, which
    keeps a single set of rules for both — years included: 2026 reads "twenty
    twenty-six", not "two thousand twenty-six". It is a no-op on text that has
    already been normalized.
    """
    if _textnorm is None or not text:
        return text
    try:
        out, _notes = _textnorm.normalize_text(text, None, level="full")
        return out or text
    except Exception:  # noqa: BLE001 - a checker must never break a render
        return text


def words(text: str) -> List[str]:
    """Put the script and the ASR output into one shape."""
    t = _as_spoken(text or "")
    t = t.lower().replace("’", "'").replace("-", " ")
    t = _CLEAN_RE.sub(" ", t)
    return [w for w in (tok.strip("'") for tok in t.split()) if w]


def _word_diff_once(sent_text: str, spoken_text: str) -> Dict:
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
    # Finer grained than dropped/inserted: what the model really failed to say,
    # separated from what the transcriber probably misheard.
    missing: List[str] = []
    extra: List[str] = []
    misheard: List[Tuple[str, str]] = []

    for op, i1, i2, j1, j2 in m.get_opcodes():
        if op in ("delete", "replace"):
            dropped += sent[i1:i2]
        if op in ("insert", "replace"):
            inserted += spoken[j1:j2]
        if op == "insert" and i1 >= len(sent):
            tail_inserted += spoken[j1:j2]

        if op == "delete":
            missing += sent[i1:i2]
        elif op == "insert":
            extra += spoken[j1:j2]
        elif op == "replace":
            a, b = sent[i1:i2], spoken[j1:j2]
            if len(a) == len(b) and all(likely_asr_artifact(x, y)
                                        for x, y in zip(a, b)):
                misheard += list(zip(a, b))
            else:
                missing += a
                extra += b

    matched = sum(b.size for b in m.get_matching_blocks())
    hard_dropped = [w for w in dropped if w not in SOFT_TOKENS]
    hard_inserted = [w for w in inserted if w not in SOFT_TOKENS]
    real_missing = [w for w in missing if w not in SOFT_TOKENS]
    real_extra = [w for w in extra if w not in SOFT_TOKENS]
    return {
        "sent_words": len(sent),
        "spoken_words": len(spoken),
        "dropped": dropped,
        "inserted": inserted,
        "tail_inserted": tail_inserted,
        "hard_dropped": hard_dropped,
        "hard_inserted": hard_inserted,
        "missing": real_missing,
        "extra": real_extra,
        "misheard": misheard,
        "word_accuracy": round(matched / max(len(sent), 1), 4),
        "ok": not hard_dropped and not hard_inserted,
    }


# Whisper writes clock times with a full stop: "7:45" comes back as "7.45",
# which reads as a decimal and expands to "seven point four five" against the
# script's "seven forty-five". Measured on the GPU box: three dropped words and
# seven inserted ones on a clip where the model said exactly the right thing.
_CLOCK_DOT_RE = re.compile(r"\b([01]?\d|2[0-3])\.([0-5]\d)\b")


# A transcriber that has just written "$4.2 million" will write the next bare
# "3 million" as "$3 million" too, because the context is money — and then the
# expansion adds a "dollars" the script never contained. Measured on the GPU
# box, and both Whisper and AssemblyAI made exactly the same inference on the
# same clip, which is what says it is notation rather than the model speaking.
_CURRENCY_MARK_RE = re.compile(r"[$£€¥]")


def _errors(diff: Dict) -> int:
    """How wrong a reading is, counting both directions.

    `word_accuracy` is matched-over-sent, so it cannot see an insertion at all
    — a clip that says every word it was sent plus a spurious "dollars" still
    scores 1.000. Choosing between two readings therefore has to count what was
    added as well as what went missing.
    """
    return len(diff["hard_dropped"]) + len(diff["hard_inserted"])


def _without_nth(text: str, regex, n: int) -> str:
    """The same text with only the nth match of `regex` removed."""
    out, last = [], 0
    for k, m in enumerate(regex.finditer(text)):
        if k == n:
            out.append(text[last:m.start()])
            last = m.end()
            break
    out.append(text[last:])
    return "".join(out)


def _best_currency_reading(sent_text: str, spoken: str, diff: Dict) -> Dict:
    """Drop the currency marks the transcriber inferred, one at a time.

    Removing all of them is no better than keeping all of them when the script
    genuinely contains one: the error simply changes from an inserted
    "dollars" to a dropped one. Only the *spurious* mark should go, so try each
    one and keep whichever removal actually reduces the error count, then look
    again. Greedy, bounded, and it can only ever improve the match.
    """
    for _ in range(_MAX_CURRENCY_TRIES):
        n = len(_CURRENCY_MARK_RE.findall(spoken))
        if not n:
            break
        best, best_text = diff, None
        for i in range(min(n, _MAX_CURRENCY_TRIES)):
            cand = _without_nth(spoken, _CURRENCY_MARK_RE, i)
            alt = _word_diff_once(sent_text, cand)
            if _errors(alt) < _errors(best):
                best, best_text = alt, cand
        if best_text is None:
            break
        diff, spoken = best, best_text
        if diff["ok"]:
            break
    return diff


_MAX_CURRENCY_TRIES = 4


def _readings(spoken_text: str):
    """Other defensible ways to read the same transcript."""
    if _CLOCK_DOT_RE.search(spoken_text):
        yield _CLOCK_DOT_RE.sub(r"\1:\2", spoken_text)


def word_diff(sent_text: str, spoken_text: str) -> Dict:
    """What was sent vs what was said, reading the transcriber charitably.

    Where the transcriber's notation is ambiguous, every defensible reading is
    scored and the best one wins. Only a reading that *improves* the match is
    accepted, so this can remove a false alarm but can never invent a clean
    result: a genuine decimal stays a decimal, because reading it as a clock
    time would not match the script either, and a "dollars" the model really
    failed to say cannot be recovered by deleting a currency symbol that is
    not in the transcript.
    """
    diff = _word_diff_once(sent_text, spoken_text)
    if diff["ok"] or not spoken_text:
        return diff
    for candidate in _readings(spoken_text):
        alt = _word_diff_once(sent_text, candidate)
        if _errors(alt) < _errors(diff):
            diff = alt
            if diff["ok"]:
                return diff
    if not diff["ok"] and _CURRENCY_MARK_RE.search(spoken_text):
        diff = _best_currency_reading(sent_text, spoken_text, diff)
    return diff


def passed(diff: Dict, min_accuracy: float = 0.995) -> bool:
    """All three conditions, because accuracy alone cannot see an insertion:
    a clip that says everything it was sent *plus* four extra words scores
    1.000 and is still broken."""
    return (not diff["hard_dropped"]
            and not diff["hard_inserted"]
            and diff["word_accuracy"] >= min_accuracy)


def clean(diff: Dict) -> bool:
    """Nothing was genuinely dropped or added. Substitutions the transcriber
    most likely misheard do not count — see `likely_asr_artifact`."""
    return not diff.get("missing") and not diff.get("extra")


def worth_regenerating(diff: Dict, garbled_fraction: float = 0.5,
                       garbled_min_words: int = 3) -> bool:
    """Should we spend GPU time making this chunk again?

    Only for a real defect. A re-generation costs as much as the original, so
    spending one on a proper noun the transcriber fumbled is pure waste — and
    the second attempt would be judged by the same fallible transcriber.

    The one exception is a clip where *most* of the words came back as
    near-misses: that is mush, not a mishearing. Both conditions are required,
    because "Bessent and Hegseth testified" is 50% near-misses and perfectly
    fine — a short sentence with two proper nouns in it, nothing more.
    """
    if diff.get("missing") or diff.get("extra"):
        return True
    heard = diff.get("misheard", [])
    if len(heard) < garbled_min_words:
        return False
    n = max(diff.get("sent_words", 0), 1)
    return len(heard) / n > garbled_fraction


def pronunciation_note(diff: Dict, limit: int = 4) -> Optional[str]:
    """Advisory: words that came back sounding like something else."""
    heard = diff.get("misheard") or []
    if not heard:
        return None
    shown = "; ".join(f"{a} heard as {b}" for a, b in heard[:limit])
    more = f" (+{len(heard) - limit} more)" if len(heard) > limit else ""
    return (f"check pronunciation: {shown}{more} — add it to lexicon.json if "
            f"it is really being said wrong")


def describe(diff: Dict, limit: int = 6) -> str:
    """One human-readable line for a log, a job card or a response header."""
    bits = []
    missing = diff.get("missing", diff["hard_dropped"])
    if missing:
        bits.append("not spoken: " + " ".join(missing[:limit])
                    + ("…" if len(missing) > limit else ""))
    if diff["tail_inserted"]:
        t = diff["tail_inserted"]
        bits.append("extra at end: " + " ".join(t[:limit])
                    + ("…" if len(t) > limit else ""))
    extra = [w for w in diff.get("extra", diff["hard_inserted"])
             if w not in diff["tail_inserted"]]
    if extra:
        bits.append("extra: " + " ".join(extra[:limit])
                    + ("…" if len(extra) > limit else ""))
    if not bits:
        note = pronunciation_note(diff, limit)
        return note or f"verified {diff['word_accuracy']:.1%}"
    return "; ".join(bits)


def only_tail_is_wrong(diff: Dict) -> bool:
    """True when the only problem is extra words after the end of the script.

    This is the repairable case: the audio for the script itself is fine and
    the artefact can be cut off, instead of paying for a whole re-generation.
    """
    if diff.get("missing", diff["hard_dropped"]):
        return False
    if not diff["tail_inserted"]:
        return False
    non_tail = [w for w in diff.get("extra", diff["hard_inserted"])
                if w not in diff["tail_inserted"]]
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
