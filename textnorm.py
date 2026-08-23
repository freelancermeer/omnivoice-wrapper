#!/usr/bin/env python3
"""Text front-end for OmniVoice — everything that happens to a script *before*
the model sees it.

Deliberately dependency-light (only `num2words`, and `omnivoice` optionally for
the language map) so it can be imported, unit-tested and iterated on a machine
with no GPU and no model weights:

    python -m pytest tests/test_textnorm.py

What it does, in order:
  1. Unicode cleanup      curly quotes/dashes/ellipsis/NBSP -> plain ASCII,
                          em dash & colon -> comma pause, unknown chars dropped.
  2. Lexicon              user-editable word -> pronunciation map (lexicon.json).
  3. Currency             "$4.7 million" -> "four point seven million dollars".
  4. Percent              "300%"         -> "three hundred percent".
  5. Ordinals / dates     "6th", "January 6" -> "sixth", "January sixth".
  6. Numbers              "11,000" -> "eleven thousand"; 7+ digit runs and
                          hyphenated phone numbers digit-by-digit; 1100-2099
                          read as years ("2024" -> "twenty twenty-four").
  7. Whitespace/punct collapse.

Sentence splitting and chunking live here too, because they have to know about
abbreviations ("U.S. Marshals" must not become two sentences).

NORMALIZE_LEVEL:
  "off"   - hand the text to the model untouched.
  "basic" - unicode cleanup + plain digits (the pre-2.0 behaviour).
  "full"  - everything above (default).
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("omnivoice.textnorm")

try:
    from num2words import num2words as _num2words
    from num2words import CONVERTER_CLASSES as _N2W_CLASSES

    N2W_LANGS = set(_N2W_CLASSES.keys())
except Exception:  # pragma: no cover - num2words is optional
    _num2words = None
    N2W_LANGS = set()

try:  # only available where omnivoice is installed
    from omnivoice.utils.lang_map import LANG_NAME_TO_ID
except Exception:  # pragma: no cover
    LANG_NAME_TO_ID = {}


# ---------------------------------------------------------------------------
# 1. Unicode cleanup
# ---------------------------------------------------------------------------
# Characters that must become their plain-ASCII equivalent before anything else.
# Doing this FIRST also fixes a real bug: the language guard below used to bail
# out on any codepoint > 0x2FF, so a single curly apostrophe ("Trump's") or em
# dash disabled number normalization for the whole script.
_CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "«": '"', "»": '"',
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "　": " ",
    "​": "", "‌": "", "‍": "", "﻿": "", "­": "",
    "−": "-", "‐": "-", "‑": "-",
}

# Em dash, en dash, horizontal bar, two-em/three-em dash -> a comma-length pause.
_DASH_RE = re.compile(r"\s*[—–―⸺⸻]+\s*")
_ASCII_DASH_RE = re.compile(r"\s*--+\s*")
_SPACED_HYPHEN_RE = re.compile(r"(?<=\S) +- +(?=\S)")
_ELLIPSIS_RE = re.compile(r"\s*…\s*")
# A colon that is not part of a clock time ("3:30") becomes a pause.
_COLON_RE = re.compile(r"(?<!\d):(?!\d)")

# Categories we silently drop rather than fail on (emoji, control chars, ...).
_DROP_CATEGORIES = {"Cc", "Cf", "Co", "Cs", "Cn", "So", "Sk"}


def unicode_cleanup(text: str, notes: Optional[List[str]] = None) -> str:
    """ASCII-fold punctuation, turn dashes/colons into pauses, drop junk."""
    notes = notes if notes is not None else []
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = "".join(_CHAR_MAP.get(ch, ch) for ch in t)
    t = _ELLIPSIS_RE.sub(". ", t)
    t = _DASH_RE.sub(", ", t)
    t = _ASCII_DASH_RE.sub(", ", t)
    t = _SPACED_HYPHEN_RE.sub(", ", t)
    t = _COLON_RE.sub(",", t)
    t = t.replace(";", ",")
    t = t.replace('"', " ")          # quotes must be silent, not spoken
    t = t.replace("\n", " ")

    dropped = set()
    kept = []
    for ch in t:
        if ch in "\t ":
            kept.append(" ")
            continue
        if unicodedata.category(ch) in _DROP_CATEGORIES:
            dropped.add(ch)
            continue
        kept.append(ch)
    if dropped:
        # "Never fail on an unknown character - say it as nothing, and log it."
        log.info("dropped unsupported char(s): %r", sorted(dropped))
        notes.append(f"dropped {len(dropped)} unsupported char(s)")
    return "".join(kept)


def collapse(text: str) -> str:
    """Tidy the spacing/punctuation left behind by the expansions above."""
    t = re.sub(r"[ \t]+", " ", text)
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    t = re.sub(r"(?:,\s*){2,}", ", ", t)
    t = re.sub(r",\s*([.!?])", r"\1", t)
    t = re.sub(r"([.!?])\s*,", r"\1 ", t)
    t = re.sub(r"([,.!?])(?=[^\s,.!?'\")\]])", r"\1 ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


# ---------------------------------------------------------------------------
# 2. Language resolution
# ---------------------------------------------------------------------------
# A label like "English" has to resolve to "en" even when omnivoice's own map
# is unavailable (this module is importable without it) or does not carry that
# spelling. Falling through to None silently disabled every number rule for a
# request that politely said language="English".
_NAME_TO_CODE = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "russian": "ru", "japanese": "ja",
    "korean": "ko", "chinese": "zh", "mandarin": "zh", "arabic": "ar",
    "hindi": "hi", "urdu": "ur", "dutch": "nl", "polish": "pl",
    "turkish": "tr", "indonesian": "id", "vietnamese": "vi", "thai": "th",
    "ukrainian": "uk", "czech": "cs", "danish": "dk", "finnish": "fi",
    "hebrew": "he", "hungarian": "hu", "norwegian": "no", "romanian": "ro",
    "serbian": "sr", "slovene": "sl", "slovenian": "sl", "swedish": "sv",
    "lithuanian": "lt", "latvian": "lv", "kazakh": "kz", "persian": "fa",
    "farsi": "fa", "catalan": "ca", "croatian": "hr", "bulgarian": "bg",
    "greek": "el", "estonian": "et", "icelandic": "is", "tamil": "ta",
    "telugu": "te", "kannada": "kn", "malay": "ms",
}


def _has_non_latin_letters(text: str) -> bool:
    # NOTE: .isalpha() is the important part — punctuation like the curly
    # apostrophe is above 0x2FF but says nothing about the script.
    return any(ch.isalpha() and ord(ch) > 0x2FF for ch in text)


def resolve_n2w_lang(language: Optional[str], text: str) -> Optional[str]:
    """Map an OmniVoice language label to a num2words code, or None to skip."""
    if _num2words is None:
        return None
    if not language or language == "Auto":
        if _has_non_latin_letters(text):
            return None
        return "en" if "en" in N2W_LANGS else None
    label = language.strip().lower()
    for code in (LANG_NAME_TO_ID.get(label), _NAME_TO_CODE.get(label), label):
        if not code:
            continue
        if code in N2W_LANGS:
            return code
        base = code.split("_")[0].split("-")[0]
        if base in N2W_LANGS:
            return base
    return None


def _is_english(code: Optional[str]) -> bool:
    return bool(code) and code.split("_")[0].split("-")[0] == "en"


def _n2w(value, code: str, to: str = "cardinal") -> Optional[str]:
    try:
        return _num2words(value, lang=code, to=to)
    except Exception:
        try:
            return _num2words(value, lang=code.split("_")[0], to=to)
        except Exception:
            return None


def _number_words(raw: str, code: str, years: bool = True) -> Optional[str]:
    """'11,000' -> eleven thousand ; '4.7' -> four point seven."""
    clean = raw.replace(",", "").strip()
    if not clean:
        return None
    try:
        if "." in clean:
            return _n2w(float(clean), code)
        n = int(clean)
    except ValueError:
        return None
    # A bare 4-digit number in this range is almost always a year in narration.
    if (years and _is_english(code) and len(clean) == 4 and 1100 <= n <= 2099
            and not clean.startswith("0")):
        w = _n2w(n, code, to="year")
        if w:
            return w
    return _n2w(n, code)


# ---------------------------------------------------------------------------
# 3. Lexicon (pronunciation overrides + pronounceable acronyms)
# ---------------------------------------------------------------------------
# Kept deliberately small. The production report verified that OmniVoice
# already spells ordinary acronyms correctly (RFK -> "r f k", CFO -> "c f o"),
# so we do NOT touch acronyms in general — only the handful that must be read
# as a word instead of letter-by-letter. Everything else belongs in
# lexicon.json, which the user owns.
DEFAULT_LEXICON: Dict[str, str] = {
    "MAGA": "Maga",
    "NASA": "Nasa",
    "NATO": "Nato",
    "OPEC": "Opec",
    "NAFTA": "Nafta",
    "FEMA": "Fema",
    "UNESCO": "Unesco",
    "UNICEF": "Unicef",
    "SCOTUS": "Scotus",
    "POTUS": "Potus",
    "FIFA": "Fifa",
}


def load_lexicon(path: Optional[str], use_defaults: bool = True) -> Dict[str, str]:
    """Merge lexicon.json over the built-in defaults. Missing file is fine."""
    lex: Dict[str, str] = dict(DEFAULT_LEXICON) if use_defaults else {}
    if path:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if k.startswith("_"):
                    continue          # "_comment" keys are documentation
                if v is None or v == "":
                    lex.pop(k, None)  # explicit removal of a default
                else:
                    lex[str(k)] = str(v)
            log.info("lexicon: loaded %d entry(ies) from %s", len(data), path)
        except FileNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001
            log.warning("lexicon %s could not be read: %s", path, e)
    return lex


def apply_lexicon(text: str, lexicon: Optional[Dict[str, str]],
                  notes: Optional[List[str]] = None) -> str:
    if not lexicon:
        return text
    notes = notes if notes is not None else []
    hits = 0
    # Longest first so "Don Jr" wins over "Don".
    for key in sorted(lexicon, key=len, reverse=True):
        if key.startswith("_"):
            continue
        pat = re.compile(r"(?<![\w'])" + re.escape(key) + r"(?![\w])")
        text, n = pat.subn(lexicon[key].replace("\\", r"\\"), text)
        hits += n
    if hits:
        notes.append(f"lexicon: {hits} replacement(s)")
    return text


# ---------------------------------------------------------------------------
# 4. Currency / percent / ordinals / dates / numbers
# ---------------------------------------------------------------------------
_CURRENCY_NAMES = {
    "$": ("dollar", "dollars"),
    "£": ("pound", "pounds"),
    "€": ("euro", "euros"),
    "¥": ("yen", "yen"),
    "₹": ("rupee", "rupees"),
    "₩": ("won", "won"),
}
_SCALE = r"(?:hundred|thousand|million|billion|trillion)"
_CURRENCY_RE = re.compile(
    r"([$£€¥₹₩])\s?(\d[\d,]*(?:\.\d+)?)"
    r"(?:\s*(" + _SCALE + r")\b)?",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*%")
_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_MONTHS = ("January|February|March|April|May|June|July|August|September|"
           "October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
_MONTH_DAY_RE = re.compile(
    r"\b(" + _MONTHS + r")\.?\s+(\d{1,2})\b(?!\s*(?:st|nd|rd|th)\b)(?!\s*,?\s*\d{3})",
    re.IGNORECASE,
)
# A digit run that may carry . , or - as internal separators.
_NUM_RE = re.compile(r"\d[\d.,\-]*\d|\d")


def expand_currency(text: str, code: str, notes: List[str]) -> str:
    def repl(m: "re.Match") -> str:
        sym, num, scale = m.group(1), m.group(2), m.group(3)
        words = _number_words(num, code, years=False)
        if words is None:
            return m.group(0)
        one, many = _CURRENCY_NAMES[sym]
        unit = many
        if not scale:
            try:
                if abs(float(num.replace(",", "")) - 1.0) < 1e-9:
                    unit = one
            except ValueError:
                pass
        out = words + (" " + scale.lower() if scale else "") + " " + unit
        return out

    out = _CURRENCY_RE.sub(repl, text)
    if out != text:
        notes.append("currency expanded")
    return out


def expand_percent(text: str, code: str, notes: List[str]) -> str:
    def repl(m: "re.Match") -> str:
        words = _number_words(m.group(1), code, years=False)
        return f"{words} percent" if words else m.group(0)

    out = _PERCENT_RE.sub(repl, text)
    out = out.replace("%", " percent")
    if out != text:
        notes.append("percent expanded")
    return out


def expand_times(text: str, code: str, notes: List[str]) -> str:
    """The colon in a clock is protected from the pause rule, so it has to be
    spoken here or it reaches the model as "three:thirty"."""
    def repl(m: "re.Match") -> str:
        hour, minute = int(m.group(1)), int(m.group(2))
        h = _n2w(hour, code)
        if h is None:
            return m.group(0)
        if minute == 0:
            return f"{h} o'clock"
        if minute < 10:
            unit = _n2w(minute, code)
            return f"{h} oh {unit}" if unit else m.group(0)
        mm = _n2w(minute, code)
        return f"{h} {mm}" if mm else m.group(0)

    out = _TIME_RE.sub(repl, text)
    if out != text:
        notes.append("times expanded")
    return out


def expand_ordinals(text: str, code: str, notes: List[str]) -> str:
    def repl(m: "re.Match") -> str:
        w = _n2w(int(m.group(1)), code, to="ordinal")
        return w or m.group(0)

    out = _ORDINAL_RE.sub(repl, text)
    if out != text:
        notes.append("ordinals expanded")
    return out


def expand_dates(text: str, code: str, notes: List[str]) -> str:
    """'January 6' -> 'January sixth' (English narration convention)."""
    def repl(m: "re.Match") -> str:
        day = int(m.group(2))
        if not 1 <= day <= 31:
            return m.group(0)
        w = _n2w(day, code, to="ordinal")
        return f"{m.group(1)} {w}" if w else m.group(0)

    out = _MONTH_DAY_RE.sub(repl, text)
    if out != text:
        notes.append("dates expanded")
    return out


def expand_numbers(text: str, code: str, notes: List[str],
                   years: bool = True, digit_run: int = 7) -> str:
    """Remaining bare numbers. Long runs (phone numbers) go digit by digit."""
    def repl(m: "re.Match") -> str:
        tok = m.group(0)
        digits = re.sub(r"\D", "", tok)
        if not digits:
            return tok
        try:
            if len(digits) >= digit_run:
                # 5551234567 and 555-123-4567 both read digit by digit.
                return " ".join(_n2w(int(d), code) or d for d in digits)
            if "-" in tok:
                return tok            # "1-2" is ambiguous; leave it alone
            return _number_words(tok, code, years=years) or tok
        except Exception:  # noqa: BLE001
            return tok

    out = _NUM_RE.sub(repl, text)
    if out != text:
        notes.append("numbers expanded")
    return out


# ---------------------------------------------------------------------------
# 5. The public entry point
# ---------------------------------------------------------------------------
def normalize_text(text: str,
                   language: Optional[str] = None,
                   level: str = "full",
                   lexicon: Optional[Dict[str, str]] = None,
                   years: bool = True) -> Tuple[str, List[str]]:
    """Return (text_for_the_model, notes). Never raises.

    `lexicon=None` uses DEFAULT_LEXICON; pass `{}` to disable it entirely.
    """
    if lexicon is None:
        lexicon = DEFAULT_LEXICON
    notes: List[str] = []
    if not text or not text.strip():
        return text, notes
    level = (level or "full").lower()
    if level == "off":
        return text, notes

    try:
        t = unicode_cleanup(text, notes)
        code = resolve_n2w_lang(language, t)
        if code is None:
            # Unsupported/non-Latin: cleanup only, never touch the digits.
            out = collapse(t)
            if out != text.strip():
                notes.append("punctuation cleaned")
            return out, notes

        if level == "full":
            t = apply_lexicon(t, lexicon, notes)
            if _is_english(code):
                # Currency/percent/ordinal word order is English-specific.
                t = expand_currency(t, code, notes)
                t = expand_percent(t, code, notes)
                t = expand_times(t, code, notes)
                t = expand_dates(t, code, notes)
                t = expand_ordinals(t, code, notes)
        t = expand_numbers(t, code, notes, years=years)
        out = collapse(t)
    except Exception as e:  # noqa: BLE001 - normalization must never break TTS
        log.warning("normalization failed (%s); using the original text", e)
        return text, notes

    if out != text.strip():
        log.info("Normalized [%s]: %r -> %r", code, text, out)
    return out, notes


# ---------------------------------------------------------------------------
# 6. Sentence splitting (abbreviation-aware) and chunking
# ---------------------------------------------------------------------------
# Titles and connectors that are always followed by more of the same sentence.
_ABBREV_ALWAYS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "mt", "vs", "approx", "cf", "eg",
    "ie", "fig", "gen", "gov", "hon", "lt", "maj", "rep", "sen", "sgt", "col",
    "capt", "cmdr", "adm", "no", "vol",
}
# These genuinely can end a sentence ("...said Don Jr. He then left."), so they
# only suppress a split when the next word is lowercase.
_ABBREV_AMBIGUOUS = {
    "jr", "sr", "inc", "ltd", "co", "corp", "dept", "est", "etc", "al",
    "min", "max", "sec", "hr", "hrs", "ft",
}
_ABBREV = _ABBREV_ALWAYS | _ABBREV_AMBIGUOUS
_SENT_END_RE = re.compile(r"[.!?。！？]+[\"')\]]*(?:\s+|$)")
_TAIL_TOKEN_RE = re.compile(r"([A-Za-z][A-Za-z.]*)\.\s*$")


def _ends_with_abbreviation(fragment: str, next_is_upper: bool = False) -> bool:
    """True when the '.' that ends `fragment` is not a sentence terminator."""
    m = _TAIL_TOKEN_RE.search(fragment)
    if not m:
        return False
    tok = m.group(1)
    if "." in tok:
        return True                      # U.S. / a.m. / e.g.
    if len(tok) == 1:
        return True                      # an initial: "Donald J. Trump"
    tok = tok.lower()
    if tok in _ABBREV_ALWAYS:
        return True
    if tok in _ABBREV_AMBIGUOUS:
        return not next_is_upper
    return False


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts: List[str] = []
    start = 0
    for m in _SENT_END_RE.finditer(text):
        nxt = text[m.end():m.end() + 1]
        if _ends_with_abbreviation(text[start:m.end()], nxt.isupper()):
            continue
        piece = text[start:m.end()].strip()
        if piece:
            parts.append(piece)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


_SOFT_SPLIT_RE = re.compile(r"(?<=[,;:])\s+")


def _split_long(sentence: str, limit: int) -> List[str]:
    """A single sentence longer than ~2x the limit is split on commas, then on
    whitespace, so it can never degenerate into the long-text 'scratching'."""
    if len(sentence) <= limit * 2:
        return [sentence]
    out, cur = [], ""
    for piece in _SOFT_SPLIT_RE.split(sentence):
        if not cur:
            cur = piece
        elif len(cur) + 1 + len(piece) <= limit:
            cur += " " + piece
        else:
            out.append(cur)
            cur = piece
    if cur:
        out.append(cur)
    final: List[str] = []
    for piece in out:
        while len(piece) > limit * 2:
            cut = piece.rfind(" ", 0, limit)
            if cut <= 0:
                break
            final.append(piece[:cut])
            piece = piece[cut + 1:]
        final.append(piece)
    return [p for p in final if p.strip()]


def chunk_text(text: str, max_chars: int = 100,
               min_chars: Optional[int] = None) -> List[str]:
    """Split into ~evenly sized chunks at sentence boundaries.

    Even sizing matters: the production report measured 133-203 wpm across one
    batch, and short inputs are the rushed ones. Greedy packing used to leave a
    5-character last chunk next to a 100-character one; balanced packing keeps
    every chunk in the same size band, and a tiny tail is merged back.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences: List[str] = []
    for s in split_sentences(text):
        sentences.extend(_split_long(s, max_chars))
    if not sentences:
        return [text]

    total = sum(len(s) for s in sentences) + max(0, len(sentences) - 1)
    n = max(1, math.ceil(total / max_chars))
    target = max(1, math.ceil(total / n))

    chunks: List[str] = []
    cur = ""
    for s in sentences:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= target:
            cur += " " + s
        else:
            chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)

    min_chars = min_chars if min_chars is not None else max(20, max_chars // 4)
    if (len(chunks) > 1 and len(chunks[-1]) < min_chars
            and len(chunks[-2]) + len(chunks[-1]) + 1 <= int(max_chars * 1.6)):
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + " " + tail
    return chunks or [text]


def word_count(text: str) -> int:
    return len(re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?|\d+", text or ""))
