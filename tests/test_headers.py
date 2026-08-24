"""The header value that killed a finished response.

Truncating the normalized script at 480 characters lands on a space often
enough, and a header value may not end with whitespace. h11 rejects it inside
`send()` — after the route has returned, after the audio was generated — so
FastAPI cannot turn it into a 500 and the caller gets a closed connection with
no status, no body and nothing in any counter, 26-41s after asking.

These run without a GPU: the sanitiser is pure string handling.
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _hdr(value, limit=480):
    """Kept in step with app._hdr, which cannot be imported without a GPU."""
    s = str(value if value is not None else "")
    s = s.encode("ascii", "replace").decode("ascii")
    s = "".join(ch if 0x20 <= ord(ch) < 0x7F else " " for ch in s)
    return s[:limit].strip()


# h11's own rule: no leading or trailing whitespace, no control characters.
_H11_FIELD_VALUE = re.compile(
    rb"[ \t]*(?:[^\x00-\x08\x0a-\x1f\x7f]*[^\x00-\x08\x0a-\x1f\x7f \t])?[ \t]*")


def _sendable(value: str) -> bool:
    raw = value.encode("latin-1", "replace")
    m = _H11_FIELD_VALUE.fullmatch(raw)
    return bool(m) and raw == raw.strip()


def test_truncation_landing_on_a_space_is_still_sendable():
    """The reported reproducer: character 479 of that segment is a space."""
    text = ("word " * 200)                      # every 5th character is a space
    out = _hdr(text)
    assert _sendable(out), repr(out[-20:])
    assert not out.endswith(" ")


def test_every_truncation_offset_produces_a_sendable_header():
    text = "".join(f"{i % 10} word here, " for i in range(200))
    for limit in range(1, 300):
        out = _hdr(text, limit=limit)
        assert _sendable(out), (limit, repr(out[-12:]))


def test_control_characters_never_reach_the_wire():
    for bad in ("a\nb", "a\rb", "a\tb", "a\x00b", "a\x7fb"):
        out = _hdr(bad)
        assert _sendable(out), repr(out)


def test_non_ascii_is_replaced_not_raised():
    out = _hdr("caf\u00e9 na\u00efve \u2014 dash")
    assert _sendable(out)
    assert out == out.strip()


def test_empty_and_none_are_safe():
    assert _hdr(None) == ""
    assert _hdr("") == ""
    assert _hdr("   ") == ""
