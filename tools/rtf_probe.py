#!/usr/bin/env python3
"""Measure RTF against clip length, with verification on and off.

A single RTF number is close to meaningless for this server, because a fixed
per-request cost (voice prompt, the verifier's Whisper pass, encode) is spread
over however much audio the clip happens to contain. A two-second clip and a
sixty-second clip give completely different answers from the same machine.

So this walks a ladder of lengths and prints RTF for each, which is the only
form in which the number can be compared to anything.

    venv\\Scripts\\python tools\\rtf_probe.py --voice RVoiceover_3_2

Run it once as-is and once against a server started with OMNIVOICE_VERIFY=0;
the difference between the two tables is what verification actually costs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

# One paragraph of ordinary voiceover copy. The ladder takes prefixes of it, so
# every rung is the same voice, the same register and the same sentence shapes.
BODY = (
    "The hearing opened at nine in the morning, and by the first recess it was "
    "already clear that the committee had lost control of its own schedule. "
    "Members who had prepared careful questions found themselves interrupting "
    "each other. The witness answered slowly, choosing every word, and the "
    "room grew quieter as he did. Outside, a line of reporters stretched the "
    "length of the corridor, waiting for a statement that never came. "
    "By noon the transcript ran to ninety pages, and the most important "
    "sentence in it had been spoken almost in passing, twenty minutes in, "
    "when nobody was writing anything down. The chairman called for order "
    "twice, and the second time he did not raise his voice at all. "
    "What followed was the part that everyone would later argue about, "
    "because the recording captured it clearly and the notes did not. "
    "Two staffers left the room. The senior counsel leaned forward, asked a "
    "question that had not been on any list, and waited a long time for the "
    "answer. It came in a single sentence, and it changed the direction of "
    "the entire afternoon. The committee adjourned without scheduling a "
    "return date, which in that building is a message of its own."
)

RUNGS = [12, 30, 60, 120, 240]  # words


def take_words(n: int) -> str:
    return " ".join(BODY.split()[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--key", default=os.environ.get("OMNIVOICE_API_KEY") or None)
    ap.add_argument("--steps", default="16")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", help="write the measurements as JSON")
    args = ap.parse_args()

    headers = {"X-API-Key": args.key} if args.key else {}
    print(f"\n=== RTF ladder{' — ' + args.label if args.label else ''} "
          f"({args.steps} steps) ===\n")
    print(f"{'words':>6} {'audio_s':>8} {'gen_s':>7} {'RTF':>7} "
          f"{'verified':>9}  warning")

    rows = []
    for n in RUNGS:
        text = take_words(n)
        for _ in range(args.repeat):
            t0 = time.time()
            r = requests.post(args.base.rstrip("/") + "/api/tts", headers=headers,
                              data={"text": text, "format": "wav",
                                    "steps": args.steps, "voice_id": args.voice,
                                    "project": f"rtf{n}"}, timeout=1800)
            wall = time.time() - t0
            if not r.ok:
                print(f"{n:>6}  FAILED HTTP {r.status_code} {r.text[:90]}")
                continue
            dur = float(r.headers.get("X-Duration-Sec", 0) or 0)
            rtf = float(r.headers.get("X-RTF", 0) or 0)
            ver = r.headers.get("X-Verified", "?")
            warn = r.headers.get("X-OmniVoice-Warning", "")
            gen = rtf * dur if rtf and dur else 0.0
            rows.append({"words": n, "audio_sec": dur, "gen_sec": round(gen, 2),
                         "rtf": rtf, "verified": ver, "wall_sec": round(wall, 2),
                         "warning": warn})
            print(f"{n:>6} {dur:>8.2f} {gen:>7.2f} {rtf:>7.3f} {ver:>9}  "
                  f"{warn[:60]}")

    if rows:
        long_rows = [r_ for r_ in rows if r_["words"] >= 60]
        aud = sum(r_["audio_sec"] for r_ in rows)
        gen = sum(r_["gen_sec"] for r_ in rows)
        print("\n" + "-" * 58)
        print(f"pooled RTF (all rungs)          : {gen / max(aud, 1e-9):.3f}")
        if long_rows:
            a2 = sum(r_["audio_sec"] for r_ in long_rows)
            g2 = sum(r_["gen_sec"] for r_ in long_rows)
            print(f"pooled RTF (60+ words, realistic): {g2 / max(a2, 1e-9):.3f}")
        print("-" * 58 + "\n")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"label": args.label, "steps": args.steps, "rows": rows},
                      f, indent=2)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
