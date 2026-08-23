#!/usr/bin/env python3
"""Generate a batch of clips and write the manifest `audit_batch.py` expects.

The batch that found the original bugs is not in this repo — only its findings
are. This rebuilds a batch in the same register (news commentary: numbers,
currency, em dashes, repeated clauses, proper nouns that Whisper mishears) so
that `audit_batch.py` has something real to measure, including the two specific
lines the bug reports named:

  * "Absolutely damning. Hegseth is a snake."  — RESEARCH.md section 2, a clip
    that came back missing its opening word.
  * the perjury / fraud / contempt triple      — README, one arm of a repeated
    structure dropped silently.

    venv\\Scripts\\python tools\\make_batch.py --voice RVoiceover_3_2
    venv\\Scripts\\python tools\\audit_batch.py manifest.json --out audit.json
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

SCRIPTS = [
    "Absolutely damning. Hegseth is a snake.",
    "He called it perjury, he called it fraud, he called it contempt.",
    "A property worth $500 million was appraised at $150 million.",
    "His penthouse went from 11,000 square feet to 30,000.",
    "Trump inflated values by 200 to 300%.",
    "His sons owe $4.7 million each.",
    "He admitted the plot—the deliberate plan—was not official.",
    "What about the January 6th speech?",
    "Bessent said the tariffs would pay for themselves by the 3rd quarter.",
    "The one word Trump's team ignores: private.",
    "Zero credible sources have verified these events.",
    "Trump sold himself as the comeback king, the guy who couldn't be stopped.",
    "The visceral outburst revealed a presidency fracturing under judicial rebuke.",
    "Kennedy forced Hegseth to admit the strike had no legal authorisation.",
    "They are sabotaging MAGA, says the former NSC official.",
    "It's not losing an election. It's not even tanking approval numbers.",
    "Let me take you back to where this story begins.",
    "The judge ordered immediate seizure of the assets on the 1st of March.",
    "Right now, his own handpicked judges are torching his executive orders.",
    "That figure rose from 2,400 to 96,000 in under eighteen months.",
    "The CFO testified for six hours, and the RFK deposition ran longer.",
    "Understanding the scale of what happened means starting at the beginning.",
    "While they brag about total victory in Iran, the report says otherwise.",
    "Nobody on that committee expected the answer they got.",
    "This is the part of the story that never made the evening news.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True, help="a registered voice_id")
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--key", default=os.environ.get("OMNIVOICE_API_KEY") or None)
    ap.add_argument("--out-dir", default="outputs/batch")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--steps", default="16")
    ap.add_argument("--limit", type=int, default=0, help="only the first N")
    ap.add_argument("--scripts", help="JSON list of scripts to use instead "
                    "of the built-in batch (e.g. tools/sample_scripts.json)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    headers = {"X-API-Key": args.key} if args.key else {}
    scripts = SCRIPTS
    if args.scripts:
        with open(args.scripts, encoding="utf-8") as f:
            scripts = json.load(f)
    if args.limit:
        scripts = scripts[: args.limit]

    manifest, rtfs, warned = [], [], 0
    t_start = time.time()
    for i, text in enumerate(scripts, 1):
        r = requests.post(
            args.base.rstrip("/") + "/api/tts", headers=headers,
            data={"text": text, "format": "wav", "steps": args.steps,
                  "voice_id": args.voice, "project": f"batch{i:02d}"},
            timeout=900)
        if not r.ok:
            print(f"[{i}/{len(scripts)}] FAILED HTTP {r.status_code} {r.text[:120]}")
            continue
        path = os.path.join(args.out_dir, f"clip{i:02d}.wav")
        with open(path, "wb") as f:
            f.write(r.content)
        rtf = r.headers.get("X-RTF", "")
        warn = r.headers.get("X-OmniVoice-Warning", "")
        if warn:
            warned += 1
        try:
            rtfs.append(float(rtf))
        except ValueError:
            pass
        manifest.append({"file": path.replace("\\", "/"), "text": text})
        print(f"[{i}/{len(scripts)}] {os.path.basename(path)} "
              f"rtf={rtf} dur={r.headers.get('X-Duration-Sec','?')}s "
              f"lufs={r.headers.get('X-LUFS','?')}"
              + (f"  WARN: {warn[:80]}" if warn else ""))

    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"{len(manifest)}/{len(scripts)} clips in {time.time() - t_start:.0f}s")
    if rtfs:
        print(f"RTF  avg {sum(rtfs) / len(rtfs):.3f}   "
              f"best {min(rtfs):.3f}   worst {max(rtfs):.3f}")
    print(f"clips carrying a warning: {warned}")
    print(f"manifest -> {args.manifest}")
    print("=" * 60)
    return 0 if manifest else 1


if __name__ == "__main__":
    sys.exit(main())
