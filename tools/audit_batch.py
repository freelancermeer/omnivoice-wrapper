#!/usr/bin/env python3
"""Re-run the measurement that found the bugs, on your own finished batch.

The original audit generated 63 clips, transcribed every one of them back, and
diffed the transcript against the text that had been sent. That is the only way
those bugs were visible at all — a caller has no other way to know that four
words were dropped three hours ago.

This does the same thing locally, using the Whisper already loaded in the
server, so it costs nothing to run after every batch.

    venv\\Scripts\\python tools\\audit_batch.py manifest.json

manifest.json is a list of what you sent and where the audio landed:

    [
      {"file": "outputs/intro.wav", "text": "the script that was sent"},
      {"file": "outputs/part2.wav", "text": "..."}
    ]

A CSV with columns file,text works too.

Output is the same table shape as the original report: words added, words
dropped, and which clips are responsible.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import verify  # noqa: E402


def load_manifest(path):
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            return [dict(r) for r in csv.DictReader(f)]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--key", default=os.environ.get("OMNIVOICE_API_KEY") or None)
    ap.add_argument("--out", help="write the full per-clip report as JSON")
    args = ap.parse_args()

    items = load_manifest(args.manifest)
    headers = {"X-API-Key": args.key} if args.key else {}

    total_added = total_dropped = 0
    clips_with_bleed = 0
    rows, tail_words, dropped_words = [], Counter(), Counter()
    sent_total = spoken_total = 0

    for i, item in enumerate(items, 1):
        path, text = item.get("file"), item.get("text", "")
        if not path or not os.path.exists(path):
            print(f"[{i}/{len(items)}] missing: {path}")
            continue
        with open(path, "rb") as f:
            r = requests.post(args.base.rstrip("/") + "/api/transcribe",
                              headers=headers, data={"text": text},
                              files={"audio": (os.path.basename(path), f)},
                              timeout=600)
        if not r.ok:
            print(f"[{i}/{len(items)}] transcribe failed ({r.status_code}) {path}")
            continue
        body = r.json()
        d = body.get("diff") or verify.word_diff(text, body["text"])
        n_add, n_drop = len(d["hard_inserted"]), len(d["hard_dropped"])
        total_added += n_add
        total_dropped += n_drop
        sent_total += d["sent_words"]
        spoken_total += d["spoken_words"]
        if d["tail_inserted"]:
            clips_with_bleed += 1
            tail_words.update(d["tail_inserted"])
        dropped_words.update(d["hard_dropped"])
        rows.append({"file": path, "added": n_add, "dropped": n_drop,
                     "word_accuracy": d["word_accuracy"],
                     "tail_inserted": d["tail_inserted"],
                     "hard_dropped": d["hard_dropped"],
                     "heard": body["text"]})
        flag = "" if verify.passed(d) else "   <-- " + verify.describe(d)
        print(f"[{i}/{len(items)}] {os.path.basename(path):<34} "
              f"acc {d['word_accuracy']:.3f}  +{n_add} -{n_drop}{flag}")

    n = len(rows)
    print("\n" + "=" * 64)
    print(f"{n} clips · {sent_total} words sent · {spoken_total} words spoken")
    print(f"words ADDED that were never sent : {total_added}")
    print(f"words DROPPED                    : {total_dropped}")
    print(f"clips with a trailing artefact   : {clips_with_bleed}"
          f"  ({clips_with_bleed / max(n, 1):.0%})")
    if tail_words:
        print("\nmost common leaked words (reference bleed):")
        for w, c in tail_words.most_common(8):
            print(f"   {w:<20} x{c}")
    if dropped_words:
        print("\nmost common dropped words:")
        for w, c in dropped_words.most_common(8):
            print(f"   {w:<20} x{c}")
    clean = sum(1 for r_ in rows if not r_["added"] and not r_["dropped"])
    print(f"\nclean clips: {clean}/{n}")
    print("=" * 64)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"per-clip report written to {args.out}")
    return 0 if (total_added == 0 and total_dropped == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
