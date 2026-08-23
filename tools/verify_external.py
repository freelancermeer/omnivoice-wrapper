#!/usr/bin/env python3
"""Check the server's own verdict against a completely different ASR.

The built-in verifier reads generated audio with Whisper. That is the right
tool, but it means one model is marking its own homework: if Whisper mishears
the same word the model mispronounced, the two errors cancel and the clip looks
clean. A second, unrelated transcriber cannot make that particular mistake.

This sends each clip to AssemblyAI, diffs the result against the script with
the same `verify.word_diff` the server uses, and prints where the two ASRs
agree and where they do not. Agreement is the evidence; disagreement is the
interesting part and is listed per clip.

    set ASSEMBLYAI_API_KEY=...
    venv\\Scripts\\python tools\\verify_external.py manifest.json

The key is read from the environment only — never pass it on the command line
and never commit it. Audio is uploaded to AssemblyAI, so run this on clips you
are willing to send to a third party.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import verify  # noqa: E402

AAI = "https://api.assemblyai.com/v2"


def transcribe_assemblyai(path: str, key: str, poll: float = 3.0,
                          timeout: float = 900.0) -> str:
    head = {"authorization": key}
    with open(path, "rb") as f:
        up = requests.post(f"{AAI}/upload", headers=head, data=f, timeout=600)
    up.raise_for_status()
    url = up.json()["upload_url"]

    job = requests.post(f"{AAI}/transcript", headers=head,
                        json={"audio_url": url, "language_code": "en_us"},
                        timeout=120)
    job.raise_for_status()
    jid = job.json()["id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{AAI}/transcript/{jid}", headers=head, timeout=120)
        r.raise_for_status()
        body = r.json()
        if body["status"] == "completed":
            return (body.get("text") or "").strip()
        if body["status"] == "error":
            raise RuntimeError(body.get("error", "assemblyai error"))
        time.sleep(poll)
    raise TimeoutError(f"assemblyai did not finish within {timeout:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--base", default="http://127.0.0.1:8001",
                    help="the OmniVoice server, for the Whisper comparison")
    ap.add_argument("--out", help="write the full comparison as JSON")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
    if not key:
        sys.exit("set ASSEMBLYAI_API_KEY in the environment first")

    with open(args.manifest, encoding="utf-8") as f:
        items = json.load(f)
    if args.limit:
        items = items[: args.limit]

    rows = []
    agree = whisper_clean = external_clean = 0
    for i, item in enumerate(items, 1):
        path, text = item["file"], item["text"]
        if not os.path.exists(path):
            print(f"[{i}/{len(items)}] missing {path}")
            continue

        with open(path, "rb") as f:
            w = requests.post(args.base.rstrip("/") + "/api/transcribe",
                              data={"text": text},
                              files={"audio": (os.path.basename(path), f)},
                              timeout=900)
        w_text = w.json()["text"] if w.ok else ""
        w_diff = verify.word_diff(text, w_text)

        try:
            a_text = transcribe_assemblyai(path, key)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(items)}] assemblyai failed: {e}")
            continue
        a_diff = verify.word_diff(text, a_text)

        w_ok, a_ok = verify.clean(w_diff), verify.clean(a_diff)
        whisper_clean += w_ok
        external_clean += a_ok
        agree += (w_ok == a_ok)

        mark = "  agree" if w_ok == a_ok else "  <-- DISAGREE"
        print(f"[{i}/{len(items)}] {os.path.basename(path):<14} "
              f"whisper {w_diff['word_accuracy']:.3f} {'clean' if w_ok else 'flag '}"
              f" | assemblyai {a_diff['word_accuracy']:.3f} "
              f"{'clean' if a_ok else 'flag '}{mark}")
        if not a_ok:
            print(f"        assemblyai: dropped={a_diff['hard_dropped']} "
                  f"extra={a_diff['hard_inserted']}")

        rows.append({"file": path, "text": text,
                     "whisper": {"text": w_text, "accuracy": w_diff["word_accuracy"],
                                 "clean": w_ok, "dropped": w_diff["hard_dropped"],
                                 "inserted": w_diff["hard_inserted"]},
                     "assemblyai": {"text": a_text, "accuracy": a_diff["word_accuracy"],
                                    "clean": a_ok, "dropped": a_diff["hard_dropped"],
                                    "inserted": a_diff["hard_inserted"]}})

    n = len(rows)
    if n:
        w_acc = sum(r["whisper"]["accuracy"] for r in rows) / n
        a_acc = sum(r["assemblyai"]["accuracy"] for r in rows) / n
        print("\n" + "=" * 66)
        print(f"{n} clips checked by two independent transcribers")
        print(f"  whisper    : {whisper_clean}/{n} clean   mean accuracy {w_acc:.4f}")
        print(f"  assemblyai : {external_clean}/{n} clean   mean accuracy {a_acc:.4f}")
        print(f"  the two agree on {agree}/{n} clips")
        print("=" * 66)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
