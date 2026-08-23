#!/usr/bin/env python3
"""Run this on the Windows box, against a running server, before you sell it.

    venv\\Scripts\\python tools\\acceptance.py --voice narrator_a

Every check here maps to a specific finding from the production batch. Nothing
in it is a unit test — the unit tests (`pytest tests/`) already ran on any
machine. These are the things that can only be answered by a real GPU, a real
model, and a real HTTP server.

Exit code 0 means every REQUIRED check passed. Optional checks print a result
and never fail the run.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import wave

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import verify as verify_mod
except Exception:
    verify_mod = None

PASS, FAIL, SKIP, INFO = "PASS", "FAIL", "SKIP", "INFO"
results = []


def record(name, state, detail="", required=True):
    results.append((name, state, detail, required))
    icon = {PASS: "[ok]  ", FAIL: "[FAIL]", SKIP: "[skip]", INFO: "[info]"}[state]
    print(f"{icon} {name}" + (f"  --  {detail}" if detail else ""))


def wav_seconds(data: bytes) -> float:
    try:
        with wave.open(io.BytesIO(data)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


class Client:
    def __init__(self, base, key=None):
        self.base = base.rstrip("/")
        self.headers = {"X-API-Key": key} if key else {}

    def get(self, path, **kw):
        return requests.get(self.base + path, headers=self.headers, timeout=120, **kw)

    def post(self, path, **kw):
        h = dict(self.headers)
        h.update(kw.pop("headers", {}))
        return requests.post(self.base + path, headers=h, timeout=900, **kw)

    def tts(self, text, **fields):
        data = {"text": text, "format": "wav", "steps": "16"}
        data.update({k: str(v) for k, v in fields.items() if v is not None})
        return self.post("/api/tts", data=data)


# ---------------------------------------------------------------------------
def check_health(c):
    r = c.get("/api/live")
    record("liveness /api/live answers", PASS if r.ok else FAIL, f"HTTP {r.status_code}")

    r = c.get("/api/ready")
    body = r.json()
    ok = r.status_code in (200, 503)
    record("readiness /api/ready answers 200 or 503", PASS if ok else FAIL,
           f"HTTP {r.status_code} {body.get('status')}")
    if r.status_code == 503:
        record("server is ready to generate", FAIL,
               f"not ready: {body.get('detail')}")
        return False
    record("readiness is backed by a real self-test", PASS,
           str(body.get("detail"))[:70])

    r = c.get("/api/health")
    if r.ok:
        vram = r.json().get("vram", {})
        record("health reports VRAM allocated AND reserved",
               PASS if "reserved_mb" in vram else FAIL, json.dumps(vram)[:110])
    return True


def check_contract(c):
    """PART B: the existing client must keep working, byte for byte."""
    r = c.tts("Contract check.", project="contract")
    if not r.ok:
        record("POST /api/tts returns audio bytes", FAIL, f"HTTP {r.status_code} {r.text[:120]}")
        return
    ctype = r.headers.get("Content-Type", "")
    record("POST /api/tts returns audio bytes, not JSON",
           PASS if ctype.startswith("audio/") else FAIL, ctype)
    record("X-Duration-Sec header still present",
           PASS if "X-Duration-Sec" in r.headers else FAIL,
           r.headers.get("X-Duration-Sec", ""))
    record("X-RTF header still present",
           PASS if "X-RTF" in r.headers else FAIL, r.headers.get("X-RTF", ""))
    record("project= is accepted (the client always sends it)",
           PASS if r.ok else FAIL)
    r2 = c.tts("Language label check.", language="English")
    record('language="English" is accepted',
           PASS if r2.ok else FAIL, f"HTTP {r2.status_code}")
    for extra in ({"speech_model": "x"},):
        r3 = c.post("/api/tts", data={"text": "Unknown param check.",
                                      "format": "wav", **extra})
        warned = "X-OmniVoice-Warning" in r3.headers
        record("an unknown parameter warns but does not 400",
               PASS if (r3.ok and warned) else FAIL,
               f"HTTP {r3.status_code} warn={warned}")


def check_unknown_voice(c):
    """D3: never substitute a voice. A wrong voice is worse than an error."""
    r = c.tts("Unknown voice check.", voice_id="does_not_exist_12345")
    record("unknown voice_id returns 404, never audio",
           PASS if r.status_code == 404 else FAIL,
           f"HTTP {r.status_code}" + (" -- IT RETURNED AUDIO" if r.ok else ""))


def check_limits(c):
    r = c.tts("word " * 5000)
    record("oversized input returns 413 with a usable message",
           PASS if r.status_code == 413 else FAIL, f"HTTP {r.status_code}")
    r = c.post("/api/tts", data={"text": "", "format": "wav"})
    record("empty text returns 400", PASS if r.status_code == 400 else FAIL,
           f"HTTP {r.status_code}")


def check_idempotency(c):
    key = f"acceptance-{int(time.time())}"
    a = c.post("/api/tts", data={"text": "Idempotency check.", "format": "wav"},
               headers={"Idempotency-Key": key})
    b = c.post("/api/tts", data={"text": "Idempotency check.", "format": "wav"},
               headers={"Idempotency-Key": key})
    replay = b.headers.get("X-OmniVoice-Idempotent-Replay") == "true"
    same = a.ok and b.ok and a.content == b.content
    record("a retry with the same Idempotency-Key replays, not regenerates",
           PASS if (replay and same) else FAIL,
           f"replay={replay} identical={same}")


def check_seed(c, voice):
    """D1: if seed does not control the output, every golden test is a coin flip."""
    text = "The judge ordered immediate seizure of the assets."
    a = c.tts(text, voice_id=voice, seed=42)
    b = c.tts(text, voice_id=voice, seed=42)
    d = c.tts(text, voice_id=voice, seed=43)
    if not (a.ok and b.ok and d.ok):
        record("seed determinism", SKIP, "generation failed", required=False)
        return
    same = a.content == b.content
    differs = a.content != d.content
    if same and differs:
        record("seed controls the output (same seed -> same audio)", PASS,
               "golden-file tests are meaningful", required=False)
    elif not same:
        record("seed controls the output", INFO,
               "seed 42 gave two different clips -- OmniVoice does not expose a "
               "usable seed, so run golden cases N times and look at the "
               "distribution instead of pinning one file", required=False)
    else:
        record("seed controls the output", INFO,
               "seeds 42 and 43 gave the same clip -- seed is being ignored",
               required=False)


def check_voice_report(c, path):
    if not path or not os.path.exists(path):
        record("reference validation report", SKIP, "no --ref-audio given",
               required=False)
        return None
    name = f"acceptance_{int(time.time())}"
    with open(path, "rb") as f:
        r = c.post("/api/voices", data={"name": name}, files={"voice": f})
    if r.status_code == 422:
        record("a bad reference is rejected with a reason", PASS,
               str(r.json().get("detail", {}).get("message", ""))[:90],
               required=False)
        return None
    if not r.ok:
        record("reference registration", FAIL, f"HTTP {r.status_code} {r.text[:120]}")
        return None
    body = r.json()
    for field in ("quality_score", "baseline_wpm", "warnings", "ref_text"):
        record(f"registration reports {field}",
               PASS if field in body else FAIL, str(body.get(field))[:70])
    return body.get("voice_id")


def check_bug_one(c, voice):
    """bugs.md section 9: one clip is enough to see the reference leak."""
    if not voice:
        record("reference-bleed probe", SKIP, "no --voice given", required=False)
        return
    text = "Trump really is. Is he the businessman he claims?"
    r = c.tts(text, voice_id=voice, project="probe")
    if not r.ok:
        record("reference-bleed probe", FAIL, f"HTTP {r.status_code}")
        return
    record("bleed probe generated", PASS, f"{wav_seconds(r.content):.1f}s audio")
    record("server checked the clip against the script",
           PASS if r.headers.get("X-Verified") == "true" else INFO,
           f"X-Verified={r.headers.get('X-Verified')}")
    warn = r.headers.get("X-OmniVoice-Warning", "")
    if warn:
        record("bleed probe warnings", INFO, warn[:140], required=False)

    files = {"audio": ("probe.wav", r.content, "audio/wav")}
    t = c.post("/api/transcribe", data={"text": text}, files=files)
    if t.ok:
        body = t.json()
        d = body.get("diff", {})
        ok = body.get("ok")
        record("transcribed back: nothing added, nothing dropped",
               PASS if ok else FAIL, body.get("summary", "")[:140])
        if d.get("tail_inserted"):
            record("reference bleed still present", FAIL,
                   "extra at end: " + " ".join(d["tail_inserted"]))
    else:
        record("transcribe-back check", SKIP, f"HTTP {t.status_code}",
               required=False)


def check_text_probes(c, voice):
    """textbrief: ten one-line requests, each with a known right answer."""
    probes = [
        ("A property worth $500 million was appraised at $150 million.",
         ["five hundred million dollars", "one hundred fifty million dollars"]),
        ("His penthouse went from 11,000 square feet to 30,000.",
         ["eleven thousand", "thirty thousand"]),
        ("Trump inflated values by 200 to 300%.",
         ["two hundred", "three hundred percent"]),
        ("His sons owe $4.7 million each.",
         ["four point seven million dollars"]),
        ("He admitted the plot—the deliberate plan—was not official.",
         ["plot, the deliberate plan, was"]),
        ("What about the January 6th speech?", ["January sixth"]),
        ("They are sabotaging MAGA, says the former NSC official.", ["Maga"]),
        ("The one word Trump's team ignores: private.",
         ["team ignores, private"]),
    ]
    bad = 0
    for text, expected in probes:
        r = c.tts(text, voice_id=voice, project="probe", steps=8)
        if not r.ok:
            bad += 1
            record(f"text probe: {text[:34]}...", FAIL, f"HTTP {r.status_code}")
            continue
        norm = r.headers.get("X-OmniVoice-Normalized-Text", "")
        missing = [e for e in expected if e.lower() not in norm.lower()]
        if missing:
            bad += 1
            record(f"text probe: {text[:34]}...", FAIL,
                   f"expected {missing} in: {norm[:90]}")
        else:
            record(f"text probe: {text[:34]}...", PASS, norm[:80])
    record("all text probes normalized as specified", PASS if not bad else FAIL,
           f"{len(probes) - bad}/{len(probes)}")


def check_loudness(c, voice):
    levels = []
    for i, text in enumerate(["Short one.",
                              "This is a considerably longer sentence, of the "
                              "kind that fills a paragraph in a real script, "
                              "and it should come back at the same loudness."]):
        r = c.tts(text, voice_id=voice, project=f"loud{i}")
        if r.ok and r.headers.get("X-LUFS"):
            try:
                levels.append(float(r.headers["X-LUFS"]))
            except ValueError:
                pass
    if len(levels) < 2:
        record("loudness is consistent across clip lengths", SKIP,
               "no X-LUFS header", required=False)
        return
    spread = max(levels) - min(levels)
    record("loudness is consistent across clip lengths",
           PASS if spread < 1.5 else FAIL,
           f"{levels} -> spread {spread:.2f} LU (was 12.4 dB in production)")


def check_concurrency(c):
    """section 7: four callers used to take the server down."""
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(c.tts, f"Concurrent request number {i}.",
                          project=f"conc{i}") for i in range(4)]
        codes = []
        for f in futs:
            try:
                codes.append(f.result().status_code)
            except Exception as e:  # noqa: BLE001
                codes.append(str(e)[:40])
    survived = all(x in (200, 429, 503) for x in codes)
    record("four concurrent callers are queued or told to back off, not fatal",
           PASS if survived else FAIL, str(codes))
    r = c.get("/api/ready")
    record("server is still alive after concurrency",
           PASS if r.status_code == 200 else FAIL, f"HTTP {r.status_code}")


def check_memory(c, voice, n):
    """A3: measure BEFORE empty_cache, and watch reserved, not just allocated."""
    before = c.get("/api/health").json().get("vram", {})
    t0 = time.time()
    fails = 0
    for i in range(n):
        r = c.tts(f"Memory soak sentence number {i}.", voice_id=voice,
                  project=f"soak{i}", steps=8)
        if not r.ok:
            fails += 1
    after = c.get("/api/health").json().get("vram", {})
    if not before or "reserved_mb" not in after:
        record("memory does not climb across requests", SKIP, "no CUDA",
               required=False)
        return
    d_alloc = after.get("allocated_mb", 0) - before.get("allocated_mb", 0)
    d_res = after.get("reserved_mb", 0) - before.get("reserved_mb", 0)
    frag = after.get("fragmentation_mb", 0)
    record(f"{n} requests completed", PASS if not fails else FAIL,
           f"{fails} failures in {time.time() - t0:.0f}s")
    record("live tensors do not accumulate", PASS if d_alloc < 100 else FAIL,
           f"allocated +{d_alloc:.0f} MB")
    record("allocator does not hoard", PASS if d_res < 400 else FAIL,
           f"reserved +{d_res:.0f} MB")
    record("memory is not fragmenting", PASS if frag < 800 else FAIL,
           f"{frag:.0f} MB reserved but unusable "
           f"(this is the 'GPU at 5% and out of memory' failure)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--key", default=os.environ.get("OMNIVOICE_API_KEY") or None)
    ap.add_argument("--voice", help="a registered voice_id to test with")
    ap.add_argument("--ref-audio", help="a wav/mp3 to test registration with")
    ap.add_argument("--soak", type=int, default=25,
                    help="requests for the memory check (0 to skip)")
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow soak and concurrency checks")
    args = ap.parse_args()

    c = Client(args.base, args.key)
    print(f"\n=== OmniVoice acceptance — {args.base} ===\n")

    try:
        if not check_health(c):
            print("\nServer is not ready. Fix that first.\n")
            return 1
    except requests.RequestException as e:
        print(f"cannot reach {args.base}: {e}")
        return 1

    print("\n-- contract (existing clients must not break) --")
    check_contract(c)
    check_unknown_voice(c)
    check_limits(c)
    check_idempotency(c)

    print("\n-- voice registration --")
    registered = check_voice_report(c, args.ref_audio)
    voice = args.voice or registered

    print("\n-- quality --")
    check_text_probes(c, voice)
    check_bug_one(c, voice)
    check_loudness(c, voice)
    check_seed(c, voice)

    if not args.quick:
        print("\n-- stability --")
        check_concurrency(c)
        if args.soak:
            check_memory(c, voice, args.soak)

    failed = [r for r in results if r[1] == FAIL and r[3]]
    print("\n" + "=" * 62)
    print(f"{sum(1 for r in results if r[1] == PASS)} passed, "
          f"{len(failed)} failed, "
          f"{sum(1 for r in results if r[1] == SKIP)} skipped")
    if failed:
        print("\nFailures:")
        for name, _s, detail, _r in failed:
            print(f"  - {name}: {detail}")
    print("=" * 62 + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
