#!/usr/bin/env python3
# =========================================================================
#  OmniVoice · OMNIVOICE_BATCH sweep on Colab T4
#  CELL 3  —  paste into a new cell BELOW the launch cell.
#
#  This cell OWNS the server: it kills the launch cell's app.py and restarts
#  it once per rung, because every knob it touches is read at import time.
#  Stop the launch cell first (it blocks); this cell restarts the server at
#  the end on the winning setting.
#
#  It answers two questions, and refuses to answer only the first:
#     1. does OMNIVOICE_BATCH>1 lower generate_s on this card?
#     2. is the audio it produces still the audio we were shipping?
#  A speed win with changed audio is reported as a LOSS.
# =========================================================================
from __future__ import annotations

import base64
import io
import json
import os
import re
import statistics as st
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave

import numpy as np

# ---- configuration ------------------------------------------------------
APP_DIR   = "/content/omnivoice-wrapper"
API_PORT  = 8001
UI_PORT   = 7860
LOG       = "/content/ov_sweep.log"
KEYFILE   = "/content/omnivoice_api_key.txt"
OUT_JSON  = "/content/batch_sweep_results.json"
BASE      = f"http://127.0.0.1:{API_PORT}"

# The ladder. 1 runs twice — first and last — so a leak or a thermal drift
# between rung 1 and rung 16 cannot be mistaken for a batching effect.
LADDER    = [1, 2, 4, 8, 16, 1]

# Fixed seeds. The FIRST is the paired-comparison seed; the rest exist only to
# build the null distribution (how much this model already varies run to run).
SEEDS     = [1234, 5678, 9012]
STEPS     = 16          # matches the production setting on this box
WARMUPS   = 2           # discarded: first generate after load pays autotune
BOOT_S    = 900

sys.path.insert(0, APP_DIR)
API_KEY = open(KEYFILE).read().strip() if os.path.exists(KEYFILE) else ""
HDR = {"X-API-Key": API_KEY}


# ---- the script ---------------------------------------------------------
# Built from the repo's own sample scripts so the register and the sentence
# shapes are the ones the product actually sees, then trimmed to land in the
# 15-17 chunk range the live Colab clips sit in.
def build_script(target_chunks=16):
    import textnorm
    pool = json.load(open(os.path.join(APP_DIR, "tools/sample_scripts.json")))
    text, n = "", 0
    for s in pool * 4:
        cand = (text + " " + s).strip()
        n = len(textnorm.chunk_text(cand, 200))
        if n > target_chunks:
            break
        text = cand
    return text, textnorm.chunk_text(text, 200)


# ---- HTTP ---------------------------------------------------------------
def api_get(path, timeout=30):
    try:
        req = urllib.request.Request(BASE + path, headers=HDR)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}
    except Exception:
        return None, {}


def _multipart(fields, files=()):
    b = b"--BOUND\r\n"
    body = b""
    for k, v in fields.items():
        if v is None:
            continue
        body += (b + f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
                 .encode())
    for k, fname, data, ctype in files:
        body += (b + f'Content-Disposition: form-data; name="{k}"; '
                 f'filename="{fname}"\r\nContent-Type: {ctype}\r\n\r\n'.encode()
                 + data + b"\r\n")
    body += b"--BOUND--\r\n"
    return body, "multipart/form-data; boundary=BOUND"


def tts_v2(text, voice_id, seed, project="sweep"):
    """/api/v2/tts: same inputs as the frozen /api/tts, JSON out.

    Carries everything the X-* headers carry (rtf, gen_sec, chunks, wpm,
    loudness, warnings) plus timing[] and info[], which the headers flatten.
    """
    body, ctype = _multipart({
        "text": text, "voice_id": voice_id, "format": "wav",
        "steps": STEPS, "seed": seed, "project": project, "inline_audio": 1,
    })
    req = urllib.request.Request(BASE + "/api/v2/tts", data=body,
                                 headers={**HDR, "Content-Type": ctype})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            j = json.loads(r.read().decode())
        j["_wall_s"] = round(time.perf_counter() - t0, 3)
        j["_http"] = 200
        return j
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        return {"_http": e.code, "_error": raw[:800],
                "_wall_s": round(time.perf_counter() - t0, 3)}


def transcribe(wav_bytes, text):
    body, ctype = _multipart({"text": text},
                             [("audio", "clip.wav", wav_bytes, "audio/wav")])
    req = urllib.request.Request(BASE + "/api/transcribe", data=body,
                                 headers={**HDR, "Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())


# ---- server lifecycle ---------------------------------------------------
# EVERY knob below is read ONCE at import into a module constant, so each rung
# needs a full restart. There is no runtime override and no API field for any
# of them:
#   OMNIVOICE_BATCH          app.py:267   <- the variable under test
#   OMNIVOICE_VERIFY         app.py:249   off, so ASR never runs during timing
#   OMNIVOICE_VERIFY_RETRIES app.py:254   0, so no chunk is regenerated at
#                                         batch 1 beside batched neighbours
#                                         (app.py:1239 hardcodes 1 there)
#   OMNIVOICE_MAX_CHARS      app.py:184   200, the shipping chunk size
#   OMNIVOICE_CHUNK_GAP      app.py:187   0.15, held constant (it lands in
#                                         audio_dur and therefore in RTF)
#   OMNIVOICE_LEVEL_MATCH    app.py:210   1, so level_corrections are computed
#                                         and reported; audio is compared
#                                         pairwise so the correction cannot
#                                         hide a difference, only report it
#   OMNIVOICE_SELFTEST       app.py:292   0, so no background clip lands in
#                                         the middle of a timed rung
#   OMNIVOICE_MAX_CONCURRENCY app.py:287  1, unchanged
# Per-request and needing NO restart: text, voice_id, steps, speed, seed,
# format, project, json/inline_audio.
FIXED_ENV = {
    "OMNIVOICE_API": "1", "OMNIVOICE_API_PORT": str(API_PORT),
    "OMNIVOICE_API_KEY": API_KEY, "OMNIVOICE_OPEN_BROWSER": "0",
    "GRADIO_SHARE": "0", "GRADIO_SERVER_NAME": "127.0.0.1",
    "GRADIO_SERVER_PORT": str(UI_PORT), "PYTHONUNBUFFERED": "1",
    "OMNIVOICE_VERIFY": "0", "OMNIVOICE_VERIFY_RETRIES": "0",
    "OMNIVOICE_MAX_CHARS": "200", "OMNIVOICE_CHUNK_GAP": "0.15",
    "OMNIVOICE_LEVEL_MATCH": "1", "OMNIVOICE_SELFTEST": "0",
    "OMNIVOICE_MAX_CONCURRENCY": "1",
}
_proc = {"p": None}


def stop_server():
    subprocess.run(["pkill", "-f", "app.py"], capture_output=True)
    for _ in range(40):
        if subprocess.run(["bash", "-c",
                           f"ss -ltn 2>/dev/null | grep -q ':{API_PORT} '"],
                          capture_output=True).returncode != 0:
            return
        time.sleep(1)


def start_server(batch):
    """Fresh process per rung. Also the only way to get an honest VRAM peak:
    torch.cuda.max_memory_allocated() is never reset anywhere in this repo, so
    peak_allocated_mb is monotone for the life of the process."""
    stop_server()
    env = dict(os.environ)
    env.update(FIXED_ENV)
    env["OMNIVOICE_BATCH"] = str(batch)
    logf = open(LOG, "w")
    _proc["p"] = subprocess.Popen([sys.executable, "app.py"], cwd=APP_DIR,
                                  env=env, stdout=logf, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < BOOT_S:
        if _proc["p"].poll() is not None:
            print(open(LOG).read()[-4000:])
            raise SystemExit(f"app.py exited ({_proc['p'].returncode}) at BATCH={batch}")
        if api_get("/api/live", timeout=5)[0] == 200:
            return
        time.sleep(3)
    raise SystemExit(f"timed out booting BATCH={batch}")


def log_text():
    try:
        return open(LOG, errors="replace").read()
    except Exception:
        return ""


# ---- OOM / masking detectors -------------------------------------------
# _generate halves the batch on OOM (app.py:1203) and the request still
# SUCCEEDS. Nothing in the response says so: the status note appends
# f"batch {BATCH}" from the module constant (app.py:1275), i.e. the CONFIGURED
# batch, never the one actually used. Three independent detectors instead —
# and a quieter fourth case they have to cover: gpu_guard.guarded retries the
# same oversized call once (gpu_guard.py:329-371) and if THAT succeeds,
# _generate never sees the exception and never halves, so no log line is ever
# written and only the counters move.
HALVE_RE = re.compile(r"CUDA OOM -> reducing batch size to (\d+)")


def oom_probe():
    _, h = api_get("/api/health")
    _, m = api_get("/api/metrics")
    v = h.get("vram", {}) or {}
    return {
        "oom_total": h.get("oom_total"),          # gpu_guard, BROAD shapes
        "metrics_oom": (m or {}).get("oom"),      # app counter, is_oom()
        "model_reloads": h.get("model_reloads"),
        "consecutive_failures": h.get("consecutive_failures"),
        "generation_ok": h.get("generation_ok"),
        "vram_refusals": (m or {}).get("vram_refusals"),
        "peak_allocated_mb": v.get("peak_allocated_mb"),
        "reserved_mb": v.get("reserved_mb"),
        "free_mb": v.get("free_mb"),
        "total_mb": v.get("total_mb"),
        "fragmentation_mb": v.get("fragmentation_mb"),
        "spilled": (m or {}).get("spilled_to_shared"),
    }


# ---- audio ---------------------------------------------------------------
def decode_wav(b64):
    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw), "rb") as w:
        sr, n, sw = w.getframerate(), w.getnframes(), w.getsampwidth()
        pcm = w.readframes(n)
    assert sw == 2, f"expected int16, got {sw*8}-bit"
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return x, sr, raw


def split_chunks(x, sr, gap_sec=0.15):
    """Segment the joined clip back into per-chunk audio.

    join_chunks inserts EXACT zeros between chunks (audio_fx.py:710-717) and
    exact zeros for the tail pad; loudness is a scalar gain and to_int16 is a
    scale, so the zeros survive to the delivered WAV bit-for-bit. A run of
    exact zeros of gap length is therefore a chunk boundary and essentially
    nothing else — model output is float noise, not digital silence.
    """
    zero = x == 0.0
    runs, i, n = [], 0, len(x)
    while i < n:
        if zero[i]:
            j = i
            while j < n and zero[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    minlen = int(0.80 * gap_sec * sr)
    cuts = [r for r in runs if (r[1] - r[0]) >= minlen]
    segs, prev = [], 0
    for a, b in cuts:
        if a > prev:
            segs.append(x[prev:a])
        prev = b
    if prev < n:
        segs.append(x[prev:n])
    return [s for s in segs if len(s) > int(0.05 * sr)]


def _mel_fb(sr, nfft, nmel=40):
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(60), hz2mel(sr / 2 * 0.95), nmel + 2))
    bins = np.floor((nfft + 1) * pts / sr).astype(int)
    fb = np.zeros((nmel, nfft // 2 + 1), np.float32)
    for m in range(1, nmel + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c == l:
            c = l + 1
        if r == c:
            r = c + 1
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


_FB = {}


def timbre_vec(seg, sr, nfft=1024, hop=256):
    """80-d spectral fingerprint: per-band mean AND spread of log-mel energy.

    Deliberately NOT an absolute quality score. It is only ever read as a
    PAIRED distance between two clips of the SAME text in the SAME voice, and
    it is only ever judged against the distance this model already produces
    between two seeds. Content is held constant, so what is left is timbre and
    delivery.
    """
    if len(seg) < nfft:
        return None
    if sr not in _FB:
        _FB[sr] = _mel_fb(sr, nfft)
    win = np.hanning(nfft).astype(np.float32)
    frames = 1 + (len(seg) - nfft) // hop
    S = np.empty((frames, nfft // 2 + 1), np.float32)
    for i in range(frames):
        S[i] = np.abs(np.fft.rfft(seg[i * hop:i * hop + nfft] * win))
    M = np.log(np.maximum(S @ _FB[sr].T, 1e-10))
    keep = M.mean(1) > (M.mean(1).max() - 4.0)      # voiced frames only
    if keep.sum() < 4:
        keep = np.ones(frames, bool)
    V = np.concatenate([M[keep].mean(0), M[keep].std(0)])
    return V / (np.linalg.norm(V) + 1e-9)


def cosd(a, b):
    if a is None or b is None:
        return None
    return float(1.0 - np.dot(a, b))


def clip_metrics(j):
    """Everything the response already knows, flattened."""
    t = j.get("timing", {}) or {}
    ch = j.get("chunks", {}) or {}
    info = j.get("info", "") or ""
    m = re.search(r"levelled (\d+) chunk\(s\) \(max ([\d.]+) dB\)", info)
    return {
        "gen_s": t.get("generate_s"), "total_s": t.get("total_s"),
        "join_s": t.get("join_s"), "norm_s": t.get("normalize_s"),
        "rtf": j.get("rtf"), "dur": j.get("duration_sec"), "wpm": j.get("wpm"),
        "baseline_wpm": j.get("baseline_wpm"),
        "chunks": ch.get("total"),
        "lufs": (j.get("loudness") or {}).get("out_lufs"),
        "peak_db": (j.get("loudness") or {}).get("true_peak_db"),
        "met_target": (j.get("loudness") or {}).get("met_target"),
        "levelled_n": int(m.group(1)) if m else 0,
        "levelled_max_db": float(m.group(2)) if m else 0.0,
        "warnings": j.get("warnings") or [],
        "info": info,
        "wall_s": j.get("_wall_s"),
    }


# =========================================================================
#  RUN
# =========================================================================
def main():
    print("=" * 74)
    print("  OMNIVOICE_BATCH sweep · T4 · VERIFY off · steps", STEPS)
    print("=" * 74)

    # ---- phase 0: what are we actually testing -------------------------
    code, voices = api_get("/api/voices")
    if code != 200 or not voices:
        raise SystemExit(
            "Register a saved voice first (POST /api/voices). A saved voice is\n"
            "required: design mode forces chunk 0 to batch 1 (app.py:1220), so\n"
            "it would test a different code path than the one that ships.")
    VOICE = (voices[0] if isinstance(voices[0], str)
             else voices[0].get("voice_id") or voices[0].get("id"))
    SCRIPT, CHUNKS = build_script()
    L = [len(c) for c in CHUNKS]
    print(f"voice        : {VOICE}")
    print(f"script       : {len(SCRIPT)} chars, {len(SCRIPT.split())} words")
    print(f"chunks       : {len(CHUNKS)}  len {min(L)}-{max(L)} "
          f"(sd {st.pstdev(L):.0f})")
    for bs in LADDER:
        if bs == 1:
            continue
        pad = sum(max(L[i:i + bs]) * len(L[i:i + bs]) - sum(L[i:i + bs])
                  for i in range(0, len(L), bs))
        print(f"  bs={bs:<2} padded-compute overhead in document order: "
              f"{100 * pad / sum(L):.1f} %   <- what has to be beaten")
    if not (12 <= len(CHUNKS) <= 20):
        print("  WARNING: chunk count is outside the 15-17 range of the live clips")

    results, ref = {}, {}
    seg_ref = {}

    for rung, batch in enumerate(LADDER):
        tag = f"B{batch}" + ("*" if rung == len(LADDER) - 1 else "")
        print("\n" + "-" * 74)
        print(f"RUNG {tag}  (restart: OMNIVOICE_BATCH is read at app.py:267)")
        print("-" * 74)
        start_server(batch)

        banner = log_text()
        mb = re.search(r"concurrency=(\d+)", banner)
        print(f"  booted · banner concurrency={mb.group(1) if mb else '?'} · "
              f"verify={'on' if 'verify=on' in banner else 'OFF'}")

        before = oom_probe()
        # warm-ups: first generate after load pays kernel autotune, and the
        # first request for a voice_id builds and caches its clone prompt.
        for _ in range(WARMUPS):
            tts_v2(SCRIPT[:300], VOICE, seed=1)

        rows, fatal = [], None
        for seed in SEEDS:
            j = tts_v2(SCRIPT, VOICE, seed=seed, project=f"{tag}_s{seed}")
            if j.get("_http") != 200:
                fatal = f"HTTP {j['_http']}: {j.get('_error','')[:300]}"
                print(f"  seed {seed}: FAILED · {fatal}")
                break
            cm = clip_metrics(j)
            x, sr, raw = decode_wav(j["audio_base64"])
            segs = split_chunks(x, sr)
            cm["n_segments"] = len(segs)
            cm["finite"] = bool(np.isfinite(x).all())
            cm["seg_rms_db"] = [round(float(20 * np.log10(
                max(float(np.sqrt(np.mean(s.astype(np.float64) ** 2))), 1e-9))), 1)
                for s in segs]
            cm["dead_segments"] = sum(1 for d in cm["seg_rms_db"] if d < -60.0)
            # ASR word diff, using the Whisper already resident here
            tr = transcribe(raw, SCRIPT)
            d = tr.get("diff", {}) or {}
            cm["asr_ok"] = bool(tr.get("ok"))
            cm["word_accuracy"] = d.get("word_accuracy")
            cm["dropped"] = len(d.get("missing") or [])
            cm["added"] = len(d.get("extra") or [])
            cm["asr_summary"] = tr.get("summary")
            cm["seed"] = seed
            vecs = [timbre_vec(s, sr) for s in segs]
            rows.append(cm)
            key = (batch, seed, rung)
            seg_ref[key] = vecs
            if batch == 1 and rung == 0:
                ref[seed] = (vecs, cm)
            print(f"  seed {seed:<5} gen {cm['gen_s']:.2f}s  dur {cm['dur']:.1f}s"
                  f"  RTF {cm['rtf']:.3f}  chunks {cm['chunks']}"
                  f"  segs {cm['n_segments']}  wpm {cm['wpm']:.0f}"
                  f"  +{cm['added']}/-{cm['dropped']}"
                  f"  acc {cm['word_accuracy']}")

        after = oom_probe()
        halved = HALVE_RE.findall(log_text())
        results[tag] = {
            "batch": batch, "rung": rung, "rows": rows, "fatal": fatal,
            "vram_before": before, "vram_after": after,
            "halve_log_lines": halved,
            "oom_delta": (after.get("oom_total") or 0) - (before.get("oom_total") or 0),
            "metrics_oom_delta": ((after.get("metrics_oom") or 0)
                                  - (before.get("metrics_oom") or 0)),
            "reload_delta": ((after.get("model_reloads") or 0)
                             - (before.get("model_reloads") or 0)),
            "refusal_delta": ((after.get("vram_refusals") or 0)
                              - (before.get("vram_refusals") or 0)),
        }
        r = results[tag]
        print(f"  VRAM peak {after.get('peak_allocated_mb')} MB of "
              f"{after.get('total_mb')} MB · reserved {after.get('reserved_mb')} MB"
              f" · frag {after.get('fragmentation_mb')} MB")
        print(f"  OOM: gpu_guard +{r['oom_delta']}  metrics +{r['metrics_oom_delta']}"
              f"  reloads +{r['reload_delta']}  refusals +{r['refusal_delta']}"
              f"  halve-log {halved or 'none'}")
        if r["oom_delta"] or r["metrics_oom_delta"] or halved:
            print("  !! MASKED: this rung did NOT actually run at batch "
                  f"{batch} throughout. The response would still have said "
                  f"'batch {batch}' (app.py:1275 reports the module constant).")
        if fatal:
            print("  stopping the ladder: a rung that dies makes every higher "
                  "rung meaningless")
            break

    # ---- analysis --------------------------------------------------------
    print("\n" + "=" * 74)
    print("  RESULTS")
    print("=" * 74)

    # null distribution: how far apart are two BATCH=1 clips at different seeds?
    null_d, null_wpm, null_dur = [], [], []
    base = results.get("B1", {})
    b1rows = {r["seed"]: r for r in base.get("rows", [])}
    seeds_ok = [s for s in SEEDS if s in ref]
    for i, a in enumerate(seeds_ok):
        for b in seeds_ok[i + 1:]:
            va, vb = ref[a][0], ref[b][0]
            if len(va) == len(vb):
                null_d += [d for d in (cosd(p, q) for p, q in zip(va, vb))
                           if d is not None]
            null_wpm.append(abs(ref[a][1]["wpm"] - ref[b][1]["wpm"]))
            null_dur.append(abs(ref[a][1]["dur"] - ref[b][1]["dur"]))
    if not null_d:
        print("NULL DISTRIBUTION EMPTY — cannot judge audio change. Stop here.")
        return
    nq95, nmax = float(np.percentile(null_d, 95)), float(max(null_d))
    print(f"null (seed-to-seed, BATCH=1): n={len(null_d)} chunk pairs · "
          f"p95 {nq95:.4f} · max {nmax:.4f} · wpm spread "
          f"{max(null_wpm):.1f} · dur spread {max(null_dur):.2f}s")
    print("  ^ this is the variation the product ALREADY ships between two "
          "runs.\n    Batching is allowed to be different; it is not allowed "
          "to be MORE\n    different than this.")

    print(f"\n{'rung':<6}{'gen_s':>9}{'vs B1':>9}{'RTF':>8}{'peakMB':>9}"
          f"{'+w':>4}{'-w':>4}{'d_p95':>9}{'d_max':>9}{'verdict':>10}")
    b1gen = st.median([r["gen_s"] for r in base.get("rows", [])]) if base.get("rows") else None
    verdicts = {}
    for tag, r in results.items():
        rows = r["rows"]
        if not rows:
            print(f"{tag:<6}{'FAILED':>9}   {r['fatal']}")
            verdicts[tag] = "FAIL(dead)"
            continue
        gen = st.median([x["gen_s"] for x in rows])
        speed = (b1gen / gen) if b1gen else float("nan")
        dmax = dq95 = None
        ds = []
        for x in rows:
            va = ref.get(x["seed"], (None,))[0]
            vb = seg_ref.get((r["batch"], x["seed"], r["rung"]))
            if va and vb and len(va) == len(vb):
                ds += [d for d in (cosd(p, q) for p, q in zip(va, vb))
                       if d is not None]
        if ds:
            dq95, dmax = float(np.percentile(ds, 95)), float(max(ds))
        added = sum(x["added"] for x in rows)
        dropped = sum(x["dropped"] for x in rows)

        fails = []
        if r["oom_delta"] or r["metrics_oom_delta"] or r["halve_log_lines"]:
            fails.append("oom/masked")
        if r["reload_delta"]:
            fails.append("model-reload")
        if not r["vram_after"].get("generation_ok", True):
            fails.append("unhealthy")
        peak = r["vram_after"].get("peak_allocated_mb") or 0
        total = r["vram_after"].get("total_mb") or 1
        if peak > 0.60 * total:
            fails.append(f"vram {peak:.0f}/{total:.0f}")
        if added or dropped:
            fails.append(f"words +{added}/-{dropped}")
        if any(x["dead_segments"] for x in rows):
            fails.append("dead-chunk")
        if any(not x["finite"] for x in rows):
            fails.append("nonfinite")
        if any(x["n_segments"] != x["chunks"] for x in rows):
            fails.append("segment-count")
        if any(x["word_accuracy"] is not None and x["word_accuracy"] < 0.995
               for x in rows):
            fails.append("accuracy")
        if r["batch"] > 1:
            if dq95 is None:
                fails.append("no-audio-comparison")
            else:
                if dq95 > nq95:
                    fails.append(f"timbre p95 {dq95:.4f}>{nq95:.4f}")
                if dmax > nmax:
                    fails.append(f"timbre max {dmax:.4f}>{nmax:.4f}")
            wsps = [abs(x["wpm"] - b1rows[x["seed"]]["wpm"])
                    for x in rows if x["seed"] in b1rows]
            wsp = max(wsps) if wsps else 0.0
            if null_wpm and wsp > max(null_wpm):
                fails.append(f"wpm {wsp:.1f}>{max(null_wpm):.1f}")
            lv = max(x["levelled_max_db"] for x in rows)
            lv1 = (max(x["levelled_max_db"] for x in base["rows"])
                   if base.get("rows") else 0.0)
            if lv > lv1 + 1.5:
                fails.append(f"level {lv:.1f}dB>{lv1:.1f}dB")
            if speed < 1.05:
                fails.append(f"no win ({speed:.2f}x)")

        v = "ADOPT" if not fails else "REJECT"
        if r["batch"] == 1:
            v = "baseline"
        verdicts[tag] = v if not fails else "REJECT: " + ", ".join(fails)
        print(f"{tag:<6}{gen:>9.2f}{speed:>8.2f}x"
              f"{st.median([x['rtf'] for x in rows]):>8.3f}{peak:>9.0f}"
              f"{added:>4}{dropped:>4}"
              f"{(dq95 if dq95 is not None else float('nan')):>9.4f}"
              f"{(dmax if dmax is not None else float('nan')):>9.4f}"
              f"{v.split(':')[0]:>10}")

    print("\nverdicts")
    for tag, v in verdicts.items():
        print(f"  {tag:<6} {v}")

    b1a = results.get("B1", {}).get("rows")
    b1b = [r for t, r in results.items() if t == "B1*"]
    if b1a and b1b and b1b[0]["rows"]:
        ga = st.median([x["gen_s"] for x in b1a])
        gb = st.median([x["gen_s"] for x in b1b[0]["rows"]])
        drift = abs(gb - ga) / ga
        print(f"\nA-B-A drift check: BATCH=1 first {ga:.2f}s, last {gb:.2f}s "
              f"({100*drift:.1f} %)"
              + ("  <- any 'win' smaller than this is noise"
                 if drift > 0.03 else "  <- box is stable"))

    json.dump({"script": SCRIPT, "chunk_lens": L, "voice": VOICE,
               "null_p95": nq95, "null_max": nmax,
               "results": {k: {kk: vv for kk, vv in v.items()}
                           for k, v in results.items()},
               "verdicts": verdicts}, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"\nfull record: {OUT_JSON}")
    print("\nWhen you are done, restart the server on the setting you adopted "
          "and re-run the launch cell.")


if __name__ == "__main__":
    main()
