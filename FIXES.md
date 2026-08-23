# What changed, and how to check it on the Windows box

Written against four documents: `omnivoicebugs.md`, `omnivoicetextbrief.md`,
`omnivoiceoombrief.md`, and `OMNIVOICE_HARNESS_CORRECTIONS.md`. Where the
corrections file disagreed with the others, it won — that was its instruction.

Nothing here has been run against a GPU. The pure-Python half (text handling,
audio repair, the verifier) has **58 unit tests that pass**, and there is an
end-to-end simulation below reproducing the original failure and its fix. The
model-facing half compiles and is reviewed, but needs the Windows machine.

---

## New files

| File | What it is | Testable without a GPU |
|---|---|---|
| `textnorm.py` | the wrapper's own text front-end | ✅ 24 tests |
| `audio_fx.py` | reference repair, BS.1770 loudness, output repair | ✅ 22 tests |
| `verify.py` | transcribe-back word diff | ✅ 12 tests |
| `gpu_guard.py` | OOM handling, VRAM snapshots, readiness | needs torch |
| `lexicon.json` | pronunciation overrides you own and edit | — |
| `tests/` | the 58 tests | ✅ |
| `tools/acceptance.py` | run against a live server before selling | GPU |
| `tools/audit_batch.py` | re-run the audit that found the bugs | GPU |
| `docs/reference_playbook.md` | hand this to customers | — |
| `run_tests.bat` | double-click the unit tests | ✅ |

`app.py` went from 1,737 to ~2,600 lines: same UI, rewired generation core, and
a substantially larger API.

---

## The root cause of the worst bug

`RVoiceover_3.mp3` says **"forcing"** at 9.8 s. `MAX_REF_SEC` was **10**. The old
`trim_reference()` did `sf.read(path, frames=int(MAX_REF_SEC * sr))` — a hard cut
at exactly 10.000 s, landing inside that word. The model treated the half-spoken
word as unfinished business and said it at the end of 86–90 % of that voice's
clips: 163 times in one batch, ~1.32 s each.

Reproduced and fixed, with the real modules:

```
reference: 61.1s
OLD  hard cut  -> ends_abruptly=True   trailing silence=0.000s   <- mid-word
NEW  smart cut -> 8.72s at_pause=True  trailing silence=0.320s

verifier: accuracy=1.0  passed=False  tail=['forcing']  repairable=True
repair:   removed 1.22s of leaked audio
loudness: -20.6 / -30.2 / -39.7 LUFS  ->  all three at -20.0 LUFS
```

Note `accuracy=1.0` with `passed=False`: everything sent *was* said, plus a word
that wasn't. That is why accuracy can never be the only gate.

---

## Item by item

### `omnivoicebugs.md`

| § | Finding | What was done |
|---|---|---|
| 1 | 218 words added; reference bleed | `audio_fx.smart_trim_reference` handles **both** ways a reference ends mid-word — one we cause (cut at the last pause before the limit, and prefer a shorter clip over a mid-word one) and one the customer causes (they stopped recording mid-word, so cut back to their last finished phrase; refused only when under 3 s would be left, and then explained). It cuts at the last pause before the limit, strips silence, fades, appends 0.3 s. `POST /api/voices` already accepted `ref_text` — it now also **validates it against the audio** and returns the transcript actually used. A reference with no pause anywhere warns at upload. Anything that still leaks is caught by the verifier and trimmed. |
| 2 | 12 words dropped, silently | `verify.word_diff` on every chunk; a drop triggers a regeneration; whatever survives is reported in `X-OmniVoice-Warning`, the v2 JSON, the job status and the UI card. |
| 3 | Loudness tracked the reference | BS.1770-4 K-weighted loudness + 4× true peak. References normalized at registration, outputs at delivery: −20 LUFS, −1 dBTP. Measured level returned in `X-LUFS` / `X-True-Peak-dB`. |
| 4 | CUDA OOM, never recovers | `inference_mode`, results to CPU immediately, named `del` → `gc.collect()` → `empty_cache()`, `expandable_segments:True` set before torch imports, concurrency semaphore, OOM retry, optional model reload with voice-prompt invalidation. |
| 5 | `/api/health` said "ok" while dead | `/api/live`, `/api/ready` (cached self-test), `/api/selftest`, and `/api/health` reporting allocated **and** reserved VRAM. `/api/ready` returns 503 + `Retry-After`. |
| 6 | 133–203 wpm | Balanced chunking (no more 100-char chunk beside a 6-char one), wpm measured per clip, compared to the voice's own baseline. Metric, not a gate. |
| 7 | 4 callers killed it; unknown params accepted; error content type | Semaphore + 429 + `Retry-After`; unknown params reported in a header and counted in `/api/metrics`; errors are JSON with proper status codes. |
| 8 | "do not fix these" | Respected. Numbers/currency/ordinals now expanded **explicitly** (same spoken result, no longer relying on the model seeing a stray `$`). Ordinary acronyms untouched — only pronounceable ones (MAGA, NASA) are in the lexicon. |

Two real bugs were found in §8's "already correct" area, both silent:

* `"1st"` → `"onest"`, `"2nd"` → `"twond"`. The old code substituted the digit
  *inside* the token, so only `-th` ordinals came out right by luck.
* One curly apostrophe or em dash disabled **all** number normalization for the
  whole script (the language guard rejected any codepoint > 0x2FF, and `’` is
  one). 26 of 63 real segments contained an em dash.

### `omnivoicetextbrief.md`

Em dashes → comma pause · numbers/currency/percent/ordinals/dates expanded ·
`U.S.` and `Jr.` no longer split a sentence · curly quotes folded, `"` silenced ·
colons → pause (clock times survive and are spoken) · unknown characters dropped
and logged, never fatal · normalized text returned in
`X-OmniVoice-Normalized-Text`. Acronyms deliberately **not** blanket-changed, per
`bugs.md` §8 — the lexicon covers the exceptions.

### `omnivoiceoombrief.md`

All eight items, in the order given. Item 5 (bound the input) is one limit in one
place — `OMNIVOICE_MAX_INPUT_CHARS` / `_WORDS` → `413` — and long-form work goes
through chunking rather than being blocked, per corrections **D8**.

### `OMNIVOICE_HARNESS_CORRECTIONS.md`

| | | |
|---|---|---|
| **A1** | word-diff must align | `verify.word_diff` uses `difflib`, with `dropped` from delete+replace and `inserted` from insert+replace, exactly as specified. A test asserts the old membership check would have passed the `rep-1` case and this one does not. |
| **A2** | `/api/ready` must really generate | Background self-test every 120 s, skipped while real work holds the slot; result cached; 503 when failed or stale. `/api/live` is separate. |
| **A3** | leak test must see fragmentation | `gpu_guard.snapshot()` reports allocated, reserved, peak and `fragmentation_mb`; `/api/health` exposes them; `tools/acceptance.py --soak` asserts on all three. |
| **A4** | `del locals()[name]` does nothing | Named `del` in a `finally`, then `gc.collect()`, then `empty_cache()`. |
| **B** | contract freeze | `POST /api/tts` unchanged: multipart in, bytes out, `X-Duration-Sec` + `X-RTF`. `project` and `language="English"` still accepted. New JSON contract on `/api/v2/tts`. Unknown params warn (`X-OmniVoice-Warning`) and are counted; `OMNIVOICE_STRICT_PARAMS=1` turns that into 400 when you are ready. `X-OmniVoice-API-Version` added. |
| **C1** | WPM is a metric | Per-voice `baseline_wpm` measured at registration from `ref_text` + duration; band 0.75–1.30 of it; warning only. No baseline → 50/280 catastrophe band. |
| **C2** | truncation detector was inverted | `audio_fx.ends_abruptly` returns True on a **loud** tail. Warning only; `dropped` is the real answer. A test asserts the direction. |
| **D1** | seed determinism | `seed` accepted on every synthesis endpoint and applied to torch/numpy/random. `tools/acceptance.py` reports whether it actually controls output — and says so plainly if OmniVoice ignores it, so the golden set can be planned accordingly. |
| **D2** | voices survive restart | `voices/index.json` written atomically, now carrying `baseline_wpm`, `quality_score`, `warnings`, `owner`, `duration_sec`, `lufs`, `snr_db`. Missing files are skipped with a log line. |
| **D3** | unknown `voice_id` → 404 | Enforced in the API and the UI. Never a substitute. |
| **D4** | idempotency | `Idempotency-Key` header; replay returns cached audio with `X-OmniVoice-Idempotent-Replay: true`; 409 while in flight; capped, TTL'd cache. |
| **D5** | `Retry-After` | On every 429, 503 and 409. |
| **D6** | verifier budget | `OMNIVOICE_VERIFY_BUDGET` (45 s). Over budget the audio ships with `X-Verified: false` and a warning. Counted in `/api/metrics`. |
| **D7** | wrapper's own normalizer | `textnorm.py`, 24 tests, not OmniVoice's. |
| **D8** | one input limit | `OMNIVOICE_MAX_INPUT_CHARS=8000`, `_WORDS=1300`, read by the UI and every endpoint. |
| **E1** | reference validation | Duration, SNR, clipping, trailing silence, LUFS, quality score, and the **`ref_text` vs audio check** (< 90 % → 422). |
| **E2** | tell the customer | `POST /api/voices` returns `accepted`, `quality_score`, `baseline_wpm`, `warnings`, `hint`; `docs/reference_playbook.md` written first, not last. |
| **E3** | tenant isolation | `OMNIVOICE_API_KEYS=k1:acme,k2:globex`. Voices carry an owner; another tenant's voice is **404**, not 403. |

---

## Two places I did not follow the note

1. **The 0.75 WPM band.** The corrections file specifies `0.75 * baseline` and
   then offers "a 210 wpm voice that drifted to 175" as the case it catches — but
   175/210 is 0.83, inside 0.75. Tightening to 0.85 would fire constantly: in the
   measured batch one voice ranged 160–203 wpm (±12 %) on ordinary clips. The
   written numbers stayed, with `OMNIVOICE_RATE_LOW`/`_HIGH` to change them. A
   17 % drift is treated as normal variation, not a warning nobody would trust.

2. **ASR on CPU.** D6 asks for the verifier's Whisper on CPU so it cannot compete
   for VRAM. Here Whisper is already resident on the GPU (`load_asr=True`) and
   moving it to CPU would make `whisper-large-v3-turbo` far too slow to run on
   every clip. Instead it runs **under the same lock as generation**, so the two
   never overlap and there is no additional concurrent peak — which is the risk
   D6 is actually guarding against. **Watch VRAM on the first long run**; if it
   is tight, `OMNIVOICE_VERIFY=0` disables checking, or point
   `OMNIVOICE_ASR_MODEL` at a smaller Whisper.

---

## Added after a research pass (see [RESEARCH.md](RESEARCH.md))

Six gaps that our own batch could not have shown, four fixed:

1. **Chunk stitching** — even edges, fixed gap, edge fades, and level matching
   to the clip's own median. The literature is unanimous that the seam is
   audible in most long-form output, and that per-chunk normalization to a fixed
   target (XTTS's approach) makes it worse.
2. **Tail padding + a stricter edge threshold** — silence stripping clipping a
   final consonant is a documented cause of lost words, and a live candidate for
   one of our own twelve drops.
3. **Voice prompt pre-warming** — `OMNIVOICE_PREWARM_VOICES=1`.
4. **Consent, watermarking and an audit trail** — EU AI Act Article 50 has been
   enforceable since **2 August 2026**.

Still open, deliberately: **two-speaker detection in a reference** (needs a
diarization model), **speaker-similarity measurement**, and the verifier's real
cost on hardware.

## What is NOT fixed

Being straight about this, because "everything is fixed" is never true of a
stochastic model:

| | |
|---|---|
| **Two speakers in one reference** | not implemented — needs a diarization model. A two-person clip registers happily and clones a blend. The `ref_text`-vs-audio check catches many indirectly, but that is a side effect, not a guarantee |
| **Speaker similarity** | nothing measures whether the output still *sounds like* the customer. That needs a speaker-embedding model and a threshold measured on your own voices |
| **Cross-lingual timbre** | upstream measures similarity collapsing to 0.22 on cross-lingual synthesis. Do not sell "clone in any language" until you have tested it |
| **The model is stochastic** | it will still occasionally drop a word or mispronounce a name. What changed is that it can no longer do so *silently* |
| **Nothing GPU-side has run** | every claim about VRAM, RTF and watermarking is reasoning until Windows says otherwise |

## What to do on Windows, in order

```powershell
git pull
REM 0. Upgrade omnivoice 0.1.5 -> 0.2.1. The app warns loudly if you skip this.
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m pip install pytest requests

REM 1. No GPU needed. Should be 58 passed.
venv\Scripts\python -m pytest tests -q

REM 2. Start the app and watch the banner line:
REM    verify=on · normalize=full · loudness=on (-20 LUFS) · concurrency=1
run.bat

REM 3. Re-register your three voices — the stored clips are repaired on the way in.
curl -X POST http://127.0.0.1:8001/api/voices -F "name=RVoiceover_3" -F "voice=@RVoiceover_3.mp3"
REM    Read "warnings" in the reply. RVoiceover_3 is the one that ended on "and".

REM 4. The probe from the bug report. If "forcing" is gone, bug 1 is fixed.
venv\Scripts\python tools\acceptance.py --voice RVoiceover_3

REM 5. Re-run your real batch, then audit it the same way you found the bugs.
venv\Scripts\python tools\audit_batch.py manifest.json --out audit.json

REM 6. Before taking public money: turn on provenance.
venv\Scripts\python -m pip install audioseal
venv\Scripts\python -c "import watermark; print(watermark.self_check())"
REM    then set OMNIVOICE_WATERMARK=1, OMNIVOICE_REQUIRE_CONSENT=1 in run.bat
```

Step 5 is the one that answers the question. Target: **0 words added, 0 words
dropped**, where the same 63 clips previously showed 218 added and 12 dropped.

### The one experiment worth running for RTF

`omnivoice` PR #239 reports a **2-2.9x lossless speedup** (2.1x at batch size 1,
which is your case) from FlashInfer. It was merged *after* the 0.2.1 release, so
it needs git main:

```powershell
venv\Scripts\python -m pip install "omnivoice @ git+https://github.com/k2-fsa/OmniVoice"
venv\Scripts\python -m pip install flashinfer-python==0.6.15.post1 ^
    "flashinfer-jit-cache==0.6.15.post1+cu128" ^
    --extra-index-url https://flashinfer.ai/whl/cu128/
set OMNIVOICE_FLASHINFER=1
```

The startup banner says whether the flag was accepted. It was measured on an
H100; your 3060 Ti is Ampere (sm_86) and untested, so run
`tools\acceptance.py` and `tools\audit_batch.py` before trusting it — "lossless"
is the author's claim, not a measurement on your hardware.

### What to watch on the first long run

| Watch | Where | Why |
|---|---|---|
| VRAM `fragmentation_mb` | `GET /api/health` | the only number that predicted the old OOM |
| RTF | `X-RTF`, job cards | verification adds one Whisper pass per chunk, roughly 15–20 % |
| `verify_skipped` | `GET /api/metrics` | above ~5 % means the budget is too tight |
| `X-OmniVoice-Warning` | every response | the whole point; empty is what you want |
| `regenerations` | `GET /api/metrics` | how often the model needed a second attempt |

### If something regresses

| Symptom | Escape hatch |
|---|---|
| Slower than before | `OMNIVOICE_VERIFY=0` |
| A number is read wrong | `OMNIVOICE_NORMALIZE_LEVEL=basic` (or `off`) |
| Clips too loud/quiet for your edit | `OMNIVOICE_OUT_LUFS=-16` (or `NORMALIZE_OUTPUT=0`) |
| A good reference gets rejected | `OMNIVOICE_STRICT_REF=0` |
| A client breaks on a 4xx | check `/api/metrics` → `unknown_params` first |

Every one of these is a single environment variable, so a regression is a
restart away from being ruled out rather than a rollback.
