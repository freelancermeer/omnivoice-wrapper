# 🔌 OmniVoice Local API — Design & Spec

A small **HTTP API** so any machine on your LAN (or scripts on this PC) can send
**text + a reference voice** and get back an **audio file (mp3/wav)** — using the
same engine and default settings as the Voiceover Studio UI.

> ✅ **LIVE** — implemented in `app.py`. It runs in the same process as
> the UI on its **own port `8001`** (UI stays on `7860`), sharing the loaded model,
> GPU lock, and job queue. Interactive docs: **`http://<pc-ip>:8001/api/docs`**.
>
> **Base URL:** `http://<pc-ip>:8001` (not 7860). Change with `OMNIVOICE_API_PORT`;
> disable the API with `OMNIVOICE_API=0`. Auth is **open** by default — set
> `OMNIVOICE_API_KEY=<key>` to require an `X-API-Key` header.

---

## 1. The big picture

```
┌────────────┐   HTTP (LAN)    ┌─────────────────────────────┐
│ Any device │ ──────────────► │  OmniVoice PC (RTX 3060 Ti) │
│ curl / app │                 │  ┌───────────────────────┐  │
│ n8n / code │ ◄────────────── │  │ FastAPI  +  Gradio UI │  │
└────────────┘   mp3 / json    │  │  (one process, 1 GPU) │  │
                               │  └───────────────────────┘  │
                               └─────────────────────────────┘
```

Same box already runs the Gradio app. We add REST endpoints **in the same process**
so they share the already-loaded model (no second 3 GB load) and the same single-GPU
queue/lock.

---

## 2. Design decisions (brainstorm → recommendation)

### 2.1 How to serve it
| Option | Notes |
|---|---|
| **A. Mount FastAPI into the Gradio app** ✅ | Gradio already runs on FastAPI. We attach our routes to the same server → **one process, one port (7860), shared model**. `curl http://<pc>:7860/api/...`. Simplest to run & deploy. |
| B. Separate FastAPI on port 8000 | Cleaner separation, but a **second process = second model load = +3 GB VRAM** (won't fit well on 8 GB). ❌ |
| C. Just use Gradio's built-in `/gradio_api` | Works via `gradio_client` (Python), but ugly for curl / other languages. Fine as a fallback, not a clean public API. |

**Recommendation:** **A** — mount our own clean `/api/*` routes on the existing server.

### 2.2 Synchronous vs job-based
| Mode | Flow | Best for |
|---|---|---|
| **Sync** ✅ | `POST /api/tts` → **waits** → returns the audio bytes | short/medium scripts, simple clients ("send text, get mp3") |
| **Async (job)** ✅ | `POST /api/tts/async` → `{job_id}` → poll `GET /api/jobs/{id}` → `GET /api/jobs/{id}/download` | long scripts, batches, fire-and-forget; plugs into the **existing queue** |

**Recommendation:** ship **both**. Sync covers the "brain-simple" case; async reuses the
queue you already built (and shares the Studio's queue board).

### 2.3 Voice input — three ways (support all)
1. **Upload per request** (multipart file field `voice`) — self-contained, simplest. ✅
2. **Saved voice by id** (`voice_id`) — register once via `POST /api/voices` (or from the
   **UI's voice library**), then just send `voice_id` on every call. ✅
3. **No voice** → designed/AI voice (leave `voice` empty). ✅

> **Permanent library (shared UI ↔ API).** Registered voices are saved to `voices/`
> with an `index.json`, so they **survive restarts** and are the *same* voices you see
> in the UI's "🎙️ Use a saved voice" dropdown. The clip is trimmed to ≤10 s and the
> transcript + prompt are built **once** (lazily) and cached.

### 2.4 Output format
- **mp3** ✅ (you asked for it) — ffmpeg is installed, so encode wav→mp3 via `pydub`.
- Keep **wav** as an option (`format=wav`) for max quality / zero re-encode.
- Default: **mp3** at 192 kbps.

### 2.5 Auth (LAN safety)
- Simple **API key** header: `X-API-Key: <key>`. Set via env `OMNIVOICE_API_KEY`.
- If the env var is unset → API is **open** (fine for a trusted home LAN).

### 2.6 Concurrency
- One GPU → all generation goes through the **same `MODEL_LOCK`/queue** already in the app.
- Sync requests queue up behind each other (and behind UI jobs) — no crashes, just waits.
- Recommend a per-request soft timeout + a max text length guard.

---

---

## 3. API reference

Base URL `http://<pc-ip>:8001`. Interactive docs: `/api/docs`.

Auth is open by default. Set `OMNIVOICE_API_KEY=<key>` to require an `X-API-Key`
header, or `OMNIVOICE_API_KEYS=k1:acme,k2:globex` to give each customer their own
key **and their own private voice library**.

### 3.0 Contract freeze

`POST /api/tts` will not change shape:

* **in:** `multipart/form-data`
* **out:** raw audio bytes
* **headers:** `X-Duration-Sec`, `X-RTF` (both still present)

Everything added since is either an **extra response header** (safe to ignore)
or lives on **`/api/v2/tts`**. Both paths run identical normalization,
verification and loudness work — only the shape of the reply differs. Unknown
request parameters are reported in `X-OmniVoice-Warning` rather than rejected,
so an older client keeps working; set `OMNIVOICE_STRICT_PARAMS=1` to turn that
into a `400` once you know nobody is sending them.

### 3.1 Health, in three parts

| Endpoint | Answers | Use it for |
|---|---|---|
| `GET /api/live` | is the process up? | container liveness |
| `GET /api/ready` | **did the model actually produce words recently?** | **load balancer, and before starting a batch** |
| `GET /api/selftest` | generate four words *right now* | manual check; `409` if busy |
| `GET /api/health` | everything, including VRAM | dashboards |
| `GET /api/metrics` | counters (generations, OOMs, verify failures, tail trims, replays) plus `rtf_overall` | monitoring |

`/api/ready` returns **503 + `Retry-After`** when the last self-test failed or
is stale. This is the one to trust: a server whose GPU is dead still answers
`/api/live` happily, and that is exactly how a twenty-minute batch gets started
against a server that cannot produce a word.

```json
GET /api/ready   → 503
{"status": "degraded",
 "detail": "AcceleratorError: CUDA error: out of memory",
 "last_selftest_age_s": 41.2,
 "queue_depth": 0,
 "vram": {"allocated_mb": 5310, "reserved_mb": 7800,
          "fragmentation_mb": 2490, "free_mb": 210, "total_mb": 8192},
 "generation_ok": false, "oom_total": 3}
```

`fragmentation_mb` is `reserved − allocated`: memory the allocator holds but
cannot hand out. That is the "GPU at 5 % utilisation and out of memory" failure,
and it is invisible if you only look at `allocated`.

### 3.2 `POST /api/tts` — synchronous, returns audio bytes

| Field | Type | Default | Notes |
|---|---|---|---|
| `text` | string | **required** | over 8,000 chars / 1,300 words → `413`, naming `/api/tts/async`, which takes 100,000. Markdown, HTML, bullets, `[brackets]`, ALL-CAPS speaker labels and URLs are stripped before synthesis on every path |
| `voice` | file | — | reference clip; omit for a designed voice |
| `voice_id` | string | — | a registered voice; **unknown id → `404`, never a substitute** |
| `ref_text` | string | — | transcript of `voice`; checked against the audio |
| `language` | string | `Auto` | `"English"` and other full names are accepted |
| `format` | `mp3`\|`wav` | `mp3` | |
| `steps` | int | `16` | 8–10 faster, 32+ better |
| `speed` | float | `1.0` | |
| `project` | string | `tts` | used for the filename |
| `seed` | int | — | seeds torch/numpy before generation |
| `json` | query `0`\|`1` | `0` | `1` returns JSON and saves the file instead |

Headers on the reply:

| Header | Meaning |
|---|---|
| `X-Duration-Sec`, `X-RTF` | unchanged, always present |
| `X-WPM` | measured speaking rate |
| `X-WPM` | measured speaking rate |
| `X-LUFS`, `X-True-Peak-dB` | measured loudness of what you got |
| `X-Verified` | `true` if the clip was transcribed back and checked |
| `X-OmniVoice-Normalized-Text` | exactly what the model was given to say |
| `X-OmniVoice-Warning` | anything wrong: words dropped, a tail trimmed, loudness missed |
| `X-OmniVoice-Idempotent-Replay` | `true` when this is a cached replay |
| `X-OmniVoice-API-Version` | `2.0` |

**Read `X-OmniVoice-Warning`.** It is the difference between shipping a video
with four missing words and knowing about them.

Send an **`Idempotency-Key`** header and a retry after a client timeout returns
the cached audio instead of paying for the GPU twice (`409` while the first one
is still running).

### 3.3 `POST /api/v2/tts` — same inputs, JSON out

Everything the verifier measured, plus the audio inline and on disk. Extra field
`inline_audio` (`1` by default) controls the base64 payload.

```json
{"ok": true, "project": "intro", "duration_sec": 12.4,
 "voice": "cloned", "voice_id": "narrator_a",
 "normalized_text": "five hundred million dollars was appraised at ...",
 "verified": true, "verification": [],
 "warnings": [],
 "wpm": 158.0, "baseline_wpm": 161.0,
 "loudness": {"in_lufs": -25.6, "out_lufs": -20.0,
              "true_peak_db": -1.0, "met_target": true, "limited_by": null},
 "rtf": 0.42, "chunks": 3, "seed": null,
 "file": "intro_9c1f2a7b.mp3", "download_url": "/api/files/intro_9c1f2a7b.mp3",
 "audio_base64": "..."}
```

When something did go wrong, `verification` names it:

```json
"verified": true,
"verification": [{"chunk": 2, "dropped": ["fraud"], "inserted": [],
                  "word_accuracy": 0.917}],
"warnings": ["chunk 2: not spoken: fraud"]
```

### 3.4 `POST /api/tts/async` → `GET /api/jobs/{id}` → `/download`

Same fields (no `format` on submit; choose it at download). The job shares the
Studio's queue, so async work and the UI cannot fight over the GPU.
`GET /api/jobs` lists recent jobs. Job status carries `warnings`, `verified`,
`verification`, `wpm`, `rtf`, `gen_sec`, `audio_sec`, `chunks`, `loudness` and
`normalized_text`.

### 3.5 `POST /api/voices` — register a reusable voice

`name` (string) + `voice` (file) + optional `ref_text`.

The clip is trimmed at a natural pause, levelled to −20 LUFS, transcribed, and
measured. The reply is **actionable** — read `warnings` before generating ten
hours of audio:

```json
{"voice_id": "narrator_a", "accepted": true,
 "ref_text": "the transcript that will actually be used",
 "quality_score": 0.72, "baseline_wpm": 104, "duration_sec": 9.4,
 "lufs": -20.0, "snr_db": 11,
 "warnings": ["reference is noisy (SNR 11 dB) — the cloned voice will carry that noise"],
 "hint": "docs/reference_playbook.md"}
```

If you send a `ref_text` that matches under 90 % of what the recording actually
says, the upload is **rejected with `422`** and the reply tells you what was
heard. A mismatched transcript is the main cause of reference words leaking into
generated clips — being told at upload beats finding out three hours in. Set
`OMNIVOICE_STRICT_REF=0` to downgrade that to a warning.

`GET /api/voices` lists them (scoped to your key when multi-tenant).
`DELETE /api/voices/{voice_id}` removes one. Another tenant's voice reports
`404`, not `403` — a `403` would confirm the id exists.

Voices survive a restart: `voices/index.json` plus the repaired clip.

### 3.6 `POST /api/transcribe` — check a clip yourself

`audio` (file) + optional `text`. Uses the Whisper already loaded here, so the
audit that found these bugs costs nothing to repeat.

```json
{"text": "what was actually said",
 "ok": false,
 "summary": "extra at end: forcing him to back down",
 "diff": {"dropped": [], "inserted": ["forcing", "him", "to", "back", "down"],
          "tail_inserted": ["forcing", "him", "to", "back", "down"],
          "word_accuracy": 1.0}}
```

Note `word_accuracy: 1.0` with a failure: everything sent *was* said, plus five
words that were not. Accuracy alone can never be the gate.

### 3.7 `GET /api/files/{name}` — download a produced file

---

## 4. Status codes

| Code | When | What to do |
|---|---|---|
| `400` | empty `text`; unknown params in strict mode | fix the request |
| `401` | bad or missing `X-API-Key` | |
| `404` | unknown `voice_id`, job or file | **never** a silent voice substitution |
| `409` | job not finished; idempotency key still running | obey `Retry-After` |
| `413` | input over the **synchronous** limit | resend to `/api/tts/async`, whose limit is far higher — the message names it |
| `422` | reference rejected (transcript mismatch) | see `detail.message` |
| `429` | GPU busy past `OMNIVOICE_QUEUE_WAIT` | obey `Retry-After` |
| `503` | `insufficient_vram` — refused before starting, in ~0.16 s | obey `Retry-After`; if `spilled_to_shared` is true the server needs a restart, not a retry |
| `500` | generation failed | |
| `503` | GPU unavailable / not ready | obey `Retry-After`; check `/api/ready` |

Errors are JSON (`application/json`) even where you asked for audio — check the
status code before treating the body as a file.

---

## 5. Examples

```bash
# simplest: designed voice, mp3 out
curl -s -X POST http://localhost:8001/api/tts \
     -F "text=Hello from the API" -F "format=mp3" -o out.mp3

# clone from an uploaded clip
curl -s -X POST http://localhost:8001/api/tts \
     -F "text=Hello there." -F "voice=@my_voice.wav" -F "format=wav" -o out.wav

# register once, reuse forever
curl -s -X POST http://localhost:8001/api/voices \
     -F "name=narrator_a" -F "voice=@my_voice.wav"
curl -s -X POST http://localhost:8001/api/tts \
     -F "text=Second clip, same voice." -F "voice_id=narrator_a" -o clip2.mp3

# see the warnings, not just the audio
curl -sD - -o out.mp3 -X POST http://localhost:8001/api/tts \
     -F "text=A property worth \$500 million." -F "voice_id=narrator_a" \
     | grep -i "^x-"

# retry-safe
curl -s -X POST http://localhost:8001/api/tts \
     -H "Idempotency-Key: 3f1c9e2a-..." \
     -F "text=Bill me once." -F "voice_id=narrator_a" -o once.mp3

# the reference-bleed probe from the bug report
curl -s -F "text=Trump really is. Is he the businessman he claims?" \
     -F format=wav -F steps=16 -F voice_id=narrator_a \
     http://localhost:8001/api/tts -o probe.wav
curl -s -F "audio=@probe.wav" \
     -F "text=Trump really is. Is he the businessman he claims?" \
     http://localhost:8001/api/transcribe
```

```python
import requests

r = requests.post("http://localhost:8001/api/v2/tts", data={
    "text": "Hello from Python.", "voice_id": "narrator_a", "format": "mp3"})
r.raise_for_status()
body = r.json()
if body["warnings"]:
    print("check this clip:", body["warnings"])
open("out.mp3", "wb").write(__import__("base64").b64decode(body["audio_base64"]))
```

---

## 5b. What a caller can actually expect

Every number here was measured on the reference box (RTX 3060 Ti 8 GB, Windows,
16 steps, verification on). Reproduce with `tools/rtf_probe.py`.

### Latency

`X-RTF` is generation time over audio length, so **lower is faster** and a
figure under 1.0 means faster than real time.

| clip | audio | RTF | wall |
|---|---:|---:|---:|
| 12 words | 3.9 s | ~0.43–0.50 | ~1.8 s |
| 60 words | 22.5 s | ~0.19 | ~4.2 s |
| 240 words | 72 s | ~0.18 | ~13 s |
| long-form | 203 s | **~0.17** | ~35 s |

**Short clips are not slower for a bad reason** — a 3.9 s clip pays the same
fixed setup as a 73 s one, so the RTF looks worse while the wall time is under
two seconds. Size your timeouts on wall time, not on RTF.

Rule of thumb for planning: **ten minutes of finished audio takes under two
minutes to render.**

### Several callers at once

The GPU serves one generation at a time (`OMNIVOICE_MAX_CONCURRENCY`, default
1 on an 8 GB card). Extra callers queue rather than fail:

| simultaneous callers | per-clip RTF | outcome |
|---|---|---|
| 1 | 0.234 | — |
| 4 | 0.212–0.222 | 4/4 succeeded |
| 8 | 0.210–0.221 | 8/8 succeeded |

**Per-clip speed does not degrade under load and nothing is refused** —
throughput across 8 simultaneous callers was RTF 0.217 against 0.234 for one
alone, so queueing costs essentially nothing. What grows is waiting: with eight
clips in flight the last caller waited 19.4 s.

That has one consequence for client design. `POST /api/tts` holds the HTTP
connection for the whole queue wait, so **anything submitting more than one
clip should use `POST /api/tts/async`** — it returns in about 10 ms with a
`job_id`, and the client polls `GET /api/jobs/{id}`. Five async jobs submitted
in 0.01 s finished in 10.3 s.

Note that `/api/tts/async` deliberately does not take `format`; the warning
saying so is the unknown-parameter mechanism working, not a bug.

### Verification, and what it costs

With checking on, every clip is transcribed back and diffed against the
script before it is returned. It is **off by default** — see below. That costs roughly **25–40 %** of render time depending on clip
length, and it is what puts `X-Verified`, `X-OmniVoice-Warning` and the
`diff` block in the v2 response.

The transcriber's notation is read charitably, because it is not the model:

* clock times come back with a full stop (`7:45` → `7.45`), which would
  otherwise read as a decimal;
* a transcriber that has just written `$4.2 million` writes the next bare
  `3 million` as `$3 million`, adding a "dollars" the script never had.

Both are scored both ways and the better reading wins, so a false alarm is
removed while a word the model genuinely dropped is still reported. Confirmed
against a second, unrelated transcriber (`tools/verify_external.py`): on ten
clips, the built-in verifier called 10/10 clean and AssemblyAI 9/10, agreeing
on 9/10 — the single disagreement being AssemblyAI's own mistake.

**Should you turn it off?** Measured both ways on the same batch: 27 s checked
against 20 s unchecked, and long-form RTF 0.170 against 0.151. The difference
that matters is not detection — `tools/audit_batch.py` finds the same defects
afterwards — but **repair**: with checking on, a chunk that came out wrong is
regenerated during the run and the caller never sees it. With it off you learn
which clips are bad and re-render them yourself. On this server the recommended
setting is **on**.

**Checking is OFF by default.** `X-Verified` reads `false`, no `diff` block is
returned, and nothing is re-rolled. Audit the batch afterwards with
`tools/audit_batch.py`, which finds the same defects at no per-clip cost.

Turn it on per server with `OMNIVOICE_VERIFY=1`. Measured on a ten-clip batch:
27 s checked against 24 s unchecked, best RTF 0.174 against 0.143.

`OMNIVOICE_VERIFY_RETRIES` is also `0` by default, so checking — when you do
turn it on — warns without re-rolling. Set it to `1` and a chunk that came out
wrong is regenerated during the run and the caller never sees it, at the cost
of re-verifying the whole clip: clips with one regenerated chunk verified 3.5x
slower per chunk than clips with none.

## 5c. Why was this clip slow, and what happens when the GPU is full

Two things a caller could not previously find out: **why** a clip took as long
as it did, and **whether the server had room** before it started.

### `X-RTF-Reason` and the `rtf_reason` block

`X-RTF` says how much. It never said why — and the two expensive causes look
identical from outside. A card so short of memory that everything crawls and a
clip whose chunks were quietly rebuilt five times both report `0.30`. One is
cured by restarting the box, the other by fixing that voice's reference.
Measured on one real batch: 396 words in 27.8 s and 394 words in 50.1 s, same
voice, same settings, two minutes apart.

Every response now carries the answer. On `/api/tts`:

```
X-RTF: 0.431
X-RTF-Reason: higher than usual: 2 of 23 chunk(s) had to be generated again
              (dropped words x2), costing 3.3s; verification took 49.6s
              against 36.4s of generation
X-Gen-Sec: 36.41      X-Verify-Sec: 49.64     X-Regen-Sec: 3.33
X-Chunks: 23          X-Chunks-Regenerated: 2  X-VRAM-Free-MB: 3448
```

A clip that was ordinary says so, and says nothing else:

```
X-RTF: 0.170      X-RTF-Reason: normal
```

`POST /api/v2/tts` returns the same thing structured:

```json
"rtf": 0.431,
"rtf_reason": {
  "rtf": 0.431, "normal": false, "normal_up_to": 0.26,
  "summary": "higher than usual: 2 of 23 chunk(s) had to be generated again …",
  "causes": ["2 of 23 chunk(s) had to be generated again (dropped words x2), costing 3.3s",
             "verification took 49.6s against 36.4s of generation"]
},
"timing": {"normalize_s": 0.003, "generate_s": 36.411,
           "verify_s": 3.4, "reverify_s": 46.2, "join_s": 0.066,
           "total_s": 86.818},
"chunks": {"total": 23, "regenerated": 2, "which": [14, 22],
           "regeneration_reasons": {"dropped_words": 2}, "regenerate_s": 3.329},
"vram_at_start_mb": 3448.0
```

`reverify_s` is separate from `verify_s` on purpose. A regenerated chunk causes
the **whole clip** to be verified again, and folding that second pass into
`verify_s` hides most of what a regeneration costs — measured across 19 clips,
clips with one regenerated chunk verified 3.5x slower per chunk than clips with
none, so a reported `regenerate_s: 1.4` was really closer to 11 s. A first pass
and a second pass no longer look alike.

The causes it will name: chunks regenerated (with the reason and which ones),
verification outweighing generation, the card being short of memory when the
clip started, VRAM having spilled into system memory, a clip too short for the
fixed per-request cost to disappear into, and a voice prompt built from scratch.
`OMNIVOICE_RTF_NORMAL_MAX` (default `0.26`) sets where "normal" ends.

### `503 insufficient_vram` — refused in milliseconds, not dead in 35 seconds

A measured batch ran the card down to `free_mb: 0` with `reserved` at 9008 MB
on an 8191 MB card — CUDA had spilled into system RAM — and from then on every
request needing real memory **died 26-41 s in and returned nothing at all**: no
status code, no body, no log line, and every counter still reading zero. A
caller could not tell a crash from a hang.

Requests are now checked for headroom before any GPU work begins:

```json
HTTP 503   Retry-After: 30
{"detail": {"error": "insufficient_vram",
            "message": "the GPU has 412 MB free and this clip needs about 940 MB. Nothing was generated; retry shortly.",
            "free_mb": 412.0, "needed_mb": 940.0, "short_by_mb": 528.0,
            "fragmentation_mb": 3440.0, "spilled_to_shared": true,
            "recovered_mb": 0.0}}
```

Measured: **0.16 s to refuse.** A cache release is attempted first, so this only
fires when the memory genuinely is not there. `spilled_to_shared: true` means
the server needs restarting rather than retrying — it is the one condition a
`Retry-After` cannot fix. Counted as `vram_refusals` in `/api/metrics`.

Tune the floor with `OMNIVOICE_MIN_FREE_MB` (default `700`).

### `/api/ready` now needs headroom as well as a working model

It used to answer `ok` on the strength of the self-test alone, and in that
batch it did so for an entire session while `free_mb` was 0 and every
substantial request was dying — because four words still generate fine on a
card with nothing left. Both conditions must now hold:

```json
HTTP 503
{"status": "degraded", "ready": false,
 "reason": "only 412 MB of VRAM free (need 700 MB to start work)"}
```

The self-test itself also generates ~60 words rather than four, for the same
reason: a probe that does less than a real request answers a question nobody
asked. Override it with `OMNIVOICE_SELFTEST_TEXT`.

### What `X-OmniVoice-Warning` will and will not say

The header is only worth reading if it is right, so two checks that were firing
on healthy clips no longer do.

**"the clip ends at full volume"** is now measured on the audio you receive,
after joining, padding and loudness normalization. It used to be judged on the
model's raw last chunk, upstream of the 300 ms of silence this pipeline appends
— which flagged 11 of 63 clips that all measured −91 dB over their final
0.25 s. A clip genuinely cut off mid-word still warns.

**Speaking rate** is compared against the voice's own baseline, and the ceiling
is now `1.40 x baseline` (from `1.30`). A clean batch warned at 194, 195 and
204 wpm against a 147 wpm baseline, all ordinary. Tune with
`OMNIVOICE_RATE_LOW` / `OMNIVOICE_RATE_HIGH`.

### Per-voice regeneration rate

`/api/metrics` gains `per_voice`. A global regeneration count says the model is
struggling; a per-voice one says **which reference** is struggling, which is the
half a customer can act on:

```json
"per_voice": {"narrator_a": {"clips": 40, "chunks": 210,
                             "regenerated": 3, "regeneration_rate": 0.0143}}
```

A voice well above the others has a problem in its reference clip, not in the
scripts it is being given.

## 6. Running it

The API starts with the app — same process, same GPU, own port.

```powershell
venv\Scripts\python app.py          REM UI :7860, API :8001
set OMNIVOICE_API=0                   REM UI only
set OMNIVOICE_API_PORT=9000           REM different port
set OMNIVOICE_API_KEY=change-me       REM require X-API-Key
set OMNIVOICE_API_KEYS=k1:acme,k2:globex   REM per-customer voice isolation
set OMNIVOICE_MAX_CONCURRENCY=1       REM 1 on an 8 GB card
set OMNIVOICE_VERIFY=1                REM per-clip checking (default: 0, off)
set OMNIVOICE_VERIFY_RETRIES=1        REM re-roll a bad chunk (default: 0)
set OMNIVOICE_ASR_BATCH=12            REM verifier windows read at once
set OMNIVOICE_ASR_DEVICE=cpu          REM move Whisper off the GPU entirely
set OMNIVOICE_ASR_BACKEND=faster      REM CTranslate2 verifier: -1.6 GB VRAM
set OMNIVOICE_MIN_FREE_MB=700         REM refuse below this much free VRAM
set OMNIVOICE_RTF_NORMAL_MAX=0.26     REM above this, a clip explains itself
```

`OMNIVOICE_ASR_BATCH` is the one worth knowing about. The verifier reads its
30 s windows as a batch instead of one after another, which on the reference
box took long-form RTF from 0.221 to 0.170 for byte-identical transcripts.
**12 is the default because the whole gain is already there at 12** (0.170 at a
5936 MB peak) while 16 buys nothing (0.169) for another 811 MB on a card that
also has a desktop on it. Lower it if VRAM is tight; raising it above the
number of chunks in a clip does nothing at all.

Before you rely on it:

```powershell
venv\Scripts\python -m pytest tests -q
venv\Scripts\python tools\acceptance.py --voice narrator_a
```

