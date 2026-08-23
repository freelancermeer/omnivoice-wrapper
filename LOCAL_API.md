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

## 6. Running it

The API starts with the app — same process, same GPU, own port.

```powershell
venv\Scripts\python app.py          REM UI :7860, API :8001
set OMNIVOICE_API=0                   REM UI only
set OMNIVOICE_API_PORT=9000           REM different port
set OMNIVOICE_API_KEY=change-me       REM require X-API-Key
set OMNIVOICE_API_KEYS=k1:acme,k2:globex   REM per-customer voice isolation
set OMNIVOICE_MAX_CONCURRENCY=1       REM 1 on an 8 GB card
```

Before you rely on it:

```powershell
venv\Scripts\python -m pytest tests -q
venv\Scripts\python tools\acceptance.py --voice narrator_a
```

