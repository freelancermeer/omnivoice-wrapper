# 🔌 OmniVoice Local API — Design & Spec

A small **HTTP API** so any machine on your LAN (or scripts on this PC) can send
**text + a reference voice** and get back an **audio file (mp3/wav)** — using the
same engine and default settings as the Voiceover Studio UI.

> ✅ **IMPLEMENTED** — the API is live in `app.py`. It runs in the same process as
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

## 3. API reference

**Base URL:** `http://<pc-ip>:8001`  (e.g. `http://192.168.100.76:8001`)
**Auth:** send `X-API-Key: <your key>` if `OMNIVOICE_API_KEY` is set.
**Content types:** `multipart/form-data` (with file) or `application/json` (no file).

### 3.1 `GET /api/health`
Liveness + model status.
```json
{ "status": "ok", "model": "OmniVoice", "device": "cuda",
  "sampling_rate": 24000, "queue": {"queued": 0, "processing": 0} }
```

### 3.2 `POST /api/tts`  — synchronous, returns audio
Send text (+ optional voice) → get the audio file back directly.

**Body (multipart/form-data):**

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `text` | string | ✅ | — | the script to speak |
| `voice` | file | — | (none) | reference clip (wav/mp3, 3–10 s). Omit = AI voice |
| `voice_id` | string | — | — | use a registered voice instead of uploading |
| `ref_text` | string | — | auto | transcript of the voice clip (else Whisper auto) |
| `language` | string | — | `Auto` | e.g. `English`, `Urdu`, or code `en` |
| `format` | string | — | `mp3` | `mp3` \| `wav` |
| `steps` | int | — | `16` | quality steps (8 fast … 32+ better) |
| `speed` | float | — | `1.0` | 0.5–1.5 |
| `project` | string | — | `tts` | used for the returned filename |

**Response:** the audio bytes.
`200 OK`, `Content-Type: audio/mpeg` (or `audio/wav`),
`Content-Disposition: attachment; filename="<project>.mp3"`,
plus headers `X-RTF`, `X-Duration-Sec`, `X-Gen-Sec`.

**Alt response (`?json=1`):** returns JSON with a download URL instead of raw bytes:
```json
{ "ok": true, "project": "intro", "file": "intro.mp3",
  "download_url": "/api/files/intro.mp3",
  "duration_sec": 12.6, "gen_sec": 1.9, "rtf": 0.15,
  "voice": "cloned", "ref_text": "Welcome to the show…" }
```

### 3.3 `POST /api/tts/async`  — enqueue, returns a job id
Same fields as `/api/tts`, but returns immediately:
```json
{ "job_id": 42, "project": "intro", "status": "queued" }
```

### 3.4 `GET /api/jobs/{id}`  — job status
```json
{ "job_id": 42, "project": "intro", "status": "done",
  "rtf": 0.15, "duration_sec": 12.6, "download_url": "/api/files/intro.mp3" }
```
`status` ∈ `queued | processing | done | error | cancelled`.

### 3.5 `GET /api/jobs/{id}/download`  — fetch the audio (when done)
Returns the audio bytes (respects `format`). `409` if not finished yet.

### 3.6 `POST /api/voices`  — register a reusable voice (permanent)
Upload a clip once; get a `voice_id` to reuse forever (survives restarts, shows up
in the UI dropdown too). The clip is auto-trimmed to ≤10 s.
**Body:** `voice` (file, ✅), `name` (string), `ref_text` (optional).
```json
{ "voice_id": "narrator_a", "ref_text": "Welcome to the show…" }
```

### 3.7 `GET /api/voices`  — list saved voices
```json
{ "voices": [ { "voice_id": "narrator_a", "ref_text": "Welcome…" } ] }
```
Also on `GET /api/health` under `"voices": [...]`.

### 3.8 `DELETE /api/voices/{voice_id}`  — delete a saved voice
Removes the clip + index entry (same library the UI uses). `404` if unknown.
```json
{ "deleted": "narrator_a" }
```
```bash
curl -X DELETE http://localhost:8001/api/voices/narrator_a
```

> **Duplicate names auto-unique:** registering a name that already exists returns a
> new unique `voice_id` (e.g. `narrator_a_2`) — existing voices are never overwritten.

### 3.8 `GET /api/files/{name}`  — download a produced file
Serves from the output folder (respects `OMNIVOICE_OUTPUT_DIR`).

---

## 4. Examples

### curl — simplest (text only, AI voice → mp3)
```bash
curl -X POST http://192.168.100.76:8001/api/tts \
  -F "text=Hello, this is a test of the local voiceover API." \
  -F "format=mp3" \
  -o out.mp3
```

### curl — clone a voice (upload clip)
```bash
curl -X POST http://192.168.100.76:8001/api/tts \
  -H "X-API-Key: mysecret" \
  -F "text=Welcome to my channel, subscribe now." \
  -F "voice=@my_voice.wav" \
  -F "language=English" -F "steps=16" \
  -o welcome.mp3
```

### curl — reuse a registered voice
```bash
# register once
curl -X POST http://192.168.100.76:8001/api/voices \
  -F "name=narrator_a" -F "voice=@narrator.wav"
# then use it forever
curl -X POST http://192.168.100.76:8001/api/tts \
  -F "text=Chapter one." -F "voice_id=narrator_a" -o ch1.mp3
```

### Python
```python
import requests
r = requests.post("http://192.168.100.76:8001/api/tts",
    data={"text": "Hello from Python", "format": "mp3", "steps": 16},
    files={"voice": open("my_voice.wav", "rb")},  # optional
    headers={"X-API-Key": "mysecret"})
open("out.mp3", "wb").write(r.content)
print("RTF:", r.headers.get("X-RTF"))
```

### JavaScript (Node / browser fetch)
```js
const fd = new FormData();
fd.append("text", "Hello from JS");
fd.append("format", "mp3");
// fd.append("voice", fileInput.files[0]);   // optional
const res = await fetch("http://192.168.100.76:8001/api/tts", { method: "POST", body: fd });
const blob = await res.blob();   // audio/mpeg
```

### n8n / Make / any HTTP node
`POST http://<pc>:7860/api/tts`, body = form-data with `text` (+ `voice` file),
response = binary → save as `.mp3`. Done.

---

## 5. Running it
No new process — the API starts inside `app.py` (a background thread) on its own
port. Same launchers:
- `run.bat` → **UI** at `http://<pc>:7860`, **API** at `http://<pc>:8001/api/...`
- Optional key: `set OMNIVOICE_API_KEY=mysecret`
- Disable API: `set OMNIVOICE_API=0` · Change port: `set OMNIVOICE_API_PORT=8080`
- Interactive docs (auto): **`http://<pc>:8001/api/docs`** (FastAPI Swagger UI) — click-to-try.

---

## 6. Implementation plan (when you say "build it")
Everything reuses code that already exists in `app.py`:

1. Grab the FastAPI instance Gradio creates: `demo.launch(...)` returns / exposes
   `app.app` (a FastAPI). Add an `APIRouter` with the routes above, or mount before launch.
2. **Sync `/api/tts`** → save uploaded `voice` to a temp file → call the existing
   `_get_or_build_prompt()` + `_gen_core_impl()` under `MODEL_LOCK` → get the numpy wav
   → if `format=mp3`, `pydub.AudioSegment(...).export(bitrate="192k")` → stream bytes back.
3. **Async `/api/tts/async`** → reuse `add_job()` / the worker queue → `job_id` = the
   job's id → `GET /api/jobs/{id}` reads the same `JOBS` list; download reads `job["file"]`.
4. **Voice registry** → a dict `{voice_id: VoiceClonePrompt}` built via `_get_or_build_prompt`
   (transcribe/encode once). Persist the source clips under `voices/` if you want them to
   survive restarts.
5. **mp3** → helper `to_mp3(wav_path_or_array) -> bytes` using pydub (ffmpeg present ✅).
6. **Auth** → tiny dependency that checks `X-API-Key` against `OMNIVOICE_API_KEY`.

Estimated: ~120–150 lines added to `app.py`, no new heavy deps (fastapi, pydub, ffmpeg
already installed).

---

## 7. Choices for you (tell me before I build)
1. **Sync only, or sync + async queue?** (I recommend both.)
2. **Voice registry?** (upload-once, reuse by id — great for repeated narrators.)
3. **Default format mp3** at 192 kbps ok? (or 128/320, or default wav?)
4. **API key** on by default, or open on the home LAN?
5. **Return style for sync:** raw audio bytes (simplest) vs JSON+download URL? (I lean raw
   bytes, with `?json=1` opt-in.)

Once you pick, I'll wire it into `app.py` and add the endpoints + a couple of tests.
