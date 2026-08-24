# 🎙️ OmniVoice Voiceover Studio

A local, offline-first web app + REST API for [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
— a massively multilingual (600+ languages) zero-shot TTS model with **voice
cloning** and **voice design**. Built with Gradio, it adds a production-style
**project queue**: pick a voice, add scripts, render them one at a time, every
clip auto-saved to a folder.

Every clip is **transcribed back and compared against your script** before it is
returned, **levelled to the same loudness** as every other clip, and generated
from a reference that was **cut at a natural pause rather than a stopwatch**.
Those three things are why this exists rather than a bare `model.generate()`.

> **Note:** the model weights (`Model/`, ~3.3 GB) and the Python `venv/` are **not**
> in this repo (see `.gitignore`). Follow the setup below to get them on a new PC.

---

## Requirements
- **NVIDIA GPU** with CUDA 12.8 drivers (tested on RTX 3060 Ti 8 GB, FP16).
- **Python 3.10 – 3.12** (developed on 3.12).
- Windows (works on Linux/macOS too; adjust the `.bat` launchers to shell scripts).
- ~5 GB disk for the models (OmniVoice ~3.3 GB + Whisper ~1.6 GB).

---

## Setup on a new PC

```powershell
git clone https://github.com/<your-user>/<your-repo>.git omnivoice
cd omnivoice
py -3.12 -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install torch==2.8.0 torchaudio==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu128
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\hf.exe download k2-fsa/OmniVoice --local-dir Model
venv\Scripts\python app.py
```

**Already have a venv?** Re-run the `-r requirements.txt` line: this targets
**omnivoice 0.2.1**, and the app prints a loud warning at startup if it finds an
older one. 0.1.5 still runs, but without control over the model's own
fade/padding, the punctuation fix, `asr_device`, or cached voices that survive a
restart.

The app auto-detects `./Model` and loads from it (fully offline afterwards). If
you skip step 4 it downloads from HuggingFace on first launch instead. The
**Whisper** ASR model (`openai/whisper-large-v3-turbo`, ~1.6 GB) downloads
automatically on first run — it transcribes reference clips *and* checks every
generated clip against its script. Pre-cache it with
`venv\Scripts\hf.exe download openai/whisper-large-v3-turbo`.

The terminal prints:

```
  UI    (local):   http://127.0.0.1:7860
  UI    (LAN):     http://<your-ip>:7860
  API   (local):   http://127.0.0.1:8001/api
  API   ready:     http://127.0.0.1:8001/api/ready   <- point monitoring here
```

---

## Running & sharing

| Launcher | What it does |
|----------|--------------|
| **`run.bat`** | Starts the app on `127.0.0.1:7860` **and** the LAN (binds `0.0.0.0`). |
| **`run_share.bat`** | Also creates a temporary public `https://xxxxx.gradio.live` link (72 h). |
| **`run_tests.bat`** | Unit tests. **No GPU needed** — runs on any machine. |

- **LAN:** others open `http://<your-ip>:7860`. If it doesn't connect, allow the
  port in Windows Firewall (admin CMD):
  `netsh advfirewall firewall add rule name="OmniVoice" dir=in action=allow protocol=TCP localport=7860`
- **Local only:** set `GRADIO_SERVER_NAME=127.0.0.1`.
- **Password:** `OMNIVOICE_AUTH=user:pass` (do this before sharing publicly).
- **Auto-start on login:** `Win+R` → `shell:startup` → drop a shortcut there.

**Works fully offline:** Gradio serves its UI from the installed package, the
models are local, and the `.bat` files set `HF_HUB_OFFLINE=1`.

---

## Using the app (single page)

1. **Pick a voice** — upload a 6–10 s clip. It is trimmed at a natural pause,
   levelled, transcribed, and checked; you are told about anything wrong with it
   *before* you generate ten hours of audio. See
   **[docs/reference_playbook.md](docs/reference_playbook.md)**. Leave it empty
   for a designed/AI voice.
2. **Add your script** — Project name + Script → **Add to queue**. Or
   **🧩 Add several at once**: a table of Project name + Script, one row per clip,
   all using the same voice.
3. **Settings (optional)** — language, quality steps (16 = fast, 32+ = better),
   speed, guidance, duration.
   *Paste as much as you like:* the queue accepts up to 100,000 characters and
   handles the splitting, cleaning and joining internally. Markdown, bullets,
   HTML, URLs and footnote brackets are all stripped before synthesis.
4. **Render queue** — jobs process one at a time. Each card carries its own
   metrics row — **RTF** (green under 1.0, amber over), audio length, time taken,
   wpm, LUFS, chunk count — plus the exact voice transcript used and **any
   warning about what came out** (words dropped, a leaked tail removed, loudness
   that could not be reached). The header shows the **average and best RTF**
   across the queue, so the number you tune is always on screen.
5. **Save & downloads** — every render auto-saves; plus **📦 Zip all completed**.
6. **🛑 Shut down PC when the queue finishes** — optional.

---

## 🔌 Local REST API

Port **8001**, same process, shared model and queue. Full spec in
**[LOCAL_API.md](LOCAL_API.md)**; interactive docs at `http://<pc-ip>:8001/api/docs`.

```bash
# text + voice clip -> mp3   (this contract is frozen and will not change)
curl -X POST http://<pc-ip>:8001/api/tts \
  -F "text=Hello from the API" -F "voice=@my_voice.wav" -F "format=mp3" -o out.mp3
```

`POST /api/tts` · `POST /api/v2/tts` (JSON + verifier metadata) ·
`POST /api/tts/async` + `GET /api/jobs/{id}` · `POST /api/transcribe` ·
voice library CRUD · `GET /api/live` · `GET /api/ready` · `GET /api/selftest` ·
`GET /api/health` · `GET /api/metrics`.

Disable with `OMNIVOICE_API=0`; secure with `OMNIVOICE_API_KEY=<key>`;
per-customer isolation with `OMNIVOICE_API_KEYS=k1:acme,k2:globex`.

---

## What this wrapper fixes

A production batch of **63 clips / 64.7 minutes / 10,269 words** was generated,
transcribed back, and diffed word by word against the scripts that were sent.
Each row below is a measured finding and where it is handled.

| Finding | Fix |
|---|---|
| **218 words spoken that were never sent.** One voice contributed 185 (`forcing` ×163). Its reference was hard-cut at exactly 10.0 s — mid-word — and the model kept finishing that word at the end of 86–90% of its clips. | References never end mid-word ([`audio_fx.smart_trim_reference`](audio_fx.py)), whether **we** cut them there (cut at the last natural pause instead, preferring a shorter clip over a mid-word one) or **the customer** stopped recording mid-word (cut back to their last finished phrase). Then faded, given 0.3 s of trailing silence, and re-transcribed so text and audio always agree. What cannot be repaired is explained at upload. Anything that still leaks is caught by the verifier and trimmed. |
| **12 words silently dropped**, mostly one arm of a repeated structure ("He called it perjury / fraud / **contempt**" lost the middle clause), with nothing reporting it. | Every chunk is transcribed back and **sequence-aligned** against the script ([`verify.word_diff`](verify.py)). A drop triggers a regeneration; what cannot be fixed is reported in `X-OmniVoice-Warning` and on the job card. |
| **Loudness tracked the reference**: −0.2 dB (edge of clipping) to −12.6 dB in one video. | ITU-R BS.1770 loudness normalization to **−20 LUFS** with a **−1 dBTP** ceiling, on both the stored reference and every output. |
| **CUDA OOM that never recovered**, while `/api/health` said `"ok"` and GPU utilisation read 5 %. | `inference_mode`, results moved to CPU immediately, explicit `del` → `gc.collect()` → `empty_cache()`, `expandable_segments:True` (**a no-op on Windows** — torch ignores it; see [GPU_RUN.md](GPU_RUN.md)), bounded concurrency, OOM retry and optional model reload ([`gpu_guard.py`](gpu_guard.py)). Measured over 73 generations: 0 OOM, 0 reloads, +0 MB allocated and +0 MB reserved across a 25-clip soak. |
| **Health lied.** | `/api/ready` is backed by a **cached self-test that actually generates words**, and returns `503` when it fails. `/api/live` is the pure liveness probe. `/api/health` reports allocated **and** reserved VRAM, so fragmentation is visible. |
| **Four concurrent callers killed the server.** | A semaphore (default 1 on 8 GB). Extra callers queue, then get `429` with `Retry-After`. |
| **Stitched long-form output has an audible seam** — measured elsewhere at ~28 dB energy jumps and 67–69 Hz F0 jumps at chunk boundaries, and XTTS's per-chunk normalization makes volume jump between sentences. | `audio_fx.join_chunks`: even edges, a fixed inter-sentence gap, 15 ms edge fades, and level matching to the clip's **own median** (never to a fixed per-chunk target). |
| **Silence stripping clips final consonants**, a documented way to lose the last word of a sentence. | A 10 dB stricter edge threshold on generated audio, and 300 ms of tail padding on every clip. |
| **133–203 wpm across one batch**; short inputs rushed. | Balanced chunking, and wpm measured per clip and compared against **that voice's own baseline** — a metric, never a gate (a documentary narrator runs ~100 wpm; a fixed 140–180 band would fail every clip). |
| **Scripts arrive as documents, not as speech** — markdown, HTML, bullet markers, footnote brackets, speaker labels, URLs. Unsupported typography is documented to make neural vocoders "fail or emit odd static noises", and OmniVoice has an open issue about exactly that. | `textnorm.sanitize_script` strips markdown/HTML, reads URLs as domains, drops `[stage directions]` and ALL-CAPS speaker labels, calms runaway punctuation, and gives bullets and headings a full stop so they become audible pauses instead of running together. |
| **A misheard proper noun was costing a full re-generation.** The batch saw `Bessent` transcribed as "bessant" 5× and `Hegseth` as "hexeth" 2× — the transcriber, not the model. | Substitutions are separated from real drops. A phonetically close swap becomes a *pronunciation note*, not a retry; numbers, money words and negations are never written off that way, because "million" vs "billion" is not a mishearing. |
| Numbers, currency, em dashes, ordinals, acronyms. | A wrapper-owned text front-end ([`textnorm.py`](textnorm.py)), with the result returned in `X-OmniVoice-Normalized-Text` so it can be checked without listening. |
| Retries after a client timeout paid for the same clip twice. | `Idempotency-Key` header; a replay returns the cached audio with `X-OmniVoice-Idempotent-Replay: true`. |
| A typo in `voice_id` silently used a different voice. | **404, always.** Never a substitute. |

Earlier fixes that are still in place: reference trimming for GitHub #50,
sentence chunking for #144, one shared voice prompt across chunks for #44, FP16
+ TF32, steps default 16.

**Verified not broken, do not "fix":** numbers, currency, percentages, decimals
and ordinals were already correct in the measured batch; ordinary acronyms
(`RFK` → "r f k", `CFO` → "c f o") are already right, so the lexicon only
carries acronyms that must be read as *words* (MAGA, NASA).

---

## Getting the lowest RTF

RTF (compute time ÷ audio duration) is the number this gets judged on. Where the
time actually goes, and what each lever costs:

| Lever | Effect | Cost |
|---|---|---|
| **Quality steps** | `16` is the default and the best value. `8–10` is noticeably faster | below ~10, quality starts to show |
| **`OMNIVOICE_VERIFY_MODE=fast`** (default) | one ASR pass over the finished clip instead of one per chunk — a five-chunk job drops from 5 extra passes to **1** | drilling into chunks only happens when something is actually wrong |
| `OMNIVOICE_VERIFY=0` | removes checking entirely | you go back to shipping bad clips silently — the thing this exists to prevent |
| *(automatic)* | a substitution the transcriber probably misheard no longer triggers a re-generation | none — it is reported as a pronunciation note instead |
| `OMNIVOICE_MAX_CHARS` | now `200`. Fewer, longer chunks mean fewer fixed per-call overheads **and** a steadier voice — upstream reports speaker switching on short generations | too large and long-text degeneration returns (upstream #144) |
| `OMNIVOICE_NUM_STEP` | the one parameter that reliably trades quality for speed | below ~10 it shows |
| *(not a lever)* | `guidance_scale` — upstream #163 reports it has no audible effect through the Python API | — |
| `OMNIVOICE_BATCH` | >1 batches chunks per GPU call | usually *raises* RTF: padding to the longest chunk wastes compute. Only for long, uniform chunks |
| `OMNIVOICE_PREWARM_VOICES=1` | no embedding rebuild on the first request per voice after a restart | none |
| **`OMNIVOICE_ASR_BATCH`** | now `12`. The verifier reads its 30 s windows as one batch instead of one after another — long-form RTF **0.221 → 0.170**, transcripts identical (word accuracy 1.0000) | ~800 MB of VRAM. `16` buys nothing further (0.169) for another 811 MB |
| *(automatic)* | clock times and inferred currency symbols in the transcript no longer count as errors, so they no longer buy a re-generation — one clip went **0.548 → 0.244** | none; both readings are scored and only a better match is accepted |
| **close Parsec** | not a code lever, but the largest one on the reference box: Parsec takes **~25 % of the GPU** continuously (desktop capture + NVENC) | you lose remote access to the machine while it is closed |
| `OMNIVOICE_ASR_BACKEND=faster` | verifier reads through CTranslate2 instead of transformers. Speed is a wash, but OmniVoice's own Whisper is never loaded: **3529 MB → 1953 MB** | `pip install faster-whisper` |
| `OMNIVOICE_COMPILE=1` | **do not.** Measured at **2x slower** (RTF 0.342 against 0.170) — an autoregressive model recompiles on every new sequence length and never amortises it | — |

Measured on this machine, the CPU-side audio work is **not** where the time
goes — on a 60 s clip: loudness normalization 126 ms, chunk joining 6 ms,
about **0.2 % of realtime**. Do not tune those; tune steps and verification.

Two things that lower RTF for free: longer scripts amortise the fixed per-call
overhead, and warm voices skip the embedding build. A one-line clip will always
show a worse RTF than a paragraph — that is arithmetic, not a fault.

**Measured on the reference box** (RTX 3060 Ti 8 GB, 16 steps, verification on):
long-form **RTF ~0.17**, 240-word clips ~0.18, 60-word clips ~0.19. Ten minutes
of finished audio renders in under two minutes. Full numbers and method in
[GPU_RUN.md](GPU_RUN.md).

**What is not available here:** upstream PR #239 (FlashInfer, "2.1x at batch
size 1") **cannot run on Windows** — `flashinfer-jit-cache` ships Linux wheels
only, and `flashinfer-python` needs `nccl4py`, which has no Windows build
because NCCL does not exist on Windows. Reaching it requires a Linux
environment; WSL2 on the same machine would qualify.

---

## Testing

Testing on a fresh machine? **[WINDOWS_SESSION.md](WINDOWS_SESSION.md)** is a
paste-ready brief: run order, the three numbers to capture, and the escape
hatch for every change.

```powershell
venv\Scripts\python -m pytest tests -q          REM no GPU needed
venv\Scripts\python tools\acceptance.py --voice narrator_a
venv\Scripts\python tools\audit_batch.py manifest.json
```

- **`tests/`** — 104 unit tests for the text front-end, the audio repair and the
  verifier. Pure Python: they run on a laptop with no CUDA and no weights, which
  is where all three were developed.
- **`tools/acceptance.py`** — run against a live server. Checks the frozen API
  contract, unknown-voice handling, limits, idempotency, the ten text probes,
  the reference-bleed probe, loudness consistency, seed behaviour, four-way
  concurrency, and whether VRAM climbs across requests.
- **`tools/audit_batch.py`** — the measurement that found these bugs, on your own
  finished batch: transcribes every clip and reports words added, words dropped,
  and which clips are responsible.

---

## Environment variables

### Text
| Var | Default | Purpose |
|-----|---------|---------|
| `OMNIVOICE_NORMALIZE_LEVEL` | `full` | `full` / `basic` / `off` |
| `OMNIVOICE_LEXICON` | `./lexicon.json` | pronunciation overrides |
| `OMNIVOICE_YEARS` | `1` | read 1100–2099 as years ("twenty twenty-four") |
| `OMNIVOICE_NUM_STYLE` | `us` | `us` = "one hundred fifty" · `uk` = "one hundred and fifty" |
| `OMNIVOICE_CHUNK` | `1` | sentence chunking for long text |
| `OMNIVOICE_MAX_CHARS` | `200` | target characters per chunk (100–250 is the recommended band) |
| `OMNIVOICE_MIN_CHUNK_CHARS` | `MAX_CHARS/3` | never emit a chunk shorter than this — short generations are where upstream reports the voice switching speaker |
| `OMNIVOICE_SPACE_BEFORE_PUNCT` | `0` | upstream #116 workaround for swallowed final consonants; A/B it before trusting it |

### Reference audio
| Var | Default | Purpose |
|-----|---------|---------|
| `OMNIVOICE_MAX_REF_SEC` | `10` | trim threshold (cut at the nearest pause below it) |
| `OMNIVOICE_REF_TAIL_SILENCE` | `0.30` | silence appended to a reference |
| `OMNIVOICE_REF_MIN_KEEP_SEC` | `3.0` | shortest clip we will cut back to when one ends mid-word |
| `OMNIVOICE_REF_HARD_MAX_SEC` | `15` | how far past the target we may go to let a sentence finish |
| `OMNIVOICE_REF_LUFS` | `-20` | reference loudness target |
| `OMNIVOICE_STRICT_REF` | `1` | reject a `ref_text` that does not match the audio |

### Output & checking
| Var | Default | Purpose |
|-----|---------|---------|
| `OMNIVOICE_VERIFY` | `1` | transcribe every clip back and diff it |
| `OMNIVOICE_VERIFY_MODE` | `fast` | `fast` = one ASR pass over the finished clip, drilling into chunks only on failure · `strict` = one pass per chunk |
| `OMNIVOICE_VERIFY_BUDGET` | `45` | seconds; over budget the audio still ships, unverified |
| `OMNIVOICE_VERIFY_RETRIES` | `1` | regenerations per bad chunk |
| `OMNIVOICE_NORMALIZE_OUTPUT` | `1` | loudness-normalize the output |
| `OMNIVOICE_OUT_LUFS` | `-20` | output loudness target |
| `OMNIVOICE_OUT_PEAK_DB` | `-1.0` | true-peak ceiling |
| `OMNIVOICE_OUT_TAIL_PAD` | `0.30` | silence appended to each clip so downstream trimming cannot eat the last consonant |
| `OMNIVOICE_LEVEL_MATCH` | `1` | pull outlier chunks toward the clip's own median level |
| `OMNIVOICE_RATE_LOW` / `_HIGH` | `0.75` / `1.30` | wpm band around a voice's own baseline (warning only) |

### Limits & stability
| Var | Default | Purpose |
|-----|---------|---------|
| `OMNIVOICE_MAX_INPUT_CHARS` | `8000` | **synchronous** `/api/tts` only → `413`, pointing at the async endpoint |
| `OMNIVOICE_MAX_INPUT_WORDS` | `1300` | same, ~8 min of speech |
| `OMNIVOICE_MAX_INPUT_CHARS_ASYNC` | `100000` | the Studio queue and `/api/tts/async` — paste a whole chapter |
| `OMNIVOICE_MAX_INPUT_WORDS_ASYNC` | `18000` | same |
| `OMNIVOICE_MAX_CONCURRENCY` | `1` | simultaneous generations |
| `OMNIVOICE_QUEUE_WAIT` | `300` | seconds a caller waits before `429` |
| `OMNIVOICE_AUTO_RELOAD` | `1` | reload the model after a fatal CUDA error |
| `OMNIVOICE_SELFTEST` | `1` | background self-test that backs `/api/ready` |
| `OMNIVOICE_SELFTEST_EVERY` | `120` | seconds between self-tests |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | set automatically; **ignored by torch on Windows** — `GET /api/health` reports whether it took effect |

### Provenance & compliance
See **[RESEARCH.md](RESEARCH.md) §5** — EU AI Act Article 50 has been enforceable
since 2 August 2026.

| Var | Default | Purpose |
|-----|---------|---------|
| `OMNIVOICE_WATERMARK` | `0` | inaudible AudioSeal mark on generated audio (needs `pip install audioseal`) |
| `OMNIVOICE_WATERMARK_ALPHA` | `1.0` | watermark strength |
| `OMNIVOICE_AUDIT` | `1` | append one line per generation/registration |
| `OMNIVOICE_AUDIT_LOG` | `./logs/generations.jsonl` | where that goes |
| `OMNIVOICE_AUDIT_TEXT` | `0` | log full scripts, not just hashes |
| `OMNIVOICE_REQUIRE_CONSENT` | `0` | refuse voice registration without `consent` + `consent_ref` |
| `OMNIVOICE_PREWARM_VOICES` | `0` | build every saved voice's prompt at startup |

### omnivoice version features
The wrapper feature-detects these, so an older install still runs. `GET /api/health`
reports `omnivoice_version` and which are available.

| Var | Default | Purpose |
|-----|---------|---------|
| `OMNIVOICE_ASR_DEVICE` | _(unset)_ | where Whisper loads (0.2.1+); `cpu` frees VRAM at the cost of speed |
| `OMNIVOICE_PAD_DURATION` | _(model default)_ | the model's own trailing silence (0.2.0+) — raise it if final consonants are being clipped |
| `OMNIVOICE_FADE_DURATION` | _(model default)_ | the model's own fade (0.2.0+) — upstream #194 reports it "sometimes adds artifact" |
| `OMNIVOICE_FLASHINFER` | `0` | 2–2.9× lossless speedup (upstream PR #239). Needs omnivoice from git main plus `flashinfer-python`; see requirements.txt |

### Model, server, API
| Var | Default | Purpose |
|-----|---------|---------|
| `OMNIVOICE_MODEL` | `./Model` if present | model path or HF id |
| `OMNIVOICE_ASR_MODEL` | `openai/whisper-large-v3-turbo` | Whisper (local path ok) |
| `OMNIVOICE_NUM_STEP` | _(unset)_ | force inference steps |
| `OMNIVOICE_ATTN` | `sdpa` | attention backend |
| `OMNIVOICE_OUTPUT_DIR` | `./outputs` | where clips auto-save |
| `OMNIVOICE_VOICES_DIR` | `./voices` | voice library |
| `OMNIVOICE_ALLOWED_PATHS` | _(unset)_ | extra folders the browser may download from |
| `GRADIO_SERVER_NAME` / `_PORT` / `GRADIO_SHARE` | `0.0.0.0` / `7860` / `0` | UI server |
| `OMNIVOICE_AUTH` | _(unset)_ | `user:pass` UI login |
| `OMNIVOICE_API` / `_API_PORT` | `1` / `8001` | REST API |
| `OMNIVOICE_API_KEY` | _(unset)_ | single shared key |
| `OMNIVOICE_API_KEYS` | _(unset)_ | `k1:acme,k2:globex` — per-customer voice isolation |
| `OMNIVOICE_STRICT_PARAMS` | `0` | `1` = reject unknown request params with `400` |
| `OMNIVOICE_HEALTH_STRICT` | `1` | `/api/health` returns `503` when generation is broken |
| `OMNIVOICE_IDEMPOTENCY_TTL` | `86400` | seconds a replay stays available |

---

## Tips
- Reference audio: **6–10 s**, clean, single speaker, **ending on a finished
  sentence**. That last part is not a style note — see the playbook.
- For max speed, drop **Quality steps** to 8–10.
- Leave **Language = Auto** unless auto-detect picks wrong.
- RTF is lower (better) on longer scripts; short one-liners show a higher RTF
  due to fixed per-call overhead.
- If throughput drops over a long day, check `GET /api/health` → `vram` →
  `fragmentation_mb` before blaming the model.
