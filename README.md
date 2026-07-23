# 🎙️ OmniVoice Voiceover Studio

A local, offline-first web app for [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
— a massively multilingual (600+ languages) zero-shot TTS model with **voice
cloning** and **voice design** (based on Qwen3-0.6B). Built with Gradio, it adds a
production-style **project queue**: pick a voice, add scripts, and render them one
at a time, with every clip auto-saved to a folder.

> **Note:** the model weights (`Model/`, ~3.3 GB) and the Python `venv/` are **not**
> in this repo (see `.gitignore`). Follow the setup below to get them on a new PC.

---

## Requirements
- **NVIDIA GPU** with CUDA 12.8 drivers (tested on RTX 3060 Ti 8 GB, FP16).
- **Python 3.10 – 3.12** (developed on 3.12).
- Windows (works on Linux too; adjust the `.bat` launchers to shell scripts).
- ~5 GB disk for the models (OmniVoice ~3.3 GB + Whisper ~1.6 GB).

---

## Setup on a new PC

### 1. Clone the repo
```powershell
git clone https://github.com/<your-user>/<your-repo>.git omnivoice
cd omnivoice
```

### 2. Create a virtual environment
```powershell
py -3.12 -m venv venv
```

### 3. Install PyTorch (CUDA 12.8) + dependencies
```powershell
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install torch==2.8.0 torchaudio==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu128
venv\Scripts\python -m pip install -r requirements.txt
```

### 4. Get the TTS model (`Model/` folder — not in git)
Download the OmniVoice weights into a local `Model/` folder:
```powershell
venv\Scripts\hf.exe download k2-fsa/OmniVoice --local-dir Model
```
The app auto-detects `./Model` and loads from it (fully offline afterwards).

> If you skip this, the app still works — on first launch it auto-downloads the
> model from HuggingFace into the HF cache (needs internet that one time).

The **Whisper** ASR model (`openai/whisper-large-v3-turbo`, ~1.6 GB, used to
auto-transcribe reference clips) downloads automatically on first run. To
pre-cache it: `venv\Scripts\hf.exe download openai/whisper-large-v3-turbo`.

### 5. Run
```powershell
venv\Scripts\python app.py
```
The browser opens automatically. The terminal prints:
```
  Local:    http://127.0.0.1:7860
  Network:  http://<your-ip>:7860   <- share on LAN
```

---

## Running & sharing
This is a **local app** — no cloud, no session limits. Everything runs on your PC.

| Launcher | What it does |
|----------|--------------|
| **`run.bat`** (or Desktop shortcut) | Starts the app. Reachable at `127.0.0.1:7860` **and** on your LAN at `<pc-ip>:7860` (binds `0.0.0.0` by default). |
| **`run_share.bat`** | Also creates a temporary **public** `https://xxxxx.gradio.live` link (needs internet; 72 h). |

- **LAN:** others on the same Wi-Fi open `http://<your-ip>:7860`. If it doesn't
  connect, allow the port in Windows Firewall (admin CMD):
  `netsh advfirewall firewall add rule name="OmniVoice" dir=in action=allow protocol=TCP localport=7860`
- **Local only:** set `GRADIO_SERVER_NAME=127.0.0.1`.
- **Password:** set `OMNIVOICE_AUTH=user:pass` before launching (recommended when sharing).
- **Auto-start on login:** `Win+R` → `shell:startup` → drop a copy of the Desktop shortcut there.

**Works fully offline:** Gradio serves its UI from the installed package (no CDN),
the models are local, and the `.bat` files set `HF_HUB_OFFLINE=1`. Only
`run_share.bat` (public link) needs internet.

### 🔌 Local REST API
The app also exposes a **REST API** on port **8001** (same process, shared model &
queue) so any device/script can generate audio programmatically. Full spec +
examples in **[LOCAL_API.md](LOCAL_API.md)**; interactive docs at
`http://<pc-ip>:8001/api/docs`.

```bash
# text + voice clip -> mp3
curl -X POST http://<pc-ip>:8001/api/tts \
  -F "text=Hello from the API" -F "voice=@my_voice.wav" -F "format=mp3" -o out.mp3
```
Endpoints: `POST /api/tts` (sync), `POST /api/tts/async` + `GET /api/jobs/{id}`,
`POST /api/voices` / `GET /api/voices` / `DELETE /api/voices/{id}` (voice library),
`GET /api/health`.
Disable with `OMNIVOICE_API=0`; secure with `OMNIVOICE_API_KEY=<key>`.

---

## Using the app (single page)
1. **Pick a voice** — upload a 3–10 s clip (auto-trimmed + auto-transcribed by
   Whisper). Leave empty for a designed/AI voice. The voice **stays loaded** and is
   cached, so adding many scripts reuses it without re-cloning.
2. **Add your script** — enter a Project name + Script → **Add to queue**. Or use
   **🧩 Add several at once**: an editable table (Project name + Script per row),
   **➕ Add row** for unlimited rows, then **Add all rows to queue** (same voice).
3. **Settings (optional)** — language, quality steps (16 = fast, 32+ = better),
   speed, guidance, duration, denoise, pre/post-process.
4. **Render queue** (right) — jobs process **one at a time** with live status
   (⏳ Queued → 🔊 Processing → ✅ Done / ❌ Error / 🚫 Cancelled), time · RTF, and
   each card shows the exact voice transcript used. Select a project to
   **Cancel / Remove**; plus **Cancel all queued** / **Clear finished**.
5. **Save & downloads** — every render **auto-saves** to your chosen folder (no
   clicking). Also **📦 Zip all completed** and per-file downloads.
6. **🛑 Shut down PC when the queue finishes** — optional; shuts down ~60 s after
   the last job (cancel with `shutdown /a`).

---

## Built-in fixes for known OmniVoice issues (in [`app.py`](app.py))
| Issue | Fix |
|-------|-----|
| Long reference audio degrades quality (#50) — trained on 3–10 s clips | Reference auto-trimmed to `OMNIVOICE_MAX_REF_SEC` (default **10 s**). Warns if < 3 s. |
| `"123"` garbled | `num2words` front-end, language-aware (en/fr/es/de/ru/pt/it/ja/ko/ar/…); non-Latin left untouched. |
| Phone / long digit runs wrong | Runs ≥ 7 digits read **digit by digit**; decimals → "three point one four". |
| Slow on consumer GPUs | FP16, **steps default 16**, TF32 on; status shows time + **RTF** (<1 = faster than realtime). |
| Long text → non-speech "scratching" (#144) | Split on sentences into ~`OMNIVOICE_MAX_CHARS` (100) chunks, rendered separately + concatenated. |
| Voice drifts between chunks (#44) | One cached `VoiceClonePrompt` reused across all chunks/scripts; design mode locks voice from the first chunk. |
| Reference transcript needed | Leave it empty → multilingual Whisper (`whisper-large-v3-turbo`) auto-transcribes. |

---

## Environment variables
| Var | Default | Purpose |
|-----|---------|---------|
| `OMNIVOICE_MODEL` | `./Model` if present, else `k2-fsa/OmniVoice` | model path or HF id |
| `OMNIVOICE_ASR_MODEL` | `openai/whisper-large-v3-turbo` | Whisper ASR (local path ok) |
| `OMNIVOICE_OUTPUT_DIR` | `./outputs` | where voices auto-save |
| `OMNIVOICE_MAX_REF_SEC` | `10` | reference-audio trim threshold |
| `OMNIVOICE_NORMALIZE` | `1` | number normalization on/off |
| `OMNIVOICE_NUM_STEP` | _(unset)_ | force inference steps (e.g. `8` for speed) |
| `OMNIVOICE_MAX_CHARS` | `100` | max characters per chunk |
| `OMNIVOICE_CHUNK` | `1` | sentence chunking for long text |
| `OMNIVOICE_ATTN` | `sdpa` | attention backend (model doesn't support flash-attn) |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | bind address (`127.0.0.1` = local only) |
| `GRADIO_SERVER_PORT` | `7860` | port (auto-tries next if busy) |
| `GRADIO_SHARE` | `0` | `1` = public gradio.live link |
| `OMNIVOICE_AUTH` | _(unset)_ | `user:pass` to require UI login |
| `OMNIVOICE_API` | `1` | run the REST API (`0` to disable) |
| `OMNIVOICE_API_PORT` | `8001` | REST API port |
| `OMNIVOICE_API_KEY` | _(unset)_ | require `X-API-Key` header on the API |

---

## Tips
- Reference audio: **3–10 s**, clean, single speaker (~6 s is the sweet spot).
- For max speed, drop **Quality steps** to 8–10 (slightly lower quality).
- Leave **Language = Auto** unless auto-detect picks wrong.
- RTF is lower (better) on longer scripts — short one-liners show a higher RTF due
  to fixed per-call overhead; that's normal.
