# What the first GPU run actually measured

23–24 August 2026 · Windows 11 · RTX 3060 Ti 8 GB · CUDA 12.8 ·
torch 2.8.0+cu128 · omnivoice **0.2.1** · Python 3.12.10

Everything in `FIXES.md` and `RESEARCH.md` up to this point was reasoning
plus 95 unit tests. This is the first time any of it ran on a GPU. Every number
below was produced on this machine and can be reproduced with the commands
given; nothing here is an estimate unless it says so.

Six defects were found and fixed during the run. Four of them could only have
been found on hardware.

---

## The three numbers that were asked for

### 1. RTF — **0.203 verified, 0.151 unverified**

A single RTF figure is meaningless for this server, because a fixed per-request
cost is spread over however much audio the clip contains. So the measurement is
a ladder (`tools/rtf_probe.py`, two runs per rung, 16 steps, voice
`RVoiceover_3_2`):

| words | audio | RTF verify=on | RTF verify=off | verification costs |
|------:|------:|--------------:|---------------:|-------------------:|
| 12 | 3.9 s | 0.489 | 0.341 | +43 % |
| 30 | 9.9 s | 0.227 | 0.151 | +50 % |
| 60 | 22.5 s | 0.188 | 0.142 | +32 % |
| 120 | 43.7 s | 0.212 | 0.154 | +37 % |
| 240 | 72.8 s | 0.203 | 0.151 | +34 % |
| **pooled, 60 words and up** | | **0.203** | **0.151** | **+35 %** |

**The remembered 0.16 was an unverified number.** It reappears exactly as soon
as verification is switched off (0.151). With the verifier doing its job the
honest figure is **0.203**.

Short clips are not slow because anything is wrong — a 3.9 s clip pays the same
fixed setup as a 73 s one. Judge the server on the 60-and-up rows; those are
what real voiceover work looks like.

### 2. Verification's exact cost — **~35 %, not the 15–20 % estimated**

`FIXES.md` guessed 15–20 %. Measured on realistic lengths it is **32–37 %**,
call it a third. That is one extra Whisper pass per clip, under the same lock
as generation, so it costs wall-clock but never a second concurrent VRAM peak.

The trade being bought: 0.151 → 0.203 RTF, in exchange for the server being
able to tell you a word went missing. **This decision is yours** — the lever is
`OMNIVOICE_VERIFY=0`, and `tools/audit_batch.py` can then re-check a whole
batch afterwards in one pass.

### 3. Batch audit — **0 added, 0 dropped, 0 % bleed**

25 clips · 266 words · `tools/make_batch.py` then `tools/audit_batch.py`:

```
words ADDED that were never sent : 0        (was 218, "forcing" x163)
words DROPPED                    : 0        (was 12)
clips with a trailing artefact   : 0  (0%)  (was 86-90% on one voice)
clean clips: 25/25
```

Two words differed, and both were the *transcriber's* spelling, not the model's
speech — `Bessent` written "besant", `authorisation` written "authorization".
The verifier classified both as pronunciation notes rather than drops, and did
not waste a regeneration on either. `audit_batch.py` now reports them on their
own line instead of burying them in the headline count.

---

## Was the reference bug really fixed?

Yes, and the evidence was sitting on disk. Five of the six stored references
were **exactly 10.0000 s** — the fingerprint of the old stopwatch cut.

Re-registering them through the repaired path:

| voice | before | after | quality | LUFS | what it said |
|---|---|---|---|---|---|
| RVoiceover_1 | 10.000 s | 9.82 s | 1.00 | −20.0 | clean, no warning |
| RVoiceover_2 | 10.000 s | 9.18 s | 0.85 | −20.0 | cut back to the last finished phrase |
| RVoiceover_3 | 10.000 s | **7.52 s** | 0.85 | −20.0 | cut back to the last finished phrase |

`RVoiceover_3` is the one that poisoned 163 clips. Its stored `ref_text` used to
end `...his executive orders and forcing.` It now ends `...his own handpicked
judges...`, and the audio was re-transcribed so text and audio agree exactly.
**"forcing" is gone at the source**, and the audit above confirms nothing leaked
downstream.

All three normalised to exactly −20.0 LUFS, against a production spread that ran
−0.2 dB to −12.6 dB.

---

## Six defects found on hardware

### 1. Verification silently did not run on any clip over 30 seconds

The worst of them. Whisper takes 30 s of mel features at a time; past that
`transformers` requires `return_timestamps=True` and raises otherwise. OmniVoice's
own `transcribe()` passes no such argument. The verifier joins every chunk and
transcribes the result in one pass — so **on any clip longer than 30 s the
transcription threw, verification was skipped, and the clip shipped**.

That is precisely the long-form work this product is sold for; the original bug
batch averaged ~62 s per clip. The check that the whole product is built around
was inoperative on the real workload, reporting only a generic "this clip could
not be verified".

`_transcribe_long()` now asks the pipeline for long-form directly, falling back
to 30 s windows. `POST /api/transcribe` was fixed the same way — otherwise
`audit_batch.py` could not audit a long clip either.

Before: `verified=false` at 120 and 240 words. After: `verified=true` at every
rung. This is also why the pre-fix long-form RTF looked good — it was fast
because nothing was checking.

### 2. The verifier failed every clip containing a number

Measured: `not spoken: ninety twenty; extra: 90 20`. The script had been
normalised to "ninety pages"; Whisper wrote "90 pages". Same number, different
notation — but the number words sit in `CRITICAL_WORDS` (rightly: "million"
heard as "billion" must never be excused), so alignment scored it as a drop
*and* an insertion, warned, and bought a regeneration that could not fix
anything.

It affected every notation Whisper uses: `$500`, `11,000`, `4.7`, `6th`, `1st`,
`300%`. In the first acceptance run this produced **22 verify failures and 25
regenerations across 49 generations**.

Fixed by canonicalising both sides through the one normalizer the script had
already been through, so there is a single set of rules — years included.
After: **0 verify failures, 0 regenerations**, batch wall time 57 s → 45 s,
acceptance soak 49 s → 36 s. The guards still hold: "million" vs "billion",
a genuinely dropped number, and the repeated-clause drop are all still caught.

### 3. `1st` → "onest" at `OMNIVOICE_NORMALIZE_LEVEL=basic`

`FIXES.md` lists this bug as fixed, and it is — but only at `level="full"`.
`expand_ordinals` runs inside the full branch while `expand_numbers` runs at
every level, so at `basic` the digit in "1st" was expanded on its own, leaving
**"onest"** (and "twond", "threerd").

`basic` is a documented escape hatch, and it is exactly what someone reaches for
when a number is read wrong — so the failure was waiting in the one setting most
likely to be used to work around it. `expand_numbers` now refuses to split an
ordinal at any level.

### 4. `OMNIVOICE_ASR_DEVICE` was silently dead

`_accepts()` deliberately ignored `**kwargs`. But `OmniVoice.from_pretrained`
is `(path, *args, **kwargs)` and pops every option it has gained since 0.1.5
straight back out — `asr_device` included. So the feature probe reported it
missing, and setting `OMNIVOICE_ASR_DEVICE=cpu` logged *"needs omnivoice >=
0.2.1 (installed: 0.2.1) — ignoring"* and did nothing.

That is the documented mitigation for VRAM pressure, dead on the one card where
VRAM is the binding constraint. `_accepts()` now also recognises a name the
callable pops out of `**kwargs`; a name nothing pops is still reported missing.
The banner now reads as the brief predicted:
`features: asr_device, pad_duration, fade_duration`.

The same bug would have silently disabled `OMNIVOICE_FLASHINFER=1` on git main,
killing the FlashInfer experiment before it started.

### 5. Empty text returned 422, not 400

`_tts_common` has the right check and raises a clean
`400 "text is required"` — but it never ran, because `text: str = Form(...)`
made the field required at FastAPI's validation layer, which rejected with a
pydantic 422 first. The intended 400 was dead code on all three synthesis
endpoints. Now both empty and missing text return `400 {"detail":"text is
required"}`.

### 6. `expandable_segments:True` does nothing on Windows

`run.bat` names this as the cure for "GPU at 5 % and out of memory", and
`README.md`/`FIXES.md` both list it among the OOM fixes. Measured on this box:

```
UserWarning: expandable_segments not supported on this platform
torch 2.8.0+cu128   allocator backend: native
```

torch accepts the variable, warns once into a log nobody reads, and carries on
with the ordinary allocator. **On Windows this mitigation has never been in
effect.**

It is not fixable from here — it is a platform limitation — so it is now
*reported* instead of assumed: `GET /api/health` carries an
`expandable_segments` block, and the startup banner prints a note. What actually
holds fragmentation down on Windows is `concurrency=1` plus the explicit
`del` → `gc.collect()` → `empty_cache()`, and the measurements below suggest
that is enough so far.

---

## Stability

`tools/acceptance.py --voice RVoiceover_3_2`: **35 passed, 0 failed, 1 skipped.**

Over 73 generations in one process:

| | |
|---|---|
| OOM events | 0 |
| model reloads | 0 |
| verify failures / skipped | 0 / 0 |
| allocated after 25-clip soak | **+0 MB** |
| reserved after 25-clip soak | **+0 MB** |
| fragmentation | 151 MB (gate: 800 MB) |
| VRAM steady state | 3521 MB allocated · 3672 MB reserved · peak 3725 MB |

Four concurrent callers: `[200, 200, 200, 200]`, server alive afterwards.

`/api/ready` is backed by a real generation — it reported *"spoke in 2656 ms"*,
and returned 503 with `Retry-After` before the first self-test had run.

**Seed is deterministic on this build**: the same seed produced byte-identical
audio and a different seed produced different audio. Golden-file tests are
therefore meaningful here — `RESEARCH.md` had flagged this as unknown.

A caveat on the VRAM result: upstream [#199](https://github.com/k2-fsa/OmniVoice/issues/199)
reports the leak appearing over *days* of continuous running, and the longest
run here was under an hour. Flat memory across 73 generations is encouraging,
not a refutation.

---

## Still not verified

Being straight about what this run did **not** establish:

- **Watermarking.** `audioseal` is still not installed; `watermark.self_check()`
  has never run. Do this before taking public money — EU AI Act Article 50 has
  been enforceable since 2 August 2026.
- **The VRAM leak over long runs.** See the caveat above.
- **FlashInfer.** Not attempted; it needs omnivoice from git main and the base
  version had to be confirmed first. Defect 4 above would have silently blocked
  it, so it is worth a try now — on Ampere it remains untested upstream.
- **Speaker similarity.** Still nothing measures whether output sounds like the
  customer, and nothing here changed that.
- **Two speakers in one reference.** Still not implemented.
- **The 63-clip production batch.** The original scripts are not in this repo,
  so `tools/make_batch.py` rebuilds a 25-clip batch in the same register
  (numbers, currency, em dashes, repeated clauses, the two lines the bug reports
  named). Re-running the real batch is still worth doing.

---

## Reproducing this

```powershell
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m pytest tests -q                    REM 104 passed
run.bat

curl -X POST http://127.0.0.1:8001/api/voices -F "name=RVoiceover_3" -F "voice=@voices\RVoiceover_3.wav"

venv\Scripts\python tools\acceptance.py --voice RVoiceover_3_2
venv\Scripts\python tools\rtf_probe.py  --voice RVoiceover_3_2 --repeat 2
venv\Scripts\python tools\make_batch.py --voice RVoiceover_3_2
venv\Scripts\python tools\audit_batch.py manifest.json --out audit.json
```

For the verification-cost half, start the server once with
`set OMNIVOICE_VERIFY=0` and run `rtf_probe.py` again.

**One operational note:** stop the server with `taskkill /PID <pid> /T /F`.
A plain terminate left the process holding port 8001 during this session, and a
"restarted" server was quietly still the old one — which invalidated one round
of measurements before it was caught. Confirm with
`netstat -ano | findstr :8001` that the port is free before starting the next
run.
