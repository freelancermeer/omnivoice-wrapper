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

## Long-form, which is what this is actually sold for

A 545-word script (non-repeating prose, numbers and currency in it) through
`POST /api/tts`:

| | |
|---|---|
| audio produced | **203.3 s** (3 min 23 s) from 61.6 s of wall time |
| RTF | 0.300, **verified** |
| words sent / spoken | 545 / 546 |
| **word accuracy** | **0.9982** |
| real drops | **0** |
| trailing artefact (reference bleed) | **none** |
| VRAM delta | +0 MB allocated, +0 MB reserved |

A 287-second (4 min 47 s) clip behaved the same: `X-Verified: true`, +8 MB
allocated, +0 MB reserved. Neither of these could have been verified at all
before the 30-second fix above.

Two things came back on the 545-word clip, and both are worth knowing about:

* One genuine extra word — "dollars", where the script said "ninety million"
  and the model said "ninety million dollars". That is the stochastic behaviour
  the docs promise will never fully go away. The point is that it arrived as a
  warning on the response rather than as a customer complaint.
* `theatre` heard as `theater`, filed as a pronunciation note. Along with
  `authorisation` → `authorization` in the batch audit, that is a pattern: if
  your scripts use **British spelling, expect pronunciation notes**. They are
  cosmetic — no regeneration is spent on them — but they will appear.

### RTF grows with clip length, because verification does

| clip length | RTF verified |
|---|---|
| ~73 s | 0.203 |
| ~203 s | 0.300 |
| ~287 s | 0.277 |

Generation itself is flat; Whisper's long-form pass is what scales. So budget
**~0.20 for clips around a minute and ~0.29 for multi-minute clips**, verified.

---

## The decision that was taken

**Verification stays on.** RTF 0.151 → 0.203 is the price, and it buys a server
that cannot drop a word silently. Two things make that cheaper than it looks:
regenerations went from 25-in-49 to 0, so the wasted work that used to sit on
top of verification is gone; and the failure it prevents costs a re-cut video,
not a re-render.

`OMNIVOICE_VERIFY=0` remains the lever if a deadline needs it, with
`tools/audit_batch.py` as the after-the-fact check.

FlashInfer was deliberately **not** attempted: 0.2.1 has just been verified end
to end, and moving to git main would give that up for a speedup that is
unmeasured on Ampere. Watermarking is likewise still untouched.

---

## Housekeeping for whoever runs this next

Re-registering a voice creates a **new** id rather than overwriting, so the
library now holds both generations:

| keep | delete when convenient |
|---|---|
| `RVoiceover_1_2`, `RVoiceover_2_2`, `RVoiceover_3_2` | `RVoiceover_1`, `RVoiceover_2`, `RVoiceover_3` |

The originals still carry the pre-fix `ref_text` — `RVoiceover_3`'s still ends
on "forcing". They were left in place rather than deleted, but **do not pick
them for real work**: they are the broken references, kept only as evidence.

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


---

# Second session — making verification cheap, and checking it against a
# second transcriber

Same machine, same day. The question this time was the one left open above:
**RTF matters, so can verification cost less rather than be switched off?**
The answer turned out to be yes, and three more false alarms fell out of it.

## The verifier's ASR was running in its slowest possible mode

Whisper can read a long clip two ways. The wrapper used the sequential
long-form path (`return_timestamps=True`), which decodes one 30 s window after
another. The pipeline will instead take the windows as a **batch**.

Measured on a 203 s clip, same model, same weights (`bench_asr.py`):

| mode | time | realtime factor | words |
|---|---:|---:|---:|
| sequential, timestamps (was) | 10.81 s | 18.8x | 543 |
| chunked, batch=4 | 5.57 s | 36.5x | 543 |
| chunked, batch=8 (now default) | 5.00 s | 40.7x | 543 |
| chunked, batch=16 | 4.12 s | 49.4x | 543 |

Word accuracy of every batched result against the sequential one: **1.0000,
zero dropped, zero inserted.** This is free speed, not a quality trade.

The per-chunk verification loop had the same shape — one model call per chunk,
each waiting for the last — and now issues a single batched call up front
(`_transcribe_many`). Both fall back to the old path if the build cannot do it.

**Long-form RTF went 0.300 to 0.226** on an identical 545-word script, wall
time 61.6 s to 46.3 s, with byte-identical warnings. `OMNIVOICE_ASR_BATCH`
tunes it; VRAM peak rises about 1.4 GB at batch=8, which this card has.

## Three more false alarms, each costing a real regeneration

### 7. Whisper writes clock times with a full stop

`7:45` comes back as `7.45`, which reads as a decimal and expands to "seven
point four five" against the script's "seven forty-five". One clip scored
**three dropped words and seven inserted ones** while having said every word
correctly — and paid for a regeneration: RTF **0.548**, against 0.244 for the
same clip once fixed.

`verify.word_diff` now scores the alternative reading too and keeps whichever
matches better. It can only ever remove a false alarm: a genuine decimal is not
improved by reading it as a clock time, so it stays a decimal.

### 8. A currency symbol the transcriber inferred

Given "$4.2 million" earlier in a sentence, a transcriber writes the following
bare "3 million" as "$3 million" — and the expansion adds a "dollars" the
script never contained. That cost a regeneration too (RTF 0.353 against 0.183).

What settled it as notation rather than the model speaking: **Whisper and
AssemblyAI independently made the same inference on the same clip.**

Removing every currency mark is no better than keeping them all when the script
genuinely contains one — the error just changes from an inserted "dollars" to a
dropped one. So they are removed one at a time, greedily, keeping only a removal
that reduces the error count. A "dollars" the model really failed to say is
still caught, including when one of two is dropped.

### A metric bug found while fixing those two

`word_accuracy` is matched-over-sent, so **it cannot see an insertion at all** —
a clip that says every word it was sent plus a spurious "dollars" scores 1.000.
Choosing between two readings on accuracy alone therefore never fired for the
currency case. `_errors()` counts both directions.

## Checked against a completely different transcriber

The built-in verifier reads generated audio with Whisper, which means one model
marking its own homework: if Whisper mishears the same word the model
mispronounced, the two errors cancel. `tools/verify_external.py` sends the same
clips to AssemblyAI and diffs both with the same `verify.word_diff`.

Ten fresh clips (`tools/sample_scripts.json`, deliberately varied — plain
narration, currency, proper nouns, clock times, a two-word line, a long literary
sentence):

| | clean | mean accuracy |
|---|---:|---:|
| Whisper (built in) | **10/10** | 0.9952 |
| AssemblyAI (independent) | 9/10 | 0.9928 |
| the two agree | **9/10** | |

The single disagreement is AssemblyAI's own error: it heard "lift the **cover**
away from you" as "lift the **COVID** away from you". Our verifier was right.

Before the two fixes above, the same batch scored Whisper 9/10 and AssemblyAI
7/10 with agreement on 8/10 — and both flagged the same spurious "dollars".

**Note:** this uploads audio to a third party. The key is read from
`ASSEMBLYAI_API_KEY` and is never written to a file.

## The reference repair, tested on a voice that never existed before

A clip was cut at exactly 10.000 s — the same stopwatch cut that caused the
original 218-word bug — and registered as a new voice:

```
duration_sec : 8.74s  (from 10.00s)
ref_text     : ...ear that the committee had lost control of its own schedule.
WARNING      : your clip stopped while you were still speaking, so it was cut
               back to the last finished phrase (8.5s of 10.0s kept)
```

Three clips generated from it: **0/3 showed any reference bleed.** The
protection is automatic on every registration, not something done by hand to
the three original voices.

## What the API does when several callers arrive at once

| | per-clip RTF | outcome |
|---|---|---|
| 1 request alone | 0.234 | — |
| 4 at the same instant | 0.212–0.222 | 4/4 succeeded |
| 8 at the same instant | 0.210–0.221 | 8/8 succeeded |

**Per-clip RTF does not move under load, and nothing is refused.** Throughput
RTF across 8 simultaneous callers was 0.217, against 0.234 for one alone — the
queue costs nothing. What grows is waiting: the eighth caller waited 19.4 s
behind seven clips.

`POST /api/tts/async` returns in **0.01 s** and the client polls
`GET /api/jobs/{id}`; five jobs finished in 10.3 s. That is the right path for
anything submitting more than one clip. It intentionally does not take
`format` — the warning saying so is the unknown-parameter mechanism working.

## RTF variance is this desktop, not the code

Every ladder run showed one slow rung per repeat — a 60-word clip rendering in
4.16 s and then 7.58 s on identical input.

The first hypothesis was the 120 s readiness self-test stealing the GPU slot.
**That was wrong.** The probe was made idle-only anyway (a real request that
succeeded seconds ago is better evidence than a synthetic one, so running it on
a busy server is not merely redundant but harmful) — and the outliers survived
unchanged.

The actual cause: **fifteen other processes are using this GPU** — explorer,
two Edge WebView instances, Parsec, WPS Office, Claude desktop, Phone Link,
Search, Start Menu. Twelve identical 60-word requests spread 4.17–5.51 s, with
temperature climbing 52 to 75 °C and clocks steady at ~1875 MHz, so nothing is
throttling; the contention simply lands mid-run.

**The numbers in this document are a floor, not a ceiling.** On a machine
without a desktop on it, expect the fast end of each range.

## The machine shut down by itself, twice

Not the app. The queue's optional shutdown runs `shutdown /s /t 60`, which logs
a clean **Event 1074**. What the log actually shows is **Event 41** ("rebooted
without cleanly shutting down") and **6008** ("the previous system shutdown was
unexpected"), at 12:34 AM on 24 Aug and again at 2:34 AM on 23 Aug — both under
sustained GPU load.

No WHEA errors, and the GPU reads 49 °C at idle. Kernel-Power 41 with no
hardware error logged and a cool card points at **power delivery** rather than
temperature: the 3060 Ti's transient spikes are large, and a marginal PSU trips
under sustained inference. Worth trying `nvidia-smi -pl 150` from an
administrator prompt before a long batch; it caps the spikes for very little
RTF.

## Where RTF stands now

| | RTF |
|---|---:|
| long-form, 203 s clip, verified | **0.221–0.230** (was 0.300) |
| 240-word clips, verified | **0.178–0.185** (was 0.198–0.208) |
| pooled 60 words and up, verified | **0.211** |
| unverified, for reference | 0.151 |

Verification now costs roughly **25–40 %** depending on clip length, down from
a straight 35 % — and the three false-alarm fixes remove regenerations that
were costing far more than the checking itself.

The one large lever left is upstream **PR #239** (FlashInfer, "2.1x at batch
size 1"), which needs `omnivoice` from git main rather than 0.2.1. It was
measured on an H100; this card is Ampere. Not attempted here.


---

## FlashInfer: attempted, and it cannot run on Windows at all

`RESEARCH.md` §8 held this out as the one large RTF lever left — upstream
[PR #239](https://github.com/k2-fsa/OmniVoice/pull/239), "2-2.9x lossless
speedup", 2.1x at batch size 1. It was tried on this machine. It is not
available here, and the blocker is not OmniVoice.

Three independent walls, in the order they were hit:

1. **`flashinfer-jit-cache` publishes Linux wheels only.** Every file on the
   cu128 index is `manylinux_2_28`, x86_64 and aarch64 alike. There is no
   Windows build of any version.
2. **`flashinfer-python` will not install either** — it requires
   `nccl4py>=0.3.1`, which has no Windows distribution.
3. **NCCL does not exist on Windows.** That is the root of (2), and it is not
   something a wrapper can work around.

So the install failed cleanly and `omnivoice` stayed at 0.2.1; nothing needed
rolling back. **Do not budget for FlashInfer on this box.** The honest way to
reach it is a Linux environment — WSL2 with CUDA passthrough on this same
machine would qualify, and would also be worth measuring against the VRAM leak
in upstream #199.

## The RTF jitter has one cause, and it is not fifteen processes

An earlier note in this document blamed "fifteen other processes" using the
GPU. That was loose. Sampling `\GPU Engine(*)\Utilization Percentage` per
process while a TTS load ran gives a much sharper answer:

| process | engine | avg % |
|---|---|---:|
| python (this server) | 3d | 32.7 |
| **parsecd** | 3d | **13.4** |
| **parsecd** | videoencode (NVENC) | **11.3** |
| claude | 3d | 0.8 |

**Parsec alone is taking ~25 % of the GPU**, continuously — desktop capture on
the 3d engine plus NVENC encoding. Everything else on the desktop (Edge
WebView, WPS, Claude, explorer, Start Menu) is rounding error at under 1 %.

That is the whole jitter story: identical 60-word requests spread 4.17-5.51 s
because a quarter of the card is being spent encoding video for a remote
session. Closing Parsec when nobody is remoting into the box is the single
cheapest RTF win available on this machine — larger than anything left in the
code.

Caveat worth stating plainly: if the box is *administered* over Parsec, closing
it costs the operator their access. It is a deployment decision, not a code
change.

## Steps stay at 16

A `num_step` sweep was offered and declined — 16 gives the voice quality this
product is being sold on. Recorded here so nobody re-opens it: the RTF numbers
in this document are all at **16 steps**, and lowering steps was not traded
against quality because the quality was judged acceptable as-is.


---

## "The UI is full of bugs" was one instance too many

Reported as a Gradio problem. It was not Gradio, and the log said so:

```
ERROR: [Errno 10048] ... bind on address ('0.0.0.0', 7860) ...
ERROR: [Errno 10048] ... bind on address ('0.0.0.0', 8001) ...
Port 7860 busy, trying 7861 ...
```

`run.bat` was started while a server was already running. What that produced:

* the second instance **could not bind the API port**, so `/api` was dead for
  it — while its own banner still printed `API (local): http://127.0.0.1:8001`
  as though it worked;
* the UI walked to **7861**, so the address in the banner was not the one being
  opened;
* and worst, **a second 3.3 GB model had already been loaded** onto an 8 GB
  card that was now hosting two of them.

Everything after that looks like a buggy UI.

### Fixed: it now refuses to be the second instance

`_refuse_if_already_running()` runs **before the model loads** — which is the
whole point, since by the time the sockets are bound the expensive and damaging
half has already happened. Measured: the second start now exits in **8.5 s
having allocated nothing**, with VRAM unchanged at 4224 MB.

```
==============================================================
  OmniVoice is ALREADY RUNNING on port 8001.
  Nothing was started, and no second model was loaded.
  Open the running one:  http://127.0.0.1:7860
  To restart instead, close the other window first.
==============================================================
```

A port held by something that is *not* OmniVoice gets a different message
pointing at `OMNIVOICE_API_PORT`.

### Fixed: the banner no longer advertises an API that failed to bind

`_start_api_server` logged the bind failure at warning level and carried on.
The banner then printed the API URLs regardless. That is precisely how a
half-dead instance passes for a working one, so the failure is now **printed**,
loudly, saying the UI still works and anything calling `/api` will not.

### The UI itself was exercised end to end and is fine

Driven through a real browser: voice dropdown repopulated with the three
re-registered voices (`demo.load` does its job), voice selected, sample and
transcript loaded, script queued, rendered, and the finished card showed
`RTF 0.242 · 9.1 s audio · 2.2 s to make · 170 wpm · -20.0 LUFS`. The clip
verified at **word accuracy 1.0000**.

That clip is also a live test of today's clock-time fix: the script said
"4:30", Whisper wrote "4.30", and the verifier scored it clean instead of
inventing three dropped words.

**One real quirk, not worth fighting.** Gradio's `gr.Timer` pauses while the
browser tab is hidden, so a queue left in a background tab looks frozen —
counters and cards both stale — and catches up the moment the tab is focused
again. Confirmed by faking `document.visibilityState`: the board went from
"1 Queued / Processing" to "0 Queued / 1 Done" immediately. Nothing is lost,
rendering continues server-side, and the display self-corrects. Worth knowing
before someone reports it as a hang.

## Voices were rebuilt from scratch

All eleven entries — the original seven from the old code plus four created
during testing — were deleted, and the three real voices re-registered from
their original 10.000 s clips so the repair ran fresh rather than inheriting
anything. The old `voices/` folder was copied to `voices_backup_<timestamp>/`
first, because those clips are the only copies that exist.

| voice | from | to | quality | wpm | note |
|---|---|---|---|---|---|
| RVoiceover_1 | 10.00 s | 9.82 s | 1.00 | 147 | clean |
| RVoiceover_2 | 10.00 s | 9.18 s | 0.85 | 196 | cut back from mid-word |
| RVoiceover_3 | 10.00 s | 7.52 s | 0.85 | 160 | the "forcing" clip |

Names are plain again — no `_2` suffixes. Re-verified on the new library:
35 acceptance checks pass, and the ten-clip batch audits **0 added, 0 dropped,
10/10 clean**.


---

# Chasing the verifier's cost, and where it actually went

"RTF is bad, make the verifier fast." The measurements below are the answer,
including the two experiments that did **not** work — which are the useful part.

## First, the number that was being misread

A batch of ten mixed clips reports `RTF avg 0.347`. That average includes a
1.6 s clip whose RTF is 1.03, because a clip that short pays the same fixed
per-request cost as a 73 s one while producing almost no audio. Its wall time
is 1.7 s. Judging the server on that average is judging it on arithmetic.

Measured properly, same batch, verification the only variable:

| | wall | RTF avg | RTF best |
|---|---:|---:|---:|
| verification **on** | 27 s | 0.325 | 0.174 |
| verification **off** | 20 s | 0.232 | 0.125 |

**Verification costs 7 s in 27, about 35 %** on a short-clip batch, and **12.6 %**
on long-form (0.170 against 0.151). The remembered "0.15" is still there; it is
the unverified number, and always was.

## Where that 7 s is not

**It is not the regenerations.** `/api/metrics` showed 7 regenerations in 61
generations, which looks like 11 % pure waste. Reading the audit log, every
regeneration-triggering warning from earlier in the day was a number the
transcriber wrote in digits — `not spoken: sixth / extra: 6th`,
`not spoken: ten / extra: 10`, `not spoken: two hundred three hundred /
extra: 200 300`. Re-running all ten of those cases against the current code:
**nine are already fixed** by today's notation work. Those log lines were
history, not a live problem.

The tenth was live and is now fixed too — see below. And the one regeneration
in the latest batch turned out to be the system **working**: the final warning
list for that clip was clean, meaning a chunk really did come out wrong, was
regenerated, and the second attempt was right. That is not waste; that is the
product. Removing it means shipping the defect.

### A bare inflection was buying a re-generation

`memory` against `memories` scores 0.714 on raw similarity — just under the
0.72 artefact threshold — so it counted as a real substitution and bought a
full re-generation that could not have fixed it. `likely_asr_artifact` now
matches on a stripped inflection first.

CRITICAL_WORDS is still checked before that, so nothing that changes meaning
gets waved through. Verified explicitly: `ninety million` → `90 billion`,
`not official` → `now official`, `nothing` → `something`, `first` → `third`,
and a dropped `dollars` are all still caught.

## Two experiments that did not pay

### Overlapping verification with the next generation — rejected on reasoning

The obvious idea for queue work: verify clip N while generating clip N+1, and
the checking becomes free. It does not, and it is worth writing down why so it
is not attempted again. **Both halves are GPU-bound on the same GPU.** The 7 s
of verification is 7 s of real GPU work; overlapping it with generation does
not reduce the total work, it only interleaves it. Pipelining pays when one
side is waiting on something else — here nothing is. Not built.

### A faster ASR runtime — built, measured, kept for a different reason

`faster-whisper` (CTranslate2) on the same large-v3-turbo weights is genuinely
faster in isolation: over 113.7 s of audio, **6.71 s → 3.06 s**, and the `base`
model is faster again at 1.72 s. On injected defects — a dropped clause, a
dropped sentence, three dropped words — `base` caught **10/10 of each**, missing
nothing large-v3-turbo caught, and called all ten untouched clips clean.

End to end, it changed almost nothing:

| | batch of 10 | long-form 203 s | VRAM allocated |
|---|---:|---:|---:|
| transformers (default) | 27 s | RTF 0.170 | 3529 MB |
| faster-whisper | 26 s | RTF 0.168 | **1953 MB** |

The reason is the morning's work: batching the verifier's windows had already
taken the ASR to ~40x realtime, so it stopped being the bottleneck. Swapping
runtimes cannot recover time that is no longer being spent there.

What it *does* buy is **1.6 GB of VRAM**, because OmniVoice's own Whisper is
not loaded when nothing calls it. On an 8 GB card shared with a desktop that is
worth having, so it stays — as `OMNIVOICE_ASR_BACKEND=faster`, off by default,
since it needs `pip install faster-whisper` and the checker is the last thing
that should change without someone deciding to. Quality was re-audited on it:
**0 added, 0 dropped, 10/10 clean.**

## So what is left

Verification costs what it costs, because reading every clip back is the thing
that stops a dropped word reaching a customer. The levers that remain are not
in the verifier:

| lever | worth | where |
|---|---|---|
| **close Parsec** | ~25 % of the GPU, continuously | deployment, not code |
| **FlashInfer** | claimed 2.1x at batch size 1 | needs Linux; impossible on Windows |
| verification off | 0.170 → 0.151 long-form | `OMNIVOICE_VERIFY=0` + `audit_batch.py` after |

The third is a real option and an honest one — the batch audit catches the same
defects, just after the run instead of during it. It is a decision about when
you want to find out, not whether.
