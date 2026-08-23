# What the rest of the world has already learned about shipping this

Research pass, August 2026. The question was: **what is still going to bite us
that our own batch did not show?** Our four bug documents came from one machine,
three voices and 63 clips. This is what everyone else running zero-shot voice
cloning in production has hit, and what regulators have started to require.

Six gaps came out of it. Four are now fixed, two are open risks you should
decide about deliberately. There is also a legal section, because you said you
are selling this to the public — that part is not optional any more.

---

## 1. Chunk stitching was our biggest untested quality risk — **fixed**

This is the one that would have generated complaints.

Long-form TTS work is unanimous that **the seam is audible in most stitched
output**. Measured on competing systems: boundary energy discontinuities around
**28 dB** and **F0 jumps of 67–69 Hz** at chunk edges
([MagpieTTS-LF](https://arxiv.org/html/2606.18485)). And the specific trap:

> **XTTS suffers severe loudness inconsistency due to independent per-chunk gain
> normalization.**

We chunk at 100 characters. A 22-minute video is **hundreds** of seams. Our old
`concat_audio` did the naive thing: hard-butt each chunk against 0.15 s of pure
silence, no fade, no level relationship, whatever leading/trailing silence the
model happened to produce.

`audio_fx.join_chunks` now does four things, each fixing a documented failure:

| | |
|---|---|
| **Even edges** | each chunk trimmed to a consistent 60 ms, then a fixed gap inserted — so inter-sentence pauses are the same length instead of whatever the model left |
| **Level matching** | outliers pulled toward the clip's **own median**, correcting only the excess beyond a 1.5 dB deadband and capped at 6 dB. Deliberately *not* per-chunk normalization to a fixed target — that is exactly the XTTS mistake |
| **Edge fades** | 15 ms in and out, because the boundary click is the usual reason a stitched clip sounds stitched. F5-TTS ships a `cross_fade_duration` for the same reason |
| **Tail padding** | 300 ms on the finished clip |

That last one deserves its own note, below.

## 2. Silence-stripping eats final consonants — **fixed**

[Murmur's TTS guide](https://www.murmurtts.com/blog/ai-voice-skips-words-punctuation-fixes)
lists four places words disappear, and the fourth is not the model at all:

> **Export Processing**: silence removal and trimming clipping final consonants.
> … Leave 250–500 ms of tail padding after each sentence.

We had `postprocess_output=True` (OmniVoice's own "remove long silences") *and*
our own silence stripping, with a threshold 35 dB below the speech peak. A soft
final `/s/` or `/f/` sits right about there.

This is a live candidate for one of the twelve dropped words in the batch —
specifically `"Absolutely damning. Hegseth is a snake"` coming back as
`"damning hegseth is a snake"`, where the clip **opens on the second word**. A
quiet opening word plus aggressive silence removal is exactly that shape.

Now: the edge threshold for generated audio is **45 dB** (10 dB stricter than
for references), and every clip gets 300 ms of tail padding. If you want to test
the hypothesis directly on Windows, generate that line with
`OMNIVOICE_VERIFY=1` and Preprocess/Postprocess **off**, and see whether
"Absolutely" comes back.

## 3. Voice prompts were rebuilt after every restart — **fixed**

> If you skip caching and recompute the embedding per request, you add 50–200 ms
> of GPU time per request for no benefit.
> — [Spheron production guide](https://www.spheron.network/blog/self-host-voice-cloning-gpu-cloud-xtts-f5-tts-openvoice-v2/)

We cache per voice, but lazily — so the first call after every restart pays it,
and on this server "after every restart" means after every OOM recovery.
`OMNIVOICE_PREWARM_VOICES=1` builds them all in the background at startup.

The same guide independently confirms two things we already do: a
`threading.Lock()` around inference because **autoregressive inference is not
thread-safe**, and a **2 GB VRAM safety margin**.

## 4. The upstream project confirms our root cause — **already fixed**

[OmniVoice issue #50](https://github.com/k2-fsa/OmniVoice/issues/50), in the
maintainers' own words:

> With 60 seconds of reference audio, the output **sounds like the speaker is
> having a stroke and fails to output about 1/4th of the words**, whereas shorter
> reference audio (around 6 seconds) produces great results.

Two things follow.

First, it independently confirms the reference-length fix is the highest-value
thing in this repo — and that **reference problems show up as dropped words**,
not only as leaked ones. Some of our twelve drops may share a cause with our 218
insertions.

Second, a caution about the 15 s ceiling you asked for. **~6 s is the documented
sweet spot**; the model is trained on 3–10 s. The overshoot penalty is set so
that a pause under the target always beats one over it, and 15 s is only ever
reached when there is nothing earlier — which still beats a mid-word cut or a
4 s reference. But if you ever see quality drop on a voice whose reference came
out at 12–15 s, lower `OMNIVOICE_REF_HARD_MAX_SEC` and re-register it first.

Also worth knowing: [issue #31](https://github.com/k2-fsa/OmniVoice/issues/31)
reports static/distortion at default settings, and #162 reports speaker
similarity collapsing to **0.22** on cross-lingual synthesis. If you sell
"clone in any language", test that specific claim before you make it.

## 5. Consent, watermarking and audit — **implemented, off by default**

This is the part that can end a business rather than annoy a customer, and it
changed while this project was being built.

**EU AI Act Article 50 became enforceable on 2 August 2026 — three weeks ago.**
It obliges the provider of a synthetic-audio system to mark output in a
**machine-readable** way, and the deployer to disclose that it is AI-generated
([overview](https://discover.oreateai.com/discover/new-federal-laws-and-the-eu-ai-act-are-redefining-voice-cloning-compliance)).

Alongside it:

| | |
|---|---|
| **Tennessee ELVIS Act** | unauthorised AI vocal synthesis violates it *regardless of purpose*; $500 per violation plus fees ([guide](https://cognitivefuture.ai/ai-voice-cloning-legal-guide/)) |
| **GDPR** | a voice is **biometric data**; processing it for cloning needs explicit consent |
| **Japan, August 2026** | ruled AI voice cloning requires consent, with **civil liability reaching developers** — not only users ([report](https://www.techtimes.com/articles/323616/20260808/japan-rules-ai-voice-cloning-requires-consent-developers-face-civil-liability.htm)) |

The industry practice is short enough to memorise:

> **consent at enrolment · watermark at generation · detect on complaint**
> — and *never* pre-stamp user-recorded audio, only model output.

What is now in the code:

- `watermark.py` — Meta's [AudioSeal](https://github.com/facebookresearch/audioseal)
  (MIT, commercial use explicitly permitted), applied to generated audio only.
  `OMNIVOICE_WATERMARK=1`. Every failure path returns the clip **unmarked with a
  flag** rather than raising, so turning it on can never cost you a render.
- `POST /api/watermark/detect` — the "detect on complaint" half. A mark nobody
  can read back is not provenance.
- Consent fields on the voice record: `speaker_name`, `consent`, `consent_ref`,
  persisted with the voice and returned at registration.
  `OMNIVOICE_REQUIRE_CONSENT=1` makes registration fail without them.
- `logs/generations.jsonl` — one line per generation and per registration:
  timestamp, tenant, voice_id, **text hash** (full text only with
  `OMNIVOICE_AUDIT_TEXT=1`), **output audio hash**, duration, wpm, verified,
  warnings, watermark status. Recommended practice is precisely this: *log every
  synthesis call with the speaker id, text and output audio hash*.

Reconstructing who made which clip from which voice, after a complaint, is
impossible. Writing 300 bytes at the time is free.

**None of this is legal advice, and I have not run the watermarking code** —
`audioseal` is not installed here. Verify it on the GPU box before you rely on
it:

```powershell
venv\Scripts\python -m pip install audioseal
venv\Scripts\python -c "import watermark; print(watermark.self_check())"
```

---

## 6. Still open — decide these deliberately

### Two speakers in one reference — **not implemented**

Real diarization needs another model (pyannote and friends). Right now a clip
with two people in it registers happily and clones a blend of both. The
`ref_text`-versus-audio check catches many of them indirectly — two speakers
usually produce a transcript that does not match cleanly — but that is a side
effect, not a guarantee. **Known gap.**

### Cross-lingual timbre collapse

Upstream #162 measures speaker similarity dropping to 0.22 when a cloned voice
speaks a different language. We do not measure similarity at all. If you sell
multilingual cloning, measure it on your own voices before you promise it.

### Nothing measures speaker similarity

The verifier proves the right *words* came out. Nothing here proves it still
sounds like the customer. That needs a speaker-embedding model, and the
threshold has to be **measured on your own fixture voices**, not assumed —
about 50 clips per voice to see the distribution, then set the gate.

### The verifier's own cost is unmeasured on real hardware

One extra Whisper pass per chunk, under the same lock as generation. I estimate
15–20 % on RTF; that number is a guess until it runs. `/api/metrics` →
`verify_skipped` tells you if the budget is too tight.

---

## So: is it production ready?

**The specific defects that were measured are fixed, and most are now
structurally impossible.** References cannot end mid-word from either cause. A
clip that does not match its script is detected and reported rather than
shipped. Loudness is deterministic. A dead GPU says it is dead. A wrong
`voice_id` is a 404.

**It is not verified.** 68 unit tests pass on the pure-Python half — the text
front-end, the audio repair, the verifier. The GPU half has never executed. Every
claim about VRAM, RTF, verification cost and watermarking is reasoning, not
measurement, until `tools/acceptance.py` and `tools/audit_batch.py` run on
Windows.

**"No customer will ever complain" is not a promise this or any TTS product can
make.** The model is stochastic; it will occasionally drop a word or mispronounce
a name. What changed is that it can no longer do so *silently* — the failure
arrives as a warning on the response, before the customer's video is cut, rather
than as a bug report afterwards. That is the achievable bar, and it is the one
worth selling on.

The honest summary: **quality-ready pending the Windows run, and one deliberate
decision away from compliance-ready** (turn on watermarking, consent and audit
before you take public money).

---

## Sources

- [MagpieTTS-LF: Inference-Time Long-Form Speech Generation](https://arxiv.org/html/2606.18485) — boundary discontinuity measurements
- [Murmur — AI Voice Skips Words: Punctuation and TTS Fixes](https://www.murmurtts.com/blog/ai-voice-skips-words-punctuation-fixes) — the four places words disappear
- [Spheron — Self-Host AI Voice Cloning production guide (2026)](https://www.spheron.network/blog/self-host-voice-cloning-gpu-cloud-xtts-f5-tts-openvoice-v2/) — VRAM, caching, thread-safety
- [k2-fsa/OmniVoice issue #50](https://github.com/k2-fsa/OmniVoice/issues/50) — 60 s reference degradation
- [k2-fsa/OmniVoice issue #31](https://github.com/k2-fsa/OmniVoice/issues/31) — static/distortion at defaults
- [k2-fsa/OmniVoice issue #162](https://github.com/k2-fsa/OmniVoice/issues/162) — cross-lingual similarity 0.22
- [facebookresearch/audioseal](https://github.com/facebookresearch/audioseal) and [the paper](https://arxiv.org/abs/2401.17264) — localized watermarking, MIT
- [EU AI Act & federal voice-cloning compliance](https://discover.oreateai.com/discover/new-federal-laws-and-the-eu-ai-act-are-redefining-voice-cloning-compliance) — Article 50, enforceable 2 Aug 2026
- [State-by-state voice cloning legal guide (2026)](https://cognitivefuture.ai/ai-voice-cloning-legal-guide/) — ELVIS Act
- [Japan rules AI voice cloning requires consent](https://www.techtimes.com/articles/323616/20260808/japan-rules-ai-voice-cloning-requires-consent-developers-face-civil-liability.htm) — developer liability
- [WaveSpeed — consent checklist for voice cloning](https://wavespeed.ai/blog/audio-and-voice-models/how-to-get-consent-for-ai-voice-cloning-checklist/) and [HuggingFace — voice consent gate](https://huggingface.co/blog/voice-consent-gate) — enrolment practice
- [Gradient Flow — F5-TTS](https://gradientflow.com/f5-tts/) — content leakage into speaker embeddings
