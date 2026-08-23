# How to record a reference clip that clones well

Give this page to anyone who uploads a voice. Most support tickets about
"the clone doesn't sound like me" or "it says words I never wrote" start here,
and every one of them is preventable in the thirty seconds it takes to record
the clip properly.

---

## The short version

| | |
|---|---|
| **Length** | 6–10 seconds |
| **Ending** | Finish a sentence, then stay quiet for a second before you stop |
| **Speakers** | One. Only you. |
| **Background** | Quiet room. No music, no TV, no traffic. |
| **Distance** | A hand's width from the mic — close enough to be clear, far enough not to distort |
| **Content** | Normal speech at your normal pace, the way you want the voiceover to sound |

---

## The one that matters most: don't stop mid-word

This is worth its own section because it caused the worst bug we have ever
measured.

A 60-second reference was uploaded. The system kept the first 10 seconds, and
those 10 seconds happened to end in the middle of the word *"forcing"*. The
model treated that half-spoken word as something it still had to finish — so it
said **"forcing"** at the end of almost every sentence it generated with that
voice. In one batch it appeared **163 times**, in 86–90% of the clips, about
1.3 seconds each. On a 22-minute video that is roughly two minutes of ruined
audio and 92 words nobody wrote.

There are **two** ways a reference ends mid-word, and both are handled:

1. **The system cut it there** — your upload was longer than 10 s. It now cuts
   at the nearest natural pause instead of at a fixed timestamp. If the only
   pause is early it will keep a *shorter* clip rather than a mid-word one.
2. **You stopped recording while still speaking** — your clip is short enough
   that nothing needed trimming, so the half word would just sit there. It is
   now cut back to your last finished phrase automatically:

   > ⚠ your clip stopped while you were still speaking, so it was cut back to
   > the last finished phrase (6.8s of 7.4s kept) — an unfinished word at the
   > end gets repeated in every clip

   Losing a whole word costs nothing: the transcript is re-derived from the
   trimmed audio afterwards, so text and audio always agree.

The one case that cannot be repaired is a clip that is **too short to cut back**
— if the only pause is one second in, cutting there would leave a one-second
reference, which clones badly for a different reason. Then you get:

> ⚠ reference ends without a pause and is too short to cut back (under 3s would
> be left) — please re-record 6-10s ending on a finished sentence

That warning is not cosmetic. Re-record.

**So: say a complete sentence, take a breath, and only then stop recording.**
Everything else the system can fix for you; that one it cannot.

---

## What the system tells you at upload

`POST /api/voices` (and the Save button in the app) returns a report:

```json
{
  "voice_id": "narrator_a",
  "accepted": true,
  "quality_score": 0.72,
  "baseline_wpm": 104,
  "duration_sec": 9.4,
  "lufs": -20.0,
  "snr_db": 11,
  "ref_text": "the transcript we will actually use",
  "warnings": [
    "reference is noisy (SNR 11 dB) — the cloned voice will carry that noise"
  ]
}
```

Read `warnings` before you generate ten hours of audio.

| Warning | What to do |
|---|---|
| `reference is Ns — under 3s clones poorly` | Record 6–10 s |
| `reference is Ns — only the first 10s matter` | Trim it yourself so you choose which 10 s |
| `reference is noisy (SNR N dB)` | Quieter room, or a closer mic |
| `reference is clipping` | Move further from the mic, or lower the input gain |
| `reference ends without a pause` | Re-record with a beat of silence at the end |
| `no pause found ... cut mid-phrase` | **Re-record.** This is the one above. |

`quality_score` is advisory (1.0 is clean). Nothing is rejected for a low score
— you are told, and you decide.

---

## `ref_text`: send it, or don't — but never send the wrong one

`ref_text` is the transcript of what your clip actually says.

- **Leave it out** and the clip is transcribed automatically. Safe and usual.
- **Send it** and it is checked against the recording. If it matches less than
  90% of what is heard, the upload is **rejected** with a `422` telling you what
  was heard instead.

That rejection is deliberate. A transcript that does not match the audio is the
main way reference words leak into generated clips: the model is told one thing,
hears another, and treats the difference as something it is supposed to say.
Being told at upload beats finding out three hours into a batch.

If the clip you send is longer than 10 seconds it has to be trimmed, so your
transcript no longer describes the audio — in that case it is replaced with a
transcript of the trimmed clip and you get a warning saying so.

To turn the rejection into a warning: `OMNIVOICE_STRICT_REF=0`.

---

## What happens to your clip after upload

1. **Trimmed** at the last natural pause within 10 s (not at a fixed 10.0 s) —
   and if your clip arrived already ending mid-word, cut back to your last
   finished phrase. A partial *opening* word is dropped too, when a pause turns
   up in the first 0.6 s.
2. **Silence stripped** from both ends, then a 25 ms fade and 0.3 s of clean
   silence appended, so the reference has a definite ending.
3. **Levelled** to −20 LUFS with a −1 dBTP ceiling, so every voice in your
   library generates at the same volume. Without this, output loudness tracks
   whatever the reference happened to be: one measured batch ran from −0.2 dB
   (on the edge of distortion) to −12.6 dB in a single video.
4. **Transcribed**, and the transcript stored with the voice.
5. **Measured** for its natural speaking rate (`baseline_wpm`), which is later
   used to notice if that voice starts drifting — compared against itself, never
   against a fixed 140–180 band that would fail a documentary narrator and pass
   a broken ads voice.

Your original file is not kept; the repaired 10-second clip is what lives in
`voices/`.

---

## Reusing a voice

Register once, then send `voice_id` on every request:

```bash
curl -X POST http://<pc-ip>:8001/api/voices \
     -F "name=narrator_a" -F "voice=@my_clip.wav"

curl -X POST http://<pc-ip>:8001/api/tts \
     -F "text=Hello there." -F "voice_id=narrator_a" -F "format=mp3" -o out.mp3
```

A `voice_id` that does not exist returns **404**. It never quietly falls back to
another voice — a wrong voice across a ten-hour job costs far more than an
error you see immediately.

Voices survive a restart: they live in `voices/index.json` with their clip.
