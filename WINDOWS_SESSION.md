# Tomorrow's prompt — paste this into a fresh Claude Code session on Windows

Everything below the line is meant to be copied as-is. It is written to be
self-contained, because the session on the Windows box will not have seen the
work that produced this code.

Keep this file; if a later round of changes happens, update the numbers in it
rather than rewriting it.

---

Ye OmniVoice TTS wrapper hai jo main sell kar raha hoon. Code Mac pe likha gaya
tha aur **GPU pe kabhi chala hi nahi** — aaj pehli baar Windows pe test kar raha
hoon. Machine: RTX 3060 Ti 8 GB, CUDA 12.8, Windows.

**Pehle `FIXES.md`, `RESEARCH.md` aur `README.md` padh lo** — poora context wahan
hai. Short version: ek production batch (63 clips / 10,269 words) transcribe
karke script se diff kiya gaya tha, usme mila:

- **218 words jo bheje hi nahi gaye the** — reference audio exactly 10.000s pe
  mid-word cut ho raha tha, aur wo aadha word har clip ke end me aa raha tha
  ("forcing" ×163)
- **12 words chup-chaap gir gaye**, kisi ne report nahi kiya
- Ek hi video me loudness −0.2 dB se −12.6 dB tak
- CUDA OOM jo kabhi recover nahi hota tha, jabki `/api/health` "ok" bolta tha

Sab fix ho chuka hai (95 unit tests pass, bina GPU ke), plus upstream issue
tracker padh ke chunking aur omnivoice version bhi update kiya. **Lekin GPU wala
hissa kabhi execute nahi hua.**

## Aaj ye run karna hai, isi tarteeb se

```powershell
git pull
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m pip install pytest requests
venv\Scripts\python -m pytest tests -q
```

`95 passed` aana chahiye. Phir `run.bat`. Banner pe ye dekhna:

```
omnivoice 0.2.1 · features: asr_device, pad_duration, fade_duration
verify=on · normalize=full · loudness=on (-20 LUFS) · concurrency=1
```

Agar `OUTDATED` likha aaye to pip wali line chali nahi — pehle wo theek karo.

Phir apni teen voices dobara register karo (clips repair ho ke andar jaati hain,
purani entries purane code se bani hain), aur reply me aane wale `warnings`
zaroor padho — `RVoiceover_3` wahi hai jo "and" pe khatam hoti thi:

```powershell
curl -X POST http://127.0.0.1:8001/api/voices -F "name=RVoiceover_3" -F "voice=@RVoiceover_3.mp3"
```

Phir:

```powershell
venv\Scripts\python tools\acceptance.py --voice RVoiceover_3
venv\Scripts\python tools\audit_batch.py manifest.json --out audit.json
```

## Teen numbers chahiye mujhe — inhi pe sab decide hoga

1. **RTF.** Pehle mujhe **0.16** milta tha 16 steps pe. Ye mera sabse ahem
   number hai. UI ke queue header pe "avg RTF" dikhta hai, API pe `X-RTF`
   header, aur `GET /api/metrics` pe `rtf_overall`.
2. **Verification ka exact cost.** `tools/acceptance.py --quick` do baar chalao,
   ek baar `OMNIVOICE_VERIFY=0` ke saath. `X-RTF` ka farq hi asli cost hai.
3. **Audit ka added/dropped count.** Target **0 added, 0 dropped** — pehle 218
   aur 12 the.

## Agar kuch toota

Har badi cheez ek env var se off hoti hai, full rollback ki zaroorat nahi:

| Symptom | Lever |
|---|---|
| RTF expectation se zyada | `OMNIVOICE_VERIFY=0` (checking batch ke baad `audit_batch.py` se) |
| Koi number galat pada gaya | `OMNIVOICE_NORMALIZE_LEVEL=basic` ya `off` |
| Clips edit ke liye zyada loud/halke | `OMNIVOICE_OUT_LUFS=-16`, ya `OMNIVOICE_NORMALIZE_OUTPUT=0` |
| Achhi reference reject ho rahi hai | `OMNIVOICE_STRICT_REF=0` |
| Voice chhoti clips pe badalti hai | `OMNIVOICE_MAX_CHARS` badhao (abhi 200) |
| Aakhri lafz kat raha hai | `OMNIVOICE_PAD_DURATION` / `OMNIVOICE_FADE_DURATION` (0.2.0+) |

Purana kaam karta version: `git checkout 7d2a387`.

## Kaam karne ka tareeqa

- Jo bhi tootey, **pehle wajah dhoondo, phir fix** — check ko band karke
  "pass" mat dikhana.
- Har result me batao ke tumne **khud verify kiya** ya sirf reasoning hai.
- Koi bhi fix karo to `pytest tests -q` dobara chalao, aur commit `main` pe
  push kar do.
- Agar RTF verification ki wajah se 0.16 se kaafi upar gaya, to mujhse poocho
  ke verification band karein ya batch ke baad chalayein — ye faisla mera hai.

## Agar RTF theek raha aur waqt bacha

`RESEARCH.md` §8 me FlashInfer wala experiment hai — upstream PR #239 kehta hai
batch size 1 pe **2.1× lossless speedup**. Wo `omnivoice` git main pe le jata
hai, to base version confirm hone ke baad hi karna.
