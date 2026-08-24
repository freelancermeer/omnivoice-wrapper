# The Windows GPU session — done, with the numbers

This file used to be a prompt to paste into a fresh session. That session has
now happened (23–24 August 2026), so it carries the results instead. The full
write-up is **[GPU_RUN.md](GPU_RUN.md)**; this is the short version.

Keep updating the numbers here rather than rewriting the file.

---

## Ye tha sawal, ye hai jawab

| Sawal | Jawab |
|---|---|
| **RTF kitna hai?** | **0.170** verification ke saath (long-form), **0.151** uske baghair. Purana 0.16 asal me *unverified* number tha. Subah ye 0.300 tha — verifier ki ASR windows batch karne se 0.226, phir batch 12 pe 0.170. |
| **Verification ka cost?** | **12.6 %** long-form pe (0.170 vs 0.151), aur ~35 % chhoti clips ke batch pe jahan fixed overhead ghaalib hai. Subah ye ~100 % tha. |
| **Audit: added / dropped?** | **0 added, 0 dropped, 0 % trailing artefact**, 25/25 clean clips. Pehle 218 aur 12 the. |

Do lafz farq the — `Bessent` → "besant", `authorisation` → "authorization" —
aur dono **transcriber ki spelling** hain, model ki ghalti nahi. Verifier ne
dono ko pronunciation note kaha, regeneration par paisa nahi lagaya.

---

## Reference wala bug — sach me theek ho gaya

Saboot disk pe hi mojood tha: 6 me se 5 references **theek 10.0000 s** ke thay —
purane stopwatch cut ka nishan.

| voice | pehle | ab | quality | LUFS |
|---|---|---|---|---|
| RVoiceover_1 | 10.000 s | 9.82 s | 1.00 | −20.0 |
| RVoiceover_2 | 10.000 s | 9.18 s | 0.85 | −20.0 |
| RVoiceover_3 | 10.000 s | **7.52 s** | 0.85 | −20.0 |

`RVoiceover_3` ka `ref_text` pehle `...and forcing.` pe khatam hota tha. Ab
`...his own handpicked judges...` pe khatam hota hai. **"forcing" jad se
khatam**, aur audit confirm karta hai ke aage kuch leak nahi hua.

---

## GPU pe chhe naye bugs mile — sab fix

Chaar sirf hardware pe hi mil sakte thay:

1. **Verifier 30 second se lambi kisi bhi clip pe chal hi nahi raha tha.**
   Sabse bura wala. Whisper ek waqt me 30 s leta hai; us se aage
   `return_timestamps=True` chahiye warna exception. Matlab long-form kaam pe —
   jo is product ka asli use case hai — check khamoshi se skip ho raha tha.
2. **Har number wali clip fail ho rahi thi.** Script "ninety", Whisper "90" —
   ek hi number, alag notation, magar drop + insert count ho raha tha.
   49 generations me **22 verify failures, 25 regenerations**. Ab **0 aur 0**.
3. **`1st` → "onest"** jab `OMNIVOICE_NORMALIZE_LEVEL=basic` ho — yani theek us
   setting me jo aap number galat parhne par khud istemal karte.
4. **`OMNIVOICE_ASR_DEVICE` chup-chaap dead tha** — feature detection `**kwargs`
   ko nahi dekhti thi. Yehi bug `OMNIVOICE_FLASHINFER=1` ko bhi mar deta.
5. **Khali text 422 deta tha, 400 nahi** — sahi check likha hua tha magar chal
   hi nahi raha tha.
6. **`expandable_segments:True` Windows pe kuch nahi karta.** torch use qubool
   karta hai, ek warning likhta hai, aur ignore kar deta hai. Ab `/api/health`
   aur banner sach batate hain.

---

## Stability

`tools/acceptance.py` → **35 passed, 0 failed, 1 skipped**.
`pytest tests -q` → **104 passed**.

73 generations me: **0 OOM, 0 model reloads, 0 verify failures/skips**, soak ke
baad **+0 MB allocated aur +0 MB reserved**, fragmentation 151 MB (gate 800).
Chaar concurrent callers: `[200, 200, 200, 200]`, server zinda.

**Seed deterministic nikla** — same seed = byte-identical audio. Iska matlab
golden-file tests yahan sach me kaam karte hain.

---

## Ab faisla aapka hai

Verification RTF **0.151 → 0.203** le jati hai. Badle me server aapko bata deta
hai ke koi lafz gir gaya. Lever: `OMNIVOICE_VERIFY=0`, aur phir batch ke baad
`tools\audit_batch.py` se ek hi pass me poora check.

Mera mashwara: **on rakhein.** 35 % ki qeemat us se kam hai jo ek dropped word
customer ki cut hui video me pakde jane par lagti hai — aur ab regenerations 0
hain, to asli kharcha pehle se kam ho gaya hai.

---

## Jo abhi bhi nahi hua

- **Watermarking** — `audioseal` install hi nahi. Public paisa lene se pehle
  karna zaroori hai (EU AI Act Article 50, 2 Aug 2026 se enforceable).
- **VRAM leak lambe run pe** — upstream #199 kehta hai ye dinon me zahir hota
  hai; yahan sab se lamba run ek ghante se kam tha. Flat memory achhi khabar
  hai, saboot nahi.
- **FlashInfer** — nahi chalaya; base version pehle confirm karna tha. Bug #4
  is ko waise bhi khamoshi se rok deta, ab raasta saaf hai.
- **Speaker similarity** aur **do-speaker reference** — jaise thay waise hain.
- **Asli 63-clip batch** — uski scripts repo me nahi hain, is liye
  `tools/make_batch.py` ne 25 clips ka batch usi register me banaya.

---

## Ek operational baat

Server ko `taskkill /PID <pid> /T /F` se band karein. Is session me plain
terminate ne process ko port 8001 pakde chhod diya, aur ek "restarted" server
asal me purana hi tha — jis se measurements ka ek round kharab hua. Agla run
shuru karne se pehle `netstat -ano | findstr :8001` se confirm karein ke port
khali hai.

## Naye tools

```powershell
venv\Scripts\python tools\rtf_probe.py  --voice RVoiceover_3_2 --repeat 2
venv\Scripts\python tools\make_batch.py --voice RVoiceover_3_2
venv\Scripts\python tools\audit_batch.py manifest.json --out audit.json
```

`rtf_probe.py` RTF ko clip length ke against naapta hai — ek akela RTF number is
server ke liye be-matlab hai, kyunke chhoti clip pe fixed cost hi ghalib rehti
hai.


---

## Doosri session (24 August) — RTF, aur teen dead ends

Poora hisab **[GPU_RUN.md](GPU_RUN.md)** me hai. Yahan sirf faisle:

| kis pe | koshish | nateeja |
|---|---|---|
| verification | ASR windows batch karna | 0.300 → 0.226 |
| verification | batch 8 → 12 | 0.226 → **0.170** |
| verification | clock-time / currency ke jhoote alarm | ek clip 0.548 → 0.244 |
| verification | plural (`memory`/`memories`) ka jhoota alarm | ek regeneration kam |
| verification | faster-whisper runtime | speed same, **−1.6 GB VRAM** |
| verification | generation ke peeche chhupana | **reject** — dono GPU-bound |
| generation | FlashInfer | **namumkin** — NCCL Windows pe hai hi nahi |
| generation | torch.compile | **2x slower** — har nayi length pe recompile |
| poora GPU | Parsec band karna | ~25 % mumkin, deployment ka faisla |

```
RTF 0.170  =  0.151 generation  +  0.019 verification
```

Verification ka hissa ab chhota hai. Baqi 0.151 OmniVoice khud hai, eager
PyTorch me, Ampere card pe — aur us se aage har documented raasta Linux maangta
hai.

## Aur kya badla

* **Do instance ek saath nahi chal saktin.** `run.bat` dobara chalane pe ab
  model load hone se **pehle** ruk jata hai (8.5 s, VRAM chhue baghair) —
  pehle wo chup-chaap doosra model load kar leta tha aur "UI buggy hai" jaisa
  lagta tha.
* **Voices dobara banayi gayin** teen asli clips se, plain naamon ke saath.
  Purani `voices/` ki copy `voices_backup_<timestamp>/` me hai — wo original
  10.000 s clips ki **wahid copy** hai, delete mat karna.
* **`tools/verify_external.py`** — apne verdict ko AssemblyAI se cross-check
  karta hai. Das clips: Whisper 10/10 clean, AssemblyAI 9/10, 9/10 par ittefaq.

## Verify on rakhein ya off?

**On.** Farq detect ka nahi, **repair** ka hai: `tools/audit_batch.py` wohi
defects baad me pakad leta hai, magar on hone pe kharab chunk **run ke doran
khud dobara banta hai** aur caller tak pahunchta hi nahi. 12.6 % pe wo sasta
bima hai.
