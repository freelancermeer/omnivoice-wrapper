@echo off
REM ============================================================
REM  OmniVoice Voiceover Studio - local launcher
REM  Bas is file ko double-click karein. Browser khud khul jayega.
REM  Saari voices outputs\ folder me save hoti hain.
REM ============================================================
title OmniVoice Voiceover Studio
cd /d "%~dp0"

REM Models local cache se load hote hain (no internet needed).
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONUTF8=1

REM Long-running GPU process ke liye zaroori. Allocator ko segment badhane
REM deta hai bajaye memory ko na-qabil-e-istemal tukdon me todne ke - yehi
REM "GPU 5%% par hai aur out of memory" wali kharabi ki wajah thi.
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM ---- Optional tweaks (line ke aage REM hata kar enable karein) ----
REM set OMNIVOICE_NUM_STEP=16
REM set GRADIO_SERVER_PORT=7860
REM set OMNIVOICE_OPEN_BROWSER=0
REM  Har clip ko transcribe kar ke script se match karta hai (default: on).
REM  Band karne ke liye:
REM set OMNIVOICE_VERIFY=0
REM  Ek waqt me kitni generations (8 GB card par 1 hi rakhein):
REM set OMNIVOICE_MAX_CONCURRENCY=1
REM  Har clip ki loudness (default -20 LUFS):
REM set OMNIVOICE_OUT_LUFS=-20
REM  Text normalization: full / basic / off
REM set OMNIVOICE_NORMALIZE_LEVEL=full
REM  Verifier ek saath kitni 30s windows padhe (default 12). Ye long-form RTF
REM  0.221 se 0.170 laaya, transcript bilkul wahi. Agar VRAM tang ho to kam
REM  karein; 12 se upar barhane ka koi faida nahi.
REM set OMNIVOICE_ASR_BATCH=12
REM  Whisper ko GPU se hata kar CPU pe (VRAM bachta hai, verification slow):
REM set OMNIVOICE_ASR_DEVICE=cpu

echo Starting OmniVoice Voiceover Studio...
echo (pehli baar model load me ~10-20 sec lagte hain)
echo.
"%~dp0venv\Scripts\python.exe" "%~dp0app.py"

echo.
echo ============================================================
echo App band ho gaya. Is window ko band karne ke liye koi key dabayein.
echo ============================================================
pause >nul
