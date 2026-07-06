@echo off
REM ============================================================
REM  OmniVoice Voiceover Studio - local launcher
REM  Bas is file ko double-click karein. Browser khud khul jayega.
REM  Saari voices D:\omnivoice\outputs\ me save hoti hain.
REM ============================================================
title OmniVoice Voiceover Studio
cd /d "%~dp0"

REM Models local cache se load hote hain (no internet needed).
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONUTF8=1

REM ---- Optional tweaks (line ke aage REM hata kar enable karein) ----
REM set OMNIVOICE_NUM_STEP=16
REM set GRADIO_SERVER_PORT=7860
REM set OMNIVOICE_OPEN_BROWSER=0

echo Starting OmniVoice Voiceover Studio...
echo (pehli baar model load me ~10-20 sec lagte hain)
echo.
"%~dp0venv\Scripts\python.exe" "%~dp0app.py"

echo.
echo ============================================================
echo App band ho gaya. Is window ko band karne ke liye koi key dabayein.
echo ============================================================
pause >nul
