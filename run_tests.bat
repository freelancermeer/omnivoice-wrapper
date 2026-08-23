@echo off
REM ============================================================
REM  Unit tests - GPU ki zaroorat NAHI. Kisi bhi machine par chalte hain.
REM  Text normalization, audio repair, aur word-diff verifier check karte hain.
REM ============================================================
title OmniVoice tests
cd /d "%~dp0"
set PYTHONUTF8=1
"%~dp0venv\Scripts\python.exe" -m pytest tests -q
echo.
echo ============================================================
echo Ab server chalu kar ke acceptance checks chalayein:
echo    venv\Scripts\python tools\acceptance.py --voice YOUR_VOICE_ID
echo ============================================================
pause >nul
