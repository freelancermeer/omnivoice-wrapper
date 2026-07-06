@echo off
REM ============================================================
REM  OmniVoice - INTERNET share (public link)
REM  Gradio ek temporary public link banata hai:
REM      https://xxxxx.gradio.live   (72 ghante valid)
REM  Link kisi ko bhi bhej do, woh internet se use kar sakta hai.
REM  NOTE: link public hai -> password zaroor lagayein.
REM ============================================================
title OmniVoice Studio (Public link)
cd /d "%~dp0"

set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONUTF8=1

set GRADIO_SHARE=1
set OMNIVOICE_OPEN_BROWSER=1

REM ---- Password (STRONGLY recommended for public links) ----
REM Neeche wali line ka REM hata kar user:pass set karein:
REM set OMNIVOICE_AUTH=admin:mypassword

echo Starting with PUBLIC share link...
echo Console me jo https://....gradio.live URL aaye wo share karein.
echo.
"%~dp0venv\Scripts\python.exe" "%~dp0app.py"
echo.
pause >nul
