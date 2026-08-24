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
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set GRADIO_SHARE=1
set OMNIVOICE_OPEN_BROWSER=1

REM ---- Password (STRONGLY recommended for public links) ----
REM Neeche wali line ka REM hata kar user:pass set karein:
REM set OMNIVOICE_AUTH=admin:mypassword
REM ---- API key bhi lagayein agar API bahar khul rahi hai ----
REM set OMNIVOICE_API_KEY=change-me

if not defined GRADIO_SERVER_PORT set GRADIO_SERVER_PORT=7860
if not defined OMNIVOICE_API_PORT set OMNIVOICE_API_PORT=8001

echo ============================================================
echo   OmniVoice Voiceover Studio  (PUBLIC share link)
echo ------------------------------------------------------------
echo   UI   (local)    http://127.0.0.1:%GRADIO_SERVER_PORT%
echo   API  (local)    http://127.0.0.1:%OMNIVOICE_API_PORT%
echo   API  docs       http://127.0.0.1:%OMNIVOICE_API_PORT%/api/docs
echo ------------------------------------------------------------
echo   Console me jo https://....gradio.live URL aaye wo share karein.
echo   NOTE: sirf UI share hoti hai. API public NAHI hoti - wo is
echo   machine par hi rehti hai.
echo ============================================================
echo.
"%~dp0venv\Scripts\python.exe" "%~dp0app.py"
echo.
pause >nul
