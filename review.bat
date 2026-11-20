@echo off
rem Start albumsplit in the browser.
rem   review.bat                     -> start page: choose the scans and work folder there
rem   review.bat work                -> straight into review of .\work
rem   review.bat D:\work --input D:\scans --port 8001
rem For "Ask Claude", set ANTHROPIC_API_KEY first (or log in with: ant auth login).
setlocal
set "HERE=%~dp0"

set "PY=python"
if exist "%HERE%.venv\Scripts\python.exe" set "PY=%HERE%.venv\Scripts\python.exe"
rem Find the albumsplit package without changing directory, so relative paths stay relative
rem to where this was started.
set "PYTHONPATH=%HERE%;%PYTHONPATH%"

rem Pass all arguments straight through.
"%PY%" -m albumsplit review %*
if errorlevel 1 pause
