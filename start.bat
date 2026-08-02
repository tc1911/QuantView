@echo off
title DeepAnalyze

cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else if exist "..\venv\Scripts\python.exe" (
    set "PY=..\venv\Scripts\python.exe"
) else (
    set "PY=python"
)

echo ========================================
echo   DeepAnalyze - Starting...
echo   Open http://localhost:5000
echo   Press Ctrl+C to stop
echo ========================================

%PY% app.py
pause
