@echo off
title DeepAnalyze - 加速节点
cd /d "%~dp0"

rem ════════════════════════════════════════════════════════════
rem  加速节点（子节点）：只提供 /analyze/sheet 工作表分析服务
rem  重要：不要设置 DEEPANALYZE_NODES，否则会变成主节点递归分发
rem  端口默认 5001（主节点脚本中 DEEPANALYZE_NODES 需与之对应）
rem ════════════════════════════════════════════════════════════
set DEEPANALYZE_PORT=5001

if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else if exist "..\venv\Scripts\python.exe" (
    set "PY=..\venv\Scripts\python.exe"
) else (
    set "PY=python"
)

echo ========================================
echo   DeepAnalyze 加速节点 - Starting...
echo   端口: 5001
echo   Press Ctrl+C to stop
echo ========================================

%PY% app.py
pause
