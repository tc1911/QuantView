@echo off
title QuantView - 主节点（分布式）
cd /d "%~dp0"

rem ════════════════════════════════════════════════════════════
rem  主节点：把工作表任务轮流分发给加速节点，主节点生成总览
rem  ! 修改下面的地址为你加速节点的实际 IP 和端口！
rem     - 子节点在别的电脑：node1=http://子节点IP:5001
rem     - 子节点在本机：    node1=http://127.0.0.1:5001
rem  多个节点用逗号分隔：node1=http://IP1:5001,node2=http://IP2:5001
rem ════════════════════════════════════════════════════════════
set DEEPANALYZE_NODES=node1=http://127.0.0.1:5001

if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else if exist "..\venv\Scripts\python.exe" (
    set "PY=..\venv\Scripts\python.exe"
) else (
    set "PY=python"
)

echo ========================================
echo   QuantView 主节点（分布式）- Starting...
echo   节点: %DEEPANALYZE_NODES%
echo   Press Ctrl+C to stop
echo ========================================

%PY% app.py
pause
