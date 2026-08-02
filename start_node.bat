@echo off
title QuantView - 加速节点
cd /d "%~dp0"

rem ════════════════════════════════════════════════════════════
rem  加速节点（无头 worker）：无 Web 界面，仅提供 /analyze/sheet
rem  任务接口，控制台打印收到的任务与耗时。
rem  重要：不要设置 DEEPANALYZE_NODES，否则会变成主节点递归分发
rem  用法：
rem    start_node.bat          → 端口 5001
rem    start_node.bat 5002     → 端口 5002（多开窗口时用不同端口）
rem  主节点 DEEPANALYZE_NODES 配置多个节点即可并行分发：
rem    node1=http://IP:5001,node2=http://IP:5002
rem ════════════════════════════════════════════════════════════
set DEEPANALYZE_HEADLESS=true
rem 加速节点任务 prompt 短，32768 上下文足够，省一半 KV 缓存内存
set DEEPANALYZE_CONTEXT=32768
if not "%1"=="" (
    set DEEPANALYZE_PORT=%1
) else (
    set DEEPANALYZE_PORT=5001
)

if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else if exist "..\venv\Scripts\python.exe" (
    set "PY=..\venv\Scripts\python.exe"
) else (
    set "PY=python"
)

echo ========================================
echo   QuantView 加速节点（无头） - Starting...
echo   端口: %DEEPANALYZE_PORT%
echo   Press Ctrl+C to stop
echo ========================================

%PY% app.py
pause
