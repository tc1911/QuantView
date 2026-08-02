$host.ui.RawUI.WindowTitle = "QuantView - 本地数据分析助手"

Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "  QuantView 本地数据分析助手" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot

# 检测 Python
if (Test-Path "venv\Scripts\python.exe") {
    Write-Host "[OK] 使用虚拟环境" -ForegroundColor Green
    $python = "venv\Scripts\python.exe"
} elseif (Test-Path "..\venv\Scripts\python.exe") {
    $python = "..\venv\Scripts\python.exe"
} else {
    Write-Host "[WARN] 未找到虚拟环境，使用系统 Python" -ForegroundColor Yellow
    $python = "python"
}

# 检查 llama-server
if (Test-Path "llama-server.exe") {
    Write-Host "[OK] 检测到 llama-server.exe (GPU 推理)" -ForegroundColor Green
}

Write-Host "启动服务..." -ForegroundColor Cyan
Write-Host "浏览器打开: http://localhost:5000" -ForegroundColor White
Write-Host "按 Ctrl+C 停止" -ForegroundColor DarkGray
Write-Host "========================================" -ForegroundColor Cyan

& $python app.py
pause
