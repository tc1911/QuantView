#!/usr/bin/env bash
# QuantView 启动脚本（macOS / Linux）
# 用法：
#   ./start.sh             单节点（默认）
#   ./start.sh master      主节点（分布式），默认分发到 node1=http://127.0.0.1:5001
#   ./start.sh master "node1=http://192.168.1.10:5001,node2=http://192.168.1.11:5001"
#   ./start.sh node        加速节点（无头），默认端口 5001
#   ./start.sh node 5002   加速节点，指定端口
set -e
cd "$(dirname "$0")"

MODE="${1:-single}"

if [ -x "venv/bin/python" ]; then
  PY="venv/bin/python"
elif [ -x "../venv/bin/python" ]; then
  PY="../venv/bin/python"
else
  PY="python3"
fi

case "$MODE" in
  node)
    export DEEPANALYZE_HEADLESS=true
    export DEEPANALYZE_PORT="${2:-5001}"
    export DEEPANALYZE_CONTEXT="${DEEPANALYZE_CONTEXT:-32768}"
    echo "========================================"
    echo "  QuantView 加速节点（无头）- 端口: $DEEPANALYZE_PORT"
    echo "  注意：不要设置 DEEPANALYZE_NODES（避免递归分发）"
    echo "========================================"
    ;;
  master)
    export DEEPANALYZE_NODES="${2:-node1=http://127.0.0.1:5001}"
    echo "========================================"
    echo "  QuantView 主节点（分布式）- 节点: $DEEPANALYZE_NODES"
    echo "========================================"
    ;;
  *)
    echo "========================================"
    echo "  QuantView 单节点 - Starting..."
    echo "  Open http://localhost:5000"
    echo "========================================"
    ;;
esac

exec "$PY" app.py
