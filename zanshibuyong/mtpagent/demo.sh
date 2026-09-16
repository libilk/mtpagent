#!/usr/bin/env bash
# 两分钟 demo:走一遍完整链路,零外部依赖(不需要 Redis / Docker / LLM key)。
#
#   ./demo.sh
#
# 它做的事见 coach/demo.py 的说明。README「快速开始」里的第二条命令就是它。
set -euo pipefail

cd "$(dirname "$0")"

# Windows 上虚拟环境在 .venv/Scripts,类 Unix 在 .venv/bin
if [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

echo "用 $PY 跑 demo..."
exec "$PY" -m coach.demo "$@"
