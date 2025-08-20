#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

OUT_DIR="output/P0"
mkdir -p "$OUT_DIR"

MONITOR="monitor_gpu.py"
if [ ! -f "$MONITOR" ]; then
  echo "错误: 未找到 $MONITOR（应与本脚本同目录）。"
  exit 1
fi

read -r -p "请输入 GPU 监控记录时长（秒），直接回车表示全程记录: " DURATION
MONITOR_ARGS=()
if [[ -n "${DURATION}" ]]; then
  if [[ "${DURATION}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    MONITOR_ARGS=(-t "${DURATION}")
    echo "GPU 监控将记录 $DURATION 秒。"
  else
    echo "输入无效（需为数字），将默认全程记录。"
  fi
else
  echo "未输入时长，GPU 监控将全程记录。"
fi

ts="$(date +"%Y%m%d_%H%M%S")"
CSV_TMP="gpu_utilization_log.csv"
CSV_FINAL="$OUT_DIR/gpu_utilization_${ts}.csv"

cleanup() {
    status=$?
    if kill -0 "${MON_PID:-0}" 2>/dev/null; then
        echo "发送 SIGINT 给 GPU 监控进程组..."
        kill -INT -"$MON_PID" || true
        for i in {1..50}; do
           if ! kill -0 "$MON_PID" 2>/dev/null; then break; fi
           sleep 0.1
        done
        if kill -0 "$MON_PID" 2>/dev/null; then
          echo "升级为 SIGTERM..."
          kill -TERM -"$MON_PID" || true
        fi
        for i in {1..30}; do
           if ! kill -0 "$MON_PID" 2>/dev/null; then break; fi
           sleep 0.1
        done
        if kill -0 "$MON_PID" 2>/dev/null; then
          echo "强制 KILL..."
          kill -KILL -"$MON_PID" || true
        fi
        wait "$MON_PID" 2>/dev/null || true
    fi
    if [ -f "$CSV_TMP" ]; then
        mv -f "$CSV_TMP" "$CSV_FINAL"
        echo "GPU 利用率日志已保存到 $CSV_FINAL"
    else
        echo "警告: 未找到 $CSV_TMP，可能监控未成功启动。"
    fi
    exit $status
}
trap cleanup EXIT INT TERM

echo "启动 GPU 利用率监控..."
setsid python3 "$MONITOR" "${MONITOR_ARGS[@]:-}" & 
MON_PID=$!
echo "GPU 监控 PID=$MON_PID"

echo "启动 piranha..."
./piranha -p 0 -c "/home/ubuntu/TDSC-2025/piranha/configs/all_relu.json" -o "$OUT_DIR" -P 0
