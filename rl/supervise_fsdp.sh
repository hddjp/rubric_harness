#!/usr/bin/env bash
# Relaunch rl/run_fsdp.sh after a crash; resume_mode=auto picks up the latest
# checkpoint (save_freq=25). Stops after a clean exit or MAX_RESTARTS crashes.
# usage: CUDA_VISIBLE_DEVICES=.. rl/supervise_fsdp.sh ARM DATASET
set -uo pipefail
RL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARM="$1"; DS="$2"; MAX_RESTARTS="${MAX_RESTARTS:-5}"
LOG_DIR="${RL_DIR}/../logs"; mkdir -p "$LOG_DIR"
for attempt in $(seq 0 "$MAX_RESTARTS"); do
  echo "[$(date -u +%FT%TZ)] attempt $attempt start" >> "$LOG_DIR/supervise_${ARM}_${DS}.log"
  "$RL_DIR/run_fsdp.sh" "$ARM" "$DS" > "$LOG_DIR/train_${ARM}_${DS}.attempt${attempt}.log" 2>&1
  rc=$?
  echo "[$(date -u +%FT%TZ)] attempt $attempt exit rc=$rc" >> "$LOG_DIR/supervise_${ARM}_${DS}.log"
  [ "$rc" -eq 0 ] && exit 0
  # free this run's GPUs before retrying (only processes on our own devices)
  for g in ${CUDA_VISIBLE_DEVICES//,/ }; do
    for p in $(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
  done
  sleep 60
done
exit 1
