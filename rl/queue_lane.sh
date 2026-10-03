#!/usr/bin/env bash
# Run a list of "ARM:DATASET" jobs sequentially on this lane's GPUs.
# usage: CUDA_VISIBLE_DEVICES=4,5 rl/queue_lane.sh agentic:science agentic:medicine ...
set -uo pipefail
RL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG="${RL_DIR}/../logs/queue_lane_${CUDA_VISIBLE_DEVICES//,/}.log"
for job in "$@"; do
  arm="${job%%:*}"; ds="${job##*:}"
  echo "[$(date -u +%FT%TZ)] start $arm $ds" >> "$LOG"
  "$RL_DIR/supervise_fsdp.sh" "$arm" "$ds"; rc=$?
  echo "[$(date -u +%FT%TZ)] end $arm $ds rc=$rc" >> "$LOG"
done
