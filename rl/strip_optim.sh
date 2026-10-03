#!/usr/bin/env bash
# Keep only model weights in checkpoints that are no longer needed for resume.
# - running run: delete optim_*.pt in global_step_N with N < latest
# - finished run (queue_lane log has "end ARM DS rc=0"): delete optim_*.pt everywhere
# Never touches the step a run is currently writing (it is > latest until saved).
RL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOGS="${RL_DIR}/../logs"
while true; do
  for run in "$RL_DIR"/checkpoints/*/rar_*; do
    [ -f "$run/latest_checkpointed_iteration.txt" ] || continue
    latest=$(cat "$run/latest_checkpointed_iteration.txt")
    arm=$(basename "$(dirname "$run")"); ds=$(basename "$run"); ds=${ds#rar_}
    finished=0; grep -qs "end $arm $ds rc=0" "$LOGS"/queue_lane_*.log && finished=1
    for d in "$run"/global_step_*; do
      n=${d##*_}
      if [ "$finished" = 1 ] || [ "$n" -lt "$latest" ]; then
        for f in "$d"/actor/optim_world_size_*_rank_*.pt; do
          [ -f "$f" ] && rm -f "$f" && echo "[$(date -u +%FT%TZ)] stripped $f" >> "$LOGS/strip_optim.log"
        done
      fi
    done
  done
  sleep 600
done
