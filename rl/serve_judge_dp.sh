#!/usr/bin/env bash
# Qwen3.8-27B judge as N independent vLLM servers (ports 8000..8000+N-1), one per GPU
# group, each in its own restart loop. Replaces the TP=4 server, whose TP workers hung
# (sample_tokens RPC timeout -> EngineDeadError) and took every request down.
# usage: JUDGE_GPU_GROUPS="0,1 2,3" rl/serve_judge_dp.sh   (TP size = GPUs per group)
# Single-GPU groups leave ~18G KV for a 27B model (~27 concurrent requests), so use TP=2.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="Qwen/Qwen3.8-27B"
GPUS=(${JUDGE_GPU_GROUPS:-0,1 2,3})
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_LOGGING_LEVEL=INFO PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs"
for i in "${!GPUS[@]}"; do
  port=$((8000 + i)); gpu=${GPUS[$i]}; tp=$(echo "$gpu" | tr "," "\n" | wc -l)
  (
    while true; do
      echo "[$(date -u +%FT%TZ)] start judge gpu=$gpu port=$port" >> "$ROOT/logs/judge_supervise.log"
      CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --served-model-name "$MODEL" --host 127.0.0.1 --port "$port" \
        --tensor-parallel-size "$tp" --gpu-memory-utilization 0.92 --max-model-len 16384 \
        --max-num-seqs "${JUDGE_MAX_NUM_SEQS:-128}" --max-num-batched-tokens 16384 \
        --no-enable-prefix-caching > "$ROOT/logs/judge_$port.log" 2>&1
      echo "[$(date -u +%FT%TZ)] judge port=$port exited rc=$?" >> "$ROOT/logs/judge_supervise.log"
      # free the GPU if a worker survived its parent
      for p in $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
      sleep 20
    done
  ) &
done
wait
