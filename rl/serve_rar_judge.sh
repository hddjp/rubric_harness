#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B is shared by the rubric-generation and judge roles. The released
# training data already contains rubrics, so the reproduction run only needs
# the judge endpoint.
RAR_JUDGE_MODEL="Qwen/Qwen3.8-27B"
RAR_JUDGE_HOST="0.0.0.0"
RAR_JUDGE_PORT="8000"
RAR_JUDGE_TP="4"
RAR_JUDGE_GPU_MEMORY_UTILIZATION="0.9"
RAR_JUDGE_MAX_NUM_SEQS="32"
RAR_JUDGE_MAX_NUM_BATCHED_TOKENS="16000"
RAR_JUDGE_MAX_LOG_LEN="100000"
RAR_JUDGE_LOG_LEVEL="INFO"
# Keep request/response disk logging disabled by default. Set to 1 to re-enable.
RAR_JUDGE_ENABLE_IO_LOG="${RAR_JUDGE_ENABLE_IO_LOG:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAR_JUDGE_IO_LOG_FILE="${PROJECT_ROOT}/outputs/rar_judge_io.log"
LOCAL_PYDEPS="${PROJECT_ROOT}/.local_pydeps"
PYTHON_SITE="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
if [[ -d "${LOCAL_PYDEPS}" ]]; then
  export PYTHONPATH="${LOCAL_PYDEPS}${PYTHONPATH:+:${PYTHONPATH}}"
fi
if [[ -d "${PYTHON_SITE}/nvidia/cuda_runtime/lib" ]]; then
  export LD_LIBRARY_PATH="${PYTHON_SITE}/nvidia/cuda_runtime/lib:${PYTHON_SITE}/nvidia/nccl/lib:${PYTHON_SITE}/nvidia/cudnn/lib:${LOCAL_PYDEPS}/transformer_engine_cu12.libs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export RAR_JUDGE_MODEL
export RAR_RUBRIC_MODEL="Qwen/Qwen3.8-27B"
export VLLM_LOGGING_LEVEL="${RAR_JUDGE_LOG_LEVEL}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

mkdir -p "$(dirname "${RAR_JUDGE_IO_LOG_FILE}")"

"${PYTHON_BIN}" - <<PY
import torch

required = int("${RAR_JUDGE_TP}")
available = torch.cuda.device_count() if torch.cuda.is_available() else 0
if available < required:
    raise SystemExit(
        f"Need at least {required} CUDA GPU(s) for RAR_JUDGE_TP={required}, "
        f"but torch sees {available}. Check the NVIDIA driver/CUDA_VISIBLE_DEVICES, "
        "or lower RAR_JUDGE_TP for the available hardware."
    )
PY

SERVER_CMD=(
  "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${RAR_JUDGE_MODEL}" \
  --served-model-name "${RAR_JUDGE_MODEL}" \
  --host "${RAR_JUDGE_HOST}" \
  --port "${RAR_JUDGE_PORT}" \
  --tensor-parallel-size "${RAR_JUDGE_TP}" \
  --gpu-memory-utilization "${RAR_JUDGE_GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${RAR_JUDGE_MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${RAR_JUDGE_MAX_NUM_BATCHED_TOKENS}" \
  --max-log-len "${RAR_JUDGE_MAX_LOG_LEN}" \
#  --dtype bfloat16 \
  --no-enable-prefix-caching \
#  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  #--speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  "$@"
)

if [[ "${RAR_JUDGE_ENABLE_IO_LOG,,}" =~ ^(1|true|yes|on)$ ]]; then
  "${SERVER_CMD[@]}" 2>&1 \
  | tee >(awk '
      function emit(id) {
          if ((id in requests) && (id in responses)) {
              print "===== request " id " ====="
              print requests[id]
              print responses[id]
              print ""
              delete requests[id]
              delete responses[id]
              fflush()
          }
      }
      /Request .* details: prompt:/ {
          id = $0
          sub(/^.*Request /, "", id)
          sub(/ details:.*$/, "", id)
          sub(/, prompt_token_ids:.*prompt_embeds shape:/, ", prompt_embeds shape:")
          requests[id] = $0
          emit(id)
      }
      /Generated response .*: output:/ {
          id = $0
          sub(/^.*Generated response /, "", id)
          sub(/:.*$/, "", id)
          sub(/, output_token_ids:.*finish_reason:/, ", finish_reason:")
          responses[id] = $0
          emit(id)
      }
  ' >>"${RAR_JUDGE_IO_LOG_FILE}")
else
  exec "${SERVER_CMD[@]}"
fi
