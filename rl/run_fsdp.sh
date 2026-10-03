#!/usr/bin/env bash
# FSDP port of run_{shipped,baseline,agentic}.sh for 80GB GPUs.
# Algorithm hyperparameters follow the README scripts except lr: 5e-6 -> 1e-6.
# 5e-6 with fp32 master weights blew up (KL to ref 0.05 -> 2.5-4.6, entropy
# collapse, val down) by step ~100; the README megatron config keeps bf16 master
# weights, which likely rounds most 5e-6 updates away (unverified).
# Fixed micro-batches, NOT use_dynamic_bsz: verl 0.7.1 dp_actor calls
# prepare_dynamic_batch without dp_group, so ranks can split into different
# micro-batch counts and FSDP deadlocks (NCCL watchdog timeout ~step 30-40).
# usage: rl/run_fsdp.sh {shipped|baseline|agentic} {medicine|science}
set -euo pipefail
ARM="${1:?arm}"; DS="${2:?dataset}"
case "${DS,,}" in
  medicine) DATASET_NAME="Medicine"; EXPERIMENT_DATASET="medicine" ;;
  science)  DATASET_NAME="Science";  EXPERIMENT_DATASET="science"  ;;
  *) echo "unsupported dataset: $DS" >&2; exit 2 ;;
esac
case "$ARM" in shipped|baseline|agentic) ;; *) echo "unsupported arm: $ARM" >&2; exit 2 ;; esac

RL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_DATA_DIR="${RL_DIR}/data/verl-newnew"
PYTHON_BIN="${PYTHON_BIN:-python}"
NGPU="${NGPU:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-300}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-25}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
LR="${LR:-1e-6}"
# Held-out validation: fixed 500-question RaR val (shipped rubric) + GPQA-Diamond (rule-scored) for science.
HELDOUT="${RL_DIR}/data/heldout"
if [[ "$EXPERIMENT_DATASET" == science ]]; then VAL_FILES="[${HELDOUT}/RaR-Science_val500.parquet,${HELDOUT}/gpqa_diamond.parquet]"
else VAL_FILES="[${HELDOUT}/RaR-Medicine_val500.parquet]"; fi
CKPT_DIR="${CKPT_DIR:-${RL_DIR}/checkpoints/${ARM}/rar_${EXPERIMENT_DATASET}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RAY_memory_usage_threshold=0.99 RAY_memory_monitor_refresh_ms=0
export VLLM_LOGGING_LEVEL=INFO PYTHONUNBUFFERED=1
export JUDGE_MAX_CONCURRENT="${JUDGE_MAX_CONCURRENT:-16}"
export JUDGE_BASE_URLS="${JUDGE_BASE_URLS:-http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1}"
export VERL_REWARD_ENABLE_DISK_LOGGING="${VERL_REWARD_ENABLE_DISK_LOGGING:-0}"

"${PYTHON_BIN}" -m verl.trainer.main_ppo --config-name ppo_trainer \
  algorithm.adv_estimator=grpo \
  data.train_files="${VERL_DATA_DIR}/RaR-${DATASET_NAME}/train.parquet" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size=32 \
  data.max_prompt_length=1024 \
  data.max_response_length=3584 \
  data.filter_overlong_prompts=True \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  actor_rollout_ref.model.path=Qwen/Qwen3.5-4B \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.max_model_len=4608 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_num_seqs=128 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  reward.reward_manager.name=naive \
  reward.custom_reward_function.path="${RL_DIR}/src/${ARM}_reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.num_workers=8 \
  algorithm.use_kl_in_reward=False \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node="${NGPU}" \
  trainer.nnodes=1 \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.project_name="${ARM}" \
  trainer.experiment_name="rar_${EXPERIMENT_DATASET}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.resume_mode=auto \
  ray_kwargs.ray_init.num_cpus=32 \
  "${@:3}"
