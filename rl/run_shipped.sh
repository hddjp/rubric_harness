if [[ $# -ne 1 ]]; then
  echo "usage: $0 {medicine|science}" >&2
  exit 2
fi
case "${1,,}" in
  medicine) DATASET_NAME="Medicine"; EXPERIMENT_DATASET="medicine" ;;
  science)  DATASET_NAME="Science";  EXPERIMENT_DATASET="science"  ;;
  *)
    echo "unsupported dataset: $1 (choose medicine or science)" >&2
    exit 2
    ;;
esac

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
RL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_DATA_DIR="${RL_DIR}/data/verl-newnew"
PYTHON_BIN="${PYTHON_BIN:-python}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAY_memory_usage_threshold=0.99
# Suppress the periodic verl replay-buffer status line (default: every 60s).
export VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS=2147483647
export TQ_LOGGING_LEVEL=WARNING
export VLLM_LOGGING_LEVEL=INFO
#export VERL_REWARD_ENABLE_DISK_LOGGING=1
#export PYTHONPATH="${RL_DIR}/.third_party/python-packages-megatron-a1a6ac3:${PYTHONPATH}"
#export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer

"${PYTHON_BIN}" -m verl.trainer.main_ppo --config-name _generated_ppo_megatron_trainer \
  algorithm.adv_estimator=grpo \
  data.train_files="${VERL_DATA_DIR}/RaR-${DATASET_NAME}/train.parquet" \
  data.val_files="${VERL_DATA_DIR}/RaR-${DATASET_NAME}/val.parquet" \
  data.train_batch_size=32 \
  data.max_prompt_length=1024 \
  data.max_response_length=3584 \
  data.filter_overlong_prompts=True \
  data.val_max_samples=500 \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.model.path=Qwen/Qwen3.5-4B \
  +actor_rollout_ref.model.override_config.model_config.text_config.mtp_num_hidden_layers=0 \
  +actor_rollout_ref.model.override_config.model_config.text_config.num_nextn_predict_layers=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.enable_activation_offload=False \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.strategy=megatron \
  actor_rollout_ref.actor.megatron.dtype=bfloat16 \
  actor_rollout_ref.actor.megatron.param_offload=True \
  actor_rollout_ref.actor.megatron.optimizer_offload=True \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
  actor_rollout_ref.actor.megatron.use_remove_padding=False \
  actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_torch_optimizer_for_cpu_offload=True \
  actor_rollout_ref.actor.megatron.dist_ckpt_optim_fully_reshardable=False \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
  actor_rollout_ref.actor.optim.lr_decay_style=constant \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4608 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.megatron.dtype=bfloat16 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.disable_log_stats=False \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASH_ATTN \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=2 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.max_model_len=4608 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8000 \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4608 \
  reward.reward_manager.name=naive \
  reward.custom_reward_function.path="${RL_DIR}/src/shipped_reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.num_workers=8 \
  algorithm.use_kl_in_reward=False \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.save_freq=25 \
  trainer.test_freq=25 \
  trainer.total_training_steps=300 \
  trainer.project_name=shipped \
  trainer.experiment_name="rar_${EXPERIMENT_DATASET}" \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.resume_mode=auto \
  ray_kwargs.ray_init.num_cpus=32 \
  actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra]' \
  actor_rollout_ref.actor.checkpoint.load_contents='[model,optimizer,extra]' \
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.main_params_dtype=torch.bfloat16 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.main_grads_dtype=torch.bfloat16 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.exp_avg_dtype=torch.bfloat16 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.exp_avg_sq_dtype=torch.bfloat16
  #actor_rollout_ref.actor.megatron.grad_offload=False \
  # actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
  # actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
  # actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
