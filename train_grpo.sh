#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$ROOT_DIR"

: "${TRAIN_FILE:?Set TRAIN_FILE to the training Parquet file}"
: "${VAL_FILE:?Set VAL_FILE to the validation Parquet file}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"
REWARD_BACKEND="${REWARD_BACKEND:-ray}"
REWARD_BASE_MODEL_PATH="${REWARD_BASE_MODEL_PATH:-}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-}"
REWARD_DEVICE="${REWARD_DEVICE:-cuda}"
REWARD_MAX_LENGTH="${REWARD_MAX_LENGTH:-5100}"
REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-16}"
TRAINING_GPUS="${TRAINING_GPUS:-4}"
ROLLOUT_N_QUERY="${ROLLOUT_N_QUERY:-4}"
ROLLOUT_N_SAMPLE="${ROLLOUT_N_SAMPLE:-4}"
MICRO_BATCH_PER_GPU="${MICRO_BATCH_PER_GPU:-2}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-2}"
RUN_NAME="${RUN_NAME:-triconlist-grpo-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo/$RUN_NAME}"

case "$REWARD_BACKEND" in
    ray|local) : "${REWARD_MODEL_PATH:?Set REWARD_MODEL_PATH to a reward checkpoint}" ;;
    heuristic) ;;
    *) printf 'Unsupported REWARD_BACKEND: %s\n' "$REWARD_BACKEND" >&2; exit 2 ;;
esac
for data_file in "$TRAIN_FILE" "$VAL_FILE"; do
    if [[ ! -f "$data_file" ]]; then
        printf 'Parquet file does not exist: %s\n' "$data_file" >&2
        exit 2
    fi
done

PPO_MINI_BATCH_SIZE=$(( TRAINING_GPUS * MICRO_BATCH_PER_GPU * GRAD_ACC_STEPS ))
TOTAL_SAMPLES=$(( ROLLOUT_N_QUERY * ROLLOUT_N_SAMPLE ))
if (( PPO_MINI_BATCH_SIZE < 1 || TOTAL_SAMPLES < 1 || TOTAL_SAMPLES % PPO_MINI_BATCH_SIZE != 0 )); then
    printf 'Rollout batch size (%s) must be a positive multiple of the PPO mini-batch size (%s).\n' \
        "$TOTAL_SAMPLES" "$PPO_MINI_BATCH_SIZE" >&2
    exit 2
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export REWARD_BACKEND REWARD_MODEL_PATH REWARD_BASE_MODEL_PATH REWARD_DEVICE REWARD_MAX_LENGTH REWARD_BATCH_SIZE
mkdir -p "$OUTPUT_DIR"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    "data.train_files='$TRAIN_FILE'" \
    "data.val_files='$VAL_FILE'" \
    "data.train_batch_size=$ROLLOUT_N_QUERY" \
    data.max_prompt_length=512 \
    data.max_response_length=8192 \
    "actor_rollout_ref.model.path='$MODEL_PATH'" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    "actor_rollout_ref.actor.ppo_mini_batch_size=$(( PPO_MINI_BATCH_SIZE / ROLLOUT_N_SAMPLE ))" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_PER_GPU" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    "actor_rollout_ref.rollout.n=$ROLLOUT_N_SAMPLE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    "trainer.logger=${VERL_LOGGERS:-[console]}" \
    trainer.project_name=TriConList \
    "trainer.experiment_name='$RUN_NAME'" \
    trainer.nnodes=1 \
    "trainer.default_local_dir='$OUTPUT_DIR'" \
    "trainer.n_gpus_per_node=$TRAINING_GPUS" \
    trainer.save_freq=512 \
    trainer.test_freq=32 \
    "trainer.total_epochs=${MAX_EPOCHS:-2}" \
    reward_model.reward_manager=batch \
    "custom_reward_function.path='$ROOT_DIR/custom_reward_fn.py'" \
    custom_reward_function.name=compute_score \
    "$@" 2>&1 | tee "$OUTPUT_DIR/training.log"
