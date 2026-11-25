#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
NPROC_PER_NODE="${NPROC_PER_NODE:-5}"
REPORT_TO="${REPORT_TO:-none}"
DATA_DIR="${DATA_DIR:-data/reward_model}"
TRAIN_DATA_FILE="${DATA_FILE:-train.jsonl}"
LOSS_TYPE="${LOSS_TYPE:-linear_bt}"
if [[ ${1:-} == bt || ${1:-} == linear_bt ]]; then
    LOSS_TYPE="$1"
    shift
fi
OUTPUT_DIR="${OUTPUT_DIR:-outputs/reward-model/${LOSS_TYPE}-$(date +%Y%m%d-%H%M%S)}"

torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="${MASTER_PORT:-29501}" \
    --module REWARD_MODEL.listwise_reward_model.listwise_train \
    --data_dir "$DATA_DIR" \
    --data_file "$TRAIN_DATA_FILE" \
    --model_name_or_path "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length 4096 \
    --num_train_epochs 30 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 13 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 5e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --dataloader_num_workers 4 \
    --report_to "$REPORT_TO" \
    --run_name "$(basename "$OUTPUT_DIR")" \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --num_lora_modules 128 \
    --use_rslora True \
    --use_listwise_head True \
    --listwise_hidden_dim 512 \
    --dropout_rate 0.1 \
    --output_dim 1 \
    --reward_token last \
    --max_candidates_per_query 3 \
    --shuffle_candidates True \
    --use_classification False \
    --loss_type "$LOSS_TYPE" \
    --ranking_temperature 1.0 \
    --eval_ranking_metrics True \
    --remove_unused_columns False \
    --do_train True \
    --do_eval True \
    --seed 42 \
    --ddp_find_unused_parameters True "$@"
