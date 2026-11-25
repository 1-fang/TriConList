#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
NPROC_PER_NODE="${NPROC_PER_NODE:-5}"
REPORT_TO="${REPORT_TO:-none}"
TRAIN_DATASET="${TRAIN_DATASET:-data/reward_model/train.jsonl}"
EVAL_DATASET="${EVAL_DATASET:-data/reward_model/eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/reward-model/classification-$(date +%Y%m%d-%H%M%S)}"

torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="${MASTER_PORT:-29501}" \
    --module REWARD_MODEL.listwise_train_classification \
    --base_model_name_or_path "$BASE_MODEL" \
    --train_data_path "$TRAIN_DATASET" \
    --eval_data_path "$EVAL_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --model_max_length 4096 \
    --num_train_epochs 10 \
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
    --metric_for_best_model accuracy \
    --greater_is_better True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,v_proj" \
    --listwise_hidden_dim 512 \
    --dropout_rate 0.1 \
    --max_candidates_per_query 3 \
    --output_dim 3 \
    --task_type classification \
    --ddp_find_unused_parameters True "$@"
