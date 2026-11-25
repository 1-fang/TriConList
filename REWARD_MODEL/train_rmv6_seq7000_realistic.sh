#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-data/reward_model}"
DATA_FILE="${DATA_FILE:-dataset.jsonl}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
REPORT_TO="${REPORT_TO:-none}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/reward-model/combined-7000-$(date +%Y%m%d-%H%M%S)}"

torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port=29506 \
    --module REWARD_MODEL.listwise_reward_model.listwise_train \
    --data_dir "$DATA_DIR" \
    --data_file "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path "$MODEL_PATH" \
    --max_seq_length 7000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-4 \
    --num_train_epochs 14 \
    --warmup_ratio 0.08 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 0.7 \
    --lr_scheduler_type cosine \
    --eval_strategy steps \
    --eval_steps 25 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 8 \
    --logging_steps 5 \
    --loss_type combined \
    --use_classification False \
    --loss_weight_listmle 0.95 \
    --loss_weight_listnet 0.0 \
    --loss_weight_ranknet 0.0 \
    --loss_weight_ndcg 0.05 \
    --ranking_temperature 1.0 \
    --ranking_margin 1.0 \
    --seed 42 \
    --run_name "$(basename "$OUTPUT_DIR")" \
    --do_train \
    --do_eval \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --bf16 \
    --dataloader_num_workers 1 \
    --dataloader_pin_memory false \
    --torch_compile false \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --eval_ranking_metrics \
    --remove_unused_columns False \
    --use_listwise_head \
    --listwise_hidden_dim 256 \
    --dropout_rate 0.08 \
    --lora_enable \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.08 \
    --num_lora_modules 32 \
    --use_rslora \
    --report_to "$REPORT_TO" \
    --logging_first_step \
    --optim adamw_torch_fused \
    --skip_memory_metrics \
    --save_safetensors \
    --ddp_find_unused_parameters true \
    --torch_empty_cache_steps 5 "$@"
