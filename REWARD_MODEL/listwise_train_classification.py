#!/usr/bin/env python3
"""
Listwise Reward Model
: best (2) / middle (1) / worst (0)
"""

import os
import json
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from .listwise_reward_model.listwise_model import ListwiseRewardModel
from .listwise_reward_model.listwise_data import (
    ListwiseDataConfig,
    ListwiseDataset,
    ListwiseDataCollator,
)
from .listwise_reward_model.classification_trainer import create_classification_trainer
from .listwise_reward_model.listwise_trainer import ListwiseTrainingArguments

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """"""

    base_model_name_or_path: str = field(metadata={"help": ""})
    output_dim: int = field(default=3, metadata={"help": " (3=: best/middle/worst)"})
    reward_token: str = field(
        default="last", metadata={"help": "token: last, mean, special"}
    )
    use_listwise_head: bool = field(default=True, metadata={"help": "listwise head"})
    listwise_hidden_dim: int = field(default=512, metadata={"help": "Listwise head"})
    dropout_rate: float = field(default=0.1, metadata={"help": "Dropout"})
    task_type: str = field(
        default="classification", metadata={"help": ": regression classification"}
    )


@dataclass
class DataArguments:
    """"""

    train_data_path: str = field(metadata={"help": ""})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": ""})
    model_max_length: int = field(default=5100, metadata={"help": ""})
    max_candidates_per_query: int = field(default=3, metadata={"help": "query"})


@dataclass
class LoraArguments:
    """LoRA"""

    use_lora: bool = field(default=True, metadata={"help": "LoRA"})
    lora_r: int = field(
        default=16,  # 16, 64，
        metadata={"help": "LoRA rank"},
    )
    lora_alpha: int = field(
        default=32,  # 32, 128，
        metadata={"help": "LoRA alpha"},
    )
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={
            "help": "LoRA，"
        },  # "q_proj,k_proj,v_proj","q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, LoraArguments, ListwiseTrainingArguments)
    )
    model_args, data_args, lora_args, training_args = (
        parser.parse_args_into_dataclasses()
    )
    set_seed(training_args.seed)
    logger.info("=" * 80)
    logger.info(" Reward Model")
    logger.info("=" * 80)
    logger.info(f" : {model_args.base_model_name_or_path}")
    logger.info(f" : {model_args.output_dim} ()")
    logger.info(f" : best (2) / middle (1) / worst (0)")
    logger.info(f" : {data_args.train_data_path}")
    logger.info(f" : {training_args.output_dir}")
    logger.info(f" LoRA: r={lora_args.lora_r}, alpha={lora_args.lora_alpha}")
    logger.info("=" * 80)
    # 1. tokenizer
    logger.info("\n tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.base_model_name_or_path,
        trust_remote_code=False,
        padding_side="right",
        model_max_length=data_args.model_max_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f" pad_tokeneos_token: {tokenizer.eos_token}")
    # 2.
    logger.info("Loading reward model")
    model = ListwiseRewardModel.from_pretrained(
        model_args.base_model_name_or_path,
        output_dim=model_args.output_dim,  # 3
        reward_token=model_args.reward_token,
        use_listwise_head=model_args.use_listwise_head,
        listwise_hidden_dim=model_args.listwise_hidden_dim,
        dropout_rate=model_args.dropout_rate,
        num_quality_classes=3,  # 3
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    # embedding
    model.resize_token_embeddings(len(tokenizer))
    # 3. LoRA
    if lora_args.use_lora:
        logger.info("\n LoRA...")
        # target modules
        target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",")]
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        # LoRA + Gradient Checkpointing input_require_grads
        # Trainergradient_checkpointing_enable(),
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
            logger.info(" input_require_grads ( LoRA + Gradient Checkpointing)")
    # 4.
    logger.info("Loading ranked data")
    train_data_dir = os.path.dirname(data_args.train_data_path)
    train_data_file = os.path.basename(data_args.train_data_path)
    data_config = ListwiseDataConfig(
        data_dir=train_data_dir,  #
        data_file=train_data_file,  #
        model_max_length=data_args.model_max_length,
        max_candidates_per_query=data_args.max_candidates_per_query,
        shuffle_candidates=True,  #
    )
    train_dataset = ListwiseDataset(data_config=data_config)
    logger.info(f" : {len(train_dataset)}")
    eval_dataset = None
    if data_args.eval_data_path:
        eval_data_dir = os.path.dirname(data_args.eval_data_path)
        eval_data_file = os.path.basename(data_args.eval_data_path)
        eval_data_config = ListwiseDataConfig(
            data_dir=eval_data_dir,
            data_file=eval_data_file,  #
            model_max_length=data_args.model_max_length,
            max_candidates_per_query=data_args.max_candidates_per_query,
            shuffle_candidates=True,
        )
        eval_dataset = ListwiseDataset(data_config=eval_data_config)
        logger.info(f" : {len(eval_dataset)}")
    # 5. data collator
    data_collator = ListwiseDataCollator(
        tokenizer=tokenizer, max_length=data_args.model_max_length
    )
    # 6.
    logger.info("\n ...")
    trainer = create_classification_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        training_args=training_args,
        output_dir=training_args.output_dir,
    )
    # 7.
    logger.info("\n ...")
    config_dict = {
        "model_config": {
            "base_model": model_args.base_model_name_or_path,
            "output_dim": model_args.output_dim,
            "reward_token": model_args.reward_token,
            "use_listwise_head": model_args.use_listwise_head,
            "listwise_hidden_dim": model_args.listwise_hidden_dim,
            "dropout_rate": model_args.dropout_rate,
            "task_type": "classification",
            "num_classes": 3,
            "class_names": ["worst", "middle", "best"],
        },
        "data_config": {
            "train_data_path": data_args.train_data_path,
            "eval_data_path": data_args.eval_data_path,
            "model_max_length": data_args.model_max_length,
            "max_candidates_per_query": data_args.max_candidates_per_query,
        },
        "lora_config": {
            "use_lora": lora_args.use_lora,
            "lora_r": lora_args.lora_r,
            "lora_alpha": lora_args.lora_alpha,
            "lora_dropout": lora_args.lora_dropout,
            "lora_target_modules": target_modules if lora_args.use_lora else None,
        },
        "training_config": {
            "num_train_epochs": training_args.num_train_epochs,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "warmup_ratio": training_args.warmup_ratio,
            "bf16": training_args.bf16,
            "gradient_checkpointing": training_args.gradient_checkpointing,
        },
    }
    os.makedirs(training_args.output_dir, exist_ok=True)
    config_path = os.path.join(
        training_args.output_dir, "listwise_training_config.json"
    )
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
        logger.info(f" : {config_path}")
    # 8.
    logger.info("\n" + "=" * 80)
    logger.info(" ...")
    logger.info("=" * 80)
    train_result = trainer.train()
    # 9.
    logger.info("\n ...")
    trainer.save_model(training_args.output_dir)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    # 10.
    if eval_dataset is not None:
        logger.info("\n ...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    logger.info("\n" + "=" * 80)
    logger.info("✅ ！")
    logger.info("=" * 80)
    logger.info(f" : {training_args.output_dir}")
    logger.info(f" : best (2) / middle (1) / worst (0)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
