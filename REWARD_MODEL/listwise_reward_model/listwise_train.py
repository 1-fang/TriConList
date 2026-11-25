"""
Train a listwise reward model for full-stack code generation.
"""

import os
import json
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from datetime import datetime

from transformers import AutoTokenizer, HfArgumentParser, set_seed
from peft import LoraConfig, get_peft_model


from .listwise_data import (
    ListwiseDataConfig,
    create_listwise_dataset,
    ListwiseDataCollator,
)
from .listwise_model import ListwiseRewardModel
from .listwise_trainer import ListwiseTrainer, ListwiseTrainingArguments


from .utils import PEFTLoraConfig


@dataclass
class ListwiseModelConfig:
    """
    Reward model architecture and backbone settings.
    """

    # model_name_or_path: str = "./models/Qwen2.5-7B-Instruct"
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    model_revision: str = "main"
    torch_dtype: str = "auto"
    use_flash_attn2: bool = False
    use_special_tokens: bool = True

    use_listwise_head: bool = True
    listwise_hidden_dim: int = 512
    dropout_rate: float = 0.1
    output_dim: int = 1
    reward_token: str = "last"  # "last", "mean", "special"

    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass
class WandBConfig:
    """
    Optional Weights & Biases settings.
    """

    enable_wandb: bool = False
    project_name: str = "listwise-reward-model"
    tags: List[str] = field(
        default_factory=lambda: ["listwise", "reward-model", "ranking"]
    )
    log_model: bool = True
    wandb_log_gradients: bool = False
    watch_model: bool = True

    experiment_name: Optional[str] = None
    notes: Optional[str] = None
    group: Optional[str] = None


def save_training_config(
    data_config: ListwiseDataConfig,
    model_config: ListwiseModelConfig,
    training_args: ListwiseTrainingArguments,
    peft_config: Optional[PEFTLoraConfig] = None,
    wandb_config: Optional[WandBConfig] = None,
    output_dir: str = None,
):
    """
    Write the resolved settings to a JSON file.
    """
    config_dict = {
        "data_config": asdict(data_config),
        "model_config": asdict(model_config),
        "training_args": training_args.to_dict(),
    }

    if peft_config is not None:
        config_dict["peft_config"] = asdict(peft_config)

    if wandb_config is not None:
        config_dict["wandb_config"] = asdict(wandb_config)

    if "local_rank" in config_dict["training_args"]:
        del config_dict["training_args"]["local_rank"]
    if "_n_gpu" in config_dict["training_args"]:
        del config_dict["training_args"]["_n_gpu"]

    save_dir = output_dir or training_args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "listwise_training_config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"Saved training configuration to: {config_path}")


def setup_model_and_tokenizer(
    model_config: ListwiseModelConfig, peft_config: Optional[PEFTLoraConfig] = None
):
    """
    Load the reward backbone and prepare its tokenizer.
    """
    print(f"Loading model: {model_config.model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, padding_side="right", trust_remote_code=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = ["<|reward|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
        print(f"Registered special tokens: {special_tokens} -> {special_token_ids}")

    model_kwargs = {
        "torch_dtype": getattr(torch, model_config.torch_dtype)
        if model_config.torch_dtype != "auto"
        else "auto",
        "attn_implementation": "flash_attention_2"
        if model_config.use_flash_attn2
        else "eager",
        "trust_remote_code": False,
    }

    if model_config.load_in_4bit or model_config.load_in_8bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_config.load_in_4bit,
            load_in_8bit=model_config.load_in_8bit,
            bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, model_config.bnb_4bit_compute_dtype),
        )
        model_kwargs["quantization_config"] = quantization_config

    model = ListwiseRewardModel.from_pretrained(
        model_config.model_name_or_path,
        output_dim=model_config.output_dim,
        reward_token=model_config.reward_token,
        special_token_ids=special_token_ids,
        use_listwise_head=model_config.use_listwise_head,
        listwise_hidden_dim=model_config.listwise_hidden_dim,
        dropout_rate=model_config.dropout_rate,
        **model_kwargs,
    )

    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized embeddings to vocabulary size: {len(tokenizer)}")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.tokenizer_padding_side = tokenizer.padding_side

    if peft_config and peft_config.lora_enable:
        print("Applying LoRA adapters")

        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if peft_config.lora_namespan_exclude:
                    if any(
                        exclude in name for exclude in peft_config.lora_namespan_exclude
                    ):
                        continue
                target_modules.append(name)

        if peft_config.num_lora_modules > 0:
            target_modules = target_modules[-peft_config.num_lora_modules :]

        print(f"LoRA target modules ({len(target_modules)}): {target_modules[:5]}...")

        lora_config = LoraConfig(
            r=peft_config.lora_r,
            lora_alpha=peft_config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=peft_config.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            use_rslora=peft_config.use_rslora,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    """
    Run reward model training.
    """

    parser = HfArgumentParser(
        (
            ListwiseDataConfig,
            ListwiseModelConfig,
            ListwiseTrainingArguments,
            PEFTLoraConfig,
            WandBConfig,
        )
    )

    data_config, model_config, training_args, peft_config, wandb_config = (
        parser.parse_args_into_dataclasses()
    )

    set_seed(training_args.seed)

    print("=" * 80)
    print("TriConList reward model training")
    print("=" * 80)
    print(f"Data directory: {data_config.data_dir}")
    print(f"Model: {model_config.model_name_or_path}")
    print(f"Output directory: {training_args.output_dir}")
    print(f"Loss: {training_args.loss_type}")
    print(f"LoRA enabled: {peft_config.lora_enable}")
    print(f"W&B enabled: {wandb_config.enable_wandb}")
    if wandb_config.enable_wandb:
        print(f"W&B project: {wandb_config.project_name}")
        print(f"Experiment: {wandb_config.experiment_name or 'auto-generated'}")

    print("=" * 80)

    os.makedirs(training_args.output_dir, exist_ok=True)

    if training_args.should_save:
        save_training_config(
            data_config, model_config, training_args, peft_config, wandb_config
        )

    print("Loading ranked examples")
    train_dataset, eval_dataset = create_listwise_dataset(data_config)

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")

    model, tokenizer = setup_model_and_tokenizer(model_config, peft_config)

    data_collator = ListwiseDataCollator(
        tokenizer=tokenizer, max_length=getattr(training_args, "max_seq_length", 2048)
    )

    print("Collating a sample batch")
    sample_batch = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
    test_batch = data_collator(sample_batch)
    print(f"Sample batch shapes:")
    print(f"  input_ids: {test_batch['input_ids'].shape}")
    print(f"  attention_mask: {test_batch['attention_mask'].shape}")
    print(f"  labels: {test_batch['labels'].shape}")
    print(f"  num_candidates: {test_batch['num_candidates']}")

    loss_kwargs = {
        "temperature": training_args.ranking_temperature,
    }

    if training_args.loss_type == "combined":
        loss_kwargs["weights"] = [
            training_args.loss_weight_listmle,
            training_args.loss_weight_listnet,
            training_args.loss_weight_ranknet,
            training_args.loss_weight_ndcg,
        ]

    callbacks = []
    if wandb_config.enable_wandb:
        print("Initializing W&B")

        if training_args.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            exp_name = wandb_config.experiment_name or "listwise-reward"
            training_args.run_name = f"{exp_name}-{timestamp}"

        wandb_full_config = {
            "model_config": asdict(model_config),
            "data_config": asdict(data_config),
            "training_config": training_args.to_dict(),
            "loss_config": loss_kwargs,
        }

        if wandb_config.experiment_name:
            wandb_full_config["experiment"] = {
                "name": wandb_config.experiment_name,
                "notes": wandb_config.notes,
                "group": wandb_config.group,
            }

        # Simple WandB initialization (direct import)
        try:
            import wandb

            # Ensure API key is set (should be set via environment variable)
            wandb.init(
                project=wandb_config.project_name,
                name=training_args.run_name,
                config=wandb_full_config,
            )
            print(f"WandB initialized - Project: {wandb_config.project_name}")
        except ImportError:
            print("WandB not installed, skipping logging")

    trainer = ListwiseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        loss_type=training_args.loss_type,
        loss_kwargs=loss_kwargs,
        ranking_metrics_k=training_args.ranking_k_values,
        eval_ranking_metrics=training_args.eval_ranking_metrics,
        callbacks=callbacks,
    )

    print(f"Training device: {training_args.device}")
    print(f"Per-device batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"Training epochs: {training_args.num_train_epochs}")
    print(f"Learning rate: {training_args.learning_rate}")

    print("\nStarting training")
    try:
        train_result = trainer.train()

        print("Training complete")
        print(f"Training statistics: {train_result}")

        print("Saving final reward checkpoint")
        trainer.save_model()

        trainer.save_state()

        if training_args.do_eval:
            print("Running final validation")
            try:
                eval_results = trainer.evaluate()
                print(f"Validation results: {eval_results}")
            except Exception as e:
                print(f"Validation failed: {e}")
                print("Training has already finished")

        print(f"Saved output to: {training_args.output_dir}")

    except KeyboardInterrupt:
        print("Training interrupted")
        print("Saving the current checkpoint")
        trainer.save_model()
        trainer.save_state()

    except Exception as e:
        print(f"Training failed: {e}")
        print("Attempting to save the current checkpoint")
        try:
            trainer.save_model()
            trainer.save_state()
        except:
            print("Unable to save the checkpoint")
        raise


if __name__ == "__main__":
    main()
