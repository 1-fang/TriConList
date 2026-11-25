"""
Listwise Ranking
listwise reward model
"""

import os
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import logging
from peft import PeftModel
from .listwise_losses import create_listwise_loss
from .ranking_metrics import RankingMetrics

logger = logging.get_logger(__name__)


class ListwiseTrainer(Trainer):
    """
    Listwise Ranking
    reward model
    listwise
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        data_collator=None,
        compute_metrics=None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple = (None, None),
        preprocess_logits_for_metrics=None,
        # Listwise specific arguments
        loss_type: str = "combined",
        loss_kwargs: Optional[Dict] = None,
        ranking_metrics_k: List[int] = [1, 3, 5],
        eval_ranking_metrics: bool = True,
        **kwargs,
    ):
        """
        Listwise
               Args:
        loss_type: ['listmle', 'listnet', 'ranknet', 'ndcg', 'combined']
        loss_kwargs:
        ranking_metrics_k: @kk
        eval_ranking_metrics:
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs,
        )
        # listwise
        self.loss_kwargs = loss_kwargs or {}
        self.loss_function = create_listwise_loss(loss_type, **self.loss_kwargs)
        self.eval_ranking_metrics = eval_ranking_metrics
        if self.eval_ranking_metrics:
            self.ranking_evaluator = RankingMetrics(k_values=ranking_metrics_k)
        self.loss_type = loss_type
        logger.info(f"Listwise，: {loss_type}")
        logger.info(f"k: {ranking_metrics_k}")

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        listwise
               Args:
        model:
        inputs:
        return_outputs:
        num_items_in_batch:
               Returns:
        """
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "return_dict": True,
        }
        outputs = model(**model_inputs)
        logits = outputs[
            "logits"
        ]  # [batch_size, max_candidates] or [batch_size, max_candidates, output_dim]
        # logits2D
        if logits.dim() == 3 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)  # [batch_size, max_candidates]
        labels = inputs["labels"]  # [batch_size, max_candidates]
        num_candidates = inputs["num_candidates"]  # [batch_size]
        # listwise
        loss = self.loss_function(logits, labels, num_candidates)
        if hasattr(self.loss_function, "_debug"):
            self.loss_function._debug = (
                self.args.logging_steps > 0
                and self.state.global_step % self.args.logging_steps == 0
            )
        outputs_dict = {
            "logits": logits,
            "labels": labels,
            "num_candidates": num_candidates,
            "loss": loss,
        }
        if return_outputs:
            return loss, outputs_dict
        else:
            return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        （）
        """
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss, None, None)
        logits = outputs["logits"]  # [batch_size, max_candidates]
        labels = outputs["labels"]  # [batch_size, max_candidates]
        loss = loss.detach()
        logits = nested_detach(logits)
        labels = nested_detach(labels)
        return loss, logits, (labels, nested_detach(outputs["num_candidates"]))

    def compute_ranking_metrics(self, eval_pred) -> Dict[str, float]:
        """
               Args:
        eval_pred: EvalPrediction，predictionslabels
               Returns:
        """
        if not self.eval_ranking_metrics:
            return {}
        predictions, (labels, num_candidates) = eval_pred
        return self.ranking_evaluator.compute_all_metrics(
            torch.as_tensor(predictions),
            torch.as_tensor(labels),
            torch.as_tensor(num_candidates),
        )

    def log_metrics(
        self, split: str, metrics: Dict[str, float], combined: bool = True
    ) -> None:
        """
        （）
        """
        if not self.is_world_process_zero():
            return
        # (: Transformers  log_metrics  split  metrics)
        super().log_metrics(split, metrics)
        if split == "eval" and self.eval_ranking_metrics:
            ranking_metrics = {
                k: v
                for k, v in metrics.items()
                if any(
                    metric_name in k
                    for metric_name in [
                        "ndcg",
                        "map",
                        "precision",
                        "kendall",
                        "spearman",
                        "ranking_acc",
                        "pairwise",
                    ]
                )
            }
            if ranking_metrics:
                logger.info(f"\n=== {split.upper()} ===")
                basic_metrics = [
                    "kendall_tau",
                    "spearman_rho",
                    "pairwise_accuracy",
                    "mrr",
                ]
                for metric in basic_metrics:
                    if metric in ranking_metrics:
                        logger.info(f"{metric}: {ranking_metrics[metric]:.4f}")
                # @k
                k_values = sorted(
                    set(
                        [
                            int(k.split("@")[1])
                            for k in ranking_metrics.keys()
                            if "@" in k
                        ]
                    )
                )
                for k in k_values:
                    logger.info(f"\n--- Top-{k} ---")
                    for metric_name in ["ndcg", "map", "precision", "ranking_acc"]:
                        key = f"{metric_name}@{k}"
                        if key in ranking_metrics:
                            logger.info(f"{key}: {ranking_metrics[key]:.4f}")

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        （）
        """
        # ，compute_metrics
        if self.eval_ranking_metrics and self.compute_metrics is None:
            self.compute_metrics = self.compute_ranking_metrics
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        return metrics

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        model = self.accelerator.unwrap_model(self.model)
        if not isinstance(model, PeftModel):
            return

        import json
        from pathlib import Path
        from safetensors.torch import save_file

        output = Path(output_dir or self.args.output_dir)
        base_model = model.get_base_model()
        metadata = base_model.config.triconlist_reward
        (output / "reward_model_config.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        if base_model.use_listwise_head:
            # PEFT saves the LoRA deltas. Save the head's underlying weights as well.
            base_weights = {}
            for name, layer in base_model.listwise_head.named_children():
                if hasattr(layer, "get_base_layer"):
                    layer = layer.get_base_layer()
                for key, weight in layer.state_dict().items():
                    base_weights[f"{name}.{key}"] = weight.detach().cpu().contiguous()
            save_file(base_weights, output / "reward_head.safetensors")

    def save_metrics(
        self, split: str, metrics: Dict[str, float], combined: bool = True
    ) -> None:
        """ """
        super().save_metrics(split, metrics)
        if split == "eval" and self.eval_ranking_metrics:
            ranking_metrics = {
                k: v
                for k, v in metrics.items()
                if any(
                    metric_name in k
                    for metric_name in [
                        "ndcg",
                        "map",
                        "precision",
                        "kendall",
                        "spearman",
                        "ranking_acc",
                        "pairwise",
                    ]
                )
            }
            if ranking_metrics:
                ranking_metrics_file = os.path.join(
                    self.args.output_dir, f"{split}_ranking_metrics.json"
                )
                try:
                    import json

                    with open(ranking_metrics_file, "w") as f:
                        json.dump(ranking_metrics, f, indent=2)
                except Exception as e:
                    logger.warning(f": {e}")


@dataclass
class ListwiseTrainingArguments(TrainingArguments):
    """
    ，listwise
    """

    # Listwise specific arguments
    loss_type: str = "combined"
    loss_weight_listmle: float = 1.0
    loss_weight_listnet: float = 0.5
    loss_weight_ranknet: float = 0.3
    loss_weight_ndcg: float = 0.7
    ranking_temperature: float = 1.0
    ranking_margin: float = 1.0
    ranking_k_values: Optional[List[int]] = field(default_factory=lambda: [1, 2, 3])
    eval_ranking_metrics: bool = True
    save_ranking_predictions: bool = False
    max_seq_length: int = 7451
    remove_unused_columns: bool = False
    eval_strategy: str = "steps"

    def __post_init__(self):
        """， __post_init__"""
        eval_strategy = self.eval_strategy
        super().__post_init__()
        if eval_strategy != "no":
            self.eval_strategy = eval_strategy
        # ranking_k_values
        if self.ranking_k_values is None:
            self.ranking_k_values = [1, 2, 3]

    def to_dict(self) -> Dict[str, Any]:
        """（）"""
        d = super().to_dict()
        # listwise
        listwise_args = {
            "loss_type": self.loss_type,
            "loss_weight_listmle": self.loss_weight_listmle,
            "loss_weight_listnet": self.loss_weight_listnet,
            "loss_weight_ranknet": self.loss_weight_ranknet,
            "loss_weight_ndcg": self.loss_weight_ndcg,
            "ranking_temperature": self.ranking_temperature,
            "ranking_margin": self.ranking_margin,
            "ranking_k_values": self.ranking_k_values,
            "eval_ranking_metrics": self.eval_ranking_metrics,
            "save_ranking_predictions": self.save_ranking_predictions,
            "max_seq_length": self.max_seq_length,
        }
        d.update(listwise_args)
        return d


def create_listwise_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset,
    eval_dataset=None,
    data_collator=None,
    output_dir: str = "./listwise_output",
    **trainer_kwargs,
) -> ListwiseTrainer:
    """
    Listwise
       Args:
    model:
    tokenizer:
    train_dataset:
    eval_dataset:
    data_collator:
    output_dir:
    **trainer_kwargs:
       Returns:
    ListwiseTrainer
    """
    default_args = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 3,
        "learning_rate": 5e-5,
        "warmup_steps": 100,
        "logging_steps": 50,
        "save_steps": 500,
        "eval_steps": 500,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "dataloader_pin_memory": True,
        "remove_unused_columns": False,
    }
    args_dict = {**default_args, **trainer_kwargs.get("training_args", {})}
    if "training_args" in trainer_kwargs:
        training_args = trainer_kwargs.pop("training_args")
        if not isinstance(training_args, ListwiseTrainingArguments):
            training_args = ListwiseTrainingArguments(**args_dict)
    else:
        training_args = ListwiseTrainingArguments(**args_dict)
    trainer = ListwiseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **trainer_kwargs,
    )
    return trainer


if __name__ == "__main__":
    """
 Listwise
    """
    print("Listwise！")
    print(":")
    print("1. listwise")
    print("2. ")
    print("3. PEFT")
    print("4. ")
