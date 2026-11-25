"""
Reward Model
: best (2), middle (1), worst (0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
from .listwise_trainer import ListwiseTrainer


class ClassificationRewardTrainer(ListwiseTrainer):
    """
    Reward Model
    ListwiseTrainer:
    - CrossEntropyLossranking loss
    -
    - ranking metrics
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CrossEntropyLoss
        self.loss_function = nn.CrossEntropyLoss()
        print(" ")
        print(" : best (2) / middle (1) / worst (0)")
        print(" : CrossEntropyLoss")

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
               Args:
        model:
        inputs:
        return_outputs:
               Returns:
        """
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "return_dict": True,
        }
        outputs = model(**model_inputs)
        logits = outputs["logits"]  # [batch_size, max_candidates, 3] or [batch_size, 3]
        labels = inputs["labels"]  # [batch_size, max_candidates] -
        num_candidates = inputs["num_candidates"]  # [batch_size]
        # 3.0 ->  2 (best)
        # 1.0 ->  1 (middle)
        # 0.0 ->  0 (worst)
        class_labels = self._scores_to_classes(labels, num_candidates)
        # CrossEntropyLoss
        if logits.dim() == 3:
            # Listwise: [batch_size, max_candidates, 3]
            batch_size, max_candidates, num_classes = logits.shape
            # [batch_size * max_candidates, 3]
            logits_flat = logits.view(-1, num_classes)
            labels_flat = class_labels.view(-1)
            # mask，
            valid_mask = torch.zeros(
                batch_size * max_candidates, dtype=torch.bool, device=logits.device
            )
            for i in range(batch_size):
                n_cands = num_candidates[i].item()
                valid_mask[i * max_candidates : i * max_candidates + n_cands] = True
            loss = F.cross_entropy(
                logits_flat[valid_mask], labels_flat[valid_mask], reduction="mean"
            )
        else:
            # : [batch_size, 3]
            loss = F.cross_entropy(logits, class_labels, reduction="mean")
        outputs_dict = {
            "logits": logits,
            "labels": class_labels,  #
            "num_candidates": num_candidates,
            "loss": loss,
        }
        if return_outputs:
            return loss, outputs_dict
        else:
            return loss

    def _scores_to_classes(
        self, scores: torch.Tensor, num_candidates: torch.Tensor
    ) -> torch.Tensor:
        """
        ：
        1. : 0=best, 1=middle, 2=worst ()
        2. : 3.0=best, 1.0=middle, 0.0=worst ()
               Args:
        scores: [batch_size, max_candidates] -
        num_candidates: [batch_size] -
               Returns:
        class_labels: [batch_size, max_candidates] - (2=best, 1=middle, 0=worst)
        """
        batch_size, max_candidates = scores.shape
        class_labels = torch.zeros_like(scores, dtype=torch.long)
        for i in range(batch_size):
            n_cands = num_candidates[i].item()
            sample_scores = scores[i, :n_cands]
            for j in range(n_cands):
                score = sample_scores[j].item()
                # :  0, 1, 2 ()
                # :  3.0, 1.0, 0.0
                if score > 2.5:
                    # : 3.0 -> best (class 2)
                    class_labels[i, j] = 2
                elif abs(score - 2.0) < 0.1:
                    # : 2 -> worst (class 0)
                    class_labels[i, j] = 0
                elif abs(score - 1.0) < 0.1:
                    # : 1.0/1 -> middle (class 1)
                    class_labels[i, j] = 1
                elif abs(score - 0.0) < 0.1:
                    # : 0 -> best (class 2)
                    # : 0.0 -> worst (class 0)
                    #: score=2 ()
                    has_score_2 = any(
                        abs(sample_scores[k].item() - 2.0) < 0.1 for k in range(n_cands)
                    )
                    if has_score_2:
                        # : 0 = best
                        class_labels[i, j] = 2
                    else:
                        # : 0.0 = worst
                        class_labels[i, j] = 0
                else:
                    # worst
                    class_labels[i, j] = 0
        return class_labels

    def compute_ranking_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Returns:
        """
        predictions, (labels, num_candidates) = eval_pred
        pred_classes = torch.as_tensor(predictions).argmax(dim=-1)
        labels = torch.as_tensor(labels)
        num_candidates = torch.as_tensor(num_candidates)
        true_classes = self._scores_to_classes(labels, num_candidates)
        valid_mask = torch.arange(labels.shape[1]).unsqueeze(
            0
        ) < num_candidates.unsqueeze(1)
        correct = (pred_classes == true_classes) & valid_mask
        accuracy = correct.sum().float() / valid_mask.sum().float()
        metrics = {
            "accuracy": accuracy.item(),
        }
        for class_id, class_name in enumerate(["worst", "middle", "best"]):
            class_mask = (true_classes == class_id) & valid_mask
            if class_mask.sum() > 0:
                class_correct = ((pred_classes == class_id) & class_mask).sum().float()
                class_acc = class_correct / class_mask.sum().float()
                metrics[f"accuracy_{class_name}"] = class_acc.item()
        return metrics


def create_classification_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    data_collator=None,
    output_dir: str = "./classification_output",
    **trainer_kwargs,
):
    """
       Args:
    model: (output_dim3)
    tokenizer:
    train_dataset:
    eval_dataset:
    data_collator:
    output_dir:
    **trainer_kwargs:
       Returns:
    ClassificationRewardTrainer
    """
    from .listwise_trainer import ListwiseTrainingArguments

    default_args = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3,
        "learning_rate": 5e-6,
        "warmup_ratio": 0.1,
        "logging_steps": 5,
        "save_steps": 100,
        "eval_steps": 50,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,  #
        "remove_unused_columns": False,
        "bf16": True,
        "gradient_checkpointing": True,
    }
    user_training_args = trainer_kwargs.get("training_args", None)
    if user_training_args is not None:
        if isinstance(user_training_args, ListwiseTrainingArguments):
            training_args = user_training_args
            # kwargs，
            trainer_kwargs.pop("training_args")
        else:
            args_dict = {**default_args, **user_training_args}
            training_args = ListwiseTrainingArguments(**args_dict)
            trainer_kwargs.pop("training_args")
    else:
        training_args = ListwiseTrainingArguments(**default_args)
    trainer = ClassificationRewardTrainer(
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
    print("Reward Model")
    print(": best (2) / middle (1) / worst (0)")
