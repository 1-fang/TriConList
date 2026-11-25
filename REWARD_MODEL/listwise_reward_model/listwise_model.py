"""
Listwise Ranking
reward model
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union, Dict, Tuple
from transformers import (
    Qwen2ForSequenceClassification,
    Qwen2Config,
)
from transformers.cache_utils import Cache


class ListwiseRewardModel(Qwen2ForSequenceClassification):
    """
       Listwise Ranking Reward Model
    listwise，
    query，
    """

    def __init__(
        self,
        config: Qwen2Config,
        output_dim: int = 1,
        reward_token: str = "last",  # 'last', 'mean', 'special'
        special_token_ids: Optional[List[int]] = None,
        use_listwise_head: bool = True,  # listwise head
        listwise_hidden_dim: int = 512,  # listwise head
        dropout_rate: float = 0.1,
        num_quality_classes: int = 3,  # : (best/middle/worst)
    ):
        """
        Listwise Reward Model
               Args:
        config:
        output_dim: （1=, 3=logits）
        reward_token:
        special_token_ids: tokenID
        use_listwise_head: listwise ranking head
        listwise_hidden_dim: listwise head
        dropout_rate: dropout
        num_quality_classes: （3=best/middle/worst）
        """
        config.triconlist_reward = {
            "output_dim": output_dim,
            "reward_token": reward_token,
            "special_token_ids": special_token_ids,
            "use_listwise_head": use_listwise_head,
            "listwise_hidden_dim": listwise_hidden_dim,
            "dropout_rate": dropout_rate,
            "num_quality_classes": num_quality_classes,
        }
        config.num_labels = output_dim
        super().__init__(config)
        self.output_dim = output_dim
        self.reward_token = reward_token
        self.special_token_ids = special_token_ids
        self.use_listwise_head = use_listwise_head
        self.num_quality_classes = num_quality_classes  #
        if self.special_token_ids is not None:
            self.reward_token = "special"
        # listwise head，
        if self.use_listwise_head:
            #: output_dim > 1，logits
            final_output_dim = num_quality_classes if output_dim > 1 else output_dim
            self.listwise_head = nn.Sequential(
                nn.Linear(config.hidden_size, listwise_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(listwise_hidden_dim, listwise_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(listwise_hidden_dim // 2, final_output_dim),  # 3logits
            )
            self._init_listwise_head()
            #: score，DDP
            # listwise_head，score
            for param in self.score.parameters():
                param.requires_grad = False

    def _init_listwise_head(self):
        """listwise head"""
        for module in self.listwise_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """
               Args:
        input_ids: [batch_size, max_candidates, seq_length] [batch_size, seq_length]
        attention_mask: attention mask
        Qwen2ForSequenceClassification
               Returns:
        ，logits
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if input_ids is not None:
            input_shape = input_ids.shape
            if len(input_shape) == 3:
                # Listwise input: [batch_size, max_candidates, seq_length]
                return self._forward_listwise(
                    input_ids,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    use_cache,
                    output_attentions,
                    output_hidden_states,
                    return_dict,
                )
            elif len(input_shape) == 2:
                # Single input: [batch_size, seq_length]
                return self._forward_single(
                    input_ids,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    use_cache,
                    output_attentions,
                    output_hidden_states,
                    return_dict,
                )
            else:
                raise ValueError(f"input_ids: {input_shape}")
        else:
            # inputs_embeds
            return self._forward_single(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )

    def _forward_single(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """ """
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]  # [batch_size, seq_length, hidden_size]
        pooled_features = self._extract_features(
            hidden_states, input_ids, attention_mask
        )  # [batch_size, hidden_size]
        # logits
        if self.use_listwise_head:
            logits = self.listwise_head(pooled_features)
        else:
            logits = self.score(pooled_features)
        return {"logits": logits}

    def _forward_listwise(
        self,
        input_ids: torch.LongTensor,  # [batch_size, max_candidates, seq_length]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Listwise：
        """
        batch_size, max_candidates, seq_length = input_ids.shape
        # 2D
        input_ids_flat = input_ids.view(
            -1, seq_length
        )  # [batch_size * max_candidates, seq_length]
        if attention_mask is not None:
            attention_mask_flat = attention_mask.view(-1, seq_length)
        else:
            attention_mask_flat = None
        transformer_outputs = self.model(
            input_ids_flat,
            attention_mask=attention_mask_flat,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[
            0
        ]  # [batch_size * max_candidates, seq_length, hidden_size]
        pooled_features = self._extract_features(
            hidden_states, input_ids_flat, attention_mask_flat
        )  # [batch_size * max_candidates, hidden_size]
        # logits
        if self.use_listwise_head:
            logits_flat = self.listwise_head(
                pooled_features
            )  # [batch_size * max_candidates, output_dim]
        else:
            logits_flat = self.score(pooled_features)
        # listwise
        logits = logits_flat.view(
            batch_size, max_candidates, self.output_dim
        )  # [batch_size, max_candidates, output_dim]
        # output_dim=1，
        if self.output_dim == 1:
            logits = logits.squeeze(-1)  # [batch_size, max_candidates]
        return {"logits": logits}

    def _extract_features(
        self,
        hidden_states: torch.Tensor,  # [batch_size, seq_length, hidden_size]
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        hidden states
               Returns:
                   pooled_features: [batch_size, hidden_size]
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        if self.reward_token == "last":
            # padding token
            if input_ids is not None and self.config.pad_token_id is not None:
                # padding token
                non_pad_mask = (input_ids != self.config.pad_token_id).to(
                    device, torch.int32
                )
                token_indices = torch.arange(input_ids.shape[-1], device=device)
                last_non_pad_indices = (token_indices * non_pad_mask).argmax(-1)
                pooled_features = hidden_states[
                    torch.arange(batch_size, device=device), last_non_pad_indices
                ]
            else:
                # token
                pooled_features = hidden_states[:, -1, :]
        elif self.reward_token == "mean":
            # token
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1)
                pooled_features = sum_hidden / (sum_mask + 1e-9)
            else:
                pooled_features = hidden_states.mean(dim=1)
        elif self.reward_token == "special":
            # token
            if self.special_token_ids is None or input_ids is None:
                raise ValueError("special tokenspecial_token_idsinput_ids")
            # token mask
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (
                    input_ids == special_token_id
                )
            # token (mask)
            special_token_mask = special_token_mask.to(hidden_states.device)
            special_features = hidden_states[
                special_token_mask, :
            ]  # [num_special_tokens, hidden_size]
            if special_features.shape[0] == 0:
                # token，last
                pooled_features = hidden_states[:, -1, :]
            else:
                # batch
                pooled_features = special_features.view(
                    batch_size, -1, hidden_states.shape[-1]
                )
                pooled_features = pooled_features.mean(dim=1)  # token
        else:
            raise ValueError(f"reward_token: {self.reward_token}")
        return pooled_features

    def get_listwise_scores(
        self,
        input_ids: torch.LongTensor,  # [batch_size, max_candidates, seq_length]
        attention_mask: Optional[torch.Tensor] = None,
        num_candidates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        listwise（）
               Args:
                   input_ids: [batch_size, max_candidates, seq_length]
                   attention_mask: [batch_size, max_candidates, seq_length]
        num_candidates: [batch_size] - query
               Returns:
        scores: [batch_size, max_candidates] -
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            scores = outputs["logits"]  # [batch_size, max_candidates]
            if num_candidates is not None:
                batch_size, max_candidates = scores.shape
                for i in range(batch_size):
                    n_cands = num_candidates[i].item()
                    if n_cands < max_candidates:
                        scores[i, n_cands:] = -float("inf")
            return scores

    def rank_candidates(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_candidates: Optional[torch.Tensor] = None,
        return_scores: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
               Args:
                   input_ids: [batch_size, max_candidates, seq_length]
                   attention_mask: [batch_size, max_candidates, seq_length]
        num_candidates: [batch_size] - query
        return_scores:
               Returns:
        rankings: [batch_size, max_candidates] - （）
        scores: [batch_size, max_candidates] - （return_scores=True）
        """
        scores = self.get_listwise_scores(input_ids, attention_mask, num_candidates)
        _, rankings = torch.sort(scores, descending=True, dim=-1)
        if return_scores:
            return rankings, scores
        else:
            return rankings

    def predict_quality_classes(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_candidates: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        : (best/middle/worst)
               Args:
                   input_ids: [batch_size, max_candidates, seq_length]
                   attention_mask: [batch_size, max_candidates, seq_length]
        num_candidates: [batch_size] - query
        return_logits: logits
               Returns:
        predicted_classes: [batch_size, max_candidates] - (0=best, 1=middle, 2=worst)
        logits: [batch_size, max_candidates, num_classes] - logits（return_logits=True）
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            logits = outputs["logits"]  # [batch_size, max_candidates, num_classes]
            # （output_dim=1），None
            if logits.dim() == 2:
                print("⚠️ ，。output_dim=3。")
                return None
            predicted_classes = torch.argmax(
                logits, dim=-1
            )  # [batch_size, max_candidates]
            # -1
            if num_candidates is not None:
                batch_size, max_candidates = predicted_classes.shape
                for i in range(batch_size):
                    n_cands = num_candidates[i].item()
                    if n_cands < max_candidates:
                        predicted_classes[i, n_cands:] = -1
            if return_logits:
                return predicted_classes, logits
            else:
                return predicted_classes


def create_listwise_reward_model(
    model_name_or_path: str,
    use_listwise_head: bool = True,
    listwise_hidden_dim: int = 512,
    dropout_rate: float = 0.1,
    special_tokens: Optional[List[str]] = None,
    **kwargs,
) -> ListwiseRewardModel:
    """
    Listwise Reward Model
       Args:
    model_name_or_path:
    use_listwise_head: listwise head
    listwise_hidden_dim: listwise head
    dropout_rate: dropout
    special_tokens: token
    **kwargs: from_pretrained
       Returns:
    ListwiseRewardModel
    """
    model = ListwiseRewardModel.from_pretrained(
        model_name_or_path,
        use_listwise_head=use_listwise_head,
        listwise_hidden_dim=listwise_hidden_dim,
        dropout_rate=dropout_rate,
        **kwargs,
    )
    return model


if __name__ == "__main__":
    """
 Listwise Reward Model
    """
    from transformers import AutoTokenizer

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print("tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("...")
    model = create_listwise_reward_model(
        model_name, use_listwise_head=True, listwise_hidden_dim=256
    )
    batch_size, max_candidates, seq_length = 2, 3, 128
    input_ids = torch.randint(1, 1000, (batch_size, max_candidates, seq_length))
    attention_mask = torch.ones_like(input_ids)
    num_candidates = torch.tensor([3, 2])  # query2
    print(f": {input_ids.shape}")
    print("...")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    print(f"logits: {outputs['logits'].shape}")
    print("...")
    rankings, scores = model.rank_candidates(
        input_ids, attention_mask, num_candidates, return_scores=True
    )
    print(f": {rankings.shape}")
    print(f": {scores.shape}")
    print(f"query: {rankings[0]}")
    print(f"query: {scores[0]}")
    print("！")


def load_reward_checkpoint(
    model_path, tokenizer_path=None, base_model_path=None, device="cpu"
):
    """Load a full reward checkpoint or a PEFT adapter and its reward head."""
    import json
    import warnings
    from pathlib import Path

    from peft import PeftModel
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoTokenizer

    checkpoint = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or model_path, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    adapter_config = checkpoint / "adapter_config.json"
    if adapter_config.is_file():
        adapter = json.loads(adapter_config.read_text())
        backbone = base_model_path or adapter["base_model_name_or_path"]
        metadata = checkpoint / "reward_model_config.json"
        head_weights = checkpoint / "reward_head.safetensors"
        if metadata.is_file():
            model_options = json.loads(metadata.read_text())
        else:
            model_options = {"listwise_hidden_dim": 256}
            adapter_weights = checkpoint / "adapter_model.safetensors"
            if adapter_weights.is_file():
                with safe_open(adapter_weights, framework="pt") as weights:
                    for name in weights.keys():
                        if "listwise_head.0." in name and (
                            name.endswith(".lora_B.weight")
                            or name.endswith(".0.weight")
                        ):
                            model_options["listwise_hidden_dim"] = weights.get_slice(
                                name
                            ).get_shape()[0]
                            break
            warnings.warn(
                "Legacy adapter lacks reward-head metadata. Its scores may differ from the training run.",
                stacklevel=2,
            )
        model = ListwiseRewardModel.from_pretrained(
            backbone, torch_dtype=dtype, **model_options
        )
        model.resize_token_embeddings(len(tokenizer))
        if head_weights.is_file():
            model.listwise_head.load_state_dict(load_file(head_weights))
        model = PeftModel.from_pretrained(model, checkpoint)
    else:
        config = AutoConfig.from_pretrained(model_path)
        model_options = getattr(config, "triconlist_reward", {})
        model = ListwiseRewardModel.from_pretrained(
            model_path, config=config, torch_dtype=dtype, **model_options
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    return model.to(device).eval(), tokenizer
