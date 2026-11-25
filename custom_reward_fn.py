"""Batch reward scoring for TriConList and veRL."""

import os
from functools import lru_cache
from typing import Dict, List, Optional

import torch

try:
    import ray
except ImportError:
    ray = None

from REWARD_MODEL.listwise_reward_model.listwise_model import load_reward_checkpoint


class LocalListwiseRewardModel:
    """
    Local ListwiseRewardModel loader

    Directly loads LoRA adapter for scoring, avoiding additional GPU usage
    """

    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None,
        device: str = "cuda",
        max_length: int = 5100,
    ):
        """
        Args:
            model_path: LoRA adapter path
            base_model_path: Base model path
            device: Device (cuda/cpu)
            max_length: Maximum sequence length
        """
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.device = device
        self.max_length = max_length

        print(f"[LocalListwiseRewardModel] Initializing...")
        print(f"  Device: {self.device}")
        print(f"  Model path: {model_path}")
        print(f"  Base model: {base_model_path}")

        self.model, self.tokenizer = load_reward_checkpoint(
            model_path, base_model_path=base_model_path, device=device
        )

    def score_batch(self, prompts: List[str], responses: List[str]) -> List[float]:
        """
        Batch scoring

        Args:
            prompts: List of prompts
            responses: List of responses

        Returns:
            List of scores
        """
        if len(prompts) != len(responses):
            raise ValueError("Prompts and responses must have the same length")
        return [
            self._score_single(prompt, response)
            for prompt, response in zip(prompts, responses)
        ]

    def _score_single(self, prompt: str, response: str) -> float:
        """Score a single sample"""
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        # 3. Apply chat template
        text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        # 4. Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        # 5. Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 6. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            score = outputs["logits"].item()

        return score


if ray is not None:

    @ray.remote
    class RemoteListwiseRewardActor:
        """Ray actor wrapper so that reward model stays on independent GPU"""

        def __init__(
            self,
            model_path: str,
            base_model_path: str,
            device: str = "cuda",
            max_length: int = 5100,
        ) -> None:
            self._local_model = LocalListwiseRewardModel(
                model_path=model_path,
                base_model_path=base_model_path,
                device=device,
                max_length=max_length,
            )

        def score_batch(self, prompts: List[str], responses: List[str]) -> List[float]:
            return self._local_model.score_batch(prompts, responses)


class SimpleCodeRewardFunction:
    """
    Simple rule-based scoring function

    If you don't want to load reward model, you can use this simple rule-based scoring
    """

    def __init__(self):
        print("[SimpleCodeRewardFunction] Initializing simple rule-based scoring")

    def __call__(self, prompts: List[str], responses: List[str]) -> List[float]:
        """
        Rule-based scoring

        This is an example implementation, you should customize the scoring logic for your task
        """
        scores = []

        for prompt, response in zip(prompts, responses):
            score = 0.0

            # Rule 1: Response is not empty
            if len(response.strip()) > 0:
                score += 0.2

            # Rule 2: Contains code structure
            code_keywords = [
                "def ",
                "class ",
                "function ",
                "return ",
                "import ",
                "const ",
                "let ",
                "var ",
            ]
            if any(keyword in response for keyword in code_keywords):
                score += 0.3

            # Rule 3: Reasonable length
            if 50 < len(response) < 2000:
                score += 0.2

            # Rule 4: No obvious errors
            error_keywords = ["error", "exception", "undefined", "null pointer"]
            if not any(keyword in response.lower() for keyword in error_keywords):
                score += 0.3

            scores.append(score)

        return scores


# Factory function to create the appropriate reward function
def create_custom_reward_fn(
    use_local_model: bool = True,
    model_path: str = None,
    base_model_path: str = None,
    device: str = "cuda",
    max_length: Optional[int] = None,
    use_ray_actor: bool = False,
    ray_actor_num_gpus: int = 1,
    ray_actor_resources: Optional[Dict[str, float]] = None,
):
    """
    Factory function to create custom reward function

    Args:
        use_local_model: Whether to use local ListwiseRewardModel
        model_path: LoRA adapter path
        base_model_path: Base model path
        device: Device

    Returns:
        Callable reward function
    """
    if use_local_model:
        # Use local ListwiseRewardModel
        if model_path is None:
            raise ValueError("model_path is required for learned reward scoring")

        max_length = max_length or 5100

        if use_ray_actor:
            if ray is None:
                raise RuntimeError(
                    "Ray is not installed, cannot create remote reward actor"
                )

            if not ray.is_initialized():
                # If user has manually started Ray, this will automatically connect
                ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))

            options = {"num_gpus": ray_actor_num_gpus}
            if ray_actor_resources:
                options["resources"] = ray_actor_resources

            actor = RemoteListwiseRewardActor.options(**options).remote(
                model_path=model_path,
                base_model_path=base_model_path,
                device=device,
                max_length=max_length,
            )

            def remote_scorer(prompts: List[str], responses: List[str]) -> List[float]:
                return ray.get(actor.score_batch.remote(prompts, responses))

            return remote_scorer

        client = LocalListwiseRewardModel(
            model_path=model_path,
            base_model_path=base_model_path,
            device=device,
            max_length=max_length,
        )
        return client.score_batch
    else:
        # Use simple rule-based scoring
        simple_fn = SimpleCodeRewardFunction()
        return simple_fn


@lru_cache(maxsize=1)
def _configured_scorer():
    backend = os.environ.get("REWARD_BACKEND", "ray")
    return create_custom_reward_fn(
        use_local_model=backend != "heuristic",
        model_path=os.environ.get("REWARD_MODEL_PATH") or None,
        base_model_path=os.environ.get("REWARD_BASE_MODEL_PATH") or None,
        device=os.environ.get("REWARD_DEVICE", "cuda"),
        max_length=int(os.environ.get("REWARD_MAX_LENGTH", "5100")),
        use_ray_actor=backend == "ray",
    )


def compute_score(data_sources, solution_strs, ground_truths, extra_infos):
    """Score the decoded completions received from veRL's batch reward manager."""
    backend = os.environ.get("REWARD_BACKEND", "ray")
    if not solution_strs:
        return []
    if backend == "heuristic":
        prompts = [""] * len(solution_strs)
    else:
        prompts = [info.get("reward_prompt") if info else None for info in extra_infos]
        if len(prompts) != len(solution_strs) or any(
            not isinstance(prompt, str) for prompt in prompts
        ):
            raise ValueError(
                "Each Parquet row must include a string extra_info.reward_prompt"
            )
    scorer = _configured_scorer()
    batch_size = int(os.environ.get("REWARD_BATCH_SIZE", "16"))
    if batch_size < 1:
        raise ValueError("REWARD_BATCH_SIZE must be positive")
    scores = []
    for start in range(0, len(solution_strs), batch_size):
        end = start + batch_size
        scores.extend(scorer(prompts[start:end], solution_strs[start:end]))
    return scores
