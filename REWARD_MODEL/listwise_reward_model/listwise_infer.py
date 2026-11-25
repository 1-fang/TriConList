"""
Listwise Ranking Reward Model
"""

import json
import torch
import argparse
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from .listwise_model import load_reward_checkpoint
from .listwise_data import (
    ListwiseDataCollator,
    build_fullstack_prompt,
    format_source_code,
)
from .ranking_metrics import RankingMetrics


class ListwiseRewardInference:
    """
    Listwise Reward Model
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "auto",
        max_length: int = 7451,
        batch_size: int = 4,
        base_model_path: Optional[str] = None,
    ):
        """
               Args:
        model_path:
        tokenizer_path: tokenizer（model_path）
        device:
        max_length:
        batch_size:
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_length = max_length
        self.batch_size = batch_size
        self.model, self.tokenizer = load_reward_checkpoint(
            model_path, self.tokenizer_path, base_model_path, self.device
        )
        self.data_collator = ListwiseDataCollator(
            tokenizer=self.tokenizer, max_length=max_length
        )

    def rank_candidates(
        self,
        prompt: str,
        candidates: List[Union[str, List[Dict]]],
        return_scores: bool = True,
        return_details: bool = False,
    ) -> Dict:
        """
               Args:
        prompt:
        candidates: ，
        return_scores:
        return_details:
               Returns:
        """
        # - ：candidate
        formatted_candidates = []
        full_prompt = build_fullstack_prompt(prompt)
        for candidate in candidates:
            if isinstance(candidate, str):
                formatted_code = candidate
            elif isinstance(candidate, list):
                formatted_code = format_source_code(candidate)
            else:
                raise ValueError(f": {type(candidate)}")
            conversation = [
                {"role": "user", "content": full_prompt},
                {"role": "assistant", "content": formatted_code},
            ]
            formatted_candidates.append(conversation)
        # ：candidate
        candidate_scores = []
        for i, candidate_conversation in enumerate(formatted_candidates):
            # candidatesample
            single_sample = {
                "candidates": [candidate_conversation],  # candidate
                "scores": [0.0],  #
                "query_id": f"inference_{hash(prompt) % 10000}_{i}",
                "num_candidates": 1,
            }
            # candidate
            single_batch = self.data_collator([single_sample])
            for key in ["input_ids", "attention_mask", "labels", "num_candidates"]:
                if key in single_batch and isinstance(single_batch[key], torch.Tensor):
                    single_batch[key] = single_batch[key].to(self.device)
            # candidate
            with torch.no_grad():
                model_inputs = {
                    "input_ids": single_batch["input_ids"],
                    "attention_mask": single_batch["attention_mask"],
                    "return_dict": True,
                }
                outputs = self.model(**model_inputs)
                logits = outputs["logits"]  # [1, 1] or [1, 1, 1]
                if logits.dim() == 3:
                    score = float(logits[0, 0, 0])
                elif logits.dim() == 2:
                    score = float(logits[0, 0])
                else:
                    score = float(logits[0])
                candidate_scores.append(score)
        scored_candidates = [(score, i) for i, score in enumerate(candidate_scores)]
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        results = {
            "ranked_indices": [idx for _, idx in scored_candidates],
            "ranked_candidates": [candidates[idx] for _, idx in scored_candidates],
        }
        if return_scores:
            results["scores"] = [score for score, _ in scored_candidates]
            results["original_scores"] = candidate_scores
        if return_details:
            results["prompt"] = prompt
            results["num_candidates"] = len(candidates)
            results["model_info"] = {
                "model_path": self.model_path,
                "max_length": self.data_collator.max_length,
            }
        return results

    def evaluate_single_candidate(
        self, prompt: str, candidate: Union[str, List[Dict]]
    ) -> float:
        """
               Args:
        prompt:
        candidate:
               Returns:
        """
        result = self.rank_candidates(prompt, [candidate], return_scores=True)
        return result["scores"][0]

    def batch_rank_candidates(
        self, queries_and_candidates: List[Tuple[str, List]], return_scores: bool = True
    ) -> List[Dict]:
        """
        query
               Args:
        queries_and_candidates: (prompt, candidates)
        return_scores:
               Returns:
        """
        results = []
        for prompt, candidates in queries_and_candidates:
            try:
                result = self.rank_candidates(
                    prompt, candidates, return_scores=return_scores
                )
                results.append(result)
            except Exception as e:
                print(f"query: {e}")
                results.append(
                    {
                        "ranked_indices": list(range(len(candidates))),
                        "num_candidates": len(candidates),
                        "error": str(e),
                    }
                )
        return results

    def compare_candidates(
        self,
        prompt: str,
        candidate_a: Union[str, List[Dict]],
        candidate_b: Union[str, List[Dict]],
    ) -> Dict:
        """
               Args:
        prompt:
        candidate_a: A
        candidate_b: B
               Returns:
        """
        result = self.rank_candidates(
            prompt, [candidate_a, candidate_b], return_scores=True, return_details=True
        )
        scores = result["original_scores"]
        ranking = result["ranked_indices"]
        return {
            "winner": "A" if ranking[0] == 0 else "B",
            "score_a": scores[0],
            "score_b": scores[1],
            "score_difference": abs(scores[0] - scores[1]),
            "confidence": "high"
            if abs(scores[0] - scores[1]) > 1.0
            else "medium"
            if abs(scores[0] - scores[1]) > 0.5
            else "low",
        }


def load_test_data(data_path: str) -> List[Dict]:
    """Read ranked examples from JSON, JSONL, or a legacy directory."""
    path = Path(data_path)
    if path.is_dir():
        return [
            json.loads(item.read_text()) for item in sorted(path.glob("*/data.json"))
        ]
    with path.open(encoding="utf-8") as source:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in source if line.strip()]
        data = json.load(source)
    return data if isinstance(data, list) else [data]


def evaluate_on_test_data(
    inference_engine: ListwiseRewardInference,
    test_data: List[Dict],
    output_file: Optional[str] = None,
) -> Dict:
    """
       Args:
    inference_engine:
    test_data:
    output_file:
       Returns:
    """
    print(f"Evaluating {len(test_data)} ranked examples")
    evaluator = RankingMetrics(k_values=[1, 2, 3])
    all_predictions = []
    all_labels = []
    all_num_candidates = []
    detailed_results = []
    for i, item in enumerate(test_data):
        prompt = item.get("prompt", "")
        responses = item.get("responses", [])
        true_ranking = item.get("ranking", [])
        if len(responses) < 2:
            continue
        candidates = []
        for response in responses:
            if "source_code" in response:
                candidates.append(response["source_code"])
            else:
                candidates.append(response.get("text", ""))
        try:
            result = inference_engine.rank_candidates(
                prompt, candidates, return_scores=True
            )
            pred_ranking = result["ranked_indices"]
            pred_scores = result["original_scores"]
            # ：ranking[0]，
            true_scores = [0.0] * len(responses)
            for rank, response_id in enumerate(true_ranking):
                # response_id
                for j, response in enumerate(responses):
                    if response.get("id") == response_id:
                        true_scores[j] = len(true_ranking) - rank - 1  #
                        break
            all_predictions.append(pred_scores)
            all_labels.append(true_scores)
            all_num_candidates.append(len(candidates))
            detailed_result = {
                "query_id": i,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "true_ranking": true_ranking,
                "pred_ranking": pred_ranking,
                "true_scores": true_scores,
                "pred_scores": pred_scores,
                "num_candidates": len(candidates),
            }
            detailed_results.append(detailed_result)
            if (i + 1) % 10 == 0:
                print(f" {i + 1}/{len(test_data)} ")
        except Exception as e:
            print(f"Unable to score example {i + 1}: {e}")
            continue
    if all_predictions:
        predictions_tensor = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(scores) for scores in all_predictions], batch_first=True
        )
        labels_tensor = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(scores) for scores in all_labels], batch_first=True
        )
        num_candidates_tensor = torch.tensor(all_num_candidates)
        metrics = evaluator.compute_all_metrics(
            predictions_tensor, labels_tensor, num_candidates_tensor
        )
        print("\nRanking metrics")
        print(evaluator.format_metrics_string(metrics))
        if output_file:
            results = {
                "metrics": metrics,
                "detailed_results": detailed_results,
                "summary": {
                    "total_samples": len(test_data),
                    "processed_samples": len(all_predictions),
                    "success_rate": len(all_predictions) / len(test_data),
                },
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Saved results to {output_file}")
        return metrics
    else:
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a TriConList reward checkpoint"
    )
    parser.add_argument(
        "--model_path", required=True, help="Local reward checkpoint directory"
    )
    parser.add_argument(
        "--base_model_path", help="Override the adapter's base model ID or path"
    )
    parser.add_argument("--tokenizer_path", help="Override the checkpoint tokenizer")
    parser.add_argument(
        "--test_data", required=True, help="Ranked examples as JSONL or JSON"
    )
    parser.add_argument("--output_file", help="Optional path for detailed JSON results")
    parser.add_argument(
        "--device", default="auto", help="Device such as cpu, cuda, or cuda:0"
    )
    parser.add_argument("--max_length", type=int, default=7451)
    args = parser.parse_args()
    test_data = load_test_data(args.test_data)
    if not test_data:
        parser.error("--test_data contains no examples")
    engine = ListwiseRewardInference(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        max_length=args.max_length,
    )
    if not evaluate_on_test_data(engine, test_data, args.output_file):
        parser.exit(1, "No valid ranked examples were scored.\n")


if __name__ == "__main__":
    main()
