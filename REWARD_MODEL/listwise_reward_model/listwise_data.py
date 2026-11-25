"""
Listwise Ranking
，listwise reward model
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
import random


# from prompt_template.action import ACTION_PROMPT  # ，prompt
@dataclass
class ListwiseDataConfig:
    """
    Listwise
    """

    data_dir: str = "data/reward_model"
    data_file: str = "dataset.jsonl"  # ( train.jsonl eval.jsonl)
    max_samples_per_folder: int = -1  # ，-1
    train_test_split: float = 0.8  # Legacy configuration field
    max_candidates_per_query: int = 3  # query
    shuffle_candidates: bool = True  #
    model_max_length: int = 5100
    use_classification: bool = True  # : (best/middle/worst)


def build_fullstack_prompt(prompt_text: str) -> str:
    """
    prompt
    """
    return f"""You are an expert full-stack developer. Generate high-quality full-stack web application code based on the following requirements:
{prompt_text}
Ensure your code meets these critical quality standards:
**Frontend-Backend Consistency (MOST IMPORTANT)**:
- API endpoint paths must match exactly between frontend and backend
- Request/response data structures must correspond precisely
- Technology stack versions and dependencies must be unified
- Proper error handling and status codes
**UI/UX Excellence**:
- Complete form validation with clear feedback messages
- Loading states and progress indicators for all async operations
- Consistent styling and visual hierarchy
- Responsive design with proper breakpoints
- Interactive elements with hover/focus states
- Unified icon and emoji standards
- Error messages with consistent styling and placement
**Code Quality**:
- Standardized and semantic function/variable naming
- JSDoc comments for key functions
- Complete error handling and edge cases
- Clear code structure, easy to maintain
- Proper async/await usage and state management
**Architecture Design**:
- Modular organization with separation of concerns
- High component reusability
- Follow technology stack best practices
- Extensible and testable code
- Proper data flow and state management
**Focus Areas for Evaluation**:
- Perfect implementation should have flawless frontend-backend integration AND excellent UI/UX
- Lower quality implementations may have UI/UX issues like: missing form validation feedback, absent loading indicators, inconsistent styling, broken responsive design, or missing interactive effects
- These UI/UX problems significantly impact user experience even when core functionality works
Priority: Frontend-Backend Consistency > UI/UX Excellence > Code Quality > Architecture Design
"""


def format_source_code(source_code_list: List[Dict]) -> str:
    """ """
    formatted_code = ""
    for file_info in source_code_list:
        file_path = file_info.get("path", "unknown")
        file_content = file_info.get("content", "")
        formatted_code += f"\n\n=== {file_path} ===\n"
        formatted_code += file_content
    return formatted_code.strip()


def convert_listwise_data_to_training_format(
    example: Dict,
    shuffle_order: bool = True,
    use_classification: bool = False,  # :
) -> Dict:
    """
    listwise JSON
       Args:
    example: prompt, responses, ranking
    shuffle_order: （）
       Returns:
    """
    prompt = example.get("prompt", "")
    responses = example.get("responses", [])
    ranking = example.get("ranking", [])
    # prompt
    full_prompt = build_fullstack_prompt(prompt)
    # ranking
    ranked_candidates = []
    ranking_scores = []  # listwise training
    for idx, response_id in enumerate(ranking):
        # response
        response_data = None
        for resp in responses:
            if resp.get("id") == response_id:
                response_data = resp
                break
        if response_data is None:
            continue
        if "source_code" in response_data:
            formatted_code = format_source_code(response_data["source_code"])
        else:
            formatted_code = response_data.get("text", "")
        candidate_data = [
            {"role": "user", "content": full_prompt},
            {"role": "assistant", "content": formatted_code},
        ]
        ranked_candidates.append(candidate_data)
        if use_classification:
            # idx=0 (best) -> class 0
            # idx=1 (middle) -> class 1
            # idx=2 (worst) -> class 2
            class_label = idx
            ranking_scores.append(int(class_label))
        else:
            # ：3， [3, 1, 0]
            score = 2 ** (len(ranking) - 1 - idx) - 1
            ranking_scores.append(float(score))
    if shuffle_order and len(ranked_candidates) > 0:
        indices = list(range(len(ranked_candidates)))
        random.shuffle(indices)
        # candidatesscores，
        ranked_candidates = [ranked_candidates[i] for i in indices]
        ranking_scores = [ranking_scores[i] for i in indices]
    return {
        "candidates": ranked_candidates,
        "scores": ranking_scores,
        "query_id": f"query_{hash(prompt) % 10000}",
        "num_candidates": len(ranked_candidates),
    }


class ListwiseDataset(Dataset):
    """
    Listwise
    queryranked candidates
    """

    def __init__(self, data_config: ListwiseDataConfig):
        self.data_config = data_config
        self.samples = []
        self._load_data()

    def _load_data(self):
        """"""
        data_dir = Path(self.data_config.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Reward data directory does not exist: {data_dir}")
        jsonl_file = data_dir / self.data_config.data_file
        if jsonl_file.exists():
            print(f" JSONL : {jsonl_file}")
            self._load_jsonl_data(jsonl_file)
            return
        if self.data_config.data_file != "dataset.jsonl":
            raise FileNotFoundError(
                f"Ranked examples file does not exist: {jsonl_file}"
            )
        default_jsonl_file = data_dir / "dataset.jsonl"
        if default_jsonl_file.exists():
            print(f" JSONL : {default_jsonl_file}")
            self._load_jsonl_data(default_jsonl_file)
            return
        # JSONL ，
        data_folders = [f for f in data_dir.iterdir() if f.is_dir()]
        data_folders.sort()  #
        print(f" {len(data_folders)} ")
        total_samples = 0
        for folder in data_folders:
            data_file = folder / "data.json"
            if not data_file.exists():
                continue
            try:
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # ( use_classification )
                converted_data = convert_listwise_data_to_training_format(
                    data,
                    shuffle_order=self.data_config.shuffle_candidates,
                    use_classification=self.data_config.use_classification,
                )
                if converted_data["num_candidates"] >= 2:
                    self.samples.append(converted_data)
                    total_samples += 1
                    if (
                        self.data_config.max_samples_per_folder > 0
                        and total_samples >= self.data_config.max_samples_per_folder
                    ):
                        break
            except Exception as e:
                print(f" {data_file}: {e}")
                continue
        print(f" {len(self.samples)} ")
        if self.data_config.shuffle_candidates:
            random.shuffle(self.samples)

    def _load_jsonl_data(self, jsonl_file: Path):
        """JSONL"""
        total_samples = 0
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        # JSON
                        data = json.loads(line)
                        # ( use_classification )
                        converted_data = convert_listwise_data_to_training_format(
                            data,
                            shuffle_order=self.data_config.shuffle_candidates,
                            use_classification=self.data_config.use_classification,
                        )
                        if converted_data["num_candidates"] >= 2:
                            self.samples.append(converted_data)
                            total_samples += 1
                            if (
                                self.data_config.max_samples_per_folder > 0
                                and total_samples
                                >= self.data_config.max_samples_per_folder
                            ):
                                break
                    except json.JSONDecodeError as e:
                        print(f" {line_num} JSON: {e}")
                        continue
                    except Exception as e:
                        print(f" {line_num} : {e}")
                        continue
        except Exception as e:
            print(f"JSONL {jsonl_file}: {e}")
            return
        print(f"Loaded {len(self.samples)} ranked examples from {jsonl_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        max_candidates = self.data_config.max_candidates_per_query
        if max_candidates > 0 and sample["num_candidates"] > max_candidates:
            candidates = sample["candidates"]
            scores = sample["scores"]
            indices = [0, len(candidates) - 1]
            middle_indices = list(range(1, len(candidates) - 1))
            random.shuffle(middle_indices)
            indices.extend(middle_indices[: max_candidates - 2])
            # : indices.sort(key=lambda i: scores[i], reverse=True)
            random.shuffle(indices)
            candidates = [candidates[i] for i in indices]
            scores = [scores[i] for i in indices]
            sample = {
                "candidates": candidates,
                "scores": scores,
                "query_id": sample["query_id"],
                "num_candidates": len(candidates),
            }
        return sample


class ListwiseDataCollator:
    """
    Listwise
    batchlistwise
    """

    def __init__(self, tokenizer, max_length: int = 7451):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict[str, Any]:
        """
        batch
               Args:
                   features: List of samples from ListwiseDataset
               Returns:
                   Batch data ready for model training
        """
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],  # ranking scores
            "num_candidates": [],
            "query_ids": [],
        }
        for feature in features:
            candidates = feature["candidates"]
            scores = feature["scores"]
            # input
            candidate_inputs = []
            for candidate in candidates:
                # tokenize
                text = self.tokenizer.apply_chat_template(
                    candidate, tokenize=False, add_generation_prompt=False
                )
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    return_tensors="pt",
                )
                candidate_inputs.append(
                    {
                        "input_ids": inputs["input_ids"].squeeze(0),
                        "attention_mask": inputs["attention_mask"].squeeze(0),
                    }
                )
            # batch
            batch["input_ids"].append([c["input_ids"] for c in candidate_inputs])
            batch["attention_mask"].append(
                [c["attention_mask"] for c in candidate_inputs]
            )
            batch["labels"].append(torch.tensor(scores, dtype=torch.float))
            batch["num_candidates"].append(len(candidates))
            batch["query_ids"].append(feature["query_id"])
        return self._pad_batch(batch)

    def _pad_batch(self, batch: Dict) -> Dict[str, Any]:
        """
        batchpadding，
        """
        max_candidates = max(batch["num_candidates"])
        max_seq_length = 0
        for sample_inputs in batch["input_ids"]:
            for candidate_input in sample_inputs:
                max_seq_length = max(max_seq_length, len(candidate_input))
        padded_batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "num_candidates": torch.tensor(batch["num_candidates"]),
            "query_ids": batch["query_ids"],
        }
        for i in range(len(batch["input_ids"])):
            sample_inputs = batch["input_ids"][i]
            sample_masks = batch["attention_mask"][i]
            sample_labels = batch["labels"][i]
            # padmax_candidates
            padded_inputs = []
            padded_masks = []
            padded_labels = []
            for j in range(max_candidates):
                if j < len(sample_inputs):
                    # pad
                    input_ids = sample_inputs[j]
                    attention_mask = sample_masks[j]
                    if len(input_ids) < max_seq_length:
                        pad_length = max_seq_length - len(input_ids)
                        input_ids = torch.cat(
                            [
                                input_ids,
                                torch.full((pad_length,), self.tokenizer.pad_token_id),
                            ]
                        )
                        attention_mask = torch.cat(
                            [attention_mask, torch.zeros(pad_length)]
                        )
                    padded_inputs.append(input_ids)
                    padded_masks.append(attention_mask)
                    padded_labels.append(sample_labels[j])
                else:
                    # dummy
                    padded_inputs.append(
                        torch.full((max_seq_length,), self.tokenizer.pad_token_id)
                    )
                    padded_masks.append(torch.zeros(max_seq_length))
                    padded_labels.append(torch.tensor(0.0))
            padded_batch["input_ids"].append(torch.stack(padded_inputs))
            padded_batch["attention_mask"].append(torch.stack(padded_masks))
            padded_batch["labels"].append(torch.stack(padded_labels))
        # tensor
        padded_batch["input_ids"] = torch.stack(padded_batch["input_ids"])
        padded_batch["attention_mask"] = torch.stack(padded_batch["attention_mask"])
        padded_batch["labels"] = torch.stack(padded_batch["labels"])
        # Shape: [batch_size, max_candidates, max_seq_length]
        return padded_batch


def create_listwise_dataset(data_config: ListwiseDataConfig) -> tuple:
    """
    Returns:
        (train_dataset, test_dataset)
    """
    full_dataset = ListwiseDataset(data_config)
    total_samples = len(full_dataset)
    eval_size = 100
    if total_samples <= 200:
        raise ValueError(
            f"Expected at least 201 ranked examples, found {total_samples}"
        )
    all_samples = full_dataset.samples.copy()
    random.shuffle(all_samples)
    test_samples = all_samples[:eval_size]
    train_samples = all_samples[eval_size:]
    train_dataset = ListwiseDataset.__new__(ListwiseDataset)
    train_dataset.data_config = data_config
    train_dataset.samples = train_samples
    test_dataset = ListwiseDataset.__new__(ListwiseDataset)
    test_dataset.data_config = data_config
    test_dataset.samples = test_samples
    return train_dataset, test_dataset
