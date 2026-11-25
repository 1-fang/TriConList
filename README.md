# TriConList

**Learning Tri-tier Contract Consistency for Full-Stack Repository Generation**

EACL camera-ready code for TriConList. The code trains a reward model on ranked full-stack repository candidates and uses its scores as rewards for Group Relative Policy Optimization (GRPO).

## Installation

Use Python 3.10 or 3.11. Training the reference 7B models uses Linux and NVIDIA GPUs. Install the dependencies from the repository root.

```bash
git clone https://github.com/1-fang/TriConList.git
cd TriConList
python -m pip install -r REWARD_MODEL/requirements.txt
```

For GRPO, use a separate environment and install the [veRL v0.5 stack](https://verl.readthedocs.io/en/v0.5.x/start/install.html) with its CUDA dependencies. The launcher targets veRL 0.5.0 and vLLM 0.8.5.

```bash
python -m pip install -r requirements-grpo.txt
```

## 1. Train the reward model

The loader reads JSON Lines. Each record contains a `prompt`, a list of `responses` with unique IDs, and a `ranking` of those IDs from best to worst. Each response contains either a `text` string or a `source_code` list of objects with `path` and `content` fields. See [a synthetic example](examples/ranked_candidates.jsonl).

```bash
DATA_DIR=data/reward_model DATA_FILE=train.jsonl \
OUTPUT_DIR=outputs/reward-model/triconlist \
bash REWARD_MODEL/train_lire.sh
```

This launcher uses Qwen2.5-Coder-7B-Instruct and linear Bradley–Terry loss. Set `BASE_MODEL` to a local model directory or another compatible Hugging Face model ID, and `NPROC_PER_NODE` to the number of GPU workers. Paths in the examples resolve from the repository root. The launcher also accepts `bt` as its first argument.

The listwise trainer reserves 100 examples for validation and requires at least 201 valid ranked examples. [The combined-loss launcher](REWARD_MODEL/train_rmv6.sh), [its 7,000-token variant](REWARD_MODEL/train_rmv6_seq7000_realistic.sh), and [the classification baseline](REWARD_MODEL/train_classification.sh) expose their hyperparameters in the scripts. The classification baseline takes separate `TRAIN_DATASET` and `EVAL_DATASET` files.

To evaluate a scalar reward checkpoint, supply a held-out file in the ranked data format.

```bash
python -m REWARD_MODEL.listwise_reward_model.listwise_infer \
    --model_path outputs/reward-model/triconlist \
    --test_data data/reward_model/eval.jsonl \
    --max_length 4096 \
    --output_file outputs/reward-model/eval.json
```

Keep the full reward checkpoint directory together, including the tokenizer, adapter, and reward-head files. Pass `--base_model_path` when moving an adapter whose configuration points to a local copy of the backbone.

## 2. Optimize the policy with GRPO

Prepare train and validation Parquet files in the [veRL data format](https://verl.readthedocs.io/en/v0.5.x/preparation/prepare_data.html). The reward adapter also expects `extra_info.reward_prompt` to contain the text presented to the reward model. The policy consumes the Parquet row's `prompt` chat messages.

```bash
TRAIN_FILE=data/grpo/train.parquet \
VAL_FILE=data/grpo/eval.parquet \
REWARD_MODEL_PATH=outputs/reward-model/triconlist \
TRAINING_GPUS=4 \
bash train_grpo.sh
```

Expose four GPUs for the policy and one additional GPU for the default Ray reward actor. The launcher accepts veRL Hydra overrides as additional arguments. Set `REWARD_BACKEND=local REWARD_DEVICE=cpu` for in-process CPU scoring or `REWARD_BACKEND=heuristic` for the rule-based baseline. Set `REWARD_BASE_MODEL_PATH` to override the reward checkpoint's backbone. Logs and policy checkpoints are written to `outputs/grpo/` by default.

Training data, trained checkpoints, and paper result files are absent from this code release. The synthetic example illustrates the input format.

## Citation and license

Cite *TriConList: Learning Tri-tier Contract Consistency for Full-Stack Repository Generation* when using this code. The source is distributed under the [MIT License](LICENSE).

The GRPO launcher builds on [veRL](https://github.com/verl-project/verl), and the reward training examples use [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct).
