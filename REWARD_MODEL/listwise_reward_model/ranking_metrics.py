"""
listwise reward model
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from scipy.stats import kendalltau, spearmanr


def dcg_at_k(
    scores: torch.Tensor, labels: torch.Tensor, k: Optional[int] = None
) -> float:
    """
    Discounted Cumulative Gain @ k
       Args:
    scores: [n_candidates]
    labels: [n_candidates]
    k: ，None
       Returns:
    DCG@k
    """
    _, indices = torch.sort(scores, descending=True)
    sorted_labels = labels[indices]
    # k
    if k is not None:
        sorted_labels = sorted_labels[:k]
    # DCG：gain
    # [0,1,2]，
    if len(sorted_labels) == 0:
        return 0.0
    label_min, label_max = labels.min().item(), labels.max().item()
    if label_max <= label_min:
        return 0.0  #
    # [1, 4]，gain > 0
    normalized_labels = (sorted_labels - label_min) / (label_max - label_min) * 3 + 1
    gains = normalized_labels  # gain
    discounts = torch.log2(
        torch.arange(2, len(gains) + 2, dtype=torch.float, device=gains.device)
    )
    dcg = (gains / discounts).sum().item()
    return dcg


def ndcg_at_k(
    scores: torch.Tensor, labels: torch.Tensor, k: Optional[int] = None
) -> float:
    """
    Normalized Discounted Cumulative Gain @ k
       Args:
    scores: [n_candidates]
    labels: [n_candidates]
    k:
       Returns:
    NDCG@k
    """
    if len(scores) == 0 or len(labels) == 0:
        return 0.0
    if labels.max() <= labels.min():
        return 0.0  # 0
    # DCG
    dcg = dcg_at_k(scores, labels, k)
    # DCG (IDCG) -
    sorted_labels, _ = torch.sort(labels, descending=True)
    idcg = dcg_at_k(sorted_labels, sorted_labels, k)  # ：=，
    # NDCG
    if idcg <= 1e-8:  # 0
        return 0.0
    ndcg_value = dcg / idcg
    # NDCG[0,1]
    return max(0.0, min(1.0, ndcg_value))


def map_at_k(
    scores: torch.Tensor, labels: torch.Tensor, k: Optional[int] = None
) -> float:
    """
    Mean Average Precision @ k
       Args:
    scores: [n_candidates]
    labels: [n_candidates] (1=, 0=)
    k:
       Returns:
    MAP@k
    """
    _, indices = torch.sort(scores, descending=True)
    sorted_labels = labels[indices]
    # k
    if k is not None:
        sorted_labels = sorted_labels[:k]
    relevant_count = 0
    precision_sum = 0.0
    for i, label in enumerate(sorted_labels):
        if label > 0:  #
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
    total_relevant = (labels > 0).sum().item()
    if total_relevant == 0:
        return 0.0
    return precision_sum / total_relevant


def mrr(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Mean Reciprocal Rank
       Args:
    scores: [n_candidates]
    labels: [n_candidates]
       Returns:
    MRR
    """
    _, indices = torch.sort(scores, descending=True)
    sorted_labels = labels[indices]
    for i, label in enumerate(sorted_labels):
        if label > 0:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """
    Precision @ k
       Args:
    scores: [n_candidates]
    labels: [n_candidates]
    k:
       Returns:
    P@k
    """
    _, indices = torch.sort(scores, descending=True)
    sorted_labels = labels[indices]
    # k
    top_k_labels = sorted_labels[:k]
    relevant_count = (top_k_labels > 0).sum().item()
    return relevant_count / k


def kendall_tau(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Kendall's Tau
       Args:
    scores: [n_candidates]
    labels: [n_candidates]
       Returns:
    Kendall's Tau (-11)
    """
    scores_np = scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    try:
        tau, _ = kendalltau(scores_np, labels_np)
        return tau if not np.isnan(tau) else 0.0
    except:
        return 0.0


def spearman_correlation(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Spearman
       Args:
    scores: [n_candidates]
    labels: [n_candidates]
       Returns:
    Spearman (-11)
    """
    scores_np = scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    try:
        rho, _ = spearmanr(scores_np, labels_np)
        return rho if not np.isnan(rho) else 0.0
    except:
        return 0.0


def ranking_accuracy_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """
    top-k
    top-ktop-k
       Args:
    scores: [n_candidates]
    labels: [n_candidates]
    k:
       Returns:
    """
    # top-k
    _, pred_top_k = torch.topk(scores, k)
    # top-k
    _, true_top_k = torch.topk(labels, k)
    pred_set = set(pred_top_k.tolist())
    true_set = set(true_top_k.tolist())
    overlap = len(pred_set & true_set)
    return overlap / k


def pairwise_accuracy(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    ，
       Args:
    scores: [n_candidates]
    labels: [n_candidates]
       Returns:
    """
    n = len(scores)
    if n <= 1:
        return 1.0
    correct_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            true_order = labels[i] > labels[j]
            pred_order = scores[i] > scores[j]
            if true_order == pred_order:
                correct_pairs += 1
            total_pairs += 1
    return correct_pairs / total_pairs if total_pairs > 0 else 1.0


class RankingMetrics:
    """ """

    def __init__(self, k_values: List[int] = [1, 2, 3]):
        self.k_values = k_values

    def compute_all_metrics(
        self,
        batch_scores: torch.Tensor,
        batch_labels: torch.Tensor,
        num_candidates: torch.Tensor,
    ) -> Dict[str, float]:
        """
        batch
               Args:
        batch_scores: [batch_size, max_candidates] -
        batch_labels: [batch_size, max_candidates] -
        num_candidates: [batch_size] - query
               Returns:
        """
        batch_size = batch_scores.shape[0]
        metrics = {
            "kendall_tau": 0.0,
            "spearman_rho": 0.0,
            "pairwise_accuracy": 0.0,
            "mrr": 0.0,
        }
        # @k
        for k in self.k_values:
            metrics[f"ndcg@{k}"] = 0.0
            metrics[f"map@{k}"] = 0.0
            metrics[f"precision@{k}"] = 0.0
            metrics[f"ranking_acc@{k}"] = 0.0
        valid_samples = 0
        # batch
        for i in range(batch_size):
            n_cands = num_candidates[i].item()
            if n_cands <= 1:
                continue
            scores = batch_scores[i, :n_cands]
            labels = batch_labels[i, :n_cands]
            metrics["kendall_tau"] += kendall_tau(scores, labels)
            metrics["spearman_rho"] += spearman_correlation(scores, labels)
            metrics["pairwise_accuracy"] += pairwise_accuracy(scores, labels)
            metrics["mrr"] += mrr(scores, labels)
            # @k
            for k in self.k_values:
                if k <= n_cands:
                    metrics[f"ndcg@{k}"] += ndcg_at_k(scores, labels, k)
                    # （MAPPrecision）
                    binary_labels = (labels >= labels.median()).float()
                    metrics[f"map@{k}"] += map_at_k(scores, binary_labels, k)
                    metrics[f"precision@{k}"] += precision_at_k(
                        scores, binary_labels, k
                    )
                    metrics[f"ranking_acc@{k}"] += ranking_accuracy_at_k(
                        scores, labels, k
                    )
            valid_samples += 1
        if valid_samples > 0:
            for key in metrics:
                metrics[key] /= valid_samples
        metrics["valid_samples"] = valid_samples
        metrics["total_samples"] = batch_size
        return metrics

    def compute_single_query_metrics(
        self, scores: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        query
               Args:
        scores: [n_candidates] -
        labels: [n_candidates] -
               Returns:
        """
        n_cands = len(scores)
        metrics = {}
        metrics["kendall_tau"] = kendall_tau(scores, labels)
        metrics["spearman_rho"] = spearman_correlation(scores, labels)
        metrics["pairwise_accuracy"] = pairwise_accuracy(scores, labels)
        metrics["mrr"] = mrr(scores, labels)
        # @k
        for k in self.k_values:
            if k <= n_cands:
                metrics[f"ndcg@{k}"] = ndcg_at_k(scores, labels, k)
                binary_labels = (labels >= labels.median()).float()
                metrics[f"map@{k}"] = map_at_k(scores, binary_labels, k)
                metrics[f"precision@{k}"] = precision_at_k(scores, binary_labels, k)
                metrics[f"ranking_acc@{k}"] = ranking_accuracy_at_k(scores, labels, k)
            else:
                metrics[f"ndcg@{k}"] = 0.0
                metrics[f"map@{k}"] = 0.0
                metrics[f"precision@{k}"] = 0.0
                metrics[f"ranking_acc@{k}"] = 0.0
        return metrics

    def format_metrics_string(self, metrics: Dict[str, float]) -> str:
        """ """
        lines = []
        lines.append("Overall")
        lines.append(f"Kendall's Tau: {metrics.get('kendall_tau', 0):.4f}")
        lines.append(f"Spearman Rho: {metrics.get('spearman_rho', 0):.4f}")
        lines.append(f"Pairwise Accuracy: {metrics.get('pairwise_accuracy', 0):.4f}")
        lines.append(f"MRR: {metrics.get('mrr', 0):.4f}")
        # @k
        for k in self.k_values:
            if f"ndcg@{k}" in metrics:
                lines.append(f"\n=== Top-{k} ===")
                lines.append(f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
                lines.append(f"MAP@{k}: {metrics[f'map@{k}']:.4f}")
                lines.append(f"Precision@{k}: {metrics[f'precision@{k}']:.4f}")
                lines.append(f"Ranking Acc@{k}: {metrics[f'ranking_acc@{k}']:.4f}")
        if "valid_samples" in metrics:
            lines.append("\nExamples")
            lines.append(f"Scored: {metrics['valid_samples']}")
            lines.append(f"Total: {metrics['total_samples']}")
        return "\n".join(lines)


if __name__ == "__main__":
    """
    """
    batch_size, max_candidates = 2, 3
    batch_scores = torch.tensor(
        [
            [3.2, 1.8, 2.5],  # query 1
            [2.1, 3.0, 1.5],  # query 2
        ]
    )
    batch_labels = torch.tensor(
        [
            [2.0, 0.0, 1.0],  # query 1: best, worst, middle
            [0.0, 2.0, 1.0],  # query 2: worst, best, middle
        ]
    )
    num_candidates = torch.tensor([3, 3])
    evaluator = RankingMetrics(k_values=[1, 2, 3])
    metrics = evaluator.compute_all_metrics(batch_scores, batch_labels, num_candidates)
    print(":")
    print(evaluator.format_metrics_string(metrics))
    # query
    print("\n\nQuery:")
    single_metrics = evaluator.compute_single_query_metrics(
        batch_scores[0, :4], batch_labels[0, :4]
    )
    print(evaluator.format_metrics_string(single_metrics))
