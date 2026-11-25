"""
Listwise Ranking
listwise learningreward model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ListMLELoss(nn.Module):
    """
       ListMLE (List Maximum Likelihood Estimation) Loss
    listwise
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, scores: torch.Tensor, labels: torch.Tensor, num_candidates: torch.Tensor
    ) -> torch.Tensor:
        """
               Args:
        scores: [batch_size, max_candidates] -
        labels: [batch_size, max_candidates] - (=)
        num_candidates: [batch_size] - query
        """
        batch_size, max_candidates = scores.shape
        device = scores.device
        # temperature scaling
        scores = scores / self.temperature
        total_loss = 0.0
        valid_samples = 0
        for i in range(batch_size):
            n_cands = num_candidates[i].item()
            if n_cands <= 1:
                continue
            sample_scores = scores[i, :n_cands]  # [n_cands]
            sample_labels = labels[i, :n_cands]  # [n_cands]
            _, sorted_indices = torch.sort(sample_labels, descending=True)
            # ListMLE -
            sample_loss = 0.0
            remaining_indices = torch.arange(n_cands, device=sample_scores.device)
            for j in range(n_cands):
                # j
                target_idx = sorted_indices[j]
                remaining_scores = sample_scores[remaining_indices]
                log_prob = F.log_softmax(remaining_scores, dim=0)
                # target_idxremaining_indices
                target_pos = (remaining_indices == target_idx).nonzero(as_tuple=True)[
                    0
                ][0]
                sample_loss -= log_prob[target_pos]
                if j < n_cands - 1:
                    mask = torch.ones_like(remaining_indices, dtype=torch.bool)
                    mask[target_pos] = False
                    remaining_indices = remaining_indices[mask]
            total_loss += sample_loss
            valid_samples += 1
        if valid_samples == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total_loss / valid_samples


class BradleyTerryLoss(nn.Module):
    """
       Bradley-Terry Pairwise Loss
    LiRE BT ：
    - listwise （ query ）
    - list (i, j)
    - 2 logits [s_i, s_j] softmax， [1,0] / [0,1] / [0.5,0.5]
    labels （ [3,1,0] best > middle > worst）
    """

    def __init__(self, threshold: float = 0.0):
        """
               Args:
        threshold: |label_i - label_j| <= threshold “”， [0.5, 0.5]
        """
        super().__init__()
        self.threshold = threshold

    def forward(
        self, scores: torch.Tensor, labels: torch.Tensor, num_candidates: torch.Tensor
    ) -> torch.Tensor:
        """
               Args:
        scores: [batch_size, max_candidates] -
        labels: [batch_size, max_candidates] - “”（）
        num_candidates: [batch_size] - query
        """
        batch_size = scores.shape[0]
        device = scores.device
        total_loss = 0.0
        total_pairs = 0
        for b in range(batch_size):
            n_cands = int(num_candidates[b].item())
            if n_cands <= 1:
                continue
            s = scores[b, :n_cands]
            y = labels[b, :n_cands]
            for i in range(n_cands):
                for j in range(i + 1, n_cands):
                    diff = (y[i] - y[j]).item()
                    if abs(diff) <= self.threshold:
                        target = torch.tensor(
                            [0.5, 0.5], device=device, dtype=torch.float32
                        )
                    elif diff > 0:
                        # i  j
                        target = torch.tensor(
                            [1.0, 0.0], device=device, dtype=torch.float32
                        )
                    else:
                        # j  i
                        target = torch.tensor(
                            [0.0, 1.0], device=device, dtype=torch.float32
                        )
                    # 2  logits: [s_i, s_j]
                    pair_logits = torch.stack([s[i], s[j]], dim=0).unsqueeze(
                        0
                    )  # [1, 2]
                    logprobs = F.log_softmax(pair_logits, dim=1)  # [1, 2]
                    loss_ij = -(target * logprobs.squeeze(0)).sum()
                    total_loss += loss_ij
                    total_pairs += 1
        if total_pairs == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total_loss / total_pairs


class LinearBradleyTerryLoss(nn.Module):
    """
    Linear BT Loss（LiRE linear_BT ）
    BradleyTerryLoss ：
    - logits softmax
    - ，，
    """

    def __init__(self, threshold: float = 0.0, epsilon: float = 1e-5):
        super().__init__()
        self.threshold = threshold
        self.epsilon = epsilon

    def forward(
        self, scores: torch.Tensor, labels: torch.Tensor, num_candidates: torch.Tensor
    ) -> torch.Tensor:
        batch_size = scores.shape[0]
        device = scores.device
        total_loss = 0.0
        total_pairs = 0
        for b in range(batch_size):
            n_cands = int(num_candidates[b].item())
            if n_cands <= 1:
                continue
            s = scores[b, :n_cands]
            y = labels[b, :n_cands]
            for i in range(n_cands):
                for j in range(i + 1, n_cands):
                    diff = (y[i] - y[j]).item()
                    if abs(diff) <= self.threshold:
                        target = torch.tensor(
                            [0.5, 0.5], device=device, dtype=torch.float32
                        )
                    elif diff > 0:
                        target = torch.tensor(
                            [1.0, 0.0], device=device, dtype=torch.float32
                        )
                    else:
                        target = torch.tensor(
                            [0.0, 1.0], device=device, dtype=torch.float32
                        )
                    pair_scores = torch.stack([s[i], s[j]], dim=0).unsqueeze(
                        0
                    )  # [1, 2]
                    # （ LiRE  pred_hat += segment_size）
                    min_val = pair_scores.min(dim=1, keepdim=True).values
                    pair_scores_pos = pair_scores - min_val + self.epsilon  # > 0
                    prob = pair_scores_pos / pair_scores_pos.sum(
                        dim=1, keepdim=True
                    )  # [1, 2]
                    loss_ij = -(target * torch.log(prob.squeeze(0))).sum()
                    total_loss += loss_ij
                    total_pairs += 1
        if total_pairs == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total_loss / total_pairs


class ListNetLoss(nn.Module):
    """
       ListNet Loss
    listwise
    KL
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def _prob_distribution(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Plackett-Luce
        """
        n = len(scores)
        exp_scores = torch.exp(scores / self.temperature)
        probs = torch.zeros(n, n, device=scores.device, dtype=scores.dtype)
        remaining_mask = torch.ones(n, dtype=torch.bool, device=scores.device)
        for i in range(n):
            current_exp_scores = exp_scores.clone()
            current_exp_scores[~remaining_mask] = 0  # 0
            current_sum = current_exp_scores.sum()
            if current_sum > 0:
                prob = current_exp_scores / current_sum
            else:
                prob = torch.zeros_like(current_exp_scores)
            probs[i] = prob
            if i < n - 1:
                remaining_prob = prob[remaining_mask]
                if len(remaining_prob) > 0:
                    max_idx_in_remaining = torch.argmax(remaining_prob)
                    remaining_indices = torch.where(remaining_mask)[0]
                    selected_idx = remaining_indices[max_idx_in_remaining]
                    remaining_mask[selected_idx] = False
        return probs

    def forward(
        self, scores: torch.Tensor, labels: torch.Tensor, num_candidates: torch.Tensor
    ) -> torch.Tensor:
        """
               Args:
        scores: [batch_size, max_candidates] -
        labels: [batch_size, max_candidates] -
        num_candidates: [batch_size] - query
        """
        batch_size = scores.shape[0]
        total_loss = 0.0
        valid_samples = 0
        for i in range(batch_size):
            n_cands = num_candidates[i].item()
            if n_cands <= 1:
                continue
            pred_scores = scores[i, :n_cands]
            true_scores = labels[i, :n_cands]
            pred_probs = self._prob_distribution(pred_scores)
            true_probs = self._prob_distribution(true_scores)
            # KL（pred_probstrue_probs）
            kl_div = F.kl_div(
                torch.log(pred_probs.flatten() + 1e-8),
                true_probs.flatten(),
                reduction="sum",
            )
            total_loss += kl_div
            valid_samples += 1
        return total_loss / max(valid_samples, 1)


class RankNetLoss(nn.Module):
    """
    RankNet Loss (listwise)
    ，listwise
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, scores: torch.Tensor, labels: torch.Tensor, num_candidates: torch.Tensor
    ) -> torch.Tensor:
        """ """
        batch_size = scores.shape[0]
        total_loss = 0.0
        total_pairs = 0
        for i in range(batch_size):
            n_cands = num_candidates[i].item()
            if n_cands <= 1:
                continue
            pred_scores = scores[i, :n_cands]
            true_scores = labels[i, :n_cands]
            for j in range(n_cands):
                for k in range(j + 1, n_cands):
                    if true_scores[j] > true_scores[k]:
                        # jk
                        diff = pred_scores[j] - pred_scores[k]
                        loss = torch.clamp(self.margin - diff, min=0.0)
                    elif true_scores[k] > true_scores[j]:
                        # kj
                        diff = pred_scores[k] - pred_scores[j]
                        loss = torch.clamp(self.margin - diff, min=0.0)
                    else:
                        continue
                    total_loss += loss
                    total_pairs += 1
        return total_loss / max(total_pairs, 1)


class ApproxNDCGLoss(nn.Module):
    """
       Approximate NDCG Loss
    NDCG
    smooth approximation
    """

    def __init__(self, temperature: float = 1.0, k: Optional[int] = None):
        super().__init__()
        self.temperature = temperature
        self.k = k  # top-k，None

    def _dcg(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Discounted Cumulative Gain
        """
        _, indices = torch.sort(scores, descending=True)
        sorted_labels = labels[indices]
        # top-k
        if self.k is not None:
            sorted_labels = sorted_labels[: self.k]
        # DCG
        gains = 2**sorted_labels - 1
        discounts = torch.log2(
            torch.arange(2, len(gains) + 2, device=scores.device).float()
        )
        dcg = (gains / discounts).sum()
        return dcg

    def _ideal_dcg(self, labels: torch.Tensor) -> torch.Tensor:
        """
        DCG (IDCG)
        """
        sorted_labels, _ = torch.sort(labels, descending=True)
        # top-k
        if self.k is not None:
            sorted_labels = sorted_labels[: self.k]
        if len(sorted_labels) == 0:
            return torch.tensor(0.0, device=labels.device)
        # _dcg
        label_min, label_max = labels.min(), labels.max()
        if label_max <= label_min:
            return torch.tensor(0.0, device=labels.device)
        # [1, 4]
        normalized_labels = (sorted_labels - label_min) / (
            label_max - label_min
        ) * 3 + 1
        gains = normalized_labels
        discounts = torch.log2(
            torch.arange(2, len(gains) + 2, device=labels.device).float()
        )
        idcg = (gains / discounts).sum()
        return idcg

    def forward(
        self, scores: torch.Tensor, labels: torch.Tensor, num_candidates: torch.Tensor
    ) -> torch.Tensor:
        """
        NDCG（NDCG = NDCG）
        """
        batch_size = scores.shape[0]
        total_ndcg = 0.0
        valid_samples = 0
        for i in range(batch_size):
            n_cands = num_candidates[i].item()
            if n_cands <= 1:
                continue
            pred_scores = scores[i, :n_cands] / self.temperature
            true_labels = labels[i, :n_cands]
            # DCGIDCG
            dcg = self._dcg(pred_scores, true_labels)
            idcg = self._ideal_dcg(true_labels)
            # NDCG
            if idcg > 0:
                ndcg = dcg / idcg
                total_ndcg += ndcg
                valid_samples += 1
        avg_ndcg = total_ndcg / max(valid_samples, 1)
        return 1.0 - avg_ndcg  # NDCG


class CombinedListwiseLoss(nn.Module):
    """
    listwise
    """

    def __init__(
        self,
        use_listmle: bool = True,
        use_listnet: bool = True,
        use_ranknet: bool = True,
        use_ndcg: bool = True,
        weights: Optional[List[float]] = None,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.losses = []
        default_weights = []
        if use_listmle:
            self.losses.append(("listmle", ListMLELoss(temperature=temperature)))
            default_weights.append(0.5)  # ，
        if use_listnet:
            self.losses.append(("listnet", ListNetLoss(temperature=temperature)))
            default_weights.append(0.2)
        if use_ranknet:
            self.losses.append(("ranknet", RankNetLoss()))
            default_weights.append(0.15)
        if use_ndcg:
            self.losses.append(("ndcg", ApproxNDCGLoss(temperature=temperature)))
            default_weights.append(0.35)  # NDCG
        if weights is not None:
            if len(weights) != len(self.losses):
                print(f": ({len(weights)})({len(self.losses)})")
                print(f": {[name for name, _ in self.losses]}")
                weight_mapping = {
                    "listmle": weights[0] if len(weights) > 0 else 1.0,
                    "listnet": weights[1] if len(weights) > 1 else 0.0,
                    "ranknet": weights[2] if len(weights) > 2 else 1.0,
                    "ndcg": weights[3] if len(weights) > 3 else 1.0,
                }
                self.weights = [weight_mapping[name] for name, _ in self.losses]
                print(f": {self.weights}")
            else:
                self.weights = weights
        else:
            self.weights = default_weights
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
        else:
            raise ValueError("Listwise loss weights must have a positive sum")

    def forward(
        self, scores: torch.Tensor, labels: torch.Tensor, num_candidates: torch.Tensor
    ) -> torch.Tensor:
        """ """
        total_loss = 0.0
        for (name, loss_fn), weight in zip(self.losses, self.weights):
            try:
                loss_value = loss_fn(scores, labels, num_candidates)
                total_loss += weight * loss_value
                if hasattr(self, "_debug") and self._debug:
                    print(f"{name}_loss: {loss_value.item():.4f}")
            except Exception as e:
                print(f"{name}: {e}")
                continue
        return total_loss


class ClassificationLoss(nn.Module):
    """
    :
    (best/middle/worst)
    """

    def __init__(
        self, num_classes: int = 3, class_weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, num_candidates: torch.Tensor
    ) -> torch.Tensor:
        """
               Args:
        logits: [batch_size, max_candidates, num_classes] - logits
        labels: [batch_size, max_candidates] - (0=best, 1=middle, 2=worst)
        num_candidates: [batch_size] - query
        """
        batch_size, max_candidates, num_classes = logits.shape
        device = logits.device
        # class_weights
        if self.class_weights is not None:
            class_weights = self.class_weights.to(device)
        else:
            class_weights = None
        total_loss = 0.0
        valid_samples = 0
        for i in range(batch_size):
            n_cands = num_candidates[i].item()
            if n_cands <= 0:
                continue
            sample_logits = logits[i, :n_cands, :]  # [n_cands, num_classes]
            sample_labels = labels[i, :n_cands].long()  # [n_cands]
            loss = F.cross_entropy(
                sample_logits, sample_labels, weight=class_weights, reduction="mean"
            )
            total_loss += loss
            valid_samples += 1
        if valid_samples == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total_loss / valid_samples


def create_listwise_loss(loss_type: str = "combined", **kwargs):
    """
    listwise
       Args:
    loss_type: ['listmle', 'listnet', 'ranknet', 'ndcg', 'combined', 'classification', 'bt', 'linear_bt']
    **kwargs:
       Returns:
    """
    if loss_type == "listmle":
        return ListMLELoss(**kwargs)
    elif loss_type == "listnet":
        return ListNetLoss(**kwargs)
    elif loss_type == "ranknet":
        return RankNetLoss(**kwargs)
    elif loss_type == "ndcg":
        return ApproxNDCGLoss(**kwargs)
    elif loss_type == "combined":
        return CombinedListwiseLoss(**kwargs)
    elif loss_type == "classification":  #
        return ClassificationLoss(**kwargs)
    elif loss_type == "bt":
        # BT  threshold ，（ temperature）
        bt_kwargs = {k: v for k, v in kwargs.items() if k in ["threshold"]}
        return BradleyTerryLoss(**bt_kwargs)
    elif loss_type == "linear_bt":
        # Linear BT  threshold  epsilon ，（ temperature）
        linear_bt_kwargs = {
            k: v for k, v in kwargs.items() if k in ["threshold", "epsilon"]
        }
        return LinearBradleyTerryLoss(**linear_bt_kwargs)
    else:
        raise ValueError(f": {loss_type}")


if __name__ == "__main__":
    """
    """
    batch_size, max_candidates = 2, 3
    scores = torch.tensor(
        [
            [3.2, 1.8, 2.5],  # query 1
            [2.1, 3.0, 1.5],  # query 2
        ]
    )
    labels = torch.tensor(
        [
            [2.0, 0.0, 1.0],  # query 1: best=2, worst=0, middle=1
            [0.0, 2.0, 1.0],  # query 2: worst=0, best=2, middle=1
        ]
    )
    num_candidates = torch.tensor([3, 3])
    loss_functions = {
        "ListMLE": ListMLELoss(),
        "ListNet": ListNetLoss(),
        "RankNet": RankNetLoss(),
        "ApproxNDCG": ApproxNDCGLoss(),
        "Combined": CombinedListwiseLoss(),
    }
    print(":")
    print(":", scores)
    print(":", labels)
    print(":", num_candidates)
    print()
    for name, loss_fn in loss_functions.items():
        try:
            loss_value = loss_fn(scores, labels, num_candidates)
            print(f"{name}: {loss_value.item():.4f}")
        except Exception as e:
            print(f"{name}: - {e}")
