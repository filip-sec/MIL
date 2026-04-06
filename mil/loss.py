"""Loss functions for MIL training."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralLoss(nn.Module):
    """CORAL ordinal loss (Cao et al., 2020).

    Converts integer label y ∈ {0,..,K-1} to K-1 binary targets:
        target_k = 1 if y > k else 0, for k = 0,..,K-2
    Then binary cross-entropy on each threshold.
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, ordinal_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        K = self.num_classes
        levels = torch.arange(K - 1, device=labels.device).unsqueeze(0)
        targets = (labels.unsqueeze(1) > levels).float()
        return F.binary_cross_entropy_with_logits(ordinal_logits, targets)


class OrdinalWithCELoss(nn.Module):
    """Combined CORAL + auxiliary cross-entropy loss."""

    def __init__(self, num_classes: int = 6, ce_weight: float = 0.3):
        super().__init__()
        self.coral = CoralLoss(num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight

    def forward(self, model_output: dict, labels: torch.Tensor) -> torch.Tensor:
        loss_coral = self.coral(model_output["ordinal_logits"], labels)
        loss_ce = self.ce(model_output["ce_logits"], labels)
        return loss_coral + self.ce_weight * loss_ce
