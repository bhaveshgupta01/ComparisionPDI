"""
Base DTI Model
==============
Abstract base class shared by all four variants.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class BaseDTIModel(nn.Module, ABC):
    """
    Every variant must implement `forward` with the same interface.
    """

    @abstractmethod
    def forward(
        self,
        drug_tokens: Tensor,       # [B, L_d]
        drug_mask: Tensor,         # [B, L_d] bool — True = padding
        prot_tokens: Tensor,       # [B, L_p]
        prot_mask: Tensor,         # [B, L_p] bool — True = padding
    ) -> Tensor:
        """Returns predicted affinity [B]."""
        ...

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> str:
        total = self.count_parameters()
        return f"{self.__class__.__name__}: {total:,} trainable parameters"
