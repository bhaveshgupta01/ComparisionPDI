"""
Prediction Head
===============
Two-layer MLP that maps pooled representation → scalar affinity (§7.5).
"""
import torch.nn as nn
from torch import Tensor


class PredictionHead(nn.Module):
    """
    §7.5 — Three-layer MLP:
        d_in → d_hidden → d_hidden/2 → 1
    """

    def __init__(self, d_in: int, d_hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, d_in] → [B]"""
        return self.net(x).squeeze(-1)
