"""
Transformer Encoder
===================
Pre-norm transformer encoder block and stacked encoder (§7.3).
Attention weights are saved as `attn_weights` for analysis.
"""
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class TransformerEncoderBlock(nn.Module):
    """
    Pre-norm single transformer encoder block.

    Saves the *last-batch* attention weight matrix on `self.attn_weights`
    for use in the analysis module (Part A).
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[Tensor] = None  # [B, L, L] (averaged heads)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        x                : [B, L, D]
        key_padding_mask : [B, L] bool — True = ignore (padding position)
        """
        x_norm = self.ln1(x)
        attn_out, attn_w = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        self.attn_weights = attn_w.detach()          # [B, L, L]
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of N TransformerEncoderBlock layers."""

    def __init__(
        self,
        n_layers: int = 4,
        d_model: int = 64,
        n_heads: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x
