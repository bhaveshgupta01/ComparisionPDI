"""
Embedding Module
================
Token embeddings + sinusoidal positional encoding (§7.2).
"""
import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017).
    Registers the encoding matrix as a buffer (not trainable).
    """

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, L, D]  →  returns positional encoding [1, L, D]"""
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    """
    Learnable token embedding + sinusoidal positional encoding + LayerNorm + Dropout.
    Implements §7.2.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        max_len: int = 1200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = SinusoidalPositionalEncoding(d_model, max_len)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Parameters
        ----------
        token_ids : LongTensor [B, L]

        Returns
        -------
        Tensor [B, L, d_model]
        """
        x = self.token_embed(token_ids)      # [B, L, D]
        x = x + self.pos_embed(x)            # add positional
        x = self.layer_norm(x)
        return self.dropout(x)
