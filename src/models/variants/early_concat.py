"""
Variant 1 — Early Concatenation
=================================
Strategy: concatenate drug and protein embeddings at the *embedding* level
(before any deep encoding), prepend a [CLS] token, run through a single
shared transformer encoder, and read off the [CLS] representation.

Architecture (§7.6):
    drug_emb  = DrugEmbedding(drug_tokens)          # [B, L_d, D]
    prot_emb  = ProtEmbedding(prot_tokens)           # [B, L_p, D]
    combined  = concat([CLS, drug_emb, SEP, prot_emb], dim=1)
    encoded   = SharedEncoder(combined)              # 4 layers
    cls_repr  = encoded[:, 0]                        # [B, D]
    pred      = PredictionHead(cls_repr)
"""
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseDTIModel
from src.models.embeddings import TokenEmbedding
from src.models.encoders import TransformerEncoder
from src.models.prediction_head import PredictionHead


class EarlyConcatDTI(BaseDTIModel):
    """Variant 1: Early Concatenation."""

    def __init__(
        self,
        drug_vocab_size: int,
        prot_vocab_size: int,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 4,
        d_ff: int = 128,
        dropout: float = 0.1,
        max_drug_len: int = 64,
        max_prot_len: int = 512,
        head_hidden: int = 128,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model

        # Separate embeddings for drug and protein
        self.drug_embedding = TokenEmbedding(drug_vocab_size, d_model, max_drug_len, dropout)
        self.prot_embedding = TokenEmbedding(prot_vocab_size, d_model, max_prot_len, dropout)

        # Learnable [CLS] and [SEP] token embeddings injected into combined sequence
        self.cls_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sep_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_embed, std=0.02)
        nn.init.normal_(self.sep_embed, std=0.02)

        # Shared encoder — 4 layers total
        self.encoder = TransformerEncoder(n_layers, d_model, n_heads, d_ff, dropout)

        # Prediction head: d_model → scalar
        self.head = PredictionHead(d_model, head_hidden, head_dropout)

    def forward(
        self,
        drug_tokens: Tensor,   # [B, L_d]
        drug_mask: Tensor,     # [B, L_d] bool
        prot_tokens: Tensor,   # [B, L_p]
        prot_mask: Tensor,     # [B, L_p] bool
    ) -> Tensor:
        B = drug_tokens.size(0)

        drug_emb = self.drug_embedding(drug_tokens)   # [B, L_d, D]
        prot_emb = self.prot_embedding(prot_tokens)   # [B, L_p, D]

        cls = self.cls_embed.expand(B, -1, -1)        # [B, 1, D]
        sep = self.sep_embed.expand(B, -1, -1)        # [B, 1, D]

        # combined: [CLS | drug_emb | SEP | prot_emb]
        combined = torch.cat([cls, drug_emb, sep, prot_emb], dim=1)  # [B, 2+L_d+L_p, D]

        # Build combined padding mask
        # CLS and SEP are never padding (False)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=drug_tokens.device)
        sep_mask = torch.zeros(B, 1, dtype=torch.bool, device=drug_tokens.device)
        combined_mask = torch.cat([cls_mask, drug_mask, sep_mask, prot_mask], dim=1)  # [B, 2+L_d+L_p]

        encoded = self.encoder(combined, key_padding_mask=combined_mask)  # [B, L, D]

        cls_repr = encoded[:, 0]   # [B, D]
        return self.head(cls_repr)
