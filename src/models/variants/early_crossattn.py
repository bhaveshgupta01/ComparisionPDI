"""
Variant 2 — Early Cross-Attention
===================================
Strategy: apply bidirectional cross-attention *immediately after embedding*
(before any self-attention encoding), then concatenate and pass through a
shared transformer encoder.

Architecture (§7.7):
    drug_emb              = DrugEmbedding(drug_tokens)
    prot_emb              = ProtEmbedding(prot_tokens)
    drug_attn, prot_attn  = CrossAttention(drug_emb, prot_emb)  # at embed level
    combined              = concat([CLS, drug_attn, SEP, prot_attn], dim=1)
    encoded               = SharedEncoder(combined)              # 4 layers
    pred                  = PredictionHead(encoded[:, 0])
"""
import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseDTIModel
from src.models.cross_attention import BidirectionalCrossAttention
from src.models.embeddings import TokenEmbedding
from src.models.encoders import TransformerEncoder
from src.models.prediction_head import PredictionHead


class EarlyCrossAttnDTI(BaseDTIModel):
    """Variant 2: Early Cross-Attention."""

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

        self.drug_embedding = TokenEmbedding(drug_vocab_size, d_model, max_drug_len, dropout)
        self.prot_embedding = TokenEmbedding(prot_vocab_size, d_model, max_prot_len, dropout)

        # Cross-attention at the embedding level (before shared encoder)
        self.cross_attn = BidirectionalCrossAttention(d_model, n_heads, dropout)

        # Shared encoder after cross-attention
        self.encoder = TransformerEncoder(n_layers, d_model, n_heads, d_ff, dropout)

        self.cls_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sep_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_embed, std=0.02)
        nn.init.normal_(self.sep_embed, std=0.02)

        self.head = PredictionHead(d_model, head_hidden, head_dropout)

    def forward(
        self,
        drug_tokens: Tensor,
        drug_mask: Tensor,
        prot_tokens: Tensor,
        prot_mask: Tensor,
    ) -> Tensor:
        B = drug_tokens.size(0)

        drug_emb = self.drug_embedding(drug_tokens)   # [B, L_d, D]
        prot_emb = self.prot_embedding(prot_tokens)   # [B, L_p, D]

        # Early interaction via bidirectional cross-attention
        drug_attn, prot_attn = self.cross_attn(
            drug_emb, prot_emb,
            drug_mask=drug_mask,
            prot_mask=prot_mask,
        )

        cls = self.cls_embed.expand(B, -1, -1)
        sep = self.sep_embed.expand(B, -1, -1)

        combined = torch.cat([cls, drug_attn, sep, prot_attn], dim=1)

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=drug_tokens.device)
        sep_mask = torch.zeros(B, 1, dtype=torch.bool, device=drug_tokens.device)
        combined_mask = torch.cat([cls_mask, drug_mask, sep_mask, prot_mask], dim=1)

        encoded = self.encoder(combined, key_padding_mask=combined_mask)
        cls_repr = encoded[:, 0]
        return self.head(cls_repr)
