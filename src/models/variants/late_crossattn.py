"""
Variant 3 — Late Cross-Attention
==================================
Strategy: encode drug and protein independently, then apply bidirectional
cross-attention *after* deep encoding, pool, and predict.

Architecture (§7.9):
    drug_emb              = DrugEmbedding(drug_tokens)
    prot_emb              = ProtEmbedding(prot_tokens)
    drug_enc              = DrugEncoder(drug_emb)    # deep encoding
    prot_enc              = ProtEncoder(prot_emb)    # deep encoding
    drug_fused, prot_fused = CrossAttention(drug_enc, prot_enc)
    drug_pool             = mean_pool(drug_fused, drug_mask)
    prot_pool             = mean_pool(prot_fused, prot_mask)
    combined              = concat([drug_pool, prot_pool])
    pred                  = PredictionHead(combined)
"""
import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseDTIModel
from src.models.cross_attention import BidirectionalCrossAttention
from src.models.embeddings import TokenEmbedding
from src.models.encoders import TransformerEncoder
from src.models.prediction_head import PredictionHead


def _mean_pool(x: Tensor, mask: Tensor) -> Tensor:
    real = (~mask).float().unsqueeze(-1)
    summed = (x * real).sum(dim=1)
    counts = real.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class LateCrossAttnDTI(BaseDTIModel):
    """Variant 3: Late Cross-Attention."""

    def __init__(
        self,
        drug_vocab_size: int,
        prot_vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        n_layers_drug: int = None,
        n_layers_prot: int = None,
        xattn_layers: int = 1,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_drug_len: int = 100,
        max_prot_len: int = 1200,
        head_hidden: int = 256,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        # Default to even split if not specified
        n_d = n_layers_drug if n_layers_drug is not None else n_layers // 2
        n_p = n_layers_prot if n_layers_prot is not None else n_layers // 2

        self.drug_embedding = TokenEmbedding(drug_vocab_size, d_model, max_drug_len, dropout)
        self.prot_embedding = TokenEmbedding(prot_vocab_size, d_model, max_prot_len, dropout)

        self.drug_encoder = TransformerEncoder(n_d, d_model, n_heads, d_ff, dropout)
        self.prot_encoder = TransformerEncoder(n_p, d_model, n_heads, d_ff, dropout)

        # Cross-attention after deep encoding (late interaction)
        # Stack multiple layers if xattn_layers > 1
        self.cross_attn_layers = nn.ModuleList([
            BidirectionalCrossAttention(d_model, n_heads, dropout)
            for _ in range(xattn_layers)
        ])

        self.head = PredictionHead(2 * d_model, head_hidden, head_dropout)

    def forward(
        self,
        drug_tokens: Tensor,
        drug_mask: Tensor,
        prot_tokens: Tensor,
        prot_mask: Tensor,
    ) -> Tensor:
        drug_emb = self.drug_embedding(drug_tokens)
        prot_emb = self.prot_embedding(prot_tokens)

        drug_enc = self.drug_encoder(drug_emb, key_padding_mask=drug_mask)
        prot_enc = self.prot_encoder(prot_emb, key_padding_mask=prot_mask)

        # Late interaction via bidirectional cross-attention (stackable)
        drug_fused, prot_fused = drug_enc, prot_enc
        for xattn in self.cross_attn_layers:
            drug_fused, prot_fused = xattn(
                drug_fused, prot_fused,
                drug_mask=drug_mask,
                prot_mask=prot_mask,
            )

        drug_pool = _mean_pool(drug_fused, drug_mask)
        prot_pool = _mean_pool(prot_fused, prot_mask)

        combined = torch.cat([drug_pool, prot_pool], dim=-1)
        return self.head(combined)
