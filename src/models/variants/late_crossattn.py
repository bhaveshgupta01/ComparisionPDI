"""
Variant 4 — Late Cross-Attention
==================================
Strategy: encode drug and protein independently, then apply bidirectional
cross-attention *after* deep encoding, pool, and predict.

Architecture (§7.9):
    drug_emb              = DrugEmbedding(drug_tokens)
    prot_emb              = ProtEmbedding(prot_tokens)
    drug_enc              = DrugEncoder(drug_emb)    # 2 layers
    prot_enc              = ProtEncoder(prot_emb)    # 2 layers  (total = 4)
    drug_fused, prot_fused = CrossAttention(drug_enc, prot_enc)
    drug_pool             = mean_pool(drug_fused, drug_mask)
    prot_pool             = mean_pool(prot_fused, prot_mask)
    combined              = concat([drug_pool, prot_pool])
    pred                  = PredictionHead(combined)
"""
import torch
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
    """Variant 4: Late Cross-Attention."""

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
        n_per = max(1, n_layers // 2)

        self.drug_embedding = TokenEmbedding(drug_vocab_size, d_model, max_drug_len, dropout)
        self.prot_embedding = TokenEmbedding(prot_vocab_size, d_model, max_prot_len, dropout)

        self.drug_encoder = TransformerEncoder(n_per, d_model, n_heads, d_ff, dropout)
        self.prot_encoder = TransformerEncoder(n_per, d_model, n_heads, d_ff, dropout)

        # Cross-attention after deep encoding (late interaction)
        self.cross_attn = BidirectionalCrossAttention(d_model, n_heads, dropout)

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

        # Late interaction via bidirectional cross-attention
        drug_fused, prot_fused = self.cross_attn(
            drug_enc, prot_enc,
            drug_mask=drug_mask,
            prot_mask=prot_mask,
        )

        drug_pool = _mean_pool(drug_fused, drug_mask)
        prot_pool = _mean_pool(prot_fused, prot_mask)

        combined = torch.cat([drug_pool, prot_pool], dim=-1)
        return self.head(combined)
