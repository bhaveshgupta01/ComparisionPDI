"""
Variant 3 — Late Concatenation
================================
Strategy: encode drug and protein *independently* through separate encoders,
then pool each modality and concatenate before the prediction head.

Architecture (§7.8):
    drug_emb  = DrugEmbedding(drug_tokens)
    prot_emb  = ProtEmbedding(prot_tokens)
    drug_enc  = DrugEncoder(drug_emb)    # 2 layers
    prot_enc  = ProtEncoder(prot_emb)    # 2 layers  (total = 4)
    drug_pool = mean_pool(drug_enc, drug_mask)   # [B, D]
    prot_pool = mean_pool(prot_enc, prot_mask)   # [B, D]
    combined  = concat([drug_pool, prot_pool])   # [B, 2D]
    pred      = PredictionHead(combined)
"""
import torch
from torch import Tensor

from src.models.base import BaseDTIModel
from src.models.embeddings import TokenEmbedding
from src.models.encoders import TransformerEncoder
from src.models.prediction_head import PredictionHead


def _mean_pool(x: Tensor, mask: Tensor) -> Tensor:
    """
    Mean-pool over non-padding positions.

    Parameters
    ----------
    x    : [B, L, D]
    mask : [B, L] bool — True = padding position

    Returns
    -------
    [B, D]
    """
    # Invert: 1 where real token, 0 where padding
    real = (~mask).float().unsqueeze(-1)       # [B, L, 1]
    summed = (x * real).sum(dim=1)             # [B, D]
    counts = real.sum(dim=1).clamp(min=1e-9)   # [B, 1]
    return summed / counts


class LateConcatDTI(BaseDTIModel):
    """Variant 3: Late Concatenation."""

    def __init__(
        self,
        drug_vocab_size: int,
        prot_vocab_size: int,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 4,      # split evenly between drug and protein encoders
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

        # Head takes 2*d (concatenated drug + prot pools)
        self.head = PredictionHead(2 * d_model, head_hidden, head_dropout)

    def forward(
        self,
        drug_tokens: Tensor,
        drug_mask: Tensor,
        prot_tokens: Tensor,
        prot_mask: Tensor,
    ) -> Tensor:
        drug_emb = self.drug_embedding(drug_tokens)        # [B, L_d, D]
        prot_emb = self.prot_embedding(prot_tokens)        # [B, L_p, D]

        drug_enc = self.drug_encoder(drug_emb, key_padding_mask=drug_mask)   # [B, L_d, D]
        prot_enc = self.prot_encoder(prot_emb, key_padding_mask=prot_mask)   # [B, L_p, D]

        drug_pool = _mean_pool(drug_enc, drug_mask)   # [B, D]
        prot_pool = _mean_pool(prot_enc, prot_mask)   # [B, D]

        combined = torch.cat([drug_pool, prot_pool], dim=-1)   # [B, 2D]
        return self.head(combined)
