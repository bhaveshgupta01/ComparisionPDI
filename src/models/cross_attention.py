"""
Bidirectional Cross-Attention
==============================
Drug attends to protein and protein attends to drug simultaneously (§7.4).
Cross-attention weights are saved for analysis (Part A.2).
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class BidirectionalCrossAttention(nn.Module):
    """
    §7.4 — Two simultaneous cross-attention operations:
      - drug queries protein keys/values
      - protein queries drug keys/values

    The residual-connected outputs replace the original representations.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.drug_attends_protein = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.protein_attends_drug = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln_drug = nn.LayerNorm(d_model, eps=1e-5)
        self.ln_prot = nn.LayerNorm(d_model, eps=1e-5)

        # Saved for analysis
        self.drug_attn_weights: Optional[Tensor] = None   # drug→prot
        self.prot_attn_weights: Optional[Tensor] = None   # prot→drug

    def forward(
        self,
        drug: Tensor,
        prot: Tensor,
        drug_mask: Optional[Tensor] = None,
        prot_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        drug      : [B, L_d, D]
        prot      : [B, L_p, D]
        drug_mask : [B, L_d] bool — True = padding (for protein→drug attn)
        prot_mask : [B, L_p] bool — True = padding (for drug→protein attn)

        Returns
        -------
        drug_out : [B, L_d, D]
        prot_out : [B, L_p, D]
        """
        drug_n = self.ln_drug(drug)
        prot_n = self.ln_prot(prot)

        # Drug queries protein
        drug_ctx, dap_w = self.drug_attends_protein(
            drug_n, prot_n, prot_n,
            key_padding_mask=prot_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        # Protein queries drug
        prot_ctx, pad_w = self.protein_attends_drug(
            prot_n, drug_n, drug_n,
            key_padding_mask=drug_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        self.drug_attn_weights = dap_w.detach()
        self.prot_attn_weights = pad_w.detach()

        return drug + drug_ctx, prot + prot_ctx
