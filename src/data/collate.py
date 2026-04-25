"""
Collate Function
================
Pads drug and protein token sequences within a batch, and returns
attention masks (1 = real token, 0 = padding) for each modality.
"""
from typing import List, Tuple

import torch
from torch import Tensor

PAD_ID = 0  # matches tokenizer convention: <pad> → index 0


def collate_fn(
    batch: List[Tuple[Tensor, Tensor, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Collate a list of (drug_ids, prot_ids, affinity) tuples.

    Returns
    -------
    drug_tokens : LongTensor [B, L_d]
    drug_mask   : BoolTensor [B, L_d]   True where padding
    prot_tokens : LongTensor [B, L_p]
    prot_mask   : BoolTensor [B, L_p]   True where padding
    affinities  : FloatTensor [B]
    """
    drug_ids_list, prot_ids_list, affinities = zip(*batch)

    # All tensors in the batch already have fixed length from the tokenizer
    # (pre-padded to MAX_DRUG_LEN / MAX_PROT_LEN). We still stack cleanly.
    drug_tokens = torch.stack(list(drug_ids_list))   # [B, MAX_DRUG_LEN]
    prot_tokens = torch.stack(list(prot_ids_list))   # [B, MAX_PROT_LEN]
    affinities = torch.stack(list(affinities))        # [B]

    # Attention mask: True where token == PAD (PyTorch convention for
    # key_padding_mask in nn.MultiheadAttention)
    drug_mask = drug_tokens == PAD_ID   # [B, L_d]
    prot_mask = prot_tokens == PAD_ID   # [B, L_p]

    return drug_tokens, drug_mask, prot_tokens, prot_mask, affinities
