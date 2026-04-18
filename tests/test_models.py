"""
Tests for all four DTI model variants.
Verifies:
  - Forward pass produces correct output shape [B]
  - Parameter counts are within expected range (< 500K for small model)
  - Gradients flow (loss.backward() works)
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.variants.early_concat import EarlyConcatDTI
from src.models.variants.early_crossattn import EarlyCrossAttnDTI
from src.models.variants.late_concat import LateConcatDTI
from src.models.variants.late_crossattn import LateCrossAttnDTI
from src.models import build_model

# ─── Shared test config (small) ───────────────────────────────────────────
DRUG_VOCAB = 72
PROT_VOCAB = 24
B = 4        # batch size
L_D = 64     # drug seq len
L_P = 128    # protein seq len

SMALL_KWARGS = dict(
    drug_vocab_size=DRUG_VOCAB,
    prot_vocab_size=PROT_VOCAB,
    d_model=32,
    n_heads=2,
    n_layers=4,
    d_ff=64,
    dropout=0.0,
    max_drug_len=L_D,
    max_prot_len=L_P,
    head_hidden=64,
    head_dropout=0.0,
)

VARIANTS = [
    ("early_concat", EarlyConcatDTI),
    ("early_crossattn", EarlyCrossAttnDTI),
    ("late_concat", LateConcatDTI),
    ("late_crossattn", LateCrossAttnDTI),
]


def make_batch(pad_frac: float = 0.3):
    """Make a fake batch of padded token IDs and masks."""
    # drug tokens: mostly 1..DRUG_VOCAB-1, some padding=0
    drug_tokens = torch.randint(1, DRUG_VOCAB, (B, L_D))
    prot_tokens = torch.randint(1, PROT_VOCAB, (B, L_P))

    # Simulate padding on the right
    pad_drug = int(L_D * pad_frac)
    pad_prot = int(L_P * pad_frac)
    drug_tokens[:, L_D - pad_drug :] = 0
    prot_tokens[:, L_P - pad_prot :] = 0

    drug_mask = drug_tokens == 0   # [B, L_D] True = padding
    prot_mask = prot_tokens == 0   # [B, L_P] True = padding
    return drug_tokens, drug_mask, prot_tokens, prot_mask


@pytest.mark.parametrize("variant_name,cls", VARIANTS)
def test_forward_shape(variant_name, cls):
    model = cls(**SMALL_KWARGS)
    model.eval()
    drug_tokens, drug_mask, prot_tokens, prot_mask = make_batch()

    with torch.no_grad():
        out = model(drug_tokens, drug_mask, prot_tokens, prot_mask)

    assert out.shape == (B,), (
        f"[{variant_name}] Expected output shape ({B},), got {out.shape}"
    )


@pytest.mark.parametrize("variant_name,cls", VARIANTS)
def test_parameter_count(variant_name, cls):
    model = cls(**SMALL_KWARGS)
    n_params = model.count_parameters()
    print(f"\n[{variant_name}] {n_params:,} parameters")
    # For this very small config, expect < 200K params
    assert n_params < 200_000, (
        f"[{variant_name}] Parameter count {n_params:,} exceeds 200K threshold"
    )


@pytest.mark.parametrize("variant_name,cls", VARIANTS)
def test_backward(variant_name, cls):
    model = cls(**SMALL_KWARGS)
    model.train()
    drug_tokens, drug_mask, prot_tokens, prot_mask = make_batch()
    targets = torch.randn(B)

    out = model(drug_tokens, drug_mask, prot_tokens, prot_mask)
    loss = torch.nn.functional.mse_loss(out, targets)
    loss.backward()

    # Check at least one parameter has a gradient
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, f"[{variant_name}] No gradients after backward()"


def test_build_model_registry():
    """Test that all variants are accessible via the registry."""
    for name, _ in VARIANTS:
        model = build_model(name, **SMALL_KWARGS)
        assert model is not None

    with pytest.raises(ValueError):
        build_model("nonexistent_variant", **SMALL_KWARGS)


@pytest.mark.parametrize("variant_name,cls", VARIANTS)
def test_no_nan_outputs(variant_name, cls):
    """Outputs must be finite, not NaN or Inf."""
    model = cls(**SMALL_KWARGS)
    model.eval()
    drug_tokens, drug_mask, prot_tokens, prot_mask = make_batch()

    with torch.no_grad():
        out = model(drug_tokens, drug_mask, prot_tokens, prot_mask)

    assert torch.isfinite(out).all(), (
        f"[{variant_name}] Output contains NaN or Inf: {out}"
    )
