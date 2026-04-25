from .variants.early_concat import EarlyConcatDTI
from .variants.early_crossattn import EarlyCrossAttnDTI
from .variants.late_concat import LateConcatDTI
from .variants.late_crossattn import LateCrossAttnDTI

VARIANT_REGISTRY = {
    "early_concat": EarlyConcatDTI,
    "early_crossattn": EarlyCrossAttnDTI,
    "late_concat": LateConcatDTI,
    "late_crossattn": LateCrossAttnDTI,
}


def build_model(variant: str, **kwargs):
    if variant not in VARIANT_REGISTRY:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(VARIANT_REGISTRY.keys())}")
    return VARIANT_REGISTRY[variant](**kwargs)
