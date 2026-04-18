"""
Data Splits
===========
Three splitting strategies as defined in Technical Specification §6.5.
All return (train_indices, val_indices, test_indices) as lists of ints.
"""
import random
from typing import List, Tuple

from torch.utils.data import Dataset


def random_split(
    dataset: Dataset,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    §6.5.1 — Random split: 80/10/10 by default.
    """
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def cold_drug_split(
    dataset,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    §6.5.2 — Cold-drug split.
    Unique drugs are partitioned into train/val/test pools;
    each data point inherits the split of its drug.
    """
    rng = random.Random(seed)
    smiles_list = dataset.smiles_list  # list[str], one per sample

    # Build unique drug → sample-indices mapping
    drug_to_indices: dict = {}
    for i, smi in enumerate(smiles_list):
        drug_to_indices.setdefault(smi, []).append(i)

    unique_drugs = list(drug_to_indices.keys())
    rng.shuffle(unique_drugs)

    n_drugs = len(unique_drugs)
    n_train_d = int(n_drugs * train_frac)
    n_val_d = int(n_drugs * val_frac)

    train_drugs = set(unique_drugs[:n_train_d])
    val_drugs = set(unique_drugs[n_train_d : n_train_d + n_val_d])
    test_drugs = set(unique_drugs[n_train_d + n_val_d :])

    train_idx, val_idx, test_idx = [], [], []
    for drug, idxs in drug_to_indices.items():
        if drug in train_drugs:
            train_idx.extend(idxs)
        elif drug in val_drugs:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    return train_idx, val_idx, test_idx


def cold_target_split(
    dataset,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    §6.5.3 — Cold-target split.
    Unique proteins are partitioned; each pair inherits the protein split.
    """
    rng = random.Random(seed)
    prot_list = dataset.prot_list  # list[str], one per sample

    prot_to_indices: dict = {}
    for i, seq in enumerate(prot_list):
        prot_to_indices.setdefault(seq, []).append(i)

    unique_prots = list(prot_to_indices.keys())
    rng.shuffle(unique_prots)

    n_prots = len(unique_prots)
    n_train_p = int(n_prots * train_frac)
    n_val_p = int(n_prots * val_frac)

    train_prots = set(unique_prots[:n_train_p])
    val_prots = set(unique_prots[n_train_p : n_train_p + n_val_p])

    train_idx, val_idx, test_idx = [], [], []
    for prot, idxs in prot_to_indices.items():
        if prot in train_prots:
            train_idx.extend(idxs)
        elif prot in val_prots:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    return train_idx, val_idx, test_idx
