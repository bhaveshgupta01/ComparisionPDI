"""
BindingDB Ki Dataset
====================
Loads dataset/BindingDB/BindingDB_PDSPKi.tsv, filters to valid rows,
and converts Ki (nM) → pKi = -log10(Ki * 1e-9).
"""
import math
import os
import re
import sys
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.tokenizers import SMILESTokenizer, ProteinTokenizer

# ────────────────────────────────────────────────────────────────────────────
# Column names in BindingDB_PDSPKi.tsv
# ────────────────────────────────────────────────────────────────────────────
COL_SMILES = "Ligand SMILES"
COL_PROT = "BindingDB Target Chain Sequence 1"
COL_KI = "Ki (nM)"

PKI_MIN = 3.0   # clip floor
PKI_MAX = 12.0  # clip ceiling

# SMILES must not contain SMARTS query wildcards like [#6] or [#7]
_SMARTS_PAT = re.compile(r"\[#\d+\]")


def _is_valid_smiles(s: str) -> bool:
    """Basic validity: non-empty string, no SMARTS wildcards."""
    if not isinstance(s, str) or not s.strip():
        return False
    # reject SMARTS query notation that appears in some BindingDB entries
    if _SMARTS_PAT.search(s):
        return False
    return True


def ki_to_pki(ki_nm: float) -> float:
    """Convert Ki in nM to pKi = -log10(Ki * 1e-9)."""
    ki_mol = ki_nm * 1e-9          # convert nM → M
    pki = -math.log10(ki_mol)      # pKi
    return float(max(PKI_MIN, min(PKI_MAX, pki)))


class BindingDBKiDataset(Dataset):
    """
    PyTorch Dataset for BindingDB PDSPKi Ki measurements.

    Parameters
    ----------
    tsv_path : str
        Path to BindingDB_PDSPKi.tsv.
    smiles_tokenizer : SMILESTokenizer
        Pre-built (or pre-loaded) SMILES tokenizer.
    protein_tokenizer : ProteinTokenizer
        Pre-built protein tokenizer.
    max_rows : int, optional
        Limit rows read from TSV (useful for smoke tests).
    """

    def __init__(
        self,
        tsv_path: str,
        smiles_tokenizer: SMILESTokenizer,
        protein_tokenizer: ProteinTokenizer,
        max_rows: Optional[int] = None,
    ):
        self.smiles_tok = smiles_tokenizer
        self.prot_tok = protein_tokenizer

        print(f"[BindingDBKiDataset] Loading {tsv_path} …")
        df = pd.read_csv(
            tsv_path,
            sep="\t",
            usecols=[COL_SMILES, COL_PROT, COL_KI],
            nrows=max_rows,
            low_memory=False,
        )

        n_raw = len(df)
        # Drop rows with missing essential fields
        df = df.dropna(subset=[COL_SMILES, COL_PROT, COL_KI])

        # Convert Ki: strip whitespace, coerce to float, drop non-positive
        df[COL_KI] = pd.to_numeric(df[COL_KI].astype(str).str.strip(), errors="coerce")
        df = df[df[COL_KI] > 0].copy()

        # Filter invalid SMILES
        df = df[df[COL_SMILES].apply(_is_valid_smiles)].copy()

        # Filter empty protein sequences
        df = df[df[COL_PROT].apply(lambda s: isinstance(s, str) and len(s.strip()) > 0)].copy()

        df = df.reset_index(drop=True)
        n_clean = len(df)
        print(
            f"[BindingDBKiDataset] {n_raw} raw rows → {n_clean} valid rows "
            f"({n_raw - n_clean} dropped)"
        )

        # Pre-compute pKi
        df["pKi"] = df[COL_KI].apply(ki_to_pki)

        self.smiles_list: list[str] = df[COL_SMILES].tolist()
        self.prot_list: list[str] = df[COL_PROT].tolist()
        self.pki_list: list[float] = df["pKi"].tolist()

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int):
        drug_ids = torch.tensor(
            self.smiles_tok.encode(self.smiles_list[idx]), dtype=torch.long
        )
        prot_ids = torch.tensor(
            self.prot_tok.encode(self.prot_list[idx]), dtype=torch.long
        )
        affinity = torch.tensor(self.pki_list[idx], dtype=torch.float32)
        return drug_ids, prot_ids, affinity
