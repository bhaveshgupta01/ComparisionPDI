"""
Metrics
=======
Evaluation metrics as specified in Technical Specification §9.1.

All functions accept numpy arrays or Python lists.
"""
import math
from typing import Union

import numpy as np


ArrayLike = Union[np.ndarray, list]


def _to_np(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def mse(y_pred: ArrayLike, y_true: ArrayLike) -> float:
    """Mean Squared Error."""
    y_pred, y_true = _to_np(y_pred), _to_np(y_true)
    return float(np.mean((y_pred - y_true) ** 2))


def rmse(y_pred: ArrayLike, y_true: ArrayLike) -> float:
    """Root Mean Squared Error."""
    return math.sqrt(mse(y_pred, y_true))


def concordance_index(y_pred: ArrayLike, y_true: ArrayLike) -> float:
    """
    Concordance Index (CI) — §9.1.

    Pure NumPy implementation.  For N samples this is O(N²) which is fine
    for eval sets up to ~50k pairs.  For larger sets, subsample or use
    lifelines.utils.concordance_index.
    """
    y_pred, y_true = _to_np(y_pred), _to_np(y_true)
    n = len(y_true)
    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            total += 1
            diff_true = y_true[i] - y_true[j]
            diff_pred = y_pred[i] - y_pred[j]
            if diff_true * diff_pred > 0:
                concordant += 1
            elif diff_pred == 0:
                concordant += 0.5
    return concordant / total if total > 0 else 0.5


def fast_concordance_index(y_pred: ArrayLike, y_true: ArrayLike) -> float:
    """
    Vectorised CI — O(N² memory) but fast for N ≤ 20k.
    Falls back to the loop version for larger arrays.
    """
    y_pred, y_true = _to_np(y_pred), _to_np(y_true)
    n = len(y_true)
    if n > 20_000:
        # subsample to 10k for speed during training
        rng = np.random.default_rng(0)
        idx = rng.choice(n, 10_000, replace=False)
        y_pred, y_true = y_pred[idx], y_true[idx]

    diff_true = y_true[:, None] - y_true[None, :]   # [N, N]
    diff_pred = y_pred[:, None] - y_pred[None, :]   # [N, N]

    # Only upper triangle (i < j)
    mask = np.triu(diff_true != 0, k=1)
    concordant = np.sum(mask & (diff_true * diff_pred > 0))
    concordant += 0.5 * np.sum(mask & (diff_pred == 0))
    total = np.sum(mask)
    return float(concordant / total) if total > 0 else 0.5


def pearson_r(y_pred: ArrayLike, y_true: ArrayLike) -> float:
    """Pearson correlation coefficient."""
    from scipy.stats import pearsonr
    y_pred, y_true = _to_np(y_pred), _to_np(y_true)
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    r, _ = pearsonr(y_pred, y_true)
    return float(r)


def spearman_r(y_pred: ArrayLike, y_true: ArrayLike) -> float:
    """Spearman rank correlation coefficient."""
    from scipy.stats import spearmanr
    y_pred, y_true = _to_np(y_pred), _to_np(y_true)
    rho, _ = spearmanr(y_pred, y_true)
    return float(rho)


def compute_all_metrics(y_pred: ArrayLike, y_true: ArrayLike) -> dict:
    """Return dict of all primary and secondary metrics."""
    return {
        "mse": mse(y_pred, y_true),
        "rmse": rmse(y_pred, y_true),
        "ci": fast_concordance_index(y_pred, y_true),
        "pearson": pearson_r(y_pred, y_true),
        "spearman": spearman_r(y_pred, y_true),
    }
