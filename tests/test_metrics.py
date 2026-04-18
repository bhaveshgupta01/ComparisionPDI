"""
Tests for evaluation metrics.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.metrics import (
    mse,
    rmse,
    concordance_index,
    fast_concordance_index,
    pearson_r,
    spearman_r,
    compute_all_metrics,
)


def test_mse_perfect():
    y = [1.0, 2.0, 3.0]
    assert mse(y, y) == pytest.approx(0.0)


def test_mse_basic():
    pred = [2.0, 4.0]
    true = [1.0, 3.0]
    assert mse(pred, true) == pytest.approx(1.0)


def test_rmse_basic():
    pred = [4.0]
    true = [1.0]
    assert rmse(pred, true) == pytest.approx(3.0)


def test_ci_perfect_ranking():
    # When predictions perfectly agree with true ordering
    true = [1.0, 2.0, 3.0, 4.0, 5.0]
    pred = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert concordance_index(pred, true) == pytest.approx(1.0)


def test_ci_random_ranking():
    # Hard to hit exactly 0.5, but reversed ranking should be 0.0
    true = [1.0, 2.0, 3.0, 4.0, 5.0]
    pred = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert concordance_index(pred, true) == pytest.approx(0.0)


def test_ci_matches_fast_ci():
    rng = np.random.default_rng(42)
    y_true = rng.uniform(3, 12, 100)
    y_pred = y_true + rng.normal(0, 0.5, 100)
    ci_exact = concordance_index(y_pred, y_true)
    ci_fast = fast_concordance_index(y_pred, y_true)
    assert abs(ci_exact - ci_fast) < 0.01, (
        f"Exact CI={ci_exact:.4f} vs Fast CI={ci_fast:.4f}"
    )


def test_pearson_perfect():
    y = [1.0, 2.0, 3.0, 4.0]
    assert pearson_r(y, y) == pytest.approx(1.0, abs=1e-5)


def test_pearson_anticorrelated():
    y = [1.0, 2.0, 3.0, 4.0]
    y_neg = [4.0, 3.0, 2.0, 1.0]
    assert pearson_r(y_neg, y) == pytest.approx(-1.0, abs=1e-5)


def test_spearman_perfect():
    y = [1.0, 3.0, 2.0, 4.0]
    assert spearman_r(y, y) == pytest.approx(1.0, abs=1e-5)


def test_compute_all_metrics_keys():
    y = [1.0, 2.0, 3.0, 4.0]
    metrics = compute_all_metrics(y, y)
    expected_keys = {"mse", "rmse", "ci", "pearson", "spearman"}
    assert expected_keys == set(metrics.keys())


def test_compute_all_metrics_perfect():
    y = [5.0, 6.0, 7.0, 8.0, 9.0]
    m = compute_all_metrics(y, y)
    assert m["mse"] == pytest.approx(0.0)
    assert m["ci"] == pytest.approx(1.0)
    assert m["pearson"] == pytest.approx(1.0, abs=1e-5)
    assert m["spearman"] == pytest.approx(1.0, abs=1e-5)
