"""Sanity checks on the evaluation and metrics helpers.

No real data, no GAN, no grid search. Whole suite runs in a couple of
seconds and validates the arithmetic and the contracts that the
notebooks rely on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from solder_reliability import evaluate, find_repo_root


# --- closed-form arithmetic ------------------------------------------------

def test_regression_metrics_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true.copy()
    m = evaluate.regression_metrics(y_true, y_pred)
    assert m["mse"] == pytest.approx(0.0, abs=1e-12)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert m["mae"] == pytest.approx(0.0, abs=1e-12)
    assert m["r2"] == pytest.approx(1.0, abs=1e-12)


def test_regression_metrics_constant_prediction():
    # y_true mean is 2.0, SS_tot = 2.0, SS_res = 5.0, R2 = 1 - 2.5 = -1.5
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    m = evaluate.regression_metrics(y_true, y_pred)
    assert m["mse"] == pytest.approx(5.0 / 3.0, rel=1e-12)
    assert m["rmse"] == pytest.approx(np.sqrt(5.0 / 3.0), rel=1e-12)
    assert m["mae"] == pytest.approx(1.0, rel=1e-12)
    assert m["r2"] == pytest.approx(-1.5, rel=1e-12)


# --- denormalisation round trip --------------------------------------------

def test_denormalize_round_trip():
    y = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    scaler = MinMaxScaler(feature_range=(0.2, 0.8)).fit(y.reshape(-1, 1))
    y_scaled = scaler.transform(y.reshape(-1, 1)).ravel()
    y_rec = evaluate.denormalize(y_scaled, scaler)
    assert np.allclose(np.asarray(y_rec).ravel(), y, rtol=1e-12, atol=1e-9)


def test_rmse_in_hours_matches_closed_form():
    # Scaler maps [1000, 5000] to [0.2, 0.8]. A scaled RMSE of 0.05
    # corresponds to 0.05 * (5000 - 1000) / (0.8 - 0.2) = 333.33 hours.
    y = np.array([1000.0, 5000.0])
    scaler = MinMaxScaler(feature_range=(0.2, 0.8)).fit(y.reshape(-1, 1))
    y_true_scaled = np.array([0.2, 0.8])
    y_pred_scaled = np.array([0.25, 0.85])
    result = evaluate.rmse_in_hours(y_true_scaled, y_pred_scaled, scaler)
    expected = 0.05 * (5000.0 - 1000.0) / 0.6
    assert result == pytest.approx(expected, rel=1e-9)


# --- cross validation ------------------------------------------------------

def test_cross_val_rmse_linear_fixture():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 4))
    coef = np.array([1.0, -2.0, 0.5, 0.0])
    y = X @ coef + 0.1 * rng.normal(size=90)

    rmse = evaluate.cross_val_rmse(LinearRegression(), X, y, cv=3)
    assert np.isfinite(rmse)
    assert rmse > 0
    # Linear model with small noise should beat the mean predictor by a lot.
    assert rmse < y.std()


# --- learning curve contract ----------------------------------------------

def test_learning_curve_data_contract():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 3))
    y = X @ np.array([1.0, 0.5, -1.0]) + 0.1 * rng.normal(size=80)
    X_tr, y_tr = X[:60], y[:60]
    X_te, y_te = X[60:], y[60:]

    out = evaluate.learning_curve_data(
        lambda: LinearRegression(), X_tr, y_tr, X_te, y_te
    )
    assert isinstance(out, dict)
    assert set(out.keys()) == {"train_sizes", "train_rmse", "test_rmse"}

    sizes = np.asarray(out["train_sizes"])
    tr = np.asarray(out["train_rmse"])
    te = np.asarray(out["test_rmse"])
    assert sizes.shape == tr.shape == te.shape
    assert len(sizes) >= 2
    assert np.all(np.isfinite(tr))
    assert np.all(np.isfinite(te))
    # training sizes must be monotonically increasing
    assert np.all(np.diff(sizes) > 0)
    # a fresh factory per subset means the same fixture twice gives
    # the same curve
    out2 = evaluate.learning_curve_data(
        lambda: LinearRegression(), X_tr, y_tr, X_te, y_te
    )
    assert np.allclose(out["train_rmse"], out2["train_rmse"])
    assert np.allclose(out["test_rmse"],  out2["test_rmse"])


# --- evaluate_model and summarize smoke tests ------------------------------

def test_evaluate_model_returns_metric_dict():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    y = X @ np.array([1.0, 0.5, -1.0]) + 0.1 * rng.normal(size=60)
    X_tr, y_tr = X[:48], y[:48]
    X_te, y_te = X[48:], y[48:]

    # evaluate_model scores an already-fitted estimator; the docstring
    # says "fit" but the implementation only predicts.
    lr = LinearRegression().fit(X_tr, y_tr)
    res = evaluate.evaluate_model(lr, X_tr, y_tr, X_te, y_te)
    assert isinstance(res, dict)
    # Expect at least one of the common metric keys.
    expected_any = {"r2", "rmse", "mse", "mae", "test_r2", "test_rmse"}
    assert expected_any & set(res.keys()), sorted(res.keys())


# --- find_repo_root ---------------------------------------------------------

def test_find_repo_root_from_nested_paths():
    root = find_repo_root(Path(__file__).parent)
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "solder_reliability").is_dir()
    assert root == find_repo_root(root / "src")
    assert root == find_repo_root(root / "src" / "solder_reliability")