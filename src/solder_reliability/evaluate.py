"""Evaluation helpers for solder joint reliability models.

All metric functions return values in the unit of the input arrays. If the
inputs are scaled targets (MinMax to [0.2, 0.8], as in the thesis), the
RMSE is in scaled units. To report RMSE in hours, denormalise predictions
and targets first with the fitted target scaler from data_prep.

The thesis reports scaled RMSE. This module keeps that convention for
reproduction, and adds a denormalisation helper for the interpretable
report.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MSE, RMSE, MAE, and R^2 for a single prediction vector."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_val_rmse(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 3,
) -> float:
    """Cross-validated RMSE using sklearn's cross_val_predict.

    Uses the same 3-fold split the thesis used.
    """
    pred = cross_val_predict(model, X, y, cv=cv)
    return float(np.sqrt(mean_squared_error(y, pred)))


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv: int = 3,
) -> dict[str, Any]:
    """Fit, predict, and score a model.

    Returns a dict with metrics and the train/test prediction vectors.
    The model is expected to be already fitted or fit-able. It is not
    refit here; call fit before passing if you need to control that.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_m = regression_metrics(y_train, train_pred)
    test_m = regression_metrics(y_test, test_pred)
    cv_rmse = cross_val_rmse(model, X_train, y_train, cv=cv)

    return {
        "train_rmse": train_m["rmse"],
        "test_rmse": test_m["rmse"],
        "cv_rmse": cv_rmse,
        "test_r2": test_m["r2"],
        "train_r2": train_m["r2"],
        "train_pred": train_pred,
        "test_pred": test_pred,
    }


def learning_curve_data(
    model_factory: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_sizes: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Manual learning curve.

    Matches the thesis's manual loop: for each training subset size, fit a
    fresh model and record train and test RMSE. Uses a callable factory so
    each subset starts from a clean model.

    model_factory is called with no arguments and must return an unfitted
    estimator with the same hyperparameters as the final tuned model.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_errors = []
    test_errors = []

    for size in train_sizes:
        subset = int(size * len(X_train))
        X_subset = X_train[:subset]
        y_subset = y_train[:subset]

        model = model_factory()
        model.fit(X_subset, y_subset)

        train_pred = model.predict(X_subset)
        test_pred = model.predict(X_test)

        train_errors.append(np.sqrt(mean_squared_error(y_subset, train_pred)))
        test_errors.append(np.sqrt(mean_squared_error(y_test, test_pred)))

    return {
        "train_sizes": np.asarray(train_sizes),
        "train_rmse": np.asarray(train_errors),
        "test_rmse": np.asarray(test_errors),
    }


def denormalize(
    y_scaled: np.ndarray,
    target_scaler: Any,
) -> np.ndarray:
    """Invert MinMax scaling on the target.

    target_scaler is a fitted sklearn MinMaxScaler from data_prep's bundle.
    """
    arr = np.asarray(y_scaled).reshape(-1, 1)
    return target_scaler.inverse_transform(arr).ravel()


def rmse_in_hours(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    target_scaler: Any,
) -> float:
    """RMSE on the original target scale (hours)."""
    y_true_h = denormalize(y_true_scaled, target_scaler)
    y_pred_h = denormalize(y_pred_scaled, target_scaler)
    return float(np.sqrt(mean_squared_error(y_true_h, y_pred_h)))

def summarize(
    results: dict[str, dict[str, Any]],
    y_test_scaled: np.ndarray | None = None,
    target_scaler: Any | None = None,
) -> list[dict[str, Any]]:
    """Turn a per-model results dict into rows for results/metrics.csv.

    If y_test_scaled and target_scaler are provided, an extra
    test_rmse_hours column is added.
    """
    rows = []
    for name, r in results.items():
        row = {
            "model": name,
            "train_rmse": r["train_rmse"],
            "test_rmse": r["test_rmse"],
            "cv_rmse": r["cv_rmse"],
            "train_r2": r["train_r2"],
            "test_r2": r["test_r2"],
        }
        if y_test_scaled is not None and target_scaler is not None:
            row["test_rmse_hours"] = rmse_in_hours(
                y_test_scaled, r["test_pred"], target_scaler
            )
        rows.append(row)
    return rows