"""Training orchestration for the four-stage narrative.

One entry point per track (real, synthetic). Each entry point runs all
four models through their applicable stages, does the GridSearchCV and
the cross-validated scoring, and returns a DataFrame in
``results/metrics.csv`` shape plus a dict of fitted estimators.

Stage definitions:

  1. reproduce:  layout=original, scale=original,   grid=original
  2. layout fix: layout=fixed,    scale=original,   grid=original
  3. scale fix:  layout=fixed,    scale=principled, grid=original
  4. regrid:     layout=fixed,    scale=principled, grid=extended

Stages 1 through 3 apply to all four models. Stage 4 applies to SVR and
GPR only, because their original grids were tuned against the scrambled
layout. Random forest and XGBoost grids are layout-agnostic, so their
stage 3 and a hypothetical stage 4 would be identical.

Scaling strategies:

  original:   one MinMaxScaler(feature_range=(0.2, 0.8)) fit on the
              full (450, 25) array, then train/test split. Matches the
              thesis notebook, and is the behaviour that leaks test
              statistics into the scaler.

  principled: split first, fit the feature scaler on X_train only and
              the target scaler on y_train only. No leakage.

Metrics are reported in both scaled and hour units. ``rmse_hours =
rmse_scaled * target_spread_hours`` where ``target_spread_hours`` is
the raw target range divided by the scaler's (0.8 - 0.2) span. R2 is
invariant to linear rescaling of the target and is reported once.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import MinMaxScaler

from solder_reliability import models

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
SCALER_RANGE = (0.2, 0.8)
_SCALER_SPAN = SCALER_RANGE[1] - SCALER_RANGE[0]


STAGES: dict[str, dict[str, str]] = {
    "1. reproduce":  {"layout": "original", "scale_strategy": "original",   "grid": "original"},
    "2. layout fix": {"layout": "fixed",    "scale_strategy": "original",   "grid": "original"},
    "3. scale fix":  {"layout": "fixed",    "scale_strategy": "principled", "grid": "original"},
    "4. regrid":     {"layout": "fixed",    "scale_strategy": "principled", "grid": "extended"},
}

MODELS_WITH_REGRID = frozenset({"svr", "gpr"})

STAGE_ORDER = (
    "1. reproduce",
    "2. layout fix",
    "3. scale fix",
    "4. regrid",
)


@dataclass
class StageResult:
    model: str
    stage: str
    layout: str
    scale_strategy: str
    grid: str
    best_params: dict[str, Any]
    train_rmse_scaled: float
    test_rmse_scaled: float
    cv_rmse_scaled: float
    r2: float
    target_spread_hours: float
    train_rmse_hours: float
    test_rmse_hours: float
    cv_rmse_hours: float


def clone_factory(estimator: Any):
    """Return a zero-arg callable that produces an unfitted clone.

    For use with ``evaluate.learning_curve_data``, which needs a fresh
    estimator per training-subset size.
    """
    return lambda: clone(estimator)


def _split_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED
    )
    return train_idx, test_idx


def _scale_original(
    X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Thesis behaviour: fit one scaler on the full (n, 25) array."""
    data = np.hstack([X, y.reshape(-1, 1)])
    scaler = MinMaxScaler(feature_range=SCALER_RANGE).fit(data)
    scaled = scaler.transform(data)
    X_s, y_s = scaled[:, :-1], scaled[:, -1]
    spread_hours = float((y.max() - y.min()) / _SCALER_SPAN)
    return (
        X_s[train_idx], X_s[test_idx],
        y_s[train_idx], y_s[test_idx],
        spread_hours,
    )


def _scale_principled(
    X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit feature scaler on X_train, target scaler on y_train. No leakage."""
    x_scaler = MinMaxScaler(feature_range=SCALER_RANGE).fit(X[train_idx])
    y_scaler = MinMaxScaler(feature_range=SCALER_RANGE).fit(
        y[train_idx].reshape(-1, 1)
    )
    X_tr = x_scaler.transform(X[train_idx])
    X_te = x_scaler.transform(X[test_idx])
    y_tr = y_scaler.transform(y[train_idx].reshape(-1, 1)).ravel()
    y_te = y_scaler.transform(y[test_idx].reshape(-1, 1)).ravel()
    y_train_raw = y[train_idx]
    spread_hours = float((y_train_raw.max() - y_train_raw.min()) / _SCALER_SPAN)
    return X_tr, X_te, y_tr, y_te, spread_hours


def run_stage(
    model_name: str,
    stage_name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_jobs: int = 1,
) -> tuple[StageResult, Any]:
    """Run one stage for one model on one raw dataset.

    Returns ``(result, fitted_estimator)``. The fitted estimator carries
    the best hyperparameters from GridSearchCV and is refit on the full
    training split.
    """
    if stage_name not in STAGES:
        raise KeyError(
            f"unknown stage {stage_name!r}; known: {list(STAGES)}"
        )
    cfg = STAGES[stage_name]
    grid = cfg["grid"]
    if grid == "extended" and model_name not in MODELS_WITH_REGRID:
        raise ValueError(
            f"{model_name} has no extended grid; stage '4. regrid' "
            f"is not defined for it."
        )

    train_idx, test_idx = _split_indices(len(X))

    if cfg["scale_strategy"] == "original":
        X_tr, X_te, y_tr, y_te, spread_hours = _scale_original(
            X, y, train_idx, test_idx
        )
    else:
        X_tr, X_te, y_tr, y_te, spread_hours = _scale_principled(
            X, y, train_idx, test_idx
        )

    _, estimator, param_grid = models.make(model_name, grid=grid)

    gs = GridSearchCV(
        estimator,
        param_grid,
        cv=CV_FOLDS,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
    )
    gs.fit(X_tr, y_tr)
    best = gs.best_estimator_

    train_pred = best.predict(X_tr)
    test_pred = best.predict(X_te)
    cv_pred = cross_val_predict(best, X_tr, y_tr, cv=CV_FOLDS)

    train_rmse_s = float(np.sqrt(np.mean((y_tr - train_pred) ** 2)))
    test_rmse_s = float(np.sqrt(np.mean((y_te - test_pred) ** 2)))
    cv_rmse_s = float(np.sqrt(np.mean((y_tr - cv_pred) ** 2)))
    r2 = float(
        1 - np.sum((y_te - test_pred) ** 2) / np.sum((y_te - y_te.mean()) ** 2)
    )

    result = StageResult(
        model=model_name,
        stage=stage_name,
        layout=cfg["layout"],
        scale_strategy=cfg["scale_strategy"],
        grid=grid,
        best_params=dict(gs.best_params_),
        train_rmse_scaled=train_rmse_s,
        test_rmse_scaled=test_rmse_s,
        cv_rmse_scaled=cv_rmse_s,
        r2=r2,
        target_spread_hours=spread_hours,
        train_rmse_hours=train_rmse_s * spread_hours,
        test_rmse_hours=test_rmse_s * spread_hours,
        cv_rmse_hours=cv_rmse_s * spread_hours,
    )
    return result, best


def _stages_for(model_name: str) -> list[str]:
    return [
        s for s in STAGE_ORDER
        if s in STAGES
        and not (
            STAGES[s]["grid"] == "extended"
            and model_name not in MODELS_WITH_REGRID
        )
    ]


def run_all(
    data_by_layout: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    model_names: list[str] | None = None,
    verbose: bool = True,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict[tuple[str, str], Any]]:
    """Run all models through their applicable stages.

    ``data_by_layout`` maps ``"fixed"`` and ``"original"`` to ``(X, y)``
    raw arrays. Both keys must be present because stage 1 uses the
    original layout.

    Returns ``(results_df, fitted_models)`` where ``fitted_models`` is
    keyed by ``(model, stage)``.
    """
    if "fixed" not in data_by_layout or "original" not in data_by_layout:
        raise ValueError(
            "data_by_layout must have keys 'fixed' and 'original'"
        )

    if model_names is None:
        model_names = sorted(models.REGISTRY)

    rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, str], Any] = {}

    for model_name in model_names:
        for stage in _stages_for(model_name):
            X, y = data_by_layout[STAGES[stage]["layout"]]
            if verbose:
                print(f"{model_name:14s} {stage} ...", end=" ", flush=True)
            res, est = run_stage(
                model_name, stage, X, y, n_jobs=n_jobs
            )
            rows.append(asdict(res))
            fitted[(model_name, stage)] = est
            if verbose:
                print(
                    f"r2={res.r2:+.4f}  "
                    f"test_rmse_hours={res.test_rmse_hours:,.0f}h"
                )

    df = pd.DataFrame(rows)
    df["stage_order"] = df["stage"].map(
        {name: i for i, name in enumerate(STAGE_ORDER)}
    )
    df = df.sort_values(
        ["model", "stage_order"]
    ).drop(columns="stage_order").reset_index(drop=True)
    return df, fitted