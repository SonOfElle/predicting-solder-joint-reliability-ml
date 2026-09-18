"""Model factories for solder joint reliability regression.

Every factory returns ``(name, estimator, param_grid)``. The estimator is
unfitted. The grid is a dict suitable for ``GridSearchCV``.

Two grid families are exposed:

``"original"``
    Exact grids from the MSc thesis notebook. These were tuned while the
    feature matrix was in the buggy ``(450, 24)`` layout, so for SVR and
    GPR the searched ranges are calibrated against scrambled features.

``"extended"``
    Wider ranges sized for the fixed per-sample layout. Defined for SVR
    and GPR only. Random forest and XGBoost grids are layout-agnostic
    because tree splits do not depend on feature ordering or scaling.

Usage::

    from solder_reliability.models import make_svr
    name, est, grid = make_svr(grid="extended")
"""

from __future__ import annotations

from typing import Any

import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    WhiteKernel,
)
from sklearn.svm import SVR

SEED = 42


# --- Random forest ---------------------------------------------------------

_RF_GRID: dict[str, list[Any]] = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}


def make_random_forest(
    grid: str = "original",
) -> tuple[str, BaseEstimator, dict[str, list[Any]]]:
    if grid != "original":
        raise ValueError(
            f"random_forest only supports grid='original', got {grid!r}"
        )
    estimator = RandomForestRegressor(n_estimators=100, random_state=SEED)
    return "random_forest", estimator, dict(_RF_GRID)


# --- XGBoost ---------------------------------------------------------------

_XGB_GRID: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.2],
}


def make_xgboost(
    grid: str = "original",
) -> tuple[str, BaseEstimator, dict[str, list[Any]]]:
    if grid != "original":
        raise ValueError(f"xgboost only supports grid='original', got {grid!r}")
    estimator = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=SEED
    )
    return "xgboost", estimator, dict(_XGB_GRID)


# --- SVR -------------------------------------------------------------------

_SVR_GRID_ORIGINAL: dict[str, list[Any]] = {
    "C": [0.1, 1.0, 10.0],
    "gamma": [0.01, 0.1, 1.0],
    "kernel": ["rbf"],
}

# Fixed layout has 24 features on a common [0.2, 0.8] scale. For an RBF
# kernel gamma = 1 / (2 * length_scale**2), and the pairwise squared
# distances on 24 features occupy a wider range than on the buggy layout.
# Widen C and gamma by two decades on either side, and include gamma='scale'
# as the sklearn default baseline.
_SVR_GRID_EXTENDED: dict[str, list[Any]] = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "gamma": ["scale", 0.001, 0.01, 0.1, 1.0],
    "kernel": ["rbf"],
}


def make_svr(
    grid: str = "original",
) -> tuple[str, BaseEstimator, dict[str, list[Any]]]:
    if grid == "original":
        g = _SVR_GRID_ORIGINAL
    elif grid == "extended":
        g = _SVR_GRID_EXTENDED
    else:
        raise ValueError(f"grid must be 'original' or 'extended', got {grid!r}")
    return "svr", SVR(), dict(g)


# --- Gaussian process ------------------------------------------------------

_GPR_GRID_ORIGINAL: dict[str, list[Any]] = {
    "kernel": [RBF()],
    "alpha": [0.1, 0.01, 0.001],
}

# The original bare RBF() has length_scale=1.0 with bounds (1e-5, 1e5).
# On the fixed layout the marginal-likelihood optimiser drives
# length_scale to the upper bound and emits ConvergenceWarning, so the
# kernel is effectively not being fit. The extended grid gives the kernel
# an explicit amplitude, a wider length-scale window, an optional
# WhiteKernel for noise, and a Matern alternative.
_GPR_KERNELS_EXTENDED: list[Any] = [
    ConstantKernel(1.0, (1e-3, 1e3))
    * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)),
    ConstantKernel(1.0, (1e-3, 1e3))
    * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0)),
    ConstantKernel(1.0, (1e-3, 1e3))
    * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=1.5),
]

# With an explicit WhiteKernel in play, alpha should be small. Keeping
# three orders of magnitude lets the grid show which regularisation route
# the data prefers.
_GPR_GRID_EXTENDED: dict[str, list[Any]] = {
    "kernel": _GPR_KERNELS_EXTENDED,
    "alpha": [1e-6, 1e-3, 1e-1],
}


def make_gpr(
    grid: str = "original",
) -> tuple[str, BaseEstimator, dict[str, list[Any]]]:
    if grid == "original":
        g = _GPR_GRID_ORIGINAL
    elif grid == "extended":
        g = _GPR_GRID_EXTENDED
    else:
        raise ValueError(f"grid must be 'original' or 'extended', got {grid!r}")
    return "gpr", GaussianProcessRegressor(random_state=SEED), dict(g)


# --- registry --------------------------------------------------------------

REGISTRY = {
    "random_forest": make_random_forest,
    "xgboost": make_xgboost,
    "svr": make_svr,
    "gpr": make_gpr,
}


def make(
    name: str, grid: str = "original"
) -> tuple[str, BaseEstimator, dict[str, list[Any]]]:
    try:
        factory = REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown model {name!r}; known: {sorted(REGISTRY)}"
        ) from exc
    return factory(grid=grid)