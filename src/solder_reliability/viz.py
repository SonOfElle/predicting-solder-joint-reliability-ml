"""Plotting helpers for solder joint reliability results.

Every public function returns a matplotlib Axes. If ``ax`` is None, one
is created and stored on a figure the caller can save. No function calls
``plt.show()``; that is the caller's job. Figures are written to
``reports/figures`` via ``save_figure`` with consistent dpi and naming.

Boundary: this module imports ``evaluate`` and matplotlib. ``evaluate``
imports neither this module nor matplotlib. Notebooks compose them.

Colors, stage labels and stage ordering are defined once here so all
figures across the write-up agree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solder_reliability import evaluate

REPORTS_FIGURES_DIR = Path("reports/figures")
DEFAULT_DPI = 200
DEFAULT_FIGSIZE = (7.0, 4.5)

MODEL_COLORS: dict[str, str] = {
    "random_forest": "#1f77b4",
    "xgboost": "#ff7f0e",
    "svr": "#2ca02c",
    "gpr": "#d62728",
}

STAGE_COLORS: dict[str, str] = {
    "1. reproduce": "#7f7f7f",
    "2. layout fix": "#1f77b4",
    "3. scale fix": "#2ca02c",
    "4. regrid": "#d62728",
}

STAGE_ORDER: tuple[str, ...] = (
    "1. reproduce",
    "2. layout fix",
    "3. scale fix",
    "4. regrid",
)

# Maps the joined (layout|scale_strategy|grid) key to a human stage label.
# Callers can override via the ``stage_labels`` argument.
_JOINED_TO_STAGE: dict[str, str] = {
    "original|original|original": "1. reproduce",
    "fixed|original|original": "2. layout fix",
    "fixed|principled|original": "3. scale fix",
    "fixed|principled|extended": "4. regrid",
}


def _new_ax(ax: plt.Axes | None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    return ax


def save_figure(
    fig: plt.Figure,
    name: str,
    *,
    out_dir: Path | str = REPORTS_FIGURES_DIR,
    dpi: int = DEFAULT_DPI,
    fmt: str = "png",
    tight: bool = True,
) -> Path:
    """Write ``fig`` to ``out_dir/name`` with consistent dpi and naming.

    Creates ``out_dir`` if missing. Appends the format suffix when not
    present. Returns the path written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{fmt}"
    if not name.endswith(suffix):
        name = f"{name}{suffix}"
    path = out_dir / name
    kwargs: dict[str, Any] = {"dpi": dpi}
    if tight:
        kwargs["bbox_inches"] = "tight"
    fig.savefig(path, **kwargs)
    return path


def plot_learning_curve(
    model_factory: Callable[[], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    train_sizes: Sequence[float] | None = None,
    title: str = "Learning curve",
    label: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Compute and draw a learning curve.

    Delegates the fit loop to ``evaluate.learning_curve_data``. The RMSE
    values returned are in the units of ``y``, so pass denormalised
    ``y_train`` and ``y_test`` when you want the axis in hours.
    """
    ax = _new_ax(ax)
    sizes_arg = None if train_sizes is None else np.asarray(train_sizes)
    out = evaluate.learning_curve_data(
        model_factory, X_train, y_train, X_test, y_test,
        train_sizes=sizes_arg,
    )
    sizes = out["train_sizes"]
    train_rmse = out["train_rmse"]
    test_rmse = out["test_rmse"]
    ax.plot(sizes, train_rmse, marker="o", color="#1f77b4", label="train")
    ax.plot(sizes, test_rmse, marker="s", color="#d62728", label="test")
    ax.set_xlabel("training set size")
    ax.set_ylabel("RMSE")
    ax.set_title(title if label is None else f"{title}: {label}")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(frameon=False)
    return ax


def plot_predictions(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    *,
    title: str = "Predictions",
    one_to_one: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter of predicted vs true values for train and test splits."""
    ax = _new_ax(ax)
    ax.scatter(
        y_true_train, y_pred_train, s=14, alpha=0.5,
        color="#1f77b4", label="train",
    )
    ax.scatter(
        y_true_test, y_pred_test, s=18, alpha=0.75,
        color="#d62728", label="test",
    )
    if one_to_one:
        lo = float(min(
            y_true_train.min(), y_true_test.min(),
            y_pred_train.min(), y_pred_test.min(),
        ))
        hi = float(max(
            y_true_train.max(), y_true_test.max(),
            y_pred_train.max(), y_pred_test.max(),
        ))
        ax.plot(
            [lo, hi], [lo, hi], linestyle="--", linewidth=1,
            color="#333333", label="y = x",
        )
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    ax.set_title(title)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(frameon=False)
    return ax


def plot_stage_bars(
    df: pd.DataFrame,
    metric: str = "r2",
    *,
    model_col: str = "model",
    stage_cols: Sequence[str] = ("layout", "scale_strategy", "grid"),
    stage_labels: Mapping[str, str] | None = None,
    stage_order: Sequence[str] = STAGE_ORDER,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar chart of one metric across the four-stage narrative.

    ``df`` is expected in ``results/metrics.csv`` shape: one row per
    (model, layout, scale_strategy, grid). SVR and GPR have a fourth
    stage, random forest and XGBoost have three.
    """
    ax = _new_ax(ax)
    df = df.copy()
    joined = df[list(stage_cols)].astype(str).agg("|".join, axis=1)
    labels = dict(_JOINED_TO_STAGE)
    if stage_labels:
        labels.update(stage_labels)
    df["_stage"] = joined.map(labels).fillna(joined)

    pivot = df.pivot_table(
        index=model_col, columns="_stage", values=metric, aggfunc="mean"
    )
    ordered = [s for s in stage_order if s in pivot.columns]
    leftover = [c for c in pivot.columns if c not in ordered]
    pivot = pivot[ordered + leftover]

    colors = [STAGE_COLORS.get(s, "#444444") for s in pivot.columns]
    pivot.plot(kind="bar", ax=ax, color=colors, width=0.78, edgecolor="white")

    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by stage")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(title="stage", frameon=False, loc="best")
    return ax


def plot_gan_losses(
    history: Mapping[str, Sequence[float]],
    *,
    title: str = "GAN training loss",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Generator and discriminator loss per epoch from ``gan.train_gan``."""
    ax = _new_ax(ax)
    gen = np.asarray(history["gen_loss"])
    disc = np.asarray(history["disc_loss"])
    epochs = np.arange(1, len(gen) + 1)
    ax.plot(epochs, gen, color="#1f77b4", label="generator")
    ax.plot(epochs, disc, color="#d62728", label="discriminator")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(frameon=False)
    return ax