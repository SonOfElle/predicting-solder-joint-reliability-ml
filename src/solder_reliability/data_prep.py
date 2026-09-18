"""Data loading and preparation for solder joint reliability modelling.

The raw dataset contains 450 samples. Each sample is stored as one plain
text file. The file is a 7 x 4 matrix:

    rows 0 to 5, all 4 columns  ->  24 feature candidates
    row 6, column 0             ->  measured lifetime in hours

This module loads those files into a feature matrix and a target vector,
optionally applies the layout convention from the original thesis, scales
the values, and returns train/test splits.

Layout conventions
------------------

The original thesis code reshaped the (6, 4, 450) feature array with NumPy's
default C-order:

    features.reshape(450, 24)

Because the sample axis is last in (6, 4, 450), this operation does not
produce one row per sample. It produces rows that mix feature values across
many different samples, paired with the target of a single sample. The model
therefore never sees the 24 features of one joint in one row.

This module reproduces that behaviour under layout="original" so that the
published numbers can be reproduced, and offers layout="fixed" for the
correct per-sample construction. Notebooks and the results table report both.

Scale conventions
-----------------

The original thesis applied MinMaxScaler(feature_range=(0.2, 0.8)) to the
combined feature and target array before splitting. This leaks test
information into the scaler fit. This module reproduces that under
scale_strategy="original" and offers the principled alternative
scale_strategy="principled", which fits the scaler on the training split
only. Default is "principled".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

Layout = Literal["original", "fixed"]
ScaleStrategy = Literal["original", "principled"]

N_FEATURES = 24
N_SAMPLES = 450


def _natural_key(path: Path) -> list:
    """Sort key that treats embedded numbers as integers.

    So Data_Matrix_2.txt sorts before Data_Matrix_10.txt.
    """
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p for p in parts]


def _list_sample_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*.txt"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No .txt sample files found in {data_dir}")
    return files


def _read_sample(path: Path) -> tuple[np.ndarray, float]:
    raw = np.loadtxt(path)
    if raw.shape != (7, 4):
        raise ValueError(f"{path.name}: expected a 7 x 4 matrix, got {raw.shape}")
    features = raw[:6, :].flatten()
    target = float(raw[6, 0])
    return features, target


def load_raw(
    data_dir: str | Path,
    layout: Layout = "fixed",
) -> tuple[np.ndarray, np.ndarray]:
    """Load the 450 sample files into X (450, 24) and y (450,).

    layout="fixed"     X[i] is the 24 features of sample i, y[i] is its target.
    layout="original"  reproduces the thesis's C-order reshape of the
                       (6, 4, 450) array, which mixes samples within rows.
                       y is unchanged.
    """
    data_dir = Path(data_dir)
    files = _list_sample_files(data_dir)
    n = len(files)

    features_stack = np.zeros((6, 4, n))
    y = np.zeros(n)

    for i, path in enumerate(files):
        features, target = _read_sample(path)
        features_stack[:, :, i] = features.reshape(6, 4)
        y[i] = target

    if layout == "original":
        X = np.reshape(features_stack, (n, N_FEATURES))
    elif layout == "fixed":
        X = features_stack.transpose(2, 0, 1).reshape(n, N_FEATURES)
    else:
        raise ValueError(f"Unknown layout: {layout!r}")

    return X, y


def scale_and_split(
    X: np.ndarray,
    y: np.ndarray,
    scale_strategy: ScaleStrategy = "principled",
    feature_range: tuple[float, float] = (0.2, 0.8),
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Scale X and y, then split into train and test.

    Returns a dict with keys:
        X_train, X_test, y_train, y_test  (scaled)
        scaler_X, scaler_y                (fitted MinMaxScaler instances)
    """
    if scale_strategy == "original":
        # Thesis behaviour: fit scalers on the full dataset, then split.
        scaler_X = MinMaxScaler(feature_range=feature_range).fit(X)
        scaler_y = MinMaxScaler(feature_range=feature_range).fit(y.reshape(-1, 1))
        X_s = scaler_X.transform(X)
        y_s = scaler_y.transform(y.reshape(-1, 1)).ravel()
        X_train, X_test, y_train, y_test = train_test_split(
            X_s, y_s, test_size=test_size, random_state=seed
        )
    elif scale_strategy == "principled":
        # Fit on train only, transform test with the same scaler.
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        scaler_X = MinMaxScaler(feature_range=feature_range).fit(X_train_raw)
        scaler_y = MinMaxScaler(feature_range=feature_range).fit(
            y_train_raw.reshape(-1, 1)
        )
        X_train = scaler_X.transform(X_train_raw)
        X_test = scaler_X.transform(X_test_raw)
        y_train = scaler_y.transform(y_train_raw.reshape(-1, 1)).ravel()
        y_test = scaler_y.transform(y_test_raw.reshape(-1, 1)).ravel()
    else:
        raise ValueError(f"Unknown scale_strategy: {scale_strategy!r}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
    }


def load_and_prepare(
    data_dir: str | Path,
    layout: Layout = "fixed",
    scale_strategy: ScaleStrategy = "principled",
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Convenience wrapper: load, scale, split in one call."""
    X, y = load_raw(data_dir, layout=layout)
    return scale_and_split(
        X,
        y,
        scale_strategy=scale_strategy,
        test_size=test_size,
        seed=seed,
    )