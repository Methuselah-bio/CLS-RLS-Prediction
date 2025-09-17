"""Tests for high‑dimensionality and robustness of the pipeline.

These tests exercise the z‑score computation on wide tables and verify
that the training routine can handle more features than samples without
crashing.  They use small random datasets to remain fast.
"""

import numpy as np
import pandas as pd

from src.datasets import compute_zscores
from src.train import build_model


def test_compute_zscores_high_dimensionality() -> None:
    """Z‑score computation should return the same number of rows and columns."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.rand(50, 100), columns=[f"feat{i}" for i in range(100)])
    z_df = compute_zscores(df)
    # The number of z‑score columns should equal the number of numeric columns in the original
    z_cols = [c for c in z_df.columns if c.startswith("z_")]
    assert len(z_cols) == 100
    assert z_df.shape[0] == df.shape[0]


def test_train_with_high_dimensionality() -> None:
    """Training logistic regression on high‑dimensional data should work."""
    rng = np.random.RandomState(1)
    X = pd.DataFrame(rng.rand(20, 100), columns=[f"feat{i}" for i in range(100)])
    y = rng.randint(0, 2, size=20)
    model = build_model("logreg", {}, False, y)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 20