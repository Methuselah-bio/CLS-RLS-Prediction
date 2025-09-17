"""Fairness auditing for yeast lifespan predictions.

Although fairness is more commonly discussed in the context of human
data, biases can arise in yeast ageing models due to strain background
differences, media conditions or experimental batch effects.  This
module provides a simple wrapper around the ``aif360`` toolkit to
compute fairness metrics such as equalised odds or demographic parity
across defined sensitive groups (e.g., strain background or mating
type).  If ``aif360`` is not installed, the audit functions will raise
an informative error.

Usage example::

    from fairness_audit import audit_fairness
    metrics = audit_fairness(df, sensitive_attribute="strain", target_col="lifespan", predictions=pred)
    print(metrics)

Note that ``aif360`` expects binary labels; multi‑class problems need
to be binarised or analysed per class.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    from aif360.datasets import BinaryLabelDataset  # type: ignore
    from aif360.metrics import ClassificationMetric  # type: ignore
except ImportError:
    BinaryLabelDataset = None  # type: ignore
    ClassificationMetric = None  # type: ignore


def audit_fairness(
    df: pd.DataFrame,
    sensitive_attribute: str,
    target_col: str,
    predictions: Iterable,
    privileged_values: Optional[Iterable[Any]] = None,
    unprivileged_values: Optional[Iterable[Any]] = None,
) -> Dict[str, float]:
    """Compute fairness metrics using the AIF360 library.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the sensitive attribute and true labels.
    sensitive_attribute : str
        Column name of the sensitive attribute (e.g., strain background).
    target_col : str
        Column name of the true labels.  Must be binary for AIF360 metrics.
    predictions : iterable
        Predicted labels from the classifier.
    privileged_values : iterable, optional
        Values of the sensitive attribute considered privileged.  If
        ``None``, the function will take the most common value as
        privileged.
    unprivileged_values : iterable, optional
        Values considered unprivileged.  If ``None``, all other values
        are unprivileged.

    Returns
    -------
    dict
        Dictionary of fairness metrics including demographic parity
        difference and equalised odds difference.
    """
    if BinaryLabelDataset is None or ClassificationMetric is None:
        raise ImportError(
            "aif360 is required for fairness auditing; install via pip install aif360"
        )
    if sensitive_attribute not in df.columns or target_col not in df.columns:
        raise ValueError("Sensitive attribute or target column missing from DataFrame")
    y_true = df[target_col].values
    y_pred = np.asarray(list(predictions))
    # Default privileged/unprivileged assignment
    if privileged_values is None:
        # Use the most frequent value as privileged
        counts = df[sensitive_attribute].value_counts()
        privileged_values = [counts.idxmax()]
    if unprivileged_values is None:
        unprivileged_values = [v for v in df[sensitive_attribute].unique() if v not in privileged_values]
    dataset = BinaryLabelDataset(
        df=df[[sensitive_attribute]],
        label_names=[target_col],
        protected_attribute_names=[sensitive_attribute],
        favorable_label=1,
        unfavorable_label=0,
    )
    # Assign predictions
    pred_dataset = dataset.copy()
    pred_dataset.labels = y_pred.reshape(-1, 1)
    metric = ClassificationMetric(
        dataset,
        pred_dataset,
        unprivileged_groups=[{sensitive_attribute: val} for val in unprivileged_values],
        privileged_groups=[{sensitive_attribute: val} for val in privileged_values],
    )
    results: Dict[str, float] = {}
    # Demographic parity difference
    results["demographic_parity_difference"] = metric.statistical_parity_difference()
    # Equal opportunity difference (true positive rate difference)
    results["equal_opportunity_difference"] = metric.equal_opportunity_difference()
    # Average odds difference
    results["average_odds_difference"] = metric.average_odds_difference()
    return results