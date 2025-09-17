"""Causal analysis utilities.

This module illustrates how to perform basic causal inference on gene
expression or perturbation datasets using the DoWhy library.  In the
context of chronological lifespan studies, one might treat nutrient
perturbations (e.g., caloric restriction or rapamycin treatment) as
interventions and estimate their causal effect on survival or gene
expression changes.  The functions in this module are thin wrappers
around DoWhy and are intended for exploratory analyses.

Usage example::

    from causal_analysis import estimate_effect
    df = prepare_some_dataframe()
    effect = estimate_effect(df, treatment_col="rapamycin", outcome_col="lifespan", confounders=["age", "batch"])
    print(effect.summary())

DoWhy must be installed separately.  If the import fails the functions
will raise an informative error.
"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

try:
    from dowhy import CausalModel  # type: ignore
except ImportError:
    CausalModel = None  # type: ignore


def estimate_effect(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounders: Optional[Iterable[str]] = None,
    method: str = "backdoor.linear_regression",
) -> object:
    """Estimate the causal effect of a treatment on an outcome.

    This function constructs a simple causal graph in which the
    treatment influences the outcome and optionally has confounders.
    It then calls DoWhy's ``estimate_effect`` to compute the average
    treatment effect (ATE) using the specified method.  See the
    DoWhy documentation for available estimation methods.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the treatment, outcome and confounder variables.
    treatment_col : str
        Name of the binary or continuous treatment/intervention column.
    outcome_col : str
        Name of the outcome variable (e.g., survival time).
    confounders : iterable of str, optional
        List of column names that confound the treatment–outcome relationship.
    method : str, default 'backdoor.linear_regression'
        Estimator name passed to DoWhy's ``estimate_effect``.

    Returns
    -------
    object
        A DoWhy estimate object with attributes such as ``value`` and
        ``summary()``.
    """
    if CausalModel is None:
        raise ImportError(
            "DoWhy is required for causal analysis. Install dowhy via pip install dowhy"
        )
    if treatment_col not in df.columns or outcome_col not in df.columns:
        raise ValueError("Treatment or outcome column not found in DataFrame")
    confounders = list(confounders or [])
    # Build a causal graph in the form of a string; include confounders
    # connecting to both treatment and outcome
    # Example graph: 'treatment -> outcome; confounder -> treatment; confounder -> outcome'
    graph_lines = [f"{treatment_col} -> {outcome_col}"]
    for c in confounders:
        graph_lines.append(f"{c} -> {treatment_col}")
        graph_lines.append(f"{c} -> {outcome_col}")
    graph = "; ".join(graph_lines)
    model = CausalModel(
        data=df,
        treatment=treatment_col,
        outcome=outcome_col,
        graph=graph,
    )
    identified = model.identify_effect()
    estimate = model.estimate_effect(identified, method_name=method)
    return estimate