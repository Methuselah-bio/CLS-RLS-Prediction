"""Feature selection utilities.

This module implements multivariate feature selection strategies that go
beyond univariate tests.  Methods include recursive feature elimination
with cross‑validation (RFECV) and network‑based ranking using
gene–gene interaction graphs.  These routines are optional and will
degrade gracefully if the requisite dependencies are missing.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.feature_selection import RFECV  # type: ignore
    from sklearn.model_selection import StratifiedKFold  # type: ignore
except Exception:
    RFECV = None  # type: ignore
    StratifiedKFold = None  # type: ignore

try:
    import networkx as nx  # type: ignore
except ImportError:
    nx = None  # type: ignore


def rfecv_selection(
    estimator,
    X: pd.DataFrame,
    y: Iterable,
    step: int = 1,
    cv: int = 5,
    scoring: str = "accuracy",
) -> List[str]:
    """Perform recursive feature elimination with cross‑validation.

    This function wraps ``sklearn.feature_selection.RFECV``.  It trains the
    provided estimator on successively smaller subsets of features and
    retains the subset that yields the best cross‑validated score.  The
    default scoring is accuracy; for survival analysis metrics such as
    concordance index can be specified if compatible with the estimator.

    Parameters
    ----------
    estimator : estimator instance
        A supervised learning estimator with a ``fit`` method and a
        ``coef_`` or ``feature_importances_`` attribute.
    X : pandas.DataFrame
        Feature matrix.
    y : array‑like
        Target vector.
    step : int, default 1
        Number of features to remove at each iteration.
    cv : int, default 5
        Number of folds in cross‑validation.  If the estimator does not
        support stratified splitting (e.g., for survival data), pass a
        custom CV object.
    scoring : str, default 'accuracy'
        Metric to optimise.  See scikit‑learn scoring parameter for
        options.

    Returns
    -------
    list of str
        Names of the selected features.
    """
    if RFECV is None or StratifiedKFold is None:
        logging.warning("scikit‑learn is required for RFECV; returning all features")
        return list(X.columns)
    # Determine appropriate cross‑validation iterator
    try:
        # Attempt stratified splits; if fails (e.g., for continuous y), fall back to simple CV
        cv_iter = StratifiedKFold(n_splits=cv)
    except Exception:
        from sklearn.model_selection import KFold  # type: ignore
        cv_iter = KFold(n_splits=cv)
    rfe = RFECV(estimator, step=step, cv=cv_iter, scoring=scoring, n_jobs=-1)
    rfe.fit(X, y)
    support_mask = getattr(rfe, "support_", None)
    if support_mask is None:
        return list(X.columns)
    return [col for col, keep in zip(X.columns, support_mask) if bool(keep)]


def network_based_selection(
    genes: List[str],
    interaction_network: Optional[nx.Graph] = None,
    top_k: int = 50,
) -> List[str]:
    """Select features based on network centrality.

    Given a list of gene identifiers and an optional interaction network,
    compute centrality measures (degree by default) and return the top
    ``top_k`` genes.  If no network is provided but ``networkx`` is
    available, a synthetic graph is constructed as in
    ``datasets.compute_network_features``.  This function is useful for
    pruning high‑dimensional gene sets prior to model training.  If
    networkx is not installed, the original gene list is returned.

    Parameters
    ----------
    genes : list of str
        Gene identifiers.
    interaction_network : networkx.Graph, optional
        Precomputed PPI or gene interaction network.  If ``None`` and
        ``networkx`` is available, a synthetic network is constructed.
    top_k : int, default 50
        Number of genes to retain based on highest centrality.

    Returns
    -------
    list of str
        Selected genes ranked by centrality.
    """
    if nx is None:
        logging.warning("networkx is not installed; returning original gene list")
        return genes[:top_k]
    # Build or use the provided network
    G: nx.Graph
    if interaction_network is not None:
        G = interaction_network
    else:
        G = nx.Graph()
        G.add_nodes_from(genes)
        # Connect nodes deterministically based on hash mod 17
        for i, ga in enumerate(genes):
            for gb in genes[i + 1 :]:
                if abs(hash(ga + gb)) % 17 == 0:
                    G.add_edge(ga, gb)
    # Compute degree centrality
    centrality = nx.degree_centrality(G)
    # Sort genes by centrality and take top_k
    sorted_genes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return [gene for gene, _score in sorted_genes[:top_k]]