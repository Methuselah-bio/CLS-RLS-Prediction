"""
train_nested_cv.py
------------------

This module provides a skeleton implementation of nested cross‑
validation for hyperparameter tuning and model evaluation.  Nested
cross‑validation performs an inner loop of model selection on each
training fold and an outer loop of performance estimation, reducing
optimism from tuning the model on the full data set.  Scaling to
large datasets is supported via parallel execution using `joblib` or
`dask`.  Users can modify this script to integrate with `optuna` or
other Bayesian optimisation libraries.

Example usage::

    python src/train_nested_cv.py --config configs/base.yaml

At present, this script is a template and prints the structure of
nested cross‑validation without executing a full training loop.  It
should be extended to call the existing `train` functions with
appropriate subsets of the data and to record metrics for each fold.

Note: To enable distributed computation across multiple cores or
machines, consider using `dask.distributed` or `ray`.  This skeleton
uses `joblib.Parallel` for parallel execution on a single node.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, ParameterGrid
from sklearn.base import clone
from joblib import Parallel, delayed

from .train import build_model, build_survival_model, compute_metrics, compute_survival_metrics
from .prepare_data import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested cross‑validation for Methuselah models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run (use -1 to utilise all cores)",
    )
    return parser.parse_args()


def nested_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    config: Dict[str, Any],
    seed: int,
    n_jobs: int = 1,
) -> List[Dict[str, float]]:
    """Perform nested cross‑validation and return per‑fold metrics.

    This function splits the data into outer folds and, for each outer
    fold, performs an inner cross‑validation to tune hyperparameters.
    It then trains the best model on the inner training data and
    evaluates it on the outer test fold.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : ndarray
        Target vector.
    config : dict
        Experiment configuration containing model and hyperparameter grid.
    seed : int
        Random seed.
    n_jobs : int, default 1
        Number of parallel processes to use.

    Returns
    -------
    list of dict
        A list of metric dictionaries for each outer fold.
    """
    outer_folds = config.get("experiment", {}).get("cv_folds", 5)
    inner_folds = max(2, outer_folds - 1)
    model_name = config.get("model", {}).get("name", "logreg")
    param_grid = config.get("experiment", {}).get("param_grids", {}).get(model_name, {})
    use_class_weights = bool(config.get("model", {}).get("use_class_weights", False))
    problem_type = config.get("task", {}).get("problem_type", "classification").lower()
    if problem_type == "survival":
        # Nested CV for survival analysis is currently not implemented
        raise NotImplementedError("Nested cross‑validation for survival models is not yet implemented")
    # Build parameter grid
    if param_grid:
        grid = list(ParameterGrid(param_grid))
    else:
        grid = [{}]
    outer_skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)

    def evaluate_fold(train_index: np.ndarray, test_index: np.ndarray) -> Dict[str, float]:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Inner CV to select hyperparameters
        inner_skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
        best_score = -np.inf
        best_params: Dict[str, Any] = {}
        for params in grid:
            scores = []
            for inner_train_idx, inner_val_idx in inner_skf.split(X_train, y_train):
                X_inner_train, X_inner_val = X_train.iloc[inner_train_idx], X_train.iloc[inner_val_idx]
                y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
                model = build_model(model_name, params, use_class_weights, y_inner_train, seed)
                model.fit(X_inner_train, y_inner_train)
                metrics = compute_metrics(model, X_inner_val, y_inner_val)
                # Use AUROC as optimisation criterion if available, else accuracy
                score = metrics.get("auroc", metrics.get("accuracy", 0.0))
                scores.append(score)
            mean_score = float(np.mean(scores)) if scores else -np.inf
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        # Train model with best parameters on full training fold
        final_model = build_model(model_name, best_params, use_class_weights, y_train, seed)
        final_model.fit(X_train, y_train)
        fold_metrics = compute_metrics(final_model, X_test, y_test)
        return fold_metrics

    # Run outer folds in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_fold)(train_idx, test_idx) for train_idx, test_idx in outer_skf.split(X, y)
    )
    return results


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_config(args.config)
    seed = config.get("seed", 42)
    data_path = Path(config["paths"]["processed"])
    if not data_path.exists():
        logging.error("Processed data not found at %s; run prepare_data first", data_path)
        return
    df = pd.read_csv(data_path)
    target_col = config["task"]["target"]
    if target_col not in df.columns:
        logging.error("Target column '%s' missing from processed data", target_col)
        return
    # Separate features and labels
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    # Run nested CV
    logging.info("Starting nested cross‑validation with %d folds", config.get("experiment", {}).get("cv_folds", 5))
    try:
        metrics_list = nested_cv(X, y, config, seed, n_jobs=args.n_jobs)
    except NotImplementedError as exc:
        logging.error(exc)
        return
    # Summarise results
    df_metrics = pd.DataFrame(metrics_list)
    summary = df_metrics.mean().to_dict()
    logging.info("Nested CV complete. Mean metrics: %s", summary)


if __name__ == "__main__":  # pragma: no cover
    main()