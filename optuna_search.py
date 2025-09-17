#!/usr/bin/env python3
"""
optuna_search.py
----------------

Perform Bayesian hyperparameter optimisation using Optuna.  The search
space is defined by entries in ``config['experiment']['param_grids']``
for the specified algorithm.  If a grid is not provided, default
spaces are used.  Objective functions evaluate cross‑validated AUROC
on the training set and Optuna minimises the negative of that score.

Example usage::

    python src/optuna_search.py --config configs/base.yaml --model balanced_rf --n-trials 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

try:
    import optuna  # type: ignore
except ImportError:
    raise ImportError("optuna must be installed to use this script")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from .train import build_model, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter optimisation with Optuna")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML configuration file")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to tune (e.g., xgboost, balanced_rf)")
    parser.add_argument("--n-trials", type=int, default=25, help="Number of Optuna trials to run")
    parser.add_argument("--study-name", type=str, default="optuna_study", help="Name of the Optuna study")
    return parser.parse_args()


def get_search_space(trial: optuna.Trial, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Sample a hyperparameter configuration from Optuna based on a parameter grid.

    Each key in ``param_grid`` is treated as categorical.  Optuna will
    suggest one of the provided values.
    """
    params: Dict[str, Any] = {}
    for name, values in param_grid.items():
        params[name] = trial.suggest_categorical(name, values)
    return params


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    # Load data
    data_path = Path(config["paths"]["processed"])
    df = pd.read_csv(data_path)
    y = df[config["task"]["target"]].values
    X = df.drop(columns=[config["task"]["target"]])
    # Determine param grid for the model
    grid = config.get("experiment", {}).get("param_grids", {}).get(args.model, {})
    # Fallback grids
    if not grid:
        if args.model == "balanced_rf":
            grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
            }
        elif args.model == "xgboost":
            grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 4, 5],
                "learning_rate": [0.01, 0.1, 0.2],
            }
        else:
            grid = {}
    # Define objective function
    def objective(trial: optuna.Trial) -> float:
        params = get_search_space(trial, grid)
        # Build model; do not use class weights during tuning
        model = build_model(args.model, params, use_class_weights=False, y=y)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.get("seed", 42))
        aucs: List[float] = []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X_val)
                auc = roc_auc_score(y_val, probas, multi_class="ovr", average="macro")
            else:
                # Use accuracy if probabilities are unavailable
                preds = model.predict(X_val)
                auc = (preds == y_val).mean()
            aucs.append(auc)
        return -float(np.mean(aucs))  # Minimise negative AUROC
    # Create or load study
    study = optuna.create_study(direction="minimize", study_name=args.study_name)
    study.optimize(objective, n_trials=args.n_trials)
    print("Best parameters:", study.best_params)
    print("Best objective:", study.best_value)


if __name__ == "__main__":  # pragma: no cover
    main()