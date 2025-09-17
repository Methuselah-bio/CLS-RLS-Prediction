#!/usr/bin/env python3
"""
train.py
--------

Train a classifier on the processed dataset.  This script generalises the
original implementation by supporting additional models and cost‑sensitive
learning.  Supported model names include:

* ``logreg`` – standard logistic regression (default).  When
  ``use_class_weights`` is true in the configuration the class
  distribution is used to compute ``class_weight``.
* ``rf`` – random forest classifier.
* ``xgboost`` – gradient boosted trees using XGBoost (if installed).
* ``balanced_rf`` – balanced random forest from ``imbalanced‑learn``
  to handle imbalanced classes.
* ``cost_sensitive_logreg`` – logistic regression with inverse class
  weighting regardless of ``use_class_weights``.
* Advanced models exposed via ``advanced_models.build_advanced_models``
  can be selected by their keys (e.g., ``cox``, ``transformer``, ``graphnn``).

The processed CSV is split into training/validation/test sets using
stratification on the target.  Metrics (AUROC, AUPRC, accuracy, Brier)
are computed on each split and saved into JSON.  The trained model is
persisted with joblib.

Usage:
    python src/train.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize

try:
    import xgboost as xgb  # type: ignore
except ImportError:  # pragma: no cover
    xgb = None  # type: ignore

try:
    from imblearn.ensemble import BalancedRandomForestClassifier  # type: ignore
except ImportError:  # pragma: no cover
    BalancedRandomForestClassifier = None  # type: ignore

try:
    # Advanced models may depend on optional packages
    from .advanced_models import build_advanced_models  # type: ignore
except Exception:
    build_advanced_models = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifier on processed data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(name: str, params: Dict[str, Any], use_class_weights: bool, y: np.ndarray, seed: int = 42):
    """Instantiate a classifier according to its name and configuration.

    Parameters
    ----------
    name : str
        Name of the classifier to build.
    params : dict
        Hyperparameters to pass to the underlying estimator.
    use_class_weights : bool
        Whether to compute class weights from ``y`` and supply them to
        classifiers that accept a ``class_weight`` argument.
    y : numpy.ndarray
        Target labels used for computing class weights when needed.
    seed : int, default 42
        Random seed for estimators.

    Returns
    -------
    sklearn estimator or object with fit/predict methods
        A pipeline that standardises features and trains the chosen model.
    """
    name_lower = (name or "").lower()
    class_weight = None
    if use_class_weights or name_lower == "cost_sensitive_logreg":
        # Compute inverse frequency weights
        unique, counts = np.unique(y, return_counts=True)
        total = counts.sum()
        weights = {cls: float(total) / (len(unique) * cnt) for cls, cnt in zip(unique, counts)}
        class_weight = weights
    # Define base estimator
    # Survival models are handled separately via build_survival_model; only
    # classification models are constructed here.  See ``build_survival_model``
    # below for Cox and random survival forests.
    if name_lower in {"rf", "random_forest"}:
        clf = RandomForestClassifier(random_state=seed, **(params or {}))
    elif name_lower in {"xgboost", "xgb"} and xgb is not None:
        clf = xgb.XGBClassifier(
            random_state=seed,
            use_label_encoder=False,
            eval_metric="logloss",
            **(params or {})
        )
    elif name_lower in {"balanced_rf", "balanced_random_forest"}:
        if BalancedRandomForestClassifier is None:
            raise ImportError("imbalanced-learn is required for BalancedRandomForestClassifier")
        clf = BalancedRandomForestClassifier(random_state=seed, **(params or {}))
    elif name_lower in {"cost_sensitive_logreg"}:
        clf = LogisticRegression(max_iter=500, multi_class="auto", class_weight=class_weight, **(params or {}))
    elif name_lower in {"logreg", "logistic_regression", ""}:
        clf = LogisticRegression(max_iter=500, multi_class="auto", class_weight=class_weight, **(params or {}))
    else:
        # Attempt to use an advanced model
        if build_advanced_models is not None:
            adv_models = build_advanced_models({"advanced_models": {"param_grids": {}}})
            if name_lower in adv_models:
                model_obj, _grid = adv_models[name_lower]
                # Advanced models may not require scaling
                return model_obj
        raise ValueError(f"Unknown model name '{name}'.")
    # Wrap in a pipeline with feature scaling
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])
    return pipeline


def build_survival_model(name: str, params: Dict[str, Any] | None = None, seed: int = 42):
    """Instantiate a survival analysis model.

    Supported names include:

    * ``cox`` – Cox proportional hazards model (lifelines).
    * ``random_survival_forest`` – Random survival forest (scikit‑survival).

    Parameters
    ----------
    name : str
        Identifier of the survival model.
    params : dict or None
        Hyperparameters passed through to the underlying estimator.
    seed : int, default 42
        Random seed used by ensemble models.

    Returns
    -------
    object
        A survival model object; callers are responsible for fitting and
        evaluating it appropriately.
    """
    name = (name or "").lower()
    params = params or {}
    if name in {"cox", "coxph", "cox_ph", "coxphfitter"}:
        try:
            from lifelines import CoxPHFitter  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "lifelines is required for Cox proportional hazards models; install via pip"
            ) from exc
        # CoxPHFitter accepts penalizer and strata arguments; others are ignored
        return CoxPHFitter(**params)
    if name in {"random_survival_forest", "rsf", "rsf_classifier"}:
        try:
            from sksurv.ensemble import RandomSurvivalForest  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "scikit-survival is required for RandomSurvivalForest; install via pip"
            ) from exc
        # Provide sensible defaults for number of trees
        n_estimators = params.get("n_estimators", 100)
        max_features = params.get("max_features", "sqrt")
        return RandomSurvivalForest(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=seed,
            **{k: v for k, v in params.items() if k not in {"n_estimators", "max_features"}},
        )
    raise ValueError(f"Unknown survival model '{name}'.")


def compute_survival_metrics(model: Any, X: pd.DataFrame, durations: np.ndarray, events: np.ndarray) -> Dict[str, float]:
    """Compute concordance and Brier score for survival models.

    This helper attempts to compute the concordance index (C‑index) and
    integrated Brier score.  If the required packages are unavailable,
    placeholders (NaN) are returned.

    Parameters
    ----------
    model : object
        Fitted survival model with methods appropriate for predicting risk or
        survival functions.
    X : pandas.DataFrame
        Feature matrix.
    durations : ndarray
        Array of observed survival times.
    events : ndarray
        Binary array indicating event occurrence (1 if event occurred, 0 if
        censored).

    Returns
    -------
    dict
        Dictionary containing concordance and integrated Brier score.
    """
    metrics: Dict[str, float] = {}
    # Try scikit‑survival metrics first
    try:
        from sksurv.metrics import concordance_index_censored, integrated_brier_score  # type: ignore
        from sksurv.metrics import cumulative_dynamic_auc  # type: ignore
        # Convert events/durations to structured array dtype required by sksurv
        y_struct = np.array([(bool(e), float(d)) for e, d in zip(events, durations)],
                            dtype=[('event', '?'), ('time', '<f8')])
        # Concordance index
        try:
            risk_scores = -model.predict(X)  # lower scores indicate higher survival
        except Exception:
            # Fallback to baseline hazard or partial hazard predictions
            if hasattr(model, "predict_partial_hazard"):
                risk_scores = model.predict_partial_hazard(X)
            else:
                risk_scores = np.zeros_like(durations)
        # Use censoring information to compute concordance
        c_index = concordance_index_censored(events.astype(bool), durations, risk_scores)[0]
        metrics["concordance_index"] = float(c_index)
        # Integrated Brier score at discrete times
        # Choose a grid of times (10 quantiles)
        times = np.percentile(durations, np.linspace(10, 90, 5))
        try:
            surv_fn = getattr(model, "predict_survival_function", None)
            if surv_fn is not None:
                # scikit‑survival's integrated_brier_score expects survival functions
                preds = np.asarray([fn(times) for fn in surv_fn(X)])
                ibs = integrated_brier_score(y_struct, y_struct, preds, times)
                metrics["integrated_brier"] = float(ibs)
        except Exception:
            metrics["integrated_brier"] = float('nan')
    except Exception:
        # Try lifelines C‑index
        try:
            from lifelines.utils import concordance_index  # type: ignore
            # For lifelines, higher partial hazard values correspond to higher risk
            risk_scores = -model.predict_partial_hazard(X)
            metrics["concordance_index"] = float(concordance_index(durations, risk_scores, events))
        except Exception:
            metrics["concordance_index"] = float('nan')
        metrics["integrated_brier"] = float('nan')
    return metrics


def compute_metrics(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Compute evaluation metrics for a classifier."""
    probs = None
    preds = None
    metrics: Dict[str, float] = {}
    # Some advanced models may lack predict_proba; handle gracefully
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        preds = model.predict(X)
    else:
        # Use decision function or raw predictions
        if hasattr(model, "predict"):
            preds = model.predict(X)
        if hasattr(model, "decision_function"):
            dfc = model.decision_function(X)
            # Convert decision function outputs into pseudo‑probabilities via sigmoid
            probs = 1 / (1 + np.exp(-dfc))
    # Binarise for multi‑class metrics
    classes = sorted(set(y))
    y_bin = label_binarize(y, classes=classes)
    # Compute metrics conditionally
    try:
        if probs is not None:
            metrics["auroc"] = roc_auc_score(y, probs, multi_class="ovr", average="macro")
        else:
            metrics["auroc"] = np.nan
    except Exception:
        metrics["auroc"] = np.nan
    try:
        if probs is not None:
            metrics["auprc"] = average_precision_score(y_bin, probs, average="macro")
        else:
            metrics["auprc"] = np.nan
    except Exception:
        metrics["auprc"] = np.nan
    try:
        if preds is not None:
            metrics["accuracy"] = accuracy_score(y, preds)
        else:
            metrics["accuracy"] = np.nan
    except Exception:
        metrics["accuracy"] = np.nan
    try:
        if probs is not None:
            metrics["brier"] = ((probs - y_bin) ** 2).sum(axis=1).mean()
        else:
            metrics["brier"] = np.nan
    except Exception:
        metrics["brier"] = np.nan
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = config.get("seed", 42)
    data_path = Path(config["paths"]["processed"])
    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    problem_type = config.get("task", {}).get("problem_type", "classification").lower()
    target_col = config["task"]["target"]
    duration_col = config["task"].get("target_time")
    event_col = config["task"].get("target_event")
    # Load processed data
    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in processed data.")
    # Separate features and labels; drop non‑numeric columns except the targets
    if problem_type == "survival":
        if duration_col is None or event_col is None:
            raise ValueError(
                "For survival analysis, task.target_time and task.target_event must be set"
            )
        if duration_col not in df.columns or event_col not in df.columns:
            raise KeyError(
                f"Duration column '{duration_col}' or event column '{event_col}' missing from data"
            )
        # Extract duration and event arrays
        durations = df[duration_col].astype(float).values
        events = df[event_col].astype(int).values
        X = df.drop(columns=[duration_col, event_col])
        # Train/validation/test split; do not stratify survival data
        test_size = config.get("split", {}).get("test_size", 0.2)
        val_size = config.get("split", {}).get("val_size", 0.2)
        X_train_val, X_test, dur_train_val, dur_test, ev_train_val, ev_test = train_test_split(
            X, durations, events, test_size=test_size, random_state=seed
        )
        if 0.0 < val_size < 1.0:
            val_frac = val_size / (1.0 - test_size)
            X_train, X_val, dur_train, dur_val, ev_train, ev_val = train_test_split(
                X_train_val,
                dur_train_val,
                ev_train_val,
                test_size=val_frac,
                random_state=seed,
            )
        else:
            X_train, X_val, dur_train, dur_val, ev_train, ev_val = (
                X_train_val,
                None,
                dur_train_val,
                None,
                ev_train_val,
                None,
            )
        # Build a survival model
        model_cfg = config.get("model", {})
        model_name = model_cfg.get("name", "cox")
        params = model_cfg.get("params", {}) or {}
        # Construct survival model
        survival_model = build_survival_model(model_name, params, seed=seed)
        # Fit model; interface differs between lifelines and scikit‑survival
        # For lifelines CoxPHFitter we pass a DataFrame with duration and event columns
        if model_name.lower() in {"cox", "coxph", "cox_ph", "coxphfitter"}:
            # Combine X and survival columns
            train_df = pd.concat([
                X_train.reset_index(drop=True),
                pd.Series(dur_train, name=duration_col),
                pd.Series(ev_train, name=event_col),
            ], axis=1)
            survival_model.fit(train_df, duration_col=duration_col, event_col=event_col)
        else:
            # scikit‑survival models expect structured arrays for y
            from sksurv.util import Surv  # type: ignore
            y_train_struct = Surv.from_arrays(ev_train.astype(bool), dur_train)
            survival_model.fit(X_train, y_train_struct)
        # Evaluate
        metrics: Dict[str, Any] = {}
        # Test metrics
        if model_name.lower() in {"cox", "coxph", "cox_ph", "coxphfitter"}:
            test_df = pd.concat([
                X_test.reset_index(drop=True),
                pd.Series(dur_test, name=duration_col),
                pd.Series(ev_test, name=event_col),
            ], axis=1)
            test_metrics = compute_survival_metrics(
                survival_model, X_test, dur_test, ev_test
            )
        else:
            test_metrics = compute_survival_metrics(
                survival_model, X_test, dur_test, ev_test
            )
        metrics["test"] = test_metrics
        # Validation metrics if split
        if X_val is not None:
            if model_name.lower() in {"cox", "coxph", "cox_ph", "coxphfitter"}:
                val_metrics = compute_survival_metrics(
                    survival_model, X_val, dur_val, ev_val
                )
            else:
                val_metrics = compute_survival_metrics(
                    survival_model, X_val, dur_val, ev_val
                )
            metrics["val"] = val_metrics
        # Save model using joblib (lifelines models can be pickled)
        model_path = results_dir / "model.joblib"
        joblib.dump(survival_model, model_path)
        metrics_path = results_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Survival training complete. Metrics saved to {metrics_path}")
        return

    # Classification pipeline (default)
    # Separate features and labels; drop non‑numeric columns except identifier
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    # Determine test/validation splits
    test_size = config.get("split", {}).get("test_size", 0.2)
    val_size = config.get("split", {}).get("val_size", 0.2)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    if 0.0 < val_size < 1.0:
        val_frac = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_frac,
            random_state=seed,
            stratify=y_train_val,
        )
    else:
        X_train, X_val, y_train, y_val = X_train_val, None, y_train_val, None
    # Build model
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "logreg")
    params = model_cfg.get("params", {}) or {}
    use_class_weights = bool(model_cfg.get("use_class_weights", False))
    model = build_model(model_name, params, use_class_weights, y_train, seed=seed)
    # Fit model
    model.fit(X_train, y_train)
    # Evaluate
    metrics: Dict[str, Any] = {}
    metrics["test"] = compute_metrics(model, X_test, y_test)
    if X_val is not None:
        metrics["val"] = compute_metrics(model, X_val, y_val)
    # Save metrics and model
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    model_path = results_dir / "model.joblib"
    joblib.dump(model, model_path)
    print(f"Training complete. Metrics saved to {metrics_path}")


if __name__ == "__main__":  # pragma: no cover
    main()