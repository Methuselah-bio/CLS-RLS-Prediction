#!/usr/bin/env python3
"""
interpret_shap.py
------------------

Compute SHAP (SHapley Additive exPlanations) values for a trained model
and visualise feature importances.  In addition to per‑feature importance,
this script supports grouping features into biological modules (e.g.,
KEGG pathways or gene modules).  A YAML file mapping group names to
feature lists can be supplied via ``--group-config``.  When provided,
the script aggregates absolute SHAP values across features within each
group and plots a bar chart of group importances.

Outputs:

* ``shap_summary.png`` – bar chart of mean |SHAP value| per feature or group.
* ``shap_values.npy`` – optional save of raw SHAP values for further analysis.

Usage::

    python src/interpret_shap.py --config configs/base.yaml [--max-samples 200] [--group-config path/to/groups.yaml]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap  # type: ignore
except ImportError:
    shap = None  # type: ignore

import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SHAP values for a trained model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum number of samples to use for SHAP estimation",
    )
    parser.add_argument(
        "--group-config",
        type=str,
        default=None,
        help="YAML file mapping group names to feature names for grouped SHAP aggregation",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Produce an interactive Plotly HTML chart instead of a static PNG",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_group_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    if shap is None:
        print(
            "The 'shap' package is not installed. Please install it with 'pip install shap' to use this script."
        )
        return
    config = load_config(args.config)
    processed_path = Path(config["paths"]["processed"])
    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    target_col = config["task"]["target"]
    seed = config.get("seed", 42)

    df = pd.read_csv(processed_path)
    if target_col not in df.columns:
        print(f"Target column '{target_col}' missing from processed data; cannot compute SHAP.")
        return
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])  # drop non‑numeric features
    y = df[target_col]
    # Load trained model from results directory
    model_path_candidates = [results_dir / "best_model.joblib", results_dir / "stacking_model.joblib", results_dir / "model.joblib"]
    model_path = None
    for cand in model_path_candidates:
        if cand.exists():
            model_path = cand
            break
    if model_path is None:
        print("No trained model found in results directory; please train a model first.")
        return
    model = joblib.load(model_path)
    # Sample to reduce computation
    if args.max_samples < len(X):
        rng = np.random.RandomState(seed)
        sample_indices = rng.choice(len(X), size=args.max_samples, replace=False)
        X_sample = X.iloc[sample_indices]
    else:
        X_sample = X
    # Choose appropriate explainer
    try:
        if hasattr(model, "predict_proba") and (
            "xgb" in model.__class__.__module__ or "sklearn.ensemble" in model.__class__.__module__ or "forest" in model.__class__.__name__.lower()
        ):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            background = shap.sample(X_sample, min(50, len(X_sample)), random_state=seed)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_sample)
    except Exception as exc:
        print(f"Failed to compute SHAP values: {exc}")
        return
    # Aggregate SHAP values across classes for multi‑class problems
    if isinstance(shap_values, list):
        abs_vals = np.mean([np.abs(vals) for vals in shap_values], axis=0)
    else:
        abs_vals = np.abs(shap_values)
    mean_abs = abs_vals.mean(axis=0)
    feature_names = np.array(X_sample.columns)

    if args.group_config:
        # Load group mapping and aggregate scores
        group_map = load_group_config(args.group_config)
        group_scores: Dict[str, float] = {}
        for group_name, feats in group_map.items():
            # Determine indices of features in this group
            indices = [i for i, f in enumerate(feature_names) if f in feats]
            if not indices:
                continue
            group_scores[group_name] = float(mean_abs[indices].mean())
        # Sort groups by importance
        sorted_items = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
        labels = [name for name, _ in sorted_items]
        importances = [score for _, score in sorted_items]
        title = "Grouped SHAP Feature Importance"
    else:
        # Standard per‑feature importance
        sorted_indices = np.argsort(mean_abs)[::-1]
        labels = feature_names[sorted_indices].tolist()
        importances = mean_abs[sorted_indices].tolist()
        title = "SHAP Feature Importance"
    # Plot bar chart
    if args.interactive:
        # Use plotly for an interactive bar chart
        try:
            import plotly.express as px  # type: ignore
            fig = px.bar(
                x=list(range(len(labels))),
                y=importances,
                labels={"x": "Feature", "y": "Mean |SHAP value|"},
            )
            fig.update_layout(
                title=title,
                xaxis=dict(tickmode="array", tickvals=list(range(len(labels))), ticktext=labels, tickangle=90),
            )
            out_html = results_dir / "shap_summary.html"
            fig.write_html(str(out_html))
            print(f"Interactive SHAP summary saved to {out_html}")
        except Exception as exc:
            print(f"Failed to generate interactive plot: {exc}; falling back to static PNG")
            args.interactive = False
    if not args.interactive:
        plt.figure(figsize=(max(6, len(labels) * 0.3), 4))
        plt.bar(range(len(labels)), importances, color="mediumorchid")
        plt.xticks(range(len(labels)), labels, rotation=90, ha="right")
        plt.ylabel("Mean |SHAP value|")
        plt.title(title)
        plt.tight_layout()
        out_path = results_dir / "shap_summary.png"
        plt.savefig(out_path)
        plt.close()
        print(f"SHAP summary plot saved to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()