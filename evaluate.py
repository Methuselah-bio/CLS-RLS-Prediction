"""Model evaluation script.

This script loads a previously trained model and evaluates it on a
provided processed dataset.  It mirrors the evaluation performed
during training but can be run independently to assess new test sets
or perform fairness audits.  When evaluating survival models the
concordance index and integrated Brier score are computed.

Usage::

    python src/evaluate.py --config configs/base.yaml --data data/processed/new.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from .train import compute_metrics, compute_survival_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a dataset")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--data", type=str, help="Path to processed CSV to evaluate")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model joblib file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    data_path = Path(args.data)
    model_path = Path(args.model) if args.model else None
    # Load config
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    problem_type = config.get("task", {}).get("problem_type", "classification").lower()
    # Determine model path
    if model_path is None:
        model_path = Path(config["paths"]["results"]) / "model.joblib"
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    metrics: Dict[str, Any] = {}
    if problem_type == "survival":
        duration_col = config["task"]["target_time"]
        event_col = config["task"]["target_event"]
        durations = df[duration_col].astype(float).values
        events = df[event_col].astype(int).values
        X = df.drop(columns=[duration_col, event_col])
        metrics = compute_survival_metrics(model, X, durations, events)
    else:
        target_col = config["task"]["target"]
        X = df.drop(columns=[target_col])
        y = df[target_col].values
        metrics = compute_metrics(model, X, y)
    out_path = Path(config["paths"]["results"]) / "evaluation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation complete. Metrics saved to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()