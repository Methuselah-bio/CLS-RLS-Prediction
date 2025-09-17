"""Pipeline runner script.

This module orchestrates the data preparation, model training and
interpretation stages in a fault‑tolerant manner.  It can be used as
the main entry point for experiments, reducing the boilerplate of
invoking individual scripts.  Common exceptions (e.g., missing files,
package import errors) are caught and logged to aid debugging.

Usage::

    python src/runner.py --config configs/base.yaml

The runner will execute ``prepare_data.main`` followed by ``train.main``.
Additional stages such as feature selection or fairness auditing can be
incorporated in the future.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Methuselah pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Adjust logging to include time and level
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error("Configuration file %s does not exist", config_path)
        sys.exit(1)
    # Stage 1: prepare data
    try:
        from . import prepare_data
        prepare_data.main()
    except Exception as exc:
        logging.error("Data preparation failed: %s", exc)
        sys.exit(1)
    # Stage 2: train model
    try:
        from . import train
        train.main()
    except Exception as exc:
        logging.error("Model training failed: %s", exc)
        sys.exit(1)
    logging.info("Pipeline finished successfully")


if __name__ == "__main__":  # pragma: no cover
    main()