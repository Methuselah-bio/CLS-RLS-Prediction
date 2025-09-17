"""Top‑level package for the enhanced Methuselah‑prediction.

This package exposes modules for data preparation, model training,
advanced models, hyperparameter optimisation, causal and fairness
analysis, and data ingestion utilities.  It is structured as a
Python package so that notebooks and scripts can import project
components without relying on relative paths.  For example::

    from methuselah_prediction import prepare_data, train, fetch_biodata

These imports will resolve to the corresponding modules within
``src/``.  New modules such as ``fetch_biodata`` and
``train_nested_cv`` are also exported here for convenience.
"""

from . import prepare_data  # noqa: F401
from . import train  # noqa: F401
from . import evaluate  # noqa: F401
from . import advanced_models  # noqa: F401
from . import feature_selection  # noqa: F401
from . import causal_analysis  # noqa: F401
from . import fairness_audit  # noqa: F401
from . import interpret_shap  # noqa: F401
from . import optuna_search  # noqa: F401
from . import fetch_biodata  # noqa: F401
from . import train_nested_cv  # noqa: F401

__all__ = [
    "prepare_data",
    "train",
    "evaluate",
    "advanced_models",
    "feature_selection",
    "causal_analysis",
    "fairness_audit",
    "interpret_shap",
    "optuna_search",
    "fetch_biodata",
    "train_nested_cv",
]