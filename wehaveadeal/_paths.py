"""Filesystem locations shipped inside the package.

The trained weights of the learned baselines (ML, ML2, RL, RL2) live in
`wehaveadeal/baselines_data/`, so `pip install` gives a working benchmark
without retraining, and paths resolve the same from any working directory.
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
BASELINES_DATA_DIR = PACKAGE_DIR / "baselines_data"


def baseline_artifact(relative: str) -> str:
    """Absolute path (as str) of a bundled baseline artifact, e.g. "ml/carrier_ml_model.joblib"."""
    return str(BASELINES_DATA_DIR / relative)


def load_sklearn_model(path):
    """Load a bundled scikit-learn model without the version-mismatch warning.

    The ML models were saved with scikit-learn 1.8.0. Every version allowed by
    pyproject.toml (>=1.5,<1.9) has been checked to reproduce the paper, so the
    InconsistentVersionWarning is noise for users rather than a real risk.
    """
    import warnings

    import joblib

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
        return joblib.load(path)
