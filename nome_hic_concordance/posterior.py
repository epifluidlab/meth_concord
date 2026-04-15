"""Posterior classification of read ends given a fitted mixture model."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .mixture_model import BetaBinomialMixture
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def compute_cell_posteriors(
    parquet_path: str,
    mixture_model: BetaBinomialMixture,
    output_dir: str,
    cell_name: str,
) -> str:
    """Add ``posterior1`` and ``posterior2`` columns to one cell's read-pair data.

    ``posteriorX = P(accessible | kX, nX)`` under the fitted mixture.
    Read ends with ``n == 0`` get ``NaN`` posteriors.
    """
    df = pd.read_parquet(parquet_path)

    df["posterior1"] = mixture_model.predict_proba(df["k1"].values, df["n1"].values)
    df["posterior2"] = mixture_model.predict_proba(df["k2"].values, df["n2"].values)

    out_path = ensure_dir(output_dir) / f"{cell_name}.posteriors.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)


def compute_all_posteriors(
    cell_parquet_paths: dict[str, str],
    mixture_model: BetaBinomialMixture,
    output_dir: str,
    n_threads: int = 4,
) -> dict[str, str]:
    """Run :func:`compute_cell_posteriors` for every cell (parallel)."""
    from joblib import Parallel, delayed

    ensure_dir(output_dir)

    items = list(cell_parquet_paths.items())
    if n_threads > 1 and len(items) > 1:
        results = Parallel(n_jobs=n_threads)(
            delayed(compute_cell_posteriors)(path, mixture_model, output_dir, cell_name)
            for cell_name, path in items
        )
    else:
        results = [
            compute_cell_posteriors(path, mixture_model, output_dir, cell_name)
            for cell_name, path in items
        ]

    return dict(zip([name for name, _ in items], results))
