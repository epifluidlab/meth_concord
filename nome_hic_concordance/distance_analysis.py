"""Concordance vs. genomic distance decay analysis.

Bins read pairs by genomic distance (log-spaced), computes the Pearson
correlation of ``(posterior1, posterior2)`` within each bin, and plots
per-cell curves alongside the population mean ± 95% CI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .utils import ensure_dir

logger = logging.getLogger(__name__)


DEFAULT_DISTANCE_BINS: list[tuple[int, int]] = [
    (1_000, 5_000),
    (5_000, 20_000),
    (20_000, 100_000),
    (100_000, 500_000),
    (500_000, 1_000_000),
    (1_000_000, 5_000_000),
    (5_000_000, 20_000_000),
]


def _process_cell(
    cell_name: str,
    path: str,
    distance_bins: Sequence[tuple[int, int]],
    min_gch_sites: int,
) -> list[dict]:
    """Compute per-distance-bin concordance for one cell."""
    df = pd.read_parquet(path)
    df = df[
        (df["n1"] >= min_gch_sites)
        & (df["n2"] >= min_gch_sites)
        & df["posterior1"].notna()
        & df["posterior2"].notna()
    ]

    rows: list[dict] = []
    for d_start, d_end in distance_bins:
        mask = (df["distance"] >= d_start) & (df["distance"] < d_end)
        sub = df.loc[mask]
        n_pairs = len(sub)
        if n_pairs >= 5 and sub["posterior1"].std() > 1e-10 and sub["posterior2"].std() > 1e-10:
            r = float(np.corrcoef(sub["posterior1"], sub["posterior2"])[0, 1])
        else:
            r = np.nan
        rows.append({
            "cell_name": cell_name,
            "dist_bin_start": int(d_start),
            "dist_bin_end": int(d_end),
            "r": r,
            "n_pairs": int(n_pairs),
            "dist_label": _format_dist_label(d_start, d_end),
        })
    return rows


def _format_dist_label(start: int, end: int) -> str:
    def _fmt(v: int) -> str:
        if v >= 1_000_000 and v % 1_000_000 == 0:
            return f"{v // 1_000_000}Mb"
        if v >= 1_000 and v % 1_000 == 0:
            return f"{v // 1_000}kb"
        return str(v)
    return f"{_fmt(start)}-{_fmt(end)}"


def concordance_vs_distance(
    cell_posterior_paths: dict[str, str],
    distance_bins: Sequence[tuple[int, int]] | None = None,
    min_gch_sites: int = 3,
    n_threads: int = 4,
    output_dir: str = "output/05_concordance",
) -> pd.DataFrame:
    """Compute per-cell concordance within log-spaced distance bins.

    Also writes ``concordance_vs_distance.tsv`` and, best-effort, a summary
    plot ``concordance_vs_distance.png``.
    """
    from joblib import Parallel, delayed

    if distance_bins is None:
        distance_bins = DEFAULT_DISTANCE_BINS

    out_path = ensure_dir(output_dir)

    items = list(cell_posterior_paths.items())
    if n_threads > 1 and len(items) > 1:
        all_rows = Parallel(n_jobs=n_threads)(
            delayed(_process_cell)(name, path, distance_bins, min_gch_sites)
            for name, path in items
        )
    else:
        all_rows = [
            _process_cell(name, path, distance_bins, min_gch_sites)
            for name, path in items
        ]

    flat_rows = [r for cell_rows in all_rows for r in cell_rows]
    result = pd.DataFrame(flat_rows)
    result.to_csv(out_path / "concordance_vs_distance.tsv", sep="\t", index=False)

    try:
        _plot_distance_decay(result, out_path)
    except Exception as exc:
        logger.warning(f"Distance-decay plot failed: {exc}")

    return result


def _plot_distance_decay(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for cell in df["cell_name"].unique():
        cell_df = df[df["cell_name"] == cell].sort_values("dist_bin_start")
        midpoints = np.sqrt(cell_df["dist_bin_start"] * cell_df["dist_bin_end"])
        ax.plot(midpoints, cell_df["r"], alpha=0.15, color="steelblue", linewidth=0.6)

    grouped = df.groupby("dist_bin_start").agg(
        r_mean=("r", "mean"),
        r_std=("r", "std"),
        r_count=("r", lambda x: x.notna().sum()),
        dist_bin_end=("dist_bin_end", "first"),
    ).reset_index()

    midpoints = np.sqrt(grouped["dist_bin_start"] * grouped["dist_bin_end"])
    ci = 1.96 * grouped["r_std"] / np.sqrt(grouped["r_count"].replace(0, np.nan))

    ax.fill_between(
        midpoints, grouped["r_mean"] - ci, grouped["r_mean"] + ci,
        alpha=0.3, color="darkblue",
    )
    ax.plot(midpoints, grouped["r_mean"], "o-", color="darkblue", linewidth=2,
            label="Mean ± 95% CI")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Genomic distance (bp)")
    ax.set_ylabel("Concordance (Pearson r)")
    ax.set_title("Long-range chromatin accessibility concordance vs. distance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path / "concordance_vs_distance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
