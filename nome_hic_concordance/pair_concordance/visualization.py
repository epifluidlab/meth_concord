"""Pair-concordance-specific visualisation: UMAP plots, forest plots, diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_umap(
    umap_coords: np.ndarray,
    cell_names: list[str],
    labels: pd.Series | None = None,
    title: str = "UMAP",
    output_path: str = "umap.png",
) -> None:
    """2D UMAP scatter, optionally coloured by cell-type label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))

    if labels is not None:
        label_arr = labels.reindex(cell_names).values
        unique = [g for g in pd.unique(label_arr) if pd.notna(g)]
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique), 1)))
        for g, c in zip(unique, colors):
            mask = label_arr == g
            ax.scatter(
                umap_coords[mask, 0], umap_coords[mask, 1],
                c=[c], label=str(g), s=15, alpha=0.75, edgecolors="none",
            )
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    else:
        ax.scatter(
            umap_coords[:, 0], umap_coords[:, 1],
            s=15, alpha=0.75, c="steelblue", edgecolors="none",
        )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_three_umaps(
    embeddings: dict[str, dict],
    labels: pd.Series | None,
    output_dir: str,
) -> None:
    """Generate UMAP plots for E_contact, E_coord, and E_full."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, emb in embeddings.items():
        if "umap" not in emb or emb["umap"] is None:
            continue
        plot_umap(
            emb["umap"],
            emb["cell_names"],
            labels=labels,
            title=f"UMAP — {name}",
            output_path=str(out / f"umap_{name}.png"),
        )
        # Save coordinates
        coords = pd.DataFrame({
            "cell": emb["cell_names"],
            "UMAP1": emb["umap"][:, 0],
            "UMAP2": emb["umap"][:, 1],
        })
        if labels is not None:
            coords["label"] = labels.reindex(emb["cell_names"]).values
        coords.to_csv(out / f"umap_{name}_coordinates.tsv", sep="\t", index=False)


def plot_auc_forest(
    auc_df: pd.DataFrame,
    output_path: str = "auc_forest.png",
) -> None:
    """Forest plot of AUC per (category, distance_band) for each embedding."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if auc_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(auc_df) * 0.35)))

    labels = auc_df["category"].astype(str)
    if "distance_band" in auc_df.columns:
        labels = labels + " / " + auc_df["distance_band"].astype(str)

    y = np.arange(len(auc_df))
    ax.barh(y, auc_df["auc_mean"], xerr=auc_df.get("auc_std", 0),
            height=0.6, color="steelblue", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Classification AUC")
    ax.set_title("Cell-type AUC by pair category")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(fontsize=8)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_delta_distribution(
    delta_matrix: pd.DataFrame,
    panel: pd.DataFrame,
    output_path: str = "delta_distribution.png",
) -> None:
    """Histogram of Δ values per category."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if delta_matrix.empty:
        return

    categories = panel.set_index("pair_id")["category"]
    unique_cats = categories.unique()
    n_cats = len(unique_cats)
    if n_cats == 0:
        return

    fig, axes = plt.subplots(1, min(n_cats, 4), figsize=(4 * min(n_cats, 4), 3.5),
                              squeeze=False)

    for i, cat in enumerate(unique_cats[:4]):
        ax = axes[0, i]
        pair_ids = categories[categories == cat].index
        cols = [c for c in pair_ids if c in delta_matrix.columns]
        if not cols:
            ax.set_title(f"{cat}\n(no data)")
            continue
        vals = delta_matrix[cols].values.ravel()
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            ax.hist(vals, bins=50, alpha=0.7, color="steelblue", density=True)
        ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
        ax.set_title(cat, fontsize=9)
        ax.set_xlabel("Δ")
        if i == 0:
            ax.set_ylabel("density")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_diagnostic(
    pair_counts_stacked: pd.DataFrame,
    panel: pd.DataFrame,
    output_path: str = "coverage_diagnostic.png",
) -> None:
    """Histogram of n_linked per pair across cells."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if pair_counts_stacked.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: distribution of n_linked per (pair, cell)
    ax = axes[0]
    vals = pair_counts_stacked["n_linked"].values
    vals = vals[vals > 0]
    if len(vals) > 0:
        ax.hist(vals, bins=50, alpha=0.7, color="steelblue")
    ax.set_xlabel("n_linked per (pair, cell)")
    ax.set_ylabel("count")
    ax.set_title("Linked reads per pair-cell")
    ax.set_yscale("log")

    # Panel 2: fraction of pairs with >=1, >=5, >=10 linked reads per cell
    ax = axes[1]
    per_cell = pair_counts_stacked.groupby("cell_name")["n_linked"].agg(
        total="sum",
        ge1=lambda x: (x >= 1).sum(),
        ge5=lambda x: (x >= 5).sum(),
        ge10=lambda x: (x >= 10).sum(),
    )
    for col, label in [("ge1", "≥1"), ("ge5", "≥5"), ("ge10", "≥10")]:
        ax.hist(per_cell[col], bins=30, alpha=0.5, label=label)
    ax.set_xlabel("# pairs with ≥N linked reads")
    ax.set_ylabel("# cells")
    ax.set_title("Per-cell pair coverage")
    ax.legend()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
