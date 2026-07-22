#!/usr/bin/env python3
"""Cell-type-variable concordance bins: a marker heatmap + an out-of-selection UMAP.

Two panels that together make -- and DEFEND -- the claim that long-range
concordance is cell-type-specific:

  HEATMAP (F5C)  bins with HIGH between-cell-type variance and LOW within-
                 cell-type between-batch variance, shown as concordance across
                 cells, with cell-type AND batch colour bars.  Selecting on
                 cell-type variance AND against batch variance is what makes the
                 blocks track biology rather than sample.

  UMAP (F5D)     the SAME cells embedded on the bins that were NOT selected.
                 This is the control that keeps the heatmap honest: picking the
                 top cell-type-variable bins and then showing they separate cell
                 types is CIRCULAR -- it works on pure noise.  If the bins we did
                 NOT pick still separate cell types, the structure is genome-
                 wide, not an artefact of selection.

Selection is done on a TRAIN split of cells and everything is displayed on the
HELD-OUT split, so even the heatmap is a genuine out-of-sample demonstration
rather than a restatement of the selection.

Per-bin variance is decomposed by one-way ANOVA (the ICC): the between-group
share of variance, with sampling noise in the denominator rather than counted as
signal.  Cell-type ICC ranks the bins; batch ICC (computed WITHIN cell type)
filters out sample-driven bins.

Input is the cells x bins concordance parquet -- e.g.
context_GCH/05_concordance/concordance_signed.parquet -- one per context.

Usage
-----
    python scripts/celltype_marker_bins.py \\
        --matrix gch=out/context_GCH/05_concordance/concordance_signed.parquet \\
        --matrix hcg=out/context_HCG/05_concordance/concordance_signed.parquet \\
        --samples samples.tsv --cell-type-column cell_type --batch-column sample \\
        --exclude-chrom chrM,chrX,chrY --output-dir output/marker_bins
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _icc_mod():
    """Reuse the tested one-way-ANOVA ICC from the sibling script."""
    path = Path(__file__).resolve().parent / "celltype_variance_vs_distance.py"
    spec = importlib.util.spec_from_file_location("_ctv", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_ctv"] = m
    spec.loader.exec_module(m)
    return m


def _batch_icc(X, cell_type_ids, batch_ids, n_batch, icc_fn, min_cells, min_g):
    """Between-batch ICC computed WITHIN cell type, then averaged over types.

    A bin whose two samples separate *inside* a cell type is batch-driven, and
    that is exactly what must be filtered out -- so batch variance is measured
    within each cell type and pooled, not globally (globally, cell-type
    structure would swamp it).
    """
    n_reg = X.shape[1]
    acc = np.zeros(n_reg)
    wsum = np.zeros(n_reg)
    for ct in np.unique(cell_type_ids):
        m = cell_type_ids == ct
        if m.sum() < 2 * min_cells:
            continue
        icc, usable = icc_fn(X[m], batch_ids[m], n_batch, min_cells, min_g)
        w = usable.astype(float)
        acc += np.where(np.isfinite(icc), icc, 0.0) * w
        wsum += w
    return np.where(wsum > 0, acc / np.maximum(wsum, 1), np.nan)


def _stratified_split(labels, frac, seed):
    """Train/test cell split, stratified by label so every type is in both."""
    rng = np.random.default_rng(seed)
    train = np.zeros(len(labels), bool)
    for lv in pd.unique(labels):
        idx = np.flatnonzero(labels == lv)
        rng.shuffle(idx)
        train[idx[: max(1, int(round(frac * len(idx))))]] = True
    return train


def _svd_umap(M, n_svd, skip_top, seed):
    """SVD (drop the top `skip_top`) then UMAP; returns (n,2) or None."""
    try:
        import umap
    except Exception:
        return None
    from sklearn.decomposition import TruncatedSVD
    X = M.to_numpy(np.float64)
    X = np.where(np.isfinite(X), X, np.nanmean(X, axis=0, keepdims=True))
    X = np.where(np.isfinite(X), X, 0.0)
    k = min(n_svd + skip_top, X.shape[0] - 1, X.shape[1])
    if k > 1:
        emb = TruncatedSVD(n_components=k, random_state=seed).fit_transform(X)
        X = emb[:, skip_top:]
    nn = min(15, max(2, X.shape[0] - 1))
    return umap.UMAP(n_neighbors=nn, random_state=seed).fit_transform(X)


def _celltype_r2(coords, labels):
    ok = pd.notna(labels)
    P = np.asarray(coords)[ok]
    lab = np.asarray(labels)[ok]
    if len(np.unique(lab)) < 2:
        return np.nan
    tot = ((P - P.mean(0)) ** 2).sum()
    w = sum(((P[lab == g] - P[lab == g].mean(0)) ** 2).sum()
            for g in np.unique(lab))
    return float(1 - w / tot) if tot > 0 else np.nan


def select_marker_bins(X, cell_type_ids, n_ct, batch_ids, n_batch, icc_fn,
                       args):
    """Bins ranked by cell-type ICC, filtered to low batch ICC.

    Returns (selected_idx, ct_icc, batch_icc).  Selection uses ONLY the rows in
    X (the train split), so the held-out split can validate it.
    """
    ct_icc, _ = icc_fn(X, cell_type_ids, n_ct, args.min_cells_per_group,
                       args.min_groups)
    # Batch has its own group count -- typically just 2 samples -- so the
    # cell-type --min-groups (default 3) would render EVERY bin unusable and
    # silently disable the batch filter.  Batch needs >=2 groups covering it.
    b_icc = _batch_icc(X, cell_type_ids, batch_ids, n_batch, icc_fn,
                       args.min_cells_per_group, min(2, n_batch))
    ok = np.isfinite(ct_icc)
    # batch filter: keep bins whose within-type batch ICC is below the cut
    # (NaN batch ICC = batch not measurable there -> keep, it is not batchy)
    batch_ok = ~(np.isfinite(b_icc) & (b_icc > args.max_batch_icc))
    cand = np.flatnonzero(ok & batch_ok)
    if cand.size == 0:
        return np.array([], int), ct_icc, b_icc
    order = cand[np.argsort(ct_icc[cand])[::-1]]
    top = order[: args.n_bins]
    return top, ct_icc, b_icc


def _nanmedian(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


def _cluster_rows(Z):
    """Order rows (bins) by hierarchical clustering so blocks band together."""
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
    except Exception:
        return np.arange(Z.shape[0])
    if Z.shape[0] < 3:
        return np.arange(Z.shape[0])
    return leaves_list(linkage(Z, method="average", metric="correlation"))


def _heatmap(M, cells_order, bins_order, ct_labels, batch_labels, ct_colors,
             title, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    Z = M.loc[cells_order, bins_order].to_numpy(np.float64).T   # bins x cells
    mu = np.nanmean(Z, axis=1, keepdims=True)
    sd = np.nanstd(Z, axis=1, keepdims=True)
    Z = np.where(sd > 0, (Z - mu) / np.where(sd > 0, sd, 1), 0.0)
    Z = np.clip(np.where(np.isfinite(Z), Z, 0.0), -2.5, 2.5)
    Z = Z[_cluster_rows(Z)]                          # band similar bins together

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 2, height_ratios=[0.5, 0.5, 12], width_ratios=[40, 1],
                  hspace=0.05, wspace=0.02)
    # cell-type strip
    ax_ct = fig.add_subplot(gs[0, 0])
    uniq = list(dict.fromkeys(ct_labels.loc[cells_order]))
    cmap_ct = {g: ct_colors.get(g, plt.get_cmap("tab20")(i % 20))
               for i, g in enumerate(uniq)}
    ax_ct.imshow([[list(matplotlib.colors.to_rgb(cmap_ct[g]))
                   for g in ct_labels.loc[cells_order]]], aspect="auto")
    ax_ct.set_xticks([]); ax_ct.set_yticks([]); ax_ct.set_ylabel(
        "cell type", rotation=0, ha="right", va="center", fontsize=8)
    # batch strip
    ax_b = fig.add_subplot(gs[1, 0])
    buniq = list(dict.fromkeys(batch_labels.loc[cells_order]))
    cmap_b = {g: plt.get_cmap("Set2")(i) for i, g in enumerate(buniq)}
    ax_b.imshow([[list(matplotlib.colors.to_rgb(cmap_b[g]))
                  for g in batch_labels.loc[cells_order]]], aspect="auto")
    ax_b.set_xticks([]); ax_b.set_yticks([]); ax_b.set_ylabel(
        "batch", rotation=0, ha="right", va="center", fontsize=8)
    # heatmap
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                   interpolation="nearest")
    ax.set_xlabel(f"cells (n={len(cells_order)}, ordered by cell type)")
    ax.set_ylabel(f"selected bins (n={len(bins_order)})")
    ax.set_xticks([]); ax.set_yticks([])
    cax = fig.add_subplot(gs[2, 1])
    fig.colorbar(im, cax=cax, label="concordance (z per bin)")
    # legends
    import matplotlib.patches as mp
    ax_ct.legend([mp.Patch(color=matplotlib.colors.to_rgb(cmap_ct[g]))
                  for g in uniq], uniq, loc="upper left",
                 bbox_to_anchor=(1.01, 1.4), fontsize=7, ncol=1,
                 title="cell type", title_fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _umap_plot(coords, ct_labels, batch_labels, ct_colors, title, out, note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, (labels, name, colors) in zip(
            axes, [(ct_labels, "cell type", ct_colors),
                   (batch_labels, "batch", None)]):
        uniq = list(dict.fromkeys(labels))
        cmap = plt.get_cmap("tab20" if len(uniq) > 10 else "tab10")
        for i, g in enumerate(uniq):
            m = (labels == g).to_numpy()
            c = (colors.get(g) if colors else None) or cmap(i % cmap.N)
            ax.scatter(coords[m, 0], coords[m, 1], s=14, color=c, alpha=0.8,
                       edgecolors="none", label=str(g))
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.set_title(f"coloured by {name}", fontsize=10)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.text(0.5, 0.01, note, ha="center", fontsize=8.5, color="0.35")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ctv = _icc_mod()
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", action="append", required=True,
                    metavar="NAME=PATH", help="cells x bins concordance "
                    "parquet, e.g. gch=.../concordance_signed.parquet. "
                    "Repeatable.")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cell-type-column", default="cell_type")
    ap.add_argument("--batch-column", default="sample",
                    help="Column whose within-cell-type variance selects "
                         "AGAINST a bin. Comma-separated for several.")
    ap.add_argument("--color-column", default=None,
                    help="Samples-sheet column giving a colour per cell type.")
    ap.add_argument("--exclude-chrom", action="append", default=None)
    ap.add_argument("--n-bins", type=int, default=500,
                    help="Number of marker bins for the heatmap.")
    ap.add_argument("--max-batch-icc", type=float, default=0.1,
                    help="Drop a candidate bin whose within-cell-type between-"
                         "batch ICC exceeds this — it is sample-driven.")
    ap.add_argument("--train-frac", type=float, default=0.5,
                    help="Fraction of cells used to SELECT bins; the rest are "
                         "held out for the heatmap and UMAP.")
    ap.add_argument("--min-cells-per-group", type=int, default=3)
    ap.add_argument("--min-groups", type=int, default=3)
    ap.add_argument("--n-svd", type=int, default=10)
    ap.add_argument("--skip-top-svd", type=int, default=1)
    ap.add_argument("--n-permutations", type=int, default=200)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--plot-format", choices=["png", "pdf", "svg"],
                    default="pdf")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from nome_hic_concordance.cli import load_samples
    from nome_hic_concordance.utils import (bin_label_chrom, chrom_excluded,
                                            ensure_dir, parse_exclude_chroms)

    samples = load_samples(args.samples)
    for col in (args.cell_type_column, *args.batch_column.split(",")):
        if col not in samples.columns:
            sys.exit(f"column {col!r} not in --samples "
                     f"(columns: {samples.columns.tolist()}).")
    samples = samples.copy()
    samples["cell_name"] = samples["cell_name"].astype(str)
    ct_of = dict(zip(samples["cell_name"], samples[args.cell_type_column].astype(str)))
    bcols = [c.strip() for c in args.batch_column.split(",")]
    batch_of = dict(zip(samples["cell_name"],
                        samples[bcols].astype(str).agg("|".join, axis=1)))
    ct_colors = {}
    if args.color_column and args.color_column in samples.columns:
        for ct, c in zip(samples[args.cell_type_column].astype(str),
                         samples[args.color_column].astype(str)):
            if c and c.lower() not in ("nan", "none", ""):
                ct_colors.setdefault(ct, c)

    excl = parse_exclude_chroms(args.exclude_chrom)
    out_dir = ensure_dir(Path(args.output_dir))
    summary = {}

    for spec in args.matrix:
        if "=" not in spec:
            sys.exit(f"--matrix must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not Path(path).exists():
            sys.exit(f"no such matrix: {path}")
        M = pd.read_parquet(path)
        M.index = M.index.astype(str)
        keep = [c for c in M.index if c in ct_of]
        M = M.loc[keep]
        if excl:
            M = M[[b for b in M.columns
                   if not chrom_excluded(bin_label_chrom(b), excl)]]
        if M.shape[0] < 2 * args.min_cells_per_group * args.min_groups:
            print(f"  {name}: only {M.shape[0]} matched cells — skipped",
                  file=sys.stderr)
            continue

        ct = pd.Series([ct_of[c] for c in M.index], index=M.index)
        batch = pd.Series([batch_of[c] for c in M.index], index=M.index)
        ct_u = sorted(ct.unique())
        b_u = sorted(batch.unique())
        ct_id = np.searchsorted(np.array(ct_u), ct.to_numpy())
        b_id = np.searchsorted(np.array(b_u), batch.to_numpy())

        train = _stratified_split(ct.to_numpy(), args.train_frac, args.seed)
        sel, ct_icc, b_icc = select_marker_bins(
            M.to_numpy()[train], ct_id[train], len(ct_u), b_id[train],
            len(b_u), ctv.icc_per_region, args)
        if sel.size == 0:
            print(f"  {name}: no bins passed selection — "
                  f"loosen --max-batch-icc?", file=sys.stderr)
            continue
        sel_bins = M.columns[sel]
        rest_bins = M.columns[np.setdiff1d(np.arange(M.shape[1]), sel)]

        test = ~train
        test_cells = M.index[test]
        # order held-out cells by cell type for the heatmap
        order = sorted(test_cells, key=lambda c: (ct[c], c))
        fig_hm = out_dir / f"marker_heatmap_{name}.{args.plot_format}"
        _heatmap(M.loc[test_cells], order, list(sel_bins), ct, batch,
                 ct_colors, f"{name.upper()}: {len(sel_bins)} cell-type-variable, "
                 f"batch-invariant bins (held-out cells)", fig_hm)

        # F5D: UMAP on the NON-selected bins, held-out cells
        coords = _svd_umap(M.loc[test_cells, rest_bins], args.n_svd,
                           args.skip_top_svd, args.seed)
        umap_r2 = perm_p = np.nan
        if coords is not None:
            umap_r2 = _celltype_r2(coords, ct.loc[test_cells])
            rng = np.random.default_rng(args.seed)
            null = []
            lab = ct.loc[test_cells].to_numpy()
            for _ in range(args.n_permutations):
                null.append(_celltype_r2(coords, pd.Series(rng.permutation(lab))))
            null = np.array([x for x in null if np.isfinite(x)])
            if null.size:
                perm_p = (1 + int((null >= umap_r2).sum())) / (1 + null.size)
            fig_um = out_dir / f"nonselected_umap_{name}.{args.plot_format}"
            _umap_plot(coords, ct.loc[test_cells], batch.loc[test_cells],
                       ct_colors,
                       f"{name.upper()}: UMAP on NON-selected bins "
                       f"({len(rest_bins)}), held-out cells",
                       fig_um,
                       f"cell-type R²={umap_r2:.3f} (perm p={perm_p:.3f}) on bins "
                       f"NOT chosen as markers — the out-of-selection control")

        pd.DataFrame({"bin": M.columns, "celltype_icc": ct_icc,
                      "batch_icc": b_icc,
                      "selected": M.columns.isin(sel_bins)}
                     ).to_csv(out_dir / f"bin_scores_{name}.tsv",
                              sep="\t", index=False)
        summary[name] = {
            "n_cells": int(M.shape[0]), "n_bins": int(M.shape[1]),
            "n_train": int(train.sum()), "n_test": int(test.sum()),
            "n_selected": int(sel.size),
            "median_celltype_icc_selected": _nanmedian(ct_icc[sel]),
            "median_batch_icc_selected": _nanmedian(b_icc[sel]),
            "nonselected_umap_celltype_r2": (None if not np.isfinite(umap_r2)
                                             else float(umap_r2)),
            "nonselected_umap_perm_p": (None if not np.isfinite(perm_p)
                                        else float(perm_p))}
        print(f"\n=== {name}: {M.shape[0]} cells x {M.shape[1]} bins ===")
        print(f"  selected {sel.size} bins (train n={int(train.sum())}): "
              f"median cell-type ICC {_nanmedian(ct_icc[sel]):.3f}, "
              f"median batch ICC {_nanmedian(b_icc[sel]):.3f}")
        print(f"  heatmap on {int(test.sum())} held-out cells -> {fig_hm.name}")
        if np.isfinite(umap_r2):
            verdict = ("SEPARATES" if perm_p < 0.05 else "does NOT separate")
            print(f"  NON-selected bins {verdict} cell types: "
                  f"R²={umap_r2:.3f} (perm p={perm_p:.3f}) -> the "
                  f"non-circular proof")

    if not summary:
        sys.exit("no matrix produced output.")
    with open(out_dir / "marker_bins_summary.json", "w") as fh:
        json.dump({"matrices": summary, "n_bins": args.n_bins,
                   "max_batch_icc": args.max_batch_icc,
                   "train_frac": args.train_frac,
                   "note": "bins selected on the TRAIN split; heatmap and UMAP "
                           "on the HELD-OUT split; UMAP uses NON-selected bins"},
                  fh, indent=2)
    print(f"\nRead it: the heatmap shows cell-type-variable, batch-invariant "
          "concordance on\n  HELD-OUT cells; the UMAP proves the structure is "
          "genome-wide by separating\n  cell types on the bins that were NOT "
          "selected.  If the UMAP does NOT separate,\n  the heatmap was a "
          "selection artefact and the claim fails.")


if __name__ == "__main__":
    main()
