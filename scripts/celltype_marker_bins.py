#!/usr/bin/env python3
"""Cell-type-variable concordance bins: a marker heatmap (F5C).

Two panels that together make -- and DEFEND -- the claim that long-range
concordance is cell-type-specific:

  HEATMAP (F5C)  bins with HIGH between-cell-type variance and LOW within-
                 cell-type between-batch variance, shown as concordance across
                 cells, with cell-type AND batch colour bars.  Selecting on
                 cell-type variance AND against batch variance is what makes the
                 blocks track biology rather than sample.

  CONTROL        the out-of-selection honesty check: do the bins we did NOT
                 select still separate cell types?  Picking the top cell-type-
                 variable bins and then showing they separate cell types is
                 CIRCULAR -- it works on pure noise.  So the non-selected bins
                 are scored by cell-type silhouette against a permutation null;
                 if they separate too (p<0.05), the heatmap is real.  The UMAP
                 of the full / non-selected bins (F5D) is already produced by
                 nome_hic_concordance and nome_hic_compare, so only the NUMBER
                 is emitted here.

Bins are selected on ALL cells and the heatmap shows ALL cells; the honesty
comes from the CONTROL (the bins we did NOT pick still separate cell types),
not from holding cells out.  Selection prefers batch-invariant bins but fills
to --n-bins by cell-type ICC so the heatmap has enough rows on real data.

Per-bin variance is decomposed by one-way ANOVA (the ICC): the between-group
share of variance, with sampling noise in the denominator rather than counted as
signal.  Cell-type ICC ranks the bins; batch ICC is computed WITHIN cell type,
separately for EACH batch variable (exp/plate/lib), and a bin is filtered out if
ANY variable drives it.  Rows (bins) are hierarchically clustered on the CELL-
TYPE-MEAN pattern, so the row order reflects cell type, never batch.

The matrix is normalised exactly as nome_hic_concordance / nome_hic_compare do
before the UMAP (quantile per-bin + per-cell rank + recenter for the signed
method), so the marker bins live in the same feature space as the embedding.

Input is the cells x bins concordance parquet.  Use one built with the SIGNED
method and a minimum pair distance > 1 kb (i.e. run nome_hic_concordance with
--concordance-method signed --min-distance 1000 or larger); this script reads
that matrix and does not recompute concordance, so the distance floor is set
upstream.

Usage
-----
    python scripts/celltype_marker_bins.py \\
        --matrix gch=out/context_GCH/05_concordance/concordance_signed.parquet \\
        --matrix hcg=out/context_HCG/05_concordance/concordance_signed.parquet \\
        --samples samples.tsv --cell-type-column cell_type \\
        --batch-column exp_batch,plate_batch,lib_batch \\
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



def _nonselected_separation(M, labels, n_svd, skip_top, seed, n_perm):
    """Do the NON-selected bins still separate cell types?  (silhouette + perm)

    This is the out-of-selection control that keeps the heatmap honest -- picking
    the top cell-type-variable bins and showing they separate cell types is
    circular, but if the bins we did NOT pick separate cell types too, the
    structure is genome-wide.  The UMAP of these bins is already produced by
    nome_hic_concordance / nome_hic_compare, so this returns just the NUMBER:
    the cell-type silhouette on the non-selected-bin SVD scores, against a
    label-permutation null.  Returns (silhouette, perm_p).
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import silhouette_score
    X = M.to_numpy(np.float64)
    X = np.where(np.isfinite(X), X, np.nanmean(X, axis=0, keepdims=True))
    X = np.where(np.isfinite(X), X, 0.0)
    k = min(n_svd + skip_top, X.shape[0] - 1, X.shape[1])
    if k > 1:
        emb = TruncatedSVD(n_components=k, random_state=seed).fit_transform(X)
        X = emb[:, skip_top:]
    lab = np.asarray(labels)
    if len(np.unique(lab)) < 2 or X.shape[0] <= len(np.unique(lab)):
        return np.nan, np.nan
    obs = float(silhouette_score(X, lab))
    rng = np.random.default_rng(seed)
    null = np.array([silhouette_score(X, rng.permutation(lab))
                     for _ in range(n_perm)])
    # (1 + count)/(1 + B): a finite permutation cannot justify p = 0
    p = (1 + int((null >= obs).sum())) / (1 + n_perm)
    return obs, float(p)


def select_marker_bins(X, cell_type_ids, n_ct, batch_specs, icc_fn, args):
    """Top ``n_bins`` by cell-type ICC, PREFERRING batch-invariant bins.

    Returns (selected_idx, ct_icc, batch_icc, n_clean).  Bins are ranked by
    between-cell-type ICC (high between cell types, low within -- exactly "most
    variable across cell types, least within"); those passing the batch cut are
    taken first, and if fewer than ``n_bins`` pass, the target is FILLED with the
    next-best by cell-type ICC so the heatmap has enough rows.  ``n_clean``
    reports how many passed the cut, so the trade-off is visible.

    ``batch_specs`` is a list of (batch_ids, n_levels), ONE per batch variable
    (exp/plate/lib), and a bin is 'batchy' if ANY of them drives it -- the
    within-cell-type batch ICC is taken as the MAX over the variables.
    """
    ct_icc, _ = icc_fn(X, cell_type_ids, n_ct, args.min_cells_per_group,
                       args.min_groups)
    # Batch has its own group count -- often just 2 levels -- so the cell-type
    # --min-groups (default 3) would render EVERY bin unusable and silently
    # disable the batch filter.  Each batch variable needs >=2 groups covering
    # it; a bin batchy in ANY variable is flagged (np.fmax ignores NaN).
    b_icc = np.full(X.shape[1], np.nan)
    for batch_ids, n_batch in batch_specs:
        bi = _batch_icc(X, cell_type_ids, batch_ids, n_batch, icc_fn,
                        args.min_cells_per_group, min(2, n_batch))
        b_icc = np.fmax(b_icc, bi)
    usable = np.flatnonzero(np.isfinite(ct_icc))
    if usable.size == 0:
        return np.array([], int), ct_icc, b_icc, 0
    ranked = usable[np.argsort(ct_icc[usable])[::-1]]   # best cell-type ICC first
    is_clean = ~(np.isfinite(b_icc) & (b_icc > args.max_batch_icc))
    clean = ranked[is_clean[ranked]]
    if clean.size >= args.n_bins:
        return clean[: args.n_bins], ct_icc, b_icc, int(args.n_bins)
    dirty = ranked[~is_clean[ranked]]
    sel = np.concatenate([clean, dirty[: args.n_bins - clean.size]])
    return sel, ct_icc, b_icc, int(clean.size)


def _prep(M, name, out_dir, args):
    """Normalise a cells x bins matrix exactly as the pipeline UMAP does.

    Reuses nome_hic_compare's _prep_matrix (which reuses matrix_ops), so the
    marker bins are chosen in the SAME feature space as the embedding: quantile
    per-bin normalisation, per-cell rank, and post-norm recenter for the signed
    method.
    """
    from nome_hic_concordance.comparison import _prep_matrix
    return _prep_matrix(M, name, out_dir, args.matrix_norm,
                        args.min_cell_coverage, args.per_cell_norm,
                        args.post_norm_recenter == "on")


def _nanmedian(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan



# Distinct diverging colour schemes so GCH-GCH and HCG-HCG never look alike.
_CMAPS = {"gch": "RdBu_r", "hcg": "PuOr_r"}
_CMAP_FALLBACK = ["RdBu_r", "PuOr_r", "BrBG_r", "PiYG_r"]

# What the mean-signal companion heatmaps display (GCH=accessibility,
# HCG=methylation; both stored as accessibility_mean.parquet per context).
_MEAN_LABEL = {"gch": "mean accessibility", "hcg": "mean methylation"}


def _mean_cmap(name):
    """Sequential colormap for a mean-signal companion heatmap, matching the
    conventional NOMe-seq view: GCH accessibility is white(low) -> teal(high);
    HCG methylation is blue(low) -> yellow(high).

    GCH is a custom white->teal ramp; HCG uses matplotlib's 'cividis', the
    canonical perceptually-uniform, colour-blind-safe blue->yellow map."""
    from matplotlib.colors import LinearSegmentedColormap
    n = name.lower()
    if n == "gch":
        return LinearSegmentedColormap.from_list(
            "acc_white_teal", ["#ffffff", "#7fcccc", "#1f9e9e", "#0b5c5c"])
    if n == "hcg":
        return "cividis"
    return "cividis"


def _remove_batch(M, ct_labels, batch_df):
    """Center each (cell type, batch) group to its cell-type mean, per bin.

    Removes WITHIN-cell-type batch structure -- the main effect AND the
    cell-type-specific interaction the plain global correction leaves behind --
    while preserving the cell-type means (the signal the heatmap is about).
    Applied once per batch variable in turn.  Because cell-type composition is
    balanced across batches here, this cannot move a cell type's mean.
    """
    cells = M.index
    df = M.copy()
    ctk = ct_labels.loc[cells].astype(str)
    ct_mean = df.groupby(ctk.to_numpy()).transform("mean")
    for col in batch_df.columns:
        grp = (ctk + "|" + batch_df[col].loc[cells].astype(str)).to_numpy()
        df = df - df.groupby(grp).transform("mean") + ct_mean
    return df



def _row_linkage(Z):
    """Linkage for the bins, or None; returns (linkage, leaf_order)."""
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
    except Exception:
        return None, np.arange(Z.shape[0])
    if Z.shape[0] < 3:
        return None, np.arange(Z.shape[0])
    lk = linkage(Z, method="average", metric="correlation")
    return lk, leaves_list(lk)


def _qual_palette():
    """A stable list of ~50 visually distinct RGB colours for batch strips.

    This is the family for the FIRST batch column (e.g. exp_batch).  Leads with
    10 DISTINCT-HUE colours (tab10, reordered so the first two are green then
    orange) so a 2-level strip like exp_batch gets HIGH-CONTRAST colours (green
    vs orange) instead of the dark/light same-hue pair tab20's first two slots
    give (dark blue vs light blue).  tab20b + tab20c extend it for many-level
    strips.  Levels are coloured by position in a FIXED, sorted order, so a
    label keeps its colour (and legend slot) across every heatmap (gch, hcg, …).
    Later batch columns use their OWN distinct colormaps (see _column_palette),
    so the stacked strips never share a palette.
    """
    import matplotlib
    tab10 = [matplotlib.colors.to_rgb(c)
             for c in matplotlib.colormaps["tab10"].colors]
    # tab10 is [blue, orange, green, red, purple, brown, pink, grey, olive, cyan]
    # -> reorder to [green, orange, blue, red, ...] so slot 0/1 = green/orange.
    out = [tab10[i] for i in (2, 1, 0, 3, 4, 5, 6, 7, 8, 9)]
    for name in ("tab20b", "tab20c"):
        out += [matplotlib.colors.to_rgb(c)
                for c in matplotlib.colormaps[name].colors]
    return out


# One distinct colour FAMILY per batch-column POSITION, so stacked batch strips
# (exp_batch / plate_batch / lib_batch ...) are easy to tell apart at a glance.
# Position 0 keeps the qualitative green/orange-lead hues from _qual_palette (so
# exp_batch stays batch2=green, scNH=orange); each later column samples a
# DIFFERENT sequential colormap -- visually unmistakable from the qualitative
# strip and from one another, with unlimited distinct levels for a
# high-cardinality column like lib_batch (dates) and a natural dark->light
# ordering for ordered levels (plates, dates).
_BATCH_CMAPS = ("viridis", "plasma", "cividis", "spring", "winter", "copper")


def _column_palette(index, n):
    """``n`` RGB colours for the batch column at position ``index``.

    Column 0 uses the qualitative distinct-hue palette (green/orange lead);
    columns 1+ each sample a DIFFERENT sequential colormap, so no two batch
    strips share a palette and a many-level column (e.g. lib_batch dates) still
    gets ``n`` distinct, naturally-ordered colours.
    """
    import matplotlib
    n = max(int(n), 1)
    if index <= 0:
        pal = _qual_palette()
        return [pal[i % len(pal)] for i in range(n)]
    cmap = matplotlib.colormaps[_BATCH_CMAPS[(index - 1) % len(_BATCH_CMAPS)]]
    if n == 1:
        return [tuple(cmap(0.5)[:3])]
    return [tuple(cmap(t)[:3]) for t in np.linspace(0.06, 0.96, n)]


def _batch_color_maps(batch_df, bcols):
    """Shared {column: {level: rgb}} and {column: [levels]} across ALL matrices.

    Built once from the full samples sheet, with levels in a deterministic
    sorted order, so batch colours and legend order are identical between the
    GCH and HCG heatmaps.  Each batch COLUMN gets its OWN colour family (see
    _column_palette / _BATCH_CMAPS) so the stacked strips are easy to tell apart.
    """
    order = {c: sorted(batch_df[c].dropna().astype(str).unique().tolist())
             for c in bcols}
    colors = {}
    for i, c in enumerate(bcols):
        pal = _column_palette(i, len(order[c]))
        colors[c] = {lev: pal[j] for j, lev in enumerate(order[c])}
    return colors, order


def _heatmap(M, cells, bins_order, ct_labels, batch_df, ct_colors,
             batch_colors, batch_order, title, out, cmap, legend_order=None,
             legend_max=40, order=None, value_label="concordance (z per bin)",
             zscore=True):
    """Marker heatmap: bins x cells.  Cells (columns) are hierarchically
    clustered and the COLUMN dendrogram is drawn on top of the cell-type strip;
    no cell names are shown.  A cell-type strip and one strip per batch variable
    (exp/plate/lib) sit below it, so it is visible that the cell clustering
    tracks cell type and not any batch.

    ``zscore`` (default) z-scores each bin across cells for display (diverging
    cmap, +/-2.5).  Set False to show the RAW value on a sequential scale (the
    mean accessibility / methylation companions), with a robust 2-98 percentile
    range.

    When ``order`` is supplied (a previous heatmap's layout) the columns are NOT
    re-clustered, so the dendrogram is OMITTED -- drawing one would falsely imply
    clustering was done on this matrix.

    Returns ``(cells_order, bins_order, col_linkage)`` -- the exact layout it
    drew.  Pass that tuple back as ``order`` to render a DIFFERENT matrix (e.g.
    mean accessibility) in the IDENTICAL layout instead of re-clustering."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    from matplotlib.gridspec import GridSpec

    # Cluster the CELLS (columns) on the selected-bin values and order columns by
    # that clustering; the dendrogram drawn on top uses the same linkage.  When
    # `order` is supplied (a previous heatmap's layout), reuse it verbatim so a
    # second matrix appears in the EXACT same row/column order.
    if order is None:
        Yc = M.loc[cells, bins_order].to_numpy(np.float64)
        Yc = np.where(np.isfinite(Yc), Yc, 0.0)
        col_lk, col_order = _row_linkage(Yc)          # cluster over cells
        cells_o = [cells[i] for i in col_order]
        bins_seq = list(bins_order)
    else:
        cells_o, bins_seq, col_lk = order
        cells_o = list(cells_o)
        bins_seq = list(bins_seq)

    Z = M.loc[cells_o, bins_seq].to_numpy(np.float64).T   # bins x cells
    if zscore:
        mu = np.nanmean(Z, axis=1, keepdims=True)
        sd = np.nanstd(Z, axis=1, keepdims=True)
        Z = np.where(sd > 0, (Z - mu) / np.where(sd > 0, sd, 1), 0.0)
        Z = np.clip(np.where(np.isfinite(Z), Z, 0.0), -2.5, 2.5)
        vmin, vmax = -2.5, 2.5
    else:
        # Raw value (mean accessibility / methylation): keep the LEVEL, pick a
        # robust display range for contrast; missing -> the low (white) end.
        finite = Z[np.isfinite(Z)]
        if finite.size:
            vmin, vmax = (float(v) for v in np.percentile(finite, [2, 98]))
            if vmax <= vmin:
                vmin, vmax = float(finite.min()), float(finite.max())
            if vmax <= vmin:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0
        Z = np.where(np.isfinite(Z), Z, vmin)
    if order is None:
        # Rows (bins) ordered by clustering on the cell-type-MEAN pattern (no row
        # dendrogram drawn), so bins band by which cell types they mark.
        ct_arr = np.asarray(ct_labels.loc[cells_o])
        means = np.stack([Z[:, ct_arr == g].mean(1) for g in
                          dict.fromkeys(ct_arr)], axis=1)
        row_perm = _row_linkage(means)[1]
        Z = Z[row_perm]
        bins_seq = [bins_seq[i] for i in row_perm]

    strips = [("cell type", ct_labels)] + [(c, batch_df[c]) for c in
                                           batch_df.columns]
    n_strip = len(strips)
    # Draw the dendrogram ONLY when we clustered here (order is None).  A
    # borrowed layout must NOT show one -- it would falsely imply this matrix was
    # clustered.
    show_dend = order is None and col_lk is not None
    n_top = 1 if show_dend else 0
    uniq_ct = (legend_order or list(dict.fromkeys(ct_labels.loc[cells_o])))
    fig = plt.figure(figsize=(12.5 + 0.13 * len(uniq_ct), 9.0 + 0.3 * n_strip))
    # (optional dendrogram row), then the strips, then the heatmap.
    gs = GridSpec(n_strip + 1 + n_top, 2,
                  height_ratios=([1.6] if show_dend else []) + [0.5] * n_strip
                  + [12], width_ratios=[24, 1], hspace=0.06, wspace=0.03)

    # Column dendrogram (cells), aligned to the heatmap columns.
    if show_dend:
        from scipy.cluster.hierarchy import dendrogram
        ax_d = fig.add_subplot(gs[0, 0])
        dendrogram(col_lk, ax=ax_d, no_labels=True, color_threshold=0,
                   above_threshold_color="0.4")
        ax_d.set_xticks([]); ax_d.set_yticks([])
        for s in ax_d.spines.values():
            s.set_visible(False)

    legends = []          # (title, {level: colour}, levels)
    for r, (sname, labels) in enumerate(strips):
        ax_s = fig.add_subplot(gs[r + n_top, 0])
        ordered = labels.loc[cells_o]
        if sname == "cell type":
            uniq = list(dict.fromkeys(ordered))
            cols = {g: ct_colors.get(g, plt.get_cmap("tab20")(i % 20))
                    for i, g in enumerate(uniq)}
            row = [list(matplotlib.colors.to_rgb(cols[g])) for g in ordered]
        else:
            # Batch strips use the SHARED per-level colours and the SHARED
            # sorted level order, so gch and hcg match exactly.
            gray = (0.75, 0.75, 0.75)
            shared = batch_colors.get(sname, {})
            uniq = list(batch_order.get(sname, list(dict.fromkeys(ordered))))
            for g in dict.fromkeys(ordered):        # any stray level -> append
                if g not in uniq:
                    uniq.append(g)
            cols = {g: shared.get(g, gray) for g in uniq}
            row = [list(shared.get(g, gray)) for g in ordered]
        ax_s.imshow([row], aspect="auto")
        ax_s.set_xticks([]); ax_s.set_yticks([])
        ax_s.set_ylabel(sname, rotation=0, ha="right", va="center", fontsize=8)
        legends.append((sname, cols, uniq))

    ax = fig.add_subplot(gs[n_strip + n_top, 0])
    im = ax.imshow(Z, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    layout_note = "clustered" if order is None else "borrowed order"
    ax.set_xlabel(f"cells (n={len(cells_o)}, {layout_note})")
    ax.set_ylabel(f"selected bins (n={len(bins_seq)}, {layout_note})")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, cax=fig.add_subplot(gs[n_strip + n_top, 1]),
                 label=value_label)

    # Legends to the RIGHT of the colour bar, stacked top-to-bottom.  The cell
    # type legend follows --group-order; a strip with many levels (plate/lib)
    # shows a count instead of an unreadable list.
    y = 0.95
    for sname, cols, uniq in legends:
        if sname == "cell type" and legend_order:
            uniq = [g for g in legend_order if g in cols] + \
                   [g for g in uniq if g not in legend_order]
        n = len(uniq)
        # Too many levels (e.g. lib_batch) to name legibly: colours are still on
        # the strip and consistent between heatmaps, but a legend of dozens of
        # near-duplicate swatches is unreadable, so just report the count.
        if n > legend_max:
            fig.text(1.0, y, f"{sname}: {n} levels (see strip; legend omitted)",
                     fontsize=8, transform=fig.transFigure, va="top")
            y -= 0.05
            continue
        # Wrap many-but-plottable levels into columns of <=12 so the legend
        # stays compact instead of running off the bottom.
        ncol = 1 if n <= 12 else int(np.ceil(n / 12))
        leg = fig.legend([mp.Patch(color=matplotlib.colors.to_rgb(cols[g]))
                          for g in uniq], uniq, loc="upper left",
                         bbox_to_anchor=(1.0, y), fontsize=7.5, title=sname,
                         title_fontsize=8, frameon=False, ncol=ncol,
                         columnspacing=1.0, handletextpad=0.5)
        fig.add_artist(leg)
        y -= 0.06 + 0.035 * int(np.ceil(n / ncol))
    fig.suptitle(title, fontsize=12)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return cells_o, bins_seq, col_lk


def main():
    ctv = _icc_mod()
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", action="append", required=True,
                    metavar="NAME=PATH", help="cells x bins concordance "
                    "parquet, e.g. gch=.../concordance_signed.parquet. "
                    "Repeatable.")
    ap.add_argument("--mean-matrix", action="append", default=None,
                    metavar="NAME=PATH",
                    help="cells x bins MEAN-signal parquet (accessibility for "
                         "gch, methylation for hcg) matching a --matrix NAME, "
                         "e.g. gch=.../accessibility_mean.parquet.  Used for the "
                         "two companion heatmaps (see --mean-heatmaps).  If "
                         "omitted, accessibility_mean.parquet beside the "
                         "concordance parquet is used automatically.  Repeatable.")
    ap.add_argument("--mean-heatmaps", choices=["auto", "on", "off"],
                    default="auto",
                    help="For each --matrix, also write two mean-signal "
                         "heatmaps: '<name>_mean_fixedorder' (the concordance "
                         "heatmap's EXACT cell+bin order, recoloured by the mean "
                         "value) and '<name>_mean_reclustered' (same cells+bins, "
                         "clustered on the mean value itself).  'auto' (default) "
                         "makes them when a mean matrix is found, else skips; "
                         "'on' errors if none is found; 'off' disables them.")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cell-type-column", default="cell_type")
    ap.add_argument("--batch-column", default="exp_batch,plate_batch,lib_batch",
                    help="Batch variables. A bin is dropped if its within-cell-"
                         "type variance is high in ANY of them, and EACH is "
                         "drawn as its own colour strip. Comma-separated, e.g. "
                         "exp_batch,plate_batch,lib_batch.")
    ap.add_argument("--color-column", default=None,
                    help="Samples-sheet column giving a colour per cell type.")
    ap.add_argument("--exclude-chrom", action="append", default=None)
    ap.add_argument("--n-bins", type=int, default=500,
                    help="Number of marker bins for the heatmap.")
    ap.add_argument("--cmap", action="append", default=None,
                    metavar="NAME=CMAP",
                    help="Override the diverging colormap per matrix, e.g. "
                         "gch=RdBu_r. Defaults: gch=RdBu_r, hcg=PuOr_r.")
    ap.add_argument("--max-batch-icc", type=float, default=0.1,
                    help="Drop a candidate bin whose within-cell-type between-"
                         "batch ICC exceeds this — it is sample-driven.")
    ap.add_argument("--min-cells-per-group", type=int, default=3)
    ap.add_argument("--min-groups", type=int, default=3)
    ap.add_argument("--max-legend-levels", type=int, default=40,
                    help="A batch strip with more levels than this (e.g. "
                         "lib_batch) is drawn but its legend is replaced by a "
                         "count; at/below it, the legend wraps into columns of "
                         "12.  Colours on the strip stay consistent regardless.")
    ap.add_argument("--matrix-norm", default="quantile",
                    choices=["quantile", "zscore", "none"],
                    help="Per-bin normalisation, matching the pipeline "
                         "(--matrix-norm quantile for signed).")
    ap.add_argument("--per-cell-norm", default="rank",
                    choices=["off", "rank", "center"],
                    help="Per-cell normalisation, matching the signed pipeline "
                         "(rank).")
    ap.add_argument("--post-norm-recenter", default="on", choices=["on", "off"])
    ap.add_argument("--min-cell-coverage", type=float, default=0.0,
                    help="Drop bins covered by fewer than this fraction of "
                         "cells (0 keeps all; the imputed concordance is dense).")
    ap.add_argument("--group-order", default=None,
                    help='Cell-type order for the column blocks and legend, '
                         'e.g. "IT-L2/3,IT-L4,CT-L6,...". Unlisted types follow '
                         'alphabetically.')
    ap.add_argument("--remove-batch", choices=["none", "within-celltype"],
                    default="none",
                    help="Center each (cell type, batch) group to its cell-type "
                         "mean per bin, for every batch variable — removes the "
                         "within-cell-type batch structure (main effect AND "
                         "interaction) while keeping the cell-type means. Use "
                         "this if the batch strips still show a pattern.")
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
    bcols = [c.strip() for c in args.batch_column.split(",") if c.strip()]
    batch_df_full = samples.set_index("cell_name")[bcols].astype(str)
    # One shared colour + legend order per batch level, across ALL matrices, so
    # a given batch label is the same colour and legend slot in every heatmap.
    batch_colors, batch_order = _batch_color_maps(batch_df_full, bcols)
    ct_colors = {}
    if args.color_column and args.color_column in samples.columns:
        for ct, c in zip(samples[args.cell_type_column].astype(str),
                         samples[args.color_column].astype(str)):
            if c and c.lower() not in ("nan", "none", ""):
                ct_colors.setdefault(ct, c)

    excl = parse_exclude_chroms(args.exclude_chrom)
    cmap_override = {}
    for c in (args.cmap or []):
        if '=' in c:
            k, v = c.split('=', 1); cmap_override[k.strip().lower()] = v.strip()
    mean_explicit = {}
    for c in (args.mean_matrix or []):
        if "=" not in c:
            sys.exit(f"--mean-matrix must be NAME=PATH, got {c!r}")
        k, v = c.split("=", 1); mean_explicit[k.strip()] = v.strip()
    out_dir = ensure_dir(Path(args.output_dir))
    summary = {}

    for mi, spec in enumerate(args.matrix):
        if "=" not in spec:
            sys.exit(f"--matrix must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not Path(path).exists():
            sys.exit(f"no such matrix: {path}")
        # Resolve the mean-signal matrix for the companion heatmaps: explicit
        # --mean-matrix NAME=PATH wins; else auto-derive the accessibility_mean
        # sibling written next to the concordance parquet.
        mean_path = mean_explicit.get(name)
        if mean_path and not Path(mean_path).exists():
            sys.exit(f"no such --mean-matrix for {name}: {mean_path}")
        if not mean_path:
            cand = Path(path).with_name("accessibility_mean.parquet")
            mean_path = str(cand) if cand.exists() else None
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

        # Normalise exactly as nome_hic_concordance / nome_hic_compare do before
        # the UMAP (quantile per-bin + per-cell rank + recenter for signed), so
        # the selected bins live in the same feature space as the embedding.
        M = _prep(M, name, str(out_dir), args)
        if M.shape[1] < 2:
            print(f"  {name}: {M.shape[1]} bins after normalisation — skipped",
                  file=sys.stderr)
            continue

        ct = pd.Series([ct_of[c] for c in M.index], index=M.index)
        ct_u = sorted(ct.unique())
        ct_id = np.searchsorted(np.array(ct_u), ct.to_numpy())
        batch_df = batch_df_full.reindex(M.index)

        # Optionally remove within-cell-type batch structure from the VALUES so
        # the batch strips (and the heatmap) look random.
        if args.remove_batch == "within-celltype":
            M = _remove_batch(M, ct, batch_df)

        batch_specs = []
        for col in bcols:
            u = sorted(batch_df[col].dropna().unique())
            batch_specs.append(
                (np.searchsorted(np.array(u), batch_df[col].to_numpy()), len(u)))

        # Select on ALL cells: more coverage means more bins pass than a train
        # split would (a hard filter on half the cells left only a couple dozen).
        sel, ct_icc, b_icc, n_clean = select_marker_bins(
            M.to_numpy(), ct_id, len(ct_u), batch_specs, ctv.icc_per_region, args)
        if sel.size == 0:
            print(f"  {name}: no usable bins — lower --min-groups?",
                  file=sys.stderr)
            continue
        sel_bins = M.columns[sel]
        rest_bins = M.columns[np.setdiff1d(np.arange(M.shape[1]), sel)]

        # Cells (columns) are clustered inside _heatmap; --group-order only sets
        # the legend order now.
        want = ([g.strip() for g in args.group_order.split(",") if g.strip()]
                if args.group_order else [])
        block_order = [g for g in want if g in set(ct_u)] + \
                      [g for g in ct_u if g not in want]
        cmap = (cmap_override.get(name.lower())
                or _CMAPS.get(name.lower(),
                              _CMAP_FALLBACK[mi % len(_CMAP_FALLBACK)]))
        fig_hm = out_dir / f"marker_heatmap_{name}.{args.plot_format}"
        clean_note = ("batch-invariant" if n_clean >= sel.size
                      else f"{n_clean}/{sel.size} batch-invariant")
        main_order = _heatmap(
            M, list(M.index), list(sel_bins), ct, batch_df, ct_colors,
            batch_colors, batch_order,
            f"{name.upper()}: {len(sel_bins)} cell-type-variable "
            f"({clean_note}) bins", fig_hm, cmap, legend_order=block_order,
            legend_max=args.max_legend_levels)

        # Companion mean-signal heatmaps (accessibility for gch, methylation for
        # hcg): (A) the concordance heatmap's EXACT layout recoloured by the mean
        # value (NO dendrogram -- the order is borrowed, not clustered here) ->
        # is the concordance structure present in the mean signal?  (B) the same
        # cells+bins clustered on the mean value itself -> what layout does the
        # mean signal make on its own?  (C) when --remove-batch is on, a second
        # reclustered view on the batch-corrected mean.  All show the RAW mean
        # LEVEL (no concordance normalisation) on the conventional NOMe-seq
        # sequential palette (white->teal for accessibility, blue->yellow for
        # methylation), not the z-scored diverging scale of the main heatmap.
        macc = None
        if args.mean_heatmaps != "off" and mean_path:
            macc = pd.read_parquet(mean_path)
            macc.index = macc.index.astype(str)
            miss_b = sum(b not in macc.columns for b in sel_bins)
            miss_c = sum(c not in macc.index for c in M.index)
            if miss_b == len(sel_bins) or miss_c == M.shape[0]:
                print(f"  {name}: mean matrix {mean_path} shares no "
                      f"{'bins' if miss_b == len(sel_bins) else 'cells'} with "
                      f"the concordance matrix — skipping companion heatmaps "
                      f"(wrong --mean-matrix?).", file=sys.stderr)
                macc = None
            elif miss_b or miss_c:
                print(f"  {name}: mean matrix is missing {miss_b}/{len(sel_bins)}"
                      f" bins and {miss_c}/{M.shape[0]} cells (shown blank).",
                      file=sys.stderr)
        if macc is not None:
            macc = macc.reindex(index=list(M.index), columns=list(sel_bins))
            lab = _MEAN_LABEL.get(name.lower(), "mean level")
            mcmap = _mean_cmap(name)
            fig_a = out_dir / f"marker_heatmap_{name}_mean_fixedorder.{args.plot_format}"
            _heatmap(macc, list(macc.index), list(sel_bins), ct, batch_df,
                     ct_colors, batch_colors, batch_order,
                     f"{name.upper()}: {lab}, concordance order "
                     f"({len(sel_bins)} marker bins)", fig_a, mcmap,
                     legend_order=block_order, legend_max=args.max_legend_levels,
                     order=main_order, value_label=lab, zscore=False)
            fig_b = out_dir / f"marker_heatmap_{name}_mean_reclustered.{args.plot_format}"
            _heatmap(macc, list(macc.index), list(sel_bins), ct, batch_df,
                     ct_colors, batch_colors, batch_order,
                     f"{name.upper()}: {lab}, own clustering "
                     f"({len(sel_bins)} marker bins)", fig_b, mcmap,
                     legend_order=block_order, legend_max=args.max_legend_levels,
                     value_label=lab, zscore=False)
            made = [fig_a.name, fig_b.name]
            # When the main figure removes batch (--remove-batch), also write a
            # reclustered companion on the mean signal AFTER the SAME within-
            # cell-type correction, so you can see how it re-clusters once the
            # batch structure is gone.
            if args.remove_batch != "none":
                macc_bc = _remove_batch(macc, ct, batch_df)
                fig_c = (out_dir / f"marker_heatmap_{name}_mean_"
                         f"reclustered_batchcorrected.{args.plot_format}")
                _heatmap(macc_bc, list(macc_bc.index), list(sel_bins), ct,
                         batch_df, ct_colors, batch_colors, batch_order,
                         f"{name.upper()}: {lab}, own clustering, batch-"
                         f"corrected ({len(sel_bins)} marker bins)", fig_c, mcmap,
                         legend_order=block_order,
                         legend_max=args.max_legend_levels,
                         value_label=f"{lab} (batch-corrected)", zscore=False)
                made.append(fig_c.name)
            print(f"  mean-signal companions -> {', '.join(made)}")
        elif args.mean_heatmaps == "on":
            sys.exit(f"--mean-heatmaps on but no mean matrix for {name!r}: pass "
                     f"--mean-matrix {name}=PATH or place "
                     f"accessibility_mean.parquet beside {path}.")

        # Out-of-selection CONTROL (number only; the UMAP figure is produced by
        # nome_hic_concordance / nome_hic_compare): do the bins we did NOT pick
        # still separate cell types?
        sil, perm_p = _nonselected_separation(
            M.loc[:, rest_bins], ct.to_numpy(), args.n_svd, args.skip_top_svd,
            args.seed, args.n_permutations)

        pd.DataFrame({"bin": M.columns, "celltype_icc": ct_icc,
                      "batch_icc": b_icc,
                      "selected": M.columns.isin(sel_bins)}
                     ).to_csv(out_dir / f"bin_scores_{name}.tsv",
                              sep="\t", index=False)
        summary[name] = {
            "n_cells": int(M.shape[0]), "n_bins": int(M.shape[1]),
            "n_selected": int(sel.size),
            "n_selected_batch_clean": int(n_clean),
            "median_celltype_icc_selected": _nanmedian(ct_icc[sel]),
            "median_batch_icc_selected": _nanmedian(b_icc[sel]),
            "nonselected_silhouette": (None if not np.isfinite(sil)
                                       else float(sil)),
            "nonselected_perm_p": (None if not np.isfinite(perm_p)
                                   else float(perm_p))}
        print(f"\n=== {name}: {M.shape[0]} cells x {M.shape[1]} bins ===")
        print(f"  selected {sel.size} bins ({n_clean} batch-invariant): "
              f"median cell-type ICC {_nanmedian(ct_icc[sel]):.3f}, "
              f"median batch ICC {_nanmedian(b_icc[sel]):.3f}")
        if n_clean < sel.size:
            print(f"  NOTE: only {n_clean} bins passed --max-batch-icc "
                  f"{args.max_batch_icc}; the rest were filled by cell-type ICC "
                  f"(raise the cut to admit more, or lower --n-bins).")
        print(f"  heatmap on all {M.shape[0]} cells -> {fig_hm.name}")
        if np.isfinite(sil):
            verdict = "SEPARATES" if perm_p < 0.05 else "does NOT separate"
            print(f"  out-of-selection control: NON-selected bins {verdict} "
                  f"cell types (silhouette={sil:.3f}, perm p={perm_p:.3f}) — "
                  f"see the nome_hic_concordance/compare UMAP for the figure")

    if not summary:
        sys.exit("no matrix produced output.")
    with open(out_dir / "marker_bins_summary.json", "w") as fh:
        json.dump({"matrices": summary, "n_bins": args.n_bins,
                   "max_batch_icc": args.max_batch_icc,
                   "note": "bins selected on ALL cells by cell-type ICC, "
                           "preferring batch-invariant bins; heatmap on all "
                           "cells; nonselected_* validates the claim on the "
                           "bins NOT selected (the UMAP figure is produced by "
                           "nome_hic_concordance / nome_hic_compare)"},
                  fh, indent=2)
    print(f"\nRead it: the heatmap shows cell-type-variable, batch-invariant "
          "concordance across\n  all cells.  The nonselected_* silhouette + "
          "permutation p is the honesty\n  check: if the bins NOT selected also "
          "separate cell types (p<0.05), the heatmap\n  is not a selection "
          "artefact.  If they do NOT, the claim fails.")


if __name__ == "__main__":
    main()
