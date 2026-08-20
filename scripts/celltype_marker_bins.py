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

Three further bin sets are written as BED alongside the marker set, each ranked
by its own statistic but filtered for batch the same way:

  STABLE          the invariant counterpart -- bins whose cell-type means sit
                  closest together (--stable-bins).  Ranked by the SD of those
                  means, NOT by low cell-type ICC, which would return the
                  noisiest bins rather than the flattest.  Stability says
                  NOTHING about the level of concordance: the ranking runs on
                  the per-bin quantile-normalised matrix, where every bin shares
                  one distribution across cells, so a bin flat at low
                  concordance and one flat at high concordance are equally
                  invariant and both are taken.  bin_mean_raw is reported on
                  every output so that can be checked, and --stable-level
                  high/low narrows the pool first when what is wanted is
                  "constitutively coordinated" rather than "unchanging".

  CO-OPEN /       for a DIRECTIONAL matrix (D = C(2F-1)), the most co-open
  CO-CLOSED       and the most co-closed bins (--sign-split), read off the RAW
                  values since the normalisation used for selection destroys
                  the sign that carries the meaning.  Top --n-sign-bins by
                  rank in each direction, not everything past a threshold:
                  that keeps them the same size as the marker and invariant
                  sets for a genomic-enrichment comparison, and a reference
                  bias common to every bin slides a threshold but not a rank.
                  --sign-reference decides what "co-open" is measured against.

  CONTRAST        bins separating a NAMED set of cell types from all the others
                  (--contrast LABEL=g1,g2), e.g. two meningioma tumours against
                  a panel of GM12878, IMR90 and monocyte.  Welch t with BH-FDR
                  and Cohen's d, plus the association between the contrast and
                  each batch variable, since a condition processed on its own
                  makes "separates the groups" and "separates the batches" the
                  same statement.  --cnv-seg adds the copy-number check, which
                  matters most for exactly this comparison: copy number moves
                  coverage, coverage moves the DM shrinkage and what RWR
                  imputes, so a tumour-vs-line contrast can separate on a
                  deletion rather than on coordination.

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


def _group_stats(X, group_ids, n_groups, min_cells):
    """(means, usable) per group per bin; a group with too few finite cells is
    masked out rather than contributing a mean of one cell."""
    X = np.asarray(X, float)
    means = np.full((n_groups, X.shape[1]), np.nan)
    usable = np.zeros((n_groups, X.shape[1]), bool)
    for g in range(n_groups):
        rows = X[group_ids == g]
        if not rows.shape[0]:
            continue
        n = np.isfinite(rows).sum(axis=0)
        usable[g] = n >= min_cells
        with np.errstate(invalid="ignore"):
            means[g] = np.where(n > 0, np.nanmean(rows, axis=0), np.nan)
    return np.where(usable, means, np.nan), usable


def celltype_spread(X, cell_type_ids, n_ct, args):
    """SD of the cell-type MEANS per bin — small means invariant.

    Ranking stability by LOW cell-type ICC is only safe once the bins are on a
    common scale.  On raw values it returns the NOISIEST bins rather than the
    flattest, because ICC puts sampling noise in its denominator and a bin that
    is pure noise has a between-group share near zero.  The spread of the group
    means penalises noise directly — with n cells per group it is
    ~sqrt(var_between + var_within / n), small only when the bin is both flat
    across cell types AND measured reliably.

    Under the default --matrix-norm quantile the point is largely moot: per-bin
    QN gives every bin the same total variance, so low ICC and low spread rank
    almost identically.  The spread is used anyway because it survives the
    settings where that is not true (--matrix-norm zscore, or an unnormalised
    caller), and because it reads in the same units as the heatmap.
    """
    means, usable = _group_stats(X, cell_type_ids, n_ct,
                                 args.min_cells_per_group)
    covered = usable.sum(axis=0)
    with np.errstate(invalid="ignore"):
        spread = np.nanstd(means, axis=0, ddof=1)
    return np.where(covered >= max(args.min_groups, 2), spread, np.nan)


def select_stable_bins(X, cell_type_ids, n_ct, b_icc, args, n_want,
                       eligible=None):
    """The invariant counterpart of :func:`select_marker_bins`.

    The batch filter applies exactly as it does to the marker set: a bin that
    is flat across cell types but driven by batch is not invariant, it is
    sample.  Same clean-first, fill-by-rank policy, so the two sets are
    selected by the same rules and can be compared.

    ``eligible`` restricts the POOL before ranking (see --stable-level).  It has
    to be a restriction rather than part of the ranking: the ranking runs on the
    normalised matrix, where per-bin level does not exist, so "flat AND high"
    cannot be expressed as one score.  Stability alone says nothing about the
    level -- a bin flat at low concordance and one flat at high concordance are
    equally invariant, and both are selected unless this narrows the pool.
    """
    spread = celltype_spread(X, cell_type_ids, n_ct, args)
    if eligible is not None:
        spread = np.where(eligible, spread, np.nan)
    usable = np.flatnonzero(np.isfinite(spread))
    if usable.size == 0:
        return np.array([], int), spread, 0
    ranked = usable[np.argsort(spread[usable])]          # flattest first
    is_clean = ~(np.isfinite(b_icc) & (b_icc > args.max_batch_icc))
    clean = ranked[is_clean[ranked]]
    if clean.size >= n_want:
        return clean[:n_want], spread, int(n_want)
    dirty = ranked[~is_clean[ranked]]
    sel = np.concatenate([clean, dirty[: n_want - clean.size]])
    return sel, spread, int(clean.size)


def expected_directional(marginal, coordination):
    """D under no direction beyond the bin's own level: C·(2·E[F] − 1).

    F is the realized share of concordant pairs that are co-open.  Under
    independence at marginal p̄ the two co-states have probabilities p̄² and
    (1−p̄)², so

        E[F] = p̄² / (p̄² + (1−p̄)²)

    and subtracting C·(2E[F]−1) from D leaves 2C·(F − E[F]) — the EXCESS
    co-openness over what the bin's mean level already implies.  That is the
    same coordination-versus-level control the rest of the analysis rests on,
    applied to direction instead of magnitude, and unlike a genome-wide median
    it gives each bin its own reference.

    One approximation: the stored marginal is the MEAN of the two ends'
    marginals, and P(both open) = p̄₁p̄₂ ≤ p̄², so using p̄² slightly
    over-predicts co-openness when the two ends differ.  Correcting it would
    need the per-end marginals, which are not kept.  The denominator is
    p̄² + (1−p̄)² ≥ 0.5, so it never divides by zero.
    """
    p = np.clip(np.asarray(marginal, dtype=float), 0.0, 1.0)
    with np.errstate(invalid="ignore"):
        ef = p ** 2 / (p ** 2 + (1.0 - p) ** 2)
    return np.asarray(coordination, dtype=float) * (2.0 * ef - 1.0)


def sign_reference(grand, mode: str) -> float:
    """The value a bin's D is called co-open / co-closed against.

    Zero is the neutral point for COORDINATION but NOT for DIRECTION, and that
    distinction decides what this analysis can see.  For a pair with marginals
    p1, p2,

        P(both open) - P(both closed) = p1*p2 - (1-p1)(1-p2) = p1 + p2 - 1

    so sign(2F-1) is the sign of (mean level - 0.5), and D = C*(2F-1) inherits
    it.  Mean GCH accessibility over 250 kb bins sits well below 0.5 and mean
    HCG methylation well above it, so against zero essentially every bin comes
    back co-closed for GCH and co-methylated for HCG.  That is the genome's
    base rate restated, not a property of any region.

    'median' therefore calls a bin against the genome-wide median of the
    per-bin means instead: "more co-open than the typical bin", which is the
    question that has an answer that varies along the genome.  'zero' keeps
    the literal reading for anyone who wants it.
    """
    if mode == "zero":
        return 0.0
    v = np.asarray(grand, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else 0.0


def sign_summary(raw, cell_type_ids, n_ct, args, expected=None):
    """Per-bin co-open / co-closed evidence for a DIRECTIONAL matrix.

    Must be computed on the RAW values.  D = C·(2F−1) carries its meaning in
    its sign — positive is coordination between OPEN states, negative between
    CLOSED ones — and the normalisation used for selection destroys exactly
    that: per-bin quantile normalisation re-maps every bin onto a common
    distribution and the per-cell rank re-centres each cell, after which "above
    zero" no longer means co-open.

    Returns the per-cell-type mean range and the fraction of cells on the
    positive side, so a bin is called only when EVERY cell type agrees and the
    cells within them mostly do too.  Both are measured against
    :func:`sign_reference`, not necessarily against zero.
    """
    X = np.asarray(raw, float)
    with np.errstate(invalid="ignore"):
        grand = np.nanmean(X, axis=0)
        if args.sign_reference == "expected":
            if expected is None:
                sys.exit("--sign-reference expected needs the marginal and "
                         "coordination components; pass --marginal-matrix / "
                         "--coordination-matrix, or keep them beside the "
                         "directional parquet.")
            # A per-CELL, per-bin reference, so it is subtracted before any
            # aggregation rather than after.
            ref_mat = np.asarray(expected, float)
            ref_report = np.nanmean(ref_mat, axis=0)
        else:
            ref_mat = sign_reference(grand, args.sign_reference)
            ref_report = np.full(X.shape[1], float(ref_mat))
        Z = X - ref_mat
    means, usable = _group_stats(Z, cell_type_ids, n_ct,
                                 args.min_cells_per_group)
    covered = usable.sum(axis=0)
    with np.errstate(invalid="ignore"):
        lo = np.nanmin(means, axis=0)
        hi = np.nanmax(means, axis=0)
        fin = np.isfinite(Z).sum(axis=0)
        pos = np.nansum(Z > 0, axis=0)
        frac_pos = np.where(fin > 0, pos / np.maximum(fin, 1), np.nan)
    enough = covered >= max(args.min_groups, 2)
    lo = np.where(enough, lo, np.nan)
    hi = np.where(enough, hi, np.nan)
    c = args.sign_min_consistency
    co_open = np.isfinite(lo) & (lo > 0) & (frac_pos >= c)
    co_closed = np.isfinite(hi) & (hi < 0) & ((1.0 - frac_pos) >= c)
    return {"d_mean": grand, "d_reference": ref_report,
            "d_group_min": lo, "d_group_max": hi,
            "frac_cells_positive": frac_pos, "n_groups_used": covered,
            "always_co_open": co_open, "always_co_closed": co_closed}


def select_sign_bins(sg, key, rank_by, ascending, n_want):
    """The n_want most co-open (or co-closed) bins, agreeing ones first.

    A fixed size rather than a threshold, for two reasons.  It makes this set
    directly comparable to the marker and invariant sets, which are already
    top-N, so a genomic-enrichment comparison between them is not really a
    comparison of set sizes.  And it is robust to the reference: any bias
    common to every bin — and the expected-D reference carries one, since the
    stored marginal is the mean of the two ends' — slides the whole
    distribution across a threshold while leaving the RANKING untouched.  A
    threshold turns that into "almost every bin is co-open"; a rank does not
    notice it.

    Ranked by the per-cell-type extreme (min for co-open, max for co-closed):
    "how co-open is this bin even in its LEAST co-open cell type".  Bins where
    every cell type agrees come first; if fewer than n_want do, the rest are
    filled by the same ranking, and the agreeing count is reported so the
    trade-off is visible — the same clean-first-then-fill policy as
    :func:`select_marker_bins`.
    """
    v = np.asarray(sg[rank_by], float)
    usable = np.flatnonzero(np.isfinite(v))
    if usable.size == 0:
        return np.array([], int), 0
    order = usable[np.argsort(v[usable], kind="stable")]
    if not ascending:
        order = order[::-1]
    agree = np.asarray(sg[key], bool)[order]
    clean = order[agree]
    if clean.size >= n_want:
        return clean[:n_want], int(n_want)
    rest = order[~agree]
    return (np.concatenate([clean, rest[:max(n_want - clean.size, 0)]]),
            int(clean.size))


def mwu_min_pvalue(n1, n2):
    """Smallest two-sided p a rank test can return for these arm sizes.

    A rank test has only C(n1+n2, n1) distinct orderings, so its p-value has a
    FLOOR.  At 3 against 3 that floor is 0.1 and no bin can ever be called,
    however large the difference; at 8 against 8 it is 1.6e-4, which survives
    BH across a handful of bins but not across thousands.  Worth checking
    before a run rather than reading an all-empty table afterwards.
    """
    from math import comb
    n1, n2 = int(n1), int(n2)
    if n1 < 1 or n2 < 1:
        return float("nan")
    return min(1.0, 2.0 / comb(n1 + n2, n1))


def contrast_stats(X, in_mask, min_cells, out_mask=None, test="mwu"):
    """Per-bin two-sample test of one group set against a reference arm.

    ``test='mwu'`` is a two-sided Mann-Whitney U on the cells: it makes no
    normality assumption, which single-cell values do not honour, and one
    outlying cell cannot carry a bin the way it can carry a t.  ``'welch'`` is
    the Welch t -- unequal variances, since the two arms are a handful of
    samples of one condition against a reference.

    The reported difference is the difference of MEANS either way.  That is
    the volcano's x-axis and the quantity in D units; the rank test supplies
    the p-value, not the effect size.  Returns NaN for bins either arm cannot
    cover, which keeps them out of the multiple-testing correction rather than
    in it at p = 1.

    ``out_mask`` defaults to everything outside ``in_mask``.  Passing it
    explicitly names the reference instead, and cells in NEITHER arm are
    dropped rather than folded into the reference -- a tumour-vs-monocyte
    contrast is not a tumour-vs-everything contrast with a different label, and
    silently widening the reference would change both the effect size and which
    bins clear the threshold.
    """
    from scipy.stats import t as _t

    if test not in ("mwu", "welch"):
        raise ValueError(f"test must be 'mwu' or 'welch', got {test!r}")
    X = np.asarray(X, float)
    if out_mask is None:
        out_mask = ~np.asarray(in_mask, bool)
    out = {}
    for tag, m in (("in", in_mask), ("out", out_mask)):
        rows = X[m]
        n = np.isfinite(rows).sum(axis=0)
        with np.errstate(invalid="ignore"):
            out[f"n_{tag}"] = n
            out[f"mean_{tag}"] = np.where(n > 0, np.nanmean(rows, axis=0),
                                          np.nan)
            out[f"var_{tag}"] = np.where(n > 1, np.nanvar(rows, axis=0, ddof=1),
                                         np.nan)
    n1, n2 = out["n_in"], out["n_out"]
    v1, v2 = out["var_in"], out["var_out"]
    diff = out["mean_in"] - out["mean_out"]
    with np.errstate(invalid="ignore", divide="ignore"):
        s1, s2 = v1 / np.maximum(n1, 1), v2 / np.maximum(n2, 1)
        se = np.sqrt(s1 + s2)
        tstat = diff / se
        df = (s1 + s2) ** 2 / (s1 ** 2 / np.maximum(n1 - 1, 1)
                               + s2 ** 2 / np.maximum(n2 - 1, 1))
        p = 2.0 * _t.sf(np.abs(tstat), df)
        # Cohen's d on the pooled SD, so the effect size is readable next to
        # the p-value that a large n would otherwise make look decisive.
        sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2)
                     / np.maximum(n1 + n2 - 2, 1))
        d = diff / sp
    if test == "mwu":
        from scipy.stats import mannwhitneyu
        with np.errstate(invalid="ignore"):
            try:
                p = mannwhitneyu(X[in_mask], X[out_mask], axis=0,
                                 alternative="two-sided",
                                 nan_policy="omit").pvalue
            except ValueError:
                # Raised when a bin is constant in BOTH arms together; there
                # is no ordering to test, so it is unusable rather than p = 1.
                p = np.full(X.shape[1], np.nan)
        p = np.asarray(p, float)
    bad = (n1 < min_cells) | (n2 < min_cells) | ~np.isfinite(se) | (se <= 0)
    for k, v in (("diff", diff), ("t", tstat), ("p", p), ("d", d)):
        out[k] = np.where(bad, np.nan, v)
    return out


def volcano_figure(diff, fdr, sig, path, title, fdr_cut, min_diff=0.0):
    """Mean difference against -log10(FDR), significant bins coloured.

    The whole point of showing it rather than a ranked list: a volcano makes
    the two ways a bin can be uninteresting visible at once -- significant but
    tiny, and large but noisy -- which a top-N table hides by construction,
    since it reports its N either way.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = np.isfinite(diff) & np.isfinite(fdr)
    x = np.asarray(diff, float)[ok]
    # FDR of exactly 0 would be an infinite y; floor at the smallest non-zero.
    q = np.asarray(fdr, float)[ok]
    nz = q[q > 0]
    q = np.maximum(q, nz.min() if nz.size else 1e-300)
    y = -np.log10(q)
    m = np.asarray(sig, bool)[ok]
    up, down = m & (x > 0), m & (x < 0)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(x[~m], y[~m], s=5, c="#C6CBD1", lw=0, alpha=0.55,
               rasterized=True, label=f"n.s. ({int((~m).sum()):,})")
    ax.scatter(x[down], y[down], s=7, c="#2C6FAF", lw=0, rasterized=True,
               label=f"down ({int(down.sum()):,})")
    ax.scatter(x[up], y[up], s=7, c="#D1495B", lw=0, rasterized=True,
               label=f"up ({int(up.sum()):,})")
    if np.isfinite(fdr_cut) and 0 < fdr_cut < 1:
        ax.axhline(-np.log10(fdr_cut), color="#6B7280", lw=0.9, ls="--")
    if min_diff > 0:
        for v in (-min_diff, min_diff):
            ax.axvline(v, color="#6B7280", lw=0.9, ls=":")
    ax.axvline(0.0, color="#9AA0A6", lw=0.8, zorder=0)
    ax.set_xlabel("mean difference (contrast - reference)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return int(up.sum()), int(down.sum())


def joint_contrast_table(per_context, label):
    """Align two contexts' contrast results on their shared bins.

    Returns a frame indexed by bin key with diff/fdr/sig per context and a
    ``category`` naming which modality the bin is aberrant in.  Bins missing
    from either context are dropped: a bin GCH never measured is not evidence
    that the aberration is HCG-specific, and letting it in would inflate the
    modality-specific classes with what is really missing data.
    """
    names = list(per_context)
    if len(names) != 2:
        raise ValueError(f"need exactly two contexts, got {names}")
    a, b = names
    fa, fb = per_context[a], per_context[b]
    keys = fa.index.intersection(fb.index)
    out = pd.DataFrame(index=keys)
    for nm, fr in ((a, fa), (b, fb)):
        out[f"{nm}_diff"] = fr.loc[keys, "diff"].to_numpy()
        out[f"{nm}_fdr"] = fr.loc[keys, "fdr"].to_numpy()
        out[f"{nm}_sig"] = fr.loc[keys, "sig"].to_numpy()
    sa, sb = out[f"{a}_sig"].to_numpy(), out[f"{b}_sig"].to_numpy()
    out["category"] = np.where(
        sa & sb, "both",
        np.where(sa, f"{a}_only",
                 np.where(sb, f"{b}_only", "neither")))
    return out, a, b


def joint_concordance(tab, a, b):
    """Counts and sign agreement over the bins significant in both.

    Sign agreement is only meaningful there.  Asking whether a bin moves the
    same way in both modalities when one of them did not move at all is a
    coin flip dressed as a result.
    """
    both = tab[tab["category"] == "both"]
    da, db = both[f"{a}_diff"].to_numpy(), both[f"{b}_diff"].to_numpy()
    same = int(np.sum(np.sign(da) == np.sign(db)))
    n = int(len(both))
    stats = {"n_shared_bins": int(len(tab)),
             "n_both": n,
             f"n_{a}_only": int((tab["category"] == f"{a}_only").sum()),
             f"n_{b}_only": int((tab["category"] == f"{b}_only").sum()),
             "n_both_same_direction": same,
             "frac_both_same_direction": (same / n) if n else float("nan")}
    if n > 2:
        from scipy.stats import binomtest, pearsonr, spearmanr
        stats["pearson_r_diff"] = float(pearsonr(da, db)[0])
        stats["spearman_rho_diff"] = float(spearmanr(da, db)[0])
        # Against the null that direction is independent between modalities.
        stats["p_same_direction"] = float(
            binomtest(same, n, 0.5, alternative="two-sided").pvalue)
    return stats


def joint_figure(tab, a, b, stats, mats, cell_types, path, label):
    """One PDF: the two mean differences against each other, then a clustered
    heatmap per modality over the union of significant bins.

    The heatmaps share one bin order -- clustered once, on the two differences
    together -- so a row is the same bin in both panels.  Clustering each
    panel separately would put a different bin on each row and make the two
    impossible to read against each other, which is the only question here.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.cluster.hierarchy import leaves_list, linkage

    colour = {"both": "#7B3294", f"{a}_only": "#D1495B",
              f"{b}_only": "#2C6FAF", "neither": "#DADDE1"}
    with PdfPages(path) as pdf:
        # (1) scatter of the two effects
        fig, ax = plt.subplots(figsize=(5.4, 5.2))
        for cat in ("neither", f"{a}_only", f"{b}_only", "both"):
            sub = tab[tab["category"] == cat]
            if not len(sub):
                continue
            ax.scatter(sub[f"{a}_diff"], sub[f"{b}_diff"], s=9 if cat != "neither" else 4,
                       c=colour[cat], lw=0, alpha=0.75 if cat != "neither" else 0.4,
                       rasterized=True, label=f"{cat} ({len(sub):,})")
        ax.axhline(0, color="#9AA0A6", lw=0.8, zorder=0)
        ax.axvline(0, color="#9AA0A6", lw=0.8, zorder=0)
        ax.set_xlabel(f"{a} mean difference (contrast - reference)")
        ax.set_ylabel(f"{b} mean difference (contrast - reference)")
        sub = stats.get("frac_both_same_direction")
        note = (f"{stats['n_both']:,} significant in both, "
                f"{sub:.0%} same direction" if stats["n_both"] else
                "no bin significant in both")
        ax.set_title(f"{label}: {note}", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="best")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # (2) clustered heatmaps over the union, one bin order for both
        sel = tab[tab["category"] != "neither"]
        if not len(sel):
            return
        keys = list(sel.index)
        eff = np.column_stack([sel[f"{a}_diff"].to_numpy(),
                               sel[f"{b}_diff"].to_numpy()])
        eff = np.nan_to_num(eff)
        order = (leaves_list(linkage(eff, method="average"))
                 if len(keys) > 2 else np.arange(len(keys)))
        keys = [keys[i] for i in order]

        fig, axes = plt.subplots(1, len(mats), figsize=(6.0 * len(mats), 7.2),
                                 squeeze=False)
        for ax, (nm, mat) in zip(axes[0], mats.items()):
            cols = [k for k in keys if k in mat.columns]
            sub = mat.loc[:, cols]
            ct = pd.Series(cell_types).reindex(sub.index)
            cell_order = np.argsort(ct.to_numpy().astype(str), kind="stable")
            v = sub.to_numpy()[cell_order].T
            lim = np.nanpercentile(np.abs(v), 98) if np.isfinite(v).any() else 1
            im = ax.imshow(v, aspect="auto", cmap="RdBu_r",
                           vmin=-lim, vmax=lim, interpolation="nearest")
            ax.set_title(f"{nm}  ({len(cols):,} bins)", fontsize=10)
            ax.set_xlabel("cells (grouped by cell type)")
            ax.set_ylabel("bins (shared order, clustered on both effects)")
            ax.set_yticks([])
            ax.set_xticks([])
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.suptitle(f"{label}: union of significant bins", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def read_segments(path, sample_col=None):
    """A copy-number segmentation → DataFrame[sample, chrom, start, end, value].

    Accepts a headed file with those names, or the classic 6-column .seg
    (ID, chrom, start, end, num_mark, seg.mean).  Values are assumed to be
    log2 ratios, so 0 is copy-neutral; the analysis below only ever uses
    DIFFERENCES between groups, so any centred scale works.
    """
    df = pd.read_csv(path, sep="\t", dtype={0: str}, comment="#")
    cols = {c.lower().strip(): c for c in df.columns}
    want = ("sample", "chrom", "start", "end", "value")
    if all(w in cols for w in want):
        out = df[[cols[w] for w in want]].copy()
    elif df.shape[1] >= 6:
        out = df.iloc[:, [0, 1, 2, 3, 5]].copy()          # .seg, skipping num_mark
    else:
        sys.exit(f"{path}: expected sample/chrom/start/end/value columns or a "
                 f"6-column .seg (ID chrom start end num_mark seg.mean); got "
                 f"{list(df.columns)}.")
    out.columns = list(want)
    out["sample"] = out["sample"].astype(str)
    out["chrom"] = out["chrom"].astype(str).map(_canon_chrom)
    for c in ("start", "end"):
        out[c] = out[c].astype(float).astype(np.int64)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"])


def segment_bin_means(bins, seg):
    """Per-bin, overlap-weighted mean segment value → (n_samples, n_bins).

    Overlap-weighted rather than midpoint-assigned: a 250 kb bin routinely
    spans a segment boundary, and taking whichever segment covers the midpoint
    would give the bin a copy number it only half has.
    """
    samples = sorted(seg["sample"].unique())
    chrom = pd.Index(bins).str.split(":").str[0].map(_canon_chrom).to_numpy()
    start = pd.Index(bins).str.split(":").str[1].astype(np.int64).to_numpy()
    width = infer_bin_size_from(bins)
    end = start + width
    out = np.full((len(samples), len(bins)), np.nan)
    for si, sample in enumerate(samples):
        sub = seg[seg["sample"] == sample]
        for c, g in sub.groupby("chrom"):
            hit = np.flatnonzero(chrom == c)
            if hit.size == 0:
                continue
            ss = g["start"].to_numpy()
            ee = g["end"].to_numpy()
            vv = g["value"].to_numpy()
            order = np.argsort(ss)
            ss, ee, vv = ss[order], ee[order], vv[order]
            for i in hit:
                lo = np.searchsorted(ee, start[i], "right")
                hi = np.searchsorted(ss, end[i], "left")
                if hi <= lo:
                    continue
                ov = (np.minimum(ee[lo:hi], end[i])
                      - np.maximum(ss[lo:hi], start[i])).astype(float)
                keep = ov > 0
                if keep.any():
                    out[si, i] = float(np.average(vv[lo:hi][keep],
                                                  weights=ov[keep]))
    return np.array(samples), out


def infer_bin_size_from(bins) -> int:
    """Bin width from consecutive starts on one chromosome (GCD of the gaps)."""
    from nome_hic_concordance.blacklist import infer_bin_size
    w = infer_bin_size(pd.Index(list(bins)))
    if not w:
        sys.exit("cannot infer the bin width from the column labels, so a "
                 "segment cannot be mapped onto a bin.")
    return int(w)


def cnv_group_delta(bins, seg, cell_to_sample, in_cells, out_cells):
    """Per-bin (mean CN in the contrast group) − (mean CN outside it).

    The copy-number analogue of the batch check.  In a tumour-vs-line contrast
    the two arms differ in copy number by construction, and copy number moves
    coverage, which moves the DM shrinkage and what RWR imputes -- so a bin can
    separate the groups for reasons that have nothing to do with coordination.
    A bin over a recurrent deletion is the clearest case.
    """
    samples, mat = segment_bin_means(bins, seg)
    idx = {s: i for i, s in enumerate(samples)}
    def rows(cells):
        got = sorted({cell_to_sample[c] for c in cells
                      if cell_to_sample.get(c) in idx})
        return [idx[s] for s in got]
    ri, ro = rows(in_cells), rows(out_cells)
    if not ri or not ro:
        return None, samples, (len(ri), len(ro))
    with np.errstate(invalid="ignore"):
        delta = np.nanmean(mat[ri], axis=0) - np.nanmean(mat[ro], axis=0)
    return delta, samples, (len(ri), len(ro))


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Association between two categorical vectors, 0 (none) to 1 (identical
    partition).  Used to warn when a requested contrast is confounded with
    batch, where 'the bins that separate these groups' and 'the bins that
    separate these batches' are the same bins."""
    from scipy.stats import chi2_contingency

    tab = pd.crosstab(a, b)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(tab.to_numpy())[0]
    n = int(tab.to_numpy().sum())
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


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


def _bh_fdr(p):
    """Benjamini-Hochberg FDR (q-values) over the FINITE p-values; NaN in ->
    NaN out, so unusable bins do not enter the multiple-testing correction."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if idx.size == 0:
        return out
    pv = p[idx]
    order = np.argsort(pv)
    m = pv.size
    q = pv[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]          # enforce monotonicity
    out[idx[order]] = np.clip(q, 0.0, 1.0)
    return out


def _canon_chrom(c):
    """Contig name with a leading 'chr'/'Chr' stripped, so chr1 == 1."""
    s = str(c)
    return s[3:] if s[:3].lower() == "chr" else s


def _read_chrom_sizes(path):
    """{contig (without 'chr'): length} from a chrom.sizes / .fai (first 2
    columns)."""
    sizes = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    sizes[_canon_chrom(parts[0])] = int(parts[1])
                except ValueError:
                    continue
    return sizes


def _bins_to_bed(bins, bin_size, chrom_sizes=None):
    """(chrom, start, end) per bin label 'chrom:start' (0-based, half-open).

    ``bin_size`` sets the end; if it could not be inferred, end = start + 1.
    When ``chrom_sizes`` is given (keyed by the 'chr'-stripped contig name), the
    last bin of each chromosome is clamped so its end never runs past the
    chromosome length -- otherwise a fixed-width grid overshoots the final bin
    and breaks downstream tools (bedtools, enrichment)."""
    width = int(bin_size) if bin_size else 1
    sizes = chrom_sizes or {}
    rows = []
    for b in bins:
        s = str(b)
        if ":" in s:
            chrom, pos = s.rsplit(":", 1)
            try:
                start = int(pos)
            except ValueError:
                chrom, start = s, 0
        else:
            chrom, start = s, 0
        end = start + width
        size = sizes.get(_canon_chrom(chrom))
        if size is not None:
            end = max(min(end, int(size)), start + 1)   # clamp; keep end > start
        rows.append((chrom, start, end))
    return rows


# Distinct diverging colour schemes so GCH-GCH and HCG-HCG never look alike.
_CMAPS = {"gch": "RdBu_r", "hcg": "PuOr_r"}
_CMAP_FALLBACK = ["RdBu_r", "PuOr_r", "BrBG_r", "PiYG_r"]

# Compartment (A/B) heatmaps: diverging + centred at 0, but a DIFFERENT hue
# family per context, so they are distinct from the concordance maps above AND
# from each other (gch purple<->green, hcg brown<->teal).
_COMPARTMENT_CMAPS = {"gch": "PRGn_r", "hcg": "BrBG_r"}
_COMPARTMENT_FALLBACK = ["PRGn_r", "BrBG_r", "PiYG_r", "RdGy_r"]

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
             zscore=True, symmetric=False):
    """Marker heatmap: bins x cells.  Cells (columns) are hierarchically
    clustered and the COLUMN dendrogram is drawn on top of the cell-type strip;
    no cell names are shown.  A cell-type strip and one strip per batch variable
    (exp/plate/lib) sit below it, so it is visible that the cell clustering
    tracks cell type and not any batch.

    ``zscore`` (default) z-scores each bin across cells for display (diverging
    cmap, +/-2.5).  Set False to show the RAW value with a robust 2-98
    percentile range: sequential (mean accessibility / methylation), or, with
    ``symmetric``, a signed range centred at 0 (compartment A/B).

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
        # Raw value: keep the LEVEL, pick a robust display range for contrast.
        # `symmetric` centres a signed/diverging signal (compartment A/B) at 0;
        # otherwise a sequential 2-98 percentile range (mean accessibility).
        finite = Z[np.isfinite(Z)]
        if not finite.size:
            vmin, vmax = (-1.0, 1.0) if symmetric else (0.0, 1.0)
        elif symmetric:
            m = float(np.percentile(np.abs(finite), 98)) or float(
                np.abs(finite).max()) or 1.0
            vmin, vmax = -m, m
        else:
            vmin, vmax = (float(v) for v in np.percentile(finite, [2, 98]))
            if vmax <= vmin:
                vmin, vmax = float(finite.min()), float(finite.max())
            if vmax <= vmin:
                vmax = vmin + 1.0
        # missing -> 0 (neutral) for a centred diverging map, else the low end.
        Z = np.where(np.isfinite(Z), Z, 0.0 if symmetric else vmin)
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


def _find_compartment(conc_path):
    """Best-effort auto-derive: compartment_scores.parquet under a
    ``compartments/`` directory at the run root (a sibling of the per-context
    dirs that hold the concordance parquet).  Returns a path or None."""
    for anc in Path(conc_path).resolve().parents:
        cand = anc / "compartments" / "compartment_scores.parquet"
        if cand.exists():
            return str(cand)
    return None


def _align_signal(sig, cells, sel_bins, name, infix, grid_align):
    """Align a companion signal onto the (cells x sel_bins) grid.

    ``grid_align`` maps a COARSER bin grid onto sel_bins via align_to_grid (each
    marker bin inherits the covering coarse bin -- e.g. compartments); otherwise
    an exact reindex (the mean matrix is on the same grid).  Returns the aligned
    frame, or None if it shares no cells / no bins with the concordance."""
    sig = sig.copy()
    sig.index = sig.index.astype(str)
    n_cell = sum(c in sig.index for c in cells)
    if n_cell == 0:
        print(f"  {name}: {infix} matrix shares no cells with the concordance "
              f"— skipping {infix} heatmaps.", file=sys.stderr)
        return None
    sig = sig.reindex(index=list(cells))
    if grid_align:
        from nome_hic_concordance.comparison import align_to_grid
        try:
            sig = align_to_grid(sig, list(sel_bins))
        except Exception as e:                       # unparseable bins, etc.
            print(f"  {name}: could not align {infix} to the marker-bin grid "
                  f"({e}); skipping {infix} heatmaps.", file=sys.stderr)
            return None
    else:
        sig = sig.reindex(columns=list(sel_bins))
    n_bin = int((~sig.isna()).any(axis=0).sum())
    if n_bin == 0:
        print(f"  {name}: {infix} matrix covers none of the marker bins "
              f"— skipping {infix} heatmaps.", file=sys.stderr)
        return None
    if n_bin < len(sel_bins) or n_cell < len(cells):
        print(f"  {name}: {infix} covers {n_bin}/{len(sel_bins)} bins, "
              f"{n_cell}/{len(cells)} cells (rest shown blank).", file=sys.stderr)
    return sig


def _companion_group(sig, infix, cmap, value_label, name, sel_bins, main_order,
                     ct, batch_df, ct_colors, batch_colors, batch_order,
                     block_order, out_dir, args, zscore=False, symmetric=False):
    """Render one companion GROUP for an already-aligned signal ``sig`` (cells x
    sel_bins).

    Two orderings x two value versions:

      fixedorder    cell order borrowed from the concordance heatmap, NO
                    dendrogram, so the companion can be read against it column
                    by column.
      reclustered   clustered on the signal itself.

    and, when --remove-batch is on, both again on the batch-corrected signal.
    The corrected FIXED-ORDER panel is the one that makes the comparison honest:
    --remove-batch already corrects the concordance matrix, so ``main_order``
    comes from corrected values, and pairing it with a raw companion asks the
    reader to compare a corrected ordering against an uncorrected signal.

    Returns the filenames.
    """
    made = []
    kw = dict(legend_order=block_order, legend_max=args.max_legend_levels,
              zscore=zscore, symmetric=symmetric)
    versions = [("", sig, "", value_label)]
    if args.remove_batch != "none":
        versions.append(("_batchcorrected", _remove_batch(sig, ct, batch_df),
                         ", batch-corrected",
                         f"{value_label} (batch-corrected)"))
    for tag, mat, note, vlabel in versions:
        for suffix, order, how in (("fixedorder", main_order,
                                    "concordance order"),
                                   ("reclustered", None, "own clustering")):
            fig = (out_dir / f"marker_heatmap_{name}_{infix}_{suffix}{tag}"
                             f".{args.plot_format}")
            _heatmap(mat, list(mat.index), list(sel_bins), ct, batch_df,
                     ct_colors, batch_colors, batch_order,
                     f"{name.upper()}: {value_label}, {how}{note} "
                     f"({len(sel_bins)} marker bins)", fig, cmap, order=order,
                     value_label=vlabel, **kw)
            made.append(fig.name)
    return made


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
    ap.add_argument("--compartment-matrix", default=None, metavar="PATH",
                    help="cells x bins compartment-score parquet (the 3D "
                         "baseline; context-independent, so ONE file is shared "
                         "by every --matrix), e.g. "
                         ".../compartments/compartment_scores.parquet.  A "
                         "coarser bin grid is mapped onto the marker bins.  If "
                         "omitted, compartments/compartment_scores.parquet at "
                         "the run root is auto-derived.  Used for the "
                         "compartment companion heatmaps.")
    ap.add_argument("--compartment-heatmaps", choices=["auto", "on", "off"],
                    default="auto",
                    help="Like --mean-heatmaps, but a third companion group in "
                         "the COMPARTMENT signal ('<name>_compartment_"
                         "fixedorder/_reclustered[/_reclustered_batchcorrected]')"
                         ".  'auto' (default) makes them when a compartment "
                         "matrix is found, else skips.")
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
    ap.add_argument("--chrom-sizes", default=None, metavar="FILE",
                    help="Chromosome sizes (a UCSC chrom.sizes or .fai; the "
                         "first two columns 'contig<TAB>length' are read).  Used "
                         "to clamp the last bin of each chromosome in the output "
                         "BEDs so it never overshoots the chromosome end.  "
                         "Strongly recommended -- without it a fixed-width bin "
                         "grid produces a final bin that runs past the contig "
                         "and breaks bedtools / enrichment tools.")
    ap.add_argument("--n-bins", type=int, default=500,
                    help="Number of marker bins for the heatmap.")
    ap.add_argument("--cmap", action="append", default=None,
                    metavar="NAME=CMAP",
                    help="Override the diverging colormap per matrix, e.g. "
                         "gch=RdBu_r. Defaults: gch=RdBu_r, hcg=PuOr_r.")
    ap.add_argument("--compartment-cmap", action="append", default=None,
                    metavar="NAME=CMAP",
                    help="Override the diverging colormap for the COMPARTMENT "
                         "companion heatmaps per matrix, e.g. gch=PRGn_r. "
                         "Defaults: gch=PRGn_r, hcg=BrBG_r (kept distinct from "
                         "the concordance maps and from each other).")
    ap.add_argument("--stable-bins", choices=["on", "off"], default="on",
                    help="Also emit the INVARIANT set: the bins whose "
                         "cell-type means sit closest together, ranked by the "
                         "SD of those means rather than by low cell-type ICC "
                         "(low ICC returns the noisiest bins, since ICC has "
                         "sampling noise in its denominator).  Same batch "
                         "filter as the marker set — a bin flat across cell "
                         "types but driven by batch is sample, not biology.  "
                         "Writes bin_scores_<name>.stable.bed.")
    ap.add_argument("--n-stable-bins", type=int, default=0,
                    help="Size of the invariant set (0 = same as --n-bins).")
    ap.add_argument("--stable-level", choices=["any", "high", "low"],
                    default="any",
                    help="Restrict the invariant set by CONCORDANCE LEVEL.  "
                         "'any' (default) is stability alone, which says "
                         "nothing about the level: selection ranks on the "
                         "per-bin quantile-normalised matrix, where every bin "
                         "shares one distribution across cells, so a bin flat "
                         "at LOW concordance and one flat at HIGH concordance "
                         "are equally invariant and both are taken.  'high' / "
                         "'low' keep only bins whose raw per-bin mean is above "
                         "/ below the genome-wide median before ranking, for "
                         "'constitutively coordinated' rather than merely "
                         "'unchanging'.  The raw mean is reported as "
                         "bin_mean_raw either way.")
    ap.add_argument("--sign-split", choices=["auto", "on", "off"],
                    default="auto",
                    help="For a DIRECTIONAL matrix (D = C·(2F−1)), also emit "
                         "the bins that are co-open in every cell type and "
                         "those co-closed in every cell type.  'auto' turns "
                         "this on for a matrix whose filename contains "
                         "'directional', which is the only combine mode whose "
                         "sign carries that meaning: for C the sign is "
                         "concordant vs anti-concordant, not open vs closed.  "
                         "Computed on the RAW values, since the normalisation "
                         "used for selection destroys the sign.  Writes "
                         "bin_scores_<name>.always_co_{open,closed}.bed.")
    ap.add_argument("--marginal-matrix", action="append", default=None,
                    metavar="NAME=PATH",
                    help="Marginal component p̄ for --sign-reference expected "
                         "(repeatable).  Defaults to concordance_marginal."
                         "parquet beside the --matrix parquet.")
    ap.add_argument("--coordination-matrix", action="append", default=None,
                    metavar="NAME=PATH",
                    help="Coordination component C for --sign-reference "
                         "expected (repeatable).  Defaults to "
                         "concordance_coordination.parquet beside the "
                         "--matrix parquet.")
    ap.add_argument("--sign-reference",
                    choices=["zero", "median", "expected"], default="zero",
                    help="What co-open / co-closed is judged against.  'zero' "
                         "is the literal reading, and on real data it is "
                         "almost always degenerate: sign(2F−1) equals "
                         "sign(mean level − 0.5), so with mean GCH "
                         "accessibility below 0.5 and mean HCG methylation "
                         "above it, essentially every bin comes back "
                         "co-closed for GCH and co-methylated for HCG — the "
                         "genome's base rate restated, not a property of any "
                         "region.  'median' judges each bin against the "
                         "genome-wide median instead ('more co-open than the "
                         "typical bin'), which is the version that varies "
                         "along the genome.  'expected' gives each bin its "
                         "OWN reference, C·(2·E[F]−1) with E[F] = p̄²/(p̄²+"
                         "(1−p̄)²) from that bin's marginal, so a bin counts "
                         "as co-open only when it is MORE co-open than its own "
                         "mean level already implies — the coordination-versus-"
                         "level control applied to direction rather than to "
                         "magnitude.  Needs the marginal and coordination "
                         "components, taken from beside the --matrix parquet "
                         "unless given explicitly.")
    ap.add_argument("--n-sign-bins", type=int, default=0,
                    help="Bins per direction in the co-open / co-closed BEDs "
                         "(0 = same as --n-bins).  A fixed SIZE, not a "
                         "threshold: it keeps this set comparable with the "
                         "marker and invariant sets for a genomic-enrichment "
                         "comparison, and it is unmoved by a reference that is "
                         "biased for every bin alike — which slides a "
                         "threshold but not a ranking.")
    ap.add_argument("--sign-min-consistency", type=float, default=0.6,
                    help="Fraction of cells that must fall on the dominant "
                         "side for a bin to be called always co-open / "
                         "co-closed, on top of every cell-type mean agreeing.")
    ap.add_argument("--contrast", action="append", default=None,
                    metavar="LABEL=GROUP[,GROUP...]",
                    help="Rank bins by how well they separate the named "
                         "cell-type groups from ALL the others (repeatable), "
                         "e.g. --contrast meningioma=tumor1,tumor2 against a "
                         "panel of GM12878, IMR90 and monocyte.  Welch t per "
                         "bin with BH-FDR and Cohen's d.  Writes "
                         "bin_scores_<name>.contrast_<label>.bed.  The "
                         "association between the contrast and each batch "
                         "variable is reported: when samples of one condition "
                         "were processed together, 'the bins separating these "
                         "groups' and 'the bins separating these batches' are "
                         "the same bins, and the result means nothing.")
    ap.add_argument("--cnv-seg", default=None, metavar="FILE",
                    help="A copy-number segmentation (sample/chrom/start/end/"
                         "value, or a 6-column .seg).  Turns on the CNV check "
                         "for every --contrast: copy number moves coverage, "
                         "coverage moves the DM shrinkage and what RWR "
                         "imputes, so in a tumour-vs-line contrast a bin can "
                         "separate the groups for reasons that are not "
                         "coordination at all.  Reports how strongly the "
                         "contrast tracks the group copy-number difference and "
                         "how many selected bins sit in CN-different regions.")
    ap.add_argument("--cnv-sample-column", default=None,
                    help="Column in --samples whose values match the "
                         "segmentation's sample IDs (default: the cell name).  "
                         "Use it when the CNV calls are per donor or per "
                         "library rather than per cell.")
    ap.add_argument("--cnv-max-delta", type=float, default=0.0,
                    help="Drop bins whose |group copy-number difference| "
                         "exceeds this from the contrast pool BEFORE ranking "
                         "(0 = report only).  0.2-0.3 on a log2 scale removes "
                         "the clearly CN-different regions while keeping the "
                         "copy-neutral genome.")
    ap.add_argument("--n-contrast-bins", type=int, default=0,
                    help="Bins per contrast BED with --contrast-select top-n "
                         "(0 = same as --n-bins).  Ignored under the default "
                         "fdr rule, which has no fixed count.")
    ap.add_argument("--contrast-select", choices=["fdr", "top-n"],
                    default="fdr",
                    help="How a contrast BED is filled.  'fdr' (default) keeps "
                         "every bin below --contrast-fdr, which is what the "
                         "volcano colours and can legitimately come back "
                         "empty.  'top-n' keeps the --n-contrast-bins largest "
                         "|t| whether or not any of them is significant, so it "
                         "always returns a set and never says the contrast is "
                         "empty.")
    ap.add_argument("--contrast-test", choices=["mwu", "welch"], default="mwu",
                    help="Per-bin two-sample test for a contrast.  'mwu' "
                         "(default) is a two-sided Mann-Whitney U over cells: "
                         "no normality assumption, and one outlying cell "
                         "cannot carry a bin.  'welch' is the Welch t.  Either "
                         "way the volcano's x-axis is the difference of means, "
                         "in D units.  A rank test has a p-value FLOOR set by "
                         "the arm sizes -- 0.1 at 3 vs 3 -- and the run warns "
                         "when that floor puts FDR out of reach.")
    ap.add_argument("--contrast-fdr", type=float, default=0.05,
                    help="BH-FDR cut for --contrast-select fdr (0.05).")
    ap.add_argument("--contrast-min-diff", type=float, default=0.0,
                    help="Also require |mean difference| at least this large. "
                         "0 disables.  With many cells a tiny shift reaches "
                         "significance, and this is where 'significant' is "
                         "separated from 'big enough to care about'.")
    ap.add_argument("--volcano", action="store_true",
                    help="Draw a volcano per contrast per context: mean "
                         "difference against -log10(FDR), significant bins "
                         "coloured by direction.")
    ap.add_argument("--contrast-joint", action="store_true",
                    help="After every context, compare the contrast ACROSS "
                         "them: the union of significant bins, a scatter of "
                         "the two mean differences, and a clustered heatmap "
                         "of each modality over those bins -- which shows "
                         "whether a bin is aberrant in both or in only one.")
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
    from nome_hic_concordance.blacklist import infer_bin_size

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
    chrom_sizes = None
    if args.chrom_sizes:
        chrom_sizes = _read_chrom_sizes(args.chrom_sizes)
        print(f"chrom sizes: {len(chrom_sizes)} contigs (last bins clamped to "
              f"the contig end)", file=sys.stderr)
    else:
        print("NOTE: --chrom-sizes not given; the last bin of each chromosome "
              "in the output BEDs may run past the contig end (pass "
              "--chrom-sizes to clamp it).", file=sys.stderr)
    cmap_override = {}
    for c in (args.cmap or []):
        if '=' in c:
            k, v = c.split('=', 1); cmap_override[k.strip().lower()] = v.strip()
    comp_cmap_override = {}
    for c in (args.compartment_cmap or []):
        if '=' in c:
            k, v = c.split('=', 1)
            comp_cmap_override[k.strip().lower()] = v.strip()
    def _named_map(specs, flag):
        out = {}
        for c in (specs or []):
            if "=" not in c:
                sys.exit(f"{flag} must be NAME=PATH, got {c!r}")
            k, v = c.split("=", 1)
            out[k.strip()] = v.strip()
        return out

    marginal_explicit = _named_map(args.marginal_matrix, "--marginal-matrix")
    coord_explicit = _named_map(args.coordination_matrix,
                                "--coordination-matrix")
    mean_explicit = {}
    for c in (args.mean_matrix or []):
        if "=" not in c:
            sys.exit(f"--mean-matrix must be NAME=PATH, got {c!r}")
        k, v = c.split("=", 1); mean_explicit[k.strip()] = v.strip()
    contrasts = []
    for spec in (args.contrast or []):
        if "=" not in spec:
            sys.exit(f"--contrast must be LABEL=GROUP[,GROUP...][:REF[,REF]], "
                     f"got {spec!r}")
        lab, groups = spec.split("=", 1)
        lab = lab.strip()
        # "in:ref" names the reference arm; without it the reference is every
        # other cell, which is a different question with a different answer.
        ref = None
        if ":" in groups:
            groups, ref_s = groups.split(":", 1)
            ref = [g.strip() for g in ref_s.split(",") if g.strip()]
            if not ref:
                sys.exit(f"--contrast {spec!r}: ':' given but no reference "
                         f"group follows it")
        want = [g.strip() for g in groups.split(",") if g.strip()]
        if not lab or not want:
            sys.exit(f"--contrast needs a label and at least one group: "
                     f"{spec!r}")
        overlap = set(want) & set(ref or [])
        if overlap:
            sys.exit(f"--contrast {spec!r}: {sorted(overlap)} is on both sides")
        contrasts.append((lab, want, ref))

    seg = cell_to_sample = None
    if args.cnv_seg:
        if not contrasts:
            print("  NOTE --cnv-seg does nothing without --contrast: the check "
                  "is about a contrast being copy number rather than biology.",
                  file=sys.stderr)
        seg = read_segments(args.cnv_seg)
        col = args.cnv_sample_column
        if col and col not in samples.columns:
            sys.exit(f"--cnv-sample-column {col!r} not in --samples "
                     f"(columns: {samples.columns.tolist()}).")
        cell_to_sample = dict(zip(
            samples["cell_name"],
            samples[col].astype(str) if col else samples["cell_name"]))
        known = set(seg["sample"])
        matched = {v for v in cell_to_sample.values() if v in known}
        if not matched:
            sys.exit(
                f"--cnv-seg: none of its sample IDs match "
                f"{col or 'cell_name'} in --samples.  Its IDs look like "
                f"{sorted(known)[:3]}; the sheet has "
                f"{sorted(set(cell_to_sample.values()))[:3]}.  Point "
                f"--cnv-sample-column at the column that matches.")
        print(f"CNV segmentation: {len(seg):,} segments over {len(known)} "
              f"samples, {len(matched)} of them matched to cells.",
              file=sys.stderr)

    out_dir = ensure_dir(Path(args.output_dir))
    summary = {}

    # {contrast label: {context: frame(diff, fdr, sig) indexed by bin key}}
    joint = {}
    joint_mats = {}
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
        # Compartment matrix (shared across matrices; explicit flag wins, else
        # auto-derive compartments/compartment_scores.parquet at the run root).
        comp_path = args.compartment_matrix
        if comp_path and not Path(comp_path).exists():
            sys.exit(f"no such --compartment-matrix: {comp_path}")
        if not comp_path:
            comp_path = _find_compartment(path)
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

        # The co-open / co-closed call needs the values BEFORE normalisation:
        # per-bin QN and the per-cell rank both destroy the sign that carries
        # the meaning.  Kept here and re-aligned to the surviving bins below.
        M_raw = M.copy()

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
        #
        # The contrast needs the same correction, on the RAW scale it runs on.
        # _remove_batch centres each (cell type, batch) group to its cell-type
        # mean, so it takes out within-cell-type batch variance while leaving
        # the between-cell-type means alone -- which is exactly the split a
        # tumour-vs-reference contrast wants: less nuisance spread in each arm,
        # the difference between the arms untouched.
        M_contrast = M_raw.reindex(index=M.index, columns=M.columns)
        if args.remove_batch == "within-celltype":
            M = _remove_batch(M, ct, batch_df)
            M_contrast = _remove_batch(M_contrast, ct, batch_df)

        batch_specs = []
        for col in bcols:
            u = sorted(batch_df[col].dropna().unique())
            batch_specs.append(
                (np.searchsorted(np.array(u), batch_df[col].to_numpy()), len(u)))

        # Select on ALL cells: more coverage means more bins pass than a train
        # split would (a hard filter on half the cells left only a couple dozen).
        # One materialised copy of the selection matrix, shared by every
        # ranking below so they cannot disagree about what was measured.
        Msel = M.to_numpy()
        sel, ct_icc, b_icc, n_clean = select_marker_bins(
            Msel, ct_id, len(ct_u), batch_specs, ctv.icc_per_region, args)
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

        # Companion heatmap GROUPS on the mean signal (accessibility for gch,
        # methylation for hcg) and the compartment signal (the 3D baseline).
        # Each group has: (A) the concordance heatmap's EXACT layout (NO
        # dendrogram -- borrowed, not clustered here) -> is the concordance
        # structure present in this signal?  (B) the same cells+bins reclustered
        # on the signal -> what layout does it make on its own?  (C) when
        # --remove-batch is on, reclustered on the batch-corrected signal.  All
        # show the RAW value (no concordance normalisation): mean on the
        # conventional NOMe-seq sequential palette (white->teal / blue->yellow),
        # compartment on a diverging A/B (red/blue) scale centred at 0.
        cg = dict(name=name, sel_bins=sel_bins, main_order=main_order, ct=ct,
                  batch_df=batch_df, ct_colors=ct_colors,
                  batch_colors=batch_colors, batch_order=batch_order,
                  block_order=block_order, out_dir=out_dir, args=args)

        mean_made = []
        if args.mean_heatmaps != "off" and mean_path:
            # Mean is on the concordance bin grid -> exact reindex.
            macc = _align_signal(pd.read_parquet(mean_path), list(M.index),
                                 sel_bins, name, "mean", grid_align=False)
            if macc is not None:
                mean_made = _companion_group(
                    macc, "mean", _mean_cmap(name),
                    _MEAN_LABEL.get(name.lower(), "mean level"), **cg)
        if mean_made:
            print(f"  mean-signal companions -> {', '.join(mean_made)}")
        elif args.mean_heatmaps == "on":
            sys.exit(f"--mean-heatmaps on but no usable mean matrix for {name!r}:"
                     f" pass --mean-matrix {name}=PATH or place "
                     f"accessibility_mean.parquet beside {path}.")

        comp_made = []
        if args.compartment_heatmaps != "off" and comp_path:
            # Compartment may be on a coarser grid -> map onto the marker bins.
            comp = _align_signal(pd.read_parquet(comp_path), list(M.index),
                                 sel_bins, name, "compartment", grid_align=True)
            if comp is not None:
                comp_cmap = (comp_cmap_override.get(name.lower())
                             or _COMPARTMENT_CMAPS.get(
                                 name.lower(),
                                 _COMPARTMENT_FALLBACK[mi % len(_COMPARTMENT_FALLBACK)]))
                comp_made = _companion_group(
                    comp, "compartment", comp_cmap, "compartment score",
                    symmetric=True, **cg)
        if comp_made:
            print(f"  compartment companions -> {', '.join(comp_made)}")
        elif args.compartment_heatmaps == "on":
            sys.exit(f"--compartment-heatmaps on but no usable compartment "
                     f"matrix for {name!r}: pass --compartment-matrix PATH or "
                     f"place compartments/compartment_scores.parquet at the run "
                     f"root.")

        # Out-of-selection CONTROL (number only; the UMAP figure is produced by
        # nome_hic_concordance / nome_hic_compare): do the bins we did NOT pick
        # still separate cell types?
        sil, perm_p = _nonselected_separation(
            M.loc[:, rest_bins], ct.to_numpy(), args.n_svd, args.skip_top_svd,
            args.seed, args.n_permutations)

        # Per-bin statistics: the cell-type ICC (the ranking statistic) with its
        # one-way ANOVA F-test p-value + BH-FDR (same model as the ICC), plus the
        # within-cell-type batch ICC (the filter statistic).  Computed on the
        # SAME matrix used for selection (batch-removed if --remove-batch).
        _, _, _ct_f, ct_p = ctv.icc_per_region(
            M.to_numpy(), ct_id, len(ct_u), args.min_cells_per_group,
            args.min_groups, return_stats=True)
        ct_fdr = _bh_fdr(ct_p)
        # The per-bin RAW mean, purely as a reported column.  Selection runs on
        # the per-bin quantile-normalised matrix, where every bin has been
        # mapped onto the same distribution across cells, so the level is
        # invisible to every ranking here -- a stable bin is flat across cell
        # types and may sit anywhere on the level axis.  Without this column
        # that could not even be checked afterwards.
        raw_mean = np.asarray(
            M_raw.reindex(index=M.index, columns=M.columns).mean(axis=0),
            dtype=float)
        scores_df = pd.DataFrame({
            "bin": M.columns, "bin_mean_raw": raw_mean, "celltype_icc": ct_icc,
            "celltype_pvalue": ct_p, "celltype_fdr": ct_fdr,
            "batch_icc": b_icc, "selected": M.columns.isin(sel_bins)})
        extra_cols: list[str] = []
        subsets: dict[str, np.ndarray] = {}

        # (2) INVARIANT set: flattest across cell types, same batch filter.
        if args.stable_bins == "on":
            n_want = args.n_stable_bins or args.n_bins
            eligible = None
            if args.stable_level != "any":
                # Level is not part of the ranking and cannot be: after per-bin
                # QN it is gone.  So it is applied as a POOL restriction before
                # ranking, against the genome-wide median -- the same relative
                # reading --sign-reference median uses, and for the same reason
                # (an absolute cut on D is a cut on the base rate).
                med = float(np.nanmedian(raw_mean))
                eligible = (raw_mean > med if args.stable_level == "high"
                            else raw_mean < med)
                print(f"  {name}: --stable-level {args.stable_level} keeps "
                      f"{int(np.sum(eligible & np.isfinite(raw_mean))):,} of "
                      f"{len(raw_mean):,} bins eligible for the invariant set "
                      f"(raw per-bin mean vs the genome-wide median "
                      f"{med:+.4g}).", file=sys.stderr)
            stable_sel, spread, n_stable_clean = select_stable_bins(
                Msel, ct_id, len(ct_u), b_icc, args, n_want, eligible)
            is_stable = np.zeros(M.shape[1], bool)
            is_stable[stable_sel] = True
            scores_df["celltype_mean_spread"] = spread
            scores_df["stable"] = is_stable
            extra_cols += ["celltype_mean_spread", "stable"]
            subsets["stable"] = is_stable
            summary_stable = {
                "n_stable": int(stable_sel.size),
                "n_stable_batch_clean": int(n_stable_clean),
                "median_celltype_icc_stable": _nanmedian(ct_icc[stable_sel]),
                "median_spread_stable": _nanmedian(spread[stable_sel]),
                "median_spread_marker": _nanmedian(spread[sel])}
        else:
            summary_stable = {}

        # (3) Directional sign: always co-open / always co-closed, on RAW values.
        sign_on = (args.sign_split == "on"
                   or (args.sign_split == "auto"
                       and "directional" in Path(path).name.lower()))
        summary_sign = {}
        if sign_on:
            raw = M_raw.reindex(index=M.index, columns=M.columns)
            expected = None
            if args.sign_reference == "expected":
                comps = {}
                for kind, explicit in (("marginal", marginal_explicit),
                                       ("coordination", coord_explicit)):
                    cp = explicit.get(name)
                    if not cp:
                        cand = Path(path).with_name(
                            f"concordance_{kind}.parquet")
                        cp = str(cand) if cand.exists() else None
                    if not cp or not Path(cp).exists():
                        sys.exit(
                            f"--sign-reference expected needs the {kind} "
                            f"component for {name!r}: no --{kind}-matrix given "
                            f"and no concordance_{kind}.parquet beside "
                            f"{path}.")
                    # Reindexed onto the SAME cells and bins as D, so the
                    # reference is per (cell, bin) and aligned, not broadcast
                    # from a different grid.
                    comps[kind] = pd.read_parquet(cp).reindex(
                        index=M.index, columns=M.columns)
                expected = expected_directional(comps["marginal"],
                                                comps["coordination"])
                print(f"  {name}: per-bin expected-D reference from "
                      f"{Path(path).parent}/concordance_{{marginal,"
                      f"coordination}}.parquet", file=sys.stderr)
            sg = sign_summary(raw, ct_id, len(ct_u), args, expected)
            n_want = args.n_sign_bins or args.n_bins
            for key, rank_by, ascending in (
                    ("always_co_open", "d_group_min", False),
                    ("always_co_closed", "d_group_max", True)):
                hit, n_agree = select_sign_bins(sg, key, rank_by, ascending,
                                                n_want)
                mask = np.zeros(M.shape[1], bool)
                mask[hit] = True
                scores_df[key] = mask
                subsets[key] = mask
                summary_sign[f"n_{key}"] = int(mask.sum())
                summary_sign[f"n_{key}_agreeing"] = int(n_agree)
                summary_sign[f"n_{key}_total"] = int(sg[key].sum())
            for c in ("d_mean", "d_reference", "d_group_min", "d_group_max",
                      "frac_cells_positive"):
                scores_df[c] = sg[c]
            extra_cols += ["d_mean", "d_reference", "d_group_min",
                           "d_group_max", "frac_cells_positive",
                           "always_co_open", "always_co_closed"]
            called = int(sg["always_co_open"].sum()
                         + sg["always_co_closed"].sum())
            big = max(int(sg["always_co_open"].sum()),
                      int(sg["always_co_closed"].sum()))
            if called and big / called > 0.9:
                why = ("sign(2F−1) = sign(mean level − 0.5), so a mark whose "
                       "genome-wide mean is not near 0.5 lands on one side "
                       "everywhere; try --sign-reference median or expected"
                       if args.sign_reference == "zero" else
                       "a reference offset shared by every bin does this, and "
                       "the expected-D one carries one (the stored marginal is "
                       "the MEAN of the two ends')")
                print(f"  NOTE {name}: {100 * big / called:.1f}% of the bins "
                      f"that PASS the sign test fall on one side.  {why}.  The "
                      f"BEDs are unaffected: they take the top "
                      f"{args.n_sign_bins or args.n_bins} by rank in each "
                      f"direction, which a common offset cannot move.",
                      file=sys.stderr)

        # (4) Group-set contrasts: these groups against every other.
        summary_contrast = {}
        for label, want, ref_groups in contrasts:
            present = [g for g in want if g in set(ct_u)]
            missing = [g for g in want if g not in set(ct_u)]
            if missing:
                print(f"  contrast {label!r}: no cells for {missing} "
                      f"(available: {ct_u})", file=sys.stderr)
            if not present:
                print(f"  contrast {label!r}: no group present — skipped",
                      file=sys.stderr)
                continue
            in_mask = ct.isin(present).to_numpy()
            if ref_groups:
                ref_present = [g for g in ref_groups if g in set(ct_u)]
                ref_missing = [g for g in ref_groups if g not in set(ct_u)]
                if ref_missing:
                    print(f"  contrast {label!r}: no cells for reference "
                          f"{ref_missing} (available: {ct_u})",
                          file=sys.stderr)
                if not ref_present:
                    print(f"  contrast {label!r}: no reference cells — "
                          f"skipped", file=sys.stderr)
                    continue
                out_mask = ct.isin(ref_present).to_numpy()
                arm = f"vs {'+'.join(ref_present)}"
            else:
                if len(present) == len(ct_u):
                    print(f"  contrast {label!r}: every group is in the set, "
                          f"so nothing is left to compare against — skipped",
                          file=sys.stderr)
                    continue
                out_mask = ~in_mask
                arm = "vs all other cell types"
            print(f"  contrast {label!r}: {int(in_mask.sum())} cells "
                  f"({'+'.join(present)}) {arm}, {int(out_mask.sum())} cells",
                  file=sys.stderr)
            # On the batch-corrected RAW values, not the normalised ones.
            # Per-bin QN
            # followed by per-cell rank is COMPOSITIONAL: a real shift in one
            # block of bins pushes every other bin the other way in rank
            # space, so bins that are identical in every cell acquire a
            # difference between the arms out of nothing.  On a fixture whose
            # blocks 2 and 3 are constant across all cells, the normalised
            # matrix calls 25% and 20% of them significant at p<0.01 against
            # 5% and 0% raw.  A ranked top-N hid this by taking only the
            # strongest bins; an FDR cut does not, and would fill the BED with
            # normalisation artefacts.  The mean difference is also only a
            # difference in D on this scale -- on the normalised one it is a
            # difference in per-cell rank, which is not what the volcano's
            # x-axis claims to show.
            cs = contrast_stats(M_contrast.to_numpy(), in_mask,
                                args.min_cells_per_group, out_mask,
                                args.contrast_test)
            q = _bh_fdr(cs["p"])
            if args.contrast_test == "mwu":
                n_hi = int(np.nanmax(cs["n_in"])), int(np.nanmax(cs["n_out"]))
                floor = mwu_min_pvalue(*n_hi)
                n_tested = int(np.isfinite(cs["p"]).sum())
                # BH needs the most significant bin at or below q/m, so a
                # floor above that means NOTHING can be called, whatever the
                # data say.
                need = args.contrast_fdr / max(n_tested, 1)
                if np.isfinite(floor) and floor > need:
                    print(f"  contrast {label!r}: WARNING a rank test on "
                          f"{n_hi[0]} vs {n_hi[1]} cells cannot return a "
                          f"p below {floor:.2g}, and BH over {n_tested:,} "
                          f"bins needs {need:.2g} for even the top bin, so no "
                          f"bin can reach FDR<{args.contrast_fdr}.  Use "
                          f"--contrast-test welch, or accept that this "
                          f"contrast has too few cells to call.",
                          file=sys.stderr)
            # The copy-number check, before anything is ranked: a bin whose two
            # arms differ in copy number differs in coverage, and coverage is
            # upstream of every value in this matrix.
            cnv = cnv_info = None
            if seg is not None:
                cells_in = list(M.index[in_mask])
                cells_out = list(M.index[out_mask])
                cnv, _samples, (n_in, n_out) = cnv_group_delta(
                    M.columns, seg, cell_to_sample, cells_in, cells_out)
                if cnv is None:
                    print(f"  contrast {label!r}: no CNV samples on one side "
                          f"({n_in} in, {n_out} out) — check skipped.",
                          file=sys.stderr)
                else:
                    ok_c = np.isfinite(cnv) & np.isfinite(cs["d"])
                    from scipy.stats import pearsonr, spearmanr
                    r = sr = np.nan
                    if int(ok_c.sum()) > 2:
                        r = float(pearsonr(cnv[ok_c], cs["d"][ok_c])[0])
                        sr = float(spearmanr(cnv[ok_c], cs["d"][ok_c])[0])
                    cnv_info = {"n_samples_in": n_in, "n_samples_out": n_out,
                                "n_bins_with_cnv": int(ok_c.sum()),
                                "pearson_r_effect_vs_cnv": r,
                                "spearman_rho_effect_vs_cnv": sr}
                    scores_df[f"contrast_{label}_cnv_delta"] = cnv
                    extra_cols.append(f"contrast_{label}_cnv_delta")
                    if args.cnv_max_delta > 0:
                        drop = np.isfinite(cnv) & (np.abs(cnv)
                                                   > args.cnv_max_delta)
                        cs["t"] = np.where(drop, np.nan, cs["t"])
                        cnv_info["n_bins_dropped"] = int(drop.sum())
                        print(f"  contrast {label!r}: --cnv-max-delta "
                              f"{args.cnv_max_delta} removed "
                              f"{int(drop.sum()):,} bins from the pool.",
                              file=sys.stderr)
            # Rank on the chosen test's p, not on |t|: under mwu the t is not
            # the statistic that produced the p-value.
            rank = np.where(np.isfinite(cs["p"]),
                            -np.log10(np.maximum(cs["p"], 1e-300)), -np.inf)
            mask = np.zeros(M.shape[1], bool)
            if args.contrast_select == "fdr":
                # Significance, not a fixed count.  A rank cut returns N bins
                # whether or not anything differs, so "the top 500" reads as a
                # finding even when the contrast is empty; an FDR cut can
                # return nothing, which is the honest answer in that case.
                sig = (np.isfinite(q) & (q < args.contrast_fdr)
                       & np.isfinite(cs["diff"]))
                if args.contrast_min_diff > 0:
                    sig &= np.abs(cs["diff"]) >= args.contrast_min_diff
                if args.cnv_max_delta > 0 and cnv is not None:
                    sig &= ~(np.isfinite(cnv)
                             & (np.abs(cnv) > args.cnv_max_delta))
                mask = sig
            else:
                keep_rank = np.argsort(rank)[::-1][: args.n_contrast_bins
                                                   or args.n_bins]
                keep_rank = keep_rank[np.isfinite(rank[keep_rank])]
                mask[keep_rank] = True
            # A contrast bin driven by batch is worthless in exactly the same
            # way a marker bin is, so it faces the same filter.
            clean = ~(np.isfinite(b_icc) & (b_icc > args.max_batch_icc))
            pre = f"contrast_{label}"
            if args.volcano:
                vpath = out_dir / f"volcano_{name}_{label}.pdf"
                n_up, n_dn = volcano_figure(
                    cs["diff"], q, mask, vpath,
                    f"{name}  {label} ({'+'.join(present)} {arm})",
                    args.contrast_fdr, args.contrast_min_diff)
                print(f"  contrast {label!r}: {n_up:,} up / {n_dn:,} down "
                      f"-> {vpath.name}", file=sys.stderr)
            joint.setdefault(label, {})[name] = pd.DataFrame(
                {"diff": cs["diff"], "fdr": q, "sig": mask},
                index=pd.Index(M.columns, name="bin"))
            joint_mats[name] = M
            scores_df[f"{pre}_diff"] = cs["diff"]
            scores_df[f"{pre}_cohens_d"] = cs["d"]
            scores_df[f"{pre}_pvalue"] = cs["p"]
            scores_df[f"{pre}_fdr"] = q
            scores_df[pre] = mask
            extra_cols += [f"{pre}_diff", f"{pre}_cohens_d", f"{pre}_pvalue",
                           f"{pre}_fdr", pre]
            subsets[pre] = mask
            # Confounding check: if the contrast IS a batch, the bins that
            # separate the groups are the bins that separate the batches.
            conf = {c: _cramers_v(pd.Series(in_mask, index=M.index),
                                  batch_df[c]) for c in bcols}
            worst = max(conf, key=conf.get) if conf else None
            # How many selected bins the batch filter could even JUDGE.  It is
            # a WITHIN-cell-type statistic, so when a batch is nested inside
            # cell type -- the usual shape of a confounded design -- there is
            # no within-cell-type batch contrast to measure, every bin comes
            # back NaN and "passed the batch filter" means "was never tested".
            judged = int(np.sum(mask & np.isfinite(b_icc)))
            if cnv_info is not None:
                sel_cnv = np.abs(cnv[mask & np.isfinite(cnv)])
                cut = args.cnv_max_delta or 0.2
                cnv_info["median_abs_cnv_delta_selected"] = (
                    float(np.median(sel_cnv)) if sel_cnv.size else np.nan)
                cnv_info["frac_selected_over_delta"] = (
                    float(np.mean(sel_cnv > cut)) if sel_cnv.size else np.nan)
                cnv_info["delta_cut_used"] = float(cut)
            summary_contrast[label] = {
                "groups_in": present, "n_cells_in": int(in_mask.sum()),
                # The REFERENCE arm, not "everything else": with an explicit
                # reference the two differ, and reporting the complement would
                # describe a comparison that was never run.
                "groups_out": ref_present if ref_groups else "rest",
                "n_cells_out": int(out_mask.sum()),
                "n_selected": int(mask.sum()),
                "n_fdr_0.05": int(np.nansum(q < 0.05)),
                "n_selected_batch_clean": int(np.sum(mask & clean)),
                "n_selected_batch_judged": judged,
                "batch_confounding_cramers_v": conf,
                **({"cnv": cnv_info} if cnv_info else {})}
            if cnv_info and np.isfinite(cnv_info.get("frac_selected_over_delta",
                                                      np.nan)):
                rr = cnv_info["pearson_r_effect_vs_cnv"]
                frac = cnv_info["frac_selected_over_delta"]
                # The warning keys on WHERE THE SELECTED BINS ARE, not on the
                # correlation.  Two step functions over complementary regions
                # correlate near-perfectly while the effect sits exactly where
                # the copy number does not, so a correlation alone would flag a
                # clean result; it is reported as a diagnostic instead.
                if frac > 0.3:
                    print(f"  WARNING contrast {label!r} tracks COPY NUMBER: "
                          f"the per-bin effect correlates with the group "
                          f"copy-number difference at r = {rr:+.2f}, and "
                          f"{100 * frac:.0f}% of the selected bins sit where "
                          f"the two arms differ by more than "
                          f"{cnv_info['delta_cut_used']:.2f}.  Copy number "
                          f"moves coverage and coverage is upstream of every "
                          f"value here, so these bins are not evidence of "
                          f"differential coordination until the effect "
                          f"survives inside copy-neutral regions — re-run with "
                          f"--cnv-max-delta.", file=sys.stderr)
            if worst and conf[worst] > 0.5:
                print(f"  WARNING contrast {label!r} is confounded with "
                      f"{worst} (Cramer's V {conf[worst]:.2f}): these cells "
                      f"were largely processed apart from the rest, so a bin "
                      f"that separates the groups cannot be distinguished "
                      f"from one that separates the batches.", file=sys.stderr)
                if judged < mask.sum():
                    print(f"    and --max-batch-icc cannot help: it is a "
                          f"WITHIN-cell-type statistic and only "
                          f"{judged}/{int(mask.sum())} selected bins even have "
                          f"one, because {worst} does not vary inside a cell "
                          f"type here.  These bins are a hypothesis to test on "
                          f"samples processed together, not a result.",
                          file=sys.stderr)

        scores_df.to_csv(out_dir / f"bin_scores_{name}.tsv", sep="\t",
                         index=False)

        # BED files (0-based, half-open) with the same statistics: one with ALL
        # bins, one per selected subset.
        coords = _bins_to_bed(M.columns, infer_bin_size(M.columns), chrom_sizes)
        bed = pd.DataFrame(coords, columns=["chrom", "start", "end"])
        for c in ("bin_mean_raw", "celltype_icc", "celltype_pvalue",
                  "celltype_fdr", "batch_icc", "selected", *extra_cols):
            bed[c] = scores_df[c].values
        bed = bed.sort_values(["chrom", "start"]).reset_index(drop=True)
        bed.to_csv(out_dir / f"bin_scores_{name}.all.bed", sep="\t",
                   index=False, na_rep="NA")
        flags = ["selected", *subsets]
        for key in flags:
            # Every membership flag is dropped from a subset file: the one it
            # was selected by is constant True there, and the others belong to
            # a different question.  The per-bin STATISTICS all stay, since
            # they are what the bins are for.
            bed[bed[key]].drop(columns=flags).to_csv(
                out_dir / f"bin_scores_{name}.{key}.bed", sep="\t",
                index=False, na_rep="NA")
        summary[name] = {
            "n_cells": int(M.shape[0]), "n_bins": int(M.shape[1]),
            "n_selected": int(sel.size),
            "n_selected_batch_clean": int(n_clean),
            "median_celltype_icc_selected": _nanmedian(ct_icc[sel]),
            "median_batch_icc_selected": _nanmedian(b_icc[sel]),
            "nonselected_silhouette": (None if not np.isfinite(sil)
                                       else float(sil)),
            "nonselected_perm_p": (None if not np.isfinite(perm_p)
                                   else float(perm_p)),
            **summary_stable, **summary_sign,
            **({"contrasts": summary_contrast} if summary_contrast else {})}
        print(f"\n=== {name}: {M.shape[0]} cells x {M.shape[1]} bins ===")
        print(f"  selected {sel.size} bins ({n_clean} batch-invariant): "
              f"median cell-type ICC {_nanmedian(ct_icc[sel]):.3f}, "
              f"median batch ICC {_nanmedian(b_icc[sel]):.3f}")
        if n_clean < sel.size:
            print(f"  NOTE: only {n_clean} bins passed --max-batch-icc "
                  f"{args.max_batch_icc}; the rest were filled by cell-type ICC "
                  f"(raise the cut to admit more, or lower --n-bins).")
        if summary_stable:
            print(f"  invariant set: {summary_stable['n_stable']} bins "
                  f"({summary_stable['n_stable_batch_clean']} batch-invariant), "
                  f"median cell-type-mean spread "
                  f"{summary_stable['median_spread_stable']:.4f} against "
                  f"{summary_stable['median_spread_marker']:.4f} for the marker "
                  f"set -> bin_scores_{name}.stable.bed")
        for key in ("always_co_open", "always_co_closed"):
            if f"n_{key}" in summary_sign:
                print(f"  {key.replace('_', ' ')}: "
                      f"{summary_sign[f'n_{key}']} written of "
                      f"{summary_sign[f'n_{key}_total']} qualifying "
                      f"-> bin_scores_{name}.{key}.bed")
        for lab, s in summary_contrast.items():
            ref = s.get("groups_out", "rest")
            ref = "+".join(ref) if isinstance(ref, list) else ref
            print(f"  contrast {lab} ({'+'.join(s['groups_in'])} vs {ref}, "
                  f"{s['n_cells_in']} vs {s['n_cells_out']} cells): "
                  f"{s['n_selected']} bins written, "
                  f"{s['n_fdr_0.05']} at FDR<0.05, "
                  f"{s['n_selected_batch_clean']}/{s['n_selected']} "
                  f"batch-invariant -> bin_scores_{name}.contrast_{lab}.bed")
        print(f"  heatmap on all {M.shape[0]} cells -> {fig_hm.name}")
        print(f"  per-bin stats -> bin_scores_{name}.{{all,selected}}.bed "
              f"(chrom/start/end, cell-type ICC, ANOVA p, BH-FDR, batch ICC)")
        if np.isfinite(sil):
            verdict = "SEPARATES" if perm_p < 0.05 else "does NOT separate"
            print(f"  out-of-selection control: NON-selected bins {verdict} "
                  f"cell types (silhouette={sil:.3f}, perm p={perm_p:.3f}) — "
                  f"see the nome_hic_concordance/compare UMAP for the figure")

    if not summary:
        sys.exit("no matrix produced output.")

    # (5) The contrast ACROSS contexts: is a bin aberrant in both modalities,
    # or in only one?  Needs both, so it waits until every matrix is done.
    if args.contrast_joint:
        for label, per_ctx in joint.items():
            if len(per_ctx) != 2:
                print(f"  contrast-joint {label!r}: needs exactly two "
                      f"contexts, have {sorted(per_ctx)} — skipped",
                      file=sys.stderr)
                continue
            tab, a, b = joint_contrast_table(per_ctx, label)
            stats = joint_concordance(tab, a, b)
            tab.to_csv(out_dir / f"contrast_joint_{label}.tsv", sep="\t")
            sel = tab[tab["category"] != "neither"]
            bed = out_dir / f"contrast_joint_{label}.bed"
            coords = _bins_to_bed(sel.index, infer_bin_size(sel.index),
                                  chrom_sizes)
            jb = pd.DataFrame(coords, columns=["chrom", "start", "end"])
            for c in (f"{a}_diff", f"{a}_fdr", f"{b}_diff", f"{b}_fdr",
                      "category"):
                jb[c] = sel[c].to_numpy()
            jb.sort_values(["chrom", "start"]).to_csv(
                bed, sep="\t", index=False, na_rep="NA")
            joint_figure(tab, a, b, stats,
                         {k: joint_mats[k] for k in (a, b)},
                         ct_of, out_dir / f"contrast_joint_{label}.pdf",
                         label)
            summary.setdefault("contrast_joint", {})[label] = stats
            print(f"  contrast-joint {label!r}: {stats['n_both']:,} in both, "
                  f"{stats[f'n_{a}_only']:,} {a} only, "
                  f"{stats[f'n_{b}_only']:,} {b} only; "
                  f"{stats['frac_both_same_direction']:.0%} of the shared ones "
                  f"move the same way -> {bed.name}", file=sys.stderr)
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
