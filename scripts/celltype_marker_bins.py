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
import warnings
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


def variance_from_counts(counts, X, in_mask, out_mask, eps=1e-12):
    """Approximate per-estimate variance from read-pair counts.

    Delta is a difference of two proportions estimated from the pairs in a
    (cell, bin), so its sampling variance falls as 1/n -- exact in form, with
    only the constant unknown, since that constant involves the underlying
    proportions.  The constant is recovered per bin and per arm by moments:
    with v_i = c/n_i, the weighted residual sum Sum n_i (x_i - m)^2 has
    expectation (k-1)c, so c is that sum over (k-1).

    Anything the counts do not explain -- real between-cell variation -- lands
    in c and inflates every variance in that arm equally.  That makes this
    CONSERVATIVE rather than anticonservative, which is the right way to be
    wrong when the exact variances are missing.

    Use concordance_delta_var.parquet when the pipeline wrote one; this is the
    fallback, not the equal.
    """
    n = np.asarray(counts, float)
    n = np.where(np.isfinite(n) & (n > 0), n, np.nan)
    X = np.asarray(X, float)
    V = np.full(X.shape, np.nan)
    for m in (np.asarray(in_mask, bool), np.asarray(out_mask, bool)):
        x, ni = X[m], n[m]
        ok = np.isfinite(x) & np.isfinite(ni)
        w = np.where(ok, ni, np.nan)
        sw = np.nansum(w, axis=0)
        mu = np.nansum(w * np.where(ok, x, 0.0), axis=0) / np.maximum(sw, eps)
        k = np.sum(ok, axis=0)
        resid = np.nansum(w * (np.where(ok, x, np.nan) - mu) ** 2, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.where(k > 1, resid / np.maximum(k - 1, 1), np.nan)
        c = np.where(np.isfinite(c) & (c > 0), c, np.nan)
        # A bin whose arm has one usable cell -- or none that differ -- gives
        # no residual to estimate c from.  Falling back to NaN would discard
        # the BIN, not just the estimate, and at 250 kb in a sparse context
        # that is most of the genome.  The median c over the bins that could
        # be estimated is the honest stand-in: it says "as noisy as this arm
        # usually is" rather than "unmeasurable".
        pooled = np.nanmedian(c) if np.isfinite(c).any() else np.nan
        c = np.where(np.isfinite(c), c, pooled)
        V[m] = c[None, :] / ni
    return V


def _dl_tau2(x, v, axis=0):
    """DerSimonian-Laird between-unit variance, per bin.

    Each (cell, bin) estimate carries its own sampling variance from the
    pipeline, so the spread between cells has two parts: measurement error,
    which is known, and real biological variation, which is not.  tau^2 is the
    second.  Closed form, so it runs over tens of thousands of bins without
    fitting anything iteratively.
    """
    w = 1.0 / v
    sw = np.nansum(w, axis=axis)
    m = np.nansum(w * x, axis=axis) / sw
    q = np.nansum(w * (x - m) ** 2, axis=axis)
    k = np.sum(np.isfinite(x) & np.isfinite(v), axis=axis)
    denom = sw - np.nansum(w ** 2, axis=axis) / sw
    with np.errstate(invalid="ignore", divide="ignore"):
        tau2 = (q - (k - 1)) / denom
    return np.where(np.isfinite(tau2) & (tau2 > 0), tau2, 0.0)


def _weighted_arm(x, v, tau2=None):
    """Inverse-variance pooled mean and its variance, per bin.

    With tau2 this is the random-effects estimate: a cell with few read pairs
    stops dominating on noise, and a cell with many stops dominating on
    precision alone once real between-cell variation is accounted for.
    """
    vv = v if tau2 is None else v + tau2
    w = 1.0 / vv
    w = np.where(np.isfinite(w) & np.isfinite(x), w, np.nan)
    sw = np.nansum(w, axis=0)
    m = np.nansum(w * x, axis=0) / sw
    return m, 1.0 / sw


def weighted_contrast(X, V, in_mask, out_mask, min_cells, mode="random-effects"):
    """Two-arm comparison weighting each cell by the precision of its estimate.

    D is not measured, it is ESTIMATED from the read pairs in a (cell, bin),
    and the pipeline records the variance of that estimate.  An unweighted
    test treats a cell with five pairs in a bin as being worth as much as one
    with five hundred.  That is wasteful when coverage merely varies, and
    actively misleading when it varies BETWEEN THE ARMS -- a noisier estimate
    is an attenuated one, so a coverage difference becomes a genome-wide
    difference in D with no biology behind it.

    'precision' is the fixed-effect form, which assumes every cell in an arm
    shares one true value and is anticonservative when they do not.
    'random-effects' adds the DerSimonian-Laird between-cell variance, so a
    bin that genuinely varies within an arm is not called on the strength of
    its coverage.  Returns the same keys as contrast_stats.
    """
    X = np.asarray(X, float)
    V = np.asarray(V, float)
    V = np.where(np.isfinite(V) & (V > 0), V, np.nan)
    out = {}
    means, varis = [], []
    for tag, m in (("in", in_mask), ("out", out_mask)):
        x, v = X[m], V[m]
        n = np.sum(np.isfinite(x) & np.isfinite(v), axis=0)
        tau2 = _dl_tau2(x, v) if mode == "random-effects" else None
        mu, vr = _weighted_arm(x, v, tau2)
        out[f"n_{tag}"] = n
        out[f"mean_{tag}"] = mu
        out[f"var_{tag}"] = vr
        out[f"tau2_{tag}"] = tau2 if tau2 is not None else np.zeros_like(mu)
        means.append(mu)
        varis.append(vr)
    diff = means[0] - means[1]
    se = np.sqrt(varis[0] + varis[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        z = diff / se
    from scipy.stats import norm
    p = 2.0 * norm.sf(np.abs(z))
    bad = ((out["n_in"] < min_cells) | (out["n_out"] < min_cells)
           | ~np.isfinite(se) | (se <= 0))
    out["diff"] = np.where(bad, np.nan, diff)
    out["t"] = np.where(bad, np.nan, z)
    out["p"] = np.where(bad, np.nan, p)
    # Standardised on the between-cell SD, so it stays comparable with the
    # unweighted Cohen's d rather than exploding as the pooled SE shrinks.
    sd = np.sqrt(np.maximum(out["tau2_in"] + out["tau2_out"], 1e-300) / 2.0)
    out["d"] = np.where(bad | (sd <= 0), np.nan, diff / sd)
    return out


def _logit(x, eps=1e-4):
    """logit on the D scale, with the out-of-range values pinned.

    D is a Pearson r, so sampling noise puts some bins at or below 0 where the
    logit does not exist.  Those are bins with no reliable signal, and pinning
    them to eps keeps them in the matrix at the floor rather than turning them
    into NaN, which would silently change WHICH bins are testable between the
    corrected and uncorrected runs -- the one difference that would make the
    two runs incomparable.
    """
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    v = np.clip(x[ok], eps, 1.0 - eps)
    out[ok] = np.log(v / (1.0 - v))
    return out


def _expit(z):
    z = np.asarray(z, float)
    out = np.full(z.shape, np.nan)
    ok = np.isfinite(z)
    out[ok] = 1.0 / (1.0 + np.exp(-z[ok]))
    return out


def correct_by_reference(X, ref_d, eps=1e-4):
    """Divide out each cell's assay reliability, measured on a reference.

    D is a Pearson r between the posteriors at the two ends of a read pair,
    which under a noise model is a RELIABILITY coefficient rather than a
    structural correlation:

        D = S / (S + N),   S = e^2 var_A,   N = Vbar / n_sites

    with ``e`` the GpC methyltransferase efficiency in that cell.  On the
    logit scale that is additive,

        logit(D) = 2 log e + log var_A + log n_sites - log Vbar

    so subtracting the logit of a reference D measured in the SAME cell
    cancels the enzyme term exactly (to first order -- Vbar depends on e too,
    but that is second order beside 2 log e).

    This is why the transform is logit and not Fisher z: Fisher z stabilises
    the variance of a sampling correlation, which is a different goal.  Logit
    is the scale on which the attenuation being removed is additive.

    Cells with no reference (too few pairs, or a degenerate one) keep their
    uncorrected values.  Correcting some cells and not others would put two
    scales in one matrix, so the grand mean is added back from the corrected
    cells only and the untouched cells are reported.
    """
    X = np.asarray(X, float)
    ref = np.asarray(ref_d, float)
    if ref.shape[0] != X.shape[0]:
        raise ValueError(f"reference has {ref.shape[0]} cells, "
                         f"matrix has {X.shape[0]}")
    # This transform is logit, and logit is only defined on (0, 1).  _logit
    # PINS everything outside to the floor, which is the right thing for a
    # Pearson r whose sampling noise dips below zero and a disaster for a
    # matrix that legitimately spans both signs: direction_logor is a log
    # odds ratio centred near zero, so half of it would collapse onto the
    # single value logit(eps).  Not an approximation -- the whole negative
    # half becomes one number.  Refuse rather than return it.
    fin = np.isfinite(X)
    frac_out = (float(np.mean((X[fin] < 0.0) | (X[fin] > 1.0)))
                if fin.any() else 0.0)
    med = float(np.median(X[fin])) if fin.any() else float("nan")
    # A Pearson r whose sampling noise dips below zero is the case this was
    # built for: on a simulated reliability matrix 12% of bins land below 0
    # while the median sits at +0.26, and flooring those is right -- they are
    # bins with no reliable signal.  A statistic that is genuinely SIGNED is
    # a different animal: direction_logor sits at 50% and a median of ~0,
    # D = C(2F-1) came back 100% negative.  There the clip is not a floor
    # under a tail, it is the whole matrix, and each cell collapses to one
    # value -- measured: bin-to-bin sd 0.153 -> 2.7e-20, one distinct value
    # per cell, frac_at_floor 1.0, with the run reporting success.
    if frac_out >= 0.30 or (fin.any() and med <= eps):
        raise ValueError(
            f"the reference correction is logit-based and assumes a "
            f"reliability coefficient in (0, 1); this matrix has "
            f"{frac_out:.1%} of its values outside that range and a median "
            f"of {med:+.6g}, so the clip inside logit would flatten it -- "
            f"on a wholly-signed matrix every bin in a cell collapses to a "
            f"single value.  The attenuation enters as e^2, so the right "
            f"operation follows the scale of the statistic: subtract "
            f"logit(ref) for a Pearson r, DIVIDE by a reference C for a "
            f"covariance (coordination), and for direction_logor it is "
            f"multiplicative on the log odds ratio -- none of which is what "
            f"this does.  Drop --contrast-reference-d for this matrix; "
            f"--contrast-center is scale-free and still applies.")
    usable = np.isfinite(ref) & (ref > 0)
    info = {"n_cells": int(X.shape[0]), "n_corrected": int(usable.sum()),
            "n_uncorrected": int((~usable).sum())}
    if not usable.any():
        info["note"] = "no cell had a usable reference"
        return X, info
    finite = np.isfinite(X)
    info["frac_at_floor"] = float(np.mean(X[finite] <= eps)) if finite.any() \
        else float("nan")
    z = _logit(X, eps)
    zr = _logit(ref, eps)
    out = np.array(z, copy=True)
    out[usable] = z[usable] - zr[usable, None] + float(np.nanmean(zr[usable]))
    corrected = _expit(out)
    # cells without a reference are returned exactly as they came in, not
    # round-tripped through logit/expit, so eps-clipping cannot move them
    corrected[~usable] = X[~usable]
    info["reference_d_median"] = float(np.nanmedian(ref[usable]))
    info["reference_d_min"] = float(np.nanmin(ref[usable]))
    info["reference_d_max"] = float(np.nanmax(ref[usable]))
    return corrected, info


def scale_by_reference(X, ref, min_frac=0.1, zero_floor=1e-6):
    """Rescale each cell so its reference matches the cohort median.

    For an EFFECT SIZE the attenuation is multiplicative, not additive on a
    logit: C carries e^2 var_A, so C_cell / C_reference is free of the
    enzyme term whatever the cell's marginal.  Measured on a simulated
    reference pool, C fell by 0.49x and 0.16x as e went 1.0 -> 0.7 -> 0.4 --
    exactly e^2 -- and it did so at every marginal skew tested, which is what
    makes a reference taken at chrM's marginal valid for a genome at a
    different one.

    THE SAME IS NOT TRUE OF A LOG ODDS RATIO, and that is the whole reason
    this function takes a column rather than assuming one.  L's attenuation
    depends on the marginal as well as on e: at skew -1.5 it fell 0.29x for
    the same enzyme drop that took it 0.12x at skew -3.  A reference measured
    on chrM's marginal therefore does NOT cancel the nuclear L's attenuation,
    and where chrM's marginal is near symmetric L_ref is ~0 (measured:
    -0.0003) and dividing by it is meaningless rather than merely imprecise.

    Cells whose reference is not distinguishable from zero are left alone.
    Dividing by a noisy near-zero number does not just add variance, it flips
    signs, so a bad reference is worse than none.  A cell whose reference is
    under ``min_frac`` of the cohort median, or of the opposite sign, is left
    as it came in and reported.
    """
    X = np.asarray(X, float)
    ref = np.asarray(ref, float)
    if ref.shape[0] != X.shape[0]:
        raise ValueError(f"reference has {ref.shape[0]} cells, "
                         f"matrix has {X.shape[0]}")
    finite = np.isfinite(ref)
    centre = float(np.median(ref[finite])) if finite.any() else float("nan")
    info = {"n_cells": int(X.shape[0]), "reference_median": centre,
            "min_frac_of_median": float(min_frac)}
    # The cohort's SPREAD is the thing being divided out, so it is not the
    # scale to judge "distinguishable from zero" against.  What matters is
    # whether the reference has a definite magnitude and sign at all: a
    # reference centred on zero -- which is what a log odds ratio on chrM is
    # when the marginal is near symmetric, measured at -0.0003 -- makes the
    # ratio meaningless rather than merely imprecise.
    spread = _robust_sd(ref[finite]) if finite.sum() > 1 else float("nan")
    info["reference_spread"] = float(spread)
    # Two ways a reference is unusable, and both are the same failure: the
    # denominator has no definite value.  Either it sits on zero outright, or
    # its spread across cells swamps its own median -- a coefficient of
    # variation at or above 1 means the sign of the next cell's reference is
    # close to a coin flip.
    if (not np.isfinite(centre) or abs(centre) < float(zero_floor)
            or (np.isfinite(spread) and spread >= abs(centre))):
        info["note"] = (f"the reference is centred on {centre:+.3g} with a "
                        f"spread of {spread:.3g}, so it has no definite "
                        f"magnitude or sign; dividing by it would amplify "
                        f"noise and flip signs, and nothing was rescaled")
        info.update(n_corrected=0, n_uncorrected=int(X.shape[0]))
        return X, info
    # Same sign as the cohort, and not a small fraction of it.  A cell whose
    # reference is a tenth of everyone else's would be multiplied up tenfold,
    # and one of the opposite sign would have its whole profile inverted.
    usable = (finite & (np.sign(ref) == np.sign(centre))
              & (np.abs(ref) >= float(min_frac) * abs(centre)))
    info.update(n_corrected=int(usable.sum()),
                n_uncorrected=int((~usable).sum()))
    if not usable.any():
        info["note"] = "no cell had a usable reference; nothing was rescaled"
        return X, info
    out = np.array(X, copy=True)
    out[usable] = X[usable] * (centre / ref[usable, None])
    info.update({"reference_min": float(np.min(ref[usable])),
                 "reference_max": float(np.max(ref[usable]))})
    return out, info


def _robust_sd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    mad = float(np.median(np.abs(x - np.median(x))))
    return 1.4826 * mad if mad > 0 else float(np.std(x))


def regress_out_coverage(X, counts, n_strata=24):
    """Subtract the shared relationship between D and sequencing depth.

    D is estimated from the read pairs in a (cell, bin), and a shallow
    estimate is pulled toward the prior, so D carries a monotone function of
    depth that has nothing to do with biology.  Where the two arms differ in
    depth bin by bin, that function becomes a bin-specific difference -- and
    a within-cell permutation null KEEPS it, because it genuinely is one.
    Shuffling a cell's values across bins cannot break a confound that is
    itself indexed by bin.

    The curve is fitted ONCE over every (cell, bin) rather than per bin, and
    that is the load-bearing choice.  Within a single bin, coverage is
    collinear with the arm whenever the tumour cells are the thin ones, so a
    per-bin fit cannot separate depth from biology and would subtract the
    finding along with the artefact.  Pooled, the curve is estimated across
    the whole matrix, where depth varies for reasons unrelated to any one
    contrast, and a real per-bin effect survives it.

    Non-parametric: log2 coverage is cut into quantile strata and D is
    centred within each, so no functional form is imposed on an attenuation
    whose shape depends on a prior nobody measured.  The grand mean is added
    back so the result stays on the D scale.
    """
    X = np.asarray(X, float)
    n = np.asarray(counts, float)
    x = np.log2(np.where(np.isfinite(n) & (n > 0), n, np.nan))
    ok = np.isfinite(X) & np.isfinite(x)
    if int(ok.sum()) < 100:
        return X, {"n_strata": 0, "note": "too few covered bins"}
    edges = np.unique(np.nanpercentile(x[ok],
                                       np.linspace(0, 100, n_strata + 1)))
    if edges.size < 3:
        return X, {"n_strata": 0, "note": "coverage does not vary"}
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0,
                  edges.size - 2)
    out = np.array(X, copy=True)
    grand = float(np.nanmean(X[ok]))
    # Track what was actually centred.  A stratum too thin to give a stable
    # mean is left alone, and the grand mean must NOT be added back to it --
    # doing so would shift those values by ~the mean of D while leaving them
    # uncorrected, which is worse than not touching them at all.
    done = np.zeros(X.shape, bool)
    used = 0
    for k in range(edges.size - 1):
        sel = ok & (idx == k)
        if int(sel.sum()) > 10:
            out[sel] = X[sel] - float(np.nanmean(X[sel]))
            done |= sel
            used += 1
    if not used:
        return X, {"n_strata": 0, "note": "no stratum had enough observations"}
    out[done] += grand
    before = float(np.nanstd(X[done]))
    after = float(np.nanstd(out[done]))
    return out, {"n_strata": used, "sd_before": before, "sd_after": after,
                 "frac_corrected": float(done.sum()) / float(ok.sum()),
                 "frac_variance_removed": (1.0 - (after / before) ** 2
                                           if before > 0 else float("nan"))}


def coverage_confound(counts, in_mask, out_mask, diff):
    """Does the contrast track the per-bin COVERAGE gap between the arms?

    The failure a permutation null cannot catch.  Permuting bins within a cell
    absorbs anything whole-cell -- an offset, a purity shift, a uniform depth
    difference -- because those survive the shuffle.  A coverage gap that
    varies BY BIN does not: mappability, GC and capture make one arm relatively
    thinner in some bins than others, D is estimated from the pairs in a bin,
    and shallower estimates are pulled toward the prior.  That is a genuine
    bin-specific difference, so permutation keeps it, and it is technical.

    Its signature is unmistakable once looked for: an effect that is almost
    entirely one-directional, tiny in size, and correlated with the coverage
    gap.  On a fixture with identical biology everywhere and only a bin-varying
    gap: 1,431 of 3,000 bins called, 1,431 down and 0 up, r = 0.97.
    """
    n = np.asarray(counts, float)
    n = np.where(np.isfinite(n) & (n > 0), n, np.nan)
    with np.errstate(invalid="ignore"):
        a = np.nanmean(n[np.asarray(in_mask, bool)], axis=0)
        b = np.nanmean(n[np.asarray(out_mask, bool)], axis=0)
        gap = np.log2((a + 1.0) / (b + 1.0))
    d = np.asarray(diff, float)
    ok = np.isfinite(gap) & np.isfinite(d)
    if int(ok.sum()) < 10:
        return {}
    if float(np.nanstd(gap[ok])) < 1e-12:
        # The arms are matched bin for bin, so there is no gap to correlate
        # with.  Reporting NaN silently would read as "the check ran and
        # found nothing" when it is "there was nothing to check".
        return {"pearson_r_effect_vs_coverage_gap": np.nan,
                "spearman_rho_effect_vs_coverage_gap": np.nan,
                "median_log2_coverage_gap": float(np.nanmedian(gap)),
                "n_bins": int(ok.sum()), "gap_is_constant": True}
    from scipy.stats import pearsonr, spearmanr
    r = float(pearsonr(gap[ok], d[ok])[0])
    rho = float(spearmanr(gap[ok], d[ok])[0])
    return {"pearson_r_effect_vs_coverage_gap": r,
            "spearman_rho_effect_vs_coverage_gap": rho,
            "median_log2_coverage_gap": float(np.nanmedian(gap)),
            "n_bins": int(ok.sum())}


def direction_skew(diff, sig):
    """How one-sided the significant bins are.

    Real regional biology goes both ways somewhere.  A result that is 99%
    one-directional is usually a monotone technical gradient -- attenuation
    pulls toward the prior and never away from it.
    """
    d = np.asarray(diff, float)
    m = np.asarray(sig, bool) & np.isfinite(d)
    n = int(m.sum())
    if not n:
        return {}
    up = int(np.sum(d[m] > 0))
    return {"n_significant": n, "n_up": up, "n_down": n - up,
            "frac_one_direction": max(up, n - up) / n}


def permute_within_cells(X, rng, permutable=None):
    """Shuffle each cell's values across bins, keeping its NaN pattern.

    Destroys WHICH bin a value belongs to while preserving everything about
    the cell: its offset, its spread, and which bins it covers.  So a
    difference that survives this is not bin-specific -- it is a property of
    the cells, which is what a genome-wide batch or purity shift is.

    The NaN positions stay put deliberately.  Moving them would shuffle
    coverage as well, and coverage differs between bins for reasons that have
    nothing to do with the contrast; holding it fixed keeps the null about bin
    IDENTITY alone.

    THAT PROTECTION REACHES NOTHING ON A POST-RWR MATRIX.  RWR fills every
    (cell, bin), so concordance_direction_logor.parquet and its siblings
    carry 0% NaN whatever a cell's real coverage was -- there is no pattern
    left to hold, and every entry is shuffled as a measurement.

    And the filled values are not neutral.  RWR borrows across the k-NN
    graph, which is built from accessibility and is therefore cell-type
    aware, so an imputed entry carries its arm's signature BY CONSTRUCTION.
    A sparsely covered bin is then mostly arm signature; the permuted
    column at that position holds each cell's genome-wide MIX of measured
    and imputed instead, which dilutes it.  The observed clears a null that
    no longer describes it.  Measured on a fixture with no signal at all:
    57.7% of sparse bins called at p < 0.05, and 0% once the entries with
    no read pairs of their own were masked.

    A rank test is why the same argument does NOT go through for structural
    zeros -- ties do not move a Mann-Whitney, and four fixtures built that
    way came back at a nominal 0.040.  It is the cell-type tint that does
    the damage, not the fill.

    ``permutable`` restores the defence without discarding anything.  Pass
    the entries that have read pairs of THEIR OWN and the imputed ones stay
    where they are, in the null exactly as in the observed -- so a bin's
    imputation-borne arm signature is present on both sides and the bin
    only clears by differing MORE than the imputation explains.  That is
    the same argument the whole function rests on, applied to a second
    nuisance.  It also keeps every bin testable, which masking the imputed
    entries outright does not: on the fixture above masking left 6 sparse
    bins of 300, while holding them fixed left all of them and took the
    false-positive rate from 0.577 to 0.077.
    """
    out = np.array(X, dtype=float, copy=True)
    ok = np.isfinite(out)
    if permutable is not None:
        ok &= np.asarray(permutable, bool)
    for i in range(out.shape[0]):
        idx = np.flatnonzero(ok[i])
        if idx.size > 1:
            out[i, idx] = out[i, rng.permutation(idx)]
    return out


# Fixed bands, not quantiles.  Over a mostly sparse genome the quantile
# edges fall inside the sparse class and put its better-covered bins in with
# the fully measured ones, which poisons the pool -- measured as power 0.00
# where fixed bands gave 1.00.  These separate the classes by construction.
COVERAGE_BANDS = (0.25, 0.5, 0.75, 0.999)


def coverage_strata(permutable, bands=COVERAGE_BANDS):
    """Bin each column by the fraction of cells that measured it."""
    frac = np.asarray(permutable, bool).mean(axis=0)
    return np.digitize(frac, list(bands))


def permutation_null(X, in_mask, out_mask, min_cells, test, n_perm,
                     V=None, mode="none", strata=None, counts=None, seed=0,
                     permutable=None, pool_strata=None, min_pool=5_000):
    """Empirical p per bin against a within-cell permutation null.

    The parametric p asks whether the arms differ.  It cannot tell a
    bin-specific difference from a whole-cell one, so a genome-wide offset --
    batch, purity, coverage -- makes every bin significant, which is exactly
    what a real run produced.  Permuting bins WITHIN each cell leaves that
    offset untouched and removes bin identity, so the null it generates
    already contains the offset, and a bin only clears it by differing MORE
    than the cell-level shift explains.

    The null is pooled over bins and permutations: after permutation the bins
    are exchangeable, so every permuted statistic is a draw from the same null
    and pooling buys n_bins x n_perm draws instead of n_perm.  That matters --
    a per-bin null of 100 permutations cannot resolve a p below 0.01.

    ``pool_strata`` IS WHAT MAKES THAT EXCHANGEABILITY TRUE when coverage
    varies.  With ``permutable`` holding a bin's imputed entries in place,
    a sparsely covered bin's permuted statistic is close to its observed one
    -- which is the point, it is how the bin stops clearing a null that does
    not describe it -- but it also means such a bin contributes ENORMOUS
    draws to a shared pool.  Measured: sparse bins at |t| median 9.3 against
    a dense-only null whose 99.9th percentile is 3.67.  Pool them together
    and a real effect at |t| 5.3 does not clear, so power goes to zero
    everywhere while the run still reports thousands of parametric calls.

    Pooling within a coverage stratum restores it: on the same fixture,
    power 1.00 at the planted bins with false-positive rates of 0.032 and
    0.057 in the two classes, against 0.00 / 0.000 / 0.057 unstratified.
    The bands have to SEPARATE the classes -- quantile edges over a mostly
    sparse genome put the better-covered sparse bins in with the dense ones
    and the pool is poisoned again, which measured as power 0.00.
    """
    rng = np.random.default_rng(seed)

    def stat_of(mat):
        if strata is not None:
            cs, _ = stratified_contrast(mat, strata, in_mask, out_mask,
                                        min_cells, test, V, mode,
                                        counts=counts, min_strata=1)
            if cs is None:
                cs = (weighted_contrast(mat, V, in_mask, out_mask, min_cells,
                                        mode) if V is not None
                      else contrast_stats(mat, in_mask, min_cells, out_mask,
                                          test))
        elif V is not None:
            cs = weighted_contrast(mat, V, in_mask, out_mask, min_cells, mode)
        else:
            cs = contrast_stats(mat, in_mask, min_cells, out_mask, test)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.abs(np.asarray(cs["t"], float)), cs

    obs, cs_obs = stat_of(np.asarray(X, float))
    draws = []
    for _ in range(int(n_perm)):
        v, _ = stat_of(permute_within_cells(X, rng, permutable))
        draws.append(np.asarray(v, float))
    if not draws:
        return None
    D = np.vstack(draws)                       # n_perm x n_bins, bins kept
    if D[np.isfinite(D)].size == 0:
        return None

    # +1 in numerator and denominator: an observed value beyond every draw
    # gets 1/(N+1), not zero, because N draws cannot evidence p = 0.
    def _p_from(pool, where):
        pool = np.sort(pool[np.isfinite(pool)])
        if pool.size == 0:
            return None
        rank = np.searchsorted(pool, obs[where], side="left")
        return (pool.size - rank + 1.0) / (pool.size + 1.0), pool.size

    emp = np.full(obs.shape, np.nan)
    n_used, n_strat = 0, 0
    if pool_strata is None:
        r = _p_from(D.ravel(), slice(None))
        if r is None:
            return None
        emp, n_used = r[0], r[1]
        n_strat = 1
    else:
        ps = np.asarray(pool_strata)
        globally = np.sort(D[np.isfinite(D)].ravel())
        for s in np.unique(ps[np.isfinite(ps.astype(float))]
                           if ps.dtype.kind == "f" else np.unique(ps)):
            col = ps == s
            if not col.any():
                continue
            r = _p_from(D[:, col].ravel(), col)
            # A stratum too thin to resolve the tail falls back to the whole
            # pool, which is conservative, rather than to a handful of draws.
            if r is None or r[1] < min_pool:
                rank = np.searchsorted(globally, obs[col], side="left")
                emp[col] = ((globally.size - rank + 1.0)
                            / (globally.size + 1.0))
            else:
                emp[col] = r[0]
                n_used += r[1]
                n_strat += 1
        if n_used == 0:
            n_used = int(globally.size)
    emp = np.where(np.isfinite(obs), emp, np.nan)
    return {"stat": obs, "p_empirical": emp, "n_null": int(n_used),
            "n_perm": int(n_perm), "n_pool_strata": int(n_strat),
            "cs": cs_obs}


def batch_negative_control(X, ct, batch, min_cells, test, fdr_cut,
                           V=None, mode="none", counts=None):
    """Run the contrast with the arms defined by BATCH instead of biology.

    The decisive check that a confounding statistic cannot give.  Cramer's V
    says whether the contrast and a batch are associated; --max-batch-icc is a
    within-cell-type statistic and reports nothing when a batch is nested
    inside cell type.  Neither answers the question actually being asked --
    how many bins would this pipeline call at this FDR if the arms differed
    ONLY by processing?

    So: same matrix, same test, same weighting, same correction, arms taken
    from the batch column, stratified by cell type so the comparison is
    within-cell-type and cannot be biology.  The count that comes back is the
    number of bins this design yields from processing alone.  Compared with
    the real contrast's count it is either reassuring or fatal, and it needs
    no assumption about how batch and biology relate.
    """
    levels = [v for v in pd.unique(batch) if v == v]
    if len(levels) < 2:
        return None
    counts_by = pd.Series(batch).value_counts()
    top = [v for v in counts_by.index[:2]]
    a = (batch == top[0])
    b = (batch == top[1])
    if int(a.sum()) < min_cells or int(b.sum()) < min_cells:
        return None
    # Stratified by cell type, so only cell types present in BOTH batches
    # contribute -- which is exactly a bridge sample if you have one.  A cell
    # type split across two experiments measures the batch effect with the
    # biology held constant, and is the best estimate of it available.
    comb, per = stratified_contrast(X, np.asarray(ct), a, b, min_cells,
                                    test, V, mode, counts=counts,
                                    min_strata=1)
    if comb is None:
        # NOT falling back to an unstratified comparison.  Where a batch is
        # nested inside cell type, comparing the batches directly compares the
        # cell types too, and the count that comes back is biology wearing a
        # batch label -- the exact confusion this control exists to resolve.
        return {"levels": [str(top[0]), str(top[1])],
                "n_cells": [int(a.sum()), int(b.sum())],
                "n_significant": None, "n_tested": 0,
                "n_strata": 0, "strata": []}
    q = _bh_fdr(comb["p"])
    sig = int(np.sum(np.isfinite(q) & (q < fdr_cut)))
    return {"levels": [str(top[0]), str(top[1])],
            "n_cells": [int(a.sum()), int(b.sum())],
            "n_significant": sig,
            "n_tested": int(np.isfinite(comb["p"]).sum()),
            "n_strata": len(per), "strata": sorted(per)}


def dropout_report(X, in_mask, out_mask, min_cells, strata=None):
    """Why bins are not testable, counted at each requirement in turn.

    A contrast can look weak for two very different reasons -- the effect is
    small, or most of the genome was never tested -- and the selected count
    alone cannot tell them apart.  Stratifying multiplies the requirement:
    each arm needs min_cells covered cells in EVERY stratum, so two arms and
    two patients is four hurdles, and a sparse context clears far fewer of
    them than a dense one.
    """
    X = np.asarray(X, float)
    ok = np.isfinite(X)
    out = {"n_bins": X.shape[1]}
    ain = ok[np.asarray(in_mask, bool)].sum(axis=0)
    aout = ok[np.asarray(out_mask, bool)].sum(axis=0)
    out["n_testable_pooled"] = int(((ain >= min_cells)
                                    & (aout >= min_cells)).sum())
    out["median_cells_in"] = float(np.median(ain))
    out["median_cells_out"] = float(np.median(aout))
    if strata is not None:
        strata = np.asarray(strata)
        per_ok = []
        for lev in [x for x in pd.unique(strata) if x == x]:
            k = strata == lev
            a = ok[np.asarray(in_mask, bool) & k].sum(axis=0)
            b = ok[np.asarray(out_mask, bool) & k].sum(axis=0)
            hit = (a >= min_cells) & (b >= min_cells)
            per_ok.append(hit)
            out[f"n_testable_{lev}"] = int(hit.sum())
        if per_ok:
            out["n_testable_all_strata"] = int(np.all(per_ok, axis=0).sum())
    return out


def stratified_contrast(X, strata, in_mask, out_mask, min_cells, test,
                        V=None, mode="random-effects", min_strata=2,
                        counts=None):
    """Run the contrast WITHIN each stratum, then combine across them.

    The right treatment for a blocking factor that is biological rather than
    technical -- patient above all.  Comparing tumour against monocyte inside
    one patient removes that patient's whole contribution exactly, including
    the parts no covariate would capture (CNV background, purity, tissue
    handling), because both arms carry it equally.  Adjusting for patient as a
    covariate assumes its effect is a constant added to every cell, which for
    two patients is neither testable nor likely.

    Combination is Stouffer's weighted z on the per-stratum results: it works
    for whichever per-bin test was chosen, including a rank test that has no
    standard error to pool.  Weights are sqrt(n) per stratum, so a patient
    contributing twice the cells carries more evidence but not proportionally
    more, which is right when the cells inside a patient are correlated.

    Returns (combined, per_stratum).  ``combined['diff']`` is the mean of the
    per-stratum differences: an average of within-patient effects, which is
    the estimand here, rather than a pooled difference that a patient
    imbalance would tilt.
    """
    from scipy.stats import norm

    strata = np.asarray(strata)
    per, zs, ws, diffs = {}, [], [], []
    for lev in [x for x in pd.unique(strata) if x == x]:
        keep = strata == lev
        a, b = in_mask & keep, out_mask & keep
        if int(a.sum()) < min_cells or int(b.sum()) < min_cells:
            continue
        # The variance approximated from counts must be estimated INSIDE the
        # stratum.  Estimated over the whole arm it absorbs the between-strata
        # difference -- the patient effect -- into the noise constant, and
        # then inflates every within-patient comparison by it, which is
        # exactly the variation stratifying was meant to remove.
        Vs = V
        if V is None and counts is not None:
            Vs = variance_from_counts(counts, X, a, b)
        if Vs is not None:
            cs = weighted_contrast(X, Vs, a, b, min_cells, mode)
        else:
            cs = contrast_stats(X, a, min_cells, b, test)
        per[str(lev)] = cs
        with np.errstate(invalid="ignore"):
            z = np.sign(cs["diff"]) * norm.isf(np.clip(cs["p"], 1e-300, 1) / 2)
        zs.append(z)
        diffs.append(cs["diff"])
        ws.append(np.sqrt(min(int(a.sum()), int(b.sum()))))
    if len(per) < min_strata:
        return None, per
    Z = np.vstack(zs)
    D = np.vstack(diffs)
    w = np.asarray(ws, float)[:, None]
    ok = np.isfinite(Z).all(axis=0) & np.isfinite(D).all(axis=0)
    with np.errstate(invalid="ignore"):
        zc = np.nansum(w * Z, axis=0) / np.sqrt(np.nansum(w ** 2, axis=0))
    combined = {
        "diff": np.where(ok, np.nanmean(D, axis=0), np.nan),
        "t": np.where(ok, zc, np.nan),
        "p": np.where(ok, 2.0 * norm.sf(np.abs(zc)), np.nan),
        "n_strata": len(per),
        "n_in": np.nansum([per[k]["n_in"] for k in per], axis=0),
        "n_out": np.nansum([per[k]["n_out"] for k in per], axis=0)}
    combined["d"] = combined["diff"]
    return combined, per


def group_consistency(X, ct, groups, out_mask, min_cells, test, V=None,
                      mode="random-effects"):
    """Run the contrast once per group in the arm; return per-group diffs.

    With two tumour populations pooled into one arm, a bin can reach
    significance on one of them alone.  That is not a meningioma signature, it
    is one population's idiosyncrasy, and pooling hides which.  Testing each
    separately against the same reference and asking for agreement is a
    REPRODUCIBILITY criterion -- weaker than a p-value in theory, and stronger
    than one here, where the p-value's null (independent cells) is false.
    """
    per = {}
    for g in groups:
        m = (ct == g).to_numpy() if hasattr(ct, "to_numpy") else (ct == g)
        if int(m.sum()) < min_cells:
            continue
        if V is not None:
            cs = weighted_contrast(X, V, m, out_mask, min_cells, mode)
        else:
            cs = contrast_stats(X, m, min_cells, out_mask, test)
        per[g] = cs
    return per


def consistency_mask(per_group, fdr_cut=0.05):
    """Bins that move the SAME way in every group, significantly in each."""
    if not per_group:
        return None, {}
    signs, sig = [], []
    for g, cs in per_group.items():
        q = _bh_fdr(cs["p"])
        signs.append(np.sign(cs["diff"]))
        sig.append(np.isfinite(q) & (q < fdr_cut))
    signs = np.vstack(signs)
    sig = np.vstack(sig)
    same = np.all(signs == signs[0], axis=0) & np.all(signs != 0, axis=0)
    everywhere = np.all(sig, axis=0)
    mask = same & everywhere
    return mask, {"n_consistent": int(mask.sum()),
                  "n_significant_in_any": int(np.any(sig, axis=0).sum()),
                  "n_significant_in_all": int(everywhere.sum()),
                  "groups": list(per_group)}


def coverage_report(counts, in_mask, out_mask):
    """Per-arm read-pair coverage, the first thing to check on a global offset.

    D is estimated from the pairs in a (cell, bin), so a coverage difference
    between the arms attenuates one side's estimates more than the other and
    shows up as a genome-wide difference in D with no biology under it.  If
    the arms are matched here, a global offset is more likely to be real.
    """
    c = np.asarray(counts, float)
    out = {}
    for tag, m in (("in", in_mask), ("out", out_mask)):
        v = c[m]
        v = v[np.isfinite(v)]
        out[f"median_pairs_{tag}"] = float(np.median(v)) if v.size else np.nan
    a, b = out["median_pairs_in"], out["median_pairs_out"]
    out["pairs_ratio"] = (a / b) if b else np.nan
    return out


def _check_companion(df, matrix, src, flag, min_overlap=0.95):
    """A companion must cover the matrix it describes.

    Reindexing a mismatched file onto the matrix succeeds and fills the gaps
    with NaN, which is the worst possible outcome: every cell it does not
    cover silently loses its variance, drops out of the contrast, and the run
    reports a handful of testable bins with no indication that a stale file is
    the reason.  A file from another run -- different cells, a different bin
    size, a rerun with new filters -- looks identical to a correct one until
    the results are inspected closely enough to notice they are impossible.
    """
    cells = len(matrix.index.intersection(df.index)) / max(len(matrix.index), 1)
    bins = (len(matrix.columns.intersection(df.columns))
            / max(len(matrix.columns), 1))
    if cells < min_overlap or bins < min_overlap:
        sys.exit(
            f"{flag}: {src} does not match the matrix it is meant to "
            f"describe.\n"
            f"  it covers {cells:.0%} of the matrix's {len(matrix.index):,} "
            f"cells and {bins:.0%} of its {len(matrix.columns):,} bins "
            f"(the file itself is {df.shape[0]:,} x {df.shape[1]:,}).\n"
            f"  every cell or bin it does not cover would lose its variance "
            f"and drop out of the contrast, leaving a handful of testable "
            f"bins and nothing to say why.\n"
            f"  this is usually a file left from an earlier run with "
            f"different cells, a different bin size, or different filters.  "
            f"Re-run the concordance step, or point {flag} at the file that "
            f"belongs to this matrix.")
    return df


def per_matrix(specs, name, cast, default, flag):
    """A threshold that may be global or named per --matrix.

    GCH and HCG do not want the same cut: one is dense enough that a
    coverage filter barely bites, the other is mostly imputed.  A single
    scalar forces the looser of the two on both, and the looser one is the
    one that lets imputation-driven bins through.

    A bare value applies to every matrix; NAME=VALUE overrides it for that
    matrix.  The last matching NAME wins, and a bare value given after a
    NAME= still only sets the default.
    """
    value = default
    for spec in (specs or []):
        text = str(spec)
        if "=" in text:
            k, v = text.split("=", 1)
            if k.strip() == name:
                value = cast(v.strip())
        else:
            try:
                value = cast(text.strip())
            except ValueError:
                sys.exit(f"{flag} wants a number or NAME=number, got "
                         f"{spec!r}")
    return value


def _companion(matrix_path, specs, name, filename, flag, matrix=None):
    """The named companion parquet, or the one sitting beside the matrix."""
    for spec in (specs or []):
        if "=" not in spec:
            sys.exit(f"{flag} must be NAME=PATH, got {spec!r}")
        k, v = spec.split("=", 1)
        if k.strip() == name:
            pth = Path(v.strip())
            if not pth.exists():
                sys.exit(f"{flag} {name}: {pth} does not exist")
            df = pd.read_parquet(pth)
            return (_check_companion(df, matrix, str(pth), flag)
                    if matrix is not None else df)
    beside = Path(matrix_path).parent / filename
    if not beside.exists():
        return None
    df = pd.read_parquet(beside)
    return (_check_companion(df, matrix, str(beside), flag)
            if matrix is not None else df)


GLM_COUNT_KEYS = ("k_open", "k_closed", "k_mixed", "n_linked",
                  "k_dm_open", "k_dm_closed", "k_dm_mixed", "n_dm")

# Which two count columns each statistic is a log odds ratio OF, and which
# expected pair supplies the offset.  Taken from the definitions in
# decompose_components: the GLM's group coefficient is then the SAME
# quantity the matrix holds, estimated from counts instead of averaged from
# per-cell values.
GLM_MODELS = {
    # C is an excess of concordant mass over the expected.  On the log-odds
    # scale that is concordant against mixed, offset by the expected split.
    "coordination": (("k_open", "k_closed"), ("k_mixed",),
                     ("k_dm_open", "k_dm_closed"), ("k_dm_mixed",)),
    # L = log(O_open/O_closed) - log(E_open/E_closed) exactly, so co-open
    # against co-closed with the expected log odds as the offset makes the
    # group coefficient equal to the difference in L between the arms.
    "direction_logor": (("k_open",), ("k_closed",),
                        ("k_dm_open",), ("k_dm_closed",)),
}


def _design(group, batch_df, cells):
    """Intercept, the group indicator, and one column per batch level.

    Batch enters as a COVARIATE rather than as a pre-correction.  Centring
    the matrix by batch first and then testing spends the batch's degrees of
    freedom without telling the test it spent them, so the standard errors
    come back too small; here they are paid for.
    """
    cols = [np.ones(len(cells)), np.asarray(group, float)]
    names = ["const", "group"]
    if batch_df is not None:
        for c in batch_df.columns:
            v = batch_df[c].reindex(cells).astype(str).to_numpy()
            levels = [l for l in sorted(set(v)) if l == l][1:]   # drop base
            for l in levels:
                col = (v == l).astype(float)
                # A level that is constant, or perfectly collinear with the
                # arm, contributes nothing and would make the fit singular.
                if 0 < col.sum() < len(col):
                    cols.append(col)
                    names.append(f"{c}={l}")
    X = np.column_stack(cols)
    return X, names


def donor_logits(counts, keep, donor, statistic, min_pairs=1):
    """One offset-referenced logit per (donor, bin), from POOLED counts.

    Counts are summed within the donor first and the statistic formed once,
    which is the same reason ``--from-counts`` exists in
    scaled_region_profile.py: a ratio of aggregated counts is not the mean
    of per-cell ratios, and a thin cell's ratio is mostly the prior.

    The value is exactly the statistic the matrix holds -- for
    ``direction_logor``, log(O_open/O_closed) - log(E_open/E_closed) on that
    donor's pooled pairs.
    """
    succ_k, fail_k, e_succ_k, e_fail_k = GLM_MODELS[statistic]
    dv = np.asarray(donor, dtype=object)
    order = [d for d in dict.fromkeys(dv[keep])]
    n_bins = counts["n_linked"].shape[1]
    L = np.full((len(order), n_bins), np.nan)
    for i, d in enumerate(order):
        rows = np.flatnonzero(keep & (dv == d))
        if not rows.size:
            continue
        deep = counts["n_linked"][rows] >= float(min_pairs)
        g = lambda ks: np.sum(
            [np.where(deep, counts[k][rows], 0.0) for k in ks],
            axis=0).sum(axis=0)
        s, f = g(succ_k), g(fail_k)
        es, ef = g(e_succ_k), g(e_fail_k)
        ok = (s > 0) & (f > 0) & (es > 0) & (ef > 0)
        L[i, ok] = (np.log(s[ok]) - np.log(f[ok])
                    - np.log(es[ok]) + np.log(ef[ok]))
    return L, order


def moderated_t(a, b, prior_df=4.0, strata=None):
    """Two-sample t per bin with the variance moderated across bins.

    WHY NOT WELCH, AND WHY NOT A RANK TEST.  Measured on a fixture with
    donor-level bin effects, two donors against three:

        cell-level GLM p                 power 1.00   FPR 0.605
        GLM, donor-clustered SEs         power 0.80   FPR 0.189
        Welch t on donor logits          power 0.15   FPR 0.032
        Mann-Whitney on donor logits     power 0.00   FPR 0.000
        POOLED t on donor logits         power 0.55   FPR 0.050
        pooled + moderation (d0 = 4)     power 0.60   FPR 0.037

    against a nominal 0.05.  The cell-level p is not a p-value at these
    cluster counts, and neither is a donor-clustered SE -- five clusters is
    far below the thirty-odd those need.  A rank test cannot fire at all:
    its floor at 2 against 3 is 0.2.  Pooling the variance across the arms
    recovers the third degree of freedom Welch spends, and moderating it
    across bins buys the rest, which is exactly what limma exists for at
    this sample size.

    ``prior_df`` is how many bins' worth of borrowed variance to add to the
    3 the design has.  Larger is more powerful and eventually
    anti-conservative: d0 = 8 measured 0.70 / 0.045 and d0 = 20 measured
    0.75 / 0.058 on the same fixture.

    WHY NOT A LIKELIHOOD RATIO TEST UNDER A COUNT MODEL.  Measured on the
    same fixture, at the CELL level:

        binomial GLM, Wald      power 1.00   FPR 0.563
        binomial GLM, LRT       power 1.00   FPR 0.563
        beta-binomial, LRT      power 1.00   FPR 0.496

    The test statistic is irrelevant -- Wald and LRT agree to three decimal
    places -- and beta-binomial buys almost nothing, because the
    over-dispersion it models is WITHIN a replication unit while the error
    is BETWEEN them.  Poisson and negative binomial are the wrong family
    outright: the denominator here is known and fixed (the pairs in the
    bin), not a nuisance library size, so this is a proportion and the
    binomial variance np(1-p) is the correct one; both also need integer
    counts, which soft counts are not.

    A beta-binomial LRT on unit-POOLED counts is the one real contender.
    Its dispersion parameter is estimated from 2 residual degrees of
    freedom, so the chi-square(1) reference is far from asymptotic and it
    measured FPR 0.148; with the dispersion shared across bins and an
    F(1, 3) reference it becomes competitive -- power 0.87 at FPR 0.000 --
    but ONLY while dispersion really is homogeneous.  Re-run with the
    per-bin dispersion varying tenfold, which is what a genome looks like,
    and it collapses to power 0.25 while this test holds at 0.75.  Sharing
    a variance is a strong assumption; moderating one is a weak one.
    """
    from scipy.stats import t as _t
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        na = np.isfinite(a).sum(axis=0)
        nb = np.isfinite(b).sum(axis=0)
        diff = np.nanmean(a, axis=0) - np.nanmean(b, axis=0)
        va = np.nanvar(a, axis=0, ddof=1)
        vb = np.nanvar(b, axis=0, ddof=1)
    ok = (na >= 2) & (nb >= 2)
    df = np.where(ok, na + nb - 2, np.nan)
    s2 = np.where(ok, ((na - 1) * np.nan_to_num(va)
                       + (nb - 1) * np.nan_to_num(vb)) / np.where(df > 0, df, 1),
                  np.nan)
    good = np.isfinite(s2) & (s2 > 0)
    # WHICH VARIANCE TO BORROW.  Deep bins have tighter unit-level logits
    # than thin ones, so one global median is the wrong target at both ends:
    # it inflates the deep bins' variance and deflates the thin ones'.
    # Taking the median WITHIN a depth band instead moved the measured
    # false-positive rate from 0.071 to 0.060 against a nominal 0.05, at a
    # cost of 0.07 in power -- worth it when the result is FDR-corrected
    # over ten thousand bins.  Quantile edges are fine HERE, unlike for the
    # permutation pool: borrowing a slightly wrong variance degrades
    # gracefully, while a poisoned null pool takes power to zero.
    if strata is not None and good.any():
        st = np.asarray(strata)
        s0v = np.full(s2.shape, np.nan)
        for k in np.unique(st[good]):
            m = good & (st == k)
            if m.sum() >= 20:
                s0v[st == k] = float(np.median(s2[m]))
        s0 = float(np.median(s2[good]))
        s0v = np.where(np.isfinite(s0v), s0v, s0)
    else:
        s0 = float(np.median(s2[good])) if good.any() else np.nan
        s0v = np.full(s2.shape, s0)
    d0 = float(prior_df)
    s2m = np.where(np.isfinite(s0v), (d0 * s0v + df * s2) / (d0 + df), s2)
    se = np.sqrt(s2m * (1.0 / np.where(na > 0, na, np.nan)
                        + 1.0 / np.where(nb > 0, nb, np.nan)))
    tstat = np.where(se > 0, diff / se, np.nan)
    p = np.where(np.isfinite(tstat), 2 * _t.sf(np.abs(tstat), df + d0), np.nan)
    return {"diff": diff, "p": p, "t": tstat, "n_in": na, "n_out": nb,
            "prior_df": d0, "s0": s0,
            "n_var_strata": int(len(np.unique(np.asarray(strata))))
            if strata is not None else 1}


def _design(group, batch_df, cells):
    """Intercept, the group indicator, and one column per batch level.

    Batch enters as a COVARIATE rather than as a pre-correction.  Centring
    the matrix by batch first and then testing spends the batch's degrees of
    freedom without telling the test it spent them, so the standard errors
    come back too small; here they are paid for.
    """
    cols = [np.ones(len(cells)), np.asarray(group, float)]
    names = ["const", "group"]
    if batch_df is not None:
        for c in batch_df.columns:
            v = batch_df[c].reindex(cells).astype(str).to_numpy()
            for l in [x for x in sorted(set(v))][1:]:      # first level = base
                col = (v == l).astype(float)
                # A level that is constant, or perfectly aliased with the
                # arm, contributes nothing and makes the fit singular.
                if 0 < col.sum() < len(col):
                    cols.append(col)
                    names.append(f"{c}={l}")
    return np.column_stack(cols), names


def glm_effect(counts, in_mask, out_mask, statistic, cells, batch_df=None,
               min_cells=3, min_pairs=1, n_jobs=1):
    """Per-bin binomial GLM on the RAW soft counts: the EFFECT SIZE only.

    WHY COUNTS RATHER THAN THE PER-CELL VALUES.  A rank test over cells
    gives a bin measured on 3 read pairs the same vote as one measured on
    500.  Worse, at n = 3 the Dirichlet-Multinomial posterior is about 97
    percent prior, so that vote is very nearly the prior itself and the test
    compares two arms' shrinkage rather than their data.  A binomial GLM on
    the counts weights every cell by the evidence it carries, and its group
    coefficient IS the statistic: for ``direction_logor``, the difference in
    L between the arms, on the log-odds scale the matrix already lives on.
    Measured against a planted 0.9: this returns +0.79 where the mean of the
    per-cell values is pulled toward the prior.

    The expected pool enters as an OFFSET, not a covariate, because that is
    what "referenced to E" means -- with logit(E) pinned at coefficient 1
    the intercept is the departure from E.  Fitting it as a covariate would
    let the model rescale the baseline it is meant to be measured against.

    IT RETURNS NO P-VALUE, DELIBERATELY.  Cells from one donor are not
    replicates and the binomial treats every read pair as independent; the
    same fixture that gives the coefficient above gives a false-positive
    rate of 0.605 at a nominal 0.05, and 0.189 even with donor-clustered
    standard errors.  The p belongs to ``moderated_t`` on donor-pooled
    logits, which measured 0.037.
    """
    import warnings as _w
    try:
        import statsmodels.api as sm
    except Exception:
        sys.exit(
            "--contrast-test glm needs statsmodels, which is a declared "
            "dependency of this package -- this environment predates it.\n"
            "  conda install -c conda-forge statsmodels\n"
            "  # or: pip install 'statsmodels>=0.13'\n"
            "  # or, to pick up every dependency: pip install -e .")

    succ_k, fail_k, e_succ_k, e_fail_k = GLM_MODELS[statistic]
    n_bins = counts["n_linked"].shape[1]
    in_mask, out_mask = np.asarray(in_mask, bool), np.asarray(out_mask, bool)
    keep = in_mask | out_mask
    rows_all = np.flatnonzero(keep)
    cells = list(cells)
    Xf, names = _design(in_mask[keep].astype(float), batch_df,
                        [cells[i] for i in rows_all])

    def _sum(keys, j, rows):
        return np.sum([counts[k][rows, j] for k in keys], axis=0)

    def _one(j):
        n_l = counts["n_linked"][rows_all, j]
        ok = np.isfinite(n_l) & (n_l >= float(min_pairs))
        g = in_mask[rows_all][ok].astype(float)
        n_a, n_b = int(g.sum()), int(g.size - g.sum())
        if n_a < int(min_cells) or n_b < int(min_cells):
            return np.nan, n_a, n_b
        X = Xf[ok]
        keepc = np.ptp(X, axis=0) > 0
        keepc[0] = keepc[1] = True          # const and group always stay
        X = X[:, keepc]
        succ, fail = _sum(succ_k, j, rows_all[ok]), _sum(fail_k, j, rows_all[ok])
        e_s, e_f = _sum(e_succ_k, j, rows_all[ok]), _sum(e_fail_k, j, rows_all[ok])
        eps = 1e-9
        off = np.log(np.clip(e_s, eps, None)) - np.log(np.clip(e_f, eps, None))
        endog = np.column_stack([np.clip(succ, 0.0, None),
                                 np.clip(fail, 0.0, None)])
        if not (np.isfinite(endog).all() and np.isfinite(off).all()) \
                or endog.sum() <= 0:
            return np.nan, n_a, n_b
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                # NO scale="X2".  statsmodels computes the Pearson
                # dispersion on the row scale for a two-column binomial
                # endog, which on genuinely over-dispersed donor-pooled data
                # came back at 0.012 -- shrinking the standard error ninefold
                # and turning a null into p = 8e-27.
                res = sm.GLM(endog, X, family=sm.families.Binomial(),
                             offset=off).fit()
            b = float(res.params[1])
        except Exception:
            return np.nan, n_a, n_b
        return (b if np.isfinite(b) else np.nan), n_a, n_b

    if int(n_jobs) > 1:
        from joblib import Parallel, delayed
        got = Parallel(n_jobs=int(n_jobs))(delayed(_one)(j)
                                           for j in range(n_bins))
    else:
        got = [_one(j) for j in range(n_bins)]
    diff, na, nb = (np.asarray(v) for v in zip(*got))
    return {"diff": diff.astype(float), "n_in": na.astype(int),
            "n_out": nb.astype(int), "design": names}


def glm_contrast(counts, in_mask, out_mask, statistic, cells, batch_df=None,
                 donor=None, min_cells=3, min_pairs=1, prior_df=4.0,
                 n_jobs=1):
    """Coverage-weighted effect size from the cells, p-value from the donors.

    The two halves answer different questions and neither can do the
    other's job at these sample sizes -- see ``glm_effect`` for why the
    cell-level p is not a p-value, and ``moderated_t`` for the measured
    comparison of every alternative.
    """
    eff = glm_effect(counts, in_mask, out_mask, statistic, cells,
                     batch_df=batch_df, min_cells=min_cells,
                     min_pairs=min_pairs, n_jobs=n_jobs)
    out = dict(eff)
    out["d"] = np.full(eff["diff"].shape, np.nan)
    if donor is None:
        out["p"] = np.full(eff["diff"].shape, np.nan)
        out["p_unit"] = None
        return out
    keep = np.asarray(in_mask, bool) | np.asarray(out_mask, bool)
    L, order = donor_logits(counts, keep, donor, statistic,
                            min_pairs=min_pairs)
    dv = np.asarray(donor, dtype=object)
    is_in = np.array([bool(np.asarray(in_mask, bool)[dv == d][0])
                      for d in order])
    # Depth bands for the variance target, from the pairs each bin actually
    # pooled across the units that could be tested.
    tot = np.nansum(np.where(np.isfinite(counts["n_linked"][keep]),
                             counts["n_linked"][keep], 0.0), axis=0)
    fin = np.isfinite(tot) & (tot > 0)
    if fin.sum() >= 80:
        edges = np.quantile(tot[fin], [0.25, 0.5, 0.75])
        vstrata = np.where(fin, np.digitize(tot, edges), -1)
    else:
        vstrata = None
    mt = moderated_t(L[is_in], L[~is_in], prior_df=prior_df, strata=vstrata)
    out["p"] = mt["p"]
    out["donor_diff"] = mt["diff"]
    out["n_donor_in"] = mt["n_in"]
    out["n_donor_out"] = mt["n_out"]
    out["p_unit"] = "donor"
    out["n_donors"] = len(order)
    out["n_var_strata"] = mt.get("n_var_strata", 1)
    return out


def empirical_null_qvalue(p_obs, p_null, p_perm=None):
    """A q-value against the RELABELLED null instead of against uniform.

    Benjamini-Hochberg assumes the p-values are uniform under the null.  On
    this data they are not -- a nominal welch p ran at 0.24 on relabelled
    GCH and 0.38 on HCG, against 0.05 -- so BH at q = 0.05 is really
    controlling something near 0.24 or 0.38.  The fix is not a different
    threshold on the same p, which only trades one arbitrary cut for
    another; it is to count the false positives from the null that was
    actually measured:

        expected false at cut t  =  P(null p <= t) x n_tested
        q at the k-th smallest   =  expected(p_k) / k

    monotonised upward the way BH's step-up is, so a bin is never assigned
    a smaller q than a more extreme one.  Then `q <= 0.1` means what it
    says, whatever the nominal p is worth.

    IT ONLY WORKS IF THE NULL IS BIG ENOUGH.  With a pooled null of tens of
    thousands the smallest resolvable P(null <= t) is 1/pool, so the finest
    q is that times n_tested divided by the rank -- reported by the caller
    so a q of 0 is never mistaken for certainty.
    """
    p = np.asarray(p_obs, float)
    null = np.sort(np.asarray(p_null, float)[
        np.isfinite(np.asarray(p_null, float))])
    q = np.full(p.shape, np.nan)
    fin = np.isfinite(p)
    if not fin.any() or null.size < 100:
        return q, {"n_null": int(null.size), "finest_q": float("nan")}
    obs = p[fin]
    # SORT BY WHAT IS BEING CORRECTED.  The step-up ranks bins by the
    # p-value the FDR is computed on, and with a STRATIFIED permutation p
    # that is not the nominal one: two strata have different null shapes, so
    # a larger nominal p can carry a smaller permutation p.  Sorting by the
    # nominal p and then applying rank arithmetic to the permutation p
    # measured a true false-discovery rate of 0.79 at a nominal q of 0.10.
    key = (np.asarray(p_perm, float)[fin] if p_perm is not None else obs)
    key = np.where(np.isfinite(key), key, np.inf)
    order = np.argsort(key, kind="stable")
    ps = obs[order]
    n = ps.size
    # P(null <= t) at each observed p, then the expected count at that cut.
    #
    # ``p_perm`` SUPPLIES THAT PROBABILITY PER BIN when the caller already
    # measured it against a STRATIFIED null -- bins of similar coverage
    # pooled together rather than the whole genome at once, the same
    # correction the permutation null and the drawn-null rank both needed.
    # Given it, this is exactly Benjamini-Hochberg applied to a p-value
    # that is calibrated by construction, which is what makes BH valid here
    # when it is not on the nominal p.  Checked: the two routes agreed to
    # 0.0015 in q and returned identical call counts and identical true
    # false-discovery rates on a fixture.
    if p_perm is not None:
        pp = np.asarray(p_perm, float)[fin][order]
        frac = np.where(np.isfinite(pp), pp,
                        np.searchsorted(null, ps, side="right") / null.size)
    else:
        frac = np.searchsorted(null, ps, side="right") / null.size
    expected = frac * n
    ranks = np.arange(1, n + 1)
    raw = expected / ranks
    # Step-up monotonisation: q_(k) = min over j >= k.
    qs = np.minimum.accumulate(raw[::-1])[::-1]
    qs = np.clip(qs, 0.0, 1.0)
    out = np.empty(n)
    out[order] = qs
    q[fin] = out
    # The smallest q this null could actually express, reported so a small
    # one is never mistaken for a precise one: the empirical P(null <= t)
    # bottoms out at one draw, so q bottoms out with it.
    return q, {"n_null": int(null.size),
               "finest_q": float(np.min(qs)) if qs.size else float("nan")}


def unit_relabellings(units, n_a, true_a, max_n=0, seed=0, scope="all"):
    """Ways of calling some units the contrast arm, other than the real one.

    ``scope="all"`` takes every BALANCED relabelling -- the same arm sizes as
    the real contrast, because power and variance both depend on them, so a
    1-vs-4 split would not be the test that was actually run.

    ``scope="reference"`` DROPS THE CONTRAST ARM'S UNITS ENTIRELY and
    relabels only within the reference.  The balanced null is contaminated
    and cannot not be: with two tumours among five units, three of the nine
    relabellings put BOTH tumours in the reference arm and the other six put
    one in each, so every one of them inherits some of the signal being
    tested.  That makes the measured inflation an upper bound, and on a run
    where the real contrast came back at 1.0x chance it is the difference
    between "no effect" and "a null built partly from the effect".
    Excluding the contrast arm gives a null with none of it -- at the cost
    of smaller, unbalanced arms, so read it as a check on the inflation
    rather than as the same test.
    """
    import itertools
    true_a = frozenset(true_a)
    if scope == "reference":
        rest = sorted(set(units) - true_a)
        if len(rest) < 2:
            return []
        n_ref_a = max(1, len(rest) // 2)
        return [frozenset(c)
                for c in itertools.combinations(rest, n_ref_a)][
            :None if not max_n else int(max_n)]
    out = [frozenset(c) for c in itertools.combinations(sorted(units), int(n_a))
           if frozenset(c) != true_a]
    if max_n and len(out) > int(max_n):
        rng = np.random.default_rng(int(seed))
        out = [out[i] for i in sorted(rng.choice(len(out), int(max_n),
                                                 replace=False))]
    return out


def unit_permutation_p(p_obs, p_null, strata=None, min_pool=2_000):
    """An empirical p from relabelled units, pooled across bins.

    WHY THIS EXISTS.  A test can be the most powerful one on a dataset and
    still have a nominal p that means nothing -- measured on one run, welch
    and mwu on per-cell values recovered far more spiked signal than the
    unit-level GLM and carried a permuted false-positive rate two to three
    times nominal.  Those are not in conflict: the extra power is real, and
    so is the inflation, because cells within a cell type are not
    replicates.  Replacing the nominal p with the rank of the observed
    statistic in a null built by RELABELLING WHICH UNITS ARE THE CONTRAST
    ARM keeps the power and fixes the calibration, because the relabelled
    runs inherit exactly the same pseudo-replication.

    Extremeness is measured as -log10(p) so any test can be fed in -- the
    nominal p never has to be believed, only ordered.

    POOLED ACROSS BINS, because with five units there are only nine
    relabellings and a per-bin null of nine cannot resolve a p below 0.1.
    Pooling within a coverage stratum buys n_bins x 9 draws, at the cost of
    assuming bins within a stratum are exchangeable -- which is what the
    strata are chosen to make true.

    IT IS AN UPPER BOUND ON THE P, not an exact one: a relabelled split such
    as {tumor1, GM12878} against the rest still carries real biology, so the
    null is contaminated with signal and the p is conservative.  A bin that
    clears it has cleared something harder than the nominal test.
    """
    obs = -np.log10(np.clip(np.asarray(p_obs, float), 1e-300, 1.0))
    nulls = [-np.log10(np.clip(np.asarray(q, float), 1e-300, 1.0))
             for q in p_null]
    out = np.full(obs.shape, np.nan)
    if not nulls:
        return out, {"n_relabellings": 0, "median_pool": 0}
    st = (np.zeros(obs.shape, int) if strata is None
          else np.asarray(strata).astype(int))
    testable = np.isfinite(obs)
    pools = []
    for s in np.unique(st[testable]):
        m = testable & (st == s)
        pool = np.concatenate([q[m] for q in nulls])
        pool = np.sort(pool[np.isfinite(pool)])
        if pool.size < int(min_pool):
            everything = np.concatenate(
                [q[testable] for q in nulls])
            pool = np.sort(everything[np.isfinite(everything)])
        if pool.size < 2:
            continue
        # One-sided: bigger -log10(p) is more extreme, whichever tail the
        # underlying test used.
        n_ge = pool.size - np.searchsorted(pool, obs[m], side="left")
        out[m] = (n_ge + 1.0) / (pool.size + 1.0)
        pools.append(pool.size)
    return out, {"n_relabellings": len(nulls),
                 "median_pool": int(np.median(pools)) if pools else 0}


def _glm_statistic_for(matrix_path):
    """Which 2x2 table the matrix's statistic is a log odds ratio of."""
    stem = Path(matrix_path).name
    base = stem[len("concordance_"):-len(".parquet")] \
        if stem.startswith("concordance_") and stem.endswith(".parquet") \
        else ""
    for suf in ("_z", "_rank"):
        if base.endswith(suf):
            base = base[:-len(suf)]
    if base not in GLM_MODELS:
        sys.exit(
            f"--contrast-test glm cannot model {stem!r}: it fits the counts "
            f"behind {' or '.join(sorted(GLM_MODELS))}, and the statistic is "
            f"read from the matrix filename.  Point --matrix at "
            f"concordance_coordination.parquet or "
            f"concordance_direction_logor.parquet (the _z and _rank forms of "
            f"those are accepted too -- the GLM refits from raw counts "
            f"either way).")
    return base


def rank_filename_for(matrix_path):
    """The rank matrix that belongs beside ``matrix_path``.

    Derived from the matrix's own name rather than hardcoded, because the
    rank is per statistic: concordance_coordination.parquet is answered by
    concordance_coordination_rank.parquet and the log odds ratio by its
    own.  Pairing a coordination matrix with a direction rank would gate
    one statistic on another's evidence.
    """
    stem = Path(matrix_path).name
    if stem.startswith("concordance_") and stem.endswith(".parquet"):
        base = stem[:-len(".parquet")]
        # A _z matrix is the same statistic already referenced to its own
        # draws; its evidence still lives in that statistic's rank file, not
        # in a "_z_rank" that was never written.
        if base.endswith("_z"):
            base = base[:-2]
        return f"{base}_rank.parquet"
    return "concordance_coordination_rank.parquet"


def rank_departure(R, in_mask, out_mask, min_score):
    """Per bin: does either arm depart from its own drawn null, and by how
    much.

    The contrast asks whether the arms DIFFER.  It has no way to ask whether
    either is doing anything, and at a few hundred cells per arm a small
    consistent offset between two arms that both sit at their own null
    clears FDR comfortably -- a difference in nothing.  This is the missing
    half: the fraction of each arm's cells whose bin departs from the null
    the pipeline produces at that depth and that marginal.

    ONE ARM IS ENOUGH.  A bin coordinated in the reference and flat in the
    tumour is the finding, not a failure of it, so the gate takes the max
    over arms rather than the min.

    Cells with no finite rank are excluded from their arm's denominator, not
    counted as non-departing: an unmeasured cell is not evidence of absence,
    and treating it as one would penalise exactly the sparse bins that most
    need an honest answer.
    """
    R = np.asarray(R, float)
    hit = np.isfinite(R) & (np.abs(R) >= float(min_score))
    out = {}
    for tag, m in (("a", in_mask), ("b", out_mask)):
        sub, hs = R[m], hit[m]
        n = np.isfinite(sub).sum(axis=0)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[f"rank_frac_{tag}"] = np.where(n > 0, hs.sum(axis=0) / n,
                                               np.nan)
            out[f"rank_mean_{tag}"] = np.nanmean(sub, axis=0)
        out[f"rank_n_{tag}"] = n
    out["rank_frac_max"] = np.fmax(out["rank_frac_a"], out["rank_frac_b"])
    return out


def global_shift_report(diff, sig=None):
    """Is the contrast a genome-wide offset rather than local aberration?

    A bin-specific result has differences scattered either side of zero, with
    the significant ones in both tails.  If instead nearly every bin moves the
    same way, the two arms differ by a constant across the whole genome -- and
    at a few hundred cells per arm even a modest constant makes almost every
    bin significant, so the BED comes back holding most of the genome and is
    useless as a foreground: an enrichment of the genome against itself.

    Such an offset can be real (one arm genuinely less coordinated) or
    technical (different coverage, purity, or cell quality changing how D is
    estimated).  This cannot tell which, so it reports rather than corrects.
    """
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {}
    med = float(np.median(d))
    same = float(max(np.mean(d > 0), np.mean(d < 0)))
    mad = float(np.median(np.abs(d - med))) or float("nan")
    out = {"median_diff": med, "frac_bins_same_direction": same,
           "median_abs_deviation": mad,
           "median_over_mad": abs(med) / mad if mad == mad and mad else np.nan}
    if sig is not None:
        m = np.asarray(sig, bool)[np.isfinite(np.asarray(diff, float))]
        if m.any():
            out["frac_significant_up"] = float(np.mean(d[m] > 0))
    return out


def center_cells(X, mask=None):
    """Subtract each cell's own median across bins.

    Removes a per-CELL offset -- the mechanism behind a genome-wide arm
    difference, since each cell carries its own overall level -- while leaving
    the shape along the genome alone.  Gentler than the per-cell RANK used
    before the UMAP, which redistributes every value; this only slides each
    cell up or down.

    It is still compositional in principle: a bin's centred value depends on
    the other bins through the median.  That is harmless while genuine
    aberrations are a small share of the genome (a few hundred bins cannot
    move a median over tens of thousands) and wrong when they are not.  If
    half the genome really differs, centring will subtract the finding.
    """
    X = np.asarray(X, float)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(X[:, mask] if mask is not None else X, axis=1)
    return X - med[:, None]


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


def volcano_figure(diff, fdr, sig, path, title, fdr_cut, min_diff=0.0,
                   passed_fdr=None):
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
    # A bin can clear the FDR line and still be dropped -- by the consistency
    # filter, the batch filter or an effect-size floor.  Colouring it "n.s."
    # would put a point above the threshold line in the not-significant class,
    # which reads as a broken plot rather than as a second criterion.
    dropped = np.zeros_like(m)
    if passed_fdr is not None:
        dropped = np.asarray(passed_fdr, bool)[ok] & ~m
    rest = ~m & ~dropped

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(x[rest], y[rest], s=5, c="#C6CBD1", lw=0, alpha=0.55,
               rasterized=True, label=f"n.s. ({int(rest.sum()):,})")
    if dropped.any():
        ax.scatter(x[dropped], y[dropped], s=6, c="#F0C808", lw=0, alpha=0.8,
                   rasterized=True,
                   label=f"FDR only, not kept ({int(dropped.sum()):,})")
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


def shared_grid(mats, keys):
    """Cells and bins measured in EVERY modality -> ((cells, bins), dropped).

    The two panels only mean anything side by side if position i is the same
    cell and row j the same bin in both.  A cell present in one matrix and
    absent from the other, or covered in one and all-NaN in the other, cannot
    be compared and is removed from both rather than drawn as a blank stripe
    in one -- a blank column reads as "no signal here", which is a claim,
    where the truth is "not measured here".
    """
    names = list(mats)
    cells = None
    for nm in names:
        idx = mats[nm].index
        cells = idx if cells is None else cells.intersection(idx)
    cols = [k for k in keys
            if all(k in mats[nm].columns for nm in names)]
    if cells is None or not len(cells) or not cols:
        return None, {"bins_dropped": len(keys), "cells_dropped": 0}

    # Present is not the same as measured: require real values on both sides.
    ok_cols, ok_cells = [], np.ones(len(cells), bool)
    for k in cols:
        if all(np.isfinite(mats[nm].loc[cells, k].to_numpy()).any()
               for nm in names):
            ok_cols.append(k)
    if not ok_cols:
        return None, {"bins_dropped": len(keys), "cells_dropped": 0}
    for nm in names:
        block = mats[nm].loc[cells, ok_cols].to_numpy()
        ok_cells &= np.isfinite(block).any(axis=1)
    if not ok_cells.any():
        return None, {"bins_dropped": len(keys), "cells_dropped": len(cells)}
    kept = cells[ok_cells]
    n_union = len(set().union(*[set(mats[nm].index) for nm in names]))
    return (kept, ok_cols), {"bins_dropped": len(keys) - len(ok_cols),
                             "cells_dropped": n_union - len(kept)}


# What each component IS, for an axis.  A panel labelled D while it draws
# C names a quantity whose sign convention is the opposite of the one
# plotted, since D = C(2F-1) is an inverted copy of C at these marginals.
JOINT_STAT_LABEL = {
    "coordination": "C",
    "direction": "F",
    "direction_expected": "F_exp",
    "direction_excess": "F \u2212 F_exp",
    "direction_logor": "log-OR",
    "directional": "D = C\u00b7(2F\u22121)",
    "directional_corrected": "D' = C\u00b7(F\u2212F_exp)",
    "marginal": "p\u0304",
}


def statistic_label(matrix_path):
    """The short name of the component a concordance_<name>.parquet holds."""
    stem = Path(matrix_path).name if matrix_path else ""
    if stem.startswith("concordance_") and stem.endswith(".parquet"):
        got = stem[len("concordance_"):-len(".parquet")]
        if got in JOINT_STAT_LABEL:
            return JOINT_STAT_LABEL[got]
    return None


def _point_density(x, y, bins=180, smooth=1.5, rng=None):
    """Local density at each point, from a 2-D histogram rather than a KDE.

    A gaussian_kde evaluated at every point is O(n^2) and there are tens of
    thousands of bins here.  Binning, smoothing once, then reading each
    point's cell back gives the same picture in linear time.
    """
    from scipy.ndimage import gaussian_filter
    # Binned over the VIEW when one is given, so the grid resolves the cloud
    # instead of spending every cell on empty space between outliers.
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=rng)
    H = gaussian_filter(H, smooth)
    xi = np.clip(np.searchsorted(xe, x, side="right") - 1, 0, H.shape[0] - 1)
    yi = np.clip(np.searchsorted(ye, y, side="right") - 1, 0, H.shape[1] - 1)
    return H[xi, yi]


def joint_scatter(ax, tab, a, b, stats, label, stat_labels=None):
    """Every bin as a dot, coloured by how many bins are near it.

    NO CATEGORY HIGHLIGHT.  Marking the both-significant bins in red asked
    the eye to find a subset that was defined by two thresholds, and against
    tens of thousands of grey points that subset reads as the finding
    whatever its size.  The claim this panel actually supports is about the
    CLOUD -- whether the two modalities move together across the genome --
    and a density surface shows that where scattered dots cannot: at this n
    the overplotted core is solid ink either way, so the shape of the middle
    is invisible until it is coloured by density.

    The fit is ordinary least squares over EVERY bin.  Its slope is
    ATTENUATED and the annotation says so: both axes are estimated with
    error, and error on x biases a least-squares slope toward zero, so the
    slope describes the cloud while the correlation carries the association.

    TWO P-VALUES, NOT THREE.  For a simple regression the test of slope = 0
    and the test of r = 0 are the same t-statistic, so the "fitted line's
    p-value" and the Pearson p-value are one number, and reporting them
    separately would suggest two pieces of evidence where there is one.
    Spearman is the second, genuinely different, test: rank-based, so it
    survives the heavy tails these differences have.

    BOTH ARE ANTI-CONSERVATIVE HERE and the annotation says that too.  They
    assume the bins are independent; neighbouring 250 kb bins are strongly
    correlated, so the effective n is far below the printed one.  Read the
    coefficient, not the exponent.
    """
    from scipy.stats import pearsonr, spearmanr, linregress

    sl = stat_labels or {}
    x, y = tab[f"{a}_diff"].to_numpy(float), tab[f"{b}_diff"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    xo, yo = x[ok], y[ok]

    # THE VIEW IS CLIPPED TO A ROBUST RANGE, and it has to be.  These
    # differences have heavy tails -- a handful of bins sit tens of units
    # out -- and letting them set the axes squeezes the entire cloud into a
    # few pixels, which is the failure the density colouring exists to fix.
    # The fit and every statistic still use ALL the points; only the view is
    # trimmed, and the count outside it is stated so the trim is visible.
    lim = None
    if xo.size > 200:
        xl = np.percentile(xo, [0.3, 99.7])
        yl = np.percentile(yo, [0.3, 99.7])
        mx, my = 0.06 * np.ptp(xl), 0.06 * np.ptp(yl)
        lim = ((xl[0] - mx, xl[1] + mx), (yl[0] - my, yl[1] + my))
    dens = (_point_density(xo, yo, rng=lim) if xo.size > 50
            else np.ones(xo.size))
    order = np.argsort(dens)          # dense points drawn last, on top
    sc = ax.scatter(xo[order], yo[order], c=dens[order], s=5, lw=0,
                    cmap="turbo", rasterized=True)
    if lim is not None:
        n_out = int(np.sum((xo < lim[0][0]) | (xo > lim[0][1])
                           | (yo < lim[1][0]) | (yo > lim[1][1])))
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
        if n_out:
            ax.annotate(f"{n_out:,} bins outside the view (all are in the "
                        f"fit)", (0.99, 0.01), xycoords="axes fraction",
                        fontsize=6.5, color="0.4", ha="right", va="bottom")
    cb = ax.figure.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("bins per neighbourhood (smoothed 2-D count)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.axhline(0, color="#9AA0A6", lw=0.8, zorder=0)
    ax.axvline(0, color="#9AA0A6", lw=0.8, zorder=0)

    fit = {}
    if xo.size >= 3 and np.ptp(xo) > 0:
        lr = linregress(xo, yo)
        r, p_r = pearsonr(xo, yo)
        rho, p_rho = spearmanr(xo, yo)
        xs = np.array([xo.min(), xo.max()])
        ax.plot(xs, lr.slope * xs + lr.intercept, "-", color="#1A1A1A",
                lw=1.4, zorder=5)
        fit = {"slope": float(lr.slope), "intercept": float(lr.intercept),
               "slope_stderr": float(lr.stderr),
               "slope_pvalue": float(lr.pvalue),
               "pearson_r": float(r), "pearson_p": float(p_r),
               "spearman_rho": float(rho), "spearman_p": float(p_rho),
               "n_fit": int(xo.size)}
        _f = lambda v: "< 1e-300" if v < 1e-300 else f"= {v:.2g}"
        ax.annotate(
            f"slope {lr.slope:+.3g} \u00b1 {lr.stderr:.2g}   "
            f"p {_f(lr.pvalue)}\n"
            f"Pearson r = {r:+.3f}, p {_f(p_r)}  "
            f"(same test as the slope)\n"
            f"Spearman \u03c1 = {rho:+.3f}, p {_f(p_rho)}\n"
            f"n = {xo.size:,} bins \u2014 neighbouring bins are correlated, "
            f"so both p are anti-conservative\n"
            f"slope attenuated by error on x",
            (0.03, 0.97), xycoords="axes fraction", fontsize=7,
            va="top", color="#1A1A1A",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85",
                      alpha=0.85))

    xa = f"{a} mean difference" + (f"  ({sl[a]})" if sl.get(a) else "")
    yb = f"{b} mean difference" + (f"  ({sl[b]})" if sl.get(b) else "")
    ax.set_xlabel(f"{xa}, contrast \u2212 reference")
    ax.set_ylabel(f"{yb}, contrast \u2212 reference")
    same = stats.get("frac_both_same_direction")
    note = (f"{stats['n_both']:,} significant in both, {same:.0%} same "
            f"direction" if stats["n_both"] else "no bin significant in both")
    ax.set_title(f"{label}: {note}", fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return fit


def joint_figure(tab, a, b, stats, mats, cell_types, path, label,
                 ct_colors=None, stat_labels=None):
    """One PDF: the two mean differences against each other, then a clustered
    heatmap per modality over the union of significant bins.

    The heatmaps are drawn on ONE grid: the cells and bins measured in BOTH
    modalities, in one order, clustered once on the two differences together.
    Anything else makes the panels uncomparable in a way that is invisible --
    letting each panel keep its own bins puts a different bin on each row, and
    letting each keep its own cells puts a different cell in each column, so
    a reader comparing the two positions is comparing nothing.  The count that
    survives is in the title, because a small shared grid is itself the
    result: it says the two modalities were measured on different parts of the
    genome, not that they disagree.
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
        fit = joint_scatter(ax, tab, a, b, stats, label, stat_labels)
        stats.update({f"scatter_{k}": v for k, v in fit.items()})
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

        grid, dropped = shared_grid(mats, keys)
        _ = ct_colors
        if grid is None:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, f"no bin is measured in both {a} and {b}\n"
                              f"across a shared set of cells",
                    ha="center", va="center", fontsize=11)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
            return
        cells, cols = grid
        ct = pd.Series(cell_types).reindex(cells).astype(str)
        cell_order = np.argsort(ct.to_numpy(), kind="stable")
        ct_sorted = ct.to_numpy()[cell_order]
        # Columns are cells in cell-type order, but that order is invisible
        # without a key: which block of columns is the tumour and which the
        # reference is the first thing this figure has to answer.
        levels = list(dict.fromkeys(ct_sorted))
        palette = dict(ct_colors or {})
        missing = [lv for lv in levels if lv not in palette]
        if missing:
            fallback = plt.get_cmap("tab10")
            for i, lv in enumerate(missing):
                palette[lv] = fallback(i % 10)
        strip = np.array([[matplotlib.colors.to_rgb(palette[c])
                           for c in ct_sorted]])

        # The colourbar gets its OWN gridspec column rather than being carved
        # out of the heatmap axis.  Carved out, the heatmap narrows while the
        # strip above it does not, and the key stops sitting over the columns
        # it labels -- which is worse than no key at all.
        n = len(mats)
        fig = plt.figure(figsize=(6.0 * n, 7.6))
        gs = fig.add_gridspec(2, 3 * n, height_ratios=[1, 26], hspace=0.03,
                              width_ratios=[16, 1, 3] * n, wspace=0.05)
        axes = [[fig.add_subplot(gs[1, 3 * i]) for i in range(n)]]
        cbars = [fig.add_subplot(gs[1, 3 * i + 1]) for i in range(n)]
        for i in range(n):
            sax = fig.add_subplot(gs[0, 3 * i])
            sax.imshow(strip, aspect="auto", interpolation="nearest")
            sax.set_xticks([])
            sax.set_yticks([])
            sax.set_frame_on(False)
        for ax, cax, (nm, mat) in zip(axes[0], cbars, mats.items()):
            v = mat.reindex(index=cells, columns=cols).to_numpy()[
                cell_order].T
            lim = np.nanpercentile(np.abs(v), 98) if np.isfinite(v).any() else 1
            im = ax.imshow(v, aspect="auto", cmap="RdBu_r",
                           vmin=-lim, vmax=lim, interpolation="nearest")
            ax.set_title(nm, fontsize=10)
            ax.set_xlabel(f"{len(cells):,} cells measured in both "
                          f"(grouped by cell type)")
            ax.set_ylabel(f"{len(cols):,} bins measured in both "
                          f"(shared order)")
            ax.set_yticks([])
            ax.set_xticks([])
            fig.colorbar(im, cax=cax)
            cax.tick_params(labelsize=7)
        handles = [plt.Rectangle((0, 0), 1, 1,
                                 facecolor=palette[lv], edgecolor="#374151",
                                 label=f"{lv} ({int(np.sum(ct_sorted == lv))})")
                   for lv in levels]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(levels), 6),
                   frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
        note = (f"; {dropped['bins_dropped']:,} bins and "
                f"{dropped['cells_dropped']:,} cells dropped for lacking one "
                f"modality" if (dropped["bins_dropped"]
                                or dropped["cells_dropped"]) else "")
        fig.suptitle(f"{label}: significant bins on the shared grid{note}",
                     fontsize=11)
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
    """Linkage for the bins, or None; returns (linkage, leaf_order).

    CORRELATION DISTANCE IS UNDEFINED FOR A CONSTANT ROW, and constant rows
    are normal here rather than exotic.  A companion signal need not cover
    every selected bin -- the null rank is written only where the bin
    measured something, so on one real run it covered 446 of 500 -- and the
    uncovered rows are filled with a single neutral value before display.
    Their correlation with anything is 0/0, scipy's pdist hands back NaN,
    and linkage refuses the condensed matrix outright.  The same happens to
    a cell that is blank everywhere in the column clustering.

    So the distances are computed here and the undefined ones set to 1.0 --
    the value correlation distance gives to two uncorrelated vectors.  That
    is what a row carrying no pattern should contribute: no evidence of
    similarity to anything, so it clusters last, with the other blanks.
    Dropping the rows instead would silently change the layout the caller
    asked for, and this function's whole job is to return an order over the
    rows it was given.
    """
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist
    except Exception:
        return None, np.arange(Z.shape[0])
    if Z.shape[0] < 3:
        return None, np.arange(Z.shape[0])
    with np.errstate(invalid="ignore", divide="ignore"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = pdist(np.asarray(Z, np.float64), metric="correlation")
    d = np.where(np.isfinite(d), d, 1.0)
    lk = linkage(d, method="average")
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
    ap.add_argument("--contrast-unit-p", type=float, default=0.05,
                    metavar="P",
                    help="Threshold for --contrast-select unit-p, on the "
                         "relabelled-null p rather than on an FDR.  It is a "
                         "PER-BIN rate: at p <= 0.05 over 10,000 tested "
                         "bins, about 500 clear by chance however well "
                         "calibrated the p is, so a set selected this way is "
                         "a screen and not a list of bins each claimed to "
                         "differ.  The run prints the expected count beside "
                         "the observed one.")
    ap.add_argument("--contrast-select",
                    choices=["fdr", "top-n", "unit-fdr", "unit-p"],
                    default="fdr",
                    help="How a contrast BED is filled.  'fdr' (default) keeps "
                         "every bin below --contrast-fdr, which is what the "
                         "volcano colours and can legitimately come back "
                         "empty.  'top-n' keeps the --n-contrast-bins largest "
                         "|t| whether or not any of them is significant, so it "
                         "always returns a set and never says the contrast is "
                         "empty.  'unit-fdr' keeps every bin below "
                         "--contrast-fdr on the RELABELLED-NULL q instead of "
                         "on BH of the nominal p, and needs "
                         "--contrast-permute-units.  Use it when the "
                         "relabellings say the nominal p is inflated: BH "
                         "assumes that p is uniform under the null, and on a "
                         "measured run it was not -- welch ran at 0.24 on "
                         "GCH and 0.38 on HCG against a nominal 0.05, so "
                         "'FDR < 0.05' was really controlling near those.  "
                         "Selecting on the calibrated q makes the BEDs, the "
                         "volcano, the joint contrast and the heatmaps all "
                         "reflect the rate that was actually measured, "
                         "instead of leaving it to a column the reader has "
                         "to filter by hand.  'unit-p' keeps every bin below "
                         "--contrast-unit-p on that same relabelled-null p "
                         "WITHOUT the multiple-testing step -- legitimate "
                         "when the set is a screen, such as a foreground for "
                         "genomic enrichment, and not a list of bins each "
                         "claimed to differ.  Both need "
                         "--contrast-permute-units.")
    ap.add_argument("--delta-var", action="append", default=None,
                    metavar="NAME=PATH",
                    help="concordance_delta_var.parquet for the matching "
                         "--matrix NAME, giving the sampling variance of every "
                         "(cell, bin) estimate.  Auto-discovered next to the "
                         "matrix when not given.  Enables --contrast-weight.")
    ap.add_argument("--min-pairs", action="append", default=None,
                    metavar="N|NAME=N",
                    help="Hold (cell, bin) entries with fewer than N read "
                         "pairs of their own FIXED while the permutation "
                         "null is built.  They are still used as "
                         "observations: RWR filled them from cells the k-NN "
                         "graph says are similar, which is evidence, and "
                         "the component matrices carry 0%% NaN so every "
                         "entry is finite whatever its real coverage.  The "
                         "problem is only in the null -- that graph is "
                         "cell-type aware, so an imputed entry carries its "
                         "arm's signature by construction, and shuffling it "
                         "scatters that signature across every bin position "
                         "so the null holds each cell's genome-wide MIX "
                         "rather than the bin's own share.  A sparsely "
                         "covered bin then clears a null that no longer "
                         "describes it.  Measured on a fixture with no "
                         "signal at all: 57.7%% of sparse bins called at "
                         "p < 0.05, 7.7%% once held fixed, and every bin "
                         "still testable -- against 6 of 300 if the imputed "
                         "entries are dropped instead.  Only acts when "
                         "--contrast-permutations > 0.  Needs --pair-counts "
                         "(auto-discovered from beside --matrix, so a matrix "
                         "whose directory holds one is filtered even when "
                         "--pair-counts named a different matrix).  A bare "
                         "number applies to every --matrix; NAME=N sets it "
                         "for one, since a dense context and a sparse one "
                         "do not want the same cut.")
    ap.add_argument("--min-bin-frac", action="append", default=None,
                    metavar="F|NAME=F",
                    help="Mark a bin UNTESTABLE unless at least this "
                         "fraction of cells measured it with --min-pairs "
                         "read pairs of their own.  Such a bin's statistic "
                         "is mostly what RWR borrowed from the k-NN graph, "
                         "and that graph is built from accessibility, so it "
                         "is cell-type aware and the borrowed value carries "
                         "the arm's signature by construction.  Measured on "
                         "a fixture: with the null built only from measured "
                         "bins, 73%% to 100%% of sparse bins came back "
                         "significant with nothing real in them, the "
                         "fraction rising with how strongly the imputation "
                         "is tinted.  Setting this drops them from the null "
                         "AND from the calls, which is the honest pair.  "
                         "Left at 0 they are still tested, but against a "
                         "null pooled within their own coverage class, "
                         "where they clear only by exceeding the tint that "
                         "every such bin carries -- calibrated, though what "
                         "it measures is the imputation rather than the "
                         "concordance.  Needs --min-pairs.  A bare number "
                         "applies to every --matrix; NAME=F sets it for "
                         "one.")
    ap.add_argument("--null-rank", action="append", default=None,
                    metavar="NAME=PATH",
                    help="concordance_<stat>_rank.parquet for the matching "
                         "--matrix NAME: each cell-bin's signed -log10(p) "
                         "against a null drawn from that bin's own ends "
                         "(step 5 --null-rank).  Auto-discovered from the "
                         "matrix filename.  WHAT IT ADDS THAT THE CONTRAST "
                         "CANNOT SEE.  A two-arm test asks whether the arms "
                         "DIFFER; it cannot ask whether either of them is "
                         "doing anything.  Two arms both sitting at their "
                         "own null, separated by a small consistent offset, "
                         "clear FDR at a few hundred cells per arm -- a "
                         "difference in nothing.  The rank says, per cell "
                         "and per bin, whether that bin departs from what "
                         "the pipeline produces with no coordination at that "
                         "depth and that marginal, so it gates on presence "
                         "where --min-pairs gates on coverage.  Used for the "
                         "gate, for the reported per-arm departure rates, "
                         "and for a companion heatmap in the concordance "
                         "heatmap's own layout.")
    ap.add_argument("--null-rank-min", type=float, default=1.30103,
                    metavar="S",
                    help="How far from its own null a cell-bin has to be to "
                         "count as departing, as |signed -log10(p)|.  "
                         "Default 1.30103 = p 0.05.  Note the score is "
                         "CENSORED at -log10(2/(pool+1)) -- about 4.4 for a "
                         "50,000-draw pool -- so it saturates in "
                         "well-measured bins and is a poor effect size; that "
                         "is why it gates and annotates here rather than "
                         "replacing the tested value.")
    ap.add_argument("--null-rank-min-frac", type=float, default=0.0,
                    metavar="F",
                    help="Require at least this fraction of ONE arm's cells "
                         "to depart from the null before a bin can be "
                         "called.  0 (default) reports the rates without "
                         "gating.  One arm, not both: a bin that is "
                         "coordinated in the reference and flat in the "
                         "tumour is exactly the finding, and requiring both "
                         "would discard it.")
    ap.add_argument("--null-rank-heatmaps", choices=["auto", "on", "off"],
                    default="auto",
                    help="Companion heatmap of the rank over the selected "
                         "bins, in the concordance heatmap's EXACT layout.  "
                         "It answers a question the main heatmap cannot: is "
                         "the block structure there a departure from the "
                         "null, or the raw statistic's own scale tracking "
                         "depth and openness across cell types.  'auto' "
                         "draws it when a rank matrix is available.")
    ap.add_argument("--pair-counts", action="append", default=None,
                    metavar="NAME=PATH",
                    help="pair_counts_per_bin.parquet for the matching "
                         "--matrix NAME.  Auto-discovered.  Used only to "
                         "report per-arm coverage, which is the first thing "
                         "to check when a contrast shows a genome-wide "
                         "offset.")
    ap.add_argument("--contrast-weight",
                    choices=["none", "precision", "random-effects"],
                    default="none",
                    help="Weight each cell by the precision of its D "
                         "estimate.  D is estimated from the read pairs in a "
                         "(cell, bin), so an unweighted test treats a cell "
                         "with five pairs as worth one with five hundred -- "
                         "wasteful when coverage varies, and misleading when "
                         "it varies BETWEEN arms, since a noisier estimate is "
                         "an attenuated one.  'random-effects' adds the "
                         "DerSimonian-Laird between-cell variance and is the "
                         "one to use; 'precision' is the fixed-effect form "
                         "and is anticonservative when cells within an arm "
                         "genuinely differ.  Needs --delta-var.")
    ap.add_argument("--contrast-stratify", default=None, metavar="COLUMN",
                    help="Run the contrast WITHIN each level of this --samples "
                         "column and combine across them (Stouffer's weighted "
                         "z).  The right treatment for a BIOLOGICAL blocking "
                         "factor -- patient above all: comparing the two arms "
                         "inside one patient removes that patient's entire "
                         "contribution, including CNV background and purity, "
                         "which no covariate would capture.  Adjusting for "
                         "patient instead assumes its effect is a constant "
                         "added to every cell, which two patients cannot even "
                         "test.  Technical batches (plate, sequencing run) "
                         "belong in --batch-column and --remove-batch, not "
                         "here.  Per-stratum differences are written to the "
                         "BED so a bin carried by one stratum is visible.")
    ap.add_argument("--contrast-reference-d", metavar="NAME=PATH",
                    action="append", default=[],
                    help="Per-cell reference concordance from "
                         "scripts/reference_concordance.py, used to divide "
                         "out each cell's assay reliability.  D is a Pearson "
                         "r, hence a reliability coefficient S/(S+N) with "
                         "S = e^2 var_A, so the GpC enzyme efficiency e is "
                         "ADDITIVE on the logit scale and subtracting the "
                         "reference's logit cancels it.  Measured on chrM, "
                         "which -- unlike cLADs, replication timing or PMDs "
                         "-- has no chromatin state for a tumour to "
                         "rearrange.  Cells with no usable reference keep "
                         "their uncorrected values.")
    ap.add_argument("--contrast-regress-coverage", action="store_true",
                    help="Subtract the shared D-versus-depth curve before the "
                         "contrast.  A shallow D is pulled toward the prior, "
                         "so where the arms differ in depth bin by bin that "
                         "becomes a bin-specific difference -- which a "
                         "permutation null KEEPS, because it is one.  The "
                         "curve is fitted once over the whole matrix, not per "
                         "bin: within a bin, depth is collinear with the arm "
                         "whenever the tumour cells are the thin ones, so a "
                         "per-bin fit would subtract the finding along with "
                         "the artefact.  Requires --pair-counts.")
    ap.add_argument("--contrast-permutations", type=int, default=0,
                    metavar="N",
                    help="Replace the parametric p with an empirical one from "
                         "N within-cell permutations.  Shuffling each cell's "
                         "values across bins keeps everything about the CELL "
                         "-- its offset, its spread, its coverage -- and "
                         "removes which bin is which, so the null already "
                         "contains any genome-wide batch or purity shift and "
                         "a bin clears it only by differing MORE than that "
                         "shift explains.  100 is usually enough; the null is "
                         "pooled over bins, so N permutations give N x n_bins "
                         "draws.")
    ap.add_argument("--contrast-batch-control", action="store_true",
                    help="After each contrast, run the SAME pipeline with the "
                         "arms taken from each --batch-column instead of from "
                         "biology, stratified by cell type.  It answers the "
                         "question a confounding statistic cannot: how many "
                         "bins this design calls at this FDR when the arms "
                         "differ only by processing.  A nuisance variable "
                         "yielding as many bins as the biology means the two "
                         "are not separable here, whatever Cramer's V says.")
    ap.add_argument("--contrast-consistency", action="store_true",
                    help="Also run the contrast SEPARATELY for each group in "
                         "the arm and keep only bins that move the same way, "
                         "significantly, in every one.  With two tumour "
                         "populations pooled, a bin can otherwise reach "
                         "significance on one of them alone.  This is a "
                         "reproducibility criterion, and where cells are not "
                         "independent replicates it is worth more than the "
                         "p-value it replaces.")
    ap.add_argument("--contrast-center", choices=["none", "cell-median"],
                    default="none",
                    help="'cell-median' subtracts each cell's own median "
                         "across bins before the contrast, removing the "
                         "per-cell offset that makes two arms differ "
                         "genome-wide.  Use it when the run warns that nearly "
                         "every bin moves the same way: it changes the "
                         "question from 'does this bin differ' -- which a "
                         "global offset answers yes to everywhere -- to 'does "
                         "this bin differ MORE than the genome does'.  It is "
                         "compositional, so it is safe while real aberrations "
                         "are a small share of the genome and will subtract "
                         "the finding when they are not.")
    ap.add_argument("--from-counts", action="append", default=None,
                    metavar="NAME=DIR",
                    help="The 05_concordance directory (or its counts/ "
                         "subdirectory) holding the RAW per-(cell, bin) soft "
                         "counts for --matrix NAME, from a step-5 run made "
                         "with --emit-counts.  Required by --contrast-test "
                         "glm.  Auto-discovered from the matrix's own "
                         "directory.")
    ap.add_argument("--replicate-column", "--donor-column", default=None,
                    metavar="COLUMN", dest="replicate_column",
                    help="Column of --samples naming the unit of BIOLOGICAL "
                         "REPLICATION -- the thing there are only a handful "
                         "of, and across which the p-value is taken.  "
                         "Defaults to --cell-type-column, which is usually "
                         "right here: tumor1 and tumor2 are two tumours and "
                         "GM12878, IMR90 and monocyte are three separate "
                         "sources, so a tumour-vs-reference contrast has two "
                         "clusters against three however many cells each "
                         "contributes.  Cells within one of them are not "
                         "replicates of each other, which is why the "
                         "cell-level p is not a p-value: it measured a 0.605 "
                         "false-positive rate against a nominal 0.05.  Point "
                         "this at a donor column instead when several cell "
                         "types come from one donor, or the shared donor "
                         "would be counted as independent evidence.")
    ap.add_argument("--glm-prior-df", type=float, default=4.0, metavar="D",
                    help="Bins' worth of borrowed variance added to the "
                         "donor design's own degrees of freedom, limma-style. "
                         "Measured on a fixture with donor-level bin effects, "
                         "two donors against three, at a nominal 0.05: 0 gave "
                         "power 0.55 at FPR 0.050, 4 gave 0.60 / 0.037, 8 "
                         "gave 0.70 / 0.045 and 20 gave 0.75 / 0.058.  Raise "
                         "it for power, and watch the FPR.")
    ap.add_argument("--contrast-permute-units", type=int, default=0,
                    metavar="N",
                    help="MEASURE how inflated this run's nominal p is on "
                         "THIS dataset, by relabelling which replication "
                         "units are the contrast arm and re-running the "
                         "identical test.  N IS A CAP, NOT A REPETITION "
                         "COUNT: unit labels are exhaustible, so with five "
                         "cell types split two against three there are "
                         "C(5,2) - 1 = 9 relabellings and asking for a "
                         "thousand still returns nine (with "
                         "--contrast-permute-scope reference and three "
                         "reference units, three).  What supplies the "
                         "resolution is not their number but the pooling "
                         "across bins -- 3 relabellings over 10,000 bins is "
                         "a pool of 30,000 -- and what would sharpen it is "
                         "more cell types, not more permutations.  Do NOT "
                         "manufacture units by splitting a cell type in "
                         "half: the halves share a donor, which reintroduces "
                         "exactly the pseudo-replication this is measuring.  "
                         "N caps how many "
                         "balanced relabellings to use; 0 (default) is off "
                         "and any positive number takes all of them when "
                         "there are fewer (with five cell types, two against "
                         "three, there are nine).  WHY.  A test can be the "
                         "most powerful on a dataset and still have a "
                         "nominal p that means nothing -- measured on one "
                         "run, welch and mwu on per-cell values recovered "
                         "far more spiked signal than the unit-level GLM and "
                         "carried a permuted false-positive rate two to "
                         "three times nominal.  Those do not contradict each "
                         "other: the extra power is real and so is the "
                         "inflation, because cells within a cell type are "
                         "not replicates.  The relabelled runs inherit the "
                         "SAME pseudo-replication, so what they return IS "
                         "this dataset's false-positive rate for this test.  "
                         "Reported with the nominal cut that buys a true 5 "
                         "percent and how many bins clear it.  IT DOES NOT "
                         "REPLACE THE P: an empirical p from these "
                         "relabellings is far too conservative, because a "
                         "split such as tumor1+GM12878 against the rest "
                         "still carries real cell-type biology -- measured, "
                         "it returned zero bins at FDR 0.05 whether 25, 5 or "
                         "2 percent of bins were genuinely differential.  It "
                         "rides along as contrast_<label>_pvalue_unitperm "
                         "for inspection.  For the same reason the measured "
                         "inflation is an UPPER bound.  Works with any "
                         "--contrast-test.  "
                         "Needs --replicate-column (or its cell-type "
                         "default) with 2+ units per arm.  Cost is N times "
                         "the contrast, so pair it with --glm-threads.")
    ap.add_argument("--contrast-permute-scope",
                    choices=["all", "reference"], default="all",
                    help="Which relabellings form the null.  'all' (default) "
                         "takes every BALANCED one, matching the real arm "
                         "sizes.  It is contaminated and cannot not be: with "
                         "two tumours among five units, three of the nine "
                         "relabellings put both tumours in the reference arm "
                         "and the other six put one in each, so every one "
                         "inherits some of the signal being tested -- which "
                         "is why the reported inflation is an upper bound.  "
                         "'reference' DROPS the contrast arm's units "
                         "entirely and relabels only within the reference, "
                         "giving a null with none of the effect in it.  Its "
                         "arms are smaller and unbalanced, so read it as a "
                         "check on the inflation rather than as the same "
                         "test.  Run both when the balanced null says the "
                         "real split is no more extreme than a relabelling: "
                         "that reading is only trustworthy if the clean null "
                         "agrees.")
    ap.add_argument("--contrast-permute-min-pool", type=int, default=2_000,
                    metavar="M",
                    help="Smallest per-stratum pool of relabelled statistics "
                         "to use on its own; below it, every testable bin's "
                         "draws are pooled instead.  Nine relabellings "
                         "cannot resolve a p below 0.1 per bin, so pooling "
                         "across bins within a coverage stratum is what "
                         "makes the empirical p usable at all.")
    ap.add_argument("--glm-threads", type=int, default=1, metavar="N",
                    help="Parallel workers for the per-bin GLM fits (about "
                         "15 ms each, so ~3 min for 10,000 bins on one "
                         "core).")
    ap.add_argument("--glm-batch", action="store_true",
                    help="Put --batch-column in the GLM design as a "
                         "covariate.  Preferable to correcting the matrix "
                         "first and then testing, which spends the batch's "
                         "degrees of freedom without telling the test it "
                         "spent them, so the standard errors come back too "
                         "small.  Off by default because a batch perfectly "
                         "aliased with the arm makes the fit singular and "
                         "the column is silently dropped.")
    ap.add_argument("--contrast-test", choices=["mwu", "welch", "glm"],
                    default="mwu",
                    help="Per-bin two-sample test for a contrast.  'mwu' "
                         "(default) is a two-sided Mann-Whitney U over cells: "
                         "no normality assumption, and one outlying cell "
                         "cannot carry a bin.  'welch' is the Welch t.  Either "
                         "way the volcano's x-axis is the difference of means, "
                         "in D units.  A rank test has a p-value FLOOR set by "
                         "the arm sizes -- 0.1 at 3 vs 3 -- and the run warns "
                         "when that floor puts FDR out of reach.  'glm' is "
                         "different in kind: it goes back to the RAW soft "
                         "counts (--from-counts) and fits a binomial GLM with "
                         "the expected pool as an offset, so its coefficient "
                         "IS the statistic and every cell is weighted by the "
                         "evidence it carries rather than one vote each -- a "
                         "bin measured on 3 read pairs is about 97 percent "
                         "prior, and mwu gives it the same vote as one "
                         "measured on 500.  Its p comes from a moderated t "
                         "over DONORS, not from the GLM; see --donor-column "
                         "and --glm-prior-df.")
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
    ap.add_argument("--min-cells-per-group", action="append", default=None,
                    metavar="N|NAME=N",
                    help="Cells a group needs in a bin before that bin is "
                         "used.  Default 3.  A bare number applies to every "
                         "--matrix; NAME=N sets it for one, since a dense "
                         "context and a sparse one do not want the same "
                         "cut -- GCH keeps most bins at any setting while "
                         "HCG is mostly imputed, and a single scalar forces "
                         "the looser of the two on both.")
    ap.add_argument("--min-groups", type=int, default=3)
    ap.add_argument("--max-legend-levels", type=int, default=40,
                    help="A batch strip with more levels than this (e.g. "
                         "lib_batch) is drawn but its legend is replaced by a "
                         "count; at/below it, the legend wraps into columns of "
                         "12.  Colours on the strip stay consistent regardless.")
    ap.add_argument("--blacklist-bed", action="append", default=None,
                    metavar="BED",
                    help="Repeatable.  Drop every bin overlapping these "
                         "regions -- ENCODE blacklist, custom masks -- before "
                         "anything is computed or drawn, so the filter "
                         "reaches the differential tests, the heatmaps, the "
                         "volcano and the emitted BEDs alike.  A blacklisted "
                         "bin is not a low signal, it is an artefact, and "
                         "leaving it in lets it be called differential and "
                         "then drawn as a result.  Matched against the bin "
                         "grid by overlap, so partial overlap drops the bin; "
                         "the run says how many bins went, and WARNS if the "
                         "file matched none -- an empty filter and a clean "
                         "genome look identical downstream.")
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
    from nome_hic_concordance.blacklist import (drop_blacklisted_bins,
                                                infer_bin_size)

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

    ref_specs = _named_map(args.contrast_reference_d, "--contrast-reference-d")
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
    joint_stat = {}
    base_args = args
    for mi, spec in enumerate(base_args.matrix):
        if "=" not in spec:
            sys.exit(f"--matrix must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        name = name.strip()
        # A PER-MATRIX view of the settings that a dense and a sparse
        # context do not want to share.  --min-cells-per-group is read by a
        # dozen call sites and by helpers that take `args` whole, so it is
        # resolved here and shadowed for this iteration rather than threaded
        # through each one.  Rebuilt from base_args every time, so one
        # matrix's value cannot leak into the next.
        args = argparse.Namespace(**vars(base_args))
        args.min_cells_per_group = per_matrix(
            base_args.min_cells_per_group, name, int, 3,
            "--min-cells-per-group")
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
        _n_matrix = M.shape[0]
        keep = [c for c in M.index if c in ct_of]
        # Say WHICH side is short when most cells fall away.  "only 16
        # matched cells" names the symptom; the usual cause is that the
        # matrix and the sample sheet disagree about cell naming, and
        # nothing in that message would tell you so.
        if len(keep) < 0.5 * _n_matrix:
            _miss = [c for c in M.index if c not in ct_of]
            print(f"  {name}: {len(keep):,} of {_n_matrix:,} matrix cells "
                  f"carry a {args.cell_type_column!r} in --samples "
                  f"({len(ct_of):,} cells there).\n"
                  f"    unmatched matrix cells look like: "
                  f"{_miss[:3]}\n"
                  f"    --samples cells look like:        "
                  f"{list(ct_of)[:3]}\n"
                  f"    If those two do not look like the same naming, that "
                  f"is the problem, not the data.", file=sys.stderr)
        M = M.loc[keep]
        if excl:
            M = M[[b for b in M.columns
                   if not chrom_excluded(bin_label_chrom(b), excl)]]
        # Applied to the MATRIX, before anything is computed or drawn, so the
        # filter reaches the tests, the heatmaps, the volcano and the emitted
        # BEDs alike.  --from-counts reindexes onto these columns, so the raw
        # counts follow without a second filter that could drift from this
        # one.
        if args.blacklist_bed:
            _before = M.shape[1]
            M = drop_blacklisted_bins(M, args.blacklist_bed, who=f"{name}")
            _gone = _before - M.shape[1]
            # Reported here rather than left to the library's logger: this
            # script does not configure logging, so that INFO line goes
            # nowhere and the run looks as though no filter was applied.
            if _gone:
                print(f"  {name}: blacklist dropped {_gone:,} of "
                      f"{_before:,} bins", file=sys.stderr)
            else:
                print(f"  {name}: WARNING the blacklist matched NO bin of "
                      f"{_before:,}.  Check the chromosome naming and the "
                      f"coordinates -- an empty filter and a clean genome "
                      f"look identical downstream.", file=sys.stderr)
        _need = 2 * args.min_cells_per_group * args.min_groups
        if M.shape[0] < _need:
            print(f"  {name}: only {M.shape[0]} matched cells, need "
                  f"{_need} (2 x --min-cells-per-group {args.min_cells_per_group}"
                  f" x --min-groups {args.min_groups}) — skipped",
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
        # The sampling variance and the read-pair counts the pipeline wrote
        # beside this matrix.  Auto-discovered so the common case needs no
        # flag; a bad explicit path is an error rather than a silent skip.
        V_df = _companion(path, args.delta_var, name,
                          "concordance_delta_var.parquet", "--delta-var",
                          matrix=M)
        C_df = _companion(path, args.pair_counts, name,
                          "pair_counts_per_bin.parquet", "--pair-counts",
                          matrix=M)
        # Named after the MATRIX's statistic, so a coordination matrix is
        # never gated on the direction's evidence.
        # Raw soft counts for --contrast-test glm.  The statistic is taken
        # from the MATRIX's filename, so a coordination matrix is never
        # modelled as the direction's 2x2 table.
        glm_counts = glm_statistic = None
        if args.contrast_test == "glm":
            glm_statistic = _glm_statistic_for(path)
            _cd = None
            for spec in (args.from_counts or []):
                if "=" not in spec:
                    sys.exit(f"--from-counts must be NAME=DIR, got {spec!r}")
                k, v = spec.split("=", 1)
                if k.strip() == name:
                    _cd = Path(v.strip())
            if _cd is None:
                _cd = Path(path).parent
            _cd = _cd / "counts" if (_cd / "counts").is_dir() else _cd
            missing = [k for k in GLM_COUNT_KEYS
                       if not (_cd / f"{k}.parquet").exists()]
            if missing:
                sys.exit(
                    f"--contrast-test glm: {_cd} is missing "
                    f"{', '.join(missing)}.  Re-run step 5 with "
                    f"--emit-counts, or point --from-counts {name}=<dir> at "
                    f"the run that has them.")
            glm_counts = {
                k: pd.read_parquet(_cd / f"{k}.parquet").reindex(
                    index=M.index, columns=M.columns).to_numpy(float)
                for k in GLM_COUNT_KEYS}
            print(f"  {name}: raw soft counts for the GLM from {_cd} "
                  f"(statistic {glm_statistic!r}); the imputed --matrix "
                  f"supplies only the cell x bin grid for this test.",
                  file=sys.stderr)
        donor_v = None
        if args.contrast_test == "glm" or args.contrast_permute_units:
            # Default to the cell type: it is the unit there are only a
            # handful of, and a contrast of two tumours against three
            # reference sources has 2 clusters against 3 no matter how many
            # cells each contributes.
            _rep_col = args.replicate_column or args.cell_type_column
            if _rep_col not in samples.columns:
                sys.exit(f"--replicate-column {_rep_col!r} not in --samples; "
                         f"have: {', '.join(map(str, samples.columns))}")
            donor_v = samples.set_index(samples.columns[0])[
                _rep_col].reindex(M.index).astype(str).to_numpy()
            _how = ("--replicate-column" if args.replicate_column
                    else "--cell-type-column (default)")
            print(f"  {name}: replication unit = {_rep_col!r} from {_how}, "
                  f"{len(set(donor_v[pd.notna(donor_v)]))} levels.  The "
                  f"p-value is taken ACROSS these, not across cells.",
                  file=sys.stderr)
        NR_df = _companion(path, args.null_rank, name,
                           rank_filename_for(path), "--null-rank", matrix=M)
        if NR_df is None and args.null_rank:
            sys.exit(f"--null-rank was given but nothing matched {name!r}; "
                     f"pass --null-rank {name}=<path>")
        R_ser = None
        if name in ref_specs:
            _rp, _, _rcol = str(ref_specs[name]).partition(":")
            _rcol = _rcol or "reference_d"
            rpath = Path(_rp)
            if not rpath.exists():
                sys.exit(f"--contrast-reference-d {name}: no such file "
                         f"{rpath}")
            rdf = pd.read_csv(rpath, sep="\t")
            if "cell" not in rdf.columns or _rcol not in rdf.columns:
                sys.exit(f"--contrast-reference-d {name}: {rpath} needs "
                         f"'cell' and {_rcol!r} columns (write it with "
                         f"scripts/reference_concordance.py)")
            R_ser = rdf.set_index("cell")[_rcol]
            shared = R_ser.index.intersection(M.index)
            if len(shared) < 0.5 * len(M.index):
                sys.exit(
                    f"--contrast-reference-d {name}: the reference covers "
                    f"{len(shared)} of {len(M.index)} cells in the matrix.  "
                    f"Correcting a minority would leave two scales in one "
                    f"matrix; regenerate it over the same cells.")
        if args.contrast_weight != "none" and V_df is None and C_df is None:
            here = sorted(x.name for x in Path(path).parent.glob("*.parquet"))
            sys.exit(
                f"--contrast-weight {args.contrast_weight} needs the "
                f"precision of each (cell, bin) estimate, and neither "
                f"concordance_delta_var.parquet nor pair_counts_per_bin"
                f".parquet is beside {path!r}.\n"
                f"  that directory holds: {', '.join(here) or '(no parquet)'}\n"
                f"  pass either --delta-var {name}=PATH (exact variances) or "
                f"--pair-counts {name}=PATH (read-pair counts, from which the "
                f"variance is approximated as c/n).  pair_counts_per_bin"
                f".parquet is written into the same 05_concordance directory "
                f"by the concordance step.")
        if args.contrast_weight != "none" and V_df is None:
            print(f"  {name}: no concordance_delta_var.parquet; approximating "
                  f"the per-estimate variance as c/n from "
                  f"pair_counts_per_bin.parquet, with c recovered per bin by "
                  f"moments.  Real between-cell spread lands in c, so this "
                  f"errs conservative.", file=sys.stderr)
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
        rank_made = []
        if args.null_rank_heatmaps != "off" and NR_df is not None:
            # Raw and SYMMETRIC, never z-scored.  The rank is already on a
            # calibrated per-cell scale with a meaningful zero -- "at its own
            # null" -- and z-scoring each bin across cells would throw that
            # away and manufacture contrast in a bin where every cell sits at
            # the null.  That is the one thing this panel exists to rule out.
            nrs = _align_signal(NR_df, list(M.index), sel_bins, name,
                                "null-rank", grid_align=False)
            if nrs is not None:
                rank_made = _companion_group(
                    nrs, "nullrank", "RdBu_r",
                    "signed -log10(p) vs drawn null", symmetric=True, **cg)
        if rank_made:
            print(f"  null-rank companions -> {', '.join(rank_made)}")
        elif args.null_rank_heatmaps == "on":
            sys.exit(f"--null-rank-heatmaps on but no rank matrix for "
                     f"{name!r}: pass --null-rank {name}=<path>, or re-run "
                     f"step 5 with --null-rank")
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
                # C(2E[F]-1) is on D's scale, and only D's.  Handed a
                # direction_logor matrix -- a log odds ratio of order 0.1 to
                # 1 against a reference of order C, about 1e-3 -- it is not
                # an approximation of the right reference, it is a different
                # quantity that happens to be numerically small enough to
                # look harmless.  Compared by robust scale rather than by
                # name, because the matrix arrives as an arbitrary path.
                _ms = _robust_sd(np.asarray(raw, float).ravel())
                _es = _robust_sd(np.asarray(expected, float).ravel())
                if (np.isfinite(_ms) and np.isfinite(_es) and _ms > 0
                        and _es > 0 and not (0.05 < _es / _ms < 20.0)):
                    sys.exit(
                        f"--sign-reference expected builds C*(2E[F]-1), "
                        f"which is on D's scale.  The reference's robust sd "
                        f"is {_es:.3g} against the matrix's {_ms:.3g}, a "
                        f"factor of {max(_es / _ms, _ms / _es):.0f}, so this "
                        f"matrix is not D and the reference does not belong "
                        f"to it.  For a log odds ratio the marginal is "
                        f"already divided out inside the statistic -- "
                        f"F_exp comes from the same bin's own ends -- so "
                        f"use --sign-reference zero, which is the honest "
                        f"reading there.")
                print(f"  {name}: per-bin expected-D reference from "
                      f"{Path(path).parent}/concordance_{{marginal,"
                      f"coordination}}.parquet (robust sd {_es:.3g} against "
                      f"the matrix's {_ms:.3g})", file=sys.stderr)
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
            Xc = M_contrast.to_numpy()
            # The reference correction runs BEFORE centring, not after.
            # It takes logit, so it needs the matrix on its natural (0, 1)
            # reliability scale -- and centring is exactly what takes it off
            # that scale, putting the median at 0 so the clip inside logit
            # swallows half the matrix.  Run the other way round the two
            # together collapsed every bin in a cell onto one value while
            # the run reported success.  Both are per-cell operations, so
            # the order is free to be the correct one: divide out the
            # measured reliability on the scale it was measured on, then
            # centre whatever location is left.
            Vc = Cc = NRc = None
            if V_df is not None:
                Vc = V_df.reindex(index=M.index,
                                  columns=M.columns).to_numpy()
            if C_df is not None:
                Cc = C_df.reindex(index=M.index,
                                  columns=M.columns).to_numpy()
            if NR_df is not None:
                # Deliberately NOT put through the per-cell corrections
                # below.  The rank is already referenced to each cell's own
                # null, which is what --contrast-center cell-median and
                # --contrast-reference-d approximate; centring it a second
                # time would subtract a per-cell offset that is not there.
                NRc = NR_df.reindex(index=M.index,
                                    columns=M.columns).to_numpy(float)
            if args.min_pairs == 0 and Cc is not None:
                _imp = float(np.mean(Cc == 0))
                if _imp > 0.05:
                    print(f"  {name}: WARNING {_imp:.1%} of (cell, bin) "
                          f"entries have NO read pairs of their own.  RWR "
                          f"filled every one of them, so the matrix carries "
                          f"0% NaN and the within-cell permutation -- which "
                          f"defends the null by holding NaN in place -- has "
                          f"nothing to hold.  Worse, RWR borrows from the "
                          f"k-NN graph, which is cell-type aware, so an "
                          f"imputed entry carries the arm's signature by "
                          f"construction.  A sparsely covered bin is then "
                          f"mostly arm signature, while the permuted column "
                          f"holds each cell's genome-wide MIX and dilutes "
                          f"it -- the bin clears a null that no longer "
                          f"describes it.  Measured on a fixture with no "
                          f"signal at all: 57.7% of sparse bins called at "
                          f"p < 0.05, against 0% once masked.  Pass "
                          f"--min-pairs 1.", file=sys.stderr)
            # HELD FIXED IN THE NULL, KEPT IN THE OBSERVED.  The imputed
            # value is still evidence -- RWR borrowed it from cells the
            # k-NN graph says are similar -- so it stays in the statistic.
            # What it must not do is move during permutation: shuffled, a
            # cell's imputed entries scatter their arm signature across
            # every bin position and the null is diluted by the genome-wide
            # MIX instead of carrying the bin's own share.  Held in place,
            # the signature sits in the null exactly as it sits in the
            # observed, and a bin clears only by differing MORE than the
            # imputation explains.
            _min_pairs = per_matrix(args.min_pairs, name, int, 0,
                                    "--min-pairs")
            _min_bin_frac = per_matrix(args.min_bin_frac, name, float, 0.0,
                                       "--min-bin-frac")
            _permutable = None
            if _min_pairs > 0:
                if Cc is None:
                    sys.exit(
                        f"--min-pairs needs the read-pair counts; pass "
                        f"--pair-counts {name}=<pair_counts_per_bin.parquet>")
                _permutable = Cc >= float(_min_pairs)
                print(f"  {name}: {1 - _permutable.mean():.1%} of "
                      f"(cell, bin) entries have < {_min_pairs} read "
                      f"pairs of their own; they stay in the observed "
                      f"statistic and are held fixed while the null is "
                      f"permuted", file=sys.stderr)
                _frac = _permutable.mean(axis=0)
                if _min_bin_frac > 0:
                    _drop = _frac < float(_min_bin_frac)
                    Xc[:, _drop] = np.nan
                    print(f"  {name}: {int(_drop.sum()):,} of "
                          f"{len(_drop):,} bins are measured in under "
                          f"{_min_bin_frac:.0%} of cells and are marked "
                          f"UNTESTABLE -- so they neither contribute to the "
                          f"null nor receive a call.  Their statistic would "
                          f"be mostly what RWR borrowed from the k-NN "
                          f"graph, and that graph is cell-type aware.",
                          file=sys.stderr)
                if args.contrast_permutations <= 0:
                    print(f"  {name}: NOTE --min-pairs only acts on the "
                          f"permutation null, and --contrast-permutations "
                          f"is 0, so it is doing nothing here",
                          file=sys.stderr)
            _rd = {}
            if R_ser is not None:
                Xc, _rd = correct_by_reference(
                    Xc, R_ser.reindex(M.index).to_numpy())
                if _rd.get("n_corrected"):
                    print(f"  {name}: divided out assay reliability using "
                          f"the reference for {_rd['n_corrected']}/"
                          f"{_rd['n_cells']} cells (reference D "
                          f"{_rd['reference_d_min']:.3f}.."
                          f"{_rd['reference_d_max']:.3f}, median "
                          f"{_rd['reference_d_median']:.3f})",
                          file=sys.stderr)
                if _rd.get("n_uncorrected"):
                    print(f"  {name}: WARNING {_rd['n_uncorrected']} cells "
                          f"had no usable reference and keep their "
                          f"uncorrected values -- two scales in one matrix.  "
                          f"Drop them, or lower --n-pairs when building the "
                          f"reference.", file=sys.stderr)
                if args.contrast_center == "cell-median":
                    print(f"  {name}: NOTE --contrast-center cell-median and "
                          f"--contrast-reference-d both remove a per-cell "
                          f"effect.  Using both is not wrong, but the "
                          f"reference is the measured version and centring "
                          f"the inferred one; the second has nothing left to "
                          f"take.", file=sys.stderr)
            if args.contrast_center == "cell-median":
                Xc = center_cells(Xc)
            _rg = {}
            if args.contrast_regress_coverage:
                if Cc is None:
                    sys.exit(
                        f"--contrast-regress-coverage needs the read-pair "
                        f"counts and none were loaded for {name}; pass "
                        f"--pair-counts {name}=pair_counts_per_bin.parquet")
                Xc, _rg = regress_out_coverage(Xc, Cc)
                if _rg.get("n_strata"):
                    print(f"  {name}: regressed out depth over "
                          f"{_rg['n_strata']} coverage strata, removing "
                          f"{_rg['frac_variance_removed']:.1%} of the "
                          f"variance in D", file=sys.stderr)
                else:
                    print(f"  {name}: depth not regressed out "
                          f"({_rg.get('note', 'unavailable')})",
                          file=sys.stderr)
            if Vc is None and Cc is not None and args.contrast_weight != "none":
                Vc = variance_from_counts(Cc, Xc, in_mask, out_mask)
            strat_per, cs = {}, None
            _lev = None
            if args.contrast_stratify:
                _lev = (batch_df_full.reindex(M.index)[args.contrast_stratify]
                        if args.contrast_stratify in batch_df_full.columns
                        else samples.set_index("cell_name")[
                            args.contrast_stratify].reindex(M.index)).to_numpy()
            dropout = dropout_report(Xc, in_mask, out_mask,
                                     args.min_cells_per_group, _lev)
            keep = dropout.get(
                "n_testable_all_strata", dropout["n_testable_pooled"])
            if keep < 0.5 * dropout["n_bins"]:
                detail = ", ".join(
                    f"{k[len('n_testable_'):]}: {v:,}"
                    for k, v in dropout.items()
                    if k.startswith("n_testable_")
                    and k not in ("n_testable_pooled", "n_testable_all_strata"))
                print(f"  contrast {label!r}: only {keep:,} of "
                      f"{dropout['n_bins']:,} bins are testable at all "
                      f"({dropout['n_testable_pooled']:,} without stratifying"
                      + (f"; per stratum {detail}" if detail else "") + ").  "
                      f"A bin needs {args.min_cells_per_group} covered cells "
                      f"in EACH arm"
                      + (" of EACH stratum" if _lev is not None else "")
                      + f"; the median bin has {dropout['median_cells_in']:.0f} "
                      f"in the arm and {dropout['median_cells_out']:.0f} in the "
                      f"reference.  Most of the genome is UNTESTED rather than "
                      f"unremarkable, so a small hit count here is reach, not "
                      f"effect size.", file=sys.stderr)
            if args.contrast_stratify:
                use_v = (Vc if (args.contrast_weight != "none"
                                and V_df is not None) else None)
                cs, strat_per = stratified_contrast(
                    Xc, _lev, in_mask, out_mask,
                    args.min_cells_per_group, args.contrast_test,
                    use_v, args.contrast_weight,
                    counts=(Cc if args.contrast_weight != "none" else None))
                if cs is None:
                    print(f"  contrast {label!r}: --contrast-stratify "
                          f"{args.contrast_stratify!r} left fewer than two "
                          f"strata with both arms present "
                          f"({sorted(strat_per)}) -- falling back to the "
                          f"unstratified contrast.", file=sys.stderr)
                else:
                    print(f"  contrast {label!r}: stratified by "
                          f"{args.contrast_stratify} over "
                          f"{cs['n_strata']} levels ({', '.join(strat_per)}); "
                          f"the difference reported is the mean of the "
                          f"within-stratum differences.", file=sys.stderr)
                    for lv, gcs in strat_per.items():
                        scores_df[f"contrast_{label}_diff_{lv}"] = gcs["diff"]
                        extra_cols.append(f"contrast_{label}_diff_{lv}")
            if args.contrast_test == "glm":
                # RAW counts, not the imputed matrix.  RWR's k-NN graph is
                # built from accessibility, so an imputed entry carries its
                # cell type's signature by construction: testing arms on it
                # asks whether the imputation knew which arm a cell was in.
                if glm_counts is None:
                    sys.exit(
                        "--contrast-test glm needs the raw soft counts: pass "
                        "--from-counts <05_concordance dir> from a step-5 run "
                        "made with --emit-counts")
                cs = glm_contrast(
                    glm_counts, in_mask, out_mask, glm_statistic,
                    list(M.index), batch_df=(batch_df if args.glm_batch
                                             else None),
                    donor=donor_v, min_cells=args.min_cells_per_group,
                    min_pairs=max(1, int(_min_pairs)),
                    prior_df=args.glm_prior_df, n_jobs=args.glm_threads)
                print(f"  contrast {label!r}: binomial GLM on raw counts, "
                      f"design {' + '.join(cs['design'])}, statistic "
                      f"{glm_statistic!r}.  The coefficient IS the "
                      f"difference in that statistic between the arms, "
                      f"coverage-weighted.", file=sys.stderr)
                if cs.get("p_unit") == "donor":
                    print(f"  contrast {label!r}: p from a moderated t over "
                          f"{cs['n_donors']} replication units on their "
                          f"pooled logits "
                          f"(prior df {args.glm_prior_df:g}).  NOT from the "
                          f"GLM: on a fixture with donor-level bin effects "
                          f"the cell-level p ran at a 0.605 false-positive "
                          f"rate against a nominal 0.05, and 0.189 even with "
                          f"donor-clustered standard errors -- five clusters "
                          f"is far below what those need.",
                          file=sys.stderr)
                else:
                    print(f"  contrast {label!r}: WARNING no replication "
                          f"units resolved, so there is NO p-value -- only "
                          f"the effect size.  A cell-level p here measured a "
                          f"0.605 false-positive rate against a nominal 0.05, "
                          f"because cells from one cell type or donor are "
                          f"not replicates of each other.", file=sys.stderr)
            if cs is None:                     # not stratified, or it fell back
                if args.contrast_weight != "none":
                    cs = weighted_contrast(Xc, Vc, in_mask, out_mask,
                                           args.min_cells_per_group,
                                           args.contrast_weight)
                else:
                    cs = contrast_stats(Xc, in_mask, args.min_cells_per_group,
                                        out_mask, args.contrast_test)
            # ---- Empirical p from relabelled units ---------------------
            summary_unitperm = locals().get("summary_unitperm") or {}
            unitperm_nulls = locals().get("unitperm_nulls") or {}
            unitnull_q = unitperm_p = None
            if args.contrast_permute_units and cs is not None \
                    and donor_v is not None:
                _u_all = np.asarray(donor_v, dtype=object)
                _ua = sorted(set(_u_all[in_mask]))
                _ub = sorted(set(_u_all[out_mask]))
                _units = _ua + _ub
                if len(_ua) < 2 or len(_ub) < 2:
                    print(f"  contrast {label!r}: --contrast-permute-units "
                          f"needs 2+ replication units per arm, have "
                          f"{len(_ua)} vs {len(_ub)}; skipped.",
                          file=sys.stderr)
                else:
                    _rel = unit_relabellings(
                        _units, len(_ua), _ua,
                        max_n=args.contrast_permute_units, seed=args.seed,
                        scope=args.contrast_permute_scope)
                    if not _rel:
                        print(f"  contrast {label!r}: "
                              f"--contrast-permute-scope "
                              f"{args.contrast_permute_scope} leaves no "
                              f"relabelling (needs 2+ units outside the "
                              f"contrast arm); skipped.", file=sys.stderr)
                    _pool_units = ([u for u in _units if u not in _ua]
                                   if args.contrast_permute_scope == "reference"
                                   else _units)
                    _nulls = []
                    for _combo in _rel:
                        _mi = np.isin(_u_all, list(_combo))
                        _mo = np.isin(_u_all, [u for u in _pool_units
                                               if u not in _combo])
                        if args.contrast_test == "glm":
                            _r = glm_contrast(
                                glm_counts, _mi, _mo, glm_statistic,
                                list(M.index),
                                batch_df=(batch_df if args.glm_batch
                                          else None),
                                donor=donor_v,
                                min_cells=args.min_cells_per_group,
                                min_pairs=max(1, int(_min_pairs)),
                                prior_df=args.glm_prior_df,
                                n_jobs=args.glm_threads)
                        else:
                            _r = contrast_stats(Xc, _mi,
                                                args.min_cells_per_group,
                                                _mo, args.contrast_test)
                        _nulls.append(_r["p"])
                    _cstrata = (coverage_strata(Cc >= max(1, int(_min_pairs)))
                                if Cc is not None else None)
                    _pe, _pi = unit_permutation_p(
                        cs["p"], _nulls, strata=_cstrata,
                        min_pool=args.contrast_permute_min_pool)
                    _flat = np.concatenate(
                        [q[np.isfinite(q)] for q in _nulls])
                    _fpr = float(np.mean(_flat <= 0.05))
                    _thr = float(np.quantile(_flat, 0.05)) \
                        if _flat.size >= 200 else float("nan")
                    _n_nom = int(np.sum(np.isfinite(cs["p"])
                                        & (cs["p"] <= 0.05)))
                    _n_cal = int(np.sum(np.isfinite(cs["p"])
                                        & (cs["p"] <= _thr))) \
                        if np.isfinite(_thr) else 0
                    # DIAGNOSTIC, NOT A REPLACEMENT.  Swapping the nominal p
                    # for this empirical one was the first design and it is
                    # wrong: a relabelled split such as {tumor1, GM12878}
                    # against the rest still carries real cell-type biology,
                    # so the null is contaminated with signal and the p is
                    # far too conservative -- measured, it returned ZERO
                    # bins at FDR 0.05 whether 25, 5 or 2 percent of bins
                    # were genuinely differential.  What the relabellings DO
                    # settle is how badly the nominal p is inflated on this
                    # dataset, and what nominal cut buys a true 5 percent.
                    # Both are reported; the p that goes into FDR is left
                    # alone, and the empirical one rides along as a column.
                    # THE NUMBER THAT SETTLES IT.  A calibrated per-bin
                    # threshold is not a multiple-testing correction: at a
                    # true 5 percent over N bins, 0.05 N of them clear by
                    # chance.  What says whether the contrast found anything
                    # is how many MORE than that it found.
                    _n_test = int(np.sum(np.isfinite(cs["p"])))
                    _exp_nom = _fpr * _n_test
                    _exp_cal = 0.05 * _n_test
                    _enr_nom = _n_nom / _exp_nom if _exp_nom > 0 else np.nan
                    _enr_cal = _n_cal / _exp_cal if _exp_cal > 0 else np.nan
                    # HOW MANY RELABELLINGS EXIST, not how many were
                    # asked for.  --contrast-permute-units is a CAP: the
                    # unit labels are exhaustible, so with five cell types
                    # split two against three there are C(5,2) - 1 = 9 and
                    # asking for a thousand still returns nine.  Saying only
                    # "9 relabellings" invites it to be read as a repetition
                    # count that could be raised.
                    _n_poss = len(unit_relabellings(
                        _units, len(_ua), _ua,
                        scope=args.contrast_permute_scope))
                    _cap_note = (
                        "the complete enumeration; the unit labels are "
                        "exhaustible, so raising --contrast-permute-units "
                        "cannot add any"
                        if _pi["n_relabellings"] >= _n_poss
                        else f"capped from {_n_poss} by "
                             f"--contrast-permute-units")
                    print(f"  contrast {label!r}: "
                          f"{_pi['n_relabellings']} of {_n_poss} possible "
                          f"unit relabellings ({_cap_note}).  Median pool "
                          f"{_pi['median_pool']:,} bin-statistics -- but "
                          f"they reuse the same cells and neighbouring bins "
                          f"are correlated, so the real resolution is "
                          f"coarser than 1/{_pi['median_pool'] + 1:,}.  More "
                          f"cell types, not more permutations, is what would "
                          f"sharpen this.  ON THIS DATA the "
                          f"nominal {args.contrast_test!r} p runs at "
                          f"{_fpr:.3f} against a nominal 0.05 "
                          f"({_fpr / 0.05:.1f}x); a true 5 percent needs "
                          f"p <= {_thr:.2g}.  "
                          + (f"The contrast arm is excluded from this null, "
                             f"so the effect being tested cannot leak into "
                             f"it -- though the reference cell types still "
                             f"differ from each other, which is what a "
                             f"cell-type contrast is meant to survive."
                             if args.contrast_permute_scope == "reference"
                             else f"The relabelled null still mixes real "
                                  f"cell types, so {_fpr:.3f} is an UPPER "
                                  f"bound on the inflation."),
                          file=sys.stderr)
                    print(f"  contrast {label!r}: at p<=0.05, {_n_nom:,} "
                          f"bins against {_exp_nom:,.0f} expected from a "
                          f"relabelling -> {_enr_nom:.2f}x.  At p<={_thr:.2g},"
                          f" {_n_cal:,} against {_exp_cal:,.0f} -> "
                          f"{_enr_cal:.2f}x.  THE ENRICHMENT IS THE RESULT, "
                          f"not the count: a calibrated per-bin cut is not a "
                          f"multiple-testing correction, and at a true 5 "
                          f"percent over {_n_test:,} bins about "
                          f"{_exp_cal:,.0f} clear by chance.",
                          file=sys.stderr)
                    if max(_enr_nom, _enr_cal) < 1.3:
                        print(f"  contrast {label!r}: WARNING the real split "
                              f"produces no more extreme bins than a random "
                              f"relabelling of the same cell types "
                              f"({_enr_cal:.2f}x at the calibrated cut).  "
                              f"Whatever list you take here is "
                              f"indistinguishable from one an arbitrary "
                              f"cell-type split would give.  Before "
                              f"concluding there is no effect, note this "
                              f"null is CONTAMINATED -- with 2 units in the "
                              f"contrast arm, some relabellings put both of "
                              f"them in the reference and inherit the very "
                              f"signal being tested.  "
                              + ("This null already excludes the contrast "
                                 "arm, so that escape is closed."
                                 if args.contrast_permute_scope == "reference"
                                 else "Re-run with --contrast-permute-scope "
                                      "reference for a null the contrast arm "
                                      "is excluded from."),
                              file=sys.stderr)
                    elif _fpr > 0.10:
                        print(f"  contrast {label!r}: WARNING at "
                              f"{_fpr / 0.05:.1f}x nominal, a BH-FDR at "
                              f"q=0.05 on these p-values is controlling "
                              f"something nearer q={_fpr / 0.05 * 0.05:.2f}. "
                              f"Read the {_n_cal:,} bins clearing "
                              f"p <= {_thr:.2g} ({_enr_cal:.2f}x chance), or "
                              f"switch to --contrast-test glm, whose p is "
                              f"taken across replication units.",
                              file=sys.stderr)
                    unitperm_nulls[f"{name}:{label}"] = [
                        np.asarray(v, float) for v in _nulls]
                    scores_df[f"contrast_{label}_pvalue_unitperm"] = _pe
                    extra_cols.append(f"contrast_{label}_pvalue_unitperm")
                    # THE COLUMN TO THRESHOLD ON.  BH assumes the p-values
                    # are uniform under the null and on this data they are
                    # not, so BH at q=0.05 controls something nearer the
                    # measured rate.  This counts the false positives from
                    # the null that was actually observed, so q <= 0.1 means
                    # what it says whatever the nominal p is worth.
                    _q, _qi = empirical_null_qvalue(cs["p"], _flat,
                                                    p_perm=_pe)
                    unitnull_q = _q
                    unitperm_p = _pe
                    scores_df[f"contrast_{label}_qvalue_unitnull"] = _q
                    extra_cols.append(f"contrast_{label}_qvalue_unitnull")
                    _at = {t: int(np.sum(np.isfinite(_q) & (_q <= t)))
                           for t in (0.05, 0.10, 0.20)}
                    print(f"  contrast {label!r}: FDR against that null "
                          f"(column contrast_{label}_qvalue_unitnull) -- "
                          + ", ".join(f"{v:,} bins at q<={t:g}"
                                      for t, v in _at.items())
                          + f".  Threshold THIS, not the nominal p: it "
                            f"counts false positives from the null that was "
                            f"measured rather than assuming a uniform one.  "
                            f"Finest resolvable q here is {_qi['finest_q']:.3f}"
                            f" (null pool {_qi['n_null']:,}).",
                          file=sys.stderr)
                    summary_unitperm[f"{name}:{label}:qvalue"] = {
                        f"n_q{int(t * 100):02d}": v for t, v in _at.items()}
                    summary_unitperm[f"{name}:{label}"] = {
                        "n_relabellings": _pi["n_relabellings"],
                        "nominal_fpr_at_0.05": _fpr,
                        "threshold_for_true_5pct": _thr,
                        "n_bins_nominal_p05": _n_nom,
                        "n_bins_calibrated": _n_cal,
                        "n_bins_tested": _n_test,
                        "enrichment_at_0.05": _enr_nom,
                        "enrichment_at_calibrated": _enr_cal,
                        "scope": args.contrast_permute_scope,
                        "test": args.contrast_test}
            if C_df is not None:
                cov = coverage_report(
                    C_df.reindex(index=M.index, columns=M.columns)
                    .to_numpy(), in_mask, out_mask)
                summary_cov = cov
                if (np.isfinite(cov["pairs_ratio"])
                        and not 0.67 < cov["pairs_ratio"] < 1.5):
                    print(f"  contrast {label!r}: NOTE the arms differ in "
                          f"coverage -- median {cov['median_pairs_in']:.0f} "
                          f"vs {cov['median_pairs_out']:.0f} read pairs per "
                          f"bin ({cov['pairs_ratio']:.2f}x).  D is estimated "
                          f"from those pairs, so the thinner arm's estimates "
                          f"are attenuated and a genome-wide difference can "
                          f"follow from coverage alone.  --contrast-weight "
                          f"random-effects weights each cell by the precision "
                          f"of its own estimate.", file=sys.stderr)
            else:
                summary_cov = {}
            if args.contrast_permutations > 0:
                pn = permutation_null(
                    Xc, in_mask, out_mask, args.min_cells_per_group,
                    args.contrast_test, args.contrast_permutations,
                    Vc if args.contrast_weight != "none" else None,
                    args.contrast_weight,
                    strata=_lev if args.contrast_stratify else None,
                    counts=(Cc if args.contrast_weight != "none" else None),
                    permutable=_permutable,
                    pool_strata=(coverage_strata(_permutable)
                                 if _permutable is not None else None))
                if pn is None:
                    print(f"  contrast {label!r}: permutation null produced "
                          f"no usable statistic -- keeping the parametric p",
                          file=sys.stderr)
                else:
                    n_para = int(np.sum(np.isfinite(_bh_fdr(cs["p"]))
                                        & (_bh_fdr(cs["p"])
                                           < args.contrast_fdr)))
                    cs["p_parametric"] = cs["p"]
                    cs["p"] = pn["p_empirical"]
                    n_emp = int(np.sum(np.isfinite(_bh_fdr(cs["p"]))
                                       & (_bh_fdr(cs["p"])
                                          < args.contrast_fdr)))
                    print(f"  contrast {label!r}: {args.contrast_permutations} "
                          f"within-cell permutations, {pn['n_null']:,} null "
                          f"draws -> {n_emp:,} bins at FDR<"
                          f"{args.contrast_fdr} against {n_para:,} from the "
                          f"parametric p.  The gap is what a whole-cell shift "
                          f"was buying.", file=sys.stderr)
            q = _bh_fdr(cs["p"])
            # The threshold the SELECTION uses.  It is --contrast-fdr for
            # every mode that selects on an error RATE, and
            # --contrast-unit-p for the one that selects on a raw p.
            sel_thr = args.contrast_fdr
            if args.contrast_select == "unit-p":
                if unitperm_p is None:
                    sys.exit(
                        "--contrast-select unit-p needs the relabelled null: "
                        "add --contrast-permute-units 9 (and, for a null the "
                        "contrast arm is excluded from, "
                        "--contrast-permute-scope reference).")
                # SELECTING ON A P, NOT ON A RATE.  Legitimate when the set
                # is a screen -- a foreground for enrichment, say -- and not
                # a list of bins each claimed to differ.  What it is not is
                # a multiple-testing correction: at a calibrated p <= t over
                # N tested bins, t x N of them clear by chance whatever the
                # calibration.  The count is printed so that cannot be
                # missed.
                scores_df[f"contrast_{label}_fdr_bh"] = q
                extra_cols.append(f"contrast_{label}_fdr_bh")
                sel_thr = args.contrast_unit_p
                _n_t = int(np.sum(np.isfinite(unitperm_p)))
                _n_p = int(np.sum(np.isfinite(unitperm_p)
                                  & (unitperm_p < sel_thr)))
                _exp = sel_thr * _n_t
                print(f"  contrast {label!r}: selecting on the relabelled-"
                      f"null p at {sel_thr:g}, NOT on an FDR -> {_n_p:,} "
                      f"bins, against {_exp:,.0f} expected by chance over "
                      f"{_n_t:,} tested ({_n_p / max(_exp, 1e-9):.2f}x).  "
                      f"This is a per-bin rate, so about {_exp:,.0f} of "
                      f"those are false however well calibrated the p is.  "
                      f"BH on the same p is kept as "
                      f"contrast_{label}_fdr_bh; --contrast-select unit-fdr "
                      f"selects on it.", file=sys.stderr)
                q = unitperm_p
            if args.contrast_select == "unit-fdr":
                if unitnull_q is None:
                    sys.exit(
                        "--contrast-select unit-fdr needs the relabelled "
                        "null: add --contrast-permute-units 9 (and, for a "
                        "null the contrast arm is excluded from, "
                        "--contrast-permute-scope reference).")
                # THE POINT OF MEASURING IT.  Until now the relabellings
                # were a diagnostic and the call set still came from BH on
                # the nominal p -- which on this data is the p that ran at
                # 0.24 and 0.38 against 0.05, so "FDR < 0.05" was not 0.05.
                # Selecting on the calibrated q makes every downstream
                # artefact -- the BEDs, the volcano, the joint contrast, the
                # heatmaps -- reflect the rate that was actually measured.
                _n_bh = int(np.sum(np.isfinite(q) & (q < args.contrast_fdr)))
                _n_un = int(np.sum(np.isfinite(unitnull_q)
                                   & (unitnull_q < args.contrast_fdr)))
                # THE FLOOR IS SET BY THE RELABELLINGS, NOT THE POOL.  The
                # empirical P(null <= t) bottoms out at one draw in
                # R x n_bins, and the q multiplies it back by n_bins over
                # the rank -- so the most extreme bin can never score below
                # about 1/R, and with three relabellings that is 0.33.
                # Pooling across bins buys resolution in the middle of the
                # list, where a real set of discoveries lives, but nothing
                # at rank 1.  If the whole curve sits above the threshold
                # there is nothing to select and the reason is arithmetic.
                _fq = float(np.nanmin(unitnull_q)) \
                    if np.isfinite(unitnull_q).any() else float("nan")
                if _n_un == 0 and np.isfinite(_fq) \
                        and _fq >= args.contrast_fdr:
                    print(f"  contrast {label!r}: WARNING no bin can clear "
                          f"q<{args.contrast_fdr} here -- the smallest q "
                          f"this null expresses is {_fq:.2f}, and with "
                          f"{_pi['n_relabellings']} relabellings the floor "
                          f"is about 1/{_pi['n_relabellings']} = "
                          f"{1 / max(_pi['n_relabellings'], 1):.2f} at the "
                          f"top of the list.  Either the signal is not "
                          f"strong enough to push a large SET of bins down "
                          f"(which is what buys a small q), or you need more "
                          f"cell types.  --contrast-permute-scope all gives "
                          f"9 relabellings instead of 3, at the cost of a "
                          f"null the contrast arm leaks into.",
                          file=sys.stderr)
                print(f"  contrast {label!r}: selecting on the relabelled-"
                      f"null q, not on BH of the nominal p -> {_n_un:,} bins "
                      f"at q<{args.contrast_fdr} against {_n_bh:,} from BH.  "
                      f"BH assumes the nominal p is uniform under the null "
                      f"and on this data it is not, which is what the "
                      f"relabellings just measured.", file=sys.stderr)
                q = unitnull_q
            gs = global_shift_report(cs["diff"],
                                     np.isfinite(q) & (q < args.contrast_fdr))
            if gs and gs["frac_bins_same_direction"] > 0.9:
                n_sig = int(np.sum(np.isfinite(q) & (q < args.contrast_fdr)))
                n_tot = int(np.isfinite(cs["diff"]).sum())
                print(f"  contrast {label!r}: WARNING "
                      f"{gs['frac_bins_same_direction']:.0%} of bins move the "
                      f"same way (median difference {gs['median_diff']:+.3g}, "
                      f"{gs['median_over_mad']:.1f}x the spread between "
                      f"bins), and {n_sig:,}/{n_tot:,} reach FDR<"
                      f"{args.contrast_fdr}.  That is a genome-wide OFFSET "
                      f"between the arms, not local aberration: the BED holds "
                      f"most of the genome and cannot serve as a foreground, "
                      f"since the enrichment would be the genome against "
                      f"itself.  The offset may be real (one arm less "
                      f"coordinated overall) or technical (coverage, purity, "
                      f"cell quality changing how D is estimated) -- check "
                      f"per-cell mean D against depth.  To ask instead which "
                      f"bins depart from the genome-wide difference, rerun "
                      f"with --contrast-center cell-median.",
                      file=sys.stderr)
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
            rank_stats = (rank_departure(NRc, in_mask, out_mask,
                                         args.null_rank_min)
                          if NRc is not None else None)
            if rank_stats is not None:
                _fa = np.nanmean(rank_stats["rank_frac_a"])
                _fb = np.nanmean(rank_stats["rank_frac_b"])
                print(f"  contrast {label!r}: departs from its own drawn "
                      f"null (|score| >= {args.null_rank_min:.2f}) in "
                      f"{_fa:.1%} of contrast cell-bins and {_fb:.1%} of "
                      f"reference ones", file=sys.stderr)
            rank = np.where(np.isfinite(cs["p"]),
                            -np.log10(np.maximum(cs["p"], 1e-300)), -np.inf)
            mask = np.zeros(M.shape[1], bool)
            if args.contrast_select in ("fdr", "unit-fdr", "unit-p"):
                # Significance, not a fixed count.  A rank cut returns N bins
                # whether or not anything differs, so "the top 500" reads as a
                # finding even when the contrast is empty; an FDR cut can
                # return nothing, which is the honest answer in that case.
                sig = (np.isfinite(q) & (q < sel_thr)
                       & np.isfinite(cs["diff"]))
                if args.contrast_min_diff > 0:
                    sig &= np.abs(cs["diff"]) >= args.contrast_min_diff
                if args.cnv_max_delta > 0 and cnv is not None:
                    sig &= ~(np.isfinite(cnv)
                             & (np.abs(cnv) > args.cnv_max_delta))
                # DOES EITHER ARM DEPART FROM ITS OWN NULL.  The test above
                # asks only whether the arms differ, and at a few hundred
                # cells per arm a small consistent offset between two arms
                # that both sit at their null clears FDR -- a difference in
                # nothing.  Applied here rather than by NaN-ing the column
                # so a bin that cleared FDR and failed this shows on the
                # volcano as "FDR only, not kept": how many calls rested on
                # nothing is worth seeing, not hiding.
                if rank_stats is not None and args.null_rank_min_frac > 0:
                    _dep = (np.isfinite(rank_stats["rank_frac_max"])
                            & (rank_stats["rank_frac_max"]
                               >= float(args.null_rank_min_frac)))
                    _lost = int((sig & ~_dep).sum())
                    sig &= _dep
                    print(f"  contrast {label!r}: {_lost:,} bins cleared "
                          f"FDR but no arm departs from its own drawn null "
                          f"in {args.null_rank_min_frac:.0%} of its cells "
                          f"(|score| >= {args.null_rank_min:.2f}); dropped, "
                          f"and drawn on the volcano as 'FDR only, not "
                          f"kept'", file=sys.stderr)
                mask = sig
            else:
                keep_rank = np.argsort(rank)[::-1][: args.n_contrast_bins
                                                   or args.n_bins]
                keep_rank = keep_rank[np.isfinite(rank[keep_rank])]
                mask[keep_rank] = True
            cov_conf = {}
            if Cc is not None:
                cov_conf = coverage_confound(Cc, in_mask, out_mask,
                                             cs["diff"])
            skew = direction_skew(cs["diff"],
                                  np.isfinite(q) & (q < args.contrast_fdr))
            if cov_conf and skew and _rg.get("n_strata"):
                r = cov_conf["pearson_r_effect_vs_coverage_gap"]
                # The regression already ran.  A surviving correlation is a
                # STRONGER signal than the original warning, not a resolved
                # one: the pooled curve did not describe this gap, so do not
                # read the corrected numbers as clean.
                if abs(r) > 0.5 and skew["frac_one_direction"] > 0.9:
                    print(f"  contrast {label!r}: WARNING the effect STILL "
                          f"tracks the per-bin coverage gap (r = {r:+.2f}) "
                          f"after regressing depth out over "
                          f"{_rg['n_strata']} strata, and "
                          f"{skew['frac_one_direction']:.0%} of the "
                          f"{skew['n_significant']:,} significant bins point "
                          f"the same way.  The pooled curve did not describe "
                          f"this gap -- treat these bins as unresolved, not "
                          f"corrected.", file=sys.stderr)
            elif cov_conf and skew:
                r = cov_conf["pearson_r_effect_vs_coverage_gap"]
                if abs(r) > 0.5 and skew["frac_one_direction"] > 0.9:
                    print(f"  contrast {label!r}: WARNING the effect tracks "
                          f"the per-bin COVERAGE GAP between the arms "
                          f"(r = {r:+.2f}) and {skew['frac_one_direction']:.0%}"
                          f" of the {skew['n_significant']:,} significant bins "
                          f"point the same way ({skew['n_down']:,} down, "
                          f"{skew['n_up']:,} up).  A gap that varies by bin is "
                          f"bin-specific, so a permutation null keeps it, and "
                          f"shallower D is pulled toward the prior -- which is "
                          f"why it only ever goes one way.  On a fixture with "
                          f"identical biology and only such a gap this "
                          f"produced 1,431 bins, all down, at r = 0.97.  "
                          f"Re-run with --contrast-regress-coverage, which "
                          f"subtracts the shared D-versus-depth curve fitted "
                          f"over the whole matrix; on that same fixture it "
                          f"took 1,492 false positives down to 2 while "
                          f"keeping all 40 planted real bins.  Whatever "
                          f"survives it is not depth.", file=sys.stderr)
            cons_info = {}
            # When stratifying, consistency runs across the STRATA: combining
            # per-stratum evidence (Stouffer) asks whether there is an average
            # effect, which a bin real in one patient alone still passes.
            # Requiring it in every patient is the different, stricter
            # question, and the one "specific to this tumour type" means.
            per_g = None
            if args.contrast_consistency and strat_per and cs is not None:
                per_g = strat_per
            elif args.contrast_consistency and len(present) > 1:
                per_g = group_consistency(
                    Xc, ct, present, out_mask, args.min_cells_per_group,
                    args.contrast_test,
                    Vc if args.contrast_weight != "none" else None,
                    args.contrast_weight)
            if per_g is not None:
                cmask, cons_info = consistency_mask(per_g, args.contrast_fdr)
                if cmask is not None:
                    for g, gcs in per_g.items():
                        scores_df[f"contrast_{label}_diff_{g}"] = gcs["diff"]
                        extra_cols.append(f"contrast_{label}_diff_{g}")
                    before = int(mask.sum())
                    mask = mask & cmask
                    print(f"  contrast {label!r}: consistency across "
                          f"{'+'.join(cons_info['groups'])} kept "
                          f"{int(mask.sum()):,} of {before:,} bins "
                          f"({cons_info['n_significant_in_any']:,} were "
                          f"significant in at least one group, "
                          f"{cons_info['n_consistent']:,} in all of them with "
                          f"the same sign).", file=sys.stderr)
            elif args.contrast_consistency:
                print(f"  contrast {label!r}: --contrast-consistency needs "
                      f"more than one group in the arm, or "
                      f"--contrast-stratify; the arm {present} has one group "
                      f"and no strata.", file=sys.stderr)
            # A contrast bin driven by batch is worthless in exactly the same
            # way a marker bin is, so it faces the same filter.
            clean = ~(np.isfinite(b_icc) & (b_icc > args.max_batch_icc))
            pre = f"contrast_{label}"
            if args.volcano:
                vpath = out_dir / f"volcano_{name}_{label}.pdf"
                n_up, n_dn = volcano_figure(
                    cs["diff"], q, mask, vpath,
                    f"{name}  {label} ({'+'.join(present)} {arm})",
                    args.contrast_fdr, args.contrast_min_diff,
                    passed_fdr=np.isfinite(q) & (q < args.contrast_fdr))
                print(f"  contrast {label!r}: {n_up:,} up / {n_dn:,} down "
                      f"-> {vpath.name}", file=sys.stderr)
            joint.setdefault(label, {})[name] = pd.DataFrame(
                {"diff": cs["diff"], "fdr": q, "sig": mask},
                index=pd.Index(M.columns, name="bin"))
            joint_mats[name] = M
            joint_stat[name] = statistic_label(path)
            scores_df[f"{pre}_diff"] = cs["diff"]
            scores_df[f"{pre}_cohens_d"] = cs["d"]
            scores_df[f"{pre}_pvalue"] = cs["p"]
            scores_df[f"{pre}_fdr"] = q
            scores_df[pre] = mask
            extra_cols += [f"{pre}_diff", f"{pre}_cohens_d", f"{pre}_pvalue",
                           f"{pre}_fdr", pre]
            # Reported whether or not they gated anything: "the arms differ"
            # and "either arm is doing something" are separate claims, and a
            # table that carries only the first invites the second to be
            # assumed.
            if rank_stats is not None:
                for _k in ("rank_frac_a", "rank_frac_b", "rank_frac_max",
                           "rank_mean_a", "rank_mean_b"):
                    scores_df[f"{pre}_{_k}"] = rank_stats[_k]
                    extra_cols.append(f"{pre}_{_k}")
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
                **{f"shift_{k}": v for k, v in gs.items()},
                **{f"dropout_{k}": v for k, v in dropout.items()},
                **{f"coverage_{k}": v for k, v in summary_cov.items()},
                **{f"covconf_{k}": v for k, v in cov_conf.items()},
                **{f"depthreg_{k}": v for k, v in _rg.items()},
                **{f"refd_{k}": v for k, v in _rd.items()},
                "regressed_coverage": bool(args.contrast_regress_coverage),
                **{f"skew_{k}": v for k, v in skew.items()},
                **{f"consistency_{k}": v for k, v in cons_info.items()},
                "weight": args.contrast_weight, "test": args.contrast_test,
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
            if args.contrast_batch_control:
                for c in bcols:
                    nc = batch_negative_control(
                        Xc, ct.to_numpy(), batch_df[c].to_numpy(),
                        args.min_cells_per_group, args.contrast_test,
                        args.contrast_fdr,
                        Vc if args.contrast_weight != "none" else None,
                        args.contrast_weight,
                        counts=(Cc if args.contrast_weight != "none"
                                else None))
                    if nc is None:
                        print(f"  negative control {c}: fewer than two levels "
                              f"with {args.min_cells_per_group} cells -- not "
                              f"run", file=sys.stderr)
                        continue
                    if nc["n_significant"] is None:
                        print(f"  negative control {c}: no cell type appears "
                              f"in both {nc['levels'][0]} and "
                              f"{nc['levels'][1]}, so this batch is NESTED "
                              f"inside cell type and cannot be measured "
                              f"separately from it -- comparing the batches "
                              f"would compare the cell types.  A bridge "
                              f"sample, one cell type split across both "
                              f"batches, is what makes this answerable.",
                              file=sys.stderr)
                        cons_info[f"negcontrol_{c}"] = nc
                        continue
                    summary_contrast.setdefault(label, {})
                    verdict = ""
                    if nc["n_significant"] >= int(mask.sum()):
                        verdict = ("   <- AS MANY AS THE BIOLOGY: these bins "
                                   "are not separable from processing")
                    elif nc["n_significant"] > 0.25 * max(int(mask.sum()), 1):
                        verdict = "   <- a sizeable share of the biology"
                    via = (f"within {', '.join(nc['strata'])}"
                           if nc.get("strata") else "unstratified")
                    print(f"  negative control {c} "
                          f"({nc['levels'][0]} vs {nc['levels'][1]}, "
                          f"{nc['n_cells'][0]} vs {nc['n_cells'][1]} cells, "
                          f"{via}): {nc['n_significant']:,} bins at "
                          f"FDR<{args.contrast_fdr} against "
                          f"{int(mask.sum()):,} for {label!r}{verdict}",
                          file=sys.stderr)
                    if len(nc.get("strata", [])) == 1:
                        print(f"    that one cell type spans both batches, so "
                              f"it is a BRIDGE: the same biology on either "
                              f"side of the batch, and the cleanest estimate "
                              f"of the batch effect this design offers.",
                              file=sys.stderr)
                    cons_info[f"negcontrol_{c}"] = nc
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
        # Per-contrast up / down / ns, so an enrichment run can take the
        # changed bins as foreground and the UNCHANGED-BUT-MEASURED bins as
        # background, instead of every bin in the genome.
        #
        # 'ns' is the one that has to be defined carefully.  It means TESTED
        # and not significant -- never merely "not selected".  A bin whose
        # statistic is NaN was never testable (too few cells, no coverage),
        # and untestable bins are not a random sample of the genome: they sit
        # in poor mappability, repeats and pericentromeric sequence.  Letting
        # them into the background makes the enrichment partly a measure of
        # testable-versus-untestable rather than changed-versus-unchanged,
        # which is the whole reason not to use the all-bins file.
        #
        # Bins significant at FDR but dropped by a later filter (--contrast-
        # min-diff, --contrast-consistency) land in NEITHER set.  They are
        # changed, so they do not belong in the background, and they were not
        # kept, so they do not belong in the foreground.  The count is
        # reported rather than silently absorbed.
        for lab in summary_contrast:
            pre = f"contrast_{lab}"
            dcol, qcol = f"{pre}_diff", f"{pre}_fdr"
            if not {pre, dcol, qcol} <= set(bed.columns):
                continue
            d = bed[dcol].to_numpy(float)
            qv = bed[qcol].to_numpy(float)
            keep = bed[pre].to_numpy(bool)
            tested = np.isfinite(qv)
            parts = {"up": keep & (d > 0), "down": keep & (d < 0),
                     "ns": tested & (qv >= args.contrast_fdr),
                     # Every bin that HAD a statistic.  A superset of
                     # up|down|ns by the significant-but-filtered bins, and
                     # deliberately so: this is the universe, the honest
                     # answer to "which bins could this analysis have
                     # called?".  Use it to draw a matched random control --
                     # matching against all genomic bins would sample
                     # sequence the contrast could never have reported.
                     #
                     # As a BACKGROUND it is the other reasonable choice:
                     # the script removes the foreground from it, so testing
                     # 'up' against it means up-vs-(down+ns).  That differs
                     # from up-vs-ns only when the annotation marks BOTH
                     # directions -- with 400 up, 5,775 down and 3,000 ns, an
                     # annotation covering 40% of up and 20% of everything
                     # else gives OR 2.67 either way, but one covering 40% of
                     # up AND 40% of down gives 2.67 against ns and 1.34
                     # here.  That divergence is the useful signal: it says
                     # the annotation is not about direction, which is what
                     # an assay artefact looks like.
                     "tested": tested}
            for suf, msk in parts.items():
                bed[msk].drop(columns=flags).to_csv(
                    out_dir / f"bin_scores_{name}.{pre}.{suf}.bed",
                    sep="\t", index=False, na_rep="NA")
            # A MATCHED NULL FOREGROUND, one per relabelling.
            #
            # Genomic enrichment on this foreground has a trap that no
            # per-bin threshold closes: the false positives are not random
            # bins.  They come from cell-type structure, and cell-type-
            # variable sequence is itself clustered in compartments and
            # domains -- so a foreground that is half noise can return a
            # clean-looking enrichment for the same features the real bins
            # sit in.  Tightening the p does not help, because the noise
            # tightens with it.
            #
            # The control is a foreground of the SAME SIZE taken from a run
            # where the labels were relabelled: it carries the cell-type
            # structure and none of the tumour effect.  Run the enrichment on
            # these too and read the real one against them, rather than
            # against a genome background.
            _nl = (locals().get("unitperm_nulls") or {}).get(f"{name}:{lab}")
            _n_fg = int(parts["up"].sum() + parts["down"].sum())
            if _nl and _n_fg > 0:
                _tested_idx = np.flatnonzero(tested)
                for _k, _np_ in enumerate(_nl):
                    _v = np.asarray(_np_, float)
                    _v = _v[_tested_idx] if _v.size == tested.size else _v
                    _ok = np.isfinite(_v)
                    if _ok.sum() < _n_fg:
                        continue
                    _sel = _tested_idx[_ok][np.argsort(_v[_ok],
                                                       kind="stable")[:_n_fg]]
                    _m = np.zeros(len(bed), bool)
                    _m[_sel] = True
                    bed[_m].drop(columns=flags).to_csv(
                        out_dir / f"bin_scores_{name}.{pre}.nullfg{_k}.bed",
                        sep="\t", index=False, na_rep="NA")
                print(f"  {name}: {len(_nl)} matched null foregrounds of "
                      f"{_n_fg:,} bins each -> "
                      f"bin_scores_{name}.{pre}.nullfg*.bed.  Run the "
                      f"genomic enrichment on these too: the false positives "
                      f"here are cell-type structure, which is itself "
                      f"clustered in the genome, so a noisy foreground can "
                      f"return a clean-looking enrichment for the same "
                      f"features.  Read the real foreground against these, "
                      f"not against a genome background.", file=sys.stderr)
            gap = int((tested & (qv < args.contrast_fdr) & ~keep).sum())
            summary_contrast[lab].update({
                "n_bed_up": int(parts["up"].sum()),
                "n_bed_down": int(parts["down"].sum()),
                "n_bed_ns": int(parts["ns"].sum()),
                "n_bed_tested": int(parts["tested"].sum()),
                "n_bed_untested": int((~tested).sum()),
                "n_bed_ambiguous": gap})
            print(f"  contrast {lab}: {int(parts['up'].sum()):,} up / "
                  f"{int(parts['down'].sum()):,} down / "
                  f"{int(parts['ns'].sum()):,} ns / "
                  f"{int(parts['tested'].sum()):,} tested "
                  f"-> bin_scores_{name}.{pre}.{{up,down,ns,tested}}.bed  "
                  f"({int((~tested).sum()):,} bins never testable and in "
                  f"none of them"
                  + (f"; {gap:,} significant but filtered, also in none)"
                     if gap else ")"), file=sys.stderr)
            if int(parts["ns"].sum()) < 50:
                print(f"  contrast {lab}: WARNING only "
                      f"{int(parts['ns'].sum()):,} bins in the ns "
                      f"background.  An enrichment against that has almost "
                      f"no resolution; widen the testable set or fall back "
                      f"to bin_scores_{name}.all.bed and accept that it "
                      f"mixes in untestable sequence.", file=sys.stderr)
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
        if summary_contrast:
            print(f"  for enrichment: --bed {name}=bin_scores_{name}."
                  f"contrast_<label>.up.bed --background-bed "
                  f"{name}=bin_scores_{name}.contrast_<label>.ns.bed",
                  file=sys.stderr)
            print(f"    ns is the reference for a claim about CHANGE, since "
                  f"it is the only set that means measured-and-unchanged.  "
                  f"Re-running against .tested.bed (= down+ns once the "
                  f"foreground is removed) answers a different question and "
                  f"gives the same answer unless the annotation marks BOTH "
                  f"directions -- if the two disagree, that is the finding: "
                  f"test up against down directly to see what is "
                  f"directional.", file=sys.stderr)
        print(f"  per-bin stats -> bin_scores_{name}.{{all,selected}}.bed "
              f"(chrom/start/end, cell-type ICC, ANOVA p, BH-FDR, batch ICC)")
        if np.isfinite(sil):
            verdict = "SEPARATES" if perm_p < 0.05 else "does NOT separate"
            print(f"  out-of-selection control: NON-selected bins {verdict} "
                  f"cell types (silhouette={sil:.3f}, perm p={perm_p:.3f}) — "
                  f"see the nome_hic_concordance/compare UMAP for the figure")

    # Back to the un-shadowed settings: everything after the loop is
    # cross-matrix, and leaving the LAST matrix's per-matrix value bound
    # would apply it to work that belongs to neither.
    args = base_args
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
                         label, ct_colors=ct_colors,
                         stat_labels={k: joint_stat.get(k) for k in (a, b)})
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
