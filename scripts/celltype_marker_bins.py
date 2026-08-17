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
                  noisiest bins rather than the flattest.

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
                  same statement.

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


def select_stable_bins(X, cell_type_ids, n_ct, b_icc, args, n_want):
    """The invariant counterpart of :func:`select_marker_bins`.

    The batch filter applies exactly as it does to the marker set: a bin that
    is flat across cell types but driven by batch is not invariant, it is
    sample.  Same clean-first, fill-by-rank policy, so the two sets are
    selected by the same rules and can be compared.
    """
    spread = celltype_spread(X, cell_type_ids, n_ct, args)
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


def contrast_stats(X, in_mask, min_cells):
    """Welch two-sample t per bin for one group set against all the others.

    Welch rather than pooled: the two arms are a handful of samples of one
    condition against everything else, so equal variances are not a safe
    assumption.  Returns NaN for bins either arm cannot cover, which keeps them
    out of the multiple-testing correction rather than in it at p = 1.
    """
    from scipy.stats import t as _t

    X = np.asarray(X, float)
    out = {}
    for tag, m in (("in", in_mask), ("out", ~in_mask)):
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
    bad = (n1 < min_cells) | (n2 < min_cells) | ~np.isfinite(se) | (se <= 0)
    for k, v in (("diff", diff), ("t", tstat), ("p", p), ("d", d)):
        out[k] = np.where(bad, np.nan, v)
    return out


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
    ap.add_argument("--n-contrast-bins", type=int, default=0,
                    help="Bins per contrast BED (0 = same as --n-bins).")
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
            sys.exit(f"--contrast must be LABEL=GROUP[,GROUP...], got {spec!r}")
        lab, groups = spec.split("=", 1)
        lab = lab.strip().replace("/", "_").replace(" ", "_")
        want = [g.strip() for g in groups.split(",") if g.strip()]
        if not lab or not want:
            sys.exit(f"--contrast needs a label and at least one group: {spec!r}")
        contrasts.append((lab, want))

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
        if args.remove_batch == "within-celltype":
            M = _remove_batch(M, ct, batch_df)

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
        scores_df = pd.DataFrame({
            "bin": M.columns, "celltype_icc": ct_icc,
            "celltype_pvalue": ct_p, "celltype_fdr": ct_fdr,
            "batch_icc": b_icc, "selected": M.columns.isin(sel_bins)})
        extra_cols: list[str] = []
        subsets: dict[str, np.ndarray] = {}

        # (2) INVARIANT set: flattest across cell types, same batch filter.
        if args.stable_bins == "on":
            n_want = args.n_stable_bins or args.n_bins
            stable_sel, spread, n_stable_clean = select_stable_bins(
                Msel, ct_id, len(ct_u), b_icc, args, n_want)
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
        for label, want in contrasts:
            present = [g for g in want if g in set(ct_u)]
            missing = [g for g in want if g not in set(ct_u)]
            if missing:
                print(f"  contrast {label!r}: no cells for {missing} "
                      f"(available: {ct_u})", file=sys.stderr)
            if not present or len(present) == len(ct_u):
                print(f"  contrast {label!r}: needs at least one group in the "
                      f"set and one outside — skipped", file=sys.stderr)
                continue
            in_mask = ct.isin(present).to_numpy()
            cs = contrast_stats(Msel, in_mask, args.min_cells_per_group)
            q = _bh_fdr(cs["p"])
            rank = np.where(np.isfinite(cs["t"]), np.abs(cs["t"]), -np.inf)
            keep_rank = np.argsort(rank)[::-1][: args.n_contrast_bins
                                               or args.n_bins]
            keep_rank = keep_rank[np.isfinite(rank[keep_rank])]
            # A contrast bin driven by batch is worthless in exactly the same
            # way a marker bin is, so it faces the same filter.
            clean = ~(np.isfinite(b_icc) & (b_icc > args.max_batch_icc))
            mask = np.zeros(M.shape[1], bool)
            mask[keep_rank] = True
            pre = f"contrast_{label}"
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
            summary_contrast[label] = {
                "groups_in": present, "n_cells_in": int(in_mask.sum()),
                "n_cells_out": int((~in_mask).sum()),
                "n_selected": int(mask.sum()),
                "n_fdr_0.05": int(np.nansum(q < 0.05)),
                "n_selected_batch_clean": int(np.sum(mask & clean)),
                "n_selected_batch_judged": judged,
                "batch_confounding_cramers_v": conf}
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
        for c in ("celltype_icc", "celltype_pvalue", "celltype_fdr",
                  "batch_icc", "selected", *extra_cols):
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
            print(f"  contrast {lab} ({'+'.join(s['groups_in'])}, "
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
