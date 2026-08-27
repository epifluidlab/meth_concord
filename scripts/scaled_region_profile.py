#!/usr/bin/env python3
"""Mean GCH-GCH concordance across scaled domains, tumour against reference.

WHAT.  The deepTools ``computeMatrix scale-regions`` shape: each domain is
compressed to the same number of body bins regardless of its true width, with
unscaled flanks on either side, so domains of very different length can be
averaged onto one axis.  Then one line per GROUP with a 95% interval.

WHY SCALE AT ALL.  A boundary-anchored profile (scripts/boundary_profile.py)
answers "what happens near an edge".  It cannot answer "what happens across a
domain", because PMDs run from tens of kb to tens of Mb: aligned on their
left edge, a fixed-width axis is inside some domains and outside others at
the same x, and the average is over a mixture of two states.  Scaling puts
every domain's interior onto the same coordinate.

WHAT THE SCALING COSTS.  x is no longer distance.  A step in the body is a
different number of base pairs in every domain, so body features that are
genuinely fixed-width -- anything anchored to an edge -- get smeared across
different fractions of the axis and flattened.  The flanks stay in bp for
exactly that reason, and the two halves of the plot must be read
differently: bp on the outside, fraction-of-domain inside.

THE INTERVAL IS ACROSS CELLS.  Bins inside one domain are heavily
correlated, so treating (cell, bin) pairs as independent would give a band
far too tight to mean anything.  Cells are the closest thing to a replicate.
That is still generous: cells from one tumour are not independent replicates
of "tumour" either, and with two donors the honest n is 2, not the cell
count.  The per-donor curves are drawn behind the group line for that
reason -- if tumour1 and tumour2 do not agree, the band is telling you
about cells, not about tumours.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from nome_hic_concordance.concordance import decompose_components as _dc

_HERE = Path(__file__).resolve().parent


def _bp_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bprof", _HERE / "boundary_profile.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["bprof"] = m
    spec.loader.exec_module(m)
    return m


def scaled_index(bins, domains, n_body, flank_bp, n_flank):
    """Position on the scaled axis for every (bin, domain) overlap.

    Returns ``(bin_idx, pos_idx)``: a bin may appear more than once, because
    a bin can sit in the flank of one domain and the body of the next.  Both
    placements are real and both are kept -- dropping either would silently
    thin the axis wherever domains are close together.

    Layout: ``[0, n_flank)`` upstream, ``[n_flank, n_flank + n_body)`` body,
    then ``n_flank`` downstream.
    """
    total = 2 * n_flank + n_body
    if not len(domains):
        return (np.array([], int), np.array([], int), total,
                np.array([], int))
    bi, pi, di = [], [], []
    step_f = flank_bp / float(n_flank) if n_flank else 1.0
    by_chrom = {}
    for c, sub in bins.groupby("chrom"):
        by_chrom[str(c)] = (sub.index.to_numpy(),
                            sub["start"].to_numpy(float),
                            sub["end"].to_numpy(float))
    for _dom_no, (_, d) in enumerate(domains.iterrows()):
        got = by_chrom.get(str(d["chrom"]))
        if got is None:
            continue
        idx, bs, be = got
        lo, hi = float(d["start"]), float(d["end"])
        if hi <= lo:
            continue
        mid = (bs + be) / 2.0
        sel = (mid >= lo - flank_bp) & (mid < hi + flank_bp)
        if not sel.any():
            continue
        m = mid[sel]
        p = np.empty(m.size, float)
        up = m < lo
        dn = m >= hi
        body = ~(up | dn)
        p[up] = (m[up] - (lo - flank_bp)) / step_f
        p[body] = n_flank + (m[body] - lo) / (hi - lo) * n_body
        p[dn] = n_flank + n_body + (m[dn] - hi) / step_f
        q = np.clip(p.astype(int), 0, total - 1)
        bi.append(idx[sel])
        pi.append(q)
        # WHICH domain each placement came from.  The scaled axis flattens
        # thousands of domains onto one set of positions, which is what the
        # panel wants and what a per-domain test cannot have: without this
        # the only units left are cells, and cells from one donor are not
        # replicates of anything.
        di.append(np.full(q.size, _dom_no, int))
    if not bi:
        return (np.array([], int), np.array([], int), total,
                np.array([], int))
    return (np.concatenate(bi), np.concatenate(pi), total,
            np.concatenate(di))


# What each component IS, for the axis.  A panel labelled D while it draws
# C is not a cosmetic problem: D = C(2F-1) is an inverted, rescaled copy of
# C at the marginals this data has, so the label names a quantity whose sign
# convention is the opposite of the one plotted.
STAT_LABEL = {
    "coordination": "coordination C (marginal-free)",
    "direction": "direction F (co-open share)",
    "direction_expected": "expected direction F_exp",
    "direction_excess": "direction excess F \u2212 F_exp",
    "direction_logor": "direction log-OR  logit(F) \u2212 logit(F_exp)",
    "directional": "directional D = C\u00b7(2F\u22121) (legacy)",
    "directional_corrected": "directional D' = C\u00b7(F\u2212F_exp)",
    "marginal": "marginal p\u0304 (mean openness)",
}


def statistic_of(matrix_path):
    """The component a concordance_<name>.parquet holds, or None."""
    stem = Path(matrix_path).name
    if not (stem.startswith("concordance_") and stem.endswith(".parquet")):
        return None
    got = stem[len("concordance_"):-len(".parquet")]
    return got if got in STAT_LABEL else None


COUNT_KEYS = ("k_open", "k_closed", "k_mixed", "n_linked",
              "k_dm_open", "k_dm_closed", "k_dm_mixed", "n_dm")

# Taken from decompose_components rather than retyped, so the two cannot
# drift: every component is a function of the same four pooled rates, and
# none of them is privileged by the pooling.
POOLABLE = frozenset(_dc(*(np.zeros(1),) * 4))


def _counts_dir(d):
    d = Path(d)
    return d / "counts" if (d / "counts").is_dir() else d


def load_counts(counts_dir, index, columns):
    """The raw per-(cell, bin) soft counts, aligned to the matrix."""
    d = _counts_dir(counts_dir)
    out = {}
    for k in COUNT_KEYS:
        f = d / f"{k}.parquet"
        if not f.exists():
            sys.exit(f"--from-counts: {f} not found.  Re-run step 5 with "
                     f"--emit-counts.")
        out[k] = pd.read_parquet(f).reindex(
            index=index, columns=columns).to_numpy(float)
    return out


def load_prior(counts_dir):
    """The prior the run fitted, so a pooled estimate is shrunk like the
    per-bin ones were rather than by something invented here."""
    d = Path(counts_dir)
    if d.name == "counts":
        d = d.parent
    f = d / "dirichlet_multinomial_prior.json"
    if not f.exists():
        return {"open": 1.0, "closed": 1.0, "mixed": 1.0}
    import json
    pri = json.load(open(f))
    return pri.get("shared", pri.get("linked", {}))["alpha"]


def which_statistic_matches(M, counts, alpha, cap=200_000, seed=0):
    """Which component the --matrix values actually look like.

    Under --from-counts the statistic is chosen by the --matrix FILENAME
    and the file's values are never read, so a rename, a symlink or a stale
    path picks a different quantity in silence.  Recomputing every
    component per (cell, bin) from the raw counts and ranking them by
    agreement with the matrix says whether the name is telling the truth.

    Correlation, not equality: the matrix is post-RWR and the counts are
    raw, so even the right component will not match exactly.  What
    identifies it is being the BEST match, by a clear margin.
    """
    from scipy.stats import spearmanr
    a0 = float(sum(alpha.values()))
    ao, ac = float(alpha["open"]), float(alpha["closed"])
    n_l, n_e = counts["n_linked"], counts["n_dm"]
    with np.errstate(invalid="ignore", divide="ignore"):
        comp = _dc((ao + counts["k_open"]) / (a0 + n_l),
                   (ac + counts["k_closed"]) / (a0 + n_l),
                   (ao + counts["k_dm_open"]) / (a0 + n_e),
                   (ac + counts["k_dm_closed"]) / (a0 + n_e))
    target = np.asarray(M, float).ravel()
    ok = np.isfinite(target)
    if ok.sum() > cap:
        idx = np.random.default_rng(seed).choice(np.flatnonzero(ok), cap,
                                                 replace=False)
    else:
        idx = np.flatnonzero(ok)
    out = {}
    for k, v in comp.items():
        w = np.asarray(v, float).ravel()[idx]
        t = target[idx]
        m = np.isfinite(w) & np.isfinite(t)
        if m.sum() > 10 and np.ptp(w[m]) > 0 and np.ptp(t[m]) > 0:
            out[k] = float(abs(spearmanr(w[m], t[m])[0]))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def pooled_profiles(counts, rows, bin_idx, pos_idx, total, alpha, stat):
    """Sum the counts at each position, THEN compute the statistic once.

    Averaging a per-bin statistic over a region is not the region's
    statistic.  C and the log odds ratio are ratios of aggregated counts,
    so they are not additive across bins, and a thin bin's value is mostly
    the prior.  Measured on a 20-bin region with 16 thin bins: the
    unweighted mean of the per-bin log odds ratio came to +0.043 against a
    true +0.350, while pooling the counts first gave +0.222.

    The attenuation follows the SHARE of thin bins, which is what makes it
    a bias rather than a scale factor: a domain and its flank differ in
    accessibility, hence in depth, so they are attenuated by different
    amounts and the contrast between them is distorted.  Domains also vary
    in size, so a small one is a single bin stretched across the axis while
    a large one is forty averaged -- pooling makes both one estimate from
    whatever pairs they contain.
    """
    import numpy as _np
    from nome_hic_concordance.concordance import decompose_components
    a0 = float(sum(alpha.values()))
    ao, ac = float(alpha["open"]), float(alpha["closed"])
    P = _np.full((len(rows), total), _np.nan)
    n_bins = _np.zeros(total, int)
    order = _np.argsort(pos_idx, kind="stable")
    bi, pi = bin_idx[order], pos_idx[order]
    starts = _np.searchsorted(pi, _np.arange(total), side="left")
    ends = _np.searchsorted(pi, _np.arange(total), side="right")
    for i in range(total):
        cols = bi[starts[i]:ends[i]]
        n_bins[i] = cols.size
        if cols.size == 0:
            continue
        g = lambda k: _np.nansum(counts[k][:, cols], axis=1)
        n_l, n_e = g("n_linked"), g("n_dm")
        o_o = (ao + g("k_open")) / (a0 + n_l)
        o_c = (ac + g("k_closed")) / (a0 + n_l)
        e_o = (ao + g("k_dm_open")) / (a0 + n_e)
        e_c = (ac + g("k_dm_closed")) / (a0 + n_e)
        v = decompose_components(o_o, o_c, e_o, e_c)[stat]
        P[:, i] = _np.where(n_l > 0, v, _np.nan)
    return P, rows, n_bins


def cell_profiles(M, ct, members, bin_idx, pos_idx, total):
    """One curve per CELL: mean D at each axis position.

    A domain contributing many bins to a position does not thereby count as
    many observations -- they are averaged into that cell's single value, so
    the observation stays the cell.
    """
    cells = ct.index[ct.isin(members)]
    rows = M.loc[M.index.intersection(cells)]
    P = np.full((len(rows), total), np.nan)
    n_bins = np.zeros(total, int)
    if not len(rows):
        return P, rows.index, n_bins
    v = rows.to_numpy(float)
    order = np.argsort(pos_idx, kind="stable")
    bi, pi = bin_idx[order], pos_idx[order]
    starts = np.searchsorted(pi, np.arange(total), side="left")
    ends = np.searchsorted(pi, np.arange(total), side="right")
    for i in range(total):
        cols = bi[starts[i]:ends[i]]
        n_bins[i] = cols.size
        if cols.size == 0:
            continue
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            P[:, i] = np.nanmean(v[:, cols], axis=1)
    return P, rows.index, n_bins


def anchor_to_flank(P, n_flank, n_body):
    """Subtract each cell's OWN mean over the flanks.

    Makes the outside baseline identical between groups by construction, so
    the panel shows the change across the domain rather than the level.
    That is a real loss as well as a gain: any genuine genome-wide
    difference between the arms is gone from the figure, and 'tumour is
    higher overall' can no longer be read off it -- only 'tumour changes
    more from its own flank'.

    Per CELL, not per group.  Subtracting a group mean would leave every
    cell's deviation from that mean intact and shrink nothing, so the
    interval would still be dominated by the between-cell level differences
    the anchoring is meant to remove.
    """
    if not P.size:
        return P
    total = P.shape[1]
    flank = np.concatenate([np.arange(n_flank),
                            np.arange(n_flank + n_body, total)])
    if flank.size == 0:
        return P
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = np.nanmean(P[:, flank], axis=1)
    return P - base[:, None]


def group_profile(M, ct, members, bin_idx, pos_idx, total, conf=0.95,
                  baseline="none", n_flank=0, n_body=0, pooled=None):
    """Mean and interval per axis position for one group of cell types.

    The interval is across CELLS.  Bins inside one domain are heavily
    correlated, so treating (cell, bin) pairs as independent would give a
    band far too tight to mean anything.
    """
    from scipy.stats import t as _t

    if pooled is not None:
        cells = ct.index[ct.isin(members)]
        keep = M.index.intersection(cells)
        rows = M.index.get_indexer(keep)
        sub = {k: v[rows] for k, v in pooled["counts"].items()}
        P, idx, n_bins = pooled_profiles(sub, keep, bin_idx, pos_idx, total,
                                         pooled["alpha"], pooled["stat"])
    else:
        P, idx, n_bins = cell_profiles(M, ct, members, bin_idx, pos_idx,
                                       total)
    if baseline == "flank":
        P = anchor_to_flank(P, n_flank, n_body)
    # One number per CELL: its mean over the body.  With --baseline flank
    # that is the domain minus this cell's own flank, which is the contrast
    # the panel is read for; without it, an absolute level dominated by the
    # genome-wide offset.  Kept so a test can use exactly what is drawn
    # rather than recomputing it a second, subtly different way.
    # A cell with no finite value anywhere in the body is a legitimate
    # outcome, not a problem: it means this domain gave that cell nothing.
    # nanmean says so with a RuntimeWarning per cell per domain, which over
    # thousands of domains buries the run's actual output.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        body = (np.nanmean(P[:, n_flank:n_flank + n_body], axis=1)
                if P.shape[0] and n_body else np.array([]))
    out = {"mean": np.full(total, np.nan), "lo": np.full(total, np.nan),
           "hi": np.full(total, np.nan), "n_bins": n_bins,
           "n_cells": int(P.shape[0]), "body": body, "cells": idx,
           "members": {g: int((ct.loc[idx] == g).sum()) for g in members}}
    if not P.shape[0]:
        return out
    for i in range(total):
        per_cell = P[:, i]
        per_cell = per_cell[np.isfinite(per_cell)]
        if per_cell.size == 0:
            continue
        out["mean"][i] = per_cell.mean()
        if per_cell.size > 1:
            se = per_cell.std(ddof=1) / np.sqrt(per_cell.size)
            half = _t.ppf(0.5 + conf / 2.0, per_cell.size - 1) * se
            out["lo"][i] = out["mean"][i] - half
            out["hi"][i] = out["mean"][i] + half
    return out


def per_domain_values(M, ct, members, bin_idx, pos_idx, dom_idx, n_flank,
                      n_body, total, baseline="none", pooled=None):
    """One value per DOMAIN: the group's mean over that domain's body,
    minus its own flank when anchored.

    The scaled panel flattens thousands of domains onto one axis, which is
    what makes it readable and what leaves only cells to test on.  Cells
    from one donor are not replicates, so that test answers "do these cells
    differ" rather than "do these conditions differ".

    A domain is a different unit and a better-powered one: with thousands
    of PMDs, asking whether the tumour-minus-reference difference is
    consistently non-zero ACROSS domains has thousands of degrees of
    freedom.  It is not free of dependence either -- domains cluster and a
    genome-wide offset would move every one of them the same way -- which
    is exactly why the flank anchor matters here: each domain is measured
    against its OWN flank, so a global shift cancels inside every domain
    and what is left is local.
    """
    n_dom = int(dom_idx.max()) + 1 if dom_idx.size else 0
    out = np.full(n_dom, np.nan)
    if not n_dom:
        return out
    body_lo, body_hi = n_flank, n_flank + n_body
    cells = ct.index[ct.isin(members)]
    keep = M.index.intersection(cells)
    rows = M.index.get_indexer(keep) if pooled is not None else None
    sub = ({k: v[rows] for k, v in pooled["counts"].items()}
           if pooled is not None else None)
    # TWO positions, not the full scaled axis: this domain's flank and its
    # body.  The scaled axis exists to draw a shape, and a shape is not
    # what a per-domain value is -- collapsing it also avoids recomputing
    # the confidence interval at every one of `total` positions for every
    # one of thousands of domains.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for d in range(n_dom):
            sel = dom_idx == d
            if not sel.any():
                continue
            bi_d, pi_d = bin_idx[sel], pos_idx[sel]
            two = np.where((pi_d >= body_lo) & (pi_d < body_hi), 1,
                           0).astype(int)
            if pooled is not None:
                P, _r, _n = pooled_profiles(sub, keep, bi_d, two, 2,
                                            pooled["alpha"], pooled["stat"])
            else:
                P, _r, _n = cell_profiles(M, ct, members, bi_d, two, 2)
            if not P.shape[0]:
                continue
            # Anchored on THIS domain's own flank, which is what makes the
            # per-domain difference local: a genome-wide offset sits in
            # both terms and cancels inside every domain.
            v = P[:, 1] - P[:, 0] if baseline == "flank" else P[:, 1]
            if np.isfinite(v).any():
                out[d] = float(np.nanmean(v))
    return out


def domain_paired_test(a_vals, b_vals):
    """Is the difference between the arms consistent across domains?

    Paired on the domain -- both arms are measured on the same one -- so a
    Wilcoxon signed-rank on the per-domain differences.  Signed-rank rather
    than a t because these differences are not normal, and paired rather
    than two-sample because a domain's depth, size and compartment affect
    both arms and cancel in the difference.
    """
    from scipy.stats import wilcoxon
    d = np.asarray(a_vals, float) - np.asarray(b_vals, float)
    d = d[np.isfinite(d)]
    out = {"n_domains": int(d.size),
           "median_diff": float(np.median(d)) if d.size else float("nan"),
           # The MEAN beside the median, because their gap is the whole
           # story when the two units disagree: a per-cell test averages
           # over domains and so follows the mean, while the signed-rank
           # follows the middle.  A mean several times the median means a
           # minority of domains is carrying the cell-level result, and
           # "PMDs differ" is then the wrong sentence for it.
           "mean_diff": float(np.mean(d)) if d.size else float("nan"),
           "q25_diff": float(np.percentile(d, 25)) if d.size else float("nan"),
           "q75_diff": float(np.percentile(d, 75)) if d.size else float("nan"),
           "frac_positive": float(np.mean(d > 0)) if d.size else float("nan"),
           "p_domain": float("nan")}
    if d.size >= 6 and np.any(d != 0):
        out["p_domain"] = float(wilcoxon(d, alternative="two-sided")[1])
    return out


def region_group_test(prof, ct, donor=None, baseline="none",
                      domain_vals=None):
    """Are the two arms different across this region set?

    One value per CELL -- its mean over the domain body, which under
    --baseline flank is that cell's domain-minus-flank -- then a two-sided
    Mann-Whitney between the arms.  Rank-based because these values are not
    normal and one outlying cell should not carry the result.

    THE CELL-LEVEL p IS NOT A BIOLOGICAL REPLICATE TEST, and on this design
    it cannot be made into one.  Cells from one donor share that donor's
    biology, its library prep and its coverage, so with a few hundred cells
    per arm the test has enough power to call any systematic difference --
    including the donor itself.  It answers "do these cells differ", which
    is nearly always yes, not "do these conditions differ".

    Passing a donor column gives the test that does answer the second
    question, by averaging each donor's cells first.  It is reported
    alongside and it will usually be far less significant; that is the
    honest arithmetic of two donors against three references, not a defect
    in the test.  Where the donor-level n is too small to reach any p at
    all, the smallest attainable one is reported so the reader can see the
    ceiling rather than mistake it for a null result.
    """
    from itertools import combinations
    from math import comb
    from scipy.stats import mannwhitneyu

    rows = []
    labs = [g for g in prof if np.size(prof[g].get("body", ())) > 0]
    for a, b in combinations(labs, 2):
        rec = {"group_a": a, "group_b": b, "baseline": baseline}
        for unit in ("cell", "donor"):
            if unit == "donor" and donor is None:
                continue
            xs = []
            for g in (a, b):
                v = np.asarray(prof[g]["body"], float)
                idx = prof[g]["cells"]
                if unit == "donor":
                    d = pd.Series(v, index=idx).groupby(
                        donor.reindex(idx).astype(str)).mean()
                    v = d.to_numpy(float)
                xs.append(v[np.isfinite(v)])
            na, nb = len(xs[0]), len(xs[1])
            rec[f"n_{unit}_a"], rec[f"n_{unit}_b"] = na, nb
            rec[f"mean_{unit}_a"] = float(np.mean(xs[0])) if na else np.nan
            rec[f"mean_{unit}_b"] = float(np.mean(xs[1])) if nb else np.nan
            rec[f"diff_{unit}"] = rec[f"mean_{unit}_a"] - rec[f"mean_{unit}_b"]
            if na >= 1 and nb >= 1 and (na + nb) >= 3:
                rec[f"p_{unit}"] = float(
                    mannwhitneyu(xs[0], xs[1], alternative="two-sided")[1])
            else:
                rec[f"p_{unit}"] = float("nan")
            # The smallest p this many units could ever produce, so a
            # non-significant donor test is read as a ceiling and not as
            # evidence of no difference.
            rec[f"p_floor_{unit}"] = (2.0 / comb(na + nb, na)
                                      if na and nb else float("nan"))
        if domain_vals is not None and a in domain_vals and b in domain_vals:
            rec.update(domain_paired_test(domain_vals[a], domain_vals[b]))
        rows.append(rec)
    return pd.DataFrame(rows)


def member_curves(M, ct, members, bin_idx, pos_idx, total, baseline="none",
                  n_flank=0, n_body=0, pooled=None):
    """Mean per axis position for each constituent cell type, no interval.

    Drawn behind the group line so a group mean carried by ONE of its
    members is visible.  With two tumours that is the difference between a
    tumour finding and a tumour-1 finding, and no interval across cells can
    show it.  Anchored the same way as the group, or the faint lines would
    sit on a different baseline from the line they are meant to explain.
    """
    return {g: group_profile(M, ct, [g], bin_idx, pos_idx, total,
                             baseline=baseline, n_flank=n_flank,
                             n_body=n_body, pooled=pooled)["mean"]
            for g in members}


def scaled_figure(results, path, title, ylabel, n_body, n_flank, flank_bp,
                  colors=None, conf=0.95, members=None):
    """One panel per domain set; one line and band per group."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(len(names), 1, figsize=(7.0, 2.9 * len(names)),
                             squeeze=False, sharex=True)
    total = 2 * n_flank + n_body
    x = np.arange(total)
    default = ["#c0392b", "#2c3e50", "#16a085", "#8e44ad", "#d35400"]
    for ax, nm in zip(axes[:, 0], names):
        prof = results[nm]
        for k, (g, r) in enumerate(prof.items()):
            col = (colors or {}).get(g, default[k % len(default)])
            ok = np.isfinite(r["mean"])
            if (members or {}).get(nm, {}).get(g):
                for sub, yv in members[nm][g].items():
                    s = np.isfinite(yv)
                    ax.plot(x[s], yv[s], color=col, lw=0.7, alpha=0.35,
                            zorder=1)
            band = np.isfinite(r["lo"]) & np.isfinite(r["hi"])
            if band.any():
                ax.fill_between(x[band], r["lo"][band], r["hi"][band],
                                color=col, alpha=0.18, lw=0, zorder=2)
            ax.plot(x[ok], r["mean"][ok], color=col, lw=1.8, zorder=3,
                    label=f"{g} (n={r['n_cells']})")
        for b in (n_flank, n_flank + n_body):
            ax.axvline(b, color="0.35", lw=0.8, ls="--", zorder=0)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(nm, fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False)
        ax.tick_params(labelsize=7)
    axes[-1, 0].set_xticks([0, n_flank, n_flank + n_body, total - 1])
    fl = f"{flank_bp / 1e6:g} Mb"
    axes[-1, 0].set_xticklabels([f"-{fl}", "start", "end", f"+{fl}"],
                                fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _named(specs, flag):
    out = {}
    for spec in (specs or []):
        if "=" not in spec:
            sys.exit(f"{flag} must be NAME=VALUE, got {spec!r}")
        k, v = spec.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default=None, metavar="NAME=PATH",
                    help="Concordance matrix (cells x bins) parquet.  "
                         "Optional under --from-counts, which reads none of "
                         "its values -- give --from-counts-statistic "
                         "instead and the grid comes from the count "
                         "matrices themselves, which carry the same cells "
                         "and bins.  Naming a matrix there is still useful: "
                         "the run cross-checks the statistic against its "
                         "values.")
    ap.add_argument("--name", default=None,
                    help="Label for the outputs.  Default: the --matrix "
                         "NAME, or the statistic when there is no "
                         "--matrix.")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cell-type-column", default="cell_type")
    ap.add_argument("--color-column", default=None)
    ap.add_argument("--group", action="append", required=True,
                    metavar="LABEL=CT[,CT...]",
                    help="A curve, e.g. --group tumor=tumor1,tumor2 "
                         "--group other=GM12878,IMR90,monocyte.  Cells of "
                         "every named type are pooled; each constituent type "
                         "is also drawn as a faint line, because a group "
                         "mean carried by one member is a different finding "
                         "from one its members agree on.")
    ap.add_argument("--region", action="append", required=True,
                    metavar="LABEL=BED[::CLASS]",
                    help="Domain set to scale, repeatable: --region "
                         "PMD=pmd.bed::PMD --region late_RT=late.bed "
                         "--region cLAD=clad.bed")
    ap.add_argument("--batch-column", default=None)
    ap.add_argument("--remove-batch", choices=["none", "within-celltype"],
                    default="none",
                    help="Centre each bin within cell type across batches "
                         "before profiling.")
    ap.add_argument("--exclude-chrom", default=None, metavar="CHROM[,...]",
                    help="Contigs to drop before anything else.  Must match "
                         "the detection run: --contrast-center subtracts each "
                         "cell's median over the bins PRESENT, so a different "
                         "contig set gives a different centring and a "
                         "different picture.")
    ap.add_argument("--contrast-center", choices=["none", "cell-median"],
                    default="none",
                    help="Same flag, same meaning, as in "
                         "celltype_marker_bins.py.  Without it the panel "
                         "shows the genome-wide offset between the arms and "
                         "the domain structure rides on top of it, invisible.")
    ap.add_argument("--domain-test", action="store_true",
                    help="Also test across DOMAINS: each domain's "
                         "tumour-minus-reference difference, then a "
                         "Wilcoxon signed-rank on those.  With thousands of "
                         "PMDs that has thousands of degrees of freedom "
                         "where the per-cell test has none that are "
                         "biological -- cells from one donor are not "
                         "replicates.  Paired on the domain, since its "
                         "depth, size and compartment affect both arms and "
                         "cancel in the difference.  Domains are not "
                         "independent either, and that is why --baseline "
                         "flank matters here: each domain is measured "
                         "against its OWN flank, so a genome-wide offset "
                         "cancels inside every one and what is left is "
                         "local.  Costs one profile pass per domain, so it "
                         "is off by default.")
    ap.add_argument("--donor-column", default=None, metavar="COLUMN",
                    help="Column in --samples naming the biological "
                         "replicate.  The per-cell test is reported either "
                         "way, and it is not a replicate test: cells from "
                         "one donor share that donor's biology, prep and "
                         "coverage, so a few hundred per arm will call any "
                         "systematic difference including the donor itself. "
                         "Giving this averages each donor's cells first and "
                         "reports the test that answers whether the "
                         "CONDITIONS differ.  Expect it to be far less "
                         "significant, and read p_floor_donor beside it -- "
                         "with two donors against three references the "
                         "smallest p attainable is 0.2, so a null result "
                         "there is a ceiling, not evidence.")
    ap.add_argument("--from-counts", default=None, metavar="DIR",
                    help="Compute each axis position from the POOLED raw "
                         "counts of the bins landing there, instead of "
                         "averaging their per-bin statistics.  Point at the "
                         "05_concordance directory (or its counts/ "
                         "subdirectory) of a run made with --emit-counts.  "
                         "A region's statistic is not the mean of its bins': "
                         "C and the log odds ratio are ratios of aggregated "
                         "counts, and a thin bin's value is mostly the "
                         "prior.  Measured on a 20-bin region with 16 thin "
                         "bins, the unweighted mean of the per-bin log odds "
                         "ratio came to +0.043 against a true +0.350, while "
                         "pooling gave +0.222 -- and the attenuation follows "
                         "the SHARE of thin bins, so a domain and its flank "
                         "are attenuated differently and the contrast "
                         "between them is distorted.  Domains also vary in "
                         "size, so a small one is one bin stretched across "
                         "the axis while a large one is forty averaged; "
                         "pooling makes both one estimate from whatever "
                         "pairs they contain.  The per-cell corrections "
                         "(--contrast-center, --contrast-reference-*, "
                         "--contrast-regress-coverage) act on the statistic "
                         "and cannot be applied to counts, so they are "
                         "refused here; --baseline flank still applies, and "
                         "is the per-cell control that matters for a "
                         "profile.")
    ap.add_argument("--from-counts-statistic", default=None,
                    choices=sorted(POOLABLE),
                    help="Which statistic to build from the pooled counts.  "
                         "Default: read from the --matrix filename.  Any "
                         "component decompose_components emits works -- they "
                         "are all functions of the same four pooled rates, "
                         "so none is privileged.")
    ap.add_argument("--contrast-reference-scale", action="append",
                    default=[], metavar="NAME=PATH[:COLUMN]",
                    help="Divide out each cell's assay reliability on the "
                         "scale an EFFECT SIZE lives on, using a column of "
                         "the reference_concordance.py table (default "
                         "'reference_c').  C carries e^2 var_A, so the ratio "
                         "C_cell/C_reference is free of the enzyme term at "
                         "any marginal -- measured, C fell by exactly e^2 at "
                         "every marginal skew tested.  Use this for a "
                         "coordination matrix, not --contrast-reference-d, "
                         "which is logit-based and only fits a reliability "
                         "coefficient in (0, 1).  A log odds ratio matrix "
                         "fits NEITHER: L's attenuation depends on the "
                         "marginal as well as on e, so a reference taken at "
                         "chrM's marginal does not cancel the genome's, and "
                         "where chrM's marginal is near symmetric "
                         "reference_logor is ~0 and the ratio is "
                         "meaningless.  The run refuses rather than dividing "
                         "by it.")
    ap.add_argument("--contrast-reference-d", action="append", default=[],
                    metavar="NAME=PATH[:COLUMN]",
                    help="Per-cell reference from "
                         "scripts/reference_concordance.py, as in "
                         "celltype_marker_bins.py (default column "
                         "'reference_d').  The transform is logit, so the "
                         "MATRIX has to be a proportion in (0, 1) too.  With "
                         ":reference_f on a concordance_direction.parquet "
                         "(raw F) this asks whether a bin's concordant mass "
                         "sits co-open or co-closed relative to DNA with no "
                         "nucleosomes and no 3D structure in the SAME cell, "
                         "which makes cells comparable.  It does not control "
                         "the per-bin marginal: F's null position depends on "
                         "how accessible the bin is, so a less accessible "
                         "region reads co-closed against a chrM baseline "
                         "whether or not anything is coordinated -- and in "
                         "heterochromatin the artefact and the biology "
                         "predict the same sign, so agreeing with intuition "
                         "is not evidence.  Profile "
                         "concordance_marginal.parquet through the same "
                         "flags to see how much of the shape is the "
                         "marginal.")
    ap.add_argument("--contrast-regress-coverage", action="store_true",
                    help="Subtract the shared D-versus-depth curve, as in "
                         "celltype_marker_bins.py.  Needs --pair-counts.")
    ap.add_argument("--pair-counts", action="append", default=[],
                    metavar="NAME=PATH",
                    help="pair_counts_per_bin.parquet for the matching "
                         "matrix.")
    ap.add_argument("--body-bins", type=int, default=40,
                    help="Bins the domain body is scaled to (default 40).  "
                         "Past roughly the median domain width divided by "
                         "the matrix bin size, extra bins interpolate rather "
                         "than resolve anything.")
    ap.add_argument("--flank", type=int, default=2_000_000,
                    help="Unscaled bp kept on each side (default 2 Mb).")
    ap.add_argument("--flank-bins", type=int, default=20,
                    help="Bins per flank (default 20).")
    ap.add_argument("--min-domain-width", type=int, default=0, metavar="BP",
                    help="Drop domains narrower than this.  A domain below "
                         "the matrix bin size contributes one bin smeared "
                         "over the whole body axis, which is noise given a "
                         "confident shape.")
    ap.add_argument("--baseline", choices=["none", "flank"], default="none",
                    help="'flank' subtracts each CELL's own mean over the "
                         "unscaled flanks, so the outside baseline is "
                         "identical between groups by construction and the "
                         "panel shows CHANGE across the domain rather than "
                         "level.  That is a loss as well as a gain: a "
                         "genuine genome-wide difference between the arms "
                         "disappears from the figure, and 'tumour is higher "
                         "overall' can no longer be read off it.  Per cell, "
                         "not per group -- subtracting a group mean would "
                         "leave every cell's own level intact and the "
                         "interval would still be dominated by it.")
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--ylabel", default=None,
                    help="Default: read from the --matrix filename, so a "
                         "panel drawing C is not labelled D.")
    ap.add_argument("--members", choices=["show", "hide"], default="show",
                    help="The faint per-cell-type lines.  They exist as the "
                         "pseudo-replication check -- a group difference "
                         "carried by one member is not a group difference, "
                         "and no interval across cells can show that -- so "
                         "'hide' is for a final panel after that check has "
                         "been made, not instead of making it.")
    ap.add_argument("--title", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--plot-format", default="pdf",
                    choices=["pdf", "png", "svg"])
    args = ap.parse_args(argv)
    # Checked before any I/O: a contradiction between flags should not wait
    # on a file read to be reported.
    if args.from_counts:
        _clash = [f for f, on in (
            ("--contrast-center", args.contrast_center != "none"),
            ("--contrast-reference-d", bool(args.contrast_reference_d)),
            ("--contrast-reference-scale",
             bool(args.contrast_reference_scale)),
            ("--contrast-regress-coverage", args.contrast_regress_coverage),
        ) if on]
        if _clash:
            sys.exit(
                f"--from-counts computes the statistic from pooled counts, "
                f"so {', '.join(_clash)} cannot be applied -- they act on "
                f"the statistic, not on counts.  Drop them; --baseline flank "
                f"still works and is the per-cell control that matters for a "
                f"profile.")

    bp = _bp_mod()
    if args.matrix is None and not args.from_counts:
        sys.exit("one of --matrix or --from-counts is required")
    if args.matrix is None and not args.from_counts_statistic:
        sys.exit("--from-counts without --matrix needs "
                 "--from-counts-statistic: with no matrix there is no "
                 "filename to read the statistic from, and guessing one "
                 "would decide the panel silently")
    if args.matrix is not None:
        name, mpath = (args.matrix.split("=", 1) if "=" in args.matrix
                       else ("gch", args.matrix))
        M = pd.read_parquet(mpath)
    else:
        # The grid, from the counts themselves.  They carry the same cells
        # and bins, so a matrix whose values are never read is not needed
        # to supply it -- only to be cross-checked against.
        name, mpath = args.from_counts_statistic, None
        _g = _counts_dir(args.from_counts) / "n_linked.parquet"
        if not _g.exists():
            sys.exit(f"--from-counts without --matrix needs {_g} for the "
                     f"cell x bin grid; re-run step 5 with --emit-counts")
        _gr = pd.read_parquet(_g)
        M = pd.DataFrame(np.nan, index=_gr.index, columns=_gr.columns)
        print(f"  no --matrix: cell x bin grid taken from {_g.name} "
              f"({len(M):,} cells x {M.shape[1]:,} bins)", file=sys.stderr)
    name = args.name or name
    samples = pd.read_csv(args.samples, sep="\t")
    if args.cell_type_column not in samples.columns:
        sys.exit(f"--cell-type-column {args.cell_type_column!r} not in "
                 f"--samples ({list(samples.columns)})")
    ct = samples.set_index("cell_name")[args.cell_type_column].astype(str)
    ct = ct.reindex(M.index).dropna()
    M = M.loc[ct.index]

    if args.exclude_chrom:
        drop = {c.strip().lower().replace("chr", "")
                for c in args.exclude_chrom.split(",") if c.strip()}
        drop = {"m" if c in ("m", "mt") else c for c in drop}
        keep = [c for c in M.columns
                if str(c).rsplit(":", 1)[0].lower().replace("chr", "")
                not in drop]
        if len(keep) < len(M.columns):
            print(f"  dropped {len(M.columns) - len(keep):,} bins on "
                  f"{args.exclude_chrom}", file=sys.stderr)
        M = M[keep]

    if args.remove_batch == "within-celltype":
        if not args.batch_column:
            sys.exit("--remove-batch needs --batch-column")
        cols = [c.strip() for c in args.batch_column.split(",") if c.strip()]
        missing = [c for c in cols if c not in samples.columns]
        if missing:
            sys.exit(f"--batch-column {missing} not in --samples")
        bdf = samples.set_index("cell_name")[cols].astype(str).reindex(M.index)
        M = bp._marker_mod()._remove_batch(M, ct, bdf)
        print(f"  batch-corrected within cell type on {'+'.join(cols)}",
              file=sys.stderr)

    # The SAME chain as celltype_marker_bins.py, in the same order: batch,
    # then centring, then the reference, then depth.  Profiling raw values of
    # bins that were DETECTED on the normalised matrix shows the genome-wide
    # offset between the arms with the domain structure riding invisibly on
    # top -- the panel is then a picture of the nuisance the contrast removed.
    mk = bp._marker_mod()
    steps = []
    Xc = M.to_numpy(float)
    # Reference first, then centring -- see celltype_marker_bins.py: the
    # reference correction takes logit and needs the natural (0, 1) scale,
    # which centring destroys.
    ref_specs = _named(args.contrast_reference_d, "--contrast-reference-d")
    if name in ref_specs:
        _rp, _, _rcol = str(ref_specs[name]).partition(":")
        _rcol = _rcol or "reference_d"
        rp = Path(_rp)
        if not rp.exists():
            sys.exit(f"--contrast-reference-d {name}: no such file {rp}")
        rdf = pd.read_csv(rp, sep="\t")
        if "cell" not in rdf.columns or _rcol not in rdf.columns:
            sys.exit(f"--contrast-reference-d {name}: {rp} needs 'cell' and "
                     f"{_rcol!r} columns")
        rs = rdf.set_index("cell")[_rcol].reindex(M.index)
        Xc, rinfo = mk.correct_by_reference(Xc, rs.to_numpy(float))
        steps.append(f"referenced to {_rcol} for "
                     f"{rinfo.get('n_corrected', 0)}/{len(M)} cells")
    sc_specs = _named(args.contrast_reference_scale,
                      "--contrast-reference-scale")
    if name in sc_specs:
        spec = sc_specs[name]
        sp, _, col = spec.partition(":")
        col = col or "reference_c"
        if not Path(sp).exists():
            sys.exit(f"--contrast-reference-scale {name}: no such file {sp}")
        sdf = pd.read_csv(sp, sep="\t")
        if "cell" not in sdf.columns or col not in sdf.columns:
            sys.exit(f"--contrast-reference-scale {name}: {sp} needs 'cell' "
                     f"and {col!r} columns; it has "
                     f"{list(sdf.columns)[:8]}")
        ss = sdf.set_index("cell")[col].reindex(M.index)
        Xc, sinfo = mk.scale_by_reference(Xc, ss.to_numpy(float))
        if sinfo.get("note"):
            sys.exit(f"--contrast-reference-scale {name} ({col}): "
                     f"{sinfo['note']}")
        steps.append(f"rescaled by {col} for "
                     f"{sinfo['n_corrected']}/{len(M)} cells")
    if args.contrast_center == "cell-median":
        Xc = mk.center_cells(Xc)
        steps.append("centred each cell on its median")
    cnt_specs = _named(args.pair_counts, "--pair-counts")
    if args.contrast_regress_coverage:
        if name not in cnt_specs:
            sys.exit(f"--contrast-regress-coverage needs --pair-counts "
                     f"{name}=pair_counts_per_bin.parquet")
        C = pd.read_parquet(cnt_specs[name]).reindex(
            index=M.index, columns=M.columns)
        Xc, ginfo = mk.regress_out_coverage(Xc, C.to_numpy(float))
        steps.append(f"regressed out depth over "
                     f"{ginfo.get('n_strata', 0)} strata")
    if steps:
        M = pd.DataFrame(Xc, index=M.index, columns=M.columns)
        print("  normalised: " + "; ".join(steps), file=sys.stderr)
    if args.baseline == "flank":
        # Reported apart from the matrix normalisation, and not counted as
        # it: anchoring is a DISPLAY choice made after the fact, and it
        # cannot stand in for matching what the detection run did to the
        # values.  Conflating them would let --baseline flank silence the
        # warning below while the profile still disagreed with the contrast.
        print("  anchored each cell to its own flank mean: the outside "
              "baseline is now identical between groups by construction, so "
              "read the panel as change across the domain, not as level.",
              file=sys.stderr)
    if not steps and not args.from_counts:
        print("  WARNING no normalisation applied.  If these regions came "
              "from a contrast, its bins were called on a matrix that had "
              "been centred and corrected, so a raw profile shows the "
              "genome-wide arm offset instead of the structure -- pass the "
              "same --contrast-center / --contrast-reference-d / "
              "--contrast-regress-coverage as the detection run.",
              file=sys.stderr)

    groups = {}
    for lab, spec in _named(args.group, "--group").items():
        mem = [g.strip() for g in spec.split(",") if g.strip()]
        bad = [g for g in mem if g not in set(ct)]
        if bad:
            sys.exit(f"--group {lab}: not in --samples "
                     f"{args.cell_type_column!r}: {bad}\n"
                     f"  have: {sorted(set(ct))}")
        groups[lab] = mem

    _pooled = None
    if args.from_counts:
        _stat = args.from_counts_statistic or statistic_of(mpath)
        if mpath is None:
            _stat = args.from_counts_statistic
        if _stat not in POOLABLE:
            sys.exit(f"--from-counts cannot tell which statistic "
                     f"{Path(mpath).name!r} holds; pass "
                     f"--from-counts-statistic.")
        _alpha = load_prior(args.from_counts)
        _pooled = {"counts": load_counts(args.from_counts, M.index,
                                         M.columns),
                   "alpha": _alpha, "stat": _stat}
        print(f"  the per-cell corrections do not apply here: the "
              f"statistic is rebuilt from each position's own pooled pairs, "
              f"so there is no genome-wide arm offset riding on it to "
              f"remove.  --baseline flank still anchors each cell.",
              file=sys.stderr)
        print(f"  pooling raw counts per axis position and computing "
              f"{_stat} from them (prior nu={sum(_alpha.values()):.1f})",
              file=sys.stderr)
        _how = ("--from-counts-statistic" if args.from_counts_statistic
                else f"the --matrix filename {Path(mpath).name!r}")
        _cdir = _counts_dir(args.from_counts)
        print(f"  read: {_cdir}/{{{','.join(COUNT_KEYS)}}}.parquet "
              f"and dirichlet_multinomial_prior.json.  --matrix supplies "
              f"only the cell x bin grid and the statistic's NAME; none of "
              f"its values are read.", file=sys.stderr)
        print(f"  the statistic came from {_how}", file=sys.stderr)
        _rank = ({} if mpath is None
                 else which_statistic_matches(M, _pooled["counts"], _alpha))
        if mpath is None:
            print(f"  no --matrix, so nothing to cross-check the statistic "
                  f"against; it is whatever --from-counts-statistic said",
                  file=sys.stderr)
        if _rank:
            _best = next(iter(_rank))
            _line = "  ".join(f"{k}={v:.2f}" for k, v in
                              list(_rank.items())[:4])
            print(f"  --matrix values look most like {_best!r} "
                  f"(|rho| {_line})", file=sys.stderr)
            if _best != _stat:
                print(f"  WARNING the name says {_stat!r} but the values in "
                      f"{Path(mpath).name} rank {_best!r} higher.  Nothing "
                      f"reads those values, so a renamed, symlinked or "
                      f"stale --matrix would change the plotted quantity in "
                      f"silence -- check the path, or state the statistic "
                      f"outright with --from-counts-statistic.",
                      file=sys.stderr)

    bins = pd.DataFrame({
        "chrom": [str(c).rsplit(":", 1)[0] for c in M.columns],
        "start": [int(str(c).rsplit(":", 1)[1]) for c in M.columns]})
    widths = np.diff(np.sort(bins.start.unique()))
    step_bp = int(np.median(widths)) if widths.size else 250_000
    bins["end"] = bins.start + step_bp

    colors = {}
    if args.color_column and args.color_column in samples.columns:
        for g, c in zip(samples[args.cell_type_column].astype(str),
                        samples[args.color_column].astype(str)):
            if c and str(c).lower() not in ("nan", "none", ""):
                colors.setdefault(g, c)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _stat_name = (_stat if args.from_counts else statistic_of(mpath))
    results, members, rows, tests = {}, {}, [], []
    _donor = None
    if args.donor_column:
        if args.donor_column not in samples.columns:
            sys.exit(f"--donor-column {args.donor_column!r} not in "
                     f"--samples: {list(samples.columns)[:10]}")
        _donor = samples.set_index("cell_name")[args.donor_column].astype(str)
    for rname, rspec in _named(args.region, "--region").items():
        rpath, _, rclass = rspec.partition("::")
        dom = bp.read_bed(rpath, merge=True, keep_class=rclass or None)
        n_all = len(dom)
        if args.min_domain_width:
            dom = dom[(dom["end"] - dom["start"]) >= args.min_domain_width]
        if not len(dom):
            print(f"  {rname}: no domain left", file=sys.stderr)
            continue
        w = (dom["end"] - dom["start"]).to_numpy(float)
        thin = int((w < step_bp).sum())
        bi, pi, total, dmi = scaled_index(bins, dom, args.body_bins,
                                          args.flank, args.flank_bins)
        kw = dict(baseline=args.baseline, n_flank=args.flank_bins,
                  n_body=args.body_bins)
        if _pooled is not None:
            kw["pooled"] = _pooled
        prof = {lab: group_profile(M, ct, mem, bi, pi, total, args.conf, **kw)
                for lab, mem in groups.items()}
        results[rname] = prof
        _dv = None
        if args.domain_test and len(groups) >= 2:
            _dv = {lab: per_domain_values(M, ct, mem, bi, pi, dmi,
                                          args.flank_bins, args.body_bins,
                                          total, baseline=args.baseline,
                                          pooled=_pooled)
                   for lab, mem in groups.items()}
        _t = region_group_test(prof, ct, donor=_donor,
                               baseline=args.baseline, domain_vals=_dv)
        if len(_t):
            _t.insert(0, "region", rname)
            tests.append(_t)
        members[rname] = ({} if args.members == "hide" else
                          {lab: member_curves(M, ct, mem, bi, pi, total,
                                              **kw)
                           for lab, mem in groups.items() if len(mem) > 1})
        print(f"  {rname}: {len(dom):,} of {n_all:,} domains "
              f"(median width {np.median(w) / 1e6:.2f} Mb), "
              f"{bi.size:,} bin placements", file=sys.stderr)
        if thin:
            print(f"  {rname}: {thin:,} domains are narrower than the "
                  f"{step_bp / 1e3:g} kb matrix bin, so each contributes a "
                  f"single value stretched across the whole body axis.  "
                  f"--min-domain-width {step_bp} drops them.",
                  file=sys.stderr)
        body = slice(args.flank_bins, args.flank_bins + args.body_bins)
        for lab, r in prof.items():
            inside = float(np.nanmean(r["mean"][body]))
            flank = float(np.nanmean(np.concatenate(
                [r["mean"][:args.flank_bins],
                 r["mean"][args.flank_bins + args.body_bins:]])))
            rows.append({"region": rname, "group": lab,
                         "n_cells": r["n_cells"], "mean_body": inside,
                         "mean_flank": flank, "body_minus_flank":
                         inside - flank,
                         "members": ";".join(f"{k}:{v}" for k, v in
                                             r["members"].items())})
            if len(groups[lab]) > 1 and r["n_cells"]:
                n_don = len([1 for v in r["members"].values() if v])
                if n_don < 3:
                    print(f"  {rname}/{lab}: the band is across "
                          f"{r['n_cells']} CELLS drawn from {n_don} "
                          f"cell types.  Cells from one donor are not "
                          f"independent replicates of the group, so read it "
                          f"as within-group precision, not as evidence about "
                          f"{lab} in general; the faint per-member lines are "
                          f"the check that matters.", file=sys.stderr)
    if not results:
        sys.exit("no region set produced a profile")

    tab = pd.DataFrame(rows)
    tab.to_csv(out / f"scaled_profile_{name}.tsv", sep="\t", index=False)

    if tests:
        tt = pd.concat(tests, ignore_index=True)
        tt.to_csv(out / f"region_group_test_{name}.tsv", sep="\t",
                  index=False)
        _unit = ("domain minus flank" if args.baseline == "flank"
                 else "absolute level")
        print(f"\n  region test ({_unit} per cell, two-sided "
              f"Mann-Whitney):", file=sys.stderr)
        for r in tt.itertuples():
            line = (f"  {r.region:>10s} {r.group_a} vs {r.group_b}: "
                    f"diff={r.diff_cell:+.3g}  p_cell={r.p_cell:.3g} "
                    f"({r.n_cell_a} vs {r.n_cell_b} cells)")
            if "p_domain" in tt.columns and np.isfinite(r.p_domain):
                line += (f"   |  p_domain={r.p_domain:.3g} "
                         f"({r.n_domains} domains, median "
                         f"{r.median_diff:+.3g}, mean "
                         f"{r.mean_diff:+.3g}, {r.frac_positive:.0%} "
                         f"positive)")
            if "p_donor" in tt.columns and np.isfinite(r.p_donor):
                line += (f"   |  p_donor={r.p_donor:.3g} "
                         f"({r.n_donor_a} vs {r.n_donor_b}, floor "
                         f"{r.p_floor_donor:.3g})")
            print(line, file=sys.stderr)
        if "p_donor" not in tt.columns:
            print(f"  the per-cell p is NOT a replicate test -- cells from "
                  f"one donor share its biology, prep and coverage, so a "
                  f"few hundred per arm call any systematic difference "
                  f"including the donor.  Pass --donor-column for the test "
                  f"that answers whether the CONDITIONS differ.",
                  file=sys.stderr)
        print(f"  -> region_group_test_{name}.tsv", file=sys.stderr)
    fig = out / f"scaled_profile_{name}.{args.plot_format}"
    _stat_lbl = STAT_LABEL.get(_stat_name, _stat_name or "concordance")
    scaled_figure(results, fig,
                  args.title or f"{name}: mean {_stat_lbl} across scaled "
                  f"domains",
                  args.ylabel or f"{name.upper()} {_stat_lbl}",
                  args.body_bins, args.flank_bins,
                  args.flank, colors, args.conf, members)
    print(f"  -> {fig.name}, scaled_profile_{name}.tsv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
