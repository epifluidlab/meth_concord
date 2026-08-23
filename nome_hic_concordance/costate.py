"""Cross-mark co-state analysis, at the level the co-states are measured.

Two identities constrain everything here, and both are exact for any joint
distribution over two binary ends with marginals p1, p2:

    pi_open - pi_closed          = p1 + p2 - 1
    O_open - E_open              = O_closed - E_closed  = Cov

The second is the one that decides the design.  The excess of co-open pairs
over chance and the excess of co-closed pairs over chance are THE SAME NUMBER,
so coordination has no direction of its own; by the first identity, whatever
direction is observed is the marginal level.  "Co-open coordination versus
co-closed coordination" is therefore not a third degree of freedom -- there are
two, level and association -- and any classification of BINS into co-open and
co-closed is a classification by level wearing a different name.

So this module does not try to separate them at the bin level.  It offers the
two questions that do have answers:

``costate_*``   Classify each (CELL, BIN) observation, and count the 3x3
                STRATIFIED BY BIN.  Within one bin the genomic level is fixed,
                so what varies across cells is which cells are locally co-open
                there, and the association between the marks is measured with
                the confound held constant by design rather than adjusted for.
                Pooled with a Mantel-Haenszel odds ratio.

``level_grid``  Accept that direction is level, and ask the question that
                phrasing makes available: given a bin's accessibility level and
                its methylation level, how STRONG is its long-range
                coordination?  Level on both axes, C in the cells.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Code order is fixed so the 2x2 core of the table is always [2,0] / [0,2]:
#   2 = the HIGH co-state (co-open for GCH, co-methylated for HCG)
#   1 = mixed (no positive coordination to give a direction)
#   0 = the LOW co-state (co-closed, co-unmethylated)
CO_LOW, CO_MIXED, CO_HIGH, UNUSABLE = 0, 1, 2, -1
GCH_STATES = ("co-closed", "mixed", "co-open")
HCG_STATES = ("co-unmethylated", "mixed", "co-methylated")


def costate_codes(coordination, direction, min_coordination=0.0,
                  expected=None):
    """Per (cell, bin) co-state code from the components.

    A cell-bin is given a direction only when it is actually coordinated
    (C > min_coordination); with C <= 0 the two ends agree no more than chance
    and F is describing noise, so it is 'mixed' rather than being forced into
    a side.  Among coordinated observations the direction is F vs 0.5: more of
    the concordant mass co-high than co-low.

    Note what this does NOT claim.  F > 0.5 happens where the local level is
    above 0.5, by the identity above; the classification is honest about a
    single (cell, bin) and becomes a statement about coordination only once it
    is compared ACROSS CELLS WITHIN A BIN, where the level is held fixed.

    ``expected`` is the fix for that when an absolute call is wanted.  Pass
    the direction under the marginal-matched null, F_exp = E_open/(E_open +
    E_closed) (emitted as ``concordance_direction_excess``'s reference), and
    the split is taken against it rather than against 0.5.  It matters more
    than it looks: under independence at marginal p-bar the concordant mass
    already splits F_indep = p^2/(p^2 + (1-p)^2), which is 0.31 at
    p-bar = 0.40.  Thresholding such data at 0.5 puts EVERY coordinated
    cell-bin in CO_LOW and leaves CO_HIGH empty -- not because nothing is
    co-open, but because the threshold is unreachable.
    """
    C = np.asarray(coordination, dtype=float)
    F = np.asarray(direction, dtype=float)
    out = np.full(C.shape, CO_MIXED, dtype=np.int8)
    if expected is None:
        ref = np.full(C.shape, 0.5)
    else:
        ref = np.broadcast_to(np.asarray(expected, dtype=float),
                              C.shape).astype(float)
    ok = np.isfinite(C) & np.isfinite(F) & np.isfinite(ref)
    coord = ok & (C > min_coordination)
    out[coord & (F > ref)] = CO_HIGH
    out[coord & (F < ref)] = CO_LOW
    out[~ok] = UNUSABLE
    return out


def stratified_counts(g_code, h_code, batch_codes=None):
    """One contingency table per stratum, over cells → (n_strata, 3, 3).

    A stratum is a bin, or a (bin, batch) pair when ``batch_codes`` is given.
    Adding batch to the stratification asks the harder question: batch is a
    CELL property, so it survives stratifying by bin alone -- if one batch's
    cells read co-open in GCH and co-unmethylated in HCG for technical reasons,
    that lands in the within-bin association.  Splitting the strata by batch as
    well means the association is only ever measured between cells processed
    together.
    """
    g = np.asarray(g_code)
    h = np.asarray(h_code)
    if g.shape != h.shape:
        raise ValueError(f"shape mismatch: {g.shape} vs {h.shape}")
    if batch_codes is None:
        groups = [slice(None)]
    else:
        b = np.asarray(batch_codes)
        groups = [np.flatnonzero(b == u) for u in np.unique(b)]
    out = np.zeros((g.shape[1] * len(groups), 3, 3), dtype=np.int64)
    for k, rows in enumerate(groups):
        gg, hh = g[rows], h[rows]
        off = k * g.shape[1]
        for i in range(3):
            gi = gg == i
            for j in range(3):
                out[off:off + g.shape[1], i, j] = np.count_nonzero(
                    gi & (hh == j), axis=0)
    return out


def _abcd(tab):
    """The 2x2 core: co-open/co-closed against co-unmethylated/co-methylated.

    'mixed' is dropped rather than folded into a side -- an observation with no
    coordination has no direction to contribute, and folding it in would let
    the number of UNCOORDINATED cell-bins drive an odds ratio about direction.
    """
    t = np.asarray(tab)
    a = t[..., CO_HIGH, CO_LOW]      # co-open  x co-unmethylated  (canonical)
    b = t[..., CO_HIGH, CO_HIGH]     # co-open  x co-methylated
    c = t[..., CO_LOW, CO_LOW]       # co-closed x co-unmethylated
    d = t[..., CO_LOW, CO_HIGH]      # co-closed x co-methylated   (canonical)
    return (a.astype(float), b.astype(float), c.astype(float),
            d.astype(float))


def odds_ratio(tab):
    """Crude OR of the pooled 2x2.  > 1 means the canonical pairing."""
    a, b, c, d = _abcd(np.asarray(tab).sum(axis=0) if np.ndim(tab) == 3
                       else tab)
    return float(a * d / (b * c)) if b > 0 and c > 0 else np.nan


def mantel_haenszel(tab):
    """Bin-stratified common odds ratio.

    sum_i(a_i d_i / n_i) / sum_i(b_i c_i / n_i), the Mantel-Haenszel estimator.
    Each bin is its own stratum, so the level -- which is what makes a bin
    co-open in the first place -- is constant inside every term and cannot
    contribute to the pooled estimate.  The difference between this and the
    crude OR is exactly the amount of the naive answer that was level.
    """
    a, b, c, d = _abcd(tab)
    n = a + b + c + d
    ok = n > 0
    if not ok.any():
        return np.nan
    num = float(np.sum(a[ok] * d[ok] / n[ok]))
    den = float(np.sum(b[ok] * c[ok] / n[ok]))
    return num / den if den > 0 else np.inf if num > 0 else np.nan


def per_stratum_log_or(tab, haldane=0.5):
    """log OR per bin, Haldane-corrected so a zero cell does not remove it."""
    a, b, c, d = _abcd(tab)
    keep = (a + b + c + d) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        lor = np.log(((a + haldane) * (d + haldane))
                     / ((b + haldane) * (c + haldane)))
    return np.where(keep, lor, np.nan)


def _ci_and_p(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if v.size < 20:
        return None
    lo, hi = np.percentile(v, [2.5, 97.5])
    # Two-sided tail against OR = 1, floored at 2/(B+1): a finite bootstrap
    # cannot justify p = 0.
    n_side = min(int(np.sum(v <= 1.0)), int(np.sum(v >= 1.0)))
    return float(lo), float(hi), min(1.0, 2.0 * (1 + n_side) / (v.size + 1))


def bootstrap_ors(g_code, h_code, n_boot=200, seed=42, batch_codes=None):
    """Cell-level bootstrap CIs for the crude and the MH odds ratio.

    Cells, not (cell, bin) observations: one cell contributes an observation to
    every bin, so a table of 15 million rows rests on ~1,500 independent units.
    The Robins-Breslow-Greenland variance assumes the strata are independent
    samples and would be far too small here by orders of magnitude.

    Both estimators come off the SAME resamples, so the gap between them --
    which is the part of the crude answer that was level -- is not itself a
    sampling artifact of comparing two independent bootstraps.
    """
    g = np.asarray(g_code)
    h = np.asarray(h_code)
    n_cells = g.shape[0]
    if n_cells < 3 or n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    mh, crude = [], []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n_cells, n_cells)
        tab = stratified_counts(g[idx], h[idx], batch_codes=(
            None if batch_codes is None else np.asarray(batch_codes)[idx]))
        mh.append(mantel_haenszel(tab))
        crude.append(odds_ratio(tab))
    out = {"mh": _ci_and_p(mh), "crude": _ci_and_p(crude)}
    return None if out["mh"] is None else out


def bootstrap_mh(g_code, h_code, n_boot=200, seed=42, batch_codes=None):
    """The MH entry of :func:`bootstrap_ors` → (lo, hi, p)."""
    got = bootstrap_ors(g_code, h_code, n_boot, seed, batch_codes)
    return None if got is None else got["mh"]


def level_grid(level_x, level_y, values, n_bins=3, labels=None):
    """Bin two per-bin LEVELS into quantile bands and summarise a value in each.

    The honest use of the identity: since direction is level, stop calling it
    direction and put level on both axes, then ask what the coordination
    STRENGTH does across that grid.  Quantile bands rather than fixed cuts, so
    the grid is populated whatever the two marks' genome-wide levels are.

    ``values`` maps a name to a per-bin array; each is summarised by its mean
    and standard error within every cell of the grid.
    """
    x = np.asarray(level_x, float)
    y = np.asarray(level_y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < n_bins ** 2:
        raise ValueError(f"only {int(ok.sum())} usable bins for a "
                         f"{n_bins}x{n_bins} grid")
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    xe = np.percentile(x[ok], qs)
    ye = np.percentile(y[ok], qs)
    xi = np.digitize(x, xe)
    yi = np.digitize(y, ye)
    lab = labels or [f"Q{i + 1}" for i in range(n_bins)]
    rows = []
    for i in range(n_bins):
        for j in range(n_bins):
            m = ok & (xi == i) & (yi == j)
            row = {"x_band": lab[i], "y_band": lab[j], "n_bins": int(m.sum())}
            for name, v in values.items():
                vv = np.asarray(v, float)[m]
                vv = vv[np.isfinite(vv)]
                row[f"mean_{name}"] = float(np.mean(vv)) if vv.size else np.nan
                row[f"sem_{name}"] = (float(np.std(vv, ddof=1) / np.sqrt(vv.size))
                                      if vv.size > 1 else np.nan)
                row[f"n_{name}"] = int(vv.size)
            rows.append(row)
    return pd.DataFrame(rows), xe, ye
