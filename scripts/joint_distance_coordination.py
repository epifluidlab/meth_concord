#!/usr/bin/env python3
"""Distance–coordination curves for joint GCH+HCG: contact vs distance-matched null.

For every cis read-pair (two ends in 3-D contact) this measures how the two
ends' epigenetic states co-vary as a function of the genomic distance between
them, for three pairings of the per-end Beta-Binomial mixture posteriors stored
in the ``*.posteriors.parquet`` (``a = P(accessible|GpC)``, ``m = P(methylated|
CpG)``):

  * GCH–GCH : r(accessibility_i, accessibility_j)  — do two contacting ends
              share accessibility?
  * HCG–HCG : r(methylation_i,   methylation_j)     — ... share methylation?
  * GCH–HCG : r(accessibility_i, methylation_j)     — cross-END, cross-modality
              (canonically NEGATIVE: open ↔ unmethylated, even across ends).
              Pools both directions (a_i,m_j) and (a_j,m_i).

By default (``--cell-centering pooled``) every linked pair from every cell whose
two ends both clear the min-site gate enters ONE Pearson correlation per
distance bin (sufficient stats summed across all cells, centred once).  With
``--cell-centering within-cell`` each cell is instead centred by its own
(cell, bin) mean first, removing cell-type covariation from the contact curve.
(This is the BETWEEN-end coordination the concordance method rides on;
``joint_state_2x2.py`` measures the orthogonal WITHIN-end a–m coupling.)

Each curve is drawn with two nulls and a **95% CI** (cell-resampling bootstrap,
shaded):

  * **1-D autocorrelation** — the SAME loci paired by genomic distance instead
    of by ligation: the co-methylation / co-accessibility you'd see from 1-D
    proximity ALONE (so it is naturally >0 at short range, where nearby CpGs are
    co-methylated).  CONTACT *above* this line is 3-D-contact-specific.
  * **distance-matched shuffle** — within (cell, distance-bin) the partner end
    is permuted (a different contact), so it traces the marginal-independence
    floor (pooled, the cell-type baseline).

So contact − autocorr is the 3-D-specific coordination beyond 1-D proximity,
and the bootstrap CI says whether a gap is real given the cell-to-cell spread.

Distance is binned log-spaced from --min-distance (default 1 bp, the finest
within-read scale) to --max-distance (default 1 Mb), --bins-per-decade per
decade.  Output: a multi-panel PDF and a sibling TSV of the per-bin values.

Usage
-----
    python scripts/joint_distance_coordination.py \\
        --gch-dir .../context_GCH/04_posteriors \\
        --hcg-dir .../context_HCG/04_posteriors \\
        --out joint_distance_coordination.pdf

    # --region-bin-size denoises per-end posteriors to their (chrom, pos//SIZE)
    # genomic-bin mean POOLED ACROSS ALL CELLS (a single cell is too sparse to
    # denoise a small bin) before correlating, surfacing the strong region/
    # compartment-scale GCH-HCG coupling that single-end noise hides — the
    # distance-resolved analogue of joint_state_2x2.py's r(region):
    #     ... --region-bin-size 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: the three between-end measures, in (key, x-modality, y-modality, label) form.
_MEASURES = (("gg", "GCH–GCH  r(acc_i, acc_j)"),
             ("hh", "HCG–HCG  r(meth_i, meth_j)"),
             ("gh", "GCH–HCG  r(acc_i, meth_j)  [cross-end]"))
_TLAB = {1: "1bp", 10: "10bp", 100: "100bp", 1_000: "1kb", 10_000: "10kb",
         100_000: "100kb", 1_000_000: "1Mb", 10_000_000: "10Mb"}


def _cells(d: Path) -> dict:
    return {p.name.replace(".posteriors.parquet", ""): p
            for p in d.glob("*.posteriors.parquet")}


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["chrom", "pos1", "pos2", "n1", "n2",
                                        "posterior1", "posterior2"])
    df["chrom"] = df["chrom"].astype(str)
    return df


def _sums(bins, x, y, mask, nb):
    """Per-bin raw sufficient statistics (n, Σx, Σy, Σx², Σy², Σxy) for the rows
    selected by ``mask`` — a (6, nb) array so two directions can be added."""
    b = bins[mask]; X = x[mask]; Y = y[mask]
    return np.stack([
        np.bincount(b, minlength=nb).astype(np.float64),
        np.bincount(b, weights=X, minlength=nb),
        np.bincount(b, weights=Y, minlength=nb),
        np.bincount(b, weights=X * X, minlength=nb),
        np.bincount(b, weights=Y * Y, minlength=nb),
        np.bincount(b, weights=X * Y, minlength=nb),
    ])


def _center(s):
    """Raw sums → this cell's mean-centred partial sums (cov, var_x, var_y, n).

    cov = Σxy − ΣxΣy/n is the cell-centred cross-sum; summing it across cells
    gives the within-cell pooled covariance numerator (Σ_c Σ_i (x−x̄_c)(y−ȳ_c)),
    so r = Σcov / √(Σvar_x · Σvar_y) is the within-cell Pearson correlation."""
    n, sx, sy, sxx, syy, sxy = s
    nz = n > 0
    cov = np.zeros_like(n); vx = np.zeros_like(n); vy = np.zeros_like(n)
    cov[nz] = sxy[nz] - sx[nz] * sy[nz] / n[nz]
    vx[nz] = sxx[nz] - sx[nz] * sx[nz] / n[nz]
    vy[nz] = syy[nz] - sy[nz] * sy[nz] / n[nz]
    return np.stack([cov, vx, vy, n])


def _within_bin_partner(bins, rng):
    """For each row, the index of a partner row drawn uniformly from the SAME
    distance bin — a within-bin permutation (distance-matched shuffle).

    ``argsort(bins)`` and ``lexsort((rand, bins))`` both group rows by bin in the
    same order, so position k of each is in the same bin; mapping one onto the
    other pairs every row with a random row in its bin."""
    n = bins.size
    sort_orig = np.argsort(bins, kind="stable")
    sort_rand = np.lexsort((rng.random(n), bins))
    partner = np.empty(n, dtype=np.int64)
    partner[sort_orig] = sort_rand
    return partner


def _acc_into(arr, bi, x, y, mask):
    """Add (n, Σx, Σy, Σx², Σy², Σxy) for masked x,y into column ``bi`` of a
    (6, nb) sums array."""
    x = x[mask]; y = y[mask]
    if x.size == 0:
        return
    arr[0, bi] += x.size
    arr[1, bi] += x.sum(); arr[2, bi] += y.sum()
    arr[3, bi] += (x * x).sum(); arr[4, bi] += (y * y).sum()
    arr[5, bi] += (x * y).sum()


def _autocorr_sums(chrom, pos, A, M, AO, MO, edges, rng):
    """1-D genomic autocorrelation null: pair each locus with ONE other locus in
    the same genomic-distance bin (sampled, same chromosome) → the co-methylation
    / co-accessibility you'd see from 1-D proximity ALONE, ignoring 3-D contact.

    Uses the SAME loci as the contact curve, paired by genomic distance instead
    of by ligation, so contact-minus-autocorr isolates the 3-D-contact-specific
    coordination (and the autocorr line is naturally >0 at short range, where
    nearby CpGs are co-methylated).  Returns raw (6, nb) sums per measure."""
    nb = len(edges) - 1
    out = {k: np.zeros((6, nb)) for k in ("gg", "hh", "gh")}
    order = np.lexsort((pos, chrom))
    cs = chrom[order]; ps = pos[order].astype(np.int64)
    A, M, AO, MO = A[order], M[order], AO[order], MO[order]
    uniq, starts = np.unique(cs, return_index=True)
    bounds = np.append(starts, cs.size)
    for k in range(uniq.size):
        s, e = bounds[k], bounds[k + 1]
        P = ps[s:e]
        if P.size < 2:
            continue
        av, mv, ao, mo = A[s:e], M[s:e], AO[s:e], MO[s:e]
        ie = [np.searchsorted(P, P + ed, side="left") for ed in edges]
        for bi in range(nb):
            cnt = ie[bi + 1] - ie[bi]
            has = cnt > 0
            if not has.any():
                continue
            off = np.minimum((rng.random(P.size) * cnt).astype(np.int64),
                             np.maximum(cnt - 1, 0))
            j = ie[bi] + off
            ip = np.flatnonzero(has); jp = j[ip]
            _acc_into(out["gg"], bi, av[ip], av[jp], ao[ip] & ao[jp])
            _acc_into(out["hh"], bi, mv[ip], mv[jp], mo[ip] & mo[jp])
            _acc_into(out["gh"], bi, av[ip], mv[jp], ao[ip] & mo[jp])
            _acc_into(out["gh"], bi, mv[ip], av[jp], mo[ip] & ao[jp])
    return out


def _bootstrap_ci(per_cell, pooled, min_pairs, n_boot, seed):
    """Cell-level bootstrap 95% CI for each bin's r.  Resamples CELLS (the
    independent units) with replacement — honest about the real uncertainty,
    unlike a per-pair Fisher CI that, with millions of non-independent pairs,
    would be absurdly tight.  Returns (lo, hi) arrays (NaN where undefined)."""
    nb = per_cell[0].shape[1]; nc = len(per_cell)
    nanrow = np.full(nb, np.nan)
    if n_boot <= 0 or nc < 2:
        return nanrow, nanrow
    stack = (np.stack(per_cell) if pooled
             else np.stack([_center(s) for s in per_cell]))   # (nc, 6|4, nb)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, nb))
    for b in range(n_boot):
        agg = stack[rng.integers(0, nc, nc)].sum(axis=0)
        cov, vx, vy, n = _center(agg) if pooled else agg
        denom = np.sqrt(vx * vy)
        boots[b] = np.where((denom > 0) & (n >= min_pairs),
                            cov / np.where(denom > 0, denom, 1.0), np.nan)
    ok = np.isfinite(boots).any(axis=0)
    lo, hi = nanrow.copy(), nanrow.copy()
    if ok.any():
        lo[ok] = np.nanpercentile(boots[:, ok], 2.5, axis=0)
        hi[ok] = np.nanpercentile(boots[:, ok], 97.5, axis=0)
    return lo, hi


def _load_kept(gpath, hpath, edges, min_gpc, min_cpg):
    """Shared load+join+distance-filter for both passes.  Returns
    (chrom, p1, p2, bins, a1, a2, m1, m2, ga1, ga2, hm1, hm2, nb) over the kept
    pairs, or None."""
    g = _load(Path(gpath)).rename(columns={"n1": "n1g", "n2": "n2g",
                                           "posterior1": "a1", "posterior2": "a2"})
    h = _load(Path(hpath)).rename(columns={"n1": "n1h", "n2": "n2h",
                                           "posterior1": "m1", "posterior2": "m2"})
    m = g.merge(h, on=["chrom", "pos1", "pos2"], how="inner")
    if len(m) == 0:
        return None
    pos1 = m["pos1"].to_numpy(np.int64); pos2 = m["pos2"].to_numpy(np.int64)
    dist = np.abs(pos2 - pos1)
    keep = (dist >= edges[0]) & (dist <= edges[-1])
    if not keep.any():
        return None
    nb = len(edges) - 1
    bins = np.clip(np.digitize(dist[keep], edges) - 1, 0, nb - 1).astype(np.int64)

    def col(name):
        return m[name].to_numpy(np.float64)[keep]
    a1, a2 = col("a1"), col("a2"); m1, m2 = col("m1"), col("m2")
    ga1 = (col("n1g") >= min_gpc) & np.isfinite(a1)
    ga2 = (col("n2g") >= min_gpc) & np.isfinite(a2)
    hm1 = (col("n1h") >= min_cpg) & np.isfinite(m1)
    hm2 = (col("n2h") >= min_cpg) & np.isfinite(m2)
    a1 = np.nan_to_num(a1); a2 = np.nan_to_num(a2)
    m1 = np.nan_to_num(m1); m2 = np.nan_to_num(m2)
    return (m["chrom"].to_numpy()[keep], pos1[keep], pos2[keep], bins,
            a1, a2, m1, m2, ga1, ga2, hm1, hm2, nb)


def cell_region_sums(gpath, hpath, edges, min_gpc, min_cpg, B):
    """PASS 1 — one cell's per (chrom, pos//B) genomic-bin sums: Σa and count
    over GpC-observed ends, Σm and count over CpG-observed ends.

    A single cell is far too sparse to denoise a small bin on its own (~1 end
    per 2 kb bin), so these are summed ACROSS ALL CELLS in :func:`_build_global_
    means` to form well-pooled region means — matching how joint_state_2x2.py's
    region coupling is pooled over the whole dataset."""
    loaded = _load_kept(gpath, hpath, edges, min_gpc, min_cpg)
    if loaded is None:
        return None
    chrom, p1, p2, _, a1, a2, m1, m2, ga1, ga2, hm1, hm2, _ = loaded
    ec = np.concatenate([chrom, chrom])
    erb = np.concatenate([p1, p2]) // B
    ea = np.concatenate([a1, a2]); em = np.concatenate([m1, m2])
    eao = np.concatenate([ga1, ga2]); emo = np.concatenate([hm1, hm2])
    return pd.DataFrame({"chrom": ec, "rbin": erb,
                         "sa": np.where(eao, ea, 0.0), "na": eao.astype(np.int64),
                         "sm": np.where(emo, em, 0.0), "nm": emo.astype(np.int64)}
                        ).groupby(["chrom", "rbin"], as_index=False).sum()


def _build_global_means(sum_dfs, B):
    """Aggregate the per-cell region sums into a pooled, lookup-ready table:
    (chroms tuple, MULT, sorted int keys, mean_a, mean_m).  ``mean_*`` is NaN
    where the region had no observed end of that modality (→ masked out)."""
    g = pd.concat(sum_dfs, ignore_index=True).groupby(
        ["chrom", "rbin"], as_index=False).sum()
    na = g["na"].to_numpy(np.float64); nm = g["nm"].to_numpy(np.float64)
    mean_a = np.where(na > 0, g["sa"].to_numpy() / np.where(na > 0, na, 1.0), np.nan)
    mean_m = np.where(nm > 0, g["sm"].to_numpy() / np.where(nm > 0, nm, 1.0), np.nan)
    chroms = tuple(sorted(g["chrom"].unique()))
    cmap = {c: i for i, c in enumerate(chroms)}
    cc = g["chrom"].map(cmap).to_numpy(np.int64)
    rb = g["rbin"].to_numpy(np.int64)
    mult = int(rb.max()) + 1 if rb.size else 1
    key = cc * mult + rb
    o = np.argsort(key)
    return (chroms, mult, key[o], mean_a[o], mean_m[o])


def _region_lookup(chrom, pos, gmeans, B):
    """Each (chrom, pos//B) end → its pooled region (mean_a, mean_m), NaN if the
    region is absent.  Vectorized: map chrom→code, searchsorted on the keys."""
    chroms, mult, keys, ma, mm = gmeans
    cc = np.full(chrom.shape, -1, dtype=np.int64)
    for i, c in enumerate(chroms):
        cc[chrom == c] = i
    k = cc * mult + (pos // B)
    idx = np.clip(np.searchsorted(keys, k), 0, max(len(keys) - 1, 0))
    hit = (cc >= 0) & (keys[idx] == k) if keys.size else np.zeros(k.shape, bool)
    return np.where(hit, ma[idx], np.nan), np.where(hit, mm[idx], np.nan)


def process_cell(gpath, hpath, edges, min_gpc, min_cpg, seed, cell_idx,
                 region_bin=0, gmeans=None):
    """PASS 2 — one cell → RAW per-bin partial sums (6, nb) per (measure,
    real/shuf), so the caller can pool ACROSS cells (region mode) or centre
    within each cell (default).  In region mode each end's a/m is replaced by
    its POOLED region mean (``gmeans``) before correlating."""
    loaded = _load_kept(gpath, hpath, edges, min_gpc, min_cpg)
    if loaded is None:
        return None
    chrom, p1, p2, bins, a1, a2, m1, m2, ga1, ga2, hm1, hm2, nb = loaded

    if gmeans is not None:                              # substitute pooled region means
        a1, m1 = _region_lookup(chrom, p1, gmeans, region_bin)
        a2, m2 = _region_lookup(chrom, p2, gmeans, region_bin)
        ga1, hm1 = np.isfinite(a1), np.isfinite(m1)
        ga2, hm2 = np.isfinite(a2), np.isfinite(m2)
        a1 = np.nan_to_num(a1); a2 = np.nan_to_num(a2)
        m1 = np.nan_to_num(m1); m2 = np.nan_to_num(m2)

    def measures(a2_, m2_, ga2_, hm2_):
        return {
            "gg": _sums(bins, a1, a2_, ga1 & ga2_, nb),
            "hh": _sums(bins, m1, m2_, hm1 & hm2_, nb),
            # cross pools both end orderings so accessibility is always x, meth y
            "gh": _sums(bins, a1, m2_, ga1 & hm2_, nb)
                  + _sums(bins, a2_, m1, ga2_ & hm1, nb),
        }

    real = measures(a2, m2, ga2, hm2)
    rng = np.random.default_rng(int(seed) + int(cell_idx) * 7919)
    p = _within_bin_partner(bins, rng)
    shuf = measures(a2[p], m2[p], ga2[p], hm2[p])
    # 1-D autocorrelation null over the same loci, paired by genomic distance
    auto = _autocorr_sums(np.concatenate([chrom, chrom]),
                          np.concatenate([p1, p2]),
                          np.concatenate([a1, a2]), np.concatenate([m1, m2]),
                          np.concatenate([ga1, ga2]), np.concatenate([hm1, hm2]),
                          edges, rng)
    return {"real": real, "shuf": shuf, "auto": auto}


def _corr(acc, min_pairs):
    """(4, nb) summed partial sums → (r per bin, n per bin), NaN where n<min."""
    cov, vx, vy, n = acc
    denom = np.sqrt(vx * vy)
    ok = (denom > 0) & (n >= min_pairs)
    r = np.where(ok, cov / np.where(denom > 0, denom, 1.0), np.nan)
    return r, n


def _r_scalar(a4):
    """Length-4 partial sums (cov, var_x, var_y, n) → (r, n) for one pooled band."""
    cov, vx, vy, n = a4
    d = np.sqrt(vx * vy)
    return (cov / d if d > 0 else np.nan), n


def _human_dist(x):
    x = int(round(x))
    if x >= 1_000_000:
        return f"{x // 1_000_000}Mb" if x % 1_000_000 == 0 else f"{x / 1e6:.1f}Mb"
    if x >= 1_000:
        return f"{x // 1_000}kb" if x % 1_000 == 0 else f"{x / 1e3:.1f}kb"
    return f"{x}bp"


def _band_edges(min_d, max_d):
    """Decade-boundary band edges spanning [min_d, max_d] for the summary."""
    inner = [d for d in (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
             if min_d < d < max_d]
    return [min_d, *inner, max_d]


def _pool_band(acc, centers, lo, hi, last):
    """Sum a (4, nb) array's columns whose bin centre falls in [lo, hi)
    (inclusive of hi for the last band) → length-4 pooled partial sums."""
    sel = (centers >= lo) & ((centers <= hi) if last else (centers < hi))
    return acc[:, sel].sum(axis=1)


def _build_edges(min_d, max_d, per_decade):
    lo, hi = np.log10(max(min_d, 1)), np.log10(max_d)
    n_edges = max(2, int(round((hi - lo) * per_decade)) + 1)
    edges = np.logspace(lo, hi, n_edges)
    edges[0], edges[-1] = float(min_d), float(max_d)   # exact bounds (FP-safe)
    return edges


def _logx(ax, lo, hi):
    ax.set_xscale("log")
    ticks = [t for t in _TLAB if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_TLAB[t] for t in ticks])
    ax.set_xlabel("distance between ends")


def _plot(centers, edges, curves, n_cells, min_gpc, min_cpg, out, region_bin=0,
          pooled=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = edges[0], edges[-1]
    ylab = ("pooled" if pooled else "within-cell") + " Pearson r"
    colors = {"gg": "#1f77b4", "hh": "#d62728", "gh": "#2ca02c"}
    # (kind, linestyle, marker, line-alpha, band-alpha, label)
    kinds = [("real", "-", "o", 1.0, 0.22, "contact"),
             ("auto", "--", "", 0.9, 0.16, "1D autocorrelation"),
             ("shuf", ":", "", 0.7, 0.10, "shuffle (dist-matched)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    panels = dict(zip(("gg", "hh", "gh"), axes.flat[:3]))

    def draw(ax, key, kind, ls, mk, al, ba, lbl):
        c = colors[key]
        r, n, clo, chi = curves[key][kind]
        if np.isfinite(clo).any():
            ax.fill_between(centers, clo, chi, color=c, alpha=ba, lw=0)
        ax.plot(centers, r, ls, marker=mk, color=c, ms=3.2, lw=1.5, alpha=al,
                label=lbl)

    for key, label in _MEASURES:
        ax = panels[key]
        ax.axhline(0, color="0.6", lw=0.8, zorder=0)
        for kind, ls, mk, al, ba, lbl in kinds:
            draw(ax, key, kind, ls, mk, al, ba, lbl)
        _logx(ax, lo, hi)
        ax.set_ylabel(ylab)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8, loc="best")
    # 4th panel: overlay the three contact curves (with CI) + pair-count context
    ax = axes.flat[3]
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    for key, label in _MEASURES:
        draw(ax, key, "real", "-", "o", 1.0, 0.18, label.split("  ")[0])
    _logx(ax, lo, hi); ax.set_ylabel(ylab)
    ax.set_title("contact curves overlaid", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    axn = ax.twinx()
    axn.plot(centers, curves["gg"]["real"][1], color="0.5", lw=0.9, alpha=0.5)
    axn.set_yscale("log"); axn.set_ylabel("GCH–GCH pair count", color="0.5",
                                          fontsize=8)
    axn.tick_params(axis="y", colors="0.5", labelsize=7)
    lvl = ("pooled across cells" if pooled else "within-cell") + " Pearson r"
    reg = (f"  |  region-avg {region_bin:,} bp" if region_bin > 0 else "")
    fig.suptitle(
        f"Distance–coordination: contact vs 1D-autocorrelation vs shuffle  "
        f"(95% CI shaded)  |  {n_cells} cells  |  GpC≥{min_gpc}, CpG≥{min_cpg} "
        f"sites/end  |  {lvl}{reg}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


def _write_tsv(centers, edges, curves, path):
    rows = {"bin_lo": edges[:-1].astype(np.int64),
            "bin_hi": edges[1:].astype(np.int64),
            "center_bp": np.round(centers).astype(np.int64)}
    names = {"real": "contact", "auto": "autocorr", "shuf": "shuffle"}
    for key, _ in _MEASURES:
        for kind, nm in names.items():
            r, n, clo, chi = curves[key][kind]
            rows[f"{key}_{nm}_r"] = r
            rows[f"{key}_{nm}_lo"] = clo
            rows[f"{key}_{nm}_hi"] = chi
            rows[f"{key}_{nm}_n"] = n.astype(np.int64)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, na_rep="nan")


def _print_band_summary(raw, curves, centers, min_d, max_d, eps=0.05):
    """Summary stats log: pool the per-bin concordance into decade distance
    bands (partial sums summed across the band's bins, then a single band-level
    r), and a per-measure headline.  The baseline is now the 1-D autocorrelation,
    so ``Δ = contact - autocorr`` is the 3-D-contact-specific excess (beyond 1-D
    proximity)."""
    edges = _band_edges(min_d, max_d)
    nbd = len(edges) - 1
    names = [lab.split("  ")[0] for _, lab in _MEASURES]
    print("\n=== concordance summary by distance band "
          "(contact r | 1D-autocorr r | Δ=contact-autocorr) ===")
    print(f"  {'band':<14}{'n(GG)':>9}  "
          + "".join(f"{nm + ' c|a|Δ':>22}" for nm in names))
    for bi in range(nbd):
        lo, hi, last = edges[bi], edges[bi + 1], bi == nbd - 1
        label = f"{_human_dist(lo)}-{_human_dist(hi)}"
        cols, n_gg = "", 0
        for key, _ in _MEASURES:
            rc, n = _r_scalar(_pool_band(raw[key]["real"], centers, lo, hi, last))
            ra, _ = _r_scalar(_pool_band(raw[key]["auto"], centers, lo, hi, last))
            if key == "gg":
                n_gg = int(n)
            ex = rc - ra if (np.isfinite(rc) and np.isfinite(ra)) else np.nan
            cols += (f"{_sgn(rc)}|{_sgn(ra)}|{_sgn(ex)}").rjust(22)
        print(f"  {label:<14}{n_gg:>9,d}  {cols}")

    print("  headline (contact vs 1D-autocorr baseline):")
    for key, lab in _MEASURES:
        nm = lab.split("  ")[0]
        rc = curves[key]["real"][0]; ra = curves[key]["auto"][0]
        ex = rc - ra
        fin = np.isfinite(rc)
        if not fin.any():
            print(f"    {nm:<8}: n/a (no bin above --min-pairs)")
            continue
        idx = int(np.nanargmax(np.where(fin, np.abs(rc), np.nan)))
        sig = fin & np.isfinite(ex) & (np.abs(ex) >= eps)
        reach = _human_dist(centers[sig].max()) if sig.any() else "none"
        lr, _ = _r_scalar(_pool_band(raw[key]["real"], centers,
                                     edges[-2], edges[-1], True))
        print(f"    {nm:<8}: peak {rc[idx]:+.3f} @ {_human_dist(centers[idx])}"
              f"  ·  3D-specific |contact-autocorr|>={eps} out to ~{reach}"
              f"  ·  {_human_dist(edges[-2])}-{_human_dist(edges[-1])} pooled "
              f"{lr:+.3f}")


def _sgn(x):
    return f"{x:+.2f}" if np.isfinite(x) else "  n/a"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gch-dir", required=True, help="context_GCH/04_posteriors")
    ap.add_argument("--hcg-dir", required=True, help="context_HCG/04_posteriors")
    ap.add_argument("--out", default="joint_distance_coordination.pdf",
                    help="Output PDF (a sibling .tsv of per-bin values is written too).")
    ap.add_argument("--min-distance", type=int, default=1,
                    help="Finest distance (bp); 1 = down to within-read site pairs.")
    ap.add_argument("--max-distance", type=int, default=1_000_000)
    ap.add_argument("--bins-per-decade", type=int, default=8,
                    help="Log-spaced distance bins per decade.")
    ap.add_argument("--min-gpc-sites", type=int, default=2)
    ap.add_argument("--min-cpg-sites", type=int, default=2)
    ap.add_argument("--min-pairs", type=int, default=100,
                    help="Drop a bin whose pooled informative-pair count is below this.")
    ap.add_argument("--region-bin-size", type=int, default=0,
                    help="If >0, average each end's a/m over its (cell, chrom, "
                         "pos//SIZE) genomic bin BEFORE correlating — denoises "
                         "the per-end posteriors to surface the region/"
                         "compartment-scale coupling (e.g. the strong GCH–HCG "
                         "anti-correlation). 0 = raw single-end posteriors.")
    ap.add_argument("--cell-centering", choices=["pooled", "within-cell"],
                    default="pooled",
                    help="'pooled' (default): correlate ALL linked pairs from "
                         "every cell together per distance bin — the shuffle "
                         "then traces the cell-type baseline and real-minus-"
                         "shuffle is the contact-specific excess. 'within-cell': "
                         "centre each cell first, removing cell-type covariation "
                         "from the contact curve. (Region mode is always pooled.)")
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--n-bootstrap", type=int, default=200,
                    help="Cell-resampling bootstrap iterations for the 95%% CI "
                         "shaded bands (0 = no CI).")
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.min_distance < 1 or args.max_distance <= args.min_distance:
        sys.exit("require 1 <= --min-distance < --max-distance")

    edges = _build_edges(args.min_distance, args.max_distance, args.bins_per_decade)
    nb = len(edges) - 1
    centers = np.sqrt(edges[:-1] * edges[1:])

    gch = _cells(Path(args.gch_dir)); hcg = _cells(Path(args.hcg_dir))
    shared = sorted(set(gch) & set(hcg))
    if not shared:
        sys.exit("No cells shared between --gch-dir and --hcg-dir.")
    if args.max_cells and args.max_cells > 0:
        shared = shared[: args.max_cells]
    reg = (f" | region-avg {args.region_bin_size:,} bp"
           if args.region_bin_size > 0 else "")
    print(f"[dist-coord] {len(shared)} shared cells | {nb} bins "
          f"{args.min_distance:,}–{args.max_distance:,} bp "
          f"({args.bins_per_decade}/decade) | min sites GpC>={args.min_gpc_sites} "
          f"CpG>={args.min_cpg_sites}{reg}")

    region = args.region_bin_size > 0
    par = None
    if args.n_threads > 1 and len(shared) > 1:
        from joblib import Parallel, delayed
        par = Parallel(n_jobs=args.n_threads, verbose=5)

    gmeans = None
    if region:                                        # PASS 1: pool region means over ALL cells
        sjobs = [(gch[c], hcg[c], edges, args.min_gpc_sites, args.min_cpg_sites,
                  args.region_bin_size) for c in shared]
        sums = (par(delayed(cell_region_sums)(*j) for j in sjobs) if par
                else [cell_region_sums(*j) for j in sjobs])
        sums = [s for s in sums if s is not None]
        if not sums:
            sys.exit("No ends in range to pool region means.")
        gmeans = _build_global_means(sums, args.region_bin_size)
        print(f"  region means: {len(gmeans[2]):,} genomic bins of "
              f"{args.region_bin_size:,} bp pooled across all cells (pass 1)")

    jobs = [(gch[c], hcg[c], edges, args.min_gpc_sites, args.min_cpg_sites,
             args.seed, i, args.region_bin_size, gmeans)
            for i, c in enumerate(shared)]
    res = (par(delayed(process_cell)(*j) for j in jobs) if par
           else [process_cell(*j) for j in jobs])
    res = [r for r in res if r is not None]
    if not res:
        sys.exit("No qualifying pairs found in the distance range.")

    # 'pooled' sums the sufficient stats across ALL cells then centres ONCE, so
    # every linked pair (both ends >= min sites) enters one correlation per
    # distance bin (the within-cell-partner shuffle then traces the cell-type
    # baseline; real-minus-shuffle is the contact-specific excess).  Region mode
    # is always pooled.  'within-cell' instead centres each cell first, removing
    # cell-type covariation from the contact curve.
    pooled = region or args.cell_centering == "pooled"
    raw, curves = {}, {}
    for key, _ in _MEASURES:
        raw[key], curves[key] = {}, {}
        for kind in ("real", "shuf", "auto"):
            sums = [r[kind][key] for r in res]         # (6, nb) raw per cell
            acc = _center(sum(sums)) if pooled else sum(_center(s) for s in sums)
            raw[key][kind] = acc                       # (4, nb) cov/var/var/n
            r_, n_ = _corr(acc, args.min_pairs)
            lo, hi = _bootstrap_ci(sums, pooled, args.min_pairs,
                                   args.n_bootstrap, args.seed)
            curves[key][kind] = (r_, n_, lo, hi)        # r, n, CI lo, CI hi

    # console summary at the decade marks
    lvl = "pooled across cells" if pooled else "within-cell"
    print(f"\n=== {lvl} Pearson r (contact / 1D-autocorr / shuffle), "
          f"{len(res)} cells ===")
    print(f"  {'distance':>10}{'n_pairs':>11} | "
          + "".join(f"{lab.split('  ')[0]:>22}" for _, lab in _MEASURES))
    for bi in range(nb):
        if not np.isfinite(curves['gg']['real'][0][bi]):
            continue
        cells_txt = ""
        for key, _ in _MEASURES:
            rc = curves[key]["real"][0][bi]; ra = curves[key]["auto"][0][bi]
            rs = curves[key]["shuf"][0][bi]
            cells_txt += f"{rc:+.2f}/{ra:+.2f}/{rs:+.2f}  "
        npairs = int(curves["gg"]["real"][1][bi])
        print(f"  {centers[bi]:>10,.0f}{npairs:>11,d} | {cells_txt}")

    if args.region_bin_size > 0:
        print(f"\n  (region-avg {args.region_bin_size:,} bp, pooled across all "
              f"cells: same-modality GCH–GCH / HCG–HCG ≈ 1 below the bin size — "
              f"both ends share the region mean; read the GCH–HCG cross curve "
              f"for the denoised region/compartment-scale a–m coupling, the "
              f"distance-resolved analogue of joint_state_2x2.py's r(region).)")

    _print_band_summary(raw, curves, centers, args.min_distance,
                        args.max_distance)

    out = Path(args.out)
    _plot(centers, edges, curves, len(res), args.min_gpc_sites,
          args.min_cpg_sites, out, args.region_bin_size, pooled)
    tsv = out.with_suffix(".tsv")
    _write_tsv(centers, edges, curves, tsv)
    print(f"\nwrote {out}\n      {tsv}")
    print("\nRead the figure (95% CI shaded): the 1D-AUTOCORRELATION line is the "
          "co-\n  methylation / co-accessibility expected from 1-D genomic "
          "proximity ALONE\n  (same loci, paired by distance not contact) — so it "
          "is naturally >0 at\n  short range. CONTACT sitting ABOVE the "
          "autocorrelation (Δ in the band\n  summary) is 3-D-contact-specific "
          "coordination; the shuffle is the\n  marginal-independence floor.")


if __name__ == "__main__":
    main()
