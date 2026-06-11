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

Each is a **within-cell** Pearson correlation: per (cell, distance-bin) the two
ends are centred by that cell/bin mean, then the centred cross-products are
pooled across cells — so a cell's global accessibility/methylation level can't
masquerade as coordination.  (This is the BETWEEN-end coordination that the
concordance method rides on; ``joint_state_2x2.py`` measures the orthogonal
WITHIN-end a–m coupling.)

For each curve a **distance-matched shuffle** null is drawn: within (cell,
distance-bin) the partner end is permuted, so every end keeps its own value and
distance class but is paired with an end from a DIFFERENT contact.  The gap
between a real curve and its shuffle is the contact-specific coordination —
agreement beyond what 1-D genomic distance alone implies.

Distance is binned log-spaced from --min-distance (default 1 bp, the finest
within-read scale) to --max-distance (default 1 Mb), --bins-per-decade per
decade.  Output: a multi-panel PDF and a sibling TSV of the per-bin values.

Usage
-----
    python scripts/joint_distance_coordination.py \\
        --gch-dir .../context_GCH/04_posteriors \\
        --hcg-dir .../context_HCG/04_posteriors \\
        --out joint_distance_coordination.pdf
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


def process_cell(gpath, hpath, edges, min_gpc, min_cpg, seed, cell_idx):
    """One cell → within-cell centred partial sums per (measure, real/shuf, bin).

    Returns ``{"real": {gg,hh,gh}, "shuf": {...}}`` where each value is a
    (4, nb) array (cov, var_x, var_y, n), or None if the cell has no joined
    pairs in the distance range."""
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

    def measures(a2_, m2_, ga2_, hm2_):
        return {
            "gg": _center(_sums(bins, a1, a2_, ga1 & ga2_, nb)),
            "hh": _center(_sums(bins, m1, m2_, hm1 & hm2_, nb)),
            # cross pools both end orderings so accessibility is always x, meth y
            "gh": _center(_sums(bins, a1, m2_, ga1 & hm2_, nb)
                          + _sums(bins, a2_, m1, ga2_ & hm1, nb)),
        }

    real = measures(a2, m2, ga2, hm2)
    rng = np.random.default_rng(int(seed) + int(cell_idx) * 7919)
    p = _within_bin_partner(bins, rng)
    shuf = measures(a2[p], m2[p], ga2[p], hm2[p])
    return {"real": real, "shuf": shuf}


def _corr(acc, min_pairs):
    """(4, nb) summed partial sums → (r per bin, n per bin), NaN where n<min."""
    cov, vx, vy, n = acc
    denom = np.sqrt(vx * vy)
    ok = (denom > 0) & (n >= min_pairs)
    r = np.where(ok, cov / np.where(denom > 0, denom, 1.0), np.nan)
    return r, n


def _build_edges(min_d, max_d, per_decade):
    lo, hi = np.log10(max(min_d, 1)), np.log10(max_d)
    n_edges = max(2, int(round((hi - lo) * per_decade)) + 1)
    return np.logspace(lo, hi, n_edges)


def _logx(ax, lo, hi):
    ax.set_xscale("log")
    ticks = [t for t in _TLAB if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_TLAB[t] for t in ticks])
    ax.set_xlabel("distance between ends")


def _plot(centers, edges, curves, n_cells, min_gpc, min_cpg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = edges[0], edges[-1]
    colors = {"gg": "#1f77b4", "hh": "#d62728", "gh": "#2ca02c"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    panels = dict(zip(("gg", "hh", "gh"), axes.flat[:3]))
    for key, label in _MEASURES:
        ax = panels[key]; c = colors[key]
        r_real = curves[key]["real"][0]; r_shuf = curves[key]["shuf"][0]
        ax.axhline(0, color="0.6", lw=0.8, zorder=0)
        ax.plot(centers, r_real, "-o", color=c, ms=3.5, lw=1.6, label="contact")
        ax.plot(centers, r_shuf, "--", color=c, ms=3, lw=1.2, alpha=0.7,
                label="shuffle (dist-matched)")
        _logx(ax, lo, hi)
        ax.set_ylabel("within-cell Pearson r")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8, loc="best")
    # 4th panel: overlay the three contact curves + pair-count context
    ax = axes.flat[3]
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    for key, label in _MEASURES:
        ax.plot(centers, curves[key]["real"][0], "-o", ms=3, lw=1.5,
                color=colors[key], label=label.split("  ")[0])
    _logx(ax, lo, hi); ax.set_ylabel("within-cell Pearson r")
    ax.set_title("contact curves overlaid", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    axn = ax.twinx()
    axn.plot(centers, curves["gg"]["real"][1], color="0.5", lw=0.9, alpha=0.5)
    axn.set_yscale("log"); axn.set_ylabel("GCH–GCH pair count", color="0.5",
                                          fontsize=8)
    axn.tick_params(axis="y", colors="0.5", labelsize=7)
    fig.suptitle(
        f"Distance–coordination: contact vs distance-matched shuffle  |  "
        f"{n_cells} cells  |  GpC≥{min_gpc}, CpG≥{min_cpg} sites/end  |  "
        f"within-cell Pearson r", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


def _write_tsv(centers, edges, curves, path):
    rows = {"bin_lo": edges[:-1].astype(np.int64),
            "bin_hi": edges[1:].astype(np.int64),
            "center_bp": np.round(centers).astype(np.int64)}
    for key, _ in _MEASURES:
        rows[f"{key}_r_contact"] = curves[key]["real"][0]
        rows[f"{key}_r_shuffle"] = curves[key]["shuf"][0]
        rows[f"{key}_n"] = curves[key]["real"][1].astype(np.int64)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, na_rep="nan")


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
    ap.add_argument("--max-cells", type=int, default=0)
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
    print(f"[dist-coord] {len(shared)} shared cells | {nb} bins "
          f"{args.min_distance:,}–{args.max_distance:,} bp "
          f"({args.bins_per_decade}/decade) | min sites GpC>={args.min_gpc_sites} "
          f"CpG>={args.min_cpg_sites}")

    jobs = [(gch[c], hcg[c], edges, args.min_gpc_sites, args.min_cpg_sites,
             args.seed, i) for i, c in enumerate(shared)]
    if args.n_threads > 1 and len(jobs) > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=args.n_threads, verbose=5)(
            delayed(process_cell)(*j) for j in jobs)
    else:
        res = [process_cell(*j) for j in jobs]
    res = [r for r in res if r is not None]
    if not res:
        sys.exit("No qualifying pairs found in the distance range.")

    curves = {}
    for key, _ in _MEASURES:
        curves[key] = {}
        for kind in ("real", "shuf"):
            acc = np.zeros((4, nb))
            for r in res:
                acc += r[kind][key]
            curves[key][kind] = _corr(acc, args.min_pairs)

    # console summary at the decade marks
    print(f"\n=== within-cell Pearson r (contact | shuffle), {len(res)} cells ===")
    print(f"  {'distance':>10}{'n_pairs':>11} | "
          + "".join(f"{lab.split('  ')[0]:>20}" for _, lab in _MEASURES))
    for bi in range(nb):
        if not np.isfinite(curves['gg']['real'][0][bi]):
            continue
        d = centers[bi]
        cells_txt = ""
        for key, _ in _MEASURES:
            rr = curves[key]["real"][0][bi]; rs = curves[key]["shuf"][0][bi]
            cells_txt += f"{rr:+7.3f}|{rs:+6.3f}   "
        npairs = int(curves["gg"]["real"][1][bi])
        print(f"  {d:>10,.0f}{npairs:>11,d} | {cells_txt}")

    out = Path(args.out)
    _plot(centers, edges, curves, len(res), args.min_gpc_sites,
          args.min_cpg_sites, out)
    tsv = out.with_suffix(".tsv")
    _write_tsv(centers, edges, curves, tsv)
    print(f"\nwrote {out}\n      {tsv}")
    print("\nRead the figure: a POSITIVE GCH–GCH / HCG–HCG contact curve that "
          "decays\n  with distance and sits ABOVE its flat ~0 shuffle = "
          "contact-specific\n  state agreement; a NEGATIVE GCH–HCG curve at "
          "short range that relaxes\n  toward 0 = the canonical open↔unmethylated "
          "coupling is a local property.")


if __name__ == "__main__":
    main()
