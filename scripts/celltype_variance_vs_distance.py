#!/usr/bin/env python3
"""Where along the distance axis does cell-type information live?

The genome-wide concordance-vs-distance curve is an AVERAGE over millions of
loci, and an average is blind to cell-type structure: two cell types can share
one curve exactly while their per-region patterns are completely different.
That is why the GCH-GCH curves collapse onto a single line and the UMAP built
from the same data still separates cell types cleanly.

This closes that gap.  For each distance band it asks, per genomic region, how
much of the variation in concordance is BETWEEN cell types versus WITHIN them:

    ICC = sigma2_between / (sigma2_between + sigma2_within)

from a one-way ANOVA variance decomposition, so sampling noise sits in the
denominator instead of being mistaken for signal.  Plotted against distance,
ICC says where the cell-type information actually is -- which is what turns
"long-range coordination is cell-type-specific" from a claim into a measurement.

Because ICC rises with group size even under the null, the honest reference is
a PERMUTATION null: shuffle the cell -> cell-type labels and recompute.  The
shaded band is that null, so only excursions above it are real.

Input is the same cells x regions matrices that produce the UMAP, one per
distance band -- run nome_hic_concordance at several --min-distance/--max-distance
settings.  Any matrix can be passed, so a 1D baseline (mean methylation) can be
plotted alongside concordance to show concordance carries cell-type information
the baseline does not.

Usage
-----
    python scripts/celltype_variance_vs_distance.py \\
        --samples samples.tsv --group-column group \\
        --matrix gch:35000=out_20k_50k/context_GCH/05_concordance/concordance.parquet \\
        --matrix gch:100000=out_50k_200k/context_GCH/05_concordance/concordance.parquet \\
        --matrix hcg:35000=out_20k_50k/context_HCG/05_concordance/concordance.parquet \\
        --matrix gch_mean:35000=out_20k_50k/context_GCH/05_concordance/accessibility_mean.parquet \\
        --output-dir output/celltype_variance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_matrix_spec(spec):
    """``series:distance_bp=path`` -> (series, distance, path)."""
    if "=" not in spec:
        raise SystemExit(f"--matrix must be SERIES:DIST_BP=PATH, got {spec!r}")
    head, path = spec.split("=", 1)
    if ":" not in head:
        raise SystemExit(f"--matrix must be SERIES:DIST_BP=PATH, got {spec!r}")
    series, dist = head.rsplit(":", 1)
    try:
        dist = float(dist)
    except ValueError:
        raise SystemExit(f"distance must be a number in bp, got {dist!r}")
    return series.strip(), dist, path.strip()


def icc_per_region(X, group_ids, n_groups, min_cells_per_group=3,
                   min_groups=3):
    """One-way ANOVA ICC per column (region), NaN-aware.

    Returns (icc, usable_mask).  A region is usable only when enough groups
    carry enough cells; otherwise ICC is undefined rather than 0, because an
    unmeasurable region and a region with no cell-type structure are different
    statements.
    """
    X = np.asarray(X, dtype=np.float64)
    n_reg = X.shape[1]
    obs = np.isfinite(X)

    n_g = np.zeros((n_groups, n_reg))
    s_g = np.zeros((n_groups, n_reg))
    q_g = np.zeros((n_groups, n_reg))
    for g in range(n_groups):
        sel = group_ids == g
        if not sel.any():
            continue
        m = obs[sel]
        v = np.where(m, X[sel], 0.0)
        n_g[g] = m.sum(0)
        s_g[g] = v.sum(0)
        q_g[g] = (v * v).sum(0)

    enough = n_g >= min_cells_per_group
    n_g = np.where(enough, n_g, 0.0)
    s_g = np.where(enough, s_g, 0.0)
    q_g = np.where(enough, q_g, 0.0)

    k = enough.sum(0).astype(np.float64)              # groups per region
    N = n_g.sum(0)
    usable = (k >= min_groups) & (N > k)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_g = np.where(n_g > 0, s_g / np.where(n_g > 0, n_g, 1.0), 0.0)
        grand = np.where(N > 0, s_g.sum(0) / np.where(N > 0, N, 1.0), 0.0)
        ss_b = (n_g * (mean_g - grand) ** 2).sum(0)
        ss_w = (q_g - n_g * mean_g ** 2).sum(0)
        df_b = np.maximum(k - 1.0, 1.0)
        df_w = np.maximum(N - k, 1.0)
        ms_b, ms_w = ss_b / df_b, ss_w / df_w
        # n0: the effective group size, which corrects for unequal group sizes.
        n0 = (N - (n_g ** 2).sum(0) / np.where(N > 0, N, 1.0)) / df_b
        n0 = np.where(n0 > 0, n0, np.nan)
        var_b = np.maximum((ms_b - ms_w) / n0, 0.0)   # negative -> no structure
        icc = var_b / (var_b + ms_w)
        icc = np.where(np.isfinite(icc), icc, np.nan)
    return np.where(usable, icc, np.nan), usable


def _summarise(X, group_ids, n_groups, args, rng, n_perm):
    """Observed ICC and its permutation null for one matrix."""
    icc, usable = icc_per_region(X, group_ids, n_groups,
                                 args.min_cells_per_group, args.min_groups)
    if not np.isfinite(icc).any():
        return None
    obs = float(np.nanmedian(icc))

    null = []
    for _ in range(n_perm):
        perm = rng.permutation(group_ids)
        ni, _ = icc_per_region(X, perm, n_groups,
                               args.min_cells_per_group, args.min_groups)
        if np.isfinite(ni).any():
            null.append(float(np.nanmedian(ni)))
    null = np.asarray(null, float)
    if null.size:
        lo, hi = np.percentile(null, [2.5, 97.5])
        # (1 + count)/(1 + B): a finite permutation cannot justify p = 0
        p = (1.0 + int((null >= obs).sum())) / (1.0 + null.size)
    else:
        lo = hi = np.nan
        p = np.nan
    return {"icc": obs, "null_lo": float(lo), "null_hi": float(hi),
            "p": float(p), "n_regions": int(np.isfinite(icc).sum()),
            "n_regions_total": int(icc.size)}


def _plot(rows, out, n_perm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows).sort_values(["series", "distance"])
    series = list(dict.fromkeys(df["series"]))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    for i, s in enumerate(series):
        d = df[df["series"] == s]
        color = cmap(i % 10)
        ax.fill_between(d["distance"], d["null_lo"], d["null_hi"],
                        color=color, alpha=0.15, lw=0)
        ax.plot(d["distance"], d["icc"], "-o", ms=5, lw=1.8, color=color,
                label=s)
        sig = d[d["p"] < 0.05]
        if len(sig):
            ax.plot(sig["distance"], sig["icc"], "o", ms=9, mfc="none",
                    mec="0.15", mew=1.2, ls="none", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("genomic distance band (bp)")
    ax.set_ylabel("ICC — fraction of per-region variance that is\n"
                  "BETWEEN cell types")
    ax.set_title("Where cell-type information lives along the distance axis",
                 fontsize=11)
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    ax.legend(fontsize=9, title="matrix", title_fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.015,
             f"shaded = permutation null ({n_perm} shuffles of the cell-type "
             f"labels); rings mark p < 0.05.\nICC is a variance decomposition, "
             f"so sampling noise sits in the denominator rather than counting "
             f"as signal.",
             ha="center", fontsize=7.5, color="0.4")
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", action="append", required=True,
                    metavar="SERIES:DIST_BP=PATH",
                    help="cells x regions parquet for one series at one "
                         "distance band. Repeatable. The series name groups "
                         "points into one line, so pass a 1D baseline as its "
                         "own series to compare against concordance.")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--group-column", default="group")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-cells-per-group", type=int, default=3,
                    help="A group must have this many cells covering a region "
                         "for that region to count toward the ICC.")
    ap.add_argument("--min-groups", type=int, default=3,
                    help="Minimum groups covering a region.")
    ap.add_argument("--n-permutations", type=int, default=200,
                    help="Shuffles of the cell-type labels for the null. ICC "
                         "is positively biased with many groups, so the null "
                         "is the reference, not zero.")
    ap.add_argument("--plot-format", choices=["png", "pdf", "svg"],
                    default="pdf")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from nome_hic_concordance.cli import load_samples
    from nome_hic_concordance.utils import ensure_dir

    samples = load_samples(args.samples)
    if args.group_column not in samples.columns:
        sys.exit(f"--group-column {args.group_column!r} not in --samples "
                 f"(columns: {samples.columns.tolist()}).")
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    cell_group = dict(zip(samples["cell_name"].astype(str),
                          samples[args.group_column].astype(str)))
    rng = np.random.default_rng(args.seed)
    rows = []

    for spec in args.matrix:
        series, dist, path = _parse_matrix_spec(spec)
        if not Path(path).exists():
            sys.exit(f"no such matrix: {path}")
        M = pd.read_parquet(path)
        M.index = M.index.astype(str)
        keep = [c for c in M.index if c in cell_group]
        if len(keep) < args.min_cells_per_group * args.min_groups:
            print(f"  {series}@{dist:,.0f}: only {len(keep)} cells match "
                  f"--samples — skipped", file=sys.stderr)
            continue
        M = M.loc[keep]
        labels = np.array([cell_group[c] for c in M.index])
        uniq = sorted(set(labels))
        gid = np.searchsorted(np.array(uniq), labels)

        res = _summarise(M.to_numpy(), gid, len(uniq), args, rng,
                         args.n_permutations)
        if res is None:
            print(f"  {series}@{dist:,.0f}: no usable regions — skipped",
                  file=sys.stderr)
            continue
        res.update(series=series, distance=dist, path=path,
                   n_cells=len(keep), n_groups=len(uniq))
        rows.append(res)
        star = "*" if res["p"] < 0.05 else " "
        print(f"  {series:>12} @ {dist:>10,.0f} bp | ICC={res['icc']:.4f}{star} "
              f"null [{res['null_lo']:.4f}, {res['null_hi']:.4f}] "
              f"p={res['p']:.3f} | {res['n_regions']:,}/"
              f"{res['n_regions_total']:,} regions, {len(keep)} cells, "
              f"{len(uniq)} groups")

    if not rows:
        sys.exit("no matrix produced a usable ICC.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "celltype_variance_vs_distance.tsv", sep="\t",
              index=False)
    fig = out_dir / f"celltype_variance_vs_distance.{args.plot_format}"
    _plot(rows, fig, args.n_permutations)
    with open(out_dir / "celltype_variance_summary.json", "w") as fh:
        json.dump({"rows": rows, "n_permutations": args.n_permutations,
                   "group_column": args.group_column,
                   "statistic": "one-way ANOVA ICC of per-region values, "
                                "median over regions; null = label "
                                "permutation"}, fh, indent=2)
    print(f"\nwrote {fig}")
    print("Read it: ICC ABOVE the shaded null means cell types differ in "
          "their per-region\n  values at that distance by more than sampling "
          "noise. A concordance series\n  sitting above its 1D-baseline series "
          "is concordance carrying cell-type\n  information the baseline does "
          "not.")


if __name__ == "__main__":
    main()
