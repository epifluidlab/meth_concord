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
        return np.array([], int), np.array([], int), total
    bi, pi = [], []
    step_f = flank_bp / float(n_flank) if n_flank else 1.0
    by_chrom = {}
    for c, sub in bins.groupby("chrom"):
        by_chrom[str(c)] = (sub.index.to_numpy(),
                            sub["start"].to_numpy(float),
                            sub["end"].to_numpy(float))
    for _, d in domains.iterrows():
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
    if not bi:
        return np.array([], int), np.array([], int), total
    return np.concatenate(bi), np.concatenate(pi), total


def group_profile(M, ct, members, bin_idx, pos_idx, total, conf=0.95):
    """Mean and interval per axis position for one group of cell types.

    The per-cell value at a position is the mean over every bin placed there,
    so a domain contributing many bins to a position does not thereby count
    as many observations: the observation is the CELL.
    """
    from scipy.stats import t as _t

    cells = ct.index[ct.isin(members)]
    rows = M.loc[M.index.intersection(cells)]
    out = {"mean": np.full(total, np.nan), "lo": np.full(total, np.nan),
           "hi": np.full(total, np.nan), "n_bins": np.zeros(total, int),
           "n_cells": int(len(rows)),
           "members": {g: int((ct.loc[rows.index] == g).sum())
                       for g in members}}
    if not len(rows):
        return out
    v = rows.to_numpy(float)
    order = np.argsort(pos_idx, kind="stable")
    bi, pi = bin_idx[order], pos_idx[order]
    starts = np.searchsorted(pi, np.arange(total), side="left")
    ends = np.searchsorted(pi, np.arange(total), side="right")
    for i in range(total):
        cols = bi[starts[i]:ends[i]]
        out["n_bins"][i] = cols.size
        if cols.size == 0:
            continue
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            per_cell = np.nanmean(v[:, cols], axis=1)
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


def member_curves(M, ct, members, bin_idx, pos_idx, total):
    """Mean per axis position for each constituent cell type, no interval.

    Drawn behind the group line so a group mean carried by ONE of its
    members is visible.  With two tumours that is the difference between a
    tumour finding and a tumour-1 finding, and no interval across cells can
    show it.
    """
    return {g: group_profile(M, ct, [g], bin_idx, pos_idx, total)["mean"]
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
    ap.add_argument("--matrix", required=True, metavar="NAME=PATH",
                    help="Concordance matrix (cells x bins) parquet.")
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
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--ylabel", default="GCH-GCH concordance (D)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--plot-format", default="pdf",
                    choices=["pdf", "png", "svg"])
    args = ap.parse_args(argv)

    bp = _bp_mod()
    name, mpath = (args.matrix.split("=", 1) if "=" in args.matrix
                   else ("gch", args.matrix))
    M = pd.read_parquet(mpath)
    samples = pd.read_csv(args.samples, sep="\t")
    if args.cell_type_column not in samples.columns:
        sys.exit(f"--cell-type-column {args.cell_type_column!r} not in "
                 f"--samples ({list(samples.columns)})")
    ct = samples.set_index("cell_name")[args.cell_type_column].astype(str)
    ct = ct.reindex(M.index).dropna()
    M = M.loc[ct.index]

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

    groups = {}
    for lab, spec in _named(args.group, "--group").items():
        mem = [g.strip() for g in spec.split(",") if g.strip()]
        bad = [g for g in mem if g not in set(ct)]
        if bad:
            sys.exit(f"--group {lab}: not in --samples "
                     f"{args.cell_type_column!r}: {bad}\n"
                     f"  have: {sorted(set(ct))}")
        groups[lab] = mem

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
    results, members, rows = {}, {}, []
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
        bi, pi, total = scaled_index(bins, dom, args.body_bins, args.flank,
                                     args.flank_bins)
        prof = {lab: group_profile(M, ct, mem, bi, pi, total, args.conf)
                for lab, mem in groups.items()}
        results[rname] = prof
        members[rname] = {lab: member_curves(M, ct, mem, bi, pi, total)
                          for lab, mem in groups.items() if len(mem) > 1}
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
    fig = out / f"scaled_profile_{name}.{args.plot_format}"
    scaled_figure(results, fig, args.title or f"{name}: mean D across scaled "
                  f"domains", args.ylabel, args.body_bins, args.flank_bins,
                  args.flank, colors, args.conf, members)
    print(f"  -> {fig.name}, scaled_profile_{name}.tsv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
