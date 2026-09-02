#!/usr/bin/env python3
"""Signed-distance density of bin sets to a domain boundary.

For every bin, the distance to the NEAREST edge of the domain it is closest
to, signed so that

    NEGATIVE = inside the domain,  POSITIVE = outside

and plotted as a density per bin set, with the background drawn behind them.
(Note the sign is the opposite of scripts/boundary_profile.py, which makes
inside positive; this script follows the "outside is positive" convention
asked for at the boundary of late replication timing.)

WHY THE BACKGROUND MATTERS.  Domains have a length distribution and bins are
not uniformly placed, so the distance density of ANY bin set is non-flat.
A set that piles up near a boundary can only be read against the same
geometry applied to the universe it was drawn from -- which is what
--background supplies, and why it is required rather than optional.

LATE-RT DOMAINS COME FROM THE TWO REPLISEQ TRACKS, not from a BED: late
replication is where the late signal exceeds the early one.  Pass the pair
with --rt-early / --rt-late and the domains are derived, merged and
width-filtered here, so the boundary is defined by the data rather than by
somebody's threshold in a supplementary file.  LAD domains are already
intervals and go in as --domain.

    python scripts/bin_set_boundary_distance.py \\
        --background GCH=.../bin_scores_gch.all.bed \\
        --bin-set GCH:variable=.../bin_scores_gch.selected.bed \\
        --bin-set GCH:invariant=.../bin_scores_gch.stable.bed \\
        --rt-early NPC=.../repliseq.early.mm10.bw \\
        --rt-late  NPC=.../repliseq.late.mm10.bw \\
        --domain cLAD=annots/cLAD_mm10.bed \\
        --output-dir out/boundary_distance
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _boundary_mod():
    """Reuse boundary_profile's BED reader and signed-distance geometry."""
    p = Path(__file__).resolve().parent / "boundary_profile.py"
    spec = importlib.util.spec_from_file_location("_bp", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_bp"] = m
    spec.loader.exec_module(m)
    return m


def _named(spec, flag, sep="="):
    if sep not in spec:
        raise SystemExit(f"{flag} expects NAME{sep}PATH, got {spec!r}")
    name, path = spec.split(sep, 1)
    if not Path(path).exists():
        raise SystemExit(f"{flag} {name}: {path} not found")
    return name, path


def late_rt_domains(early_path, late_path, bin_size=50_000,
                    min_width=200_000, chroms=None):
    """Intervals where the LATE repliseq signal exceeds the EARLY one.

    Binned means rather than raw values: repliseq tracks are noisy at base
    resolution and a per-base comparison would shatter every domain into
    thousands of fragments, which then dominate the boundary set.  Adjacent
    late bins are merged and anything under ``min_width`` is dropped for the
    same reason -- a 50 kb speck has two edges and would contribute distances
    as readily as a 5 Mb domain.
    """
    try:
        import pyBigWig
    except ImportError:
        raise SystemExit(
            "reading bigWig needs pyBigWig: pip install 'pyBigWig>=0.3.18' "
            "(or install this package with the [bigwig] extra)")
    rows = []
    with pyBigWig.open(str(early_path)) as be, \
            pyBigWig.open(str(late_path)) as bl:
        common = set(be.chroms()) & set(bl.chroms())
        if chroms:
            common &= set(chroms)
        if not common:
            raise SystemExit(
                f"no shared contigs between {Path(early_path).name} and "
                f"{Path(late_path).name}"
                + (f" within {sorted(chroms)}" if chroms else ""))
        for c in sorted(common):
            n = min(be.chroms()[c], bl.chroms()[c])
            nb = max(1, n // bin_size)
            e = np.array(be.stats(c, 0, nb * bin_size, nBins=nb, type="mean"),
                         dtype=float)
            l = np.array(bl.stats(c, 0, nb * bin_size, nBins=nb, type="mean"),
                         dtype=float)
            late = np.where(np.isfinite(e) & np.isfinite(l), l > e, False)
            if not late.any():
                continue
            # merge runs of consecutive late bins
            idx = np.flatnonzero(late)
            brk = np.flatnonzero(np.diff(idx) > 1)
            for lo, hi in zip(np.r_[0, brk + 1], np.r_[brk, len(idx) - 1]):
                s, ee = idx[lo] * bin_size, (idx[hi] + 1) * bin_size
                if ee - s >= min_width:
                    rows.append((c, s, ee))
    if not rows:
        raise SystemExit("no late-RT domain survived the width filter; "
                         "lower --rt-min-width or --rt-bin-size")
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def density_figure(dist, path, title, xlim, bw, plot_format):
    """One panel per domain; a line per bin set, background behind."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    domains = sorted({d for d, _ in dist})
    fig, axes = plt.subplots(1, len(domains), squeeze=False,
                             figsize=(4.6 * len(domains) + 0.6, 3.6))
    grid = np.linspace(-xlim, xlim, 512)
    for ax, dom in zip(axes[0], domains):
        for label in sorted({s for d, s in dist if d == dom}):
            v = dist[(dom, label)]
            v = v[np.isfinite(v)]
            n_all = int(v.size)
            v = v[np.abs(v) <= xlim]
            if v.size < 5:
                continue
            # Say when bins fell outside the axis.  The legend count is what
            # is DRAWN, and quoting it as if it were the set size would
            # understate every set differently, by however much of it lies
            # beyond --xlim.
            shown = (f"n={v.size:,}" if v.size == n_all
                     else f"n={v.size:,} of {n_all:,} within axis")
            is_bg = label.endswith("(background)")
            try:
                y = gaussian_kde(v, bw_method=bw)(grid)
            except Exception:      # a degenerate set has no bandwidth
                y, edges = np.histogram(v, bins=80, range=(-xlim, xlim),
                                        density=True)
                grid_h = 0.5 * (edges[1:] + edges[:-1])
                ax.plot(grid_h, y, lw=1.0, label=f"{label} ({shown})")
                continue
            ax.plot(grid, y, lw=2.0 if is_bg else 1.4,
                    color="0.55" if is_bg else None,
                    zorder=1 if is_bg else 2,
                    label=f"{label} ({shown})")
        ax.axvline(0, color="0.3", lw=0.9, ls="--")
        ax.set_title(dom, fontsize=10, loc="left")
        ax.set_xlabel("signed distance to nearest boundary (bp)\n"
                      "negative = inside, positive = outside", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--background", action="append", required=True,
                    metavar="NAME=BED",
                    help="Universe per context, e.g. GCH=bin_scores_gch.all.bed."
                         "  Required: the distance density of any bin set is "
                         "non-flat, so a set can only be read against the "
                         "same geometry applied to the bins it was drawn "
                         "from.")
    ap.add_argument("--bin-set", action="append", required=True,
                    metavar="NAME:LABEL=BED",
                    help="Repeatable, e.g. GCH:variable=...selected.bed.  "
                         "NAME must match a --background.")
    ap.add_argument("--domain", action="append", default=None,
                    metavar="NAME=BED", help="Interval domains, e.g. cLAD.")
    ap.add_argument("--rt-early", action="append", default=None,
                    metavar="NAME=BW")
    ap.add_argument("--rt-late", action="append", default=None,
                    metavar="NAME=BW",
                    help="Paired with --rt-early by NAME; late-RT domains are "
                         "derived as late > early.")
    ap.add_argument("--rt-bin-size", type=int, default=50_000)
    ap.add_argument("--rt-min-width", type=int, default=200_000)
    ap.add_argument("--xlim", type=int, default=5_000_000,
                    help="Distance axis limit in bp (5 Mb).")
    ap.add_argument("--kde-bw", type=float, default=None,
                    help="gaussian_kde bandwidth; default is Scott's rule.")
    ap.add_argument("--plot-format", default="pdf",
                    choices=["pdf", "png", "svg"])
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args(argv)

    bp = _boundary_mod()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    backgrounds = dict(_named(s, "--background") for s in a.background)
    sets = []
    for spec in a.bin_set:
        head, path = _named(spec, "--bin-set")
        if ":" not in head:
            raise SystemExit(f"--bin-set expects NAME:LABEL=PATH, got {spec!r}")
        name, label = head.split(":", 1)
        if name not in backgrounds:
            raise SystemExit(f"--bin-set {name}:{label} has no matching "
                             f"--background {name}= "
                             f"(have {sorted(backgrounds)})")
        sets.append((name, label, path))

    domains = {}
    for spec in (a.domain or []):
        name, path = _named(spec, "--domain")
        domains[name] = bp.read_bed(path, merge=True)
    early = dict(_named(s, "--rt-early") for s in (a.rt_early or []))
    late = dict(_named(s, "--rt-late") for s in (a.rt_late or []))
    for name in sorted(set(early) | set(late)):
        if name not in early or name not in late:
            raise SystemExit(f"--rt-early/--rt-late must be paired by NAME; "
                             f"{name!r} has only one of them")
        d = late_rt_domains(early[name], late[name], a.rt_bin_size,
                            a.rt_min_width)
        domains[f"{name} late-RT"] = d
        print(f"{name} late-RT: {len(d):,} domains, median width "
              f"{int((d.end - d.start).median()):,} bp")
    if not domains:
        raise SystemExit("nothing to measure against; pass --domain and/or "
                         "--rt-early/--rt-late")

    dist, rows = {}, []
    for dom_name, dom in domains.items():
        for ctx, path in backgrounds.items():
            beds = [(f"{ctx} (background)", path)] + [
                (f"{ctx}:{lab}", p) for c, lab, p in sets if c == ctx]
            for label, p in beds:
                b = bp.read_bed(p)
                # boundary_profile makes INSIDE positive; this script's
                # convention is the opposite, so negate once here.
                v = -bp.signed_distance(b, dom)
                dist[(dom_name, label)] = v
                fin = v[np.isfinite(v)]
                rows.append({
                    "domain": dom_name, "set": label, "n_bins": int(len(v)),
                    "n_with_distance": int(fin.size),
                    "frac_inside": float(np.mean(fin < 0)) if fin.size else np.nan,
                    "median_distance": float(np.median(fin)) if fin.size else np.nan,
                    "q25": float(np.percentile(fin, 25)) if fin.size else np.nan,
                    "q75": float(np.percentile(fin, 75)) if fin.size else np.nan,
                })
    tab = pd.DataFrame(rows)
    tsv = out / "boundary_distance.tsv"
    tab.to_csv(tsv, sep="\t", index=False)
    fig = out / f"boundary_distance.{a.plot_format}"
    density_figure(dist, fig, "Distance to domain boundary", a.xlim,
                   a.kde_bw, a.plot_format)
    print(tab.to_string(index=False))
    print(f"\nwrote {tsv}\n      {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
