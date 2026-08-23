#!/usr/bin/env python3
"""Mechanistic scheme: where long-range epigenetic coordination goes in tumour.

A MODEL, not a measurement.  The panel it belongs beside reports what was
observed -- GCH-GCH and HCG-HCG concordance rising inside cLADs,
late-replicating domains and PMDs, and falling outside them.  This draws the
arrangement that would produce that; nothing here is fitted and the arc
weights are chosen, not measured.

WHAT IS DELIBERATELY NOT DRAWN.  No arc connects one heterochromatin domain
to another.  The measurement is a concordance between two ends of a single
read pair, so it speaks to loci within one bin -- kilobases to a few hundred
kilobases apart, inside a domain.  It carries no information about whether
two separate cLADs coordinate with each other, and an arc between them would
be a claim the data cannot support.  Coordination is therefore drawn only
along the fibre, within a single domain, in both panels.

The two nuclei differ in exactly one respect -- which compartment carries the
coordination -- because a schematic that changes several things at once lets a
reader attribute the effect to whichever one they already believed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TUMOUR = "#c0392b"
OTHER = "#2c3e50"
LAMINA = "#7f8c8d"
B_FILL = "#d5dbdb"
A_FILL = "#fdf2e9"
GENE = "#b9770e"


def _arc(ax, p0, p1, bulge, color, lw, alpha=1.0, ls="-", zorder=5):
    """A quadratic curve between two loci -- one coordinated pair."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    mid = (p0 + p1) / 2.0
    d = p1 - p0
    normal = np.array([-d[1], d[0]])
    n = np.linalg.norm(normal)
    ctrl = mid + (normal / n * bulge if n else 0.0)
    ax.add_patch(PathPatch(
        MPath([tuple(p0), tuple(ctrl), tuple(p1)],
              [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]),
        fc="none", ec=color, lw=lw, alpha=alpha, ls=ls,
        zorder=zorder, capstyle="round"))


def _envelope(ax, cx, cy, rx, ry, n_pores=14):
    """Nuclear envelope: two membranes with pores, and the lamina inside it.

    Drawn as a real double membrane rather than one ring because the lamina
    -- what the B domains are tethered to -- sits against the INNER one, and
    a single line leaves nothing for them to attach to.
    """
    from matplotlib.patches import Ellipse

    ax.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ry, fc=A_FILL, ec="none",
                         zorder=0))
    gap = 0.022
    ax.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ry, fc="none", ec=LAMINA,
                         lw=1.6, zorder=1))                     # outer
    ax.add_patch(Ellipse((cx, cy), 2 * (rx - gap), 2 * (ry - gap),
                         fc="none", ec=LAMINA, lw=1.6, zorder=1))  # inner
    ax.add_patch(Ellipse((cx, cy), 2 * (rx - gap - 0.012),
                         2 * (ry - gap - 0.012), fc="none", ec="#5d6d7e",
                         lw=2.6, alpha=0.55, zorder=1))            # lamina
    for k in range(n_pores):
        t = 2 * np.pi * k / n_pores + 0.12
        ct, st = np.cos(t), np.sin(t)
        ax.plot([cx + rx * ct, cx + (rx - gap) * ct],
                [cy + ry * st, cy + (ry - gap) * st],
                color="white", lw=3.0, solid_capstyle="butt", zorder=1.4)
        ax.plot([cx + rx * ct, cx + (rx - gap) * ct],
                [cy + ry * st, cy + (ry - gap) * st],
                color=LAMINA, lw=1.2, solid_capstyle="butt", zorder=1.5)


def _flat_bump(d, half, ramp=0.42):
    """1 across the middle of a sector, smoothly to 0 at its edge.

    A raised cosine peaks at a POINT, so a fibre following it touches the
    lamina only at the apex and the resulting domain is a nub.  A flat top is
    what makes a domain wide enough to carry coordination along it.
    """
    a = half * (1.0 - ramp)
    x = np.clip((np.abs(d) - a) / max(half - a, 1e-9), 0.0, 1.0)
    return 1.0 - (3.0 * x ** 2 - 2.0 * x ** 3)


LAMINA_SECTORS = ((0.85, 0.62), (2.45, 0.50), (3.95, 0.66), (5.45, 0.44))


def chromosome_path(cx, cy, rx, ry, n=1400, r_lo=0.30, r_hi=0.985):
    """One chromatin fibre wandering between the lamina and the interior.

    The lamina contacts are placed as explicit sectors of differing width
    rather than left to a harmonic, for two reasons: a single harmonic makes
    an n-fold rosette that reads as a decorative motif, and its peaks are
    points, so the domains come out too short to draw coordination along.
    Real lamina contacts differ in extent, which the sector widths say.

    Deterministic, so both panels draw the same chromosome and any visible
    difference between them is the arc weight, not a different draw.
    """
    t = np.linspace(0.30, 2 * np.pi - 0.18, n)
    m = np.zeros_like(t)
    for c, half in LAMINA_SECTORS:
        d = np.angle(np.exp(1j * (t - c)))
        m = np.maximum(m, _flat_bump(d, half))
    r = r_lo + (r_hi - r_lo) * m
    r = r + 0.020 * np.sin(11.0 * t) + 0.010 * np.sin(19.0 * t + 0.7)
    r = np.clip(r, 0.16, 1.0)
    inner = rx - 0.040, ry - 0.040
    return (cx + inner[0] * r * np.cos(t), cy + inner[1] * r * np.sin(t), r)


def _runs(mask):
    """Contiguous True stretches of *mask*, as (start, stop) index pairs."""
    out, i, n = [], 0, mask.size
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _fibre(ax, x, y, is_b):
    """The fibre itself: compact and dark where it meets the lamina."""
    for a, b in _runs(~is_b):
        ax.plot(x[a:b + 1], y[a:b + 1], color="#c2ab8e", lw=2.2, alpha=0.98,
                solid_capstyle="round", zorder=3)
    for a, b in _runs(is_b):
        # compact: a broad pale sheath with a dark core, so it reads as
        # condensed chromatin against the thin open fibre
        ax.plot(x[a:b + 1], y[a:b + 1], color="#9fb0b0", lw=6.6, alpha=0.95,
                solid_capstyle="round", zorder=3)
        ax.plot(x[a:b + 1], y[a:b + 1], color="#4d5656", lw=2.4, alpha=0.95,
                solid_capstyle="round", zorder=3.1)
    # the free end, so the fibre reads as a chromosome and not a ring
    ax.plot([x[0]], [y[0]], marker="o", ms=3.4, color="#cbb69c",
            zorder=3.4)


def _within_arcs(ax, x, y, runs, strength, color, n_per=3, span=0.42):
    """Coordination between loci on the SAME domain, and only there.

    span is a fraction of the domain's own length, so every arc stays inside
    the stretch it belongs to.  There is deliberately no path from one run to
    another: the measurement cannot see domain-to-domain coordination.
    """
    for a, b in runs:
        L = b - a
        if L < 12:
            continue
        for k in range(n_per):
            f0 = 0.10 + 0.80 * k / max(n_per - 1, 1) - span / 2
            f0 = min(max(f0, 0.04), 1.0 - span - 0.04)
            i = a + int(f0 * L)
            j = a + int((f0 + span) * L)
            _arc(ax, (x[i], y[i]), (x[j], y[j]), 0.055,
                 color, 0.6 + 3.2 * strength,
                 alpha=0.30 + 0.60 * strength, zorder=5)


def draw(ax, title, tag, b_strength, a_strength, note, cross=False):
    """One nucleus.  b_strength / a_strength scale the arcs, nothing else."""
    cx, cy, rx, ry = 0.5, 0.50, 0.40, 0.35
    _envelope(ax, cx, cy, rx, ry)
    x, y, r = chromosome_path(cx, cy, rx, ry)
    is_b = r > 0.72
    _fibre(ax, x, y, is_b)

    _within_arcs(ax, x, y, _runs(is_b), b_strength, OTHER, n_per=3)
    _within_arcs(ax, x, y, _runs(~is_b), a_strength, GENE, n_per=3)

    ax.text(0.02, 0.975, tag, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top")
    ax.set_title(title, fontsize=11, pad=6)
    ax.text(0.5, 0.045, note, transform=ax.transAxes, fontsize=8.2,
            ha="center", va="center", color="0.25")
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(-0.02, 0.96)
    ax.set_aspect("equal")
    ax.axis("off")


def legend_panel(ax):
    """Horizontal key at fixed columns.

    Square swatches, not ellipses stretched by the axes aspect, and fixed
    positions rather than a cumulative advance -- text width is not knowable
    before rendering, so advancing by a guess is how legend items end up on
    top of one another.
    """
    from matplotlib.patches import Rectangle

    ax.axis("off")
    items = [
        (0.015, "patch", B_FILL, "cLAD / late-RT / PMD"),
        (0.290, "line", LAMINA, "nuclear lamina"),
        (0.490, "line", OTHER, "B-domain co-accessibility"),
        (0.760, "line", GENE, "A-hub co-accessibility"),
    ]
    for x, kind, col, lab in items:
        if kind == "patch":
            ax.add_patch(Rectangle((x, 0.36), 0.020, 0.34, fc=col,
                                   ec="#95a5a6", lw=0.8,
                                   transform=ax.transAxes))
        else:
            ax.plot([x, x + 0.020], [0.53, 0.53], color=col, lw=2.6,
                    solid_capstyle="round", transform=ax.transAxes)
        ax.text(x + 0.028, 0.53, lab, fontsize=7.8, va="center",
                transform=ax.transAxes)


def _direction(ax, xy, text, color, up=True):
    """Name the measured direction on the drawing itself.

    A schematic that shows a rearrangement without saying which way the
    number moved leaves the reader to infer it from line weight, which is a
    choice made by whoever drew it.
    """
    ax.annotate(text, xy=xy, xycoords="axes fraction", fontsize=9,
                fontweight="bold", color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color,
                          lw=1.0, alpha=0.92), zorder=20)


def figure(path, title, subtitle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.4, 4.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.13], hspace=0.02,
                          wspace=0.01, left=0.01, right=0.99,
                          top=0.855, bottom=0.02)
    ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axl = fig.add_subplot(gs[1, :])

    draw(ax0, "Reference cell types", "a", b_strength=0.10, a_strength=0.95,
         note="coordination held in the active\nregulatory compartment")
    draw(ax1, "Meningioma", "b", b_strength=0.95, a_strength=0.14,
         note="coordination shifts into heterochromatin;\n"
              "the regulatory hub decoheres", cross=True)
    _direction(ax1, (0.300, 0.815), "D \u2191", OTHER)
    _direction(ax1, (0.560, 0.470), "D \u2193", GENE)
    legend_panel(axl)

    fig.suptitle(title, fontsize=11.5, y=0.985)
    fig.text(0.5, 0.905, subtitle, ha="center", fontsize=7.8, color="0.32")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.
                                 RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True,
                    help="Output figure (.pdf keeps it vector for the "
                         "journal; .png only for a draft).")
    ap.add_argument("--title",
                    default="A compartment-scale redistribution of long-range "
                            "epigenetic coordination")
    ap.add_argument("--subtitle",
                    default="Model consistent with GCH-GCH and HCG-HCG "
                            "concordance rising inside cLADs, late-replicating "
                            "domains and PMDs, and falling outside them")
    args = ap.parse_args(argv)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure(out, args.title, args.subtitle)
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
