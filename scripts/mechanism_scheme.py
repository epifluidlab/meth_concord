#!/usr/bin/env python3
"""Mechanistic scheme: long-range coordination is LOST in tumour heterochromatin.

A MODEL, not a measurement.  What the accompanying profiles report, on the
marginal-free coordination magnitude C against a distance-pooled non-contact
null with a shared Dirichlet prior:

  inside cLADs, late-replicating domains and PMDs, C sits BELOW the flanking
  euchromatin in every cell type, and further below it in meningioma.

Nothing here is fitted; the arc weights are chosen to carry that ordering and
nothing else.

THIS REVERSES AN EARLIER VERSION OF THIS FIGURE, which drew coordination
moving INTO heterochromatin in tumour.  That reading came from
D = C(2F - 1), whose direction term centres at F = 0.5 -- correct only when
the mean per-end openness is 0.5.  At the measured 0.4009 the term is -0.354,
corr(D, -C) = 0.970, and D was an inverted, rescaled copy of C.  The sign of
the whole claim came from that.

WHAT IS DELIBERATELY NOT DRAWN, and why the two active compartments are
IDENTICAL between the panels:

  No arc connects one heterochromatin domain to another.  The measurement is
  a concordance between the two ends of a single read pair, so it speaks to
  loci within one bin.  It carries no information about whether two separate
  cLADs coordinate, and an arc between them would be a claim the data cannot
  support.

  The profiles are baselined on each domain's own flank and each cell is
  centred on its own median, so they measure domain-versus-flank and nothing
  else.  Whether euchromatin itself differs between tumour and reference is
  NOT answerable from them, and drawing a change there would invent a second
  finding.  The panels therefore differ in one compartment only.

DIRECTION IS A SEPARATE AXIS AND IS NOT ENCODED IN ARC WEIGHT.  On F - F_exp
the concordant mass inside these domains sits co-OPEN of the marginal-matched
null, less so in tumour -- orthogonal to the magnitude by construction, since
scaling the concordant mass at a fixed co-open:co-closed ratio moves C and
leaves the direction where it was.  It is stated in the caption rather than
drawn, because a reader cannot be expected to separate two quantities encoded
in one line.
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


# The whole claim, as two numbers per panel.  Module level so a test can
# assert on what SHIPS rather than on values retyped beside it.
#
#   b < a in both panels        heterochromatin coordinates less than its
#                               flanking euchromatin, which is the profile's
#                               domain-versus-flank result
#   b_tumour < b_reference      and less again in meningioma
#   a identical                 euchromatin is NOT compared by a
#                               flank-baselined profile, so the figure makes
#                               no claim about it
CAVEAT = (
    "Direction is a separate axis and is not encoded in arc weight: on "
    "F \u2212 F_exp the residual concordant mass inside these domains sits "
    "co-open of the marginal-matched null, less so in tumour.  Euchromatin "
    "is drawn alike in both panels because a flank-baselined, per-cell-"
    "centred profile measures domain versus flank and cannot compare it.  "
    "No arc joins two domains: the measurement sees the two ends of one "
    "read pair."
)

PANEL_REFERENCE = {"b_strength": 0.55, "a_strength": 0.95}
PANEL_TUMOUR = {"b_strength": 0.10, "a_strength": 0.95}


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


def draw(ax, title, tag, b_strength, a_strength, note):
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
        (0.470, "line", OTHER, "coordination within a B domain"),
        (0.760, "line", GENE, "coordination within an A domain"),
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

    fig = plt.figure(figsize=(7.4, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.13], hspace=0.02,
                          wspace=0.01, left=0.01, right=0.99,
                          top=0.855, bottom=0.10)
    ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axl = fig.add_subplot(gs[1, :])

    # A_STRENGTH IS THE SAME IN BOTH PANELS, deliberately.  The profiles are
    # baselined on each domain's own flank, so they cannot compare
    # euchromatin between the groups; drawing a difference there would be a
    # second finding the measurement does not contain.  Only b_strength
    # moves, and it moves DOWN -- heterochromatin already carries less
    # coordination than its flank in reference cells, and less again in
    # tumour.
    draw(ax0, "Reference cell types", "a", **PANEL_REFERENCE,
         note="heterochromatin already coordinates\n"
              "less than its flanking euchromatin")
    draw(ax1, "Meningioma", "b", **PANEL_TUMOUR,
         note="the residual coordination inside\n"
              "heterochromatin is lost further")
    # Just outside the envelope, beside an upper-left B sector: inside it
    # the badge sits on the fibre it is labelling.
    _direction(ax0, (0.185, 0.915), "C < flank", OTHER)
    _direction(ax1, (0.185, 0.915), "C \u226a flank", OTHER)
    legend_panel(axl)

    fig.suptitle(title, fontsize=11.5, y=0.985)
    # WRAPPED.  A one-line caption at this length makes bbox_inches="tight"
    # widen the whole figure to fit it, and the nuclei end up as two small
    # marks adrift in white space.
    import textwrap
    fig.text(0.5, 0.945, textwrap.fill(subtitle, 104), ha="center",
             va="top", fontsize=7.6, color="0.32", linespacing=1.45)
    # What the figure does NOT claim, kept clear of what it does.
    fig.text(0.5, 0.012, textwrap.fill(CAVEAT, 132), ha="center",
             va="bottom", fontsize=6.5, color="0.45", linespacing=1.4)
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
                    default="Long-range epigenetic coordination is lost in "
                            "tumour heterochromatin")
    ap.add_argument("--subtitle",
                    default="Model consistent with the coordination magnitude C "
                            "sitting below the flanking euchromatin inside "
                            "cLADs, late-replicating domains and PMDs, and "
                            "further below it in meningioma")
    args = ap.parse_args(argv)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure(out, args.title, args.subtitle)
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
