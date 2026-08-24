#!/usr/bin/env python3
"""How R and theta separate five coordination states that no single axis can.

The companion to the C-vs-D panel.  That figure showed two states; this one
shows why five need a PLANE rather than another scalar.

Each state is a point in (C, F - F_exp).  Every scalar summary is a
projection of that plane onto a line, and every projection collides at least
one pair:

  C      co-open and co-closed carry identical C -- Cov is unchanged by
         swapping open for closed.
  F-F_exp  uncoordinated, balanced and anti-coordinated all sit at 0.
  D = C(2F-1)  is ~ -0.35 C when p-bar is 0.40, so it collapses the same
         pair C does, and inverts the sign while doing it.
  R      alone cannot separate co-open from co-closed, nor balanced from
         anti-coordinated: they are equidistant from the origin.

Only the pair (R, theta) separates all five, which is the point: the state
is two-dimensional and no scalar recovers it.  R says how far from the null,
theta says which way, and the classification is a sector plus a radius cut.

A MODEL panel: the five points are chosen to sit one per sector, not fitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

C_OPEN = "#0e8f7e"
C_CLOSED = "#2c3e50"
C_MAG = "#7d5ba6"
C_NULL = "#95a5a6"
C_ANTI = "#c0392b"

# (label, C, dF, colour, label offset, ha, va) in standardised units.
# Offsets are per state rather than derived from the sign of C: that rule
# put the anti-coordinated label off the left edge and dropped the
# uncoordinated one on top of the radius annotation.
STATES = [
    ("uncoordinated",         0.00,  0.00, C_NULL,   (0.55,  0.95), "left", "bottom"),
    ("coordinated balanced",  3.00,  0.00, C_MAG,    (0.00, -0.85), "center", "top"),
    ("co-open shifted",       2.10,  2.10, C_OPEN,   (0.30,  0.50), "left", "bottom"),
    ("co-closed shifted",     2.10, -2.10, C_CLOSED, (0.30, -0.50), "left", "top"),
    ("anti-coordinated",     -3.00,  0.00, C_ANTI,   (0.00,  0.85), "center", "bottom"),
]

R_CUT = 2.0
SECTOR = 30.0
F_BAR = 0.3230          # measured; sets the legacy D multiplier
D_MULT = 2 * F_BAR - 1


def _sectors(ax, lim):
    """Shade the four angular sectors and the uncoordinated disc."""
    from matplotlib.patches import Circle, Wedge

    bands = [(-SECTOR, SECTOR, C_MAG, "balanced"),
             (SECTOR, 180 - SECTOR, C_OPEN, "co-open"),
             (180 - SECTOR, 180 + SECTOR, C_ANTI, "anti"),
             (180 + SECTOR, 360 - SECTOR, C_CLOSED, "co-closed")]
    for a0, a1, col, _ in bands:
        ax.add_patch(Wedge((0, 0), lim * 1.5, a0, a1, fc=col, alpha=0.07,
                           ec="none", zorder=0))
    for a in (SECTOR, 180 - SECTOR, 180 + SECTOR, 360 - SECTOR):
        t = np.radians(a)
        ax.plot([0, lim * 1.5 * np.cos(t)], [0, lim * 1.5 * np.sin(t)],
                color="0.75", lw=0.7, ls=(0, (3, 3)), zorder=1)
    ax.add_patch(Circle((0, 0), R_CUT, fc="white", ec="0.45", lw=1.1,
                        ls="--", alpha=0.92, zorder=2))
    ax.annotate(f"R < {R_CUT:g}: uncoordinated", xy=(-1.41, -1.41),
                xytext=(-3.7, -3.3), fontsize=7.2, color="0.35",
                ha="left", zorder=3,
                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.7,
                                shrinkA=0, shrinkB=2))


def panel_plane(ax):
    lim = 4.0
    _sectors(ax, lim)
    ax.axhline(0, color="0.55", lw=0.8, zorder=1)
    ax.axvline(0, color="0.55", lw=0.8, zorder=1)
    for lab, c, f, col, (ox, oy), ha, va in STATES:
        R = float(np.hypot(c, f))
        ax.plot([0, c], [0, f], color=col, lw=1.0, alpha=0.55, zorder=4)
        ax.scatter([c], [f], s=95, color=col, ec="white", lw=1.2, zorder=6)
        sub = ("R = 0" if R < 1e-9 else
               "R = %.1f,  $\\theta$ = %.0f$\\degree$"
               % (R, np.degrees(np.arctan2(f, c))))
        ax.text(c + ox, f + oy, "%s\n%s" % (lab, sub), fontsize=7.4,
                color=col, ha=ha, va=va, zorder=7, linespacing=1.45)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("C  —  coordination magnitude  (standardised)", fontsize=8.5)
    ax.set_ylabel(r"F $-$ F$_{exp}$  —  direction  (standardised)",
                  fontsize=8.5)
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("a   the plane: R is how far, "
                 r"$\theta$ is which way", fontsize=10, loc="left", pad=8)
    # A key rather than labels on the wedges themselves: at this scale a
    # label at a wedge bisector lands on the state sitting in it.
    ax.text(0.0, -0.155,
            r"sectors:   $|\theta|<30\degree$ balanced   $\cdot$   "
            r"$30$ to $150\degree$ co-open",
            transform=ax.transAxes, fontsize=6.9, color="0.35", va="bottom")
    ax.text(0.0, -0.205,
            r"               $|\theta|>150\degree$ anti-coordinated   "
            r"$\cdot$   $-150$ to $-30\degree$ co-closed",
            transform=ax.transAxes, fontsize=6.9, color="0.35", va="bottom")


def panel_collapse(ax):
    """Which pairs each scalar cannot tell apart."""
    ax.axis("off")
    cols = ["C", r"F$-$F$_{exp}$", r"D = C(2F$-$1)", "R", r"$\theta$",
            r"(R, $\theta$)"]
    vals = []
    for lab, c, f, col, _o, _h, _v in STATES:
        R = float(np.hypot(c, f))
        th = np.degrees(np.arctan2(f, c)) if R > 1e-9 else np.nan
        vals.append([c, f, c * D_MULT, R, th])
    arr = np.array(vals, float)

    n_r, n_c = len(STATES), len(cols)
    x0, y0, w, h = 0.315, 0.86, 0.112, 0.138
    for j, cname in enumerate(cols):
        ax.text(x0 + j * w + w / 2, y0 + 0.055, cname, ha="center",
                va="bottom", fontsize=8, fontweight="bold",
                transform=ax.transAxes)
    for i, (lab, c, f, col, _o, _h, _v) in enumerate(STATES):
        yy = y0 - i * h
        ax.add_patch(plt_rect(0.005, yy - h * 0.34, 0.018, h * 0.62, col,
                              ax))
        ax.text(0.032, yy, lab, fontsize=8, va="center",
                transform=ax.transAxes)
        for j in range(n_c):
            if j == n_c - 1:
                txt, clash = "distinct", False
            else:
                v = arr[i, j]
                txt = ("--" if not np.isfinite(v)
                       else (f"{v:+.0f}" + "°" if j == 4
                             else f"{v:+.2f}"))
                other = np.delete(arr[:, j], i)
                clash = bool(np.any(np.isfinite(other)
                                    & np.isclose(other, v, atol=0.02))
                             ) if np.isfinite(v) else False
            cx = x0 + j * w + w / 2
            if clash:
                ax.add_patch(plt_rect(x0 + j * w + 0.008, yy - h * 0.34,
                                      w - 0.016, h * 0.62, "#c0392b", ax,
                                      alpha=0.13))
            ax.text(cx, yy, txt, ha="center", va="center", fontsize=7.8,
                    color=("#c0392b" if clash else "0.15"),
                    fontweight=("bold" if clash else "normal"),
                    transform=ax.transAxes)
    ax.text(0.005, y0 - n_r * h - 0.02,
            "shaded = this statistic gives the same value to another state, "
            "so it cannot tell them apart",
            fontsize=7.6, color="#c0392b", transform=ax.transAxes)
    ax.text(0.005, y0 - n_r * h - 0.09,
            "Every scalar collides at least one pair -- each is a projection "
            "of the plane onto a line.\nOnly the pair separates all five.  "
            r"D also inverts: with $\bar{p}$ = 0.40 it is $\approx$ "
            r"$-0.35\,$C, so C falling reads as D rising.",
            fontsize=7.6, color="0.30", transform=ax.transAxes)
    ax.set_title("b   what each summary can and cannot separate",
                 fontsize=10, loc="left", pad=8)


def plt_rect(x, y, w, h, color, ax, alpha=1.0):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, fc=color, ec="none", alpha=alpha,
                     transform=ax.transAxes, zorder=0)


def figure(path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.0, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.30], wspace=0.13,
                          left=0.055, right=0.985, top=0.87, bottom=0.14)
    panel_plane(fig.add_subplot(gs[0, 0]))
    panel_collapse(fig.add_subplot(gs[0, 1]))
    fig.suptitle(title, fontsize=11.5, y=0.97)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True,
                    help=".pdf keeps it vector for the journal.")
    ap.add_argument("--title",
                    default="Coordination is two-dimensional: no single "
                            "statistic separates the five states")
    args = ap.parse_args(argv)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure(out, args.title)
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
