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

# 12 linked pairs per state as (co-open, co-closed, mixed), chosen so the
# implied C and F - F_exp reproduce the points drawn in panel a exactly.
# 'balanced' is NOT a 50/50 split of the concordant pairs: it is the same
# open:closed RATIO as the null, which at F_exp = 1/3 is one open to two
# closed.  Drawing it 50/50 would make it a co-open shift, which is the
# confusion the whole re-referencing exists to remove.
PAIRS_NULL = (2, 4, 6)
PAIRS = {
    "uncoordinated":         (2, 4, 6),      # exactly the null
    "coordinated balanced":  (4, 8, 0),      # every pair concordant, null ratio
    "co-open shifted":       (12, 0, 0),     # every pair co-open
    "co-closed shifted":     (0, 12, 0),     # every pair co-closed
    "anti-coordinated":      (1, 2, 9),      # fewest concordant, null ratio
}

R_CUT = 2.0
SECTOR = 30.0
F_BAR = 0.3230          # measured; sets the legacy D multiplier
D_MULT = 2 * F_BAR - 1


def _pair(ax, x, y, left_open, right_open, w, r=0.0062):
    """One linked read pair: two ends joined by a contact arc.

    Filled = accessible end, hollow = inaccessible.  Both filled is a
    co-open pair, both hollow co-closed, one of each is mixed -- the three
    categories every statistic on this figure is built from.
    """
    from matplotlib.patches import Circle
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    ax.add_patch(PathPatch(
        MPath([(x, y), (x + w / 2, y + w * 0.78), (x + w, y)],
              [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]),
        fc="none", ec="0.62", lw=0.7, transform=ax.transAxes, zorder=3))
    for cx, is_open in ((x, left_open), (x + w, right_open)):
        ax.add_patch(Circle(
            (cx, y), r, transform=ax.transAxes, zorder=4,
            fc=(C_OPEN if is_open else "white"),
            ec=(C_OPEN if is_open else "0.35"), lw=0.8))


def _cartoon(ax, x0, y0, counts, cell_w, n_col=6, gap=0.007):
    """12 pairs in two rows: co-open first, then co-closed, then mixed."""
    n_open, n_closed, n_mixed = counts
    kinds = ([(True, True)] * n_open + [(False, False)] * n_closed
             + [(True, False)] * n_mixed)
    pw = (cell_w - (n_col - 1) * gap) / n_col
    for i, (lo, ro) in enumerate(kinds):
        r, c = divmod(i, n_col)
        _pair(ax, x0 + c * (pw + gap), y0 - r * 0.046, lo, ro, pw * 0.52)


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
    # What the radius actually MEANS.  R^2 is chi-square(2), so every circle
    # is a p contour; R_CUT is a display default, not a threshold anyone
    # derived, and with millions of cell-bins the real cut is FDR on
    # polar_p, which sits far outside this frame.
    for pv, col in ((0.05, "0.55"), (0.01, "0.65")):
        rr = float(np.sqrt(-2.0 * np.log(pv)))
        ax.add_patch(Circle((0, 0), rr, fc="none", ec=col, lw=0.8,
                            ls=(0, (1.5, 2.5)), zorder=2))
        ax.text(rr * np.cos(np.radians(112.5)),
                rr * np.sin(np.radians(112.5)), f"p={pv:g}", fontsize=6.4,
                color=col, ha="center", va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none",
                          alpha=0.85))
    ax.annotate(f"R < {R_CUT:g}  (p > {np.exp(-R_CUT ** 2 / 2):.2f}):\n"
                f"class default, not a test",
                xy=(-R_CUT * 0.71, -R_CUT * 0.71), xytext=(-3.85, -3.45),
                fontsize=6.9, color="0.35", ha="left", zorder=3,
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
    """Which pairs each scalar cannot tell apart, beside what each state is."""
    ax.axis("off")
    cols = ["C", r"F$-$F$_{exp}$", "R", r"$\theta$", r"polar $p$"]
    rows = []
    for lab, c, f, col, _o, _h, _v in STATES:
        R = float(np.hypot(c, f))
        th = np.degrees(np.arctan2(f, c)) if R > 1e-9 else np.nan
        rows.append((lab, col, [c, f, R, th, float(np.exp(-R * R / 2))]))
    arr = np.array([r[2] for r in rows], float)

    x_lab, x_art, w_art = 0.005, 0.150, 0.252
    x0, w, h = 0.470, 0.104, 0.136
    y0 = 0.655

    # The null, drawn once above the table: every row below is read against
    # it, and without it "more concordant than expected" has no referent.
    ax.text(x_art, y0 + 0.292, "expected under the null (E)", fontsize=7.4,
            style="italic", color="0.35", va="bottom",
            transform=ax.transAxes)
    _cartoon(ax, x_art, y0 + 0.256, PAIRS_NULL, w_art)
    ax.plot([x_lab, 0.412], [y0 + 0.163] * 2, color="0.8", lw=0.8,
            transform=ax.transAxes, clip_on=False)

    for j, cname in enumerate(cols):
        ax.text(x0 + j * w + w / 2, y0 + 0.115, cname, ha="center",
                va="bottom", fontsize=7.8, fontweight="bold",
                transform=ax.transAxes)
    ax.text(x_art, y0 + 0.115, "observed linked pairs (O)", fontsize=7.4,
            fontweight="bold", va="bottom", transform=ax.transAxes)

    for i, (lab, col, _v) in enumerate(rows):
        yy = y0 - i * h
        ax.add_patch(plt_rect(x_lab, yy - h * 0.30, 0.016, h * 0.56, col, ax))
        ax.text(x_lab + 0.026, yy, lab, fontsize=7.8, va="center",
                transform=ax.transAxes)
        _cartoon(ax, x_art, yy + 0.020, PAIRS[lab], w_art)
        for j in range(len(cols)):
            v = arr[i, j]
            if not np.isfinite(v):
                txt = "--"
            elif j == 3:
                txt = f"{v:+.0f}\u00b0"
            elif j == 4:
                txt = "1.00" if v > 0.995 else f"{v:.3f}"
            else:
                txt = f"{v:+.2f}"
            other = np.delete(arr[:, j], i)
            atol = 0.002 if j == 4 else 0.02
            clash = bool(np.any(np.isfinite(other)
                                & np.isclose(other, v, atol=atol))
                         ) if np.isfinite(v) else False
            if clash:
                ax.add_patch(plt_rect(x0 + j * w + 0.006, yy - h * 0.30,
                                      w - 0.012, h * 0.56, "#c0392b", ax,
                                      alpha=0.13))
            ax.text(x0 + j * w + w / 2, yy, txt, ha="center", va="center",
                    fontsize=7.6, color=("#c0392b" if clash else "0.15"),
                    fontweight=("bold" if clash else "normal"),
                    transform=ax.transAxes)

    yb = y0 - len(rows) * h
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x_lab + 0.006, yb + 0.045), 0.0075, fc=C_OPEN,
                        ec=C_OPEN, lw=0.8, transform=ax.transAxes))
    ax.text(x_lab + 0.020, yb + 0.045, "accessible end", fontsize=7,
            va="center", transform=ax.transAxes)
    ax.add_patch(Circle((x_lab + 0.135, yb + 0.045), 0.0075, fc="white",
                        ec="0.35", lw=0.8, transform=ax.transAxes))
    ax.text(x_lab + 0.149, yb + 0.045, "inaccessible end", fontsize=7,
            va="center", transform=ax.transAxes)
    ax.text(x_lab + 0.300, yb + 0.045,
            "both filled = co-open pair  ·  both hollow = co-closed  ·  "
            "one of each = mixed", fontsize=7, va="center", color="0.35",
            transform=ax.transAxes)
    ax.text(x_lab, yb - 0.015,
            "shaded = this statistic gives the same value to another state, "
            "so it cannot tell them apart", fontsize=7.4, color="#c0392b",
            transform=ax.transAxes)
    ax.text(x_lab, yb - 0.075,
            "Every column collides at least one pair -- each is a projection "
            "of the plane onto a line.  Only R and "
            r"$\theta$ TOGETHER separate all five." "\n"
            r"polar $p$ = exp($-R^2/2$): whitened, $R^2$ is "
            r"$\chi^2_2$ under the null, so ONE calibrated test covers both "
            "axes and needs no reconciling of two FDRs.\n"
            r"It says how far from the null, never which way -- $\theta$ "
            "does that -- so balanced and anti-coordinated share a "
            r"$p$ and are told apart only by angle." "\n"
            "'balanced' keeps the NULL's open:closed ratio, not 50/50: at "
            r"F$_{exp}$ = 1/3 a 50/50 split is itself a co-open shift.",
            fontsize=7.4, color="0.30", va="top", linespacing=1.6,
            transform=ax.transAxes)
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

    fig = plt.figure(figsize=(13.2, 6.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.05], wspace=0.10,
                          left=0.055, right=0.985, top=0.87, bottom=0.19)
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
