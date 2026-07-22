#!/usr/bin/env python3
"""Figure 5A — the long-range epigenetic concordance scheme, as ONE panel.

Combines two earlier drafts into a single figure:

  * the nome_hic_concordance workflow (formerly method_schematic.py, panels
    a-f) with the panel letters removed and the wording matched to the current
    pipeline, and
  * the toy demonstration (formerly a separate figure) that closes the panel:
    two cell states with IDENTICAL mean methylation / accessibility (0.5) but
    OPPOSITE long-range concordance (1.0 vs 0.0) -- mean cannot separate them,
    signed concordance does.

Top two tiers = the method (single-cell contact -> per-end posterior -> soft
co-state -> long-range vs local baseline -> signed Δ per (cell, bin) -> RWR
denoising over the cell graph).  Bottom tier = the payoff: a mean-level UMAP
where the two states overlap, next to a concordance UMAP where they split.

Emits a vector PDF + a 300-dpi PNG into figures/.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                Rectangle, PathPatch)
from matplotlib.path import Path
from scipy.stats import beta as betadist

# ---- palette -------------------------------------------------------------
FILL   = "#26303A"   # methylated / accessible (filled lollipop)
DNA    = "#9AA0A6"
PANEL  = "#F6F7F9"
EDGE   = "#D5D8DF"
INK    = "#23262B"
MUT    = "#6B7280"
ACC    = "#3D7DCA"   # arrows / brackets
SLATE  = "#5B6B7A"
S1     = "#7E6FCB"   # cell state 1 (purple)
S2     = "#E76F51"   # cell state 2 (orange)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "text.color": INK, "axes.edgecolor": INK,
})

fig = plt.figure(figsize=(16.4, 11.8), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 164)
ax.set_ylim(0, 118)
ax.axis("off")

# panel grid for the two workflow tiers
PW, PH = 49, 31
XS = [5, 57.5, 110]
TIER1, TIER2 = 78, 43          # y of each tier's bottom edge


def panel(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.0,rounding_size=1.6",
                 fc=PANEL, ec=EDGE, lw=1.2, zorder=1))
    ax.text(x + w / 2, y + h - 2.4, title, fontsize=10.8, fontweight="bold",
            va="top", ha="center", color=INK)


def lolly(cx, cy, filled, r=0.95, stem=2.4):
    ax.plot([cx, cx], [cy, cy + stem], color=DNA, lw=1.3, zorder=3)
    ax.add_patch(Circle((cx, cy + stem + r), r,
                        fc=(FILL if filled else "white"), ec=FILL, lw=1.4,
                        zorder=4))


def arc(x1, x2, ybase, h, color=DNA, lw=1.6, ls="-"):
    xm = (x1 + x2) / 2
    p = PathPatch(Path([(x1, ybase), (xm, ybase + h), (x2, ybase)],
                       [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                  fc="none", ec=color, lw=lw, ls=ls, zorder=2)
    ax.add_patch(p)


def flow(x1, y1, x2, y2, rad=0.0, color=ACC, lw=2.0, mut=15):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                 mutation_scale=mut, lw=lw, color=color, zorder=6))


def loop(cx, cy, sl, sr, span=6.2, r=1.15, h=3.4):
    """A within-cell contact: two loci (filled=methylated) joined by an arc."""
    xl, xr = cx - span / 2, cx + span / 2
    arc(xl, xr, cy + r, h, color="0.5", lw=1.9)
    for xx, s in ((xl, sl), (xr, sr)):
        ax.add_patch(Circle((xx, cy), r, fc=(FILL if s else "white"),
                            ec=FILL, lw=1.5, zorder=4))


# =====================================================================
# Title
# =====================================================================
ax.text(82, 114.6, "Long-range epigenetic concordance", fontsize=16.5,
        fontweight="bold", ha="center", va="center")
ax.text(82, 110.8,
        "a per-cell, per-bin signal that separates cell states even when mean "
        "methylation / accessibility is identical",
        fontsize=10.2, color=MUT, ha="center", va="center", style="italic")

# =====================================================================
# Tier 1, panel 1 — single-cell NOMe-HiC contact
# =====================================================================
ax0 = XS[0]
panel(ax0, TIER1, PW, PH, "Single-cell NOMe-HiC contact")
cy = TIER1 + 11
liA = (ax0 + 6, ax0 + 20)
liB = (ax0 + 29, ax0 + 43)
for (lx0, lx1) in (liA, liB):
    ax.plot([lx0, lx1], [cy, cy], color=DNA, lw=3.0, solid_capstyle="round",
            zorder=2)
for lx0, lx1, states in [(liA[0], liA[1], [1, 1, 0, 1]),
                         (liB[0], liB[1], [1, 0, 1, 1])]:
    for xx, s in zip(np.linspace(lx0 + 2, lx1 - 2, len(states)), states):
        lolly(xx, cy, s)
arc(np.mean(liA), np.mean(liB), cy, 12.5, color=ACC, lw=2.0)
ax.text((np.mean(liA) + np.mean(liB)) / 2, cy + 13.2, "chromatin contact",
        ha="center", va="bottom", fontsize=8.8, color=ACC, style="italic")
ax.text(np.mean(liA), cy - 4.0, "locus $i$", ha="center", fontsize=9.2)
ax.text(np.mean(liB), cy - 4.0, "locus $j$", ha="center", fontsize=9.2)
ax.text(np.mean(liA), cy - 7.4, r"$k_i/n_i$", ha="center", fontsize=8.8,
        color=MUT)
ax.text(np.mean(liB), cy - 7.4, r"$k_j/n_j$", ha="center", fontsize=8.8,
        color=MUT)
ax.text(ax0 + PW / 2, TIER1 + 2.4, "GCH accessibility  ·  HCG (CpG) methylation",
        ha="center", fontsize=8.4, color=MUT)

# =====================================================================
# Tier 1, panel 2 — per-end Beta-Binomial posterior
# =====================================================================
bx = XS[1]
panel(bx, TIER1, PW, PH, "Per-end state posterior")
gx0, gx1 = bx + 8, bx + 42
gy0, gy1 = TIER1 + 8, TIER1 + 22
ax.plot([gx0, gx1], [gy0, gy0], color=INK, lw=1.1)
ax.plot([gx0, gx0], [gy0, gy1], color=INK, lw=1.1)
tt = np.linspace(0.001, 0.999, 200)
for (a_, b_, col, lab) in [(9, 2, S2, r"$p_i$"), (5, 5, SLATE, r"$p_j$")]:
    pdf = betadist.pdf(tt, a_, b_)
    pdf = pdf / pdf.max()
    X = gx0 + tt * (gx1 - gx0)
    Y = gy0 + pdf * (gy1 - gy0) * 0.92
    ax.plot(X, Y, color=col, lw=2.3)
    ax.fill_between(X, gy0, Y, color=col, alpha=0.12)
    pm = tt[np.argmax(pdf)]
    ax.text(gx0 + pm * (gx1 - gx0), gy0 + (gy1 - gy0) * 0.98, lab, color=col,
            fontsize=11, ha="center", va="bottom", fontweight="bold")
ax.text((gx0 + gx1) / 2, gy0 - 3.2, r"$p=P(\mathrm{open}\,|\,k,n)$",
        ha="center", fontsize=10.2)
ax.text((gx0 + gx1) / 2, gy0 - 6.6, "Beta–Binomial mixture (EM)", ha="center",
        fontsize=8.4, color=MUT, style="italic")
ax.text(gx0 - 1.2, gy1, "density", rotation=90, va="top", ha="right",
        fontsize=8.0, color=MUT)
ax.text(gx1, gy0 - 0.8, "P(open)", va="top", ha="right", fontsize=8.0,
        color=MUT)

# =====================================================================
# Tier 1, panel 3 — soft pairwise co-state
# =====================================================================
cx0 = XS[2]
panel(cx0, TIER1, PW, PH, "Soft pairwise co-state")
rows = [("co-open", 1, 1, r"$p_i\,p_j$"),
        ("co-closed", 0, 0, r"$(1-p_i)(1-p_j)$"),
        ("mixed", 1, 0, r"$p_i(1-p_j)+(1-p_i)p_j$")]
ry = [TIER1 + 21, TIER1 + 13.5, TIER1 + 6.5]
for (label, s1, s2, expr), yy in zip(rows, ry):
    for k, s in enumerate((s1, s2)):
        ax.add_patch(Circle((cx0 + 7 + k * 4.0, yy), 1.45,
                            fc=(FILL if s else "white"), ec=FILL, lw=1.5,
                            zorder=4))
    ax.text(cx0 + 15, yy + 1.4, label, fontsize=9.4, fontweight="bold",
            va="center", color=INK)
    ax.text(cx0 + 15, yy - 2.0, expr, fontsize=8.6, va="center", color=MUT)
ax.text(cx0 + PW / 2, TIER1 + 2.2, "soft expected counts (no thresholding)",
        ha="center", fontsize=8.2, color=MUT, style="italic")

# =====================================================================
# Tier 2, panel 4 — long-range vs local baseline
# =====================================================================
dx = XS[0]
panel(dx, TIER2, PW, PH, "Long-range vs local baseline")
axis_y = TIER2 + 12
ax.annotate("", xy=(dx + 45, axis_y), xytext=(dx + 5, axis_y),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
ax.text(dx + 45, axis_y - 3.0, "contact distance", ha="right", fontsize=8.2,
        color=MUT)
thx = dx + 21
ax.plot([thx, thx], [axis_y - 1.4, axis_y + 12], color=ACC, lw=1.4, ls="--")
ax.text(thx, axis_y + 13, r"$d_0\,(\approx\!20\,\mathrm{kb})$", ha="center",
        fontsize=8.4, color=ACC)
ax.plot([dx + 8, dx + 15], [axis_y, axis_y], color=DNA, lw=2.4,
        solid_capstyle="round")
arc(dx + 8, dx + 15, axis_y, 3.4, color=MUT, lw=1.6)
ax.text(dx + 11.5, axis_y - 4.2, "short-range\n(< $d_0$, baseline)",
        ha="center", va="top", fontsize=8.0, color=MUT)
ax.plot([dx + 26, dx + 43], [axis_y, axis_y], color=DNA, lw=2.4,
        solid_capstyle="round")
arc(dx + 26, dx + 43, axis_y, 7.0, color=ACC, lw=2.0)
ax.text(dx + 34.5, axis_y - 4.2, "linked\n(long-range)", ha="center",
        va="top", fontsize=8.4, color=ACC, fontweight="bold")
by = TIER2 + 22
ax.plot([dx + 6, dx + 44], [by, by], color=INK, lw=1.1)
for xx in (dx + 6, dx + 44):
    ax.plot([xx, xx], [by, by - 1.3], color=INK, lw=1.1)
ax.text(dx + PW / 2, by + 1.0, "aggregate within 250-kb bin", ha="center",
        va="bottom", fontsize=8.4)

# =====================================================================
# Tier 2, panel 5 — signed long-range concordance Δ
# =====================================================================
ex = XS[1]
panel(ex, TIER2, PW, PH, "Signed long-range concordance")
ax.text(ex + PW / 2, TIER2 + 24.2,
        r"$\Delta_{\mathrm{signed}}=(\hat\pi_{\mathrm{co\text{-}open}}-\hat\pi_{\mathrm{co\text{-}closed}})_{\mathrm{linked}}$",
        ha="center", fontsize=9.6)
ax.text(ex + PW / 2, TIER2 + 20.0,
        r"$-\,(\hat\pi_{\mathrm{co\text{-}open}}-\hat\pi_{\mathrm{co\text{-}closed}})_{\mathrm{short\text{-}range}}$",
        ha="center", fontsize=9.6)
hm_x0, hm_y0, hm_w, hm_h = ex + 11, TIER2 + 4.5, 27, 12.5
rng = np.random.default_rng(1)
nC, nB = 12, 18
block = np.vstack([rng.normal(0.5, 0.35, (6, nB)),
                   rng.normal(-0.5, 0.35, (6, nB))])
block[:, ::3] += rng.normal(0, 0.5, (nC, len(range(0, nB, 3))))
ax.imshow(block, cmap="RdBu_r", vmin=-1.4, vmax=1.4, aspect="auto",
          extent=[hm_x0, hm_x0 + hm_w, hm_y0, hm_y0 + hm_h], zorder=3)
ax.add_patch(Rectangle((hm_x0, hm_y0), hm_w, hm_h, fc="none", ec=INK, lw=1.0,
                       zorder=4))
ax.text(hm_x0 + hm_w / 2, hm_y0 - 1.6, "bins", ha="center", fontsize=8.0,
        color=MUT)
ax.text(hm_x0 - 1.4, hm_y0 + hm_h / 2, "cells", rotation=90, va="center",
        ha="right", fontsize=8.0, color=MUT)
ax.text(hm_x0 + hm_w + 1.4, hm_y0 + hm_h, r"$+$", color="#B2182B", fontsize=10,
        va="top", fontweight="bold")
ax.text(hm_x0 + hm_w + 1.4, hm_y0, r"$-$", color="#2166AC", fontsize=10,
        va="bottom", fontweight="bold")

# =====================================================================
# Tier 2, panel 6 — RWR denoising over the cell graph
# =====================================================================
fx = XS[2]
panel(fx, TIER2, PW, PH, "Cell-graph denoising (RWR)")
gcx, gcy = fx + 15, TIER2 + 14
rngp = np.random.default_rng(7)
nodes = np.array([[gcx, gcy]] +
                 [[gcx + 6.5 * np.cos(t), gcy + 6.5 * np.sin(t)]
                  for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)])
for i in range(1, len(nodes)):
    ax.plot([nodes[0, 0], nodes[i, 0]], [nodes[0, 1], nodes[i, 1]],
            color=EDGE, lw=1.1, zorder=2)
for i in (1, 2, 4):
    j = i + 1 if i + 1 < len(nodes) else 1
    ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]],
            color=EDGE, lw=1.1, zorder=2)
for (nx, ny) in nodes:
    ax.add_patch(Circle((nx, ny), 1.35, fc="white", ec=ACC, lw=1.6, zorder=4))
ax.text(fx + PW / 2, TIER2 + 24.0,
        "random-walk-with-restart smoothing\nover the cell $k$-NN graph",
        ha="center", fontsize=8.4, color=INK, linespacing=1.3)
ax.text(fx + PW / 2, TIER2 + 3.0,
        "quantile-normalised, imputed  →  UMAP",
        ha="center", fontsize=8.2, color=MUT, style="italic")

# =====================================================================
# Tier 3 — the payoff (identical mean, opposite concordance)
# =====================================================================
ax.text(82, 37.6,
        "Result:  identical mean methylation / accessibility, opposite "
        "long-range concordance", fontsize=11.5, fontweight="bold",
        ha="center", va="center")

# ---- two toy cell states (left) -----------------------------------------
def state_box(x, y, w, h, color, name, grid, conc):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.0,rounding_size=1.6",
                 fc="white", ec=EDGE, lw=1.2, zorder=1))
    ax.add_patch(FancyBboxPatch((x + 2.4, y + h - 6.6), 20, 4.6,
                 boxstyle="round,pad=0.0,rounding_size=1.0",
                 fc=color, ec="none", alpha=0.16, zorder=2))
    ax.text(x + 12.4, y + h - 4.3, name, fontsize=10.2, fontweight="bold",
            color=color, ha="center", va="center", zorder=3)
    # 2 x 2 grid of within-cell contact loops (grid[row][col] = (left, right))
    cxs = [x + w * 0.30, x + w * 0.70]
    cys = [y + h - 13.5, y + h - 22.5]
    for r, row in zip(cys, grid):
        for c, (sl, sr) in zip(cxs, row):
            loop(c, r, sl, sr)
    ax.text(x + w / 2, y + 5.4, "mean HCG = 0.5", fontsize=8.6, color=MUT,
            ha="center", va="center")
    ax.text(x + w / 2, y + 2.4, f"concordance = {conc}", fontsize=10.2,
            fontweight="bold", color=INK, ha="center", va="center")


# state 1: both ends agree (co-methylated / co-unmethylated) -> concordance 1
state_box(5, 3, 33, 30, S1, "cell state 1",
          [[(1, 1), (1, 1)], [(0, 0), (0, 0)]], "1.0")
# state 2: ends disagree (mixed) -> concordance 0, same mean (0.5)
state_box(41, 3, 33, 30, S2, "cell state 2",
          [[(1, 0), (0, 1)], [(1, 0), (0, 1)]], "0.0")

# ---- bridge: apply the workflow -----------------------------------------
flow(75.5, 18, 82.5, 18, color=ACC, lw=2.2)
ax.text(79, 20.4, "workflow", ha="center", va="bottom", fontsize=7.8,
        color=ACC, style="italic")

# ---- two UMAPs (right): mean overlaps, concordance separates -------------
def umap_box(x, y, w, h, title, sep, caption):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.0,rounding_size=1.6",
                 fc="white", ec=EDGE, lw=1.2, zorder=1))
    ax.text(x + w / 2, y + h - 2.6, title, fontsize=9.6, fontweight="bold",
            ha="center", va="top")
    cxc, cyc = x + w / 2, y + h / 2 - 1.5
    rng2 = np.random.default_rng(4)
    if sep:
        a = rng2.normal([cxc - 6.5, cyc], [2.2, 3.0], (22, 2))
        b = rng2.normal([cxc + 6.5, cyc], [2.2, 3.0], (22, 2))
    else:
        a = rng2.normal([cxc, cyc], [3.4, 3.2], (22, 2))
        b = rng2.normal([cxc, cyc], [3.4, 3.2], (22, 2))
    ax.scatter(a[:, 0], a[:, 1], s=12, c=S1, edgecolors="none", zorder=4)
    ax.scatter(b[:, 0], b[:, 1], s=12, c=S2, edgecolors="none", zorder=4)
    ax.text(x + w / 2, y + 2.4, caption, fontsize=8.2, color=MUT,
            ha="center", va="center", style="italic")


umap_box(84, 3, 37, 30, "UMAP: mean HCG / GCH", False,
         "states overlap — not separable")
umap_box(124, 3, 37, 30, "UMAP: signed concordance", True,
         "states separate")

# =====================================================================
# flow arrows between tiers / panels
# =====================================================================
flow(XS[0] + PW, TIER1 + PH / 2, XS[1], TIER1 + PH / 2)          # P1 -> P2
flow(XS[1] + PW, TIER1 + PH / 2, XS[2], TIER1 + PH / 2)          # P2 -> P3
# wrap P3 -> P4: down into the gap, left, then into P4's clear top-left corner
_wrap = Path([(XS[2] + PW / 2, TIER1 - 0.3), (XS[2] + PW / 2, TIER1 - 2.3),
              (XS[0] + 4, TIER1 - 2.3), (XS[0] + 4, TIER2 + PH - 0.4)],
             [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])
ax.add_patch(FancyArrowPatch(path=_wrap, arrowstyle="-|>", mutation_scale=15,
                             lw=1.8, color=ACC, zorder=6))
flow(XS[0] + PW, TIER2 + PH / 2, XS[1], TIER2 + PH / 2)          # P4 -> P5
flow(XS[1] + PW, TIER2 + PH / 2, XS[2], TIER2 + PH / 2)          # P5 -> P6
flow(XS[2] + PW / 2, TIER2 - 0.3, 142, 34, rad=0.0, lw=1.8)     # P6 -> UMAPs

# =====================================================================
# legend
# =====================================================================
ax.add_patch(Circle((7, 0.9 + 0.0 * 1), 1.15, fc=FILL, ec=FILL, lw=1.4))
ax.add_patch(Circle((44, 1.0), 1.15, fc="white", ec=FILL, lw=1.4))
ax.text(9.4, 1.0, "methylated / accessible", fontsize=8.6, va="center")
ax.text(46.4, 1.0, "unmethylated / closed", fontsize=8.6, va="center")
ax.text(120, 1.0, "each loop = two loci in contact within one cell",
        fontsize=8.6, va="center", ha="center", color=MUT)

fig.savefig("figures/figure5a_scheme.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig("figures/figure5a_scheme.pdf", bbox_inches="tight",
            facecolor="white")
print("wrote figures/figure5a_scheme.png and .pdf")
