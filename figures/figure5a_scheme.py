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

fig = plt.figure(figsize=(16.4, 17.4), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 164)
ax.set_ylim(0, 174)
ax.axis("off")

# panel grid for the two workflow tiers (taller panels so the enlarged titles
# clear the in-panel notes)
PW, PH = 49, 37
XS = [5, 57.5, 110]
TIER1, TIER2 = 125, 84         # y of each tier's bottom edge
TIER3 = 41                     # the toy states + the matrix they produce
TIER4 = 5                      # what the matrix is used for


def panel(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.0,rounding_size=1.6",
                 fc=PANEL, ec=EDGE, lw=1.2, zorder=1))
    ax.text(x + w / 2, y + h - 2.4, title, fontsize=16.2, fontweight="bold",
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
ax.text(82, 170.0, "Long-range epigenetic concordance", fontsize=24.8,
        fontweight="bold", ha="center", va="center")
ax.text(82, 165.5,
        "a per-cell, per-bin signal that separates cell states even when mean "
        "methylation / accessibility is identical",
        fontsize=15.3, color=MUT, ha="center", va="center", style="italic")

# =====================================================================
# Tier 1, panel 1 — single-cell NOMe-HiC contact
# =====================================================================
ax0 = XS[0]
panel(ax0, TIER1, PW, PH, "Single-cell NOMe-HiC contact")
cy = TIER1 + 13
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
        ha="center", va="bottom", fontsize=13.2, color=ACC, style="italic")
ax.text(np.mean(liA), cy - 4.0, "locus $i$", ha="center", fontsize=13.8)
ax.text(np.mean(liB), cy - 4.0, "locus $j$", ha="center", fontsize=13.8)
ax.text(np.mean(liA), cy - 7.0, r"$k_i/n_i$", ha="center", fontsize=13.2,
        color=MUT)
ax.text(np.mean(liB), cy - 7.0, r"$k_j/n_j$", ha="center", fontsize=13.2,
        color=MUT)
ax.text(ax0 + PW / 2, TIER1 + 2.4, "GCH accessibility  ·  HCG (CpG) methylation",
        ha="center", fontsize=12.6, color=MUT)

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
            fontsize=16.5, ha="center", va="bottom", fontweight="bold")
ax.text((gx0 + gx1) / 2, gy0 - 3.2, r"$p=P(\mathrm{open}\,|\,k,n)$",
        ha="center", fontsize=15.3)
ax.text((gx0 + gx1) / 2, gy0 - 6.6, "Beta–Binomial mixture (EM)", ha="center",
        fontsize=12.6, color=MUT, style="italic")
ax.text(gx0 - 1.2, gy1, "density", rotation=90, va="top", ha="right",
        fontsize=12.0, color=MUT)
ax.text(gx1, gy0 - 0.8, "P(open)", va="top", ha="right", fontsize=12.0,
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
    ax.text(cx0 + 15, yy + 1.4, label, fontsize=14.1, fontweight="bold",
            va="center", color=INK)
    ax.text(cx0 + 15, yy - 2.0, expr, fontsize=12.9, va="center", color=MUT)
ax.text(cx0 + PW / 2, TIER1 + 2.2, "soft expected counts (no thresholding)",
        ha="center", fontsize=12.3, color=MUT, style="italic")

# =====================================================================
# Tier 2, panel 4 — the DISTANCE-POOLED expected pool
#
# Not "short-range as the baseline".  The expected pool is built from the
# bin's OWN read ends, re-paired: every non-contact end pair at or beyond
# d0 is pooled, so E carries the bin's marginal accessibility and its
# depth while carrying none of its coordination.  A short-range baseline
# would instead compare two different distance regimes.
# =====================================================================
dx = XS[0]
panel(dx, TIER2, PW, PH, "Distance-pooled expected")
# Bracket first, high enough to clear the title; then the axis well below
# it so the observed arcs above and the re-paired arcs below both fit
# INSIDE the panel.
by = TIER2 + 29.5
ax.plot([dx + 6, dx + 44], [by, by], color=INK, lw=1.1)
for xx in (dx + 6, dx + 44):
    ax.plot([xx, xx], [by, by - 1.3], color=INK, lw=1.1)
ax.text(dx + PW / 2, by + 0.9, "aggregate within 250-kb bin", ha="center",
        va="bottom", fontsize=12.0)

axis_y = TIER2 + 14.5
ax.annotate("", xy=(dx + 45, axis_y), xytext=(dx + 5, axis_y),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
# Below the axis and hard right, where neither the solid arcs above nor
# the dashed ones below reach.
ax.text(dx + 45.5, axis_y - 2.4, "distance", ha="right", va="top",
        fontsize=11.4, color=MUT)
thx = dx + 13
ax.plot([thx, thx], [axis_y - 8.0, axis_y + 6.5], color=ACC, lw=1.4, ls="--")
ax.text(thx, axis_y + 7.2, r"$d_0\,(\approx\!1\,\mathrm{kb})$", ha="center",
        va="bottom", fontsize=12.0, color=ACC)
# OBSERVED: real ligations, solid, above the axis.
ax.plot([dx + 17, dx + 43], [axis_y, axis_y], color=DNA, lw=2.4,
        solid_capstyle="round")
arc(dx + 19, dx + 29, axis_y, 4.4, color=ACC, lw=2.0)
arc(dx + 25, dx + 41, axis_y, 6.2, color=ACC, lw=2.0)
ax.text(dx + 31, axis_y + 10.4, "observed: linked contacts",
        ha="center", va="bottom", fontsize=11.7, color=ACC, fontweight="bold")
# EXPECTED: the SAME ends re-paired, dashed, below the axis.
for (a_, b_, h_) in ((dx + 18, dx + 34, 2.9), (dx + 22, dx + 27, 1.5),
                     (dx + 26, dx + 38, 2.2)):
    xm = (a_ + b_) / 2
    ax.add_patch(PathPatch(Path([(a_, axis_y - 0.9),
                                 (xm, axis_y - 0.9 - 2 * h_),
                                 (b_, axis_y - 0.9)],
                                [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                           fc="none", ec=MUT, lw=1.5, ls=(0, (3, 2)),
                           zorder=3))
ax.text(dx + PW / 2, TIER2 + 2.6,
        "expected: the SAME ends re-paired,\n"
        r"every separation $\geq d_0$ pooled",
        ha="center", va="bottom", fontsize=11.4, color=MUT, linespacing=1.25)

# =====================================================================
# Tier 2, panel 5 — the two statistics actually reported
#
# C is the magnitude: excess concordant mass over the expected.  The log
# odds ratio is the direction, referred to the SAME expected pool, which
# is what makes it marginal-free -- logit(F) - logit(F_exp), i.e. the log
# odds ratio of F against the log odds ratio of F_exp.
# =====================================================================
ex = XS[1]
panel(ex, TIER2, PW, PH, "Long-range concordance")
ax.text(ex + PW / 2, TIER2 + 29.0,
        r"$C=(O_{\mathrm{open}}+O_{\mathrm{closed}})-"
        r"(E_{\mathrm{open}}+E_{\mathrm{closed}})$",
        ha="center", fontsize=13.8)
ax.text(ex + PW / 2, TIER2 + 24.6,
        r"$F=\dfrac{O_{\mathrm{open}}}{O_{\mathrm{open}}+O_{\mathrm{closed}}}"
        r"\qquad F_{\mathrm{exp}}=\dfrac{E_{\mathrm{open}}}"
        r"{E_{\mathrm{open}}+E_{\mathrm{closed}}}$",
        ha="center", fontsize=12.4)
ax.text(ex + PW / 2, TIER2 + 19.4,
        r"$L=\mathrm{logit}\,F-\mathrm{logit}\,F_{\mathrm{exp}}"
        r"=\log\dfrac{O_{\mathrm{open}}}{O_{\mathrm{closed}}}"
        r"-\log\dfrac{E_{\mathrm{open}}}{E_{\mathrm{closed}}}$",
        ha="center", fontsize=13.2)
ax.text(ex + PW / 2, TIER2 + 15.4,
        "C = magnitude   ·   L = direction (marginal-free)",
        ha="center", fontsize=11.7, color=MUT, style="italic")
hm_x0, hm_y0, hm_w, hm_h = ex + 11, TIER2 + 2.6, 27, 10.5
rng = np.random.default_rng(1)
nC, nB = 12, 18
block = np.vstack([rng.normal(0.5, 0.35, (6, nB)),
                   rng.normal(-0.5, 0.35, (6, nB))])
block[:, ::3] += rng.normal(0, 0.5, (nC, len(range(0, nB, 3))))
ax.imshow(block, cmap="RdBu_r", vmin=-1.4, vmax=1.4, aspect="auto",
          extent=[hm_x0, hm_x0 + hm_w, hm_y0, hm_y0 + hm_h], zorder=3)
ax.add_patch(Rectangle((hm_x0, hm_y0), hm_w, hm_h, fc="none", ec=INK, lw=1.0,
                       zorder=4))
ax.text(hm_x0 + hm_w / 2, hm_y0 - 1.6, "bins", ha="center", fontsize=12.0,
        color=MUT)
ax.text(hm_x0 - 1.4, hm_y0 + hm_h / 2, "cells", rotation=90, va="center",
        ha="right", fontsize=12.0, color=MUT)
ax.text(hm_x0 + hm_w + 1.4, hm_y0 + hm_h, r"$+$", color="#B2182B",
        fontsize=15.0, va="top", fontweight="bold")
ax.text(hm_x0 + hm_w + 1.4, hm_y0, r"$-$", color="#2166AC", fontsize=15.0,
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
        ha="center", fontsize=12.6, color=INK, linespacing=1.3)
ax.text(fx + PW / 2, TIER2 + 3.0,
        "quantile-normalised, imputed  →  UMAP",
        ha="center", fontsize=12.3, color=MUT, style="italic")

# =====================================================================
# Tier 3 — the payoff (identical mean, opposite concordance)
# =====================================================================
ax.text(74, TIER3 + 34.5,
        "Same mean methylation / accessibility, opposite long-range concordance",
        fontsize=16.4, fontweight="bold", ha="center", va="center")


def state_box(x, y, w, h, color, name, grid, conc):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.0,rounding_size=1.6",
                 fc="white", ec=EDGE, lw=1.2, zorder=1))
    ax.add_patch(FancyBboxPatch((x + 2.4, y + h - 6.0), 20, 4.4,
                 boxstyle="round,pad=0.0,rounding_size=1.0",
                 fc=color, ec="none", alpha=0.16, zorder=2))
    ax.text(x + 12.4, y + h - 3.8, name, fontsize=14.6, fontweight="bold",
            color=color, ha="center", va="center", zorder=3)
    cxs = [x + w * 0.30, x + w * 0.70]
    cys = [y + h - 12.6, y + h - 20.4]
    for r, row in zip(cys, grid):
        for c, (sl, sr) in zip(cxs, row):
            loop(c, r, sl, sr)
    ax.text(x + w / 2, y + 3.4, f"concordance = {conc}", fontsize=14.6,
            fontweight="bold", color=INK, ha="center", va="center")


state_box(5, TIER3, 33, 30, S1, "cell state 1",
          [[(1, 1), (1, 1)], [(0, 0), (0, 0)]], "1.0")
state_box(41, TIER3, 33, 30, S2, "cell state 2",
          [[(1, 0), (0, 1)], [(1, 0), (0, 1)]], "0.0")

# The concordance matrix is what every downstream analysis reads, so the
# panel on the right is the matrix itself rather than one of its outputs.
ax.add_patch(FancyBboxPatch((86, TIER3), 58, 30,
             boxstyle="round,pad=0.0,rounding_size=1.6",
             fc="white", ec=EDGE, lw=1.2, zorder=1))
ax.text(115, TIER3 + 27.4, "per-cell, per-bin  $C$  and  $L$",
        fontsize=14.4, fontweight="bold", ha="center", va="top")
_mx, _my, _mw, _mh = 96, TIER3 + 7.5, 38, 14.5
rng3 = np.random.default_rng(11)
_blk = np.vstack([rng3.normal(0.55, 0.34, (7, 26)),
                  rng3.normal(-0.55, 0.34, (7, 26))])
_blk[:, ::4] += rng3.normal(0, 0.55, (14, len(range(0, 26, 4))))
ax.imshow(_blk, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto",
          extent=[_mx, _mx + _mw, _my, _my + _mh], zorder=3)
ax.add_patch(Rectangle((_mx, _my), _mw, _mh, fc="none", ec=INK, lw=1.0,
                       zorder=4))
ax.text(_mx + _mw / 2, _my - 1.6, "bins", ha="center", fontsize=11.7,
        color=MUT)
ax.text(_mx - 1.4, _my + _mh / 2, "cells", rotation=90, va="center",
        ha="right", fontsize=11.7, color=MUT)
ax.text(115, TIER3 + 3.2, "the input to every analysis below",
        fontsize=12.0, color=MUT, ha="center", va="center", style="italic")
flow(76, TIER3 + 15, 84, TIER3 + 15, color=ACC, lw=2.2)
ax.text(80, TIER3 + 17.4, "workflow", ha="center", va="bottom",
        fontsize=11.7, color=ACC, style="italic")

# =====================================================================
# Tier 4 — what the matrix is used for
#
# Five outputs on one strip, because the point is that they all come from
# the SAME matrix: an embedding, a differential test and its volcano, the
# bins those calls make, a profile around annotated regions, and the
# pathways the genes in those bins fall into.
# =====================================================================
ax.text(74, TIER4 + 33.0, "Downstream analyses", fontsize=16.4,
        fontweight="bold", ha="center", va="center")

OW, OGAP = 26.4, 2.6
OX0 = 5.0
_orng = np.random.default_rng(21)


def out_box(k, title, caption):
    x = OX0 + k * (OW + OGAP)
    y = TIER4
    ax.add_patch(FancyBboxPatch((x, y), OW, 29,
                 boxstyle="round,pad=0.0,rounding_size=1.4",
                 fc="white", ec=EDGE, lw=1.2, zorder=1))
    ax.text(x + OW / 2, y + 27.0, title, fontsize=12.9, fontweight="bold",
            ha="center", va="top")
    ax.text(x + OW / 2, y + 2.2, caption, fontsize=10.8, color=MUT,
            ha="center", va="center", style="italic")
    return x, y


# --- 1. UMAP ---------------------------------------------------------
x, y = out_box(0, "UMAP", "cell states separate")
cxc, cyc = x + OW / 2, y + 14.5
a = _orng.normal([cxc - 4.6, cyc], [2.0, 2.8], (22, 2))
b = _orng.normal([cxc + 4.6, cyc], [2.0, 2.8], (22, 2))
ax.scatter(a[:, 0], a[:, 1], s=8, c=S1, edgecolors="none", zorder=4)
ax.scatter(b[:, 0], b[:, 1], s=8, c=S2, edgecolors="none", zorder=4)

# --- 2. heatmap ------------------------------------------------------
x, y = out_box(1, "Heatmap", "differential bins x cells")
hx, hy, hw, hh = x + 4.5, y + 7.0, OW - 9, 15
_hb = np.vstack([_orng.normal(0.6, 0.3, (8, 22)),
                 _orng.normal(-0.6, 0.3, (8, 22))])
ax.imshow(_hb, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto",
          extent=[hx, hx + hw, hy, hy + hh], zorder=3)
ax.add_patch(Rectangle((hx, hy), hw, hh, fc="none", ec=INK, lw=1.0, zorder=4))

# --- 3. volcano ------------------------------------------------------
x, y = out_box(2, "Volcano", "effect size vs FDR")
vx, vy, vw, vh = x + 5.0, y + 6.5, OW - 10, 15.5
ax.plot([vx, vx + vw], [vy, vy], color=INK, lw=1.0, zorder=4)
ax.plot([vx + vw / 2, vx + vw / 2], [vy, vy + vh], color="#9AA0A6", lw=0.9,
        zorder=3)
_vx = _orng.normal(0, 1, 260)
_vy = np.abs(_orng.normal(0, 1, 260)) * 0.8 + 0.35 * np.abs(_vx) ** 2
_sig = (np.abs(_vx) > 1.5) & (_vy > 1.6)
_px = vx + vw / 2 + _vx / 3.4 * (vw / 2)
_py = vy + np.clip(_vy / 4.6, 0, 1) * vh
ax.scatter(_px[~_sig], _py[~_sig], s=3.2, c="#C6CBD1", edgecolors="none",
           zorder=4)
ax.scatter(_px[_sig & (_vx > 0)], _py[_sig & (_vx > 0)], s=4.6, c="#D1495B",
           edgecolors="none", zorder=5)
ax.scatter(_px[_sig & (_vx < 0)], _py[_sig & (_vx < 0)], s=4.6, c="#2C6FAF",
           edgecolors="none", zorder=5)
ax.plot([vx, vx + vw], [vy + vh * 0.42] * 2, color=MUT, lw=0.9,
        ls=(0, (3, 2)), zorder=4)

# --- 4. scaled profile -----------------------------------------------
x, y = out_box(3, "Scaled profile", "PMD / LAD / RT domains")
px0, py0, pw, ph = x + 4.5, y + 7.0, OW - 9, 15
ax.plot([px0, px0 + pw], [py0, py0], color=INK, lw=1.0, zorder=4)
for xx in (px0 + pw * 0.28, px0 + pw * 0.72):
    ax.plot([xx, xx], [py0, py0 + ph], color=MUT, lw=0.9, ls=(0, (3, 2)),
            zorder=3)
_t = np.linspace(0, 1, 120)
_shape = np.where((_t > 0.28) & (_t < 0.72), 0.30, 0.78)
for col, off, amp in ((S2, 0.10, 1.0), (S1, -0.06, 0.75)):
    _yv = (_shape * amp + off) + _orng.normal(0, 0.022, _t.size)
    ax.plot(px0 + _t * pw, py0 + np.clip(_yv, 0.04, 0.97) * ph, color=col,
            lw=1.7, zorder=5)
ax.text(px0 + pw / 2, py0 - 1.5, "start        end", ha="center",
        fontsize=9.6, color=MUT)

# --- 5. KEGG / GO ----------------------------------------------------
x, y = out_box(4, "KEGG / GO", "genes in differential bins")
bx, by0, bw = x + 5.0, y + 7.5, OW - 12
for k, frac in enumerate((0.95, 0.78, 0.62, 0.48, 0.33)):
    yy = by0 + (4 - k) * 2.9
    ax.add_patch(Rectangle((bx, yy), bw * frac, 2.1, fc=ACC,
                           ec="none", alpha=0.35 + 0.13 * k, zorder=4))
ax.plot([bx, bx], [by0 - 0.6, by0 + 14.4], color=INK, lw=1.0, zorder=5)
ax.text(bx + bw / 2, by0 - 2.2, r"$-\log_{10}$ FDR", ha="center",
        fontsize=9.6, color=MUT)

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
flow(XS[2] + PW / 2, TIER2 - 0.3, 115, TIER3 + 30.4, rad=0.0, lw=1.8)
flow(115, TIER3 - 0.3, 74, TIER4 + 29.6, rad=0.0, lw=1.8)

# =====================================================================
# legend
# =====================================================================
ax.add_patch(Circle((7, 1.6), 1.3, fc=FILL, ec=FILL, lw=1.4))
ax.add_patch(Circle((52, 1.6), 1.3, fc="white", ec=FILL, lw=1.4))
ax.text(9.6, 1.6, "methylated / accessible", fontsize=12.6, va="center")
ax.text(54.6, 1.6, "unmethylated / closed", fontsize=12.6, va="center")
ax.text(126, 1.6, "each loop = two loci in contact within one cell",
        fontsize=12.6, va="center", ha="center", color=MUT)

fig.savefig("figures/figure5a_scheme.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig("figures/figure5a_scheme.pdf", bbox_inches="tight",
            facecolor="white")
print("wrote figures/figure5a_scheme.png and .pdf")
