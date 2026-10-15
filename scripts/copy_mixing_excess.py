#!/usr/bin/env python3
"""What the two X copies' different levels add to X's contact concordance.

In a cell a contact's two ends sit on one copy of X.  The link-shuffle null
of annotation_class_concordance.py --chrx draws partners from other cells at
the partner's place -- from either copy.  So where the copies differ in
level, a real partner agrees with its anchor more than a donor does with no
coordination between them: per contact the expected ΔC = 2(xy - xy1) is
½ δ_a δ_p, δ the copies' difference in level at each end.  Over X's contacts
at distance s that is ½ E[δ(x) δ(x + s)]: the copies' difference correlated
with itself at the contact's distance.

allelic_imbalance.py's per-SNP table gives δ at each heterozygous SNP of a
clonal line (hap1 - hap2 = Xi - Xa when every cell silences the same X).
Pairs of SNPs at distance s estimate E[δ(x) δ(x + s)] without bias: each δ
is read on its own reads, so their noise does not multiply.  The
autosomes' SNPs, whose haplotypes do not differ (imprinting aside), give
the same with no copy difference, and X less autosomes is the prediction
for chrx_concordance.tsv's 'chrX − autosomes' (PAR1 and the escape genes
left out of X, as there).  Contacts are not spread over distance the way
SNP pairs are, so a band's prediction weighs its 0.1-decade bins by X's
contacts (--ps, haplotype_contacts_ps.tsv); intervals resample --block bp
blocks of the SNPs.

Units: δ is a methylation difference over sites, ΔC is computed on the
ends' posteriors.  With --haplotype-concordance (chrx_haplotype_concordance
.tsv) the run reports the copies' difference in posterior units over their
tagged contacts' ends against δ over the same part of X: their ratio k, by
which the prediction would scale as k².

    python scripts/copy_mixing_excess.py \\
        --allelic-snps allelic_imbalance_gm12878_elements/allelic_snps.tsv \\
        --concordance chrx_concordance.tsv \\
        --ps haplotype_contacts/haplotype_contacts_ps.tsv \\
        --haplotype-concordance chrx_haplotype_concordance.tsv \\
        --group GM12878 --genome hg38 --output-dir copy_mixing_excess
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance.xci import parse_escape, parse_par, x_part  # noqa: E402

logger = logging.getLogger("copy_mixing_excess")

#: 0.1-decade distance bins the prediction is made in, 1 kb to 10 Mb.
FINE = np.round(np.arange(3.0, 7.01, 0.1), 2)
_UNIT = {"bp": 1, "kb": 1_000, "mb": 1_000_000}


def parse_band(label: str):
    """'10kb-100kb' -> (10000, 100000); None for anything else."""
    m = re.fullmatch(r"\s*([\d.]+)\s*(bp|kb|mb)\s*-\s*([\d.]+)\s*(bp|kb|mb)\s*",
                     str(label).lower())
    if not m:
        return None
    return (float(m.group(1)) * _UNIT[m.group(2)],
            float(m.group(3)) * _UNIT[m.group(4)])


def lag_sums(pos, v, edges, block: int, offset: int = 0):
    """Over the pairs of points i < j with edges[k] <= pos_j - pos_i <
    edges[k + 1]: per bin k and block of the left point (pos // block +
    ``offset``), the sum of v_i v_j and the number of pairs.  Returns
    (blocks, sums, counts), sums and counts bins x blocks."""
    pos = np.asarray(pos, np.int64)
    v = np.asarray(v, float)
    o = np.argsort(pos, kind="stable")
    pos, v = pos[o], v[o]
    cs = np.concatenate([[0.0], np.cumsum(v)])
    blk, inv = np.unique(pos // block + offset, return_inverse=True)
    nb = len(edges) - 1
    S = np.zeros((nb, blk.size))
    N = np.zeros((nb, blk.size))
    for k in range(nb):
        a = np.searchsorted(pos, pos + edges[k], side="left")
        b = np.searchsorted(pos, pos + edges[k + 1], side="left")
        S[k] = np.bincount(inv, v * (cs[b] - cs[a]), blk.size)
        N[k] = np.bincount(inv, (b - a).astype(float), blk.size)
    return blk, S, N


def snp_delta(t: pd.DataFrame, ctx: str, min_reads: int) -> pd.DataFrame:
    """The SNPs with ``ctx``'s haplotype difference read on at least
    ``min_reads`` reads of each allele: chrom, pos, delta (hap1 - hap2),
    weight (the alleles' harmonic mean of reads)."""
    r1, r2 = t[f"{ctx}_reads_ref"], t[f"{ctx}_reads_alt"]
    d = t[f"{ctx}_hap1_minus_hap2"]
    ok = (r1 >= min_reads) & (r2 >= min_reads) & d.notna()
    u = t.loc[ok, ["chrom", "pos"]].copy()
    u["delta"] = d[ok].to_numpy(float)
    u["weight"] = (2 * r1[ok] * r2[ok] / (r1[ok] + r2[ok])).to_numpy(float)
    return u


def predict(u: pd.DataFrame, par, escape, block: int):
    """δ's products over SNP pairs per FINE bin and block: X outside PAR1
    and the escape genes, and the autosomes.  Returns {'X': (S, N),
    'autosomes': (S, N)} and X's SNPs' read-weighted mean δ."""
    edges = 10.0 ** FINE
    isx = u["chrom"].astype(str).str.replace("chr", "", regex=False) \
        .str.upper() == "X"
    x = u[isx]
    part = x_part(x["pos"].to_numpy(np.int64), par, escape, {})
    if part is not None:
        x = x[np.asarray(part) == "rest"]
    out = {}
    if len(x):
        _, S, N = lag_sums(x["pos"], x["delta"], edges, block)
        out["X"] = (S, N)
    auto = u[u["chrom"].astype(str).str.replace("chr", "", regex=False)
             .str.isdigit()]
    parts = [lag_sums(g["pos"], g["delta"], edges, block, offset=i * 10_000)
             for i, (_, g) in enumerate(auto.groupby("chrom", sort=True))]
    if parts:
        out["autosomes"] = (np.concatenate([p[1] for p in parts], 1),
                            np.concatenate([p[2] for p in parts], 1))
    mean_x = (float(np.average(x["delta"], weights=x["weight"]))
              if len(x) else np.nan)
    return out, mean_x, len(x)


def _curve(S, N):
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * S.sum(1) / N.sum(1)


def band_values(f, weights, bands):
    """Each band's prediction: its FINE bins' weighted mean."""
    lo = FINE[:-1]
    out = []
    for a, b in bands:
        m = ((10.0 ** lo >= a * (1 - 1e-9)) & (10.0 ** lo < b * (1 - 1e-9))
             & np.isfinite(f) & (weights > 0))
        out.append(float(np.sum(f[m] * weights[m]) / weights[m].sum())
                   if m.any() else np.nan)
    return np.array(out)


def prediction_table(sums, weights, bands, n_boot: int, seed: int):
    """X less autosomes per FINE bin and per band, with intervals over
    blocks resampled with replacement."""
    SX, NX = sums["X"]
    SA, NA = sums.get("autosomes", (np.zeros_like(SX), np.ones_like(NX)))

    def one(SX, NX, SA, NA):
        f = _curve(SX, NX) - np.nan_to_num(_curve(SA, NA))
        return f, band_values(f, weights, bands)
    f, bv = one(SX, NX, SA, NA)
    fa = _curve(SA, NA)
    rng = np.random.default_rng(seed)
    fb, bb = [], []
    for _ in range(n_boot):
        ix = rng.integers(0, SX.shape[1], SX.shape[1])
        ia = rng.integers(0, SA.shape[1], SA.shape[1])
        f_, b_ = one(SX[:, ix], NX[:, ix], SA[:, ia], NA[:, ia])
        fb.append(f_)
        bb.append(b_)
    with np.errstate(invalid="ignore"):
        flo, fhi = np.nanpercentile(np.array(fb), [2.5, 97.5], 0)
        blo, bhi = np.nanpercentile(np.array(bb), [2.5, 97.5], 0)
    fine = pd.DataFrame({"dist_lo": 10.0 ** FINE[:-1],
                         "dist_hi": 10.0 ** FINE[1:],
                         "snp_pairs_X": NX.sum(1),
                         "autosomes": fa, "predicted": f,
                         "predicted_lo": flo, "predicted_hi": fhi})
    band = pd.DataFrame({"dist_lo": [a for a, _ in bands],
                         "dist_hi": [b for _, b in bands],
                         "predicted": bv, "predicted_lo": blo,
                         "predicted_hi": bhi})
    return fine, band


def contact_weights(ps_path, group: str | None) -> np.ndarray:
    """X's contacts per FINE bin (both copies), from haplotype_contacts
    _ps.tsv; uniform when there is none."""
    w = np.ones(len(FINE) - 1)
    if ps_path is None:
        return w
    p = pd.read_csv(ps_path, sep="\t")
    p = p[p["set"] == "chrX"]
    if group is not None and "group" in p and (p["group"] == group).any():
        p = p[p["group"] == group]
    c = p.groupby(p["log10_lo"].round(2))["contacts"].sum()
    w = np.array([float(c.get(round(lo, 2), 0.0)) for lo in FINE[:-1]])
    return w if w.sum() > 0 else np.ones(len(FINE) - 1)


def observed_rows(path, contexts) -> pd.DataFrame:
    """chrx_concordance.tsv's 'chrX − autosomes' ΔC per group and band."""
    t = pd.read_csv(path, sep="\t")
    t = t[(t["comparison"] == "chrX − autosomes")
          & (t["statistic"] == "delta")
          & t["context"].isin(contexts)].copy()
    rng_ = t["band"].map(parse_band)
    t = t[rng_.notna()]
    t["dist_lo"] = [r[0] for r in rng_[rng_.notna()]]
    t["dist_hi"] = [r[1] for r in rng_[rng_.notna()]]
    return t[["context", "group", "band", "dist_lo", "dist_hi",
              "pairs_test", "diff", "diff_lo", "diff_hi"]]


def posterior_ratio(path, group: str, ctx: str, site_delta: float,
                    xi_hap: int):
    """The copies' level difference (Xi - Xa) over their tagged contacts'
    ends, in posterior units, against ``site_delta``: (k, k_lo, k_hi,
    posterior difference)."""
    t = pd.read_csv(path, sep="\t")
    t = t[(t["statistic"] == "level") & (t["context"] == ctx)
          & (t["group"] == group) & (t["comparison"] == "chrX: hap2 − hap1")
          & t["band"].astype(str).str.startswith("all")]
    if not len(t) or not np.isfinite(site_delta) or site_delta == 0:
        return None
    r = t.iloc[0]
    s = -1.0 if xi_hap == 1 else 1.0          # hap2 - hap1 -> Xi - Xa
    d = s * float(r["diff"])
    k_lo, k_hi = sorted((s * float(r["diff_lo"]) / site_delta,
                         s * float(r["diff_hi"]) / site_delta))
    return d / site_delta, k_lo, k_hi, d


def plot(pred: dict, obs: pd.DataFrame, path, group: str):
    """Per context, X less autosomes ΔC by distance: each observed group's
    bands (points, the band as the horizontal bar) and the prediction from
    ``group``'s copies (line, 95% band; diamonds, each band's mean)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ctxs = list(pred)
    fig, axes = plt.subplots(1, len(ctxs), figsize=(3.3 * len(ctxs), 2.8),
                             squeeze=False)
    gcol = {}
    palette = ["#222222", "#8c8c8c", "#e08214", "#5e3c99"]
    for ax, ctx in zip(axes[0], ctxs):
        fine, band = pred[ctx]
        xm = np.sqrt(fine["dist_lo"] * fine["dist_hi"])
        ok = np.isfinite(fine["predicted"])
        ax.plot(xm[ok], 100 * fine["predicted"][ok], color="#b2182b", lw=1,
                label=f"predicted from {group}'s copies")
        ax.fill_between(xm[ok], 100 * fine["predicted_lo"][ok],
                        100 * fine["predicted_hi"][ok], color="#b2182b",
                        alpha=0.18, lw=0)
        bm = np.sqrt(band["dist_lo"] * band["dist_hi"])
        ax.scatter(bm, 100 * band["predicted"], marker="D", s=14,
                   facecolor="white", edgecolor="#b2182b", zorder=3)
        o = obs[obs["context"] == ctx]
        groups = list(dict.fromkeys(o["group"]))
        for j, g in enumerate(groups):
            gcol.setdefault(g, palette[len(gcol) % len(palette)])
            u = o[o["group"] == g]
            xc = np.sqrt(u["dist_lo"] * u["dist_hi"]) * (1.12 ** (j - (len(
                groups) - 1) / 2))
            ax.errorbar(xc, 100 * u["diff"],
                        yerr=[100 * (u["diff"] - u["diff_lo"]),
                              100 * (u["diff_hi"] - u["diff"])],
                        fmt="o", ms=3, color=gcol[g], lw=0.8, capsize=1.5,
                        label=f"observed, {g}", zorder=4)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlim(1e3, 1e7)
        ax.set_xlabel("distance between the contact's ends, bp", fontsize=7)
        ax.set_ylabel("ΔC, chrX − autosomes (pp)", fontsize=7)
        ax.set_title(f"{ctx}-{ctx}", loc="left", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--allelic-snps", required=True,
                    help="allelic_imbalance.py's allelic_snps.tsv.")
    ap.add_argument("--concordance", required=True,
                    help="annotation_class_concordance.py --chrx's "
                         "chrx_concordance.tsv: the observed excess.")
    ap.add_argument("--ps", default=None,
                    help="haplotype_contacts.py's haplotype_contacts_ps.tsv: "
                         "X's contacts by distance, to weigh each band's "
                         "bins (uniform in log distance without).")
    ap.add_argument("--haplotype-concordance", default=None,
                    help="annotation_class_concordance.py's chrx_haplotype_"
                         "concordance.tsv: the copies' difference in "
                         "posterior units, for the ratio k.")
    ap.add_argument("--group", default="GM12878",
                    help="The clonal line whose SNPs give δ (its rows of "
                         "--allelic-snps).")
    ap.add_argument("--observed-groups", default=None,
                    help="Groups of --concordance to plot (comma-"
                         "separated; default every one but 'all cells').")
    ap.add_argument("--contexts", default="HCG,GCH")
    ap.add_argument("--xi-hap", type=int, default=1, choices=(1, 2),
                    help="The haplotype that is the inactive X in --group "
                         "(allelic_summary.tsv's xi_hap); orients δ as "
                         "Xi - Xa for the report.  The prediction does not "
                         "depend on it.")
    ap.add_argument("--genome", default="hg38")
    ap.add_argument("--chrx-par", default="auto")
    ap.add_argument("--chrx-escape", default="auto")
    ap.add_argument("--min-reads", type=int, default=1,
                    help="Reads of each allele a SNP needs (default 1).")
    ap.add_argument("--block", type=int, default=5_000_000,
                    help="Block, bp, the intervals resample (default 5 Mb).")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-format", default="pdf")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    contexts = [c.strip() for c in args.contexts.split(",") if c.strip()]
    par = parse_par(args.chrx_par, args.genome)
    escape = parse_escape(args.chrx_escape, args.genome, par)
    cols = ["group", "chrom", "pos"] + [
        f"{c}_{s}" for c in contexts
        for s in ("hap1_minus_hap2", "reads_ref", "reads_alt")]
    snps = pd.read_csv(args.allelic_snps, sep="\t", usecols=cols,
                       dtype={"chrom": str})
    snps = snps[snps["group"] == args.group]
    if not len(snps):
        sys.exit(f"--allelic-snps has no rows of group {args.group!r}")
    obs = observed_rows(args.concordance, contexts)
    if args.observed_groups:
        keep = [g.strip() for g in args.observed_groups.split(",")]
        obs = obs[obs["group"].isin(keep)]
    else:
        obs = obs[obs["group"] != "all cells"]
    bands = sorted({(a, b) for a, b in zip(obs["dist_lo"], obs["dist_hi"])})
    if not bands:
        bands = [(1e3, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 1e7)]
    weights = contact_weights(args.ps, args.group)
    sign = 1.0 if args.xi_hap == 1 else -1.0
    pred, fine_rows, band_rows = {}, [], []
    for ctx in contexts:
        u = snp_delta(snps, ctx, args.min_reads)
        sums, mean_x, n_x = predict(u, par, escape, args.block)
        if "X" not in sums:
            logger.warning(f"{ctx}: no SNP on X's rest; skipped")
            continue
        fine, band = prediction_table(sums, weights, bands, args.n_boot,
                                      args.seed)
        pred[ctx] = (fine, band)
        k = (posterior_ratio(args.haplotype_concordance, args.group, ctx,
                             sign * mean_x, args.xi_hap)
             if args.haplotype_concordance else None)
        for df, rows in ((fine, fine_rows), (band, band_rows)):
            d = df.copy()
            d.insert(0, "context", ctx)
            d["site_delta_xi_minus_xa"] = sign * mean_x
            d["snps_X"] = n_x
            if k is not None:
                d["posterior_over_site_k"] = k[0]
                d["k_lo"], d["k_hi"] = k[1], k[2]
            rows.append(d)
        o = obs[obs["context"] == ctx]
        msg = [f"{ctx}: {n_x:,} SNPs on X's rest, Xi - Xa {sign * mean_x:+.3f}"
               + (f"; posterior/site k = {k[0]:.2f} [{k[1]:.2f}, {k[2]:.2f}]"
                  if k is not None else "")]
        for (a, b), p, lo, hi in zip(bands, band["predicted"],
                                     band["predicted_lo"],
                                     band["predicted_hi"]):
            seen = o[(o["dist_lo"] == a) & (o["dist_hi"] == b)]
            obs_s = ", ".join(f"{g} {v:+.4f} [{l_:+.4f}, {h_:+.4f}]"
                              for g, v, l_, h_ in zip(
                                  seen["group"], seen["diff"],
                                  seen["diff_lo"], seen["diff_hi"]))
            msg.append(f"  {a:,.0f}-{b:,.0f} bp: predicted {p:+.4f} "
                       f"[{lo:+.4f}, {hi:+.4f}]; observed {obs_s}")
        logger.info("\n".join(msg))
    if not pred:
        sys.exit("no context had SNPs on X")
    pd.concat(fine_rows).to_csv(out / "copy_mixing_excess_fine.tsv",
                                sep="\t", index=False)
    bt = pd.concat(band_rows)
    bt = bt.merge(obs.rename(columns={"diff": "observed",
                                      "diff_lo": "observed_lo",
                                      "diff_hi": "observed_hi",
                                      "pairs_test": "contacts_X"}),
                  on=["context", "dist_lo", "dist_hi"], how="left")
    bt.to_csv(out / "copy_mixing_excess.tsv", sep="\t", index=False)
    plot(pred, obs, out / f"copy_mixing_excess.{args.plot_format}",
         args.group)
    return 0


if __name__ == "__main__":
    sys.exit(main())
