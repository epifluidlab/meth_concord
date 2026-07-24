#!/usr/bin/env python3
"""Functional enrichment of the cell-type-specific marker bins.

Takes the BED files written by ``celltype_marker_bins.py``
(``bin_scores_<ctx>.all.bed``) and asks WHERE the significant (cell-type-
specific) bins are and WHAT they mean:

  * chromosomal distribution  -- are the significant bins over-represented on
    particular chromosomes?
  * genomic features          -- promoter / gene-body / intergenic coverage
  * gene overlap              -- which genes the significant bins cover
  * chromatin states          -- ChromHMM state coverage enrichment
  * other annotations         -- any ``--annotation LABEL=BED`` (enhancers, CpG
    islands, TF peaks, ...)
  * GO / KEGG / gene sets      -- over-representation of the covered genes

Every analysis is OPTIONAL and runs only when its annotation is supplied, so the
script degrades gracefully; results go to per-context TSVs and a multi-page PDF.

Foreground vs background.  A bin is FOREGROUND if it was ``selected`` for the
heatmap (or, with ``--fdr``, if ``celltype_fdr`` is below the cut); the rest are
BACKGROUND.  The universe is every bin in the input, so the enrichment is
unbiased by the bin-selection itself.

Enrichment model.  The bins are large (250 kb by default), so the natural
question is COVERAGE, not point overlap: for each region annotation the fraction
of a bin covered is compared foreground-vs-background with a two-sided
Mann-Whitney U test, reported with the log2 fold-change of the mean coverage and
a BH-FDR across categories (positive log2FC = enriched, negative = depleted).
Gene-set (GO/KEGG) enrichment is a hypergeometric over-representation test of the
foreground-bin genes against the universe of all-bin genes.

Chromosome names are matched with the leading ``chr`` stripped, so ``chr1`` and
``1`` are treated as the same contig.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_EPS = 1e-9


# --------------------------------------------------------------- small utils --

def _canon(chrom) -> str:
    """Contig name with a leading 'chr'/'Chr' stripped, so chr1 == 1."""
    s = str(chrom)
    return s[3:] if s[:3].lower() == "chr" else s


def _chrom_key(chrom):
    """Sort key giving a natural chromosome order (1,2,...,10,...,X,Y,M)."""
    c = _canon(chrom)
    return (0, int(c)) if c.isdigit() else (1, c)


def _bh(p) -> np.ndarray:
    """Benjamini-Hochberg FDR over the finite p-values; NaN in -> NaN out."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if idx.size == 0:
        return out
    pv = p[idx]
    order = np.argsort(pv)
    m = pv.size
    q = pv[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[idx[order]] = np.clip(q, 0.0, 1.0)
    return out


def _merge(iv: np.ndarray) -> np.ndarray:
    """Union of (start, end) intervals -> sorted, non-overlapping array."""
    if len(iv) == 0:
        return iv.reshape(0, 2)
    iv = iv[np.argsort(iv[:, 0])]
    out = [iv[0].tolist()]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return np.asarray(out, dtype=np.int64)


# ------------------------------------------------------------------ readers --

def _read_marker_bed(path: str) -> pd.DataFrame:
    """The celltype_marker_bins BED (has a header: chrom/start/end/stats)."""
    df = pd.read_csv(path, sep="\t", dtype={"chrom": str})
    need = {"chrom", "start", "end"}
    if not need <= set(df.columns):
        sys.exit(f"{path}: expected a header with chrom/start/end "
                 f"(the bin_scores_<ctx>.all.bed from celltype_marker_bins.py); "
                 f"got columns {list(df.columns)}.")
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return df


def _read_bed(path: str, want_name=False):
    """A plain (headerless) BED: chrom, start, end[, name/state]."""
    df = pd.read_csv(path, sep="\t", header=None, comment="#",
                     compression="infer", dtype={0: str})
    out = pd.DataFrame({"chrom": df[0].astype(str),
                        "start": df[1].astype(int), "end": df[2].astype(int)})
    if want_name:
        out["name"] = (df[3].astype(str) if df.shape[1] > 3
                       else out["chrom"] + ":" + out["start"].astype(str))
    return out


def _read_genes(path: str) -> pd.DataFrame:
    """Gene bodies from a GTF/GFF ('gene' features) or a BED6 (name=col4,
    strand=col6).  Returns chrom/start/end/name/strand, 0-based half-open."""
    p = str(path).lower()
    if p.endswith((".gtf", ".gtf.gz", ".gff", ".gff3", ".gff.gz", ".gff3.gz")):
        g = pd.read_csv(path, sep="\t", comment="#", header=None,
                        compression="infer", dtype={0: str},
                        names=["chrom", "src", "feature", "start", "end",
                               "score", "strand", "frame", "attr"])
        g = g[g["feature"] == "gene"]
        if g.empty:
            sys.exit(f"{path}: no 'gene' features found in the GTF.")

        def _name(a):
            m = (re.search(r'gene_name "([^"]+)"', a)
                 or re.search(r'gene_id "([^"]+)"', a))
            return m.group(1) if m else None
        out = pd.DataFrame({"chrom": g["chrom"].astype(str),
                            "start": g["start"].astype(int) - 1,  # GTF 1-based
                            "end": g["end"].astype(int),
                            "name": g["attr"].map(_name),
                            "strand": g["strand"].astype(str)})
        return out.dropna(subset=["name"]).reset_index(drop=True)
    df = pd.read_csv(path, sep="\t", header=None, comment="#",
                     compression="infer", dtype={0: str})
    return pd.DataFrame({
        "chrom": df[0].astype(str), "start": df[1].astype(int),
        "end": df[2].astype(int),
        "name": (df[3].astype(str) if df.shape[1] > 3
                 else df[0].astype(str) + ":" + df[1].astype(str)),
        "strand": (df[5].astype(str) if df.shape[1] > 5 else "+")})


def _read_gmt(path: str) -> dict:
    """MSigDB/Enrichr GMT -> {term: set(genes)} (genes upper-cased for join)."""
    sets = {}
    opener = __import__("gzip").open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0]
            genes = {g.strip().upper() for g in parts[2:] if g.strip()}
            if genes:
                sets[term] = genes
    return sets


# ------------------------------------------------------ interval operations --

def _coverage(bins: pd.DataFrame, annot: pd.DataFrame) -> np.ndarray:
    """Per-bin fraction (0-1) covered by the UNION of the annotation intervals.

    Robust to overlapping annotation intervals (they are merged first)."""
    frac = np.zeros(len(bins), dtype=float)
    if annot is None or len(annot) == 0:
        return frac
    width = (bins["end"].values - bins["start"].values).astype(float)
    merged = {c: _merge(g[["start", "end"]].values.astype(np.int64))
              for c, g in annot.groupby(annot["chrom"].map(_canon))}
    bc = bins["chrom"].map(_canon).values
    bs = bins["start"].values
    be = bins["end"].values
    for i in range(len(bins)):
        m = merged.get(bc[i])
        if m is None or len(m) == 0:
            continue
        starts, ends = m[:, 0], m[:, 1]
        lo = int(np.searchsorted(ends, bs[i], "right"))
        hi = int(np.searchsorted(starts, be[i], "right"))
        if hi <= lo:
            continue
        ov = np.minimum(be[i], ends[lo:hi]) - np.maximum(bs[i], starts[lo:hi])
        frac[i] = ov[ov > 0].sum() / width[i] if width[i] > 0 else 0.0
    return np.clip(frac, 0.0, 1.0)


def _genes_over_bins(bins: pd.DataFrame, genes: pd.DataFrame) -> set:
    """Names of genes whose body overlaps ANY of the (non-overlapping) bins."""
    hits = set()
    idx = {}
    for c, g in bins.groupby(bins["chrom"].map(_canon)):
        v = g[["start", "end"]].values
        idx[c] = (np.sort(v[:, 0]), np.sort(v[:, 1]))
    for c, g in genes.groupby(genes["chrom"].map(_canon)):
        if c not in idx:
            continue
        st, en = idx[c]
        gs = g["start"].values
        ge = g["end"].values
        n_ov = (np.searchsorted(st, ge, "right")
                - np.searchsorted(en, gs, "right"))
        hits.update(g["name"].values[n_ov > 0])
    return hits


# ------------------------------------------------------------- enrichment ----

def _mwu_row(label, cov, fg):
    from scipy.stats import mannwhitneyu
    a, b = cov[fg], cov[~fg]
    m_fg = float(a.mean()) if a.size else np.nan
    m_bg = float(b.mean()) if b.size else np.nan
    lfc = float(np.log2((m_fg + _EPS) / (m_bg + _EPS)))
    p = np.nan
    if a.size and b.size:
        try:                      # MWU is valid when the groups differ, even if
            p = float(mannwhitneyu(  # each is internally constant (fg=1, bg=0);
                a, b, alternative="two-sided").pvalue)  # it only errors when
        except ValueError:        # EVERY value (both groups) is identical.
            p = np.nan
    return {"annotation": label, "mean_frac_fg": m_fg, "mean_frac_bg": m_bg,
            "log2_fold_change": lfc, "pvalue": p}


def _region_enrichment(bins, fg, categories):
    """MWU coverage enrichment (fg vs bg) for a dict {label: intervals_df}."""
    rows, covs = [], {}
    for label, iv in categories.items():
        cov = _coverage(bins, iv)
        covs[label] = cov
        rows.append(_mwu_row(label, cov, fg))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["fdr"] = _bh(df["pvalue"].values)
        df = df.sort_values("log2_fold_change", ascending=False).reset_index(
            drop=True)
    return df, covs


def _chrom_enrichment(bins, fg):
    """Per-chromosome Fisher enrichment of the foreground bins."""
    from scipy.stats import fisher_exact
    rows = []
    chroms = bins["chrom"].values
    n_fg = int(fg.sum())
    for c in sorted(pd.unique(chroms), key=_chrom_key):
        onc = chroms == c
        a = int((fg & onc).sum())
        b = int((fg & ~onc).sum())
        cc = int((~fg & onc).sum())
        d = int((~fg & ~onc).sum())
        try:
            orr, p = fisher_exact([[a, b], [cc, d]])
        except Exception:
            orr, p = np.nan, np.nan
        rows.append({"chrom": c, "n_fg": a, "n_bins": int(onc.sum()),
                     "frac_of_fg": a / n_fg if n_fg else np.nan,
                     "odds_ratio": float(orr), "pvalue": float(p)})
    df = pd.DataFrame(rows)
    df["fdr"] = _bh(df["pvalue"].values)
    return df


def _ora(fg_genes, universe_genes, gene_sets, min_size, max_size):
    """Hypergeometric over-representation of fg genes vs the bin-gene universe."""
    from scipy.stats import hypergeom
    universe = {g.upper() for g in universe_genes}
    fg = {g.upper() for g in fg_genes} & universe
    big_n, small_n = len(universe), len(fg)
    if small_n == 0 or big_n == 0:
        return pd.DataFrame()
    rows = []
    for term, genes in gene_sets.items():
        s = genes & universe
        big_k = len(s)
        if big_k < min_size or big_k > max_size:
            continue
        k = len(fg & s)
        if k == 0:
            continue
        p = float(hypergeom.sf(k - 1, big_n, big_k, small_n))
        fold = (k / small_n) / (big_k / big_n) if big_k else np.nan
        rows.append({"term": term, "n_overlap": k, "set_size": big_k,
                     "fg_genes": small_n, "universe": big_n,
                     "fold_enrichment": float(fold), "pvalue": p,
                     "genes": ",".join(sorted(fg & s))})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["fdr"] = _bh(df["pvalue"].values)
        df = df.sort_values("pvalue").reset_index(drop=True)
    return df


# --------------------------------------------------------------- plotting ----

def _stars(q):
    return ("***" if q < 1e-3 else "**" if q < 1e-2 else "*" if q < 0.05
            else "")


def _barh_region(ax, df, title):
    """Horizontal bars of log2 fold-change, coloured by direction, FDR-starred."""
    if df is None or df.empty:
        ax.text(0.5, 0.5, f"{title}\n(no data)", ha="center", va="center")
        ax.axis("off")
        return
    d = df.iloc[::-1]
    y = np.arange(len(d))
    vals = d["log2_fold_change"].values
    colors = ["#c0392b" if v >= 0 else "#2c6fbb" for v in vals]
    ax.barh(y, vals, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(d["annotation"], fontsize=8)
    ax.axvline(0, color="0.4", lw=0.8)
    for i, (v, q) in enumerate(zip(vals, d["fdr"].values)):
        s = _stars(q) if np.isfinite(q) else ""
        if s:
            ax.text(v + (0.02 if v >= 0 else -0.02), i, s, va="center",
                    ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel("log2 fold-change of mean coverage (fg / bg)")
    ax.set_title(title, fontsize=11)


def _barh_ora(ax, df, title, max_terms):
    if df is None or df.empty:
        ax.text(0.5, 0.5, f"{title}\n(no enriched terms)", ha="center",
                va="center")
        ax.axis("off")
        return
    d = df.head(max_terms).iloc[::-1]
    y = np.arange(len(d))
    vals = -np.log10(np.clip(d["fdr"].values, 1e-300, 1.0))
    ax.barh(y, vals, color="#3b7a57", edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([t[:48] for t in d["term"]], fontsize=7.5)
    ax.axvline(-np.log10(0.05), color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("-log10 BH-FDR")
    ax.set_title(title, fontsize=11)


def _chrom_page(ax, df):
    y = np.arange(len(df))
    sig = df["fdr"].values < 0.05
    ax.barh(y, df["frac_of_fg"].values,
            color=np.where(sig, "#c0392b", "#8a8a8a"))
    ax.set_yticks(y)
    ax.set_yticklabels(df["chrom"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of significant bins")
    ax.set_title("Chromosomal distribution (red = Fisher FDR < 0.05)",
                 fontsize=11)


# ------------------------------------------------------------------- driver --

def _run_context(name, bed_path, args, genes, gene_features, chromhmm,
                 annotations, gene_sets, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    bins = _read_marker_bed(bed_path)
    if args.fdr is not None and "celltype_fdr" in bins.columns:
        fg = (bins["celltype_fdr"].values < args.fdr)
        fg_desc = f"celltype_fdr < {args.fdr}"
    elif "selected" in bins.columns:
        fg = bins["selected"].astype(str).str.lower().isin(
            ["true", "1", "yes"]).values
        fg_desc = "selected == True"
    else:
        fg = np.ones(len(bins), bool)
        fg_desc = "ALL bins (no 'selected' column; provide the .all.bed)"
    n_fg, n_bg = int(fg.sum()), int((~fg).sum())
    print(f"\n=== {name}: {len(bins)} bins, {n_fg} foreground ({fg_desc}), "
          f"{n_bg} background ===")
    if n_fg == 0:
        print(f"  {name}: no foreground bins — skipping.", file=sys.stderr)
        return None

    tsv = lambda s: out_dir / f"{name}_{s}.tsv"
    summary = {"n_bins": len(bins), "n_foreground": n_fg, "foreground": fg_desc}
    pages = []          # (kind, payload) to render into the PDF

    # (1) chromosomal distribution -----------------------------------------
    chrom_df = _chrom_enrichment(bins, fg)
    chrom_df.to_csv(tsv("chrom_enrichment"), sep="\t", index=False)
    pages.append(("chrom", chrom_df))

    # (2) genomic features + gene overlap ----------------------------------
    fg_genes = bg_genes = None
    if genes is not None:
        feat_df, _ = _region_enrichment(bins, fg, gene_features)
        feat_df.to_csv(tsv("feature_enrichment"), sep="\t", index=False)
        pages.append(("region", ("Genomic features", feat_df)))
        bg_genes = _genes_over_bins(bins, genes)                 # universe
        fg_genes = _genes_over_bins(bins[fg], genes)
        pd.DataFrame({"gene": sorted(fg_genes)}).to_csv(
            tsv("foreground_genes"), sep="\t", index=False)
        summary["n_foreground_genes"] = len(fg_genes)
        summary["n_universe_genes"] = len(bg_genes)
        print(f"  genes covered: {len(fg_genes)} foreground / "
              f"{len(bg_genes)} universe -> {tsv('foreground_genes').name}")

    # (3) chromatin states -------------------------------------------------
    if chromhmm is not None:
        states = {f"state:{s}": g for s, g in chromhmm.groupby("name")}
        hmm_df, _ = _region_enrichment(bins, fg, states)
        hmm_df.to_csv(tsv("chromhmm_enrichment"), sep="\t", index=False)
        pages.append(("region", ("Chromatin-state coverage", hmm_df)))

    # (4) other annotations ------------------------------------------------
    if annotations:
        ann_df, _ = _region_enrichment(bins, fg, annotations)
        ann_df.to_csv(tsv("annotation_enrichment"), sep="\t", index=False)
        pages.append(("region", ("Other annotations", ann_df)))

    # (5) gene-set (GO / KEGG / ...) over-representation --------------------
    if gene_sets and fg_genes is not None:
        for label, sets in gene_sets.items():
            ora = _ora(fg_genes, bg_genes, sets, args.min_gene_set,
                       args.max_gene_set)
            ora.to_csv(tsv(f"{label}_enrichment"), sep="\t", index=False)
            n_sig = int((ora["fdr"] < 0.05).sum()) if not ora.empty else 0
            print(f"  {label}: {n_sig} terms at FDR<0.05 -> "
                  f"{tsv(label + '_enrichment').name}")
            pages.append(("ora", (f"{label} (top {args.max_terms})", ora)))
    elif gene_sets:
        print("  gene sets given but no --gene-annotation -> skipping GO/KEGG.",
              file=sys.stderr)

    # ---- render the multi-page PDF ----
    pdf_path = out_dir / f"{name}_enrichment.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 5))
        fig.text(0.5, 0.9, f"Marker-bin enrichment — {name.upper()}",
                 ha="center", fontsize=15)
        lines = [f"bins: {len(bins)}   foreground: {n_fg} ({fg_desc})   "
                 f"background: {n_bg}"]
        if "n_foreground_genes" in summary:
            lines.append(f"genes covered: {summary['n_foreground_genes']} "
                         f"foreground / {summary['n_universe_genes']} universe")
        lines.append("")
        lines.append("Coverage enrichment: two-sided Mann-Whitney U (fg vs bg),")
        lines.append("log2 fold-change of mean coverage, BH-FDR.  Gene sets:")
        lines.append("hypergeometric over-representation vs the bin-gene universe.")
        fig.text(0.1, 0.68, "\n".join(lines), fontsize=9, va="top")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, max(3, 0.28 * len(chrom_df) + 1.5)))
        _chrom_page(ax, chrom_df)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for kind, payload in pages:
            if kind == "chrom":
                continue
            if kind == "region":
                title, df = payload
                fig, ax = plt.subplots(
                    figsize=(8.5, max(3, 0.32 * len(df) + 1.5)))
                _barh_region(ax, df, title)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            elif kind == "ora":
                title, df = payload
                rows = min(len(df), args.max_terms) if df is not None else 0
                fig, ax = plt.subplots(figsize=(8.5, max(3, 0.34 * rows + 1.5)))
                _barh_ora(ax, df, title, args.max_terms)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    print(f"  report -> {pdf_path.name}")
    return summary


def _parse_pairs(items, what):
    """['a=x','y.bed'] -> {'a':'x','y':'y.bed'}; a bare value is keyed by its
    filename stem, so multiple bare paths do not collide."""
    out = {}
    for it in (items or []):
        if "=" in it:
            k, v = it.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out[Path(it).stem] = it.strip()
    for v in out.values():
        if not Path(v).exists():
            sys.exit(f"no such {what}: {v}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bed", action="append", required=True, metavar="NAME=PATH",
                    help="bin_scores_<ctx>.all.bed from celltype_marker_bins.py, "
                         "e.g. gch=out/bin_scores_gch.all.bed. Repeatable.")
    ap.add_argument("--fdr", type=float, default=None,
                    help="Define foreground by celltype_fdr < FDR instead of the "
                         "'selected' column.")
    ap.add_argument("--gene-annotation", default=None, metavar="GTF|BED",
                    help="Genes for the feature/gene-overlap/GO analyses: a GTF/"
                         "GFF ('gene' features) or a BED6 (name=col4, "
                         "strand=col6).")
    ap.add_argument("--promoter-window", type=int, default=2000,
                    help="TSS +/- this many bp defines the promoter (default "
                         "2000).")
    ap.add_argument("--chromhmm", default=None, metavar="BED",
                    help="ChromHMM (or any state) BED; column 4 is the state "
                         "label.  Coverage enrichment per state.")
    ap.add_argument("--annotation", action="append", default=None,
                    metavar="LABEL=BED",
                    help="Any functional annotation BED (enhancers, CpG islands, "
                         "TF peaks, ...) for coverage enrichment. Repeatable.")
    ap.add_argument("--gene-sets", action="append", default=None,
                    metavar="LABEL=GMT",
                    help="Gene-set database in GMT format (e.g. Enrichr/MSigDB "
                         "GO_Biological_Process, KEGG) for over-representation of "
                         "the covered genes. Repeatable, e.g. GO=go.gmt "
                         "KEGG=kegg.gmt.")
    ap.add_argument("--min-gene-set", type=int, default=5,
                    help="Ignore gene sets smaller than this (in the universe).")
    ap.add_argument("--max-gene-set", type=int, default=2000,
                    help="Ignore gene sets larger than this (in the universe).")
    ap.add_argument("--max-terms", type=int, default=15,
                    help="Top gene-set terms to plot per database.")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genes = gene_features = None
    if args.gene_annotation:
        genes = _read_genes(args.gene_annotation)
        tss = np.where(genes["strand"].values == "-", genes["end"].values - 1,
                       genes["start"].values)
        gene_features = {
            "promoter": pd.DataFrame({
                "chrom": genes["chrom"],
                "start": np.maximum(tss - args.promoter_window, 0),
                "end": tss + args.promoter_window}),
            "gene_body": genes[["chrom", "start", "end"]].copy()}
        print(f"gene annotation: {len(genes)} genes", file=sys.stderr)

    chromhmm = None
    if args.chromhmm:
        chromhmm = _read_bed(args.chromhmm, want_name=True)
        print(f"chromhmm: {chromhmm['name'].nunique()} states, "
              f"{len(chromhmm)} intervals", file=sys.stderr)

    annotations = {k or Path(v).stem: _read_bed(v)
                   for k, v in _parse_pairs(args.annotation, "annotation").items()}
    gene_sets = {k or Path(v).stem: _read_gmt(v)
                 for k, v in _parse_pairs(args.gene_sets, "gene-set GMT").items()}

    beds = _parse_pairs(args.bed, "--bed")
    summaries = {}
    for name, path in beds.items():
        name = name or Path(path).stem
        s = _run_context(name, path, args, genes, gene_features, chromhmm,
                         annotations, gene_sets, out_dir)
        if s is not None:
            summaries[name] = s
    if not summaries:
        sys.exit("no context produced output.")
    import json
    with open(out_dir / "enrichment_summary.json", "w") as fh:
        json.dump(summaries, fh, indent=2)


if __name__ == "__main__":
    main()
