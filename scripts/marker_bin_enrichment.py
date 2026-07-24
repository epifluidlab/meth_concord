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


def _read_chrom_sizes(path) -> dict:
    """{contig (chr-stripped): length} from a chrom.sizes / .fai (first 2 cols)."""
    sizes = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    sizes[_canon(parts[0])] = int(parts[1])
                except ValueError:
                    continue
    return sizes


def _read_genes(path: str) -> pd.DataFrame:
    """Gene bodies from a GTF/GFF ('gene' features) or a BED6 (name=col4,
    strand=col6).  Returns chrom/start/end/name/strand, 0-based half-open."""
    p = str(path).lower()
    if p.endswith((".gtf", ".gtf.gz", ".gff", ".gff3", ".gff.gz", ".gff3.gz")):
        # Stream and keep only 'gene' lines -- a full GENCODE GTF is ~1.5 GB of
        # exon/transcript rows we do not need.
        import gzip
        opener = gzip.open if p.endswith(".gz") else open
        rows = []
        with opener(path, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 9 or f[2] != "gene":
                    continue
                m = (re.search(r'gene_name "([^"]+)"', f[8])
                     or re.search(r'gene_id "([^"]+)"', f[8]))
                if m:
                    rows.append((f[0], int(f[3]) - 1, int(f[4]),  # GTF 1-based
                                 m.group(1), f[6]))
        if not rows:
            sys.exit(f"{path}: no 'gene' features found in the GTF.")
        return pd.DataFrame(rows, columns=["chrom", "start", "end", "name",
                                           "strand"])
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


def _read_ucsc_bed(path, cols=(0, 1, 2)):
    """A UCSC 'database' table dump (chrom/start/end at the given 0-based column
    indices -- cpgIslandExt has a leading 'bin' column, so use (1, 2, 3))."""
    c, s, e = cols
    df = pd.read_csv(path, sep="\t", header=None, comment="#",
                     compression="infer", dtype={c: str})
    return pd.DataFrame({"chrom": df[c].astype(str), "start": df[s].astype(int),
                         "end": df[e].astype(int)})


def _read_cytoband(path):
    """UCSC cytoBand/cytoBandIdeo -> {'band:<gieStain>': intervals}, so the
    coverage enrichment reports dark (gpos*, heterochromatin) vs light (gneg)
    Giemsa bands."""
    df = pd.read_csv(path, sep="\t", header=None, comment="#",
                     compression="infer", dtype={0: str})
    stain = df[4].astype(str) if df.shape[1] > 4 else pd.Series(
        ["band"] * len(df))
    out = {}
    for s, g in df.groupby(stain.values):
        out[f"band:{s}"] = pd.DataFrame({
            "chrom": g[0].astype(str), "start": g[1].astype(int),
            "end": g[2].astype(int)})
    return out


def _label_to_group(labels, ann, group_col, state_col=None):
    """Map each ChromHMM state label to its biological group, so a fine-grained
    model (e.g. the 100-state full-stack, whose states are semantically
    redundant) collapses to a handful of interpretable groups.

    A state BED label is often '<N>_<mnemonic>' (e.g. '15_mHET9'); it is matched
    to the annotation by the FULL label or by the mnemonic (the suffix after the
    first '_'), NEVER by the leading number -- the segmentation's numbering can
    differ from the annotation's.  ``state_col`` is auto-detected when omitted:
    the annotation column that matches the most labels wins.  Returns
    (label->group, n_matched, n_total, column_used)."""
    labels = list(dict.fromkeys(str(x) for x in labels))
    suffix = {x: (x.split("_", 1)[1] if "_" in x else x) for x in labels}
    cand = ([state_col] if state_col else
            [c for c in ann.columns if c != group_col])
    best, best_hits, best_col = {}, -1, None
    for c in cand:
        if c not in ann.columns:
            continue
        mp = dict(zip(ann[c].astype(str), ann[group_col].astype(str)))
        hits = sum((x in mp) or (suffix[x] in mp) for x in labels)
        if hits > best_hits:
            best_hits, best, best_col = hits, mp, c
    out = {x: best.get(x, best.get(suffix[x], "unmapped")) for x in labels}
    return out, best_hits, len(labels), best_col


# --------------------------------------------------- annotation auto-download -

# Reference URLs per genome for the standard annotations (validated; a coarser
# ChromHMM has no single canonical source, so it stays user-supplied).  The
# Enrichr GO/KEGG GMTs are gene-symbol based and matched case-insensitively, so
# the human-style symbols work for mouse too.
_ENRICHR = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName="
_UCSC = "https://hgdownload.soe.ucsc.edu/goldenPath"
_GENCODE = "https://ftp.ebi.ac.uk/pub/databases/gencode"
_FULLSTACK = ("https://public.hoffman2.idre.ucla.edu/ernst/2K9RS//full_stack/"
              "full_stack_annotation_public_release")
_GENOME = {
    "mm10": {
        "chrom_sizes": f"{_UCSC}/mm10/bigZips/mm10.chrom.sizes",
        "gene_annotation": f"{_GENCODE}/Gencode_mouse/release_M25/"
                           f"gencode.vM25.annotation.gtf.gz",
        "cytoband": f"{_UCSC}/mm10/database/cytoBand.txt.gz",
        "cpg_islands": f"{_UCSC}/mm10/database/cpgIslandExt.txt.gz",
        # Ernst-lab full-stack ChromHMM (universal, 100 states -> 16 groups).
        "chromhmm": f"{_FULLSTACK}/mm10/mm10_100_segments_segments.bed.gz",
        "chromhmm_groups": f"{_FULLSTACK}/mm10/state_annotation_processed.tsv",
        "gene_sets": {"GO_BP": f"{_ENRICHR}GO_Biological_Process_2021",
                      "KEGG": f"{_ENRICHR}KEGG_2019_Mouse"},
    },
    "mm39": {
        "chrom_sizes": f"{_UCSC}/mm39/bigZips/mm39.chrom.sizes",
        "gene_annotation": f"{_GENCODE}/Gencode_mouse/release_M33/"
                           f"gencode.vM33.annotation.gtf.gz",
        "cytoband": f"{_UCSC}/mm39/database/cytoBandIdeo.txt.gz",
        "cpg_islands": f"{_UCSC}/mm39/database/cpgIslandExt.txt.gz",
        "gene_sets": {"GO_BP": f"{_ENRICHR}GO_Biological_Process_2021",
                      "KEGG": f"{_ENRICHR}KEGG_2019_Mouse"},
    },
    "hg38": {
        "chrom_sizes": f"{_UCSC}/hg38/bigZips/hg38.chrom.sizes",
        "gene_annotation": f"{_GENCODE}/Gencode_human/release_44/"
                           f"gencode.v44.annotation.gtf.gz",
        "cytoband": f"{_UCSC}/hg38/database/cytoBand.txt.gz",
        "cpg_islands": f"{_UCSC}/hg38/database/cpgIslandExt.txt.gz",
        "gene_sets": {"GO_BP": f"{_ENRICHR}GO_Biological_Process_2021",
                      "KEGG": f"{_ENRICHR}KEGG_2021_Human"},
    },
    "hg19": {
        "chrom_sizes": f"{_UCSC}/hg19/bigZips/hg19.chrom.sizes",
        "gene_annotation": f"{_GENCODE}/Gencode_human/release_19/"
                           f"gencode.v19.annotation.gtf.gz",
        "cytoband": f"{_UCSC}/hg19/database/cytoBand.txt.gz",
        "cpg_islands": f"{_UCSC}/hg19/database/cpgIslandExt.txt.gz",
        "gene_sets": {"GO_BP": f"{_ENRICHR}GO_Biological_Process_2021",
                      "KEGG": f"{_ENRICHR}KEGG_2021_Human"},
    },
}


def _is_url(s) -> bool:
    return str(s).startswith(("http://", "https://", "ftp://", "file://"))


def _fetch(url, cache_dir):
    """Download ``url`` into ``cache_dir`` (skipped if already cached); return
    the local path.  Enrichr query URLs are named by their libraryName."""
    from urllib.parse import urlparse, parse_qs
    from urllib.request import urlopen, Request
    import shutil
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    q = parse_qs(urlparse(url).query)
    if "libraryName" in q:
        fname = q["libraryName"][0] + ".gmt"
    else:
        fname = Path(urlparse(url).path).name or "download"
    dest = cache_dir / fname
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {fname}", file=sys.stderr)
        return dest
    print(f"  downloading {fname} ...", file=sys.stderr)
    tmp = dest.with_name(dest.name + ".part")
    req = Request(url, headers={"User-Agent": "marker_bin_enrichment"})
    with urlopen(req, timeout=180) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh)
    tmp.replace(dest)
    return dest


def _resolve(value, cache_dir, what):
    """A local path is returned as-is (validated); an http/ftp/file URL is
    fetched+cached.  Returns None on download failure so the caller can skip
    that analysis gracefully."""
    if value is None:
        return None
    if _is_url(value):
        try:
            return str(_fetch(value, cache_dir))
        except Exception as e:                       # network / 404 / timeout
            print(f"  WARNING: could not download {what} ({value}): {e}; "
                  f"skipping it.", file=sys.stderr)
            return None
    if not Path(value).exists():
        sys.exit(f"no such {what}: {value}")
    return value


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
                 annotations, gene_sets, chrom_sizes, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    bins = _read_marker_bed(bed_path)
    if chrom_sizes:                      # defensively clamp over-long bins
        cap = bins["chrom"].map(_canon).map(chrom_sizes)     # size or NaN
        clamped = np.minimum(bins["end"], cap.fillna(bins["end"]))
        bins["end"] = np.maximum(clamped, bins["start"] + 1).astype(int)
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

    # (3) chromatin states -- collapse to biological GROUPS when a mapping is
    # available (e.g. the 100-state full-stack model -> ~16 groups), which is
    # interpretable and higher-powered than testing 100 redundant states.
    if chromhmm is not None:
        grouped = "group" in chromhmm.columns
        if grouped:                              # group name directly (HET, ...)
            cats = {s: g for s, g in chromhmm.groupby("group")}
        else:                                    # per state, back-compat label
            cats = {f"state:{s}": g for s, g in chromhmm.groupby("name")}
        hmm_df, _ = _region_enrichment(bins, fg, cats)
        hmm_df.to_csv(tsv("chromhmm_enrichment"), sep="\t", index=False)
        pages.append(("region", (
            f"Chromatin states (by {'group' if grouped else 'state'})", hmm_df)))
        if args.chromhmm_per_state and grouped:
            st = {f"state:{s}": g for s, g in chromhmm.groupby("name")}
            _region_enrichment(bins, fg, st)[0].to_csv(
                tsv("chromhmm_state_enrichment"), sep="\t", index=False)

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
        head = it.split("=", 1)[0]
        # 'LABEL=value' only when the head is a plain label (a bare URL may
        # contain '=' in its query string, e.g. ...?libraryName=GO).
        if "=" in it and "://" not in head and "/" not in head:
            k, v = it.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out[Path(it.split("?")[0]).stem or it] = it.strip()
    return out                                 # resolved/validated by _resolve


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
    ap.add_argument("--genome", default=None,
                    choices=sorted(_GENOME),
                    help="Auto-DOWNLOAD the standard annotations for this genome "
                         "(GENCODE genes, UCSC chrom.sizes + cytoBand + CpG "
                         "islands, Enrichr GO_BP + KEGG) into --cache-dir and use "
                         "them for any input not given explicitly.  e.g. mm10 for "
                         "GENCODE vM25.  (A tissue ChromHMM has no single "
                         "canonical source, so pass --chromhmm yourself.)")
    ap.add_argument("--cache-dir", default=None, metavar="DIR",
                    help="Where downloaded annotations are cached and reused "
                         "(default: ~/.cache/marker_bin_enrichment/<genome>).")
    ap.add_argument("--chrom-sizes", default=None, metavar="FILE|URL",
                    help="chrom.sizes / .fai; bin ends are clamped to the contig "
                         "length on read (fixes over-long last bins).  Auto-"
                         "filled by --genome.")
    ap.add_argument("--gene-annotation", default=None, metavar="GTF|BED|URL",
                    help="Genes for the feature/gene-overlap/GO analyses: a GTF/"
                         "GFF ('gene' features) or a BED6 (name=col4, "
                         "strand=col6).  A local path or an http/ftp URL.")
    ap.add_argument("--promoter-window", type=int, default=2000,
                    help="TSS +/- this many bp defines the promoter (default "
                         "2000).")
    ap.add_argument("--chromhmm", default=None, metavar="BED|URL",
                    help="ChromHMM (or any state) BED; column 4 is the state "
                         "label.  Coverage enrichment per state.  Path or URL.  "
                         "Auto-filled by --genome mm10 (the 100-state full-stack "
                         "model).")
    ap.add_argument("--chromhmm-groups", default=None, metavar="TSV|URL",
                    help="State-annotation TSV that maps each ChromHMM state to a "
                         "biological GROUP, so a fine-grained model (e.g. the "
                         "100-state full-stack, whose states are semantically "
                         "redundant) is collapsed to a handful of interpretable "
                         "groups for the enrichment.  Matched to the BED state "
                         "label by the full label or its mnemonic (suffix after "
                         "'N_').  Auto-filled by --genome mm10.")
    ap.add_argument("--chromhmm-group-col", default="group",
                    help="Group column in --chromhmm-groups (default 'group').")
    ap.add_argument("--chromhmm-state-col", default=None,
                    help="Column in --chromhmm-groups to match the state label "
                         "against (default: auto-detect, e.g. 'mneumonics').")
    ap.add_argument("--chromhmm-per-state", action="store_true",
                    help="Also write the per-state (un-collapsed) enrichment "
                         "table alongside the per-group one.")
    ap.add_argument("--annotation", action="append", default=None,
                    metavar="LABEL=BED|URL",
                    help="Any functional annotation BED (enhancers, CpG islands, "
                         "TF peaks, ...) for coverage enrichment.  Path or URL. "
                         "Repeatable.")
    ap.add_argument("--gene-sets", action="append", default=None,
                    metavar="LABEL=GMT|URL",
                    help="Gene-set database in GMT format (e.g. Enrichr/MSigDB "
                         "GO_Biological_Process, KEGG) for over-representation of "
                         "the covered genes.  Path or URL. Repeatable, e.g. "
                         "GO=go.gmt KEGG=kegg.gmt.")
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

    preset = _GENOME.get(args.genome, {}) if args.genome else {}
    cache = Path(args.cache_dir) if args.cache_dir else (
        Path.home() / ".cache" / "marker_bin_enrichment" / (args.genome or "custom"))
    if preset:
        print(f"genome {args.genome}: filling missing annotations from downloads "
              f"in {cache}", file=sys.stderr)

    # Chromosome sizes -> clamp over-long bins on read (from --chrom-sizes or the
    # genome preset).
    chrom_sizes = None
    cs = _resolve(args.chrom_sizes or preset.get("chrom_sizes"), cache,
                  "chrom-sizes")
    if cs:
        chrom_sizes = _read_chrom_sizes(cs)

    # Genes -> feature intervals + gene overlap + the GO/KEGG gene universe.
    genes = gene_features = None
    ga = _resolve(args.gene_annotation or preset.get("gene_annotation"), cache,
                  "gene-annotation")
    if ga:
        genes = _read_genes(ga)
        tss = np.where(genes["strand"].values == "-", genes["end"].values - 1,
                       genes["start"].values)
        gene_features = {
            "promoter": pd.DataFrame({
                "chrom": genes["chrom"],
                "start": np.maximum(tss - args.promoter_window, 0),
                "end": tss + args.promoter_window}),
            "gene_body": genes[["chrom", "start", "end"]].copy()}
        print(f"gene annotation: {len(genes)} genes", file=sys.stderr)

    # Chromatin states.  A fine-grained model is collapsed to biological groups
    # via --chromhmm-groups (auto-filled for mm10 = the full-stack model).
    chromhmm = None
    ch = _resolve(args.chromhmm or preset.get("chromhmm"), cache, "chromhmm")
    if ch:
        chromhmm = _read_bed(ch, want_name=True)
        gtsv = _resolve(args.chromhmm_groups or preset.get("chromhmm_groups"),
                        cache, "chromhmm-groups")
        if gtsv:
            ann = pd.read_csv(gtsv, sep="\t", dtype=str)
            if args.chromhmm_group_col not in ann.columns:
                sys.exit(f"--chromhmm-group-col {args.chromhmm_group_col!r} not "
                         f"in {gtsv}: {list(ann.columns)}")
            mp, hit, tot, col = _label_to_group(
                chromhmm["name"].unique(), ann, args.chromhmm_group_col,
                args.chromhmm_state_col)
            chromhmm["group"] = chromhmm["name"].map(mp)
            print(f"chromhmm groups: matched {hit}/{tot} states via column "
                  f"{col!r} -> {chromhmm['group'].nunique()} groups",
                  file=sys.stderr)
        print(f"chromhmm: {chromhmm['name'].nunique()} states"
              + (f", {chromhmm['group'].nunique()} groups"
                 if "group" in chromhmm.columns else "")
              + f", {len(chromhmm)} intervals", file=sys.stderr)

    # Generic annotations: the user's, plus the genome's cytoBand (per Giemsa
    # band -> heterochromatin gpos* vs euchromatin gneg) and CpG islands.
    annotations = {}
    for k, v in _parse_pairs(args.annotation, "annotation").items():
        rv = _resolve(v, cache, f"annotation {k}")
        if rv is not None:
            annotations[k] = _read_bed(rv)
    if preset.get("cytoband"):
        cb = _resolve(preset["cytoband"], cache, "cytoBand")
        if cb is not None:
            for stain, df in _read_cytoband(cb).items():
                annotations.setdefault(stain, df)
    if preset.get("cpg_islands") and "CpG_island" not in annotations:
        cp = _resolve(preset["cpg_islands"], cache, "CpG islands")
        if cp is not None:
            annotations["CpG_island"] = _read_ucsc_bed(cp, cols=(1, 2, 3))

    # Gene sets: the user's, plus the genome's GO/KEGG for labels not overridden.
    gene_sets = {}
    for k, v in _parse_pairs(args.gene_sets, "gene-set GMT").items():
        rv = _resolve(v, cache, f"gene-set {k}")
        if rv is not None:
            gene_sets[k] = _read_gmt(rv)
    for label, url in preset.get("gene_sets", {}).items():
        if label not in gene_sets:
            rv = _resolve(url, cache, f"gene-set {label}")
            if rv is not None:
                gene_sets[label] = _read_gmt(rv)

    summaries = {}
    for name, path in _parse_pairs(args.bed, "--bed").items():
        rp = _resolve(path, cache, f"--bed {name}")
        if rp is None:
            continue
        s = _run_context(name, rp, args, genes, gene_features, chromhmm,
                         annotations, gene_sets, chrom_sizes, out_dir)
        if s is not None:
            summaries[name] = s
    if not summaries:
        sys.exit("no context produced output.")
    import json
    with open(out_dir / "enrichment_summary.json", "w") as fh:
        json.dump(summaries, fh, indent=2)


if __name__ == "__main__":
    main()
