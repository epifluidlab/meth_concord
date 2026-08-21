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

Foreground vs background.  Given the ``.all.bed``, a bin is FOREGROUND if it
was ``selected`` for the heatmap (or, with ``--fdr``, if ``celltype_fdr`` is
below the cut) and the rest are BACKGROUND, so the universe is every bin in the
input and the enrichment is unbiased by the bin-selection itself.

The subset BEDs -- ``.selected.bed``, ``.stable.bed``, ``.always_co_open.bed``,
``.contrast_<label>.bed`` -- drop their membership flags by design, so they
carry no column to split on and are foreground ONLY.  Give each one a universe
with ``--background-bed NAME=PATH``: a marker BED or a plain 3-column BED, and
usually the matching ``.all.bed``.  Bins in both files count as foreground
alone, so a background that CONTAINS the foreground does the right thing rather
than putting those bins on both sides of every comparison.

Enrichment model.  The bins are large (250 kb by default), so the natural
question is COVERAGE, not point overlap: for each region annotation the fraction
of a bin covered is compared foreground-vs-background with a two-sided
Mann-Whitney U test, reported with the log2 fold-change of the mean coverage and
a BH-FDR across categories (positive log2FC = enriched, negative = depleted).
Gene-set (GO/KEGG) enrichment is a hypergeometric over-representation test of the
foreground-bin genes against the universe of all-bin genes.

Continuous tracks (``--signal-track``, e.g. Repli-seq early/late) get the
per-bin mean compared foreground-vs-background, reported as BOTH the signed
difference of means and the rank AUC.  The AUC is the effect size belonging to
the Mann-Whitney p-value and is scale-free, so it is the one that can be
compared BETWEEN tracks -- a difference of means cannot, since an early track
and a late one are not on the same scale.  Its interval comes from a block
bootstrap over megabase blocks (--signal-block-bp), because these signals are
organised into domains far larger than a bin and the p-value beside it treats
the bins as independent.  scripts/bin_set_signal_summary.py draws several bin
sets and contexts from those TSVs as one panel.

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


def _read_bins_any(path: str) -> pd.DataFrame:
    """A bin set from either a marker BED (with a header) or a plain BED.

    The foreground-only files celltype_marker_bins writes -- .selected.bed,
    .stable.bed, .always_co_open.bed, .contrast_<label>.bed -- have their
    membership flags dropped by design, so they carry no 'selected' column to
    split on and can only be used together with an explicit background.  A
    plain three-column BED is accepted too, so the background can be any region
    set rather than something this pipeline produced.
    """
    with open(path) as fh:
        first = fh.readline()
    header = first.lower().startswith(("chrom", "#chrom"))
    if header:
        return _read_marker_bed(path)
    df = _read_bed(path)
    return df[["chrom", "start", "end"]]


def _split_with_background(fg_path: str, bg_path: str, name: str):
    """Foreground and background from two files → (bins, fg_mask, description).

    A bin present in BOTH is foreground: the common case is a background of
    every bin (the .all.bed), which contains the foreground, and leaving the
    duplicates in would put those bins on both sides of every comparison --
    inflating the background toward the foreground and shrinking every effect
    toward zero.
    """
    fg_df = _read_bins_any(fg_path)
    bg_df = _read_bins_any(bg_path)
    key = ["chrom", "start", "end"]
    fg_keys = set(map(tuple, fg_df[key].astype(str).to_numpy()))
    dup = np.array([tuple(r) in fg_keys
                    for r in bg_df[key].astype(str).to_numpy()], dtype=bool)
    if dup.any():
        print(f"  {name}: {int(dup.sum()):,} of {len(bg_df):,} background bins "
              f"are also foreground and were removed from the background "
              f"(a background that CONTAINS the foreground is the usual case, "
              f"e.g. the .all.bed).", file=sys.stderr)
    bg_df = bg_df.loc[~dup]
    if len(fg_df) == 0:
        sys.exit(f"{name}: --bed {fg_path} has no bins.")
    if len(bg_df) == 0:
        sys.exit(f"{name}: no background bins left after removing the "
                 f"foreground from {bg_path}.  An enrichment needs something "
                 f"to be enriched AGAINST; pass a wider background (the "
                 f".all.bed, or a genome-wide bin set).")
    bins = pd.concat([fg_df, bg_df], ignore_index=True, sort=False)
    fg = np.zeros(len(bins), bool)
    fg[:len(fg_df)] = True
    return bins, fg, f"--background-bed ({len(fg_df):,} fg / {len(bg_df):,} bg)"


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


def _read_pmd(path) -> dict:
    """The Zhou et al. 2018 PMD file → {"PMD": ivals, "HMD": ivals}.

    Six columns: chrom, start, end, score, class, label — where the label is
    commonPMD / commonHMD / Neither.  Loading it whole would merge partially
    METHYLATED domains with highly methylated ones and with the unclassified
    remainder, and test their union against the genome, which is not a
    question anyone means to ask.  'Neither' is dropped rather than kept as a
    third set: it is the residue of the classification, not a domain type.
    """
    df = pd.read_csv(path, sep="\t", header=None, comment="#",
                     compression="infer", dtype={0: str})
    if df.shape[1] < 6:
        raise ValueError(f"{path}: expected the 6-column PMD file "
                         f"(chrom/start/end/score/class/label), got "
                         f"{df.shape[1]} columns")
    lab = df[5].astype(str)
    out = {}
    for key, want in (("PMD", "commonPMD"), ("HMD", "commonHMD")):
        sub = df[lab == want]
        if len(sub):
            out[key] = pd.DataFrame({"chrom": sub[0].astype(str),
                                     "start": sub[1].astype(int),
                                     "end": sub[2].astype(int)})
    return out


def _read_annotation_manifest(path) -> dict:
    """``label<TAB>path_or_url`` lines → dict.  '#' comments, blanks ignored.

    A file rather than a heap of --annotation flags, because the domain-scale
    sets worth testing at 250 kb (LADs, PMDs, compartments, a lifted-over
    segmentation) are third-party downloads whose URLs move, and a manifest can
    be fixed without touching a command line or this script.
    """
    out = {}
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split(None, 1)
            if len(parts) < 2:
                sys.exit(f"{path}:{ln}: expected 'label<TAB>path_or_url', "
                         f"got {line!r}")
            out[parts[0].strip()] = parts[1].strip()
    return out


def _merge_manifests(paths) -> dict:
    """Every --annotation-manifest, merged into one {label: path_or_url}.

    The flag repeats, so LADs and replication timing can come from separate
    files.  A label appearing in two manifests is an error: silently keeping
    one would drop a whole annotation from the report, and its absence is not
    visible in the output -- nothing says "this was meant to be here".
    """
    out, source = {}, {}
    for path in paths:
        for label, value in _read_annotation_manifest(path).items():
            if label in out and out[label] != value:
                sys.exit(f"annotation {label!r} is defined in both "
                         f"{source[label]} and {path}, pointing at different "
                         f"files.  Rename one, or drop the duplicate line.")
            out[label], source[label] = value, path
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
_ZHOU = "https://zhouserver.research.chop.edu/GenomeAnnotation"
_ERNST_GH = ("https://raw.githubusercontent.com/ernstlab/"
             "full_stack_ChromHMM_annotations/main")
_FULLSTACK_PUB = ("https://public.hoffman2.idre.ucla.edu/ernst/2K9RS//"
                  "full_stack/full_stack_annotation_public_release")
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
        # Common PMD/HMD, Zhou et al. Nat Genet 2018 -- megabase domains, and
        # where pan-cancer hypomethylation lives, so the set most worth testing
        # against HCG concordance at this bin width.
        "annotations": {"PMD": {"url": f"{_ZHOU}/hg38/PMD_coordinates_hg38"
                                       f".bed.gz", "reader": "pmd"}},
        # Full-stack ChromHMM (Vu & Ernst, Genome Biol 2021), lifted to hg38 by
        # the authors.  No LAD entry: UCSC carries the Guelen table for hg19
        # only, so an hg38 LAD set has to come through --annotation-manifest --
        # see resources/annotations_human_domains.tsv.
        "chromhmm": f"{_FULLSTACK_PUB}/hg38/hg38_genome_100_segments.bed.gz",
        "chromhmm_groups": f"{_ERNST_GH}/state_annotations_processed.csv",
    },
    "hg19": {
        "chrom_sizes": f"{_UCSC}/hg19/bigZips/hg19.chrom.sizes",
        "gene_annotation": f"{_GENCODE}/Gencode_human/release_19/"
                           f"gencode.v19.annotation.gtf.gz",
        "cytoband": f"{_UCSC}/hg19/database/cytoBand.txt.gz",
        "cpg_islands": f"{_UCSC}/hg19/database/cpgIslandExt.txt.gz",
        # Lamin-B1 LADs (Guelen et al. Nature 2008, Tig3 DamID) as UCSC ships
        # them, and common PMD/HMD (Zhou et al. Nat Genet 2018).  Both are
        # megabase domains, so they survive a 250 kb bin -- unlike anything
        # peak-scale, which at that width measures gene density instead.
        "annotations": {
            "LAD": f"{_UCSC}/hg19/database/laminB1Lads.txt.gz",
            "PMD": {"url": f"{_ZHOU}/hg19/PMD_coordinates_hg19.bed.gz",
                    "reader": "pmd"}},
        "chromhmm": f"{_FULLSTACK_PUB}/hg19/hg19_genome_100_segments.bed.gz",
        "chromhmm_groups": f"{_ERNST_GH}/state_annotations_processed.csv",
        "gene_sets": {"GO_BP": f"{_ENRICHR}GO_Biological_Process_2021",
                      "KEGG": f"{_ENRICHR}KEGG_2021_Human"},
    },
}


def _is_url(s) -> bool:
    return str(s).startswith(("http://", "https://", "ftp://", "file://"))


def _fetch(url, cache_dir):
    """Download ``url`` into ``cache_dir`` (skipped if already cached); return
    the local path.  Enrichr query URLs are named by their libraryName.

    The cache name carries a hash of the FULL url, not just its basename.  UCSC
    serves cytoBand.txt.gz and cpgIslandExt.txt.gz at the same basename under
    every assembly, so a basename-keyed cache hands an mm10 run's cytobands to
    a later hg38 run in the same --cache-dir, silently and with no error --
    mouse chr1-19 match human chr1-19 by name and simply carry wrong
    coordinates.  Hashing the url keeps both assemblies side by side."""
    from urllib.parse import urlparse, parse_qs
    from urllib.request import urlopen, Request
    import hashlib
    import shutil
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    q = parse_qs(urlparse(url).query)
    if "libraryName" in q:
        fname = q["libraryName"][0] + ".gmt"
    else:
        fname = Path(urlparse(url).path).name or "download"
    key = hashlib.sha1(url.encode()).hexdigest()[:8]
    dest = cache_dir / f"{key}_{fname}"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {fname}", file=sys.stderr)
        return dest
    # A file left by a pre-hash release cannot be trusted: its name does not
    # record which assembly it came from.  Re-download rather than guess.
    legacy = cache_dir / fname
    if legacy.exists():
        print(f"  re-downloading {fname} (cached under the old name, which "
              f"did not record the source assembly)", file=sys.stderr)
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


# ---------------------------------------------- continuous (bigWig) signals --

def _binned_mean(bins: pd.DataFrame, sig: pd.DataFrame) -> np.ndarray:
    """Per-bin overlap-length-weighted mean of a valued interval signal
    (bedGraph).  Intervals are assumed non-overlapping (a bedGraph partition).
    NaN where a bin has no covered signal."""
    out = np.full(len(bins), np.nan)
    by = {}
    for c, g in sig.groupby(sig["chrom"].map(_canon)):
        v = g[["start", "end"]].values.astype(np.int64)
        order = np.argsort(v[:, 0])
        by[c] = (v[order, 0], v[order, 1],
                 g["value"].values.astype(float)[order])
    bc = bins["chrom"].map(_canon).values
    bs = bins["start"].values
    be = bins["end"].values
    for i in range(len(bins)):
        t = by.get(bc[i])
        if t is None:
            continue
        starts, ends, vals = t
        lo = int(np.searchsorted(ends, bs[i], "right"))
        hi = int(np.searchsorted(starts, be[i], "right"))
        if hi <= lo:
            continue
        ov = np.minimum(be[i], ends[lo:hi]) - np.maximum(bs[i], starts[lo:hi])
        vv = vals[lo:hi]
        m = (ov > 0) & np.isfinite(vv)
        if m.any() and ov[m].sum() > 0:
            out[i] = float(np.average(vv[m], weights=ov[m]))
    return out


def _bigwig_bin_means(bins: pd.DataFrame, path: str) -> np.ndarray:
    """Per-bin mean of a bigWig via pyBigWig (mean over covered bases)."""
    import pyBigWig
    bw = pyBigWig.open(str(path))
    have = {c.lower(): c for c in bw.chroms().keys()}
    out = np.full(len(bins), np.nan)
    for i, (c, s, e) in enumerate(zip(bins["chrom"], bins["start"],
                                      bins["end"])):
        name = have.get(str(c).lower()) or have.get(("chr" + _canon(c)).lower()) \
            or have.get(_canon(c).lower())
        if name is None:
            continue
        e = min(int(e), bw.chroms()[name])
        if e <= int(s):
            continue
        try:
            v = bw.stats(name, int(s), e, type="mean", exact=True)[0]
        except (RuntimeError, ValueError):
            v = None
        if v is not None:
            out[i] = float(v)
    bw.close()
    return out


def _signal_bin_means(bins: pd.DataFrame, path: str) -> np.ndarray:
    """Per-bin mean signal from a bigWig (.bw/.bigwig) or a bedGraph/BED (4th
    numeric column)."""
    p = str(path).lower()
    if p.endswith((".bw", ".bigwig")):
        return _bigwig_bin_means(bins, path)
    df = pd.read_csv(path, sep="\t", header=None, comment="#",
                     compression="infer", dtype={0: str})
    sig = pd.DataFrame({"chrom": df[0].astype(str), "start": df[1].astype(int),
                        "end": df[2].astype(int),
                        "value": pd.to_numeric(df[3], errors="coerce")})
    return _binned_mean(bins, sig)


def _auc(a, b) -> float:
    """P(a random fg bin exceeds a random bg one), ties counted as half.

    The rank statistic the Mann-Whitney p-value already tests, expressed as an
    effect size.  0.5 is no effect.  Scale-free, which ``difference`` is not:
    raw signal units differ between tracks, so a difference of means cannot be
    compared between an early-replication track and a late one without knowing
    both scales.  Comparing several tracks in one figure needs this.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([a, b]))          # ties -> average rank
    return float((r[:a.size].sum() - a.size * (a.size + 1) / 2)
                 / (a.size * b.size))


def _block_ids(bins, block_bp):
    """Contiguous genomic blocks, for a bootstrap that respects autocorrelation.

    Replication timing (and most of what these tracks carry) is organised into
    domains far larger than a bin, so neighbouring bins are not independent
    draws.  Resampling BINS would understate the uncertainty badly; resampling
    megabase blocks keeps whole domains together.
    """
    chrom = bins["chrom"].map(_canon).to_numpy()
    blk = (bins["start"].to_numpy() // max(int(block_bp), 1)).astype(np.int64)
    return np.array([f"{c}:{b}" for c, b in zip(chrom, blk)])


def _auc_block_ci(val, fg, blocks, n_boot, seed=42):
    """Block-bootstrap CI for the AUC → (lo, hi)."""
    ok = np.isfinite(val)
    if n_boot <= 0 or not ok.any():
        return np.nan, np.nan
    uniq = np.unique(blocks[ok])
    if uniq.size < 5:
        return np.nan, np.nan
    idx_of = {u: np.flatnonzero(ok & (blocks == u)) for u in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n_boot)):
        pick = rng.choice(uniq, uniq.size, replace=True)
        sel = np.concatenate([idx_of[u] for u in pick])
        a, b = val[sel][fg[sel]], val[sel][~fg[sel]]
        if a.size and b.size:
            out.append(_auc(a, b))
    if len(out) < 20:
        return np.nan, np.nan
    return tuple(float(x) for x in np.percentile(out, [2.5, 97.5]))


def _signal_enrichment(bins, fg, tracks, block_bp=10_000_000, n_boot=200):
    """fg-vs-bg Mann-Whitney on the per-bin MEAN of each continuous track
    (replication timing, ...).

    Reports the signed difference of means (not a ratio -- the signal may be
    negative) AND the rank AUC, which is what makes several tracks comparable
    in one figure.  The AUC's interval comes from a BLOCK bootstrap over
    genomic blocks, because the p-value beside it treats 250 kb bins as
    independent and megabase-scale domains mean they are not.
    """
    from scipy.stats import mannwhitneyu
    rows, series = [], {}
    blocks = _block_ids(bins, block_bp)
    for label, path in tracks.items():
        val = _signal_bin_means(bins, path)
        series[label] = val
        keep = np.isfinite(val)
        a = val[fg & keep]
        b = val[(~fg) & keep]
        m_fg = float(a.mean()) if a.size else np.nan
        m_bg = float(b.mean()) if b.size else np.nan
        p = np.nan
        if a.size and b.size:
            try:
                p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
            except ValueError:
                p = np.nan
        auc = _auc(a, b)
        lo, hi = _auc_block_ci(val, fg, blocks, n_boot)
        sd_fg = float(a.std(ddof=1)) if a.size > 1 else np.nan
        sd_bg = float(b.std(ddof=1)) if b.size > 1 else np.nan
        pooled = np.sqrt(((max(a.size - 1, 0) * (sd_fg ** 2 if a.size > 1 else 0)
                           + max(b.size - 1, 0)
                           * (sd_bg ** 2 if b.size > 1 else 0))
                          / max(a.size + b.size - 2, 1)))
        rows.append({"annotation": label, "mean_signal_fg": m_fg,
                     "mean_signal_bg": m_bg, "difference": m_fg - m_bg,
                     "sd_signal_fg": sd_fg, "sd_signal_bg": sd_bg,
                     "cohens_d": ((m_fg - m_bg) / pooled if pooled > 0
                                  else np.nan),
                     "auc": auc, "auc_lo": lo, "auc_hi": hi,
                     "n_fg": int(a.size), "n_bg": int(b.size), "pvalue": p})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["fdr"] = _bh(df["pvalue"].values)
        df = df.sort_values("difference", ascending=False).reset_index(drop=True)
    return df, series


# ------------------------------------------------------------- enrichment ----

def _is_binary_coverage(cov, eps=1e-3) -> bool:
    """Is this annotation all-or-nothing on this bin grid?

    An annotation BUILT on the analysis grid -- the LAD partition, the
    replication-timing classes, anything written per bin -- covers each bin
    either fully or not at all, so its coverage takes exactly two values.  The
    graded ones (PMDs, cytobands, a segmentation) land anywhere in between.
    """
    v = np.asarray(cov, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return False
    return bool(np.all((v <= eps) | (v >= 1.0 - eps)))


def _fisher_row(label, cov, fg, min_frac=0.5):
    """Fisher's exact on the 2x2 of (foreground, in the annotation).

    A bin counts as IN when at least ``min_frac`` of it is covered.  For a set
    defined on this grid -- the LAD partition, the RT classes -- coverage is 0
    or 1 and any threshold in between gives the same table.  For a set with
    its own coordinates (PMDs, HMDs) the bins that straddle a boundary go to
    whichever side holds most of them, which is the usual call for domains
    megabases wide against a 250 kb bin.

    This is the classic enrichment question -- are foreground bins more often
    in the annotation -- answered exactly, which matters when the foreground
    is small, and with an odds ratio.  Running it for EVERY region annotation
    rather than switching per annotation is deliberate: one panel comparing
    LADs, PMDs and replication timing should not mix two tests and two effect
    measures, because then no row can be read against its neighbour.
    """
    from scipy.stats import fisher_exact
    hit = np.asarray(cov, float) >= min_frac
    a = int(np.sum(fg & hit))
    b = int(np.sum(fg & ~hit))
    c = int(np.sum(~fg & hit))
    d = int(np.sum(~fg & ~hit))
    try:
        orr, p = fisher_exact([[a, b], [c, d]])
    except Exception:
        orr, p = np.nan, np.nan
    m_fg = a / max(a + b, 1)
    m_bg = c / max(c + d, 1)
    return {"annotation": label, "test": "fisher",
            "mean_frac_fg": m_fg, "mean_frac_bg": m_bg,
            "log2_fold_change": float(np.log2((m_fg + _EPS) / (m_bg + _EPS))),
            "odds_ratio": float(orr), "n_fg_in": a, "n_fg_out": b,
            "n_bg_in": c, "n_bg_out": d, "pvalue": float(p)}


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
    return {"annotation": label, "test": "mannwhitney",
            "mean_frac_fg": m_fg, "mean_frac_bg": m_bg,
            "log2_fold_change": lfc, "pvalue": p}


def _dotplot(df, x, size, color, labels, path, title, xlabel,
             size_label="Count", color_label="FDR", vline=None, top=20,
             ascending=False):
    """A dot plot: position is the effect, area the count, colour the FDR.

    One panel carrying all three is what makes an enrichment readable at a
    glance -- a bar of fold change alone hides that it rests on four genes,
    and a table of p-values hides which effects are large.  Area, not radius,
    scales with the count, because area is what the eye compares.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.dropna(subset=[x, color]).copy()
    if not len(d):
        return None
    d = d.sort_values(color).head(top).sort_values(x, ascending=ascending)
    y = np.arange(len(d))
    area = d[size].to_numpy(float)
    # np.ptp, not ndarray.ptp: the method was removed in NumPy 2.0, and this
    # module is otherwise version-agnostic.
    span = float(np.ptp(area)) if area.size else 0.0
    area = 40 + 260 * ((area - area.min()) / (span or 1.0))

    fig, ax = plt.subplots(figsize=(8.6, max(3.0, 0.34 * len(d) + 1.8)))
    sc = ax.scatter(d[x], y, s=area, c=d[color], cmap="plasma_r",
                    edgecolor="#2A2E35", linewidth=0.5, zorder=3)
    if vline is not None:
        ax.axvline(vline, color="#9AA0A6", lw=1.0, ls="--", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([str(v) for v in d[labels]], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", color="#E5E7EB", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, aspect=28)
    cb.set_label(color_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.5)

    # A size key, since area carries a number the axes cannot show.
    lo, hi = int(np.nanmin(d[size])), int(np.nanmax(d[size]))
    ticks = sorted({lo, (lo + hi) // 2, hi})
    handles = [plt.scatter([], [], s=40 + 260 * ((t - lo) / ((hi - lo) or 1)),
                           c="#B0B5BC", edgecolor="#2A2E35", linewidth=0.5,
                           label=str(t)) for t in ticks]
    # Below the colourbar rather than beside it: the two keys otherwise share
    # the same strip of figure and overlap once the bars grow.
    ax.legend(handles=handles, title=size_label, loc="lower left",
              bbox_to_anchor=(1.14, 0.0), frameon=False, fontsize=8,
              title_fontsize=9, labelspacing=1.3, borderpad=0.8,
              handletextpad=1.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _annotation_dotplot(df, path, name, exclude=r"^band:|^CpG_island$"):
    """Publication panel for the region annotations, Fisher rows only.

    Giemsa bands and CpG islands are dropped by default and for opposite
    reasons: a band is a stain, not a functional annotation, and it is the
    coarsest thing in the table; a CpG island is ~1 kb and never fills a bin,
    so its row is empty rather than negative.  Neither belongs beside LADs,
    PMDs and replication timing in a figure about domain organisation.
    """
    d = df[df.get("test", "fisher") == "fisher"].copy()
    if exclude:
        d = d[~d["annotation"].astype(str).str.contains(exclude, regex=True)]
    need = {"n_fg_in", "n_fg_out", "n_bg_in", "n_bg_out"}
    if not need <= set(d.columns) or not len(d):
        return None
    # Haldane-Anscombe: half a count in every cell of the 2x2.  An annotation
    # the foreground never enters has an odds ratio of exactly 0, whose log is
    # -inf -- and that is the STRONGEST depletion in the table, so dropping it
    # for being unplottable removes the biggest result from the figure while
    # leaving it in the TSV.  The correction keeps it finite and it stays the
    # most extreme point, which is the honest place for it.
    a = d["n_fg_in"].to_numpy(float) + 0.5
    b = d["n_fg_out"].to_numpy(float) + 0.5
    c = d["n_bg_in"].to_numpy(float) + 0.5
    e = d["n_bg_out"].to_numpy(float) + 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        d["log2_odds_ratio"] = np.log2((a * e) / (b * c))
    d = d[np.isfinite(d["log2_odds_ratio"])]
    if not len(d):
        return None
    return _dotplot(d, "log2_odds_ratio", "n_fg_in", "fdr", "annotation",
                    path, f"{name}: domain-scale enrichment (Fisher's exact)",
                    "log2 odds ratio (Haldane-corrected)",
                    size_label="bins in set", color_label="FDR", vline=0.0)


def _geneset_dotplot(df, path, name, label, n_fg_genes):
    """Publication panel for a gene-set database, in the usual dot-plot form.

    x is the gene ratio -- the share of the foreground's genes in the term --
    because that is the quantity a reader can hold against the term's own
    size; area is the overlap count and colour the adjusted p.
    """
    d = df.copy()
    if not len(d) or "n_overlap" not in d:
        return None
    denom = float(n_fg_genes) or 1.0
    d["GeneRatio"] = d["n_overlap"].to_numpy(float) / denom
    return _dotplot(d, "GeneRatio", "n_overlap", "fdr", "term", path,
                    f"{name}: {label}", "GeneRatio", size_label="Count",
                    color_label="p.adjust", ascending=True)


def _region_enrichment(bins, fg, categories, test="fisher", min_frac=0.5):
    """Region enrichment (fg vs bg) for a dict {label: intervals_df}.

    Fisher's exact for every annotation by default, on 'is this bin in the
    set'.  These are all domain-scale sets asking one question, and they are
    read side by side in a single panel; a per-annotation choice of test would
    put an odds ratio next to a rank statistic and two different nulls in one
    figure, which no reader can compare.

    'mwu' keeps the graded version -- Mann-Whitney on the coverage fraction --
    for a set where the partial bins are the point.  'auto' picks per
    annotation, which is what the graded/binary split would suggest in
    isolation; it is not the default for the reason above.  The test that ran
    is recorded in the 'test' column either way.
    """
    rows, covs = [], {}
    for label, iv in categories.items():
        cov = _coverage(bins, iv)
        covs[label] = cov
        use = test
        if use == "auto":
            use = "fisher" if _is_binary_coverage(cov) else "mwu"
        if use == "fisher":
            total = float(np.nansum(cov))
            if total > 0 and not np.any(cov >= min_frac):
                # Peak-scale set at a domain-scale bin.  It never fills half a
                # bin, so every cell of the 2x2 on one side is zero and the
                # row comes out p = 1 -- which reads as "no enrichment" when
                # it means "this test cannot see this annotation here".
                print(f"  NOTE {label}: covers at most "
                      f"{100 * float(np.nanmax(cov)):.2g}% of any bin, never "
                      f"the {100 * min_frac:.0f}% needed to count as IN, so "
                      f"its row is empty rather than negative.  It is "
                      f"peak-scale against a domain-scale bin: turn it into a "
                      f"density track (bp or peaks per bin) and pass it with "
                      f"--signal-track, or lower --region-overlap-frac.",
                      file=sys.stderr)
        rows.append(_fisher_row(label, cov, fg, min_frac) if use == "fisher"
                    else _mwu_row(label, cov, fg))
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


def _barh_region(ax, df, title, value_col="log2_fold_change",
                 xlabel="log2 fold-change of mean coverage (fg / bg)"):
    """Horizontal bars of a signed effect, coloured by direction, FDR-starred."""
    if df is None or df.empty:
        ax.text(0.5, 0.5, f"{title}\n(no data)", ha="center", va="center")
        ax.axis("off")
        return
    d = df.iloc[::-1]
    y = np.arange(len(d))
    vals = d[value_col].values
    span = np.nanmax(np.abs(vals)) or 1.0
    colors = ["#c0392b" if v >= 0 else "#2c6fbb" for v in vals]
    ax.barh(y, vals, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(d["annotation"], fontsize=8)
    ax.axvline(0, color="0.4", lw=0.8)
    for i, (v, q) in enumerate(zip(vals, d["fdr"].values)):
        s = _stars(q) if np.isfinite(q) else ""
        if s:
            ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), i, s,
                    va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel(xlabel)
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


def _signal_figure(series, fg, df, name, out_path):
    """Publication figure: per-bin signal DISTRIBUTION, foreground vs
    background, one grouped violin pair per track (ordered as the enrichment
    table), with the BH-FDR significance annotated.

    The y-axis is set to a ROBUST range (1st-99th percentile of the pooled
    signal) so a few outlier bins do not flatten the distributions; the violin
    KDE is trimmed to that range (else it is evaluated over the full 0..outlier
    span and the bulk renders as a flat blob).  The mean line is the true
    full-data mean (matching the enrichment table)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = df["annotation"].tolist()
    n = len(order)
    pooled = np.concatenate([np.asarray(series[l], float) for l in order])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size:
        # IQR (Tukey) fence -> robust to ANY outlier fraction; a plain high
        # percentile is itself contaminated once >1% of bins are outliers (as
        # happens for low-coverage bins with inflated normalised signal).
        q1, q3 = (float(x) for x in np.percentile(pooled, [25, 75]))
        iqr = q3 - q1
        if iqr > 1e-12:
            lo = min(0.0, q1 - 1.5 * iqr)
            hi = q3 + 3.0 * iqr
        else:
            lo = min(0.0, float(np.percentile(pooled, 1)))
            hi = float(np.percentile(pooled, 99))
        if not (hi > lo):
            lo, hi = float(pooled.min()), float(pooled.max())
    else:
        lo, hi = 0.0, 1.0
    rng = (hi - lo) or 1.0
    fg_c, bg_c = "#c0392b", "#95a5a6"
    fig, ax = plt.subplots(figsize=(max(4.5, 1.15 * n + 1.5), 4.4))
    for i, lab in enumerate(order):
        v = np.asarray(series[lab], dtype=float)
        for mask, off, color in [((~fg), 0.22, bg_c), (fg, -0.22, fg_c)]:
            full = v[mask & np.isfinite(v)]
            if full.size < 2:
                continue
            shown = full[(full >= lo) & (full <= hi)]     # trim KDE to the axis
            if shown.size >= 2:
                pc = ax.violinplot([shown], positions=[i + off], widths=0.38,
                                   showmeans=False, showextrema=False)
                pc["bodies"][0].set_facecolor(color)
                pc["bodies"][0].set_alpha(0.8)
                pc["bodies"][0].set_edgecolor("none")
            med = float(np.median(full))    # median: robust, sits in the bulk
            ax.plot([i + off - 0.17, i + off + 0.17], [med, med], color="black",
                    lw=1.1, zorder=3)         # (rank-based, matches the MWU test)
        q = df.loc[df["annotation"] == lab, "fdr"].iloc[0]
        s = _stars(q) if np.isfinite(q) else "ns"
        ax.text(i, hi + 0.04 * rng, s or "ns", ha="center", va="bottom",
                fontsize=9)
    ax.set_ylim(lo - 0.03 * rng, hi + 0.14 * rng)
    ax.set_xticks(range(n))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("per-bin mean signal")
    ax.set_title(f"{name.upper()}: signal at cell-type-variable bins vs the rest",
                 fontsize=11)
    ax.legend(handles=[
        plt.Line2D([], [], color=fg_c, lw=7, alpha=0.8,
                   label=f"foreground ({int(fg.sum())})"),
        plt.Line2D([], [], color=bg_c, lw=7, alpha=0.8,
                   label=f"background ({int((~fg).sum())})")],
        frameon=False, fontsize=8, loc="upper left",
        bbox_to_anchor=(1.01, 1.0))         # outside, clear of the star row
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- driver --

def _run_context(name, bed_path, args, genes, gene_features, chromhmm,
                 annotations, gene_sets, signal_tracks, chrom_sizes, out_dir,
                 bg_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if bg_path:
        bins, fg, fg_desc = _split_with_background(bed_path, bg_path, name)
        if args.fdr is not None:
            print(f"  {name}: --fdr is ignored when --background-bed gives the "
                  f"split explicitly.", file=sys.stderr)
    else:
        bins = _read_marker_bed(bed_path)
    if chrom_sizes:                      # defensively clamp over-long bins
        cap = bins["chrom"].map(_canon).map(chrom_sizes)     # size or NaN
        clamped = np.minimum(bins["end"], cap.fillna(bins["end"]))
        bins["end"] = np.maximum(clamped, bins["start"] + 1).astype(int)
    if bg_path:
        pass                                    # split already decided above
    elif args.fdr is not None and "celltype_fdr" in bins.columns:
        fg = (bins["celltype_fdr"].values < args.fdr)
        fg_desc = f"celltype_fdr < {args.fdr}"
    elif "selected" in bins.columns:
        fg = bins["selected"].astype(str).str.lower().isin(
            ["true", "1", "yes"]).values
        fg_desc = "selected == True"
    else:
        sys.exit(
            f"{name}: {bed_path} has no 'selected' column, so there is no "
            f"background to compare against and every enrichment would be the "
            f"foreground against itself.  Either pass the bin_scores_<ctx>."
            f"all.bed (which carries the flag), or give this foreground its "
            f"own background with --background-bed {name}=<bed>.")
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
        feat_df, _ = _region_enrichment(bins, fg, gene_features,
                                        args.region_test,
                                        args.region_overlap_frac)
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
        hmm_df, _ = _region_enrichment(bins, fg, cats,
                                       args.chromhmm_test,
                                       args.region_overlap_frac)
        hmm_df.to_csv(tsv("chromhmm_enrichment"), sep="\t", index=False)
        pages.append(("region", (
            f"Chromatin states (by {'group' if grouped else 'state'})", hmm_df)))
        if args.chromhmm_per_state and grouped:
            st = {f"state:{s}": g for s, g in chromhmm.groupby("name")}
            _region_enrichment(bins, fg, st, args.chromhmm_test,
                               args.region_overlap_frac)[0].to_csv(
                tsv("chromhmm_state_enrichment"), sep="\t", index=False)

    # (4) other annotations ------------------------------------------------
    if annotations:
        ann_df, _ = _region_enrichment(bins, fg, annotations,
                                       args.region_test,
                                       args.region_overlap_frac)
        ann_df.to_csv(tsv("annotation_enrichment"), sep="\t", index=False)
        pages.append(("region", ("Other annotations", ann_df)))
        got = _annotation_dotplot(ann_df, out_dir / f"{name}_annotation_dot"
                                  f"plot.pdf", name, args.figure_exclude)
        if got is not None:
            print(f"  figure -> {got.name}")

    # (4b) continuous signal tracks (replication timing, ...) --------------
    if signal_tracks:
        sig_df, series = _signal_enrichment(
            bins, fg, signal_tracks, block_bp=args.signal_block_bp,
            n_boot=args.signal_bootstrap)
        sig_df.to_csv(tsv("signal_enrichment"), sep="\t", index=False)
        pages.append(("signal", ("Continuous signals (fg - bg mean)", sig_df)))
        # Per-bin values (for custom re-plotting) + a standalone publication
        # figure of the foreground-vs-background distributions.
        vals = bins[["chrom", "start", "end"]].copy()
        vals["foreground"] = fg
        for lab, v in series.items():
            vals[lab] = v
        vals.to_csv(out_dir / f"{name}_signal_bin_values.tsv.gz", sep="\t",
                    index=False)
        _signal_figure(series, fg, sig_df, name,
                       out_dir / f"{name}_signal_figure.pdf")

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
            got = _geneset_dotplot(
                ora[ora["fdr"] < args.figure_fdr] if not ora.empty else ora,
                out_dir / f"{name}_{label}_dotplot.pdf", name, label,
                len(fg_genes))
            if got is not None:
                print(f"  figure -> {got.name}")
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
            elif kind == "signal":
                title, df = payload
                fig, ax = plt.subplots(
                    figsize=(8.5, max(3, 0.32 * len(df) + 1.5)))
                _barh_region(ax, df, title, value_col="difference",
                             xlabel="mean signal difference (fg - bg)")
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
    ap.add_argument("--background-bed", action="append", default=None,
                    metavar="NAME=PATH",
                    help="Background bin set for the matching --bed NAME, when "
                         "the foreground file carries no 'selected' column — "
                         "which is every subset BED celltype_marker_bins "
                         "writes (.selected, .stable, .always_co_open, "
                         ".contrast_*), since those drop the membership flags.  "
                         "A marker BED or a plain 3-column BED.  Bins present "
                         "in both are counted as foreground only, so passing "
                         "the .all.bed as the background does the right thing.")
    ap.add_argument("--fdr", type=float, default=None,
                    help="Define foreground by celltype_fdr < FDR instead of the "
                         "'selected' column.  Ignored when --background-bed "
                         "gives the split explicitly.")
    ap.add_argument("--genome", default=None,
                    choices=sorted(_GENOME),
                    help="Auto-DOWNLOAD the standard annotations for this genome "
                         "into --cache-dir and use them for any input not given "
                         "explicitly: GENCODE genes, UCSC chrom.sizes + cytoBand "
                         "+ CpG islands, Enrichr GO_BP + KEGG.  hg38 and hg19 "
                         "add the domain-scale sets that a 250 kb bin can "
                         "actually resolve -- common PMD/HMD (Zhou et al. 2018, "
                         "split by class) and the full-stack ChromHMM "
                         "segmentation collapsed to biological groups -- plus "
                         "Lamin-B1 LADs for hg19, which is the only build UCSC "
                         "ships them for.  The segmentation is ~39 MB and is "
                         "cached after the first run.")
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
    ap.add_argument("--region-test", choices=["fisher", "mwu", "auto"],
                    default="fisher",
                    help="Test for region enrichment.  'fisher' (default) is "
                         "Fisher's exact on 'is this bin in the set', for "
                         "EVERY annotation, so one panel of LADs, PMDs and "
                         "replication timing carries one test and one effect "
                         "measure.  'mwu' is Mann-Whitney on the coverage "
                         "fraction, for when the partly-covered bins are the "
                         "point.  'auto' picks per annotation.  Signal tracks "
                         "(--signal-track) are continuous and always use "
                         "Mann-Whitney; this flag does not touch them.")
    ap.add_argument("--figure-fdr", type=float, default=0.05,
                    help="FDR cut for the gene-set dot plot (0.05).  The TSV "
                         "keeps every term regardless.")
    ap.add_argument("--figure-exclude", default=r"^band:|^CpG_island$",
                    metavar="REGEX",
                    help="Annotations kept out of the publication dot plot "
                         "(the TSV keeps everything).  Default drops the "
                         "Giemsa bands and CpG islands: a band is a stain "
                         "rather than a functional annotation, and a CpG "
                         "island is ~1 kb so its Fisher row is empty rather "
                         "than negative.  Pass an empty string to keep all.")
    ap.add_argument("--chromhmm-test", choices=["mwu", "fisher", "auto"],
                    default="mwu",
                    help="Test for the chromatin-state enrichment, separately "
                         "from --region-test.  Mann-Whitney on the coverage "
                         "fraction by default, because a full-stack segment "
                         "has a MEDIAN length of 200 bp -- 0.08%% of a 250 kb "
                         "bin -- and even a whole state group rarely covers "
                         "20%% of one, so a majority-overlap Fisher test "
                         "returns an empty table for every state.  The graded "
                         "coverage is the only thing there is to test here.")
    ap.add_argument("--region-overlap-frac", type=float, default=0.5,
                    help="A bin is IN an annotation when at least this "
                         "fraction of it is covered (0.5).  Irrelevant for a "
                         "set defined on the bin grid, where coverage is 0 or "
                         "1; for PMDs and the like it decides the "
                         "boundary-straddling bins, 23%% of them at 250 kb.")
    ap.add_argument("--annotation-manifest", action="append", default=None,
                    metavar="TSV",
                    help="A file of 'label<TAB>path_or_url' lines, one per "
                         "annotation, as an alternative to many --annotation "
                         "flags.  REPEATABLE -- pass one per source (LADs, "
                         "replication timing, ...) and they are merged; a "
                         "label defined in two of them is an error rather "
                         "than one quietly winning.  See resources/annotations_human_domains.tsv "
                         "for a starting point with the sources named: at a "
                         "250 kb bin the sets worth testing are DOMAIN-scale "
                         "(LADs, PMDs, compartments), since a peak-scale set "
                         "covers ~0.1%% of a bin and its coverage fraction "
                         "mostly measures gene density.")
    ap.add_argument("--annotation", action="append", default=None,
                    metavar="LABEL=BED|URL",
                    help="Any functional annotation BED (enhancers, CpG islands, "
                         "TF peaks, ...) for coverage enrichment.  Path or URL. "
                         "Repeatable.")
    ap.add_argument("--signal-track", action="append", default=None,
                    metavar="LABEL=BW|BEDGRAPH|URL",
                    help="A CONTINUOUS signal (bigWig .bw or bedGraph; e.g. E/L "
                         "Repli-seq replication timing) for signal enrichment: "
                         "the per-bin MEAN signal is compared foreground-vs-"
                         "background (Mann-Whitney) and reported as a signed "
                         "difference of means.  bigWig needs pyBigWig.  Path or "
                         "URL. Repeatable, e.g. RT_early=early.bw RT_late=late.bw.")
    ap.add_argument("--signal-block-bp", type=int, default=10_000_000,
                    help="Block size for the AUC's block bootstrap (10 Mb).  "
                         "Bins inside one block move together, which is what "
                         "makes the interval honest for a signal organised "
                         "into domains -- replication timing above all.")
    ap.add_argument("--signal-bootstrap", type=int, default=200,
                    help="Block-bootstrap resamples for the AUC interval "
                         "(0 disables).")
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
            # Sniffed, not assumed to be tab: the full-stack state annotation
            # ships as TSV for mouse and CSV for human.
            ann = pd.read_csv(gtsv, sep=None, engine="python", dtype=str)
            gcol = args.chromhmm_group_col
            if gcol not in ann.columns:
                # Case-insensitive, for the same reason: the human file names
                # the column 'Group' and the mouse one 'group'.
                hit_ci = [c for c in ann.columns if c.lower() == gcol.lower()]
                if not hit_ci:
                    sys.exit(f"--chromhmm-group-col {gcol!r} not in {gtsv}: "
                             f"{list(ann.columns)}")
                gcol = hit_ci[0]
            mp, hit, tot, col = _label_to_group(
                chromhmm["name"].unique(), ann, gcol,
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
    manifest = _merge_manifests(args.annotation_manifest or [])
    for k, v in manifest.items():
        if k not in annotations:
            rv = _resolve(v, cache, f"annotation {k} (manifest)")
            if rv is not None:
                annotations[k] = _read_bed(rv)
    for k, v in preset.get("annotations", {}).items():
        spec = v if isinstance(v, dict) else {"url": v}
        if k in annotations:
            continue
        rv = _resolve(spec["url"], cache, f"annotation {k}")
        if rv is None:
            continue
        if spec.get("reader") == "pmd":
            # One file, several domain classes: it expands into named sets
            # rather than being tested as their union.
            for sub_k, sub_df in _read_pmd(rv).items():
                annotations.setdefault(sub_k, sub_df)
        elif str(spec["url"]).endswith(".txt.gz"):
            # UCSC tables carry a leading bin column; _read_ucsc_bed skips it.
            annotations[k] = _read_ucsc_bed(rv, cols=(1, 2, 3))
        else:
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

    # Continuous signal tracks (replication timing, ...): the user's plus the
    # genome preset's.  Resolved to a local path; the per-bin mean is read lazily
    # per context (bigWig via pyBigWig, or a bedGraph).
    signal_tracks = {}
    for k, v in _parse_pairs(args.signal_track, "signal-track").items():
        rv = _resolve(v, cache, f"signal-track {k}")
        if rv is not None:
            signal_tracks[k] = rv
    for label, url in preset.get("signal_tracks", {}).items():
        if label not in signal_tracks:
            rv = _resolve(url, cache, f"signal-track {label}")
            if rv is not None:
                signal_tracks[label] = rv

    backgrounds = _parse_pairs(args.background_bed, "--background-bed")
    beds = _parse_pairs(args.bed, "--bed")
    unknown = set(backgrounds) - set(beds)
    if unknown:
        sys.exit(f"--background-bed names with no matching --bed: "
                 f"{sorted(unknown)}; --bed has {sorted(beds)}.")
    summaries = {}
    for name, path in beds.items():
        rp = _resolve(path, cache, f"--bed {name}")
        if rp is None:
            continue
        bg = backgrounds.get(name)
        if bg:
            bg = _resolve(bg, cache, f"--background-bed {name}")
        s = _run_context(name, rp, args, genes, gene_features, chromhmm,
                         annotations, gene_sets, signal_tracks, chrom_sizes,
                         out_dir, bg_path=bg)
        if s is not None:
            summaries[name] = s
    if not summaries:
        sys.exit("no context produced output.")
    import json
    with open(out_dir / "enrichment_summary.json", "w") as fh:
        json.dump(summaries, fh, indent=2)


if __name__ == "__main__":
    main()
