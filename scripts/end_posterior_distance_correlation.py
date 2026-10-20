#!/usr/bin/env python3
"""Correlation between the two ENDS' posterior levels, versus their separation.

site_distance_correlation.py correlates individual 0/1 site calls, which at one
cytosine on one molecule is close to a coin flip: most of what it measures at
long range is that Bernoulli noise, and the real coordination has to be read
through it.  This script instead uses the Step-4 per-end POSTERIOR — the
probability that the end is open (GCH) or methylated (HCG) given ALL of its
informative sites — so each observation already averages over n sites.

For every read pair the two ends give (p1, p2); pairs are binned by the
separation of their midpoints and a Pearson r is taken per bin.  Three
measures, as in the site-level script:

    gg   GCH(end 1) x GCH(end 2)
    hh   HCG(end 1) x HCG(end 2)
    gh   HCG(one end) x GCH(the other)   -- cross-context

The cross-context measure needs both contexts for the SAME read pair, so the
two posterior tables are joined on (chrom, pos1, pos2); a pair that failed the
site threshold in either context simply does not contribute to gh.

Nulls mirror the site-level script:

    --null-type distance-matched   re-pair two ends of the SAME cell at the
        observed separation (+/- the bin), excluding real ligations.  Answers
        "is this contact more concordant than any two loci that far apart in
        this cell".
    --null-type locus-matched      keep end 1 and swap end 2 for another end
        within +/---null-window of the true partner.  Answers "same two places,
        different molecule".

    --null-type link-shuffle       re-pair contacts across cells: keep end 1
        of a contact and give it, as end 2, an end of ANOTHER cell's contact
        that lies in the same --link-shuffle-bin bin (250 kb) as the real end
        2, at a distance from end 1 in the same log distance bin, from the
        same stratum (cell type from --group-column x --null-match-column,
        e.g. exp_batch,lib_batch).  Where that end's own partner sits does
        not matter, so the curve continues past the bin size.  Answers "same
        place, same distance, same kind of cell -- but the two ends were not
        ligated in one nucleus".  That is --null-scope pooled.  Under
        --null-scope cell the stand-in end 2 is instead another contact end of
        the SAME cell in that bin at that distance, never the real partner:
        same nucleus AND same place, so only the ligation itself differs --
        the strictest null here.  It thins at short range, where a 250 kb bin
        rarely holds a same-cell end at the right distance; there the
        distance-matched null applies anyway.  The pooled version centres
        each end on its own cell's mean in every arm (--cell-center auto), or
        the shared nucleus of a linked pair's two ends -- absent from a
        cross-cell null -- would count as concordance; the same-cell one
        needs no centring.  Within one cell an end at the real partner's
        place can be its own molecule, or the end of a near-copy of the
        contact (a third of a cortex cell's contacts >= 1 kb have another
        with both ends within 1 kb: the same ligation captured twice), so
        --null-exclude W refuses a stand-in within W of the real end 2 or
        ligated to within W of end 1.

    --null-scope cell     donors come from the anchor's own cell.
    --null-scope pooled   donors come from every cell in the group, optionally
        restricted to cells sharing --null-match-column (e.g.
        exp_batch,plate_batch,lib_batch), so a donor never crosses a batch.

Both nulls stay on the anchor's CHROMOSOME.  Ends are placed on a genome-wide
coordinate (chromosome code x 2^33 + position) before any window is searched,
so "two loci d apart" means two loci d apart in cis.  Before this, a cell's
ends were sorted by position alone and a partner "10 kb away" was usually the
same coordinate on another chromosome: on a 19-autosome cell, 93.6% of the
distance-matched null pairs crossed chromosomes, at every distance, so the
long-range null was effectively a trans baseline.

Under either scope a draw that would re-create a real ligation is refused: read
pairs carry run-wide identifiers, so two donor ends belonging to one read pair
are recognised as such even when the pool spans hundreds of cells and the pair
is not the anchor's own.

Linked is tested against null twice, on the same contrast and with the cell as
the unit of inference both times, because end pairs within a cell are nowhere
near independent.  end_posterior_dist_<label> carries a cell-level bootstrap of
Δ = r_linked − r_null; end_posterior_dist_<label>_fisherz carries the Fisher
z version (--fisher-z), which z-transforms each cell's two correlations, pairs
them within the cell and tests the mean difference with a one-sample t.  Both
are BH-adjusted across every bin and measure in their figure.  Agreement
between the two files is the point: one is resampled, the other parametric.

With --group-column, a figure and table are written per cell-type group as well
as for the pooled "all".

--per-cell on (default) also asks whether concordance is detectable inside
EACH cell -- panel a's contrast, one cell at a time, beyond --per-cell-min-
distance (1 kb).  Writes end_posterior_percell.tsv (cell x mark: delta, its
block-bootstrap SE, z, and a negative control from a second independent null
draw) and end_posterior_percell.<fmt>, the Figure 5c panel.  See
nome_hic_concordance/percell_existence.py for why the SE is a block bootstrap
and why the control is a second null draw.

--haplotypes (allelic_imbalance.py --haplotype-dir) asks the same of each
copy of a chromosome.  Every cell is split into units -- chrX's inactive
copy, its active copy and both together (outside PAR1 and the escape
genes), and the same on the autosomes (--units) -- each run as a cell of its
own, so --null-scope cell draws non-ligated pairs of the same cell AND the
same copy: what bulk data, which cannot keep one cell's reads apart, cannot
do.  The inactive X per line: --xi GROUP=1|2, or each cell's own (cells,
--xi-cells).  Per line, end_posterior_units_<line>.tsv/.<fmt>, by
--unit-bands: each copy's Δ; 'same copy', the mean of the two copies' Δ;
'both copies', their pairs pooled against donors of either copy; X against
the autosomes either way; and what pooling the copies makes ('both - same')
on X less on the autosomes -- the cells resampled jointly, since a cell's
units are paired -- and the same over all the bands together.
--unit-element NAME=BED asks it again of the contacts with an end in the
BED's intervals (promoters, CpG islands, ...): that end is kept, the other
replaced by an end of the same cell and copy at the partner's place
(--null-type link-shuffle), drawn from all of the copy's ends, so the
element shapes the contacts tested, not their stand-ins; rows under its
NAME in the table's 'element' column ('all' for every contact).
--unit-split NAME=CHROM:START-END asks it of the contacts crossing the
region -- an end on each side -- and of those with both ends on one side
(DXZ4, the inactive X's megadomain boundary: does concordance across it
differ from concordance within a megadomain, on the Xi and not the Xa?).
A stand-in replaces end 2 near the real one, so a crossing contact's
stand-ins cross too.  Rows under NAME: each copy crossing and on one side,
crossing less one side per copy, and the copies' difference in that.

    python scripts/end_posterior_distance_correlation.py \\
        --posteriors GCH=output/signed_250kb_rwr/context_GCH/04_posteriors \\
        --posteriors HCG=output/signed_250kb_rwr/context_HCG/04_posteriors \\
        --samples mouse_cortex_1577_cells.samples.with_batches.tsv \\
        --group-column group --null-type distance-matched --null-scope cell \\
        --output-dir output/end_posterior_dist
"""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance.distance_stats import (      # noqa: E402
    accumulate, bootstrap_delta, build_edges, corr_from_acc, fdr,
    fisher_z_cell, fisher_z_pooled, new_acc, stars)
from nome_hic_concordance.progress import Progress      # noqa: E402
from nome_hic_concordance import percell_existence as pce  # noqa: E402

logger = logging.getLogger("end_posterior_dist")

_MEASURES = (("gg", "GCH–GCH  r(end$_i$, end$_j$)"),
             ("hh", "HCG–HCG  r(end$_i$, end$_j$)"),
             ("gh", "HCG–GCH  r(end$_i$, end$_j$)  [cross-context]"))
_CONDS = (("linked", "linked (true partner)"),
          ("matched", "matched (pairs the null placed)"),
          ("null", "non-linked"))

#: Which pair set each measure is computed on.  "per-mark" uses every pair
#: informative for the mark(s) a measure needs; "joint" puts all three on the
#: pairs informative for BOTH marks at both ends, as before -- which kept a
#: cell with no HCG output out of GCH-GCH altogether and restricted GCH-GCH to
#: CpG-rich ends.
_SET_OF = {"per-mark": {"gg": "GCH", "hh": "HCG", "gh": "joint"},
           "joint": {"gg": "joint", "hh": "joint", "gh": "joint"}}
#: End values each pair set carries: name -> (end-1 column, end-2 column).
_SET_COLS = {"GCH": {"g": ("p1", "p2")},
             "HCG": {"h": ("p1", "p2")},
             "joint": {"g": ("p1_g", "p2_g"), "h": ("p1_h", "p2_h")}}
_COLOR = {"gg": "#1f77b4", "hh": "#d62728", "gh": "#2ca02c"}

# Pair identifiers must be unique across the WHOLE run, not just within a cell,
# or a pooled donor pool cannot tell a real ligation from a coincidence: the
# per-cell indices collide, and rejecting on them would throw away every draw.
# Striding by the cell's position in the group needs no bookkeeping between the
# pool build and the per-cell pass, since both enumerate the same list.
_PAIR_STRIDE = 1 << 32


def _pair_base(cell_index: int) -> int:
    return int(cell_index) * _PAIR_STRIDE


#: Per-chromosome offset for genome-wide coordinates.  2^33 bp is far beyond
#: any chromosome plus the widest search window (anchor window + max
#: distance), so no null window can reach a neighbouring chromosome; and the
#: largest code times this still fits int64 and float64 exactly.
_GENOME_STRIDE = 1 << 33


def _chrom_code(name) -> int:
    """Stable integer per chromosome NAME, identical in every process.

    Built from the name rather than from a dict filled on first sight: the
    process backend and the pooled donor pool both need two different cells
    to agree on a chromosome's code without sharing state.
    """
    c = _canon(name).upper()
    if c.isdigit():
        return int(c)
    special = {"X": 1001, "Y": 1002, "M": 1003, "MT": 1003}
    if c in special:
        return special[c]
    return 2000 + zlib.crc32(c.encode()) % 100_000


def genome_coords(chrom, pos) -> np.ndarray:
    """``code(chrom) * 2^33 + pos`` as int64, so position sorts stay in cis."""
    pos = np.asarray(pos, dtype=np.int64)
    ch = pd.Series(chrom)
    if isinstance(ch.dtype, pd.CategoricalDtype):
        lut = np.array([_chrom_code(c) for c in ch.cat.categories],
                       dtype=np.int64)
        codes = lut[ch.cat.codes.to_numpy()]
    else:
        names, inv = np.unique(ch.astype(str).to_numpy(), return_inverse=True)
        codes = np.array([_chrom_code(c) for c in names], dtype=np.int64)[inv]
    return codes * _GENOME_STRIDE + pos


# ---------------------------------------------------------------- exclusion --

def _canon(c: str) -> str:
    """Chromosome name without the 'chr' prefix, so a BED and a BAM that
    disagree about it still line up."""
    c = str(c)
    return c[3:] if c.startswith("chr") else c


def load_exclusions(bed_paths, exclude_chrom=()):
    """Merged intervals per chromosome from one or more BED files.

    Merging matters for the lookup below: with non-overlapping, sorted
    intervals a position hits at most the predecessor of its insertion point,
    so the test is one searchsorted rather than a scan.
    """
    excl_chrom = {_canon(c) for c in exclude_chrom if c}
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for path in bed_paths or []:
        with (gzip.open(path, "rt") if str(path).endswith(".gz")
              else open(path)) as fh:
            for line in fh:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                f = line.split()
                if len(f) < 3:
                    continue
                by_chrom.setdefault(_canon(f[0]), []).append(
                    (int(f[1]), int(f[2])))
    merged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c, ivs in by_chrom.items():
        ivs.sort()
        out: list[list[int]] = []
        for s0, e0 in ivs:
            if out and s0 <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e0)
            else:
                out.append([s0, e0])
        merged[c] = (np.array([a for a, _ in out], np.int64),
                     np.array([b for _, b in out], np.int64))
    if merged:
        n = sum(len(v[0]) for v in merged.values())
        logger.info(f"Exclusion set: {n:,} merged intervals over "
                    f"{len(merged)} chromosome(s).")
    return merged, excl_chrom


def _keep_mask(df: pd.DataFrame, merged, excl_chrom) -> np.ndarray:
    """Keep a pair only if NEITHER end falls in an excluded interval.

    Pair-level, matching the site-level script's blacklist rule: a contact with
    one end in a dark region is not half-usable.  Only the end midpoints are
    available here (the read spans are not carried in the posteriors), so a
    read overlapping an interval edge can survive -- acceptable for regions
    that are large relative to a read, which is what these exclusions are.

    Chromosomes are handled through categorical CODES.  The obvious
    ``chrom.map(_canon)`` runs a Python call per row -- 1.6 M of them for two
    contexts of a 400 k-pair cell, which profiling showed was the single
    largest cost in the whole script -- and ``np.unique`` on the string column
    then sorts every row again.  Canonicalising the ~20 category labels
    instead makes both costs independent of the number of pairs.
    """
    n = len(df)
    keep = np.ones(n, dtype=bool)
    cat = df["chrom"].astype("category")
    codes = cat.cat.codes.to_numpy()
    canon = np.array([_canon(c) for c in cat.cat.categories])   # ~20 calls

    if excl_chrom:
        bad = np.flatnonzero(np.isin(canon, list(excl_chrom)))
        if bad.size:
            keep &= ~np.isin(codes, bad)
    if not merged:
        return keep

    p1 = df["pos1"].to_numpy(np.int64)
    p2 = df["pos2"].to_numpy(np.int64)
    for ci, c in enumerate(canon):                 # ~20 iterations, not n
        iv = merged.get(c)
        if iv is None:
            continue
        rows = np.flatnonzero(codes == ci)
        if rows.size == 0:
            continue
        starts, ends = iv
        last = len(ends) - 1
        for pos in (p1[rows], p2[rows]):
            # Intervals are merged and sorted, so only the predecessor of the
            # insertion point can contain the position.
            i = np.searchsorted(starts, pos, side="right") - 1
            hit = (i >= 0) & (ends[np.clip(i, 0, last)] > pos)
            keep[rows[hit]] = False
    return keep


# ------------------------------------------------------------------ loading --

def _load_cell(paths: dict[str, Path], cell: str, min_sites: int,
               merged=None, excl_chrom=()):
    """One cell's per-end posteriors, per context, keyed by read pair.

    Returns ``{ctx: DataFrame[chrom,pos1,pos2,distance,p1,p2]}``; contexts with
    no file for this cell are omitted rather than faked.

    ``chrom`` comes back CATEGORICAL, read straight out of the parquet
    dictionary instead of being materialised as one Python string per row.
    The join in :func:`joined_pairs` is the hottest step of the run, and
    factorising an object column there hashes a Python object per row while
    HOLDING THE GIL -- measured at 1.63x scaling on 8 threads against 2.48x
    once the column is categorical.  The contexts are then given a common
    category set, because pandas falls back to object when merging
    categoricals whose categories differ, which would hand the cost straight
    back; the union is over ~25 chromosome names, not rows.
    """
    out = {}
    for ctx, d in paths.items():
        p = d / f"{cell}.posteriors.parquet"
        if not p.exists():
            continue
        ms = (min_sites.get(ctx, 2) if isinstance(min_sites, dict)
              else min_sites)
        df = pq.read_table(p, columns=["chrom", "pos1", "pos2", "n1", "n2",
                                       "distance", "posterior1", "posterior2"]
                           ).to_pandas(categories=["chrom"])
        df = df[(df["n1"] >= ms) & (df["n2"] >= ms)
                & df["posterior1"].notna() & df["posterior2"].notna()]
        if df.empty:
            continue
        if merged or excl_chrom:
            df = df[_keep_mask(df, merged, excl_chrom)]
            if df.empty:
                continue
        out[ctx] = df.rename(columns={"posterior1": "p1", "posterior2": "p2"})[
            ["chrom", "pos1", "pos2", "distance", "p1", "p2"]]
    if len(out) > 1:
        cats = None
        for f in out.values():
            c = f["chrom"].cat.categories
            cats = c if cats is None else cats.union(c)
        for ctx, f in out.items():
            out[ctx] = f.assign(chrom=f["chrom"].cat.set_categories(cats))
    return out


#: --haplotypes: a cell's read pairs split into units, each run as if it were
#: a cell of its own, so the within-cell null (--null-scope cell) draws from
#: the unit's own ends -- the same cell AND the same copy; a pooled null's
#: strata are kept per unit, so its donors are of the same copy too.  The
#: unit rides in the cell's name after UNIT_SEP.
UNIT_SEP = "::"
#: (chromosomes, copy) per unit: chrX outside PAR1 and the escape genes, or
#: the autosomes; the copy on the cell's inactive X's haplotype, the other
#: one, or both together -- whose within-cell null draws from either copy.
UNITS = {"x_xi": ("X", "xi"), "x_xa": ("X", "xa"), "x_both": ("X", "both"),
         "a_xi": ("A", "xi"), "a_xa": ("A", "xa"), "a_both": ("A", "both")}


def unit_label(key: str, named: bool = True) -> str:
    """A unit's name in groups and figures; ``named``: the line has an
    inactive X called (else the copies are hap1 and hap2)."""
    chrom, copy = UNITS[key]
    one, two = ("Xi", "Xa") if named else ("hap1", "hap2")
    if chrom == "X":
        return {"xi": f"chrX {one}", "xa": f"chrX {two}",
                "both": "chrX both copies"}[copy]
    return {"xi": f"autosomes {one}'s haplotype" if named
            else "autosomes hap1",
            "xa": f"autosomes {two}'s haplotype" if named
            else "autosomes hap2",
            "both": "autosomes both copies"}[copy]


def _hap_frame(hap_dir, cell: str):
    """A cell's read pairs on one haplotype (allelic_imbalance.py
    --haplotype-dir), keyed as the posteriors key a read pair -- chrom and
    its two ends' midpoints, low and high -- a pair given both haplotypes
    left out: a Series of the haplotype by (chrom, lo, hi); None without a
    file."""
    p = Path(hap_dir) / f"{cell}.haplotypes.parquet"
    if not p.exists():
        return None
    t = pd.read_parquet(p, columns=["chrom", "pos1", "pos2", "hap"])
    p1 = t["pos1"].to_numpy(np.int64)
    p2 = t["pos2"].to_numpy(np.int64)
    g = pd.DataFrame({"chrom": t["chrom"].astype(str).to_numpy(),
                      "lo": np.minimum(p1, p2), "hi": np.maximum(p1, p2),
                      "hap": t["hap"].to_numpy(np.int8)}).groupby(
        ["chrom", "lo", "hi"], sort=False)["hap"].agg(["min", "max"])
    return g.loc[g["min"] == g["max"], "min"]


def _unit_filter(per_ctx: dict, haps, key: str, xi_hap: int, par=None,
                 escape=None) -> dict:
    """Each context's read pairs of unit ``key`` (UNITS): tagged in
    ``haps`` (:func:`_hap_frame`), on its chromosomes -- on X with both ends
    outside PAR1 (``par``) and the escape genes (``escape``) -- and on its
    copy."""
    from nome_hic_concordance.xci import in_intervals
    chroms, copy = UNITS[key]
    want = np.array(sorted({"xi": {xi_hap}, "xa": {3 - xi_hap},
                            "both": {1, 2}}[copy]), np.int8)
    out = {}
    for ctx, f in per_ctx.items():
        name = f["chrom"].astype(str).to_numpy()
        canon = np.array([_canon(c).upper() for c in name], dtype=object)
        on = (canon == "X") if chroms == "X" else np.array(
            [c.isdigit() for c in canon], bool)
        if not on.any() or haps is None or not len(haps):
            continue
        p1 = f["pos1"].to_numpy(np.int64)
        p2 = f["pos2"].to_numpy(np.int64)
        lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
        h = haps.reindex(pd.MultiIndex.from_arrays([name, lo, hi])).to_numpy(
            float)
        keep = on & np.isin(np.nan_to_num(h, nan=0).astype(np.int8), want)
        if chroms == "X":
            for a_, b_ in ([par] if par else []):
                keep &= ~((lo >= a_) & (lo < b_)) & ~((hi >= a_) & (hi < b_))
            if escape is not None and len(escape):
                keep &= ~in_intervals(lo, escape) & ~in_intervals(hi, escape)
        if keep.any():
            out[ctx] = f[keep]
    return out


def _unit_key(key: str):
    """A unit's key split: its UNITS key, and the --unit-element its
    contacts are focused on (None: every contact)."""
    base, _, elem = key.partition("@")
    return base, (elem or None)


def read_unit_elements(specs, flank: int) -> dict:
    """--unit-element NAME=BED: per name, the BED's intervals widened by
    ``flank`` each side, as sorted, disjoint (start, end) rows on the
    genome-wide coordinate (:func:`genome_coords`)."""
    out = {}
    for spec in specs or []:
        name, _, path = str(spec).partition("=")
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise SystemExit(f"--unit-element {spec}: NAME=BED")
        if "@" in name or UNIT_SEP in name:
            raise SystemExit(f"--unit-element {spec}: no '@' or "
                             f"'{UNIT_SEP}' in the name")
        b = pd.read_csv(path, sep="\t", header=None, comment="#",
                        usecols=[0, 1, 2], dtype={0: str})
        b = b[~b[0].str.startswith(("track", "browser"))]
        chrom = b[0].to_numpy()
        lo = genome_coords(chrom, np.maximum(
            b[1].to_numpy(np.int64) - flank, 0))
        hi = genome_coords(chrom, b[2].to_numpy(np.int64) + flank)
        o = np.argsort(lo, kind="stable")
        lo, hi = lo[o], hi[o]
        # merged, so the rows are disjoint for in_intervals
        new = np.r_[True, lo[1:] > np.maximum.accumulate(hi)[:-1]]
        out[name] = np.column_stack(
            [lo[new], np.maximum.reduceat(hi, np.flatnonzero(new))])
    return out


#: --unit-split's two kinds of contact: an end on each side of the region,
#: or both ends on one side of it (neither inside).
SPLIT_PARTS = ("cross", "side")


def read_unit_splits(specs) -> dict:
    """--unit-split NAME=CHROM:START-END: per name, the region and its
    chromosome's bounds on the genome-wide coordinate (:func:`genome_coords`)
    -- 'lo', 'hi', 'c0', 'c1' -- and 'set', the units' chromosomes it lies on
    (UNITS: 'X' or 'A')."""
    out = {}
    for spec in specs or []:
        name, _, reg = str(spec).partition("=")
        name, reg = name.strip(), reg.strip().replace(",", "")
        chrom, _, span = reg.rpartition(":")
        a_, _, b_ = span.partition("-")
        if not name or not chrom or not (a_.isdigit() and b_.isdigit()
                                         and int(a_) < int(b_)):
            raise SystemExit(f"--unit-split {spec}: NAME=CHROM:START-END")
        if any(c in name for c in ("@", ":", UNIT_SEP)):
            raise SystemExit(f"--unit-split {spec}: no '@' or ':' in the "
                             f"name")
        c = _canon(chrom).upper()
        if c != "X" and not c.isdigit():
            raise SystemExit(f"--unit-split {spec}: on chrX or an autosome, "
                             f"as the units are")
        c0 = int(genome_coords([chrom], [0])[0])
        out[name] = {"lo": c0 + int(a_), "hi": c0 + int(b_), "c0": c0,
                     "c1": c0 + _GENOME_STRIDE, "set": "X" if c == "X"
                     else "A", "region": f"{chrom}:{int(a_)}-{int(b_)}"}
    return out


def _split_frame(j: pd.DataFrame, region: dict, part: str, base: int):
    """``j``'s pairs crossing ``region`` (:func:`read_unit_splits`) -- an
    end on each side -- or, ``part`` 'side', with both ends on one side of
    it, neither inside; and their ids in the whole of ``j`` (base + row)."""
    g1 = j["gpos1"].to_numpy(np.int64)
    g2 = j["gpos2"].to_numpy(np.int64)
    lo, hi = np.minimum(g1, g2), np.maximum(g1, g2)
    on = (lo >= region["c0"]) & (hi < region["c1"])
    if part == "cross":
        m = on & (lo < region["lo"]) & (hi >= region["hi"])
    else:
        m = on & ((hi < region["lo"]) | (lo >= region["hi"]))
    sel = np.flatnonzero(m)
    return (j.iloc[sel].reset_index(drop=True),
            base + sel.astype(np.int64))


def _focus_frame(j: pd.DataFrame, cols: dict, iv, base: int):
    """``j``'s pairs with an end in the intervals ``iv`` (genome
    coordinates), each turned so that end is end 1 -- the end a
    link-shuffle keeps -- and their ids in the whole of ``j``, as its pool
    knows them (base + row)."""
    from nome_hic_concordance.xci import in_intervals
    g1 = j["gpos1"].to_numpy(np.int64)
    g2 = j["gpos2"].to_numpy(np.int64)
    in1, in2 = in_intervals(g1, iv), in_intervals(g2, iv)
    sel = np.flatnonzero(in1 | in2)
    f = j.iloc[sel].reset_index(drop=True)
    flip = ~in1[sel]
    if flip.any():
        upd = {}
        for a, b in [("gpos1", "gpos2"), ("pos1", "pos2"),
                     *(tuple(c) for c in cols.values())]:
            if a in f.columns and b in f.columns:
                va, vb = f[a].to_numpy(), f[b].to_numpy()
                upd[a], upd[b] = np.where(flip, vb, va), np.where(flip, va,
                                                                  vb)
        f = f.assign(**upd)
    return f, base + sel.astype(np.int64)


def _unit_focus(cell: str, args):
    """How a unit's contacts are focused, or None: a function of (frame,
    its columns, the cell's pair base) giving the contacts it keeps and
    their ids in the whole frame -- those with an end in an element
    (:func:`_focus_frame`), or crossing a region or on one side of it
    (:func:`_split_frame`)."""
    cfg = getattr(args, "units_cfg", None)
    if not cfg or UNIT_SEP not in cell:
        return None
    elem = _unit_key(cell.split(UNIT_SEP, 1)[1])[1]
    if elem is None:
        return None
    if elem in cfg.get("elements", {}):
        iv = cfg["elements"][elem]
        return lambda j, cols, base: _focus_frame(j, cols, iv, base)
    name, _, part = elem.rpartition(":")
    region = cfg["splits"][name]
    return lambda j, cols, base: _split_frame(j, region, part, base)


def _load_any(paths: dict, cell: str, min_sites, merged, excl_chrom, args):
    """:func:`_load_cell`, or with --haplotypes and a unit in the name
    (UNIT_SEP) that cell's read pairs of the unit -- all of them for a unit
    focused on an element, whose stand-ins are drawn from every end."""
    cfg = getattr(args, "units_cfg", None)
    if not cfg or UNIT_SEP not in cell:
        return _load_cell(paths, cell, min_sites, merged, excl_chrom)
    name, key = cell.split(UNIT_SEP, 1)
    per_ctx = _load_cell(paths, name, min_sites, merged, excl_chrom)
    if not per_ctx:
        return {}
    return _unit_filter(per_ctx, _hap_frame(cfg["dir"], name),
                        _unit_key(key)[0], int(cfg["xi"].get(name, 1)),
                        cfg.get("par"), cfg.get("escape"))


#: The units' comparisons (unit_comparisons): (name, {unit: weight}).
#: "same" is the mean of the two copies' Δ, each against donors of its own
#: copy -- not their pairs pooled into one correlation, where the copies'
#: different levels would enter the linked and the null r alike and squeeze
#: their difference; "both" is the same pairs pooled against donors of
#: either copy, so "both - same" is what pooling the copies makes.
UNIT_COMPARISONS = (
    ("chrX Xi − chrX Xa", {"x_xi": 1, "x_xa": -1}),
    # the control of the one above: the same haplotypes on the autosomes
    ("autosomes: Xi's − Xa's haplotype", {"a_xi": 1, "a_xa": -1}),
    ("chrX − autosomes, same copy", {"x_same": 1, "a_same": -1}),
    ("chrX − autosomes, both copies", {"x_both": 1, "a_both": -1}),
    ("chrX: both copies − same copy", {"x_both": 1, "x_same": -1}),
    ("autosomes: both copies − same copy", {"a_both": 1, "a_same": -1}),
    ("copy part: chrX − autosomes", {"x_both": 1, "x_same": -1,
                                     "a_both": -1, "a_same": 1}))
#: Each unit alone, as the comparisons' rows name it.
UNIT_SINGLES = ("x_xi", "x_xa", "x_same", "x_both", "a_xi", "a_xa",
                "a_same", "a_both")


def unit_comparisons(kept: dict, labels: dict, edges, bands, args,
                     named: bool = True, combos=None) -> pd.DataFrame:
    """One line's units against each other, by distance band: Δ (the
    --delta-arm arm less the null) of each unit and of UNIT_COMPARISONS'
    combinations, with 95% intervals and two-sided p from resampling the
    CELLS jointly -- a cell's units move together, so a comparison between
    them is paired.  ``kept``: {group label: per-cell accumulators, in the
    cells' order}; ``labels``: {unit key: its group label}; ``combos``:
    [(name, {unit key: weight})] in place of the units and
    UNIT_COMPARISONS."""
    arm = getattr(args, "delta_arm", "linked")
    have = {k: kept.get(lab) for k, lab in labels.items()}
    have = {k: v for k, v in have.items() if v is not None}
    if not have:
        return pd.DataFrame()
    n = len(next(iter(have.values())))
    if n < 3 or any(len(v) != n for v in have.values()):
        return pd.DataFrame()
    centers = np.sqrt(edges[:-1] * edges[1:])
    bix = np.searchsorted(bands, centers, side="right") - 1
    nbd = len(bands) - 1
    # the bands, then all of them together
    band_of = [(bands[j], bands[j + 1], f"{_bp(bands[j])}-{_bp(bands[j + 1])}")
               for j in range(nbd)]
    band_of.append((bands[0], bands[-1],
                    f"all {_bp(bands[0])}-{_bp(bands[-1])}"))
    rng = np.random.default_rng(args.seed + 99)
    W = np.vstack([np.ones((1, n)),
                   rng.multinomial(n, np.full(n, 1.0 / n),
                                   size=max(args.n_bootstrap, 1))
                   .astype(float)])
    rows = []
    for m, _ in _MEASURES:
        delta = {}
        for k, pc in have.items():
            arrs = []
            for cond in (arm, "null"):
                a = np.stack([x[(m, cond)] for x in pc]).astype(float)
                bd = np.zeros(a.shape[:2] + (nbd + 1,))
                for j in range(nbd):
                    sel = bix == j
                    if sel.any():
                        bd[:, :, j] = a[:, :, sel].sum(-1)
                bd[:, :, nbd] = bd[:, :, :nbd].sum(-1)
                arrs.append(bd)
            delta[k] = arrs
        d_of = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            for k, (ao, an) in delta.items():
                ro = corr_from_acc((W @ ao.reshape(n, -1)).reshape(
                    W.shape[0], 6, nbd + 1), args.min_pairs)[0]
                rn = corr_from_acc((W @ an.reshape(n, -1)).reshape(
                    W.shape[0], 6, nbd + 1), args.min_pairs)[0]
                d_of[k] = ro - rn
        for tot, (u1, u2) in (("x_same", ("x_xi", "x_xa")),
                              ("a_same", ("a_xi", "a_xa"))):
            if u1 in d_of and u2 in d_of:
                d_of[tot] = (d_of[u1] + d_of[u2]) / 2
        todo = combos if combos is not None else (
            [(unit_label(k, named) if k in UNITS else
              {"x_same": "chrX same copy",
               "a_same": "autosomes same copy"}[k], {k: 1})
             for k in UNIT_SINGLES] + list(UNIT_COMPARISONS))
        for name, wts in todo:
            if any(k not in d_of for k in wts):
                continue
            if not named:
                name = name.replace("Xi", "hap1").replace("Xa", "hap2")
            v = sum(w_ * d_of[k] for k, w_ in wts.items())
            for j, (b_lo, b_hi, b_lab) in enumerate(band_of):
                b = v[1:, j]
                b = b[np.isfinite(b)]
                lo, hi = (np.percentile(b, [2.5, 97.5]) if b.size >= 20
                          else (np.nan, np.nan))
                p = (min(1.0, 2.0 * (1 + min((b <= 0).sum(), (b >= 0).sum()))
                         / (b.size + 1)) if b.size >= 20 else np.nan)
                rows.append({"measure": m, "comparison": name,
                             "band_lo": b_lo, "band_hi": b_hi,
                             "band": b_lab, "value": v[0, j], "lo": lo,
                             "hi": hi, "p": p, "cells": n})
    return pd.DataFrame(rows)


def split_combos(name: str, bases, named: bool = True) -> list:
    """--unit-split NAME's rows: per copy (UNITS key in ``bases``) its
    contacts crossing the region and on one side, crossing less one side;
    for two copies, the first's crossing less one side less the second's,
    and the copies' difference among the crossing and among the one-side
    contacts.  Keys 'unit|cross' and 'unit|side'."""
    out = []
    for b in bases:
        lab = unit_label(b, named)
        out += [(f"{lab} crossing {name}", {f"{b}|cross": 1}),
                (f"{lab} same side of {name}", {f"{b}|side": 1}),
                (f"{lab}: crossing − same side",
                 {f"{b}|cross": 1, f"{b}|side": -1})]
    if len(bases) == 2:
        u, v = bases
        lu, lv = unit_label(u, named), unit_label(v, named)
        out += [(f"crossing − same side: {lu} − {lv}",
                 {f"{u}|cross": 1, f"{u}|side": -1, f"{v}|cross": -1,
                  f"{v}|side": 1}),
                (f"crossing: {lu} − {lv}", {f"{u}|cross": 1,
                                            f"{v}|cross": -1}),
                (f"same side: {lu} − {lv}", {f"{u}|side": 1,
                                             f"{v}|side": -1})]
    return out


def split_figure(t: pd.DataFrame, line: str, name: str, path) -> bool:
    """--unit-split NAME, per measure: a, each copy's Δ for the contacts
    crossing the region and those on one side, by distance band; b,
    crossing less one side per copy, and the copies' difference in that."""
    if t is None or not len(t):
        return False
    t = t[~t["band"].astype(str).str.startswith("all")]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meas = [m for m, _ in _MEASURES if (t["measure"] == m).any()]
    comps = list(dict.fromkeys(t["comparison"]))
    singles = [c for c in comps if c.endswith(f"crossing {name}")
               or c.endswith(f"same side of {name}")]
    diffs = [c for c in comps if ": crossing − same side" in c
             or c.startswith("crossing − same side:")]
    # the first copy red, the second blue, crossing dark and one side light;
    # the copies' difference black
    pal = (["#b2182b", "#ef8a62", "#2166ac", "#67a9cf"],
           ["#b2182b", "#2166ac", "black"])
    fig, axes = plt.subplots(2, len(meas), figsize=(3.4 * len(meas), 5.2),
                             squeeze=False)
    for j, m in enumerate(meas):
        for row, items in ((0, singles), (1, diffs)):
            ax = axes[row, j]
            for i, nm in enumerate(items):
                u = t[(t["measure"] == m) & (t["comparison"] == nm)]
                if not len(u):
                    continue
                x = np.sqrt(u["band_lo"] * u["band_hi"]) * (
                    1.12 ** (i - (len(items) - 1) / 2))
                ax.errorbar(x, u["value"], yerr=[u["value"] - u["lo"],
                                                 u["hi"] - u["value"]],
                            fmt="o", ms=3, lw=0.8, capsize=1.5,
                            color=pal[row][i % len(pal[row])], label=nm)
            ax.axhline(0, color="0.6", lw=0.5)
            _log_x(ax, float(t["band_lo"].min()), float(t["band_hi"].max()))
            ax.set_xlabel("distance between the ends, bp", fontsize=6)
            ax.set_ylabel("Δr, linked − within-cell null" if row == 0
                          else "difference in Δr", fontsize=6)
            ax.tick_params(labelsize=5.5)
            ax.legend(fontsize=4.8, frameon=False)
            ax.set_title(f"{'ab'[row]}  {dict(_MEASURES)[m].split('  ')[0]}",
                         loc="left", fontsize=6.5)
    fig.suptitle(f"{line}: contacts crossing {name} against those on one "
                 f"side, each copy against donors of its own", fontsize=7,
                 y=1.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return True


def _bp(v) -> str:
    """1000 -> '1kb', 10000000 -> '10Mb', 300 -> '300bp'."""
    v = int(round(float(v)))
    for unit, f in (("Mb", 1_000_000), ("kb", 1_000)):
        if v >= f and v % f == 0:
            return f"{v // f}{unit}"
    return f"{v}bp"


def unit_figure(t: pd.DataFrame, line: str, path) -> bool:
    """Per measure: a, each copy's Δ against its within-cell null on X and
    on the autosomes, by distance band; b, the comparisons -- X against the
    autosomes, and the part the copies differing makes."""
    if t is None or not len(t):
        return False
    if "element" in t.columns:
        t = t[t["element"] == "all"]
    if "band" in t.columns:
        t = t[~t["band"].astype(str).str.startswith("all")]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meas = [m for m, _ in _MEASURES if (t["measure"] == m).any()]
    singles = (("chrX same copy", "#b2182b"), ("chrX both copies", "#ef8a62"),
               ("autosomes same copy", "0.25"),
               ("autosomes both copies", "0.6"))
    comps = (("chrX − autosomes, same copy", "#b2182b"),
             ("chrX − autosomes, both copies", "#ef8a62"),
             ("copy part: chrX − autosomes", "black"))
    fig, axes = plt.subplots(2, len(meas), figsize=(3.4 * len(meas), 5.2),
                             squeeze=False)
    for j, m in enumerate(meas):
        for row, items in ((0, singles), (1, comps)):
            ax = axes[row, j]
            for i, (nm, col) in enumerate(items):
                u = t[(t["measure"] == m) & (t["comparison"] == nm)]
                if not len(u):
                    continue
                x = np.sqrt(u["band_lo"] * u["band_hi"]) * (
                    1.12 ** (i - (len(items) - 1) / 2))
                ax.errorbar(x, u["value"], yerr=[u["value"] - u["lo"],
                                                 u["hi"] - u["value"]],
                            fmt="o", ms=3, lw=0.8, capsize=1.5, color=col,
                            label=nm)
            ax.axhline(0, color="0.6", lw=0.5)
            _log_x(ax, float(t["band_lo"].min()), float(t["band_hi"].max()))
            ax.set_xlabel("distance between the ends, bp", fontsize=6)
            ax.set_ylabel("Δr, linked − within-cell null" if row == 0
                          else "difference in Δr", fontsize=6)
            ax.tick_params(labelsize=5.5)
            ax.legend(fontsize=4.8, frameon=False)
            ax.set_title(f"{'ab'[row]}  {dict(_MEASURES)[m].split('  ')[0]}",
                         loc="left", fontsize=6.5)
    fig.suptitle(f"{line}: each copy against donors of the same cell -- of "
                 f"its own copy, or of either", fontsize=7, y=1.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return True


def _unit_setup(samples: pd.DataFrame, cells, args, batch_of, excl_chrom,
                paths=None):
    """--haplotypes: the cells split into units (UNITS), one group per line
    (--group-column) and unit; each cell's inactive X (--xi); a pooled
    null's strata kept per unit.  Returns (groups, the units' strata,
    {line: ({unit key: group label}, the copies named)})."""
    from nome_hic_concordance.xci import (cell_xi_table, parse_escape,
                                          parse_par)
    if any(_canon(c).upper() == "X" for c in excl_chrom):
        raise SystemExit("--haplotypes: chrX is in --exclude-chrom")
    keys = [k.strip() for k in args.units.split(",") if k.strip()]
    bad = set(keys) - set(UNITS)
    if bad:
        raise SystemExit(f"--units: unknown {sorted(bad)}; choose from "
                         f"{list(UNITS)}")
    elements = read_unit_elements(args.unit_element,
                                  int(args.unit_element_flank))
    if elements and args.null_type != "link-shuffle":
        raise SystemExit("--unit-element needs --null-type link-shuffle: "
                         "it keeps the element's end and replaces the other")
    splits = read_unit_splits(getattr(args, "unit_split", None))
    if splits and args.null_type != "link-shuffle":
        raise SystemExit("--unit-split needs --null-type link-shuffle: its "
                         "stand-ins must stay on the real end's side")
    clash = set(splits) & set(elements)
    if clash:
        raise SystemExit(f"--unit-split and --unit-element share "
                         f"{sorted(clash)}")
    # per element the copies one by one -- 'both' is about pooling copies
    ekeys = [f"{k}@{nm}" for nm in elements for k in keys
             if UNITS[k][1] != "both"]
    # per region the copies one by one, of the chromosomes it is on
    ekeys += [f"{k}@{nm}:{part}" for nm, reg in splits.items()
              for part in SPLIT_PARTS for k in keys
              if UNITS[k][1] != "both" and UNITS[k][0] == reg["set"]]
    for nm, reg in splits.items():
        if not any(k.startswith(f"{u}@{nm}:") for k in ekeys for u in keys):
            raise SystemExit(f"--unit-split {nm}: none of --units "
                             f"{','.join(keys)} is a single copy of "
                             f"{'chrX' if reg['set'] == 'X' else 'the autosomes'}")
    xi = {}
    for v in args.xi or []:
        for tok in str(v).split(","):
            g, _, m = tok.partition("=")
            if g.strip():
                if m.strip() not in ("1", "2", "cells"):
                    raise SystemExit(f"--xi {tok}: GROUP=1, 2 or cells")
                xi[g.strip()] = m.strip()
    calls = {}
    if args.xi_cells:
        t = cell_xi_table(args.xi_cells, args.xi_context,
                          args.min_cell_sites)
        calls = dict(zip(t["cell"].astype(str), t["xi_hap"]))
    elif "cells" in xi.values():
        raise SystemExit("--xi GROUP=cells needs --xi-cells")
    idx = samples.set_index(args.cell_column)
    line_of = (idx[args.group_column].astype(str).to_dict()
               if args.group_column else {})
    genome = None if str(args.genome).lower() == "none" else args.genome
    par = parse_par(args.chrx_par, genome)
    esc = parse_escape(args.chrx_escape, genome, par)
    hd = Path(args.haplotypes)
    xi_of, groups, lines, strata = {}, {}, {}, {}
    n_file = n_call = 0
    for c in cells:
        if not (hd / f"{c}.haplotypes.parquet").exists():
            n_file += 1
            continue
        line = line_of.get(c, "all")
        mode = xi.get(line)
        if mode == "cells":
            h = calls.get(c)
            if h is None:
                n_call += 1
                continue
        else:
            h = int(mode) if mode else 1
        xi_of[c] = int(h)
        labs = lines.setdefault(line, (
            {k: f"{line} {unit_label(_unit_key(k)[0], mode is not None)}"
                + _focus_suffix(_unit_key(k)[1])
             for k in keys + ekeys},
            mode is not None))[0]
        for k in keys + ekeys:
            u = f"{c}{UNIT_SEP}{k}"
            groups.setdefault(labs[k], []).append(u)
            strata[u] = (f"{batch_of.get(c, '')}|{k}" if batch_of
                         else f"|{k}")
    if not xi_of:
        raise SystemExit(f"--haplotypes: no cell has {hd}/<cell>"
                         f".haplotypes.parquet (and an inactive X call)")
    if paths:
        _probe_haplotypes(paths, hd, list(xi_of)[:3], args)
    args.units_cfg = {"dir": str(hd), "xi": xi_of, "par": par,
                      "escape": esc, "elements": elements, "splits": splits}
    n_split = sum(":" in (_unit_key(k)[1] or "") for k in ekeys)
    logger.info(f"--haplotypes: {len(xi_of):,} cells in "
                f"{len(keys) + len(ekeys)} units each "
                f"({len(ekeys) - n_split} at --unit-element's "
                f"{len(elements)} elements, {n_split} crossing or beside "
                f"--unit-split's {len(splits)} regions"
                + "".join(f" [{nm} {r['region']}]" for nm, r in splits.items())
                + f"; {n_file} cells without a haplotype file, {n_call} "
                f"without an inactive-X call); lines: {sorted(lines)}")
    return groups, strata, lines


def _focus_suffix(elem) -> str:
    """A focused unit's label after its copy's: ' at ELEMENT', ' crossing
    NAME' or ' same side of NAME'; '' for every contact."""
    if elem is None:
        return ""
    name, sep, part = elem.rpartition(":")
    if sep and part in SPLIT_PARTS:
        return f" crossing {name}" if part == "cross" else \
            f" same side of {name}"
    return f" at {elem}"


def _probe_haplotypes(paths: dict, hd: Path, cells, args) -> None:
    """A few cells' posteriors' read pairs against their haplotype files, as
    the units find them (chrom and the two ends): how many carry a
    haplotype, stopping where none does -- files made from other read pairs
    would leave every unit empty."""
    tagged = total = 0
    for c in cells:
        haps = _hap_frame(hd, c)
        if haps is None:
            continue
        for f in _load_cell(paths, c, _min_sites(args)).values():
            p1 = f["pos1"].to_numpy(np.int64)
            p2 = f["pos2"].to_numpy(np.int64)
            key = pd.MultiIndex.from_arrays(
                [f["chrom"].astype(str).to_numpy(), np.minimum(p1, p2),
                 np.maximum(p1, p2)])
            tagged += int(haps.reindex(key).notna().sum())
            total += len(f)
    logger.info(f"--haplotypes: of {total:,} read pairs in {len(cells)} "
                f"cells' posteriors, {tagged:,} carry a haplotype")
    if total and not tagged:
        raise SystemExit(
            f"--haplotypes: none of {total:,} read pairs in {len(cells)} "
            f"cells' posteriors is in {hd}'s files (chrom, pos1, pos2): "
            f"were the haplotypes made from these posteriors' read pairs?")
    if not total:
        raise SystemExit(f"--haplotypes: {len(cells)} cells with haplotype "
                         f"files have no posteriors' read pairs: check "
                         f"--posteriors")


def _pair_frames(per_ctx: dict[str, pd.DataFrame]):
    """The (x, y, distance) inputs for each measure.

    gg / hh are within one context.  gh joins the two contexts on the read-pair
    key so the two levels come from the SAME molecule pair; both orderings are
    emitted (HCG@end1 x GCH@end2 and the reverse), matching the site-level
    script's treatment.
    """
    frames: dict[str, list[tuple]] = {"gg": [], "hh": [], "gh": []}
    g, h = per_ctx.get("GCH"), per_ctx.get("HCG")
    if g is not None:
        frames["gg"].append((g["p1"].to_numpy(), g["p2"].to_numpy(),
                             g["distance"].to_numpy()))
    if h is not None:
        frames["hh"].append((h["p1"].to_numpy(), h["p2"].to_numpy(),
                             h["distance"].to_numpy()))
    if g is not None and h is not None:
        j = h.merge(g, on=["chrom", "pos1", "pos2"], suffixes=("_h", "_g"))
        if len(j):
            d = j["distance_h"].to_numpy()
            frames["gh"].append((j["p1_h"].to_numpy(), j["p2_g"].to_numpy(), d))
            frames["gh"].append((j["p2_h"].to_numpy(), j["p1_g"].to_numpy(), d))
    return frames


def _min_sites(args):
    """Per-context minimum sites per END.

    ``--min-sites`` given explicitly still applies to both, so older commands
    mean what they meant; otherwise GCH and HCG take their own floors.  A
    minimal args object (the tests build them) falls back to ``min_sites``.
    """
    explicit = getattr(args, "min_sites", None)
    if explicit is not None:
        return {"GCH": int(explicit), "HCG": int(explicit)}
    g = getattr(args, "min_sites_gch", None)
    h = getattr(args, "min_sites_hcg", None)
    return {"GCH": int(2 if g is None else g), "HCG": int(2 if h is None else h)}


def set_frames(per_ctx: dict[str, pd.DataFrame], sets):
    """One frame per pair set, each carrying genome-wide coordinates.

    "GCH" / "HCG" are that context's own pairs; "joint" is the join of both.
    A set with no rows is simply absent, so a cell with GCH but no HCG output
    still contributes GCH-GCH.
    """
    out = {}
    for s in sets:
        if s == "joint":
            f = joined_pairs(per_ctx)
        else:
            f = per_ctx.get(s)
            if f is not None and len(f):
                f = f.assign(gpos1=genome_coords(f["chrom"], f["pos1"]),
                             gpos2=genome_coords(f["chrom"], f["pos2"]))
        if f is not None and len(f):
            out[s] = f
    return out


def _linked_xy(m, frame, cols, idx=None):
    """(x, y) arrays of the linked arm of measure ``m`` (gh: both orders)."""
    def col(name, end):
        v = frame[cols[name][end]].to_numpy()
        return v if idx is None else v[idx]
    if m == "gg":
        return [(col("g", 0), col("g", 1))]
    if m == "hh":
        return [(col("h", 0), col("h", 1))]
    return [(col("h", 0), col("g", 1)), (col("h", 1), col("g", 0))]


def _null_xy(m, xa, xb):
    """(x, y) arrays of a null arm of measure ``m`` from donor values.

    gh keeps HCG on x and GCH on y in BOTH orders, exactly as the linked arm
    does.  The two orders are pooled into one correlation, so swapping the
    roles in the second -- (GCH of a, HCG of b), as every version before this
    did -- put methylation (mean ~0.65) and accessibility (mean ~0.3) into
    the same variable, and the mean difference alone made the null r about
    -0.2 at every distance: the whole of the cross-context "excess".
    """
    if m == "gg":
        return [(xa["g"], xb["g"])]
    if m == "hh":
        return [(xa["h"], xb["h"])]
    return [(xa["h"], xb["g"]), (xb["h"], xa["g"])]


def joined_pairs(per_ctx: dict[str, pd.DataFrame]):
    """One frame of read pairs carrying BOTH contexts' end posteriors.

    Joined on (chrom, pos1, pos2) so a row's GCH and HCG levels come from the
    same molecule pair.  Rows missing from either context are dropped -- the
    cross-context measure is undefined for them, and using a different pair set
    per measure would make the three panels incomparable.
    """
    g, h = per_ctx.get("GCH"), per_ctx.get("HCG")
    if g is None or h is None:
        return None
    j = h.merge(g, on=["chrom", "pos1", "pos2"], suffixes=("_h", "_g"))
    if j.empty:
        return None
    j = j.rename(columns={"distance_h": "distance"})[
        ["chrom", "pos1", "pos2", "distance",
         "p1_h", "p2_h", "p1_g", "p2_g"]]
    # Genome-wide coordinates for the null searches; pos1/pos2 stay raw.
    return j.assign(gpos1=genome_coords(j["chrom"], j["pos1"]),
                    gpos2=genome_coords(j["chrom"], j["pos2"]))


def end_table(df: pd.DataFrame, cols: dict[str, tuple[str, str]],
              base: int = 0, coords: str = "genome"):
    """Both ends of every pair, sorted by position.

    ``cols`` maps an output name to the (end-1, end-2) column pair, so one call
    builds an end table carrying however many per-end values the caller needs
    (here: the GCH and HCG posteriors together).  Returns
    ``(pos, {name: values}, pair_index)``; the pair index lets a null refuse to
    re-create a real ligation.  Both ends of a pair carry the SAME id, so the
    check also stops an end from being paired with itself.

    ``coords="genome"`` sorts on the genome-wide coordinate when the frame has
    one, so a null window never reaches another chromosome; ``"raw"`` sorts on
    the per-chromosome position, which is the ORIGINAL behaviour behind
    ``--null-chrom any``.  Either way the genome coordinate rides along as
    ``vals["_gpos"]`` when available, so a null pair can still be placed in a
    genomic block.

    ``base`` offsets the ids into a run-wide namespace (see :func:`_pair_base`)
    so that a pooled donor pool spanning many cells stays unambiguous.
    """
    n = len(df)
    have_g = "gpos1" in df.columns
    c1, c2 = (("gpos1", "gpos2") if coords == "genome" and have_g
              else ("pos1", "pos2"))
    pos = np.concatenate([df[c1].to_numpy(), df[c2].to_numpy()])
    ids = base + np.arange(n, dtype=np.int64)
    pi = np.concatenate([ids, ids])
    order = np.argsort(pos, kind="stable")
    vals = {name: np.concatenate([df[a].to_numpy(), df[b].to_numpy()])[order]
            for name, (a, b) in cols.items()}
    if have_g:
        vals["_gpos"] = np.concatenate([df["gpos1"].to_numpy(),
                                        df["gpos2"].to_numpy()])[order]
    return pos[order], vals, pi[order]


def _center_on(args) -> bool:
    """--cell-center: 'auto' centres only under pooled link-shuffle, the one
    null whose two ends always come from two different cells.  The same-cell
    version keeps both ends in one nucleus, as the linked pair does."""
    mode = getattr(args, "cell_center", "auto")
    if mode == "auto":
        return (getattr(args, "null_type", "") == "link-shuffle"
                and getattr(args, "null_scope", "cell") == "pooled")
    return mode == "on"


def _center_frame(f: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """Each value centred on this cell's mean over all its ends in the set.

    A linked pair's two ends share a nucleus; under a cross-cell null the
    two ends never do.  Without this, cell-to-cell differences in level
    within a stratum -- global methylation, conversion -- would raise the
    linked correlation and not the null's, and count as concordance.
    """
    f = f.copy()
    for c1, c2 in cols.values():
        mu = float(np.nanmean(np.concatenate([f[c1].to_numpy(float),
                                              f[c2].to_numpy(float)])))
        f[c1] = f[c1].to_numpy(float) - mu
        f[c2] = f[c2].to_numpy(float) - mu
    return f


def _merge_end_tables(parts: list):
    """Concatenate end tables ``(pos, vals, pair_ids[, mates])`` and sort by
    position; ``mates``, each end's read pair's other end, when every part
    has them."""
    pos = np.concatenate([p[0] for p in parts])
    names = parts[0][1].keys()
    vals = {n_: np.concatenate([p[1][n_] for p in parts]) for n_ in names}
    pi = np.concatenate([p[2] for p in parts])
    o = np.argsort(pos, kind="stable")
    out = (pos[o], {n_: v[o] for n_, v in vals.items()}, pi[o])
    if all(len(p) > 3 for p in parts):
        out += (np.concatenate([p[3] for p in parts])[o],)
    return out


def _contact_ends(f: pd.DataFrame, edges) -> pd.DataFrame:
    """The pairs whose ends can stand in for an end 2: linked pairs in range."""
    d = f["distance"].to_numpy()
    return f[(d >= edges[0]) & (d <= edges[-1])]


def _own_contact_pool(j: pd.DataFrame, cols: dict, base: int, edges):
    """Same-cell link-shuffle donors: this cell's contact ends.

    Built from the FULL frame so a pair's id is ``base + its row``, which is
    how the draw recognises -- and refuses -- the real partner.
    """
    d = j["distance"].to_numpy()
    keep = np.flatnonzero((d >= edges[0]) & (d <= edges[-1]))
    ids = base + keep.astype(np.int64)
    g1 = j["gpos1"].to_numpy(np.int64)[keep]
    g2 = j["gpos2"].to_numpy(np.int64)[keep]
    # each end with its read pair's other end: --null-exclude refuses a
    # stand-in ligated to the anchor's locus
    parts = [(g1, {n_: j[c1].to_numpy(float)[keep]
                   for n_, (c1, _c2) in cols.items()}, ids, g2),
             (g2, {n_: j[c2].to_numpy(float)[keep]
                   for n_, (_c1, c2) in cols.items()}, ids, g1)]
    return _merge_end_tables(parts)


def _draw_link_shuffle(j, d, cols, pool, base, edges, args, rng,
                       max_rounds=16, ids=None):
    """Link-shuffle null: keep end 1, take another cell's contact end as end 2.

    The donor is an end of a linked pair (in range) from ANOTHER cell of the
    anchor's stratum -- the pool is per stratum -- or, under --null-scope
    cell, from the anchor's OWN cell but not its own read pair, lying in the
    same
    --link-shuffle-bin bin as the real end 2 (bin y), on the same side of
    end 1, at a distance from end 1 inside the anchor pair's own distance
    bin.  Where the donor's own partner sits does not matter, so pairs far
    beyond the bin size keep their donors.  Same return shape as
    :func:`_draw_null`.  ``ids``: each pair's id as the pool knows it, where
    ``j`` is part of the frame the pool was built from (default base +
    row).
    """
    if pool is None:
        return None
    d_pos, d_vals, d_pi = pool[:3]
    d_mate = pool[3] if len(pool) > 3 else None
    n = len(j)
    if d_pos.size == 0 or n == 0:
        return None
    nb = len(edges) - 1
    B = int(getattr(args, "link_shuffle_bin", 250_000))
    g1 = j["gpos1"].to_numpy(np.int64)
    g2 = j["gpos2"].to_numpy(np.int64)
    in_range = (d >= edges[0]) & (d <= edges[-1])
    k = np.clip(np.digitize(d, edges) - 1, 0, nb - 1)
    e_lo, e_hi = edges[k], edges[k + 1]
    right = g2 >= g1
    # Distance band on the real end 2's side of end 1, half-open like the
    # bins themselves: [lo, hi) to the right, (lo, hi] to the left.
    lo = np.where(right, np.searchsorted(d_pos, g1 + e_lo, side="left"),
                  np.searchsorted(d_pos, g1 - e_hi, side="right"))
    hi = np.where(right, np.searchsorted(d_pos, g1 + e_hi, side="left"),
                  np.searchsorted(d_pos, g1 - e_lo, side="right"))
    # ... intersected with bin y, the bin of the real end 2, on the
    # pipeline's per-chromosome grid (chr:0, chr:250000, ...).  Binning the
    # genome-wide coordinate instead would shift every boundary by
    # (chromosome code x 2^33) mod B -- 184,592 bp on chr1.  A bin never
    # holds two chromosomes, so the null pair is always cis.
    y0 = g2 - ((g2 & (_GENOME_STRIDE - 1)) % B)
    lo = np.maximum(lo, np.searchsorted(d_pos, y0, side="left"))
    hi = np.minimum(hi, np.searchsorted(d_pos, y0 + B, side="left"))
    me = int(base // _PAIR_STRIDE)
    same_cell = getattr(args, "null_scope", "pooled") == "cell"
    # --null-exclude W: within one cell, a stand-in within W of the real end
    # 2 is likely its own molecule, and one whose read pair's other end lies
    # within W of end 1 is ligated to end 1's locus -- a near-copy of the
    # contact or another contact of the anchor -- so neither is unligated
    W = int(getattr(args, "null_exclude", 0) or 0)
    W = W if same_cell and d_mate is not None else 0

    rid = (base + np.arange(n, dtype=np.int64) if ids is None
           else np.asarray(ids, np.int64))

    def ok_cell(kk, ii):
        good = d_pi[kk] != rid[ii]
        if W:
            good &= ((np.abs(d_pos[kk] - g2[ii]) > W)
                     & (np.abs(d_mate[kk] - g1[ii]) > W))
        return good

    alive = in_range & (hi > lo)
    if W:
        # windows wholly within W of the real end 2 hold no stand-in
        a_ = np.flatnonzero(alive)
        near = ((d_pos[lo[a_]] >= g2[a_] - W)
                & (d_pos[hi[a_] - 1] <= g2[a_] + W))
        alive[a_[near]] = False
    kb_out = np.full(n, -1, dtype=np.int64)
    for _ in range(max_rounds):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        span = (hi[idx] - lo[idx]).astype(np.int64)
        kk = lo[idx] + (rng.random(idx.size) * span).astype(np.int64)
        if same_cell:
            # the cell's own ends: anything but the real partner's read pair
            good = ok_cell(kk, idx)
        else:
            # another cell (pair ids carry the cell): never a real ligation
            good = (d_pi[kk] // _PAIR_STRIDE) != me
        sel = idx[good]
        kb_out[sel] = kk[good]
        alive[sel] = False
    if W:
        # the few left after the rounds: their window, end by end
        for i in np.flatnonzero(alive):
            cand = np.arange(lo[i], hi[i])
            cand = cand[ok_cell(cand, np.full(cand.size, i))]
            if cand.size:
                kb_out[i] = rng.choice(cand)
    placed = kb_out >= 0
    if not placed.any():
        return None
    ai = np.flatnonzero(placed)
    kb = kb_out[placed]
    xa = {name: j[c1].to_numpy(float)[ai] for name, (c1, _c2) in cols.items()}
    xb = {name: d_vals[name][kb] for name in cols}
    dn = np.abs(d_pos[kb] - g1[ai]).astype(float)
    return xa, xb, dn, g1[ai], ai


def _draw(j, d, cols, pool, base, edges, args, rng, ids=None):
    """Dispatch one null draw to the null type's own function."""
    if getattr(args, "null_type", "") == "link-shuffle":
        return _draw_link_shuffle(j, d, cols, pool, base, edges, args, rng,
                                  ids=ids)
    d_pos, d_vals, d_pi = pool
    return _draw_null(j, d, cols, d_pos, d_vals, d_pi, base, edges, args,
                      rng)


def _coords(args) -> str:
    """'genome' (partners on the same chromosome) or 'raw' (any chromosome).

    link-shuffle is always 'genome': it looks donors up by the bin of the
    real end 2, which only means something on a chromosome-aware coordinate.
    """
    if getattr(args, "null_type", "") == "link-shuffle":
        return "genome"
    return "raw" if getattr(args, "null_chrom", "same") == "any" else "genome"


# -------------------------------------------------------------------- nulls --

def _null_indices_distance_matched(anchor_pos, anchor_dist, d_pos, d_pi,
                                   edges, rng, window, anchor_window,
                                   max_rounds=16, return_anchors=False):
    """Indices (ka, kb) into the donor end table for a distance-matched null.

    Returning INDICES rather than values is what lets one re-pairing serve
    every measure: gg takes (g[ka], g[kb]), hh takes (h[ka], h[kb]) and the
    cross-context measure takes (h[ka], g[kb]).  Building a separate null per
    measure would compare them against different synthetic pairs.

    Vectorised over anchors, with the retry as an OUTER loop of at most
    ``max_rounds`` fully-vectorised passes over whatever is still unplaced.
    The obvious per-anchor loop is pure Python and holds the GIL for the whole
    run, which made --n-threads do nothing at all.
    """
    n = int(anchor_pos.size)
    if d_pos.size < 2 or n == 0:
        return None
    nb = len(edges) - 1
    in_range = (anchor_dist >= edges[0]) & (anchor_dist <= edges[-1])
    bins = np.clip(np.digitize(anchor_dist, edges) - 1, 0, nb - 1)
    # Target separation = the observed pair's own distance BIN, widened to at
    # least +/-window: the analysis cannot resolve finer than a bin, and an
    # absolute window is unfindable at long range.
    tlo = np.minimum(edges[bins], anchor_dist - window)
    thi = np.maximum(edges[bins + 1], anchor_dist + window)

    if anchor_window > 0:
        alo = np.searchsorted(d_pos, anchor_pos - anchor_window, side="left")
        ahi = np.searchsorted(d_pos, anchor_pos + anchor_window, side="right")
    else:
        alo = np.zeros(n, dtype=np.int64)
        ahi = np.full(n, d_pos.size, dtype=np.int64)

    alive = in_range & ((ahi - alo) >= 2)
    ka_out = np.full(n, -1, dtype=np.int64)
    kb_out = np.full(n, -1, dtype=np.int64)

    for _ in range(max_rounds):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        span = (ahi[idx] - alo[idx]).astype(np.int64)
        ka = alo[idx] + (rng.random(idx.size) * span).astype(np.int64)
        up = rng.random(idx.size) < 0.5
        pa = d_pos[ka]
        lo = np.where(up, pa + tlo[idx], pa - thi[idx])
        hi = np.where(up, pa + thi[idx], pa - tlo[idx])
        blo = np.searchsorted(d_pos, lo, side="left")
        bhi = np.searchsorted(d_pos, hi, side="right")
        has = bhi > blo
        width = np.maximum(bhi - blo, 1).astype(np.int64)
        kb = blo + (rng.random(idx.size) * width).astype(np.int64)
        kb = np.clip(kb, 0, d_pos.size - 1)
        # Two ends of ONE read pair are a real ligation, not a null.  With a
        # pooled pool this can happen for any cell's pair, not just the
        # anchor's, which is why the ids are run-wide.
        good = has & (kb != ka) & (d_pi[ka] != d_pi[kb])
        sel = idx[good]
        ka_out[sel] = ka[good]
        kb_out[sel] = kb[good]
        alive[sel] = False

    placed = ka_out >= 0
    if not placed.any():
        return None
    ka_f, kb_f = ka_out[placed], kb_out[placed]
    dist = np.abs(d_pos[kb_f].astype(np.int64) - d_pos[ka_f].astype(np.int64))
    if return_anchors:
        # Which REAL pairs got a null partner -- the matched arm is those
        # pairs, so linked and null are compared on the same loci set.
        return ka_f, kb_f, dist.astype(float), np.flatnonzero(placed)
    return ka_f, kb_f, dist.astype(float)


def _null_indices_locus_matched(anchor_pos2, anchor_pos1, anchor_idx, d_pos,
                                d_pi, rng, window, max_rounds=8):
    """Indices for the locus-matched null: keep end 1, swap end 2 for another
    end within +/-window of the true partner.

    The first return value indexes the ANCHOR's own end 1 (not the donor
    table), so the caller pairs anchor-side values with donor-side ones.
    Vectorised the same way as above.

    ``anchor_idx`` carries the anchor pair's run-wide id, so rejecting donors
    that share it removes both the true partner and the anchor's own end 1 —
    the latter sits inside the window whenever the pair is closer than
    ``window``, and would otherwise be correlated with itself.
    """
    n = int(anchor_pos2.size)
    if d_pos.size < 2 or n == 0:
        return None
    lo = np.searchsorted(d_pos, anchor_pos2 - window, side="left")
    hi = np.searchsorted(d_pos, anchor_pos2 + window, side="right")
    alive = (hi - lo) >= 2
    kb_out = np.full(n, -1, dtype=np.int64)

    for _ in range(max_rounds):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        span = (hi[idx] - lo[idx]).astype(np.int64)
        k = lo[idx] + (rng.random(idx.size) * span).astype(np.int64)
        k = np.clip(k, 0, d_pos.size - 1)
        good = d_pi[k] != anchor_idx[idx]
        sel = idx[good]
        kb_out[sel] = k[good]
        alive[sel] = False

    placed = kb_out >= 0
    if not placed.any():
        return None
    ai = np.flatnonzero(placed)
    kb_f = kb_out[placed]
    dist = np.abs(d_pos[kb_f].astype(np.int64)
                  - anchor_pos1[ai].astype(np.int64))
    return ai, kb_f, dist.astype(float)


# ----------------------------------------------------------------- plotting --

#: How the observed curve is labelled, by --delta-arm.
_OBS_LABEL = {"matched": "linked (pairs with a null partner)",
              "linked": "linked (every true pair)"}


def _log_x(ax, lo: float, hi: float) -> None:
    """A log distance axis over [lo, hi] whatever is drawn on it: a measure
    with no pairs placed in any bin -- an inactive X's sparse HCG pairs
    under the within-cell null -- would otherwise leave matplotlib no
    positive value to scale, and the figure would stop the run."""
    ax.set_xlim(lo, hi)
    ax.set_xscale("log")


def _draw_measure(ax, axd, key, title, curves, centers, args, boot,
                  fisher=None, first=True):
    """One measure's column: observed and null on ``ax``, the contrast below.

    Shared by the full figure and the main-figure layout, so the two draw
    every curve, band and FDR ring identically.
    """
    obs = getattr(args, "delta_arm", "linked")
    # Both lines are straight segments between bin centres, no
    # smoothing; the null gets hollow markers so each bin's value can be
    # read off it too -- without them the gap to the observed curve could
    # not be checked against row 2 point by point.
    for cond, clab, style in (
            (obs, _OBS_LABEL.get(obs, obs),
             dict(marker="o", ms=3.0, ls="-", alpha=1.0, lw=1.6)),
            ("null", "non-linked",
             dict(marker="o", ms=3.0, mfc="white", ls="--", alpha=0.8,
                  lw=1.4))):
        if (key, cond) not in curves:
            continue
        r, n = curves[(key, cond)]
        ax.plot(centers, r, label=clab, color=_COLOR[key], **style)
    ax.axhline(0, color="#9AA0A6", lw=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Pearson r of end posteriors" if first else "")
    _log_x(ax, centers[0] / 1.3, centers[-1] * 1.3)
    if not any(np.isfinite(curves[(key, c)][0]).any() for c in (obs, "null")
               if (key, c) in curves):
        ax.text(0.5, 0.5, f"too few pairs (--min-pairs {args.min_pairs})",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="0.4")
    ax.legend(fontsize=8, frameon=False)

    if fisher is not None:
        delta, lo, hi, pv, q, _ = fisher.get(
            key, (np.full(len(centers), np.nan),) * 5 + (None,))
        by_cell = args.fisher_z == "cell"
        band = ("95% CI (t across cells)" if by_cell
                else "95% CI (Fisher z)")
        line = ("Δz = mean$_{cells}$(z$_{linked}$ − z$_{null}$)" if by_cell
                else "Δz = z(r$_{linked}$) − z(r$_{null}$)")
        ylab = "Δz (Fisher)"
    else:
        arm = getattr(args, "delta_arm", "linked")
        delta = curves[(key, arm)][0] - curves[(key, "null")][0]
        lo, hi, pv, q = boot.get(key, (None, None, None, None))
        band, line = "95% CI (cell bootstrap)", f"Δ = {arm} − null"
        ylab = f"Δ = {arm} − null"
    if lo is not None:
        axd.fill_between(centers, lo, hi, color=_COLOR[key], alpha=0.18,
                         lw=0, label=band)
    axd.plot(centers, delta, color=_COLOR[key], lw=1.6, marker="o", ms=3,
             label=line)
    if q is not None:
        # Ring the bins that survive FDR, and only those: one test per bin
        # per measure means several would clear raw p<0.05 by chance.
        sig = np.isfinite(q) & (q < 0.05) & np.isfinite(delta)
        if sig.any():
            axd.plot(centers[sig], delta[sig], "o", ms=8,
                     mfc="none", mec=_COLOR[key], mew=1.5,
                     label=f"q < 0.05  ({int(sig.sum())} bins)")
        for xi, di, qi in zip(centers, delta, q):
            mark = stars(qi)
            if mark and np.isfinite(di):
                axd.annotate(mark, (xi, di), textcoords="offset points",
                             xytext=(0, 7), ha="center", fontsize=8,
                             color=_COLOR[key])
    axd.axhline(0, color="#9AA0A6", lw=0.8)
    _log_x(axd, centers[0] / 1.3, centers[-1] * 1.3)
    axd.set_xlabel("genomic distance between the two ends (bp)")
    axd.set_ylabel(ylab if first else "")
    axd.legend(fontsize=8, frameon=False)


def _figure(curves, centers, label, out_path, args, note="", boot=None,
            fisher=None):
    """Row 1 the observed and null curves, row 2 the contrast between them.

    The observed curve is the --delta-arm arm, so row 1 shows the very pairs
    row 2's Δ is taken on: under 'matched' (the default) only the linked
    pairs the null placed a partner for.  Two lines, not three -- linked and
    matched sit on top of each other wherever most pairs are placed, and the
    third line only made the figure harder to read.  Both are in the TSV.

    With ``fisher`` the second row switches to Δz and its parametric interval;
    the first row is untouched, so the two files can be read side by side and
    differ only in which test drew the marks.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    boot = boot or {}

    fig, axes = plt.subplots(2, len(_MEASURES),
                             figsize=(5.2 * len(_MEASURES), 8), sharex=True)
    for j, (key, title) in enumerate(_MEASURES):
        _draw_measure(axes[0, j], axes[1, j], key, title, curves, centers,
                      args, boot, fisher, first=(j == 0))
    test = ("" if fisher is None else
            ("   [Fisher z, paired within cell, t across cells]"
             if args.fisher_z == "cell" else
             "   [Fisher z, two-sample, pooled pairs — anti-conservative]"))
    fig.suptitle(f"{label}: end-posterior correlation vs distance"
                 + (f"   ({note})" if note else "") + test, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def _figure_by_group(per_group, out_path, args, colors=None):
    """One panel per measure, every cell-type group overlaid on a shared scale.

    The per-group figures cannot be compared by eye because each is drawn to
    its own y-range; this is the one that shows whether the groups actually
    differ.  Δ is plotted rather than the raw curves, since that is the
    quantity the per-group panels are read for.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    labels = [g for g in per_group if g != "all"]
    if len(labels) < 2:
        return False
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(labels), 1)))
    colors = colors or {g: cmap[i] for i, g in enumerate(labels)}

    fig, axes = plt.subplots(2, len(_MEASURES),
                             figsize=(5.2 * len(_MEASURES), 8), sharex=True)
    arm = getattr(args, "delta_arm", "linked")
    for j, (key, title) in enumerate(_MEASURES):
        for row, what in enumerate(("observed", "delta")):
            ax = axes[row, j]
            for g in labels:
                curves, centers = per_group[g]
                if what == "observed":
                    # the --delta-arm arm, as in the per-group figures
                    y = curves[(key, arm)][0]
                else:
                    y = (curves[(key, arm)][0]
                         - curves[(key, "null")][0])
                ax.plot(centers, y, lw=1.4, color=colors[g], label=g,
                        marker="o", ms=2.5, alpha=0.9)
            ax.axhline(0, color="#9AA0A6", lw=0.8)
            _log_x(ax, centers[0] / 1.3, centers[-1] * 1.3)
            if row == 0:
                ax.set_title(title, fontsize=11)
                ax.set_ylabel(("linked r (pairs with a null partner)"
                               if arm == "matched" else "linked r")
                              if j == 0 else "")
            else:
                ax.set_xlabel("genomic distance between the two ends (bp)")
                ax.set_ylabel(f"Δ = {getattr(args, 'delta_arm', 'linked')}"
                              f" − null" if j == 0 else "")
        # Shared y-scale per row so the groups are visually comparable.
    for row in range(2):
        ylo = min(ax.get_ylim()[0] for ax in axes[row])
        yhi = max(ax.get_ylim()[1] for ax in axes[row])
        for ax in axes[row]:
            ax.set_ylim(ylo, yhi)
    axes[0, -1].legend(fontsize=7, frameon=False, ncol=2,
                       loc="upper right")
    fig.suptitle(f"end-posterior correlation by cell-type group "
                 f"({len(labels)} groups, shared y-scale)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


# --------------------------------------------------------------------- main --

# Read-only context handed to a worker process ONCE, at start-up.  Passing it
# per task instead would pickle the exclusion intervals and the donor pools
# 1577 times over, which is what made the process backend lose to threads.
_W: dict = {}


#: Cell count at or above which --backend auto switches to processes.  Worker
#: start-up is a flat cost; below this it outweighs the GIL it buys out of.
_PROCESS_FROM_CELLS = 96


def _pin_inner_threads() -> None:
    """Stop pyarrow opening a full thread pool inside every worker.

    pyarrow defaults BOTH its cpu and io pools to the machine's core count, so
    ``--n-threads 10`` really asked for 10 workers x 16 inner threads on a
    10-core box.  The oversubscription is why utilisation looked high for the
    first few cells and then collapsed.  Parallelism here is per CELL, so one
    inner thread per worker is what we want.
    """
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def _init_worker(paths, edges, args, pools, batch_of, merged, excl_chrom):
    _pin_inner_threads()
    _W.update(paths=paths, edges=edges, args=args, pools=pools,
              batch_of=batch_of, merged=merged, excl_chrom=excl_chrom)


def _worker(job):
    """Only (cell, index[, want_percell]) crosses the process boundary."""
    cell, i = job[0], job[1]
    want = bool(job[2]) if len(job) > 2 else False
    return _accumulate_cell(cell, i, _W["paths"], _W["edges"], _W["args"],
                            _W["pools"], _W["batch_of"], _W["merged"],
                            _W["excl_chrom"], want)


def _drain(results, accs, per_cell, total, label):
    """Fold each cell's accumulators in AS IT ARRIVES, reporting progress.

    ``list(executor.map(...))`` would hold every cell's accumulators before
    folding any of them, and would show nothing until the last one landed.
    These runs are thousands of cells long, so both matter: the fold keeps
    only the running totals plus what the bootstrap genuinely needs, and the
    count gives an ETA instead of a silent hour.
    """
    n_pairs = n_null = 0
    bar = Progress(total, f"{label}: cells")
    try:
        for a, np_, nn_ in results:
            for k in accs:
                accs[k] += a[k]
            per_cell.append(a)
            n_pairs += np_
            n_null += nn_
            bar.update()
    finally:
        bar.close()
    return n_pairs, n_null


def _percell_set_of(args):
    """Pair sets the per-cell test runs on.  Per-mark by default whatever
    --pair-set says: the per-cell test needs every usable pair a cell has, and
    unlike panel a it has no earlier figure to reproduce."""
    if getattr(args, "per_cell_pair_set", "per-mark") == "same":
        return _SET_OF[getattr(args, "pair_set", "joint")]
    return _SET_OF["per-mark"]


def _accumulate_cell(cell, seed_i, paths, edges, args, pools, batch_of,
                     merged, excl_chrom, want_percell=False):
    """One cell's contribution, as its own accumulators.

    Returned rather than added in place so the cells can run on a thread pool
    and be merged afterwards; the accumulators are additive, so summing them is
    exact.  The RNG is seeded per cell rather than shared: a shared generator
    would both race between threads and make the run depend on the order they
    happen to finish in.

    One null draw per PAIR SET, shared by every measure computed on that set.
    Under --pair-set joint that is a single draw feeding all three measures,
    exactly as before, consumed from the generator in the same order -- so with
    --null-chrom any the panel a numbers reproduce the original run exactly.

    Three arms per measure: LINKED (every true pair), MATCHED (the true pairs
    the null managed to place a partner for) and NULL.

    The per-cell test may run on DIFFERENT pair sets from panel a (per-mark by
    default).  When it does, it gets its own pass with its own generators, so
    panel a is untouched by it.
    """
    rng = np.random.default_rng(args.seed + seed_i)
    nb = len(edges) - 1
    accs = {(m, c): new_acc(nb) for m, _ in _MEASURES for c, _ in _CONDS}
    n_pairs = n_null = 0

    set_of = _SET_OF[getattr(args, "pair_set", "joint")]
    a_sets = sorted(set(set_of.values()))
    pc_set_of = _percell_set_of(args) if want_percell else None
    separate = want_percell and pc_set_of != set_of
    pc_sets = sorted(set(pc_set_of.values())) if separate else []
    per_ctx = _load_any(paths, cell, _min_sites(args), merged, excl_chrom,
                        args)
    frames = set_frames(per_ctx, sorted(set(a_sets) | set(pc_sets)))
    if _center_on(args):
        frames = {sk: _center_frame(f, _SET_COLS[sk])
                  for sk, f in frames.items()}
    blocks = {} if want_percell else None
    if not frames:
        if want_percell:
            accs["_percell"] = blocks
        return accs, 0, 0

    base = _pair_base(seed_i)
    bkey = batch_of.get(cell, "") if batch_of else ""
    focus = _unit_focus(cell, args)

    def donors(sk, j, cols):
        if args.null_scope == "pooled":
            return (pools.get(sk) or {}).get(bkey)
        if args.null_type == "link-shuffle":
            return _own_contact_pool(j, cols, base, edges)
        return end_table(j, cols, base, _coords(args))

    # ---- panel a: one pass per panel-a pair set, generator order unchanged.
    present = [sk for sk in a_sets if sk in frames]
    for ix, sk in enumerate(present):
        j = frames[sk]
        cols = _SET_COLS[sk]
        measures = [m for m, _ in _MEASURES if set_of[m] == sk]
        ids = None
        if focus is not None:
            # the stand-ins from all of the unit's ends; only the focused
            # contacts count -- an end in the element, that end kept, or
            # crossing a region or on one side of it
            pool = donors(sk, j, cols)
            j, ids = focus(j, cols, base)
            if not len(j):
                continue
        d = j["distance"].to_numpy()
        keep = (d >= edges[0]) & (d <= edges[-1])
        bins = np.clip(np.digitize(d, edges) - 1, 0, nb - 1)
        for m in measures:
            for x, y in _linked_xy(m, j, cols):
                accumulate(accs[(m, "linked")], bins[keep], x[keep], y[keep])
        n_pairs += int(keep.sum())

        if focus is None:
            pool = donors(sk, j, cols)
        if pool is None:
            if want_percell and not separate:
                blocks.update(_percell_blocks(j, d, cols, measures, None,
                                              None, edges, args))
            continue
        drawn = _draw(j, d, cols, pool, base, edges, args, rng, ids=ids)
        if drawn is not None:
            xa, xb, dn, _anchor, aidx = drawn
            nkeep = (dn >= edges[0]) & (dn <= edges[-1])
            bn = np.clip(np.digitize(dn, edges) - 1, 0, nb - 1)
            for m in measures:
                for x, y in _null_xy(m, xa, xb):
                    accumulate(accs[(m, "null")], bn[nkeep], x[nkeep],
                               y[nkeep])
            # MATCHED: the true pairs behind the null pairs that were kept,
            # binned by their own distance.
            mi = aidx[nkeep]
            mi = mi[keep[mi]]
            for m in measures:
                for x, y in _linked_xy(m, j, cols, mi):
                    accumulate(accs[(m, "matched")], bins[mi], x, y)
            n_null += int(nkeep.sum())

        if want_percell and not separate:
            # A SECOND, independent null draw on its own generator, so the
            # arm above -- and with it every panel a number -- is identical
            # with or without --per-cell.
            rng2 = np.random.default_rng([args.seed, seed_i, 2, ix])
            drawn2 = _draw(j, d, cols, pool, base, edges, args, rng2)
            blocks.update(_percell_blocks(j, d, cols, measures, drawn,
                                          drawn2, edges, args))

    # ---- per-cell on its own pair sets: separate generators throughout.
    for ix, sk in enumerate(s_ for s_ in pc_sets if s_ in frames):
        j = frames[sk]
        cols = _SET_COLS[sk]
        measures = [m for m, _ in _MEASURES if pc_set_of[m] == sk]
        d = j["distance"].to_numpy()
        pool = donors(sk, j, cols)
        if pool is None:
            blocks.update(_percell_blocks(j, d, cols, measures, None, None,
                                          edges, args))
            continue
        r1 = np.random.default_rng([args.seed, seed_i, 3, ix])
        r2 = np.random.default_rng([args.seed, seed_i, 4, ix])
        drawn = _draw(j, d, cols, pool, base, edges, args, r1)
        drawn2 = _draw(j, d, cols, pool, base, edges, args, r2)
        blocks.update(_percell_blocks(j, d, cols, measures, drawn, drawn2,
                                      edges, args))

    if want_percell:
        accs["_percell"] = blocks
    return accs, n_pairs, n_null


def _draw_null(j, d, cols, d_pos, d_vals, d_pi, base, edges, args, rng):
    """One null re-pairing of one pair set.

    Returns ``(xa, xb, distance, anchor_genome_pos, anchor_index)`` or None.
    ``anchor_index`` names the REAL pairs that got a partner, which is what
    the matched arm is built from.  One helper for both draws, so the
    per-cell control is built exactly the way panel a's null is.
    """
    raw = _coords(args) == "raw"
    p1 = j["pos1" if raw else "gpos1"].to_numpy()
    p2 = j["pos2" if raw else "gpos2"].to_numpy()
    if args.null_type == "distance-matched":
        got = _null_indices_distance_matched(
            p1, d, d_pos, d_pi, edges, rng,
            args.null_window, args.null_anchor_window, return_anchors=True)
        if got is None:
            return None
        ka, kb, dn, aidx = got
        xa = {k: v[ka] for k, v in d_vals.items()}
        xb = {k: v[kb] for k, v in d_vals.items()}
        anchor = d_vals["_gpos"][ka] if "_gpos" in d_vals else d_pos[ka]
        return xa, xb, dn, anchor, aidx
    got = _null_indices_locus_matched(
        p2, p1, base + np.arange(len(j), dtype=np.int64),
        d_pos, d_pi, rng, args.null_window)
    if got is None:
        return None
    ai, kb, dn = got
    # end 1 stays with the anchor; end 2 comes from the donor pool
    xa = {name: j[c1].to_numpy()[ai] for name, (c1, _c2) in cols.items()}
    xb = {k: v[kb] for k, v in d_vals.items()}
    return xa, xb, dn, j["gpos1"].to_numpy()[ai], ai


def _percell_blocks(j, d, cols, measures, drawn, drawn2, edges, args):
    """Per-genomic-block raw sums for each arm of each measure on one set.

    Arms: L (linked), M (matched), N (null), N2 (second null, the control).
    Blocks are --per-cell-block wide on the genome-wide coordinate, so a block
    never spans two chromosomes; the bootstrap resamples them, which is what
    carries the spatial dependence between pairs into the per-cell SE.
    """
    lo = max(float(args.per_cell_min_distance), float(edges[0]))
    hi = float(edges[-1])
    B = int(args.per_cell_block)
    g1 = j["gpos1"].to_numpy()
    empty = (np.empty(0, np.int64), np.empty((0, 6)))

    def from_xy(blk, pairs):
        if not pairs:
            return empty
        return pce.block_sums(np.concatenate([blk] * len(pairs)),
                              np.concatenate([p[0] for p in pairs]),
                              np.concatenate([p[1] for p in pairs]))

    lk = np.flatnonzero((d >= lo) & (d <= hi))
    # Per-REGION sums ride along (--per-region): the same arms keyed by
    # region x distance band on the per-chromosome grid, summed over cells
    # later by write_regions.
    regions = getattr(args, "per_region", "off") == "on"
    if regions:
        bands = _region_bands(args, lo, hi)

        def rk(gpos, dist):
            return _region_key(gpos, dist, int(args.region_size), bands)
    out = {}
    for m in measures:
        arms = {"L": from_xy(g1[lk] // B, _linked_xy(m, j, cols, lk))}
        reg = ({"L": from_xy(rk(g1[lk], d[lk]), _linked_xy(m, j, cols, lk))}
               if regions else None)
        for tag, dr in (("N", drawn), ("N2", drawn2)):
            if dr is None:
                arms[tag] = empty if tag == "N" else None
                if tag == "N":
                    arms["M"] = empty
                continue
            xa, xb, dn, anchor, aidx = dr
            k = (dn >= lo) & (dn <= hi)
            nxy = [(x[k], y[k]) for x, y in _null_xy(m, xa, xb)]
            arms[tag] = from_xy(anchor[k] // B, nxy)
            if regions:
                reg[tag] = from_xy(rk(anchor[k], dn[k]), nxy)
            if tag == "N":
                mi = aidx[k]
                mi = mi[(d[mi] >= lo) & (d[mi] <= hi)]
                arms["M"] = from_xy(g1[mi] // B, _linked_xy(m, j, cols, mi))
                if regions:
                    reg["M"] = from_xy(rk(g1[mi], d[mi]),
                                       _linked_xy(m, j, cols, mi))
        if regions:
            arms["R"] = reg
        out[m] = arms
    return out


def _region_bands(args, lo: float, hi: float) -> np.ndarray:
    """Distance-band edges for --per-region: --region-bands, or one band."""
    raw = getattr(args, "region_bands", None)
    if not raw:
        return np.array([lo, hi], dtype=float)
    e = sorted({float(x) for x in str(raw).split(",") if x.strip()}
               | {lo, hi})
    return np.array([x for x in e if lo <= x <= hi], dtype=float)


def _region_key(gpos, dist, size: int, bands: np.ndarray) -> np.ndarray:
    """One int64 per pair: region start (per-chromosome grid) x band."""
    g = np.asarray(gpos, np.int64)
    start = g - ((g & (_GENOME_STRIDE - 1)) % int(size))
    nb = len(bands) - 1
    band = np.clip(np.searchsorted(bands, np.asarray(dist, float),
                                   side="right") - 1, 0, nb - 1)
    return start * nb + band


def _chrom_name(code: int) -> str:
    """Inverse of :func:`_chrom_code` for the names it maps directly."""
    code = int(code)
    if code < 1000:
        return f"chr{code}"
    return {1001: "chrX", 1002: "chrY", 1003: "chrM"}.get(code,
                                                         f"contig{code}")


def _pool_sets(args):
    """Pair sets a donor pool must cover: panel a's, plus the per-cell test's."""
    sets = set(_SET_OF[getattr(args, "pair_set", "joint")].values())
    if getattr(args, "per_cell", "off") == "on":
        sets |= set(_percell_set_of(args).values())
    return sorted(sets)


def _build_pools(cells, paths, args, batch_of, merged, excl_chrom):
    """Donor end pools for --null-scope pooled: ``{pair_set: {batch: pool}}``.

    Not used by link-shuffle, which builds one small pool per stratum inside
    the stratum's own task (:func:`_stratum_job`).

    One pool per pair set, because each set's null draws from ends carrying
    that set's values (GCH-only, HCG-only, or both).  Parallel over cells,
    because this is a SECOND full pass of the same load-and-join that
    dominates the run: left serial it silently doubled the wall clock of every
    --null-scope pooled run while nine of ten cores idled.  Threads rather
    than processes -- the per-cell end tables are large and would cost more to
    pickle back than the join costs to compute, and with ``chrom`` categorical
    the join no longer pins the GIL.
    """
    sets = _pool_sets(args)
    ms = _min_sites(args)

    def _one(job):
        ci, cell = job
        frames = set_frames(_load_any(paths, cell, ms, merged, excl_chrom,
                                      args), sets)
        if not frames:
            return None
        key = batch_of.get(cell, "") if batch_of else ""
        # Same enumeration as run_group's job list, so a cell's ids here and
        # in its own pass agree without either side recording an offset.
        return key, {sk: end_table(f, _SET_COLS[sk], _pair_base(ci),
                                   _coords(args))
                     for sk, f in frames.items()}

    jobs = list(enumerate(cells))
    parts: dict = {}

    def _take(got):
        if got is None:
            return
        key, tabs = got
        for sk, tab in tabs.items():
            parts.setdefault(sk, {}).setdefault(key, []).append(tab)

    bar = Progress(len(jobs), "pooled donors: cells")
    try:
        # getattr: callers that build a minimal args for this one helper
        # never had to supply a thread count, and should not start to.
        if getattr(args, "n_threads", 1) > 1 and len(jobs) > 1:
            from concurrent.futures import ThreadPoolExecutor
            _pin_inner_threads()
            with ThreadPoolExecutor(max_workers=args.n_threads) as ex:
                for got in ex.map(_one, jobs):
                    _take(got)
                    bar.update()
        else:
            for job in jobs:
                _take(_one(job))
                bar.update()
    finally:
        bar.close()
    pools: dict = {}
    for sk, by_key in parts.items():
        # Run-wide ids, so a donor pair drawn out of ANY cell in the stratum
        # is recognisable as a real ligation and refused.
        pools[sk] = {k: _merge_end_tables(ps) for k, ps in by_key.items()}
    logger.info("  pooled donor ends: " + ", ".join(
        f"{sk}: {sum(int(v[0].size) for v in by.values()):,}"
        for sk, by in pools.items()))
    return pools


def _link_strata(samples: pd.DataFrame, args, batch_of):
    """Stratum per cell for link-shuffle: cell type x --null-match-column.

    The cell type is added from --group-column so that even the all-cells
    run never hands a cell another type's link.
    """
    idx = samples.set_index(args.cell_column)
    parts = []
    if args.group_column:
        parts.append(idx[args.group_column].astype(str))
    else:
        logger.warning("--null-type link-shuffle without --group-column: "
                       "donor links are not restricted to the anchor's cell "
                       "type.")
    if batch_of:
        parts.append(pd.Series(batch_of))
    if not parts:
        return None
    strata = pd.concat(parts, axis=1).astype(str).agg("|".join, axis=1)
    logger.info(f"link-shuffle strata: {strata.nunique()} "
                f"(median {int(strata.value_counts().median())} cells each)")
    return strata.to_dict()


def _stratum_pool(members, paths, edges, args, merged, excl_chrom):
    """link-shuffle donors for ONE stratum: its cells' contact ends.

    Both ends of every linked pair in the distance range, centred on their
    own cell's mean, values as float32 and without the genome-coordinate
    copy (the position already is one) -- roughly a third of a full end
    table, and only ever one stratum's worth at a time per worker.
    """
    ms = _min_sites(args)
    sets = _pool_sets(args)
    parts: dict = {}
    for cell, i in members:
        frames = set_frames(_load_any(paths, cell, ms, merged, excl_chrom,
                                      args), sets)
        for sk, f in frames.items():
            if _center_on(args):
                f = _center_frame(f, _SET_COLS[sk])
            f = _contact_ends(f, edges)
            if not len(f):
                continue
            pos, vals, pi = end_table(f, _SET_COLS[sk], _pair_base(i),
                                      "genome")
            vals = {n_: v.astype(np.float32) for n_, v in vals.items()
                    if n_ != "_gpos"}
            parts.setdefault(sk, []).append((pos, vals, pi))
    return {sk: _merge_end_tables(ps) for sk, ps in parts.items()}


def _stratum_job(task, paths, edges, args, batch_of, merged, excl_chrom):
    """One stratum: build its pool, run its cells, let the pool go.

    Returns ``[(cell_index, accumulate_result), ...]``.  The cell index is
    the cell's position in the group, as everywhere else, so seeds and pair
    ids are what a cell-by-cell run would have used.
    """
    key, members, want = task
    pool = _stratum_pool(members, paths, edges, args, merged, excl_chrom)
    pools = {sk: {key: p} for sk, p in pool.items()}
    return [(i, _accumulate_cell(cell, i, paths, edges, args, pools,
                                 batch_of, merged, excl_chrom, want))
            for cell, i in members]


def _stratum_worker(task):
    """Process-backend entry: only (key, members, want) crosses over."""
    return _stratum_job(task, _W["paths"], _W["edges"], _W["args"],
                        _W["batch_of"], _W["merged"], _W["excl_chrom"])


def _run_strata(cells, paths, edges, args, batch_of, merged, excl_chrom,
                executor, want, accs, per_cell, label):
    """link-shuffle, one stratum per task.

    A run-wide donor pool -- every contact end of every cell, for every pair
    set -- was the whole run's pairs held at once, briefly twice while it
    was merged, and handed to each worker process: on 1,577 cells the job
    was killed for memory the moment the pool was built.  A stratum (cell
    type x batch, median 7 cells) is its own closed world for this null, so
    it can be built, used and dropped on its own.  Results go back into
    cell order, because per_cell[i] has to belong to cells[i].
    """
    by: dict = {}
    for i, c in enumerate(cells):
        by.setdefault(batch_of.get(c, "") if batch_of else "", []).append(
            (c, i))
    # Largest strata first, so a big one cannot be left running alone at
    # the end while the other workers idle.
    tasks = sorted(((k, m, want) for k, m in by.items()),
                   key=lambda t: -len(t[1]))
    sizes = [len(t[1]) for t in tasks]
    logger.info(f"  link-shuffle: {len(tasks)} strata, "
                f"{min(sizes)}-{max(sizes)} cells each")
    if executor is not None and len(tasks) > 1:
        if args.backend == "process":
            it = executor.map(_stratum_worker, tasks)
        else:
            it = executor.map(lambda t: _stratum_job(
                t, paths, edges, args, batch_of, merged, excl_chrom), tasks)
    else:
        it = (_stratum_job(t, paths, edges, args, batch_of, merged,
                           excl_chrom) for t in tasks)
    slots: list = [None] * len(cells)
    bar = Progress(len(cells), f"{label}: cells")
    try:
        for res in it:
            for i, r in res:
                slots[i] = r
            bar.update(len(res))
    finally:
        bar.close()
    n_pairs = n_null = 0
    for a, np_, nn_ in slots:
        for k in accs:
            accs[k] += a[k]
        per_cell.append(a)
        n_pairs += np_
        n_null += nn_
    return n_pairs, n_null


def run_group(cells, paths, edges, args, label, out_dir, batch_of=None,
              merged=None, excl_chrom=(), executor=None, percell=None,
              keep=None):
    nb = len(edges) - 1
    accs = {(m, c): new_acc(nb) for m, _ in _MEASURES for c, _ in _CONDS}
    links = (args.null_type == "link-shuffle"
             and args.null_scope == "pooled")
    pools = ({} if args.null_scope != "pooled" or links
             else _build_pools(cells, paths, args, batch_of, merged,
                               excl_chrom))

    n_pairs = n_null = 0
    per_cell: list = []   # kept for the cell-level bootstrap
    want = percell is not None
    jobs = [(c, i, want) for i, c in enumerate(cells)]
    if links:
        n_pairs, n_null = _run_strata(cells, paths, edges, args, batch_of,
                                      merged, excl_chrom, executor, want,
                                      accs, per_cell, label)
    elif executor is not None and len(cells) > 1:
        # The pool is created once for the whole run and reused across groups.
        # Only ProcessPoolExecutor.map honours chunksize; threads ignore it.
        # Keep chunks small so one straggling worker cannot be left holding a
        # long tail of cells while the rest idle -- cells differ several-fold
        # in depth, which is exactly when a coarse chunking strands work.
        chunk = max(1, min(8, len(jobs) // (args.n_threads * 16)))
        if args.backend == "process" and pools:
            # Pooled donors are per GROUP, so they cannot ride on a pool built
            # once for the whole run: a shared initializer would hand every
            # group the same (or an empty) donor set and silently produce the
            # wrong null.  Threads close over `pools` directly and are fine.
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(
                    max_workers=args.n_threads, initializer=_init_worker,
                    initargs=(paths, edges, args, pools, batch_of, merged,
                              excl_chrom)) as gex:
                out = _drain(gex.map(_worker, jobs, chunksize=chunk),
                             accs, per_cell, len(jobs), label)
        elif args.backend == "process":
            out = _drain(executor.map(_worker, jobs, chunksize=chunk),
                         accs, per_cell, len(jobs), label)
        else:
            out = _drain(executor.map(
                lambda ci: _accumulate_cell(ci[0], ci[1], paths, edges, args,
                                            pools, batch_of, merged,
                                            excl_chrom, ci[2]), jobs),
                         accs, per_cell, len(jobs), label)
        n_pairs, n_null = out
    else:
        for c, i, w in jobs:
            a, np_, nn_ = _accumulate_cell(c, i, paths, edges, args, pools,
                                           batch_of, merged, excl_chrom, w)
            for k in accs:
                accs[k] += a[k]
            per_cell.append(a)
            n_pairs += np_
            n_null += nn_
            if (i + 1) % 200 == 0:
                logger.info(f"  {label}: {i + 1}/{len(cells)} cells")

    curves = {k: corr_from_acc(a, args.min_pairs) for k, a in accs.items()}
    if keep is not None:            # --haplotypes: the units' comparisons
        keep[label] = per_cell

    # Cell-level bootstrap of Δ = linked − null, then FDR across the whole
    # figure.  Resampling CELLS rather than end pairs is the honest unit: a
    # cell contributes hundreds of thousands of massively non-independent
    # pairs, so a test treating those as independent would star everything.
    # Δ is taken against MATCHED by default: the true pairs the null could
    # place, so both arms are scored on the same loci.  "linked" keeps the old
    # all-pairs contrast for comparison.
    arm = getattr(args, "delta_arm", "linked")
    boot = {}
    if args.n_bootstrap > 0 and len(per_cell) >= 3:
        raw = {}
        for m, _ in _MEASURES:
            got = bootstrap_delta([a[(m, arm)] for a in per_cell],
                                  [a[(m, "null")] for a in per_cell],
                                  args.min_pairs, args.n_bootstrap, args.seed)
            if got is not None:
                raw[m] = got
        if raw:
            # One test per bin per measure, so adjust over the figure as a
            # whole rather than per panel.
            flat = np.concatenate([raw[m][2] for m in raw])
            q_all = fdr(flat, args.fdr)
            off = 0
            for m in raw:
                lo, hi, pv = raw[m]
                boot[m] = (lo, hi, pv, q_all[off:off + pv.size])
                off += pv.size

    # Companion parametric test on the same contrast, written to its own
    # figure.  It shares the bootstrap's unit of inference (the cell) under
    # --fisher-z cell and deliberately abandons it under 'pooled', so the two
    # files together show how much of the significance is real replication.
    fz = {}
    if args.fisher_z != "none" and (args.fisher_z == "pooled"
                                    or len(per_cell) >= 3):
        raw = {}
        for m, _ in _MEASURES:
            if args.fisher_z == "cell":
                raw[m] = fisher_z_cell([a[(m, arm)] for a in per_cell],
                                       [a[(m, "null")] for a in per_cell],
                                       args.fisher_min_cell_pairs)
            else:
                raw[m] = fisher_z_pooled(accs[(m, arm)],
                                         accs[(m, "null")], args.min_pairs)
        flat = np.concatenate([raw[m][3] for m in raw])
        q_all = fdr(flat, args.fdr)
        off = 0
        for m in raw:
            dz, lo, hi, pv, k = raw[m]
            fz[m] = (dz, lo, hi, pv, q_all[off:off + pv.size], k)
            off += pv.size

    centers = np.sqrt(edges[:-1] * edges[1:])
    rows = {"bin_lo": edges[:-1], "bin_hi": edges[1:], "center_bp": centers}
    for (m, c), (r, n) in curves.items():
        rows[f"{m}_{c}_r"] = r
        rows[f"{m}_{c}_n"] = n.astype(np.int64)
    for m, (lo, hi, pv, q) in boot.items():
        rows[f"{m}_delta_lo"] = lo
        rows[f"{m}_delta_hi"] = hi
        rows[f"{m}_delta_p"] = pv
        rows[f"{m}_delta_q"] = q
    for m, (dz, lo, hi, pv, q, k) in fz.items():
        rows[f"{m}_fz_dz"] = dz
        rows[f"{m}_fz_lo"] = lo
        rows[f"{m}_fz_hi"] = hi
        rows[f"{m}_fz_p"] = pv
        rows[f"{m}_fz_q"] = q
        # Cells contributing under 'cell'; the smaller arm's pair count under
        # 'pooled' — the n that test wrongly treats as independent.
        rows[f"{m}_fz_n"] = k
    tsv = Path(out_dir) / f"end_posterior_dist_{label}.tsv"
    pd.DataFrame(rows).to_csv(tsv, sep="\t", index=False)
    note = (f"{args.null_type}, --null-scope {args.null_scope}"
            + (f", donors matched on {args.null_match_column}"
               if args.null_match_column and args.null_scope == "pooled"
               else "")
            # Said on the figure itself, so the legend cannot drift from it:
            # under --null-chrom any most null partners are on another
            # chromosome, i.e. matched coordinate offset, not distance.
            + (", null partners on any chromosome"
               if getattr(args, "null_chrom", "same") == "any"
               and args.null_type != "link-shuffle" else "")
            + (f", same {args.link_shuffle_bin / 1000:g} kb bin, "
               + ("other cells" if args.null_scope == "pooled"
                  else "same cell")
               if args.null_type == "link-shuffle" else "")
            + (", ends centred per cell" if _center_on(args) else ""))
    _figure(curves, centers, label,
            Path(out_dir) / f"end_posterior_dist_{label}.{args.plot_format}",
            args, note, boot)
    if fz:
        _figure(curves, centers, label,
                Path(out_dir) / f"end_posterior_dist_{label}_fisherz."
                                f"{args.plot_format}",
                args, note, fisher=fz)
    logger.info(f"  {label}: {n_pairs:,} linked and {n_null:,} null "
                f"end-pairs; wrote {tsv}")
    if want:
        write_percell(cells, per_cell, args, out_dir, percell)
        if getattr(args, "per_region", "off") == "on":
            got = write_regions(per_cell, args, out_dir)
            if got is not None:
                main = (Path(out_dir)
                        / f"end_posterior_dist_{label}_main.{args.plot_format}")
                if _main_figure(curves, centers, label, main, args, *got,
                                note=note, boot=boot) is not None:
                    logger.info(f"  wrote {main}  (GCH, HCG and the regions; "
                                f"the cross-context column stays in "
                                f"end_posterior_dist_{label}.{args.plot_format})")
    return curves, centers


def write_percell(cells, per_cell, args, out_dir, meta):
    """end_posterior_percell.tsv and the Figure 5c panel from it.

    ``per_cell[i]`` belongs to ``cells[i]``: both _drain and the serial loop
    keep job order (executor.map preserves it).
    """
    rows = []
    for i, (cell, acc) in enumerate(zip(cells, per_cell)):
        blocks = acc.get("_percell")
        if not blocks:
            continue
        for m, arms in blocks.items():
            test_arm = arms["M"] if getattr(args, "delta_arm",
                                            "matched") == "matched" else arms["L"]
            st = pce.percell_stats(test_arm, arms["N"], arms.get("N2"),
                                   n_boot=args.per_cell_bootstrap,
                                   seed=args.seed + i,
                                   min_pairs=args.per_cell_min_pairs)
            n_linked = int(arms["L"][1][:, 0].sum()) if len(arms["L"][1]) else 0
            rows.append({"cell_id": cell,
                         "group": meta["group_of"].get(cell, "all"),
                         "mark": pce.MEASURE_TO_MARK[m], "n_linked": n_linked,
                         **st})
    if not rows:
        logger.warning("  per-cell: no cell produced a usable row")
        return None
    df = pce.add_q(pd.DataFrame(rows))
    df["min_distance"] = int(args.per_cell_min_distance)
    df["delta_arm"] = getattr(args, "delta_arm", "matched")
    df["pair_set"] = getattr(args, "pair_set", "joint")
    tsv = Path(out_dir) / "end_posterior_percell.tsv"
    df.to_csv(tsv, sep="\t", index=False)
    for mark, d in df.groupby("mark"):
        ok = d["delta"].notna()
        pos = float((d.loc[ok, "delta"] > 0).mean()) if ok.any() else float("nan")
        okc = d["delta_null"].notna()
        pos0 = (float((d.loc[okc, "delta_null"] > 0).mean())
                if okc.any() else float("nan"))
        logger.info(
            f"  per-cell {mark}: {int(ok.sum()):,} cells; delta>0 in "
            f"{100 * pos:.0f}% (control {100 * pos0:.0f}%); "
            f"{int((d['q'] < 0.05).sum()):,} individually q<0.05")
    plotted = df[df["mark"].isin(pce.MARKS) & df["delta"].notna()
                 & df["delta_null"].notna()]
    if len(plotted):
        # full panel, plus main-figure (column i) and Extended Data (ii-iii)
        prefix = Path(out_dir) / "end_posterior_percell"
        pce.plot_layouts(plotted, out_prefix=str(prefix),
                         group_order=meta.get("order"),
                         group_colors=meta.get("colors"),
                         formats=(args.plot_format,))
        logger.info(f"  wrote {prefix}{{,_main,_ED}}.{args.plot_format}")
    logger.info(f"  wrote {tsv}")
    return df


def _sum_keyed(parts):
    """Sum several (keys, sums) block tables onto one key axis."""
    parts = [p for p in parts if p is not None and p[0].size]
    if not parts:
        return np.empty(0, np.int64), np.empty((0, 6))
    keys = np.concatenate([p[0] for p in parts])
    sums = np.concatenate([p[1] for p in parts])
    uniq, inv = np.unique(keys, return_inverse=True)
    out = np.zeros((uniq.size, 6))
    np.add.at(out, inv, sums)
    return uniq, out


def _rotation_p(x, y, min_shift: int):
    """Two-sided p for corr(x, y) against y rotated along the genome.

    Neighbouring regions are not independent, so a Pearson p would be far
    too small; rotating one track keeps each track's own autocorrelation and
    breaks only their alignment.  Shifts below ``min_shift`` regions are
    skipped -- they barely move the track.
    """
    n = len(x)
    shifts = range(min_shift, n - min_shift + 1)
    if n < 8 or len(shifts) < 5:
        return float("nan")
    r0 = np.corrcoef(x, y)[0, 1]
    rs = np.array([np.corrcoef(x, np.roll(y, k))[0, 1] for k in shifts])
    return float((1 + np.sum(np.abs(rs) >= abs(r0))) / (1 + len(rs)))


def region_summary(reg: pd.DataFrame, args) -> pd.DataFrame:
    """Δ_GCH against Δ_HCG across regions, one row per distance band."""
    from scipy import stats
    out = []
    min_n = int(getattr(args, "region_min_pairs", 500))
    for (blo, bhi), d in reg.groupby(["band_lo", "band_hi"]):
        ok = d[(d["n_obs"] >= min_n) & (d["n_null"] >= min_n)]
        w = ok.pivot_table(index=["chrom_code", "start"], columns="mark",
                           values=["delta", "delta_ctl"]).dropna()
        if not {("delta", "GCH"), ("delta", "HCG")} <= set(w.columns) \
                or len(w) < 8:
            continue
        w = w.sort_index()
        x, y = w[("delta", "GCH")].to_numpy(), w[("delta", "HCG")].to_numpy()
        cx = w[("delta_ctl", "GCH")].to_numpy()
        cy = w[("delta_ctl", "HCG")].to_numpy()
        r, _ = stats.pearsonr(x, y)
        rho, _ = stats.spearmanr(x, y)
        rc, _ = stats.pearsonr(cx, cy)
        # Rotations shorter than ~50 Mb leave neighbouring regions aligned;
        # capped at an eighth of the track so a short one still has shifts.
        min_shift = max(2, min(int(np.ceil(50_000_000
                                           / int(args.region_size))),
                               len(x) // 8))

        def rel(v, c):
            # control Δ (second null minus first) carries only noise, so its
            # spread across regions estimates the noise in a region's Δ
            return max(0.0, 1.0 - np.var(c, ddof=1) / np.var(v, ddof=1))

        rg, rh = rel(x, cx), rel(y, cy)
        ts = stats.theilslopes(y, x)
        rx, ry = stats.rankdata(x), stats.rankdata(y)
        out.append({"band_lo": blo, "band_hi": bhi, "n_regions": len(w),
                    "pearson_r": r, "spearman_rho": rho,
                    "rotation_p": _rotation_p(x, y, min_shift),
                    # the same rotation on ranks: whether the BULK trend, not
                    # just a tail of regions high in both, beats the null
                    "rotation_p_spearman": _rotation_p(rx, ry, min_shift),
                    "ols_slope": float(np.polyfit(x, y, 1)[0]),
                    "theilsen_slope": float(ts[0]),
                    "theilsen_lo": float(ts[2]), "theilsen_hi": float(ts[3]),
                    "control_r": rc, "reliability_GCH": rg,
                    "reliability_HCG": rh,
                    "disattenuated_r": (r / np.sqrt(rg * rh)
                                        if rg > 0 and rh > 0 else np.nan)})
    return pd.DataFrame(out)


def _density_panel(ax, x, y, sparse_frac=0.1):
    """smoothScatter-style panel: kernel density, sparsest points, two fits.

    The density is shown on a fourth-root scale, as R's smoothScatter does,
    so the thin edges stay visible next to the dense core; only the points
    in the sparsest ``sparse_frac`` of the cloud are drawn individually.
    Least squares (with its 95% band) and Theil-Sen are both drawn: when a
    tail of points carries the least-squares slope the two part, and that
    is worth seeing rather than averaging away.
    """
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    px = 0.06 * (np.ptp(x) or 1.0)
    py = 0.06 * (np.ptp(y) or 1.0)
    x0, x1, y0, y1 = x.min() - px, x.max() + px, y.min() - py, y.max() + py
    kde = stats.gaussian_kde(np.vstack([x, y]))
    gx, gy = np.mgrid[x0:x1:220j, y0:y1:220j]
    z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.imshow((z / z.max()).T ** 0.25, origin="lower", aspect="auto",
              extent=[x0, x1, y0, y1], cmap="Blues", interpolation="bilinear")
    dens = kde(np.vstack([x, y]))
    sparse = dens <= np.quantile(dens, sparse_frac)
    ax.scatter(x[sparse], y[sparse], s=7, color="black", lw=0, zorder=3)
    xs = np.linspace(x0, x1, 200)
    slope, icpt = np.polyfit(x, y, 1)
    n = len(x)
    resid = y - (icpt + slope * x)
    s_res = np.sqrt(np.sum(resid ** 2) / max(n - 2, 1))
    sxx = np.sum((x - x.mean()) ** 2) or 1.0
    half = (stats.t.ppf(0.975, max(n - 2, 1)) * s_res
            * np.sqrt(1.0 / n + (xs - x.mean()) ** 2 / sxx))
    fit = icpt + slope * xs
    ax.fill_between(xs, fit - half, fit + half, color="#d62728", alpha=0.18,
                    lw=0, zorder=4)
    ax.plot(xs, fit, color="#d62728", lw=1.6, zorder=5,
            label=f"least squares, slope {slope:+.2f}")
    ts = stats.theilslopes(y, x)
    ax.plot(xs, ts[1] + ts[0] * xs, color="#d62728", lw=1.4, ls="--",
            zorder=5, label=f"Theil–Sen (robust), slope {ts[0]:+.2f}")
    ax.axhline(0, color="#9AA0A6", lw=0.8, zorder=2)
    ax.axvline(0, color="#9AA0A6", lw=0.8, zorder=2)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.legend(fontsize=8, frameon=False, loc="upper left")


def _region_figure(reg: pd.DataFrame, summ: pd.DataFrame, out_path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    min_n = int(getattr(args, "region_min_pairs", 500))
    rows = list(summ.itertuples())
    if not rows:
        return False
    fig, axes = plt.subplots(len(rows), 3, squeeze=False,
                             figsize=(17, 5.0 * len(rows)),
                             gridspec_kw={"width_ratios": [1, 1, 2.0]})
    for i, s in enumerate(rows):
        d = reg[(reg["band_lo"] == s.band_lo) & (reg["band_hi"] == s.band_hi)
                & (reg["n_obs"] >= min_n) & (reg["n_null"] >= min_n)]
        w = d.pivot_table(index=["chrom_code", "start"], columns="mark",
                          values=["delta", "delta_ctl", "n_obs"]).dropna()
        w = w.sort_index()
        band = f"{s.band_lo / 1e3:g}-{s.band_hi / 1e3:g} kb"
        for k, (val, lab) in enumerate((("delta", "Δ = observed − null"),
                                        ("delta_ctl",
                                         "control: null − null"))):
            ax = axes[i, k]
            _density_panel(ax, w[(val, "GCH")].to_numpy(),
                           w[(val, "HCG")].to_numpy())
            ax.set_xlabel(f"GCH–GCH {lab}")
            ax.set_ylabel(f"HCG–HCG {lab}")
            if k == 0:
                ax.set_title(
                    f"{band}: {s.n_regions} regions\n"
                    f"r = {s.pearson_r:+.2f} (rotation p {s.rotation_p:.3f}),"
                    f"  ρ = {s.spearman_rho:+.2f} (rotation p "
                    f"{s.rotation_p_spearman:.3f})", fontsize=10)
            else:
                ax.set_title(f"control r = {s.control_r:+.2f}", fontsize=10)
        ax = axes[i, 2]
        idx = np.arange(len(w))
        for mark, col in (("GCH", "#1f77b4"), ("HCG", "#d62728")):
            v = w[("delta", mark)].to_numpy()
            z = (v - v.mean()) / (v.std() or 1.0)
            ax.plot(idx, z, color=col, lw=1.1, marker="o", ms=2.5,
                    label=f"{mark} Δ (z)")
        codes = np.asarray([c for c, _ in w.index])
        for b in np.flatnonzero(np.diff(codes)) + 0.5:
            ax.axvline(b, color="#DDDDDD", lw=0.8)
        ax.set_xlabel(f"regions in genome order ({args.region_size / 1e6:g} "
                      f"Mb; grey lines: chromosome boundaries)")
        ax.set_ylabel("Δ, z-scored per mark")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Is long-range concordance coordinated in the same places "
                 "for both marks?  (one point per region, all cells pooled)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def _main_band(summ: pd.DataFrame):
    """The band the main figure shows: the only one, else the longest-range."""
    if summ is None or not len(summ):
        return None
    return summ.sort_values("band_lo").iloc[-1]


def _main_figure(curves, centers, label, out_path, args, reg, summ, note="",
                 boot=None):
    """Main-figure layout: GCH and HCG against distance, then the regions.

    The cross-context column is replaced by the per-region answer to the
    question it was meant to ask -- is the two marks' coordination in the
    same places?  Observed region scatter on top, its null-draw control
    below, aligned with the distance panels' two rows.  The three-measure
    figure is still written for Extended Data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    s = _main_band(summ)
    if s is None:
        return None
    min_n = int(getattr(args, "region_min_pairs", 500))
    d = reg[(reg["band_lo"] == s["band_lo"]) & (reg["band_hi"] == s["band_hi"])
            & (reg["n_obs"] >= min_n) & (reg["n_null"] >= min_n)]
    w = d.pivot_table(index=["chrom_code", "start"], columns="mark",
                      values=["delta", "delta_ctl"]).dropna().sort_index()
    if len(w) < 8:
        return None
    titles = dict(_MEASURES)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4),
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for j, key in enumerate(("gg", "hh")):
        axes[0, j].sharex(axes[1, j])
        _draw_measure(axes[0, j], axes[1, j], key, titles[key], curves,
                      centers, args, boot or {}, None, first=(j == 0))
        axes[0, j].tick_params(labelbottom=False)
    size = f"{int(args.region_size) / 1e6:g} Mb"
    band = f"{s['band_lo'] / 1e3:g}-{s['band_hi'] / 1e3:g} kb"
    _density_panel(axes[0, 2], w[("delta", "GCH")].to_numpy(),
                   w[("delta", "HCG")].to_numpy())
    axes[0, 2].set_title(
        f"Same places? one point per {size} region ({int(s['n_regions'])})\n"
        f"r = {s['pearson_r']:+.2f}, ρ = {s['spearman_rho']:+.2f}  "
        f"(genome-rotation p {s['rotation_p']:.3f} / "
        f"{s['rotation_p_spearman']:.3f})", fontsize=10)
    axes[0, 2].set_xlabel(f"GCH–GCH Δ, pairs {band}")
    axes[0, 2].set_ylabel("HCG–HCG Δ")
    _density_panel(axes[1, 2], w[("delta_ctl", "GCH")].to_numpy(),
                   w[("delta_ctl", "HCG")].to_numpy())
    axes[1, 2].set_title(f"control: second null − null   r = "
                         f"{s['control_r']:+.2f}", fontsize=10)
    axes[1, 2].set_xlabel("GCH–GCH control Δ")
    axes[1, 2].set_ylabel("HCG–HCG control Δ")
    fig.suptitle(f"{label}: end-posterior correlation vs distance"
                 + (f"   ({note})" if note else "")
                 + "   |   right: the two marks per region", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def write_regions(per_cell, args, out_dir):
    """end_posterior_regions.tsv, its summary, and the figure.

    Each region's Δ uses every cell's pairs in that region -- the per-cell
    pass's arms, summed -- so it is precise where a single cell's is not.
    The control Δ (second null draw minus the first) has no signal by
    construction: it is what the correlation looks like from noise alone.
    """
    arm = "M" if getattr(args, "delta_arm", "matched") == "matched" else "L"
    rows = []
    measures = sorted({m for acc in per_cell
                       for m in (acc.get("_percell") or {})})
    for m in measures:
        tabs = {t: _sum_keyed([(acc["_percell"][m].get("R") or {}).get(t)
                               for acc in per_cell
                               if m in (acc.get("_percell") or {})])
                for t in (arm, "N", "N2")}
        keys, mats = pce._align([tabs[arm], tabs["N"], tabs["N2"]])
        if not keys.size:
            continue
        r = [pce.corr_from_sums(mt) for mt in mats]
        rows.append(pd.DataFrame({
            "key": keys, "mark": pce.MEASURE_TO_MARK[m],
            "n_obs": mats[0][:, 0].astype(np.int64),
            "n_null": mats[1][:, 0].astype(np.int64),
            "r_obs": r[0], "r_null": r[1], "r_null2": r[2]}))
    if not rows:
        logger.warning("  per-region: nothing to write")
        return None
    reg = pd.concat(rows, ignore_index=True)
    edges_lo = max(float(args.per_cell_min_distance), float(args.min_distance))
    bands = _region_bands(args, edges_lo, float(args.max_distance))
    nb = len(bands) - 1
    band = (reg["key"] % nb).astype(int)
    start = reg["key"] // nb
    reg["chrom_code"] = (start // _GENOME_STRIDE).astype(np.int64)
    reg["chrom"] = [_chrom_name(c) for c in reg["chrom_code"]]
    reg["start"] = (start - reg["chrom_code"] * _GENOME_STRIDE).astype(np.int64)
    reg["end"] = reg["start"] + int(args.region_size)
    reg["band_lo"] = bands[band]
    reg["band_hi"] = bands[band + 1]
    reg["delta"] = reg["r_obs"] - reg["r_null"]
    reg["delta_ctl"] = reg["r_null2"] - reg["r_null"]
    reg = reg.drop(columns="key").sort_values(
        ["band_lo", "chrom_code", "start", "mark"]).reset_index(drop=True)
    cols = ["chrom", "start", "end", "band_lo", "band_hi", "mark", "n_obs",
            "n_null", "r_obs", "r_null", "r_null2", "delta", "delta_ctl",
            "chrom_code"]
    reg = reg[cols]
    tsv = Path(out_dir) / "end_posterior_regions.tsv"
    reg.to_csv(tsv, sep="\t", index=False)
    summ = region_summary(reg, args)
    summ.to_csv(Path(out_dir) / "end_posterior_regions_summary.tsv",
                sep="\t", index=False)
    for s in summ.itertuples():
        logger.info(
            f"  per-region {s.band_lo:g}-{s.band_hi:g} bp: Δ_GCH vs Δ_HCG "
            f"across {s.n_regions} regions r = {s.pearson_r:+.3f} "
            f"(rho {s.spearman_rho:+.3f}, rotation p {s.rotation_p:.3g}); "
            f"control r = {s.control_r:+.3f}")
    fig = Path(out_dir) / f"end_posterior_regions.{args.plot_format}"
    if _region_figure(reg, summ, fig, args):
        logger.info(f"  wrote {fig}")
    logger.info(f"  wrote {tsv}")
    return reg, summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--posteriors", action="append", required=True,
                    metavar="CTX=DIR",
                    help="Repeatable, e.g. GCH=.../04_posteriors.")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cell-column", default="cell_name")
    ap.add_argument("--group-column", default=None,
                    help="Also emit one figure/table per group.")
    ap.add_argument("--null-match-column", default=None,
                    help="With --null-scope pooled, restrict donor ends to "
                         "cells sharing these columns (e.g. "
                         "exp_batch,plate_batch,lib_batch).")
    ap.add_argument("--null-type",
                    choices=["distance-matched", "locus-matched",
                             "link-shuffle"],
                    default="distance-matched")
    ap.add_argument("--null-exclude", type=int, default=0,
                    help="link-shuffle under --null-scope cell: refuse a "
                         "stand-in end within this many bp of the real end "
                         "2 (likely its own molecule) or whose read pair's "
                         "other end lies within it of end 1 (ligated to "
                         "end 1's locus: a near-copy of the contact, or "
                         "another contact of the anchor).  0 (default): off, "
                         "as before; 1000 recommended -- a third of a "
                         "cell's contacts >= 1 kb have a near-copy.")
    ap.add_argument("--link-shuffle-bin", type=int, default=250_000,
                    help="link-shuffle: the donor end must lie in the same "
                         "bin of this size as the real end 2 (250000).")
    ap.add_argument("--cell-center", choices=["auto", "on", "off"],
                    default="auto",
                    help="Centre each end's posterior on its own cell's mean, "
                         "in every arm.  'auto' (default): on for pooled "
                         "link-shuffle only, whose null ends always come from "
                         "two cells while a linked pair's share one -- so "
                         "cell-to-cell differences in level would otherwise "
                         "count as concordance.  It lowers the linked curve "
                         "accordingly.  The same reasoning applies to a "
                         "pooled locus-matched null; 'on' covers it.")
    ap.add_argument("--null-scope", choices=["cell", "pooled"], default="cell")
    ap.add_argument("--null-window", type=int, default=1000,
                    help="Tolerance in bp; also the minimum width of the "
                         "distance-matched target band (1 kb).")
    ap.add_argument("--null-anchor-window", type=int, default=10_000_000,
                    help="Distance-matched only: draw the synthetic anchor "
                         "within ±this many bp of the real one (10 Mb); "
                         "0 disables.")
    ap.add_argument("--min-sites", type=int, default=None,
                    help="Minimum informative sites per END for BOTH contexts. "
                         "Unset: --min-sites-gch / --min-sites-hcg apply.")
    ap.add_argument("--min-sites-gch", type=int, default=2,
                    help="Minimum GCH sites per END (2).")
    ap.add_argument("--min-sites-hcg", type=int, default=2,
                    help="Minimum HCG sites per END (2, as before).  1 keeps "
                         "roughly twice the pairs, but an end called from a "
                         "single CpG has a near-binary posterior, which "
                         "attenuates every HCG correlation -- the raw linked "
                         "curve at 100 bp fell from 0.75 to 0.54 at 1.  Use 1 "
                         "for the per-cell test if depth is the limit, never "
                         "silently for the panel a curve.")
    ap.add_argument("--pair-set", choices=["per-mark", "joint"],
                    default="per-mark",
                    help="'per-mark' (default): GCH-GCH from every "
                         "GCH-informative pair, HCG-HCG from every "
                         "HCG-informative pair, the cross-context measure from "
                         "pairs informative for both.  'joint' puts all three "
                         "on the last set -- comparable across panels, but a "
                         "cell with no HCG output then drops out of GCH-GCH "
                         "and GCH-GCH is restricted to CpG-rich ends.")
    ap.add_argument("--null-chrom", choices=["same", "any"], default="same",
                    help="'same' (default): null partners on the anchor's own "
                         "chromosome, so 'matched distance' is a genomic "
                         "distance.  'any': the ORIGINAL behaviour -- ends "
                         "sorted by per-chromosome position, so most partners "
                         "sit on another chromosome at a similar coordinate.  "
                         "Denser donors, but the null is then 'same cell, "
                         "matched coordinate offset', and should be described "
                         "that way.")
    ap.add_argument("--per-cell-pair-set", choices=["per-mark", "same"],
                    default="per-mark",
                    help="Pair sets for the per-cell test.  'per-mark' "
                         "(default) whatever --pair-set is, because the "
                         "per-cell test needs every usable pair a cell has; "
                         "'same' follows --pair-set.")
    ap.add_argument("--delta-arm", choices=["matched", "linked"],
                    default="matched",
                    help="Arm Δ is taken against the null with.  'matched' "
                         "(default): the true pairs the null could place a "
                         "partner for, so both arms cover the same loci.  "
                         "'linked': every true pair -- a different loci set "
                         "from the null's whenever eligibility is selective.")
    ap.add_argument("--min-distance", type=int, default=100)
    ap.add_argument("--max-distance", type=int, default=10_000_000)
    ap.add_argument("--bins-per-decade", type=int, default=8)
    ap.add_argument("--min-pairs", type=int, default=200,
                    help="Mask a bin with fewer end-pairs than this.")
    ap.add_argument("--exclude-bed", action="append", default=None,
                    metavar="BED",
                    help="Repeatable.  Drop a read pair when EITHER end falls "
                         "in one of these intervals — a blacklist, a "
                         "CpG-island+shore set, or any custom region file.  "
                         "Pair-level, matching the site-level script: a "
                         "contact with one end in a dark region is not half "
                         "usable.  .gz is fine.")
    ap.add_argument("--exclude-chrom", default="",
                    help="Comma-separated chromosomes to drop entirely "
                         "(e.g. chrM,chrX,chrY).")
    ap.add_argument("--haplotypes", default=None, metavar="DIR",
                    help="allelic_imbalance.py --haplotype-dir: per cell the "
                         "read pairs a phased SNP puts on one copy.  Each "
                         "cell is then split into --units -- chrX's inactive "
                         "and active copy and both together, and the same on "
                         "the autosomes -- each run as a cell of its own, so "
                         "--null-scope cell draws from the same cell AND the "
                         "same copy (or, for 'both', either copy).  One group "
                         "per line (--group-column) and unit, and per line "
                         "end_posterior_units_<line>.tsv/.<fmt>: each copy "
                         "and the comparisons between them, cells resampled "
                         "jointly.  chrX must not be in --exclude-chrom.")
    ap.add_argument("--xi", action="append", default=None,
                    metavar="GROUP=1|2|cells",
                    help="--haplotypes: a line's inactive X -- the "
                         "haplotype, the same in every cell, or each cell's "
                         "own from --xi-cells.  Without: hap1 and hap2.")
    ap.add_argument("--xi-cells", default=None,
                    help="allelic_imbalance.py's allelic_cells.tsv.")
    ap.add_argument("--xi-context", default="HCG_island")
    ap.add_argument("--min-cell-sites", type=int, default=20)
    ap.add_argument("--units", default=",".join(UNITS),
                    help="--haplotypes: the units, of " + ", ".join(UNITS)
                         + ".")
    ap.add_argument("--unit-bands", default="1000,10000,100000,1000000,"
                                            "10000000",
                    help="--haplotypes: distance-band edges (bp) of the "
                         "units' comparisons.")
    ap.add_argument("--unit-element", action="append", default=None,
                    metavar="NAME=BED",
                    help="--haplotypes, repeatable: also each copy over the "
                         "contacts with an end in the BED's intervals, that "
                         "end kept and the other replaced by an end of the "
                         "same cell and copy (--null-type link-shuffle), "
                         "drawn from all of the copy's ends; e.g. "
                         "promoter=PLS.bed.  Its rows carry NAME in "
                         "end_posterior_units_<line>.tsv's 'element'.")
    ap.add_argument("--unit-split", action="append", default=None,
                    metavar="NAME=CHROM:START-END",
                    help="--haplotypes, repeatable: also each copy over the "
                         "contacts crossing the region (an end on each "
                         "side) and over those with both ends on one side, "
                         "neither inside -- e.g. DXZ4=chrX:115630000-"
                         "115970000, the inactive X's megadomain boundary "
                         "(--null-type link-shuffle, whose stand-in keeps "
                         "the real end's side).  Rows under NAME in "
                         "end_posterior_units_<line>.tsv's 'element'; "
                         "figure end_posterior_units_<line>_NAME.  Crossing "
                         "contacts are long: raise --max-distance and "
                         "--unit-bands to 100 Mb for chrX.")
    ap.add_argument("--unit-element-flank", type=int, default=100,
                    help="--unit-element: bp each side of an interval an "
                         "end's midpoint may lie and count as in it (100, "
                         "about half a read).")
    ap.add_argument("--genome", default="hg38",
                    help="--haplotypes: for PAR1 and the escape genes.")
    ap.add_argument("--chrx-par", default="auto")
    ap.add_argument("--chrx-escape", default="auto")
    ap.add_argument("--backend", choices=["auto", "thread", "process"],
                    default="auto",
                    help="'auto' (default) takes 'thread' below "
                         f"{_PROCESS_FROM_CELLS} cells and 'process' at or "
                         "above it.  Starting the workers costs a flat ~1.1 s, "
                         "so on 8 workers threads win small runs (8 cells: "
                         "0.24 s against 1.15 s) and lose big ones (192 "
                         "cells: 3.26 s against 2.50 s), crossing near "
                         f"{_PROCESS_FROM_CELLS}.  The margin grows with core "
                         "count, since the GIL the processes escape is "
                         "contended by every extra thread.  Force either to "
                         "override.")
    ap.add_argument("--n-threads", type=int, default=4,
                    help="Cells are scored in parallel and their accumulators "
                         "summed; each cell gets its own seeded RNG, so the "
                         "result does not depend on thread count or order.")
    ap.add_argument("--n-bootstrap", type=int, default=500,
                    help="Cell-level bootstrap resamples for the Δ CI and "
                         "p-value (500; 0 disables).  Cells are the unit "
                         "because end pairs within one are not independent.")
    ap.add_argument("--fisher-z", choices=["cell", "pooled", "none"],
                    default="cell",
                    help="Companion parametric test, written to a separate "
                         "'_fisherz' figure with the same layout.  'cell' "
                         "(default) z-transforms each CELL's linked and null "
                         "r, pairs them within the cell and tests the mean "
                         "difference with a one-sample t: the cell is the "
                         "unit, as in the bootstrap, so this is a genuine "
                         "second opinion on the same contrast and can resolve "
                         "p below the bootstrap's 2/(B+1) floor.  'pooled' is "
                         "the textbook two-sample Fisher z on the pooled "
                         "curves; its n is the number of end pairs, which are "
                         "not independent, so it is anti-conservative by "
                         "orders of magnitude and is offered only to show "
                         "that.  'none' skips it.")
    ap.add_argument("--fisher-min-cell-pairs", type=int, default=30,
                    help="--fisher-z cell: minimum end-pairs for ONE cell to "
                         "contribute an r to a bin (30).  --min-pairs is a "
                         "threshold on the pooled count across every cell and "
                         "is far too high to apply per cell.")
    ap.add_argument("--fdr", choices=["bh", "by", "none"], default="bh",
                    help="Multiple-testing adjustment across every bin and "
                         "measure in a figure.  'bh' assumes positive "
                         "dependence, which adjacent distance bins have.")
    ap.add_argument("--per-cell", choices=["on", "off"], default="on",
                    help="Also test each cell on its own (Figure 5c): panel "
                         "a's delta per cell, a block-bootstrap SE, and a "
                         "second independent null draw as the negative "
                         "control.  Panel a's numbers do not change either "
                         "way.")
    ap.add_argument("--per-cell-min-distance", type=int, default=1000,
                    help="Per-cell test uses pairs at least this far apart "
                         "(1 kb): below it most pairs are the two ends of one "
                         "unligated fragment, not contacts.")
    ap.add_argument("--per-region", choices=["on", "off"], default="on",
                    help="With --per-cell on: also compute each region's Δ "
                         "from every cell's pairs there, and correlate GCH's "
                         "with HCG's across regions -- is the two marks' "
                         "long-range coordination in the same places?")
    ap.add_argument("--region-size", type=int, default=10_000_000,
                    help="Region size for --per-region (10 Mb), on the "
                         "per-chromosome grid.")
    ap.add_argument("--region-bands", default=None,
                    help="Comma-separated distance-band edges (bp) for "
                         "--per-region, e.g. 1000,100000,10000000; default "
                         "one band from --per-cell-min-distance to "
                         "--max-distance.")
    ap.add_argument("--region-min-pairs", type=int, default=500,
                    help="A region enters the correlation only with at least "
                         "this many observed and null pairs for a mark.")
    ap.add_argument("--per-cell-block", type=int, default=10_000_000,
                    help="Genomic block (bp) the per-cell bootstrap resamples "
                         "(10 Mb), so pairs that share a domain move together.")
    ap.add_argument("--per-cell-bootstrap", type=int, default=200)
    ap.add_argument("--per-cell-min-pairs", type=int, default=100,
                    help="A cell needs this many pairs in BOTH arms, beyond "
                         "--per-cell-min-distance, to be tested.")
    ap.add_argument("--group-order", default=None,
                    help="Comma-delimited group order for the per-cell panel; "
                         "names may contain '/', so only ',' splits.")
    ap.add_argument("--color-column", default=None,
                    help="Per-group colour column in --samples for the "
                         "per-cell panel.")
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot-format", choices=["pdf", "png", "svg"],
                    default="pdf")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    paths = {}
    for spec in args.posteriors:
        if "=" not in spec:
            raise SystemExit(f"--posteriors expects CTX=DIR, got {spec!r}")
        ctx, d = spec.split("=", 1)
        paths[ctx.strip()] = Path(d)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(args.samples, sep="\t")
    cells = samples[args.cell_column].astype(str).tolist()
    if args.max_cells:
        cells = cells[:args.max_cells]
    # a cell without a file is skipped where it is read, silently: say how
    # many have one, and stop where none does -- a wrong directory would
    # otherwise run to the end on no pairs at all
    have = {ctx: sum((d / f"{c}.posteriors.parquet").exists() for c in cells)
            for ctx, d in paths.items()}
    for ctx, n_ in have.items():
        logger.info(f"--posteriors {ctx}: {n_:,} of {len(cells):,} cells "
                    f"have {paths[ctx]}/<cell>.posteriors.parquet")
    if not any(have.values()):
        raise SystemExit(
            f"no cell of --samples has a <cell>.posteriors.parquet in "
            f"{', '.join(str(d) for d in paths.values())}: check --posteriors "
            f"and --cell-column")
    batch_of = None
    if args.null_match_column:
        cols = [c.strip() for c in args.null_match_column.split(",")
                if c.strip()]
        missing = [c for c in cols if c not in samples.columns]
        if missing:
            raise SystemExit(f"--null-match-column not in --samples: {missing}")
        idx = samples.set_index(args.cell_column)
        batch_of = (idx[cols].astype(str).agg("|".join, axis=1)).to_dict()

    merged, excl_chrom = load_exclusions(
        args.exclude_bed,
        [c.strip() for c in args.exclude_chrom.split(",") if c.strip()])

    if args.null_type == "link-shuffle" and args.null_scope == "pooled":
        batch_of = _link_strata(samples, args, batch_of)
    if args.null_chrom == "any" and args.null_type != "link-shuffle":
        logger.warning(
            "--null-chrom any: null partners are drawn by per-chromosome "
            "position, so most lie on ANOTHER chromosome at a similar "
            "coordinate.  The non-linked curve is then 'same cell, matched "
            "coordinate offset', not 'matched genomic distance' -- describe it "
            "that way in the legend and Methods.")
    if args.fisher_z == "pooled":
        logger.warning(
            "--fisher-z pooled treats every end pair as an independent "
            "observation.  They are not: one cell contributes most of the "
            "pairs in a bin and one end sits in many of them, so the reported "
            "p is anti-conservative by orders of magnitude.  Read it only "
            "against --fisher-z cell, never on its own.")

    edges = build_edges(args.min_distance, args.max_distance,
                        args.bins_per_decade)
    logger.info(f"{len(cells)} cells, {len(edges) - 1} distance bins, "
                f"contexts {sorted(paths)}")

    groups = {"all": cells}
    if args.group_column:
        gmap = samples.set_index(args.cell_column)[args.group_column]
        for c in cells:
            g = str(gmap.get(c, "NA"))
            groups.setdefault(g, []).append(c)
    unit_lines = {}
    if args.haplotypes:
        groups, batch_of, unit_lines = _unit_setup(samples, cells, args,
                                                   batch_of, excl_chrom,
                                                   paths=paths)
        cells = [u for us in groups.values() for u in us]
    executor = None
    if args.n_threads > 1 and len(cells) > 1:
        # Resolve 'auto' ONCE, here, so run_group and the workers all see a
        # concrete backend rather than each re-deciding it.
        if args.backend == "auto":
            args.backend = ("process" if len(cells) >= _PROCESS_FROM_CELLS
                            else "thread")
            logger.info(f"--backend auto -> {args.backend} ({len(cells)} "
                        f"cells, threshold {_PROCESS_FROM_CELLS}).")
        # Our parallelism is per cell, so pyarrow must not also fan out: in the
        # parent this stops its pool competing with our threads, and
        # _init_worker repeats it inside each process.
        _pin_inner_threads()
        if args.backend == "process":
            from concurrent.futures import ProcessPoolExecutor
            executor = ProcessPoolExecutor(
                max_workers=args.n_threads, initializer=_init_worker,
                initargs=(paths, edges, args, {}, batch_of, merged,
                          excl_chrom))
        else:
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=args.n_threads)
        logger.info(f"Parallel over cells: {args.n_threads} "
                    f"{args.backend}(s), pool reused across groups.")
    elif args.backend == "auto":            # serial: nothing to choose
        args.backend = "thread"
    percell = None
    if args.per_cell == "on" and not unit_lines:
        idx = samples.set_index(args.cell_column)
        group_of = ({c: str(idx[args.group_column].get(c, "NA")) for c in cells}
                    if args.group_column else {})
        order = ([g.strip() for g in args.group_order.split(",") if g.strip()]
                 if args.group_order else None)
        colors = None
        if args.color_column and args.group_column:
            if args.color_column not in samples.columns:
                raise SystemExit(f"--color-column {args.color_column!r} not in "
                                 f"--samples")
            colors = (samples.dropna(subset=[args.color_column])
                      .groupby(args.group_column)[args.color_column].first()
                      .astype(str).to_dict())
        percell = {"group_of": group_of, "order": order, "colors": colors}
    kept: dict = {}
    try:
        per_group = {}
        for label, gcells in groups.items():
            safe = label.replace("/", "_").replace(" ", "_")
            logger.info(f"--- {label} ({len(gcells)} cells) ---")
            # Per-cell statistics depend on the cell, not on the group being
            # summarised, so they are computed once: in the "all" pass.
            per_group[label] = run_group(
                gcells, paths, edges, args, safe, out_dir, batch_of, merged,
                excl_chrom, executor,
                percell=percell if label == "all" else None,
                keep=kept if unit_lines else None)
        bands = np.array([float(x) for x in args.unit_bands.split(",")
                          if x.strip()])
        for line, (labels, named) in unit_lines.items():
            safe = str(line).replace("/", "_").replace(" ", "_")
            # run_group keeps each group under its file-safe label
            safe_of = {k: lab.replace("/", "_").replace(" ", "_")
                       for k, lab in labels.items()}
            parts = []
            for elem in ["all"] + list(args.units_cfg.get("elements", {})):
                lab_e = {_unit_key(k)[0]: v for k, v in safe_of.items()
                         if (_unit_key(k)[1] or "all") == elem}
                t_ = unit_comparisons(kept, lab_e, edges, bands, args, named)
                if len(t_):
                    parts.append(t_.assign(element=elem))
            # each --unit-split region: crossing against one side, per copy
            for nm in args.units_cfg.get("splits", {}):
                lab_s = {}
                for k, v in safe_of.items():
                    base_, el = _unit_key(k)
                    if el and el.rpartition(":")[0] == nm:
                        lab_s[f"{base_}|{el.rpartition(':')[2]}"] = v
                bases = list(dict.fromkeys(k.split("|")[0] for k in lab_s))
                t_ = unit_comparisons(kept, lab_s, edges, bands, args, named,
                                      combos=split_combos(nm, bases, named))
                if len(t_):
                    parts.append(t_.assign(element=nm))
            t = (pd.concat(parts, ignore_index=True) if parts
                 else pd.DataFrame())
            if len(t):
                t.insert(0, "element", t.pop("element"))
                t.insert(0, "line", line)
                f_ = out_dir / f"end_posterior_units_{safe}.tsv"
                t.to_csv(f_, sep="\t", index=False)
                unit_figure(t, line, out_dir / f"end_posterior_units_{safe}."
                                               f"{args.plot_format}")
                for nm in args.units_cfg.get("splits", {}):
                    split_figure(t[t["element"] == nm], line, nm,
                                 out_dir / f"end_posterior_units_{safe}_{nm}."
                                           f"{args.plot_format}")
                logger.info(f"{line}: the copies against each other -> {f_}")
            else:
                logger.warning(f"{line}: too few cells in some unit for the "
                               f"comparisons")
        overlay = out_dir / f"end_posterior_dist_by_group.{args.plot_format}"
        if _figure_by_group(per_group, overlay, args):
            logger.info(f"wrote {overlay}  "
                        f"({len(per_group) - 1} groups overlaid, shared "
                        f"y-scale)")
    finally:
        if executor is not None:
            executor.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
