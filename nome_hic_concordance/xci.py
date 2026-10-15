"""X inactivation: the parts of chrX where a cell's two copies are alike --
PAR1, the genes that escape X inactivation -- the elements specific to the
inactive X, and X's genes split into those that escape and those it
silences, shared by the chrX analyses (annotation_class_concordance.py,
allelic_imbalance.py, haplotype_contacts.py)."""

from __future__ import annotations

import gzip
import re

import numpy as np
import pandas as pd

from .utils import chrom_key

#: PAR1 on chrX (0-based, half-open) per --genome.  It escapes X
#: inactivation -- a cell's two copies are alike there -- unlike PAR2, two of
#: whose four genes the inactive X silences.
PAR1 = {"hg38": (10_000, 2_781_479), "hg19": (60_000, 2_699_520)}
#: Genes outside the PARs that escape X inactivation: Tukiainen et al. 2017
#: (Nature 550:244), Suppl. Table 13, reported XCI status 'Escape' -- the
#: consensus of earlier studies -- XIST left out: expressed from the inactive
#: X alone, it is where the two copies differ most.  (Ensembl ID, name as
#: published, hg19 start, end, hg38 start, end), 0-based half-open: hg19 as
#: published (GENCODE v19), hg38 lifted with UCSC's hg19ToHg38 chain (every
#: gene to chrX, its length kept).
XCI_ESCAPE = (
    ("ENSG00000124343", "XG", 2670090, 2734539, 2752049, 2816498),
    ("ENSG00000056998", "GYG2", 2746828, 2800859, 2828787, 2882818),
    ("ENSG00000006756", "ARSD", 2822010, 2847392, 2903969, 2929351),
    ("ENSG00000157399", "ARSE", 2852698, 2886286, 2934657, 2968245),
    ("ENSG00000205667", "ARSH", 2924653, 2951612, 3006612, 3033571),
    ("ENSG00000062096", "ARSF", 2959511, 3030767, 3041470, 3112726),
    ("ENSG00000101825", "MXRA5", 3226605, 3264682, 3308564, 3346641),
    ("ENSG00000183943", "PRKX", 3522410, 3631649, 3604369, 3713608),
    ("ENSG00000205664", "RP11-706O15.1", 3735568, 3761898, 3817527, 3843857),
    ("ENSG00000130021", "HDHD1", 6966960, 7066231, 7048919, 7148190),
    ("ENSG00000101846", "STS", 7137496, 7272851, 7219455, 7354810),
    ("ENSG00000006757", "PNPLA4", 7866287, 7895780, 7898246, 7927739),
    ("ENSG00000011201", "KAL1", 8496914, 8700227, 8528873, 8732186),
    ("ENSG00000005302", "MSL3", 11776277, 11793870, 11758158, 11775751),
    ("ENSG00000187268", "FAM9C", 13053736, 13062801, 13035617, 13044682),
    ("ENSG00000235385", "GS1-600G8.5", 13284166, 13321571, 13266047, 13303452),
    ("ENSG00000176896", "TCEANC", 13671224, 13700083, 13653105, 13681964),
    ("ENSG00000123595", "RAB9A", 13707243, 13728625, 13689124, 13710506),
    ("ENSG00000196459", "TRAPPC2", 13730362, 13752754, 13712243, 13734635),
    ("ENSG00000046651", "OFD1", 13752831, 13787480, 13734712, 13769361),
    ("ENSG00000046653", "GPM6B", 13789149, 13956757, 13771030, 13938638),
    ("ENSG00000046647", "GEMIN8", 14026397, 14048012, 14008278, 14029893),
    ("ENSG00000130234", "ACE2", 15579155, 15620271, 15561032, 15602148),
    ("ENSG00000147003", "TMEM27", 15645440, 15683154, 15627317, 15665031),
    ("ENSG00000169239", "CA5B", 15706952, 15805747, 15688829, 15787624),
    ("ENSG00000169249", "ZRSR2", 15808594, 15841383, 15790471, 15823260),
    ("ENSG00000182287", "AP1S2", 15843928, 15873054, 15825805, 15854931),
    ("ENSG00000169895", "SYAP1", 16737754, 16783459, 16719631, 16765336),
    ("ENSG00000086712", "TXLNG", 16804549, 16862642, 16786426, 16844519),
    ("ENSG00000173674", "EIF1AX", 20142635, 20159962, 20124517, 20141844),
    ("ENSG00000130741", "EIF2S3", 24072832, 24096088, 24054715, 24077971),
    ("ENSG00000005889", "ZFX", 24167289, 24234372, 24149172, 24216255),
    ("ENSG00000169297", "NR0B1", 30322322, 30327715, 30304205, 30309598),
    ("ENSG00000165164", "CXorf22", 35937850, 36008269, 35919733, 35990152),
    ("ENSG00000185753", "CXorf38", 40488284, 40506819, 40629032, 40647567),
    ("ENSG00000180182", "MED14", 40507557, 40595110, 40648305, 40735858),
    ("ENSG00000124486", "USP9X", 40944887, 41095832, 41085634, 41236579),
    ("ENSG00000215301", "DDX3X", 41192650, 41223725, 41333397, 41364472),
    ("ENSG00000189221", "MAOA", 43515466, 43606068, 43656218, 43746821),
    ("ENSG00000069509", "FUNDC1", 44382884, 44402247, 44523638, 44543001),
    ("ENSG00000147050", "KDM6A", 44732756, 44971847, 44873510, 45112602),
    ("ENSG00000270069", "RP6-99M1.2", 45604638, 45629677, 45745210, 45770274),
    ("ENSG00000130985", "UBA1", 47050259, 47074527, 47190860, 47215128),
    ("ENSG00000102225", "CDK16", 47077258, 47089396, 47217859, 47229997),
    ("ENSG00000126012", "KDM5C", 53220502, 53254604, 53191320, 53225422),
    ("ENSG00000124313", "IQSEC2", 53262057, 53350522, 53232875, 53321324),
    ("ENSG00000072501", "SMC1A", 53401069, 53449677, 53374148, 53422728),
    ("ENSG00000158423", "RIBC1", 53449638, 53458059, 53422689, 53431111),
    ("ENSG00000198034", "RPS4X", 71475528, 71497150, 72255678, 72277300),
    ("ENSG00000225470", "JPX", 73164158, 73290243, 73944323, 74070408),
    ("ENSG00000186310", "NAP1L3", 92925928, 92928567, 93670929, 93673568),
    ("ENSG00000165194", "PCDH19", 99546641, 99665271, 100291643, 100410273),
    ("ENSG00000102362", "SYTL4", 99929487, 99987110, 100674490, 100732123),
    ("ENSG00000007952", "NOX1", 100098312, 100129334, 100843323, 100874345),
    ("ENSG00000102387", "TAF7L", 100523240, 100548059, 101268252, 101293071),
    ("ENSG00000080572", "PIH1D3", 106449861, 106487473, 107206631, 107244243),
    ("ENSG00000187823", "ZCCHC16", 111697726, 111700473, 112454498, 112457245),
    ("ENSG00000147246", "HTR2C", 113818550, 114144624, 114584077, 114910061),
    ("ENSG00000183631", "CXorf64", 125953746, 125955769, 126819763, 126821786),
    ("ENSG00000171004", "HS6ST2", 131760043, 132095423, 132626015, 132961395),
    ("ENSG00000156920", "GPR112", 135383121, 135519215, 136300962, 136437056),
    ("ENSG00000102243", "VGLL1", 135614310, 135638966, 136532151, 136556807),
    ("ENSG00000165509", "MAGEC3", 140926101, 140985618, 141838315, 141897832),
    ("ENSG00000198753", "PLXNB3", 153029650, 153044801, 153764195, 153779346),
    ("ENSG00000102032", "RENBP", 153200715, 153210232, 153935262, 153944780),
    ("ENSG00000073009", "IKBKG", 153769413, 153796782, 154541198, 154568551))


def xci_escape_regions(genome, par=None, flank: int = 10_000):
    """XCI_ESCAPE's genes on ``genome`` (hg38, hg19; else None), each
    ``flank`` bp wider, merged and kept outside PAR1 (``par``): sorted,
    disjoint (start, end) rows on chrX."""
    col = {"hg38": (4, 5), "hg19": (2, 3)}.get(str(genome))
    if col is None:
        return None
    iv = np.array([(r[col[0]] - flank, r[col[1]] + flank)
                   for r in XCI_ESCAPE], np.int64)
    return _clip_merge(iv, par)


def read_escape_bed(path, par=None):
    """A BED's chrX intervals (chrom, start, end), merged and kept outside
    PAR1 (``par``)."""
    b = pd.read_csv(path, sep="\t", header=None, comment="#",
                    usecols=[0, 1, 2], dtype={0: str})
    b = b[b[0].map(chrom_key).str.upper() == "X"]
    return _clip_merge(b[[1, 2]].to_numpy(np.int64), par)


def _clip_merge(iv, par=None):
    iv = np.asarray(iv, np.int64).reshape(-1, 2)
    iv = iv[np.argsort(iv[:, 0], kind="stable")]
    out: list = []
    for a_, b_ in iv:
        a_ = max(int(a_), 0)
        if out and a_ <= out[-1][1]:
            out[-1][1] = max(out[-1][1], int(b_))
        else:
            out.append([a_, int(b_)])
    if par:
        kept = []
        for a_, b_ in out:
            # the parts outside PAR1
            if b_ <= par[0] or a_ >= par[1]:
                kept.append([a_, b_])
            else:
                if a_ < par[0]:
                    kept.append([a_, par[0]])
                if b_ > par[1]:
                    kept.append([par[1], b_])
        out = kept
    return np.array(out, np.int64).reshape(-1, 2)


#: Elements specific to the inactive X (hg38, 0-based half-open): XIST,
#: expressed from it alone, and the anchors of its superloops -- FIRRE, ICCE
#: (the locus LOC550643, now NBDY; Bonora et al. 2016) and the macrosatellite
#: DXZ4, its megadomain boundary (hg19's array lifted with UCSC's
#: hg19ToHg38 to 115.63 Mb, its transcripts DANT1/DANT2 to 115.97 Mb).
XI_ELEMENTS = {"hg38": {"ICCE": (56_729_242, 56_819_179),
                        "XIST": (73_820_650, 73_852_753),
                        "DXZ4": (115_630_000, 115_970_000),
                        "FIRRE": (131_688_583, 131_830_928)}}


def parse_par(spec, genome=None):
    """PAR1 from 'auto' (by ``genome``), 'none' or 'START-END'."""
    v = str(spec).strip().lower().replace(",", "")
    if v == "auto":
        return PAR1.get(str(genome)) if genome else None
    if v == "none":
        return None
    a_, _, b_ = v.partition("-")
    if not (a_.isdigit() and b_.isdigit() and int(a_) < int(b_)):
        raise ValueError(f"{spec!r}: auto, none or START-END")
    return (int(a_), int(b_))


def parse_escape(spec, genome=None, par=None):
    """Escape regions from 'auto' (XCI_ESCAPE on ``genome``), 'none' or a
    BED, kept outside PAR1."""
    v = str(spec).strip()
    if v.lower() == "auto":
        return xci_escape_regions(genome, par) if genome else None
    if v.lower() == "none":
        return None
    return read_escape_bed(v, par)


def parse_elements(spec, genome=None) -> dict:
    """The inactive X's elements from 'auto' (XI_ELEMENTS on ``genome``),
    'none' or a BED of named chrX intervals (chrom, start, end, name):
    {name: (start, end)}."""
    v = str(spec).strip()
    if v.lower() == "auto":
        return dict(XI_ELEMENTS.get(str(genome), {})) if genome else {}
    if v.lower() == "none":
        return {}
    b = pd.read_csv(v, sep="\t", header=None, comment="#",
                    usecols=[0, 1, 2, 3], dtype={0: str, 3: str})
    b = b[b[0].map(chrom_key) == "X"]
    return {str(n): (int(a_), int(b_)) for a_, b_, n in
            zip(b[1], b[2], b[3])}


def x_part(pos, par=None, escape=None, elements=None) -> np.ndarray:
    """Each chrX position's part: 'PAR1', an element of the inactive X
    (``elements``, {name: (start, end)}), 'escape' (an escape gene outside
    PAR1) or 'rest'."""
    pos = np.asarray(pos, np.int64)
    out = np.full(pos.shape, "rest", dtype=object)
    if escape is not None and len(escape):
        st, en = escape[:, 0], escape[:, 1]
        i = np.searchsorted(st, pos, side="right") - 1
        out[(i >= 0) & (pos < en[np.maximum(i, 0)])] = "escape"
    for nm, (a_, b_) in (elements or {}).items():
        out[(pos >= a_) & (pos < b_)] = nm
    if par:
        out[(pos >= par[0]) & (pos < par[1])] = "PAR1"
    return out


def read_genes(path) -> pd.DataFrame:
    """Genes (chrom, start, end, name; 0-based) from a GTF ('gene'
    features, gene_name), a UCSC genePred table (refGene,
    ncbiRefSeqCurated: name2) or a BED (name in column 4)."""
    p = str(path).lower()
    opener = gzip.open if p.endswith(".gz") else open
    if p.endswith((".gtf", ".gtf.gz", ".gff", ".gff.gz", ".gff3",
                   ".gff3.gz")):
        rows = []
        with opener(path, "rt") as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if line.startswith("#") or len(f) < 9 or f[2] != "gene":
                    continue
                m = (re.search(r'gene_name "([^"]+)"', f[8])
                     or re.search(r'gene_id "([^"]+)"', f[8]))
                if m:
                    rows.append((f[0], int(f[3]) - 1, int(f[4]), m.group(1)))
        return pd.DataFrame(rows, columns=["chrom", "start", "end", "name"])
    t = pd.read_csv(path, sep="\t", header=None, comment="#", dtype=str)
    if t.shape[1] >= 13 and t[3].isin(["+", "-"]).all():
        # genePred with its bin: chrom, strand, txStart, txEnd ... name2
        g = pd.DataFrame({"chrom": t[2], "start": t[4], "end": t[5],
                          "name": t[12]})
    elif t.shape[1] >= 10 and t[2].isin(["+", "-"]).all():
        g = pd.DataFrame({"chrom": t[1], "start": t[3], "end": t[4],
                          "name": t[0]})
    else:
        g = pd.DataFrame({"chrom": t[0], "start": t[1], "end": t[2],
                          "name": t[3] if t.shape[1] > 3 else "."})
    g = g.astype({"start": np.int64, "end": np.int64})
    # one row per gene: its transcripts' extent
    return (g.groupby(["chrom", "name"], as_index=False)
            .agg(start=("start", "min"), end=("end", "max"))
            [["chrom", "start", "end", "name"]])


def in_intervals(pos, iv) -> np.ndarray:
    """Whether each position lies in the sorted, disjoint (start, end)
    rows ``iv``."""
    pos = np.asarray(pos, np.int64)
    if iv is None or not len(iv):
        return np.zeros(pos.shape, bool)
    i = np.searchsorted(iv[:, 0], pos, side="right") - 1
    return (i >= 0) & (pos < iv[np.maximum(i, 0), 1])


def subtract_intervals(iv, minus):
    """The sorted, disjoint rows ``iv`` less the sorted, disjoint rows
    ``minus``."""
    iv = np.asarray(iv, np.int64).reshape(-1, 2)
    if minus is None or not len(minus):
        return iv
    out = []
    for a_, b_ in iv:
        cur = int(a_)
        for c_, d_ in minus:
            if d_ <= cur or c_ >= b_:
                continue
            if c_ > cur:
                out.append([cur, int(c_)])
            cur = max(cur, int(d_))
            if cur >= b_:
                break
        if cur < b_:
            out.append([cur, int(b_)])
    return np.array(out, np.int64).reshape(-1, 2)


#: The two kinds of X genes compared: those that escape X inactivation, and
#: the rest of X's genes, which the inactive X silences.
GENE_CLASSES = ("escape genes", "inactivated genes")


def gene_classes(genes: pd.DataFrame, escape, par=None,
                 flank: int = 10_000) -> dict:
    """X's genes outside PAR1 in GENE_CLASSES: the escape regions
    (``escape``), and ``genes``' on chrX, each ``flank`` bp wider, less
    the escape regions -- {class: sorted, disjoint (start, end) rows}."""
    x = genes[genes["chrom"].map(chrom_key).str.upper() == "X"]
    iv = _clip_merge(np.column_stack([x["start"].to_numpy(np.int64) - flank,
                                      x["end"].to_numpy(np.int64) + flank]),
                     par)
    return {GENE_CLASSES[0]: np.asarray(escape, np.int64).reshape(-1, 2),
            GENE_CLASSES[1]: subtract_intervals(iv, escape)}


def cell_xi_table(path, context: str = "HCG_island",
                  min_sites: int = 20) -> pd.DataFrame:
    """Per cell, from allelic_imbalance.py's allelic_cells.tsv, on X
    outside PAR1 and the escape genes when the table has X's parts: the
    sites on each haplotype at ``context``, each one's level, hap1 less
    hap2, the haplotype the more methylated -- at CpG islands, the cell's
    inactive X -- and the binomial SD of the difference from its sites
    alone (sites on one read move together, so the true noise is wider).
    Cells with ``min_sites`` on each haplotype and the two levels unequal."""
    c = pd.read_csv(path, sep="\t", dtype={"cell": str})
    x = c[c["chrom"].map(chrom_key).eq("X")]
    if "part" in x and x["part"].eq("rest").any():
        x = x[x["part"].eq("rest")]
    elif "part" in x:
        x = x[x["part"].eq("all")]
    need = [f"{context}_sites_hap1", f"{context}_sites_hap2",
            f"{context}_hap1_minus_hap2"]
    if not set(need) <= set(x.columns):
        raise ValueError(f"{path}: no {context} columns")
    x = x[(x[need[0]] >= min_sites) & (x[need[1]] >= min_sites)
          & x[need[2]].notna() & x[need[2]].ne(0)]
    out = pd.DataFrame({"cell": x["cell"].to_numpy(),
                        "group": (x["group"].to_numpy() if "group" in x
                                  else ""),
                        "sites_hap1": x[need[0]].to_numpy(float),
                        "sites_hap2": x[need[1]].to_numpy(float),
                        "hap1_minus_hap2": x[need[2]].to_numpy(float)})
    for h in (1, 2):
        col = f"{context}_hap{h}"
        out[f"hap{h}"] = x[col].to_numpy(float) if col in x else np.nan
    out["xi_hap"] = np.where(out["hap1_minus_hap2"] > 0, 1, 2)
    with np.errstate(invalid="ignore"):
        out["noise_sd"] = np.sqrt(
            out["hap1"] * (1 - out["hap1"]) / out["sites_hap1"]
            + out["hap2"] * (1 - out["hap2"]) / out["sites_hap2"])
    return out.drop_duplicates("cell").reset_index(drop=True)


def xi_of_cells(path, context: str = "HCG_island",
                min_sites: int = 20) -> dict:
    """Each cell's inactive X, :func:`cell_xi_table`'s: {cell: 1 | 2}."""
    t = cell_xi_table(path, context, min_sites)
    return dict(zip(t["cell"], t["xi_hap"]))
