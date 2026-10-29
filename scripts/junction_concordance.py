#!/usr/bin/env python3
"""Concordance against the DNA between a contact's two ends when labelled.

M.CviPI labels the nuclei after proximity ligation (Methods), so when the
GpC label is laid down a contact's two ends are one molecule, joined at the
ligation junction.  If access to that molecule at labelling time made its
two ends alike, their likeness would fall with the DNA between them along
the molecule, as it does along one unligated molecule.  Concordance from the
native chromatin of the two loci has no reason to: where along each
fragment a read happens to lie is set by the random priming after
labelling, not by the loci.

``extract`` reads each cell's BAM once and keeps the read pairs the
pipeline counts (both mates MAPQ >= --mapq-min; no duplicate, secondary or
supplementary alignment), with each end's strand, alignment span and split
alignment (SA), and per mark (GCH, HCG) its informative sites -- k, n and
their mean position -- counted as nome_hic_concordance.bam_parser counts
them.  It keeps:

  cis       read pairs on one chromosome 1 kb apart or more; each end's
            junction is the first DpnII site (GATC) in the read's 3'
            direction, where the molecule runs on into the other end's
            fragment (the read may run into the site; one within
            --max-junction-gap of the read, else none) -- for a split read
            clipped at its 3' end, the clip
  trans     read pairs on two chromosomes, their junctions as cis
  molecule  a sample (--molecule-fraction) of unligated molecules: inward
            read pairs under 1 kb whose reads do not overlap

and writes one table per cell (pairs/<cell>.parquet).  ``analyze`` reads
them with the published end posteriors (--posteriors): per mark, each end's
posterior from its k and n under the published mixture fit, as Step 4
computes it, centred on the cell's mean, as the Fig. 5 runs were; each
contact (cis, >= --min-distance) against stand-ins drawn by
end_posterior_distance_correlation.py's own link-shuffle draw from the
cell's ends in the partner's 250-kb bin and distance bin, near-copies
refused (--null-exclude): control 1 of Fig. 5c,d; trans contacts against
the cell's ends in the partner's 250-kb bin.  s, the DNA between the two
ends' sites along the molecule: each end's mean site distance to its
junction, summed, for a ligation product; the distance between the two
ends' mean sites for an unligated molecule.  Per mark and bin of s:

  contacts            r, its stand-ins' r, and Δr with its cell-bootstrap
                      interval; also with both ends at >= 3 sites, and
                      without split reads, whose alignments are shorter
  trans               the same
  molecules           r, one molecule's two ends with no junction between

junction_concordance.tsv holds the bins, junction_trend.tsv the trend --
Δr (molecules: r) where s < NEAR bp against s >= FAR bp, with its
interval -- and junction_concordance.<fmt> draws both.

    python scripts/junction_concordance.py \\
        --samples samples.tsv --reference mm10.2bit \\
        --context-bed GCH:GCH_motif_seqkit.mm10.bed.gz,HCG:HCG_motif_seqkit.mm10.bed.gz \\
        --posteriors GCH=output/final/context_GCH/04_posteriors \\
        --posteriors HCG=output/final/context_HCG/04_posteriors \\
        --exclude-chrom chrM,chrX,chrY --exclude-bed mm10-blacklist.v2.bed \\
        --n-threads 10 --output-dir output/junction_concordance
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import multiprocessing as mp
import sys
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance import bam_parser as _bp  # noqa: E402
from nome_hic_concordance import reference as _ref  # noqa: E402
from nome_hic_concordance.distance_stats import (  # noqa: E402
    accumulate, bootstrap_delta, build_edges, corr_from_acc, new_acc)
from nome_hic_concordance.mixture_model import BetaBinomialMixture  # noqa: E402
from nome_hic_concordance.utils import manifest_text  # noqa: E402

logger = logging.getLogger("junction_concordance")
_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ed = _load("end_posterior_distance_correlation")
scb = _load("split_context_bed")

MARKS = ("GCH", "HCG")
#: A read on a contact may run into its junction's DpnII site by this much:
#: fill-in and blunt ligation leave GATCGATC across the junction.
OVERLAP = 8
#: A split read's junction is where its 3' end is clipped, by this much or
#: more, wherever the ligation was.
CLIP_MIN = 10
#: cis read pairs kept by extract: this far apart or more (bp); below
#: MOLECULE_MAX, the sampled unligated molecules.
CIS_MIN, MOLECULE_MAX = 1_000, 1_000
#: The null's distance bins: the published runs' (--min-distance 100,
#: --max-distance 10 Mb, 8 bins per decade).
EDGES = build_edges(100, 10_000_000, 8)
#: Bins of s, the DNA between the two ends' sites (bp); the last bin of
#: every table holds the pairs with no s (no junction found).
S_EDGES = np.array([0, 50, 100, 150, 200, 250, 300, 400, 500, 700, 1000])
NB = len(S_EDGES)            # len(S_EDGES) - 1 bins of s, and the no-s bin
#: The trend: s below NEAR against s at FAR and beyond (bp).
NEAR, FAR = 200, 300
#: The sets analyze reports, and the arms each has.
SETS = {"contacts": ("linked", "null"), "contacts_3sites": ("linked", "null"),
        "contacts_unsplit": ("linked", "null"), "trans": ("linked", "null"),
        "molecules": ("linked",)}
SET_LABEL = {"contacts": "contacts", "contacts_3sites":
             "contacts, both ends >= 3 sites", "contacts_unsplit":
             "contacts, no split read", "trans": "trans contacts",
             "molecules": "one molecule, no junction"}


# ----------------------------------------------------------------- extract --

def dpnii_sites(seq: str) -> np.ndarray:
    """The starts of every GATC in ``seq`` (sorted)."""
    a = np.frombuffer(seq.upper().encode(), dtype=np.uint8)
    if a.size < 4:
        return np.zeros(0, np.int64)
    hit = ((a[:-3] == ord("G")) & (a[1:-2] == ord("A"))
           & (a[2:-1] == ord("T")) & (a[3:] == ord("C")))
    return np.flatnonzero(hit).astype(np.int64)


def junction(sites, start: int, end: int, reverse: bool,
             max_gap: int) -> int:
    """A contact read's ligation junction: the middle of the first DpnII
    site in its 3' direction -- at or past its end on the forward strand,
    at or before its start on the reverse, the read running into it by up
    to OVERLAP bases -- or -1 with none within ``max_gap`` of the read."""
    if sites is None or not len(sites):
        return -1
    if not reverse:
        i = int(np.searchsorted(sites, end - OVERLAP, side="left"))
        if i == len(sites) or sites[i] - end > max_gap:
            return -1
        return int(sites[i]) + 2
    i = int(np.searchsorted(sites, start + OVERLAP - 4, side="right")) - 1
    if i < 0 or start - (sites[i] + 4) > max_gap:
        return -1
    return int(sites[i]) + 2


def read_sites(read, strand: str, pos, ref, trim_ends: int):
    """(k, n, sum of positions) over the read's informative sites of one
    mark, counted as bam_parser._count_context_in_read counts them;
    ``strand``, the read's conversion strand."""
    s0 = read.reference_start + trim_ends
    e0 = read.reference_end - trim_ends
    if s0 >= e0 or strand not in ("CT", "GA"):
        return 0, 0, 0
    i0 = int(np.searchsorted(pos, s0, side="left"))
    i1 = int(np.searchsorted(pos, e0, side="left"))
    seq = read.query_sequence
    if i0 >= i1 or seq is None:
        return 0, 0, 0
    want = 67 if strand == "CT" else 71
    site_pos = pos[i0:i1].tolist()
    site_ref = ref[i0:i1].tolist()
    meth, unmeth = ("C", "T") if want == 67 else ("G", "A")
    if read.reference_length == read.query_alignment_length:
        offset = read.query_alignment_start - read.reference_start
        qpos = [p + offset for p in site_pos]
        qpos = [q if 0 <= q < len(seq) else None for q in qpos]
    else:
        pairs = {r_: q_ for q_, r_ in read.get_aligned_pairs(matches_only=True)}
        qpos = [pairs.get(p) for p in site_pos]
    k = n = total = 0
    for p, rb, q in zip(site_pos, site_ref, qpos):
        if rb != want or q is None:
            continue
        b = seq[q]
        if b == meth:
            k += 1
        elif b != unmeth:
            continue
        n += 1
        total += p
    return k, n, total


def end_record(read, sites: dict, gatc, trim_ends: int, max_gap: int):
    """One end: (chrom, start, end, reverse, split, junction, per mark
    (k, n, mean site position or NaN))."""
    chrom = read.reference_name
    start, end = read.reference_start, read.reference_end
    rev = read.is_reverse
    strand = None
    marks = []
    for m in MARKS:
        pos, ref = sites[m].get(chrom, (None, None))
        if pos is None:
            marks.append((0, 0, np.nan))
            continue
        if strand is None:
            i0 = np.searchsorted(pos, start + trim_ends, side="left")
            i1 = np.searchsorted(pos, end - trim_ends, side="left")
            if i0 >= i1:
                marks.append((0, 0, np.nan))
                continue
            strand = _bp._infer_conversion_strand(read)
        k, n, total = read_sites(read, strand, pos, ref, trim_ends)
        marks.append((k, n, total / n if n else np.nan))
    split = read.has_tag("SA")
    j = junction(gatc.get(chrom), start, end, rev, max_gap)
    if split:
        # a read across its junction: where its 3' end leaves this locus
        op, length = read.cigartuples[0 if rev else -1]
        if op in (4, 5) and length >= CLIP_MIN:
            j = start if rev else end
    return (chrom, start, end, rev, split, j, marks)


#: The columns of a cell's pairs table, beyond each mark's six.
PAIR_COLUMNS = ("kind", "chrom1", "pos1", "chrom2", "pos2", "distance",
                "start1", "end1", "start2", "end2", "rev1", "rev2",
                "sa1", "sa2", "j1", "j2")


def _sampled(qname: str, frac_ppm: int) -> bool:
    """Whether a molecule is in the --molecule-fraction sample: by its
    name, so both mates agree."""
    return zlib.crc32(qname.encode()) % 1_000_000 < frac_ppm


def _pair_row(a, b, frac_ok: bool):
    """The row of a read pair -- ``a`` the read seen second, as bam_parser
    pairs them -- or None if it is neither kept kind."""
    if a[0] == b[0]:
        # the pipeline's order: the read starting first (ties: ``a``)
        r1, r2 = (a, b) if a[1] <= b[1] else (b, a)
        mid1, mid2 = (r1[1] + r1[2]) // 2, (r2[1] + r2[2]) // 2
        dist = abs(mid2 - mid1)
        if dist >= CIS_MIN:
            kind = "cis"
        elif (frac_ok and dist < MOLECULE_MAX and not r1[3] and r2[3]
              and r1[2] <= r2[1]):
            kind = "molecule"
        else:
            return None
    else:
        r1, r2 = sorted((a, b), key=lambda r: (ed._chrom_code(r[0]), r[1]))
        mid1, mid2 = (r1[1] + r1[2]) // 2, (r2[1] + r2[2]) // 2
        kind, dist = "trans", -1
    row = [kind, r1[0], mid1, r2[0], mid2, dist, r1[1], r1[2], r2[1], r2[2],
           r1[3], r2[3], r1[4], r2[4], r1[5], r2[5]]
    for i in range(len(MARKS)):
        row += [*r1[6][i], *r2[6][i]]
    return row


def mark_columns() -> list:
    return [f"{m}_{f}{e}" for m in MARKS for e in (1, 2)
            for f in ("k", "n", "c")]


def extract_cell(bam_path, sites: dict, gatc: dict, allowed, args):
    """One cell's kept read pairs (PAIR_COLUMNS and each mark's k, n and
    mean site position per end)."""
    import pysam
    pysam.set_verbosity(0)
    frac = int(round(args.molecule_fraction * 1_000_000))
    skip = _bp._SKIP_FLAGS
    rows, buf = [], {}
    # The mates' starts are within half a read's length of their midpoints'
    # distance: below CIS_MIN - SLACK apart a pair is no contact.
    slack = 300
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        # one buffer across chromosomes, so a trans pair's mates meet
        for chrom in [c for c in bam.references if c in allowed]:
            for read in bam.fetch(chrom):
                if read.flag & skip or read.mapping_quality < args.mapq_min:
                    continue
                mchrom = read.next_reference_name
                if mchrom not in allowed:
                    continue
                qname = read.query_name
                frac_ok = False
                if chrom == mchrom:
                    gap = abs(read.next_reference_start
                              - read.reference_start)
                    frac_ok = (gap < MOLECULE_MAX + slack
                               and _sampled(qname, frac))
                    if gap < CIS_MIN - slack and not frac_ok:
                        continue
                rec = end_record(read, sites, gatc, args.trim_ends,
                                 args.max_junction_gap)
                mate = buf.pop(qname, None)
                if mate is None:
                    buf[qname] = rec
                    continue
                row = _pair_row(rec, mate, frac_ok)
                if row is not None:
                    rows.append(row)
    cols = list(PAIR_COLUMNS) + mark_columns()
    t = pd.DataFrame(rows, columns=cols)
    for c in ("pos1", "pos2", "distance", "start1", "end1", "start2", "end2",
              "j1", "j2"):
        t[c] = t[c].astype(np.int64)
    for c in ("rev1", "rev2", "sa1", "sa2"):
        t[c] = t[c].astype(bool)
    for m in MARKS:
        for e in (1, 2):
            t[f"{m}_k{e}"] = t[f"{m}_k{e}"].astype(np.int32)
            t[f"{m}_n{e}"] = t[f"{m}_n{e}"].astype(np.int32)
    t["kind"] = pd.Categorical(t["kind"], categories=["cis", "trans",
                                                      "molecule"])
    return t


#: extract's genome, shared by its forked workers.
_X: dict = {}


def _extract_one(job):
    cell, bam, out = job
    t = extract_cell(bam, _X["sites"], _X["gatc"], _X["allowed"], _X["args"])
    tmp = out.with_suffix(".tmp.parquet")
    t.to_parquet(tmp, index=False)
    tmp.replace(out)
    return cell


def cell_summary(cell: str, t: pd.DataFrame) -> dict:
    """A cell's read pairs of each kind, and the share of its contacts with
    a junction found at both ends."""
    kinds = t["kind"].value_counts()
    lig = t["kind"] != "molecule"
    return {"cell": cell, **{k: int(kinds.get(k, 0)) for k in
                             ("cis", "trans", "molecule")},
            "junction_both": float(((t["j1"] >= 0) & (t["j2"] >= 0))[lig]
                                   .mean()) if lig.any() else np.nan}


def extract_summary(cells, pairs_dir: Path) -> pd.DataFrame:
    """cell_summary of every cell with a pairs table, whichever run wrote
    it."""
    rows = [cell_summary(c, pd.read_parquet(pairs_dir / f"{c}.parquet",
                                            columns=["kind", "j1", "j2"]))
            for c in cells if (pairs_dir / f"{c}.parquet").exists()]
    return pd.DataFrame(rows)


def load_genome(args, excl_chrom):
    """Each mark's sites ({chrom: (pos, ref_base)}), each chromosome's DpnII
    sites, and the chromosomes kept."""
    specs = {}
    for tok in args.context_bed.split(","):
        ctx, _, p = tok.strip().partition(":")
        specs[ctx.strip().upper()] = p.strip()
    missing = [m for m in MARKS if m not in specs]
    if missing:
        sys.exit(f"--context-bed has no {', '.join(missing)}")
    raw = _ref.load_context_from_bed({m: specs[m] for m in MARKS},
                                     n_threads=args.n_threads,
                                     c_offset=args.context_bed_offset)
    allowed = set.intersection(*(set(raw[m]) for m in MARKS))
    allowed = {c for c in allowed if ed._canon(c) not in excl_chrom}
    sites = {m: {c: (np.ascontiguousarray(a["pos"], dtype=np.int64),
                     np.ascontiguousarray(a["ref_base"], dtype=np.uint8))
                 for c, a in raw[m].items() if c in allowed}
             for m in MARKS}
    del raw
    fa = scb._genome(args.reference)
    gatc = {c: dpnii_sites(fa.fetch(c)) for c in sorted(allowed)
            if c in fa.references}
    logger.info(f"{len(allowed)} chromosomes; "
                + ", ".join(f"{m} {sum(len(v[0]) for v in sites[m].values()):,}"
                            for m in MARKS)
                + f" sites; {sum(len(v) for v in gatc.values()):,} DpnII sites")
    return sites, gatc, allowed


def run_extract(cells, bams, args, excl_chrom) -> None:
    out_dir = Path(args.output_dir) / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(c, b, out_dir / f"{c}.parquet") for c, b in zip(cells, bams)
            if args.overwrite or not (out_dir / f"{c}.parquet").exists()]
    absent = [b for _c, b, _o in jobs if not Path(b).exists()]
    if absent:
        sys.exit(f"{len(absent)} BAM files missing, e.g. {absent[0]}")
    logger.info(f"extract: {len(jobs)} of {len(cells)} cells to read "
                f"({len(cells) - len(jobs)} done before)")
    if not jobs:
        return
    sites, gatc, allowed = load_genome(args, excl_chrom)
    _X.update(sites=sites, gatc=gatc, allowed=allowed, args=args)
    for i, _cell in enumerate(_map(_extract_one, jobs, args.n_threads), 1):
        if i % 50 == 0 or i == len(jobs):
            logger.info(f"  extract: {i}/{len(jobs)} cells")
    _X.clear()                  # the genome, before analyze's workers fork


def _map(fn, jobs, n_threads):
    """``fn`` over ``jobs``: forked workers sharing the parent's arrays, or
    in turn."""
    if n_threads <= 1 or len(jobs) <= 1 \
            or "fork" not in mp.get_all_start_methods():
        for j in jobs:
            yield fn(j)
        return
    with ProcessPoolExecutor(max_workers=n_threads,
                             mp_context=mp.get_context("fork")) as ex:
        yield from ex.map(fn, jobs, chunksize=1)


# ----------------------------------------------------------------- analyze --

def separation(t: pd.DataFrame, mark: str, junctioned: bool) -> np.ndarray:
    """s per pair (bp; NaN without it): across the junction, each end's mean
    site distance to its junction, summed; along an unligated molecule,
    the distance between the two ends' mean sites."""
    c1 = t[f"{mark}_c1"].to_numpy(float)
    c2 = t[f"{mark}_c2"].to_numpy(float)
    if not junctioned:
        return np.abs(c2 - c1)
    out = np.zeros(len(t))
    for e, c in ((1, c1), (2, c2)):
        j = t[f"j{e}"].to_numpy(float)
        rev = t[f"rev{e}"].to_numpy(bool)
        d = np.where(rev, c - j, j - c)
        out += np.where(t[f"j{e}"].to_numpy() >= 0, np.clip(d, 0, None),
                        np.nan)
    return out


def s_bins(s: np.ndarray) -> np.ndarray:
    """Each s's bin; the last, NB - 1, for none or beyond the last edge."""
    b = np.searchsorted(S_EDGES, s, side="right") - 1
    ok = np.isfinite(s) & (b >= 0) & (b < NB - 1)
    return np.where(ok, b, NB - 1).astype(np.int64)


def default_fit(post_dir) -> Path:
    """The mixture fit beside a context's 04_posteriors, as the pipeline
    lays it out."""
    return Path(post_dir).parent / "03_mixture_model" / "em_fit_all.json"


def _end_keep(chrom, pos, merged, excl_chrom) -> np.ndarray:
    """One end at a time: neither an excluded chromosome nor in an excluded
    interval (trans pairs, which _keep_mask's one chrom cannot hold)."""
    df = pd.DataFrame({"chrom": chrom, "pos1": pos, "pos2": pos})
    return ed._keep_mask(df, merged, excl_chrom)


def _draw_trans(g1, g2, pool, B: int, W: int, rng, rounds: int = 16):
    """A trans contact's stand-in: another end of the cell in the partner's
    B bin, more than W from the partner; (anchor indices, pool indices)."""
    d_pos = pool[0]
    stride = ed._GENOME_STRIDE
    y0 = g2 - ((g2 & (stride - 1)) % B)
    lo = np.searchsorted(d_pos, y0, side="left")
    hi = np.searchsorted(d_pos, y0 + B, side="left")
    out = np.full(len(g1), -1, np.int64)
    alive = hi > lo
    for _ in range(rounds):
        idx = np.flatnonzero(alive)
        if not idx.size:
            break
        kk = lo[idx] + (rng.random(idx.size) * (hi[idx] - lo[idx])
                        ).astype(np.int64)
        good = np.abs(d_pos[kk] - g2[idx]) > W
        out[idx[good]] = kk[good]
        alive[idx[good]] = False
    ai = np.flatnonzero(out >= 0)
    return ai, out[ai]


def _acc_sets():
    return {(s, arm): new_acc(NB) for s, arms in SETS.items() for arm in arms}


def cell_stats(cell: str, ci: int, t: pd.DataFrame, post_dirs: dict,
               fits: dict, args, merged, excl_chrom) -> dict:
    """One cell's accumulators {(mark, set, arm): (6, NB)}, and per (mark,
    set) the pairs' summed sites and split reads per bin."""
    out = {}
    cols = {"v": ("p1", "p2")}
    ns = argparse.Namespace(null_type="link-shuffle", null_scope="cell",
                            null_exclude=args.null_exclude,
                            link_shuffle_bin=args.link_shuffle_bin)
    base = ed._pair_base(ci)
    for mi_, mark in enumerate(MARKS):
        full = ed._load_cell({mark: Path(post_dirs[mark])}, cell,
                             {mark: args.min_sites}, merged,
                             excl_chrom).get(mark)
        if full is None:
            continue
        full = full.assign(gpos1=ed.genome_coords(full["chrom"],
                                                  full["pos1"]),
                           gpos2=ed.genome_coords(full["chrom"],
                                                  full["pos2"]))
        mu = float(np.nanmean(np.concatenate([full["p1"].to_numpy(float),
                                              full["p2"].to_numpy(float)])))
        full["p1"] = full["p1"].to_numpy(float) - mu
        full["p2"] = full["p2"].to_numpy(float) - mu
        pool = ed._own_contact_pool(full, cols, base, EDGES)
        accs = _acc_sets()
        sites = {s: np.zeros((3, NB)) for s in SETS}
        rng = np.random.default_rng([args.seed, ci, mi_])
        n1, n2 = (t[f"{mark}_n{e}"].to_numpy() for e in (1, 2))
        ok = (n1 >= args.min_sites) & (n2 >= args.min_sites)
        p1 = fits[mark].predict_proba(t[f"{mark}_k1"].to_numpy(), n1) - mu
        p2 = fits[mark].predict_proba(t[f"{mark}_k2"].to_numpy(), n2) - mu
        kind = t["kind"].astype(str).to_numpy()
        split = t["sa1"].to_numpy(bool) | t["sa2"].to_numpy(bool)

        def tally(set_, rows, bins):
            sites[set_][0] += np.bincount(bins, minlength=NB)
            sites[set_][1] += np.bincount(
                bins, weights=(n1[rows] + n2[rows]) / 2, minlength=NB)
            sites[set_][2] += np.bincount(bins, weights=split[rows],
                                          minlength=NB)

        # contacts: control 1's own draw
        cis = ok & (kind == "cis")
        dist = t["distance"].to_numpy()
        cis &= (dist >= args.min_distance) & (dist <= EDGES[-1])
        rows = np.flatnonzero(cis)
        if rows.size:
            rows = rows[ed._keep_mask(t.iloc[rows].rename(
                columns={"chrom1": "chrom"}), merged, excl_chrom)]
        if rows.size and pool is not None:
            g1 = ed.genome_coords(t["chrom1"].to_numpy()[rows],
                                  t["pos1"].to_numpy()[rows])
            g2 = ed.genome_coords(t["chrom2"].to_numpy()[rows],
                                  t["pos2"].to_numpy()[rows])
            j = pd.DataFrame({"gpos1": g1, "gpos2": g2, "p1": p1[rows],
                              "p2": p2[rows]})
            drawn = ed._draw_link_shuffle(
                j, dist[rows], cols, pool, base, EDGES, ns, rng,
                ids=np.full(rows.size, -1, np.int64))
            if drawn is not None:
                xa, xb, dn, _a, ai = drawn
                keep = (dn >= EDGES[0]) & (dn <= EDGES[-1])
                ai, xa_v, xb_v = ai[keep], xa["v"][keep], xb["v"][keep]
                r = rows[ai]
                b = s_bins(separation(t.iloc[r], mark, True))
                for set_, m in (("contacts", np.ones(r.size, bool)),
                                ("contacts_3sites",
                                 (n1[r] >= 3) & (n2[r] >= 3)),
                                ("contacts_unsplit", ~split[r])):
                    accumulate(accs[(set_, "linked")], b[m], p1[r][m],
                               p2[r][m])
                    accumulate(accs[(set_, "null")], b[m], xa_v[m], xb_v[m])
                    tally(set_, r[m], b[m])
        # trans contacts: the partner's 250-kb bin
        rows = np.flatnonzero(ok & (kind == "trans"))
        if rows.size:
            sub = t.iloc[rows]
            keep = (_end_keep(sub["chrom1"].to_numpy(), sub["pos1"].to_numpy(),
                              merged, excl_chrom)
                    & _end_keep(sub["chrom2"].to_numpy(),
                                sub["pos2"].to_numpy(), merged, excl_chrom))
            rows = rows[keep]
        if rows.size and pool is not None:
            g1 = ed.genome_coords(t["chrom1"].to_numpy()[rows],
                                  t["pos1"].to_numpy()[rows])
            g2 = ed.genome_coords(t["chrom2"].to_numpy()[rows],
                                  t["pos2"].to_numpy()[rows])
            ai, kb = _draw_trans(g1, g2, pool, args.link_shuffle_bin,
                                 args.null_exclude, rng)
            r = rows[ai]
            b = s_bins(separation(t.iloc[r], mark, True))
            accumulate(accs[("trans", "linked")], b, p1[r], p2[r])
            accumulate(accs[("trans", "null")], b, p1[r], pool[1]["v"][kb])
            tally("trans", r, b)
        # one unligated molecule's two ends
        rows = np.flatnonzero(ok & (kind == "molecule"))
        if rows.size:
            rows = rows[ed._keep_mask(t.iloc[rows].rename(
                columns={"chrom1": "chrom"}), merged, excl_chrom)]
            b = s_bins(separation(t.iloc[rows], mark, False))
            accumulate(accs[("molecules", "linked")], b, p1[rows], p2[rows])
            tally("molecules", rows, b)
        for (set_, arm), a in accs.items():
            out[(mark, set_, arm)] = a
        for set_, v in sites.items():
            out[(mark, set_, "sites")] = v
    return out


#: analyze's inputs, shared by its forked workers.
_A: dict = {}


def _analyze_one(job):
    cell, ci = job
    p = Path(_A["args"].output_dir) / "pairs" / f"{cell}.parquet"
    if not p.exists():
        return cell, None
    t = pd.read_parquet(p)
    return cell, cell_stats(cell, ci, t, _A["post_dirs"], _A["fits"],
                            _A["args"], _A["merged"], _A["excl_chrom"])


def _boot_weights(k: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot
                           ).astype(np.float64)


def _pooled_r(acc: np.ndarray, mask: np.ndarray, min_pairs: int):
    """r over the bins in ``mask`` pooled, for one accumulator or a stack."""
    a = np.asarray(acc)[..., mask].sum(axis=-1, keepdims=True)
    return corr_from_acc(a, min_pairs)[0][..., 0]


def trend(per_l, per_n, min_pairs: int, n_boot: int, seed: int) -> dict:
    """Δr (r without a null) where s < NEAR against s >= FAR: each side's
    bins pooled; their difference with its cell-bootstrap interval and
    two-sided p, and their ratio with its interval."""
    lo_s, hi_s = S_EDGES[:-1], S_EDGES[1:]
    near = np.zeros(NB, bool)
    far = np.zeros(NB, bool)
    near[:NB - 1] = hi_s <= NEAR
    far[:NB - 1] = lo_s >= FAR
    L = np.stack(per_l)
    N = np.stack(per_n) if per_n is not None else None

    def value(l_, n_, m):
        v = _pooled_r(l_, m, min_pairs)
        return v - _pooled_r(n_, m, min_pairs) if n_ is not None else v

    sl, sn = L.sum(0), (N.sum(0) if N is not None else None)
    vn, vf = value(sl, sn, near), value(sl, sn, far)
    out = {"near": float(vn), "far": float(vf), "difference": float(vn - vf),
           "ratio": float(vn / vf) if vf else np.nan,
           "pairs_near": int(sl[0][near].sum()),
           "pairs_far": int(sl[0][far].sum())}
    k = L.shape[0]
    if k >= 3 and n_boot > 0:
        w = _boot_weights(k, n_boot, seed)
        bl = (w @ L.reshape(k, -1)).reshape(n_boot, 6, NB)
        bn = ((w @ N.reshape(k, -1)).reshape(n_boot, 6, NB)
              if N is not None else None)
        with np.errstate(invalid="ignore", divide="ignore"):
            dn_, df_ = value(bl, bn, near), value(bl, bn, far)
            d = dn_ - df_
            ratio = dn_ / df_
        fin = np.isfinite(d)
        if fin.sum() >= 20:
            out["difference_lo"], out["difference_hi"] = np.percentile(
                d[fin], [2.5, 97.5])
            out["p"] = float(min(1.0, 2.0 * (1.0 + min((d[fin] <= 0).sum(),
                                                       (d[fin] >= 0).sum()))
                                 / (fin.sum() + 1.0)))
            rf = np.isfinite(ratio)
            out["ratio_lo"], out["ratio_hi"] = np.percentile(ratio[rf],
                                                             [2.5, 97.5])
    return out


def _boot_r(per_l, min_pairs: int, n_boot: int, seed: int):
    """Per bin, r's cell-bootstrap interval."""
    L = np.stack(per_l)
    k = L.shape[0]
    if k < 3 or n_boot <= 0:
        return None
    w = _boot_weights(k, n_boot, seed)
    b = (w @ L.reshape(k, -1)).reshape(n_boot, 6, -1)
    with np.errstate(invalid="ignore"):
        r = corr_from_acc(b, min_pairs)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(r, [2.5, 97.5], axis=0)


def summarize(per_cell: dict, args):
    """The bins (junction_concordance.tsv) and the trends
    (junction_trend.tsv) from the cells' accumulators."""
    rows, trends = [], []
    s_lo = list(S_EDGES[:-1]) + [np.nan]
    s_hi = list(S_EDGES[1:]) + [np.nan]
    for mark in MARKS:
        for set_, arms in SETS.items():
            cells = [c for c, d in per_cell.items()
                     if d is not None and (mark, set_, "linked") in d]
            if not cells:
                continue
            per_l = [per_cell[c][(mark, set_, "linked")] for c in cells]
            per_n = ([per_cell[c][(mark, set_, "null")] for c in cells]
                     if "null" in arms else None)
            sl = np.sum(per_l, axis=0)
            rl, nl = corr_from_acc(sl, args.min_pairs)
            st = np.sum([per_cell[c][(mark, set_, "sites")] for c in cells],
                        axis=0)
            if per_n is not None:
                sn = np.sum(per_n, axis=0)
                rn, nn = corr_from_acc(sn, args.min_pairs)
                bt = bootstrap_delta(per_l, per_n, args.min_pairs,
                                     args.n_boot, args.seed)
            else:
                rn = nn = np.full(NB, np.nan)
                bt = None
                br = _boot_r(per_l, args.min_pairs, args.n_boot, args.seed)
            for b in range(NB):
                row = {"mark": mark, "set": set_, "s_lo": s_lo[b],
                       "s_hi": s_hi[b], "pairs": int(nl[b]),
                       "r": rl[b], "r_null": rn[b],
                       "delta": rl[b] - rn[b] if per_n is not None else np.nan,
                       "sites_per_end": st[1][b] / st[0][b] if st[0][b]
                       else np.nan,
                       "split_share": st[2][b] / st[0][b] if st[0][b]
                       else np.nan}
                if bt is not None:
                    row.update(delta_lo=bt[0][b], delta_hi=bt[1][b],
                               p=bt[2][b])
                elif per_n is None and br is not None:
                    row.update(r_lo=br[0][b], r_hi=br[1][b])
                rows.append(row)
            # every pair of the set, s or none
            allm = np.ones(NB, bool)
            r_all = _pooled_r(sl, allm, args.min_pairs)
            tr = {"mark": mark, "set": set_, "cells": len(cells),
                  "pairs": int(sl[0].sum()),
                  "pairs_with_s": int(sl[0][:NB - 1].sum()),
                  "r_all": float(r_all),
                  "delta_all": (float(r_all - _pooled_r(np.sum(per_n, 0),
                                                        allm, args.min_pairs))
                                if per_n is not None else np.nan)}
            tr.update(trend(per_l, per_n, args.min_pairs, args.n_boot,
                            args.seed))
            trends.append(tr)
    return pd.DataFrame(rows), pd.DataFrame(trends)


def plot(table: pd.DataFrame, trends: pd.DataFrame, path):
    """Per mark: r against s for one molecule's ends and for contacts with
    their stand-ins (top); Δr against s for contacts and trans contacts
    (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nome_hic_concordance.percell_existence import MARK_COLOR, NULL_GREY
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7,
                         "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    marks = [m for m in MARKS if (table["mark"] == m).any()]
    fig, axes = plt.subplots(2, len(marks), figsize=(3.4 * len(marks), 4.6),
                             squeeze=False)
    for j, mark in enumerate(marks):
        col = MARK_COLOR.get(mark, "0.2")
        t = table[(table["mark"] == mark) & table["s_lo"].notna()]
        ax, axd = axes[0, j], axes[1, j]

        def xy(set_, field):
            u = t[t["set"] == set_]
            return ((u["s_lo"] + u["s_hi"]).to_numpy() / 2,
                    u[field].to_numpy(float))
        x, y = xy("molecules", "r")
        ax.plot(x, y, "-o", color=NULL_GREY, ms=2.5, lw=0.9,
                label=SET_LABEL["molecules"])
        x, y = xy("contacts", "r")
        ax.plot(x, y, "-o", color=col, ms=2.5, lw=0.9, label="contacts")
        x, y = xy("contacts", "r_null")
        ax.plot(x, y, "--o", color=col, ms=2.5, lw=0.8, mfc="white",
                label="their stand-ins (control 1)")
        ax.axhline(0, color="0.7", lw=0.4)
        ax.set_title(f"{mark}: the two ends' r", loc="left", color=col)
        ax.set_ylabel("Pearson r of end posteriors")
        ax.legend(frameon=False, fontsize=5.5)
        for set_, ls in (("contacts", "-"), ("trans", ":")):
            u = t[t["set"] == set_]
            if u.empty:
                continue
            x = (u["s_lo"] + u["s_hi"]).to_numpy() / 2
            if "delta_lo" in u:
                axd.fill_between(x, u["delta_lo"], u["delta_hi"], color=col,
                                 alpha=0.15 if ls == "-" else 0.08, lw=0)
            axd.plot(x, u["delta"], ls, marker="o", color=col, ms=2.5, lw=0.9,
                     label=SET_LABEL[set_])
        tr = trends[(trends["mark"] == mark) & (trends["set"] == "contacts")]
        if len(tr):
            # the trend above the axes, clear of the curves whatever they do
            r0 = tr.iloc[0]
            txt = (f"s < {NEAR} bp: {r0['near']:+.3f}; s ≥ {FAR} bp: "
                   f"{r0['far']:+.3f}")
            if "difference_lo" in tr and np.isfinite(r0.get("difference_lo",
                                                            np.nan)):
                txt += (f"; difference {r0['difference']:+.3f} "
                        f"[{r0['difference_lo']:+.3f}, "
                        f"{r0['difference_hi']:+.3f}]")
            axd.text(0, 1.02, txt, transform=axd.transAxes, fontsize=5.3,
                     color="0.3", va="bottom")
        axd.axhline(0, color="0.7", lw=0.4)
        axd.set_title(f"{mark}: Δr, contacts − stand-ins", loc="left",
                      color=col, pad=11)
        axd.set_ylabel("Δr")
        axd.legend(frameon=False, fontsize=5.5, loc="upper right")
        for a in (ax, axd):
            a.set_xlim(0, S_EDGES[-1])
            a.set_xlabel("DNA between the two ends' sites along the "
                         "molecule, s (bp)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run_analyze(cells, args, merged, excl_chrom):
    post_dirs = {}
    for spec in args.posteriors:
        ctx, _, d = spec.partition("=")
        post_dirs[ctx.strip().upper()] = Path(d.strip())
    missing = [m for m in MARKS if m not in post_dirs]
    if missing:
        sys.exit(f"--posteriors has no {', '.join(missing)}")
    fit_paths = {m: default_fit(post_dirs[m]) for m in MARKS}
    for spec in args.mixture or []:
        ctx, _, f = spec.partition("=")
        fit_paths[ctx.strip().upper()] = Path(f.strip())
    absent = [str(p) for p in fit_paths.values() if not p.exists()]
    if absent:
        sys.exit("missing mixture fits: " + ", ".join(absent)
                 + " (give --mixture CTX=em_fit_all.json)")
    fits = {m: BetaBinomialMixture.from_json(str(p))
            for m, p in fit_paths.items()}
    _A.update(post_dirs=post_dirs, fits=fits, args=args, merged=merged,
              excl_chrom=excl_chrom)
    per_cell = {}
    jobs = [(c, i) for i, c in enumerate(cells)]
    for i, (cell, d) in enumerate(_map(_analyze_one, jobs, args.n_threads),
                                  1):
        per_cell[cell] = d
        if i % 100 == 0 or i == len(jobs):
            logger.info(f"  analyze: {i}/{len(jobs)} cells")
    absent = [c for c, d in per_cell.items() if d is None]
    if absent:
        logger.warning(f"{len(absent)} cells have no pairs table (run "
                       f"extract), e.g. {absent[0]}")
    table, trends = summarize(per_cell, args)
    out = Path(args.output_dir)
    table.to_csv(out / "junction_concordance.tsv", sep="\t", index=False,
                 float_format="%.5g")
    trends.to_csv(out / "junction_trend.tsv", sep="\t", index=False,
                  float_format="%.5g")
    plot(table, trends, out / f"junction_concordance.{args.plot_format}")
    show = trends[["mark", "set", "cells", "pairs", "delta_all", "near",
                   "far", "difference"]
                  + [c for c in ("difference_lo", "difference_hi", "p")
                     if c in trends]]
    logger.info("trend, s < %d bp against s >= %d bp (molecules: r, else "
                "Δr):\n%s", NEAR, FAR,
                show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return table, trends


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--samples", required=True,
                    help="Cells and their BAMs (--cell-column, --bam-column).")
    ap.add_argument("--cell-column", default="cell_name")
    ap.add_argument("--bam-column", default="bam_path")
    ap.add_argument("--steps", default="extract,analyze",
                    help="extract, analyze or both (default).")
    ap.add_argument("--output-dir", required=True)
    g = ap.add_argument_group("extract")
    g.add_argument("--context-bed", default=None,
                   help="GCH:PATH,HCG:PATH, as the pipeline's.")
    g.add_argument("--context-bed-offset", type=int, default=0)
    g.add_argument("--reference", default=None,
                   help="The genome (.2bit or indexed FASTA), for DpnII's "
                        "sites.")
    g.add_argument("--trim-ends", type=int, default=5)
    g.add_argument("--mapq-min", type=int, default=30)
    g.add_argument("--max-junction-gap", type=int, default=1_000,
                   help="A read's junction no further than this past its 3' "
                        "end (bp; default %(default)s, a library insert).")
    g.add_argument("--molecule-fraction", type=float, default=0.02,
                   help="The share of short read pairs read for the "
                        "unligated molecules (default %(default)s).")
    g.add_argument("--overwrite", action="store_true",
                   help="Read cells whose pairs table exists again.")
    g = ap.add_argument_group("analyze")
    g.add_argument("--posteriors", action="append", default=[],
                   metavar="CTX=DIR",
                   help="The published end posteriors, GCH and HCG.")
    g.add_argument("--mixture", action="append", default=None,
                   metavar="CTX=JSON",
                   help="A context's mixture fit (default: "
                        "03_mixture_model/em_fit_all.json beside its "
                        "04_posteriors).")
    g.add_argument("--min-distance", type=int, default=10_000,
                   help="Contacts this far apart or more (bp; default "
                        "%(default)s).")
    g.add_argument("--min-sites", type=int, default=1,
                   help="Sites per end, as the Fig. 5 runs' --min-sites-gch "
                        "and --min-sites-hcg (default %(default)s).")
    g.add_argument("--null-exclude", type=int, default=1_000,
                   help="Stand-ins near-copies refused within this (bp; "
                        "default %(default)s, control 1).")
    g.add_argument("--link-shuffle-bin", type=int, default=250_000)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--n-boot", type=int, default=1_000)
    g.add_argument("--min-pairs", type=int, default=200,
                   help="A bin's r from this many pairs or more.")
    g.add_argument("--plot-format", default="pdf")
    ap.add_argument("--exclude-chrom", default="chrM,chrX,chrY")
    ap.add_argument("--exclude-bed", action="append", default=None)
    ap.add_argument("--n-threads", type=int, default=4)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    if not steps <= {"extract", "analyze"}:
        sys.exit("--steps: extract, analyze or both")
    samples = pd.read_csv(manifest_text(args.samples), sep="\t", dtype=str)
    for col in (args.cell_column, args.bam_column):
        if col not in samples:
            sys.exit(f"--samples has no column {col!r}")
    cells = samples[args.cell_column].tolist()
    excl = [c for c in args.exclude_chrom.split(",") if c.strip()]
    merged, excl_chrom = ed.load_exclusions(args.exclude_bed or [], excl)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if "extract" in steps:
        if not args.context_bed or not args.reference:
            sys.exit("extract needs --context-bed and --reference")
        run_extract(cells, samples[args.bam_column].tolist(), args,
                    excl_chrom)
        # every cell's, so a resumed run's table covers the earlier runs too
        summ = extract_summary(cells, out / "pairs")
        if len(summ):
            summ.to_csv(out / "extract_cells.tsv", sep="\t", index=False,
                        float_format="%.4g")
            n = summ[["cis", "trans", "molecule"]].sum().to_dict()
            logger.info(f"extract: {n} read pairs over {len(summ)} cells")
    if "analyze" in steps:
        run_analyze(cells, args, merged, excl_chrom)
    return 0


if __name__ == "__main__":
    sys.exit(main())
