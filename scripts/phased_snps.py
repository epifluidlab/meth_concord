#!/usr/bin/env python3
"""Phased heterozygous SNVs on hg38 for allelic_imbalance.py: one sample's
biallelic SNVs where it is heterozygous and phased (0|1, 1|0), as a
bgzipped, tabix-indexed VCF.

1kgp -- NA12878 (GM12878) or any 1000 Genomes sample, from the
high-coverage phased panel on GRCh38 (Byrska-Bishop et al., Cell 2022).
The panel holds all 3,202 samples, ~32 GB over chr1-22 and X: each
chromosome is downloaded (resumable, md5 against the panel's manifest),
the sample's SNVs kept, and the download removed.  Autosomes were phased
with the pedigree (SHAPEIT2-duohmm), so a trio child's are phased by
transmission; chrX statistically (Eagle2), without it -- allelic_imbalance's
phase check (mates of a read pair on one haplotype) tells how well.

    python scripts/phased_snps.py 1kgp --sample NA12878 \\
        --output NA12878.het.vcf.gz --work-dir tmp_1kgp --n-jobs 4

syri -- IMR-90 (or BJ) from the haplotype-resolved assemblies of
Ranallo-Benavidez et al. (NAR 2026): SyRI's SNVs of each haplotype against
T2T-CHM13v2.0, merged by Jasmine (figshare 10.6084/m9.figshare.28924484).
The file has no genotypes: SUPP_VEC says which of the four haplotypes holds
a variant, in the order BJ hap1, BJ hap2, IMR-90 hap1, IMR-90 hap2 (BJ hap1
alone holds chrY, and its hap2 chrX; IMR-90 each holds one X).  A variant in
exactly one of the sample's two haplotypes is heterozygous, phased by the
assembly: 1|0 on hap1, 0|1 on hap2.  The SNVs are lifted to hg38 with
UCSC's hs1ToHg38 chain -- a position held by exactly one chain block, onto
the same chromosome, its bases complemented on a minus-strand chain -- and
checked against the hg38 FASTA: where hg38 carries the ALT base the
alleles and the genotype are swapped, where neither base, the SNV goes.

    python scripts/phased_snps.py syri --fasta hg38.fa \\
        --output IMR90.het.vcf.gz --work-dir tmp_imr90

(--haplotypes 1,2 for BJ, whose chrX is all on hap2 and so never
heterozygous.)  Both downloads are cached in --work-dir, and an interrupted
run picks up where it stopped.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger("phased_snps")

KGP_URL = ("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
           "1000G_2504_high_coverage/working/"
           "20220422_3202_phased_SNV_INDEL_SV")
KGP_MANIFEST = "20220804_manifest.txt"
KGP_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
SYRI_URL = "https://ndownloader.figshare.com/files/54155468"
SYRI_NAME = "BJ_IMR90_syri_SNVs_indels.vcf.gz"
SYRI_MD5 = "3ebadaa3d7495bac7f177b161cdd2b79"
CHAIN_URL = ("https://hgdownload.soe.ucsc.edu/goldenPath/hs1/liftOver/"
             "hs1ToHg38.over.chain.gz")
_COMP = str.maketrans("ACGT", "TGCA")


def kgp_file(chrom: str) -> str:
    """The panel's file of a chromosome (chrX's is the v2 release)."""
    v = ".v2" if chrom == "chrX" else ""
    return (f"1kGP_high_coverage_Illumina.{chrom}.filtered."
            f"SNV_INDEL_SV_phased_panel{v}.vcf.gz")


def fetch(url: str, dest: Path, md5: str | None = None, tries: int = 8,
          chunk: int = 1 << 22) -> Path:
    """``url`` to ``dest``, streamed, resuming a partial download (HTTP
    range) after a dropped connection, checked against ``md5``.  A local
    path is used where it is."""
    if not re.match(r"^(https?|ftp)://", str(url)):
        return Path(url)
    dest = Path(dest)
    if dest.exists() and (md5 is None or _md5(dest) == md5):
        logger.info(f"  cached {dest.name}")
        return dest
    part = dest.with_name(dest.name + ".part")
    for k in range(tries):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"}
                                     if have else {})
        try:
            with urllib.request.urlopen(req, timeout=300) as fh:
                if have and fh.status != 206:      # the range was ignored
                    have = 0
                with open(part, "ab" if have else "wb") as out:
                    while True:
                        b = fh.read(chunk)
                        if not b:
                            break
                        out.write(b)
            break
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:              # nothing left to fetch
                break
            if k == tries - 1:
                raise
            logger.warning(f"  {dest.name}: {e}; resuming ({k + 1})")
            time.sleep(min(60, 5 * 2 ** k))
        except Exception as e:                       # noqa: BLE001
            if k == tries - 1:
                raise
            logger.warning(f"  {dest.name}: {e}; resuming ({k + 1})")
            time.sleep(min(60, 5 * 2 ** k))
    if md5 is not None and _md5(part) != md5:
        part.unlink()
        raise SystemExit(f"{dest.name}: md5 differs from {md5}; removed, "
                         f"run again")
    part.replace(dest)
    return dest


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def body(records) -> str:
    """VCF lines of records (chrom, 1-based pos, id, ref, alt, gt)."""
    return "".join(f"{c}\t{p}\t{i}\t{r}\t{a}\t.\tPASS\t.\tGT\t{g}\n"
                   for c, p, i, r, a, g in records)


def write_vcf(parts, contigs: dict, sample: str, out: Path,
              source: str) -> Path:
    """VCF lines -- sorted; strings, or files holding them -- as ``out``
    (.vcf.gz), bgzipped and tabix-indexed."""
    import pysam
    out = Path(out)
    if not out.name.endswith(".vcf.gz"):
        raise SystemExit(f"--output must end in .vcf.gz: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    plain = out.with_name(out.name[:-3])
    with open(plain, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write(f"##source={source}\n")
        for c, n in contigs.items():
            fh.write(f"##contig=<ID={c},length={n}>\n")
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased '
                 'genotype">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                 f"{sample}\n")
        for part in parts:
            if isinstance(part, Path):
                with open(part) as src:
                    for line in src:
                        fh.write(line)
            else:
                fh.write(part)
    pysam.tabix_index(str(plain), preset="vcf", force=True)
    return out


# ------------------------------------------------------------------- 1kGP --


def het_snvs(path, sample: str):
    """One sample's phased heterozygous biallelic SNVs in a VCF: (records,
    {chrom: length}).  A position with more than one (a split
    multiallelic) is left out."""
    import pysam
    pysam.set_verbosity(0)          # read front to back: no index needed
    vf = pysam.VariantFile(str(path))
    if sample not in vf.header.samples:
        raise SystemExit(f"{sample} is not a sample of {path}")
    vf.subset_samples([sample])
    rows = []
    for rec in vf:
        if len(rec.alleles) != 2 or len(rec.ref) != 1 or \
                len(rec.alts[0]) != 1:
            continue
        ref, alt = rec.ref.upper(), rec.alts[0].upper()
        if ref not in "ACGT" or alt not in "ACGT":
            continue
        flt = list(rec.filter.keys())
        if flt and flt != ["PASS"]:
            continue
        s = rec.samples[0]
        gt = s["GT"]
        if gt not in ((0, 1), (1, 0)) or not s.phased:
            continue
        rows.append((rec.chrom, rec.pos, rec.id or ".", ref, alt,
                     f"{gt[0]}|{gt[1]}"))
    pos = [(r[0], r[1]) for r in rows]
    seen: dict = {}
    for k in pos:
        seen[k] = seen.get(k, 0) + 1
    rows = [r for r, k in zip(rows, pos) if seen[k] == 1]
    contigs = {c: vf.header.contigs[c].length for c in {r[0] for r in rows}
               if c in vf.header.contigs}
    vf.close()
    return rows, contigs


def _kgp_chrom(chrom, sample, base, work: Path, md5: dict, keep: bool):
    """One chromosome: download, keep the sample's SNVs (``work``/
    <sample>.<chrom>.vcf.part: VCF lines; its length beside it), remove
    the download.  Returns the SNVs, None when done before."""
    if not logging.getLogger().handlers:        # a worker process
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s")
    done = work / f"{sample}.{chrom}.vcf.part"
    if done.exists():
        return None
    name = kgp_file(chrom)
    src = (f"{base}/{name}" if re.match(r"^(https?|ftp)://", str(base))
           else str(Path(base) / name))
    logger.info(f"  {chrom}: {src}")
    path = fetch(src, work / name, md5.get(name))
    rows, contigs = het_snvs(path, sample)
    tmp = done.with_name(done.name + ".tmp")
    tmp.write_text(body(rows))
    (work / f"{sample}.{chrom}.length").write_text(
        str(contigs.get(chrom, 0)))
    tmp.replace(done)
    if not keep and Path(path).parent == work:
        Path(path).unlink(missing_ok=True)
    logger.info(f"  {chrom}: {len(rows):,} heterozygous SNVs")
    return len(rows)


def run_kgp(args) -> int:
    from joblib import Parallel, delayed
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    chroms = ([c.strip() for c in args.chroms.split(",") if c.strip()]
              if args.chroms else KGP_CHROMS)
    md5 = {}
    if re.match(r"^(https?|ftp)://", args.base_url):
        man = fetch(f"{args.base_url}/{KGP_MANIFEST}", work / KGP_MANIFEST)
        for line in man.read_text().splitlines():
            f = line.split()
            if len(f) == 2:
                md5[f[0]] = f[1]
    logger.info(f"{args.sample}: {len(chroms)} chromosomes of {args.base_url}")
    Parallel(n_jobs=args.n_jobs)(
        delayed(_kgp_chrom)(c, args.sample, args.base_url, work, md5,
                            args.keep_downloads) for c in chroms)
    parts = [work / f"{args.sample}.{c}.vcf.part" for c in chroms]
    contigs = {c: int((work / f"{args.sample}.{c}.length").read_text())
               for c in chroms}
    n = {c: sum(1 for _ in open(p_)) for c, p_ in zip(chroms, parts)}
    out = write_vcf(parts, contigs, args.sample, args.output,
                    f"phased_snps.py 1kgp {args.base_url} sample "
                    f"{args.sample}")
    logger.info(f"{out}: {sum(n.values()):,} phased heterozygous SNVs "
                f"({n.get('chrX', 0):,} on chrX)")
    return 0


# ------------------------------------------------------------------- SyRI --


def syri_het(path, haps=(3, 4), chroms=None):
    """The heterozygous SNVs of one sample in a Jasmine merge of SyRI calls
    (``haps``: its two haplotypes' 1-based places in SUPP_VEC): a frame of
    chrom, 1-based pos, ref, alt, and the haplotype with the ALT (1, 2).
    A position with more than one SNV record of the sample is left out.
    Also the counts of the sample's SNVs: heterozygous, homozygous."""
    import pandas as pd
    h1, h2 = haps[0] - 1, haps[1] - 1
    want = set(chroms) if chroms else None
    sv_re = re.compile(r"(?:^|;)SUPP_VEC=([01]+)")
    cols: dict = {"chrom": [], "pos": [], "ref": [], "alt": [], "hap": []}
    n_pos: dict = {}
    n_hom = 0
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t", 8)
            ref, alt = f[3], f[4]
            if len(ref) != 1 or len(alt) != 1:
                continue
            if want is not None and f[0] not in want:
                continue
            ref, alt = ref.upper(), alt.upper()
            if ref not in "ACGT" or alt not in "ACGT" or ref == alt:
                continue
            m = sv_re.search(f[7])
            if m is None:
                continue
            sv = m.group(1)
            b1, b2 = sv[h1] == "1", sv[h2] == "1"
            if not (b1 or b2):
                continue
            k = (f[0], int(f[1]))
            n_pos[k] = n_pos.get(k, 0) + 1
            if b1 and b2:
                n_hom += 1
                continue
            for k_, v_ in zip(cols, k + (ref, alt, 1 if b1 else 2)):
                cols[k_].append(v_)
    d = pd.DataFrame(cols)
    once = np.array([n_pos[k] == 1 for k in zip(d["chrom"], d["pos"])],
                    bool)
    return d[once].reset_index(drop=True), {
        "heterozygous": len(d), "homozygous": n_hom,
        "kept": int(once.sum())}


def read_chain(path) -> dict:
    """A UCSC chain file: {source chromosome: [(target chromosome, target
    size, target strand, block source starts, block target starts, block
    sizes), ...]}, 0-based, target starts on the chain's own strand."""
    out: dict = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    cur = None
    with opener(path, "rt") as fh:
        for line in fh:
            f = line.split()
            if not f:
                continue
            if f[0] == "chain":
                if cur:
                    _add_chain(out, cur)
                t, q = int(f[5]), int(f[10])
                cur = [f[2], f[7], int(f[8]), f[9], [], [], [], t, q]
                continue
            size = int(f[0])
            cur[4].append(cur[7])
            cur[5].append(cur[8])
            cur[6].append(size)
            if len(f) == 3:
                cur[7] += size + int(f[1])
                cur[8] += size + int(f[2])
    if cur:
        _add_chain(out, cur)
    return out


def _add_chain(out, cur):
    t_name, q_name, q_size, q_strand, ts, qs, sz = cur[:7]
    out.setdefault(t_name, []).append(
        (q_name, q_size, q_strand, np.asarray(ts, np.int64),
         np.asarray(qs, np.int64), np.asarray(sz, np.int64)))


def lift(chains: dict, chrom: str, pos0):
    """0-based positions of ``chrom`` through ``chains``: (target
    chromosome, 0-based target position, strand) of each held by exactly
    one chain block; None elsewhere, and the number of blocks that hold
    each."""
    pos0 = np.asarray(pos0, np.int64)
    n = pos0.size
    order = np.argsort(pos0, kind="stable")
    p = pos0[order]
    hits = np.zeros(n, np.int64)
    t_ch = np.empty(n, object)
    t_pos = np.full(n, -1, np.int64)
    t_str = np.empty(n, object)
    for q_name, q_size, q_strand, ts, qs, sz in chains.get(chrom, ()):
        lo, hi = np.searchsorted(p, [ts[0], ts[-1] + sz[-1]])
        if lo >= hi:
            continue
        pp = p[lo:hi]
        b = np.searchsorted(ts, pp, side="right") - 1
        inb = (b >= 0) & (pp < ts[np.maximum(b, 0)] + sz[np.maximum(b, 0)])
        j = lo + np.flatnonzero(inb)
        bi = b[inb]
        q = qs[bi] + (pp[inb] - ts[bi])
        if q_strand == "-":
            q = q_size - 1 - q
        hits[j] += 1
        t_ch[j] = q_name
        t_pos[j] = q
        t_str[j] = q_strand
    one = hits == 1
    res_ch = np.empty(n, object)
    res_pos = np.full(n, -1, np.int64)
    res_str = np.empty(n, object)
    res_ch[order[one]] = t_ch[one]
    res_pos[order[one]] = t_pos[one]
    res_str[order[one]] = t_str[one]
    h = np.empty(n, np.int64)
    h[order] = hits
    return res_ch, res_pos, res_str, h


def lift_snvs(d, chains: dict, fasta: str):
    """SNVs (a frame: chrom, 1-based pos, ref, alt, ALT haplotype) on the
    chain's
    source genome to records on its target: kept where exactly one block
    holds the SNV and it lands on the same chromosome; bases complemented
    on a minus-strand chain; checked against ``fasta`` -- the target's
    base the ALT: alleles swapped and the haplotype with them; neither:
    dropped -- and a target position reached twice dropped.  Returns
    (sorted records (chrom, pos, id, ref, alt, gt), {chrom: length},
    counts)."""
    import pysam
    fa = pysam.FastaFile(fasta)
    lens = dict(zip(fa.references, fa.lengths))
    n = {"snvs": len(d), "unmapped": 0, "several_blocks": 0,
         "other_chromosome": 0, "not_in_fasta": 0, "ref_mismatch": 0,
         "swapped": 0, "target_twice": 0}
    out = []
    for chrom, g in d.groupby("chrom", sort=False):
        tc, tp, ts, hits = lift(chains, chrom, g["pos"].to_numpy() - 1)
        n["unmapped"] += int((hits == 0).sum())
        n["several_blocks"] += int((hits > 1).sum())
        ok = hits == 1
        same = np.array([c == chrom for c in tc]) & ok
        n["other_chromosome"] += int((ok & ~same).sum())
        if chrom not in lens:
            n["not_in_fasta"] += int(same.sum())
            continue
        seq = np.frombuffer(fa.fetch(chrom).upper().encode("ascii"),
                            np.uint8)
        inside = same & (tp < seq.size)
        n["not_in_fasta"] += int((same & ~inside).sum())
        same = inside
        g = g[same].copy()
        g["tpos"] = tp[same]
        minus = ts[same] == "-"
        ref = g["ref"].to_numpy(str)
        alt = g["alt"].to_numpy(str)
        ref = np.where(minus, np.char.translate(ref, _COMP), ref)
        alt = np.where(minus, np.char.translate(alt, _COMP), alt)
        base = seq[g["tpos"].to_numpy()].tobytes().decode("ascii")
        base = np.array(list(base), dtype="<U1") if base else np.array([])
        keep_ref, swap = base == ref, base == alt
        n["ref_mismatch"] += int((~keep_ref & ~swap).sum())
        n["swapped"] += int(swap.sum())
        hap = g["hap"].to_numpy()
        m = keep_ref | swap
        new_ref = np.where(swap, alt, ref)[m]
        new_alt = np.where(swap, ref, alt)[m]
        # ALT on hap1 -> 1|0; swapped, hap1 carries the new REF
        alt_on_1 = (hap == 1) != swap
        gt = np.where(alt_on_1, "1|0", "0|1")[m]
        tpos = g["tpos"].to_numpy()[m] + 1
        u, cnt = np.unique(tpos, return_counts=True)
        once = np.isin(tpos, u[cnt == 1])
        n["target_twice"] += int((~once).sum())
        o = np.argsort(tpos[once], kind="stable")
        for p_, r_, a_, g_ in zip(tpos[once][o], new_ref[once][o],
                                  new_alt[once][o], gt[once][o]):
            out.append((chrom, int(p_), ".", r_, a_, g_))
    order = {c: i for i, c in enumerate(fa.references)}
    out.sort(key=lambda r: (order.get(r[0], 1 << 30), r[1]))
    contigs = {c: lens[c] for c in dict.fromkeys(r[0] for r in out)}
    fa.close()
    n["written"] = len(out)
    return out, contigs, n


def run_syri(args) -> int:
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    vcf = fetch(args.vcf, work / SYRI_NAME,
                SYRI_MD5 if args.vcf == SYRI_URL else None)
    chain = fetch(args.chain, work / Path(args.chain).name)
    haps = tuple(int(x) for x in args.haplotypes.split(","))
    chroms = ([c.strip() for c in args.chroms.split(",") if c.strip()]
              if args.chroms else None)
    logger.info(f"{args.sample}: SUPP_VEC places {haps} of {vcf}")
    snv, n0 = syri_het(vcf, haps, chroms)
    logger.info(f"  SNVs of the sample on CHM13: {n0['heterozygous']:,} "
                f"heterozygous, {n0['homozygous']:,} homozygous; "
                f"{n0['kept']:,} heterozygous at a position of their own")
    recs, contigs, n1 = lift_snvs(snv, read_chain(chain), args.fasta)
    logger.info("  liftover to the FASTA's genome: " + ", ".join(
        f"{k} {v:,}" for k, v in n1.items()))
    out = write_vcf([body(recs)], contigs, args.sample, args.output,
                    f"phased_snps.py syri {vcf} haplotypes {haps}, lifted "
                    f"with {chain}")
    by = {}
    for r in recs:
        k = (r[0] == "chrX", r[5])
        by[k] = by.get(k, 0) + 1
    logger.info(f"{out}: {len(recs):,} phased heterozygous SNVs; chrX "
                f"1|0 {by.get((True, '1|0'), 0):,}, 0|1 "
                f"{by.get((True, '0|1'), 0):,}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("1kgp", help="A sample of the 1000 Genomes "
                                    "high-coverage phased panel (GRCh38).")
    k.add_argument("--sample", default="NA12878")
    k.add_argument("--output", required=True, help="<name>.vcf.gz")
    k.add_argument("--work-dir", required=True,
                   help="Downloads and per-chromosome results; a rerun "
                        "resumes from them.")
    k.add_argument("--chroms", default=None,
                   help="Comma-separated (default chr1-22, chrX).")
    k.add_argument("--base-url", default=KGP_URL,
                   help="The panel's directory (a local one too).")
    k.add_argument("--n-jobs", type=int, default=4,
                   help="Chromosomes at once (each downloads 0.4-2.5 GB).")
    k.add_argument("--keep-downloads", action="store_true")
    s = sub.add_parser("syri", help="A sample of the BJ / IMR-90 SyRI "
                                    "calls, CHM13 -> hg38.")
    s.add_argument("--fasta", required=True,
                   help="The hg38 FASTA the reads were aligned to (indexed, "
                        "or writable for its .fai).")
    s.add_argument("--output", required=True, help="<name>.vcf.gz")
    s.add_argument("--work-dir", required=True)
    s.add_argument("--sample", default="IMR90")
    s.add_argument("--haplotypes", default="3,4",
                   help="The sample's hap1,hap2 places in SUPP_VEC "
                        "(1-based): IMR-90 3,4 (default), BJ 1,2.")
    s.add_argument("--vcf", default=SYRI_URL,
                   help="The merged SyRI VCF (default: figshare).")
    s.add_argument("--chain", default=CHAIN_URL,
                   help="CHM13v2.0 (hs1) -> hg38 chain (default: UCSC).")
    s.add_argument("--chroms", default=None,
                   help="Source chromosomes, comma-separated (default all).")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    return run_kgp(args) if args.cmd == "1kgp" else run_syri(args)


if __name__ == "__main__":
    sys.exit(main())
