"""junction_concordance.py: concordance against the DNA between a contact's
two ends at labelling time.

A random genome with its GCH and HCG sites (the C at the BED start, as the
seqkit BEDs) and DpnII's; bisulfite reads of the C->T strand, each GCH and
HCG cytosine kept as C when labelled or methylated.  Contacts are read
pairs pointing at their junctions: read 1 forward, d1 bp short of a GATC;
read 2 reverse, d2 bp past one.  The pairs must be the pipeline's -- its
midpoints, its k and n -- and the junctions those planted; and on synthetic
cells a shared labelling that fades with the DNA between the ends must
show as a trend, concordance shared by the two loci must not."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pysam = pytest.importorskip("pysam")

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "junction_concordance", _ROOT / "scripts" / "junction_concordance.py")
jc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(jc)
sys.path.insert(0, str(_ROOT))
from nome_hic_concordance import bam_parser as bp  # noqa: E402
from nome_hic_concordance import reference as ref_  # noqa: E402
from nome_hic_concordance.mixture_model import BetaBinomialMixture  # noqa: E402

READ = 100
CHROMS = {"chr1": 400_000, "chr2": 200_000}
#: The fit: closed ends labelled at 0.25, open at 0.75.
FIT = {"pi": 0.5, "alpha_closed": 2.5, "beta_closed": 7.5,
       "alpha_accessible": 7.5, "beta_accessible": 2.5}


def _md(read_seq: str, ref_seq: str) -> str:
    out, run = [], 0
    for q, r in zip(read_seq, ref_seq):
        if q == r:
            run += 1
        else:
            out.append(f"{run}{r}")
            run = 0
    out.append(str(run))
    return "".join(out)


def make_genome(out: Path, seed=0):
    """Random chromosomes, their GCH and HCG BEDs (BED6, the C at the
    start) and the FASTA; returns {chrom: seq} and {mark: {chrom: set of
    plus-strand site positions}}."""
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    seqs = {c: "".join(rng.choice(list("ACGT"), n)) for c, n in CHROMS.items()}
    with open(out / "g.fa", "w") as fh:
        for c, s in seqs.items():
            fh.write(f">{c}\n" + "\n".join(s[i:i + 60]
                                           for i in range(0, len(s), 60)) + "\n")
    pysam.faidx(str(out / "g.fa"))
    plus = {"GCH": {}, "HCG": {}}
    for mark in ("GCH", "HCG"):
        lines = []
        for c, s in seqs.items():
            plus[mark][c] = set()
            for i in range(1, len(s) - 1):
                a, b, d = s[i - 1], s[i], s[i + 1]
                if mark == "GCH":
                    fwd = a == "G" and b == "C" and d in "ACT"
                    rev = b == "G" and d == "C" and a in "AGT"
                else:
                    fwd = b == "C" and d == "G" and a in "ACT"
                    rev = b == "G" and a == "C" and d in "AGT"
                if fwd:
                    lines.append((c, i, "+"))
                    plus[mark][c].add(i)
                if rev:
                    lines.append((c, i, "-"))
        lines.sort()
        p = out / f"{mark}.bed"
        with open(p, "w") as fh:
            for c, i, st in lines:
                fh.write(f"{c}\t{i}\t{i + 1}\tx\t0\t{st}\n")
        pysam.tabix_index(str(p), preset="bed", force=True)
    return seqs, plus


def _read_seq(ref_seq, start, plus, chrom, label, rng):
    """A C->T read of ref_seq[start:start+READ]: a GCH or HCG C kept with
    its probability in ``label``, every other C converted."""
    q = []
    for p in range(start, start + READ):
        b = ref_seq[p]
        if b == "C":
            keep = 0.0
            for mark, pr in label.items():
                if p in plus[mark][chrom]:
                    keep = pr
            b = "C" if rng.random() < keep else "T"
        q.append(b)
    return "".join(q)


def _segment(name, chrom_id, start, seq, ref_seq, flag, mate_id, mate_start,
             cigar=None, header=None):
    s = pysam.AlignedSegment(header) if header else pysam.AlignedSegment()
    s.query_name = name
    s.query_sequence = seq
    s.flag = flag
    s.reference_id = chrom_id
    s.reference_start = start
    s.mapping_quality = 60
    s.cigarstring = cigar or f"{len(seq)}M"
    s.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
    s.next_reference_id = mate_id
    s.next_reference_start = mate_start
    s.set_tag("MD", _md(seq, ref_seq[start:start + len(seq)]))
    return s


def _gatc(seq):
    return jc.dpnii_sites(seq)


def plant_pairs(seqs, plus, rng, n_cis=60, n_trans=20, n_mol=80,
                label=None):
    """Read pairs: contacts (cis >= 20 kb and trans) pointing at planted
    junctions, and inward unligated molecules; returns (segments, truth)
    with truth {qname: (kind, j1, j2)} in the pipeline's end order."""
    label = label or {"GCH": 0.5, "HCG": 0.7}
    names = list(seqs)
    gatc = {c: _gatc(s) for c, s in seqs.items()}
    segs, truth = [], {}

    def forward_end(c):
        """(read start, junction middle): a forward read d bp short of a
        GATC with no other within the read's reach."""
        g = gatc[c]
        while True:
            i = int(rng.integers(5, len(g) - 5))
            room = g[i] - g[i - 1] - 4 - jc.OVERLAP - 1
            if room > 10:
                d = int(rng.integers(0, min(room, 400)))
                end = g[i] - d
                if end - READ > 0:
                    return end - READ, int(g[i]) + 2

    def reverse_end(c):
        g = gatc[c]
        while True:
            i = int(rng.integers(5, len(g) - 5))
            room = g[i + 1] - g[i] - 4 - jc.OVERLAP - 1
            if room > 10:
                d = int(rng.integers(0, min(room, 400)))
                start = g[i] + 4 + d
                if start + READ < len(seqs[c]):
                    return start, int(g[i]) + 2

    k = 0
    for kind, n in (("cis", n_cis), ("trans", n_trans)):
        for _ in range(n):
            c1 = names[0]
            c2 = names[0] if kind == "cis" else names[1]
            while True:
                s1, j1 = forward_end(c1)
                s2, j2 = reverse_end(c2)
                if kind == "trans" or abs(s2 - s1) >= 20_000:
                    break
            q = f"p{k}"
            k += 1
            i1, i2 = names.index(c1), names.index(c2)
            q1 = _read_seq(seqs[c1], s1, plus, c1, label, rng)
            q2 = _read_seq(seqs[c2], s2, plus, c2, label, rng)
            segs.append(_segment(q, i1, s1, q1, seqs[c1], 0x1 | 0x40 | 0x20,
                                 i2, s2))
            segs.append(_segment(q, i2, s2, q2, seqs[c2], 0x1 | 0x80 | 0x10,
                                 i1, s1))
            first = (s1 <= s2) if kind == "cis" else True
            truth[q] = (kind, j1, j2) if first else (kind, j2, j1)
    for _ in range(n_mol):
        c = names[0]
        s1 = int(rng.integers(1, len(seqs[c]) - 3 * READ - 700))
        s2 = s1 + READ + int(rng.integers(0, 600))
        q = f"m{k}"
        k += 1
        segs.append(_segment(q, 0, s1, _read_seq(seqs[c], s1, plus, c, label,
                                                 rng), seqs[c],
                             0x1 | 0x40 | 0x20, 0, s2))
        segs.append(_segment(q, 0, s2, _read_seq(seqs[c], s2, plus, c, label,
                                                 rng), seqs[c],
                             0x1 | 0x80 | 0x10, 0, s1))
        truth[q] = ("molecule", -1, -1)
    return segs, truth


def write_bam(segs, path: Path):
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": c, "LN": n} for c, n in CHROMS.items()]}
    segs = sorted(segs, key=lambda s: (s.reference_id, s.reference_start))
    with pysam.AlignmentFile(str(path), "wb", header=header) as fh:
        for s in segs:
            fh.write(s)
    pysam.index(str(path))


def _sites(g: Path):
    raw = ref_.load_context_from_bed({m: str(g / f"{m}.bed.gz")
                                      for m in jc.MARKS}, n_threads=1)
    return raw, {m: {c: (np.ascontiguousarray(a["pos"], np.int64),
                         np.ascontiguousarray(a["ref_base"], np.uint8))
                     for c, a in raw[m].items()} for m in jc.MARKS}


def _args(**kw):
    import argparse
    d = dict(trim_ends=5, mapq_min=30, max_junction_gap=1_000,
             molecule_fraction=1.0)
    d.update(kw)
    return argparse.Namespace(**d)


def test_the_junction_is_the_first_dpnii_site_in_the_read_s_3_prime_direction():
    s = jc.dpnii_sites("AAAAGATCAAAAAAAAAAGATCAAAAAAAAAAAAAAAAAAGATC")
    assert s.tolist() == [4, 18, 40]
    # forward: at or past the end; the read may run into the site, ending
    # up to OVERLAP bases past its start (GATCGATC across the junction)
    assert jc.junction(s, 0, 13, False, 1000) == 20
    assert jc.junction(s, 0, 12, False, 1000) == 6
    assert jc.junction(s, 0, 21, False, 1000) == 20
    assert jc.junction(s, 0, 30, False, 1000) == 42
    # reverse: at or before the start
    assert jc.junction(s, 30, 60, True, 1000) == 20
    assert jc.junction(s, 20, 60, True, 1000) == 20
    # none within reach
    assert jc.junction(s, 0, 13, False, 3) == -1
    assert jc.junction(s, 0, 50, False, 1000) == -1
    assert jc.junction(jc.dpnii_sites("A" * 16 + "GATC"), 5, 30, True,
                       1000) == -1


def test_a_split_read_s_junction_is_its_clipped_3_prime_end():
    seq = "ACGT" * 100
    gatc = {"chr1": np.array([300])}
    sites = {m: {} for m in jc.MARKS}
    h = pysam.AlignmentHeader.from_dict(
        {"SQ": [{"SN": c, "LN": n} for c, n in CHROMS.items()]})
    q = seq[100:180] + "T" * 20
    fwd = _segment("r", 0, 100, q, seq, 0, -1, -1, cigar="80M20S", header=h)
    fwd.set_tag("MD", "80")
    fwd.set_tag("SA", "chr1,5000,+,80S20M,60,0;")
    rev = _segment("r", 0, 200, "T" * 20 + seq[200:280], seq, 0x10, -1, -1,
                   cigar="20S80M", header=h)
    rev.set_tag("MD", "80")
    rev.set_tag("SA", "chr1,5000,-,20M80S,60,0;")
    assert jc.end_record(fwd, sites, gatc, 5, 1000)[5] == 180
    assert jc.end_record(rev, sites, gatc, 5, 1000)[5] == 200
    # clipped at its 5' end only: the DpnII site past its 3' end
    five = _segment("r", 0, 100, "T" * 20 + seq[120:200], seq, 0, -1, -1,
                    cigar="20S80M", header=h)
    five.set_tag("MD", "80")
    five.set_tag("SA", "chr1,5000,+,20M80S,60,0;")
    assert jc.end_record(five, sites, gatc, 5, 1000)[5] == 302
    # no SA: the site, whatever the clip
    fwd.set_tag("SA", None)
    assert jc.end_record(fwd, sites, gatc, 5, 1000)[5] == 302


def test_sites_are_counted_as_the_pipeline_counts_them(tmp_path):
    seqs, plus = make_genome(tmp_path / "g")
    raw, sites = _sites(tmp_path / "g")
    rng = np.random.default_rng(3)
    seq = seqs["chr1"]
    for trial in range(60):
        s = int(rng.integers(10, 50_000))
        q = _read_seq(seq, s, plus, "chr1", {"GCH": 0.5, "HCG": 0.7}, rng)
        if trial % 3 == 0:
            # an insertion: the pipeline's slow path
            q = q[:40] + "AA" + q[40:READ - 2]
            seg = _segment("r", 0, s, q, seq, 0, -1, -1,
                           cigar="40M2I58M")
            seg.set_tag("MD", _md(q[:40] + q[42:], seq[s:s + 98]))
        else:
            seg = _segment("r", 0, s, q, seq, 0, -1, -1)
        strand = bp._infer_conversion_strand(seg)
        for m in jc.MARKS:
            arr = raw[m]["chr1"]
            k0, n0 = bp._count_context_in_read(seg, arr["pos"],
                                               arr["ref_base"], 5)
            pos, rb = sites[m]["chr1"]
            k, n, total = jc.read_sites(seg, strand, pos, rb, 5)
            assert (k, n) == (k0, n0)
            if n:
                informative = [p for p in sorted(plus[m]["chr1"])
                               if s + 5 <= p < seg.reference_end - 5
                               and (trial % 3 or not 40 + s <= p < 42 + s)]
                assert total / n == pytest.approx(
                    np.mean(informative), abs=60)


def test_extract_keeps_the_pipeline_s_pairs_and_the_planted_junctions(
        tmp_path):
    seqs, plus = make_genome(tmp_path / "g")
    raw, sites = _sites(tmp_path / "g")
    segs, truth = plant_pairs(seqs, plus, np.random.default_rng(5))
    bam = tmp_path / "c.bam"
    write_bam(segs, bam)
    gatc = {c: _gatc(s) for c, s in seqs.items()}
    t = jc.extract_cell(bam, sites, gatc, set(CHROMS), _args())
    kinds = t["kind"].value_counts()
    assert kinds["cis"] == 60 and kinds["trans"] == 20
    # every non-overlapping inward molecule, and only those
    n_mol = sum(1 for q, v in truth.items() if v[0] == "molecule")
    assert 0 < kinds["molecule"] <= n_mol
    mol = t[t["kind"] == "molecule"]
    assert (mol["end1"] <= mol["start2"]).all() and (~mol["rev1"]).all()
    # the pipeline's own pairs: its midpoints, its k and n
    for m in jc.MARKS:
        ref = bp._parse_cell_bam_python(str(bam), raw[m], trim_ends=5,
                                        mapq_min=30)
        ref = ref[ref["distance"] >= jc.CIS_MIN]
        cis = t[t["kind"] == "cis"]
        got = cis[["pos1", "pos2", f"{m}_k1", f"{m}_n1", f"{m}_k2",
                   f"{m}_n2"]].to_numpy()
        want = ref[["pos1", "pos2", "k1", "n1", "k2", "n2"]].to_numpy()
        assert sorted(map(tuple, got)) == sorted(map(tuple, want))
    # the junctions planted, in the pipeline's end order
    lig = t[t["kind"] != "molecule"]
    planted = sorted((v[1], v[2]) for v in truth.values() if v[0] != "molecule")
    assert sorted(zip(lig["j1"], lig["j2"])) == planted


def _frame(rows):
    return pd.DataFrame(rows, columns=["chrom", "pos1", "pos2", "k1", "n1",
                                       "k2", "n2", "distance"])


def synthetic_cell(rng, artifact: bool, n_pool=30_000, n_contacts=4_000):
    """One cell: its published-like frame (all pairs, posteriors from k and
    n) and its pairs table (contacts with junctions; s = d1 + d2).  Native:
    a contact's two ends share an open or closed lean, whatever s.
    Artifact: the ends are independent, but labelled with a shared shift
    that fades with s."""
    fit = BetaBinomialMixture.from_dict(FIT)
    L = 40_000_000

    def ends(z, n, shift=0.0):
        p = np.clip(np.where(z, 0.75, 0.25) + shift, 0.02, 0.98)
        return rng.binomial(n, p)

    # the pool: every other read pair of the cell, unconcordant
    d = np.exp(rng.uniform(np.log(1_000), np.log(5e6), n_pool)).astype(int)
    p1 = rng.integers(0, L - 6_000_000, n_pool)
    n1, n2 = 1 + rng.poisson(3, n_pool), 1 + rng.poisson(3, n_pool)
    pool = _frame({"chrom": "chr1", "pos1": p1, "pos2": p1 + d,
                   "k1": ends(rng.random(n_pool) < 0.5, n1), "n1": n1,
                   "k2": ends(rng.random(n_pool) < 0.5, n2), "n2": n2,
                   "distance": d})
    # the contacts
    n = n_contacts
    d = np.exp(rng.uniform(np.log(10_000), np.log(5e6), n)).astype(int)
    p1 = rng.integers(0, L - 6_000_000, n)
    d1, d2 = rng.uniform(0, 300, n), rng.uniform(0, 300, n)
    s = d1 + d2
    n1, n2 = 1 + rng.poisson(4, n), 1 + rng.poisson(4, n)
    if artifact:
        z1, z2 = rng.random(n) < 0.5, rng.random(n) < 0.5
        shift = rng.choice([-1.0, 1.0], n) * 0.3 * np.exp(-s / 100)
        k1, k2 = ends(z1, n1, shift), ends(z2, n2, shift)
    else:
        lean = rng.random(n) < 0.5
        z1 = rng.random(n) < np.where(lean, 0.85, 0.15)
        z2 = rng.random(n) < np.where(lean, 0.85, 0.15)
        k1, k2 = ends(z1, n1), ends(z2, n2)
    contacts = _frame({"chrom": "chr1", "pos1": p1, "pos2": p1 + d,
                       "k1": k1, "n1": n1, "k2": k2, "n2": n2, "distance": d})
    full = pd.concat([pool, contacts], ignore_index=True)
    full["posterior1"] = fit.predict_proba(full["k1"], full["n1"])
    full["posterior2"] = fit.predict_proba(full["k2"], full["n2"])
    t = pd.DataFrame({
        "kind": pd.Categorical(["cis"] * n, categories=["cis", "trans",
                                                        "molecule"]),
        "chrom1": "chr1", "pos1": p1, "chrom2": "chr1", "pos2": p1 + d,
        "distance": d, "start1": p1 - 50, "end1": p1 + 50,
        "start2": p1 + d - 50, "end2": p1 + d + 50,
        "rev1": False, "rev2": True, "sa1": False, "sa2": False,
        "j1": (p1 + d1).astype(np.int64), "j2": (p1 + d - d2).astype(np.int64),
        "GCH_k1": k1, "GCH_n1": n1, "GCH_c1": p1.astype(float),
        "GCH_k2": k2, "GCH_n2": n2, "GCH_c2": (p1 + d).astype(float)})
    for e in (1, 2):
        t[f"HCG_k{e}"] = 0
        t[f"HCG_n{e}"] = 0
        t[f"HCG_c{e}"] = np.nan
    # the junctions, rounded down: s as planted to within 2 bp
    return full, t


def _analyze_synthetic(tmp_path: Path, artifact: bool, n_cells=8):
    import argparse
    rng = np.random.default_rng(11 if artifact else 12)
    post = tmp_path / "context_GCH" / "04_posteriors"
    post.mkdir(parents=True)
    args = argparse.Namespace(min_sites=1, min_distance=10_000,
                              null_exclude=1_000, link_shuffle_bin=250_000,
                              seed=42, n_boot=300, min_pairs=200)
    fits = {"GCH": BetaBinomialMixture.from_dict(FIT),
            "HCG": BetaBinomialMixture.from_dict(FIT)}
    per_cell = {}
    for i in range(n_cells):
        full, t = synthetic_cell(rng, artifact)
        full.to_parquet(post / f"c{i}.posteriors.parquet", index=False)
        per_cell[f"c{i}"] = jc.cell_stats(
            f"c{i}", i, t, {"GCH": post, "HCG": tmp_path / "none"}, fits,
            args, {}, set())
    return jc.summarize(per_cell, args)


def test_a_labelling_artifact_fades_with_s_and_native_concordance_does_not(
        tmp_path):
    table, trends = _analyze_synthetic(tmp_path / "native", artifact=False)
    tr = trends.set_index(["mark", "set"]).loc[("GCH", "contacts")]
    # concordant against the stand-ins, and as much near as far
    assert tr["delta_all"] > 0.1
    assert tr["difference_lo"] < 0 < tr["difference_hi"]
    assert 0.7 < tr["ratio"] < 1.4
    table2, trends2 = _analyze_synthetic(tmp_path / "artifact", artifact=True)
    tr2 = trends2.set_index(["mark", "set"]).loc[("GCH", "contacts")]
    assert tr2["difference_lo"] > 0 and tr2["near"] > 3 * abs(tr2["far"])
    # the bins: the stand-ins found, and the near bins above the far
    g = table2[(table2["mark"] == "GCH") & (table2["set"] == "contacts")
               & table2["s_lo"].notna()]
    assert g["pairs"].sum() > 0.8 * 8 * 4_000
    g = g.dropna(subset=["delta"])          # s stops at 600 bp here
    assert g["delta"].iloc[0] > g["delta"].iloc[-1]
    # no HCG for these cells: no HCG rows
    assert not (trends2["mark"] == "HCG").any()


def test_the_command_reads_bams_and_writes_the_tables(tmp_path):
    """extract then analyze on three cells, the published posteriors made
    by the pipeline's own parse and the fit beside them."""
    seqs, plus = make_genome(tmp_path / "g")
    raw, _ = _sites(tmp_path / "g")
    fit = BetaBinomialMixture.from_dict(FIT)
    rows = []
    for i in range(3):
        segs, _ = plant_pairs(seqs, plus, np.random.default_rng(20 + i),
                              n_cis=150, n_trans=40, n_mol=400)
        bam = tmp_path / f"c{i}.bam"
        write_bam(segs, bam)
        rows.append((f"c{i}", str(bam)))
        for m in jc.MARKS:
            d = tmp_path / "final" / f"context_{m}"
            (d / "04_posteriors").mkdir(parents=True, exist_ok=True)
            (d / "03_mixture_model").mkdir(exist_ok=True)
            (d / "03_mixture_model" / "em_fit_all.json").write_text(
                json.dumps(FIT))
            f = bp._parse_cell_bam_python(str(bam), raw[m], 5, 30)
            f["posterior1"] = fit.predict_proba(f["k1"], f["n1"])
            f["posterior2"] = fit.predict_proba(f["k2"], f["n2"])
            f.to_parquet(d / "04_posteriors" / f"c{i}.posteriors.parquet",
                         index=False)
    pd.DataFrame(rows, columns=["cell_name", "bam_path"]).to_csv(
        tmp_path / "samples.tsv", sep="\t", index=False)
    g, out = tmp_path / "g", tmp_path / "out"
    argv = ["--samples", str(tmp_path / "samples.tsv"),
            "--context-bed", f"GCH:{g / 'GCH.bed.gz'},HCG:{g / 'HCG.bed.gz'}",
            "--reference", str(g / "g.fa"),
            "--posteriors",
            f"GCH={tmp_path / 'final/context_GCH/04_posteriors'}",
            "--posteriors",
            f"HCG={tmp_path / 'final/context_HCG/04_posteriors'}",
            "--molecule-fraction", "1", "--min-pairs", "5", "--n-boot", "50",
            "--exclude-chrom", "chrM", "--n-threads", "1",
            "--plot-format", "png", "--output-dir", str(out)]
    assert jc.main(argv) == 0
    for f in ("junction_concordance.tsv", "junction_trend.tsv",
              "junction_concordance.png", "extract_cells.tsv"):
        assert (out / f).exists(), f
    assert len(list((out / "pairs").glob("*.parquet"))) == 3
    summ = pd.read_csv(out / "extract_cells.tsv", sep="\t")
    assert (summ["cis"] == 150).all() and (summ["trans"] == 40).all()
    assert (summ["junction_both"] == 1).all()
    tab = pd.read_csv(out / "junction_concordance.tsv", sep="\t")
    assert set(tab["set"]) == set(jc.SETS)
    # a resumed run reads only the cell it lacks, and its summary still
    # lists every cell
    (out / "pairs" / "c1.parquet").unlink()
    (out / "extract_cells.tsv").unlink()
    assert jc.main(argv + ["--steps", "extract"]) == 0
    again = pd.read_csv(out / "extract_cells.tsv", sep="\t")
    assert sorted(again["cell"]) == ["c0", "c1", "c2"]
    assert (again["cis"] == 150).all()
