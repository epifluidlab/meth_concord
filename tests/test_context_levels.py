"""split_context_bed.py and context_levels.py: GCA against GCY.

A random chromosome; its GCH sites on both strands as seqkit locate writes
them (the trinucleotide, its strand in column 4).  'N' cells (neurons) carry
endogenous mCA: their GCA sites read methylated 0.6, their GCY sites 0.3;
'G' cells (glia) 0.3 at both."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pysam = pytest.importorskip("pysam")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


scb = _load("split_context_bed")
cl = _load("context_levels")
L = 30_000
READ = 100
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


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
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    seq = "".join(rng.choice(list("ACGT"), L))
    (out / "g.fa").write_text(">chr1\n" + "\n".join(
        seq[i:i + 60] for i in range(0, L, 60)) + "\n")
    pysam.faidx(str(out / "g.fa"))
    truth = {}
    lines = []
    for i in range(1, L - 1):
        a, b, c = seq[i - 1], seq[i], seq[i + 1]
        if a == "G" and b == "C" and c in "ACT":           # plus GCH
            truth[(i, "+")] = "GCA" if c == "A" else "GCY"
            lines.append((i - 1, i + 2, "+"))
        if b == "G" and c == "C" and a in "AGT":           # minus GCH
            truth[(i, "-")] = "GCA" if COMP[a] == "A" else "GCY"
            lines.append((i - 1, i + 2, "-"))
    lines.sort()
    with open(out / "GCH.bed", "w") as fh:
        for s, e, st in lines:
            fh.write(f"chr1\t{s}\t{e}\t{st}\n")
    pysam.tabix_index(str(out / "GCH.bed"), preset="bed", force=True)
    return seq, truth


def test_split_by_the_base_after_the_c(tmp_path):
    seq, truth = make_genome(tmp_path)
    n = scb.split_bed(tmp_path / "GCH.bed.gz", tmp_path / "g.fa", 1,
                      tmp_path / "GCH")
    got = {}
    for k in ("GCA", "GCY"):
        t = pd.read_csv(tmp_path / f"GCH_{k}.bed.gz", sep="\t", header=None)
        for s, st in zip(t[1], t[3]):
            got[(s + 1, st)] = k
        assert (tmp_path / f"GCH_{k}.bed.gz.tbi").exists()
    assert got == truth and n[""] == 0
    assert 0.25 < n["GCA"] / (n["GCA"] + n["GCY"]) < 0.42
    # the wrong offset: sites that are not a GpC on their strand
    n0 = scb.split_bed(tmp_path / "GCH.bed.gz", tmp_path / "g.fa", 0,
                       tmp_path / "wrong")
    assert n0[""] > 0.9 * sum(n0.values())


def _two_bit(seq: str, path: Path, name: str = "chr1"):
    """``seq`` as a one-sequence .2bit: no N or masked blocks; T, C, A, G
    two bits each, the first base highest."""
    import struct
    code = {"T": 0, "C": 1, "A": 2, "G": 3}
    packed = bytearray()
    for i in range(0, len(seq), 4):
        b = 0
        for ch in seq[i:i + 4].ljust(4, "T"):
            b = (b << 2) | code[ch]
        packed.append(b)
    index = struct.pack("<B", len(name)) + name.encode()
    offset = 16 + len(index) + 4
    path.write_bytes(struct.pack("<IIII", 0x1A412743, 0, 1, 0) + index
                     + struct.pack("<I", offset)
                     + struct.pack("<IIII", len(seq), 0, 0, 0)
                     + bytes(packed))


def test_the_genome_can_be_a_2bit(tmp_path):
    """--reference mm10.2bit splits as the FASTA does, into a folder made
    for the BEDs."""
    pytest.importorskip("py2bit")
    seq, truth = make_genome(tmp_path)
    _two_bit(seq, tmp_path / "g.2bit")
    out = tmp_path / "new" / "GCH"
    assert scb.main(["--bed", str(tmp_path / "GCH.bed.gz"), "--reference",
                     str(tmp_path / "g.2bit"), "--c-offset", "1",
                     "--out-prefix", str(out)]) == 0
    t = pd.read_csv(f"{out}_GCY.bed.gz", sep="\t", header=None)
    assert {(s + 1, st) for s, st in zip(t[1], t[3])} == \
        {site for site, k in truth.items() if k == "GCY"}


def _reads(seq, sites, meth, rng, n_reads=1500):
    """C->T reads: a plus-strand GCH C kept with its class's probability,
    every other C converted."""
    out = []
    for _ in range(n_reads):
        s = int(rng.integers(1, L - READ - 1))
        q = []
        for j in range(s, s + READ):
            b = seq[j]
            if b == "C":
                k = sites.get(j)
                b = "C" if k is not None and rng.random() < meth[k] else "T"
            q.append(b)
        out.append((s, "".join(q)))
    return out


def test_neurons_show_their_mca_at_gca(tmp_path):
    seq, truth = make_genome(tmp_path / "g")
    scb.split_bed(tmp_path / "g" / "GCH.bed.gz", tmp_path / "g" / "g.fa", 1,
                  tmp_path / "g" / "GCH")
    plus = {i: k for (i, st), k in truth.items() if st == "+"}
    rng = np.random.default_rng(1)
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": "chr1", "LN": L}]}
    rows = []
    for grp, meth in (("N", {"GCA": 0.6, "GCY": 0.3}),
                      ("G", {"GCA": 0.3, "GCY": 0.3})):
        for ci in range(4):
            name = f"{grp}{ci}"
            segs = []
            for r, (s, q) in enumerate(sorted(_reads(seq, plus, meth, rng))):
                seg = pysam.AlignedSegment()
                seg.query_name = f"r{r}"
                seg.query_sequence = q
                seg.flag = 0
                seg.reference_id = 0
                seg.reference_start = s
                seg.mapping_quality = 60
                seg.cigarstring = f"{READ}M"
                seg.query_qualities = pysam.qualitystring_to_array("I" * READ)
                seg.set_tag("MD", _md(q, seq[s:s + READ]))
                segs.append(seg)
            p = tmp_path / f"{name}.bam"
            with pysam.AlignmentFile(str(p), "wb", header=header) as fh:
                for seg in segs:
                    fh.write(seg)
            pysam.index(str(p))
            rows.append((name, str(p), grp))
    pd.DataFrame(rows, columns=["cell_name", "bam_path", "group"]).to_csv(
        tmp_path / "samples.tsv", sep="\t", index=False)
    o = tmp_path / "out"
    g = tmp_path / "g"
    cl.main(["--samples", str(tmp_path / "samples.tsv"),
             "--context-bed", f"GCA:{g / 'GCH_GCA.bed.gz'},"
                              f"GCY:{g / 'GCH_GCY.bed.gz'}",
             "--context-bed-offset", "1", "--chroms", "chr1",
             "--trim-ends", "0", "--n-threads", "2", "--plot-format", "png",
             "--output-dir", str(o)])
    t = pd.read_csv(o / "context_levels.tsv", sep="\t")
    d = (t["GCA_level"] - t["GCY_level"]).groupby(t["group"]).median()
    assert 0.25 < d["N"] < 0.35 and abs(d["G"]) < 0.05
    assert (t["GCY_level"].between(0.25, 0.35)).all()
    gt = pd.read_csv(o / "context_levels_groups.tsv", sep="\t").set_index(
        "group")
    assert gt.loc["N", "GCA_minus_GCY_lo"] > 0.2
    assert gt.loc["G", "GCA_minus_GCY_lo"] < 0 < gt.loc["G",
                                                         "GCA_minus_GCY_hi"]
    assert (o / "context_levels.png").exists()
