"""phased_snps.py: phased heterozygous SNVs from the 1000 Genomes panel and
from SyRI calls of haplotype-resolved assemblies, lifted over a chain."""

from __future__ import annotations

import hashlib
import http.server
import importlib.util
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pysam = pytest.importorskip("pysam")

_SPEC = importlib.util.spec_from_file_location(
    "phased_snps",
    Path(__file__).resolve().parents[1] / "scripts" / "phased_snps.py")
ps = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ps)
_AI = importlib.util.spec_from_file_location(
    "allelic_imbalance",
    Path(__file__).resolve().parents[1] / "scripts" / "allelic_imbalance.py")
ai = importlib.util.module_from_spec(_AI)
_AI.loader.exec_module(ai)

# chrA 0-100 -> 50-150 (+), a gap, 110-300 -> 170-360 (+); chrB 0-100 on
# the minus strand of a 500 bp target; chrA 200-250 also -> chrC (two
# chains hold it)
CHAIN = """chain 1000 chrA 1000 + 0 300 chrA 1200 + 50 360 1
100\t10\t20
190

chain 900 chrB 400 + 0 100 chrB 500 - 0 100 2
100

chain 10 chrA 1000 + 200 250 chrC 300 + 0 50 3
50
"""


def _chain(tmp_path):
    p = tmp_path / "x.chain"
    p.write_text(CHAIN)
    return ps.read_chain(p)


def test_chain_lift_both_strands_and_overlaps(tmp_path):
    ch = _chain(tmp_path)
    tc, tp, ts, hits = ps.lift(ch, "chrA", [10, 105, 120, 220, 299, 300])
    assert list(hits) == [1, 0, 1, 2, 1, 0]
    assert list(tc[[0, 2, 4]]) == ["chrA"] * 3
    assert list(tp[[0, 2, 4]]) == [60, 180, 359]
    assert tc[1] is None and tc[3] is None and tp[3] == -1
    tc, tp, ts, hits = ps.lift(ch, "chrB", [0, 10, 99])
    assert list(tp) == [499, 489, 400] and set(ts) == {"-"}
    assert ps.lift(ch, "chrZ", [5])[3][0] == 0


def _syri(tmp_path):
    rows = [
        # pos, id, ref, alt, SUPP_VEC (BJ1 BJ2 IMR1 IMR2)
        (10, "2_SNP1", "a", "G", "0010"),     # IMR hap1
        (20, "3_SNP2", "C", "T", "0001"),     # IMR hap2
        (30, "2_SNP3", "G", "A", "0011"),     # IMR homozygous
        (40, "0_SNP4", "T", "C", "1000"),     # BJ only
        (50, "2_DEL5", "AC", "A", "0010"),    # an indel
        (60, "2_SNP6", "A", "C", "0010"),     # two records at 60
        (60, "3_SNP7", "A", "G", "0001"),
        (70, "1_SNP8", "G", "C", "1110"),     # BJ1, BJ2, IMR hap1
    ]
    lines = ["##fileformat=VCFv4.3", "##contig=<ID=chrA,length=1000>",
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"]
    for p_, i_, r_, a_, v_ in rows:
        lines.append(f"chrA\t{p_}\t{i_}\t{r_}\t{a_}\t.\tPASS\t"
                     f"ChrB=chrA_hap1;SUPP_VEC_EXT=1111;SUPP_VEC={v_};SUPP=1")
    lines.append("chrA\t5\t0_SYN1\tN\t<SYN>\t.\tPASS\tEND=900;SUPP_VEC=1111")
    p = tmp_path / "syri.vcf"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_syri_heterozygous_snvs_from_supp_vec(tmp_path):
    d, n = ps.syri_het(_syri(tmp_path))
    assert list(zip(d["pos"], d["ref"], d["alt"], d["hap"])) == [
        (10, "A", "G", 1), (20, "C", "T", 2), (70, "G", "C", 1)]
    assert n == {"heterozygous": 5, "homozygous": 1, "kept": 3}
    bj, _ = ps.syri_het(_syri(tmp_path), haps=(1, 2))
    assert list(zip(bj["pos"], bj["hap"])) == [(40, 1)]


def _fasta(tmp_path, seqs: dict):
    p = tmp_path / "t.fa"
    p.write_text("".join(f">{c}\n{s}\n" for c, s in seqs.items()))
    pysam.faidx(str(p))
    return str(p)


def test_lift_checks_the_target_base_and_swaps(tmp_path):
    ch = _chain(tmp_path)
    a = np.array(list("N" * 1200))
    # chrA 10 -> 60: target G (= REF); 20 -> 70: target T (= ALT, swap);
    # 30 -> 80: target C (neither)
    a[[60, 70, 80]] = list("GTC")
    b = np.array(list("N" * 500))
    # chrB 10 -> 489 on the minus strand: REF A / ALT C read as T / G
    b[489] = "T"
    fa = _fasta(tmp_path, {"chrA": "".join(a), "chrB": "".join(b),
                           "chrC": "N" * 300})
    d = pd.DataFrame({"chrom": ["chrA", "chrA", "chrA", "chrB", "chrA"],
                      "pos": [11, 21, 31, 11, 221],
                      "ref": ["G", "C", "A", "A", "A"],
                      "alt": ["A", "T", "G", "C", "G"],
                      "hap": [1, 1, 2, 2, 1]})
    recs, contigs, n = ps.lift_snvs(d, ch, fa)
    assert recs == [("chrA", 61, ".", "G", "A", "1|0"),
                    ("chrA", 71, ".", "T", "C", "0|1"),
                    ("chrB", 490, ".", "T", "G", "0|1")]
    assert contigs == {"chrA": 1200, "chrB": 500}
    assert n["swapped"] == 1 and n["ref_mismatch"] == 1
    assert n["several_blocks"] == 1 and n["written"] == 3


def test_syri_end_to_end(tmp_path):
    (tmp_path / "x.chain").write_text(CHAIN)
    a = np.array(list("N" * 1200))
    a[[59, 69, 119]] = list("ACC")       # pos 10 -> 60, 20 -> 70, 70 -> 120
    fa = _fasta(tmp_path, {"chrA": "".join(a)})
    out = tmp_path / "o" / "IMR90.het.vcf.gz"
    assert ps.main(["syri", "--fasta", fa, "--vcf", str(_syri(tmp_path)),
                    "--chain", str(tmp_path / "x.chain"),
                    "--output", str(out), "--work-dir",
                    str(tmp_path / "w")]) == 0
    assert Path(str(out) + ".tbi").exists()
    t = ai.read_het_snps(out)
    assert list(zip(t["pos"], t["ref"], t["alt"], t["ref_hap"])) == [
        (59, "A", "G", 2), (69, "C", "T", 1), (119, "C", "G", 1)]


def _serve(payload: bytes, plan: list):
    """A local HTTP server of ``payload`` that honours ranges; its n-th
    response does ``plan[n]``: 'short' -- a third of the body, then the
    connection closes though Content-Length said more -- 'corrupt' (one
    bit flipped), else all of it.  Returns it and the ranges' starts."""
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            start = int(rng.split("=")[1].split("-")[0]) if rng else 0
            seen.append(start)
            act = plan[len(seen) - 1] if len(seen) <= len(plan) else "ok"
            if start >= len(payload):
                self.send_response(416)
                self.end_headers()
                return
            body = payload[start:]
            self.send_response(206 if rng else 200)
            if rng:
                self.send_header("Content-Range", f"bytes {start}-"
                                 f"{len(payload) - 1}/{len(payload)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if act == "short":
                self.wfile.write(body[:len(body) // 3])
            elif act == "corrupt":
                self.wfile.write(bytes([body[0] ^ 1]) + body[1:])
            else:
                self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


@pytest.mark.parametrize("plan, starts", [
    (["short"], [0, 333_333]),          # resumed from where it stopped
    (["corrupt"], [0, 0]),              # md5: downloaded again
    (["short", "corrupt", "short"], [0, 333_333, 0, 333_333])])
def test_fetch_resumes_and_checks(tmp_path, monkeypatch, plan, starts):
    monkeypatch.setattr(ps, "RETRY_WAIT", 0)
    payload = np.random.default_rng(0).bytes(1_000_000)
    srv, seen = _serve(payload, plan)
    try:
        got = ps.fetch(f"http://127.0.0.1:{srv.server_address[1]}/f.gz",
                       tmp_path / "f.gz", hashlib.md5(payload).hexdigest(),
                       chunk=1 << 16)
    finally:
        srv.shutdown()
    assert got.read_bytes() == payload and seen == starts
    assert not (tmp_path / "f.gz.part").exists()


def test_fetch_gives_up_on_a_file_that_never_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "RETRY_WAIT", 0)
    payload = b"x" * 1000
    srv, seen = _serve(payload, ["corrupt", "corrupt"])
    try:
        with pytest.raises(RuntimeError, match="md5 differs"):
            ps.fetch(f"http://127.0.0.1:{srv.server_address[1]}/f.gz",
                     tmp_path / "f.gz", hashlib.md5(payload).hexdigest())
    finally:
        srv.shutdown()
    assert not (tmp_path / "f.gz").exists()


def _panel(panel: Path, body: dict):
    """A local copy of the panel's files: {chrom: [(pos, ref, alt,
    filter, NA12878's GT)]}, HG00096 0|0 throughout."""
    head = ["##fileformat=VCFv4.2",
            '##FILTER=<ID=PASS,Description="All filters passed">',
            '##FILTER=<ID=LowQ,Description="low">',
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">']
    for c, rows in body.items():
        p = panel / ps.kgp_file(c)
        lines = head + [f"##contig=<ID={c},length=1000>",
                        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\t"
                        "FORMAT\tHG00096\tNA12878"]
        lines += [f"{c}\t{p_}\t.\t{r_}\t{a_}\t.\t{f_}\t.\tGT\t0|0\t{g_}"
                  for p_, r_, a_, f_, g_ in rows]
        plain = p.with_name(p.name[:-3])
        plain.write_text("\n".join(lines) + "\n")
        pysam.tabix_compress(str(plain), str(p), force=True)
        plain.unlink()


def test_one_chromosome_failing_leaves_the_others(tmp_path):
    panel = tmp_path / "panel"
    panel.mkdir()
    _panel(panel, {"chr21": [(100, "A", "G", "PASS", "0|1")]})
    out = tmp_path / "NA12878.het.vcf.gz"
    args = ["1kgp", "--chroms", "chr21,chr22", "--base-url", str(panel),
            "--work-dir", str(tmp_path / "w"), "--n-jobs", "1",
            "--output", str(out)]
    assert ps.main(args) == 1                       # chr22 is missing
    assert (tmp_path / "w" / "NA12878.chr21.vcf.part").exists()
    assert not out.exists()
    _panel(panel, {"chr22": [(200, "C", "T", "PASS", "1|0")]})
    (panel / ps.kgp_file("chr21")).unlink()         # done: not read again
    assert ps.main(args) == 0
    t = ai.read_het_snps(out)
    assert list(zip(t["chrom"], t["pos"])) == [("chr21", 99), ("chr22", 199)]


def test_1kgp_sample_from_a_local_panel(tmp_path):
    panel = tmp_path / "panel"
    panel.mkdir()
    _panel(panel, {
        "chr21": [(100, "A", "G", "PASS", "0|1"),       # kept
                  (200, "C", "T", "PASS", "0/1"),       # unphased
                  (300, "G", "A", "PASS", "1|1"),       # homozygous
                  (400, "AT", "A", "PASS", "1|0"),      # indel
                  (500, "T", "C", "PASS", "1|0"),       # split multiallelic
                  (500, "T", "G", "PASS", "0|1"),
                  (600, "G", "C", "LowQ", "0|1"),       # filtered
                  (700, "c", "t", ".", "1|0")],         # kept
        "chrX": [(50, "A", "C", "PASS", "1|0")]})
    out = tmp_path / "NA12878.het.vcf.gz"
    args = ["1kgp", "--sample", "NA12878", "--chroms", "chr21,chrX",
            "--base-url", str(panel), "--work-dir", str(tmp_path / "w"),
            "--n-jobs", "1", "--output", str(out)]
    assert ps.main(args) == 0
    t = ai.read_het_snps(out)
    assert list(zip(t["chrom"], t["pos"], t["ref"], t["ref_hap"])) == [
        ("chr21", 99, "A", 1), ("chr21", 699, "C", 2), ("chrX", 49, "A", 2)]
    assert Path(str(out) + ".tbi").exists()
    # a rerun takes the finished chromosomes as they are
    (panel / ps.kgp_file("chr21")).unlink()
    assert ps.main(args) == 0
    assert len(ai.read_het_snps(out)) == 3
