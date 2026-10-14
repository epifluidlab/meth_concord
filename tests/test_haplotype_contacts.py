"""haplotype_contacts.py: the inactive and the active X's 3D genome.

Synthetic haplotype-tagged contacts on chr1 and chrX (40 Mb each).  The
active X and both copies of chr1 have TADs: a contact under 1 Mb across a
boundary (every Mb, at .5) is kept with probability 0.4.  The inactive X
has none, but a megadomain boundary at 'DXZ4' (25 Mb) -- a contact across
it kept with probability 0.1 -- more long-range contacts, and a superloop
between DXZ4 and 'FIRRE' (33 Mb).  'GM': the inactive X is hap1 in every
cell; 'IMR': hap1 in half the cells, hap2 in the others, each cell's called
from its methylation (allelic_cells.tsv)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "haplotype_contacts",
    Path(__file__).resolve().parents[1] / "scripts" / "haplotype_contacts.py")
hc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hc)

L = 40_000_000
DXZ4, FIRRE = 25_050_000, 33_050_000


def _contacts(rng, n, inactive, tads, chrom):
    lo_s, hi_s = (3.0, 7.6) if inactive else (3.0, 7.1)
    out = []
    while len(out) < n:
        p1 = int(rng.integers(0, L))
        p2 = p1 + int(10 ** rng.uniform(lo_s, hi_s))
        if p2 >= L:
            continue
        if tads and p2 - p1 < 1_000_000:
            first = (p1 - 500_000) // 1_000_000 + 1   # boundaries > p1
            if first * 1_000_000 + 500_000 <= p2 and rng.random() < 0.6:
                continue
        if inactive and p1 < DXZ4 <= p2 and rng.random() < 0.9:
            continue
        out.append((p1, p2))
    if inactive:
        for _ in range(3):
            out.append((DXZ4 + int(rng.integers(-100_000, 100_000)),
                        FIRRE + int(rng.integers(-100_000, 100_000))))
    return out


def make_data(out: Path, n_cells=40, n=600, seed=0):
    rng = np.random.default_rng(seed)
    hd = out / "haps"
    hd.mkdir(parents=True)
    rows, calls = [], []
    for grp in ("GM", "IMR"):
        for ci in range(n_cells):
            xi = 1 if grp == "GM" or ci % 2 == 0 else 2
            name = f"{grp}_{ci}"
            t = []
            for chrom in ("chr1", "chrX"):
                for hap in (1, 2):
                    inactive = chrom == "chrX" and hap == xi
                    for p1, p2 in _contacts(rng, n, inactive,
                                            tads=not inactive, chrom=chrom):
                        t.append((chrom, p1, p2, hap, 1))
            pd.DataFrame(t, columns=["chrom", "pos1", "pos2", "hap",
                                     "snp_reads"]).to_parquet(
                hd / f"{name}.haplotypes.parquet", index=False)
            rows.append((name, grp))
            # its methylation: the inactive X the more methylated at islands
            d = 0.3 if xi == 1 else -0.3
            calls.append((name, grp, "chrX", "rest", 50, 50, d))
    pd.DataFrame(rows, columns=["cell_name", "group"]).to_csv(
        out / "samples.tsv", sep="\t", index=False)
    pd.DataFrame(calls, columns=["cell", "group", "chrom", "part",
                                 "HCG_island_sites_hap1",
                                 "HCG_island_sites_hap2",
                                 "HCG_island_hap1_minus_hap2"]).to_csv(
        out / "allelic_cells.tsv", sep="\t", index=False)
    with open(out / "boundaries.bed", "w") as fh:
        for c in ("chr1", "chrX"):
            for k in range(L // 1_000_000):
                b = k * 1_000_000 + 500_000
                fh.write(f"{c}\t{b - 5000}\t{b + 5000}\n")
    (out / "el.bed").write_text(
        f"chrX\t{DXZ4 - 50_000}\t{DXZ4 + 50_000}\tDXZ4\n"
        f"chrX\t{FIRRE - 50_000}\t{FIRRE + 50_000}\tFIRRE\n")
    return out


def test_the_inactive_x_folds_apart_and_the_autosomes_do_not(tmp_path):
    d = make_data(tmp_path / "d")
    o = tmp_path / "out"
    hc.main(["--haplotypes", str(d / "haps"), "--samples",
             str(d / "samples.tsv"), "--xi", "GM=1", "--xi", "IMR=cells",
             "--xi-cells", str(d / "allelic_cells.tsv"),
             "--boundaries", str(d / "boundaries.bed"), "--genome", "none",
             "--chrx-par", "0-1000000", "--xi-elements", str(d / "el.bed"),
             "--loop-window", "150000", "--profile-exclude", "8000000",
             "--n-boot", "200", "--plot-format", "png",
             "--output-dir", str(o)])
    s = pd.read_csv(o / "haplotype_contacts.tsv", sep="\t")
    for g in ("GM", "IMR"):
        t = s[s["group"] == g].set_index(["measure", "set"])
        assert (t["copies"] == "Xi|Xa").all()
        # more long-range contacts on the inactive X, alike on chr1
        r = t.loc[("P(s): share ≥ 10 Mb", "chrX")]
        assert r["difference"] > 0.005 and r["ci_lo"] > 0
        a = t.loc[("P(s): share ≥ 10 Mb", "autosomes")]
        assert a["ci_lo"] < 0 < a["ci_hi"]
        assert t.loc[("P(s): share ≥ 10 Mb", "chrX − autosomes"),
                     "ci_lo"] > 0
        # the megadomain boundary: the inactive X's contacts avoid
        # crossing DXZ4, the active X's cross it as expected
        r = t.loc[("crossing DXZ4: log2 obs/exp", "chrX")]
        assert r["copy1"] < -1.5 and abs(r["copy2"]) < 0.3
        assert r["ci_hi"] < -1 and r["rank_along_x"] < 0.1
        # TADs: the active X's and both chr1 copies' boundaries insulate
        r = t.loc[("TAD boundary insulation", "chrX")]
        assert r["copy2"] > 0.3 and abs(r["copy1"]) < 0.15
        assert r["ci_hi"] < 0 and r["q"] < 0.05
        a = t.loc[("TAD boundary insulation", "autosomes")]
        assert a["copy1"] > 0.3 and a["copy2"] > 0.3
        assert a["ci_lo"] < 0 < a["ci_hi"]
        assert t.loc[("TAD boundary insulation", "chrX − autosomes"),
                     "ci_hi"] < 0
        # the superloop: on the inactive X only
        r = t.loc[("superloop DXZ4–FIRRE: obs/exp", "chrX")]
        assert r["copy1"] > 2 and r["copy2"] < 1.6 and r["ci_lo"] > 0
    assert (s.loc[s["group"] == "GM", "cells"] == 40).all()
    for name in ("ps", "ps_ratio", "profile", "boundaries"):
        assert (o / f"haplotype_contacts_{name}.tsv").exists()
    # P(s), copy over copy: below 1 Mb the active X's TADs keep contacts
    # from crossing their boundaries, the inactive X's do not; chr1 flat
    r = pd.read_csv(o / "haplotype_contacts_ps_ratio.tsv", sep="\t")
    r = r[r["group"] == "GM"].set_index(["set", "log10_lo"])
    assert r.loc[("chrX", 5.8), "log2_ratio"] > 0.5
    assert r.loc[("chrX", 5.8), "ci_lo"] > 0
    assert np.isnan(r.loc[("chrX", 7.2), "log2_ratio"])     # none on Xa
    # P(s) as Hi-C draws it: a share of the contacts per kb, falling
    ps = pd.read_csv(o / "haplotype_contacts_ps.tsv", sep="\t")
    a = ps[(ps["group"] == "GM") & (ps["set"] == "autosomes")
           & (ps["copy"] == "Xa")].set_index("log10_lo")
    assert abs(a["share"].sum() - 1) < 1e-9
    assert a.loc[4.0, "per_kb"] > 5 * a.loc[5.0, "per_kb"] > \
        25 * a.loc[6.0, "per_kb"]
    assert abs(r.loc[("autosomes", 5.0), "log2_ratio"]) < 0.2
    assert r.loc[("autosomes", 5.0), "ci_lo"] < 0 < r.loc[("autosomes", 5.0),
                                                          "ci_hi"]
    assert (o / "haplotype_contacts_GM.png").exists()


def test_without_a_call_the_copies_are_haplotypes_and_cancel(tmp_path):
    """IMR without --xi: hap1 against hap2, each the inactive X in half the
    cells -- the megadomain boundary shows on both, the difference
    cancels."""
    d = make_data(tmp_path / "d", n_cells=20)
    o = tmp_path / "out"
    hc.main(["--haplotypes", str(d / "haps"), "--samples",
             str(d / "samples.tsv"), "--groups", "IMR", "--genome", "none",
             "--chrx-par", "0-1000000", "--xi-elements", str(d / "el.bed"),
             "--profile-window", "5000000", "--profile-min-distance",
             "500000", "--n-boot", "100", "--plot-format", "png",
             "--output-dir", str(o)])
    s = pd.read_csv(o / "haplotype_contacts.tsv", sep="\t")
    t = s.set_index(["measure", "set"])
    assert (t["copies"] == "hap1|hap2").all() and (t["xi"] == "none").all()
    # within 5 Mb windows: each copy is the inactive X in half the cells
    r = t.loc[("crossing DXZ4: log2 obs/exp", "chrX")]
    assert r["copy1"] < -0.3 and r["copy2"] < -0.3
    assert abs(r["difference"]) < 0.5
    assert not any(s["measure"].str.startswith("TAD"))


def test_haplotypes_missing_for_every_cell_stop_early(tmp_path):
    d = make_data(tmp_path / "d", n_cells=2, n=10)
    with pytest.raises(SystemExit, match="no cell of --samples"):
        hc.main(["--haplotypes", str(tmp_path), "--samples",
                 str(d / "samples.tsv"), "--output-dir", str(tmp_path / "o")])
