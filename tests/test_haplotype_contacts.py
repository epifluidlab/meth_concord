"""haplotype_contacts.py: the inactive and the active X's 3D genome.

Synthetic haplotype-tagged contacts on chr1 and chrX (40 Mb each).  The
active X and both copies of chr1 have TADs: a contact under 1 Mb across a
boundary (every Mb, at .5) is kept with probability 0.4.  The inactive X
has none, but a megadomain boundary at 'DXZ4' (25 Mb) -- a contact across
it kept with probability 0.1 -- more long-range contacts, and a superloop
between DXZ4 and 'FIRRE' (33 Mb).  Compartments alternate every 2 Mb (A,
B, ...): a contact 2 Mb or more long between the two kinds is kept with
probability 0.3 -- but on the inactive X, which has none.  'GM': the
inactive X is hap1 in every cell; 'IMR': hap1 in half the cells, hap2 in
the others, each cell's called from its methylation (allelic_cells.tsv).
For the loci, a line's SNP levels (allelic_snps.tsv: the inactive X the
more methylated, the less accessible) and ΔC along X per haplotype
(chrx_haplotype_tracks.tsv)."""

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


def _a(pos) -> bool:
    """Compartment A: 0-2 Mb, 4-6 Mb, ..."""
    return (pos // 2_000_000) % 2 == 0


def _contacts(rng, n, inactive, tads, chrom):
    lo_s, hi_s = (3.0, 7.6) if inactive else (3.0, 7.1)
    out = []
    while len(out) < n:
        p1 = int(rng.integers(0, L))
        p2 = p1 + int(10 ** rng.uniform(lo_s, hi_s))
        if p2 >= L:
            continue
        if not inactive and p2 - p1 >= 2_000_000 and _a(p1) != _a(p2) \
                and rng.random() < 0.7:
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
    with open(out / "eigen.bedGraph", "w") as fh:
        for c in ("chr1", "chrX"):
            for b in range(0, L, 250_000):
                fh.write(f"{c}\t{b}\t{b + 250_000}\t"
                         f"{1.0 if _a(b) else -1.0}\n")
    # the SNPs: GM's hap1 (the inactive X) the more methylated, the less
    # accessible; REF on either haplotype
    pos = np.arange(1_000_000, L, 200_000)
    rh = np.where(np.arange(pos.size) % 2 == 0, 1, 2)
    xi_lv = {"HCG": 0.8, "GCH": 0.2}
    xa_lv = {"HCG": 0.3, "GCH": 0.5}
    sn = pd.DataFrame({"group": "GM", "chrom": "chrX", "pos": pos,
                       "ref_hap": rh})
    for c in ("HCG", "GCH"):
        sn[f"{c}_ref"] = np.where(rh == 1, xi_lv[c], xa_lv[c])
        sn[f"{c}_alt"] = np.where(rh == 1, xa_lv[c], xi_lv[c])
        sn[f"{c}_reads_ref"] = sn[f"{c}_reads_alt"] = 10
    sn.to_csv(out / "allelic_snps.tsv", sep="\t", index=False)
    # ΔC along X per haplotype: GM's hap1 the higher; GCH side by side in
    # 2.5 Mb bins, HCG's sparser contacts in 10 Mb windows slid 2.5 Mb, with
    # the haplotypes' difference
    tr = [("GCH", "GM", b, b + 2_500_000, np.nan, f"hap{h}", 100.0,
           0.05 if h == 1 else 0.01, 0.0, 0.1, 0.4)
          for b in range(0, L, 2_500_000) for h in (1, 2)]
    for c in range(0, L + 1, 2_500_000):
        a_, b_ = max(c - 5_000_000, 0), min(c + 5_000_000, L)
        for h, v in (("hap1", 0.08), ("hap2", 0.02), ("hap1-hap2", 0.06)):
            tr.append(("HCG", "GM", a_, b_, (a_ + b_) / 2, h, 50.0, v,
                       v - 0.05, v + 0.05, 0.5))
    pd.DataFrame(tr, columns=["context", "group", "bin_start", "bin_end",
                              "center", "haplotype", "contacts", "delta",
                              "delta_lo", "delta_hi", "level"]).to_csv(
        out / "chrx_haplotype_tracks.tsv", sep="\t", index=False)
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
             "--compartments", str(d / "eigen.bedGraph"),
             "--locus", "chrX=chrX", "--locus",
             "DXZ4=chrX:15000000-35000000",
             "--allelic-snps", str(d / "allelic_snps.tsv"),
             "--concordance-tracks", str(d / "chrx_haplotype_tracks.tsv"),
             "--n-boot", "200", "--plot-format", "png",
             "--output-dir", str(o)])
    s = pd.read_csv(o / "haplotype_contacts.tsv", sep="\t")
    for g in ("GM", "IMR"):
        t = s[s["group"] == g].set_index(["measure", "set"])
        assert (t["copies"] == "Xi|Xa").all()
        # more long-range contacts on the inactive X, alike on chr1
        r = t.loc[("P(s): share ≥ 10 Mb", "chrX")]
        assert r["difference"] > 0.005 and r["ci_lo"] > 0
        # chr1's copies are made alike: no difference beyond sampling
        a = t.loc[("P(s): share ≥ 10 Mb", "autosomes")]
        assert abs(a["difference"]) < 0.01
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
        assert abs(a["difference"]) < 0.2
        assert t.loc[("TAD boundary insulation", "chrX − autosomes"),
                     "ci_hi"] < 0
        # the superloop: on the inactive X only
        r = t.loc[("superloop DXZ4–FIRRE: obs/exp", "chrX")]
        assert r["copy1"] > 2 and r["copy2"] < 1.6 and r["ci_lo"] > 0
        # compartments: kept by the active X and both chr1 copies
        r = t.loc[("A/B compartments: ends' correlation", "chrX")]
        assert r["copy2"] > 0.3 and abs(r["copy1"]) < 0.1
        assert r["ci_hi"] < 0
        a = t.loc[("A/B compartments: ends' correlation", "autosomes")]
        assert a["copy1"] > 0.3 and a["copy2"] > 0.3
        assert abs(a["difference"]) < 0.1
        # the loci, as a browser draws them
        for nm in ("chrX", "DXZ4"):
            assert (o / f"haplotype_locus_{g}_{nm}.png").exists()
        lt = pd.read_csv(o / f"haplotype_locus_{g}_DXZ4.tsv", sep="\t")
        assert (lt["end"] - lt["start"]).max() == 500_000     # 40 bins
        at = lt[(lt["start"] <= DXZ4) & (DXZ4 < lt["end"])].iloc[0]
        assert at["insulation_Xi"] < at["insulation_Xa"] - 0.5
        # Xi over Xa, P(s) aside: the inactive X's contacts across DXZ4
        # the fewer, those on one side of it the more
        mp = pd.read_csv(o / f"haplotype_locus_{g}_DXZ4_map.tsv.gz",
                         sep="\t")
        far = (mp["start_2"] - mp["start_1"]) >= 3_000_000
        across = far & (mp["start_1"] + 500_000 <= DXZ4) & \
            (DXZ4 < mp["start_2"])
        side = far & ((mp["start_2"] + 500_000 <= DXZ4)
                      | (mp["start_1"] > DXZ4))
        assert mp.loc[across, "log2_oe_ratio"].median() < -1
        assert mp.loc[side, "log2_oe_ratio"].median() > 0.2
        w = lt.dropna(subset=["partners_compartment_Xa",
                              "partners_compartment_Xi"])
        ca = np.corrcoef(w["compartment_bulk"],
                         w["partners_compartment_Xa"])[0, 1]
        ci = np.corrcoef(w["compartment_bulk"],
                         w["partners_compartment_Xi"])[0, 1]
        assert ca > 0.5 > ci
    assert (s.loc[s["group"] == "GM", "cells"] == 40).all()
    # GM's levels per copy: from its SNPs, hap1 the inactive X
    lt = pd.read_csv(o / "haplotype_locus_GM_chrX.tsv", sep="\t")
    assert np.allclose(lt["HCG_Xi"].dropna(), 0.8)
    assert np.allclose(lt["GCH_Xa"].dropna(), 0.5)
    # and its ΔC per copy, HCG-HCG and GCH-GCH, each in its own windows,
    # and HCG's Xi less Xa
    for col, v in (("HCG_dC_Xi", 0.08), ("HCG_dC_Xa", 0.02),
                   ("HCG_dC_Xi_minus_Xa", 0.06), ("GCH_dC_Xi", 0.05),
                   ("GCH_dC_Xa", 0.01)):
        assert lt[col].notna().all() and np.allclose(lt[col], v), col
    assert lt["GCH_dC_Xi_minus_Xa"].isna().all()       # none given
    # IMR's copies are its cells' own: no SNP levels per copy
    assert "HCG_Xi" not in pd.read_csv(o / "haplotype_locus_IMR_chrX.tsv",
                                       sep="\t")
    for name in ("ps", "ps_ratio", "profile", "boundaries"):
        assert (o / f"haplotype_contacts_{name}.tsv").exists()
    # P(s), copy over copy: below 1 Mb the active X's TADs keep contacts
    # from crossing their boundaries, the inactive X's do not; chr1 flat
    r = pd.read_csv(o / "haplotype_contacts_ps_ratio.tsv", sep="\t")
    r = r[r["group"] == "GM"].set_index(["set", "log10_lo"])
    assert r.loc[("chrX", 5.8), "log2_ratio"] > 0.3
    assert r.loc[("chrX", 5.8), "ci_lo"] > 0
    assert np.isnan(r.loc[("chrX", 7.2), "log2_ratio"])     # none on Xa
    # P(s) as Hi-C draws it: a share of the contacts per kb, falling
    ps = pd.read_csv(o / "haplotype_contacts_ps.tsv", sep="\t")
    a = ps[(ps["group"] == "GM") & (ps["set"] == "autosomes")
           & (ps["copy"] == "Xa")].set_index("log10_lo")
    assert abs(a["share"].sum() - 1) < 1e-9
    assert a.loc[4.0, "per_kb"] > 5 * a.loc[5.0, "per_kb"] > \
        25 * a.loc[6.0, "per_kb"]
    assert abs(r.loc[("autosomes", 5.0), "log2_ratio"]) < 0.3
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
