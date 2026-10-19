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
    for c in range(0, L + 1, 2_500_000):          # HCG at one end, GCH
        a_, b_ = max(c - 5_000_000, 0), min(c + 5_000_000, L)
        for h, v in (("hap1", -0.04), ("hap2", 0.0), ("hap1-hap2", -0.04)):
            tr.append(("HCG×GCH", "GM", a_, b_, (a_ + b_) / 2, h, 50.0, v,
                       v - 0.02, v + 0.02, np.nan))
    pd.DataFrame(tr, columns=["context", "group", "bin_start", "bin_end",
                              "center", "haplotype", "contacts", "delta",
                              "delta_lo", "delta_hi", "level"]).to_csv(
        out / "chrx_haplotype_tracks.tsv", sep="\t", index=False)
    # by distance: GM's hap1 the more methylated, the less accessible, at
    # every distance; its HCG-HCG ΔC the higher; the autosomes' copies alike
    rows = []
    for c, lv1, lv2, d1, d2 in (("HCG", 0.8, 0.3, 0.06, 0.02),
                                ("GCH", 0.2, 0.5, 0.02, 0.02),
                                ("HCG×GCH", np.nan, np.nan, -0.04, 0.0)):
        for sname in ("chrX", "autosomes"):
            x = sname == "chrX"
            for e in np.arange(3.0, 7.0, 0.5):
                at = (c, "GM", sname, 10 ** e, 10 ** (e + 0.5))
                a1 = lv1 if x else 0.5
                a2 = lv2 if x else 0.5
                dd = (d1 - d2) if x else 0.0
                rows += [at + ("hap1", 100.0, d1, d1 - 0.01, d1 + 0.01, a1,
                               a1 - 0.01, a1 + 0.01),
                         at + ("hap2", 100.0, d2, d2 - 0.01, d2 + 0.01, a2,
                               a2 - 0.01, a2 + 0.01),
                         at + ("hap1-hap2", 200.0, dd, dd - 0.01, dd + 0.01,
                               np.log2(a1 / a2), np.log2(a1 / a2) - 0.1,
                               np.log2(a1 / a2) + 0.1)]
    pd.DataFrame(rows, columns=[
        "context", "group", "set", "dist_lo", "dist_hi", "haplotype",
        "contacts", "delta", "delta_lo", "delta_hi", "level", "level_lo",
        "level_hi"]).to_csv(out / "chrx_haplotype_distance.tsv", sep="\t",
                            index=False)
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
             "--concordance-distance",
             str(d / "chrx_haplotype_distance.tsv"),
             "--n-boot", "200", "--plot-format", "png",
             "--output-dir", str(o)])
    s = pd.read_csv(o / "haplotype_contacts.tsv", sep="\t")
    bulk = s[s["xi"] == "haplotypes (bulk view)"]
    s = s[s["xi"] != "haplotypes (bulk view)"]
    # IMR as bulk data sees it: each haplotype the inactive X in half the
    # cells, so the copies' difference cancels; GM has no such view
    assert set(bulk["group"]) == {"IMR"} and (bulk["copies"]
                                              == "hap1|hap2").all()
    b_ = bulk.set_index(["measure", "set"])
    r = b_.loc[("P(s): share ≥ 10 Mb", "chrX")]
    assert abs(r["difference"]) < 0.01 and r["ci_lo"] < 0 < r["ci_hi"]
    assert (o / "haplotype_bulk_vs_cells_IMR.png").exists()
    xc = pd.read_csv(o / "haplotype_xi_calls_IMR.tsv", sep="\t")
    assert len(xc) == 40 and set(xc["xi_hap"]) == {1, 2}
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
    assert np.allclose(lt["HCG×GCH_dC_Xi_minus_Xa"].dropna(), -0.04)
    # the copies compared by distance and along X
    for f in ("by_distance", "along_x"):
        assert (o / f"haplotype_{f}_GM.png").exists()
    lv = pd.read_csv(o / "haplotype_levels_GM_HCG.tsv", sep="\t")
    assert np.allclose(lv["log2"], np.log2(0.8 / 0.3))
    assert (lv["lo"] <= lv["log2"] + 1e-9).all()
    lv = pd.read_csv(o / "haplotype_levels_GM_GCH.tsv", sep="\t")
    assert np.allclose(lv["log2"], np.log2(0.2 / 0.5))
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


def test_the_locus_panels_put_each_position_at_one_place():
    # the maps' colour bars sit beside them, not in their width: an
    # element's dotted line runs straight down every panel
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    lo = rng.integers(0, L - 6_000_000, 4000)
    hi = lo + (10 ** rng.uniform(4, 6.7, 4000)).astype(np.int64)
    cp = rng.integers(0, 2, 4000).astype(np.int8)
    tr = hc.locus_tracks(lo, hi, cp, 0, L, None, "chrX", None, None, [], 1,
                         1.0)
    fig = hc.locus_figure(tr, "chrX", 0, L, "GM", ("Xi", "Xa"), [],
                          {"DXZ4": (DXZ4 - 50_000, DXZ4 + 50_000)},
                          np.arange(500_000, L, 1_000_000), None)
    fig.canvas.draw()
    along = [ax for ax in fig.axes if ax.get_xlim() == (0.0, L / 1e6)]
    assert len(along) == 5              # the maps, TADs, ratio, insulation
    at = [ax.transData.transform((DXZ4 / 1e6, 0))[0] for ax in along]
    assert np.ptp(at) < 0.5             # pixels
    bars = [ax for ax in fig.axes if ax not in along]
    assert len(bars) == 2 and min(ax.get_position().x0 for ax in bars) > \
        max(ax.get_position().x1 for ax in along)
    plt.close(fig)


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


#: Escape genes on the synthetic X, irregularly spaced: active on both
#: copies, so the inactive X folds and is methylated there like the active.
ESCAPE = ((4_000_000, 6_000_000), (11_000_000, 12_000_000),
          (17_000_000, 18_500_000), (26_000_000, 27_000_000),
          (33_000_000, 35_000_000))


def _in_escape(p) -> bool:
    return any(a <= p < b for a, b in ESCAPE)


def make_gene_data(out: Path, n_cells=30, n=800, seed=3):
    """GM, its inactive X hap1: its contacts the active X's kind, and a
    third more of them long-range ones (1 Mb or more) between places outside
    the escape genes; its SNPs outside the escape genes more methylated,
    less accessible, those inside read alike on both copies.  Genes 400 kb
    long every Mb."""
    rng = np.random.default_rng(seed)
    hd = out / "haps"
    hd.mkdir(parents=True)
    rows = []

    def draw(k, lo_s, hi_s, outside):
        got = []
        while len(got) < k:
            p1 = int(rng.integers(0, L))
            p2 = p1 + int(10 ** rng.uniform(lo_s, hi_s))
            if p2 >= L or (outside and (_in_escape(p1) or _in_escape(p2))):
                continue
            got.append((p1, p2))
        return got
    for ci in range(n_cells):
        name = f"GM_{ci}"
        t = []
        for chrom in ("chr1", "chrX"):
            for hap in (1, 2):
                xi = chrom == "chrX" and hap == 1
                base = draw(int(n * 0.7) if xi else n, 3.0, 7.1, False)
                extra = draw(n - int(n * 0.7), 6.0, 7.5, True) if xi else []
                t += [(chrom, p1, p2, hap, 1) for p1, p2 in base + extra]
        pd.DataFrame(t, columns=["chrom", "pos1", "pos2", "hap",
                                 "snp_reads"]).to_parquet(
            hd / f"{name}.haplotypes.parquet", index=False)
        rows.append((name, "GM"))
    pd.DataFrame(rows, columns=["cell_name", "group"]).to_csv(
        out / "samples.tsv", sep="\t", index=False)
    (out / "escape.bed").write_text(
        "".join(f"chrX\t{a}\t{b}\n" for a, b in ESCAPE))
    with open(out / "genes.bed", "w") as fh:
        for c in ("chr1", "chrX"):
            for k in range(1, L // 1_000_000):
                fh.write(f"{c}\t{k * 1_000_000 + 100_000}\t"
                         f"{k * 1_000_000 + 500_000}\tg{k}\n")
    pos = np.arange(1_000_000, L, 50_000)
    rh = np.where(np.arange(pos.size) % 2 == 0, 1, 2)
    esc = np.array([_in_escape(p) for p in pos])
    sn = pd.DataFrame({"group": "GM", "chrom": "chrX", "pos": pos,
                       "ref_hap": rh})
    for c, xi_lv, xa_lv in (("HCG", 0.8, 0.3), ("GCH", 0.2, 0.5)):
        xi_ = np.where(esc, xa_lv, xi_lv) + rng.normal(0, 0.02, pos.size)
        xa_ = np.full(pos.size, xa_lv) + rng.normal(0, 0.02, pos.size)
        sn[f"{c}_ref"] = np.where(rh == 1, xi_, xa_)
        sn[f"{c}_alt"] = np.where(rh == 1, xa_, xi_)
        sn[f"{c}_reads_ref"] = sn[f"{c}_reads_alt"] = 10
    sn.to_csv(out / "allelic_snps.tsv", sep="\t", index=False)
    return out


def test_escape_genes_fold_and_read_like_the_active_x(tmp_path):
    d = make_gene_data(tmp_path / "d")
    o = tmp_path / "out"
    hc.main(["--haplotypes", str(d / "haps"), "--samples",
             str(d / "samples.tsv"), "--xi", "GM=1", "--genome", "none",
             "--chrx-par", "0-1000000", "--xi-elements", "none",
             "--chrx-escape", str(d / "escape.bed"),
             "--genes", str(d / "genes.bed"), "--gene-flank", "0",
             "--allelic-snps", str(d / "allelic_snps.tsv"),
             "--windows", "1000000", "--locus", "none", "--n-boot", "200",
             "--plot-format", "png", "--output-dir", str(o)])
    s = pd.read_csv(o / "haplotype_contacts.tsv", sep="\t").set_index(
        ["measure", "set"])
    m = "P(s): share ≥ 1 Mb"
    # the inactivated genes' contacts: the inactive X's the longer
    r = s.loc[(m, "chrX inactivated genes")]
    assert r["difference"] > 0.2 and r["ci_lo"] > 0
    # the escape genes': alike on both copies, and below the inactivated's
    r = s.loc[(m, "chrX escape genes")]
    assert abs(r["difference"]) < 0.04 and r["ci_lo"] < 0 < r["ci_hi"]
    r = s.loc[(m, "escape − inactivated genes")]
    assert r["difference"] < -0.15 and r["ci_hi"] < 0
    # each copy's level at the genes
    g = pd.read_csv(o / "haplotype_genes_GM.tsv", sep="\t").set_index(
        ["context", "class"])
    assert abs(g.loc[("HCG", "inactivated genes"), "difference"] - 0.5) < 0.02
    assert abs(g.loc[("HCG", "escape genes"), "difference"]) < 0.02
    assert abs(g.loc[("GCH", "inactivated genes"), "difference"] + 0.3) < 0.02
    r = g.loc[("HCG", "escape − inactivated genes")]
    assert r["difference"] < -0.45 and r["ci_hi"] < 0
    assert g.loc[("HCG", "escape genes"), "snps"] > 50
    # along X: where the copies' levels differ, so does their folding
    w = pd.read_csv(o / "haplotype_windows_GM.tsv", sep="\t")
    assert len(w) == 38 and (w["escape_bp"] > 0).sum() == 8
    full = w[w["escape_bp"] == 1_000_000]
    rest = w[w["escape_bp"] == 0]
    assert full["d_HCG"].abs().max() < 0.05 and (rest["d_HCG"] > 0.45).all()
    assert rest["d_share_far"].mean() > full["d_share_far"].mean() + 0.15
    c = pd.read_csv(o / "haplotype_windows_correlation_GM.tsv",
                    sep="\t").set_index(["context", "column"])
    r = c.loc[("HCG", "d_share_far")]
    assert r["rho"] > 0.4 and r["p_shift"] < 0.1
    # the active X's level is the same everywhere: given it, the link stays
    assert r["rho_given_Xa_level"] > 0.3
    assert c.loc[("GCH", "d_share_far"), "rho"] < 0
    assert (o / "haplotype_genes_windows_GM.png").exists()


def test_window_link_through_the_active_x_alone_vanishes_given_its_level():
    """Both copies' differences made by the active X's level alone (the
    inactive X flat): a strong rank correlation, gone given that level."""
    rng = np.random.default_rng(5)
    n = 80
    xa = np.convolve(rng.normal(0, 1, n + 9), np.ones(10) / 10, "valid")
    t = pd.DataFrame({"d_HCG": -xa + rng.normal(0, 0.05, n),
                      "d_share_far": xa + rng.normal(0, 0.05, n),
                      "HCG_Xa": xa + rng.normal(0, 0.01, n)})
    c = hc.window_correlations(t, ["HCG"], ("Xi", "Xa")).set_index(
        "column")
    r = c.loc["d_share_far"]
    assert r["rho"] < -0.6 and r["p_shift"] < 0.05
    assert abs(r["rho_given_Xa_level"]) < 0.3
    assert r["p_shift_given_Xa_level"] > 0.05



def make_strength_data(out: Path, n_cells=60, n=700, seed=7, follow=True):
    """GM, its inactive X hap1 in every cell, each cell's XCI strength
    (Xi less Xa at its islands) one of 0.05, 0.25, 0.45, 0.65.  With
    ``follow`` a cell's inactive X folds the more like one the stronger
    its XCI -- that share of its contacts drawn long-range -- otherwise
    every cell's alike."""
    rng = np.random.default_rng(seed)
    hd = out / "haps"
    hd.mkdir(parents=True)
    rows, calls = [], []
    levels = (0.05, 0.25, 0.45, 0.65)
    for ci in range(n_cells):
        name = f"GM_{ci}"
        st = levels[ci % 4]
        q = st / 0.65 if follow else 0.6
        t = []
        for chrom in ("chr1", "chrX"):
            for hap in (1, 2):
                xi = chrom == "chrX" and hap == 1
                k = 0
                while k < n:
                    p1 = int(rng.integers(0, L))
                    far = xi and rng.random() < q
                    p2 = p1 + int(10 ** rng.uniform(3.0, 7.6 if far
                                                    else 7.1))
                    if p2 >= L:
                        continue
                    t.append((chrom, p1, p2, hap, 1))
                    k += 1
        pd.DataFrame(t, columns=["chrom", "pos1", "pos2", "hap",
                                 "snp_reads"]).to_parquet(
            hd / f"{name}.haplotypes.parquet", index=False)
        rows.append((name, "GM"))
        calls.append((name, "GM", "chrX", "rest", 200, 200, 0.3 + st, 0.3,
                      st))
    pd.DataFrame(rows, columns=["cell_name", "group"]).to_csv(
        out / "samples.tsv", sep="\t", index=False)
    pd.DataFrame(calls, columns=[
        "cell", "group", "chrom", "part", "HCG_island_sites_hap1",
        "HCG_island_sites_hap2", "HCG_island_hap1", "HCG_island_hap2",
        "HCG_island_hap1_minus_hap2"]).to_csv(
        out / "allelic_cells.tsv", sep="\t", index=False)
    return out


def _strata_run(d: Path, o: Path):
    hc.main(["--haplotypes", str(d / "haps"), "--samples",
             str(d / "samples.tsv"), "--xi", "GM=1", "--xi-cells",
             str(d / "allelic_cells.tsv"), "--genome", "none",
             "--chrx-par", "0-1000000", "--xi-elements", "none",
             "--locus", "none", "--windows", "0", "--xci-strata", "4",
             "--n-perm", "100", "--n-boot", "100", "--plot-format", "png",
             "--output-dir", str(o)])
    tr = pd.read_csv(o / "haplotype_xci_trend_GM.tsv", sep="\t")
    return tr.set_index(["measure", "set"]), pd.read_csv(
        o / "haplotype_xci_strata_GM.tsv", sep="\t")


def test_cells_with_stronger_xci_fold_more_like_an_inactive_x(tmp_path):
    tr, st = _strata_run(make_strength_data(tmp_path / "d"),
                         tmp_path / "o")
    m = "P(s): share ≥ 1 Mb"
    r = tr.loc[(m, "chrX")]
    assert r["slope"] > 0.04 and r["p_perm"] < 0.05
    # the autosomes' copies follow nothing
    assert tr.loc[(m, "autosomes"), "p_perm"] > 0.05
    x = st[(st["measure"] == m) & (st["set"] == "chrX")].set_index(
        "stratum")
    assert x.loc[4, "difference"] > x.loc[1, "difference"] + 0.03
    assert np.allclose(x["strength_mean"], [0.05, 0.25, 0.45, 0.65])
    c = pd.read_csv(tmp_path / "o" / "haplotype_xci_cells_GM.tsv",
                    sep="\t")
    assert len(c) == 60 and c["noise_sd"].between(0.03, 0.06).all()
    assert (tmp_path / "o" / "haplotype_xci_strata_GM.png").exists()


def test_cells_alike_in_folding_show_no_slope(tmp_path):
    tr, _ = _strata_run(make_strength_data(tmp_path / "d", follow=False),
                        tmp_path / "o")
    assert tr.loc[("P(s): share ≥ 1 Mb", "chrX"), "p_perm"] > 0.05


def _in_elem(p) -> bool:
    """The elements: 200 kb at the start of every Mb."""
    return p % 1_000_000 < 200_000


def make_element_data(out: Path, n_cells=20, n=900, seed=5):
    """GM, its inactive X hap1: its contacts with an end in an element kept
    at 0.4 -- they avoid the elements -- every other copy's at random."""
    rng = np.random.default_rng(seed)
    hd = out / "haps"
    hd.mkdir(parents=True)
    rows = []
    for ci in range(n_cells):
        t = []
        for chrom in ("chr1", "chrX"):
            for hap in (1, 2):
                xi = chrom == "chrX" and hap == 1
                got = 0
                while got < n:
                    p1 = int(rng.integers(0, L))
                    p2 = p1 + int(10 ** rng.uniform(3.0, 7.1))
                    if p2 >= L:
                        continue
                    if xi and (_in_elem(p1) or _in_elem(p2)) \
                            and rng.random() > 0.4:
                        continue
                    t.append((chrom, p1, p2, hap, 1))
                    got += 1
        pd.DataFrame(t, columns=["chrom", "pos1", "pos2", "hap",
                                 "snp_reads"]).to_parquet(
            hd / f"GM_{ci}.haplotypes.parquet", index=False)
        rows.append((f"GM_{ci}", "GM"))
    pd.DataFrame(rows, columns=["cell_name", "group"]).to_csv(
        out / "samples.tsv", sep="\t", index=False)
    with open(out / "elem.bed", "w") as fh:
        for c in ("chr1", "chrX"):
            for k in range(L // 1_000_000):
                fh.write(f"{c}\t{k * 1_000_000}\t{k * 1_000_000 + 200_000}\n")
    return out


def test_an_element_the_inactive_x_avoids_has_fewer_of_its_contacts(
        tmp_path):
    d = make_element_data(tmp_path / "d")
    o = tmp_path / "out"
    hc.main(["--haplotypes", str(d / "haps"), "--samples",
             str(d / "samples.tsv"), "--xi", "GM=1", "--genome", "none",
             "--chrx-par", "none", "--xi-elements", "none",
             "--chrx-escape", "none", "--element", f"elem={d / 'elem.bed'}",
             "--element-flank", "0", "--locus", "none", "--n-boot", "200",
             "--plot-format", "png", "--output-dir", str(o)])
    s = pd.read_csv(o / "haplotype_contacts.tsv", sep="\t").set_index(
        ["measure", "set"])
    m = hc.ELEMENT_SHARE
    # an end in the element on about a third of a copy's contacts, on the
    # inactive X half as many
    r = s.loc[(m, "chrX elem")]
    assert 0.3 < r["copy2"] < 0.45 and r["copy1"] < 0.75 * r["copy2"]
    assert r["ci_hi"] < -0.05
    r = s.loc[(m, "autosomes elem")]
    assert abs(r["difference"]) < 0.03 and r["ci_lo"] < 0 < r["ci_hi"]
    r = s.loc[(m, "elem: chrX − autosomes")]
    assert r["difference"] < -0.08 and r["ci_hi"] < 0
    # the element's contacts are as long on either copy: nothing planted
    r = s.loc[("P(s): share ≥ 1 Mb", "chrX elem")]
    assert abs(r["difference"]) < 0.04

