"""end_posterior_distance_correlation.py --haplotypes: each copy against
donors of the same cell.

Synthetic cells, chr1 and chrX, every read pair tagged with its haplotype.
On X the inactive copy (hap1) is methylated low and the active one high;
on chr1 the copies are alike.  Without coordination within a copy the
linked pairs agree only because both ends sit on one copy: against donors
of the same cell and copy that is nothing, against donors of either copy
it is the whole of X's excess.  With coordination planted on X's copies,
X stands above the autosomes copy by copy."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "end_posterior_distance_correlation.py")
L = 20_000_000


def _mod():
    spec = importlib.util.spec_from_file_location("epd_units", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["epd_units"] = m
    spec.loader.exec_module(m)
    return m


def _fixture(root: Path, n_cells=20, per=1200, x_coord=0.0, seed=0):
    rng = np.random.default_rng(seed)
    for d in ("GCH", "HCG", "haps"):
        (root / d).mkdir(parents=True, exist_ok=True)
    cells = [f"GM_{i}" for i in range(n_cells)]
    logit = {("chrX", 1): -1.5, ("chrX", 2): 1.5,
             ("chr1", 1): 0.0, ("chr1", 2): 0.0}
    for c in cells:
        recs, haps = [], []
        for (chrom, hap), mu in logit.items():
            p1 = rng.integers(0, L - 3_000_000, per)
            d = (10 ** rng.uniform(2.5, 6.5, per)).astype(np.int64)
            p2 = p1 + d
            z = (rng.normal(0, 1, per) * x_coord if chrom == "chrX"
                 else np.zeros(per))
            post = [1 / (1 + np.exp(-(mu + z + rng.normal(0, 0.8, per))))
                    for _ in range(2)]
            for i in range(per):
                recs.append((chrom, int(p1[i]), int(p2[i]), 6, 6, int(d[i]),
                             float(post[0][i]), float(post[1][i])))
                haps.append((chrom, int(p1[i]), int(p2[i]), hap))
        t = pd.DataFrame(recs, columns=["chrom", "pos1", "pos2", "n1", "n2",
                                        "distance", "posterior1",
                                        "posterior2"])
        for ctx in ("GCH", "HCG"):
            t.to_parquet(root / ctx / f"{c}.posteriors.parquet")
        pd.DataFrame(haps, columns=["chrom", "pos1", "pos2", "hap"]
                     ).to_parquet(root / "haps" / f"{c}.haplotypes.parquet")
    sp = root / "s.tsv"
    pd.DataFrame({"cell_name": cells, "group": "GM"}).to_csv(
        sp, sep="\t", index=False)
    return sp


def _run(root: Path, out: Path):
    m = _mod()
    sp = _fixture(root, x_coord=_run.x_coord)
    argv = ["--posteriors", f"GCH={root / 'GCH'}",
            "--posteriors", f"HCG={root / 'HCG'}", "--samples", str(sp),
            "--group-column", "group", "--haplotypes", str(root / "haps"),
            "--xi", "GM=1", "--genome", "none", "--chrx-par", "none",
            "--chrx-escape", "none", "--null-type", "distance-matched",
            "--null-scope", "cell", "--min-distance", "300",
            "--max-distance", "3000000", "--bins-per-decade", "4",
            "--unit-bands", "300,30000,3000000", "--min-pairs", "30",
            "--n-bootstrap", "200", "--fisher-z", "none", "--per-cell", "off",
            "--n-threads", "1", "--plot-format", "png", "--output-dir",
            str(out)]
    assert m.main(argv) == 0
    t = pd.read_csv(out / "end_posterior_units_GM.tsv", sep="\t")
    # every contact, the bands one by one
    t = t[(t["element"] == "all") & ~t["band"].str.startswith("all")]
    return t[t["measure"] == "gg"].set_index(["comparison", "band_lo"])


def test_without_coordination_x_excess_is_the_copies_differing(tmp_path):
    _run.x_coord = 0.0
    t = _run(tmp_path / "d", tmp_path / "o")
    for b in (300, 30000):
        # against donors of either copy X's pairs agree -- one copy each
        r = t.loc[("chrX both copies", b)]
        assert r["value"] > 0.2 and r["lo"] > 0
        r = t.loc[("copy part: chrX − autosomes", b)]
        assert r["value"] > 0.2 and r["lo"] > 0
        # against donors of their own copy, nothing beyond the autosomes
        for nm in ("chrX same copy", "chrX − autosomes, same copy",
                   "autosomes both copies"):
            assert abs(t.loc[(nm, b), "value"]) < 0.05, nm
    assert (tmp_path / "o" / "end_posterior_units_GM.png").exists()
    # one group per unit, each with its own curves
    assert (tmp_path / "o" / "end_posterior_dist_GM_chrX_Xi.tsv").exists()


def test_coordination_within_x_copies_stands_above_the_autosomes(tmp_path):
    _run.x_coord = 1.0
    t = _run(tmp_path / "d", tmp_path / "o")
    for b in (300, 30000):
        r = t.loc[("chrX − autosomes, same copy", b)]
        assert r["value"] > 0.3 and r["lo"] > 0
        # "same copy" is the copies' mean, not their pooled correlation
        assert abs(t.loc[("chrX same copy", b), "value"] - (
            t.loc[("chrX Xi", b), "value"]
            + t.loc[("chrX Xa", b), "value"]) / 2) < 1e-9


def test_a_pair_is_turned_so_its_element_end_is_kept():
    m = _mod()
    j = pd.DataFrame({"chrom": "chr1", "pos1": [100, 5000, 9000, 20000],
                      "pos2": [3000, 900, 12000, 30000],
                      "p1": [0.1, 0.2, 0.3, 0.4],
                      "p2": [0.9, 0.8, 0.7, 0.6], "distance": [1, 1, 1, 1]})
    j = j.assign(gpos1=m.genome_coords(j["chrom"], j["pos1"]),
                 gpos2=m.genome_coords(j["chrom"], j["pos2"]))
    iv = np.array([[m.genome_coords(["chr1"], [0])[0],
                    m.genome_coords(["chr1"], [1000])[0]]])
    f, ids = m._focus_frame(j, {"g": ("p1", "p2")}, iv, base=500)
    # the first pair's end 1 is in it, the second's end 2: turned round;
    # the last two touch it nowhere
    assert list(ids) == [500, 501]
    assert list(f["pos1"]) == [100, 900] and list(f["pos2"]) == [3000, 5000]
    assert list(f["p1"]) == [0.1, 0.8] and list(f["p2"]) == [0.9, 0.2]
    assert (f["gpos1"].to_numpy() < iv[0, 1]).all()


def test_the_real_partner_is_refused_by_its_id_in_the_whole_frame():
    m = _mod()
    j = pd.DataFrame({"gpos1": np.array([1_000, 50_000], np.int64),
                      "gpos2": np.array([11_000, 60_000], np.int64),
                      "p1": [0.5, 0.5], "p2": [0.5, 0.5],
                      "distance": [10_000.0, 10_000.0]})
    edges = np.array([1e3, 1e5])
    # the pool holds row 1's own end 2 (id 101) and another pair's (id 102)
    pool = (np.array([11_000, 60_000, 60_500], np.int64),
            {"g": np.array([0.1, 0.2, 0.3])},
            np.array([100, 101, 102], np.int64))

    class A:
        link_shuffle_bin, null_scope, null_exclude = 250_000, "cell", 0
    rng = np.random.default_rng(0)
    # row 1 of the whole frame, passed alone: its partner is id 101
    got = m._draw_link_shuffle(j.iloc[[1]].reset_index(drop=True),
                               np.array([10_000.0]), {"g": ("p1", "p2")},
                               pool, 100, edges, A(), rng,
                               ids=np.array([101]))
    assert got is not None and got[1]["g"][0] == 0.3


def _fixture_elements(root: Path, n_cells=24, per=1500, seed=3):
    """chr1 and chrX, both copies; coordination planted on the inactive X
    (hap1) alone, and there only on contacts with an end in 'elem': every
    other block of 200 kb."""
    rng = np.random.default_rng(seed)
    for d in ("GCH", "HCG", "haps"):
        (root / d).mkdir(parents=True, exist_ok=True)
    cells = [f"GM_{i}" for i in range(n_cells)]

    def in_elem(p):
        return (p // 200_000) % 2 == 0
    for c in cells:
        recs, haps = [], []
        for chrom in ("chr1", "chrX"):
            for hap in (1, 2):
                p1 = rng.integers(0, L - 3_000_000, per)
                d = (10 ** rng.uniform(3.0, 6.3, per)).astype(np.int64)
                p2 = p1 + d
                plant = ((chrom == "chrX") & (hap == 1)
                         & (in_elem(p1) | in_elem(p2)))
                z = np.where(plant, rng.normal(0, 1, per), 0.0)
                post = [1 / (1 + np.exp(-(z + rng.normal(0, 0.7, per))))
                        for _ in range(2)]
                for i in range(per):
                    recs.append((chrom, int(p1[i]), int(p2[i]), 6, 6,
                                 int(d[i]), float(post[0][i]),
                                 float(post[1][i])))
                    haps.append((chrom, int(p1[i]), int(p2[i]), hap))
        t = pd.DataFrame(recs, columns=["chrom", "pos1", "pos2", "n1", "n2",
                                        "distance", "posterior1",
                                        "posterior2"])
        for ctx in ("GCH", "HCG"):
            t.to_parquet(root / ctx / f"{c}.posteriors.parquet")
        pd.DataFrame(haps, columns=["chrom", "pos1", "pos2", "hap"]
                     ).to_parquet(root / "haps" / f"{c}.haplotypes.parquet")
    sp = root / "s.tsv"
    pd.DataFrame({"cell_name": cells, "group": "GM"}).to_csv(
        sp, sep="\t", index=False)
    bed = root / "elem.bed"
    pd.DataFrame([(ch, a, a + 200_000) for ch in ("chr1", "chrX")
                  for a in range(0, L, 400_000)]).to_csv(
        bed, sep="\t", header=False, index=False)
    return sp, bed


def test_an_element_shows_the_coordination_of_its_own_contacts(tmp_path):
    m = _mod()
    root, out = tmp_path / "d", tmp_path / "o"
    sp, bed = _fixture_elements(root)
    argv = ["--posteriors", f"GCH={root / 'GCH'}",
            "--posteriors", f"HCG={root / 'HCG'}", "--samples", str(sp),
            "--group-column", "group", "--haplotypes", str(root / "haps"),
            "--xi", "GM=1", "--genome", "none", "--chrx-par", "none",
            "--chrx-escape", "none", "--null-type", "link-shuffle",
            "--null-scope", "cell", "--min-distance", "1000",
            "--max-distance", "2000000", "--bins-per-decade", "4",
            "--unit-bands", "1000,2000000", "--min-pairs", "30",
            "--unit-element", f"elem={bed}", "--unit-element-flank", "0",
            "--n-bootstrap", "200", "--fisher-z", "none", "--per-cell", "off",
            "--n-threads", "1", "--plot-format", "png", "--output-dir",
            str(out)]
    assert m.main(argv) == 0
    t = pd.read_csv(out / "end_posterior_units_GM.tsv", sep="\t")
    t = t[(t["measure"] == "gg") & t["band"].str.startswith("all")]
    v = t.set_index(["element", "comparison"])
    # at the element, the inactive X's contacts agree, the active X's not
    assert v.loc[("elem", "chrX Xi"), "lo"] > 0.15
    assert abs(v.loc[("elem", "chrX Xa"), "value"]) < 0.06
    assert v.loc[("elem", "chrX Xi − chrX Xa"), "lo"] > 0.1
    # over every contact, the contacts away from it dilute the same
    assert (v.loc[("all", "chrX Xi − chrX Xa"), "value"]
            < v.loc[("elem", "chrX Xi − chrX Xa"), "value"])
    # the autosomes' copies, the control, alike at the element too
    assert abs(v.loc[("elem", "autosomes: Xi's − Xa's haplotype"),
                     "value"]) < 0.06
    # the element's units are the copies one by one: no pooled-copies rows
    assert not any("both copies" in c for c in v.loc["elem"].index)


def test_a_unit_with_no_pairs_for_a_measure_still_runs(tmp_path):
    """HCG read on the autosomes alone, as the inactive X's phased HCG
    pairs are too few for the within-cell null to place any: X's HCG
    columns are empty, and the run goes on to its tables."""
    m = _mod()
    root, out = tmp_path / "d", tmp_path / "o"
    sp, bed = _fixture_elements(root, n_cells=8, per=500)
    for f in (root / "HCG").glob("*.parquet"):
        t = pd.read_parquet(f)
        t[t["chrom"] != "chrX"].to_parquet(f)
    argv = ["--posteriors", f"GCH={root / 'GCH'}",
            "--posteriors", f"HCG={root / 'HCG'}", "--samples", str(sp),
            "--group-column", "group", "--haplotypes", str(root / "haps"),
            "--xi", "GM=1", "--genome", "none", "--chrx-par", "none",
            "--chrx-escape", "none", "--null-type", "link-shuffle",
            "--null-scope", "cell", "--null-exclude", "1000",
            "--unit-element", f"elem={bed}", "--min-pairs", "30",
            "--n-bootstrap", "50", "--per-cell", "off", "--n-threads", "1",
            "--plot-format", "png", "--output-dir", str(out)]
    assert m.main(argv) == 0
    t = pd.read_csv(out / "end_posterior_units_GM.tsv", sep="\t")
    x = t[(t["comparison"] == "chrX Xi") & (t["element"] == "all")]
    assert x.loc[x["measure"] == "hh", "value"].isna().all()
    assert x.loc[x["measure"] == "gg", "value"].notna().any()


def _argv_units(root, sp, out, posteriors=None):
    gch = posteriors or root / "GCH"
    return ["--posteriors", f"GCH={gch}", "--posteriors",
            f"HCG={posteriors or root / 'HCG'}", "--samples", str(sp),
            "--group-column", "group", "--haplotypes", str(root / "haps"),
            "--xi", "GM=1", "--genome", "none", "--chrx-par", "none",
            "--chrx-escape", "none", "--null-type", "link-shuffle",
            "--null-scope", "cell", "--min-pairs", "30", "--n-bootstrap",
            "20", "--per-cell", "off", "--n-threads", "1", "--plot-format",
            "png", "--output-dir", str(out)]


def test_a_posteriors_directory_with_no_cells_stops_at_once(tmp_path):
    m = _mod()
    root = tmp_path / "d"
    sp, _ = _fixture_elements(root, n_cells=4, per=200)
    (root / "empty").mkdir()
    with pytest.raises(SystemExit, match="posteriors.parquet"):
        m.main(_argv_units(root, sp, tmp_path / "o", root / "empty"))


def test_haplotypes_of_other_read_pairs_stop_at_once(tmp_path):
    m = _mod()
    root = tmp_path / "d"
    sp, _ = _fixture_elements(root, n_cells=4, per=200)
    # the haplotype files' read pairs a base off the posteriors' own
    for f in (root / "haps").glob("*.parquet"):
        t = pd.read_parquet(f)
        t.assign(pos1=t["pos1"] + 1).to_parquet(f)
    with pytest.raises(SystemExit, match="none of"):
        m.main(_argv_units(root, sp, tmp_path / "o"))



def test_a_split_keeps_the_contacts_crossing_it_or_on_one_side():
    m = _mod()
    reg = m.read_unit_splits(["R=chrX:1,000-2,000"])["R"]
    assert reg["set"] == "X" and reg["region"] == "chrX:1000-2000"
    j = pd.DataFrame({"chrom": ["chrX"] * 6 + ["chr1"],
                      "pos1": [500, 100, 9000, 1500, 500, 3000, 500],
                      "pos2": [3000, 900, 2500, 5000, 1500, 400, 3000]})
    j = j.assign(gpos1=m.genome_coords(j["chrom"], j["pos1"]),
                 gpos2=m.genome_coords(j["chrom"], j["pos2"]))
    f, ids = m._split_frame(j, reg, "cross", base=100)
    # an end each side, in either order; not chr1's at the same places
    assert list(ids) == [100, 105]
    assert list(f["pos1"]) == [500, 3000]
    f, ids = m._split_frame(j, reg, "side", base=100)
    # both ends left or both right; an end inside the region is in neither
    assert list(ids) == [101, 102]


def test_a_split_must_name_a_region():
    m = _mod()
    for bad in ("R=chrX:2000-1000", "R:x=chrX:1-2", "R=chrX", "=chrX:1-2",
                "R=chrY:1-2"):
        with pytest.raises(SystemExit, match="--unit-split"):
            m.read_unit_splits([bad])


def _fixture_split(root: Path, n_cells=20, per=1500, seed=5):
    """chr1 and chrX, both copies, the contacts round 10 Mb; coordination
    planted on the inactive X (hap1) alone, and there only on the contacts
    crossing 10.0-10.2 Mb."""
    rng = np.random.default_rng(seed)
    for d in ("GCH", "HCG", "haps"):
        (root / d).mkdir(parents=True, exist_ok=True)
    cells = [f"GM_{i}" for i in range(n_cells)]
    for c in cells:
        recs, haps = [], []
        for chrom in ("chr1", "chrX"):
            for hap in (1, 2):
                p1 = rng.integers(6_000_000, 14_000_000, per)
                d = (10 ** rng.uniform(4.0, 6.6, per)).astype(np.int64)
                p2 = p1 + d
                cross = (p1 < 10_000_000) & (p2 >= 10_200_000)
                plant = (chrom == "chrX") & (hap == 1) & cross
                z = np.where(plant, rng.normal(0, 1, per), 0.0)
                post = [1 / (1 + np.exp(-(z + rng.normal(0, 0.7, per))))
                        for _ in range(2)]
                for i in range(per):
                    recs.append((chrom, int(p1[i]), int(p2[i]), 6, 6,
                                 int(d[i]), float(post[0][i]),
                                 float(post[1][i])))
                    haps.append((chrom, int(p1[i]), int(p2[i]), hap))
        t = pd.DataFrame(recs, columns=["chrom", "pos1", "pos2", "n1", "n2",
                                        "distance", "posterior1",
                                        "posterior2"])
        for ctx in ("GCH", "HCG"):
            t.to_parquet(root / ctx / f"{c}.posteriors.parquet")
        pd.DataFrame(haps, columns=["chrom", "pos1", "pos2", "hap"]
                     ).to_parquet(root / "haps" / f"{c}.haplotypes.parquet")
    sp = root / "s.tsv"
    pd.DataFrame({"cell_name": cells, "group": "GM"}).to_csv(
        sp, sep="\t", index=False)
    return sp


def test_a_split_shows_the_coordination_of_the_contacts_crossing_it(tmp_path):
    m = _mod()
    root, out = tmp_path / "d", tmp_path / "o"
    sp = _fixture_split(root)
    argv = ["--posteriors", f"GCH={root / 'GCH'}",
            "--posteriors", f"HCG={root / 'HCG'}", "--samples", str(sp),
            "--group-column", "group", "--haplotypes", str(root / "haps"),
            "--xi", "GM=1", "--genome", "none", "--chrx-par", "none",
            "--chrx-escape", "none", "--null-type", "link-shuffle",
            "--null-scope", "cell", "--min-distance", "10000",
            "--max-distance", "4000000", "--bins-per-decade", "4",
            "--units", "x_xi,x_xa,a_xi,a_xa",
            "--unit-bands", "10000,4000000", "--min-pairs", "30",
            "--unit-split", "R=chrX:10000000-10200000",
            "--n-bootstrap", "200", "--fisher-z", "none", "--per-cell", "off",
            "--n-threads", "1", "--plot-format", "png", "--output-dir",
            str(out)]
    assert m.main(argv) == 0
    t = pd.read_csv(out / "end_posterior_units_GM.tsv", sep="\t")
    t = t[(t["measure"] == "gg") & t["band"].str.startswith("all")]
    v = t[t["element"] == "R"].set_index("comparison")
    # the inactive X's crossing contacts agree, its others and the active
    # X's do not
    assert v.loc["chrX Xi crossing R", "lo"] > 0.15
    assert abs(v.loc["chrX Xi same side of R", "value"]) < 0.06
    assert v.loc["chrX Xi: crossing − same side", "lo"] > 0.1
    assert abs(v.loc["chrX Xa: crossing − same side", "value"]) < 0.08
    assert v.loc["crossing − same side: chrX Xi − chrX Xa", "lo"] > 0.1
    # an X region splits X's copies alone
    assert not any("autosomes" in c for c in v.index)
    # every contact, as before, under 'all'
    assert (t["element"] == "all").any()
    assert (out / "end_posterior_units_GM_R.png").exists()


def test_a_split_needs_the_link_shuffle(tmp_path):
    m = _mod()
    root = tmp_path / "d"
    sp, _ = _fixture_elements(root, n_cells=4, per=200)
    argv = _argv_units(root, sp, tmp_path / "o")
    i = argv.index("link-shuffle")
    argv[i] = "distance-matched"
    with pytest.raises(SystemExit, match="link-shuffle"):
        m.main(argv + ["--unit-split", "R=chrX:1000000-1200000"])
