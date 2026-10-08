"""annotation_class_concordance.py: long-range concordance by annotation
class, per cell, against the same links reshuffled.

Synthetic cells: 2 Mb domains alternating A (accessibility 0.6) and B (0.25);
contacts prefer partners in their anchor's compartment, so the pooled
contacts carry A/B; coordination is planted in A-A contacts only, with a
per-cell strength."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "annotation_class_concordance",
    Path(__file__).resolve().parents[1] / "scripts"
    / "annotation_class_concordance.py")
acc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(acc)

DOM = 2_000_000
CHROMS = {"chr1": 20_000_000, "chr2": 20_000_000}


def is_a(pos):
    return (np.asarray(pos) // DOM) % 2 == 0


def make_data(out, n_types=3, per=25, coord=None, offset=None, n_contacts=2500,
              n_short=1500, seed=0, coord_in="A"):
    """``coord(t, c)``: the chance an A-A contact's partner copies its anchor
    in cell c of type t.  ``offset(t, c)`` is added to every posterior of
    that cell AFTER the calls are drawn, so it changes nothing but level."""
    rng = np.random.default_rng(seed)
    out = Path(out)
    post = out / "GCH"
    post.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in range(n_types):
        for c in range(per):
            chrom = rng.choice(list(CHROMS), n_contacts)
            a_pos = rng.integers(0, 17_000_000, n_contacts)
            d = np.exp(rng.uniform(np.log(1_000), np.log(3_000_000),
                                   n_contacts)).astype(np.int64)
            b_pos = a_pos + d
            # contacts prefer their own compartment
            move = (is_a(a_pos) != is_a(b_pos)) & (rng.random(n_contacts) < 0.7)
            b_pos[move] += DOM
            level = lambda x: np.where(is_a(x), 0.6, 0.25)  # noqa: E731
            pa = np.where(rng.random(n_contacts) < level(a_pos), 0.9, 0.1)
            pb = np.where(rng.random(n_contacts) < level(b_pos), 0.9, 0.1)
            if coord is not None:
                aa = (is_a(a_pos) & is_a(b_pos) if coord_in == "A"
                      else ~is_a(a_pos) & ~is_a(b_pos))
                copy = aa & (rng.random(n_contacts) < coord(t, c))
                pb[copy] = pa[copy]
            s_chr = rng.choice(list(CHROMS), n_short)
            s_pos = rng.integers(0, 19_000_000, n_short)
            s_end = s_pos + rng.integers(200, 800, n_short)
            sa = np.where(rng.random(n_short) < level(s_pos), 0.9, 0.1)
            sb = np.where(rng.random(n_short) < level(s_end), 0.9, 0.1)
            df = pd.DataFrame({
                "chrom": np.concatenate([chrom, s_chr]),
                "pos1": np.concatenate([a_pos, s_pos]),
                "pos2": np.concatenate([b_pos, s_end]),
                "posterior1": np.concatenate([pa, sa]),
                "posterior2": np.concatenate([pb, sb])})
            if offset is not None:
                df[["posterior1", "posterior2"]] += offset(t, c)
            df["n1"] = np.int32(3)
            df["n2"] = np.int32(3)
            df["distance"] = (df["pos2"] - df["pos1"]).abs()
            df.to_parquet(post / f"t{t}_c{c}.posteriors.parquet")
            rows.append((f"t{t}_c{c}", f"T{t}"))
    pd.DataFrame(rows, columns=["cell_name", "group"]).to_csv(
        out / "samples.tsv", sep="\t", index=False)
    # annotations: PMD = B, HMD = A; RT early in A; chromHMM states per domain
    tiles = [(c, s, s + 100_000) for c, n in CHROMS.items()
             for s in range(0, n, 100_000)]
    t_ = pd.DataFrame(tiles, columns=["chrom", "start", "end"])
    a_ = is_a(t_["start"])
    pd.DataFrame({"c": t_["chrom"], "s": t_["start"], "e": t_["end"],
                  "score": 0.1, "cls": np.where(a_, "HMD", "PMD"),
                  "lab": "neither"}).to_csv(out / "pmd.bed", sep="\t",
                                            header=False, index=False)
    for name, hi_in_a in (("early", True), ("late", False)):
        pd.DataFrame({"c": t_["chrom"], "s": t_["start"], "e": t_["end"],
                      "v": np.where(a_ == hi_in_a, 2.0, 0.5)}).to_csv(
            out / f"{name}.bedGraph", sep="\t", header=False, index=False)
    seg = [(c, s, s + 50_000) for c, n in CHROMS.items()
           for s in range(0, n, 50_000)]
    sg = pd.DataFrame(seg, columns=["c", "s", "e"])
    alt = (sg["s"] // 50_000) % 2 == 0
    sg["state"] = np.where(is_a(sg["s"]),
                           np.where(alt, "1_mTSS1", "2_mEnhA1"),
                           np.where(alt, "3_mHET1", "4_mQuies1"))
    sg.to_csv(out / "chromhmm.bed", sep="\t", header=False, index=False)
    pd.DataFrame({"state": ["1", "2", "3", "4"],
                  "group": ["TSS", "enhancers", "HET", "quiescent"],
                  "mneumonics": ["mTSS1", "mEnhA1", "mHET1", "mQuies1"]}
                 ).to_csv(out / "groups.tsv", sep="\t", index=False)
    return out


def run(d, systems="ab,rt,chromhmm,pmd", extra=()):
    o = Path(d) / "out"
    acc.main(["--posteriors", f"GCH={Path(d) / 'GCH'}",
              "--samples", str(Path(d) / "samples.tsv"),
              "--systems", systems,
              "--rt-early", str(Path(d) / "early.bedGraph"),
              "--rt-late", str(Path(d) / "late.bedGraph"),
              "--rt-window", "100000",
              "--chromhmm", str(Path(d) / "chromhmm.bed"),
              "--chromhmm-groups", str(Path(d) / "groups.tsv"),
              "--pmd", str(Path(d) / "pmd.bed"),
              "--ab-resolution", "500000", "--ab-min-contacts", "1000",
              "--min-pairs", "10", "--n-boot", "100", "--n-threads", "2",
              "--plot-format", "png", "--output-dir", str(o), *extra])
    return (pd.read_csv(o / "summary.tsv", sep="\t"),
            pd.read_csv(o / "percell.tsv.gz", sep="\t"), o)


def _row(s, system, k1, k2, band="all"):
    r = s[(s["system"] == system) & (s["anchor"] == k1)
          & (s["partner"] == k2) & s["band"].str.startswith(band)]
    assert len(r) == 1, (system, k1, k2, band)
    return r.iloc[0]


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    d = make_data(tmp_path_factory.mktemp("planted"),
                  coord=lambda t, c: 0.1 + 0.8 * (c % 5) / 4)
    return run(d)


# -------------------------------------------------------------- behaviour --


def test_planted_coordination_shows_in_its_class_pair_only(planted):
    s, _, _ = planted
    for system, inside, outside in (("pmd", ("HMD", "HMD"), ("PMD", "PMD")),
                                    ("rt", ("early", "early"),
                                     ("late", "late")),
                                    ("ab", ("A", "A"), ("B", "B"))):
        r = _row(s, system, *inside)
        assert r["delta_lo"] > 0 and r["delta"] > 0.05, system
        o = _row(s, system, *outside)
        assert o["delta_lo"] < 0 < o["delta_hi"] or abs(o["delta"]) < 0.01
        for k in (inside, outside):            # the control is centred
            c = _row(s, system, *k)
            assert c["control_lo"] < 0 < c["control_hi"] or abs(
                c["control"]) < 0.01


def test_cell_to_cell_variation_is_found_where_planted(planted):
    s, _, _ = planted
    r = _row(s, "pmd", "HMD", "HMD")
    assert r["excess_sd"] > 0.05
    assert r["split_half_r"] > 0.3 and abs(r["control_split_half_r"]) < 0.4
    o = _row(s, "pmd", "PMD", "PMD")
    assert o["excess_sd"] < 0.5 * r["excess_sd"]


def test_chromhmm_groups_and_the_figure(planted):
    s, pc, o = planted
    chm = s[s["system"] == "chromhmm"]
    assert set(chm["anchor"]) <= {"promoter", "enhancer", "heterochromatin",
                                  "quiescent"}
    r = _row(s, "chromhmm", "promoter", "enhancer")
    assert r["delta"] > 0
    assert (o / "fig_GCH_pmd.png").exists()
    assert (o / "main_figure.png").exists()
    assert (o / "fig_GCH_pmd_signed.png").exists()
    assert (o / "main_figure_signed.png").exists()
    assert set(pc.columns) >= {"cell", "cell_type", "delta", "control",
                               "mu_anchor", "enough_pairs"}


def test_ab_is_called_per_cell_type_and_oriented(planted):
    _, _, o = planted
    tab = pd.read_csv(o / "compartments_ab.tsv", sep="\t")
    for t, g in tab[tab["ab"].notna() & (tab["ab"] != "")].groupby(
            "cell_type"):
        mid = g["start"] + 250_000
        agree = np.mean((g["ab"] == "A") == is_a(mid))
        assert agree > 0.9, t
    qc = pd.read_csv(o / "compartments_qc.tsv", sep="\t")
    assert (qc["r_with_reference"] > 0.5).all()


def test_no_coordination_gives_nothing_to_find(tmp_path):
    s, _, _ = run(make_data(tmp_path, per=20, seed=3), systems="pmd")
    for k in (("HMD", "HMD"), ("PMD", "PMD")):
        r = _row(s, "pmd", *k)
        assert abs(r["delta"]) < 0.02
        assert r["excess_sd"] < 0.6 * r["cell_sd"]


def test_a_cells_own_level_does_not_count_as_concordance(tmp_path):
    """Centred on its own marginal, a cell shifted up or down in every
    posterior scores exactly as before -- real and reshuffled alike."""
    base = run(make_data(tmp_path / "a", per=10, seed=5,
                         coord=lambda t, c: 0.5), systems="pmd")[1]
    shifted = run(make_data(tmp_path / "b", per=10, seed=5,
                            coord=lambda t, c: 0.5,
                            offset=lambda t, c: 0.03 * t - 0.02 * (c % 3)),
                  systems="pmd")[1]
    key = ["cell", "anchor", "partner", "band"]
    m = base.merge(shifted, on=key, suffixes=("", "_s"))
    assert len(m) == len(base) > 0
    for col in ("C_obs", "delta", "control"):
        assert np.allclose(m[col], m[col + "_s"], atol=1e-5), col


# ----------------------------------------------------------------- pieces --


def test_pmd_reads_column_5_or_the_common_labels(tmp_path):
    p = tmp_path / "pmd.bed"
    pd.DataFrame([("chr1", 0, 100, 0.1, "PMD", "neither"),
                  ("chr1", 100, 200, 0.9, "HMD", "commonHMD"),
                  ("chr1", 200, 300, 0.1, "PMD", "commonPMD"),
                  ("chr1", 300, 400, 0.5, "NA", "neither")]).to_csv(
        p, sep="\t", header=False, index=False)
    iv = acc.read_pmd(p)
    assert iv.lookup(["chr1"] * 4, [50, 150, 250, 350]).tolist() == [1, 0, 1,
                                                                     -1]
    common = acc.read_pmd(p, common_only=True)
    assert common.lookup(["1"] * 3, [50, 150, 250]).tolist() == [-1, 0, 1]


def test_intervals_lookup_edges_and_chrom_names():
    iv = acc.Intervals("x", ("a", "b"), ["chr1", "chr1", "2"], [0, 100, 0],
                       [100, 200, 50], [0, 1, 1])
    got = iv.lookup(["1", "chr1", "chr1", "chr2", "chr3"],
                    [99, 100, 200, 10, 10])
    assert got.tolist() == [0, 1, -1, 1, -1]


# -------------------------------------------------------------- symmetry --


def test_both_ends_are_reshuffled_from_their_own_side_and_class():
    """Two cells, one contact each, each the other's only donor: the right
    end's donor lies to the right in the right end's class and bin, the
    left end's to the left in the left end's -- mirrored, not borrowed."""
    S = acc._ls.GENOME_STRIDE
    g_lo = np.array([1_000_000, 1_010_000], np.int64) + S
    g_hi = np.array([1_050_000, 1_060_000], np.int64) + S
    ok, right, left, _ = acc.reshuffle(
        g_lo, g_hi, g_hi - g_lo, np.array([0, 1]), np.array([0, 0]),
        np.array([1, 1]), np.array([0.4, 0.3]), np.array([-0.1, -0.2]),
        np.array(["s", "s"]), 2, acc._ls.log_dbins(1000, 10_000_000, 4),
        250_000, 2, seed=0)
    assert ok.all()
    assert right[:, 0].tolist() == [-0.2, -0.2]   # cell 1's right end
    assert left[:, 0].tolist() == [0.3, 0.3]      # cell 1's left end
    assert right[:, 1].tolist() == [-0.1, -0.1]
    assert left[:, 1].tolist() == [0.4, 0.4]


def test_pairs_are_unordered_unless_asked(planted, tmp_path):
    s, _, _ = planted
    pmd = s[s["system"] == "pmd"]
    pairs = set(zip(pmd["anchor"], pmd["partner"]))
    assert ("HMD", "PMD") in pairs and ("PMD", "HMD") not in pairs
    d = make_data(tmp_path, per=8, seed=2)
    s2, _, _ = run(d, systems="pmd", extra=("--ordered-pairs",))
    pairs = set(zip(s2["anchor"], s2["partner"]))
    assert ("HMD", "PMD") in pairs and ("PMD", "HMD") in pairs
    assert s2["ordered"].all() and not s["ordered"].any()


def test_a_plain_bed_is_inside_versus_outside(tmp_path):
    p = tmp_path / "lad.bed"
    pd.DataFrame([("chr1", 100, 200), ("chr1", 150, 300),
                  ("chr1", 500, 600)]).to_csv(p, sep="\t", header=False,
                                               index=False)
    iv = acc.read_annotation("LAD", p)
    assert iv.classes == ("LAD", "non-LAD")
    got = iv.lookup(["chr1"] * 5 + ["chr2"], [50, 120, 250, 400, 550, 10])
    assert got.tolist() == [1, 0, 0, 1, 0, -1]
    q = tmp_path / "lab.bed"
    pd.DataFrame([("chr1", 0, 100, "x"), ("chr1", 100, 200, "y")]).to_csv(
        q, sep="\t", header=False, index=False)
    lab = acc.read_annotation("st", q)
    assert lab.classes == ("x", "y")
    assert lab.lookup(["chr1"] * 3, [50, 150, 250]).tolist() == [0, 1, -1]


# ---------------------------------------------------------------- signed --


def test_signed_concordance_is_positive_when_open_and_negative_when_closed(
        planted, tmp_path):
    """Coordination in mostly-open A (level 0.6) is co-open: S > 0.  The
    same in mostly-closed B (level 0.25) is co-closed: S < 0.  The control,
    the same formula on reshuffled links alone, is centred either way."""
    s, pc, _ = planted
    r = _row(s, "pmd", "HMD", "HMD")
    assert r["F"] > 0.5 and r["signed_lo"] > 0
    assert r["signed_control_lo"] < 0 < r["signed_control_hi"] or abs(
        r["signed_control"]) < 0.005
    closed, _, _ = run(make_data(tmp_path, per=20, seed=4, coord_in="B",
                                 coord=lambda t, c: 0.6), systems="pmd")
    r = _row(closed, "pmd", "PMD", "PMD")
    assert r["F"] < 0.5 and r["signed_hi"] < 0
    assert abs(r["signed"]) > 5 * abs(r["signed_control"])
    assert {"F", "F_null", "signed", "signed_control"} <= set(pc.columns)
